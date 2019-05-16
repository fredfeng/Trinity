from collections import defaultdict
from typing import cast, Any, Callable, Dict, List, Tuple, Set, FrozenSet
import z3

from .assert_violation_handler import AssertionViolationHandler
from .blame import Blame
from .constraint_encoder import ConstraintEncoder
from .example_base import Example, ExampleDecider
from .eval_expr import eval_expr
from .result import ok, bad
from ..spec import TyrellSpec, ValueType
from ..dsl import Node, AtomNode, ParamNode, ApplyNode, NodeIndexer, dfs
from ..interpreter import Interpreter, InterpreterError
from ..logger import get_logger
from ..spec.expr import *
from ..visitor import GenericVisitor

logger = get_logger('tyrell.decider.example_constraint_pruning')


class Z3Encoder(GenericVisitor):
    _interp: Interpreter
    _indexer: NodeIndexer
    _example: Example
    _unsat_map: Dict[str, Tuple[Node, int]]
    _solver: z3.Solver

    def __init__(self, interp: Interpreter, indexer: NodeIndexer, example: Example):
        self._interp = interp
        self._indexer = indexer
        self._example = example
        self._unsat_map = dict()
        self._solver = z3.Solver()

    def get_z3_var(self, node: Node, pname: str, ptype: ExprType):
        node_id = self._indexer.get_id(node)
        var_name = '{}_n{}'.format(pname, node_id)
        if ptype is ExprType.INT:
            return z3.Int(var_name)
        elif ptype is ExprType.BOOL:
            return z3.Bool(var_name)
        else:
            raise RuntimeError('Unrecognized ExprType: {}'.format(ptype))

    def _get_constraint_var(self, node: Node, index: int):
        node_id = self._indexer.get_id(node)
        var_name = '@n{}_c{}'.format(node_id, index)
        return var_name

    def encode_param_alignment(self, node: Node, ty: ValueType, index: int):
        if not isinstance(ty, ValueType):
            raise RuntimeError(
                'Unexpected program output type: {}'.format(ty))
        for pname, pty in ty.properties:
            actual = self.get_z3_var(node, pname, pty)
            expected_expr = PropertyExpr(pname, pty, ParamExpr(index))
            expected = eval_expr(
                self._interp, self._example.input, self._example.output, expected_expr)
            if expected == -1:
                expected = self.get_z3_var(node, pname + '_sym', pty)
            self._solver.add(actual == expected)

    def encode_output_alignment(self, prog: Node):
        out_ty = cast(ValueType, prog.type)
        self.encode_param_alignment(prog, out_ty, 0)

    def visit_param_node(self, param_node: ParamNode):
        param_ty = cast(ValueType, param_node.type)
        self.encode_param_alignment(
            param_node, param_ty, param_node.index + 1)

    def visit_atom_node(self, atom_node: AtomNode):
        pass

    def visit_apply_node(self, apply_node: ApplyNode):
        def encode_property(prop_expr):
            param_expr = cast(ParamExpr, prop_expr.operand)
            if param_expr.index == 0:
                node = apply_node
            else:
                node = apply_node.args[param_expr.index - 1]
            pname = prop_expr.name
            pty = prop_expr.type
            return self.get_z3_var(node, pname, pty)
        constraint_visitor = ConstraintEncoder(encode_property)
        for index, constraint in enumerate(apply_node.production.constraints):
            cname = self._get_constraint_var(apply_node, index)
            z3_clause = constraint_visitor.visit(constraint)
            self._unsat_map[cname] = (apply_node, index)
            self._solver.assert_and_track(z3_clause, cname)
        for arg in apply_node.args:
            self.visit(arg)

    def add(self, z3_expr: z3.ExprRef):
        self._solver.add(z3_expr)

    def is_unsat(self) -> bool:
        return self._solver.check() == z3.unsat

    def get_blame_nodes(self):
        unsat_core = self._solver.unsat_core()
        if len(unsat_core) == 0:
            return None

        unsat_dict = defaultdict(list)
        for v in unsat_core:
            node, cidx = self._unsat_map[str(v)]
            unsat_dict[node].append(node.production.constraints[cidx])
        return unsat_dict


class PropertyFinder(GenericVisitor):
    _encode_property: Callable[[PropertyExpr], z3.ExprRef]

    def __init__(self, encode_property: Callable[[PropertyExpr], z3.ExprRef]):
        self._encode_property = encode_property
        self._results = []

    def visit_const_expr(self, const_expr: ConstExpr):
        pass

    def visit_property_expr(self, prop_expr: PropertyExpr):
        self._encode_property(prop_expr)

    def visit_unary_expr(self, unary_expr: UnaryExpr):
        self.visit(unary_expr.operand)

    def visit_binary_expr(self, binary_expr: BinaryExpr):
        self.visit(binary_expr.lhs)
        self.visit(binary_expr.rhs)

    def visit_cond_expr(self, cond_expr: CondExpr):
        self.visit(cond_expr.condition)
        self.visit(cond_expr.true_value)
        self.visit(cond_expr.false_value)


class PruningException(Exception):
    _node: Node

    def __init__(self, message, node):
        super().__init__(message)
        self._node = node

    @property
    def node(self):
        return self._node


class ConstraintInterpreter(GenericVisitor):
    _interp: Interpreter
    _inputs: Example
    _z3_encoder: Z3Encoder

    def __init__(self, interp: Interpreter, inputs: List[Any], z3_encoder: Z3Encoder):
        self._interp = interp
        self._inputs = inputs
        self._z3_encoder = z3_encoder

    def visit_atom_node(self, atom_node: AtomNode):
        return self._interp.eval(atom_node, self._inputs)

    def visit_param_node(self, param_node: ParamNode):
        return self._interp.eval(param_node, self._inputs)

    def visit_apply_node(self, apply_node: ApplyNode):
        in_values = [self.visit(x) for x in apply_node.args]
        method_name = self._eval_method_name(apply_node.name)
        method = getattr(self._interp, method_name, None)
        if method is None:
            raise NotImplementedError(
                'Cannot find the required eval method: {}'.format(method_name))
        method_output = method(apply_node, in_values)

        # Now that we get more info on the method output, we can use it to refine the constraints
        def encode_property(prop_expr):
            param_expr = cast(ParamExpr, prop_expr.operand)
            if param_expr.index == 0:
                node = apply_node
                value = method_output
            else:
                node = apply_node.args[param_expr.index - 1]
                value = in_values[param_expr.index - 1]
            pname = prop_expr.name
            pty = prop_expr.type
            z3_var = self._z3_encoder.get_z3_var(node, pname, pty)

            method_name = self._apply_method_name(prop_expr.name)
            method = getattr(self._interp, method_name, None)
            if method is None:
                raise ValueError(
                    'Cannot find the required apply method: {}'.format(method_name))
            property_value = method(value)
            self._z3_encoder.add(z3_var == property_value)
        property_finder = PropertyFinder(encode_property)
        for constraint in apply_node.production.constraints:
            property_finder.visit(constraint)

        if self._z3_encoder.is_unsat():
            raise PruningException(
                'Solver returns unsat when evaluating {}'.format(apply_node),
                apply_node
            )

        return method_output

    @staticmethod
    def _eval_method_name(name):
        return 'eval_' + name

    @staticmethod
    def _apply_method_name(name):
        return 'apply_' + name


class BlameFinder:
    _interp: Interpreter
    _prog: Node
    _indexer: NodeIndexer
    _blames_collection: Set[FrozenSet[Blame]]

    def __init__(self, interp: Interpreter, prog: Node):
        self._interp = interp
        self._prog = prog
        self._indexer = NodeIndexer(prog)
        self._blames_collection = set()

    def _get_raw_blames(self) -> List[List[Blame]]:
        return [list(x) for x in self._blames_collection]

    def _get_blames(self) -> List[List[Blame]]:
        return [list(x) for x in self._blames_collection]

    def process_examples(self, examples: List[Example], equal_output: Callable[[Any, Any], bool]):
        try:
            all_ok = all([self.process_example(example, equal_output) for example in examples])
            if all_ok:
                return ok()
            else:
                blames = self._get_blames()
                if len(blames) == 0:
                    return bad()
                else:
                    return bad(why=blames)
        except PruningException as e:
            node = e.node
            # Blame should include all children of node
            blame_nodes = {child for child in dfs(node)}
            # Blame should also include all non-children non-leaf nodes with constraints
            for prog_node in dfs(self._prog):
                if prog_node.is_param():
                    blame_nodes.add(node)
                elif prog_node.is_apply() and len(prog_node.production.constraints) > 0:
                    blame_nodes.add(node)
            return bad([[Blame(node, node.production) for node in blame_nodes]])

    def process_example(self, example: Example, equal_output: Callable[[Any, Any], bool]):
        z3_encoder = Z3Encoder(self._interp, self._indexer, example)
        z3_encoder.encode_output_alignment(self._prog)
        z3_encoder.visit(self._prog)

        if z3_encoder.is_unsat():
            # If abstract semantics cannot be satisfiable, perform blame analysis
            blame_nodes = z3_encoder.get_blame_nodes()
            if blame_nodes is not None:
                base_nodes = list(blame_nodes.keys())
                self._blames_collection.add(
                    frozenset([(n, n.production) for n in base_nodes])
                )
            return False
        else:
            # If abstract semantics is satisfiable, start interpretation
            constraint_interpreter = ConstraintInterpreter(self._interp, example.input, z3_encoder)
            interpreter_output = constraint_interpreter.visit(self._prog)
            return equal_output(interpreter_output, example.output)


class ExampleConstraintPruningDecider(ExampleDecider):
    assert_handler: AssertionViolationHandler

    def __init__(self,
                 spec: TyrellSpec,
                 interpreter: Interpreter,
                 examples: List[Example],
                 equal_output: Callable[[Any, Any], bool]=lambda x, y: x == y):
        super().__init__(interpreter, examples, equal_output)
        self._assert_handler = AssertionViolationHandler(spec, interpreter)

    def analyze_interpreter_error(self, error: InterpreterError):
        return self._assert_handler.handle_interpreter_error(error)

    def analyze(self, prog):
        blame_finder = BlameFinder(self.interpreter, prog)
        return blame_finder.process_examples(self.examples, self.equal_output)
