from z3 import *
from collections import deque
from .enumerator import Enumerator
from .optimizer import Optimizer

from .. import dsl as D
from ..logger import get_logger

logger = get_logger('tyrell.enumerator.smt')


class AST:
    def __init__(self):
        self.head = None


class ASTNode:
    def __init__(self, nb=None, depth=None, children=None):
        self.id = nb
        self.depth = depth
        self.children = children
        self.production = None


# FIXME: Currently this enumerator requires an "Empty" production to function properly
class SmtEnumerator(Enumerator):
    # z3 solver
    z3_solver = Solver()

    # productions that are leaf
    leaf_productions = []

    # z3 variables for each production node
    variables = []

    # z3 variables to denote if a node is a function or not
    variables_fun = []

    # map from internal k-tree to nodes of program
    program2tree = {}

    def initLeafProductions(self):
        for p in self.spec.productions():
            # FIXME: improve empty integration
            if not p.is_function() or str(p).find('Empty') != -1:
                self.leaf_productions.append(p)

    def createVariables(self, solver):
        for x in range(0, len(self.nodes)):
            name = 'n' + str(x + 1)
            v = Int(name)
            self.variables.append(v)
            # variable range constraints
            solver.add(And(v >= 0, v < self.spec.num_productions()))
            hname = 'h' + str(x + 1)
            h = Int(hname)
            self.variables_fun.append(h)
            # high variables range constraints
            solver.add(And(h >= 0, h <= 1))

    def createOutputConstraints(self, solver):
        '''The output production matches the output type'''
        ctr = None
        for p in self.spec.get_productions_with_lhs(self.spec.output):
            if ctr is None:
                # variables[0] is the root of the tree
                ctr = self.variables[0] == p.id
            else:
                ctr = Or(ctr, self.variables[0] == p.id)
        solver.add(ctr)

    def createLocConstraints(self, solver):
        '''Exactly k functions are used in the program'''
        ctr = self.variables_fun[0]
        for x in range(1, len(self.variables_fun)):
            ctr += self.variables_fun[x]
        ctr_fun = ctr == self.loc
        solver.add(ctr_fun)

    def createInputConstraints(self, solver):
        '''Each input will appear at least once in the program'''
        input_productions = self.spec.get_param_productions()
        for x in range(0, len(input_productions)):
            ctr = None
            for y in range(0, len(self.nodes)):
                if ctr is None:
                    ctr = self.variables[y] == input_productions[x].id
                else:
                    ctr = Or(self.variables[y] == input_productions[x].id, ctr)
            solver.add(ctr)

    def createFunctionConstraints(self, solver):
        '''If a function occurs then set the function variable to 1 and 0 otherwise'''
        assert len(self.nodes) == len(self.variables_fun)
        for x in range(0, len(self.nodes)):
            for p in self.spec.productions():
                # FIXME: improve empty integration
                if p.is_function() and str(p).find('Empty') == -1:
                    ctr = Implies(
                        self.variables[x] == p.id, self.variables_fun[x] == 1)
                    solver.add(ctr)
                else:
                    ctr = Implies(
                        self.variables[x] == p.id, self.variables_fun[x] == 0)
                    solver.add(ctr)

    def createLeafConstraints(self, solver):
        for x in range(0, len(self.nodes)):
            n = self.nodes[x]
            if n.children is None:
                ctr = self.variables[x] == self.leaf_productions[0].id
                for y in range(1, len(self.leaf_productions)):
                    ctr = Or(self.variables[x] ==
                             self.leaf_productions[y].id, ctr)
                solver.add(ctr)

    def createChildrenConstraints(self, solver):
        for x in range(0, len(self.nodes)):
            n = self.nodes[x]
            if n.children is not None:
                for p in self.spec.productions():
                    assert len(n.children) > 0
                    for y in range(0, len(n.children)):
                        ctr = None
                        child_type = 'Empty'
                        if p.is_function() and y < len(p.rhs):
                            child_type = str(p.rhs[y])
                        for t in self.spec.get_productions_with_lhs(child_type):
                            if ctr is None:
                                ctr = self.variables[n.children[y].id - 1] == t.id
                            else:
                                ctr = Or(
                                    ctr, self.variables[n.children[y].id - 1] == t.id)
                            ctr = Implies(self.variables[x] == p.id, ctr)
                        solver.add(ctr)

    def maxChildren(self) -> int:
        '''Finds the maximum number of children in the productions'''
        max = 0
        for p in self.spec.productions():
            if len(p.rhs) > max:
                max = len(p.rhs)
        return max

    def buildKTree(self, children, depth):
        '''Builds a K-tree that will contain the program'''
        nodes = []
        tree = AST()
        root = ASTNode(1, 1)
        nb = 1
        tree.head = root
        d = deque()
        d.append(root)
        nodes.append(root)
        while len(d) != 0:
            current = d.popleft()
            current.children = []
            for x in range(0, children):
                nb += 1
                c = ASTNode(nb, current.depth + 1)
                nodes.append(c)
                current.children.append(c)
                if c.depth < depth:
                    d.append(c)
        return tree, nodes

    @staticmethod
    def _check_arg_types(pred, python_tys):
        if pred.num_args() < len(python_tys):
            msg = 'Predicate "{}" must have at least {} arugments. Only {} is found.'.format(
                pred.name, len(python_tys), pred.num_args())
            raise ValueError(msg)
        for index, (arg, python_ty) in enumerate(zip(pred.args, python_tys)):
            if not isinstance(arg, python_ty):
                msg = 'Argument {} of predicate {} has unexpected type.'.format(
                    index, pred.name)
                raise ValueError(msg)

    def _resolve_occurs_predicate(self, pred):
        self._check_arg_types(pred, [str, (int, float)])
        prod = self.spec.get_function_production_or_raise(pred.args[0])
        weight = pred.args[1]
        self.optimizer.mk_occurs(prod, weight)

    def _resolve_not_occurs_predicate(self, pred):
        self._check_arg_types(pred, [str, (int, float)])
        prod = self.spec.get_function_production_or_raise(pred.args[0])
        weight = pred.args[1]
        self.optimizer.mk_not_occurs(prod, weight)

    def _resolve_is_not_parent_predicate(self, pred):
        self._check_arg_types(pred, [str, str, (int, float)])
        prod0 = self.spec.get_function_production_or_raise(pred.args[0])
        prod1 = self.spec.get_function_production_or_raise(pred.args[1])
        weight = pred.args[2]
        self.optimizer.mk_is_not_parent(prod0, prod1, weight)

    def _resolve_is_parent_predicate(self, pred):
        self._check_arg_types(pred, [str, str, (int, float)])
        prod0 = self.spec.get_function_production_or_raise(pred.args[0])
        prod1 = self.spec.get_function_production_or_raise(pred.args[1])
        weight = pred.args[2]
        self.optimizer.mk_is_parent(prod0, prod1, weight)

    def resolve_predicates(self):
        try:
            for pred in self.spec.predicates():
                if pred.name == 'occurs':
                    self._resolve_occurs_predicate(pred)
                elif pred.name == 'is_parent':
                    self._resolve_is_parent_predicate(pred)
                elif pred.name == 'not_occurs':
                    self._resolve_not_occurs_predicate(pred)
                elif pred.name == 'is_not_parent':
                    self._resolve_is_not_parent_predicate(pred)
                else:
                    logger.warning('Predicate not handled: {}'.format(pred))
        except (KeyError, ValueError) as e:
            msg = 'Failed to resolve predicates. {}'.format(e)
            raise RuntimeError(msg) from None

    def __init__(self, spec, depth=None, loc=None):
        self.z3_solver = Solver()
        self.leaf_productions = []
        self.variables = []
        self.variables_fun = []
        self.program2tree = {}
        self.spec = spec
        if depth <= 0:
            raise ValueError(
                'Depth cannot be non-positive: {}'.format(depth))
        self.depth = depth
        if loc <= 0:
            raise ValueError(
                'LOC cannot be non-positive: {}'.format(loc))
        self.loc = loc
        self.max_children = self.maxChildren()
        self.tree, self.nodes = self.buildKTree(self.max_children, self.depth)
        self.model = None
        self.initLeafProductions()
        self.createVariables(self.z3_solver)
        self.createOutputConstraints(self.z3_solver)
        self.createLocConstraints(self.z3_solver)
        self.createInputConstraints(self.z3_solver)
        self.createFunctionConstraints(self.z3_solver)
        self.createLeafConstraints(self.z3_solver)
        self.createChildrenConstraints(self.z3_solver)
        self.optimizer = Optimizer(
            self.z3_solver, spec, self.variables, self.nodes)
        self.resolve_predicates()

    def blockModel(self):
        assert(self.model is not None)
        # m = self.z3_solver.model()
        block = []
        # block the model using only the variables that correspond to productions
        for x in self.variables:
            block.append(x != self.model[x])
        ctr = Or(block)
        self.z3_solver.add(ctr)

    def update(self, info=None):
        # TODO: block more than one model
        # self.blockModel() # do I need to block the model anyway?
        if info is not None and not isinstance(info, str):
            for core in info:
                ctr = None
                for constraint in core:
                    if ctr is None:
                        ctr = self.variables[self.program2tree[constraint[0]
                                                               ].id - 1] != constraint[1].id
                    else:
                        ctr = Or(
                            ctr, self.variables[self.program2tree[constraint[0]].id - 1] != constraint[1].id)
                self.z3_solver.add(ctr)
        else:
            self.blockModel()

    def buildProgram(self):
        result = [0] * len(self.model)
        for x in self.model:
            c = x()
            a = str(x)
            if a[:1] == 'n':
                result[int(a[1:]) - 1] = int(str(self.model[c]))

        self.program2tree.clear()

        code = []
        for n in self.nodes:
            prod = self.spec.get_production_or_raise(result[n.id - 1])
            code.append(prod)

        builder = D.Builder(self.spec)
        builder_nodes = [None] * len(self.nodes)
        for x in range(0, len(self.nodes)):
            y = len(self.nodes) - x - 1
            if str(code[self.nodes[y].id - 1]).find('Empty') == -1:
                children = []
                if self.nodes[y].children is not None:
                    for c in self.nodes[y].children:
                        if str(code[c.id - 1]).find('Empty') == -1:
                            assert builder_nodes[c.id - 1] is not None
                            children.append(builder_nodes[c.id - 1])
                n = code[self.nodes[y].id - 1].id
                builder_nodes[y] = builder.make_node(n, children)
                self.program2tree[builder_nodes[y]] = self.nodes[y]

        assert(builder_nodes[0] is not None)
        return builder_nodes[0]

    def next(self):
        while True:
            self.model = self.optimizer.optimize(self.z3_solver)
            if self.model is not None:
                return self.buildProgram()
            else:
                return None
