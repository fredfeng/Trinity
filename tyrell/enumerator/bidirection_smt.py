from z3 import *
from collections import deque
from .enumerator import Enumerator
from functools import reduce


from .. import dsl as D
from ..logger import get_logger

logger = get_logger('tyrell.enumerator.bidirection_smt')

# Each statement has opcode, args, and lhs.
class Stmt:
    def __init__(self, opcode, args, lhs):
        self.opcode = opcode
        self.args = args
        self.lhs = lhs
        #FIXME: assuming all statements return tables.
        self.type = 'Table'
    
    def __repr__(self):
        return str(self.lhs) + ' = ' + str(self.opcode) + '(' + str(self.args) + ')'


class BidirectEnumerator(Enumerator):
    # z3 solver
    z3_solver = Solver()

    # z3 variables for each production node
    variables = []

    # map from internal k-tree to nodes of program
    program2tree = {}

    def createStmtConstraints(self):
        functions = list(filter(lambda x: x.is_function() and x.id > 0, self.spec.productions()))
        
        for i_loc in range(0, self.loc):
            st = self.lines[i_loc]
            # Opcode has to be one of the high-order functions
            opcode = st.opcode
            ctr_opcode = reduce(lambda a,b: Or(a, b.id == opcode), functions, False)
            self.z3_solver.add(ctr_opcode)

            # All vars defined beforehand.
            def_vars = list(map(lambda x: x.lhs, self.lines[:i_loc]))

            # Each opcode will enforce constraints to its children
            for i in range(0, self.max_children):
                # print('line: ', opcode, ' arg: ', i)
                arg = st.args[i]
                for p in functions:
                    if i < len(p.rhs):
                        child_type = str(p.rhs[i])
                        child_prods = self.spec.get_productions_with_lhs(child_type)
                        child_prods=list(o.id for o in filter(lambda x: (not x.is_function()), child_prods))
                        if child_type == st.type:
                            child_prods = child_prods + def_vars
                        ctr_arg = reduce(lambda a,b: Or(a, b == arg), child_prods, False)
                        self.z3_solver.add(Implies(opcode == p.id, ctr_arg))
                    else:
                        self.z3_solver.add(Implies(opcode == p.id, arg == -1))

    def createDefuseConstraints(self):
        '''All input and intermediate vars will appear at least once in the program'''
        all_args = reduce(lambda a,b: a + b.args, self.lines, [])
        for i in self.spec.get_param_productions():
            ctr_input = reduce(lambda a,b: Or(a, b == i.id), all_args, False)
            self.z3_solver.add(ctr_input)

        if self.loc > 1:
            for i in range(0, self.loc - 1):
                def_var = self.lines[i].lhs
                used_args = reduce(lambda a,b: a + b.args, self.lines[i+1:], [])
                ctr_lhs = reduce(lambda a,b: Or(a, b == def_var), used_args, False)
                self.z3_solver.add(ctr_lhs)

    def maxChildren(self) -> int:
        '''Finds the maximum number of children in the productions'''
        max = 0
        for p in self.spec.productions():
            if len(p.rhs) > max:
                max = len(p.rhs)
        return max

    def buildKLines(self, children, loc, solver):
        lines = []

        for l in range(0,loc):
            lhs_name = 'ret' + str(l)
            lhs = Int(lhs_name)
            opcode_name = 'opcode' + str(l)
            opcode = Int(opcode_name)
            args = []
            for i in range(0, children):
                arg_name = 'arg' + str(i) + '@' + str(l)
                arg_var = Int(arg_name)
                self.variables.append(arg_var)
                args.append(arg_var)
            st = Stmt(opcode, args, lhs)
            lines.append(st)
            self.variables.append(lhs)
            self.variables.append(opcode)
            self.z3_solver.add(lhs == (1000 + l))

        return lines, None

    def __init__(self, spec, depth=None, loc=None):
        self.z3_solver = Solver()
        self.variables = []
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
        self.builder = D.Builder(self.spec)
        self.lines, self.nodes = self.buildKLines(self.max_children, self.loc, self.z3_solver)
        self.model = None
        self.createStmtConstraints()
        self.createDefuseConstraints()

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
        if info is not None and not isinstance(info, str):
            for core in info:
                ctr = reduce(lambda a,b: Or(a, self.program2tree[b[0]] != b[1].id), core, False)
                # print('blocking=============', ctr)
                self.z3_solver.add(ctr)
        else:
            self.blockModel()

    def buildProgram(self):
        self.program2tree.clear()
        return self.stmtToAST(self.lines[-1])

    def stmtToAST(self, stmt):
        opcode = stmt.opcode
        opcode_val = self.model[opcode].as_long()
        args = stmt.args
        children = []
        for arg in args:
            arg_val = self.model[arg].as_long()
            if arg_val == -1:
                break
            if arg_val > 999:
                children.append(self.stmtToAST(self.lines[arg_val - 1000]))
            else:
                child_node = self.builder.make_node(arg_val)
                self.program2tree[child_node] = arg
                children.append(child_node)
                
        cur = self.builder.make_node(opcode_val, children)
        self.program2tree[cur] = opcode
        return cur

    def next(self):
        while True:
            self.model = None
            res = self.z3_solver.check()
            if res == sat:
                self.model = self.z3_solver.model()
                # print(self.model)

            if self.model is not None:
                return self.buildProgram()
            else:
                return None