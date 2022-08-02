from typing import Set, Optional
from random import Random
from .enumerator import Enumerator
from .. import dsl as D
from .. import spec as S


class RelaxedRandomEnumerator(Enumerator):
    _rand: Random
    _max_depth: int
    _min_depth: int
    _builder: D.Builder

    def __init__(self, spec: S.TyrellSpec, max_depth: int, min_depth: Optional[int]=1, seed: Optional[int]=None):
        self._rand = Random(seed)
        self._builder = D.Builder(spec)
        if max_depth < 0:
            raise ValueError(
                'Max depth cannot be negative: {}'.format(max_depth))
        if min_depth < 0:
            raise ValueError(
                'Min depth cannot be negative: {}'.format(min_depth))
        if min_depth > max_depth:
            raise ValueError(
                'Min depth should not be greater than max depth, got: {}, {}'.format(min_depth, max_depth))
        self._max_depth = max_depth
        self._min_depth = min_depth

    def _do_generate(self, curr_type: S.Type, curr_depth: int, force_leaf: bool, force_nonleaf: bool):
        assert not (force_leaf and force_nonleaf), "force_leaf and force_nonleaf can not be True at the same time"
        if curr_depth>self._max_depth:
            raise ValueError(
                'current_depth exceeds max_depth, drop')

        # First, get all the relevant production rules for current type
        productions = self._builder.get_productions_with_lhs(curr_type)

        if force_leaf:
            new_productions = list(
                filter(lambda x: not x.is_function(), productions))
            # relaxed detection: if no production is available, fall back to original productions
            if len(new_productions) > 0:
                productions = new_productions

        if force_nonleaf:
            new_productions = list(
                filter(lambda x: x.is_function(), productions))
            if len(new_productions) > 0:
                productions = new_productions

        # Pick a production rule uniformly at random
        prod = self._rand.choice(productions)
        if not prod.is_function():
            # make_node() will produce a leaf node
            return ( curr_depth, self._builder.make_node(prod) )
        else:
            # Recursively expand the right-hand-side (generating children first)
            children = []
            children_depth = curr_depth
            for x in prod.rhs:
                child_depth, child_node = self._generate(x, curr_depth + 1)
                if child_depth > children_depth:
                    children_depth = child_depth
                children.append(child_node)
            # make_node() will produce an internal node
            return ( children_depth, self._builder.make_node(prod, children) )

    def _generate(self, curr_type: S.Type, curr_depth: int):
        return self._do_generate(curr_type, curr_depth,
                                 force_leaf=(curr_depth >= self._max_depth),
                                 force_nonleaf=(curr_depth < self._min_depth))

    def next(self):
        while True:
            try:
                ret_depth, ret_node = self._generate(self._builder.output, 0)
            except ValueError as e:
                continue
            # still need to detect lower bound
            if ret_depth<=self._max_depth and ret_depth>=self._min_depth:
                return ret_node

