#!/usr/bin/env python

from sys import argv
import tyrell.spec as S
from tyrell.interpreter import PostOrderInterpreter
from tyrell.enumerator import RandomEnumerator, RelaxedRandomEnumerator
from tyrell.decider import Example, ExampleDecider
from tyrell.synthesizer import Synthesizer
from tyrell.logger import get_logger

logger = get_logger('tyrell')

toy_spec_str = '''
enum SmallInt {
  "0", "1", "2", "3"
}
value Int;
value Empty;

program Toy(Int, Int) -> Int;
func const: Int -> SmallInt;
func plus: Int -> Int, Int;
func minus: Int -> Int, Int;
func mult: Int -> Int, Int;
'''


class ToyInterpreter(PostOrderInterpreter):
    def eval_SmallInt(self, v):
        return int(v)

    def eval_const(self, node, args):
        return args[0]

    def eval_plus(self, node, args):
        return args[0] + args[1]

    def eval_minus(self, node, args):
        return args[0] - args[1]

    def eval_mult(self, node, args):
        return args[0] * args[1]


def execute(interpreter, prog, args):
    return interpreter.eval(prog, args)


def test_all(interpreter, prog, inputs, outputs):
    return all(
        execute(interpreter, prog, inputs[x]) == outputs[x]
        for x in range(0, len(inputs))
    )


def main(seed=None):
    logger.info('Parsing Spec...')
    spec = S.parse(toy_spec_str)
    logger.info('Parsing succeeded')

    logger.info('Building synthesizer...')
    synthesizer = Synthesizer(
        # RandomEnumerator samples ASTs with depths of [0, max_depth)
        enumerator=RandomEnumerator(
            spec, max_depth=1, seed=seed),
        # RelaxedRandomEnumerator will sample ASTs with depths of [min_depth, max_depth]
        # enumerator=RelaxedRandomEnumerator(
        #     spec, max_depth=2, min_depth=0, seed=seed),
        decider=ExampleDecider(
            interpreter=ToyInterpreter(),
            examples=[
                # we want to synthesize the program (x-y)*y (depth=3, loc=2)
                # which is also equivalent to x*y-y*y (depth=3, loc=3)
                Example(input=[4, 3], output=3),
                Example(input=[6, 3], output=9),
                Example(input=[1, 2], output=-2),
                Example(input=[1, 1], output=0),
            ]
        )
    )
    logger.info('Synthesizing programs...')

    prog = synthesizer.synthesize()
    if prog is not None:
        logger.info('Solution found: {}'.format(prog))
    else:
        logger.info('Solution not found!')


if __name__ == '__main__':
    logger.setLevel('DEBUG')
    seed = None
    if len(argv) > 1:
        try:
            seed = int(argv[1])
        except ValueError:
            pass
    main(seed)
