#!/usr/bin/env python

import tyrell.spec as S
from tyrell.interpreter import PostOrderInterpreter
from tyrell.enumerator import SmtEnumerator
from tyrell.decider import Example, ExampleConstraintDecider
from tyrell.synthesizer import Synthesizer
from tyrell.logger import get_logger

logger = get_logger('tyrell')


class ToyInterpreter(PostOrderInterpreter):

    def eval_const(self, node, args):
        return args[0]

    def eval_plus(self, node, args):
        return args[0] + args[1]


def main():
    logger.info('Parsing Spec...')
    spec = S.parse_file('example/simplestring.tyrell')
    logger.info('Parsing succeeded')

    logger.info('Building synthesizer...')
    synthesizer = Synthesizer(
        # enumerator=SmtEnumerator(spec, depth=3, loc=1), # plus(@param1, @param0) / plus(@param0, @param1)
        enumerator=SmtEnumerator(spec, depth=4, loc=3), # plus(plus(@param0, const(_apple_)), @param1)
        decider=ExampleConstraintDecider(
            spec=spec,
            interpreter=ToyInterpreter(),
            examples=[
                # Example(input=["a", "b"], output="ab"), # plus(@param0, @param1)
                # Example(input=["b", "a"], output="ab"), # plus(@param1, @param0)
                Example(input=["a", "b"], output="a_apple_b"),
            ],
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
    main()
