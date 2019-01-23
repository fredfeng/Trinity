===========================
Deduction-based Synthesizer
===========================

Recall the synthesizer loop we had in the beginning of :doc:`the previous tutorial <02_synthesizer_basic>`:

.. code-block:: python

  def synthesize(enumerator, decider):
      for prog in enumerator.enumerate():
          if decider.is_success(prog):
              return prog
      return None

We mentioned that for this basic search scheme there is room for improvement. Here is one observation: currently the decider does not really communicate much with the enumerator. The enumerator only gets a yes/no answer from the decider and that's it. If the program is accepted, all is good. But if it is not, the enumerator can do nothing except coming up with the next candidate in the search space.

What if the decider, in addition to rejecting the program, can provide more insights to the enumerator? What if the decider is able to tell the enumerator why a program is rejected, and what if the enumerator is able to utilize the reason returned by the decider to make sure that programs that gets rejected by the same reason never gets enumerated again, which may lead to a significant cut of its search space? This simple idea is at the heart of deduction-based synthesis.


Dynamic feedback
================

How would the decider figure out why a program gets rejected? In the setting of input-output example-based synthesis, the decider reject a program because it does not conform to some of the provided examples. However, such examples cannot be used as the feedback directly since they are not really meaningful to the enumerator: if the enumerator knows beforehand which programs are good for all the examples, there would be no point in having a decider in the first place!

To get more information, the decider needs some additional guidance from the user. Take this (somewhat contrived) example:

.. code-block:: python

  from tyrell.spec import parse
  from tyrell.interpreter import PostOrderInterpreter, GeneralError

  spec = parse(r'''
    enum IntConst {
      "-2", "-1", "0", "1", "2"
    }
    value IntValue;
    program ex0(IntValue) -> IntValue;
    func foo: IntValue -> IntValue, IntConst;
  ''')

  class Ex0Interpreter(PostOrderInterpreter):
      def eval_IntConst(self, v):
          return int(v)

      def eval_foo(self, node, args):
          return args[0] + int((args[1] - 1) ** 0.5)

Here, we human can see that the integer square root operation in ``eval_foo`` certainly requires that its argument is no less than zero. If the decider sees that ``foo(@param0, -2)`` is being generated, it should be smart enough to inform the enumerator that it should never let the second parameter of ``foo`` be less than 1.

In Tyrell, we convey this kind of dynamic information though :meth:`~tyrell.interpreter.interpreter.Interpreter.assertArg`:

.. code-block:: python

  class Ex0Interpreter(PostOrderInterpreter):
      def eval_foo(self, node, args):
          self.assertArg(
              node, args,  # These two arguments are required to be directly taken from the formal arguments of eval_foo()
              index=1,  # Specify which argument this assertion is about
              cond=lambda x: x >= 1  # Specify which condition the argument's value needs to satisfy
          )
          return args[0] + int((args[1] - 1) ** 0.5)

Sometimes we refer to the value of other parameters inside the assertion we wrote. Here's an example:

.. code-block:: python

  from tyrell.spec import parse
  from tyrell.interpreter import PostOrderInterpreter, GeneralError

  spec = parse(r'''
    enum IntConst {
      "-3", "-2", "2", "3"
    }
    value IntValue;
    program ex1(IntValue) -> IntValue;
    func bar: IntValue -> IntValue, IntConst;
  ''')

  class Ex1Interpreter(PostOrderInterpreter):
      def eval_IntConst(self, v):
          return int(v)

      def eval_bar(self, node, args):
          self.assertArg(node, args,
              index=1,
              cond=lambda x: args[0] % x == 0,
              capture_indices=[0]  # <- This is needed
          )
          return args[0] / args[1]

Note that inside the lambda we pass to ``cond``, the value of ``args[0]`` is referenced. Whenever that happens, we also need to tell ``assertArg`` that parameter index 0 is *captured* through ``capture_indices``.

.. warning:: Currently only assertions on enum nodes will be processed. But this is not a fundamental limitation and we should really extend the support to make runtime assertion like this more useful.


Static feedback
===============

Writing runtime assertion is not the only way of providing hints to the decider in Tyrell. Alternatively, we could, for each function, provide high-level descriptions on what they do in the spec:

.. code-block:: python

  from tyrell.spec import parse
  from tyrell.interpreter import PostOrderInterpreter, GeneralError

  spec = parse(r'''
    value IntValue {
      # Properties can be defined on value types.
      is_pos: bool;
    }
    program ex2(IntValue, IntValue) -> IntValue;

    # Arguments in function spec can be (optionally) named. This will come in handy when writing specs for them.
    func mult: IntValue r -> IntValue a, IntValue b {
      # Constraints can be defined on function specs
      is_pos(a) && is_pos(b) ==> is_pos(r);
      # Multiple constraints will be joined together with conjunction
      is_pos(a) && !is_pos(b) ==> !is_pos(r);
    }
  ''')

  class Ex2Interpreter(PostOrderInterpreter):
      def eval_IntConst(self, v):
          return int(v)

      def eval_mult(self, node, args):
          return args[0] * args[1]

      # Interpret the defined properties by defining 'apply_ZZZ' for each property named ZZZ
      def apply_is_pos(self, v):
          # v refers to the interpreted value
          return v > 0


Putting it together
===================

In the Tyrell framework, writing dynamic assertions and static constraints are useful only when both the decider and the enumerator is willing them to process them. Unfortunately, neither :class:`~tyrell.enumerator.exhaustive.ExhaustiveEnumerator` nor :class:`~tyrell.decider.example_base.ExampleDecider` mentioned in the :doc:`the previous tutorial <02_synthesizer_basic>` does the processing.

If we want the synthesizer to understand those assertions and constraints we wrote, the combination of :class:`~tyrell.enumerator.smt.SmtEnumerator` and :class:`~tyrell.decider.example_constraint.ExampleConstraintDecider` is needed:

.. code-block:: python

  from tyrell.spec import parse_file
  from tyrell.enumerator import SmtEnumerator
  from tyrell.interpreter import PostOrderInterpreter
  from tyrell.decider import Example, ExampleConstraintDecider
  from tyrell.synthesizer import Synthesizer

  class InterpreterWithAssert(PostOrderInterpreter):
      ...

  spec = parse_file('spec_with_constraints.tyrell')
  synthesizer = Synthesizer(
      enumerator=SmtEnumerator(spec, depth=3, loc=2),  # loc is the number of function calls in the synthesized program
      decider=ExampleConstraintDecider(
          spec=spec,  # spec is needed for this decider
          interpreter=InterpreterWithAssert(),
          examples=[
              Example(input=[4, 3], output=3),
              Example(input=[6, 3], output=9),
              Example(input=[1, 2], output=-2),
              Example(input=[1, 1], output=0),
          ]
      )
  )
  print(synthesizer.synthesize())

.. warning:: Currently due to an implementation quirk, for ``SmtEnumerator`` to work there must be a dummy ``Empty`` value type and dummy ``func empty: Empty -> Empty`` function definition included in the spec file.
