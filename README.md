# autotest

Simplest thing that could possibly work test runner, based on the notions:

0. Convoluted frameworks that increase the distance between programmer and code, as for
   example unittest and pytest, are overkill for low-tech agile software development. After
   over 20 years of developing large, robust, world class applications with a terminal an vi,
   we know what we need: simple, straightforward little tools, that is all.

0. Pythons import is enough to find and run all tests.  This mechanism is well-known,
   transparent, hierarchical, order preserving.  No magical import processes. No mixing
   of order of tests.  Run tests with:
   $ python3 <module>

0. Tests follow the hierachy of the program, in the order it is defined.  Submodule testing is straightforward and it automatically includes all the tests for the dependencies. No less, no more.  Run part of the tests for a submodule: $ python3 <submodule>

0. The same kind of tests are used for low-level modules as for integration and even
       runtime test.  It just depends on the level (module) you define the test. You can
       even use tests to build your code in phases, see bootstrapping below for an example.
       Consequently, a test is just any funtion with @test, anywhere. No magic.

0. Reporting stops at the first error, particularly useful in combination with 2, and
       gives a clear focus. First things first. There is only one real drawback: assert is
       quite limited as it has no way to tell you the values used to compare. (I tried
       inspect, traceback, dis, trace, log, audit, nothing works, pdb to the rescue).

0. O, and it runs incrementally. Each test is executed as it is discovered. Deterministically.
       No separate phases for collecting, running and reporting. Simple. Fast.
