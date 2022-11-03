Autotest: Simpler Test Runner for Python
========================================

Autotest is a simple test tool for Python.

Tests are just functions and are part of the application code; not something separate.

The most prominent differences in how autotest works are:

#) gathering tests is automatic and selects precisely what you actually use in your code, including other packages or projects that you might be developing at the same time
#) it stops on the first failure with a standard Python stack trace
#) when starting your production code, it runs all tests in this environment

In addition, it features the following:

#) works in a familiar Pythonic way, using standard libraries as much as possible,
#) based on operator, pdb, logger, difflib, etc,
#) seamlessly scales from microtests to systemtests,
#) discovers tests through the importing mechanism, no separate test tree (although you can),
#) crosses module, package and project boundaries easily,
#) makes refactorings easier, even across projects, tests just move with code,
#) executes tests immediatly after discovery,
#) stops on first failure, fits into a fast REPL-style way of working,
#) supports levels: unit, integration, performanc etc.,
#) there are fixtures (context managers) like in other test tools,
#) async tests and fixtures are fully supported,
#) most functionality is in hooks which you can extend easily,
#) there is a root tester which cvan have subtesters
#) output is send to a logger.

Although autotest enables a new, more agile, rigorous and Pythonic way of testing, since there is little magic and tests are just functions, you are free to organise them as you wish. You can even do it the Python unittest way, if you want.

1) An example tells the most

.. code:: python

    import autotest
    test = autotest.get_tester(__name__)

    @test
    def some_function():
        test.eq(1, 1)

You import a sub tester for your module using get_tester(). The result is your only and exclusive access point to all functionality of autotest.

In this case, 'test' is used a a decorator to mark a test function. The test object gives access to operators in standard module 'operator' (here 'eq').


2) Two more ways to do asserts

Hook operator.py

.. code:: python

    @test
    def another_test():
        test.all(x > 1 for x in [1,2,3])
        test.startswith("rumbush", "rum")

This shows how autotest stays close to Python as we know it. It does nothing more than looking up the given attribute in three places:

#) module operator,
#) module builtins,
#) the first argument.

The benefits of this is that we do not have to learn new methods, that the assert functions are not limited, and that autotest can print the arguments for us on failure.



3) Fixtures (context managers):

Hook fixtures.py

.. code:: python

  @test.fixture
  def answer(a=42):
    yield a

  with test.answer as p:
    test.eq(42, p)

  @test
  def prope_the_universe(answer):
    test.eq(42, answer)

  @test
  def something_wrong(answer:43):
    test.ne(42, answer)
    test.eq(43, answer)


The .fixture attribute administers the next function as a context manager. It can be used as such, but it can also be declares as argument to the test function.

Fixtures accept arguments themselves by using the ':' notation.

There standard fixtures builtin for:

#) stdout
#) stderr
#) tmp_path
#) raises
#) slow_callback_duration



4) Extended closure

Hook binder.py


5) Filtering

Hook filter.py


6) Diffs

Hook diffs.py


7) Async all the way

Hook asyncer.py


8) Wildcards

Hook wildcard.py


9) Levels

Hook levels.py


10) POD print

Hook prrint.py


11) Misc

Hooks introduce their own options, but there are two main options:

Options:

Normally, autotest runs a test as soon as it discovers it and the discards it.

.. code:: python

  @test
  def my_test():
    pass

  assert my_test is None


This can be influenced with the following options.

======  =======  =======   ==========================================================
option  type     default   Explanation
======  =======  =======   ==========================================================
keep    boolean  False     Keep the function after running instead of discarding it.
run     boolean  True      Run immediately or not.
======  =======  =======   ==========================================================


.. code:: python

  @test(keep=True, run=False)
  def another_test():
    pass

  test.isinstance(another_test, FunctionType)
  another_test()


