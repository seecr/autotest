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

1) An example
=============

Autotest has a global root tester that can have an arbitrarily deep and wide tree of child testers. A typical module uses it as follows.

.. code:: python

    import autotest
    test = autotest.get_tester(__name__)

    def area(w, h):
        return w * h

    @test
    def area_basics():
        assert 9 == area(3, 3)
        assert 6 == area(2, 3)

Its creates a subtester using get_tester(). The resuling test object main access point to all functionality of autotest.  In this case, it is used as a decorator to mark and execute a test function.


2) Basis API
============

The module level API is used once per module to get a tester object. After that the resulting test object is the main API.

Module Level
------------

The autotest core consist of the module level functions:


''basic_config(\*\*options)''

Sets options of the root tester. This can be called only once, before ''get_tester()''. If not called, default options are used. This typicalliy happens in the main of an application or in a program for running tests.


''get_tester(name=None)''

When name is ''None'' returns the root tester. Otherwise it returns a named child of the root.  Name is a potentially hierarchical name separated by dots. Each level in this hierarchy becomes a child of the one preceding it. The last tester object is returned. Thus, ''get_tester("main.sub")'' creates a child ''main'' of the root and a child 'sub' of the child ''main''. It returns that latter.

Testers created this way become globally available. A call to ''get_tester()'' with the same name repeatedly will return the same tester.

Recommended is to use ''test = get_tester(__name__)'' at the start of your module. Using subtesters is a powerful way of organising tests. See the source code of autotest for many examples.


Tester Objects
--------------

A tester object as returned from ''get_tester()'' supports the following functions:

''__call__'' as a decorator for marking functions as tests:

.. code:: python

   @test
   def function_marked_as_test():
       pass

This runs the given function as a test and returns None. Thus, ''function_marked_as_test'' becomes None and the function is garbage collected subsequently. Keeping the test is possible with an option.


''__call__'' as a callable for setting options:

.. code:: python

   @test(**options)
   def function_marked_as_test():
       pass

This creates an anonymous child tester with given options.  If you get creative, you could also run:

.. code:: python

   def function_not_marked():
       pass

   test(**options)(function_not_marked)
   # or
   test(function_not_marked, **options)

All methods are 100% equivalent.


''getChild(\*\*options)''


''addHandler(handler)''

Adds a Python Logger object (from standard module logging) as a handler for output. Child testers will delegate to their parents handlers if they have no handlers themselves. If no handler is present output will be send to the root logger (logging.getLogger()). See main for an example.


''fail(\*args, \*\*kwargs)''

Use as guard in tests. Raises AssertionError with the given args and kwargs, appending kwargs to args.


''log_stats()''

Log the current value of the statistics to the configured output.



3) Core Options
===============

The core knows three options. Hooks may support additional options.

======  =======  =======   ==========================================================
option  type     default   Explanation
======  =======  =======   ==========================================================
keep    boolean  False     Keep the function instead of discarding it.
run     boolean  True      Run immediately.
hooks   list     []        List of hooks that are invoked in order.
======  =======  =======   ==========================================================

Normally, autotest runs a test as soon as it discovers it and then discards it. The example below show how tests can be run later by keeping and invoking them.

.. code:: python

  @test
  def this_test_runs_immediately():
    pass

  assert my_test is None

  @test(keep=True, run=False)
  def another_test_for_running_later():
    pass

  test.isfunction(another_test_for_running_later)
  another_test_for_running_later()


In the code above, ''test.isfunction()'' comes from a standard hook ''operator.py'', see below.



3) API for hooks
================

''__call__(tester, func)''


''lookup(tester, name)''

Implemented by a hook.


''option_get(name, default=None)''

Returns the value for option with given name for 'this' tester or its closesed parent.


''option_setdefault(name, default)''

Set option with name on 'this' tester with 'value'.


''option_enumerate(name)''

Enumerates all values for the option with the give name, starting with this tester, up to all its parent. List and tuple values are reversed and flattened.




2) Two more ways to do asserts
==============================

Hook operator.py

.. code:: python

    @test
    def another_test():
        test.all(x > 1 for x in [1,2,3])
        test.startswith("rumbush", "rum")


This shows how autotest stays close to Python as we know it. It does nothing more than looking up the given attribute in four places:

#) module operator,
   e.g.: test.gt(2, 1)

#) module builtins,
   e.g.: test.isinstance('aa', str)

#) module inspect,
   e.g.: test.isfunction(len)

#) the first argument,
   e.g.: test.isupper(<str>)

The benefits of this is that we do not have to learn new methods, that the assert functions are not limited, and that autotest can print the arguments for us on failure.



3) Fixtures (context managers)
==============================

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



5) Filtering
============

Hook filter.py



6) Diffs
========

Hook diffs.py


7) POD diffs
=============

Hook prrint.py


8) Async all the way
====================

Hook asyncer.py


9) Wildcards
============

Hook wildcard.py


10) Levels
==========

Hook levels.py


11) Extended closure
====================

Hook binder.py


12) Runner main
===============

autotest [options] <module>

--filter


13) Misc
========

