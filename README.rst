Autotest: Simpler Test Runner for Python
========================================

Autotest is a simple an extensible test tool for Python. Tests are just functions and tests are part of the application code; not something separate.

The most prominent differences in how autotest works are:

#) gathering tests is automatic and follows the structure of your code
#) testing stops on the first failure with a standard Python stack trace
#) when starting your production code, it runs all tests in this environment

Features
--------

In addition, autotest features the following. It

#) works in a familiar Pythonic way, no magic,
#) is based on operator, pdb, logger, difflib, etc,
#) seamlessly scales from microtests to systemtests,
#) discovers tests through the importing mechanism,
#) crosses module, package and project boundaries easily,
#) makes refactorings easier, even across projects, tests just move with code,
#) executes tests immediatly after discovery,
#) stops on first failure, fits into a fast REPL-style way of working,
#) supports levels: unit, integration, performanc etc.,
#) there are fixtures (context managers) like in other test tools,
#) async tests and fixtures are fully supported,
#) most functionality is in hooks which you can extend easily,
#) there is a root tester which can have subtesters
#) output is send to a logger.

Although autotest promotoes an agile, rigorous and Pythonic way of testing, since there is little magic and tests are just functions, you are free to organise them as you wish. You can even do it the Python unittest way, if you want.

History
-------
Autotest began, as a recalcitrant move, with the following decorator above my tests:

.. code:: python

  def test(f):
      print(f.__qualname__)
      f()

  def i_need_testing(x):
      return x

  @test
  def a_test():
      assert 42 == i_need_testing(42)

That turned out to work so well that it grew out to what we have here today.



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

Its creates a subtester using ``get_tester()``. The resulting test object main access point to all functionality of autotest.  In this case, it is used as a decorator to mark and execute a test function.


2) Basic API
============

The API falls apart into four categories

#) a module level API
#) a tester object API
#) core options
#) hooks API


Module Level API
----------------

The autotest core consist of the module level functions:


``basic_config(**options)``

Sets options of the root tester. This can be called only once, before ``get_tester()``. If not called, default options are used. This typicalliy happens in the main of an application or in a program for running tests.


``get_tester(name=None)``

When name is ``None`` returns the root tester. Otherwise it returns a named child of the root.  Name is a potentially hierarchical name separated by dots. Each level in this hierarchy becomes a child of the one preceding it. The last tester object is returned. Thus, ``get_tester("main.sub")`` creates a child ``main`` of the root and a child ``sub`` of the child ``main``. It returns the latter.

Testers created this way become globally available. A call to ``get_tester()`` with the same name repeatedly will return the same tester.

Recommended is to use ``test = get_tester(__name__)`` at the start of your module. Using subtesters is a powerful way of organising tests. See the source code of autotest for many examples.


Tester Objects API
------------------

A tester object as returned from ``get_tester()`` supports the following functions:

``__call__(func)``

A decorator for marking functions as tests:

.. code:: python

   @test
   def function_marked_as_test():
       pass

This runs the given function as a test and returns None. Thus, ```function_marked_as_test()`` becomes None and the function is garbage collected subsequently. Keeping the test is possible with an option.


``__call__(**options)``

A way for setting options:

.. code:: python

   @test(keep=True, my_option=42)
   def function_marked_as_test():
       pass

This creates an anonymous child tester with given options.  If you get creative, you could also run:

.. code:: python

   def function_not_marked():
       pass

   test(keep=True, my_option=42)(function_not_marked)
   # or
   test(function_not_marked, keep=True, my_option=42)

All methods are 100% equivalent. In fact, the full signature is:

``__call__(*funcs, **options)``

So you can run multiple test functions with the given options at once.


``getChild(**options)``

This function is an alias for ``__call__(**options)``. It does exactly the same.


``addHandler(handler)``

Adds a Python Logger object (from standard module ``logging``) as a handler for output. Child testers will delegate to their parents handlers if they have no handlers themselves. If no handler is present output will be send to the root logger (``logging.getLogger()``). See ``__main__.py`` for an example.

This method is most useful on the root tester, but it can be set anywhere.


``fail(*args, **kwargs)``

Use as guard in tests. Raises ``AssertionError`` with the given ``args`` and ``kwargs``, appending ``kwargs`` to ``args``.


``log_stats()``

Log the current value of the statistics to the configured output.


Core Options
------------

The core knows three options. Hooks may support additional options. Options can be given to any of these calls:

- ``basic_config(**options)``
- ``__call__(**options)``
- ``getChild(**options)``

Child testers inherit options from their parents and can override them temporarily.

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

  another_test_for_running_later()





Hooks API
---------

Hooks are callable objects, optionally also implementing ``lookup()``.  Autotest core only dispatches to the hooks and most useful functionality is implemented in standaard hooks.

Installing a hook is done with the ``hooks`` option.


''__call__(tester, func)''

A hook is an ordinary function accepting arguments ``tester`` and ``func``. It is called when a test function is discovered, usually when the tester is used as decorator. The ``tester`` argument supports the Options API so hooks can manipulate options in the current tester. It should return the same func or a wrapper. If it returns ``None`` evaluating stops completely.

Note that all hooks get to process ``func`` in turn, so be nice to them an use ``functools.wraps`` when you wrap.


''lookup(tester, name)''

Implemented by a hook that wants to intercept attribute lookups on the tester object. The hook can no longer be a simple function but must be an object understanding ``__call__(tester, func)`` and ``lookup(tester, name)``. It is called when an attribute lookup takes place on the tester. When it returns a value, lookup stops. When it raises AttributeError, it continues with the next hook.


Options API
-----------

The Options API is meant for hook manipulating options. Options ar hierarchically registered, that is, each tester can have local values for options, and lookup missing ones in its parent.

``option_get(name, default=None)``

Returns the value for option with given name for this tester or its closesed parent.


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

