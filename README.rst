========
Autotest
========

1. Introduction
===============

A Simpler Pythonic Test Tool
----------------------------

Autotest is a simple and extensible test tool for Python. The key differences are:

- tests are *ordinary functions* and are part of the application code,
- gathering tests is *automatic* and follows the structure of your code,
- testing *stops on first failure* with a standard Python stack trace,
- *production also runs tests*, at least at startup,
- it relies on well-known *standard/builtin* modules,
- *asyncio* fully supported.

**Test Dependencies**

The most prominent feature is that it automatically tests everything your code *actually* depends on, across modules, packages and projects. This results in running less tests when working on a submodule and, when developing projects in conjunction, you need not run all their test suites after a change (and forget one).

History
-------
Autotest began, as a recalcitrant move away from the frameworks, with the following decorator above my tests:

.. code:: python

  def test(f):                            # the test decorator
      print(f.__qualname__)
      f()

  def i_need_testing(x):
      return x

  @test                                   # marks and runs the test
  def a_test():
      assert 42 == i_need_testing(42)

It would just run the test every time I imported it. That turned out to work so well that it grew out to what we have here today.


Features
--------

Meanwhile autotest gained some features. It

#) works in a familiar *Pythonic* way, no magic,
#) is based on *standard modules* operator, pdb, logger, difflib, inspect, etc,
#) seamlessly scales from *microtests* to *systemtests*,
#) discovers tests through the *importing mechanism*,
#) crosses module, package and project boundaries easily,
#) *makes refactoring easier*, even across projects, tests just move with code,
#) executes tests *immediately* after discovery,
#) *stops on first failure*, fits into a fast REPL-style way of working,
#) supports *levels*: unit, integration, performanc etc.,
#) there are *fixtures* (context managers) like in other test tools,
#) *async tests and fixtures* are fully supported,
#) most functionality is in *hooks* which you can extend easily,
#) there is a *root* tester which can have *subtesters*,
#) output is send to a *logger*.

Although autotest promotes an agile, rigorous and Pythonic way of testing, since there is little magic and tests are just functions, you are free to organise them as you wish. You can even do it the Python unittest way, if you want.



An example
----------

Autotest has a global root tester that can have an arbitrarily deep and wide tree of child testers. A typical module uses it as follows:

.. code:: python

    import autotest
    test = autotest.get_tester(__name__)

    def area(w, h):
        return w * h

    @test
    def area_basics():
        assert 9 == area(3, 3)
        assert 6 == area(2, 3)

Its creates a subtester using ``get_tester()``. The resulting tester object is the main access point to all functionality of autotest.  In this case, it is used as a decorator to mark and execute a test function.

More on assert later.



2. Basic API
============

General
-------

**Tester objects**

Autotest has a hierarchical tree of test runners called ``tester`` objects, a bit like Pythons logging facility. The main program is supposed to configure the root (although it doesn't have to) with various options. Testers lower in the tree can override these options.

**Hooks**

Tester objects only mark and execute tests, optionally with options. Any functionality beyond that is provided by hooks, which themselves are just options that can be overridden. There are hooks for:

- fixtures
- operators
- async support
- filters
- diffs
- wildcards
- levels

These hooks introduce new functionality by extending the API of the tester object.


**API**

The API falls apart into five categories:

- a module level API,
- a tester object API,
- core options,
- hooks API.
- APIs introduced by hooks


Module Level API
----------------

The autotest core consist of two module level functions:


``basic_config(**options)``

Sets options for the root tester. This can be called only once, before ``get_tester()``. If not called, default options are used. This typicalliy happens in the main of an application or in a program for running tests.


``get_tester(name=None)``

When name is ``None`` returns the root tester. Otherwise it returns a named child of the root.  Name is a potentially hierarchical name separated by dots. Each level in this hierarchy becomes a child of the one preceding it. The last tester object is returned. Thus, ``get_tester("main.sub")`` creates a child ``main`` of the root and a child ``sub`` of the child ``main``. It returns the latter.

Testers created this way become globally available. A call to ``get_tester()`` with the same name repeatedly will return the same tester.

Recommended is to use ``test = get_tester(__name__)`` at the start of your module. Using subtesters is a powerful way of organising tests. See the source code of autotest for many examples.


Tester Objects API
------------------

A tester object as returned from ``get_tester()`` support the following methods:

``__call__(func)``

A decorator for marking functions as tests:

.. code:: python

   @test
   def function_marked_as_test():
       pass

This runs the given function and returns ``None``. Thus, ``function_marked_as_test`` becomes ``None`` and the function is garbage collected subsequently. Keeping the test is possible with an option.


``__call__(**options)``

A way for setting options:

.. code:: python

   @test(keep=True, my_option=42)
   def function_marked_as_test():
       pass

**Important**: This creates an *anonymous child* tester with given options. This tester is only used once and then discarded. There is a fixtures that gives tests access to it, see ``Fixtures`` below.


If you get creative, you could also run:

.. code:: python

   def function_not_marked():
       pass

   test(keep=True, my_option=42)(function_not_marked)    # or
   test(function_not_marked, keep=True, my_option=42)

All methods are 100% equivalent. In fact, the full signature is:

``__call__(*funcs, **options)``

So you can run multiple test functions with the given options at once.


``getChild(**options)``

This function is an alias for ``__call__(**options)``. It does exactly the same.


``child(**options)``

This creates a child and returns a context manager. Afterwards it will log the number of tests found and run.

.. code:: python

   test = autotest.get_tester(__name__)
   with test.child(level=CRITICAL) as crit:
       @crit
       def a_critical_test_function():
           pass


``addHandler(handler)``

Adds a Python Logger object (from standard module ``logging``) as a handler for output. Child testers will delegate to their parents if they have no handlers themselves. If no handler is present output will be send to the a general autotest logger (``logging.getLogger('autotest')``). See ``__main__.py`` for an example.

This method is most useful on the root tester, but it can be set anywhere.


``fail(*args, **kwargs)``

Use as guard in tests. Raises ``AssertionError`` with the given ``args``, appending ``kwargs`` to ``args``.


``log_stats()``

Log the current value of the statistics to the configured output. The actual log record contains lots of data, but by default only the message is printed. See ``__main__.py`` for how to configure loggers.


Core Options
------------

The core knows three options. Hooks may support additional options. Options can be given to any of these calls:

- ``basic_config(**options)``,
- ``__call__(**options)``,
- ``getChild(**options)``,
- ``child(**options)``.


Child testers inherit options from their parents and can override them.

==========  =======  =======   ==========================================================
option      type     default   Explanation
==========  =======  =======   ==========================================================
keep        boolean  False     Keep the function instead of discarding it.
run         boolean  True      Run immediately.
hooks       list     []        List of hooks that are invoked in order.
subprocess  boolean  False     Runs test when inside a subprocess.
==========  =======  =======   ==========================================================

Normally, autotest runs a test as soon as it discovers it and then discards it. The example below show how tests can be run later by keeping and invoking them.

.. code:: python

  @test
  def this_test_runs_immediately():
    pass

  assert this_test_runs_immediately is None

  @test(keep=True, run=False)
  def another_test_for_running_later():
    pass

  another_test_for_running_later()


Tests do not run in subprocesses which are spawned because the spawn method reimports all needed modules causing all tests to run again and in an endless loop. This only happens when a test spawns of course, but is is disabled by default because these tests run in practically the same context, which add little value.


Hooks API
---------

Hooks are callable objects, optionally also implementing ``lookup()``.  Autotest core only dispatches to the hooks and most useful functionality is implemented in standaard hooks.

Installing a hook is done with the ``hooks`` option.

.. code:: python

  with test.child(hooks=[my_hook]) as hooked:
      @hooked
      def some_test():
          pass

``__call__(tester, func)``

A hook is an ordinary function accepting arguments ``tester`` and ``func``. It is called when a test function is discovered, usually when the tester is used as decorator. The ``tester`` argument supports the ``Options API`` so hooks can manipulate options in the current tester. It should return the same ``func`` or a wrapper. If it returns ``None`` evaluating stops completely.

As an example, here is the complete hook for filtering:

.. code:: python

  def filter_hook(runner, func):
      f = runner.option_get('filter', '')
      if f in func.__qualname__:
          return func

Note that all hooks get to process ``func`` in turn, so be nice to them an use ``functools.wraps`` when you wrap.


``lookup(tester, name)``

Implemented by a hook that wants to intercept attribute lookups on the tester object. The hook can no longer be a simple function but must be an object understanding both ``__call__(tester, func)`` and ``lookup(tester, name)``. It is called when an attribute lookup takes place on the tester. When it returns a value, lookup stops. When it raises AttributeError, it continues with the next hook.

As an example, here is the hook for diffs, implementing both ``__call__`` and ``lookup`` (references to diff and print functions omitted for clarity):

.. code:: python

  class DiffHook:

      def __call__(self, runner, func):
          return func

      def lookup(self, runner, name):
          if name == 'diff':
              return diff
          if name == 'diff2':
              return diff2
          if name == 'prrint':
              return prrint
          raise AttributeError



Options API
-----------

The Options API is meant for hooks manipulating options. Options are hierarchically registered, that is, each tester can have local values for options, and looks up missing ones in its parent.


``option_get(name, default=None)``

Returns the value for the option with given name for this tester or its closest parent.


``option_setdefault(name, default)``

Get option with name, searching all parents. When missing, sets the option on *this* tester with ``default`` as value and return it.


``option_enumerate(name)``

Enumerates all values for the option with the given name, starting with this tester, up to all its parent. List and tuple values are reversed and flattened (concatenated).





3) APIs from Hooks
===================

Operators
---------

Hook ``operator.py`` introduces the possibility to use various builtin operators instead of the ``assert`` statement. As a last resort, it looks up methods of the first argument to use as asserting statement. For example:

.. code:: python

    @test
    def another_test():
        test.all(x > 1 for x in [1,2,3])      # use builtin all()
        test.startswith("rumbush", "rum")     # use method of first argument

When the given operator returns ``False`` according to ``bool()`` it raises ``AssertionError`` with the actual values of the arguments.

This shows how autotest stays close to Python as we know it. It does nothing more than looking up the given attribute in four places:

#) module ``operator``, e.g.: ``test.gt(2, 1)``,
#) module ``builtins``, e.g.: ``test.isinstance('aa', str)``,
#) module ``inspect``, e.g.: ``test.isfunction(len)``,
#) the first argument, e.g.: ``test.isupper("ABC")``.

The benefits of this is that we do not have to learn new methods, that the assert functions are not limited, and that autotest can print the arguments for us on failure.

**diff**

All operators obtained this way support a keyword ``diff=<function>`` that, when present, is invoked with the actual arguments. The result is then given to the ``AssertionError`` instead of the actual arguments.

.. code:: python

    @test
    def another_test():
        a = {7, 1, 2, 8, 3, 4}
        b = {1, 2, 9, 3, 4, 6}
        test.eq(a, b, diff=set.symmetric_difference)

The code above will raise ``AssertionError`` with as argument: ``{6, 7, 8, 9}``.

For more general purpose diff functions, see the hook ``diffs.py``.

Operators is included in the default root tester.



Fixtures
--------

Hook ``fixtures.py`` introduces fixtures as seen in other test tools. The ``test.fixture`` attribute registers the next function as a fixture. A fixture is a Python ``contextmanager`` and can be used as such, or it can be specified as a formal argument to a test function. Fixtures accept arguments themselves by using the ':' notation.

.. code:: python

   @test.fixture
   def answer(a=42):
       yield a

   with test.answer as p:               # as a context manager
       test.eq(42, p)

   @test
   def probe_the_universe(answer):      # as a formal argument
       test.eq(42, answer)

   @test
   def something_wrong(answer:43):      # with argument 43
       test.ne(42, answer)
       test.eq(43, answer)


There are standard fixtures for:

#) test - gives access to the current tester, useful for anonymous subtesters,
#) stdout - captures ``sys.stdout``, including that of subprocesses, in a ``StringIO``,
#) stderr - captures ``sys.stderr``, including that of subprocesses, in a ``StringIO``,
#) tmp_path:subpath - creates a temporary ``pathlib.Path`` object, optionally with a subpath,
#) raises:(Exception, message) - raises AssertionError if given code does not raise given exception with given message,

An example for using ``raises()`` in two different ways:

.. code:: python

   @test
   def should_raise_error(raises:(AttributeError, "'list' object has no attribute 'a'")):
       [].a

   with test.raises(AttributeError, "'list' object has no attribute 'a'"):
       [].a


Fixtures can be async (``async def``) but async fixtures can only be used in async tests. Async fixtures are executed in the event loop of the async test they are declared for.

The option ``timeout=<time in s>`` specifies the maximum amount in seconds a fixture can run. After that an ``TimeoutError`` is raised.

Fixtures is included in the default root tester.


Filtering
------------

Hook ``filter.py`` supports the option ``filter=<str>`` and only executes test whose *qualified name* includes the given ``<str>``.

.. code:: python

    with test(hooks=[filter_hook]) as ftest:
        with ftest(filter='moon') as moon:
            @moon
            def phase_of_the_moon_bug():
                pass

Filtering is included in the default root tester.


Diffs
-----

Hook ``diffs.py`` provides the attributes:

- ``test.diff(a b)`` -- a Python ``pprint`` + ``difflib`` based general purpose diff for use with the operator hook,
- ``test.diff2(a, b)`` -- an Autotest ``prrint`` + ``difflib`` based diff for ``Plain Old Data`` (POD) objects,
- ``test.prrint(a)`` -- a pretty printer for POD objects. Use instead of Pythons ``pprint()``.

Diffs is included in the default root tester.


Async
-----

Hook ``asyncer.py`` supports ``asyncio`` tests defined with ``async def``. Async tests can contain other async tests, however due to limitations in Python (being that async is partially a syntax feature and not fully dynamic) this forces nested async tests to be executed in a separate event loop in a separate thread.

The option ``timeout=<time in s>`` specifies the maximum amount in seconds a test can run. After that an ``TimeoutError`` is raised.

The option ``slow_callback_duration=<time in s>`` specifies the time after which ``asyncio`` begins to emit warnings about tests running too long.

Async is included in the default root tester.


Wildcards
---------

Hook ``wildcard.py`` introduces the attribute ``test.any`` which can be used in structured data comparisons as a wildcard. Its matching can optionally be limited using a function as argument. It is nice to combine this with the operator hook:

.. code:: python

  test.eq([4, test.any,           42], [4, -3, 42])               # succeeds
  test.ne([4, test.any(test.pos), 42], [4, -3, 42])               # fails

Wildcards is included in the default root tester.


Levels
------

Hook ``levels.py`` introduces test levels such as ``unit``, ``integration``
etcetera, together with two options, ``level`` and ``threshold``, to control
it. It is meant to run only certain tests depending on the context. During
development for example, for reasons of speed, integration and performance
tests can be skipped. The levels are just numbers and a number functions as a
level/threshold, much like as in Pythons ``logging``.

The levels are:

=========== =======
level       value
=========== =======
critical      50
unit          40
integration   30
performance   20
=========== =======

The default level is ``unit``. Test levels are provides as attributes on the tester:

.. code:: python

  @test.critical
  def a_critical_test():
      pass


Tests can also be put at a certain level with an option:

.. code:: python

  @test(level='critical')
  def a_critical_test():
      pass

  with test.child(level='critical') as critical:
       @critical
       def one():
           pass


The default ``threshold`` is ``integration``.

A parent can have the ``threshold`` option set to one of the levels. This will block execution of tests in these children of levels lower than ``threshold``. Note that tests do not have levels, only Testers have. **NB:** the highest ``threshold`` in chain of parent-children determines the tests to run. This means a child tester can run integration tests (threshold='integration'), but the root tester can overrule this to run only unit tests.

Levels is included in the default root tester.


Extended closure
----------------

The hook ``binder.py`` enables binding to a class definition in the making. The namespace of a class being defined is not available inside functions being defined in the class body. See this example:

.. code:: python

  def function_a():
      a = 42
      def function_b():
          assert a = 43
      function_b()            # runs fine
  function_a()

  class class_a:
      a = 42
      def function_b():
          assert a = 43
      function_b()            # NameError: name 'a' is not defined

In order to be able to embed tests in class definitions, the binder hook extends the binding of freevars in test functions to include those of the enclosing class. Sorry for the magic.

This hook is enabled by default, but only performs binding when the option ``bind=True`` is present.

.. code:: python

  class class_a:
      a = 42
      @test(bind=True)
      def function_b():
          assert a = 43


3. Running Tests
================

**Development**

Running tests during development can be done by just running or importing your module:

.. code:: bash

  $ python <mymodule.py>
  $ python -c "import mymodule"

When you only want to develop a submodule, just ``cd`` down into that directory and do the same. Only the tests of that submodule (and everything in imports) will be tested.

The methode above just prints crude messages and has no way to use options. For that use the main that comes with autotest:

.. code:: bash

  $ autotest --help
  Usage: autotest [options] module

  Options:
    -h, --help            show this help message and exit
    -f FILTER, --filter=FILTER
                        only run tests whose qualified name contains FILTER
    -t , --threshold=THRESHOLD
                        only run tests whose level is >= THRESHOLD

For example to run your tests but not the imported ones from other packages:

.. code:: bash

  $ autotest --filter mymodule mymodule


**Production**

During production, all tests are automatically run during startup when all needed modules are imported. If an the application configures a specific root, for example by calling ``logging.basicConfig()``, the tests will automatically log there. Alternatively, you can setup a separate ``Logger`` for running tests. See Core API.

You can als filter tests or run tests for a specific level only. Or suppress them all. See the source code of ``__main__.py`` for ideas.




4. Misc
=======

**TODO**

- also implement timeout for synchrounous code

