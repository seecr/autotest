
""" Defines the Runner 

    NB: a stand alone Runner (self_test) is created as to not interfere with
        the system wide default runner, which may not exists as long as we
        are bootstrapping.

    NB: Runner is tested using Runner itself. Runner is bootstrapped in steps,i
        starting with a simple Runner of which the capabilities are gradually
        extended.

"""

import inspect
import traceback
import types
import pathlib
import tempfile
import sys
import operator
import contextlib
import functools
import asyncio
import os
import difflib
import pprint
import unittest.mock as mock


from utils import is_main_process


__all__ = ['test', 'Runner']


class Report:

    def __init__(self):
        self.total = 0
        self.ran = 0
        self.reported = 0

    def __call__(self, test, *app_args, **app_kwds):
        AUTOTEST_INTERNAL = 1
        self.start(test)
        try:
            return test(*app_args, **app_kwds)
        finally:
            self.done(self)

    def start(self, test):
        if test.runner._opts.get('report'):
            print(test, flush=True)
            self.reported += 1

    def done(self, test):
        self.ran += 1

    def found(self, context):
        self.total += 1

    def report(self):
        pass



class Runner:
    """ Main tool for running tests across modules and programs. """

    def __init__(self, reporter=None, fixtures=None, gathered=None,
            # built-in defaults for options
            skip    = not is_main_process or __name__ == '__mp_main__',
            keep    = False,      # Ditch test functions after running, or keep them in their namespace.
            report  = True,       # Do reporting when starting a test.
            gather  = False,      # Gather tests, use gathered() to get them
            timeout = 2,          # asyncio task timeout
            coverage= False,      # invoke trace module
            debug   = True,       # use debug for asyncio.run
            diff    = None,       # set a function for printing diffs on failures
            ):
        self.report = reporter if reporter else Report()
        self.fixtures = fixtures.copy() if fixtures else {}
        self.gathered = gathered if gathered else []
        self._opts = dict(skip=skip, keep=keep, report=report, gather=gather,
                timeout=timeout, coverage=coverage, debug=debug, diff=diff)

    def clone(self, opts, gathered=None):
        return Runner(reporter=self.report, fixtures=self.fixtures,
                gathered=self.gathered if gathered is None else gathered, **(self._opts | opts))

    def run(self, test_func, *app_args, **app_kwds):
        """ Binds f to stack vars and fixtures and runs it immediately. """
        AUTOTEST_INTERNAL = 1
        self.report.found(self)
        bind_func = bind_1_frame_back(test_func)
        skip = self._opts.get('skip')
        if inspect.isfunction(skip) and not skip(test_func) or not skip:
            self.report(WithFixtures(self, bind_func), *app_args, **app_kwds)
        if self._opts.get('gather'):
            self.gathered.append(bind_func)
        return bind_func if self._opts.get('keep') else None


    def __call__(self, *fs, **opts):
        """Decorator to define, run and report a test, with one-time options when given. """
        AUTOTEST_INTERNAL = 1
        if opts and not fs:
            return self.clone(opts).run       # @test(opt=value,...)
        elif len(fs) == 1:
            with self.opts(**opts):
                f, = fs
                if isinstance(f, tuple):
                    return self.run(*f)
                return self.run(*fs)           # @test or test(f, **opts)
        elif len(fs) > 1:
            with self.opts(**opts):
                for f in fs:                      # test(*suite)
                    self.run(f)
        else:
            return self.opts()                    # with test():


    @contextlib.contextmanager
    def opts(self, gathered=None, **opts):
        """ create sub tester with opts """
        yield self.clone(opts, gathered=gathered)


    def gather(self, gather=True, **opts):
        """ create sub tester which gathers tests with opts """
        return self.opts(gathered=[], gather=gather, **opts)


    def fail(self, *args, **kwds):
        if not self._opts.get('skip', False):
            args += (kwds,) if kwds else ()
            raise AssertionError(*args)


    def fixture(self, func):
        """Decorator for fixtures a la pytest. A fixture is a generator yielding exactly 1 value.
           That value is used as argument to functions declaring the fixture in their args. """
        assert inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func), func
        bound_f = bind_1_frame_back(func)
        self.fixtures[func.__name__] = bound_f
        return bound_f


    def diff(self, a, b):
        """ When str called, produces diff of textual representation of a and b. """
        return lazy_str(
            lambda:
                '\n' + '\n'.join(
                    difflib.ndiff(
                        pprint.pformat(a).splitlines(),
                        pprint.pformat(b).splitlines())))


    @staticmethod
    def diff2(a, b):
        """ experimental, own formatting more suitable for difflib """
        import autotest.prrint as prrint # late import so prrint can use autotest itself
        return '\n' + '\n'.join(
                    difflib.ndiff(
                        prrint.format(a).splitlines(),
                        prrint.format(b).splitlines()))

    def prrint(self, a):
        import autotest.prrint as prrint # late import so prrint can use autotest itself
        prrint.prrint(a)

    def isinstance(self, value, types):
        types_str = f"{[t.__name__ for t in types]}" if isinstance(types, tuple) else f"{types.__name__!r}"
        self.truth(isinstance(value, types), msg=f"{type(value).__name__!r} is not an instance of {types_str}".format)

    class Operator:
        """ Returns an function that:
            calls an operator from builtin module operator, eg:
              - test.eq(1,1), test.ne(1,2), etc.
            or calls a method of the first arg, passing it the rest of the args:
              - test.startswith("aap", "a")
            and calls given function (default bool()) on the result
        """
        def __init__(self, test=bool, diff=None):
            self.test = test
            self.diff = diff
        def __getattr__(self, opname):
            def call_operator(*args, diff=None, msg=None):
                diff = diff or msg
                AUTOTEST_INTERNAL = 1
                try:
                    op = getattr(operator, opname)
                    actual_args = args
                except AttributeError:
                    op = getattr(args[0], opname)
                    actual_args = args[1:]
                if not self.test(op(*actual_args)):
                    if diff := diff or self.diff:
                        raise AssertionError(diff(*args))
                    else:
                        raise AssertionError(op.__name__, *args)
                return True
            return call_operator

    @property
    def comp(self):
        return self.Operator(test=operator.not_, diff=self._opts.get('diff'))

    complement = comp


    def __getattr__(self, name):
        """ - test.<fixture>: returns a fixture
            - otherwise delegates to Operator() """
        if name in self.fixtures:
            fx = self.fixtures[name]
            fx_bound = WithFixtures(self, fx)
            if inspect.isgeneratorfunction(fx):
                return ArgsCollectingContextManager(fx_bound)
            if inspect.isasyncgenfunction(fx):
                return ArgsCollectingAsyncContextManager(fx_bound)
            raise ValueError(f"not an (async) generator: {fx}")
        return getattr(self.Operator(diff=self._opts.get('diff')), name)


self_test = Runner()


def iterate(f, v):
    #if isinstance(f, str):
    #    f = operator.attrgetter(f)
    while v:
        yield v
        v = f(v)


def filter_traceback(tb):
    """ Bootstrapping, placeholder, overwritten later """
    pass


""" Bootstrapping, placeholder, overwritten later """
class WithFixtures:
    def __init__(self, runner, f):
        self.runner = runner
        self.func = f
    def __call__(self, *a, **k):
        return self.func(*a, **k)
    def __str__(self):
        return f"{self.func.__module__}  \33[1m{self.func.__name__}\033[0m  "



""" Bootstrapping, placeholder, overwritten later """
ArgsCollectingContextManager = contextlib.contextmanager


""" Bootstrapping, placeholder, overwritten later """
def bind_1_frame_back(_func_):
    return _func_


class lazy_str:
    def __init__(self, f):
        self._f = f
    def __str__(self):
        return self._f()


trace = []


from numbers import Number
class any_number:
    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
    def __eq__(self, rhs):
        return isinstance(rhs, Number) and self._lo <= rhs <= self._hi


@self_test
def example_test():
    assert any_number(1,10) == 1
    assert any_number(1,10) == 2
    assert any_number(1,10) == 5
    assert any_number(1,10) == 9
    assert any_number(1,10) == 10
    assert not any_number(9,15) == 1
    assert not any_number(9,15) == 8
    assert not any_number(9,15) == 16
    assert not any_number(9,15) == 33
    assert not any_number(9,15) == None
    assert not any_number(9,15) == "A"
    assert not any_number(9,15) == object


def _(): yield
async def _a(): yield
ContextManagerType = type(contextlib.contextmanager(_)())
AsyncContextManagerType = type(contextlib.asynccontextmanager(_a)())
del _, _a


class ArgsCollector:
    """ collects args before calling a function: ArgsCollector(f, 1)(2)(3)() calls f(1,2,3) """
    def __init__(self, f, *args, **kwds):
        self.func = f
        self.args = args
        self.kwds = kwds
    def __call__(self, *args, **kwds):
        if not args and not kwds:
            return self.func(*self.args, **self.kwds)
        self.args += args
        self.kwds.update(kwds)
        return self


class ArgsCollectingContextManager(ArgsCollector, ContextManagerType):
    """ Context manager that accepts additional args everytime it is called.
        NB: Implementation closely tied to contexlib.py (self.gen)"""
    def __enter__(self):
        self.gen = self()
        return super().__enter__()


class ArgsCollectingAsyncContextManager(ArgsCollector, AsyncContextManagerType):
    async def __aenter__(self):
        self.gen = self()
        return await super().__aenter__()
    @property
    def __enter__(self):
        raise Exception(f"Use 'async with' for {self.func}.")


@self_test
def extra_args_supplying_contextmanager():
    def f(a, b, c, *, d, e, f):
        yield a, b, c, d, e, f
    c = ArgsCollectingContextManager(f, 'a', d='d')
    assert isinstance(c, contextlib.AbstractContextManager)
    c('b', e='e')
    with c('c', f='f') as v:
        assert ('a', 'b', 'c', 'd', 'e', 'f') == v, v


# using import.import_module in asyncio somehow gives us the frozen tracebacks (which were
# removed in 2012, but yet showing up again in this case. Let's get rid of them.
def asyncio_filtering_exception_handler(loop, context):
    if 'source_traceback' in context:
        context['source_traceback'] = [t for t in context['source_traceback'] if '<frozen ' not in t.filename]
    return loop.default_exception_handler(context)


# get the type of Traceback objects
try: raise Exception
except Exception as e:
    Traceback = type(e.__traceback__)


def frame_to_traceback(tb_frame, tb_next=None):
    tb = Traceback(tb_next, tb_frame, tb_frame.f_lasti, tb_frame.f_lineno)
    return create_traceback(tb_frame.f_back, tb) if tb_frame.f_back else tb


""" bootstrapping: test and instal fixture support """


def ensure_async_generator_func(f):
    if inspect.isasyncgenfunction(f):
        return f
    if inspect.isgeneratorfunction(f):
        @functools.wraps(f)
        async def wrap(*a, **k):
            for v in f(*a, **k):
                yield v
        return wrap
    assert False, f"{f} cannot be a async generator."


# redefine the placeholder with support for fixtures
class WithFixtures:
    """ Activates all fixtures recursively, then runs the test function. """

    def __init__(self, runner, func):
        self.runner = runner
        self.func = func


    def __str__(self):
        return f"{self.func.__module__}  \33[1m{self.func.__name__}\033[0m  "


    def __call__(self, *args, **kwds):
        AUTOTEST_INTERNAL = 1
        if inspect.iscoroutinefunction(self.func):
            # TODO detect already running loop and use it
            return asyncio.run(self.async_run_with_fixtures(*args, **kwds), debug=self.runner._opts.get('debug'))
        with contextlib.ExitStack() as contextmgrstack:
            # we could use the timeout here too, with a signal handler TODO
            return self.run_recursively(self.func, contextmgrstack, *args, **kwds)


    def get_fixtures_except_for(self, f, except_for):
        """ Finds all fixtures, skips those in except_for (overridden fixtures) """
        AUTOTEST_INTERNAL = 1
        def args(p):
            a = () if p.annotation == inspect.Parameter.empty else p.annotation
            assert p.default == inspect.Parameter.empty, f"Use {p.name}:{p.default} instead of {p.name}={p.default}"
            return a if isinstance(a, tuple) else (a,)
        return [(self.runner.fixtures[name], args(p)) for name, p in inspect.signature(f).parameters.items()
                if name in self.runner.fixtures and name not in except_for]


    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    def run_recursively(self, f, contextmgrstack, *args, **kwds):
        AUTOTEST_INTERNAL = 1
        fixture_values = []
        for fx, fx_args in self.get_fixtures_except_for(f, kwds.keys()):
            assert inspect.isgeneratorfunction(fx), f"function '{self.func.__name__}' cannot have async fixture '{fx.__name__}'."
            ctxmgr_func = contextlib.contextmanager(fx)
            context_mgr = self.run_recursively(ctxmgr_func, contextmgrstack, *fx_args)
            fixture_values.append(contextmgrstack.enter_context(context_mgr))
        return f(*fixture_values, *args, **kwds)

    # compare ^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v and test

    async def async_run_recursively(self, f, contextmgrstack, *args, **kwds):
        AUTOTEST_INTERNAL = 1
        fixture_values = []
        for fx, fx_args in self.get_fixtures_except_for(f, kwds.keys()):
            ctxmgr_func = contextlib.asynccontextmanager(ensure_async_generator_func(fx))
            context_mgr = await self.async_run_recursively(ctxmgr_func, contextmgrstack, *fx_args)
            fixture_values.append(await contextmgrstack.enter_async_context(context_mgr))
        return f(*fixture_values, *args, **kwds)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


    async def async_run_with_fixtures(self, *args, **kwargs):
        AUTOTEST_INTERNAL = 1
        timeout = self.runner._opts.get('timeout')
        async with contextlib.AsyncExitStack() as contextmgrstack:
            result = await self.async_run_recursively(self.func, contextmgrstack, *args, **kwargs)
            assert inspect.iscoroutine(result)
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(asyncio_filtering_exception_handler)
            done, pending = await asyncio.wait([result], timeout=timeout)
            for d in done:
                await d
            if pending:
                n = len(pending)
                p = pending.pop() # one problem at a time
                s = p.get_stack() # "only one stack frame is returned for a suspended coroutine"
                tb1 = frame_to_traceback(s[-1])
                raise asyncio.TimeoutError(f"Hanging task (1 of {n})").with_traceback(tb1)


@self_test.fixture
def fx_a():
    trace.append("A start")
    yield 67
    trace.append("A end")


@self_test.fixture
def fx_b(fx_a):
    trace.append("B start")
    yield 74
    trace.append("B end")


def test_a(fx_a, fx_b, a, b=10):
    assert 67 == fx_a
    assert 74 == fx_b
    assert 9 == a
    assert 11 == b
    trace.append("test_a done")

@self_test
def test_run_with_fixtures():
    WithFixtures(self_test, test_a)(9, b=11)
    assert ['A start', 'A start', 'B start', 'test_a done', 'B end', 'A end', 'A end'] == trace, trace


fixture_lifetime = []

@self_test.fixture
def fixture_A():
    fixture_lifetime.append('A-live')
    yield 42
    fixture_lifetime.append('A-close')


@self_test
def with_one_fixture(fixture_A):
    assert 42 == fixture_A, fixture_A


@self_test.fixture
def fixture_B(fixture_A):
    fixture_lifetime.append('B-live')
    yield fixture_A * 2
    fixture_lifetime.append('B-close')


@self_test.fixture
def fixture_C(fixture_B):
    fixture_lifetime.append('C-live')
    yield fixture_B * 3
    fixture_lifetime.append('C-close')


@self_test
def nested_fixture(fixture_B):
    assert 84 == fixture_B, fixture_B


class lifetime:

    del fixture_lifetime[:]

    @self_test
    def more_nested_fixtures(fixture_C):
        assert 252 == fixture_C, fixture_C
        assert ['A-live', 'B-live', 'C-live'] == fixture_lifetime, fixture_lifetime

    @self_test
    def fixtures_livetime():
        assert ['A-live', 'B-live', 'C-live', 'C-close', 'B-close', 'A-close'] == fixture_lifetime, fixture_lifetime

class async_tests:
    done = [False]

    #@self_test
    async def this_is_an_async_test():
        async_tests.done[0] = True

    #try:
    #    asynio.get_running_loop()
    #except RuntimeError:
    #    loop = asynio.new_event_loop()

    #self_test.truth(all(done))

class async_fixtures:

    @self_test.fixture
    async def my_async_fixture():
        await asyncio.sleep(0)
        yield 'async-42'


    @self_test
    async def async_fixture_with_async_with():
        async with self_test.my_async_fixture as f:
            assert 'async-42' == f
        try:
            with self_test.my_async_fixture:
                assert False
        except Exception as e:
            self_test.startswith(str(e), "Use 'async with' for ")


    @self_test
    async def async_fixture_as_arg(my_async_fixture):
        self_test.eq('async-42', my_async_fixture)


    try:
        @self_test
        def only_async_funcs_can_have_async_fixtures(my_async_fixture):
            assert False, "not possible"
    except AssertionError as e:
        self_test.eq(f"function 'only_async_funcs_can_have_async_fixtures' cannot have async fixture 'my_async_fixture'.", str(e))


    @self_test.fixture
    async def with_nested_async_fixture(my_async_fixture):
        yield f">>>{my_async_fixture}<<<"


    @self_test
    async def async_test_with_nested_async_fixtures(with_nested_async_fixture):
        self_test.eq(">>>async-42<<<", with_nested_async_fixture)


    @self_test
    async def mix_async_and_sync_fixtures(fixture_C, with_nested_async_fixture):
        self_test.eq(">>>async-42<<<", with_nested_async_fixture)
        self_test.eq(252, fixture_C)


def tmp_path(name=None):
    with tempfile.TemporaryDirectory() as p:
        p = pathlib.Path(p)
        if name:
            yield p/name
        else:
            yield p
self_test.fixture(tmp_path)


class tmp_files:

    path = None

    @self_test
    def temp_sync(tmp_path):
        assert tmp_path.exists()

    @self_test
    def temp_file_removal(tmp_path):
        global path
        path = tmp_path / 'aap'
        path.write_text("hello")

    @self_test
    def temp_file_gone():
        assert not path.exists()

    @self_test
    async def temp_async(tmp_path):
        assert tmp_path.exists()

    @self_test
    def temp_dir_with_file(tmp_path:'aap'):
        assert str(tmp_path).endswith('/aap')
        tmp_path.write_text('hi monkey')
        assert tmp_path.exists()


@self_test
def new_assert():
    assert self_test.eq(1, 1)
    assert self_test.lt(1, 2) # etc

    try:
        self_test.eq(1, 2)
        self_test.fail(msg="too bad")
    except AssertionError as e:
        assert ('eq', 1, 2) == e.args, e.args

    try:
        self_test.ne(1, 1)
        self_test.fail()
    except AssertionError as e:
        assert "('ne', 1, 1)" == str(e), str(e)


@self_test
def test_fail():
    try:
        self_test.fail("plz fail")
        raise AssertionError('FAILED to fail')
    except AssertionError as e:
        assert "plz fail" == str(e), e

    try:
        self_test.fail("plz fail", info="more")
        raise AssertionError('FAILED to fail')
    except AssertionError as e:
        assert "('plz fail', {'info': 'more'})" == str(e), e


@self_test
def not_operator():
    self_test.comp.contains("abc", "d")
    try:
        self_test.comp.contains("abc", "b")
        self_test.fail()
    except AssertionError as e:
        assert "('contains', 'abc', 'b')" == str(e), e


@self_test
def call_method_as_operator():
    self_test.endswith("aap", "ap")


# self_test.<op> is really just assert++ and does not need @test to run 'in'
self_test.eq(1, 1)

try:
    self_test.lt(1, 1)
except AssertionError as e:
    assert "('lt', 1, 1)" == str(e), str(e)


@self_test
def test_testop_has_args():
    try:
        @self_test(report=False)
        def nested_test_with_testop():
            x = 42
            y = 67
            self_test.eq(x, y)
            return True
    except AssertionError as e:
        assert hasattr(e, 'args')
        assert 3 == len(e.args)
        self_test.eq("('eq', 42, 67)", str(e))


with self_test.opts(report=False) as tst:
    @tst
    def nested_defaults():
        dflts0 = tst._opts
        assert not dflts0['skip']
        assert not dflts0['report']
        with tst.opts(skip=True, report=True) as tstk:
            dflts1 = tstk._opts
            assert dflts1['skip']
            assert dflts1['report']
            @tstk
            def fails_but_is_skipped():
                tstk.eq(1, 2)
            try:
                with tstk.opts(skip=False, report=False) as TeSt:
                    dflts2 = TeSt._opts
                    assert not dflts2['skip']
                    assert not dflts2['report']
                    @TeSt
                    def fails_but_is_not_reported():
                        TeSt.gt(1, 2)
                    tst.fail()
                assert tstk._opts == dflts1
            except AssertionError as e:
                tstk.eq("('gt', 1, 2)", str(e))
        assert tst._opts == dflts0

    @tst
    def shorthand_for_new_context_without_opts():
        """ Use this for defining new/tmp fixtures. """
        ctx0 = tst
        with tst() as t: # == self_test.opts()
            assert ctx0 != t
            assert ctx0.fixtures == t.fixtures
            assert ctx0.report == t.report
            assert ctx0._opts == t._opts
            @t.fixture
            def new_one():
                yield 42
            assert ctx0.fixtures != t.fixtures
            assert "new_one" in t.fixtures
            assert "new_one" not in ctx0.fixtures


@self_test
def override_fixtures_in_new_context():
    with self_test() as t:
        assert self_test != t
        @t.fixture
        def temporary_fixture():
            yield "tmp one"
        with t.temporary_fixture as tf1:
            assert "tmp one" == tf1
        with self_test():
            @self_test.fixture
            def temporary_fixture():
                yield "tmp two"
            with self_test.temporary_fixture as tf2:
                assert "tmp two" == tf2
        with t.temporary_fixture as tf1:
            assert "tmp one" == tf1
    try:
        self_test.temporary_fixture
    except AttributeError:
        pass


@self_test.fixture
def area(r, d=1):
    import math
    yield round(math.pi * r * r, d)

@self_test
def fixtures_with_1_arg(area:3):
    self_test.eq(28.3, area)
@self_test
def fixtures_with_2_args(area:(3,0)):
    self_test.eq(28.0, area)
@self_test
async def fixtures_with_2_args_async(area:(3,2)):
    self_test.eq(28.27, area)

@self_test.fixture
def answer(): yield 42
@self_test.fixture
def combine(a, area:2, answer): yield a * area * answer
@self_test
def fixtures_with_combined_args(combine:3):
    self_test.eq(1587.6, combine)


#@self_test  # test no longer modifies itself but creates a temporary runner when opts are given
def combine_test_with_options():
    trace = []
    @self_test(keep=True, my_opt=42)
    def f0():
        trace.append(self_test._opts.get('my_opt'))
    def f1(): # ordinary function; the only difference, here(!) is binding
        trace.append(self_test._opts.get('my_opt'))
    self_test(f0)
    self_test(f0, my_opt=76)
    self_test(f0, f1, my_opt=93)
    assert [None, None, 76, 93, 93] == trace, trace
    # TODO make opts available in test when specificed in @self_test
    # (@self_test(**opts) makes a one time anonymous context)
    # maybe an optional argument 'context' which tests can declare??



@self_test
def gather_tests():
    with self_test.gather() as suite:

        @suite.fixture
        def a_fixture():
            yield 16

        @suite(skip=True)
        def t0(a_fixture): return f'this is t0, with {a_fixture}'

        with suite.gather() as subsuite0:
            @subsuite0
            def sub0(): return 78

        with suite.gather(report=True) as subsuite1:
            @subsuite1
            def sub1(): return 56

        @suite
        def t1(a_fixture): return f'this is t1, with {a_fixture}'

        @suite(gather=False)
        def t2(): pass

    self_test.eq(1, len(subsuite0.gathered))
    self_test.eq(78, subsuite0.gathered[0]())
    self_test.eq(1, len(subsuite1.gathered))
    self_test.eq(56, subsuite1.gathered[0]())
    # self_test.eq(2, len(suite.gathered))  #TODO subsuite not part of super suite temporarily
    # NB: running test function without context (reporting etc):
    # self_test.eq('this is t0, with 42', suite.gathered[0](a_fixture=42))   # IDEM
    """ deprecated; we could reintroduce 'bind' option to allow this
    self_test.eq('this is t1, with 16', suite[1]())
    """
    self_test.eq(1, len(subsuite0.gathered))
    self_test.eq(1, len(subsuite1.gathered))
    # self_test.eq(2, len(suite.gathered))  # IDEM

#@self_test    TODO properly (re) implement suites
def run_suite():
    trace = []
    with self_test.gather(keep=True) as suite:
        @self_test.fixture
        def a():
            yield 42
        @self_test
        def a1(a):
            assert 0 == a % 42
            trace.append(f"a1:{a}")
        @self_test
        def a2(a):
            assert 0 == a % 42
            trace.append(f"a2:{a}")
    # run in new context
    with self_test():
        @self_test.fixture
        def a():
            yield 84
        # different ways to run a test function:
        a1(126)            # NB: no context at all, not reported
        self_test(a1)
        self_test(suite[0])
        self_test(*suite)
        self_test(a1, a2)
        # just to be sure
        self_test.eq(["a1:42", "a2:42", "a1:126", "a1:84", "a1:84", "a1:84", "a2:84", "a1:84", "a2:84"], trace)

@self_test
def test_calls_other_test():
    @self_test(keep=True, report=False)
    def test_a():
        assert 1 == 1
        return True
    @self_test(report=False)
    def test_b():
        assert test_a()


@self_test
def test_calls_other_test_with_fixture():
    @self_test(keep=True, report=False)
    def test_a(fixture_A):
        assert 42 == fixture_A
        return True
    @self_test(report=False)
    def test_b():
        assert test_a(fixture_A=42)


@self_test
def test_calls_other_test_with_fixture_and_more_args():
    @self_test(keep=True, report=False, skip=True)
    def test_a(fixture_A, value):
        assert 42 == fixture_A
        assert 16 == value
        return True
    @self_test(report=False)
    def test_b(fixture_A):
        assert test_a(fixture_A=42, value=16)


# IDEA/TODO
# I make this mistake a lot:
# Instead of @test(option=Value)
# I write def test(open=Value)
# I think supporting both is a good idea


@self_test
def call_test_with_complete_context():
    @self_test(keep=True)
    def a_test():
        assert True
    assert a_test
    self_test(a_test)


def capture(name):
    """ captures output from child processes as well """
    org_stream = getattr(sys, name)
    org_fd = org_stream.fileno()
    org_fd_backup = os.dup(org_fd)
    replacement = tempfile.TemporaryFile(mode="w+t", buffering=1)
    os.dup2(replacement.fileno(), org_fd)
    setattr(sys, name, replacement)
    def getvalue():
        replacement.flush()
        replacement.seek(0)
        return replacement.read()
    replacement.getvalue = getvalue
    try:
        yield replacement
    finally:
        os.dup2(org_fd_backup, org_fd)
        setattr(sys, name, org_stream)


# do not use @self_test.fixture as we need to install this twice
def stdout():
    yield from capture('stdout')
self_test.fixture(stdout)


# do not use @test.fixture as we need to install this twice
def stderr():
    yield from capture('stderr')
self_test.fixture(stderr)


@self_test
async def async_function():
    await asyncio.sleep(0)
    assert True


# define, test and install default fixture
# do not use @test.fixture as we need to install this twice
def raises(exception=Exception, message=None):
    AUTOTEST_INTERNAL = 1
    try:
        yield
    except exception as e:
        if message and message != str(e):
            raise AssertionError(f"should raise {exception.__name__} with message '{message}'") from e
    except BaseException as e:
        raise AssertionError(f"should raise {exception.__name__} but raised {type(e).__name__}").with_traceback(e.__traceback__) from e
    else:
        e = AssertionError(f"should raise {exception.__name__}")
        e.__suppress_context__ = True
        raise e
self_test.fixture(raises)


@self_test
def assert_raises():
    with self_test.raises:
        raise Exception
    try:
        with self_test.raises:
            pass
    except AssertionError as e:
        assert 'should raise Exception' == str(e), e


@self_test
def assert_raises_specific_exception():
    with self_test.raises(KeyError):
        raise KeyError
    try:
        with self_test.raises(KeyError):
            raise RuntimeError('oops')
    except AssertionError as e:
        assert 'should raise KeyError but raised RuntimeError' == str(e), str(e)
    try:
        with self_test.raises(KeyError):
            pass
    except AssertionError as e:
        assert 'should raise KeyError' == str(e), e


@self_test
def assert_raises_specific_message():
    with self_test.raises(RuntimeError, "hey man!"):
        raise RuntimeError("hey man!")
    try:
        with self_test.raises(RuntimeError, "hey woman!"):
            raise RuntimeError("hey man!")
    except AssertionError as e:
        assert "should raise RuntimeError with message 'hey woman!'" == str(e)



def import_syntax_error():
    with self_test.tmp_path as p:
        try:
            sys.path.append(str(p))
            (p/'my_sub_module_syntax_error.py').write_text('syntax error')
            import my_sub_module_syntax_error
        finally:
            sys.path.remove(str(p))


def is_builtin(f):
    if m := inspect.getmodule(f.f_code):
        return m.__name__ in sys.builtin_module_names

def is_internal(frame):
    nm = frame.f_code.co_filename
    return 'AUTOTEST_INTERNAL' in frame.f_code.co_varnames or \
           '<frozen importlib' in nm or \
           is_builtin(frame)   # TODO testme

#@self_test
def guess_module():
    def f():
        pass
    self_test.eq('tester', inspect.getmodule(f).__name__)
    self_test.eq('tester', inspect.getmodule(f.__code__).__name__)


def bind_names(bindings, names, frame):
    """ find names in locals on the stack and binds them """
    if not frame or not names:
        return bindings
    if not is_internal(frame):
        f_locals = frame.f_locals # rather expensive
        rest = []
        for name in names:
            try:
                bindings[name] = f_locals[name]
            except KeyError:
                rest.append(name)
    else:
        rest = names
    return bind_names(bindings, rest, frame.f_back)


def bind_1_frame_back(func):
    """ Binds the unbound vars in func to values found on the stack """
    func2 = types.FunctionType(
               func.__code__,                      # code
               bind_names(                         # globals
                   func.__globals__.copy(),
                   func.__code__.co_names,
                   inspect.currentframe().f_back),
               func.__name__,                      # name
               func.__defaults__,                  # default arguments
               func.__closure__)                   # closure/free vars
    func2.__annotations__ = func.__annotations__   # annotations not in __init__
    return func2


M = 'module scope'
a = 'scope:global'
b = 10

assert 'scope:global' == a
assert 10 == b

class X:
    a = 'scope:X'
    x = 11

    @self_test(keep=True)
    def C(fixture_A):
        a = 'scope:C'
        c = 12
        assert 'scope:C' == a
        assert 10 == b
        assert 11 == x
        assert 12 == c
        assert 10 == abs(-10) # built-in
        assert 'module scope' == M
        assert 42 == fixture_A, fixture_A
        return fixture_A

    @self_test(keep=True)
    def D(fixture_A, fixture_B):
        a = 'scope:D'
        assert 'scope:D' == a
        assert 10 == b
        assert 11 == x
        assert callable(C), C
        assert 10 == abs(-10) # built-in
        assert 'module scope' == M
        assert 42 == fixture_A, fixture_A
        assert 84 == fixture_B, fixture_B
        return fixture_B

    class Y:
        a = 'scope:Y'
        b = None
        y = 13

        @self_test
        def h(fixture_C):
            assert 'scope:Y' == a
            assert None == b
            assert 11 == x
            assert 13 == y
            assert callable(C)
            assert callable(D)
            assert C != D
            assert 10 == abs(-10) # built-in
            assert 42 == C(fixture_A=42)
            assert 84 == D(fixture_A=42, fixture_B=84) # "fixtures"
            assert 'module scope' == M
            assert 252 == fixture_C, fixture_C

    v = 45
    @self_test.fixture
    def f_A():
        yield v

    @self_test
    def fixtures_can_also_see_attrs_from_classed_being_defined(f_A):
        assert v == f_A, f_A

    @self_test
    async def coroutines_can_also_see_attrs_from_classed_being_defined(f_A):
        assert v == f_A, f_A


@self_test
def access_closure_from_enclosing_def():
    a = 46              # accessing this in another function makes it a 'freevar', which needs a closure
    @self_test
    def access_a():
        assert 46 == a


f = 16

@self_test
def dont_confuse_app_vars_with_internal_vars():
    assert 16 == f


@self_test.fixture
def fixture_D(fixture_A, a = 10):
    yield a


@self_test
def use_fixtures_as_context():
    with self_test.fixture_A as a:
        assert 42 == a
    with self_test.fixture_B as b: # or pass parameter/fixture ourselves?
        assert 84 == b
    with self_test.fixture_C as c:
        assert 252 == c
    with self_test.stdout as s:
        print("hello!")
        assert "hello!\n" == s.getvalue()
    keep = []
    with self_test.tmp_path as p:
        keep.append(p)
        (p / "f").write_text("contents")
    assert not keep[0].exists()
    # it is possible to supply additional args when used as context
    with self_test.fixture_D as d: # only fixture as arg
        assert 10 == d
    #with self_test.fixture_D() as d: # idem
    #    assert 10 == d
    with self_test.fixture_D(16) as d: # fixture arg + addtional arg
        assert 16 == d


# with self_test.<fixture> does not need @test to run 'in'
with self_test.tmp_path as p:
    assert p.exists()
assert not p.exists()


@self_test
def idea_for_dumb_diffs():
    # if you want a so called smart diff: the second arg of assert is meant for it.
    # Runner supplies a generic (lazy) diff between two pretty printed values
    a = [7, 1, 2, 8, 3, 4]
    b = [1, 2, 9, 3, 4, 6]
    d = self_test.diff(a, b)
    assert str != type(d)
    assert callable(d.__str__)
    assert str == type(str(d))

    try:
        assert a == b, self_test.diff(a,b)
    except AssertionError as e:
        assert """
- [7, 1, 2, 8, 3, 4]
?  ---      ^

+ [1, 2, 9, 3, 4, 6]
?        ^      +++
""" == str(e)


    # you can als pass a diff function to self_test.<op>().
    # diff should accept the args preceeding it
    # str is called on the result, so you can make it lazy
    try:
        self_test.eq("aap", "ape", msg=lambda x, y: f"{x} mydiff {y}")
    except AssertionError as e:
        assert "aap mydiff ape" == str(e)

    try:
        self_test.eq(a, b, msg=self_test.diff)
    except AssertionError as e:
        assert """
- [7, 1, 2, 8, 3, 4]
?  ---      ^

+ [1, 2, 9, 3, 4, 6]
?        ^      +++
""" == str(e), e


    # use of a function from elswhere
    a = set([7, 1, 2, 8, 3, 4])
    b = set([1, 2, 9, 3, 4, 6])
    try:
        assert a == b, set.symmetric_difference(a, b)
    except AssertionError as e:
        assert "{6, 7, 8, 9}" == str(e), e
    try:
        self_test.eq(a, b, msg=set.symmetric_difference)
    except AssertionError as e:
        assert "{6, 7, 8, 9}" == str(e), e

#@self_test # temp disable bc it cause system installed autotest import via diff2
def diff2_sorting_including_uncomparables():
    msg = """
  {
+   1,
-   <class 'dict'>:
?           ^^^   ^

+   <class 'str'>,
?           ^ +  ^

-     <class 'bool'>,
  }"""
    with self_test.raises(AssertionError, msg):
        self_test.eq({dict: bool}, {str, 1}, msg=self_test.diff2)


@self_test
def bind_test_functions_to_their_fixtures():

    @self_test.fixture
    def my_fix():
        yield 34

    @self_test(skip=True, keep=True, report=True)
    def override_fixture_binding_with_kwarg(my_fix):
        assert 34 != my_fix, "should be 34"
        assert 56 == my_fix
        return my_fix

    v = override_fixture_binding_with_kwarg(my_fix=56) # fixture binding overridden
    assert v == 56

    try:
        self_test(override_fixture_binding_with_kwarg)
    except AssertionError as e:
        self_test.eq("should be 34", str(e))


    @self_test(keep=True)
    def bound_fixture_1(my_fix):
        assert 34 == my_fix, my_fix
        return my_fix

    # general way to rerun a test with other fixtures:
    with self_test.opts() as t:
        @t.fixture
        def my_fix():
            yield 89
        try:
            t(bound_fixture_1)
        except AssertionError as e:
            assert "89" == str(e)
    with self_test.my_fix as x:
        assert 34 == x, x # old fixture back

    @self_test.fixture
    def my_fix(): # redefine fixture purposely to test time of binding
        yield 78

    """ deprecated
    @self_test(keep=True, skip=True) # skip, need more args
    def bound_fixture_2(my_fix, result, *, extra=None):
        assert 78 == my_fix
        return result, extra

    assert (56, "top") == bound_fixture_2(56, extra="top") # no need to pass args, already bound
    """

    class A:
        a = 34

        @self_test(keep=True) # skip, need more args
        def bound_fixture_acces_class_locals(my_fix):
            assert 78 == my_fix
            assert 34 == a
            return a
        assert 34 == bound_fixture_acces_class_locals(78)

    """ deprecated
    with self_test.opts(keep=True):

        @self_test
        def bound_by_default(my_fix):
            assert 78 == my_fix
            return my_fix

        assert 78 == bound_by_default()
    """

    trace = []

    @self_test.fixture
    def enumerate():
        trace.append('S')
        yield 1
        trace.append('E')

    @self_test(keep=True, skip=True)
    def rebind_on_every_call(enumerate):
        return True

    assert [] == trace
    self_test(rebind_on_every_call)
    assert ['S', 'E'] == trace
    self_test(rebind_on_every_call)
    assert ['S', 'E', 'S', 'E'] == trace


def filter_traceback(root):
    while root and is_internal(root.tb_frame):
        root = root.tb_next
    tb = root
    while tb and tb.tb_next:
        if is_internal(tb.tb_next.tb_frame):
           tb.tb_next = tb.tb_next.tb_next
        else:
           tb = tb.tb_next
    return root


@self_test
def trace_backfiltering():
    def eq(a, b):
        AUTOTEST_INTERNAL = 1
        assert a == b

    def B():
        eq(1, 2)

    def B_in_betwixt():
        AUTOTEST_INTERNAL = 1
        B()

    def A():
        B_in_betwixt()

    self_test.contains(B_in_betwixt.__code__.co_varnames, 'AUTOTEST_INTERNAL')

    def test_names(*should):
        _, _, tb = sys.exc_info()
        tb = filter_traceback(tb)
        names = tuple(tb.tb_frame.f_code.co_name for tb in iterate(lambda tb: tb.tb_next, tb))
        assert should == names, tuple(tb.tb_frame.f_code for tb in iterate(lambda tb: tb.tb_next, tb))

    try:
        eq(1, 2)
    except AssertionError:
        test_names('trace_backfiltering')

    try:
        B()
    except AssertionError:
        test_names('trace_backfiltering', 'B')

    try:
        B_in_betwixt()
    except AssertionError:
        test_names('trace_backfiltering', 'B')

    try:
        A()
    except AssertionError:
        test_names('trace_backfiltering', 'A', 'B')

    def C():
        A()

    def D():
        AUTOTEST_INTERNAL = 1
        C()

    def E():
        AUTOTEST_INTERNAL = 1
        D()

    try:
        C()
    except AssertionError:
        test_names('trace_backfiltering', 'C', 'A', 'B')

    try:
        D()
    except AssertionError:
        test_names('trace_backfiltering', 'C', 'A', 'B')

    try:
        E()
    except AssertionError:
        test_names('trace_backfiltering', 'C', 'A', 'B')

    try:
        @self_test
        def test_for_real_with_Runner_involved():
            self_test.eq(1, 2)
    except AssertionError:
        test_names('trace_backfiltering', 'test_for_real_with_Runner_involved')


@self_test
def test_isinstance():
    self_test.isinstance(1, int)
    self_test.isinstance(1.1, (int, float))
    with self_test.raises(AssertionError, "'int' is not an instance of 'str'"):
        self_test.isinstance(1, str)
    with self_test.raises(AssertionError, "'int' is not an instance of ['str', 'dict']"):
        self_test.isinstance(1, (str, dict))



@self_test.fixture
async def slow_callback_duration(s):
    asyncio.get_running_loop().slow_callback_duration = s
    yield


#@self_test
def probeer():
    assert False


def any(_): # name is included in diffs
    return True

class wildcard:
    def __init__(self, f=any):
        self.f = f
    def __eq__(self, x):
        return bool(self.f(x))
    def __call__(self, f):
        return wildcard(f)
    def __repr__(self):
        return self.f.__name__  + '(...)' if self.f else '*'

self_test.any = wildcard()


@self_test
def wildcard_matching():
    self_test.eq([self_test.any, 42], [16, 42])
    self_test.eq([self_test.any(lambda x: x in [1,2,3]), 78], [2, 78])
    self_test.ne([self_test.any(lambda x: x in [1,2,3]), 78], [4, 78])