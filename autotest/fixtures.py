
import tempfile
import pathlib
import inspect
import asyncio
import contextlib
import threading        # run nested async tests in thread
import asyncio          # support for async test and fixtures
import sys
import os

from .utils import asyncio_filtering_exception_handler, ensure_async_generator_func
from .utils import bind_1_frame_back # redefine placeholder
from .utils import ArgsCollectingContextManager, ArgsCollectingAsyncContextManager


__all__ = ['Fixtures', 'testing_fixtures', 'std_fixtures']


def get_fixture(runner, name):
    for value in runner._option('fixtures'):
        if name in value:
            return value[name]


def add_fixture(runner, func):
    runner._options.maps[0].setdefault('fixtures', {})[func.__name__] = func


# redefine the placeholder with support for fixtures
class _Fixtures:
    """ Activates all fixtures recursively, then runs the test function. """

    def __init__(self, runner, func):
        self.runner = runner
        self.func = func


    @classmethod
    def lookup(clz, tester, name):
        if name == 'fixture':
            def fixture(func):
                """Decorator for fixtures a la pytest. A fixture is a generator yielding exactly 1 value.
                   That value is used as argument to functions declaring the fixture in their args. """
                assert inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func), func
                bound_f = bind_1_frame_back(func)
                add_fixture(tester, bound_f)
                return bound_f
            return fixture
        if fx := get_fixture(tester, name): # tester._fixtures.get(name):
            fx_bound = _Fixtures(tester, fx)
            if inspect.isgeneratorfunction(fx):
                return ArgsCollectingContextManager(fx_bound)
            if inspect.isasyncgenfunction(fx):
                return ArgsCollectingAsyncContextManager(fx_bound)
            raise ValueError(f"not an (async) generator: {fx}")
        raise AttributeError


    def __call__(self, *args, **kwds):
        AUTOTEST_INTERNAL = 1
        if inspect.iscoroutinefunction(self.func):
            coro = self.async_run_with_fixtures(*args, **kwds)

            # dit deel moet naar Runner
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro, debug=self.runner._options.get('debug'))
            else:
                """ we get called by sync code (a decorator during def) which in turn
                    is called from async code. The only way to run this async test
                    is in a new event loop (in another thread) """
                t = threading.Thread(target=asyncio.run, args=(coro,),
                        kwargs={'debug': self.runner._options.get('debug')})
                t.start()
                t.join()
                return
            # tot hier of zo


        else:
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
        return [(get_fixture(self.runner, name), args(p)) for name, p in inspect.signature(f).parameters.items()
                if get_fixture(self.runner, name) and name not in except_for]


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
        timeout = self.runner._options.get('timeout')
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


fixtures_hook = _Fixtures


# some general purpose standard fixtures

def tmp_path(name=None):
    with tempfile.TemporaryDirectory() as p:
        p = pathlib.Path(p)
        if name:
            yield p/name
        else:
            yield p


def _capture(name):
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


def stdout():
    yield from _capture('stdout')


def stderr():
    yield from _capture('stderr')


async def slow_callback_duration(s):
    asyncio.get_running_loop().slow_callback_duration = s
    yield


def raises(exception=Exception, message=None):
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


std_fixtures = {fx.__name__: fx for fx in [tmp_path, stdout, stderr, slow_callback_duration, raises]}


def fixtures_test(self_test):
    from .binder import binder_hook
    self_test = self_test.getChild(hooks=(fixtures_hook, binder_hook), fixtures=std_fixtures)

    trace = []

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


    class tmp_files:

        path = [None]

        @self_test
        def temp_sync(tmp_path):
            assert tmp_path.exists()

        @self_test
        def temp_file_removal(tmp_path):
            path[0] = tmp_path / 'aap'
            path[0].write_text("hello")

        @self_test
        def temp_file_gone():
            assert not path[0].exists()

        @self_test
        async def temp_async(tmp_path):
            assert tmp_path.exists()

        @self_test
        def temp_dir_with_file(tmp_path:'aap'):
            assert str(tmp_path).endswith('/aap')
            tmp_path.write_text('hi monkey')
            assert tmp_path.exists()


    @self_test
    def test_fixtures():
        with self_test.stdout as s:
            print("hello!")
            assert "hello!\n" == s.getvalue()
        keep = []
        with self_test.tmp_path as p:
            keep.append(p)
            (p / "f").write_text("contents")
        assert not keep[0].exists()


    # with self_test.<fixture> does not need @test to run 'in'
    with self_test.tmp_path as p:
        assert p.exists()
    assert not p.exists()


    @self_test
    def override_fixtures_in_new_context():
        with self_test.child() as t:
            assert self_test != t
            @t.fixture
            def temporary_fixture():
                yield "tmp one"
            with t.temporary_fixture as tf1:
                assert "tmp one" == tf1, tf1
            with self_test.child():
                @self_test.fixture
                def temporary_fixture():
                    yield "tmp two"
                with self_test.temporary_fixture as tf2:
                    assert "tmp two" == tf2
            with t.temporary_fixture as tf1:
                assert "tmp one" == tf1, tf1
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


    @self_test
    def test_calls_other_test_with_fixture():
        @self_test(keep=True)
        def test_a(fixture_A):
            assert 42 == fixture_A
            return True
        @self_test
        def test_b():
            assert test_a(fixture_A=42)


    @self_test
    def test_calls_other_test_with_fixture_and_more_args():
        @self_test(keep=True, skip=True)
        def test_a(fixture_A, value):
            assert 42 == fixture_A
            assert 16 == value
            return True
        @self_test
        def test_b(fixture_A):
            assert test_a(fixture_A=42, value=16)


    M = 'module scope'
    a = 'scope:global'
    b = 10

    assert 'scope:global' == a
    assert 10 == b

    class X:
        a = 'scope:X'
        x = 11

        #@self_test(keep=True)
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

        #@self_test(keep=True)
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
            assert 'scope:Y' == a
            b = None
            y = 13

            #@self_test
            def h(fixture_C):
                global a, b # avoid binding to a and b in enclosing def
                assert 'scope:Y' == a, a
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
        # it is possible to supply additional args when used as context
        with self_test.fixture_D as d: # only fixture as arg
            assert 10 == d
        #with self_test.fixture_D() as d: # idem
        #    assert 10 == d
        with self_test.fixture_D(16) as d: # fixture arg + addtional arg
            assert 16 == d


    @self_test
    def bind_test_functions_to_their_fixtures():

        @self_test.fixture
        def my_fix():
            yield 34

        @self_test(skip=True, keep=True)
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
        with self_test.child() as t:
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
                #assert 34 == a
                #return a
            #assert 34 == bound_fixture_acces_class_locals(78)

        """ deprecated
        with self_test.child(keep=True):

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


    @self_test
    async def async_function():
        await asyncio.sleep(0)
        assert True


    class nested_async_tests:
        done = [False, False, False]

        @self_test
        async def this_is_an_async_test():
            done[0] = True
            """ A decorator is always called synchronously, so it can't call the async test
                because an event loop is already running. Solution is to start a new loop."""
            @self_test
            async def this_is_a_nested_async_test():
                done[1] = True
                @self_test
                async def this_is_a_doubly_nested_async_test():
                    done[2] = True

        self_test.all(done)


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

    @self_test
    def assert_raises_as_fixture(raises:KeyError):
        {}[0]



