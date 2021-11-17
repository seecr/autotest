#!/usr/bin/env python3

## begin license ##
#
# "Autotest" is a simplest thing that could possibly work test runner.
#
# Copyright (C) 2021 Seecr (Seek You Too B.V.) https://seecr.nl
#
# "Autotest" is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# "Autotest" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with "Autotest"; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
## end license ##



# TODO
# 2. if no args, find and run all modules from current dir, but recursively


if __name__ == '__main__':

    # experimental main; not tested

    """
    When run as main, it imports all Python modules in the current directory, or
    only the modules given as arguments (which may contain .py of /, which is ignored).

    Usage:
      $ autotest.py
      $ autotest.py a_module_dir
      $ autotest.py a_python_file.py
      $ autotest.py a_module_dir/a_python_file.py
      $ autotest.py any_path_basically_try_your_luck
    """

    import importlib
    import sys
    import pathlib
    import os
    import logging

    logging.getLogger("asyncio").setLevel(logging.DEBUG)

    cwd = pathlib.Path.cwd()
    sys.path.insert(0, str(cwd))
    if len(sys.argv) > 1:
        modules = map(lambda p: '.'.join(p.parent.parts + (p.stem,)),
                    map(pathlib.Path, sys.argv[1:]))
    else:
        modules = (pathlib.Path(p).stem for p in cwd.iterdir())
    modules = [m for m in modules if m not in ['__init__', '__pycache__'] and not m.startswith('.')]

    if 'autotest' not in modules:
        os.environ['AUTOTEST_report'] = 'False'
        print("silently", end=' ')

    print("importing \033[1mautotest\033[0m")
    from autotest import test # default test runner
    test.default(skip=lambda f: not any(f.__module__.startswith(m) for m in modules), report=True)
    if 'autotest' in modules:
        modules.remove('autotest')
    else:
        test.reset()

    for qname in modules:
        names = qname.split('.')
        for l in range(len(names)):
            name = '.'.join(names[:l+1])
            if name in sys.modules:
                print(f"already imported tests from \033[1m{name}\033[0m")
            elif importlib.util.find_spec(name):
                print(f"importing tests from \033[1m{name}\033[0m")
                importlib.import_module(name)

    print(f"Found \033[1m{test.found}\033[0m unique tests, ran \033[1m{test.ran}\033[0m, reported \033[1m{test.reported}\033[0m.")
    exit(0)


import inspect
import traceback
import types
import pathlib
import tempfile
import shutil
import sys
import pdb
import operator
import contextlib
import functools
import asyncio
import multiprocessing
import os
import difflib
import pprint
import unittest.mock as mock



__all__ = ['test', 'Runner']


is_main_process = multiprocessing.current_process().name == "MainProcess"


sys_defaults = {
    'skip'    : not is_main_process or __name__ == '__mp_main__',
    'keep'    : False,      # Ditch test functions after running, or keep them in their namespace.
    'report'  : True,       # Do reporting when starting a test.
    'bind'    : False,      # Kept functions are bound to fixtures
}

sys_defaults.update({k[len('AUTOTEST_'):]: eval(v) for k, v in os.environ.items() if k.startswith('AUTOTEST_')})


class Runner:

    def __init__(self, **opts):
        self.defaults = {**sys_defaults, **opts}
        self.fixtures = {}
        self.reset()


    def reset(self):
        self.found = 0
        self.ran = 0
        self.reported = 0


    def __call__(self, f=None, **opts):
        AUTOTEST_INTERNAL = 1
        """Decorator to define, run and report a test, with one-time options when given. """
        try:
            if opts:
                return functools.partial(self._bind, self.fixtures, **{**self.defaults, **opts})
            return self._bind(self.fixtures, f, **self.defaults)
        finally:
            self.found += 1


    def default(self, **kws):
        """ Set default options for next tests."""
        self.defaults.update(kws)


    def fail(self, *args, **kwds):
        if not self.defaults.get('skip', False):
            args += (kwds,) if kwds else ()
            raise AssertionError(*args)


    def fixture(self, func):
        """Decorator for fixtures a la pytest. A fixture is a generator yielding exactly 1 value.
           That value is used as argument to functions declaring the fixture in their args. """
        assert inspect.isgeneratorfunction(func)
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

    class Operator:
        def __init__(self, test=bool):
            self.test = test
        def __getattr__(self, opname):
            def call_operator(*args, msg=None):
                AUTOTEST_INTERNAL = 1
                try:
                    op = getattr(operator, opname)
                    actual_args = args
                except AttributeError:
                    op = getattr(args[0], opname)
                    actual_args = args[1:]
                if not self.test(op(*actual_args)):
                    if msg:
                        raise AssertionError(msg(*args))
                    else:
                        raise AssertionError(op.__name__, *args)
                return True
            return call_operator

    @property
    def comp(self):
        return self.Operator(test=operator.not_)

    complement = comp


    def __getattr__(self, name):
        """ run.eq(lhs, rhs) etc, any operator from module 'operator' really.
            or it returns a context manager if name denotes a fixture. """
        if name in self.fixtures:
            fx = self.fixtures[name]
            fx = functools.partial(run_with_fixtures, self.fixtures, fx)
            return CollectArgsContextManager(fx)
        return getattr(self.Operator(), name)


    def _bind(self, fixtures, f, *, bind, skip, keep, **opts):
        """ Binds f to stack vars and fixtures and runs it immediately. """
        AUTOTEST_INTERNAL = 1
        stack_bound_f = bind_1_frame_back(f)
        fixtures_bound_f = functools.partial(self._run, stack_bound_f, fixtures.copy(), **opts)
        if inspect.isfunction(skip) and not skip(f) or not skip:
            fixtures_bound_f()
        if not keep:
            return
        if bind:
            return fixtures_bound_f
        return functools.partial(self._run, stack_bound_f, (), **opts)


    def _run(self, f, fixtures, *app_args, report, **app_kwds):
        AUTOTEST_INTERNAL = 1
        """ Runs f with given fixtues and application args, reporting when necessary. """
        print_msg = print if report else lambda *_, **__: None
        print_msg(f"{f.__module__}  {f.__name__}  ", flush=True)
        if report:
            self.reported += 1
        try:
            return run_with_fixtures(fixtures, f, *app_args, **app_kwds)
        except SystemExit:
            raise
        except BaseException:
            et, ev, tb = sys.exc_info()
            recursive = self._run.__code__ in (f.f_code for f in iterate('f_back', tb.tb_frame.f_back))
            new_tb = filter_traceback(tb)
            if new_tb is not None: # avoid our own tests to have no traceback left
                tb = new_tb
            if ev.args == ():
                if recursive:
                    raise
                traceback.print_exception(et, ev, tb)
                post_mortem(tb, 'longlist')
                exit(-1)
            else:
                if report:
                    post_mortem(tb, 'longlist', 'exit')
                raise ev.with_traceback(tb)
        finally:
            self.ran += 1


test = Runner() # default Runner


def iterate(f, v):
    if isinstance(f, str):
        f = operator.attrgetter(f)
    while v:
        yield v
        v = f(v)


def filter_traceback(tb):
    """ Bootstrapping, placeholder, overwritten later """
    pass


def run_with_fixtures(_, f, *a, **k):
    """ Bootstrapping, placeholder, overwritten later """
    return f(*a, **k)


""" Bootstrapping, placeholder, overwritten later """
CollectArgsContextManager = contextlib.contextmanager


def bind_1_frame_back(_func_):
    """ Bootstrapping, placeholder, overwritten later """
    return _func_


def post_mortem(tb, *cmds):
    p = pdb.Pdb()
    p.rcLines.extend(cmds)
    p.reset()
    p.interaction(None, tb)


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


@test
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
ContextManagerType = type(contextlib.contextmanager(_)())
del _


class CollectArgsContextManager(ContextManagerType):
    """ Context manager that accepts additional args everytime it is called.
        NB: Implementation closely tied to contexlib.py """
    def __init__(self, f, *args, **kwds):
        self.func = f
        self.args = args
        self.kwds = kwds
    def __call__(self, *args, **kwds):
        self.args += args
        self.kwds.update(kwds)
        return self
    def __enter__(self):
        self.gen = self.func(*self.args, **self.kwds)
        return super().__enter__()


@test
def extra_args_supplying_contextmanager():
    def f(a, b, c, *, d, e, f):
        yield a, b, c, d, e, f
    c = CollectArgsContextManager(f, 'a', d='d')
    assert isinstance(c, contextlib.AbstractContextManager)
    c('b', e='e')
    with c('c', f='f') as v:
        assert ('a', 'b', 'c', 'd', 'e', 'f') == v, v


""" bootstrapping: test and instal fixture support """
# redefine the placeholder
def get_fixtures(fixtures, func, context, *args, **kwds):
    AUTOTEST_INTERNAL = 1
    return func(*(context.enter_context(
                    get_fixtures(fixtures, contextlib.contextmanager(fixtures[name]), context))
                  for name in inspect.signature(func).parameters if name in fixtures),
                *args, **kwds)


# redefine the placeholder
def run_with_fixtures(fixtures, f, *args, **kwds):
    AUTOTEST_INTERNAL = 1
    with contextlib.ExitStack() as context:
        result = get_fixtures(fixtures, f, context, *args, **kwds)
        # if we move the check below to get_fixtures, we would have async fixtures; not sure what that would bring tho.
        # TODO make fixtures async (yes, can be handy)
        if inspect.iscoroutine(result):
            return asyncio.run(result, debug=True)
        return result


@test.fixture
def fx_a():
    trace.append("A start")
    yield 67
    trace.append("A end")


@test.fixture
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

run_with_fixtures(test.fixtures, test_a, 9, b=11)

assert ['A start', 'A start', 'B start', 'test_a done', 'B end', 'A end', 'A end'] == trace, trace


fixture_lifetime = []

@test.fixture
def fixture_A():
    fixture_lifetime.append('A-live')
    yield 42
    fixture_lifetime.append('A-close')


@test
def with_one_fixture(fixture_A):
    assert 42 == fixture_A, fixture_A


@test.fixture
def fixture_B(fixture_A):
    fixture_lifetime.append('B-live')
    yield fixture_A * 2
    fixture_lifetime.append('B-close')


@test.fixture
def fixture_C(fixture_B):
    fixture_lifetime.append('C-live')
    yield fixture_B * 3
    fixture_lifetime.append('C-close')


@test
def nested_fixture(fixture_B):
    assert 84 == fixture_B, fixture_B


class lifetime:

    del fixture_lifetime[:]

    @test
    def more_nested_fixtures(fixture_C):
        assert 252 == fixture_C, fixture_C
        assert ['A-live', 'B-live', 'C-live'] == fixture_lifetime, fixture_lifetime

    @test
    def fixtures_livetime():
        assert ['A-live', 'B-live', 'C-live', 'C-close', 'B-close', 'A-close'] == fixture_lifetime, fixture_lifetime


# Async fixtures require quite a refactoring as everything becomes async, but we still want
# to support sync code.
#@test.fixture
#async def my_async_fixture():
#    await asyncio.sleep(0)
#    yield 'async-42'
#
#@test
#async def async_fixtures(my_async_fixture):
#    test.eq('async-42', my_async_fixture)


# define, test and install default fixture
# do not use @test.fixture as we need to install this twice
def tmp_path():
    with tempfile.TemporaryDirectory() as p:
        yield pathlib.Path(p)
test.fixture(tmp_path)


class tmp_files:

    path = None

    @test
    def temp_sync(tmp_path):
        assert tmp_path.exists()

    @test
    def temp_file_removal(tmp_path):
        global path
        path = tmp_path / 'aap'
        path.write_text("hello")

    @test
    def temp_file_gone():
        assert not path.exists()

    @test
    async def temp_async(tmp_path):
        assert tmp_path.exists()


@test
def new_assert():
    assert test.eq(1, 1)
    assert test.lt(1, 2) # etc

    try:
        test.eq(1, 2)
        test.fail(msg="too bad")
    except AssertionError as e:
        assert ('eq', 1, 2) == e.args, e.args

    try:
        test.ne(1, 1)
        test.fail()
    except AssertionError as e:
        assert "('ne', 1, 1)" == str(e), str(e)


@test
def test_fail():
    try:
        test.fail("plz fail")
    except AssertionError as e:
        assert "plz fail" == str(e), e

    try:
        test.fail("plz fail", info="more")
    except AssertionError as e:
        assert "('plz fail', {'info': 'more'})" == str(e), e


@test
def not_operator():
    test.comp.contains("abc", "d")
    try:
        test.comp.contains("abc", "b")
        test.fail()
    except AssertionError as e:
        assert "('contains', 'abc', 'b')" == str(e), e


@test
def call_method_as_operator():
    test.endswith("aap", "ap")


# test.<op> is really just assert++ and does not need @test to run 'in'
test.eq(1, 1)

try:
    test.lt(1, 1)
except AssertionError as e:
    assert "('lt', 1, 1)" == str(e), str(e)


@test
def test_testop_has_args():
    try:
        @test(report=False)
        def nested_test_with_testop():
            x = 42
            y = 67
            test.eq(x, y)
            return True
    except AssertionError as e:
        assert hasattr(e, 'args')
        assert 3 == len(e.args)
        test.eq("('eq', 42, 67)", str(e))


@test
def skip_until():
    test.default(skip=True)
    @test
    def fails():
        test.eq(1, 2)
    test.default(skip=False)
    try:
        @test(report=False)
        def fails():
            test.gt(1, 2)
        test.fail()
    except AssertionError as e:
        test.eq("('gt', 1, 2)", str(e))


@test
def test_calls_other_test():
    @test(keep=True, report=False)
    def test_a():
        assert 1 == 1
        return True
    @test(report=False)
    def test_b():
        assert test_a()


@test
def test_calls_other_test_with_fixture():
    @test(keep=True, report=False)
    def test_a(fixture_A):
        assert 42 == fixture_A
        return True
    @test(report=False)
    def test_b():
        assert test_a(42)


@test
def test_calls_other_test_with_fixture_and_more_args():
    @test(keep=True, report=False, skip=True)
    def test_a(fixture_A, value):
        assert 42 == fixture_A
        return True
    @test(report=False)
    def test_b(fixture_A):
        assert test_a(fixture_A, 42)


def capture(name):
    """ captures output from child processes as well """
    org_stream = getattr(sys, name)
    org_fd = org_stream.fileno()
    org_fd_backup = os.dup(org_fd)
    replacement = tempfile.TemporaryFile(mode="w+t", buffering=1)
    os.dup2(replacement.fileno(), org_fd)
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


# define, test and install default fixture
# do not use @test.fixture as we need to install this twice
def stdout():
    yield from capture('stdout')
test.fixture(stdout)


# define, test and install default fixture
# do not use @test.fixture as we need to install this twice
def stderr():
    yield from capture('stderr')
test.fixture(stderr)


if is_main_process:
    @test
    def stdout_capture():
        name = "Erik"
        msgs = []
        @test(report=False)
        def capture_all(stdout, stderr):
            print(f"Hello {name}!", file=sys.stdout)
            print(f"Bye {name}!", file=sys.stderr)
            msgs.extend([stdout.getvalue(), stderr.getvalue()])
        test.eq("Hello Erik!\n", msgs[0])
        test.eq("Bye Erik!\n", msgs[1])


if is_main_process:
    @test
    def capture_stdout_child_processes(stdout):
        def f():
            @test(report=False, skip=False)
            def in_child():
                print("hier ben ik")
                assert 1 == 1
        p = multiprocessing.Process(target=f) # NB: forks
        p.start()
        p.join(1)
        assert "hier ben ik\n" in stdout.getvalue()


@test
def reporting_tests(stdout):
    try:
        @test(report=False)
        def test_no_reporting_but_failure_raised():
            assert 1 != 1
        self.fail("should fail")
    except AssertionError as e:
        t, v, tb = sys.exc_info()
        tbi = traceback.extract_tb(tb)
        assert "test_no_reporting_but_failure_raised" == tbi[-1].name, tbi[-1].name
        assert "assert 1 != 1" == tbi[-1].line, repr(tbi[-1].line)
    m = stdout.getvalue()
    assert "" == m, m


    try:
        @test(report=True)
        def test_with_reporting_and_failure_raised():
            assert 1 != 1
        self.fail("should fail")
    except AssertionError:
        t, v, tb = sys.exc_info()
        tbi = traceback.extract_tb(tb)
        assert "test_with_reporting_and_failure_raised" == tbi[-1].name, tbi[-1].name
        assert "assert 1 != 1" == tbi[-1].line, repr(tbi[-1].line)
    m = stdout.getvalue()
    test.contains(m, "autotest  test_with_reporting_and_failure_raised")


@test
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
            e = AssertionError(f"should raise {exception.__name__} with message '{message}'")
            e.__suppress_context__ = True
            raise e
    except BaseException as e:
        e = AssertionError(f"should raise {exception.__name__} but raised {type(e).__name__}")
        e.__suppress_context__ = True
        raise e
    else:
        e = AssertionError(f"should raise {exception.__name__}")
        e.__suppress_context__ = True
        raise e
test.fixture(raises)


@test
def assert_raises():
    with test.raises:
        raise Exception
    try:
        with test.raises:
            pass
    except AssertionError as e:
        assert 'should raise Exception' == str(e), e


@test
def assert_raises_specific_exception():
    with test.raises(KeyError):
        raise KeyError
    try:
        with test.raises(KeyError):
            raise RuntimeError('oops')
    except AssertionError as e:
        assert 'should raise KeyError but raised RuntimeError' == str(e), str(e)
    try:
        with test.raises(KeyError):
            pass
    except AssertionError as e:
        assert 'should raise KeyError' == str(e), e


@test
def assert_raises_specific_message():
    with test.raises(RuntimeError, "hey man!"):
        raise RuntimeError("hey man!")
    try:
        with test.raises(RuntimeError, "hey woman!"):
            raise RuntimeError("hey man!")
    except AssertionError as e:
        assert "should raise RuntimeError with message 'hey woman!'" == str(e)



def spawn(f):
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=f)
    p.start()
    return p


def child():
    @test(report=False, skip=False) # override default skip=True in spawned process
    def in_child():
        print("I am a happy child", flush=True)
        assert 1 == 1


def import_syntax_error():
    import sub_module_syntax_error


def is_internal(frame):
    return '<frozen importlib.' in frame.f_code.co_filename or \
           'AUTOTEST_INTERNAL' in frame.f_code.co_varnames


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
    return types.FunctionType(
               func.__code__,                      # code
               bind_names(                         # globals
                   func.__globals__.copy(),
                   func.__code__.co_names,
                   inspect.currentframe().f_back),
               func.__name__,                      # name
               func.__defaults__,                  # default arguments
               func.__closure__)                   # closure/free vars


M = 'module scope'
a = 'scope:global'
b = 10

assert 'scope:global' == a
assert 10 == b

class X:
    a = 'scope:X'
    x = 11

    @test(keep=True)
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

    @test(keep=True)
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

        @test
        def h(fixture_C):
            assert 'scope:Y' == a
            assert None == b
            assert 11 == x
            assert 13 == y
            assert callable(C)
            assert callable(D)
            assert C != D
            assert 10 == abs(-10) # built-in
            assert 42 == C(42)
            assert 84 == D(42, 84) # "fixtures"
            assert 'module scope' == M
            assert 252 == fixture_C, fixture_C

    v = 45
    @test.fixture
    def f_A():
        yield v

    @test
    def fixtures_can_also_see_attrs_from_classed_being_defined(f_A):
        assert v == f_A, f_A

    @test
    async def coroutines_can_also_see_attrs_from_classed_being_defined(f_A):
        assert v == f_A, f_A


@test
def access_closure_from_enclosing_def():
    a = 46              # accessing this in another function makes it a 'freevar', which needs a closure
    @test
    def access_a():
        assert 46 == a


f = 16

@test
def dont_confuse_app_vars_with_internal_vars():
    assert 16 == f


@test.fixture
def fixture_D(fixture_A, a = 10):
    yield a


@test
def use_fixtures_as_context():
    with test.fixture_A as a:
        assert 42 == a
    with test.fixture_B as b: # or pass parameter/fixture ourselves?
        assert 84 == b
    with test.fixture_C as c:
        assert 252 == c
    with test.stdout as s:
        print("hello!")
        assert "hello!\n" == s.getvalue()
    keep = []
    with test.tmp_path as p:
        keep.append(p)
        (p / "f").write_text("contents")
    assert not keep[0].exists()
    # it is possible to supply additional args when used as context
    with test.fixture_D as d: # only fixture as arg
        assert 10 == d
    with test.fixture_D() as d: # idem
        assert 10 == d
    with test.fixture_D(16) as d: # fixture arg + addtional arg
        assert 16 == d


# with test.<fixture> does not need @test to run 'in'
with test.tmp_path as p:
    assert p.exists()
assert not p.exists()


@test
def idea_for_dumb_diffs():
    # if you want a so called smart diff: the second arg of assert is meant for it.
    # Runner supplies a generic (lazy) diff between two pretty printed values
    a = [7, 1, 2, 8, 3, 4]
    b = [1, 2, 9, 3, 4, 6]
    d = test.diff(a, b)
    assert str != type(d)
    assert callable(d.__str__)
    assert str == type(str(d))

    try:
        assert a == b, test.diff(a,b)
    except AssertionError as e:
        assert """
- [7, 1, 2, 8, 3, 4]
?  ---      ^

+ [1, 2, 9, 3, 4, 6]
?        ^      +++
""" == str(e)


    # you can als pass a diff function to test.<op>().
    # diff should accept the args preceeding it
    # str is called on the result, so you can make it lazy
    try:
        test.eq("aap", "ape", msg=lambda x, y: "mydiff")
    except AssertionError as e:
        assert "mydiff" == str(e)

    try:
        test.eq(a, b, msg=test.diff)
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
        test.eq(a, b, msg=set.symmetric_difference)
    except AssertionError as e:
        assert "{6, 7, 8, 9}" == str(e), e


@test
def bind_test_functions_to_their_fixtures():

    @test.fixture
    def my_fix():
        yield 34

    @test(bind=False, skip=True, keep=True, report=True)
    def not_bound_but_with_reporting(my_fix):
        assert 56 == my_fix
        return my_fix

    with test.stdout as s:
        not_bound_but_with_reporting(56)
    assert "  not_bound_but_with_reporting  \n" in s.getvalue(), repr(s.getvalue())


    @test(keep=True, bind=True)
    def bound_fixture_1(my_fix):
        assert 34 == my_fix
        return my_fix

    assert 34 == bound_fixture_1() # no need to pass args, already bound

    @test.fixture
    def my_fix(): # redefine fixture purposely to test time of binding
        yield 78

    @test(keep=True, bind=True, skip=True) # skip, need more args
    def bound_fixture_2(my_fix, result, *, extra=None):
        assert 78 == my_fix
        return result, extra

    assert 34 == bound_fixture_1() # no need to pass args, already bound
    assert (56, "top") == bound_fixture_2(56, extra="top") # no need to pass args, already bound

    class A:
        a = 34

        @test(keep=True, bind=True) # skip, need more args
        def bound_fixture_acces_class_locals(my_fix):
            assert 78 == my_fix
            assert 34 == a
            return a
        assert 34 == bound_fixture_acces_class_locals()

    test.default(bind=True, keep=True)

    @test
    def bound_by_default(my_fix):
        assert 78 == my_fix
        return my_fix

    assert 78 == bound_by_default()

    trace = []

    @test.fixture
    def enumerate():
        trace.append('S')
        yield 1
        trace.append('E')

    @test(keep=True, skip=True, bind=True)
    def rebind_on_every_call(enumerate):
        return True

    assert [] == trace
    assert rebind_on_every_call()
    assert ['S', 'E'] == trace
    assert rebind_on_every_call()
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


@test
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

    test.contains(B_in_betwixt.__code__.co_varnames, 'AUTOTEST_INTERNAL')

    def test_names(*should):
        _, _, tb = sys.exc_info()
        tb = filter_traceback(tb)
        names = tuple(tb.tb_frame.f_code.co_name for tb in iterate(lambda tb: tb.tb_next, tb))
        assert should == names, names

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
        @test
        def test_for_real_with_Runner_involved():
            test.eq(1, 2)
    except AssertionError:
        test_names('trace_backfiltering', 'test_for_real_with_Runner_involved')


if is_main_process:
    @test
    def silence_child_processes(stdout, stderr):
        p = spawn(child) # <= causes import of all current modules
        p.join(3)
        # if it didn't load (e.g. SyntaxError), do not run test to avoid
        # failures introduced by other modules that loaded as a result
        # of multiprocessing spawn, but failed
        if p.exitcode == 0:
            out = stdout.getvalue()
            test.contains(out, "I am a happy child")
            test.not_("in_child" in out)


    @test
    def import_submodule(stdout):
        import sub_module_ok
        m = stdout.getvalue()
        test.contains(m, "sub_module_ok")
        test.contains(m, "test_one")
        test.contains(m, "I am a happy submodule")


    try:
        with test.stdout as s:
            @test
            def import_submodule_is_silent_but_does_report_failures():
                import sub_module_fail
        test.fail("Should have failed.")
    except AssertionError as e:
        m = s.getvalue()
        test.contains(m, "5  ->	    autotest.test.eq(123, 42)")
        test.eq("('eq', 123, 42)", str(e))


    try:
        @test
        def import_syntax_error(stderr):
            spawn(import_syntax_error).join(3)
    except SyntaxError as e:
        pass

class wildcard:
    def __eq__(self, _):
        return True
test.any = wildcard()

@test
def wildcard_matching():
    test.eq([test.any, 42], [16, 42])


# keep only the standard fixtures; ditch test fixtures
test.fixtures = {}
for fx in (tmp_path, stdout, stderr, raises):
    test.fixture(fx)


###@test
def setup_correct():
    print("TEST SETUIP")
    print(os.getcwd())
    sys.argv = ['', 'sdist']
    from setup import setup
    from tarfile import open
    tf = open(name='dist/autotest-0.1.0.tar.gz', mode='r:gz')
    test.eq(['autotest-0.1.0', 'autotest-0.1.0/LICENSE', 'autotest-0.1.0/PKG-INFO', 'autotest-0.1.0/README.rst', 'autotest-0.1.0/autotest.py', 'autotest-0.1.0/setup.py', 'autotest-0.1.0/sub_module_fail.py', 'autotest-0.1.0/sub_module_ok.py', 'autotest-0.1.0/sub_module_syntax_error.py', 'autotest-0.1.0/temporary_class_namespace.py'],
            tf.getnames())
    tf.close()



# helpers
def mock_object(*functions, **more):
    """ Creates an object from a bunch of functions.
        Useful for testing methods from inside the class definition. """
    self = mock.Mock()
    self.configure_mock(**{f.__name__: types.MethodType(f, self) for f in functions}, **more)
    return self


#@test
def probeer():
    test.eq(1, 0)
#@test
def probeer():
    assert 1 == 2



