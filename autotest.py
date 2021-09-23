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


__all__ = ['test', 'Runner']


sys_defaults = {
    # We run self-tests when imported, not when run as main program. This # avoids tests run multiple times.
    # Also do not run tests when imported in a child, by multiprocessing
    'skip'  : __name__ == '__main__' or
              multiprocessing.current_process().name != "MainProcess",

    'keep'  : False,      # Ditch test functions after running, or keep them in their namespace.
    'report': True,       # Do reporting of succeeded tests.
    'bind'  : False,      # Kept functions are bound to fixtures or not
}


class Runner:

    def __init__(self, **opts):
        self.defaults = opts
        self.fixtures = {}
        self.finalize_early = []


    def __call__(self, f=None, **opts):
        """Decorator to define, run and report a test"""
        if opts:
            return self._bind_options(**{**self.defaults, **opts})
        bound_f = self._bind_options(**self.defaults)
        bound_f(f)


    def _bind_options(self, *, bind, skip, keep, **opts):
        def _do_run(f):
            f = bind_1_frame_back(f)
            fxs = self._get_fixtures(f)
            bound = functools.partial(self._run, f, fxs, **opts)
            if not skip:
                bound()
            if not keep:
                return
            if bind:
                return bound
            return f
        return _do_run


    def _run(self, f, fxs, *args, report, **kwds):
        print_msg = print if report else lambda *a, **k: None
        print_msg(f"{__name__}  {f.__module__}  {f.__name__}  ", flush=True)
        fx_values = (fx[2] for fx in fxs)
        try:
            if inspect.iscoroutinefunction(f):
                asyncio.run(f(*fx_values), debug=True)
            else:
                return f(*fx_values, *args, **kwds)
        except BaseException as e:
            et, ev, tb = sys.exc_info()
            print_msg()
            if e.args == ():
                for fx in self.finalize_early:
                    list(fx)
                traceback.print_exception(et, ev, tb.tb_next.tb_next)
                post_mortem(tb)
                exit(-1)
            raise
        finally:
            finalize_fixtures(fxs)
            del self.finalize_early[:]


    def default(self, **kws):
        """ Set default options for next tests."""
        self.defaults.update(kws)


    def fail(self, *args, **kwds):
        if not self.defaults.get('skip', False):
            args += (kwds,) if kwds else ()
            raise AssertionError(*args)


    def fixture(self, func=None, finalize_early=False):
        """Decorator for fixtures a la pytest. A fixture is a generator yielding exactly 1 value.
           That value is used as argument to functions declaring the fixture in their args.
           finalize_early means the fixture is finalized before pdb instead of at the very end."""
        def register(f):
            assert inspect.isgeneratorfunction(f), f
            bound_f = bind_1_frame_back(f)
            if finalize_early:
                def register_early(*a, **k):
                    gen = bound_f(*a, **k)
                    self.finalize_early.append(gen)
                    return gen
                self.fixtures[f.__name__] = register_early
            else:
                self.fixtures[f.__name__] = bound_f
            return bound_f
        return register(func) if func else register


    def diff(self, a, b):
        """ When str called, produces diff of textual representation of a and b. """
        return lazy_str(
            lambda:
                '\n' + '\n'.join(
                    difflib.ndiff(
                        pprint.pformat(a).splitlines(),
                        pprint.pformat(b).splitlines())))


    def __getattr__(self, name):
        """ test.eq(lhs, rhs) etc, any operator from module 'operator' really.
            or it returns a context manager if name denotes a fixture. """
        if name in self.fixtures:
            fx = self.fixtures[name]
            _, args = self._get_fixture_values(fx)
            return CollectArgsContextManager(fx, *args)
        op = getattr(operator, name)
        def call_operator(*args, msg=None):
            if not bool(op(*args)):
                if msg:
                    raise AssertionError(msg(*args))
                else:
                    raise AssertionError(op.__name__, *args)
            return True
        return call_operator


    def _get_fixtures(*_):
        """ Return info about fixtures, recursively. Placeholder function during bootstrap. """
        return []


    def _get_fixture_values(self, f):
        fxs = self._get_fixtures(f)
        return fxs, (fx[2] for fx in fxs)


test = Runner(**sys_defaults)


CollectArgsContextManager = contextlib.contextmanager
""" Bootstrapping, placeholder, overwritten later """


def bind_1_frame_back(_func_):
    """ Bootstrapping, placeholder, overwritten later """
    return _func_


def post_mortem(tb):
    p = pdb.Pdb()
    p.rcLines.extend(['longlist'])
    p.reset()
    p.interaction(None, tb)


def finalize_fixtures(fxs):
    """ Exhaust all fixture generators, recursively. """
    [(list(fx[1]), finalize_fixtures(fx[0])) for fx in fxs]


class lazy_str:
    def __init__(self, f):
        self._f = f
    def __str__(self):
        return self._f()


# Example Tests
from numbers import Number

class any_number:
    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
    def __eq__(self, rhs):
        return isinstance(rhs, Number) and self._lo <= rhs <= self._hi


@test
def AnyNumber():
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
@test
def test_get_fixtures():
    class X(Runner):
        def _get_fixtures(self, f):
            """ The real function, tested and then installed. Lambda is read as 'let'. """
            return [(lambda fx=self.fixtures[name]:
                        (lambda fxs=self._get_fixtures(fx):
                            (lambda gen=fx(*(x[2] for x in fxs)):
                                (fxs, gen, next(gen)))())())() for name in inspect.signature(f).parameters if name in self.fixtures]
    x = X()
    def f(): pass
    assert [] == x._get_fixtures(f)
    @x.fixture
    def a(): yield 7
    @x.fixture
    def b(): yield 9
    def f(a, b): pass
    fxs, gen, v = x._get_fixtures(f)[0]
    assert fxs == []
    assert inspect.isgenerator(gen)
    assert v == 7, v
    fxs, gen, v = x._get_fixtures(f)[1]
    assert fxs == []
    assert inspect.isgenerator(gen)
    assert v == 9, v
    @x.fixture
    def c(b): yield 13
    def f(c): pass
    fxs, gen, v = x._get_fixtures(f)[0]
    assert 1 == len(fxs), fxs
    assert inspect.isgenerator(gen)
    assert v == 13, v
    fxs, gen, v = fxs[0] # recursive
    assert fxs == []
    assert v == 9, v
    @x.fixture
    def d(c): yield 17
    def f(d): pass
    gs = x._get_fixtures(f)
    fxs, gen, v = gs[0][0][0][0][0]
    assert [] == fxs
    assert inspect.isgenerator(gen)
    assert v == 9, v
    # everything OK, install it
    Runner._get_fixtures = X._get_fixtures


@test.fixture
def tmp_path():
    p = pathlib.Path(tempfile.mkdtemp())
    yield p
    shutil.rmtree(p)


fixture_lifetime = []


@test.fixture
def fixture_A():
    fixture_lifetime.append('A-live')
    yield 42
    fixture_lifetime.append('A-close')


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
def with_one_fixture(fixture_A):
    assert 42 == fixture_A, fixture_A


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


class tmp_files:

    path = None

    @test
    def temp_file_removal(tmp_path):
        global path
        path = tmp_path / 'aap'
        path.write_text("hello")

    @test
    def temp_file_gone():
        assert not path.exists()


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


@test.fixture(finalize_early=True)
def stdout():
    yield from capture('stdout')


@test.fixture(finalize_early=True)
def stderr():
    yield from capture('stderr')


@test
def stdout_capture():
    name = "Erik"
    msgs = []
    @test(report=False)
    def capture_all(stdout, stderr):
        print(f"Hello {name}!", file=sys.stdout)
        print(f"Bye {name}!", file=sys.stderr)
        msgs.extend([stdout.getvalue(), stderr.getvalue()])
    assert "Hello Erik!\n" == msgs[0]
    assert "Bye Erik!\n" == msgs[1]


@test
def capture_stdout_child_processes(stdout):
    def f():
        @test(report=False, skip=False)
        def in_child():
            print("hier ben ik")
            assert 1 == 1
    p = multiprocessing.Process(target=f)
    p.start()
    p.join()
    # This gets fkd up if other output appears due to import done by multiprocessing spawn
    assert "hier ben ik\n" in stdout.getvalue()


@test
async def async_function():
    await asyncio.sleep(0)
    assert True


@test.fixture
def raises(exception=Exception):
    try:
        yield
    except exception:
        pass
    else:
        raise AssertionError(f"should raise {exception.__name__}")


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
    except RuntimeError as e:
        assert 'oops' == str(e), e
    try:
        with test.raises(KeyError):
            pass
    except AssertionError as e:
        assert 'should raise KeyError' == str(e), e



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


if multiprocessing.current_process().name == "MainProcess":
    @test
    def silence_child_processes(stdout, stderr):
        p = spawn(child) # <= causes import of all current modules
        p.join()
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
        @test
        def import_submodule_is_silent_but_does_report_failures(stdout):
            import sub_module_fail
            test.eq("sub_module  test_one  I am a happy submodule\nOK\n", stdout.getvalue())
        test.fail("Should have failed.")
    except AssertionError as e:
        assert "('eq', 123, 42)" == str(e), e


    try:
        @test
        def import_syntax_error(stderr):
            spawn(import_syntax_error).join()
    except SyntaxError as e:
        pass


def bind_names(bindings, names, frame):
    """ find names in locals on the stack and binds them """
    if not frame or not names:
        return bindings
    f_locals = frame.f_locals # rather expensive
    rest = []
    for name in names:
        try:
            bindings[name] = f_locals[name]
        except KeyError:
            rest.append(name)
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
               func.__code__.co_freevars)          # closure/free vars


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
        assert inspect.isfunction(C), C
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
            assert inspect.isfunction(C)
            assert inspect.isfunction(D)
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

import autotest
