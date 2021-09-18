## begin license ##
#
# "Autotest" is a simplest thing that could possibly work test runner.
#
# Copyright (C) 2021 Seecr (Seek You Too B.V.) https://seecr.nl
#
# This file is part of "Meresco Components"
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
import types
import pathlib
import tempfile
import shutil
import sys
import operator
import contextlib
import functools
import io
import asyncio
import multiprocessing
import os


__all__ = ['test', 'Runner']


sys_defaults = {
    # We run self-tests when imported, not when run as main program. This
    # avoids tests run multiple times.
    # Also do not run tests when imported in a child, by multiprocessing
    'skip' : __name__ == '__main__' or multiprocessing.current_process().name != "MainProcess",
}




class Runner:

    def __init__(self, **opts):
        self.defaults = opts
        self.fixtures = {}
        self.finalize_early = []


    def __call__(self, f=None, **opts):
        """Decorator to define, run and report a test"""
        if opts:
            return functools.partial(self._run, **{**self.defaults, **opts})
        return self._run(f, **self.defaults)


    def default(self, **kws):
        """ Set default options for nexts tests."""
        self.defaults.update(kws)


    def fail(self, *args):
        if not self.defaults.get('skip', False):
            raise AssertionError(*args)


    def fixture(self, f):
        """Decorator for fixtures a la pytest. A fixture is a generator yielding exactly 1 value.
           That value is used as argument to functions declaring the fixture in their args. """
        assert inspect.isgeneratorfunction(f), f
        self.fixtures[f.__name__] = f
        return f


    def early_finalize(self, fx):
        """ Fixtures finalized before debugging (i.e. stdout)"""
        self.finalize_early.append(fx)


    def __getattr__(self, name):
        """ test.eq(lhs, rhs) etc, any operator from module 'operator' really.
            or it returns a context manager if name denotes a fixture. """
        if name in self.fixtures:
            fx = self.fixtures[name]
            _, args = self._get_args(fx)
            return contextlib.contextmanager(fx)(*args)
        op = getattr(operator, name)
        def call(*args):
            if not bool(op(*args)):
                raise AssertionError(op.__name__, *args)
            return True
        return call


    def _get_fixtures(*_):
        """ Return info about fixtures, recursively. Placeholder function during bootstrap. """
        return []


    def _get_args(self, f):
        fxs = self._get_fixtures(f)
        return fxs, (fx[2] for fx in fxs)


    def _run(self, f, keep=False, silent=False, skip=False):
        if skip:
            return
        print_msg = print if not silent else lambda *a, **k: None
        print_msg(f"{__name__}  {f.__module__}  {f.__name__}  ", end='', flush=True)
        fxs, args = self._get_args(f)
        try:
            if inspect.iscoroutinefunction(f):
                asyncio.run(f(*args), debug=True)
            else:
                f = eval_with_unbound(f, *args)
        except BaseException as e:
            _, _, tb = sys.exc_info()
            print_msg()
            if e.args == ():
                for fx in self.finalize_early:
                    list(fx)
                post_mortem(tb)
                exit(-1)
            raise
        finally:
            finalize_fixtures(fxs)
            del self.finalize_early[:]
        print_msg("OK")
        return f if keep else None



test = Runner(**sys_defaults)


def eval_with_unbound(f, *args):
    """ Bootstrapping, placeholder, overwritten later """
    f(*args)
    return f


def post_mortem(tb):
    import pdb
    p = pdb.Pdb()
    p.rcLines.extend(['longlist'])
    p.reset()
    p.interaction(None, tb)


def finalize_fixtures(fxs):
    """ Exhaust all fixture generators, recursively. """
    [(list(fx[1]), finalize_fixtures(fx[0])) for fx in fxs]


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



""" bootstrapping: test and instal fixture support """
@test
def test_get_fixtures():
    class X(Runner):
        def _get_fixtures(self, f):
            """ The real function, tested and then installed. Lambda is read as 'let'. """
            return [(lambda fx=self.fixtures[name]:
                        (lambda fxs=self._get_fixtures(fx):
                            (lambda gen=fx(*(x[2] for x in fxs)):
                                (fxs, gen, next(gen)))())())() for name in inspect.signature(f).parameters]
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
    setattr(Runner, '_get_fixtures', X._get_fixtures)
    #Runner._get_fixtures = _get_fixtures


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
        test.fail()
    except AssertionError as e:
        assert ('eq', 1, 2) == e.args, e.args

    try:
        test.ne(1, 1)
        test.fail()
    except AssertionError as e:
        assert "('ne', 1, 1)" == str(e), str(e)

@test
def test_testop_has_args():

    try:
        @test(silent=True)
        def nested_test_with_testop():
            x = 42
            y = 67
            test.eq(x, y)
    except AssertionError as e:
        test.eq("('eq', 42, 67)", str(e))

@test
def skip_until():
    test.default(skip=True)
    @test
    def fails():
        test.eq(1, 2)
    test.default(skip=False)
    try:
        @test(silent=True)
        def fails():
            test.gt(1, 2)
        test.fail()
    except AssertionError as e:
        test.eq("('gt', 1, 2)", str(e))

@test
def test_calls_other_test():
    @test(keep=True, silent=True)
    def test_a():
        assert 1 == 1
        return True
    @test(silent=True)
    def test_b():
        assert test_a()


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


@test.fixture
def stdout():
    g = capture('stdout')
    test.early_finalize(g)
    yield from g


@test.fixture
def stderr():
    g = capture('stderr')
    test.early_finalize(g)
    yield from g


#@test
def stdout_capture():
    name = "Erik"
    msgs = []
    @test(silent=True)
    def capture_all(stdout, stderr):
        print(f"Hello {name}!", file=sys.stdout)
        print(f"Bye {name}!", file=sys.stderr)
        msgs.extend([stdout.getvalue(), stderr.getvalue()])
    assert "Hello Erik!\n" == msgs[0]
    assert "Bye Erik!\n" == msgs[1]


#@test
def capture_stdout_child_processes(stdout):
    def f():
        @test(silent=True)
        def in_child():
            print("hier ben ik")
            assert 1 == 1
    p = multiprocessing.Process(target=f)
    p.start()
    p.join()
    test.eq("hier ben ik\n", stdout.getvalue())


@test
async def async_function():
    await asyncio.sleep(0)
    assert True


@test(skip=True)
def assert_raises_like():
    pass


def spawn(f):
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=f)
    p.start()
    return p


def child():
    @test
    def in_child():
        print("I am a happy child", flush=True)
        assert 1 == 1

def import_syntax_error():
    import sub_module_syntax_error

if multiprocessing.current_process().name == "MainProcess":
    #@test
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
        print()


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


"""
Create a function object.

  code
    a code object
  globals
    the globals dictionary
  name
    a string that overrides the name from the code object
  argdefs
    a tuple that specifies the default argument values
  closure
    a tuple that supplies the bindings for free variables
"""

def eval_with_unbound(func, *args):
    f = types.FunctionType(
            func.__code__,
            bind_names(
                func.__globals__.copy(),
                func.__code__.co_names,
                inspect.currentframe().f_back))
    f(*args)
    return f


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


@test
def use_fixtures_as_context():
    print("WHAT:", test.tmp_path)
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


# we import ourselves to trigger running the test if/when you run autotest as main
import autotest
