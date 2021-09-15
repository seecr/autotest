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
import pathlib
import tempfile
import shutil
import sys
import pdb
import traceback
import operator


__all__ = ['test', 'fixture', 'fail'] # fixtures need not importing; some magic after all...


# hold al fixtures found during import (global)
fixtures = {}


class Test:

    def __call__(s, t):
        """Decorator to define, run and report a test"""
        print(f"{t.__module__}  {t.__name__}  ", end='', flush=True)
        fxs = get_fixtures(t)
        try:
            t(*(fx[2] for fx in fxs))
        except BaseException:
            e, v, tb = sys.exc_info()
            print()
            if v.args == ():
                pdb.post_mortem(tb)
                exit(-1)
            else:
                raise
        finally:
            finalize_fixtures(fxs)
        print("OK")


    class Op:
        """ Represents operators in test.eq etc. """
        def __init__(self, op):
            self._op = op

        def __call__(self, *args):
            if not bool(self._op(*args)):
                raise AssertionError(self._op.__name__, *args)
            return True


    def __getattr__(self, name):
        """ test.eq(lhs, rhs) etc, any operator from module 'operator' really. """
        op = getattr(operator, name)
        return Test.Op(op)


test = Test()


def fail(*args):
    raise AssertionError(*args)


def get_fixtures(_):
    """ Return info about fixtures, recursively. Placeholder function during bootstrap. """
    return []


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



def fixture(f):
    """Decorator for fixtures a la pytest. A fixture is a generator yielding exactly 1 value.
       That value is used as argument to functions declaring the fixture in their args. """
    assert inspect.isgeneratorfunction(f), f
    fixtures[f.__name__] = f
    return f


""" bootstrapping: test and instal fixture support """
@test
def test_get_fixtures():
    def _get_fixtures(f):
        """ The real function, tested and then installed. Lambda is read as 'let'. """
        return [(lambda fx=fixtures[name]:
                    (lambda fxs=_get_fixtures(fx):
                        (lambda gen=fx(*(x[2] for x in fxs)):
                            (fxs, gen, next(gen)))())())() for name in inspect.signature(f).parameters]
    def f(): pass
    assert [] == _get_fixtures(f)
    @fixture
    def a(): yield 7
    @fixture
    def b(): yield 9
    def f(a, b): pass
    fxs, gen, v = _get_fixtures(f)[0]
    assert fxs == []
    assert inspect.isgenerator(gen)
    assert v == 7, v
    fxs, gen, v = _get_fixtures(f)[1]
    assert fxs == []
    assert inspect.isgenerator(gen)
    assert v == 9, v
    @fixture
    def c(b): yield 13
    def f(c): pass
    fxs, gen, v = _get_fixtures(f)[0]
    assert 1 == len(fxs), fxs
    assert inspect.isgenerator(gen)
    assert v == 13, v
    fxs, gen, v = fxs[0] # recursive
    assert fxs == []
    assert v == 9, v
    @fixture
    def d(c): yield 17
    def f(d): pass
    gs = _get_fixtures(f)
    fxs, gen, v = gs[0][0][0][0][0]
    assert [] == fxs
    assert inspect.isgenerator(gen)
    assert v == 9, v
    # everything OK, install it
    global get_fixtures
    get_fixtures = _get_fixtures


@fixture
def tmp_path():
    p = pathlib.Path(tempfile.mkdtemp())
    yield p
    shutil.rmtree(p)


fixture_lifetime = []


@fixture
def fixture_A():
    fixture_lifetime.append('A-live')
    yield 42
    fixture_lifetime.append('A-close')


@fixture
def fixture_B(fixture_A):
    fixture_lifetime.append('B-live')
    yield fixture_A * 2
    fixture_lifetime.append('B-close')


@fixture
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
        fail()
    except AssertionError as e:
        assert ('eq', 1, 2) == e.args, e.args

    try:
        test.ne(1, 1)
        fail()
    except AssertionError as e:
        assert "('ne', 1, 1)" == str(e), str(e)

@test
def test_testop_has_args():

    try:
        @test
        def nested_test_with_testop():
            x = 42
            y = 67
            test.eq(x, y)
    except AssertionError as e:
        test.eq("('eq', 42, 67)", str(e))

