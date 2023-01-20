## begin license ##
#
# "Autotest": a simpler test runner for python
#
# Copyright (C) 2022-2023 Seecr (Seek You Too B.V.) https://seecr.nl
#
# This file is part of "Autotest"
#
# "Autotest" is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "Autotest" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with "Autotest".  If not, see <http://www.gnu.org/licenses/>.
#
## end license ##

""" Defines the Tester

    NB: a stand alone Tester (self_test) is created as to not interfere with
        the system wide default runner, which may not exists as long as we
        are bootstrapping.

    NB: Tester is tested using Tester itself. Tester is bootstrapped in steps,i
        starting with a simple Tester of which the capabilities are gradually
        extended.

    TODO nav gebruik in metastreams-server:
    2. Make async tests reuse event loop if available

"""

import inspect          # for recognizing functions, generators etc
import contextlib       # child as context
import collections      # chain maps for hierarchical Tester and Counter
import logging          # output to logger
import os               # controlling env
import sys              # maxsize
import io               # formatting tests


logger = logging.getLogger('autotest')
# in order to have tests shown when logging is unconfigured, we log at a level slightly
# higher than the default level WARNING (30)
default_loglevel = 33
logging.addLevelName(default_loglevel, "TEST") # TODO make configurable


""" by default, tests are suppressed in subprocesses because many tests run
    during import, which can easily lead to an endless loop.
    Use subprocess=True when needed.
    This is NOT the same as preventing __main__ from running multiple times!
"""
is_subprocess = 'AUTOTEST_PARENT' in os.environ
os.environ['AUTOTEST_PARENT'] = 'Y'


defaults = dict(
    keep       = False,           # Ditch test functions after running
    run        = True,            # Run test immediately
    subprocess = False            # Do not run test when in a subprocess
)


class Tester:
    """ Main tool for running tests across modules and programs. """

    def __call__(self, *functions, **options):
        """Runs tests, with options; useful as decorator. """
        AUTOTEST_INTERNAL = 1
        if functions and options:
            return self(**options)(*functions)
        if options:
            return self.getChild(**options)
        for f in functions:
            result = self._run(f)
        return result


    def getChild(self, *_, **__):
        return Tester(*_, parent=self, **__)

    get_child = getChild


    @contextlib.contextmanager
    def child(self, *_, **__):
        yield self.getChild(*_, **__)


    def fail(self, *args, **kwds):
        raise AssertionError(*args, *(kwds,) if kwds else ())


    def __init__(self, name=None, parent=None, **options):
        """ Not to be instantiated directly, use autotest.getTester() instead """
        self._parent = parent
        if parent:
            self._name = parent._name + '.' + name if name and parent._name else name or parent._name
            self._options = parent._options.new_child(m=options)
        else:
            self._name = name
            self._options = collections.ChainMap(options, defaults)
        self.stats = collections.Counter(found=0, run=0)
        self._loghandlers = []


    def option_get(self, name, default=None):
        return self._options.get(name, default)


    def option_setdefault(self, name, default):
        return self._options.maps[0].setdefault(name, default)


    def option_enumerate(self, name):
        """ enumerate options from parents, flattening sequences """
        for m in self._options.maps:
            value = m.get(name)
            if isinstance(value, (list, tuple)):
                yield from reversed(value)
            elif value is not None:
                yield value


    def _stat(self, key):
        self.stats[key] += 1
        if p := self._parent:
            p._stat(key)


    def all_hooks(self):
        return self.option_enumerate('hooks')


    def _logtest(self, f, msg):
        loggername = f"{logger.name}.{self._name}" if self._name else logger.name
        record = logging.LogRecord(
            loggername,                               # name of logger
            self.option_get('loglevel', default_loglevel), # level under which test messages get logged
            f.__code__.co_filename if f else None,    # source file where test is
            f.__code__.co_firstlineno if f else None, # line where test is
            msg,                                      # message
            (),                                       # args (passed to message.format)
            None,                                     # exc_info
            f.__qualname__ if f else None,            # name of the function invoking test.<op>
            None                                      # text representation of stack
            )
        for hook in self.all_hooks():
            try:
                record = hook.logrecord(self, f, record)
            except AttributeError:
                pass
        logger.handle(record)


    def _run(self, test_func, *app_args, **app_kwds):
        """ Runs hooks and and runs result. """
        AUTOTEST_INTERNAL = 1
        self._stat('found')
        orig_test_func = test_func
        if not is_subprocess or self.option_get('subprocess', False):
            for hook in self.all_hooks():
                if not (test_func := hook(self, test_func)):
                    break
            else:
                if self.option_get('run'):
                    self._stat('run')
                    self._logtest(orig_test_func, orig_test_func.__qualname__) # TODO pass test func?
                    test_func(*app_args, **app_kwds)
        return orig_test_func if self.option_get('keep') else None


    def log_stats(self):
        name = self._name + '.stats: ' if self._name else 'stats: '
        logger.log(self.option_get('loglevel', default_loglevel), name + ', '.join(f'{k}: {v}' for k, v in self.stats.most_common()))


    def __getattr__(self, name):
        for hook in self.all_hooks():
            try:
                return hook.lookup(self, name)
            except AttributeError:
                pass

    def __str__(self):
        return f"<Tester {self._name!r}>"



from numbers import Number
class any_number:
    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
    def __eq__(self, rhs):
        return isinstance(rhs, Number) and self._lo <= rhs <= self._hi


from .operators import operators_hook
from .levels import levels_hook

from sys import argv
if 'autotest.selftest' in argv:
    argv.remove('autotest.selftest')
    self_test = Tester('autotest-self-tests',
        hooks=(operators_hook, levels_hook), level='unit') # separate runner for bootstrapping/self testing
else:
    class Ignore:
        def __call__(self, runner, func): pass
        def lookup(self, runner, name): return self
    self_test = Tester(hooks=[Ignore()])
    #self_test.addHandler(logging.NullHandler())



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


@self_test
def new_assert():
    assert self_test.eq(1, 1)
    assert self_test.lt(1, 2) # etc

    try:
        self_test.eq(1, 2)
        self_test.fail()
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
        @self_test
        def nested_test_with_testop():
            x = 42
            y = 67
            self_test.eq(x, y)
            return True
    except AssertionError as e:
        assert hasattr(e, 'args')
        assert 3 == len(e.args)
        self_test.eq("('eq', 42, 67)", str(e))


@self_test
def test_calls_other_test():
    @self_test(keep=True)
    def test_a():
        assert 1 == 1
        return True
    @self_test.child()
    def test_b():
        assert test_a()


@self_test
def call_test_with_complete_context():
    @self_test(keep=True)
    def a_test():
        assert True
    assert a_test, a_test
    self_test(a_test)


@contextlib.contextmanager
def intercept():
    records = []
    class intercept:
        level = 33
        handle = records.append
    root_logger = logging.getLogger()
    root_logger.addHandler(intercept)
    try:
        yield records
    finally:
        root_logger.removeHandler(intercept)


@contextlib.contextmanager
def configure_stream_hander():
    autotest_logger = logging.getLogger('autotest')
    s = io.StringIO()
    handler = logging.StreamHandler(s)
    handler.setFormatter(logging.Formatter(fmt=logging._STYLES['%'][1]))
    autotest_logger.addHandler(handler)
    try:
        yield s
    finally:
        autotest_logger.removeHandler(handler)


class logging_handlers:

    @self_test
    def tester_delegates_to_root_logger():
        with intercept() as i:
            tester = Tester('free')
            @tester
            def my_output_goes_to_root_logger():
                assert 1 == 1 # hooks for operators not present
            self_test.eq('logging_handlers.tester_delegates_to_root_logger.<locals>.my_output_goes_to_root_logger', i[0].funcName)


    @self_test
    def logrecords_contents():
        tester = Tester("unconfigured_logging", hooks=[levels_hook])
        with intercept() as records:
            _line_ = inspect.currentframe().f_lineno
            @tester
            def a_test():
                assert 1 == 1
            self_test.eq((), records[0].args)
            self_test.lt(1673605231.8793523, records[0].created)
            self_test.eq(None, records[0].exc_info)
            self_test.eq('tester.py', records[0].filename)
            self_test.eq('logging_handlers.logrecords_contents.<locals>.a_test', records[0].funcName)
            self_test.eq('TEST', records[0].levelname)
            self_test.eq(33, records[0].levelno)
            self_test.eq(_line_+1, records[0].lineno)
            self_test.eq('tester', records[0].module)
            self_test.truth(10 < records[0].msecs < 1000)
            self_test.eq('UNIT:logging_handlers.logrecords_contents.<locals>.a_test', records[0].msg)
            self_test.eq('autotest.unconfigured_logging', records[0].name)
            self_test.endswith(records[0].pathname, 'autotest/autotest/tester.py')
            self_test.truth(2 < records[0].process < 100000)
            self_test.eq('MainProcess', records[0].processName)
            self_test.truth(1 < records[0].relativeCreated < 1000)
            self_test.eq(None, records[0].stack_info)
            self_test.truth(1 < records[0].thread < sys.maxsize)
            self_test.eq('MainThread', records[0].threadName)
            self_test.eq(40, records[0].testlevel)
            self_test.eq('UNIT', records[0].testlevelname)


    @self_test
    def tester_with_default_formatting():
        with configure_stream_hander() as s:
            tester = Tester("default_formatting", hooks=[levels_hook])
            @tester
            def a_test():
                assert 1 == 1  # hooks for operators not present
            self_test.eq("TEST:autotest.default_formatting:UNIT:logging_handlers.tester_with_default_formatting.<locals>.a_test\n", s.getvalue())


    @self_test
    def tester_without_a_name():
        with configure_stream_hander() as s:
            tester = Tester(hooks=[levels_hook])
            @tester
            def a_test():
                assert 1 == 1  # hooks for operators not present
            self_test.eq("TEST:autotest:UNIT:logging_handlers.tester_without_a_name.<locals>.a_test\n", s.getvalue())


    @self_test
    def sub_tester():
        with configure_stream_hander() as s:
            main = Tester("main", hooks=[levels_hook])
            sub1 = main.getChild('sub1')
            sub2 = sub1.getChild('sub2')
            @sub1
            def sub1_test():
                pass
            @main
            def main_test():
                pass
            @sub2
            def sub2_test():
                pass
            loglines = s.getvalue().splitlines()
            self_test.eq("TEST:autotest.main.sub1:UNIT:logging_handlers.sub_tester.<locals>.sub1_test", loglines[0])
            self_test.eq("TEST:autotest.main:UNIT:logging_handlers.sub_tester.<locals>.main_test", loglines[1])
            self_test.eq("TEST:autotest.main.sub1.sub2:UNIT:logging_handlers.sub_tester.<locals>.sub2_test", loglines[2])


    @self_test
    def tester_with_handler_failing():
        with configure_stream_hander() as s:
            main = Tester("main", hooks=[levels_hook])
            try:
                @main
                def a_failing_test():
                    assert 1 == 2
            except AssertionError:
                pass
            self_test.eq("TEST:autotest.main:UNIT:logging_handlers.tester_with_handler_failing.<locals>.a_failing_test\n", s.getvalue())


    @self_test
    def run_or_not():
        @self_test(run=False)
        def not_run():
            assert "do not" == "run me"


