## begin license ##
#
# "Autotest": a simpler test runner for python
#
# Copyright (C) 2022 Seecr (Seek You Too B.V.) https://seecr.nl
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

import logging

""" NB
    Tests are not assigned levels, Runners are. The level of the Runner
    determines the level of the tests run with it. So a single Runner
    will run all tests, since the level of the tests is by definition
    equal to the level of the runner.

    Filtering only becomes effective between Runners of different levels.
    If a child has level >= its parent level, all tests will run. If a child
    has a level < the level of its parent, it wil not run.

    The catch is that an individual test marked with e.g. @test.unit will
    run in a one-off temporary anonymous child of 'test' with the level
    UNIT, in this case.

    This relation extends out to arbitrarily deep nested Runners. A test
    only runs when its level is >= all parent levels. That is, any parent
    can raise the threshold in order to exclude tests.
"""

CRITICAL    = logging.CRITICAL  # 50
UNIT        = logging.ERROR     # 40
INTEGRATION = logging.WARNING   # 30   default in Python logging
PERFORMANCE = logging.INFO      # 20
NOTSET      = logging.NOTSET    #  0

CRITICAL

levels = {
    'CRITICAL':     CRITICAL,
    'UNIT':         UNIT,
    'INTEGRATION':  INTEGRATION,
    'PERFORMANCE':  PERFORMANCE,
    'NOTSET':       NOTSET,
    CRITICAL:       'CRITICAL',
    UNIT:           'UNIT',
    INTEGRATION:    'INTEGRATION',
    PERFORMANCE:    'PERFORMANCE',
    NOTSET:         'NOTSET',
}

DEFAULT_LEVEL = UNIT
DEFAULT_THRESHOLD = INTEGRATION


def numeric_level(l):
    return levels[l.upper()] if isinstance(l, str) else l


class _Levels:

    def __call__(self, tester, func):
        if tester.level >= tester.threshold:
            return func

    def lookup(self, tester, name):
        if name == 'level':
            return numeric_level(tester.option_get(name, DEFAULT_LEVEL))
        if name == 'threshold':
            return max((numeric_level(l) for l in tester.option_enumerate('threshold')), default=DEFAULT_THRESHOLD)
        if (level := levels.get(name.upper())) is not None:
            return tester(level=level)
        raise AttributeError

    def logrecord(self, tester, func, record):
        level = tester.level
        strlevel = levels[level]
        record.testlevel = level
        record.testlevelname = strlevel
        record.name = f"{record.name}:{strlevel}"
        return record


levels_hook = _Levels()


def levels_test(self_test):
    def run_various_tests(test):
        runs = [0, 0, 0, 0, 0, 0]
        @test.critical
        def a_critial_test():
            runs[0] = 1
        @test.unit
        def a_unit_test():
            runs[1] = 1
        @test.integration
        def a_integration_test():
            runs[2] = 1
        @test.performance
        def a_performance_test():
            runs[3] = 1
        @test.notset
        def a_notset_test():
            runs[4] = 1
        @test
        def a_default_test():
            runs[5] = 1
        return runs

    @self_test
    def probe_various_levels_and_thresholds():
        with self_test.child(hooks=(levels_hook,), level='unit', threshold='critical') as test:
            r = run_various_tests(test)
            assert [1, 0, 0, 0, 0, 0] == r, r

        with self_test.child(hooks=(levels_hook,), level='notset', threshold='unit') as test:
            r = run_various_tests(test)
            assert [1, 1, 0, 0, 0, 0] == r, r

        with self_test.child(hooks=(levels_hook,), level='performance', threshold='integration') as test:
            r = run_various_tests(test)
            assert [1, 1, 1, 0, 0, 0] == r, r

        with self_test.child(hooks=(levels_hook,), level='integration', threshold='performance') as test:
            r = run_various_tests(test)
            assert [1, 1, 1, 1, 0, 1] == r, r

        with self_test.child(hooks=(levels_hook,), level='critical', threshold='notset') as test:
            r = run_various_tests(test)
            assert [1, 1, 1, 1, 1, 1] == r, r


        with self_test.child(hooks=(levels_hook,), level='unit', threshold='integration') as test:
            r = run_various_tests(test)
            assert [1, 1, 1, 0, 0, 1] == r, r
            with test.child(level='performance', threshold='notset') as test2: # threshold ignored
                r = run_various_tests(test2)
                assert [1, 1, 1, 0, 0, 0] == r, r
            with test.child(level='integration', threshold='performance') as test2: # threshold ignored
                r = run_various_tests(test2)
                assert [1, 1, 1, 0, 0, 1] == r, r

