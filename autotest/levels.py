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
INTEGRATION = logging.WARNING   # 30
PERFORMANCE = logging.INFO      # 20
NOTSET      = logging.NOTSET    #  0

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


class _Levels:

    def __call__(self, tester, func):
        levels = list(tester.option_enumerate('level')) or [0]
        threshold = max(levels)
        mylevel = levels[0]
        if mylevel >= threshold:
            return func

    def lookup(self, tester, name):
        if level := levels.get(name.upper()):
            return tester(level=level)
        raise AttributeError


levels_hook = _Levels()


def levels_test(self_test):
    with self_test.child(hooks=(levels_hook,)) as test:
        runs = [None, None]
        with test.child('tst', level=CRITICAL) as tst:
            @tst.critical
            def a_critial_test_always_runs():
                runs[0] = True
            @tst.unit
            def a_unit_test_often_runs():
                runs[1] = True
        test.eq([True, None], runs)

        runs = [None, None]
        with test.child(level=INTEGRATION) as tst:
            @tst.integration
            def a_integration_test_sometimes_runs():
                runs[0] = True
            @tst.performance
            def a_performance_test_sometimes_runs():
                runs[1] = True
        test.eq([True, None], runs)

        runs = [None, None]
        with test.child(level=PERFORMANCE) as perf:
            with perf.child(level=INTEGRATION) as inte:
                with inte.child(level=UNIT) as unit:
                    @unit.critical
                    def a_critical_test():
                        runs[0] = True
                    @unit.performance
                    def a_performance_test():
                        runs[1] = True
        test.eq([True, None], runs)

        runs = [0, 0, 0, 0, 0]
        with test.child(level=INTEGRATION) as test_a:            # 30
            with test_a.child(level=UNIT) as test_b:             # 40
                with test_b.child(level=PERFORMANCE) as test_c:  # 20
                    @test_c.critical                             # 50
                    def b_critical_test():
                        runs[0] = True
                    @test_c.unit                                 # 40
                    def b_unit_test():
                        runs[1] = True
                    @test_c.integration                          # 30
                    def b_integration_test():
                        runs[2] = True
                    @test_c.performance                          # 20
                    def b_performance_test():
                        runs[3] = True
                    @test_c                                      #  0
                    def b_notset_test():
                        runs[4] = True
        test.eq([True, True, 0, 0, 0], runs)
