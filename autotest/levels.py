import logging

CRITICAL    = logging.CRITICAL  # 50
UNIT        = logging.ERROR     # 40
INTEGRATION = logging.WARNING   # 30
PERFORMANCE = logging.INFO      # 20
NOTSET      = logging.NOTSET    #  0


class _Levels:

    def __call__(self, tester, func):
        levels = list(tester.option_enumerate('level')) or [0]
        threshold = levels[-1]
        level = levels[0]
        if level >= threshold:
            return func

    def lookup(self, tester, name):
        if name == 'critical':
            return tester(level=CRITICAL)
        if name == 'unit':
            return tester(level=UNIT)
        if name == 'integration':
            return tester(level=INTEGRATION)
        if name == 'performance':
            return tester(level=PERFORMANCE)
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


