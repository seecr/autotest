import logging

CRITICAL    = logging.CRITICAL
UNIT        = logging.ERROR
INTEGRATION = logging.WARNING
PERFORMANCE = logging.INFO
NOTSET      = logging.NOTSET


class Levels:

    def __call__(self, tester, func):
        level = tester._options.get('level', UNIT)
        plevel = tester._parent._options.get('level', UNIT) if tester._parent else UNIT
        skip = level < plevel
        if skip:
            return None
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


def testing_levels(self_test):
    with self_test.child(hooks=(Levels(),)) as test:
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


