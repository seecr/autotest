
import autotest

test = autotest.get_tester(__name__).getChild(subprocess=True)

@test
def one_simple_test():
    test.eq(1, 1)

@test.integration
def one_integration_test():
    test.eq(1, 1)

@test.performance
def one_performance_test():
    test.eq(1, 1)
