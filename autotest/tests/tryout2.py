
import autotest

test = autotest.get_tester(__name__)

@test
def one_simple_test_succeeds():
    test.eq(1, 1)

