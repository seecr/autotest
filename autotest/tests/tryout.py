
import autotest

test = autotest.get_tester(__name__)

@test
def one_simple_test():
    test.eq(1, 1)

@test.integration
async def one_more_test():
    assert 1 == 2, "one is not two"
    test.eq(1, 2)
