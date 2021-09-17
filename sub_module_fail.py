import autotest

@autotest.test
def test_one():
    autotest.test.eq(123, 42)
