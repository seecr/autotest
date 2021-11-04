import autotest

@autotest.test(report=True)
def test_one():
    autotest.test.eq(123, 42)
