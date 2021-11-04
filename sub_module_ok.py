import autotest

@autotest.test(report=True)
def test_one():
    print("I am a happy submodule")
    assert 1 == 1
