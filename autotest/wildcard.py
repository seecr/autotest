
def any(_): # name is included in diffs
    return True


class _Any:
    def __init__(self, f=any):
        self.f = f

    def __eq__(self, x):
        return bool(self.f(x))

    def __call__(self, f):
        return _Any(f)

    def __repr__(self):
        return self.f.__name__  + '(...)' if self.f else '*'


class _Wildcard:

    def __call__(self, tester, func):
        return func

    def lookup(self, tester, name):
        if name == 'any':
            return _Any()
        raise AttributeError


wildcard_hook = _Wildcard()


def wildcard_test(self_test):
    with self_test.child(hooks=(wildcard_hook,)) as test:
        @test
        def wildcard_matching():
            test.eq([test.any, 42], [16, 42])
            test.eq([test.any(lambda x: x in [1,2,3]), 78], [2, 78])
            test.ne([test.any(lambda x: x in [1,2,3]), 78], [4, 78])


