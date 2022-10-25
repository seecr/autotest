
import operator         # operators for asserting
import builtins         # operators for asserting

class OperatorLookup:

    def __call__(self, tester, f):
        return f

    def lookup(self, runner, name, truth=bool):
        if name in ('comp', 'complement'):
            class Not:
                def __getattr__(inner, name):
                    return self.lookup(runner, name, truth=operator.not_)
            return Not()
        def call_operator(*args, diff=None):
            AUTOTEST_INTERNAL = 1
            if hasattr(operator, name):
                op = getattr(operator, name)
            elif hasattr(builtins, name):
                op = getattr(builtins, name)
            else:
                op = getattr(args[0], name)
                args = args[1:]
            if truth(op(*args)):
                return True
            msg = (diff(*args),) if diff else (op.__name__,) + args
            raise AssertionError(*msg)
        return call_operator

