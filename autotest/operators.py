
import operator         # operators for asserting
import builtins         # operators for asserting

class _Operators:

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
            if diff:
                raise AssertionError(diff(*args))
            else:
                raise AssertionError(op.__name__, *args)
        return call_operator


operators_hook = _Operators()


def operators_test(self_test):

    @self_test
    def test_isinstance():
        self_test.isinstance(1, int)
        self_test.isinstance(1.1, (int, float))
        try:
            self_test.isinstance(1, str)
            self.fail()
        except AssertionError as e:
            assert str(e) == "('isinstance', 1, <class 'str'>)"
        try:
            self_test.isinstance(1, (str, dict))
            self.fail()
        except AssertionError as e:
            assert str(e) == "('isinstance', 1, (<class 'str'>, <class 'dict'>))"


    @self_test
    def use_builtin():
        """ you could use any builtin as well; there are not that much useful options in module builtins though
            we could change the priority of lookups, now it is: operator, builtins, <arg[0]>. Maybe reverse? """
        self_test.all([1,2,3])
        try:
            self_test.all([False,2,3])
            self.fail()
        except AssertionError as e:
            assert str(e) == "('all', [False, 2, 3])", e
        class A: pass
        class B(A): pass
        self_test.issubclass(B, A)
        self_test.hasattr([], 'append')


