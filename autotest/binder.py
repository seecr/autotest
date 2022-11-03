
from .utils import extend_closure # redefine placeholder

class _Binder:
    def __call__(self, runner, func):
        return extend_closure(func)


binder_hook = _Binder()


def binder_test(self_test):
    self_test2 = self_test.getChild(hooks=(binder_hook,))

    class binding_context:
        a = 42
        @self_test2(keep=True)
        def one_test():
            assert a == 42
        a = 43
        #one_test()


    @self_test2
    def access_closure_from_enclosing_def():
        a = 46              # accessing this in another function makes it a 'freevar', which needs a closure
        @self_test2
        def access_a():
            assert 46 == a


    f = 16

    @self_test2
    def dont_confuse_app_vars_with_internal_vars():
        assert 16 == f

