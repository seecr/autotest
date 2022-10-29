
from .utils import bind_1_frame_back # redefine placeholder

class _Binder:
    def __call__(self, runner, func):
        return bind_1_frame_back(func)


binder_hook = _Binder()


def binder_test(self_test):
    self_test2 = self_test.getChild(hooks=(binder_hook,))
    with self_test2.child() as tst:
        @tst
        def nested_defaults():
            dflts0 = tst._options
            assert not dflts0['skip']
            with tst.child(skip=True) as tstk:
                dflts1 = tstk._options
                assert dflts1['skip']
                @tstk
                def fails_but_is_skipped():
                    tstk.eq(1, 2)
                try:
                    with tstk.child(skip=False) as TeSt:
                        dflts2 = TeSt._options
                        assert not dflts2['skip']
                        @TeSt
                        def fails_but_is_not_reported():
                            TeSt.gt(1, 2)
                        tst.fail()
                    assert tstk._options == dflts1
                except AssertionError as e:
                    tstk.eq("('gt', 1, 2)", str(e))
            assert tst._options == dflts0


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

