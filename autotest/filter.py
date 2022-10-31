

def filter_hook(runner, func):
    f = runner.option_get('filter', '')
    if f in func.__qualname__:
        return func


def filter_test(self_test):
    my_test = self_test(hooks=[filter_hook])
    @my_test
    def hook_but_no_filter_given():
        pass
    with my_test.child(filter='noot') as with_noot:
        r = [0, 0, 0]
        class aap:
            @with_noot
            def aap():
                r[0] = 1
            @with_noot
            def noot():
                r[1] = 1
                @with_noot
                def mies():
                    r[2] = 1
        assert r == [0, 1, 1], r
    assert {'found': 3, 'run': 2} == with_noot.stats, with_noot.stats
