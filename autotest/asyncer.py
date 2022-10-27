import inspect
import asyncio
import concurrent


# NB: this supports up to 10 levels of nested async tests
thread = concurrent.futures.ThreadPoolExecutor(max_workers=10).submit


""" Defines a hook for supporting async functions as test.
    It does so by executing the coroutine in an event loop in
    a separate thread. """


def async_hook(runner, func):
    def may_be_async(*a, **k):
        AUTOTEST_INTERNAL = 1
        coro_or_result = func(*a, **k)
        if inspect.iscoroutine(coro_or_result):
            thread(asyncio.run, coro_or_result, debug=runner._options.get('debug')).result()
        else:
            return coro_or_result
    return may_be_async


def async_test(self_test):
    with self_test.child(hooks=(async_hook,)) as atest:

        atest.eq({}, atest.stats)
        trace = [None, None, None, None]

        @atest
        async def an_async_test():
            trace[0] = asyncio.get_running_loop()

        @atest
        async def parent():
            trace[1] = asyncio.get_running_loop()
            @atest
            async def nested_async_test():
                trace[2] = asyncio.get_running_loop()
            @atest
            def nested_sync_test():
                trace[3] = asyncio.get_running_loop()

        assert len(trace) ==  4
        assert trace[0]
        assert trace[1]
        assert trace[1] != trace[0]
        assert trace[2]
        assert trace[2] != trace[0] # each async test has own loop
        assert trace[2] != trace[1]
        assert trace[3] == trace[1] # still same loop

        try:
            @atest
            async def with_failures():
                assert 1 == 2, '1 is not 2'
                raise Exception
        except AssertionError as e:
            assert str(e) == '1 is not 2', e

        atest.eq({'found': 5, 'run': 5}, atest.stats)
