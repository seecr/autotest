import inspect
import asyncio
import threading


def async_hook(runner, func):
    def may_be_async(*a, **k):
        coro_or_result = func(*a, **k)
        if inspect.iscoroutine(coro_or_result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(coro_or_result, debug=runner._options.get('debug'))
            else:
                """ we get called by sync code (a decorator during def) which in turn
                    is called from async code. The only way to run this async test
                    is in a new event loop (in another thread) """
                t = threading.Thread(target=asyncio.run, args=(coro_or_result,),
                        kwargs={'debug': runner._options.get('debug')})
                t.start()
                t.join()
        else:
            return coro_or_result
    return may_be_async



def async_test(self_test):
    with self_test.child(hooks=(async_hook,)) as atest:
        atest.eq({}, atest.stats)
        trace = [0, 0, 0, 0, 0]
        @atest
        async def an_async_test():
            trace[0] = 1
        atest.eq({'found': 1, 'run': 1}, atest.stats)
        @atest
        async def parent():
            @atest
            async def nested_async_test():
                trace[1] = 1
            @atest
            def nested_sync_test():
                trace[2] = 1
            trace[3] = 1
        atest.eq({'found': 4, 'run': 4}, atest.stats)

        @atest
        def normal_test():
            trace[4] = 1


        assert trace == [1, 1, 1, 1, 1], trace
        atest.eq({'found': 5, 'run': 5}, atest.stats)
