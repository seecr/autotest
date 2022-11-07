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
            async def with_options():
                AUTOTEST_INTERNAL = 1
                try:
                    if scbd := runner.option_get('slow_callback_duration'):
                        asyncio.get_running_loop().slow_callback_duration = scbd
                    await asyncio.wait_for(asyncio.shield(coro_or_result), runner.option_get('timeout', 1))
                except asyncio.TimeoutError as e:
                    raise TimeoutError(func.__name__) from None
            thread(
                asyncio.run,
                with_options(),
                debug=runner.option_get('debug', True)
                ).result()
        else:
            return coro_or_result
    return may_be_async


def async_test(self_test):
    with self_test.child(hooks=(async_hook,)) as atest:

        atest.eq({'found': 0, 'run': 0}, atest.stats)
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


        try:
            @atest(timeout=0.05)
            async def async_timeout_raises_standard_TimeoutError():
                await asyncio.sleep(0.10)
                assert False, "should have raised timeout"
        except TimeoutError as e:
            assert "async_timeout_raises_standard_TimeoutError" == str(e)


        import sys, io, time
        try:
            sys.stderr = io.StringIO()
            sys.stdout = io.StringIO()
            @atest
            async def debug_warnings_on_by_default():
                asyncio.get_running_loop().call_soon(time.sleep, 0.11)
            e = sys.stderr.getvalue()
            s = sys.stdout.getvalue()
            @atest
            def asserts():
                assert '' == s, s
                assert "Executing <Handle sleep(0.11) created at" in e, e
                assert "took 0.110 seconds" in e, e
        finally:
            sys.stderr = sys.__stderr__
            sys.stdout = sys.__stdout__


        try:
            sys.stderr = io.StringIO()
            @atest(keep=True)
            async def default_slow_callback_duration():
                assert asyncio.get_running_loop().slow_callback_duration == 0.1
            @atest(slow_callback_duration=0.12)
            async def slow_callback_duration_option():
                asyncio.get_running_loop().call_soon(time.sleep, 0.11)
            atest(default_slow_callback_duration)
            s = sys.stderr.getvalue()
            assert "Executing <Handle sleep(0.11) created at" not in s, s
            assert "took 0.110 seconds" not in s, s
        finally:
            sys.stderr = sys.__stderr__


        @atest
        def assert_stats():
            atest.eq({'found': 12, 'run': 12}, atest.stats)
