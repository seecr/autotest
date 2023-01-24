## begin license ##
#
# "Autotest": a simpler test runner for python
#
# Copyright (C) 2022 Seecr (Seek You Too B.V.) https://seecr.nl
#
# This file is part of "Autotest"
#
# "Autotest" is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "Autotest" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with "Autotest".  If not, see <http://www.gnu.org/licenses/>.
#
## end license ##

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
            try:
                asyncio.get_running_loop()
                loop=True
            except RuntimeError as e:
                loop=False # avoid this exception to be reported
            if not loop:
                asyncio.run(with_options(), debug=runner.option_get('debug', True))
            else:
                thread(
                    asyncio.run,
                    with_options(),
                    debug=runner.option_get('debug', True)
                    ).result()
        else:
            return coro_or_result
    return may_be_async


def async_test(self_test):
    import threading
    main = threading.current_thread()
    with self_test.child(hooks=(async_hook,)) as atest:

        atest.eq({'found': 0, 'run': 0}, atest.stats)
        loops = [None, None, None, None]
        threads = [None, None, None, None]

        @atest
        async def an_async_test():
            loops[0] = asyncio.get_running_loop()
            threads[0] = threading.current_thread()

        @atest
        async def parent():
            loops[1] = asyncio.get_running_loop()
            threads[1] = threading.current_thread()
            @atest
            async def nested_async_test():
                loops[2] = asyncio.get_running_loop()
                threads[2] = threading.current_thread()
            @atest
            def nested_sync_test():
                loops[3] = asyncio.get_running_loop()
                threads[3] = threading.current_thread()

            assert threads[0] == main
            assert threads[1] == main
            assert threads[2] != main
            assert threads[3] == main
            assert len(loops) ==  4
            assert loops[0]
            assert loops[1]
            assert loops[1] != loops[0]
            assert loops[2]
            assert loops[2] != loops[0] # each async test has own loop
            assert loops[2] != loops[1]
            assert loops[3] == loops[1] # still same loop


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
