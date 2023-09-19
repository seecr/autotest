## begin license ##
#
# "Autotest": a simpler test runner for python
#
# Copyright (C) 2022-2023 Seecr (Seek You Too B.V.) https://seecr.nl
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

import multiprocessing
import asyncio


def spawn(f):
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=f)
    p.start()
    return p


def fork(f):
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=f)
    p.start()
    return p


def subprocess():
    """when spawned, imports selftest freshly, potentially running al tests
        but this is prevented by default, use subprocess=True when needed
    when forked, use existing loaded selftest, not running tests
        so no special measures are taken and both tests run
    """
    import selftest

    test = selftest.get_tester("subprocess")

    @test
    def one_test_not_run():
        print("In a forked child, I run.")

    @test(subprocess=True)
    def one_does_run():
        print("In any child, I run.")


def integration_test(test):
    @test
    def import_submodule(stdout, stderr):
        from selftest.tests.sub_module_ok import marker

        test.eq("TESTER: <Tester 'selftest.tests.sub_module_ok'>\n", stdout.getvalue())
        test.eq("UNIT:selftest.tests.sub_module_ok.test_one\n", stderr.getvalue())

    try:
        with test.stdout as s, test.stderr as r:

            @test
            def import_submodule_failure():
                from .tests import sub_module_fail

        test.fail("Should have failed.")
    except AssertionError as e:
        test.eq("fail I will", str(e))
        m = r.getvalue()
        test.eq(
            m,
            "INTEGRATION:selftest.integrationtests.integration_test.import_submodule_failure\nUNIT:selftest.tests.sub_module_fail.tiedeldom\n",
        )

    with test.raises(AssertionError, "Use combine:3 instead of combine=3"):

        @test
        def fixture_args_as_annotain_iso_defaul(combine=3):
            """fixture args are not default args: '=' instead of ':' raises error"""
            pass

    @test(bind=True)
    def by_default_do_not_run_in_spawned_processes(stdout, stderr):
        spawn(subprocess).join()
        test.contains(stderr.getvalue(), "subprocess.one_does_run")
        test.eq("In any child, I run.\n", stdout.getvalue())

    @test(bind=True)
    def test_do_not_run_in_child_processes(stdout):
        fork(subprocess).join()
        test.eq("In a forked child, I run.\nIn any child, I run.\n", stdout.getvalue())

    @test.fixture
    def async_combine(a):
        yield a

    with test.raises(AssertionError, "Use async_combine:3 instead of async_combine=3"):

        @test
        async def fixture_args_as_annotain_iso_defaul(async_combine=3):
            """fixture args are not default args: '=' instead of ':' raises error"""
            pass

    @test
    def combine_test_with_options_and_test_arg():
        trace = []
        with test.child(keep=True, my_opt=42) as tst:

            @tst
            def f0(test):
                trace.append(test.option_get("my_opt"))

            def f1(test):
                trace.append(test.option_get("my_opt"))

            # at this point, f0 and f1 are equivalent; @tst does not modify f0
            # so when run, a new test context must be provided:
            test(f0)
            test(f0, my_opt=76)
            test(f0, f1, my_opt=93)
            assert [42, None, 76, 93, 93] == trace, trace

    class async_fixtures:
        """here because async fixtures need async tests"""

        @test.fixture
        async def my_async_fixture():
            await asyncio.sleep(0)
            yield "async-42"

        @test
        async def async_fixture_with_async_with():
            async with test.my_async_fixture as f:
                assert "async-42" == f
            try:
                with test.my_async_fixture:
                    assert False
                test.fail()
            except Exception as e:
                test.startswith(str(e), "Use 'async with' for ")

        @test
        async def async_fixture_as_arg(my_async_fixture):
            test.eq("async-42", my_async_fixture)

        try:

            @test
            def only_async_funcs_can_have_async_fixtures(my_async_fixture):
                assert False, "not possible"

        except AssertionError as e:
            test.eq(
                f"function 'only_async_funcs_can_have_async_fixtures' cannot have async fixture 'my_async_fixture'.",
                str(e),
            )

        @test.fixture
        async def with_nested_async_fixture(my_async_fixture):
            yield f">>>{my_async_fixture}<<<"

        @test
        async def async_test_with_nested_async_fixtures(with_nested_async_fixture):
            test.eq(">>>async-42<<<", with_nested_async_fixture)

        @test
        async def mix_async_and_sync_fixtures(fixture_C, with_nested_async_fixture):
            test.eq(">>>async-42<<<", with_nested_async_fixture)
            test.eq(252, fixture_C)

        @test.fixture
        def area(r, d=1):
            import math

            yield round(math.pi * r * r, d)

        @test
        async def fixtures_with_2_args_async(area: (3, 2)):
            test.eq(28.27, area)

    class tmp_files:
        path = [None]

        @test
        def temp_sync(tmp_path):
            assert tmp_path.exists()

        @test(bind=True)
        def temp_file_removal(tmp_path):
            path[0] = tmp_path / "aap"
            path[0].write_text("hello")

        @test(bind=True)
        def temp_file_gone():
            assert not path[0].exists()

        @test(bind=True)
        async def temp_async(tmp_path):
            assert tmp_path.exists()

        @test
        def temp_dir_with_file(tmp_path: "aap"):
            assert str(tmp_path).endswith("/aap")
            tmp_path.write_text("hi monkey")
            assert tmp_path.exists()
