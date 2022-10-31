import multiprocessing
import asyncio
import sys


is_main_process = multiprocessing.current_process().name == "MainProcess"


def spawn(f):
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=f)
    p.start()
    return p


def child():
    @test
    def in_child():
        print("I am a happy child", flush=True)
        assert 1 == 1


def integration_test(test):

    @test
    def import_submodule(stdout, stderr):
        from autotest.tests.sub_module_ok import marker
        test.eq('test_one\n', stderr.getvalue())
        test.eq("TESTER: <Runner 'autotest.tests.sub_module_ok'>\n", stdout.getvalue())


    try:
        with test.stdout as s, test.stderr as r:
            @test
            def import_submodule_failure():
                from .tests import sub_module_fail
        test.fail("Should have failed.")
    except AssertionError as e:
        test.eq("fail I will", str(e))
        m = r.getvalue()
        test.eq(m, "integration_test.<locals>.import_submodule_failure\ntiedeldom\n")


    if is_main_process:
        @test
        def silence_child_processes(stdout, stderr):
            p = spawn(child) # <= causes import of all current modules
            p.join(3)
            # if it didn't load (e.g. SyntaxError), do not run test to avoid
            # failures introduced by other modules that loaded as a result
            # of multiprocessing spawn, but failed
            if p.exitcode == 0:
                out = stdout.getvalue()
                test.contains(out, "I am a happy child")
                test.not_("in_child" in out)


        with test.raises(AssertionError, "Use combine:3 instead of combine=3"):
            @test
            def fixture_args_as_annotain_iso_defaul(combine=3):
                """ fixture args are not default args: '=' instead of ':' raises error """
                pass

        @test.fixture
        def async_combine(a):
            yield a

        with test.raises(AssertionError, "Use async_combine:3 instead of async_combine=3"):
            @test
            async def fixture_args_as_annotain_iso_defaul(async_combine=3):
                """ fixture args are not default args: '=' instead of ':' raises error """
                pass


    @test
    def combine_test_with_options_and_test_arg():
        trace = []
        with test.child(keep=True, my_opt=42) as tst:
            @tst
            def f0(test):
                trace.append(test.option_get('my_opt'))
            def f1(test):
                trace.append(test.option_get('my_opt'))
            # at this point, f0 and f1 are equivalent; @tst does not modify f0
            # so when run, a new test context must be provided:
            test(f0)
            test(f0, my_opt=76)
            test(f0, f1, my_opt=93)
            assert [42, None, 76, 93, 93] == trace, trace


    class async_fixtures:
        """ here because async fixtures need async tests """

        @test.fixture
        async def my_async_fixture():
            await asyncio.sleep(0)
            yield 'async-42'


        @test
        async def async_fixture_with_async_with():
            async with test.my_async_fixture as f:
                assert 'async-42' == f
            try:
                with test.my_async_fixture:
                    assert False
                test.fail()
            except Exception as e:
                test.startswith(str(e), "Use 'async with' for ")


        @test
        async def async_fixture_as_arg(my_async_fixture):
            test.eq('async-42', my_async_fixture)


        try:
            @test
            def only_async_funcs_can_have_async_fixtures(my_async_fixture):
                assert False, "not possible"
        except AssertionError as e:
            test.eq(f"function 'only_async_funcs_can_have_async_fixtures' cannot have async fixture 'my_async_fixture'.", str(e))


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
        async def fixtures_with_2_args_async(area:(3,2)):
            test.eq(28.27, area)


    class tmp_files:

        path = [None]

        @test
        def temp_sync(tmp_path):
            assert tmp_path.exists()

        @test
        def temp_file_removal(tmp_path):
            path[0] = tmp_path / 'aap'
            path[0].write_text("hello")

        @test
        def temp_file_gone():
            assert not path[0].exists()

        @test
        async def temp_async(tmp_path):
            assert tmp_path.exists()

        @test
        def temp_dir_with_file(tmp_path:'aap'):
            assert str(tmp_path).endswith('/aap')
            tmp_path.write_text('hi monkey')
            assert tmp_path.exists()

