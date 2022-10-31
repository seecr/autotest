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


    # We put this test last, as it captures output an thus fails when using print
    @test
    def reporting_tests(stdout):
        try:
            @test(report=False)
            def test_no_reporting_but_failure_raised():
                assert 1 != 1
            self.fail("should fail")
        except AssertionError as e:
            t, v, tb = sys.exc_info()
            tbi = traceback.extract_tb(tb)
            assert "test_no_reporting_but_failure_raised" == tbi[-1].name, tbi[-1].name
            assert "assert 1 != 1" == tbi[-1].line, repr(tbi[-1].line)
        m = stdout.getvalue()
        assert "" == m, m


        try:
            @test(report=True)
            def test_with_reporting_and_failure_raised():
                assert 1 != 1
            self.fail("should fail")
        except AssertionError:
            t, v, tb = sys.exc_info()
            tbi = traceback.extract_tb(tb)
            assert "test_with_reporting_and_failure_raised" == tbi[-1].name, tbi[-1].name
            assert "assert 1 != 1" == tbi[-1].line, repr(tbi[-1].line)
        m = stdout.getvalue()
        test.contains(m, "autotest.tester  \033[1mtest_with_reporting_and_failure_raised\033[0m")


