
def spawn(f):
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=f)
    p.start()
    return p


def child():
    @test(report=False, skip=False) # override default skip=True in spawned process
    def in_child():
        print("I am a happy child", flush=True)
        assert 1 == 1


if is_main_process:
    @test
    def stdout_capture():
        name = "Erik"
        msgs = []
        sys_stdout = sys.stdout
        sys_stderr = sys.stderr

        @test(report=False)
        def capture_all(stdout, stderr):
            print(f"Hello {name}!", file=sys.stdout)
            print(f"Bye {name}!", file=sys.stderr)
            msgs.extend([stdout.getvalue(), stderr.getvalue()])
            test.ne(sys_stdout, sys.stdout)
            test.ne(sys_stderr, sys.stderr)
        test.eq("Hello Erik!\n", msgs[0])
        test.eq("Bye Erik!\n", msgs[1])
        test.eq(sys_stdout, sys.stdout)
        test.eq(sys_stderr, sys.stderr)


if is_main_process:
    @test
    def capture_stdout_child_processes(stdout):
        def f():
            @test(report=False, skip=False)
            def in_child():
                print("hier ben ik")
                assert 1 == 1
        p = multiprocessing.Process(target=f) # NB: forks
        p.start()
        p.join(1)
        assert "hier ben ik\n" in stdout.getvalue()


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


    @test
    def import_submodule(stdout, stderr):
        from autotest.tests.sub_module_ok import marker
        test.eq('', stderr.getvalue())
        m = stdout.getvalue()
        test.contains(m, "sub_module_ok")
        test.contains(m, "test_one")
        test.contains(m, "I am a happy submodule")


    try:
        with test.stdout as s:
            @test(report=True) # force report, as might be suppressed in other context
            def import_submodule_is_silent_but_does_report_failures():
                import autotest.tests.sub_module_fail
        test.fail("Should have failed.")
    except AssertionError as e:
        m = s.getvalue()
        test.eq("('eq', 123, 42)", str(e))
        test.contains(m, "autotest.tester  \033[1mimport_submodule_is_silent_but_does_report_failures\033[0m")
        test.contains(m, "sub_module_fail  \033[1mtest_one\033[0m")


    @test
    def import_syntax_error_(stderr):
        """ what does this test??? (Thijs weet het niet)"""
        p = spawn(import_syntax_error)
        p.join(5)
        test.eq(1, p.exitcode)
        test.contains(stderr.getvalue(), "SyntaxError")
        test.contains(stderr.getvalue(), "syntax error")


    with test.opts(report=True):
        with test.stdout as s:
            try:
                @test(timeout=0.1)
                async def timeouts_test():
                    await asyncio.sleep(1)
                    assert False, "should have raised timeout"
            except asyncio.TimeoutError as e:
                assert "Hanging task (1 of 1)" in str(e), e
                msg = s.getvalue()
                assert "timeouts_test" in msg, msg
                tb = traceback.format_tb(e.__traceback__)
                assert "await asyncio.sleep(1)" in tb[-1], tb[-1]


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


