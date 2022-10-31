
""" Defines the Runner

    NB: a stand alone Runner (self_test) is created as to not interfere with
        the system wide default runner, which may not exists as long as we
        are bootstrapping.

    NB: Runner is tested using Runner itself. Runner is bootstrapped in steps,i
        starting with a simple Runner of which the capabilities are gradually
        extended.

"""

import inspect          # for recognizing functions, generators etc
import contextlib       # child as context
import collections      # chain maps for hierarchical Runners and Counter
import logging          # output to logger


defaults = dict(
    keep      = False,           # Ditch test functions after running
    run       = True,            # Run test immediately
)


class Runner: # aka Tester
    """ Main tool for running tests across modules and programs. """

    def __call__(self, *functions, **options):
        """Runs tests, with options; useful as decorator. """
        AUTOTEST_INTERNAL = 1
        if functions and options:
            return self(**options)(*functions)
        if options:
            return self.getChild(**options)
        for f in functions:
            result = self._run(f)
        return result


    def getChild(self, *_, **__):
        return Runner(*_, parent=self, **__)


    @contextlib.contextmanager
    def child(self, *_, **__):
        child = self.getChild(*_, **__)
        yield child
        child._log_stats()


    def addHandler(self, handler):
        self._loghandlers.append(handler)


    def handle(self, logrecord):
        if handlers := self._loghandlers:
            for h in handlers:
                h.handle(logrecord)
        elif self._parent:
            self._parent.handle(logrecord)
        else:
            logging.getLogger().handle(logrecord)


    def fail(self, *args, **kwds):
        raise AssertionError(*args, *(kwds,) if kwds else ())


    def __init__(self, name=None, parent=None, **options):
        """ Not to be instantiated directly, use autotest.getTester() instead """
        self._parent = parent
        if parent:
            self._name = parent._name + '.' + name if name and parent._name else name or parent._name
            self._options = parent._options.new_child(m=options)
        else:
            self._name = name
            self._options = collections.ChainMap(options, defaults)
        self.stats = collections.Counter()
        self._loghandlers = []


    def option_get(self, name, default=None):
        return self._options.get(name, default)


    def option_setdefault(self, name, default):
        return self._options.maps[0].setdefault(name, default)


    def option_enumerate(self, name):
        """ enumerate options from parents, flattening sequences """
        for m in self._options.maps:
            value = m.get(name)
            if isinstance(value, (list, tuple)):
                yield from reversed(value)
            elif value:
                yield value


    def _stat(self, key):
        self.stats[key] += 1
        if p := self._parent:
            p._stat(key)


    def _create_logrecord(self, f, msg):
        return logging.LogRecord(
            self._name,                               # name of logger
            self.option_get('level', 40),             # log level   #TODO
            f.__code__.co_filename if f else None,    # source file where test is
            f.__code__.co_firstlineno if f else None, # line where test is
            msg,                                      # message
            (),                                       # args (passed to message.format)
            None,                                     # exc_info
            f.__name__ if f else None,                # name of the function invoking test.<op>
            None                                      # text representation of stack
            )


    def _run(self, test_func, *app_args, **app_kwds):
        """ Runs hooks and and runs result. """
        AUTOTEST_INTERNAL = 1
        self._stat('found')
        orig_test_func = test_func
        for hook in self.option_enumerate('hooks'):
            if not (test_func := hook(self, test_func)):
                break
        else:
            if self.option_get('run'):
                self._stat('run')
                self.handle(self._create_logrecord(orig_test_func, orig_test_func.__qualname__))
                # TODO how to handle hook for diff?
                test_func(*app_args, **app_kwds)
        return orig_test_func if self.option_get('keep') else None


    def _log_stats(self):
        self.handle(self._create_logrecord(None, ', '.join(f'{k}: {v}' for k, v in self.stats.most_common())))


    def __getattr__(self, name):
        for hook in self.option_enumerate('hooks'):
            try:
                return hook.lookup(self, name)
            except AttributeError:
                pass

    def __str__(self):
        return f"<Runner {self._name!r}>"



from numbers import Number
class any_number:
    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
    def __eq__(self, rhs):
        return isinstance(rhs, Number) and self._lo <= rhs <= self._hi


from .operators import operators_hook

self_test = Runner('autotest-self-tests',
        hooks=(operators_hook,)) # separate runner for bootstrapping/self testing


@self_test
def example_test():
    assert any_number(1,10) == 1
    assert any_number(1,10) == 2
    assert any_number(1,10) == 5
    assert any_number(1,10) == 9
    assert any_number(1,10) == 10
    assert not any_number(9,15) == 1
    assert not any_number(9,15) == 8
    assert not any_number(9,15) == 16
    assert not any_number(9,15) == 33
    assert not any_number(9,15) == None
    assert not any_number(9,15) == "A"
    assert not any_number(9,15) == object


@self_test
def new_assert():
    assert self_test.eq(1, 1)
    assert self_test.lt(1, 2) # etc

    try:
        self_test.eq(1, 2)
        self_test.fail()
    except AssertionError as e:
        assert ('eq', 1, 2) == e.args, e.args

    try:
        self_test.ne(1, 1)
        self_test.fail()
    except AssertionError as e:
        assert "('ne', 1, 1)" == str(e), str(e)


@self_test
def test_fail():
    try:
        self_test.fail("plz fail")
        raise AssertionError('FAILED to fail')
    except AssertionError as e:
        assert "plz fail" == str(e), e

    try:
        self_test.fail("plz fail", info="more")
        raise AssertionError('FAILED to fail')
    except AssertionError as e:
        assert "('plz fail', {'info': 'more'})" == str(e), e


@self_test
def not_operator():
    self_test.comp.contains("abc", "d")
    try:
        self_test.comp.contains("abc", "b")
        self_test.fail()
    except AssertionError as e:
        assert "('contains', 'abc', 'b')" == str(e), e


@self_test
def call_method_as_operator():
    self_test.endswith("aap", "ap")


# self_test.<op> is really just assert++ and does not need @test to run 'in'
self_test.eq(1, 1)

try:
    self_test.lt(1, 1)
except AssertionError as e:
    assert "('lt', 1, 1)" == str(e), str(e)


@self_test
def test_testop_has_args():
    try:
        @self_test
        def nested_test_with_testop():
            x = 42
            y = 67
            self_test.eq(x, y)
            return True
    except AssertionError as e:
        assert hasattr(e, 'args')
        assert 3 == len(e.args)
        self_test.eq("('eq', 42, 67)", str(e))


@self_test
def combine_test_with_options():
    trace = []
    with self_test.child(keep=True, my_opt=42) as tst:
        @tst
        def f0():
            trace.append(tst.option_get('my_opt'))
        def f1():
            trace.append(tst.option_get('my_opt'))
        # at this point, f0 and f1 are equivalent; @tst does not modify f0
        # so when run, a new test context must be provided:
        self_test(f0)
        self_test(f0, my_opt=76)
        self_test(f0, f1, my_opt=93)
        # however, both f0 and f1 have no access to this context
        #assert [None, None, 76, 93, 93] == trace, trace
        assert [42, 42, 42, 42, 42] == trace, trace

        """ TODO replace static reference to tst with dynamic args 'test'?
        @tst
        def f(test):
            # here test refers to tst
            pass
        """


@self_test
def test_calls_other_test():
    @self_test(keep=True)
    def test_a():
        assert 1 == 1
        return True
    @self_test.child()
    def test_b():
        assert test_a()


# IDEA/TODO
# I make this mistake a lot:
# Instead of @test(option=Value)
# I write def test(open=Value)
# I think supporting both is a good idea


@self_test
def call_test_with_complete_context():
    @self_test(keep=True)
    def a_test():
        assert True
    assert a_test, a_test
    self_test(a_test)



def stringio_handler():
    import io
    s = io.StringIO()
    h = logging.StreamHandler(s)
    h.setFormatter(logging.Formatter(fmt="{name}-{levelno}-{pathname}-{lineno}-{message}-{exc_info}-{funcName}-{stack_info}", style='{'))
    return s, h


def logging_runner(name):
    s, myhandler = stringio_handler()
    tester = Runner(name) # No hooks
    tester.addHandler(myhandler)
    return tester, s


@contextlib.contextmanager
def intercept():
    records = []
    root_logger = logging.getLogger()
    root_logger.addFilter(records.append) # intercept only
    try:
        yield records
    finally:
        root_logger.removeFilter(records.append)


class logging_handlers:

    @self_test
    def tester_with_handler():
        tester, s = logging_runner('carl')
        _line_ = inspect.currentframe().f_lineno
        @tester
        def a_test():
            assert 1 == 1  # hooks for operators not present
        log_msg = s.getvalue()
        qname = "logging_handlers.tester_with_handler.<locals>.a_test"
        self_test.eq(log_msg, f"carl-40-{__file__}-{_line_+1}-{qname}-None-a_test-None\n")


    @self_test
    def sub_tester_propagates():
        """ propagate to parent """
        main, s = logging_runner('main')
        sub = main.getChild('sub1')
        _line_ = inspect.currentframe().f_lineno
        @sub
        def my_sub_test():
            assert 1 == 1  # hooks for operators not present
        log_msg = s.getvalue()
        self_test.startswith(log_msg, f"main.sub1-")
        qname = "logging_handlers.sub_tester_propagates.<locals>.my_sub_test"
        self_test.eq(log_msg, f"main.sub1-40-{__file__}-{_line_+1}-{qname}-None-my_sub_test-None\n")


    @self_test
    def sub_tester_does_not_propagate():
        main, main_s = logging_runner('main')
        sub_s, subhandler = stringio_handler()
        sub = main.getChild('sub1')
        sub.addHandler(subhandler)
        _line_ = inspect.currentframe().f_lineno
        @sub
        def my_sub_test():
            assert 1 == 1  # hooks for operators not present
        main_msg = main_s.getvalue()
        sub_msg = sub_s.getvalue()
        self_test.startswith(sub_msg, f"main.sub1-")
        qname = "logging_handlers.sub_tester_does_not_propagate.<locals>.my_sub_test"
        self_test.eq(sub_msg, f"main.sub1-40-{__file__}-{_line_+1}-{qname}-None-my_sub_test-None\n")
        self_test.eq('', main_msg) # do not duplicate message in parent


    @self_test
    def tester_delegates_to_root_logger():
        with intercept() as i:
            tester = Runner('free')
            @tester
            def my_output_goes_to_root_logger():
                assert 1 == 1 # hooks for operators not present
            self_test.eq('my_output_goes_to_root_logger', i[0].funcName)


    @self_test
    def tester_with_handler_failing():
        tester, s = logging_runner('esmee')
        try:
            _line_ = inspect.currentframe().f_lineno
            @tester
            def a_failing_test():
                assert 1 == 2
        except AssertionError:
            pass
        log_msg = s.getvalue()
        qname = "logging_handlers.tester_with_handler_failing.<locals>.a_failing_test"
        self_test.eq(log_msg, f"esmee-40-{__file__}-{_line_+1}-{qname}-None-a_failing_test-None\n")


    @self_test
    def log_stats():
        with intercept() as i:
            with self_test.child() as tst:
                @tst
                def one(): pass
                @tst
                def two(): pass
            tst.eq(3, len(i))
            msg = i[2].msg
            tst.contains(msg, "found: 2, run: 2")

    @self_test
    def run_or_not():
        @self_test(run=False)
        def not_run():
            assert "do not" == "run me"
