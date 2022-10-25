
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
import difflib          # show diffs on failed tests with two args
import pprint           # show diffs on failed tests with two args
import collections      # chain maps for hierarchical Runners and Counter
import operator         # not
import logging          # output to logger


from utils import is_main_process


__all__ = ['getTester']


defaults = dict(
    skip      = not is_main_process or __name__ == '__mp_main__',
    keep      = False,           # Ditch test functions after running
    timeout   = 2,               # asyncio task timeout
    debug     = True,            # use debug for asyncio.run
    format    = pprint.pformat,  # set a function for formatting diffs on failures
)


class Runner: # aka Tester
    """ Main tool for running tests across modules and programs. """

    def __call__(self, *functions, **options):
        """Runs tests, with options; usefull as decorator. """
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


    def diff(self, a, b, ff=None):
        """ Produces diff of textual representation of a and b. """
        ff = ff if ff else self._options.get('format')
        return '\n' + '\n'.join(
                    difflib.ndiff(
                        format(a).splitlines(),
                        format(b).splitlines()))


    @staticmethod
    def diff2(a, b):
        """ experimental, own formatting more suitable for difflib
            deprecated, use option 'format=autotest.pformat' """
        import autotest.prrint as prrint # late import so prrint can use autotest itself
        return Tester.diff(a, b, ff=prrint.format)


    def prrint(self, a):
        import autotest.prrint as prrint # late import so prrint can use autotest itself
        prrint.prrint(a)


    def __init__(self, name=None, parent=None, **options):
        """ Not to be instantiated directly, use autotest.getTester() instead """
        self._parent = parent
        if parent:
            self._name = parent._name + '.' + name if name else parent._name
            self._fixtures = parent._fixtures.new_child()
            self._options = parent._options.new_child(m=options)
        else:
            self._name = name
            self._fixtures = collections.ChainMap()
            self._options = collections.ChainMap(options, defaults)
        self._stats = collections.Counter()
        self._loghandlers = []


    def _hooks(self):
        for m in self._options.maps:
            for hook in reversed(m.get('hooks', ())):
                yield hook


    def _stat(self, key):
        self._stats[key] += 1
        if p := self._parent:
            p._stat(key)


    def _create_logrecord(self, f, msg):
        return logging.LogRecord(
            self._name,                               # name of logger
            self._options.get('level', 40),               # log level   #TODO
            f.__code__.co_filename if f else None,    # source file where test is
            f.__code__.co_firstlineno if f else None, # line where test is
            msg,                                      # message
            (),                                       # args (passed to message.format)
            None,                                     # exc_info
            f.__name__ if f else None,                # name of the function invoking test.<op>
            None                                      # text representation of stack
            )


    def _run(self, test_func, *app_args, **app_kwds):
        """ Binds f to stack vars and fixtures and runs it. """
        AUTOTEST_INTERNAL = 1
        self._stat('found')
        skip = self._options.get('skip')
        orig_test_func = test_func
        for hook in self._hooks():
            test_func = hook(self, test_func)
            if not test_func:
                break
        if test_func:
            if inspect.isfunction(skip) and not skip(test_func) or not skip:
                self._stat('run')
                self.handle(self._create_logrecord(orig_test_func, orig_test_func.__qualname__))
                test_func(*app_args, **app_kwds)
        return orig_test_func if self._options.get('keep') else None


    def _log_stats(self):
        self.handle(self._create_logrecord(None, ', '.join(f'{k}: {v}' for k, v in self._stats.most_common())))


    def __getattr__(self, name):
        for hook in self._hooks():
            try:
                return hook.lookup(self, name)
            except AttributeError:
                pass


from operators import OperatorLookup

self_test = Runner('autotest-self-tests',
        hooks=(OperatorLookup(),)) # separate runner for bootstrapping/self testing


from numbers import Number
class any_number:
    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
    def __eq__(self, rhs):
        return isinstance(rhs, Number) and self._lo <= rhs <= self._hi


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


#@self_test  # test no longer modifies itself but creates a temporary runner when options are given
def combine_test_with_options():
    trace = []
    @self_test(keep=True, my_opt=42)
    def f0():
        trace.append(self_test._options.get('my_opt'))
    def f1(): # ordinary function; the only difference, here(!) is binding
        trace.append(self_test._options.get('my_opt'))
    self_test(f0)
    self_test(f0, my_opt=76)
    self_test(f0, f1, my_opt=93)
    assert [None, None, 76, 93, 93] == trace, trace
    # TODO make options available in test when specificed in @self_test
    # (@self_test(**options) makes a one time anonymous context)
    # maybe an optional argument 'context' which tests can declare??



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


from fixtures import WithFixtures, fixtures_tests


from utils import bind_1_frame_back # redefine placeholder
def binder(self, func):
    return bind_1_frame_back(func)
self_test2 = self_test.getChild(hooks=(
    WithFixtures,
    binder
    )
)

assert len(self_test._options.get('hooks')) == 1
assert len(self_test2._options.get('hooks')) == 2
hooks = list(self_test2._hooks())
assert hooks[0] == binder, hooks[0]
assert hooks[1] == WithFixtures, hooks[1]
assert hooks[2].__class__ == OperatorLookup, hooks[2]
assert len(hooks) == 3

fixtures_tests(self_test2)


with self_test2.child() as tst:
    @tst
    def nested_defaults():
        dflts0 = tst._options
        assert not dflts0['skip']
        with tst.child(skip=True) as tstk:
            dflts1 = tstk._options
            assert dflts1['skip']
            @tstk
            def fails_but_is_skipped():
                tstk.eq(1, 2)
            try:
                with tstk.child(skip=False) as TeSt:
                    dflts2 = TeSt._options
                    assert not dflts2['skip']
                    @TeSt
                    def fails_but_is_not_reported():
                        TeSt.gt(1, 2)
                    tst.fail()
                assert tstk._options == dflts1
            except AssertionError as e:
                tstk.eq("('gt', 1, 2)", str(e))
        assert tst._options == dflts0

    @tst
    def shorthand_for_new_context_without_options():
        """ Use this for defining new/tmp fixtures. """
        ctx0 = tst
        with tst.child() as t:
            assert ctx0 != t
            assert ctx0._fixtures == t._fixtures
            assert ctx0._options == t._options
            @t.fixture
            def new_one():
                yield 42
            assert ctx0._fixtures != t._fixtures
            assert "new_one" in t._fixtures
            assert "new_one" not in ctx0._fixtures


# define, test and install default fixture
# do not use @test.fixture as we need to install this twice
def raises(exception=Exception, message=None):
    AUTOTEST_INTERNAL = 1
    try:
        yield
    except exception as e:
        if message and message != str(e):
            raise AssertionError(f"should raise {exception.__name__} with message '{message}'") from e
    except BaseException as e:
        raise AssertionError(f"should raise {exception.__name__} but raised {type(e).__name__}").with_traceback(e.__traceback__) from e
    else:
        e = AssertionError(f"should raise {exception.__name__}")
        e.__suppress_context__ = True
        raise e
self_test2.fixture(raises)


@self_test2
def assert_raises():
    with self_test2.raises:
        raise Exception
    try:
        with self_test2.raises:
            pass
    except AssertionError as e:
        assert 'should raise Exception' == str(e), e


@self_test2
def assert_raises_specific_exception():
    with self_test2.raises(KeyError):
        raise KeyError
    try:
        with self_test2.raises(KeyError):
            raise RuntimeError('oops')
    except AssertionError as e:
        assert 'should raise KeyError but raised RuntimeError' == str(e), str(e)
    try:
        with self_test2.raises(KeyError):
            pass
    except AssertionError as e:
        assert 'should raise KeyError' == str(e), e


@self_test2
def assert_raises_specific_message():
    with self_test2.raises(RuntimeError, "hey man!"):
        raise RuntimeError("hey man!")
    try:
        with self_test2.raises(RuntimeError, "hey woman!"):
            raise RuntimeError("hey man!")
    except AssertionError as e:
        assert "should raise RuntimeError with message 'hey woman!'" == str(e)


class binding_context:
    a = 42
    @self_test2(keep=True)
    def one_test():
        assert a == 42
    a = 43
    #one_test()


@self_test2
def access_closure_from_enclosing_def():
    a = 46              # accessing this in another function makes it a 'freevar', which needs a closure
    @self_test2
    def access_a():
        assert 46 == a


f = 16

@self_test2
def dont_confuse_app_vars_with_internal_vars():
    assert 16 == f


@self_test2
def idea_for_dumb_diffs():
    # if you want a so called smart diff: the second arg of assert is meant for it.
    # Runner supplies a generic diff between two pretty printed values
    a = [7, 1, 2, 8, 3, 4]
    b = [1, 2, 9, 3, 4, 6]
    d = self_test2.diff(a, b)
    assert str == type(str(d))

    try:
        assert a == b, self_test2.diff(a,b)
    except AssertionError as e:
        assert """
- [7, 1, 2, 8, 3, 4]
?  ---      ^

+ [1, 2, 9, 3, 4, 6]
?        ^      +++
""" == str(e)


    try:
        self_test2.eq(a, b, diff=self_test2.diff)
    except AssertionError as e:
        assert """
- [7, 1, 2, 8, 3, 4]
?  ---      ^

+ [1, 2, 9, 3, 4, 6]
?        ^      +++
""" == str(e), e


    # use of a function from elswhere
    a = set([7, 1, 2, 8, 3, 4])
    b = set([1, 2, 9, 3, 4, 6])
    try:
        assert a == b, set.symmetric_difference(a, b)
    except AssertionError as e:
        assert "{6, 7, 8, 9}" == str(e), e
    try:
        self_test2.eq(a, b, diff=set.symmetric_difference)
    except AssertionError as e:
        assert "{6, 7, 8, 9}" == str(e), e


#@self_test2 # temp disable bc it cause system installed autotest import via diff2
def diff2_sorting_including_uncomparables():
    msg = """
  {
+   1,
-   <class 'dict'>:
?           ^^^   ^

+   <class 'str'>,
?           ^ +  ^

-     <class 'bool'>,
  }"""
    with self_test2.raises(AssertionError, msg):
        self_test2.eq({dict: bool}, {str, 1}, msg=self_test2.diff2)


@self_test2
def test_isinstance():
    self_test2.isinstance(1, int)
    self_test2.isinstance(1.1, (int, float))
    with self_test2.raises(AssertionError, "('isinstance', 1, <class 'str'>)"):
        self_test2.isinstance(1, str)
    with self_test2.raises(AssertionError, "('isinstance', 1, (<class 'str'>, <class 'dict'>))"):
        self_test2.isinstance(1, (str, dict))


@self_test2
def use_builtin():
    """ you could use any builtin as well; there are not that much useful options in module builtins though
        we could change the priority of lookups, now it is: operator, builtins, <arg[0]>. Maybe reverse? """
    self_test2.all([1,2,3])
    with self_test2.raises(AssertionError):
        self_test2.all([False,2,3])
    class A: pass
    class B(A): pass
    self_test2.issubclass(B, A)
    self_test2.hasattr([], 'append')


def any(_): # name is included in diffs
    return True


class wildcard:
    def __init__(self, f=any):
        self.f = f
    def __eq__(self, x):
        return bool(self.f(x))
    def __call__(self, f):
        return wildcard(f)
    def __repr__(self):
        return self.f.__name__  + '(...)' if self.f else '*'

self_test2.any = wildcard()


@self_test2
def wildcard_matching():
    self_test2.eq([self_test2.any, 42], [16, 42])
    self_test2.eq([self_test2.any(lambda x: x in [1,2,3]), 78], [2, 78])
    self_test2.ne([self_test2.any(lambda x: x in [1,2,3]), 78], [4, 78])


@self_test2
def assert_raises_as_fixture(raises:KeyError):
    {}[0]


@self_test2.fixture
def stringio_handler():
    import io
    s = io.StringIO()
    h = logging.StreamHandler(s)
    h.setFormatter(logging.Formatter(fmt="{name}-{levelno}-{pathname}-{lineno}-{message}-{exc_info}-{funcName}-{stack_info}", style='{'))
    yield s, h


class logging_handlers:

    @self_test2.fixture
    def logging_runner(stringio_handler, name='main'):
        s, myhandler = stringio_handler
        tester = Runner(name) # No hooks for operators and fixtures
        tester.addHandler(myhandler)
        yield tester, s


    @self_test2
    def tester_with_handler(logging_runner:'carl'):
        tester, s = logging_runner
        _line_ = inspect.currentframe().f_lineno
        @tester
        def a_test():
            assert 1 == 1  # hooks for operators not present
        log_msg = s.getvalue()
        qname = "logging_handlers.tester_with_handler.<locals>.a_test"
        self_test2.eq(log_msg, f"carl-40-{__file__}-{_line_+1}-{qname}-None-a_test-None\n")


    @self_test2
    def sub_tester_propagates(logging_runner:'main'):
        """ propagate to parent """
        main, s = logging_runner
        sub = main.getChild('sub1')
        _line_ = inspect.currentframe().f_lineno
        @sub
        def my_sub_test():
            assert 1 == 1  # hooks for operators not present
        log_msg = s.getvalue()
        self_test2.startswith(log_msg, f"main.sub1-")
        qname = "logging_handlers.sub_tester_propagates.<locals>.my_sub_test"
        self_test2.eq(log_msg, f"main.sub1-40-{__file__}-{_line_+1}-{qname}-None-my_sub_test-None\n")


    @self_test2
    def sub_tester_does_not_propagate(stringio_handler, logging_runner:'main'):
        main, main_s = logging_runner
        sub_s, subhandler = stringio_handler
        sub = main.getChild('sub1')
        sub.addHandler(subhandler)
        _line_ = inspect.currentframe().f_lineno
        @sub
        def my_sub_test():
            assert 1 == 1  # hooks for operators not present
        main_msg = main_s.getvalue()
        sub_msg = sub_s.getvalue()
        self_test2.startswith(sub_msg, f"main.sub1-")
        qname = "logging_handlers.sub_tester_does_not_propagate.<locals>.my_sub_test"
        self_test2.eq(sub_msg, f"main.sub1-40-{__file__}-{_line_+1}-{qname}-None-my_sub_test-None\n")
        self_test2.eq('', main_msg) # do not duplicate message in parent


    @self_test2.fixture
    def intercept():
        records = []
        root_logger = logging.getLogger()
        root_logger.addFilter(records.append) # intercept only
        try:
            yield records
        finally:
            root_logger.removeFilter(records.append)

    @self_test2
    def tester_delegates_to_root_logger(intercept):
        tester = Runner('free')
        @tester
        def my_output_goes_to_root_logger():
            assert 1 == 1 # hooks for operators not present
        self_test2.eq('my_output_goes_to_root_logger', intercept[0].funcName)


    @self_test2
    def tester_with_handler_failing(logging_runner:'esmee'):
        tester, s = logging_runner
        try:
            _line_ = inspect.currentframe().f_lineno
            @tester
            def a_failing_test():
                assert 1 == 2
        except AssertionError:
            pass
        log_msg = s.getvalue()
        qname = "logging_handlers.tester_with_handler_failing.<locals>.a_failing_test"
        self_test2.eq(log_msg, f"esmee-40-{__file__}-{_line_+1}-{qname}-None-a_failing_test-None\n")


@self_test2
def log_stats(intercept):
    with self_test2.child() as tst:
        @tst
        def one(): pass
        @tst
        def two(): pass
    tst.eq(3, len(intercept))
    msg = intercept[2].msg
    tst.contains(msg, "found: 2, run: 2")



from levels import Levels, testing_levels
testing_levels(self_test2)


@self_test2
def check_stats():
    self_test.eq({'found': 89, 'run': 83}, self_test._stats)

