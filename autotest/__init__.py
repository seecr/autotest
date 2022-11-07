## begin license ##
#
# "Autotest": a simpler test runner for python
#
# Copyright (C) 2021-2022 Seecr (Seek You Too B.V.) https://seecr.nl
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


"""
    The structure of autotests for bootstrapping.
    ---------------------------------------------
    * tester.py contains the main Tester without any hooks, this module tests itself using a
      separate Tester called self_test. We reuse self_test to incrementally run the tests for
      the hooks.
    * self_test contains one hook: operator, in order to make testing easier. However, a
      small mistake in operator might cause all tests to fail since Tester and operator
      mutually depend on each other.
    * After all hooks have been tested, we assemble the final root Tester and tests if
      all hooks work properly.
    * Run tests for autotest itself by:
       $ python -c "import autotest" autotest.selftest
    * Integration tests are run with a partially initialized __init__.py, but it works
    * Finally we test the setup

"""


import os


from .tester import Tester, self_test
@self_test
def assert_stats():
    assert {'found': 23, 'run': 22} == self_test.stats, self_test.stats


from .levels import levels_hook, levels_test
levels_test(self_test)

from .prrint import prrint_test
prrint_test(self_test)

from .filter import filter_hook, filter_test
filter_test(self_test)

from .wildcard import wildcard_hook, wildcard_test
wildcard_test(self_test)

from .binder import binder_hook, binder_test
binder_test(self_test)

from .operators import operators_hook, operators_test
operators_test(self_test)

from .fixtures import fixtures_hook, fixtures_test, std_fixtures
fixtures_test(self_test)

from .asyncer import async_hook, async_test
async_test(self_test)

from .diffs import diff_hook, diff_test
diff_test(self_test)



@self_test
def check_stats():
    self_test.eq({'found': 112, 'run': 100}, self_test.stats)


def assemble_root_runner(**options):
    return Tester(
        # order of hook matters, processed from right to left
        hooks = [
            operators_hook,
            async_hook,
            fixtures_hook,
            diff_hook,
            levels_hook,
            wildcard_hook,
            binder_hook,
            filter_hook
            ],
        fixtures = std_fixtures,
        **options,
    )


def root_tester_assembly_test(test):
    """ only test if the root tester is assembled with all hooks """

    N = [0]

    # operators_hook
    test.eq(1, 1)
    test.isinstance({}, dict)
    test.endswith("aap", "ap")

    # fixtures hook
    @test.fixture
    def forty_two():
        yield 42
    @test
    def is_forty_two(forty_two):
        assert forty_two == 42
        N[0] += 1
    with test.forty_two as contextmanager:
        assert contextmanager == 42

    # standard fixtures
    with test.tmp_path as p:
        test.truth(p.exists())
    with test.stdout as e:
        print("one")
        assert e.getvalue() == 'one\n'
    with test.stderr as e:
        import sys
        print("two", file=sys.stderr)
        assert e.getvalue() == 'two\n'

    # binding hook
    class A:
        a = 42
        @test(bind=True)
        def bind():
            assert a == 42
            N[0] += 1

    # wildcard hook
    test.eq(test.any(int), 42)

    # levels hook
    from .levels import UNIT
    with test.child(level=UNIT) as tst:
        @tst.performance
        def performance_test():
            assert "not" == "executed"
        @tst.critical
        def critical_test():
            assert 1 == 1
            N[0] += 1
        assert {'found': 2, 'run': 1} == tst.stats, tst.stats

    # async hook (elaborate on nested stuff)
    @test.fixture
    async def nine():
        yield ['borg 1', 'borg 2', 'borg 3', 'borg 4', 'borg 5', 'borg 6', 'Annika Hansen', 'borg 8', 'borg 9']
    @test.fixture
    async def seven_of_nine(nine):
        yield nine[7-1]
    @test
    async def the_9(nine):
        assert len(nine) == 9
        N[0] += 1
    @test
    async def is_seven_of_nine(seven_of_nine):
        assert seven_of_nine == 'Annika Hansen'
        N[0] += 1

    assert N[0] == 5, N
    assert dict(found=6, run=5) == test.stats, test.stats

    # diff hook
    try:
        test.eq(1, 2, diff=test.diff)
    except AssertionError as e:
        assert str(e) == '\n- 1\n+ 2'
    try:
        test.eq(1, 2, diff=test.diff2)
    except AssertionError as e:
        assert str(e) == '\n- 1\n+ 2'

    # filter hook
    @test(filter='aa')
    def moon():
        test.fail()
    r = [0]
    @test(filter='aa')
    def maan():
        r[0] = 1
    assert r == [1]


testers = {} # initial, for more testing


def basic_config(**options):
    assert None not in testers
    testers[None] = assemble_root_runner(**options)


def get_tester(name=None):
    if None not in testers:
        testers[None] = assemble_root_runner()
    if name in testers:
        return testers[name]
    tester = testers[None]
    for namepart in name.split('.'):
        tester = tester.getChild(namepart)
        testers[tester._name] = tester
    return tester


@self_test
def set_root_opts():
    basic_config(filter='aap')
    root = get_tester()
    self_test.eq({'filter', 'fixtures', 'hooks', 'keep', 'run'}, set(root._options.keys()))
    self_test.eq('aap', root._options['filter'])


testers = {} # final


@self_test
def get_root_tester():
    root = get_tester()
    assert isinstance(root, Tester)
    root1 = get_tester()
    assert root1 is root


@self_test
def get_sub_tester():
    root = get_tester()
    mymodule = get_tester('my.module')
    assert mymodule._name == 'my.module'
    my = get_tester('my')
    assert my._name == 'my'
    assert my._parent is root
    assert mymodule._parent is my
    mymodule1 = get_tester('my.module')
    assert mymodule1 is mymodule


@self_test
def run_integration_tests():
    root_tester_assembly_test(assemble_root_runner())
    from .integrationtests import integration_test
    integration_test(assemble_root_runner().integration)



@self_test
def setup_correct():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as p:
        tmp = pathlib.Path(p)
        autotest_dev_dir = pathlib.Path(__file__).parent.resolve().parent
        if not (autotest_dev_dir/'bin/autotest').exists():
            # Not dev dir
            return
        import subprocess
        version_process = subprocess.run(['python3', 'setup.py', '--version'],
                capture_output=True,
                text=True,
                cwd=str(autotest_dev_dir),
            )
        version = version_process.stdout.strip()
        result = subprocess.run(['python3', 'setup.py', 'sdist', '--dist-dir', str(tmp)],
                capture_output=True,
                cwd=str(autotest_dev_dir))

        from tarfile import open
        tf = open(name=tmp/f'autotest-{version}.tar.gz', mode='r:gz')
        self_test.eq([
             f'autotest-{version}',
             f'autotest-{version}/LICENSE',
             f'autotest-{version}/MANIFEST.in',
             f'autotest-{version}/PKG-INFO',
             f'autotest-{version}/README.rst',
             f'autotest-{version}/autotest',
             f'autotest-{version}/autotest.egg-info',
             f'autotest-{version}/autotest.egg-info/PKG-INFO',
             f'autotest-{version}/autotest.egg-info/SOURCES.txt',
             f'autotest-{version}/autotest.egg-info/dependency_links.txt',
             f'autotest-{version}/autotest.egg-info/top_level.txt',
             f'autotest-{version}/autotest/__init__.py',
             f'autotest-{version}/autotest/__main__.py',
             f'autotest-{version}/autotest/asyncer.py',
             f'autotest-{version}/autotest/binder.py',
             f'autotest-{version}/autotest/diffs.py',
             f'autotest-{version}/autotest/filter.py',
             f'autotest-{version}/autotest/fixtures.py',
             f'autotest-{version}/autotest/integrationtests.py',
             f'autotest-{version}/autotest/levels.py',
             f'autotest-{version}/autotest/mocks.py',
             f'autotest-{version}/autotest/operators.py',
             f'autotest-{version}/autotest/prrint.py',
             f'autotest-{version}/autotest/tester.py',
             f'autotest-{version}/autotest/tests',
             f'autotest-{version}/autotest/tests/__init__.py',
             f'autotest-{version}/autotest/tests/sub_module_fail.py',
             f'autotest-{version}/autotest/tests/sub_module_ok.py',
             f'autotest-{version}/autotest/tests/temporary_class_namespace.py',
             f'autotest-{version}/autotest/tests/tryout.py',
             f'autotest-{version}/autotest/tests/tryout2.py',
             f'autotest-{version}/autotest/utils.py',
             f'autotest-{version}/autotest/wildcard.py',
             f'autotest-{version}/bin',
             f'autotest-{version}/bin/autotest',
             f'autotest-{version}/setup.cfg',
             f'autotest-{version}/setup.py',
            ],
            sorted(tf.getnames()), diff=lambda a, b: set(a).symmetric_difference(set(b)))
        tf.close()


"""
We put these last, as printing any debug/trace messages anywhere in th code causes this
to fail.
"""


if 'AUTOTESTSELFTEST' not in os.environ:
    os.putenv('AUTOTESTSELFTEST', 'ACTIVE')

    self_test2 = self_test.getChild(hooks=[fixtures_hook], fixtures=std_fixtures)


    @self_test2
    def main_without_args(stdout):
        os.system("PYTHONPATH=. python autotest")
        assert "Usage: autotest [options] module" in stdout.getvalue()


    @self_test2
    def main_without_help(stdout):
        os.system("PYTHONPATH=. python autotest --help")
        s = stdout.getvalue()
        assert "Usage: autotest [options] module" in s
        assert "-h, --help            show this help message and exit" in s
        assert "-f FILTER, --filter=FILTER" in s
        assert "only run tests whose qualified name contains FILTER" in s
        assert "-l LEVEL, --level=LEVEL" in s
        assert "only run tests whose level is >= LEVEL" in s



    @self_test2
    def main_test(stderr, stdout):
        import os
        os.system("PYTHONPATH=. python autotest autotest/tests/tryout.py")
        loglines = stdout.getvalue().splitlines()
        assert 'importing autotest.tests.tryout' in loglines[0], loglines
        assert "TEST:INTEGRATION:autotest.tests.tryout:one_simple_test:" in loglines[1], loglines
        assert "autotest/autotest/tests/tryout.py:6" in loglines[1]
        assert "TEST:INTEGRATION:autotest.tests.tryout:one_more_test:" in loglines[2]
        assert "autotest/autotest/tests/tryout.py:10" in loglines[2]
        assert len(loglines) == 3
        lines = stderr.getvalue().splitlines()
        assert "  7  \tdef one_simple_test():" == lines[0]
        assert "  8  \t    test.eq(1, 1)" == lines[1]
        assert "  9  \t" == lines[2]
        assert " 10  \t@test.integration" == lines[3]
        assert " 11  \tasync def one_more_test():" == lines[4]
        assert ' 12  ->\t    assert 1 == 2, "one is not two"' == lines[5]
        assert " 13  \t    test.eq(1, 2)" == lines[6]
        assert "[EOF]" == lines[7]
        assert "Traceback (most recent call last):" in lines[8]
        # some stuff in between we can't get rid off
        assert "in <module>" in lines[-5]
        assert "autotest/tests/tryout.py" in lines[-5]
        assert "async def one_more_test():" in lines[-4]
        assert "in one_more_test" in lines[-3]
        assert "autotest/tests/tryout.py" in lines[-3]
        assert "assert 1 == 2" in lines[-2]
        assert "AssertionError: one is not two" in lines[-1], lines[-1]
        assert 20 == len(lines), len(lines)


    @self_test2
    def main_with_selftests(stdout, stderr):
        os.system("PYTHONPATH=. python autotest autotest.selftest")
        lns = stdout.getvalue().splitlines()
        assert ['Usage: autotest [options] module', ''] == lns, lns
        lns = stderr.getvalue().splitlines()
        assert len(lns) == 142, len(lns) # number of logged tests, sort of


    @self_test2
    def main_via_bin_script_with_autotest_on_path(stdout, stderr):
        os.system(f'PATH=./bin:$PATH autotest autotest/tests/tryout2.py')
        e = stderr.getvalue()
        self_test2.eq('', e)
        o = stdout.getvalue()
        self_test2.startswith(o, "importing autotest.tests.tryout2")
        self_test2.contains(o, "TEST:INTEGRATION:autotest.tests.tryout2:one_simple_test")
        self_test2.contains(o, "TEST:INTEGRATION:autotest.tests.tryout2:one_integration_test")
        self_test2.contains(o, "TEST:PERFORMANCE:autotest.tests.tryout2:one_performance_test")
        self_test2.contains(o, "TEST:INTEGRATION:root:stats:found: 3, run: 3:")


    @self_test2
    def main_via_bin_script_in_cur_dir(stdout, stderr):
        os.system(f'(cd bin; ./autotest autotest/tests/tryout2.py)')
        e = stderr.getvalue()
        self_test2.eq('', e)
        o = stdout.getvalue()
        self_test2.startswith(o, "importing autotest.tests.tryout2")
        self_test2.contains(o, "TEST:INTEGRATION:autotest.tests.tryout2:one_simple_test")
        self_test2.contains(o, "TEST:INTEGRATION:autotest.tests.tryout2:one_integration_test")
        self_test2.contains(o, "TEST:PERFORMANCE:autotest.tests.tryout2:one_performance_test")
        self_test2.contains(o, "TEST:INTEGRATION:root:stats:found: 3, run: 3:")

    @self_test2
    def main_with_filter(stdout, stderr):
        os.system("PYTHONPATH=. python autotest autotest/tests/tryout2.py --filter one_simple")
        e = stderr.getvalue()
        assert '' == e, e
        o = stdout.getvalue()
        assert 'one_simple_test' in o
        assert 'one_integration_test' not in o
        assert 'one_performance_test' not in o
        assert 'found: 3, run: 1' in o

    @self_test2
    def main_with_level_unit(stdout, stderr):
        os.system("PYTHONPATH=. python autotest autotest/tests/tryout2.py --level unit")
        e = stderr.getvalue()
        assert '' == e, e
        o = stdout.getvalue()
        assert 'one_simple_test' in o, o
        assert 'one_integration_test' not in o, o
        assert 'one_performance_test' not in o
        assert 'found: 3, run: 1' in o, o

    @self_test2
    def main_with_level_integration(stdout, stderr):
        os.system("PYTHONPATH=. python autotest autotest/tests/tryout2.py --level integration")
        e = stderr.getvalue()
        assert '' == e, e
        o = stdout.getvalue()
        assert 'one_simple_test' in o, o
        assert 'one_integration_test' in o, o
        assert 'one_performance_test' not in o
        assert 'found: 3, run: 2' in o, o
