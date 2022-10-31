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



from .tester import Runner, self_test

from .prrint import prrint_test
prrint_test(self_test)

from .filter import filter_hook, filter_test
filter_test(self_test)

from .wildcard import wildcard_hook, wildcard_test
wildcard_test(self_test)

from .binder import binder_hook, binder_test
binder_test(self_test)

from .levels import levels_hook, levels_test
levels_test(self_test)

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
    self_test.eq({'found': 104, 'run': 97}, self_test.stats)


def assemble_root_runner():
    return Runner(
        # order of hook matters, processed from right to left
        hooks = (operators_hook, async_hook, fixtures_hook, diff_hook, levels_hook, wildcard_hook, binder_hook, filter_hook),
        fixtures = std_fixtures,
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
        @test
        def bind():
            assert a == 42
            N[0] += 1

    # wildcard hook
    test.eq(test.any(int), 42)

    # levels hook
    @test.performance
    def performance_test():
        assert "not" == "executed"
    @test.critical
    def critical_test():
        assert 1 == 1
        N[0] += 1

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



root_tester_assembly_test(assemble_root_runner())

root = assemble_root_runner()


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
             f'autotest-{version}/autotest/asyncer.py',
             f'autotest-{version}/autotest/binder.py',
             f'autotest-{version}/autotest/diffs.py',
             f'autotest-{version}/autotest/filter.py',
             f'autotest-{version}/autotest/fixtures.py',
             f'autotest-{version}/autotest/levels.py',
             f'autotest-{version}/autotest/main.py',
             f'autotest-{version}/autotest/mocks.py',
             f'autotest-{version}/autotest/moretests.py',
             f'autotest-{version}/autotest/operators.py',
             f'autotest-{version}/autotest/prrint.py',
             f'autotest-{version}/autotest/tester.py',
             f'autotest-{version}/autotest/tests',
             f'autotest-{version}/autotest/tests/__init__.py',
             f'autotest-{version}/autotest/tests/sub_module_fail.py',
             f'autotest-{version}/autotest/tests/sub_module_ok.py',
             f'autotest-{version}/autotest/tests/temporary_class_namespace.py',
             f'autotest-{version}/autotest/utils.py',
             f'autotest-{version}/autotest/wildcard.py',
             f'autotest-{version}/bin',
             f'autotest-{version}/bin/autotest',
             f'autotest-{version}/setup.cfg',
             f'autotest-{version}/setup.py',
            ],
            sorted(tf.getnames()), diff=lambda a, b: set(a).symmetric_difference(set(b)))
        tf.close()


