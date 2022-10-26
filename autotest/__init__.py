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



#TODO
#sys_defaults.update({k[len('AUTOTEST_'):]: eval(v) for k, v in os.environ.items() if k.startswith('AUTOTEST_')})



from .tester import Runner, self_test

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


test = Runner(
        # order of hook matters
        hooks = (operators_hook, fixtures_hook, levels_hook, wildcard_hook, binder_hook),
        fixtures = std_fixtures,
    )


@test
def root_tester_assembly():
    """ only test if the root tester is assembled with all hooks """

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

    # wildcard hook
    test.eq(test.any(int), 42)

    # levels hook
    @test.performance
    def critical_test():
        assert 1 == 2



@self_test
def check_stats():
    self_test.eq({'found': 89, 'run': 83}, self_test._stats)













@test
def setup_correct(tmp_path):
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
    result = subprocess.run(['python3', 'setup.py', 'sdist', '--dist-dir', str(tmp_path)],
            capture_output=True,
            cwd=str(autotest_dev_dir))

    from tarfile import open
    tf = open(name=tmp_path/f'autotest-{version}.tar.gz', mode='r:gz')
    test.eq([f'autotest-{version}',
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
             f'autotest-{version}/autotest/main.py',
             f'autotest-{version}/autotest/moretests.py',
             f'autotest-{version}/autotest/prrint.py',
             f'autotest-{version}/autotest/tester.py',
             f'autotest-{version}/autotest/tests',
             f'autotest-{version}/autotest/tests/__init__.py',
             f'autotest-{version}/autotest/tests/sub_module_fail.py',
             f'autotest-{version}/autotest/tests/sub_module_ok.py',
             f'autotest-{version}/autotest/tests/temporary_class_namespace.py',
             f'autotest-{version}/bin',
             f'autotest-{version}/bin/autotest',
             f'autotest-{version}/setup.cfg',
             f'autotest-{version}/setup.py',
            ],
            sorted(tf.getnames()), msg=test.diff)
    tf.close()



# helpers
def mock_object(*functions, **more):
    """ Creates an object from a bunch of functions.
        Useful for testing methods from inside the class definition. """
    self = mock.Mock()
    self.configure_mock(**{f.__name__: types.MethodType(f, self) for f in functions}, **more)
    return self

