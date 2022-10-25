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



from .tester import self_test

from .wildcard import Wildcard, testing_wildcard
testing_wildcard(self_test)

from .binder import Binder, testing_binder
testing_binder(self_test)

from .levels import Levels, testing_levels
testing_levels(self_test)

from .operators import Operators, testing_operators
testing_operators(self_test)

from .fixtures import Fixtures, testing_fixtures
testing_fixtures(self_test)

#assert len(self_test._options.get('hooks')) == 1
#assert len(self_test2._options.get('hooks')) == 2
#hooks = list(self_test2._hooks())
#assert hooks[0] == binder, hooks[0]
#assert hooks[1] == WithFixtures, hooks[1]
#assert hooks[2].__class__ == OperatorLookup, hooks[2]
#assert len(hooks) == 3



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

