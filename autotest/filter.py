## begin license ##
#
# "Autotest": a simpler test runner for python
#
# Copyright (C) 2022 Seecr (Seek You Too B.V.) https://seecr.nl
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


def filter_hook(runner, func):
    f = runner.option_get('filter', '')
    if f in func.__qualname__:
        return func


def filter_test(self_test):
    my_test = self_test(hooks=[filter_hook])
    @my_test
    def hook_but_no_filter_given():
        pass
    @my_test
    def aap_mies():
        with my_test.child(filter='noot') as with_noot:
            r = [0, 0, 0]
            class aap:
                @with_noot
                def aap():
                    r[0] = 1
                @with_noot
                def noot():
                    r[1] = 1
                    @with_noot
                    def mies():
                        r[2] = 1
            assert r == [0, 1, 1], r
        assert {'found': 3, 'run': 2} == with_noot.stats, with_noot.stats
