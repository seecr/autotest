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

def any(_): # name is included in diffs
    return True


class _Any:
    def __init__(self, f=any):
        self.f = f

    def __eq__(self, x):
        return bool(self.f(x))

    def __call__(self, f):
        return _Any(f)

    def __repr__(self):
        return self.f.__name__  + '(...)' if self.f else '*'


class _Wildcard:

    def __call__(self, tester, func):
        return func

    def lookup(self, tester, name):
        if name == 'any':
            return _Any()
        raise AttributeError


wildcard_hook = _Wildcard()


def wildcard_test(self_test):
    with self_test.child(hooks=(wildcard_hook,)) as test:
        @test
        def wildcard_matching():
            test.eq([test.any, 42], [16, 42])
            test.eq([test.any(lambda x: x in [1,2,3]), 78], [2, 78])
            test.ne([test.any(lambda x: x in [1,2,3]), 78], [4, 78])


