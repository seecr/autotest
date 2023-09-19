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

from .utils import extend_closure  # redefine placeholder


class _Binder:
    def __call__(self, runner, func):
        if runner.option_get("bind", False):
            return extend_closure(func)
        return func


binder_hook = _Binder()


def binder_test(self_test):
    self_test2 = self_test.getChild(hooks=(binder_hook,))

    class binding_context:
        a = 42

        @self_test2(keep=True, bind=True)
        def one_test():
            assert a == 42

        a = 43
        # one_test()

    @self_test2
    def access_closure_from_enclosing_def():
        a = 46  # accessing this in another function makes it a 'freevar', which needs a closure

        @self_test2
        def access_a():
            assert 46 == a

    f = 16

    @self_test2
    def dont_confuse_app_vars_with_internal_vars():
        assert 16 == f
