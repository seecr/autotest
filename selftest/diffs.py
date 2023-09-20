## begin license ##
#
# "selftest": a simpler test runner for python
#
# Copyright (C) 2022-2023 Seecr (Seek You Too B.V.) https://seecr.nl
#
# This file is part of "selftest"
#
# "selftest" is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "selftest" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with "selftest".  If not, see <http://www.gnu.org/licenses/>.
#
## end license ##

import difflib  # show diffs on failed tests with two args
import pprint  # show diffs on failed tests with two args
import functools
import inspect
from .prrint import prrint, format as forrmat


def diff(a, b, format=pprint.pformat):
    """Produces diff of textual representation of a and b."""
    return "\n" + "\n".join(
        difflib.ndiff(format(a).splitlines(), format(b).splitlines())
    )


def diff2(a, b):
    """diff based on own prrint; suitable for PODs"""
    return diff(a, b, format=forrmat)


class DiffHook:
    def __call__(self, runner, func):
        return func

    def lookup(self, runner, name):
        if name == "diff":
            return diff
        if name == "diff2":
            return diff2
        if name == "prrint":
            return prrint
        raise AttributeError


diff_hook = DiffHook()


def diff_test(self_test):
    self_test = self_test.getChild(hooks=[diff_hook])

    @self_test
    def idea_for_dumb_diffs():
        # if you want a so called smart diff: the second arg of assert is meant for it.
        # Runner supplies a generic diff between two pretty printed values
        a = [7, 1, 2, 8, 3, 4]
        b = [1, 2, 9, 3, 4, 6]
        d = self_test.diff(a, b)
        assert str == type(str(d))

        try:
            assert a == b, self_test.diff(a, b)
        except AssertionError as e:
            assert """
- [7, 1, 2, 8, 3, 4]
?  ---      ^

+ [1, 2, 9, 3, 4, 6]
?        ^      +++
""" == str(
                e
            )

        try:
            self_test.eq(a, b, diff=self_test.diff)
        except AssertionError as e:
            assert """
- [7, 1, 2, 8, 3, 4]
?  ---      ^

+ [1, 2, 9, 3, 4, 6]
?        ^      +++
""" == str(
                e
            ), e

        # use of a function from elswhere
        a = set([7, 1, 2, 8, 3, 4])
        b = set([1, 2, 9, 3, 4, 6])
        try:
            assert a == b, set.symmetric_difference(a, b)
        except AssertionError as e:
            assert "{6, 7, 8, 9}" == str(e), e
        try:
            self_test.eq(a, b, diff=set.symmetric_difference)
        except AssertionError as e:
            assert "{6, 7, 8, 9}" == str(e), e

    @self_test
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
        try:
            self_test.eq({dict: bool}, {str, 1}, diff=self_test.diff2)
        except AssertionError as e:
            assert msg == str(e), e
