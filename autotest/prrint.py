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

"""
    This pretty printer aims to make diffs of plain old python (POD) objects more usable.
    It does so by:
      1. making sure every atomic value (str, int, etc) is on a separate line
      2. markup symbols like (, { and : do not interfere by puttin them on a separate line
      3. long atomic values (like str) are continuous; the terminal will help you.
      4. use repr for any unsupported value
"""

import sys
from io import StringIO
from dataclasses import dataclass

def _format(data, indent, seen=None, sep=','):
    write = indent.write
    if id(data) in seen and isinstance(data, (list, dict, tuple, set)):
        write('...', sep)
    else:
        seen.add(id(data))
        if isinstance(data, list):
            if data:
                write('[')
                for v in data:
                    _format(v, indent(), seen)
                write(']', sep)
            else:
                write('[]', sep)
        elif isinstance(data, dict):
            if data:
                write('{')
                for k, v in sorted(data.items(), key=lambda item: str(item[0])):
                    _format(k, indent(), seen, sep=':')
                    _format(v, indent()(), seen)
                write('}', sep)
            else:
                write('{}', sep)
        elif isinstance(data, tuple):
            if data:
                write('(')
                for v in data:
                    _format(v, indent(), seen)
                write(')', sep)
            else:
                write('()', sep)
        elif isinstance(data, set):
            if data:
                write('{')
                for v in sorted(data, key=str):
                    _format(v, indent(), seen)
                write('}', sep)
            else:
                write('set()', sep)
        else:
            write(repr(data), sep)


@dataclass
class Indenter:
    out: StringIO
    i: int = 0
    INDENT: int = 2

    def __call__(self):
        return Indenter(self.out, self.i + self.INDENT)

    def write(self, *data):
        write = self.out.write
        write(self.i * ' ')
        for d in data:
            write(d)
        write('\n')


def format(data):
    out = StringIO()
    _format(data, Indenter(out), seen=set(), sep='')
    return out.getvalue()


def prrint(data):
    _format(data, Indenter(sys.stdout), seen=set(), sep='')


#from autotest import test
from .tester import Runner, stdout
test = Runner()
test.fixture(stdout)

@test
def empty():
    test.eq("[]\n", format([]))
    test.eq("{}\n", format({}))
    test.eq("''\n", format(""))
    test.eq("()\n", format(()))
    test.eq("set()\n", format(set()))


@test
def one_element():
    test.eq("[\n  42,\n]\n", format([42]))
    test.eq(
"""{
  1:
    'a',
}
""", format({1: "a"}))
    test.eq("''\n", format(""))
    test.eq("(\n  42,\n)\n", format((42,)))
    test.eq("[\n  42,\n]\n", format([42]))
    test.eq("{\n  42,\n}\n", format({42}))


@test
def separators():
    x = format([1, (2,), {3:3}, {4}, [5]])
    test.eq("""[
  1,
  (
    2,
  ),
  {
    3:
      3,
  },
  {
    4,
  },
  [
    5,
  ],
]
""", x)


@test
def recursion():
    a = {1: None}
    a[1] = a
    x = format(a)
    test.eq("""{
  1:
    ...,
}
""", x)


@test
def set_sorted():
    x = format({"noot", "mies", "aap"})
    test.eq("""{
  'aap',
  'mies',
  'noot',
}
""", x)


@test
def set_sorted_uncomparables():
    x = format({str, dict, bool})
    test.eq("{\n  <class 'bool'>,\n  <class 'dict'>,\n  <class 'str'>,\n}\n", x)


@test
def dict_sorted():
    x = format({"noot": 3, "mies": 2, "aap": 1})
    test.eq("""{
  'aap':
    1,
  'mies':
    2,
  'noot':
    3,
}
""", x)


@test
def dict_sorted_uncomparables():
    x = format({str: 2, dict: 1, bool: 3})
    test.eq("{\n  <class 'bool'>:\n    3,\n  <class 'dict'>:\n    1,\n  <class 'str'>:\n    2,\n}\n", x)


@test
def beetje_echt():
    x = format({'pred_a': [{'pred_b': [{'@value': "32", '@type': 'number'}, {'@value': "more", '@type':  "text"}]}]})
    test.eq("""{
  'pred_a':
    [
      {
        'pred_b':
          [
            {
              '@type':
                'number',
              '@value':
                '32',
            },
            {
              '@type':
                'text',
              '@value':
                'more',
            },
          ],
      },
    ],
}
""" , x)


@test
def allez():
    x = format({('aa', 'bb'): {('cc', ('dd',)): ('ee',)}, ('ff',): ()})
    test.eq("""{
  (
    'aa',
    'bb',
  ):
    {
      (
        'cc',
        (
          'dd',
        ),
      ):
        (
          'ee',
        ),
    },
  (
    'ff',
  ):
    (),
}
""", x)


@test
def very_long_string_uninterrupted():
    long_string = "aapnootmies" * 100
    x = format((long_string,))
    test.eq(f"""(
  '{long_string}',
)
""", x)


@test
def via_stdout(stdout):
    d = {'a': ('b', 42)}
    f = format(d)
    prrint(d)
    test.eq(f, stdout.getvalue())


def nndiff(a, b):
    import difflib
    import re
    r = re.compile("(\s*)(\S+)(\s*)")
    al = [r.fullmatch(l).group(1,2,3) for l in a]
    bl = [r.fullmatch(l).group(1,2,3) for l in b]
    dl = difflib.ndiff([l[1] for l in al], [l[1] for l in bl])
    return dl


@test
def diff2_whitespace():
    a = [" a "]
    b = ["  a "]
    d = nndiff(a, b)
    test.eq('  a', ''.join(d))


