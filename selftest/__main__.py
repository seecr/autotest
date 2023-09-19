## begin license ##
#
# "Autotest": a simpler test runner for python
#
# Copyright (C) 2022-2023 Seecr (Seek You Too B.V.) https://seecr.nl
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

import pdb
import sys
import logging
import selftest
import importlib
import pathlib
import optparse
import os
import os.path


"""
Runs selftests reporting to stdout.

Usage:
  $ selftest <path or module>

"""

if "SELFTEST_MAIN" not in os.environ:
    # avoid running main twice (during import in a spawned process)
    os.environ["SELFTEST_MAIN"] = "Y"

    def post_mortem(tb, *cmds):
        """for when you use plain assert, it'll throw you in Pdb on failure"""
        p = pdb.Pdb(stdout=sys.stderr)
        p.rcLines.extend(cmds)
        p.reset()
        p.interaction(None, tb)

    def insert_excepthook(new_hook):
        prev_excepthook = sys.excepthook

        def hook(*args):
            prev_excepthook(*new_hook(*args))

        sys.excepthook = hook

    def code_print_excepthook(t, v, tb):
        post_mortem(tb, "list", "exit")
        return t, v, tb

    cwd = pathlib.Path.cwd()
    sys.path.insert(0, cwd.as_posix())

    insert_excepthook(code_print_excepthook)
    insert_excepthook(lambda t, v, tb: (t, v, selftest.utils.filter_traceback(tb)))

    p = optparse.OptionParser(usage="usage: %prog [options] module")
    p.add_option(
        "-f", "--filter", help="only run tests whose qualified name contains FILTER"
    )
    p.add_option(
        "-t",
        "--threshold",
        help="only run tests whose level is >= THRESHOLD",
        choices=[
            l.lower() for l in selftest.levels.levels.keys() if isinstance(l, str)
        ],
    )
    options, args = p.parse_args()

    test_options = {}
    if f := options.filter:
        test_options["filter"] = f
    test_options["threshold"] = options.threshold or "integration"
    selftest.basic_config(**test_options)

    class F(str):
        def format(self, *, levelname="", message="", lineno="", pathname="", **rest):
            levelname = "\033[1mTEST\033[0m" if levelname == "TEST" else levelname
            return f"{levelname}:\033[1m{message}\033[0m:{pathname[-40:]}:{lineno}"

    logging.basicConfig(style="{", format=F("{fake}"))

    root = selftest.get_tester()
    if len(args) == 1:
        p = pathlib.Path(args[0])
        modulename = ".".join(p.parent.parts + (p.stem,))
        logging.getLogger("selftest").log(
            selftest.tester.default_loglevel, f"importing {modulename}"
        )
        importlib.import_module(modulename)
        root.log_stats()
    else:
        p.print_usage()
