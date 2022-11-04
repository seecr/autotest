
import pdb
import sys
import logging
import autotest
import importlib
import pathlib
import optparse


"""
Runs autotests reporting to stdout.

Usage:
  $ autotest <path or module>

"""

def post_mortem(tb, *cmds):
    """ for when you use plain assert, it'll throw you in Pdb on failure """
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
    post_mortem(tb, 'list', 'exit')
    return t, v, tb


class MyHandler(logging.Handler):
    def emit(self, r):
        testlevelname = autotest.levels.levels.get(r.levelno)
        print(f"TEST:{testlevelname}:{r.name}:{r.msg}:{r.pathname}:{r.lineno}")


insert_excepthook(code_print_excepthook)
insert_excepthook(lambda t, v, tb: (t, v, autotest.utils.filter_traceback(tb)))


cwd = pathlib.Path.cwd()
sys.path.insert(0, cwd.as_posix())


p = optparse.OptionParser(usage="usage: %prog [options] module")
p.add_option('-f', '--filter', help="only run tests whose qualified name contains FILTER")
p.add_option('-l', '--level', help="only run tests whose level is >= LEVEL",
        choices=[l.lower() for l in autotest.levels.levels.keys() if isinstance(l, str)])
options, args = p.parse_args()


test_options = {}
if f := options.filter:
    test_options['filter'] = f
if l := options.level:
    test_options['level'] = l
if test_options:
    autotest.basic_config(**test_options)


root = autotest.get_tester()
root.addHandler(MyHandler())


if len(args) == 1:
    p = pathlib.Path(args[0])
    modulename = '.'.join(p.parent.parts + (p.stem,))
    print("importing", modulename)
    importlib.import_module(modulename)
    root.log_stats()
else:
    p.print_usage()

