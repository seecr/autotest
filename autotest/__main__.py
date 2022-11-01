
import pdb
import sys
import logging
import autotest
import importlib
import pathlib


def post_mortem(tb, *cmds):
    """ for when you use plain assert, it'll throw you in Pdb on failure """
    p = pdb.Pdb()
    p.rcLines.extend(cmds)
    p.reset()
    p.interaction(None, tb)


def insert_excepthook(new_hook):
    prev_excepthook = sys.excepthook
    def hook(*args):
        prev_excepthook(*new_hook(*args))
    sys.excepthook = hook


def code_print_excepthook(t, v, tb):
    post_mortem(tb, 'longlist', 'exit')
    return t, v, tb


class MyHandler(logging.Handler):
    def emit(self, r):
        testlevelname = autotest.levels.levels.get(r.levelno)
        print(f"TEST:{testlevelname}:{r.name}:{r.msg}:{r.pathname}:{r.lineno}")


"""
Usage:
  $ autotest.py <path of module>
"""


insert_excepthook(code_print_excepthook)
insert_excepthook(lambda t, v, tb: (t, v, autotest.utils.filter_traceback(tb)))

root = autotest.get_tester()
root.addHandler(MyHandler())


cwd = pathlib.Path.cwd()
sys.path.insert(0, cwd.as_posix())

if len(sys.argv) == 2:
    p = pathlib.Path(sys.argv[-1])
    modulename = '.'.join(p.parent.parts + (p.stem,))
    print("importing", modulename)
    assert importlib.import_module(modulename)
    root.log_stats()
else:
    print("Please specify one module to import.")


