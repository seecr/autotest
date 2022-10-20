
import pdb
import sys


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



# TODO default runner could do fancier logging:
    #def __str__(self):
    #    return f"{self.func.__module__}  \33[1m{self.func.__name__}\033[0m  "



def main():
    print(" NEW AUTOTEST ")
    # experimental main; not tested

    """
    When run as main, it imports all Python modules in the current directory, or
    only the modules given as arguments (which may contain .py of /, which is ignored).

    Usage:
      $ autotest.py
      $ autotest.py a_module_dir
      $ autotest.py a_python_file.py
      $ autotest.py a_module_dir/a_python_file.py
      $ autotest.py any_path_basically_try_your_luck
    """

    # using import.import_module in asyncio somehow gives us the frozen tracebacks (which were
    # removed in 2012, but yet showing up again in this case. Let's get rid of them.
    def code_print_excepthook(t, v, tb):
        post_mortem(tb, 'longlist', 'exit')
        return t, v, tb
    insert_excepthook(code_print_excepthook)

    import importlib
    import pathlib
    import os

    cwd = pathlib.Path.cwd()
    sys.path.insert(0, str(cwd))
    if len(sys.argv) > 1:
        modules = map(lambda p: '.'.join(p.parent.parts + (p.stem,)),
                    map(pathlib.Path, sys.argv[1:]))
    else:
        modules = (pathlib.Path(p).stem for p in cwd.iterdir())
    modules = [m for m in modules if m not in ['__init__', '__pycache__'] and not m.startswith('.')]

    #if 'autotest' not in modules:
    #    os.environ['AUTOTEST_report'] = 'False'
    #    print("silently", end=' ')

    print("importing \033[1mautotest\033[0m")
    from tester import test, filter_traceback
    #from autotest import test, filter_traceback # default test runner
    insert_excepthook(lambda t, v, tb: (t, v, filter_traceback(tb)))

    def skip_fn(test_fn):
        return not all(test_fn.__module__.startswith(m) for m in modules)
    with test.opts(skip=skip_fn, report=True):
        if 'autotest' in modules:
            modules.remove('autotest')

        for name in modules:
            if name in sys.modules:
                print(f"already imported tests from \033[1m{name}\033[0m")
            elif importlib.util.find_spec(name):
                print(f"importing tests from \033[1m{name}\033[0m")
                importlib.import_module(name)
            else:
                print(f"WARNING: module \033[1m{name}\033[0m not found.")

        report = test.context.report
        print(f"Found \033[1m{report.total}\033[0m unique tests, ran \033[1m{report.ran}\033[0m, reported \033[1m{report.reported}\033[0m.")
        report.report()

if __name__ == '__main__':
    main()
