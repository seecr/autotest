
import tempfile
import pathlib


def tmp_path(name=None):
    with tempfile.TemporaryDirectory() as p:
        p = pathlib.Path(p)
        if name:
            yield p/name
        else:
            yield p
self_test.fixture(tmp_path)


class tmp_files:

    path = None

    @self_test
    def temp_sync(tmp_path):
        assert tmp_path.exists()

    @self_test
    def temp_file_removal(tmp_path):
        global path
        path = tmp_path / 'aap'
        path.write_text("hello")

    @self_test
    def temp_file_gone():
        assert not path.exists()

    @self_test
    async def temp_async(tmp_path):
        assert tmp_path.exists()

    @self_test
    def temp_dir_with_file(tmp_path:'aap'):
        assert str(tmp_path).endswith('/aap')
        tmp_path.write_text('hi monkey')
        assert tmp_path.exists()


def capture(name):
    """ captures output from child processes as well """
    org_stream = getattr(sys, name)
    org_fd = org_stream.fileno()
    org_fd_backup = os.dup(org_fd)
    replacement = tempfile.TemporaryFile(mode="w+t", buffering=1)
    os.dup2(replacement.fileno(), org_fd)
    setattr(sys, name, replacement)
    def getvalue():
        replacement.flush()
        replacement.seek(0)
        return replacement.read()
    replacement.getvalue = getvalue
    try:
        yield replacement
    finally:
        os.dup2(org_fd_backup, org_fd)
        setattr(sys, name, org_stream)


# do not use @self_test.fixture as we need to install this twice
def stdout():
    yield from capture('stdout')
self_test.fixture(stdout)


# do not use @test.fixture as we need to install this twice
def stderr():
    yield from capture('stderr')
self_test.fixture(stderr)


@self_test.fixture
async def slow_callback_duration(s):
    asyncio.get_running_loop().slow_callback_duration = s
    yield


@self_test.fixture
def test_fixtures():
    with self_test.stdout as s:
        print("hello!")
        assert "hello!\n" == s.getvalue()
    keep = []
    with self_test.tmp_path as p:
        keep.append(p)
        (p / "f").write_text("contents")
    assert not keep[0].exists()


# with self_test.<fixture> does not need @test to run 'in'
with self_test.tmp_path as p:
    assert p.exists()
assert not p.exists()


