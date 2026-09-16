"""An accidentally unmocked Python open cannot reach the physical scanner."""

import builtins
import io
import os

import pytest


@pytest.mark.parametrize("path", ["/dev/video0", "/dev/bus/usb/001/123", "/dev/media0"])
def test_live_device_open_is_refused_before_system_call(path):
    for opening in (builtins.open, io.open):
        with pytest.raises(AssertionError, match="live scanner open"):
            opening(path, "rb")
    with pytest.raises(AssertionError, match="live scanner open"):
        os.open(path, os.O_RDONLY)


def test_normal_fixture_file_open_still_works(tmp_path):
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"test fixture")
    with path.open("rb") as stream:
        assert stream.read() == b"test fixture"
    fd = os.open(path, os.O_RDONLY)
    try:
        assert os.read(fd, 100) == b"test fixture"
    finally:
        os.close(fd)
