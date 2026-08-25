"""Reference-probe adapter tests: SDK verification, timeouts, and cleanup.

These never touch the real SDK or the scanner; the "probe" is a stub script.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scanner3d.errors import ReferenceProbeError
from scanner3d.reference import (
    PINNED_FILES,
    _drain,
    identify_sdk,
    load_manifest,
    resolve_sdk_root,
    run_probe,
    sha256_file,
)

SERIAL = "EXAMPLESERIAL0001"


def _fake_sdk(tmp_path: Path, *, header: bytes = b"header", lib: bytes = b"lib") -> Path:
    root = tmp_path / "3DCamera"
    (root / "include").mkdir(parents=True)
    (root / "linux" / "x64").mkdir(parents=True)
    (root / "include" / "3DCamera.hpp").write_bytes(header)
    (root / "linux" / "x64" / "lib3DCamera.so").write_bytes(lib)
    return root


def _manifest_for(root: Path, tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"files": {name: sha256_file(root / name) for name in PINNED_FILES}})
    )
    return path


def _script(tmp_path: Path, body: str, name: str = "probe") -> Path:
    path = tmp_path / name
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return path


# --- SDK identity -------------------------------------------------------------


def test_identify_sdk_verified_against_matching_manifest(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    identity = identify_sdk(root, _manifest_for(root, tmp_path))
    assert identity.verified
    assert identity.missing == () and identity.mismatched == ()


def test_identify_sdk_detects_tampered_library(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    manifest = _manifest_for(root, tmp_path)
    (root / "linux" / "x64" / "lib3DCamera.so").write_bytes(b"tampered")
    identity = identify_sdk(root, manifest)
    assert not identity.verified
    assert identity.mismatched == ("linux/x64/lib3DCamera.so",)


def test_identify_sdk_detects_missing_file(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    manifest = _manifest_for(root, tmp_path)
    (root / "include" / "3DCamera.hpp").unlink()
    identity = identify_sdk(root, manifest)
    assert identity.missing == ("include/3DCamera.hpp",)
    assert not identity.verified


def test_missing_or_malformed_manifest_yields_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_manifest(bad) == {}
    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text('{"files": "nope"}')
    assert load_manifest(wrong_shape) == {}


def test_resolve_sdk_root_prefers_explicit_then_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("REVOPOINT_SDK_ROOT", raising=False)
    assert resolve_sdk_root() is None
    monkeypatch.setenv("REVOPOINT_SDK_ROOT", str(tmp_path))
    assert resolve_sdk_root() == tmp_path
    assert resolve_sdk_root("/elsewhere") == Path("/elsewhere")


# --- Guard rails --------------------------------------------------------------


def test_no_sdk_root_refuses_and_names_the_env_var(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("REVOPOINT_SDK_ROOT", raising=False)
    with pytest.raises(ReferenceProbeError, match="REVOPOINT_SDK_ROOT"):
        run_probe(SERIAL, manifest_path=tmp_path / "m.json")


def test_missing_sdk_root_directory_refuses(tmp_path: Path) -> None:
    with pytest.raises(ReferenceProbeError, match="does not exist"):
        run_probe(SERIAL, sdk_root=tmp_path / "absent", manifest_path=tmp_path / "m.json")


def test_unverified_sdk_refuses_to_run(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    manifest = _manifest_for(root, tmp_path)
    (root / "linux" / "x64" / "lib3DCamera.so").write_bytes(b"tampered")
    with pytest.raises(ReferenceProbeError, match="hash mismatch"):
        run_probe(SERIAL, sdk_root=root, probe_binary="/bin/true", manifest_path=manifest)


def test_no_pinned_hashes_refuses_to_run(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    with pytest.raises(ReferenceProbeError, match="no pinned hashes"):
        run_probe(
            SERIAL, sdk_root=root, probe_binary="/bin/true", manifest_path=tmp_path / "absent.json"
        )


def test_missing_probe_binary_refuses(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    with pytest.raises(ReferenceProbeError, match="not executable"):
        run_probe(
            SERIAL,
            sdk_root=root,
            probe_binary=tmp_path / "nope",
            manifest_path=_manifest_for(root, tmp_path),
        )


@pytest.mark.parametrize("timeout", [0, -1, 15.1, 600])
def test_timeout_outside_the_hard_ceiling_is_rejected(tmp_path: Path, timeout: float) -> None:
    root = _fake_sdk(tmp_path)
    with pytest.raises(ReferenceProbeError, match="timeout_s"):
        run_probe(
            SERIAL,
            sdk_root=root,
            probe_binary="/bin/true",
            timeout_s=timeout,
            manifest_path=_manifest_for(root, tmp_path),
        )


# --- Execution ----------------------------------------------------------------


def test_successful_probe_output_is_parsed(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    binary = _script(tmp_path, 'import json;print(json.dumps({"frames": {"depth": {"count": 1}}}))')
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=10,
        manifest_path=_manifest_for(root, tmp_path),
    )
    assert result.exit_code == 0 and not result.timed_out
    assert result.error is None
    assert result.data == {"frames": {"depth": {"count": 1}}}
    assert result.sdk is not None and result.sdk.verified


def test_probe_receives_serial_and_timeout_arguments(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    binary = _script(tmp_path, 'import json,sys;print(json.dumps({"argv": sys.argv[1:]}))')
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=10,
        manifest_path=_manifest_for(root, tmp_path),
    )
    assert result.data is not None
    argv = list(result.data["argv"])
    assert argv[:2] == ["--serial", SERIAL]
    # The child's internal budget must be strictly inside the parent's.
    assert int(argv[argv.index("--timeout-ms") + 1]) < 10_000


def test_nonzero_exit_is_reported_without_raising(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    binary = _script(tmp_path, 'import sys;sys.stderr.write("boom\\n");sys.exit(3)')
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=10,
        manifest_path=_manifest_for(root, tmp_path),
    )
    assert result.exit_code == 3
    assert result.error == "probe exited 3"
    assert "boom" in result.stderr


def test_non_json_output_is_reported_as_a_parse_error(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    binary = _script(tmp_path, 'print("not json at all")')
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=10,
        manifest_path=_manifest_for(root, tmp_path),
    )
    assert result.data is None
    assert result.error is not None and "not valid JSON" in result.error


def test_json_out_file_is_used_when_stdout_is_unusable(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    out = tmp_path / "probe.json"
    binary = _script(
        tmp_path,
        "import json,sys\n"
        'print("noise on stdout")\n'
        'i = sys.argv.index("--json-out")\n'
        'open(sys.argv[i+1], "w").write(json.dumps({"depth_scale": 0.1}))\n',
    )
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=10,
        manifest_path=_manifest_for(root, tmp_path),
        json_out=out,
    )
    assert result.data == {"depth_scale": 0.1}
    assert result.error is None


def test_timeout_kills_the_probe_and_reports_it(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    binary = _script(tmp_path, "import time;time.sleep(60)")
    started = time.monotonic()
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=1.0,
        manifest_path=_manifest_for(root, tmp_path),
    )
    elapsed = time.monotonic() - started
    assert result.timed_out
    assert result.data is None
    assert result.error is not None and "budget" in result.error
    assert elapsed < 10, "the hard timeout must bound wall clock"


def test_timeout_also_kills_orphaned_grandchildren(tmp_path: Path) -> None:
    """A probe that leaks a child must not leave a stream running behind it."""
    pidfile = tmp_path / "child.pid"
    binary = _script(
        tmp_path,
        "import os, subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(60)\n",
    )
    root = _fake_sdk(tmp_path)
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=2.0,
        manifest_path=_manifest_for(root, tmp_path),
    )
    assert result.timed_out
    child_pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("the probe's child process survived the timeout")


def test_timeout_kills_a_grandchild_that_ignores_sigterm(tmp_path: Path) -> None:
    """The adversarial cleanup case: leader dies on SIGTERM, grandchild does not.

    The grandchild also inherits the probe's stdout/stderr pipes, so if the
    group SIGKILL is skipped once the leader has exited, draining those pipes
    blocks forever and the wall-clock budget is silently broken.
    """
    pidfile = tmp_path / "stubborn.pid"
    grandchild = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
    )
    binary = _script(
        tmp_path,
        "import subprocess, sys, time\n"
        # No stdout/stderr redirection: the grandchild inherits the pipes.
        f"c = subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(120)\n",
    )
    root = _fake_sdk(tmp_path)

    started = time.monotonic()
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=2.0,
        manifest_path=_manifest_for(root, tmp_path),
    )
    elapsed = time.monotonic() - started

    assert result.timed_out
    assert elapsed < 12, f"cleanup must stay bounded; took {elapsed:.1f}s"

    child_pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("a SIGTERM-ignoring grandchild survived the hard timeout")


def test_drain_is_bounded_when_the_pipes_are_held_open() -> None:
    """Draining must give up rather than block on a survivor holding the pipes."""
    holder = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        # The leader exits immediately; its child keeps the inherited pipes open.
        proc.wait(timeout=10)
        started = time.monotonic()
        stdout, stderr, drain_failed = _drain(proc, timeout_s=0.5)
        elapsed = time.monotonic() - started
        assert drain_failed, "a held-open pipe must be reported as a drain failure"
        assert (stdout, stderr) == ("", "")
        assert elapsed < 5, f"drain must stay bounded; took {elapsed:.1f}s"
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def test_probe_result_dict_truncates_streams(tmp_path: Path) -> None:
    root = _fake_sdk(tmp_path)
    binary = _script(tmp_path, 'print("x" * 20000)')
    result = run_probe(
        SERIAL,
        sdk_root=root,
        probe_binary=binary,
        timeout_s=10,
        manifest_path=_manifest_for(root, tmp_path),
    )
    payload = result.as_dict()
    assert len(str(payload["stdout_tail"])) <= 4000
    json.dumps(payload)
