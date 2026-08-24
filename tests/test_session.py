"""Tests for the versioned session manifest and atomic artifact store.

These cover the Phase 2 exit gate directly: fixture round trips are
deterministic, malformed/unsafe/truncated/traversing/unsupported input is
rejected, and an interrupted session stays inspectable without ever being
called complete.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from scanner3d import session as session_module
from scanner3d.session import (
    ARTIFACT_BUCKETS,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    ArtifactRecord,
    ArtifactWriteIncomplete,
    CoordinateFrame,
    DeviceIdentity,
    FixedClock,
    SessionError,
    SessionManifest,
    SessionManifestInvalid,
    SessionPathNotAllowed,
    SessionSchemaUnsupported,
    SessionState,
    SessionStore,
    StageState,
    SystemClock,
    build_fixture_manifest,
    read_manifest,
)


def as_obj(value: object) -> dict[str, object]:
    """Narrow a manifest field to a dict so the type checker stays strict."""
    assert isinstance(value, dict)
    return value


def as_seq(value: object) -> list[object]:
    """Narrow a manifest field to a list so the type checker stays strict."""
    assert isinstance(value, list)
    return value


def make_store(tmp_path: Path, session_id: str = "s-0001") -> SessionStore:
    return SessionStore.create(
        tmp_path,
        session_id,
        software_version="scanner3d-test",
        clock=FixedClock(),
    )


def running_store(tmp_path: Path, stage: str = "depth_capture") -> SessionStore:
    store = make_store(tmp_path)
    store.start()
    store.begin_stage(stage)
    return store


# --------------------------------------------------------------------------
# Fixture round trips are deterministic
# --------------------------------------------------------------------------


def test_fixture_manifest_round_trips_byte_exactly():
    original = build_fixture_manifest()
    encoded = json.dumps(original.as_dict(), sort_keys=True, indent=2)
    restored = SessionManifest.from_dict(json.loads(encoded))
    assert restored == original
    assert json.dumps(restored.as_dict(), sort_keys=True, indent=2) == encoded


def test_fixture_manifest_is_stable_across_builds():
    assert build_fixture_manifest().as_dict() == build_fixture_manifest().as_dict()


def test_fixture_manifest_declares_the_current_schema():
    assert build_fixture_manifest().schema_version == SCHEMA_VERSION


def test_round_trip_through_a_real_file(tmp_path: Path):
    path = tmp_path / MANIFEST_NAME
    manifest = build_fixture_manifest()
    path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    assert read_manifest(path) == manifest


def test_written_session_reloads_identically(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/frame.y16", b"\x00\x01", kind="raw_depth_frame")
    store.finish_stage("depth_capture", StageState.COMPLETE, counters={"delivered": 1})
    store.finish(SessionState.COMPLETE)
    assert read_manifest(store.root / MANIFEST_NAME) == store.manifest


# --------------------------------------------------------------------------
# Unsupported schema versions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", [SCHEMA_VERSION + 1, SCHEMA_VERSION + 99, 0, -1])
def test_unsupported_schema_version_is_refused(version: int):
    raw = build_fixture_manifest().as_dict()
    raw["schema_version"] = version
    with pytest.raises(SessionSchemaUnsupported):
        SessionManifest.from_dict(raw)


def test_a_future_version_is_refused_rather_than_downgraded(tmp_path: Path):
    raw = build_fixture_manifest().as_dict()
    raw["schema_version"] = SCHEMA_VERSION + 1
    raw["a_field_this_build_does_not_know"] = "load bearing"
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SessionSchemaUnsupported):
        read_manifest(path)


def test_missing_schema_version_is_malformed_not_unsupported():
    raw = build_fixture_manifest().as_dict()
    del raw["schema_version"]
    with pytest.raises(SessionManifestInvalid):
        SessionManifest.from_dict(raw)


# --------------------------------------------------------------------------
# Malformed and truncated manifests
# --------------------------------------------------------------------------


def test_truncated_manifest_is_refused_with_a_reason(tmp_path: Path):
    path = tmp_path / MANIFEST_NAME
    full = json.dumps(build_fixture_manifest().as_dict())
    path.write_text(full[: len(full) // 2], encoding="utf-8")
    with pytest.raises(SessionManifestInvalid, match="not valid JSON"):
        read_manifest(path)


def test_empty_manifest_is_refused(tmp_path: Path):
    path = tmp_path / MANIFEST_NAME
    path.write_text("", encoding="utf-8")
    with pytest.raises(SessionManifestInvalid):
        read_manifest(path)


def test_non_object_manifest_is_refused(tmp_path: Path):
    path = tmp_path / MANIFEST_NAME
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SessionManifestInvalid, match="not a JSON object"):
        read_manifest(path)


def test_absent_manifest_is_refused(tmp_path: Path):
    with pytest.raises(SessionManifestInvalid, match="cannot stat"):
        read_manifest(tmp_path / "nope.json")


def test_oversized_manifest_is_refused_before_reading(tmp_path: Path, monkeypatch):
    """Lower the ceiling rather than build a 64 MB file; the branch is the point."""
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(build_fixture_manifest().as_dict()), encoding="utf-8")
    monkeypatch.setattr(session_module, "MAX_MANIFEST_BYTES", 8)
    with pytest.raises(SessionManifestInvalid, match="ceiling"):
        read_manifest(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("stages", {}),
        ("stages", "depth"),
        ("frames", 3),
        ("limitations", "a string"),
        ("limitations", [1, 2]),
        ("device", "not an object"),
        ("session_id", 17),
        ("state", "nonexistent_state"),
        ("created_wall", None),
        ("calibration", []),
    ],
)
def test_structurally_malformed_fields_are_refused(key: str, value: object):
    raw = build_fixture_manifest().as_dict()
    raw[key] = value
    with pytest.raises(SessionManifestInvalid):
        SessionManifest.from_dict(raw)


def test_unsafe_session_id_in_a_manifest_is_refused():
    raw = build_fixture_manifest().as_dict()
    raw["session_id"] = "../../etc"
    with pytest.raises(SessionManifestInvalid, match="safe identifier"):
        SessionManifest.from_dict(raw)


def test_a_boolean_is_not_accepted_as_an_integer():
    raw = build_fixture_manifest().as_dict()
    device = as_obj(raw["device"])
    device["vendor_id"] = True
    raw["device"] = device
    with pytest.raises(SessionManifestInvalid, match="integer"):
        SessionManifest.from_dict(raw)


def test_non_integer_counters_are_refused():
    raw = build_fixture_manifest().as_dict()
    stages = as_seq(raw["stages"])
    stage = as_obj(stages[0])
    stage["counters"] = {"delivered": "many"}
    stages[0] = stage
    raw["stages"] = stages
    with pytest.raises(SessionManifestInvalid, match="must be an integer"):
        SessionManifest.from_dict(raw)


def test_malformed_artifact_record_is_refused():
    raw = build_fixture_manifest().as_dict()
    stages = as_seq(raw["stages"])
    stage = as_obj(stages[0])
    stage["artifacts"] = ["not an object"]
    stages[0] = stage
    raw["stages"] = stages
    with pytest.raises(SessionManifestInvalid, match="must be an object"):
        SessionManifest.from_dict(raw)


# --------------------------------------------------------------------------
# Path traversal and unsafe artifact names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.bin",
        "raw/../../escape.bin",
        "/etc/passwd",
        "raw//double.bin",
        "raw/./x.bin",
        "raw/sub/",
        "raw/../derived/x.bin",
        "raw/sub/../../../x.bin",
        "raw\\windows.bin",
        "notabucket/x.bin",
        "raw",
        "",
        " raw/x.bin",
        "raw/x.bin ",
        "raw/.hidden",
        "raw/-leading.bin",
    ],
)
def test_unsafe_artifact_paths_are_refused(tmp_path: Path, unsafe: str):
    store = running_store(tmp_path)
    with pytest.raises(SessionPathNotAllowed):
        store.write_artifact("depth_capture", unsafe, b"x", kind="test")


def test_a_refused_path_creates_no_file(tmp_path: Path):
    store = running_store(tmp_path)
    before = sorted(p for p in store.root.rglob("*") if p.is_file())
    with pytest.raises(SessionPathNotAllowed):
        store.write_artifact("depth_capture", "../escape.bin", b"x", kind="test")
    assert sorted(p for p in store.root.rglob("*") if p.is_file()) == before
    assert not (tmp_path / "escape.bin").exists()


def test_a_symlinked_bucket_cannot_redirect_a_write(tmp_path: Path):
    store = running_store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    nested = store.root / "raw" / "nested"
    nested.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SessionPathNotAllowed, match="outside"):
        store.write_artifact("depth_capture", "raw/nested/x.bin", b"x", kind="test")
    assert not (outside / "x.bin").exists()


def test_unsafe_session_id_is_refused_at_creation(tmp_path: Path):
    for unsafe in ("../evil", "a/b", "", ".hidden", "x" * 65):
        with pytest.raises(SessionPathNotAllowed):
            SessionStore.create(tmp_path, unsafe, software_version="t", clock=FixedClock())


def test_unsafe_stage_name_is_refused(tmp_path: Path):
    store = make_store(tmp_path)
    store.start()
    for unsafe in ("../x", "a/b", ""):
        with pytest.raises(SessionPathNotAllowed):
            store.begin_stage(unsafe)


# --------------------------------------------------------------------------
# Atomic, complete-or-fail writes; raw evidence is immutable
# --------------------------------------------------------------------------


def test_artifact_is_written_with_a_matching_checksum(tmp_path: Path):
    store = running_store(tmp_path)
    payload = b"depth frame bytes"
    record = store.write_artifact("depth_capture", "raw/f.y16", payload, kind="raw_depth_frame")
    assert record.bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.immutable is True
    assert (store.root / "raw" / "f.y16").read_bytes() == payload


def test_raw_artifact_is_never_overwritten(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.y16", b"first", kind="raw_depth_frame")
    with pytest.raises(ArtifactWriteIncomplete) as caught:
        store.write_artifact("depth_capture", "raw/f.y16", b"second", kind="raw_depth_frame")
    assert caught.value.errno == errno.EEXIST
    assert (store.root / "raw" / "f.y16").read_bytes() == b"first"


def test_derived_artifact_is_also_not_overwritten(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "derived/cloud.ply", b"ply", kind="point_cloud")
    with pytest.raises(ArtifactWriteIncomplete):
        store.write_artifact("depth_capture", "derived/cloud.ply", b"other", kind="point_cloud")


def test_derived_artifact_is_not_marked_immutable(tmp_path: Path):
    store = running_store(tmp_path)
    record = store.write_artifact("depth_capture", "derived/c.ply", b"ply", kind="point_cloud")
    assert record.immutable is False


def test_oversized_artifact_is_refused_and_leaves_no_file(tmp_path: Path):
    store = running_store(tmp_path)
    with pytest.raises(ArtifactWriteIncomplete) as caught:
        store.write_artifact("depth_capture", "raw/big.bin", b"12345", kind="raw", ceiling=4)
    assert caught.value.errno == errno.EFBIG
    assert not (store.root / "raw" / "big.bin").exists()
    assert store.manifest.artifacts == ()


class WriteFaultInjector:
    """Fail or shorten writes to one target file only.

    ``os.write`` is global and pytest uses it for its own terminal output, so a
    blanket patch deadlocks the run. This watches ``os.open`` for the exact
    artifact path and misbehaves only on that descriptor; every other write is
    passed straight through.
    """

    def __init__(self, target: Path, behaviour: str) -> None:
        self.target = target
        self.behaviour = behaviour
        self.real_open = os.open
        self.real_write = os.write
        self.real_close = os.close
        self.watched: set[int] = set()
        self.calls = 0

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(os, "open", self._open)
        monkeypatch.setattr(os, "write", self._write)
        monkeypatch.setattr(os, "close", self._close)

    def _open(self, path, flags, mode=0o777, **kwargs):
        fd = self.real_open(path, flags, mode, **kwargs)
        if Path(path) == self.target:
            self.watched.add(fd)
        return fd

    def _close(self, fd: int) -> None:
        # Descriptor numbers are reused, so a watched fd must stop being watched
        # the moment it is closed or the next unrelated file inherits the fault.
        self.watched.discard(fd)
        self.real_close(fd)

    def _write(self, fd: int, data) -> int:
        if fd not in self.watched:
            return self.real_write(fd, data)
        self.calls += 1
        payload = bytes(data)
        if self.behaviour == "stall":
            return 0
        if self.behaviour == "short":
            return self.real_write(fd, payload[:1])
        if self.calls == 1:
            return self.real_write(fd, payload[:2])
        raise OSError(errno.EIO, "synthetic failure")


def test_a_failed_write_leaves_no_partial_artifact(tmp_path: Path, monkeypatch):
    store = running_store(tmp_path)
    target = store.root / "raw" / "partial.bin"
    WriteFaultInjector(target, "fail_after_partial").install(monkeypatch)
    with pytest.raises(OSError):
        store.write_artifact("depth_capture", "raw/partial.bin", b"abcdefgh", kind="raw")
    monkeypatch.undo()
    assert not target.exists()
    assert store.manifest.artifacts == ()


def test_a_zero_progress_write_is_an_error_not_a_spin(tmp_path: Path, monkeypatch):
    store = running_store(tmp_path)
    target = store.root / "raw" / "stall.bin"
    WriteFaultInjector(target, "stall").install(monkeypatch)
    with pytest.raises(ArtifactWriteIncomplete) as caught:
        store.write_artifact("depth_capture", "raw/stall.bin", b"abc", kind="raw")
    assert caught.value.errno == errno.EIO
    monkeypatch.undo()
    assert not target.exists()


def test_a_short_write_is_resumed_from_the_kernel_offset(tmp_path: Path, monkeypatch):
    store = running_store(tmp_path)
    target = store.root / "raw" / "slow.bin"
    injector = WriteFaultInjector(target, "short")
    injector.install(monkeypatch)
    store.write_artifact("depth_capture", "raw/slow.bin", b"abcdefgh", kind="raw")
    monkeypatch.undo()
    assert target.read_bytes() == b"abcdefgh"
    assert injector.calls == 8


def test_manifest_write_leaves_no_temporary_behind(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.bin", b"x", kind="raw")
    leftovers = [p.name for p in store.root.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_writing_to_an_unknown_stage_is_refused(tmp_path: Path):
    store = make_store(tmp_path)
    store.start()
    with pytest.raises(SessionError, match="never begun"):
        store.write_artifact("no_such_stage", "raw/f.bin", b"x", kind="raw")


# --------------------------------------------------------------------------
# Nothing becomes complete by omission
# --------------------------------------------------------------------------


def test_a_new_session_is_planned_not_complete(tmp_path: Path):
    store = make_store(tmp_path)
    assert store.manifest.state is SessionState.PLANNED
    assert store.manifest.is_complete is False


def test_artifacts_alone_do_not_make_a_session_complete(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.y16", b"data", kind="raw_depth_frame")
    assert store.manifest.is_complete is False
    assert store.manifest.state is SessionState.RUNNING


@pytest.mark.parametrize("stuck", [StageState.RUNNING, StageState.INCOMPLETE, StageState.PLANNED])
def test_a_complete_session_with_an_unfinished_stage_is_not_complete(stuck: StageState):
    """A writer that died between updating a stage and the session must not
    read back as finished."""
    manifest = build_fixture_manifest()
    contradictory = replace(
        manifest,
        state=SessionState.COMPLETE,
        stages=(replace(manifest.stages[0], state=stuck),),
    )
    assert contradictory.state is SessionState.COMPLETE
    assert contradictory.is_complete is False


def test_finish_requires_a_terminal_state(tmp_path: Path):
    store = running_store(tmp_path)
    for non_terminal in (SessionState.PLANNED, SessionState.RUNNING, SessionState.INCOMPLETE):
        with pytest.raises(SessionError, match="not a terminal"):
            store.finish(non_terminal)


def test_finish_stage_requires_a_terminal_state(tmp_path: Path):
    store = running_store(tmp_path)
    for non_terminal in (StageState.PLANNED, StageState.RUNNING, StageState.INCOMPLETE):
        with pytest.raises(SessionError, match="not a terminal"):
            store.finish_stage("depth_capture", non_terminal)


def test_finishing_an_unknown_stage_is_refused(tmp_path: Path):
    store = running_store(tmp_path)
    with pytest.raises(SessionError, match="never begun"):
        store.finish_stage("other", StageState.COMPLETE)


def test_a_terminal_session_refuses_further_work(tmp_path: Path):
    store = running_store(tmp_path)
    store.finish_stage("depth_capture", StageState.COMPLETE)
    store.finish(SessionState.COMPLETE)
    with pytest.raises(SessionError, match="already"):
        store.begin_stage("another")
    with pytest.raises(SessionError, match="already"):
        store.start()


def test_a_failed_stage_records_its_error(tmp_path: Path):
    store = running_store(tmp_path)
    store.finish_stage("depth_capture", StageState.FAILED, error="poll timed out")
    store.finish(SessionState.FAILED, error="capture produced no frame")
    stage = store.manifest.stage("depth_capture")
    assert stage is not None
    assert stage.state is StageState.FAILED
    assert stage.error == "poll timed out"
    assert store.manifest.is_complete is False


def test_cancellation_is_distinct_from_failure(tmp_path: Path):
    store = running_store(tmp_path)
    store.finish_stage("depth_capture", StageState.CANCELLED)
    store.finish(SessionState.CANCELLED)
    assert store.manifest.state is SessionState.CANCELLED
    assert store.manifest.is_complete is False


def test_a_genuinely_complete_session_reports_complete(tmp_path: Path):
    store = running_store(tmp_path)
    store.finish_stage("depth_capture", StageState.COMPLETE, counters={"delivered": 3})
    store.finish(SessionState.COMPLETE)
    assert store.manifest.is_complete is True


# --------------------------------------------------------------------------
# An interrupted session stays inspectable, and honest
# --------------------------------------------------------------------------


def test_reopening_an_interrupted_session_marks_it_incomplete(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.y16", b"data", kind="raw_depth_frame")
    # The owning process dies here: no finish_stage, no finish.
    reopened = SessionStore.open(store.root, clock=FixedClock())
    assert reopened.manifest.state is SessionState.INCOMPLETE
    assert reopened.manifest.is_complete is False
    stage = reopened.manifest.stage("depth_capture")
    assert stage is not None
    assert stage.state is StageState.INCOMPLETE


def test_an_interrupted_session_keeps_its_evidence(tmp_path: Path):
    store = running_store(tmp_path)
    record = store.write_artifact("depth_capture", "raw/f.y16", b"data", kind="raw_depth_frame")
    reopened = SessionStore.open(store.root, clock=FixedClock())
    assert reopened.manifest.artifacts == (record,)
    assert (reopened.root / "raw" / "f.y16").read_bytes() == b"data"
    assert reopened.verify_artifacts() == {}


def test_the_incomplete_label_is_persisted_for_the_next_reader(tmp_path: Path):
    store = running_store(tmp_path)
    SessionStore.open(store.root, clock=FixedClock())
    assert read_manifest(store.root / MANIFEST_NAME).state is SessionState.INCOMPLETE


def test_reopening_a_finished_session_does_not_relabel_it(tmp_path: Path):
    store = running_store(tmp_path)
    store.finish_stage("depth_capture", StageState.COMPLETE)
    store.finish(SessionState.COMPLETE)
    reopened = SessionStore.open(store.root, clock=FixedClock())
    assert reopened.manifest.state is SessionState.COMPLETE
    assert reopened.manifest.is_complete is True


def test_reopening_records_why_the_state_changed(tmp_path: Path):
    store = running_store(tmp_path)
    reopened = SessionStore.open(store.root, clock=FixedClock())
    assert "interrupted" in reopened.manifest.note


# --------------------------------------------------------------------------
# Integrity checking
# --------------------------------------------------------------------------


def test_verify_reports_a_clean_session(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.y16", b"data", kind="raw_depth_frame")
    assert store.verify_artifacts() == {}


def test_verify_detects_a_missing_artifact(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.y16", b"data", kind="raw_depth_frame")
    (store.root / "raw" / "f.y16").unlink()
    assert store.verify_artifacts() == {"raw/f.y16": "missing"}


def test_verify_detects_tampering(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.y16", b"data", kind="raw_depth_frame")
    (store.root / "raw" / "f.y16").write_bytes(b"XXXX")
    assert store.verify_artifacts() == {"raw/f.y16": "checksum mismatch"}


def test_verify_detects_a_size_change(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/f.y16", b"data", kind="raw_depth_frame")
    (store.root / "raw" / "f.y16").write_bytes(b"much longer than before")
    assert "expected 4 bytes" in store.verify_artifacts()["raw/f.y16"]


# --------------------------------------------------------------------------
# Creation, layout and redaction
# --------------------------------------------------------------------------


def test_create_makes_the_expected_layout(tmp_path: Path):
    store = make_store(tmp_path)
    for bucket in ARTIFACT_BUCKETS:
        assert (store.root / bucket).is_dir()
    assert (store.root / MANIFEST_NAME).is_file()


def test_create_refuses_an_existing_session_id(tmp_path: Path):
    make_store(tmp_path, "dup")
    with pytest.raises(FileExistsError):
        make_store(tmp_path, "dup")


def test_redaction_hides_the_serial_but_keeps_the_structure():
    manifest = build_fixture_manifest()
    redacted = manifest.redacted()
    device = redacted["device"]
    assert isinstance(device, dict)
    assert device["serial"] == "[redacted]"
    assert device["usb_path"] == "1-1.1"
    assert redacted["state"] == manifest.state.value
    assert len(as_seq(redacted["stages"])) == len(manifest.stages)


def test_redaction_keeps_checksums_visible():
    redacted = build_fixture_manifest().redacted()
    stages = redacted["stages"]
    assert isinstance(stages, list)
    artifact = as_seq(as_obj(stages[0])["artifacts"])[0]
    assert isinstance(artifact, dict)
    assert artifact["sha256"] == build_fixture_manifest().stages[0].artifacts[0].sha256


def test_redaction_does_not_mutate_the_manifest():
    manifest = build_fixture_manifest()
    manifest.redacted()
    assert manifest.device is not None
    assert manifest.device.serial == "FIXTURESERIAL0001"


def test_limitations_are_recorded(tmp_path: Path):
    store = running_store(tmp_path)
    store.add_limitation("depth units are unverified")
    assert "depth units are unverified" in store.manifest.limitations


def test_stage_lookup_returns_none_for_an_unknown_name():
    assert build_fixture_manifest().stage("nope") is None


def test_beginning_a_stage_twice_keeps_its_artifacts(tmp_path: Path):
    store = running_store(tmp_path)
    store.write_artifact("depth_capture", "raw/one.bin", b"1", kind="raw")
    store.begin_stage("depth_capture")
    assert [a.path for a in store.manifest.artifacts] == ["raw/one.bin"]
    restarted = store.manifest.stage("depth_capture")
    assert restarted is not None
    assert restarted.state is StageState.RUNNING


def test_coordinate_frame_round_trips():
    frame = CoordinateFrame(
        name="depth", units="mm", axes="x_right,y_down,z_forward", handedness="right"
    )
    assert CoordinateFrame.from_dict(frame.as_dict()) == frame


def test_device_identity_round_trips():
    device = DeviceIdentity(vendor_id=0x2207, product_id=0x110C, serial="ABC")
    restored = DeviceIdentity.from_dict(device.as_dict())
    assert restored == device
    assert device.as_dict()["usb_identity"] == "2207:110c"


def test_artifact_record_round_trips():
    record = ArtifactRecord(
        path="raw/x.bin",
        bytes=3,
        sha256=hashlib.sha256(b"abc").hexdigest(),
        kind="raw",
        immutable=True,
        created_wall="2026-01-01T00:00:00.000000+00:00",
    )
    assert ArtifactRecord.from_dict(record.as_dict()) == record


def test_system_clock_produces_a_parseable_stamp():
    from datetime import datetime as real_datetime

    clock = SystemClock()
    assert real_datetime.fromisoformat(clock.wall()).tzinfo is not None
    assert isinstance(clock.monotonic(), float)
