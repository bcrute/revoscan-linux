"""Offline lifecycle tests for the fixed read-only RGB control inventory."""

from __future__ import annotations

import ast
import errno
import os
import struct
from pathlib import Path

import pytest

import scanner3d.v4l2_controls as controls_module
from scanner3d.ownership import SignalReceived
from scanner3d.v4l2_controls import (
    CONTROL_FORMAT,
    CONTROL_WHITELIST,
    QUERYCTRL_FORMAT,
    V4L2_CTRL_FLAG_DISABLED,
    V4L2_CTRL_FLAG_INACTIVE,
    V4L2_CTRL_FLAG_READ_ONLY,
    V4L2_CTRL_FLAG_VOLATILE,
    V4L2_CTRL_FLAG_WRITE_ONLY,
    V4L2_CTRL_TYPE_INTEGER,
    VIDIOC_G_CTRL,
    VIDIOC_QUERYCTRL,
    ControlObservationStatus,
    InventoryStatus,
    V4l2RgbControlBackend,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder


def _query_bytes(
    control_id: int,
    *,
    control_type: int = V4L2_CTRL_TYPE_INTEGER,
    name: str = "control",
    flags: int = 0,
) -> bytes:
    return struct.pack(
        QUERYCTRL_FORMAT,
        control_id,
        control_type,
        name.encode().ljust(32, b"\0"),
        -10,
        100,
        1,
        5,
        flags,
        b"\0" * 8,
    )


class FakeControls:
    def __init__(
        self,
        node: Path,
        *,
        flags: dict[int, int] | None = None,
        query_types: dict[int, int] | None = None,
    ) -> None:
        self.node = node
        self.flags = flags or {}
        self.query_types = query_types or {}
        self.opened: list[str] = []
        self.closed: list[int] = []
        self.calls: list[tuple[int, int]] = []
        self.values = {spec.control_id: i + 10 for i, spec in enumerate(CONTROL_WHITELIST)}
        self.query_errors: dict[int, int] = {}
        self.value_errors: dict[int, int] = {}
        self.query_return_ids: dict[int, int] = {}
        self.value_return_ids: dict[int, int] = {}
        self.signal_on: tuple[int, int] | None = None

    def opener(self, path: str) -> int:
        self.opened.append(path)
        return os.open(path, os.O_RDONLY)

    def closer(self, fd: int) -> None:
        self.closed.append(fd)
        os.close(fd)

    def ioctl(self, fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        del fd, mutate
        control_id = struct.unpack_from("=I", packed)[0]
        self.calls.append((request, control_id))
        if self.signal_on == (request, control_id):
            raise SignalReceived(15, "fake control ioctl")
        if request == VIDIOC_QUERYCTRL:
            if control_id in self.query_errors:
                raise OSError(
                    self.query_errors[control_id], os.strerror(self.query_errors[control_id])
                )
            packed[:] = _query_bytes(
                self.query_return_ids.get(control_id, control_id),
                control_type=self.query_types.get(control_id, V4L2_CTRL_TYPE_INTEGER),
                name=next(spec.name for spec in CONTROL_WHITELIST if spec.control_id == control_id),
                flags=self.flags.get(control_id, 0),
            )
            return 0
        assert request == VIDIOC_G_CTRL
        if control_id in self.value_errors:
            raise OSError(self.value_errors[control_id], os.strerror(self.value_errors[control_id]))
        packed[:] = struct.pack(
            CONTROL_FORMAT,
            self.value_return_ids.get(control_id, control_id),
            self.values[control_id],
        )
        return 0


def _backend(healthy: SysfsBuilder, fake: FakeControls) -> V4l2RgbControlBackend:
    return V4l2RgbControlBackend(
        opener=fake.opener,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
    )


class BrokenShield:
    def __enter__(self) -> list[int]:
        raise SignalReceived(15, "cleanup guard")

    def __exit__(self, *_args: object) -> None:
        return None


@pytest.fixture
def rgb_tree(builder: SysfsBuilder) -> SysfsBuilder:
    builder.add_healthy_mini()
    return builder


def test_dry_run_has_fixed_plan_and_no_side_effects(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    outcome = _backend(rgb_tree, fake).dry_run()
    assert outcome.status is InventoryStatus.NOT_ATTEMPTED
    assert outcome.plan["max_ioctl_count"] == 12
    controls = outcome.plan["controls"]
    assert isinstance(controls, list)
    assert len(controls) == 6
    assert not fake.opened
    assert not fake.calls


def test_source_has_closed_read_only_ioctl_repertoire() -> None:
    source = Path(controls_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    request_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id.startswith("VIDIOC_")
    }
    assert request_names == {"VIDIOC_QUERYCTRL", "VIDIOC_G_CTRL"}

    forbidden_attributes = {
        "VIDIOC_S_CTRL",
        "VIDIOC_S_EXT_CTRLS",
        "VIDIOC_QUERYMENU",
        "VIDIOC_S_FMT",
        "VIDIOC_REQBUFS",
        "VIDIOC_QUERYBUF",
        "VIDIOC_QBUF",
        "VIDIOC_DQBUF",
        "VIDIOC_STREAMON",
        "VIDIOC_STREAMOFF",
        "mmap",
    }
    assert not {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ioctl_fn":
            assert len(node.args) >= 2
            request = node.args[1]
            assert isinstance(request, ast.Name)
            assert request.id in {"VIDIOC_QUERYCTRL", "VIDIOC_G_CTRL"}


def test_healthy_inventory_queries_then_reads_each_control_once(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.COMPLETE
    assert outcome.succeeded
    assert len(fake.opened) == 1
    assert len(fake.closed) == 1
    assert outcome.ioctl_count == 12
    assert [request for request, _ in fake.calls] == [
        item for spec in CONTROL_WHITELIST for item in (VIDIOC_QUERYCTRL, VIDIOC_G_CTRL)
    ]
    assert all(item.status is ControlObservationStatus.SUPPORTED for item in outcome.observations)
    assert outcome.post_close_identity is not None
    assert outcome.post_close_identity["stable"] is True


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (V4L2_CTRL_FLAG_DISABLED, ControlObservationStatus.DISABLED),
        (V4L2_CTRL_FLAG_WRITE_ONLY, ControlObservationStatus.WRITE_ONLY),
    ],
)
def test_non_readable_flags_skip_value_ioctl(
    rgb_tree: SysfsBuilder, flag: int, expected: ControlObservationStatus
) -> None:
    fake = FakeControls(rgb_tree.dev / "video2", flags={CONTROL_WHITELIST[0].control_id: flag})
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    first = outcome.observations[0]
    assert first.status is expected
    assert first.value_attempted is False
    assert outcome.ioctl_count == 11


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (V4L2_CTRL_FLAG_READ_ONLY, ControlObservationStatus.READ_ONLY),
        (V4L2_CTRL_FLAG_VOLATILE, ControlObservationStatus.VOLATILE),
        (V4L2_CTRL_FLAG_INACTIVE, ControlObservationStatus.INACTIVE),
    ],
)
def test_readable_flags_still_get_current_value(
    rgb_tree: SysfsBuilder, flag: int, expected: ControlObservationStatus
) -> None:
    fake = FakeControls(rgb_tree.dev / "video2", flags={CONTROL_WHITELIST[0].control_id: flag})
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    first = outcome.observations[0]
    assert first.status is expected
    assert first.current == 10
    assert first.value_attempted is True


def test_query_unsupported_and_errno_are_typed(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    fake.query_errors[CONTROL_WHITELIST[0].control_id] = errno.EINVAL
    fake.query_errors[CONTROL_WHITELIST[1].control_id] = errno.EIO
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.IOCTL_ERROR
    assert outcome.observations[0].status is ControlObservationStatus.UNSUPPORTED
    assert outcome.observations[1].status is ControlObservationStatus.QUERY_ERROR
    assert len(outcome.observations) == len(CONTROL_WHITELIST)
    assert outcome.observations[0].status is ControlObservationStatus.UNSUPPORTED
    assert outcome.observations[1].status is ControlObservationStatus.QUERY_ERROR
    assert len(fake.closed) == 1


def test_query_error_continues_fixed_inventory_without_retry(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    first = CONTROL_WHITELIST[0].control_id
    fake.query_errors[first] = errno.EIO
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.IOCTL_ERROR
    assert outcome.ioctl_count == 11
    assert [control_id for request, control_id in fake.calls].count(first) == 1
    assert len(fake.opened) == len(fake.closed) == 1


def test_unknown_type_is_typed_and_inventory_continues(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2", query_types={CONTROL_WHITELIST[0].control_id: 99})
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    first = outcome.observations[0]
    assert first.status is ControlObservationStatus.UNKNOWN_TYPE
    assert first.value_attempted is False
    assert outcome.ioctl_count == 11
    assert len(fake.opened) == len(fake.closed) == 1


def test_g_ctrl_return_id_mismatch_is_typed_without_retry(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    first = CONTROL_WHITELIST[0].control_id
    fake.value_return_ids[first] = CONTROL_WHITELIST[1].control_id
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    observation = outcome.observations[0]
    assert observation.status is ControlObservationStatus.VALUE_ERROR
    assert observation.value_attempted is True
    assert "returned id" in observation.detail
    assert outcome.ioctl_count == 12
    assert [item for item in fake.calls if item[0] == VIDIOC_G_CTRL].count(
        (VIDIOC_G_CTRL, first)
    ) == 1


def test_open_oserror_is_typed_and_never_closes_or_reopens(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    opens: list[str] = []

    def fail_open(path: str) -> int:
        opens.append(path)
        raise OSError(errno.EACCES, "denied")

    backend = V4l2RgbControlBackend(
        opener=fail_open,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.OPEN_ERROR
    assert outcome.ioctl_count == 0
    assert opens == [str(rgb_tree.dev / "video2")]
    assert fake.closed == []


def test_signal_during_open_is_typed_and_never_reopens(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    opens = 0

    def signal_open(_path: str) -> int:
        nonlocal opens
        opens += 1
        raise SignalReceived(15, "open")

    backend = V4l2RgbControlBackend(
        opener=signal_open,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.INTERRUPTED
    assert opens == 1
    assert outcome.ioctl_count == 0
    assert fake.closed == []


def test_value_errno_is_typed_and_cleanup_precedence_is_preserved(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    fake.value_errors[CONTROL_WHITELIST[0].control_id] = errno.EIO
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.IOCTL_ERROR
    assert outcome.observations[0].status is ControlObservationStatus.VALUE_ERROR
    assert outcome.observations[0].value_errno == errno.EIO
    assert len(fake.closed) == 1

    failing = FakeControls(rgb_tree.dev / "video2")
    close_outcome = V4l2RgbControlBackend(
        opener=failing.opener,
        ioctl_fn=failing.ioctl,
        closer=lambda _fd: (_ for _ in ()).throw(OSError(errno.EIO, "close")),
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    ).run(REAL_SERIAL)
    assert close_outcome.status is InventoryStatus.COMPLETE
    assert close_outcome.cleanup.ok is False
    assert close_outcome.exit_code == 9


@pytest.mark.parametrize("close_exc", [SignalReceived(15, "close"), RuntimeError("close")])
def test_close_exception_is_absorbed_once_and_has_cleanup_precedence(
    rgb_tree: SysfsBuilder, close_exc: BaseException
) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    close_calls = 0

    def fail_close(_fd: int) -> None:
        nonlocal close_calls
        close_calls += 1
        raise close_exc

    outcome = V4l2RgbControlBackend(
        opener=fake.opener,
        ioctl_fn=fake.ioctl,
        closer=fail_close,
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    ).run(REAL_SERIAL)
    assert close_calls == 1
    assert outcome.cleanup.attempted is True
    assert outcome.cleanup.ok is False
    assert outcome.exit_code == 9
    assert outcome.ioctl_count == 12


def test_signal_during_query_keeps_partial_progress_and_closes_once(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    fake.signal_on = (VIDIOC_QUERYCTRL, CONTROL_WHITELIST[0].control_id)
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.INTERRUPTED
    assert len(outcome.observations) == 1
    assert outcome.observations[0].status is ControlObservationStatus.INTERRUPTED
    assert outcome.ioctl_count == 1
    assert len(fake.closed) == 1


def test_signal_during_value_read_keeps_query_and_partial_progress(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    fake.signal_on = (VIDIOC_G_CTRL, CONTROL_WHITELIST[0].control_id)
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.INTERRUPTED
    assert outcome.observations[0].query is not None
    assert outcome.observations[0].value_attempted is True
    assert outcome.ioctl_count == 2
    assert len(fake.closed) == 1


def test_query_return_id_mismatch_is_typed_and_does_not_get(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    fake.query_return_ids[CONTROL_WHITELIST[0].control_id] = CONTROL_WHITELIST[1].control_id
    outcome = _backend(rgb_tree, fake).run(REAL_SERIAL)
    first = outcome.observations[0]
    assert first.status is ControlObservationStatus.QUERY_ERROR
    assert first.value_attempted is False
    assert fake.calls[0] == (VIDIOC_QUERYCTRL, CONTROL_WHITELIST[0].control_id)


def test_signal_on_cleanup_guard_still_attempts_one_close(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")

    backend = V4l2RgbControlBackend(
        opener=fake.opener,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        termination_shield=lambda: BrokenShield(),
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    )
    outcome = backend.run(REAL_SERIAL)
    assert len(fake.closed) == 1
    assert outcome.cleanup.ok is True


def test_identity_drift_prevents_all_control_ioctls(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    backend = V4l2RgbControlBackend(
        opener=fake.opener,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        identity_check=lambda _fd, _target: "identity changed",
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.IDENTITY_DRIFT
    assert not fake.calls
    assert len(fake.closed) == 1


def test_post_close_identity_drift_cannot_report_success(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    backend = V4l2RgbControlBackend(
        opener=fake.opener,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        post_close_check=lambda _target: {"stable": False, "detail": "devnum changed"},
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.IDENTITY_DRIFT
    assert outcome.succeeded is False
    assert outcome.post_close_identity == {"stable": False, "detail": "devnum changed"}


def test_default_post_close_identity_uses_injected_text_and_stat_seams(
    rgb_tree: SysfsBuilder,
) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")
    text_paths: list[Path] = []
    stat_paths: list[Path] = []

    def read_text(path: Path) -> str:
        text_paths.append(path)
        return path.read_text(encoding="utf-8").strip()

    def read_stat(path: Path) -> os.stat_result:
        stat_paths.append(path)
        return path.stat()

    outcome = V4l2RgbControlBackend(
        opener=fake.opener,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        text_reader=read_text,
        stat_reader=read_stat,
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    ).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.COMPLETE
    assert outcome.post_close_identity is not None
    assert outcome.post_close_identity["stable"] is True
    assert any(path.name == "interface" for path in text_paths)
    assert stat_paths == [rgb_tree.dev / "video2"]


def test_post_close_exception_is_typed_and_cannot_report_success(rgb_tree: SysfsBuilder) -> None:
    fake = FakeControls(rgb_tree.dev / "video2")

    def fail_post_close(_target: object) -> dict[str, object]:
        raise RuntimeError("re-resolve exploded")

    outcome = V4l2RgbControlBackend(
        opener=fake.opener,
        ioctl_fn=fake.ioctl,
        closer=fake.closer,
        post_close_check=fail_post_close,
        sysfs_root=rgb_tree.sysfs,
        dev_root=rgb_tree.dev,
    ).run(REAL_SERIAL)
    assert outcome.status is InventoryStatus.IDENTITY_DRIFT
    assert outcome.succeeded is False
    assert outcome.post_close_identity is not None
    assert outcome.post_close_identity["stable"] is False
    assert len(fake.opened) == len(fake.closed) == 1
