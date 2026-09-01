"""Waiting for the depth node to be USABLE, not merely to exist.

Why this exists, observed
-------------------------
On 2026-08-26 the owner opened the window and Preview failed immediately. What
the log RECORDS: the GUI's enumeration-only wait returned in 0.003 s, the open
that followed it failed with
``PermissionError: [Errno 13] Permission denied: '/dev/video0'``, and that node
had been created in the same second. Inspected afterwards it was
``root:video 0660`` with a valid per-user ACL, and this account is not in
``video``, so that ACL is what makes it openable at all.

What the log does NOT record is the node's access at the instant of the failing
open. GUI startup reads the firmware, which re-enumerates the depth function,
and a per-seat ACL being attached a moment after the kernel registers the node
is the mechanism those four observations fit - an inference, and the one this
module is named for, but not a proven timing.

The tests below do not depend on it. They ask the question that is robust
whatever the reason a registered node is not yet openable: does the sole index-0
node of the DEPTH role exist, and is it readable, and is it writable - resolved
fresh on every poll, because the device re-enumerates and its sysfs path, bus
address and ``videoN`` number can all change between one poll and the next.

Everything runs against a synthesized sysfs/dev tree. Nothing here touches the
host's ``/sys``, ``/dev``, USB or the scanner, and - asserted directly rather
than assumed - nothing opens a device node, because opening a node to find out
whether it can be opened is the race being waited out.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import v4l2_stream
from scanner3d.discovery import find_scanners
from scanner3d.errors import DoctorStatus
from scanner3d.v4l2_stream import (
    READINESS_POLL_S,
    REENUMERATION_TIMEOUT_S,
    CaptureReadiness,
    wait_for_capture_access,
    wait_for_scanner,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder

OTHER_SERIAL = "OTHERSCANNER0002"

#: A depth function that has registered its metadata node and NOT its capture
#: node. Real: ``uvcvideo`` registers them separately, and index 1 is not a
#: thing to stream depth from however present it is.
METADATA_ONLY = (("video1", 1),)


def add_scanner(
    builder: SysfsBuilder,
    *,
    usb_path: str = "1-4.1",
    serial: str = REAL_SERIAL,
    devnum: str = "24",
    depth_nodes: tuple[tuple[str, int], ...] = (("video0", 0), ("video1", 1)),
    rgb_nodes: tuple[tuple[str, int], ...] = (("video2", 0), ("video3", 1)),
    depth_mode: int = 0o660,
    rgb_mode: int = 0o660,
) -> Path:
    """One scanner, with each camera's node numbers and modes chosen per test."""
    base = builder.add_usb_device(usb_path, serial=serial, devnum=devnum)
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{serial}",
        video_nodes=depth_nodes,
        node_mode=depth_mode,
    )
    builder.add_interface(
        base,
        2,
        interface_string=f"RGBCam{serial}",
        video_nodes=rgb_nodes,
        node_mode=rgb_mode,
    )
    return base


def wait(builder: SysfsBuilder, **kwargs: object) -> CaptureReadiness:
    """The real wait, rooted at the synthesized tree instead of the host's."""
    return wait_for_capture_access(sysfs_root=builder.sysfs, dev_root=builder.dev, **kwargs)  # type: ignore[arg-type]


def probe(builder: SysfsBuilder, **kwargs: object) -> CaptureReadiness:
    """Exactly one poll, taken by giving the wait no time at all to retry."""
    return wait(builder, timeout_s=0.0, **kwargs)


class Poller:
    """The wait's own clock and sleep, replaced by a script of world changes.

    Nothing sleeps: time moves only when the wait asks it to, and each sleep
    runs the next step of the script. That is what lets "absent, then present
    but not yet usable, then usable" be written as the sequence it actually is,
    with an exact poll count rather than a race against a real ACL.

    Only :mod:`scanner3d.v4l2_stream`'s view of ``time`` is replaced, so nothing
    outside the wait under test is affected.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *steps: Callable[[], None],
        tick: float = READINESS_POLL_S,
    ) -> None:
        self.now = 0.0
        self.tick = tick
        self.steps = list(steps)
        self.sleeps = 0
        monkeypatch.setattr(
            v4l2_stream,
            "time",
            SimpleNamespace(monotonic=lambda: self.now, sleep=self._sleep),
        )

    def _sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += max(seconds, self.tick)
        if self.steps:
            self.steps.pop(0)()


@pytest.fixture
def no_devnode_opens(monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder) -> list[str]:
    """Fail loudly if anything under the fake ``/dev`` is opened.

    ``os.open`` is how every descriptor in this package is taken, so a readiness
    check that reached for one would land here. Paths outside ``/dev`` pass
    through untouched: reading sysfs attributes is exactly what discovery is
    supposed to do.
    """
    attempts: list[str] = []
    real_open = os.open

    def guarded(path, *args, **kwargs):
        if str(path).startswith(str(builder.dev)):
            attempts.append(str(path))
            raise AssertionError(f"readiness opened a device node: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded)
    return attempts


# -- the window the owner actually hit ---------------------------------------


def test_absent_then_present_but_inaccessible_then_usable_is_waited_through(
    monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder, no_devnode_opens: list[str]
) -> None:
    """The reported sequence, replayed in order and required to end in success.

    Poll 1 sees nothing attached. Poll 2 sees the re-enumerated scanner with a
    node the kernel has registered and this process cannot yet use - the state
    the failed open of 2026-08-26 is inferred to have been taken in. Poll 3 sees
    it become usable. Anything that returns ready before poll 3 has reproduced
    the defect, whatever the reason for the gap.
    """

    def attaches_with_an_unusable_node() -> None:
        add_scanner(builder, depth_mode=0o000)

    def the_acl_arrives() -> None:
        (builder.dev / "video0").chmod(0o660)

    poller = Poller(monkeypatch, attaches_with_an_unusable_node, the_acl_arrives)

    result = wait(builder)

    assert result.ready is True
    assert result.timed_out is False
    assert result.polls == 3
    assert poller.sleeps == 2
    assert result.serial == REAL_SERIAL
    assert result.node_path == str(builder.dev / "video0")
    assert result.access is not None and result.access.usable
    assert result.status is DoctorStatus.OK
    assert no_devnode_opens == []


def test_a_registered_but_unusable_node_is_not_ready(
    builder: SysfsBuilder, no_devnode_opens: list[str]
) -> None:
    """Existence is not the question. The reason has to say which one it is."""
    add_scanner(builder, depth_mode=0o000)

    result = probe(builder)

    assert result.ready is False
    assert result.node_path == str(builder.dev / "video0")
    assert "is not readable and writable by this process" in result.reason
    assert result.status is DoctorStatus.PERMISSION_DENIED
    assert no_devnode_opens == []


def test_an_absent_scanner_is_reported_as_absent_not_as_permission(
    builder: SysfsBuilder,
) -> None:
    """Two failures, two different fixes, and the message must not confuse them."""
    result = probe(builder)

    assert result.ready is False
    assert result.access is None
    assert result.status is DoctorStatus.DEVICE_ABSENT
    assert "no 2207:110c scanner is attached" in result.reason
    assert "permission problem" not in result.failure_text()


# -- which scanner is waited for ---------------------------------------------


def test_an_explicit_serial_waits_for_that_exact_scanner(
    monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder
) -> None:
    """Another scanner being ready is not this scanner being ready.

    The other unit is attached first and perfectly usable throughout. A wait
    that answered "is A scanner usable?" would return on poll 1 and the stream
    would then open the asked-for one's node, which is not.
    """
    add_scanner(
        builder,
        usb_path="1-5",
        serial=OTHER_SERIAL,
        devnum="30",
        depth_nodes=(("video4", 0), ("video5", 1)),
        rgb_nodes=(("video6", 0), ("video7", 1)),
    )
    Poller(monkeypatch, lambda: (builder.dev / "video0").chmod(0o660))
    add_scanner(builder, depth_mode=0o000)

    result = wait(builder, serial=REAL_SERIAL)

    assert result.ready is True
    assert result.polls == 2
    assert result.serial == REAL_SERIAL
    assert result.node_path == str(builder.dev / "video0")


def test_an_explicit_serial_that_is_absent_never_settles_for_another(
    builder: SysfsBuilder,
) -> None:
    """A different unit on the bus is not the one that was asked for."""
    add_scanner(builder, serial=OTHER_SERIAL)

    result = probe(builder, serial=REAL_SERIAL)

    assert result.ready is False
    assert result.serial == REAL_SERIAL
    assert f"no 2207:110c scanner with serial {REAL_SERIAL!r} is attached" in result.reason


def test_one_attached_scanner_is_still_waited_on_by_node(
    builder: SysfsBuilder,
) -> None:
    """No serial asked for and exactly one attached is not a guess."""
    add_scanner(builder)

    result = probe(builder)

    assert result.ready is True
    assert result.deferred is False
    assert result.serial == REAL_SERIAL
    assert result.node_path == str(builder.dev / "video0")


def test_several_attached_scanners_end_the_wait_and_leave_the_choice_alone(
    builder: SysfsBuilder,
) -> None:
    """Ambiguity belongs to the stream's resolver, and blocking first helps nobody.

    Choosing a node here would be exactly the guess this package refuses
    everywhere else, and waiting the full ceiling before letting the resolver
    say so would just be six silent seconds in front of the same refusal. So the
    wait ends, ready, with no node named - and says that is what it did.
    """
    add_scanner(builder)
    add_scanner(
        builder,
        usb_path="1-5",
        serial=OTHER_SERIAL,
        devnum="30",
        depth_nodes=(("video4", 0), ("video5", 1)),
        rgb_nodes=(("video6", 0), ("video7", 1)),
    )

    result = probe(builder)

    assert result.ready is True
    assert result.deferred is True
    assert result.node_path is None
    assert result.access is None
    assert "no serial was asked for" in result.reason
    assert REAL_SERIAL in result.reason and OTHER_SERIAL in result.reason


def test_serial_none_means_any_scanner_and_does_not_read_configuration(
    monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder
) -> None:
    """``None`` is "any scanner", exactly as the enumeration-only wait meant it.

    A configured serial is present and names a unit that is NOT attached. If the
    wait reached for configuration behind the caller's back, this would time out
    against an absent device; instead it answers about the one that is here. The
    caller that wants the configured scanner resolves it and says so - which is
    what the GUI does.
    """
    monkeypatch.setenv("SCANNER3D_SERIAL", OTHER_SERIAL)
    add_scanner(builder, serial=REAL_SERIAL)

    result = probe(builder)

    assert result.ready is True
    assert result.deferred is False
    assert result.serial == REAL_SERIAL
    assert result.node_path == str(builder.dev / "video0")


def test_serial_none_with_several_attached_still_does_not_read_configuration(
    monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder
) -> None:
    """The ambiguous case is deferred, not silently settled from a config file.

    Configuration could resolve this, and resolving it here would be the wait
    quietly making a choice its caller did not ask it to make. The GUI resolves
    the same configuration itself, one level up, where the answer also decides
    which scanner colour is aimed at.
    """
    monkeypatch.setenv("SCANNER3D_SERIAL", REAL_SERIAL)
    add_scanner(builder)
    add_scanner(
        builder,
        usb_path="1-5",
        serial=OTHER_SERIAL,
        devnum="30",
        depth_nodes=(("video4", 0), ("video5", 1)),
        rgb_nodes=(("video6", 0), ("video7", 1)),
    )

    result = probe(builder)

    assert result.ready is True
    assert result.deferred is True
    assert result.serial is None
    assert result.node_path is None


# -- which node, of which role -----------------------------------------------


def test_readiness_is_never_keyed_to_a_fixed_node_number(builder: SysfsBuilder) -> None:
    """Depth on video7 is answered about video7. The numbers are probe order."""
    add_scanner(
        builder,
        depth_nodes=(("video7", 0), ("video8", 1)),
        rgb_nodes=(("video3", 0), ("video4", 1)),
    )

    result = probe(builder)

    assert result.ready is True
    assert result.node_path == str(builder.dev / "video7")


def test_a_usable_rgb_node_does_not_make_the_depth_role_ready(
    builder: SysfsBuilder,
) -> None:
    """The roles are separate cameras with separate ACLs. Only depth is streamed."""
    add_scanner(builder, depth_mode=0o000, rgb_mode=0o660)

    result = probe(builder)

    assert result.ready is False
    assert result.node_path == str(builder.dev / "video0")
    assert result.status is DoctorStatus.PERMISSION_DENIED


def test_a_metadata_node_alone_is_not_a_capture_node(builder: SysfsBuilder) -> None:
    """Index 1 is the UVC metadata node, and it is present before index 0 is."""
    add_scanner(builder, depth_nodes=METADATA_ONLY)

    result = probe(builder)

    assert result.ready is False
    assert result.node_path is None
    assert "exposes no V4L2 capture node yet" in result.reason
    assert "video1 (index 1)" in result.reason
    assert result.status is DoctorStatus.DEVICE_ABSENT


def test_a_depth_function_that_has_not_appeared_yet_is_not_ready(
    builder: SysfsBuilder,
) -> None:
    """A rebind that has re-registered RGB and not yet depth is mid-flight."""
    base = builder.add_usb_device()
    builder.add_interface(
        base, 2, interface_string=f"RGBCam{REAL_SERIAL}", video_nodes=(("video2", 0),)
    )

    result = probe(builder)

    assert result.ready is False
    assert "exposes no 'depth' camera function yet" in result.reason


def test_two_capture_nodes_on_one_function_are_refused_rather_than_chosen(
    builder: SysfsBuilder,
) -> None:
    """A tree this module was not written against. Picking would be a guess."""
    add_scanner(builder, depth_nodes=(("video0", 0), ("video9", 0)))

    result = probe(builder)

    assert result.ready is False
    assert "exposes 2 capture nodes" in result.reason
    assert "refusing to choose between them" in result.reason


def test_two_devices_reporting_one_serial_are_refused_rather_than_chosen(
    builder: SysfsBuilder,
) -> None:
    add_scanner(builder)
    add_scanner(
        builder,
        usb_path="1-5",
        devnum="30",
        depth_nodes=(("video4", 0), ("video5", 1)),
        rgb_nodes=(("video6", 0), ("video7", 1)),
    )

    result = probe(builder, serial=REAL_SERIAL)

    assert result.ready is False
    assert "2 attached devices report serial" in result.reason


# -- the ceiling -------------------------------------------------------------


def test_the_wait_is_bounded_by_the_reenumeration_ceiling() -> None:
    """The ceiling is the existing one, not a second number to keep in step."""
    default = inspect.signature(wait_for_capture_access).parameters["timeout_s"].default
    assert default == REENUMERATION_TIMEOUT_S


def test_a_permanently_inaccessible_node_times_out_and_says_it_is_permission(
    monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder, no_devnode_opens: list[str]
) -> None:
    """The failure a real permission problem must produce: bounded and explicit.

    Not a spin, and specifically not a Preview that silently never starts. The
    text has to name the node, the mode and the ownership, because "permission
    denied" without them is a dead end for the person reading it.
    """
    poller = Poller(monkeypatch)
    add_scanner(builder, depth_mode=0o000)

    result = wait(builder)

    assert result.ready is False
    assert result.timed_out is True
    assert poller.now <= REENUMERATION_TIMEOUT_S + READINESS_POLL_S
    assert result.waited_s <= REENUMERATION_TIMEOUT_S + READINESS_POLL_S
    assert result.polls > 1
    assert result.status is DoctorStatus.PERMISSION_DENIED
    text = result.failure_text()
    assert str(builder.dev / "video0") in text
    assert "mode 0o0" in text
    assert "none of which opened a device node" in text
    assert "permission problem, not a scanner fault" in text
    assert no_devnode_opens == []


def test_an_absent_scanner_times_out_without_the_permission_advice(
    monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder
) -> None:
    """Advice about ACLs would send the owner looking in the wrong place."""
    Poller(monkeypatch)

    result = wait(builder)

    assert result.ready is False
    assert result.timed_out is True
    assert "permission problem" not in result.failure_text()
    assert "no 2207:110c scanner is attached" in result.failure_text()


def test_the_readiness_state_is_json_serializable(builder: SysfsBuilder) -> None:
    """It is evidence, so it has to be recordable alongside everything else."""
    import json

    add_scanner(builder)

    payload = json.loads(json.dumps(probe(builder).as_dict()))

    assert payload["ready"] is True
    assert payload["serial"] == REAL_SERIAL
    assert payload["status"] == "ok"
    assert payload["access"]["path"] == str(builder.dev / "video0")


# -- the enumeration-only wait, kept as a public API -------------------------


@pytest.fixture
def enumeration(monkeypatch: pytest.MonkeyPatch, builder: SysfsBuilder) -> None:
    """Point the root-less ``find_scanners`` at the synthesized tree.

    :func:`~scanner3d.v4l2_stream.wait_for_scanner` takes no roots - it is the
    pre-existing signature and changing it would defeat keeping it - so the
    fake tree is supplied here rather than by an argument.
    """
    monkeypatch.setattr(
        v4l2_stream,
        "find_scanners",
        lambda **_kwargs: find_scanners(sysfs_root=builder.sysfs, dev_root=builder.dev),
    )


def test_the_enumeration_only_wait_keeps_its_signature(builder: SysfsBuilder) -> None:
    """It is importable API, so its shape is part of what callers rely on.

    The GUI no longer uses it, which is not a reason to delete something another
    caller may hold: ``serial`` positional-or-keyword and defaulting to None,
    ``timeout_s`` keyword-only on the same measured ceiling.
    """
    parameters = inspect.signature(wait_for_scanner).parameters

    assert list(parameters) == ["serial", "timeout_s"]
    assert parameters["serial"].default is None
    assert parameters["serial"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["timeout_s"].default == REENUMERATION_TIMEOUT_S


def test_the_enumeration_only_wait_still_answers_about_enumeration(
    builder: SysfsBuilder, enumeration: None
) -> None:
    """Its behaviour is unchanged - including the part the GUI moved off.

    The same tree, at the same instant: a node that is registered and that this
    process cannot open. "Has a scanner come back?" is True and "may I use its
    node?" is False, which is the whole reason the GUI now asks the second
    question. Both answers are correct; only one of them is safe to open on.
    """
    add_scanner(builder, depth_mode=0o000)

    assert wait_for_scanner(timeout_s=0.0) is True
    assert probe(builder).ready is False


def test_the_enumeration_only_wait_returns_false_when_nothing_is_attached(
    builder: SysfsBuilder, enumeration: None
) -> None:
    """Bounded, and answering with a bool rather than raising, as before."""
    assert wait_for_scanner(timeout_s=0.0) is False


def test_the_enumeration_only_wait_still_matches_a_serial(
    builder: SysfsBuilder, enumeration: None
) -> None:
    """``None`` is any scanner; a serial is that scanner. Unchanged."""
    add_scanner(builder, serial=OTHER_SERIAL)

    assert wait_for_scanner(OTHER_SERIAL, timeout_s=0.0) is True
    assert wait_for_scanner(REAL_SERIAL, timeout_s=0.0) is False
    assert wait_for_scanner(timeout_s=0.0) is True
