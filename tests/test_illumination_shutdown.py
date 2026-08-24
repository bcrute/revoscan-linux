"""Offline proof for the single immutable blue-light off write.

Two things are proved here. The older one is mechanical: exactly one fixed
transfer, inside the interface-0 ownership lifecycle, with cleanup that runs on
every path. The newer one is about language. This finalizer once reported
``status=complete``, and the command it sends is now known not to darken this
unit's emitter, so several tests below exist purely to keep any report of it
from reading as a claim that light stopped -- including the tests that render
the *successful* case and search its output for the words it must not contain.
"""

from __future__ import annotations

import dataclasses
import errno
import signal
import sys
from typing import Any

import pytest

import scanner3d.cli as cli
import scanner3d.illumination_shutdown as shutdown_module
import scanner3d.ownership as ownership_module
from scanner3d.errors import IlluminationShutdownNotAllowed
from scanner3d.illumination_shutdown import (
    AUTHORIZED_SERIAL,
    COMMAND_PAYLOAD,
    COMMAND_PAYLOAD_SHA256,
    EMITTER_LIMITATION,
    EXPECTED_SETUP,
    OFF_TRANSFER,
    RESULT_LENGTH,
    IlluminationShutdownOutcome,
    ShutdownStatus,
    ShutdownTransferRecord,
    UsbfsIlluminationShutdownBackend,
    ensure_shutdown_allowed,
    not_required,
)
from scanner3d.ownership import Phase, SignalReceived
from scanner3d.rgb_frame import RGB_PROFILE, V4l2RgbFrameBackend
from scanner3d.v4l2_capture import (
    EXPECTED_FRAME_BYTES,
    VIDIOC_DQBUF,
    VIDIOC_REQBUFS,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    CaptureStatus,
    V4l2DepthFrameBackend,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.fake_v4l2 import FakeKernel
from tests.test_depth_scale import FakeUsbfs
from tests.test_jpeg_inspect import baseline_jpeg


def make_backend(builder: SysfsBuilder, fake: FakeUsbfs) -> UsbfsIlluminationShutdownBackend:
    return UsbfsIlluminationShutdownBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=builder.sysfs,
        dev_root=builder.dev,
    )


def test_fixed_transfer_is_exact_and_the_only_whitelist_member() -> None:
    assert OFF_TRANSFER.request.wire_tuple == (0x21, 0x01, 0x0700, 0x0400, 60, 1000)
    assert bytes.fromhex("2101000700043c00e8030000") == EXPECTED_SETUP
    assert (
        b"echo s 0x922 0 >/dev/rk_preisp\x00"
        + b"\x00" * (RESULT_LENGTH - len("echo s 0x922 0 >/dev/rk_preisp\x00"))
        == COMMAND_PAYLOAD
    )
    assert len(COMMAND_PAYLOAD) == RESULT_LENGTH
    assert COMMAND_PAYLOAD_SHA256 == (
        "6030364daddcc4c9855d16da552c58161bbec5cf77ea23ace9de2f14839ffb0d"
    )
    assert shutdown_module.WHITELIST == (OFF_TRANSFER,)
    assert shutdown_module.MAX_CONTROL_TRANSFERS == 1


def test_not_required_is_inert() -> None:
    outcome = not_required()
    assert not outcome.required
    assert not outcome.attempted
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.guidance is None
    # Even the never-streamed case carries the limitation: the emitter can be
    # lit from an earlier run, so no report of this module may read as "dark".
    assert outcome.as_dict()["limitation"] == EMITTER_LIMITATION
    assert outcome.as_dict()["emitter_state"] == "unknown"


def test_no_status_or_report_claims_the_emitter_went_dark(healthy: SysfsBuilder) -> None:
    """The whole point of this module's vocabulary: nothing here says "off"."""
    assert "complete" not in {member.value for member in ShutdownStatus}
    assert ShutdownStatus.COMMAND_ACCEPTED.value == "command_accepted"
    assert not hasattr(ShutdownStatus, "COMPLETE")

    fake = FakeUsbfs(control_returns={0: RESULT_LENGTH})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    payload = outcome.as_dict()

    assert payload["status"] == "command_accepted"
    assert payload["command_accepted"] is True
    assert payload["device_restored"] is True
    assert payload["emitter_state"] == "unknown"
    assert payload["limitation"] == EMITTER_LIMITATION
    assert "KNOWN NOT TO DARKEN" in EMITTER_LIMITATION
    assert "no optical readback" in EMITTER_LIMITATION
    # The best possible run still tells the reader not to conclude darkness.
    assert outcome.succeeded
    assert outcome.guidance is not None
    assert "DO NOT CLAIM THE BLUE LIGHT IS OFF" in outcome.guidance
    assert EMITTER_LIMITATION in outcome.guidance
    assert "emitter state not observed" in str(payload["detail"])


@pytest.mark.parametrize(
    "outcome",
    [
        IlluminationShutdownOutcome(
            False, True, ShutdownStatus.IOCTL_ERROR, "contradictory attempted state"
        ),
        IlluminationShutdownOutcome(
            False, False, ShutdownStatus.COMMAND_ACCEPTED, "contradictory accepted state"
        ),
        IlluminationShutdownOutcome(
            True,
            True,
            ShutdownStatus.COMMAND_ACCEPTED,
            "failed transfer presented as accepted",
            transfer=ShutdownTransferRecord(1, False, "short", 0.0),
        ),
        IlluminationShutdownOutcome(
            True,
            True,
            ShutdownStatus.IOCTL_ERROR,
            "discharged status with no transfer record to witness it",
        ),
    ],
)
def test_contradictory_outcomes_cannot_report_success(
    outcome: IlluminationShutdownOutcome,
) -> None:
    assert not outcome.accepted
    assert not outcome.succeeded
    assert outcome.exit_code != 0
    assert outcome.as_dict()["succeeded"] is False
    assert outcome.as_dict()["recovery_guidance"] is not None


def test_transfer_drift_is_rejected() -> None:
    with pytest.raises(IlluminationShutdownNotAllowed):
        ensure_shutdown_allowed(dataclasses.replace(OFF_TRANSFER, payload=b"x" * 60))
    with pytest.raises(IlluminationShutdownNotAllowed):
        ensure_shutdown_allowed(
            dataclasses.replace(
                OFF_TRANSFER,
                request=dataclasses.replace(OFF_TRANSFER.request, value=0x0800),
            )
        )


def test_success_is_one_exact_write_and_restores_interface(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_returns={0: RESULT_LENGTH})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is ShutdownStatus.COMMAND_ACCEPTED
    assert outcome.succeeded
    assert len(fake.controls) == 1
    assert fake.setups == [EXPECTED_SETUP]
    assert fake.written == [COMMAND_PAYLOAD]
    assert fake.driver_codes == [
        ownership_module.USBDEVFS_DISCONNECT,
        ownership_module.USBDEVFS_CONNECT,
    ]
    assert [phase.phase for phase in outcome.phases] == [
        Phase.CONFIRM_DRIVER,
        Phase.DISCONNECT,
        Phase.CLAIM,
        Phase.OWNED_SEQUENCE,
        Phase.RELEASE,
        Phase.RECONNECT,
    ]
    assert outcome.transfer is not None
    assert outcome.transfer.transferred == RESULT_LENGTH
    assert outcome.as_dict()["transfer"]["payload_sha256"] == COMMAND_PAYLOAD_SHA256  # type: ignore[index]


@pytest.mark.parametrize("returned", [RESULT_LENGTH - 1, RESULT_LENGTH + 1, -1])
def test_non_exact_or_negative_write_is_typed_and_never_retried(
    healthy: SysfsBuilder, returned: int
) -> None:
    """The verdict is recorded in full and scored not at all.

    The command is known not to darken this unit, so the device refusing it
    changes nothing about the emitter and must not fail the run around it. The
    typed status, the transfer record and the standing warning all survive.
    """
    fake = FakeUsbfs(control_returns={0: returned})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert len(fake.controls) == 1
    assert outcome.transfer is not None
    assert outcome.transfer.transferred == returned
    assert Phase.OWNED_SEQUENCE in [phase.phase for phase in outcome.phases]
    assert not outcome.accepted
    assert outcome.as_dict()["command_accepted"] is False
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.device_restored
    assert outcome.guidance is not None
    assert outcome.as_dict()["limitation"] == EMITTER_LIMITATION
    assert outcome.status is (
        ShutdownStatus.SHORT_WRITE
        if returned == RESULT_LENGTH - 1
        else ShutdownStatus.OVERLONG_WRITE
        if returned == RESULT_LENGTH + 1
        else ShutdownStatus.IOCTL_ERROR
    )


def test_control_error_is_typed_and_cleanup_still_runs(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_errors={0: OSError(errno.EIO, "control failed")})
    outcome = make_backend(healthy, fake).run(AUTHORIZED_SERIAL)

    assert outcome.status is ShutdownStatus.IOCTL_ERROR
    assert len(fake.controls) == 1
    assert [phase.phase for phase in outcome.cleanup.phases] == [Phase.RELEASE, Phase.RECONNECT]
    assert outcome.errno == errno.EIO
    # The attempt was made, evidenced and the interface handed back, so the
    # ineffective command's own failure does not fail the run around it.
    assert outcome.transfer is not None and not outcome.transfer.ok
    assert outcome.succeeded
    assert outcome.exit_code == 0


@pytest.mark.parametrize("interruption", [SignalReceived(signal.SIGTERM), KeyboardInterrupt()])
def test_signal_or_keyboard_interrupt_is_typed_and_restored(
    healthy: SysfsBuilder, interruption: BaseException
) -> None:
    def interrupt(_index: int) -> None:
        raise interruption

    fake = FakeUsbfs(on_control=interrupt)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is ShutdownStatus.INTERRUPTED
    assert not outcome.succeeded
    assert outcome.transfer is not None
    assert outcome.transfer.exception_type in {"SignalReceived", "KeyboardInterrupt"}
    assert [phase.phase for phase in outcome.cleanup.phases] == [Phase.RELEASE, Phase.RECONNECT]


class _BaseExceptionForTest(BaseException):
    pass


class _CloseFailureFake(FakeUsbfs):
    def close(self, fd: int) -> None:
        self.closed.append(fd)
        raise OSError(errno.EIO, "close failed")


def test_unexpected_base_exception_is_typed_without_skipping_cleanup(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(on_control=lambda _index: (_ for _ in ()).throw(_BaseExceptionForTest("boom")))
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is ShutdownStatus.UNEXPECTED_EXCEPTION
    assert outcome.transfer is not None
    assert outcome.transfer.exception_type == "_BaseExceptionForTest"
    assert [phase.phase for phase in outcome.cleanup.phases] == [Phase.RELEASE, Phase.RECONNECT]


def test_unexpected_exception_is_typed_with_phase_and_json_detail(healthy: SysfsBuilder) -> None:
    class UnexpectedException(Exception):
        pass

    fake = FakeUsbfs(on_control=lambda _index: (_ for _ in ()).throw(UnexpectedException("boom")))
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is ShutdownStatus.UNEXPECTED_EXCEPTION
    assert outcome.transfer is not None
    assert outcome.transfer.exception_type == "UnexpectedException"
    owned = next(phase for phase in outcome.phases if phase.phase is Phase.OWNED_SEQUENCE)
    assert "UnexpectedException" in owned.detail
    payload = outcome.as_dict()
    assert payload["status"] == "unexpected_exception"
    assert payload["recovery_guidance"] is not None
    assert len(fake.controls) == 1


def test_reconnect_failure_preserves_accepted_write_and_typed_cleanup(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(errors={ownership_module.USBDEVFS_CONNECT: OSError(errno.EIO, "reconnect")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is ShutdownStatus.COMMAND_ACCEPTED
    assert outcome.transfer is not None and outcome.transfer.transferred == RESULT_LENGTH
    assert len(fake.controls) == 1
    assert not outcome.succeeded
    assert outcome.cleanup.ok is False
    assert outcome.cleanup.phases[-1].phase is Phase.RECONNECT
    assert outcome.guidance is not None


def test_only_fixed_off_transfer_can_be_expressed() -> None:
    request = OFF_TRANSFER.request
    forbidden = (
        dataclasses.replace(request, request_type=0xA1),  # GET direction
        dataclasses.replace(request, value=0x0701),  # laser on
        dataclasses.replace(request, value=0x0B00),  # LED/brightness-shaped value
        dataclasses.replace(request, value=0x0800),  # arbitrary property
        dataclasses.replace(request, index=0x0401),  # arbitrary interface
    )
    for candidate in forbidden:
        with pytest.raises(IlluminationShutdownNotAllowed):
            ensure_shutdown_allowed(dataclasses.replace(OFF_TRANSFER, request=candidate))
    with pytest.raises(IlluminationShutdownNotAllowed):
        ensure_shutdown_allowed(
            dataclasses.replace(
                OFF_TRANSFER, payload=b"set arbitrary register 0xB00".ljust(60, b"\0")
            )
        )


def test_json_uses_write_status_vocabulary(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_returns={0: RESULT_LENGTH - 1})
    payload = make_backend(healthy, fake).run(REAL_SERIAL).as_dict()
    assert payload["status"] == "short_write"
    assert "short_response" not in str(payload)


def test_close_failure_preserves_write_and_driver_evidence(healthy: SysfsBuilder) -> None:
    fake = _CloseFailureFake()
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is ShutdownStatus.COMMAND_ACCEPTED
    assert outcome.transfer is not None and outcome.transfer.transferred == RESULT_LENGTH
    assert not outcome.succeeded
    assert outcome.cleanup.ok is False
    assert outcome.cleanup.phases[-1].phase is Phase.CLOSE
    assert [phase.phase for phase in outcome.phases][-2:] == [Phase.RELEASE, Phase.RECONNECT]
    assert len(fake.controls) == 1


def test_wrong_serial_does_not_send_a_write(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    wrong = make_backend(healthy, fake).run("not-authorized")
    assert wrong.status is ShutdownStatus.REFUSED
    assert not fake.controls


def test_permission_open_failure_is_typed_without_control(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(open_error=PermissionError(errno.EACCES, "denied"))
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is ShutdownStatus.PERMISSION_DENIED
    assert outcome.attempted is True
    assert outcome.transfer is None
    assert fake.controls == []
    assert fake.closed == []


def test_already_unbound_driver_is_refused_before_open(healthy: SysfsBuilder) -> None:
    interface = next(healthy.devices.rglob("*:1.0"))
    driver_link = interface / "driver"
    driver_link.unlink()
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is ShutdownStatus.REFUSED
    assert outcome.attempted is False
    assert fake.opened == []
    assert fake.controls == []


@pytest.mark.parametrize(
    ("command", "status"),
    [
        (ownership_module.USBDEVFS_DISCONNECT, ShutdownStatus.DETACH_FAILED),
        (ownership_module.USBDEVFS_CLAIMINTERFACE, ShutdownStatus.CLAIM_FAILED),
    ],
)
def test_detach_and_claim_failures_are_typed_and_never_write(
    healthy: SysfsBuilder, command: int, status: ShutdownStatus
) -> None:
    fake = FakeUsbfs(errors={command: OSError(errno.EIO, "synthetic failure")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is status
    assert outcome.transfer is None
    assert fake.controls == []
    assert len(fake.driver_codes) <= 2


class _ChangedIdentityFake(FakeUsbfs):
    def __init__(self, serial_path, **kwargs):
        super().__init__(**kwargs)
        self.serial_path = serial_path

    def open(self, path: str) -> int:
        fd = super().open(path)
        self.serial_path.write_text("replaced\n", encoding="utf-8")
        return fd


def test_post_open_identity_mutation_is_refused_without_transfer(healthy: SysfsBuilder) -> None:
    fake = _ChangedIdentityFake(healthy.devices / "1-4.1" / "serial")
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is ShutdownStatus.DEVICE_REPLACED
    assert outcome.transfer is None
    assert fake.controls == []
    assert fake.closed == [7]


def test_absent_device_does_not_send_a_write(empty_sysfs: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    absent = make_backend(empty_sysfs, fake).run(AUTHORIZED_SERIAL)
    assert absent.status is ShutdownStatus.DEVICE_ABSENT
    assert not fake.controls


def test_ambiguous_matching_serial_is_refused_before_open(healthy: SysfsBuilder) -> None:
    healthy.add_healthy_mini(usb_path="1-4.2", devpath="4.2")
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run(AUTHORIZED_SERIAL)
    assert outcome.status is ShutdownStatus.DEVICE_ABSENT
    assert outcome.attempted is False
    assert fake.opened == []
    assert fake.controls == []


def test_cleanup_failure_blocks_success(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(
        errors={ownership_module.USBDEVFS_RELEASEINTERFACE: OSError(errno.EIO, "release")}
    )
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is ShutdownStatus.COMMAND_ACCEPTED
    assert not outcome.succeeded
    assert outcome.exit_code != 0
    assert outcome.guidance is not None


def _journal_backend(
    kernel: FakeKernel,
    sysfs: SysfsBuilder,
    journal: list[str],
    *,
    rgb: bool = False,
    shutdown_result: IlluminationShutdownOutcome | None = None,
    shutdown_error: BaseException | None = None,
):
    def ioctl(fd, request, packed, mutate):
        if request == VIDIOC_STREAMON:
            journal.append("streamon")
        elif request == VIDIOC_STREAMOFF:
            journal.append("streamoff")
        elif request == VIDIOC_REQBUFS and int.from_bytes(bytes(packed[:4]), "little") == 0:
            journal.append("reqbufs0")
        return kernel.ioctl(fd, request, packed, mutate)

    class JournalMapping:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, index: slice) -> bytes:
            return self.inner[index]

        def close(self) -> None:
            journal.append("unmap")
            self.inner.close()

    def mmap(fd: int, length: int, offset: int) -> JournalMapping:
        return JournalMapping(kernel.mmap(fd, length, offset))

    def closer(fd: int) -> None:
        journal.append("close")
        kernel.closer(fd)

    class JournalShutdown:
        def run(self, _serial: str) -> IlluminationShutdownOutcome:
            journal.append("usbfs-off")
            if shutdown_error is not None:
                raise shutdown_error
            if shutdown_result is not None:
                return shutdown_result
            return IlluminationShutdownOutcome(
                True,
                True,
                ShutdownStatus.COMMAND_ACCEPTED,
                "synthetic fixed off accepted",
                transfer=ShutdownTransferRecord(60, True, "synthetic", 0.0),
            )

    def factory(_sysfs, _dev):
        journal.append("illumination-factory")
        return JournalShutdown()

    cls = V4l2RgbFrameBackend if rgb else V4l2DepthFrameBackend
    return cls(
        opener=kernel.opener,
        ioctl_fn=ioctl,
        mmap_fn=mmap,
        poll_fn=kernel.poll,
        closer=closer,
        sysfs_root=sysfs.sysfs,
        dev_root=sysfs.dev,
        illumination_backend_factory=factory,
    )


def test_single_depth_journal_places_one_off_after_v4l2_teardown(
    healthy: SysfsBuilder, tmp_path
) -> None:
    journal: list[str] = []
    kernel = FakeKernel()
    outcome = _journal_backend(kernel, healthy, journal).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )

    assert outcome.status is CaptureStatus.COMPLETE
    assert journal.count("usbfs-off") == 1
    assert journal.index("streamon") < journal.index("streamoff")
    streamoff = journal.index("streamoff")
    reqbufs0 = journal.index("reqbufs0")
    close = journal.index("close")
    off = journal.index("usbfs-off")
    unmaps = [index for index, event in enumerate(journal) if event == "unmap"]
    assert streamoff < unmaps[0] <= unmaps[-1] < reqbufs0 < close < off


def test_single_timeout_still_runs_one_off_and_rgb_never_does(
    healthy: SysfsBuilder, tmp_path
) -> None:
    journal: list[str] = []
    timeout = _journal_backend(FakeKernel(poll_revents=0), healthy, journal).run(
        REAL_SERIAL, output_dir=tmp_path / "timeout"
    )
    assert timeout.status is CaptureStatus.POLL_TIMEOUT
    assert journal.count("usbfs-off") == 1

    rgb_journal: list[str] = []
    rgb = _journal_backend(
        FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800)),
        healthy,
        rgb_journal,
        rgb=True,
    ).run(REAL_SERIAL, output_dir=tmp_path / "rgb")
    assert rgb.status is CaptureStatus.COMPLETE
    assert "illumination-factory" not in rgb_journal
    assert "usbfs-off" not in rgb_journal


def test_streamon_signal_and_v4l2_cleanup_failure_still_run_one_off(
    healthy: SysfsBuilder, tmp_path
) -> None:
    signal_journal: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        _journal_backend(
            FakeKernel(raises={VIDIOC_STREAMON: KeyboardInterrupt()}), healthy, signal_journal
        ).run(REAL_SERIAL, output_dir=tmp_path / "signal")
    assert signal_journal.count("usbfs-off") == 1
    assert signal_journal.index("streamon") < signal_journal.index("usbfs-off")

    cleanup_journal: list[str] = []
    cleanup = _journal_backend(
        FakeKernel(errors={VIDIOC_STREAMOFF: 5}), healthy, cleanup_journal
    ).run(REAL_SERIAL, output_dir=tmp_path / "cleanup")
    assert cleanup.cleanup.ok is False
    assert cleanup_journal.count("usbfs-off") == 1


@pytest.mark.parametrize(
    "kernel",
    [
        FakeKernel(bytesused=EXPECTED_FRAME_BYTES - 2),
        FakeKernel(raises={VIDIOC_STREAMON: RuntimeError("streamon")}),
        FakeKernel(raises={VIDIOC_DQBUF: RuntimeError("later exception")}),
        FakeKernel(raises={VIDIOC_DQBUF: KeyboardInterrupt()}),
    ],
)
def test_single_depth_journal_covers_malformed_and_late_failures(
    healthy: SysfsBuilder, tmp_path, kernel: FakeKernel
) -> None:
    journal: list[str] = []
    backend = _journal_backend(kernel, healthy, journal)
    try:
        outcome = backend.run(REAL_SERIAL, output_dir=tmp_path / "branch")
    except KeyboardInterrupt:
        outcome = None
    assert journal.count("usbfs-off") == 1
    assert journal.index("streamon") < journal.index("streamoff")
    assert journal.index("streamoff") < journal.index("unmap")
    assert journal.index("unmap") < journal.index("reqbufs0")
    assert journal.index("reqbufs0") < journal.index("close")
    assert journal.index("close") < journal.index("usbfs-off")
    if outcome is not None:
        assert outcome.succeeded is False


def test_pre_streamon_refusal_and_dry_paths_issue_no_off(healthy: SysfsBuilder, tmp_path) -> None:
    refused_journal: list[str] = []
    refused = _journal_backend(FakeKernel(capabilities=0), healthy, refused_journal).run(
        REAL_SERIAL, output_dir=tmp_path / "refused"
    )
    assert "illumination-factory" not in refused_journal
    assert "usbfs-off" not in refused_journal
    assert refused.illumination_shutdown.succeeded

    dry_journal: list[str] = []
    dry = _journal_backend(FakeKernel(), healthy, dry_journal).preflight(REAL_SERIAL)
    assert "illumination-factory" not in dry_journal
    assert "usbfs-off" not in dry_journal
    assert dry.illumination_shutdown.succeeded


@pytest.mark.parametrize(
    ("transferred", "status"),
    [
        (RESULT_LENGTH - 1, ShutdownStatus.SHORT_WRITE),
        (None, ShutdownStatus.IOCTL_ERROR),
    ],
)
def test_a_rejected_off_command_does_not_fail_a_good_capture(
    healthy: SysfsBuilder, tmp_path, transferred: int | None, status: ShutdownStatus
) -> None:
    """The capture is scored on the capture, not on an inert command's verdict.

    The write is still owed, still issued, and its rejection is still reported
    in full; what changed is that the frame, its artifacts and the device's
    release decide the exit code. Compare
    ``test_failed_or_raising_finalizer_blocks_depth_success_and_preserves_evidence``,
    where nothing witnesses that the write ever left the host.
    """
    rejected = IlluminationShutdownOutcome(
        True,
        True,
        status,
        "synthetic rejected off command",
        transfer=ShutdownTransferRecord(transferred, False, "rejected", 0.0),
    )
    journal: list[str] = []
    outcome = _journal_backend(FakeKernel(), healthy, journal, shutdown_result=rejected).run(
        REAL_SERIAL, output_dir=tmp_path / "rejected"
    )

    assert journal.count("usbfs-off") == 1
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.frame_bytes == EXPECTED_FRAME_BYTES
    illumination = outcome.as_dict()["illumination_shutdown"]
    assert isinstance(illumination, dict)
    assert illumination["status"] == str(status)
    assert illumination["command_accepted"] is False
    assert illumination["emitter_state"] == "unknown"
    assert illumination["limitation"] == EMITTER_LIMITATION
    assert illumination["recovery_guidance"] is not None


def test_failed_or_raising_finalizer_blocks_depth_success_and_preserves_evidence(
    healthy: SysfsBuilder, tmp_path
) -> None:
    failed = IlluminationShutdownOutcome(
        True, True, ShutdownStatus.IOCTL_ERROR, "synthetic off failure"
    )
    failed_journal: list[str] = []
    failed_outcome = _journal_backend(
        FakeKernel(), healthy, failed_journal, shutdown_result=failed
    ).run(REAL_SERIAL, output_dir=tmp_path / "failed")
    assert failed_outcome.status is CaptureStatus.COMPLETE
    assert not failed_outcome.succeeded
    assert failed_outcome.frame_bytes == EXPECTED_FRAME_BYTES
    assert failed_outcome.artifacts.ok
    assert failed_outcome.illumination_shutdown.status is ShutdownStatus.IOCTL_ERROR
    assert failed_journal.count("usbfs-off") == 1

    raised_journal: list[str] = []
    raised_outcome = _journal_backend(
        FakeKernel(), healthy, raised_journal, shutdown_error=RuntimeError("finalizer")
    ).run(REAL_SERIAL, output_dir=tmp_path / "raised")
    assert raised_outcome.status is CaptureStatus.COMPLETE
    assert not raised_outcome.succeeded
    assert raised_outcome.illumination_shutdown.status is ShutdownStatus.UNEXPECTED_EXCEPTION
    assert raised_journal.count("usbfs-off") == 1


@pytest.mark.parametrize("render", [cli._render_illumination, None])
def test_no_rendering_of_the_best_case_can_be_read_as_darkness(capsys, render) -> None:
    """The success rendering is the one that has to be checked hardest.

    A failed finalizer already shouts. The dangerous line is the one printed
    after a clean run, which used to read ``status=complete`` under a heading
    that said "Blue-light shutdown".
    """
    best = IlluminationShutdownOutcome(
        True,
        True,
        ShutdownStatus.COMMAND_ACCEPTED,
        "the device accepted all 60 bytes of the fixed off command; emitter state not observed",
        transfer=ShutdownTransferRecord(RESULT_LENGTH, True, "accepted", 0.0),
    )
    assert best.succeeded
    if render is None:
        cli._print_illumination_summary(best)
    else:
        render(best, sys.stdout)
    cli._warn_illumination(best)
    captured = capsys.readouterr()

    for text in (captured.out, captured.err):
        assert "shutdown" not in text.lower()
        assert "complete" not in text.lower()
    assert "NOT an optical result" in captured.out
    assert "command_accepted=True" in captured.out
    assert "emitter state UNKNOWN" in captured.out
    assert "does not darken this unit's emitter" in captured.out
    assert "DO NOT CLAIM THE BLUE LIGHT IS OFF" in captured.err
    assert EMITTER_LIMITATION in captured.err
