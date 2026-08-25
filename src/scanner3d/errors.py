"""Status vocabulary and exceptions shared by the diagnostic layers."""

from __future__ import annotations

import enum
from collections.abc import Iterable


class DoctorStatus(enum.StrEnum):
    """Outcome classes the ``doctor`` command must be able to distinguish.

    The ordering here is the severity ordering used by :func:`worst`.
    """

    OK = "ok"
    NO_FRAMES = "no_frames"
    UNSUPPORTED_MODE = "unsupported_mode"
    PERMISSION_DENIED = "permission_denied"
    UNEXPECTED_IDENTITY = "unexpected_identity"
    DEVICE_ABSENT = "device_absent"
    INTERNAL_ERROR = "internal_error"


_SEVERITY: dict[DoctorStatus, int] = {
    DoctorStatus.OK: 0,
    DoctorStatus.NO_FRAMES: 1,
    DoctorStatus.UNSUPPORTED_MODE: 2,
    DoctorStatus.PERMISSION_DENIED: 3,
    DoctorStatus.UNEXPECTED_IDENTITY: 4,
    DoctorStatus.DEVICE_ABSENT: 5,
    DoctorStatus.INTERNAL_ERROR: 6,
}

#: Process exit code for each status, so callers can branch without parsing JSON.
EXIT_CODES: dict[DoctorStatus, int] = {
    DoctorStatus.OK: 0,
    DoctorStatus.NO_FRAMES: 4,
    DoctorStatus.UNSUPPORTED_MODE: 5,
    DoctorStatus.PERMISSION_DENIED: 3,
    DoctorStatus.UNEXPECTED_IDENTITY: 6,
    DoctorStatus.DEVICE_ABSENT: 2,
    DoctorStatus.INTERNAL_ERROR: 7,
}


def worst(statuses: Iterable[DoctorStatus]) -> DoctorStatus:
    """Return the most severe status in ``statuses`` (``OK`` when empty)."""
    return max(statuses, key=lambda s: _SEVERITY[s], default=DoctorStatus.OK)


class ScannerError(Exception):
    """Base class for every error this package raises deliberately."""

    status: DoctorStatus = DoctorStatus.INTERNAL_ERROR


class DeviceAbsentError(ScannerError):
    """No device matching the expected USB identity is attached."""

    status = DoctorStatus.DEVICE_ABSENT


class ReferenceProbeError(ScannerError):
    """The external vendor-SDK reference probe could not be used."""


class ReferenceProbeTimeout(ReferenceProbeError):
    """The external reference probe exceeded its hard wall-clock budget."""

    status = DoctorStatus.NO_FRAMES


class TraceFormatError(ScannerError):
    """A file offered as a usbfs trace is not one this decoder can read.

    Reserved for defects in the *stream* - no header, wrong format marker, an
    unsupported version. Damage inside an otherwise valid trace (a corrupt or
    truncated record) is reported as evidence instead, never raised, so a
    partial capture stays readable.
    """


class ControlRequestNotAllowed(ScannerError):
    """A control transfer was requested that is not on the owned whitelist.

    Raised before any device node is opened, so a rejected request is never
    packed into an ioctl, let alone put on the wire.
    """


class InitSequenceNotAllowed(ScannerError):
    """A transfer was requested that is not in the owned four-transfer sequence.

    Raised by :mod:`scanner3d.init_sequence` before an ioctl argument is
    packed, so a setup packet or a payload other than the four the vendor SDK
    was observed to send can never be encoded, let alone issued.
    """


class InterfaceOwnershipNotAllowed(ScannerError):
    """A usbfs interface operation was requested that is out of owned scope.

    Raised by :mod:`scanner3d.ownership` before an ioctl argument is packed, so
    an interface other than 0 - or a driver command other than disconnect and
    connect - can never be encoded, let alone issued.
    """


class IntrinsicsReadNotAllowed(ScannerError):
    """A transfer was requested that is not in the closed intrinsics whitelist.

    Raised by :mod:`scanner3d.intrinsics` before an ioctl argument is packed,
    so a key other than ``/data/camparam/Pl.bin`` - or a setup packet, payload
    or length other than the three the recovered ``libuvcReadFile`` protocol
    uses - can never be encoded, let alone issued.
    """


class DepthScaleReadNotAllowed(ScannerError):
    """A transfer was requested that is not in the closed depth-scale pair.

    Raised by :mod:`scanner3d.depth_scale` before an ioctl argument is packed,
    so a property id other than ``0x918`` - or a setup packet, payload or
    length other than the two the recovered ``libuvcReadValueExtension`` pair
    uses - can never be encoded, let alone issued.
    """


class RgbExposureExtensionNotAllowed(ScannerError):
    """An RGB exposure-extension transfer escaped its fixed one-read whitelist."""


class RgbExposureLengthNotAllowed(ScannerError):
    """An RGB exposure-length transfer escaped its fixed one-read whitelist."""


class RgbExposureInfoNotAllowed(ScannerError):
    """An RGB exposure-info transfer escaped its fixed one-read whitelist."""


class IlluminationShutdownNotAllowed(ScannerError):
    """A blue-light shutdown transfer escaped its fixed off-only whitelist."""


class V4l2CaptureNotAllowed(ScannerError):
    """A V4L2 capture was requested outside the one fixed mode and its bounds.

    Raised by :mod:`scanner3d.v4l2_capture` before an ioctl argument is packed,
    so a format, a size, a buffer index or a struct layout other than the ones
    this repository transcribed from the kernel UAPI can never be encoded, let
    alone issued against a real descriptor.
    """


class V4l2ArtifactIncomplete(OSError):
    """An evidence file could not be written in full, so none was kept.

    An ``OSError`` rather than a :class:`ScannerError`: it is raised on the
    same path as a real write failure, is reported through the same artifact
    record, and every caller that already treats a failed write as a failed
    artifact keeps working unchanged. Carrying an ``errno`` keeps a short
    write, a zero-byte write and a kernel error describable in one field.
    """
