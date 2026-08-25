"""RGB-depth projection: colour a depth point cloud using the factory extrinsics.

Why this exists
-----------------
Every RGB-colouring idea for this scanner was blocked on one missing number:
where the colour camera sits relative to the depth camera. This repository
had depth intrinsics (read live, over the vendor's file-read protocol - see
:mod:`scanner3d.intrinsics`) but nothing that located the second camera at
all. That is now measured: the full factory calibration set was read off the
device once, over the same protocol, and is committed as fixtures at
``tests/fixtures/camparam/`` - ``Pl.bin`` (depth intrinsics), ``Prgb.bin``
(RGB intrinsics), ``LC_RT.bin`` (the depth-to-RGB extrinsics: a rotation and
a translation) and ``Distort.bin`` (RGB lens distortion, OpenCV's
``k1, k2, p1, p2, k3`` ordering). This module is what turns those four blobs,
plus a depth point and a decoded RGB frame, into a colour per point.

``camparam.yaml`` is committed alongside them and is NOT a fifth input this
module reads: its own ``deviceID`` field names a different unit
(``EXAMPLESERIAL0002``) from the one every other fixture here was read from
(a different unit's) - a factory leftover on the device's filesystem, kept
as a fixture only so nobody re-reads it hoping it applies to this unit.

What this module does NOT do
------------------------------
It does not read a device. Every function here takes bytes or already-parsed
values; the USB protocol that produced ``tests/fixtures/camparam/*.bin`` in
the first place lives in :mod:`scanner3d.intrinsics` and its unbuilt
siblings, not here.

It does not decode JPEG, or any other image format. :func:`color_points`
takes an already-decoded image - a width, a height, and a packed RGB8 byte
buffer - so this module stays about geometry: camera models, rigid
transforms, and where a 3D point lands on a 2D sensor. A caller owns turning
whatever the camera actually produced into that buffer.

It does not rescale intrinsics to a different image resolution. ``fx, fy,
cx, cy`` in :data:`Prgb.bin <parse_camera_intrinsics>` are calibrated for a
specific frame size (640x400, measured - see :data:`CameraIntrinsics`), and
:func:`color_points` requires the decoded image it is given to match that
size exactly, refusing otherwise. Scaling a calibration to a different
resolution is a real, well-defined operation, but it is a DIFFERENT one from
projection, and silently guessing at it here would be exactly the kind of
plausible-looking wrong answer this module exists to avoid elsewhere.

It does not do sub-pixel (bilinear) colour sampling. A projected point picks
the single nearest pixel by truncating its projected coordinate; a caller
that needs smoother colour needs to do that resampling itself.

It does not correct rolling-shutter, exposure-time, or capture-time skew
between the depth and RGB frames. The extrinsics this module applies are a
static rigid offset between two camera mounts, not a per-frame correction for
whatever the two sensors were doing at slightly different instants.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid, ensure_finite

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1


class RgbProjectNotAllowed(ScannerError):
    """An input to RGB-depth projection was refused before it could produce a
    silently wrong answer downstream.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class CalibrationBlobMalformed(RgbProjectNotAllowed):
    """A calibration blob is not the vendor's fixed record - wrong length, a
    non-finite float, or a structurally-fixed matrix entry that is not what
    the layout requires.

    Raised before the blob's values are trusted for anything: a camera
    matrix with the wrong shape is not "probably close enough", because
    every downstream number - focal length, principal point, distortion
    strength - is read positionally out of it.
    """


class ImageBufferMalformed(RgbProjectNotAllowed):
    """A decoded RGB image does not match its own declared size, or does not
    match the resolution the RGB calibration was measured at.

    The second case is the one worth naming: this module refuses to guess at
    rescaling ``fx, fy, cx, cy`` to a different frame size (see the module
    docstring), so a caller who decodes a frame at any resolution other than
    the one :data:`Prgb.bin <parse_camera_intrinsics>` was calibrated for
    gets a refusal here, not pixels sampled from the wrong place in a
    same-sized-by-coincidence buffer.
    """


# --- calibration blob layouts -------------------------------------------------------

#: ``Pl.bin``/``Prgb.bin``: two little-endian ``uint16`` (width, height), then
#: a row-major 3x3 pinhole camera matrix as nine little-endian float32. 40
#: bytes total. The same shape as the vendor's ``Types.hpp`` ``Intrinsics``
#: struct :data:`scanner3d.intrinsics.INTRINSICS_FORMAT` decodes live off the
#: device; re-declared here rather than imported because that module's own
#: parser returns ``None`` on a length mismatch (meant for a streaming USB
#: read that can retry) and folds in USB-descriptor plausibility checks that
#: have no meaning for a blob already committed as a fixture - a different
#: contract for a different situation, not a second copy of the same one.
_CAMERA_MATRIX_FORMAT: Final = "<HH9f"
_CAMERA_MATRIX_LENGTH: Final = struct.calcsize(_CAMERA_MATRIX_FORMAT)

#: The nine camera-matrix floats, in the order the struct unpacks them.
_CAMERA_MATRIX_FIELDS: Final[tuple[str, ...]] = (
    "fx",
    "skew",
    "cx",
    "zero_row1_col0",
    "fy",
    "cy",
    "zero_row2_col0",
    "zero_row2_col1",
    "one",
)

#: Which of those nine are structurally fixed by the pinhole layout, and what
#: they are fixed to. ``fx``, ``cx``, ``fy``, ``cy`` are the free entries.
_CAMERA_MATRIX_FIXED: Final[tuple[tuple[str, float], ...]] = (
    ("skew", 0.0),
    ("zero_row1_col0", 0.0),
    ("zero_row2_col0", 0.0),
    ("zero_row2_col1", 0.0),
    ("one", 1.0),
)

#: ``LC_RT.bin``: a row-major 3x3 rotation as nine little-endian float32,
#: then a translation as three more. 48 bytes total. "LC" is the vendor's own
#: naming for this pair - Left (the depth/IR camera) to Colour - and the
#: measured transform (10.4 degrees of rotation, a 15.8-unit baseline) is a
#: physically plausible offset between two camera mounts a few centimetres
#: apart, not a near-identity placeholder.
_EXTRINSICS_FORMAT: Final = "<12f"
_EXTRINSICS_LENGTH: Final = struct.calcsize(_EXTRINSICS_FORMAT)

#: ``Distort.bin``: five little-endian float32, OpenCV's radial/tangential
#: ordering: ``k1, k2, p1, p2, k3``. 20 bytes total.
_DISTORTION_FORMAT: Final = "<5f"
_DISTORTION_LENGTH: Final = struct.calcsize(_DISTORTION_FORMAT)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """A pinhole camera matrix and the frame size it was measured for.

    ``fx``/``fy`` are focal lengths and ``cx``/``cy`` the principal point,
    all in pixels of a ``width x height`` frame. No skew term - the vendor's
    layout fixes it at zero, checked by :func:`parse_camera_intrinsics`
    before this is ever constructed, so a nonzero skew never reaches here to
    be silently dropped.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


@dataclass(frozen=True, slots=True)
class DistortionCoefficients:
    """OpenCV's radial (``k1``, ``k2``, ``k3``) and tangential (``p1``, ``p2``)
    lens-distortion model, applied in NORMALISED (pre-intrinsics) camera
    coordinates - see :func:`project_point` for exactly where and why.
    """

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
        }


def _ensure_finite_values(values: Sequence[float], what: str) -> None:
    """The same NaN/infinity refusal as :func:`scanner3d.geometry.ensure_finite`,
    raising this module's own :class:`CalibrationBlobMalformed` instead of
    that module's :class:`~scanner3d.geometry.NonFiniteValue` - the failure
    here is "this blob is not the vendor's record", not a geometric
    complaint, and this module's own exception hierarchy is what a caller
    catching calibration-parse problems should have to know about.
    """
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise CalibrationBlobMalformed(
                f"{what}: entry {index} is {value!r}, which is not finite"
            )


def parse_camera_intrinsics(blob: bytes) -> CameraIntrinsics:
    """Decode a ``Pl.bin``/``Prgb.bin``-shaped 40-byte camera-matrix blob.

    Refuses (:class:`CalibrationBlobMalformed`) anything of the wrong
    length, any non-finite float, a non-positive frame dimension, or a
    structurally-fixed matrix entry (the skew and the two lower-row zeros,
    the bottom-right one) that is not exactly what the pinhole layout
    requires - the same checks :func:`scanner3d.intrinsics.parse_intrinsics`
    makes on the live-read path, on the same struct shape, applied here to a
    blob that already exists rather than one still arriving over USB.
    """
    if len(blob) != _CAMERA_MATRIX_LENGTH:
        raise CalibrationBlobMalformed(
            f"camera-matrix blob is {len(blob)} bytes, expected exactly {_CAMERA_MATRIX_LENGTH}"
        )
    width, height, *floats = struct.unpack(_CAMERA_MATRIX_FORMAT, blob)
    _ensure_finite_values(floats, "camera-matrix blob")
    values = dict(zip(_CAMERA_MATRIX_FIELDS, floats, strict=True))

    if width <= 0 or height <= 0:
        raise CalibrationBlobMalformed(
            f"camera-matrix blob declares a {width}x{height} frame, which cannot be projected onto"
        )
    for name, expected in _CAMERA_MATRIX_FIXED:
        actual = values[name]
        if actual != expected:
            raise CalibrationBlobMalformed(
                f"camera-matrix blob entry {name!r} is {actual!r}, but the pinhole layout fixes it "
                f"at {expected!r}: this is not a valid camera matrix"
            )

    return CameraIntrinsics(
        width=width,
        height=height,
        fx=values["fx"],
        fy=values["fy"],
        cx=values["cx"],
        cy=values["cy"],
    )


def parse_extrinsics(blob: bytes) -> Rigid:
    """Decode ``LC_RT.bin``: the rigid transform carrying a point from the
    depth camera's coordinates into the RGB camera's.

    Refuses (:class:`CalibrationBlobMalformed`) anything of the wrong
    length. The rotation is validated by :func:`scanner3d.geometry.Rigid.of`
    - reused rather than re-checked here, so "is this really a rotation"
    (orthonormal, determinant +1, no reflection - see
    :func:`scanner3d.geometry.ensure_rotation`) is answered in exactly one
    place in this codebase. A corrupted ``LC_RT.bin`` whose rotation fails
    that check raises :class:`scanner3d.geometry.NotARotation`, not this
    module's own exception - the failure is a property of the ROTATION, and
    :mod:`scanner3d.geometry` already owns naming that failure precisely.
    """
    if len(blob) != _EXTRINSICS_LENGTH:
        raise CalibrationBlobMalformed(
            f"extrinsics blob is {len(blob)} bytes, expected exactly {_EXTRINSICS_LENGTH}"
        )
    values = struct.unpack(_EXTRINSICS_FORMAT, blob)
    rotation = (values[0:3], values[3:6], values[6:9])
    translation = values[9:12]
    return Rigid.of(rotation, translation)


def parse_distortion(blob: bytes) -> DistortionCoefficients:
    """Decode ``Distort.bin``: OpenCV's ``k1, k2, p1, p2, k3`` lens-distortion
    coefficients.

    Refuses (:class:`CalibrationBlobMalformed`) anything of the wrong length
    or containing a non-finite float. Beyond that, every value is accepted
    as given: there is no "plausible distortion coefficient" range to check
    against, unlike a camera matrix's structurally-fixed zero and one
    entries.
    """
    if len(blob) != _DISTORTION_LENGTH:
        raise CalibrationBlobMalformed(
            f"distortion blob is {len(blob)} bytes, expected exactly {_DISTORTION_LENGTH}"
        )
    k1, k2, p1, p2, k3 = struct.unpack(_DISTORTION_FORMAT, blob)
    _ensure_finite_values((k1, k2, p1, p2, k3), "distortion blob")
    return DistortionCoefficients(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)


# --- projection ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RgbCalibration:
    """Everything :func:`project_point` needs, bundled: where the RGB camera
    sits relative to the depth camera, what its pinhole model is, and how it
    distorts. Built from :func:`parse_extrinsics`, :func:`parse_camera_intrinsics`
    and :func:`parse_distortion` respectively - this dataclass does not parse
    anything itself, only groups already-parsed values.
    """

    depth_to_rgb: Rigid
    rgb_intrinsics: CameraIntrinsics
    distortion: DistortionCoefficients

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "depth_to_rgb": self.depth_to_rgb.as_dict(),
            "rgb_intrinsics": self.rgb_intrinsics.as_dict(),
            "distortion": self.distortion.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProjectedPoint:
    """Where one depth-camera point lands in the RGB image - or the specific,
    named reason it does not land anywhere.

    ``rgb_point`` is always populated (the point after the extrinsics alone,
    in the RGB camera's own coordinates) - even a rejected point had SOME
    position relative to the RGB camera, and that is worth reporting rather
    than discarding, since it is exactly what a caller needs to see why a
    point was rejected (a negative ``rgb_point[2]`` is what "behind the
    camera" means). ``pixel`` is populated exactly when ``rejected`` is
    ``None``.
    """

    rgb_point: Point
    pixel: tuple[float, float] | None
    rejected: str | None

    @property
    def accepted(self) -> bool:
        return self.rejected is None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rgb_point": list(self.rgb_point),
            "pixel": list(self.pixel) if self.pixel is not None else None,
            "rejected": self.rejected,
            "accepted": self.accepted,
        }


#: Named rejection reasons :func:`project_point` reports - see
#: :attr:`ProjectedPoint.rejected`. Plain strings, not an enum: this value is
#: reported, never branched on by this module, and a plain string keeps
#: :meth:`ProjectedPoint.as_dict` trivially JSON-safe.
REJECTED_BEHIND_CAMERA: Final = "behind_camera"
REJECTED_OUTSIDE_IMAGE: Final = "outside_image"


def project_point(point: Point, calibration: RgbCalibration) -> ProjectedPoint:
    """Where ``point`` - given in DEPTH camera coordinates - lands in the RGB image.

    Three steps, in this order, and the order is load-bearing:

    1. **Extrinsics.** ``calibration.depth_to_rgb.apply(point)`` carries the
       point from depth-camera coordinates into RGB-camera coordinates. This
       has to happen first: every step after it is a property of the RGB
       camera specifically (its focal length, its principal point, its lens),
       and none of those are meaningful until the point is expressed in that
       camera's own frame.

    2. **Normalise, then distort.** The point is perspective-divided by its
       own (RGB-frame) depth to normalised coordinates ``(x/z, y/z)``, and
       OpenCV's radial/tangential model is applied THERE - not to a pixel
       coordinate. This is the step worth stating plainly, because getting
       it backwards produces a plausible-looking wrong answer rather than an
       obviously broken one: ``k1, k2, p1, p2, k3`` are calibrated in
       normalised, unit-scale coordinates, where a radius of 1 means "45
       degrees off the optical axis". Applying the identical coefficients to
       PIXEL coordinates - which run into the hundreds - would raise that
       radius to the same power against a value two to three orders of
       magnitude larger, and the correction would overwhelm the image
       instead of gently curving it; the output pixel would still be A
       number, on screen, looking like a projection, just not the right one.
       Distorting before the pixel scale ever enters the calculation is what
       keeps the correction the same physical size the calibration measured.

    3. **RGB intrinsics.** The distorted normalised coordinates are scaled by
       ``fx``/``fy`` and offset by ``cx``/``cy`` - the RGB camera's own
       matrix, never the depth camera's - to land on a pixel.

    A point at or behind the RGB camera (``rgb_point[2] <= 0``) is rejected
    with :data:`REJECTED_BEHIND_CAMERA` before step 2 ever runs: the
    perspective divide by a non-positive depth would either fail outright or
    silently reflect the point through the camera centre and report a pixel
    that looks entirely plausible for the WRONG point - the classic
    silent-wrong-answer failure mode in projection code, and the reason this
    check comes first rather than being folded into the image-bounds check
    that follows it. A point whose final pixel falls outside
    ``[0, width) x [0, height)`` is rejected with
    :data:`REJECTED_OUTSIDE_IMAGE` - visible to the depth camera, but off the
    edge of what the RGB camera actually saw.
    """
    ensure_finite(point, "point")
    rgb_point = calibration.depth_to_rgb.apply(point)
    rgb_x, rgb_y, rgb_z = rgb_point

    if rgb_z <= 0.0:
        return ProjectedPoint(rgb_point=rgb_point, pixel=None, rejected=REJECTED_BEHIND_CAMERA)

    x_norm = rgb_x / rgb_z
    y_norm = rgb_y / rgb_z

    d = calibration.distortion
    r2 = x_norm * x_norm + y_norm * y_norm
    radial = 1.0 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2
    x_distorted = (
        x_norm * radial + 2.0 * d.p1 * x_norm * y_norm + d.p2 * (r2 + 2.0 * x_norm * x_norm)
    )
    y_distorted = (
        y_norm * radial + d.p1 * (r2 + 2.0 * y_norm * y_norm) + 2.0 * d.p2 * x_norm * y_norm
    )

    intrinsics = calibration.rgb_intrinsics
    u = intrinsics.fx * x_distorted + intrinsics.cx
    v = intrinsics.fy * y_distorted + intrinsics.cy

    if not (0.0 <= u < intrinsics.width and 0.0 <= v < intrinsics.height):
        return ProjectedPoint(rgb_point=rgb_point, pixel=None, rejected=REJECTED_OUTSIDE_IMAGE)

    return ProjectedPoint(rgb_point=rgb_point, pixel=(u, v), rejected=None)


# --- colouring a cloud ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RgbImage:
    """An already-decoded RGB frame: packed RGB8, row-major, top-left origin
    (pixel row 0 is the top of the image, matching :attr:`CameraIntrinsics.cy`
    growing downward - see :mod:`scanner3d.depth_metric`'s ``CAMERA_FRAME``).

    Every way of building one goes through :meth:`of`, which is what checks
    ``pixels`` is really ``width * height * 3`` bytes - exactly the same
    discipline :meth:`scanner3d.geometry.Rigid.of` applies to a rigid
    transform, for the same reason: a buffer that is the wrong length is far
    more damaging read out-of-sync later than refused at construction.
    """

    width: int
    height: int
    pixels: bytes

    @classmethod
    def of(cls, width: int, height: int, pixels: bytes) -> RgbImage:
        if width <= 0 or height <= 0:
            raise ImageBufferMalformed(
                f"an RGB image needs positive dimensions, got {width}x{height}"
            )
        expected = width * height * 3
        if len(pixels) != expected:
            raise ImageBufferMalformed(
                f"a {width}x{height} RGB8 image needs exactly {expected} bytes, got {len(pixels)}"
            )
        return cls(width=width, height=height, pixels=pixels)

    def pixel_at(self, column: int, row: int) -> tuple[int, int, int]:
        """The RGB triple at ``(column, row)``. Caller's responsibility to
        keep both in ``[0, width)`` / ``[0, height)`` - see :func:`color_points`,
        the only caller, which only ever asks for an already-bounds-checked
        pixel.
        """
        offset = (row * self.width + column) * 3
        return (self.pixels[offset], self.pixels[offset + 1], self.pixels[offset + 2])


@dataclass(frozen=True, slots=True)
class ColoredPoint:
    """One input point's projection and, if accepted, the colour sampled for it."""

    point: Point
    projection: ProjectedPoint
    color: tuple[int, int, int] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "point": list(self.point),
            "projection": self.projection.as_dict(),
            "color": list(self.color) if self.color is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ColoringResult:
    """Every input point's outcome, in input order, plus the summary a caller
    checks first: how many landed on the image at all.
    """

    colored: tuple[ColoredPoint, ...]

    @property
    def accepted_count(self) -> int:
        return sum(1 for c in self.colored if c.color is not None)

    @property
    def rejected_count(self) -> int:
        return len(self.colored) - self.accepted_count

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "point_count": len(self.colored),
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "colored": [c.as_dict() for c in self.colored],
        }


def color_points(
    points: Sequence[Point], calibration: RgbCalibration, image: RgbImage
) -> ColoringResult:
    """Project every point in ``points`` into ``image`` and sample its colour.

    Refuses (:class:`RgbProjectNotAllowed`) an empty ``points`` - the same
    "there is nothing here to compute" refusal
    :func:`scanner3d.geometry.centroid` makes on an empty point set, for the
    same reason: an empty result is not evidence of anything, and a caller
    that meant to pass points is better served by a refusal than a silently
    empty :class:`ColoringResult`. Refuses (:class:`ImageBufferMalformed`)
    an ``image`` whose resolution does not match ``calibration``'s - see the
    module docstring's "does not rescale intrinsics" section.

    A point is coloured by NEAREST-pixel sampling of its projected,
    truncated-to-int coordinate - not bilinear. See the module docstring.
    """
    if not points:
        raise RgbProjectNotAllowed(
            "cannot colour an empty point cloud - there is nothing to project"
        )
    if (image.width, image.height) != (
        calibration.rgb_intrinsics.width,
        calibration.rgb_intrinsics.height,
    ):
        raise ImageBufferMalformed(
            f"image is {image.width}x{image.height} but the RGB calibration was measured at "
            f"{calibration.rgb_intrinsics.width}x{calibration.rgb_intrinsics.height}; this module "
            "does not rescale intrinsics to a different resolution - see the module docstring"
        )

    colored: list[ColoredPoint] = []
    for point in points:
        projection = project_point(point, calibration)
        color: tuple[int, int, int] | None = None
        if projection.accepted:
            assert projection.pixel is not None  # narrows for mypy; accepted implies a pixel
            column = int(projection.pixel[0])
            row = int(projection.pixel[1])
            color = image.pixel_at(column, row)
        colored.append(ColoredPoint(point=point, projection=projection, color=color))
    return ColoringResult(colored=tuple(colored))
