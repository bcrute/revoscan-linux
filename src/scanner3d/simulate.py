"""Render synthetic depth frames of known shapes, for testing against ground truth.

Why this exists
---------------
Every other module in this pipeline can be checked for self-consistency, but
self-consistency is not correctness. A registration that converges, a fusion
that merges and a mesh that closes can all be simultaneously wrong in the same
direction, and nothing in the pipeline would notice. What catches that is an
input whose right answer is known independently of the code under test.

This module produces exactly that: a real 512000-byte ``Y16 `` frame, byte-for-
byte the shape the kernel delivers, of a shape whose geometry is known exactly.
Deproject it and you must get the shape back. Rotate the shape by a known angle
and register the frames, and the recovered transform must be the one you asked
for. Ground truth, without hardware.

It also renders the one fault this project most needs to be able to see. The
principal point ``cx`` sits far off-centre, and if that value is wrong it acts
as a shear. Rendering with one set of intrinsics and deprojecting with another
reproduces that fault exactly, on demand, so the validation harness can be
tested against a defect whose magnitude is known rather than hoped for.

The parametrisation, and why it is exact
----------------------------------------
:mod:`~scanner3d.depth_metric` deprojects with ``x = (u - cx) * z / fx``. So a
ray through pixel ``(u, v)`` is::

    P(t) = t * ((u - cx) / fx, (v - cy) / fy, 1)

where the parameter ``t`` **is** the depth ``z``, not distance along the ray.
Rendering in that parametrisation makes this module the exact inverse of the
deprojection, so a round trip is limited only by uint16 quantisation - which is
itself worth testing, and is reported.

What this is NOT
----------------
Not a sensor model. It does not simulate the physics of structured light: no
speckle, no multipath, no subsurface scattering, no baseline shadowing. Optional
dropout and jitter exist so a consumer can be tested against imperfect input,
but a frame from here is far cleaner than a frame from the device, and no
conclusion about real-world accuracy may be drawn from it. Its purpose is to
check that arithmetic is right, not that a scanner is good.

No physical unit is implied anywhere. A radius of 20 is 20 of whatever the depth
scale measures, which remains unestablished on this device.
"""

from __future__ import annotations

import math
import random
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid, dot, normalize, subtract
from scanner3d.intrinsics import INTRINSICS_FORMAT, IntrinsicsRecord, parse_intrinsics
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

#: Serialised shape version.
SCHEMA_VERSION: Final = 1

#: A sample of 0 means no return, exactly as the device means it.
NO_RETURN: Final = 0

#: uint16 ceiling. A surface beyond this simply does not return, which is also
#: what the device does rather than wrapping around.
MAX_CODE: Final = 0xFFFF

#: One shared identity pose for shape defaults. Built once at import rather than
#: per instance; safe because :class:`~scanner3d.geometry.Rigid` is frozen.
IDENTITY_POSE: Final = Rigid.identity()


class SimulationNotAllowed(ScannerError):
    """A synthetic scene was described that cannot be rendered."""

    status = DoctorStatus.UNSUPPORTED_MODE


class ShapeUnusable(SimulationNotAllowed):
    """A shape has degenerate geometry - a non-positive radius, an empty box."""


class SceneUnusable(SimulationNotAllowed):
    """The scene as a whole cannot be rendered: no shapes, or a bad depth scale."""


class Shape(Protocol):
    """Anything a camera ray can hit.

    ``intersect`` returns the smallest positive ``t`` at which the ray
    ``P(t) = t * direction`` meets the surface, or ``None``. Because ``t`` is
    the depth in this parametrisation, the returned value is the depth sample
    directly - no conversion, and no opportunity to introduce one incorrectly.
    """

    def intersect(self, direction: Point) -> float | None: ...

    def transformed(self, pose: Rigid) -> Shape: ...

    def as_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class Sphere:
    """A sphere. The single most useful synthetic target.

    Its radius is orientation-free ground truth, and - the reason it matters
    here - a shear turns it into a tilted ellipsoid, so it is the shape that
    makes a principal-point error visible. See ``docs/capture-checklist.md``.
    """

    centre: Point
    radius: float

    def __post_init__(self) -> None:
        if not (self.radius > 0.0) or not math.isfinite(self.radius):
            raise ShapeUnusable(f"sphere radius must be positive and finite, got {self.radius!r}")

    def intersect(self, direction: Point) -> float | None:
        a = dot(direction, direction)
        b = -2.0 * dot(direction, self.centre)
        c = dot(self.centre, self.centre) - self.radius * self.radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0 or a == 0.0:
            return None
        root = math.sqrt(discriminant)
        # Nearest surface in front of the camera. The far root is kept as a
        # fallback only for a camera inside the sphere, which is a degenerate
        # scene but not one worth crashing on.
        for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if t > 0.0:
                return t
        return None

    def transformed(self, pose: Rigid) -> Sphere:
        return Sphere(centre=pose.apply(self.centre), radius=self.radius)

    def as_dict(self) -> dict[str, object]:
        return {"kind": "sphere", "centre": list(self.centre), "radius": self.radius}


@dataclass(frozen=True, slots=True)
class Plane:
    """An infinite plane, ``dot(normal, p) = offset``.

    Present mainly as the negative control for the shear test: a shear maps
    planes to planes, so a plane rendered with a wrong ``cx`` still fits a plane
    perfectly. A test that asserts that is what proves the trap is real rather
    than merely asserted in prose.
    """

    normal: Point
    offset: float

    def __post_init__(self) -> None:
        # normalize refuses a degenerate direction, which is exactly the check
        # this needs; the result is discarded because the field is frozen.
        normalize(self.normal)

    def intersect(self, direction: Point) -> float | None:
        denominator = dot(self.normal, direction)
        if abs(denominator) < 1e-12:
            return None
        t = self.offset / denominator
        return t if t > 0.0 else None

    def transformed(self, pose: Rigid) -> Plane:
        spin = Rigid.of(pose.rotation, (0.0, 0.0, 0.0))
        moved_normal = spin.apply(self.normal)
        point_on_plane = tuple(component * self.offset for component in normalize(self.normal))
        moved_point = pose.apply(point_on_plane)  # type: ignore[arg-type]
        return Plane(normal=moved_normal, offset=dot(moved_normal, moved_point))

    def as_dict(self) -> dict[str, object]:
        return {"kind": "plane", "normal": list(self.normal), "offset": self.offset}


@dataclass(frozen=True, slots=True)
class Box:
    """An oriented box: an axis-aligned extent carried by a pose.

    The asymmetric target. A sphere is useless for turntable registration
    because it looks identical from every angle and therefore carries no
    information about how far it turned; a box does not have that problem, which
    is why both shapes exist here.
    """

    half_extents: Point
    #: Shared rather than constructed per instance. Safe because Rigid is
    #: frozen, so no Box can mutate the default out from under another.
    pose: Rigid = IDENTITY_POSE

    #: The camera origin and the rotation, precomputed in the box's own frame.
    #: Derived state, never passed in. A frame is 256000 rays and every one of
    #: them needs these; computing ``pose.inverse()`` per ray made rendering a
    #: 12-frame sweep take 48 seconds. They are set through ``object.__setattr__``
    #: because the dataclass is frozen, which is the standard way to hold
    #: derived state on an immutable value.
    #: Excluded from repr and eq so two boxes with the same geometry stay
    #: equal regardless of what either has cached.
    _local_origin: Point = field(init=False, repr=False, compare=False)
    _local_rotation: Rigid = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not all(e > 0.0 and math.isfinite(e) for e in self.half_extents):
            raise ShapeUnusable(
                f"box half-extents must all be positive and finite, got {self.half_extents!r}"
            )
        inverse = self.pose.inverse()
        object.__setattr__(self, "_local_origin", inverse.apply((0.0, 0.0, 0.0)))
        object.__setattr__(self, "_local_rotation", Rigid.of(inverse.rotation, (0.0, 0.0, 0.0)))

    def intersect(self, direction: Point) -> float | None:
        # Work in the box's own frame: transform the ray rather than the box.
        origin = self._local_origin
        local = self._local_rotation.apply(direction)

        near, far = -math.inf, math.inf
        for axis in range(3):
            d = local[axis]
            o = origin[axis]
            extent = self.half_extents[axis]
            if abs(d) < 1e-12:
                if o < -extent or o > extent:
                    return None
                continue
            t0 = (-extent - o) / d
            t1 = (extent - o) / d
            if t0 > t1:
                t0, t1 = t1, t0
            near = max(near, t0)
            far = min(far, t1)
            if near > far:
                return None
        return near if near > 0.0 else (far if far > 0.0 else None)

    def transformed(self, pose: Rigid) -> Box:
        return Box(half_extents=self.half_extents, pose=pose.compose(self.pose))

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "box",
            "half_extents": list(self.half_extents),
            "pose": self.pose.as_dict(),
        }


def intrinsics_of(
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> IntrinsicsRecord:
    """Build a record through the real 40-byte parser, never by hand.

    Going through :func:`~scanner3d.intrinsics.parse_intrinsics` means a
    synthetic record is exactly as constrained as one read off the device, so a
    test cannot accidentally exercise a record shape the device could never
    produce.
    """
    blob = struct.pack(INTRINSICS_FORMAT, width, height, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
    parsed = parse_intrinsics(blob)
    if parsed is None:
        raise SceneUnusable(f"the parser refused the synthetic intrinsics fx={fx} cx={cx}")
    return parsed


#: This unit's factory record, for rendering scenes that match its geometry.
FACTORY_INTRINSICS: Final = intrinsics_of(
    fx=898.6829833984375,
    fy=898.6829833984375,
    cx=42.43747329711914,
    cy=192.8326873779297,
)


@dataclass(frozen=True, slots=True)
class SyntheticFrame:
    """A rendered frame and the truth it was rendered from."""

    raw: bytes
    intrinsics: IntrinsicsRecord
    depth_scale: float
    returned: int
    dropped: int
    beyond_range: int
    max_quantisation_error: float
    shapes: tuple[dict[str, object], ...]

    @property
    def samples(self) -> int:
        return FRAME_WIDTH * FRAME_HEIGHT

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "bytes": len(self.raw),
            "samples": self.samples,
            "returned": self.returned,
            "dropped": self.dropped,
            "beyond_range": self.beyond_range,
            "max_quantisation_error": self.max_quantisation_error,
            "depth_scale": self.depth_scale,
            "shapes": list(self.shapes),
            "limitations": [
                "Synthetic. Not a sensor model: no speckle, multipath, subsurface "
                "scattering or baseline shadowing is simulated, so this frame is far "
                "cleaner than any the device produces.",
                "No conclusion about real-world accuracy may be drawn from a synthetic "
                "frame. It checks that arithmetic is right, not that a scanner is good.",
                "No physical unit is implied. Distances are in whatever the depth scale "
                "measures, which is unestablished on this device.",
            ],
        }


def render(
    shapes: Sequence[Shape],
    *,
    intrinsics: IntrinsicsRecord = FACTORY_INTRINSICS,
    depth_scale: float = 0.004999999888241291,
    pose: Rigid | None = None,
    dropout: float = 0.0,
    jitter: float = 0.0,
    seed: int = 0,
) -> SyntheticFrame:
    """Render ``shapes`` into one real 512000-byte ``Y16 `` frame.

    ``pose`` moves the whole scene, which is how a turntable sweep is produced:
    render the same shapes at a series of :func:`geometry.rotation_about_axis`
    poses and the ground-truth transform between any two frames is known
    exactly.

    ``dropout`` randomly discards a fraction of returns and ``jitter`` perturbs
    depths, both so a consumer can be exercised against imperfect input. Both
    default to zero, because a test asserting exact recovery should not have to
    opt out of noise.
    """
    if not shapes:
        raise SceneUnusable("a scene needs at least one shape")
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise SceneUnusable(f"depth scale must be positive and finite, got {depth_scale!r}")
    if not 0.0 <= dropout <= 1.0:
        raise SceneUnusable(f"dropout must be a fraction in [0, 1], got {dropout!r}")
    if jitter < 0.0 or not math.isfinite(jitter):
        raise SceneUnusable(f"jitter must be non-negative and finite, got {jitter!r}")

    placed = [shape.transformed(pose) for shape in shapes] if pose is not None else list(shapes)
    rng = random.Random(seed)

    # Read through the same mapping the production deprojection reads, so a
    # record this module accepts is one depth_metric would also accept.
    fx, fy = intrinsics.values["fx"], intrinsics.values["fy"]
    cx, cy = intrinsics.values["cx"], intrinsics.values["cy"]
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx == 0.0 or fy == 0.0:
        raise SceneUnusable(f"intrinsics cannot carry a projection: fx={fx!r} fy={fy!r}")
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    returned = dropped = beyond = 0
    worst_quantisation = 0.0

    for v in range(FRAME_HEIGHT):
        dy = (v - cy) / fy
        row = v * FRAME_WIDTH
        for u in range(FRAME_WIDTH):
            direction = ((u - cx) / fx, dy, 1.0)
            nearest: float | None = None
            for shape in placed:
                t = shape.intersect(direction)
                if t is not None and (nearest is None or t < nearest):
                    nearest = t
            if nearest is None:
                continue
            if dropout and rng.random() < dropout:
                dropped += 1
                continue
            depth = nearest + (rng.gauss(0.0, jitter) if jitter else 0.0)
            if depth <= 0.0:
                continue
            code = round(depth / depth_scale)
            if code <= NO_RETURN or code > MAX_CODE:
                beyond += 1
                continue
            worst_quantisation = max(worst_quantisation, abs(code * depth_scale - depth))
            struct.pack_into("<H", buffer, (row + u) * 2, code)
            returned += 1

    return SyntheticFrame(
        raw=bytes(buffer),
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        returned=returned,
        dropped=dropped,
        beyond_range=beyond,
        max_quantisation_error=worst_quantisation,
        shapes=tuple(shape.as_dict() for shape in placed),
    )


def turntable_sweep(
    shapes: Sequence[Shape],
    *,
    frames: int,
    axis: Point,
    through: Point,
    intrinsics: IntrinsicsRecord = FACTORY_INTRINSICS,
    depth_scale: float = 0.004999999888241291,
    **render_options: object,
) -> tuple[tuple[SyntheticFrame, ...], tuple[Rigid, ...]]:
    """A full 360-degree sweep, and the ground-truth pose of every frame.

    The registration test this project actually needs: the recovered transforms
    must match the returned poses, and composing all of them must return to the
    identity. Because the sweep closes by construction, a loop-closure error
    measured on it is a measurement of the *algorithm*, with no capture error
    mixed in.
    """
    from scanner3d.geometry import rotation_about_axis

    if frames < 2:
        raise SceneUnusable(f"a sweep needs at least 2 frames, got {frames}")
    poses = tuple(
        rotation_about_axis(axis, 2.0 * math.pi * index / frames, through=through)
        for index in range(frames)
    )
    rendered = tuple(
        render(
            shapes,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            pose=pose,
            **render_options,  # type: ignore[arg-type]
        )
        for pose in poses
    )
    return rendered, poses


def sheared_intrinsics(source: IntrinsicsRecord, *, cx: float) -> IntrinsicsRecord:
    """The same record with a different principal point.

    Render with one and deproject with the other to reproduce the principal-
    point shear exactly, at a magnitude you chose. That is what lets the
    validation harness be tested against a defect of known size instead of
    against hope.
    """
    return intrinsics_of(
        fx=source.values["fx"],
        fy=source.values["fy"],
        cx=cx,
        cy=source.values["cy"],
        width=source.width,
        height=source.height,
    )


def points_of(frame: SyntheticFrame) -> Iterable[Point]:
    """Convenience: the frame's own points, deprojected with its own intrinsics.

    Deliberately routed through the real :mod:`~scanner3d.depth_metric` path
    rather than re-deriving the arithmetic here, so a synthetic round trip
    exercises the production deprojection and cannot silently diverge from it.
    """
    from scanner3d.cloud_export import camera_points
    from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame

    conversion = deproject_depth_frame(
        frame.raw,
        intrinsics=frame.intrinsics,
        depth_scale=frame.depth_scale,
        unit=UNVALIDATED_UNIT,
    )
    return camera_points(conversion)


def recovery_error(frame: SyntheticFrame, shape: Sphere) -> tuple[float, int]:
    """Max radial deviation of a rendered sphere's points from its true surface.

    The end-to-end round-trip check in one number: render a sphere, deproject
    it, and ask how far the recovered points sit from the surface they came
    from. Bounded below by uint16 quantisation, so a test can assert against
    ``frame.max_quantisation_error`` rather than an arbitrary tolerance.
    """
    worst = 0.0
    count = 0
    for point in points_of(frame):
        worst = max(
            worst,
            abs(
                math.sqrt(dot(subtract(point, shape.centre), subtract(point, shape.centre)))
                - shape.radius
            ),
        )
        count += 1
    return worst, count
