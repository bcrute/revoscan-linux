"""Projective truncated signed-distance fusion, for frames whose poses are known.

Why this module exists
----------------------
Everything this repository can currently turn a scan into is a *union*.
:mod:`scanner3d.fusion` unions clouds into one cloud, and
:mod:`scanner3d.mesh` unions the cells those points fell in and marches the
boundary of that solid. A union has one property that no amount of tuning
removes: it can only ever grow. A cell that one noisy frame marked solid stays
solid however many later frames disagree, so the surface stands OUTSIDE the
samples by roughly a voxel and is faceted at the cell edge. That is why the
live view starts on points.

A signed-distance field is the other thing a set of depth frames can be turned
into, and it is different in kind rather than in degree. Each frame writes,
into every voxel it can see, a *signed distance to the surface it observed*.
Voxels nearer the camera than the observed surface get a positive value, voxels
behind it get a negative one, and the surface is where the field crosses zero -
a level set that lies BETWEEN voxel centres and is found by interpolation, not
at a cell boundary. Because each voxel holds a running weighted mean rather
than a flag, N independent noisy observations of one voxel average, and the
zero crossing they imply is more accurate than any one of them. Nothing about
occupancy can do that; averaging is the whole point.

What this module is for, and what it is NOT
-------------------------------------------
This is a bounded *reference* implementation, written to establish that the
sign, the transform direction, the truncation, the weighting and the bounds are
right before any of it is wired into a product path. It produces a scalar field,
a bag of inspectable zero-level sample points, and a read-only view of the field
for :mod:`scanner3d.tsdf_surface` to triangulate. It does NOT produce a
connected mesh itself, it is not called by the GUI, the reconstruction export,
the capture path or the CLI, and neither it nor the surface module is a claim
that this repository now has TSDF reconstruction. It is the arithmetic, proved.

The two conventions this module fixes, and does not restate elsewhere
--------------------------------------------------------------------
**The pose direction.** A pose given to :meth:`TsdfVolume.integrate` maps that
frame's CAMERA coordinates into the shared MODEL frame - the same direction
:func:`scanner3d.reconstruction_export.registered_points` applies, so a caller
that has poses good enough to place points has poses good enough to fuse.
Integration needs the opposite direction, and inverts the pose itself: a voxel
lives in the model frame and has to be carried back into the camera to be
projected. Getting this backwards produces a scan that is wrong in a way that
still looks like a scan, so :meth:`integrate` takes the same direction as the
rest of the pipeline and never a "camera pose" that is silently the inverse.

**The sign.** For a voxel whose model-frame centre carries back to camera-frame
``(x, y, z)``, and a depth image whose pixel under that voxel observed distance
``d``::

    sdf = d - z            # observed camera-z minus projected voxel camera-z

so a voxel IN FRONT of the observed surface - nearer the camera, smaller z -
is POSITIVE, and one behind it is negative. Along a ray from the camera the
field therefore runs positive, crosses zero at the surface, and goes negative
inside the object. The projection itself is the one
:data:`~scanner3d.depth_metric.DEPROJECTION_EXPRESSION` inverts, read out of
the same :class:`~scanner3d.intrinsics.IntrinsicsRecord`, so this module and
the deprojection cannot disagree about where a pixel points.

What this measures, precisely, and what it does not
---------------------------------------------------
``d - z`` is a distance along the camera's z axis, not the true Euclidean
distance from the voxel to the surface. The two agree exactly when the surface
is perpendicular to the viewing ray and diverge as it tilts away, by a factor of
one over the cosine of the angle between the ray and the surface normal. This
is the standard projective approximation and it is why the truncation band has
to be several voxels wide: within a narrow band around the surface the error is
a fraction of a voxel, and outside it the values are clamped and carry no
geometry anyway. It is stated here, and in every result's ``limitations``,
because a reader who assumes a true SDF will over-trust the field's magnitude
away from the zero crossing.

Nothing here estimates, refines, repairs or scores a pose. A pose that is wrong
is integrated exactly as faithfully as one that is right, and the result is a
confidently wrong surface: fusion does not remove the tracking requirement, it
depends on it. Nor does anything here establish a physical unit. A distance of
1.0 is one unit of whatever the injected depth scale produces, and that
scalar's physical meaning remains unestablished on this device.

numpy is optional
-----------------
This module imports, and its ``"python"`` backend runs, with numpy entirely
absent - no import of numpy happens at module scope. The ``"numpy"`` backend is
an accelerator for the same arithmetic, selected by name; an explicit request
for a backend this environment cannot satisfy raises rather than quietly using
a different one. There is no GPU backend here, reserved or otherwise: this task
is the reference field, and a compute-API claim would have to be earned
separately.
"""

from __future__ import annotations

import array
import enum
import importlib
import math
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from scanner3d.depth_metric import (
    DEPROJECTION_EXPRESSION,
    NO_RETURN_SAMPLE,
    ensure_frame_length,
    ensure_intrinsics_usable,
    ensure_scale_usable,
)
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.tsdf_input import (
    MEASURED_WEIGHT,
    PROVENANCE_MEASURED,
    PROVENANCE_NAMES,
    PROVENANCE_TEMPORALLY_RECONSTRUCTED,
    WeightedDepthFrame,
)

# Imported across the module boundary for the same reason
# :mod:`scanner3d.depth_metric` does it: the one depth mode this repository
# captures is defined in :mod:`scanner3d.v4l2_capture`, and a second copy of
# 640, 400 and 512000 here would be a set of numbers that could drift apart
# from the capture that produces the bytes.
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy
    from numpy.typing import NDArray

    from scanner3d.intrinsics import IntrinsicsRecord

    # Named here rather than inline so that the numpy-shaped types stay out of
    # this module's runtime namespace entirely: they exist for the checker, and
    # the accelerated methods below still do their own local ``import numpy``.
    FloatArray = NDArray[numpy.float64]
    BoolArray = NDArray[numpy.bool_]
    IndexArray = NDArray[numpy.int64]
    ProvenanceArray = NDArray[numpy.uint8]

#: Every top-level report this module emits stamps this.
SCHEMA_VERSION: Final = 1

#: The algorithm, named once so the report and the code cannot disagree.
FUSION_MODE: Final = "projective truncated signed-distance fusion, running weighted mean"

#: The sign convention, written once for the same reason.
SIGN_CONVENTION: Final = (
    "sdf = observed camera-z - projected voxel camera-z; positive in front of the observed "
    "surface (nearer the camera), negative behind it"
)

#: The most voxels one volume may own: 256^3. The field is two float32 values
#: per voxel - the running mean and its accumulated weight - so this ceiling is
#: 128 MiB of owned storage, and the chunked integration below adds a fixed
#: temporary on top of it rather than a fraction of the volume. A caller may
#: lower it through :attr:`TsdfConfig.max_voxels` and may never raise it, and
#: it is checked before a single byte of field is allocated.
MAX_VOXELS: Final = 256 * 256 * 256

#: The lower ceiling that binds when the pure-Python backend is selected. The
#: python path touches every voxel of the volume once per frame in interpreted
#: arithmetic, measured on this host at 0.59 microseconds per voxel observation
#: against 0.073 for the vectorised path. At 64^3 that is 0.15 s a frame - a
#: few seconds over a whole sweep - while the 256^3 ceiling above would be
#: about 10 s A FRAME, or twenty minutes over a hundred-frame capture. So this
#: is a wall-clock ceiling rather than a memory one, and it exists so that a
#: backend chosen for portability cannot turn a legal volume into what looks
#: from outside like a hang. Fixed, like the one above, and never widened at
#: runtime.
PYTHON_BACKEND_MAX_VOXELS: Final = 64 * 64 * 64

#: Bytes one stored value occupies. ``array("f")`` is a C float and numpy's
#: float32 is four bytes wide; the accelerated path views the very same buffer
#: the reference path writes, so the assumption is named and checked rather
#: than left implicit in two places that could disagree.
VALUE_BYTES: Final = 4

#: Voxels per integration chunk. Integration builds about ten temporaries per
#: chunk, so this fixes the transient cost at roughly 5 MiB whatever the volume
#: owns - the point of chunking is that a bigger volume costs more passes, not
#: bigger arrays. Chunks are cut on the flat index, which means every voxel
#: appears in exactly one chunk and no scatter can collide with itself.
CHUNK_VOXELS: Final = 65_536

#: The most (frame, pose) pairs one :meth:`TsdfVolume.integrate` call accepts.
#: The volume's memory does not grow with frame count - that is the property
#: that makes fusion incremental - but its wall-clock does, linearly, so this
#: bounds one call rather than the volume's lifetime. Several calls may be made.
MAX_FRAMES_PER_CALL: Final = 4096

#: The most zero-level sample points one call may materialise. Each is a
#: three-float Python tuple costing on the order of 150 bytes once its floats
#: are counted, so this is about 150 MiB of result and is checked BEFORE the
#: first point is built. Evaluation geometry is meant to be measured, not
#: rendered; a caller that wants more than this wants a mesh, which this module
#: deliberately does not produce.
MAX_ZERO_SAMPLES: Final = 1_000_000

#: Default cap on one voxel's accumulated weight. Capping is what keeps a long
#: capture responsive to the frames arriving now: past the cap the running mean
#: becomes an exponential forget rather than an all-time average, so a voxel
#: that a hundred frames agreed on can still be moved by evidence that the
#: object has been seen better. It also bounds the stored weight, which is the
#: reason it is enforced rather than merely offered.
DEFAULT_MAX_WEIGHT: Final = 64.0

#: The hard cap on that parameter. Above this a float32 weight starts losing
#: the increments being added to it, so a "higher" cap would silently stop
#: accumulating rather than average harder.
MAX_ACCUMULATED_WEIGHT: Final = 1_048_576.0

#: Backend names this module knows, in the order ``"auto"`` tries them.
KNOWN_BACKENDS: Final[tuple[str, ...]] = ("numpy", "python")

#: The name that means "use the fastest usable backend and say which".
AUTO_BACKEND: Final = "auto"

#: True of the FIELD itself, whatever the inputs were and whatever is later read
#: out of it. Split out from :data:`INHERENT_LIMITATIONS` so that a consumer
#: which turns this same field into something that is not a bag of unconnected
#: samples - :mod:`scanner3d.tsdf_surface` turns it into a connected mesh - can
#: carry every caveat that is about the field without also carrying the one
#: caveat below that is about the sample bag and is false of a surface.
FIELD_LIMITATIONS: Final[tuple[str, ...]] = (
    "The signed distance is PROJECTIVE: it is the difference of camera-z values, not the "
    "Euclidean distance from the voxel to the surface. The two agree where the surface faces "
    "the camera and diverge as one over the cosine of the angle between the ray and the "
    "surface normal, so the field's magnitude away from the zero crossing is an "
    "over-estimate of distance on oblique geometry.",
    "Depth is read at the NEAREST pixel, with no interpolation between neighbouring pixels "
    "and no lens-distortion model, so the field inherits the depth image's own pixel "
    "quantisation and whatever distortion the injected intrinsics do not describe.",
    "A voxel no frame ever saw keeps weight zero and is not a claim about free space. This "
    "module carves only where a frame actually observed something in front of the voxel; "
    "unobserved volume is unobserved, not empty.",
    "Poses are consumed exactly as given. Nothing here estimates, refines, repairs, scores "
    "or corrects one, and a wrong pose is fused as faithfully as a right one into a "
    "confidently wrong surface. Fusion depends on tracking; it does not replace it.",
    "No dimensional or pose accuracy is claimed beyond whatever the injected intrinsics, "
    "depth scale and poses already carried. The depth scale's physical unit remains "
    "unestablished on this device, so a distance of 1.0 is one unit of an unnamed quantity.",
    "The field is stored as float32 per voxel, which resolves about one part in 10^7 - far "
    "below the voxel size, and stated only so that an exact-equality comparison against a "
    "float64 computation is not mistaken for a defect.",
)

#: The one caveat that is about :meth:`TsdfVolume.zero_level_samples` rather than
#: about the field. It is named separately because it is FALSE of a connected
#: surface extracted from the same voxels, and a result that carried it anyway
#: would be telling its reader that the thing in their hands has no topology.
UNCONNECTED_SAMPLES_LIMITATION: Final = (
    "Zero-level samples are UNCONNECTED POINTS on the level set, not a mesh. They carry no "
    "topology, no normals, no orientation and no watertightness, and nothing in this module "
    "should be read as a claim that a connected TSDF surface exists here - "
    "scanner3d.tsdf_surface extracts one from this same field, and no product path reaches "
    "either module."
)

#: True of every result THIS module produces, whatever the inputs were. Carried
#: on each result so that a reader of the JSON cannot miss them. The order is
#: the one it has always had: the sample-bag caveat sits fourth, between the
#: unobserved-volume caveat and the pose one.
INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    *FIELD_LIMITATIONS[:3],
    UNCONNECTED_SAMPLES_LIMITATION,
    *FIELD_LIMITATIONS[3:],
)


# --- refusals -----------------------------------------------------------------


class TsdfNotAllowed(ScannerError):
    """A volume or an integration was requested that this module refuses.

    Always raised before the field is allocated and before a frame is read, so
    a refused call costs no memory it did not already have and leaves no
    half-integrated volume behind it.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class VolumeBoundsUnusable(TsdfNotAllowed):
    """The model-space bounds are not a finite, strictly ordered box."""


class VoxelParametersUnusable(TsdfNotAllowed):
    """The voxel size or the truncation distance cannot define a field."""


class DepthRangeUnusable(TsdfNotAllowed):
    """The accepted depth interval is not a finite, strictly ordered, positive one."""


class WeightPolicyUnusable(TsdfNotAllowed):
    """The accumulated-weight cap is outside the interval this module can store."""


class VolumeTooLarge(TsdfNotAllowed):
    """The requested volume needs more cells than the ceiling allows.

    Raised from the shape arithmetic alone, so an oversized volume is refused
    without a single voxel being allocated.
    """


class PoseCountMismatch(TsdfNotAllowed):
    """A different number of poses than frames was offered.

    Refused rather than zipped to the shorter of the two: a silently truncated
    integration is a partial scan that reports itself as complete, and a
    silently mispaired one fuses every frame at its neighbour's pose.
    """


class FrameCountExceeded(TsdfNotAllowed):
    """More frames were offered in one call than the fixed per-call ceiling."""


class TsdfBackendUnknown(TsdfNotAllowed):
    """A backend was requested by a name this module does not implement."""


class TsdfBackendUnavailable(TsdfNotAllowed):
    """The requested backend is a real name here, but is not usable in this environment.

    Raised for an explicit request this environment cannot satisfy - numpy not
    importable - and never to mean "so something else was used instead".
    :data:`AUTO_BACKEND` is the way to ask for a fallback.
    """


class ZeroLevelCeilingExceeded(TsdfNotAllowed):
    """More zero crossings exist than the caller or the module will materialise.

    Counted first and raised before any point is built, so the refusal costs
    one bounded pass over the field rather than the memory it was refusing.
    """


class VolumeEmpty(TsdfNotAllowed):
    """A bounding box was asked for around no points at all."""


class WeightedInputUnusable(TsdfNotAllowed):
    """Something offered to :meth:`TsdfVolume.integrate_weighted` is not a weighted frame.

    Raised before a single voxel is touched, like every other refusal here.
    The frames themselves are :class:`~scanner3d.tsdf_input.WeightedDepthFrame`
    instances, which have already bound their own provenance to their own raw
    bytes and to the accepted completion result behind them - so this refusal
    is about what the CALL was handed, and never about re-litigating a frame
    that type already validated.
    """


# --- the configuration --------------------------------------------------------


class WeightPolicy(enum.StrEnum):
    """How much one accepted observation of one voxel is worth.

    Both policies are honest choices with different failure modes, which is
    why the field records which one produced it rather than hard-coding one.
    """

    #: Every accepted observation weighs exactly 1. The running mean is then the
    #: plain arithmetic mean of the observations, which is the estimator that
    #: minimises variance when the observations are equally uncertain - and the
    #: one whose behaviour is easiest to reason about when a result is wrong.
    UNIFORM = "uniform"

    #: Weight 1 in front of the observed surface, ramping linearly to 0 at the
    #: truncation distance BEHIND it. The band behind a surface is the
    #: uncertain one - it is occluded, and its signed distance is exactly where
    #: the projective approximation is worst - so this down-weights it instead
    #: of letting it pull the zero crossing outward.
    RAMPED_BEHIND = "ramped_behind"


@dataclass(frozen=True, slots=True)
class VolumeBounds:
    """The axis-aligned model-space box a field covers. Numbers, not claims."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @classmethod
    def of(
        cls,
        minimum: Point,
        maximum: Point,
    ) -> VolumeBounds:
        """Build validated bounds, or refuse.

        Ordering is strict on every axis: a degenerate box has no interior for
        a voxel centre to sit in, and a reversed one would produce a negative
        shape that later arithmetic would quietly clamp to something legal.
        """
        values = (*minimum, *maximum)
        if not all(math.isfinite(value) for value in values):
            raise VolumeBoundsUnusable(
                f"volume bounds {minimum!r}..{maximum!r} contain a non-finite value; no voxel "
                "centre computed from them would mean anything"
            )
        for axis, low, high in zip("xyz", minimum, maximum, strict=True):
            if not low < high:
                raise VolumeBoundsUnusable(
                    f"volume bounds are not strictly ordered on {axis}: {low!r} is not below "
                    f"{high!r}. A box with no extent on an axis has no interior to fuse into"
                )
        return cls(
            min_x=minimum[0],
            min_y=minimum[1],
            min_z=minimum[2],
            max_x=maximum[0],
            max_y=maximum[1],
            max_z=maximum[2],
        )

    @classmethod
    def around(cls, points: Iterable[Sequence[float]], *, margin: float) -> VolumeBounds:
        """The box that contains ``points``, grown by ``margin`` on every side.

        The margin is not decoration. A voxel exactly on the extent of the
        cloud has no neighbour on its far side to cross zero against, so a
        volume cut flush to the samples loses the outermost shell of the very
        surface it was built to recover. One or two truncation distances is the
        honest default for a caller to pass.

        Streams its input: a cloud is not held twice to be measured.
        """
        if not math.isfinite(margin) or margin < 0.0:
            raise VolumeBoundsUnusable(f"margin must be finite and non-negative, got {margin!r}")
        low = [math.inf, math.inf, math.inf]
        high = [-math.inf, -math.inf, -math.inf]
        seen = 0
        for point in points:
            # Arity first, and by refusal rather than by indexing. A short point
            # would otherwise leak an IndexError from the middle of the
            # measurement, and a long one would be measured on its first three
            # coordinates as though the rest were not there - a box quietly
            # built around the wrong quantity. Both are the caller's cloud being
            # something other than model-space points, and both are named.
            arity = len(point)
            if arity != 3:
                raise VolumeBoundsUnusable(
                    f"point {seen} carries {arity} coordinates, and a model-space point has "
                    "exactly 3 (x, y, z). No bounds were returned: a shorter point cannot be "
                    "indexed and a longer one is not this cloud's geometry with two spare "
                    "numbers on the end"
                )
            for axis in range(3):
                value = float(point[axis])
                if not math.isfinite(value):
                    raise VolumeBoundsUnusable(
                        f"point {seen} has a non-finite coordinate {value!r}; a box measured "
                        "around it would be infinite"
                    )
                low[axis] = min(low[axis], value)
                high[axis] = max(high[axis], value)
            seen += 1
        if seen == 0:
            raise VolumeEmpty(
                "no points were offered, so there is no extent to build a volume around. "
                "Refused rather than answered with a unit box, which downstream would read "
                "exactly like a volume around a scan that produced nothing"
            )
        # A cloud can be flat - a single frame of a plane is - so the margin has
        # to be able to open the box on its own. It is applied on both sides,
        # which makes every axis at least 2 * margin wide.
        minimum = (low[0] - margin, low[1] - margin, low[2] - margin)
        maximum = (high[0] + margin, high[1] + margin, high[2] + margin)
        return cls.of(minimum, maximum)

    @property
    def minimum(self) -> Point:
        return (self.min_x, self.min_y, self.min_z)

    @property
    def maximum(self) -> Point:
        return (self.max_x, self.max_y, self.max_z)

    @property
    def extent(self) -> Point:
        return (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min": list(self.minimum),
            "max": list(self.maximum),
            "extent": list(self.extent),
        }


@dataclass(frozen=True, slots=True)
class TsdfConfig:
    """Everything about the field that is decided before a frame is seen.

    Separate from the volume itself so that the parameters can be validated,
    reported and compared without allocating 128 MiB to find out they were
    wrong, and so that a result carries the exact configuration that produced
    it rather than a reader's memory of what was passed.
    """

    bounds: VolumeBounds
    voxel_size: float
    truncation: float
    min_depth: float
    max_depth: float
    weight_policy: WeightPolicy
    max_weight: float
    max_voxels: int

    @classmethod
    def of(
        cls,
        *,
        bounds: VolumeBounds,
        voxel_size: float,
        truncation: float,
        min_depth: float,
        max_depth: float,
        weight_policy: WeightPolicy = WeightPolicy.UNIFORM,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        max_voxels: int = MAX_VOXELS,
    ) -> TsdfConfig:
        """Build a validated configuration, or refuse before anything is allocated."""
        if not 0 < max_voxels <= MAX_VOXELS:
            raise VolumeTooLarge(
                f"max_voxels {max_voxels} is outside (0, {MAX_VOXELS}]; the module ceiling is "
                "never widened at run time"
            )
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise VoxelParametersUnusable(
                f"voxel size must be finite and positive, got {voxel_size!r}; it is the "
                "denominator of every index this module computes"
            )
        if not math.isfinite(truncation) or truncation <= 0.0:
            raise VoxelParametersUnusable(
                f"truncation must be finite and positive, got {truncation!r}; it is the "
                "denominator that normalises every stored value"
            )
        if truncation < voxel_size:
            raise VoxelParametersUnusable(
                f"truncation {truncation!r} is below the voxel size {voxel_size!r}. The zero "
                "crossing is found BETWEEN neighbouring voxel centres, and a band narrower "
                "than the gap between them can leave both neighbours clamped or unobserved, "
                "so the surface would disappear rather than merely coarsen. Two to four "
                "voxels is the usual band"
            )
        for name, value in (("min_depth", min_depth), ("max_depth", max_depth)):
            if not math.isfinite(value) or value <= 0.0:
                raise DepthRangeUnusable(
                    f"{name} must be finite and positive, got {value!r}; a non-positive "
                    "accepted distance describes a surface behind the camera"
                )
        if not min_depth < max_depth:
            raise DepthRangeUnusable(
                f"the accepted depth interval is not strictly ordered: min_depth "
                f"{min_depth!r} is not below max_depth {max_depth!r}, so no observation "
                "could fall inside it"
            )
        if not math.isfinite(max_weight) or not 0.0 < max_weight <= MAX_ACCUMULATED_WEIGHT:
            raise WeightPolicyUnusable(
                f"max_weight {max_weight!r} is outside (0, {MAX_ACCUMULATED_WEIGHT}]; past "
                "that a float32 weight stops resolving the increments added to it, so a "
                "higher cap would silently stop accumulating rather than average harder"
            )
        try:
            policy = WeightPolicy(weight_policy)
        except ValueError as exc:
            raise WeightPolicyUnusable(
                f"weight policy {weight_policy!r} is not one of "
                f"{tuple(str(known) for known in WeightPolicy)}"
            ) from exc
        config = cls(
            bounds=bounds,
            voxel_size=voxel_size,
            truncation=truncation,
            min_depth=min_depth,
            max_depth=max_depth,
            weight_policy=policy,
            max_weight=max_weight,
            max_voxels=max_voxels,
        )
        cells = config.cell_count
        if cells > max_voxels:
            raise VolumeTooLarge(
                f"a {config.shape[0]}x{config.shape[1]}x{config.shape[2]} volume is {cells} "
                f"voxels, past the {max_voxels} ceiling; nothing was allocated. Widen the "
                "voxel size or narrow the bounds rather than the ceiling"
            )
        return config

    @property
    def shape(self) -> tuple[int, int, int]:
        """Voxel counts along x, y and z.

        Rounded UP, so the realised volume covers the requested bounds rather
        than stopping short of them; :attr:`realised_maximum` says where it
        actually stops. Every axis holds at least one voxel because the bounds
        are strictly ordered.
        """
        extent = self.bounds.extent
        return (
            max(1, math.ceil(extent[0] / self.voxel_size)),
            max(1, math.ceil(extent[1] / self.voxel_size)),
            max(1, math.ceil(extent[2] / self.voxel_size)),
        )

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def realised_maximum(self) -> Point:
        """The far corner the whole-voxel shape actually reaches.

        At or beyond :attr:`VolumeBounds.maximum`, never short of it. Reported
        because a caller measuring against its own requested box would
        otherwise be surprised by up to one voxel of extra volume per axis.
        """
        nx, ny, nz = self.shape
        return (
            self.bounds.min_x + nx * self.voxel_size,
            self.bounds.min_y + ny * self.voxel_size,
            self.bounds.min_z + nz * self.voxel_size,
        )

    @property
    def owned_bytes(self) -> int:
        """Bytes the field itself occupies: two float32 values per voxel."""
        return self.cell_count * 8

    def voxel_centre(self, i: int, j: int, k: int) -> Point:
        """The model-frame centre of voxel ``(i, j, k)``.

        Centres, not corners: a voxel's value is a statement about the middle of
        the cell, and interpolating between corners would put the zero crossing
        half a voxel out.
        """
        nx, ny, nz = self.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise VolumeBoundsUnusable(
                f"voxel index ({i}, {j}, {k}) is outside the {nx}x{ny}x{nz} volume"
            )
        return (
            self.bounds.min_x + (i + 0.5) * self.voxel_size,
            self.bounds.min_y + (j + 0.5) * self.voxel_size,
            self.bounds.min_z + (k + 0.5) * self.voxel_size,
        )

    def as_dict(self) -> dict[str, object]:
        nx, ny, nz = self.shape
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": FUSION_MODE,
            "sign_convention": SIGN_CONVENTION,
            "projection": f"the inverse of {DEPROJECTION_EXPRESSION}",
            "bounds": self.bounds.as_dict(),
            "realised_maximum": list(self.realised_maximum),
            "grid": {
                "voxel_size": self.voxel_size,
                "truncation": self.truncation,
                "truncation_in_voxels": self.truncation / self.voxel_size,
                "shape": [nx, ny, nz],
                "cells": self.cell_count,
                "cell_ceiling": self.max_voxels,
                "module_cell_ceiling": MAX_VOXELS,
                "owned_bytes": self.owned_bytes,
            },
            "depth": {
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
                "no_return_sample": NO_RETURN_SAMPLE,
            },
            "weight": {
                "policy": str(self.weight_policy),
                "max_weight": self.max_weight,
                "max_weight_ceiling": MAX_ACCUMULATED_WEIGHT,
            },
        }


# --- the evidence -------------------------------------------------------------


class IgnoredReason(enum.StrEnum):
    """Why a voxel took nothing from a frame it was offered to.

    Declared in the order :meth:`TsdfVolume.integrate` tests them. A voxel
    observation that would qualify for two is counted once, under the first
    that applies, so the accounting stays a partition of
    ``frames x voxels`` rather than a tally that can double-count.
    """

    BEHIND_CAMERA = "behind_camera"
    OUTSIDE_IMAGE = "outside_image"
    NO_RETURN = "no_return"
    NON_FINITE_DISTANCE = "non_finite_distance"
    DEPTH_OUT_OF_RANGE = "depth_out_of_range"
    BEYOND_TRUNCATION = "beyond_truncation"


#: What each reason means, in one place so the human and JSON renderings of a
#: run cannot drift apart.
REASON_DETAIL: Final = MappingProxyType(
    {
        IgnoredReason.BEHIND_CAMERA: (
            "the voxel's centre carried back to a camera-frame z at or below zero, so it sits "
            "beside or behind the optical centre and has no projection at all"
        ),
        IgnoredReason.OUTSIDE_IMAGE: (
            f"the voxel projected outside the {FRAME_WIDTH}x{FRAME_HEIGHT} frame, so this "
            "camera saw nothing along that direction and has no evidence about the voxel"
        ),
        IgnoredReason.NO_RETURN: (
            f"the pixel the voxel projected to held the raw sample {NO_RETURN_SAMPLE}, which "
            "this repository reads as 'the sensor returned no depth here'. It is NOT a "
            "measurement of zero distance and is never fused as a surface at the camera"
        ),
        IgnoredReason.NON_FINITE_DISTANCE: (
            "raw * depth_scale was not finite, so no signed distance could be formed from it. "
            "With a finite positive scale and a uint16 sample this cannot arise; it is "
            "counted rather than assumed away"
        ),
        IgnoredReason.DEPTH_OUT_OF_RANGE: (
            "the observed distance fell outside the caller's accepted depth interval. The "
            "interval is the caller's statement about where this device's returns are "
            "trustworthy; a sample outside it is discarded rather than fused at low weight"
        ),
        IgnoredReason.BEYOND_TRUNCATION: (
            "the voxel sits further behind the observed surface than the truncation distance, "
            "so it is occluded by that surface and this frame carries no evidence about it. "
            "Values in front are clamped and fused; values this far behind are not, which is "
            "what keeps a near surface from erasing a far one"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class IgnoredVoxelObservations:
    """One reason, and how many voxel observations it accounted for."""

    reason: IgnoredReason
    count: int
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"reason": str(self.reason), "count": self.count, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class WeightSummary:
    """The accumulated weights actually present in the field.

    ``minimum`` and ``maximum`` are taken over OBSERVED voxels only - a voxel
    no frame reached has weight zero and including it would make every run
    report a minimum of zero and say nothing.
    """

    observed_voxels: int
    unobserved_voxels: int
    minimum: float | None
    maximum: float | None
    at_cap: int

    def as_dict(self) -> dict[str, object]:
        return {
            "observed_voxels": self.observed_voxels,
            "unobserved_voxels": self.unobserved_voxels,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "at_cap": self.at_cap,
            "measured_over": "voxels with a non-zero accumulated weight",
        }


@dataclass(frozen=True, slots=True)
class TsdfEvidence:
    """What one field's integration actually did, in numbers a reader can audit.

    Immutable, and a snapshot: it is built when it is asked for, and integrating
    more frames afterwards does not change a snapshot already handed out.
    """

    config: TsdfConfig
    backend: str
    frames_offered: int
    frames_integrated: int
    voxel_updates: int
    ignored: tuple[IgnoredVoxelObservations, ...]
    weights: WeightSummary
    elapsed_seconds: float | None
    limitations: tuple[str, ...]

    @property
    def voxel_observations(self) -> int:
        """Every (frame, voxel) pair offered: the total the accounting partitions."""
        return self.voxel_updates + sum(entry.count for entry in self.ignored)

    @property
    def frames_without_updates(self) -> int:
        """Frames that were projected in full and changed nothing."""
        return self.frames_offered - self.frames_integrated

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.as_dict(),
            "backend": self.backend,
            "frames": {
                "offered": self.frames_offered,
                "integrated": self.frames_integrated,
                "without_updates": self.frames_without_updates,
                "integrated_means": (
                    "the frame was projected into every voxel and changed at least one of them"
                ),
            },
            "accounting": {
                "voxel_observations": self.voxel_observations,
                "voxel_updates": self.voxel_updates,
                "ignored": [entry.as_dict() for entry in self.ignored],
                "partitions": (
                    "voxel_updates plus every ignored count equals voxel_observations, which is "
                    "frames_offered times the volume's cells"
                ),
            },
            "weights": self.weights.as_dict(),
            "timing": {
                "elapsed_seconds": self.elapsed_seconds,
                "measured_by": (
                    "the caller, and passed in; this module reads no clock, so a run is "
                    "reproducible and a timing is never invented"
                ),
            },
            "dimensional_accuracy_claimed": False,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class WeightedInputSummary:
    """Every weighted SAMPLE offered to this field, by provenance.

    Input counts, not voxel observations: these are what the frames carried,
    summed over the frames, and they do not depend on where the volume is or
    which voxels the camera could see. A partition of the samples offered.
    """

    measured: int
    temporally_reconstructed: int
    spatially_excluded: int
    missing: int
    #: The fusion weight the frames carried, before any geometry: the
    #: :func:`math.fsum` of each frame's own correctly rounded total.
    offered_weight: float

    @property
    def samples(self) -> int:
        return (
            self.measured + self.temporally_reconstructed + self.spatially_excluded + self.missing
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "measured": self.measured,
            "temporally_reconstructed": self.temporally_reconstructed,
            "spatially_excluded": self.spatially_excluded,
            "missing": self.missing,
            "samples": self.samples,
            "offered_weight": self.offered_weight,
            "partitions": (
                "measured plus temporally_reconstructed plus spatially_excluded plus missing "
                "equals samples, which is frames_offered times the samples in one frame"
            ),
            "spatially_excluded_weight": 0.0,
            "spatially_excluded_weight_is": (
                "exactly zero, always: a single-frame estimate no independent view confirmed "
                "offers a no-return code at weight zero, which integration treats as absence"
            ),
        }


@dataclass(frozen=True, slots=True)
class WeightedIntegrationEvidence:
    """What :meth:`TsdfVolume.integrate_weighted` did, in numbers a reader can audit.

    Immutable, and a snapshot, exactly as :class:`TsdfEvidence` is. It is
    published BESIDE that record rather than instead of it: the base evidence
    still partitions every ``(frame, voxel)`` pair this volume was offered,
    including the weighted ones, and this record says what the provenance and
    the weights added on top.

    ``contribution`` is the total fusion weight that actually reached voxels -
    the sum, over accepted voxel observations, of the geometric weight the
    configured :class:`WeightPolicy` gives times the sample's own fusion
    weight. It is accumulated as a compensated (Neumaier) sum over per-chunk
    :func:`math.fsum` partials taken in flat-voxel-index order, which is a
    fixed order both backends visit voxels in - so it is deterministic, and
    identical on the reference and the accelerated path rather than merely
    close. It is NOT claimed to be the exactly rounded sum of every individual
    product.
    """

    config: TsdfConfig
    backend: str
    frames_offered: int
    frames_integrated: int
    inputs: WeightedInputSummary
    voxel_updates: int
    #: Voxel observations whose offered sample carried no weight at all -
    #: spatially excluded, or missing. They took the same path a raw no-return
    #: takes, so they are also counted under
    #: :attr:`IgnoredReason.NO_RETURN` in the base accounting, and they are
    #: named separately here because a zero weight and an absent return are the
    #: same arithmetic reached for two different reasons.
    zero_weight_observations: int
    contribution: float
    #: The same total, split by the provenance of the sample that produced each
    #: accepted voxel observation. One entry per name in
    #: :data:`~scanner3d.tsdf_input.PROVENANCE_NAMES`, always present, and the
    #: four add back to :attr:`contribution`.
    #:
    #: The two excluded provenances are here to be MEASURED at zero rather than
    #: asserted at zero. Nothing in this record states that an unconfirmed
    #: single-frame estimate contributed nothing; the accumulator does, because
    #: a weighting defect that let one through would land in this mapping and
    #: nowhere else. Each entry is accumulated exactly as
    #: :attr:`contribution` is, so it is deterministic and identical on both
    #: backends.
    contribution_by_provenance: Mapping[str, float]
    #: Weighted calls this volume refused. A refused call changed no value, no
    #: weight, no count and no frame total; it is recorded so that a run which
    #: was offered a malformed frame does not look like a run that was not.
    refusals: int
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.as_dict(),
            "backend": self.backend,
            "frames": {"offered": self.frames_offered, "integrated": self.frames_integrated},
            "inputs": self.inputs.as_dict(),
            "accounting": {
                "voxel_updates": self.voxel_updates,
                "zero_weight_observations": self.zero_weight_observations,
                "contribution": self.contribution,
                "contribution_is": (
                    "the geometric weight times the sample's fusion weight, summed over accepted "
                    "voxel observations as a Neumaier sum of per-chunk math.fsum partials in "
                    "flat-voxel-index order: deterministic, and identical on both backends"
                ),
                "contribution_by_provenance": dict(self.contribution_by_provenance),
                "contribution_by_provenance_is": (
                    "the same total split by the provenance of the sample behind each accepted "
                    "voxel observation, accumulated the same way. The excluded provenances are "
                    "measured at zero here rather than asserted at zero anywhere"
                ),
                "refusals": self.refusals,
            },
            "weighting": {
                "measured_weight": MEASURED_WEIGHT,
                "excluded_provenances": ["spatially_excluded", "missing"],
                "zero_weight_is": (
                    "identical to absence: a zero-weight sample updates no voxel and cannot turn "
                    "an unobserved voxel into an observed one"
                ),
            },
            "dimensional_accuracy_claimed": False,
            "physical_improvement_claimed": False,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class VoxelSample:
    """One voxel's stored value, for auditing the field directly."""

    index: tuple[int, int, int]
    centre: Point
    value: float
    weight: float

    @property
    def observed(self) -> bool:
        return self.weight > 0.0

    def distance(self, truncation: float) -> float:
        """The stored value back in model units, un-normalised by ``truncation``."""
        return self.value * truncation

    def as_dict(self) -> dict[str, object]:
        return {
            "index": list(self.index),
            "centre": list(self.centre),
            "value": self.value,
            "value_is": "the truncated signed distance divided by the truncation distance",
            "weight": self.weight,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class ZeroLevelSamples:
    """Points on the field's zero level. Evaluation geometry, not a mesh.

    A point is one of exactly two things, and both are places the field's own
    evidence puts the level set:

    * an EDGE CROSSING: the linear interpolation between two neighbouring voxel
      centres whose stored values have strictly OPPOSITE signs and whose
      weights both reach ``min_weight``; or
    * an OBSERVED EXACTLY-ZERO VOXEL: a voxel some frame did observe whose
      stored value is exactly zero, reported once at its own centre. The level
      set passes THROUGH that centre rather than between two of them, so there
      is nothing to interpolate and nothing is invented.

    A voxel no frame observed holds a zero that means "no evidence" and is
    neither: the same weight threshold excludes it from both rules.

    There is no connectivity between the points, no normal, no orientation and
    no surface: this is what a caller measures a field with, and explicitly not
    what a caller renders or exports as a model.
    """

    points: tuple[Point, ...]
    min_weight: float
    #: Every zero-level sample the field holds at this threshold: the sum of
    #: the two counts below, taken before anything was materialised, and the
    #: number ``limit`` is checked against.
    crossings_found: int
    edge_crossings: int
    exact_zero_voxels: int
    limit: int
    limitations: tuple[str, ...]

    @property
    def point_count(self) -> int:
        return len(self.points)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "point_count": self.point_count,
            "crossings_found": self.crossings_found,
            "crossings_found_is": (
                "every zero-level sample found at this threshold, edge_crossings plus "
                "exact_zero_voxels, counted before any point was materialised"
            ),
            "edge_crossings": self.edge_crossings,
            "exact_zero_voxels": self.exact_zero_voxels,
            "min_weight": self.min_weight,
            "limit": self.limit,
            "connectivity": "none: these are unconnected samples of the zero level, not a mesh",
            "limitations": list(self.limitations),
        }


# --- backend selection --------------------------------------------------------


def _probe_numpy() -> tuple[bool, str | None]:
    """``(usable, reason-if-not)``.

    A real import, not a spec lookup, so a numpy that is installed but broken -
    a missing shared library, an ABI mismatch - is reported the same way as one
    that is missing outright, naming why it is not usable.
    """
    try:
        importlib.import_module("numpy")
    except ImportError as exc:
        return False, str(exc)
    return True, None


def resolve_backend(name: str) -> str:
    """The backend a request resolves to, or a refusal saying why not.

    Public because the same three names mean the same three things to anything
    that reads this field back out - :mod:`scanner3d.tsdf_surface` resolves an
    explicit request through this function rather than growing a second copy of
    the rule that could disagree about what ``"auto"`` does here.

    Deliberately not :func:`scanner3d.compute.select_backend`: that module
    selects an implementation of point queries - nearest neighbours, rigid
    fits - and its ``"scipy"`` backend accelerates a KD-tree that this module
    never builds. Sharing the name would mean a caller asking for ``"scipy"``
    here silently getting something else. The vocabulary is the same on purpose;
    the set of names is smaller because the set of operations is.
    """
    if name == AUTO_BACKEND:
        usable, _ = _probe_numpy()
        return "numpy" if usable else "python"
    if name not in KNOWN_BACKENDS:
        raise TsdfBackendUnknown(
            f"backend {name!r} is not one of {(AUTO_BACKEND, *KNOWN_BACKENDS)}. There is no "
            "GPU, ROCm, HIP, OpenCL or Vulkan backend here, reserved or otherwise"
        )
    if name == "numpy":
        usable, reason = _probe_numpy()
        if not usable:
            raise TsdfBackendUnavailable(
                f"backend 'numpy' was requested explicitly but is not usable here: {reason}. "
                "Explicit selection is honoured exactly, so this raises rather than silently "
                f"using a different backend - ask for {AUTO_BACKEND!r} to accept a fallback"
            )
    return name


# --- the field ----------------------------------------------------------------


def _accumulate(total: float, error: float, value: float) -> tuple[float, float]:
    """One Neumaier step: ``total + value``, carrying the lost low bits in ``error``.

    A plain running float sum over a long capture loses exactly the small
    addends it is being kept to count - a fusion weight of 0.3 added to a total
    of ten million disappears - and a list of every addend would grow without
    bound over a sweep. The compensated form is neither: constant memory, and
    the compensation term makes the result independent of how large the running
    total has become. It is deterministic for a given sequence of addends,
    which is what lets both backends publish the identical number.
    """
    updated = total + value
    if abs(total) >= abs(value):
        error += (total - updated) + value
    else:
        error += (value - updated) + total
    return updated, error


class TsdfVolume:
    """A truncated signed-distance field, and the frames folded into it so far.

    Mutable on purpose, and in exactly one way: :meth:`integrate` folds frames
    in and nothing else changes it. The two arrays it owns are allocated once,
    from :attr:`TsdfConfig.cell_count`, and never grow - which is the property
    that makes fusion affordable over a long capture, and the reason the
    ceiling is checked before they are allocated rather than after.

    Everything a caller reads back out - :meth:`evidence`,
    :meth:`zero_level_samples`, :meth:`sample_at` - is an immutable snapshot
    built when it is asked for. Integrating more frames afterwards does not
    reach back into one already handed out.
    """

    __slots__ = (
        "_backend",
        "_config",
        "_contribution",
        "_contribution_by",
        "_contribution_error",
        "_frames_integrated",
        "_frames_offered",
        "_ignored",
        "_offered_weight",
        "_offered_weight_error",
        "_tsdf",
        "_updates",
        "_weight",
        "_weighted_frames_integrated",
        "_weighted_frames_offered",
        "_weighted_inputs",
        "_weighted_refusals",
        "_weighted_updates",
        "_zero_weight",
    )

    def __init__(self, config: TsdfConfig, *, backend: str = AUTO_BACKEND) -> None:
        """Allocate an empty field, or refuse before allocating anything."""
        resolved = resolve_backend(backend)
        cells = config.cell_count
        if resolved == "python" and cells > PYTHON_BACKEND_MAX_VOXELS:
            raise VolumeTooLarge(
                f"a {cells}-voxel volume is past the {PYTHON_BACKEND_MAX_VOXELS} ceiling that "
                "binds the pure-Python backend, which touches every voxel once per frame in "
                "interpreted arithmetic; nothing was allocated. This ceiling is a wall-clock "
                "bound rather than a memory one, and it is never widened at run time - "
                "install numpy, or reduce the volume"
            )
        self._config = config
        self._backend = resolved
        # Two flat float32 arrays in C order, index (i * ny + j) * nz + k. Flat
        # because every pass over the field below is a flat-index pass, and
        # float32 because the field resolves far below the voxel size at half
        # the memory.
        self._tsdf: array.array[float] = array.array("f", [0.0]) * cells
        self._weight: array.array[float] = array.array("f", [0.0]) * cells
        if self._tsdf.itemsize != VALUE_BYTES:  # pragma: no cover - not a real platform
            raise TsdfNotAllowed(
                f"this platform's array('f') item is {self._tsdf.itemsize} bytes, not "
                f"{VALUE_BYTES}; the accelerated backend views that buffer as float32 and "
                "would read the wrong voxels"
            )
        self._frames_offered = 0
        self._frames_integrated = 0
        self._updates = 0
        self._ignored: dict[IgnoredReason, int] = dict.fromkeys(IgnoredReason, 0)
        # The weighted accounting, kept beside the base one rather than folded
        # into it: the base counters partition every (frame, voxel) pair this
        # volume was ever offered, weighted or not, and these say what the
        # provenance and the weights added on top of that.
        self._weighted_frames_offered = 0
        self._weighted_frames_integrated = 0
        self._weighted_updates = 0
        self._weighted_refusals = 0
        self._zero_weight = 0
        self._weighted_inputs: dict[str, int] = {
            "measured": 0,
            "temporally_reconstructed": 0,
            "spatially_excluded": 0,
            "missing": 0,
        }
        # Two Neumaier accumulators. A running float sum of a long capture's
        # weights loses the small addends it is meant to be counting; a list of
        # every addend would grow without bound. The compensated form is
        # neither, and - because both backends add the same per-chunk partials
        # in the same order - it produces the identical float on each.
        self._contribution = 0.0
        self._contribution_error = 0.0
        # The same accumulator, once per provenance, so the split is measured
        # in the fusion loop rather than reasoned about afterwards. Fixed at
        # four entries, whatever a capture contains.
        self._contribution_by: dict[str, tuple[float, float]] = dict.fromkeys(
            PROVENANCE_NAMES, (0.0, 0.0)
        )
        self._offered_weight = 0.0
        self._offered_weight_error = 0.0

    @property
    def config(self) -> TsdfConfig:
        return self._config

    @property
    def backend(self) -> str:
        """The backend this volume resolved to, never the name that was asked for."""
        return self._backend

    @property
    def frames_offered(self) -> int:
        return self._frames_offered

    @property
    def frames_integrated(self) -> int:
        return self._frames_integrated

    @property
    def voxel_updates(self) -> int:
        return self._updates

    # -- reading the field itself ----------------------------------------------

    @property
    def values(self) -> memoryview:
        """A READ-ONLY float32 view of the stored running means. No copy.

        Flat, C order, index ``(i * ny + j) * nz + k`` - the volume's own
        layout, not a reshaped one, so a consumer that wants a grid says so
        itself and cannot be handed a differently-ordered buffer by accident.

        A ``memoryview`` rather than a list, a tuple or an ``array`` copy for
        two reasons that are both about the ceiling this module already
        enforces. At :data:`MAX_VOXELS` the field is 16.7 million voxels, and
        turning it into Python floats would cost around 400 MB of objects to
        say what 64 MB of buffer already says; and a copy, or the live
        ``array`` itself, would let a caller write to the field behind
        :meth:`integrate`'s back. ``toreadonly`` makes that a ``TypeError`` at
        the point of the write rather than a corrupted scan later.

        The view aliases the live buffer, so a later :meth:`integrate` is
        visible through it. That is the opposite of what :meth:`evidence` and
        :meth:`zero_level_samples` do, and it is why this is a raw view named
        after the storage rather than a snapshot named after a result.
        """
        return memoryview(self._tsdf).toreadonly()

    @property
    def weights(self) -> memoryview:
        """A READ-ONLY float32 view of the accumulated weights, laid out as
        :attr:`values` is and aliasing the live buffer the same way.

        Paired with :attr:`values` on purpose: a value without its weight
        cannot be told apart from the stored zero of a voxel no frame ever
        observed, so anything reading the field back out needs both or it will
        invent surface out of emptiness.
        """
        return memoryview(self._weight).toreadonly()

    # -- integration -----------------------------------------------------------

    def integrate(
        self,
        frames: Sequence[bytes],
        poses: Sequence[Rigid],
        *,
        intrinsics: IntrinsicsRecord,
        depth_scale: float,
    ) -> None:
        """Fold ``frames`` into the field at ``poses``.

        Each pose maps ITS FRAME'S CAMERA coordinates into the model frame -
        the direction :func:`scanner3d.reconstruction_export.registered_points`
        applies to place a cloud. This method inverts it to carry voxels the
        other way; a caller that passes the inverse gets a scan that is wrong
        in a way that still looks like a scan, which is why the direction is
        stated here, in the module docstring, and in the config's report.

        Every refusal is raised **before a single voxel is touched**, so a
        rejected call leaves the field exactly as it found it: frame count
        against pose count, then the ceiling on frames, then - per frame, still
        before any projection - the intrinsics, the scale and the frame length,
        through the same :mod:`scanner3d.depth_metric` gates the production
        deprojection uses.
        """
        if len(frames) != len(poses):
            raise PoseCountMismatch(
                f"{len(frames)} frames were offered with {len(poses)} poses. Nothing was "
                "integrated: pairing them by position after truncating to the shorter would "
                "fuse frames at other frames' poses, which is not a partial scan but a wrong "
                "one"
            )
        if len(frames) > MAX_FRAMES_PER_CALL:
            raise FrameCountExceeded(
                f"{len(frames)} frames in one call is past the {MAX_FRAMES_PER_CALL} ceiling; "
                "nothing was integrated. The volume's memory does not grow with frame count, "
                "so split the sequence across several calls rather than widening the ceiling"
            )
        fx, fy, cx, cy = ensure_intrinsics_usable(intrinsics)
        scale = ensure_scale_usable(depth_scale)
        for frame in frames:
            ensure_frame_length(frame)

        for frame, pose in zip(frames, poses, strict=True):
            before = self._updates
            self._integrate_one(frame, pose, fx=fx, fy=fy, cx=cx, cy=cy, scale=scale)
            self._frames_offered += 1
            if self._updates > before:
                self._frames_integrated += 1

    def integrate_weighted(
        self,
        frames: Sequence[WeightedDepthFrame],
        poses: Sequence[Rigid],
        *,
        intrinsics: IntrinsicsRecord,
        depth_scale: float,
    ) -> None:
        """Fold provenance-weighted ``frames`` into the field at ``poses``.

        The same projective arithmetic :meth:`integrate` does, with one change
        and only one: the geometric weight the configured
        :class:`WeightPolicy` gives an accepted voxel observation is multiplied
        by the sample's own
        :meth:`~scanner3d.tsdf_input.WeightedDepthFrame.weight_at`. A measured
        sample weighs :data:`~scanner3d.tsdf_input.MEASURED_WEIGHT`, so

            ``volume.integrate_weighted([weighted_raw_frame(raw)], poses, ...)``

        produces byte-for-byte the field that

            ``volume.integrate([raw], poses, ...)``

        produces - the equivalence is not approximate, it is multiplication by
        one, and :mod:`tests.test_tsdf` compares the buffers directly.

        A sample carrying weight zero - a single-frame spatial estimate no
        independent view confirmed, or a sample nothing recovered at all -
        takes the SAME path a raw no-return takes: it updates no voxel, it is
        counted under :attr:`IgnoredReason.NO_RETURN` in the base accounting
        and under
        :attr:`WeightedIntegrationEvidence.zero_weight_observations` here, and
        it cannot turn an unobserved voxel into an observed one. Absence and
        zero weight are the same thing, deliberately.

        Poses run in the same direction :meth:`integrate` documents: each maps
        ITS FRAME'S camera coordinates into the model frame.

        Validated whole, before the field is touched. A frame that is not a
        :class:`~scanner3d.tsdf_input.WeightedDepthFrame`, a pose count that
        does not match, or more frames than the per-call ceiling leaves every
        value, every weight, every evidence count and every frame total exactly
        as it found them - and increments only the refusal counter, so a run
        that was offered something malformed does not look like a run that was
        not.
        """
        try:
            for position, frame in enumerate(frames):
                if not isinstance(frame, WeightedDepthFrame):
                    raise WeightedInputUnusable(
                        f"frame {position} is {frame!r}, and weighted integration takes a "
                        "scanner3d.tsdf_input.WeightedDepthFrame - the type that has bound its "
                        "own provenance to its own raw bytes and to the accepted completion "
                        "result behind it. Raw bytes carry no provenance and go through "
                        "integrate()"
                    )
            if len(frames) != len(poses):
                raise PoseCountMismatch(
                    f"{len(frames)} weighted frames were offered with {len(poses)} poses. "
                    "Nothing was integrated, for the same reason integrate() refuses it: a "
                    "silently mispaired fusion is not a partial scan but a wrong one"
                )
            if len(frames) > MAX_FRAMES_PER_CALL:
                raise FrameCountExceeded(
                    f"{len(frames)} weighted frames in one call is past the "
                    f"{MAX_FRAMES_PER_CALL} ceiling; nothing was integrated. Split the sequence "
                    "across several calls rather than widening the ceiling"
                )
            fx, fy, cx, cy = ensure_intrinsics_usable(intrinsics)
            scale = ensure_scale_usable(depth_scale)
        except ScannerError:
            # Every refusal, including the intrinsics and depth-scale gates
            # scanner3d.depth_metric owns. A refused call is still a call this
            # volume was offered, and a run that was handed something malformed
            # should not read like a run that was not.
            self._weighted_refusals += 1
            raise

        for frame, pose in zip(frames, poses, strict=True):
            before = self._updates
            self._integrate_one(
                frame.codes, pose, fx=fx, fy=fy, cx=cx, cy=cy, scale=scale, weighted=frame
            )
            self._frames_offered += 1
            self._weighted_frames_offered += 1
            summary = frame.summary
            self._weighted_inputs["measured"] += summary.measured
            self._weighted_inputs["temporally_reconstructed"] += summary.temporally_reconstructed
            self._weighted_inputs["spatially_excluded"] += summary.spatially_excluded
            self._weighted_inputs["missing"] += summary.missing
            self._offered_weight, self._offered_weight_error = _accumulate(
                self._offered_weight, self._offered_weight_error, summary.total_weight
            )
            if self._updates > before:
                self._frames_integrated += 1
                self._weighted_frames_integrated += 1

    def _integrate_one(
        self,
        frame: bytes,
        pose: Rigid,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        scale: float,
        weighted: WeightedDepthFrame | None = None,
    ) -> None:
        """One frame, at one pose, through the selected backend.

        ``weighted`` is the provenance behind ``frame``'s codes, or ``None``
        for the raw path. It is threaded rather than branched on here so that
        both paths run the identical arithmetic: the raw path is the weighted
        one with every fusion weight equal to
        :data:`~scanner3d.tsdf_input.MEASURED_WEIGHT`, and a multiplication by
        one is exact.
        """
        # The single most consequential line in this module: a pose places
        # CAMERA points into the MODEL frame, and a voxel is a model point, so
        # projecting it needs the other direction.
        to_camera = pose.inverse()
        if self._backend == "numpy":
            self._integrate_numpy(
                frame, to_camera, fx=fx, fy=fy, cx=cx, cy=cy, scale=scale, weighted=weighted
            )
        else:
            self._integrate_python(
                frame, to_camera, fx=fx, fy=fy, cx=cx, cy=cy, scale=scale, weighted=weighted
            )

    def _accumulate_by_provenance(self, applied: Sequence[Sequence[float]]) -> None:
        """Fold one chunk's applied weights into the per-provenance accumulators.

        ``applied`` is indexed by provenance byte, and each entry is that
        chunk's applied weights in flat-voxel-index order - the same order and
        the same per-chunk :func:`math.fsum` partials the total uses, which is
        what makes the split identical on both backends rather than merely
        close.
        """
        for code, values in enumerate(applied):
            if not values:
                continue
            name = PROVENANCE_NAMES[code]
            total, error = self._contribution_by[name]
            self._contribution_by[name] = _accumulate(total, error, math.fsum(values))

    def _integrate_python(
        self,
        frame: bytes,
        to_camera: Rigid,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        scale: float,
        weighted: WeightedDepthFrame | None = None,
    ) -> None:
        """The reference arithmetic, one voxel at a time and nothing hidden."""
        config = self._config
        nx, ny, nz = config.shape
        codes = array.array("H")
        codes.frombytes(frame)
        if sys.byteorder != "little":  # pragma: no cover - x86/ARM hosts are little-endian
            codes.byteswap()

        r = to_camera.rotation
        r00, r01, r02 = r[0]
        r10, r11, r12 = r[1]
        r20, r21, r22 = r[2]
        t0, t1, t2 = to_camera.translation

        tsdf = self._tsdf
        weight = self._weight
        ignored = self._ignored
        truncation = config.truncation
        min_depth = config.min_depth
        max_depth = config.max_depth
        max_weight = config.max_weight
        ramped = config.weight_policy is WeightPolicy.RAMPED_BEHIND
        voxel = config.voxel_size
        origin_x = config.bounds.min_x + 0.5 * voxel
        origin_y = config.bounds.min_y + 0.5 * voxel
        origin_z = config.bounds.min_z + 0.5 * voxel
        updates = 0
        # The provenance behind this frame's codes, or nothing at all. Read
        # once into locals so the inner loop is a byte index and a dict lookup
        # rather than two attribute walks per voxel.
        provenance = None if weighted is None else weighted.provenance
        reconstructed = {} if weighted is None else weighted.weights
        zero_weight = 0

        # This path builds no temporary arrays at all - it holds one voxel's
        # arithmetic in locals and writes it back before moving on - so the
        # chunking below bounds nothing about memory. It exists so that the
        # weight accounting adds up the SAME per-chunk partials, in the same
        # flat-index order, that :meth:`_integrate_numpy` does, which is what
        # makes the published contribution identical on both backends rather
        # than merely close.
        cells = nx * ny * nz
        for start in range(0, cells, CHUNK_VOXELS):
            stop = min(start + CHUNK_VOXELS, cells)
            applied: list[float] = []
            # One bucket per provenance byte, in the same flat order, so the
            # split is the loop's own arithmetic rather than a claim about it.
            applied_by: tuple[list[float], ...] = ([], [], [], [])
            for flat in range(start, stop):
                i, remainder = divmod(flat, ny * nz)
                j, k = divmod(remainder, nz)
                mx = origin_x + i * voxel
                my = origin_y + j * voxel
                mz = origin_z + k * voxel
                z = r20 * mx + r21 * my + r22 * mz + t2
                if z <= 0.0:
                    ignored[IgnoredReason.BEHIND_CAMERA] += 1
                    continue
                x = r00 * mx + r01 * my + r02 * mz + t0
                y = r10 * mx + r11 * my + r12 * mz + t1
                # Nearest pixel, rounded half up. floor(v + 0.5) and not
                # round(), whose half-to-even tie-breaking would disagree with
                # the vectorised path on exact half-pixels.
                u_index = math.floor(fx * x / z + cx + 0.5)
                v_index = math.floor(fy * y / z + cy + 0.5)
                if not (0 <= u_index < FRAME_WIDTH and 0 <= v_index < FRAME_HEIGHT):
                    ignored[IgnoredReason.OUTSIDE_IMAGE] += 1
                    continue
                pixel = v_index * FRAME_WIDTH + u_index
                code = codes[pixel]
                fusion = MEASURED_WEIGHT
                state = PROVENANCE_MEASURED
                if provenance is not None:
                    state = provenance[pixel]
                    if state == PROVENANCE_MEASURED:
                        fusion = MEASURED_WEIGHT
                    elif state == PROVENANCE_TEMPORALLY_RECONSTRUCTED:
                        fusion = reconstructed[pixel]
                    else:
                        fusion = 0.0
                if code == NO_RETURN_SAMPLE or fusion == 0.0:
                    # One branch for both, because they are one fact. A
                    # weighted frame offers a no-return code exactly where it
                    # offers no weight, and neither can update a voxel.
                    ignored[IgnoredReason.NO_RETURN] += 1
                    if provenance is not None:
                        zero_weight += 1
                    continue
                distance = code * scale
                if not math.isfinite(distance):  # pragma: no cover - unreachable, counted anyway
                    ignored[IgnoredReason.NON_FINITE_DISTANCE] += 1
                    continue
                if distance < min_depth or distance > max_depth:
                    ignored[IgnoredReason.DEPTH_OUT_OF_RANGE] += 1
                    continue
                signed = distance - z
                if signed <= -truncation:
                    ignored[IgnoredReason.BEYOND_TRUNCATION] += 1
                    continue
                normalised = signed / truncation
                if normalised > 1.0:
                    normalised = 1.0
                sample_weight = fusion
                if ramped and normalised < 0.0:
                    sample_weight = fusion * (1.0 + normalised)
                old_weight = weight[flat]
                total = old_weight + sample_weight
                tsdf[flat] = (tsdf[flat] * old_weight + normalised * sample_weight) / total
                weight[flat] = total if total < max_weight else max_weight
                updates += 1
                if provenance is not None:
                    applied.append(sample_weight)
                    applied_by[state].append(sample_weight)
            if applied:
                self._contribution, self._contribution_error = _accumulate(
                    self._contribution, self._contribution_error, math.fsum(applied)
                )
                self._accumulate_by_provenance(applied_by)
        self._updates += updates
        if provenance is not None:
            self._weighted_updates += updates
            self._zero_weight += zero_weight

    def _integrate_numpy(
        self,
        frame: bytes,
        to_camera: Rigid,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        scale: float,
        weighted: WeightedDepthFrame | None = None,
    ) -> None:
        """The same arithmetic, a bounded chunk of voxels at a time.

        Chunks are cut on the FLAT index, which is what makes the scatter at the
        end safe: every voxel appears in exactly one chunk exactly once, so
        ``field[indices] = values`` can never collide with itself and lose an
        update to numpy's last-writer-wins.
        """
        import numpy as np

        config = self._config
        nx, ny, nz = config.shape
        cells = nx * ny * nz
        codes = np.frombuffer(frame, dtype="<u2")
        tsdf = np.frombuffer(self._tsdf, dtype=np.float32)
        weight = np.frombuffer(self._weight, dtype=np.float32)

        r = to_camera.rotation
        t = to_camera.translation
        truncation = config.truncation
        voxel = config.voxel_size
        origin = (
            config.bounds.min_x + 0.5 * voxel,
            config.bounds.min_y + 0.5 * voxel,
            config.bounds.min_z + 0.5 * voxel,
        )
        ramped = config.weight_policy is WeightPolicy.RAMPED_BEHIND
        ignored = self._ignored
        plane = ny * nz
        updates = 0
        zero_weight = 0
        # One dense fusion weight per SAMPLE - not per voxel - built once for
        # the whole frame from the provenance bytes and the sparse
        # reconstruction weights. That is 2 MiB for a 640x400 frame, a fixed
        # cost beside the volume rather than a fraction of it, and it is what
        # lets the per-chunk arithmetic below stay a single indexed multiply.
        fusion_of_sample: FloatArray | None = None
        states_of_sample: ProvenanceArray | None = None
        if weighted is not None:
            states = np.frombuffer(weighted.provenance, dtype=np.uint8)
            states_of_sample = states
            fusion_of_sample = (states == PROVENANCE_MEASURED).astype(np.float64)
            fusion_of_sample *= MEASURED_WEIGHT
            if weighted.weights:
                count = len(weighted.weights)
                fusion_of_sample[
                    np.fromiter(weighted.weights.keys(), dtype=np.int64, count=count)
                ] = np.fromiter(weighted.weights.values(), dtype=np.float64, count=count)

        for start in range(0, cells, CHUNK_VOXELS):
            stop = min(start + CHUNK_VOXELS, cells)
            flat = np.arange(start, stop, dtype=np.int64)
            i, remainder = np.divmod(flat, plane)
            j, k = np.divmod(remainder, nz)
            mx = origin[0] + i * voxel
            my = origin[1] + j * voxel
            mz = origin[2] + k * voxel

            z = r[2][0] * mx + r[2][1] * my + r[2][2] * mz + t[2]
            in_front = z > 0.0
            ignored[IgnoredReason.BEHIND_CAMERA] += int(np.count_nonzero(~in_front))
            if not in_front.any():
                continue
            x = r[0][0] * mx + r[0][1] * my + r[0][2] * mz + t[0]
            y = r[1][0] * mx + r[1][1] * my + r[1][2] * mz + t[1]
            # Division is done against a z that is never zero, so no warning is
            # raised and no invalid value is produced for voxels already
            # excluded above; the mask, not the arithmetic, decides.
            safe_z = np.where(in_front, z, 1.0)
            u_index = np.floor(fx * x / safe_z + cx + 0.5).astype(np.int64)
            v_index = np.floor(fy * y / safe_z + cy + 0.5).astype(np.int64)
            inside = (
                in_front
                & (u_index >= 0)
                & (u_index < FRAME_WIDTH)
                & (v_index >= 0)
                & (v_index < FRAME_HEIGHT)
            )
            ignored[IgnoredReason.OUTSIDE_IMAGE] += int(np.count_nonzero(in_front & ~inside))
            if not inside.any():
                continue
            pixel = np.where(inside, v_index * FRAME_WIDTH + u_index, 0)
            code = codes[pixel]
            returned = inside & (code != NO_RETURN_SAMPLE)
            if fusion_of_sample is not None:
                # One branch for a no-return code and a zero weight, because
                # they are one fact: a weighted frame offers a no-return
                # exactly where it offers no weight.
                returned &= fusion_of_sample[pixel] > 0.0
            excluded = int(np.count_nonzero(inside & ~returned))
            ignored[IgnoredReason.NO_RETURN] += excluded
            if fusion_of_sample is not None:
                zero_weight += excluded

            distance = code.astype(np.float64) * scale
            finite = returned & np.isfinite(distance)
            ignored[IgnoredReason.NON_FINITE_DISTANCE] += int(np.count_nonzero(returned & ~finite))
            in_range = finite & (distance >= config.min_depth) & (distance <= config.max_depth)
            ignored[IgnoredReason.DEPTH_OUT_OF_RANGE] += int(np.count_nonzero(finite & ~in_range))

            signed = distance - z
            accepted = in_range & (signed > -truncation)
            ignored[IgnoredReason.BEYOND_TRUNCATION] += int(np.count_nonzero(in_range & ~accepted))
            if not accepted.any():
                continue

            indices = flat[accepted]
            normalised = np.minimum(signed[accepted] / truncation, 1.0)
            if fusion_of_sample is None:
                fusion = MEASURED_WEIGHT
            else:
                fusion = fusion_of_sample[pixel[accepted]]
            if ramped:
                sample_weight = fusion * np.where(normalised < 0.0, 1.0 + normalised, 1.0)
            else:
                sample_weight = fusion * np.ones_like(normalised)
            old_weight = weight[indices].astype(np.float64)
            total = old_weight + sample_weight
            merged = tsdf[indices].astype(np.float64) * old_weight + normalised * sample_weight
            tsdf[indices] = merged / total
            weight[indices] = np.minimum(total, config.max_weight)
            updates += int(indices.size)
            if states_of_sample is not None and indices.size:
                self._contribution, self._contribution_error = _accumulate(
                    self._contribution,
                    self._contribution_error,
                    math.fsum(sample_weight.tolist()),
                )
                chosen = states_of_sample[pixel[accepted]]
                self._accumulate_by_provenance(
                    tuple(
                        sample_weight[chosen == code].tolist()
                        for code in range(len(PROVENANCE_NAMES))
                    )
                )
        self._updates += updates
        if fusion_of_sample is not None:
            self._weighted_updates += updates
            self._zero_weight += zero_weight

    # -- reading the field back out --------------------------------------------

    def sample_at(self, i: int, j: int, k: int) -> VoxelSample:
        """One voxel's stored value and weight, for auditing the field directly."""
        nx, ny, nz = self._config.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise VolumeBoundsUnusable(
                f"voxel index ({i}, {j}, {k}) is outside the {nx}x{ny}x{nz} volume"
            )
        flat = (i * ny + j) * nz + k
        return VoxelSample(
            index=(i, j, k),
            centre=self._config.voxel_centre(i, j, k),
            value=self._tsdf[flat],
            weight=self._weight[flat],
        )

    def _weight_summary(self) -> WeightSummary:
        """Min, max and cap-count over observed voxels, in bounded passes."""
        cells = self._config.cell_count
        cap = self._config.max_weight
        observed = 0
        at_cap = 0
        minimum = math.inf
        maximum = -math.inf
        if self._backend == "numpy":
            import numpy as np

            weight = np.frombuffer(self._weight, dtype=np.float32)
            for start in range(0, cells, CHUNK_VOXELS):
                chunk = weight[start : start + CHUNK_VOXELS]
                seen = chunk[chunk > 0.0]
                if seen.size == 0:
                    continue
                observed += int(seen.size)
                minimum = min(minimum, float(seen.min()))
                maximum = max(maximum, float(seen.max()))
                at_cap += int(np.count_nonzero(seen >= cap))
        else:
            weights = self._weight
            for flat in range(cells):
                value = weights[flat]
                if value <= 0.0:
                    continue
                observed += 1
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                if value >= cap:
                    at_cap += 1
        if observed == 0:
            return WeightSummary(
                observed_voxels=0,
                unobserved_voxels=cells,
                minimum=None,
                maximum=None,
                at_cap=0,
            )
        return WeightSummary(
            observed_voxels=observed,
            unobserved_voxels=cells - observed,
            minimum=minimum,
            maximum=maximum,
            at_cap=at_cap,
        )

    def evidence(self, *, elapsed_seconds: float | None = None) -> TsdfEvidence:
        """An immutable snapshot of what this field's integration did.

        ``elapsed_seconds`` is the CALLER's measurement. This module reads no
        clock, so a run is reproducible and a timing is never invented; a caller
        that did not measure passes nothing and the report says so.
        """
        if elapsed_seconds is not None and (
            not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0
        ):
            raise TsdfNotAllowed(
                f"elapsed_seconds {elapsed_seconds!r} is not a finite non-negative duration"
            )
        weights = self._weight_summary()
        limitations = list(INHERENT_LIMITATIONS)
        if self._frames_offered == 0:
            limitations.append(
                "No frame has been integrated, so this field is empty and every voxel is "
                "unobserved. It is not a scan of anything."
            )
        elif weights.observed_voxels == 0:
            limitations.append(
                f"{self._frames_offered} frames were projected and NO voxel was observed. The "
                "volume and the poses do not agree about where the object is - the usual cause "
                "is bounds that do not contain the surface, or a pose direction that is "
                "inverted relative to the one this module documents."
            )
        without = self._frames_offered - self._frames_integrated
        if without:
            limitations.append(
                f"{without} of {self._frames_offered} frames changed no voxel at all. They were "
                "projected in full, so the volume simply held nothing those cameras could see."
            )
        if weights.at_cap:
            limitations.append(
                f"{weights.at_cap} voxels reached the {self._config.max_weight} weight cap, so "
                "their running mean has become an exponential forget rather than an all-time "
                "average: the newest frames weigh more there than the earliest ones."
            )
        return TsdfEvidence(
            config=self._config,
            backend=self._backend,
            frames_offered=self._frames_offered,
            frames_integrated=self._frames_integrated,
            voxel_updates=self._updates,
            ignored=tuple(
                IgnoredVoxelObservations(
                    reason=reason, count=self._ignored[reason], detail=REASON_DETAIL[reason]
                )
                for reason in IgnoredReason
            ),
            weights=weights,
            elapsed_seconds=elapsed_seconds,
            limitations=tuple(limitations),
        )

    @property
    def weighted_frames_offered(self) -> int:
        """Frames offered to :meth:`integrate_weighted`. A subset of :attr:`frames_offered`."""
        return self._weighted_frames_offered

    @property
    def weighted_refusals(self) -> int:
        """Weighted calls this volume refused. A refusal changed nothing else."""
        return self._weighted_refusals

    def weighted_evidence(self) -> WeightedIntegrationEvidence:
        """An immutable snapshot of what the provenance and the weights did here.

        Published beside :meth:`evidence` rather than instead of it. It is
        meaningful on a volume that has been offered no weighted frame at all -
        every count is zero, and the limitations say so - so a caller can ask
        without first knowing which path a field was built through.
        """
        inputs = WeightedInputSummary(
            measured=self._weighted_inputs["measured"],
            temporally_reconstructed=self._weighted_inputs["temporally_reconstructed"],
            spatially_excluded=self._weighted_inputs["spatially_excluded"],
            missing=self._weighted_inputs["missing"],
            offered_weight=self._offered_weight + self._offered_weight_error,
        )
        limitations = [
            "Synthetic and offline. No weighting here has been shown to improve a scan of a "
            "real object on the attached device.",
            "A fusion weight is evidence about PROVENANCE and not about accuracy: a "
            "reconstruction at high confidence is still a reconstruction, and the field records "
            "how much it was allowed to count for rather than claiming it was right.",
        ]
        if self._weighted_frames_offered == 0:
            limitations.append(
                "No weighted frame has been offered to this volume, so every count below is "
                "zero. If the field holds anything it came through the raw integrate() path, "
                "where every returned sample weighs the same."
            )
        if inputs.spatially_excluded:
            limitations.append(
                f"{inputs.spatially_excluded} sample(s) carried an accepted single-frame spatial "
                "estimate that no independent view confirmed. Every one of them contributed "
                "exactly zero: they are counted so the field's silence about them is visible, "
                "and not because they were fused at a low weight."
            )
        if self._weighted_frames_offered and not inputs.temporally_reconstructed:
            limitations.append(
                "No sample offered here was temporally reconstructed, so this field is exactly "
                "the field the same raw frames produce. The weighting changed nothing, which is "
                "a result and not an error."
            )
        return WeightedIntegrationEvidence(
            config=self._config,
            backend=self._backend,
            frames_offered=self._weighted_frames_offered,
            frames_integrated=self._weighted_frames_integrated,
            inputs=inputs,
            voxel_updates=self._weighted_updates,
            zero_weight_observations=self._zero_weight,
            contribution=self._contribution + self._contribution_error,
            contribution_by_provenance=MappingProxyType(
                {name: total + error for name, (total, error) in self._contribution_by.items()}
            ),
            refusals=self._weighted_refusals,
            limitations=tuple(limitations),
        )

    def zero_level_samples(
        self,
        *,
        min_weight: float = 1.0,
        limit: int = MAX_ZERO_SAMPLES,
    ) -> ZeroLevelSamples:
        """Points where this field's own evidence puts the zero level.

        Two kinds, and the level set is only complete with both:

        * an EDGE CROSSING between two neighbouring voxels whose stored values
          have strictly opposite signs and whose weights BOTH reach
          ``min_weight``; and
        * an OBSERVED EXACTLY-ZERO VOXEL, whose weight reaches ``min_weight``
          and whose stored value is exactly zero. That value is not a missing
          sign to be skipped: it is the level set passing through the voxel
          centre instead of between two centres, so the centre itself is
          reported, once. Dropping it erases any surface that happens to lie on
          the grid - a plane aligned with the voxel centres has no strictly
          opposite-sign pair anywhere along it, and would vanish entirely.

        What is never a point is the OTHER zero: a voxel no frame observed
        holds a stored zero that means "no evidence", and the weight threshold
        excludes it from both rules. Pairing against it, or reporting it, would
        invent surface out of emptiness - the precise failure that makes an
        unbounded field look like a closed object.

        Each kind is counted in one bounded pass and the ceiling is enforced
        BEFORE any point is built, so a refusal costs the passes rather than
        the memory it is refusing. The observed exact zeros come first in flat
        index order, then the edge crossings with the axes walked x, then y,
        then z, each in flat index order, so the result is deterministic and
        comparable between backends.
        """
        if not math.isfinite(min_weight) or min_weight <= 0.0:
            raise TsdfNotAllowed(
                f"min_weight {min_weight!r} must be finite and positive; a threshold of zero "
                "would pair against voxels no frame ever observed"
            )
        if not 0 < limit <= MAX_ZERO_SAMPLES:
            raise ZeroLevelCeilingExceeded(
                f"limit {limit} is outside (0, {MAX_ZERO_SAMPLES}]; the module ceiling is "
                "never widened at run time"
            )
        edges = self._count_crossings(min_weight)
        exact = self._count_exact_zeros(min_weight)
        found = edges + exact
        if found > limit:
            raise ZeroLevelCeilingExceeded(
                f"{found} zero-level samples exist ({edges} edge crossings and {exact} "
                f"observed exactly-zero voxels) and the limit is {limit}; nothing was "
                "materialised. Coarsen the volume or raise min_weight rather than the ceiling "
                f"- the module ceiling is {MAX_ZERO_SAMPLES} and a caller may only lower it"
            )
        points = self._build_exact_zeros(min_weight) + self._build_crossings(min_weight)
        limitations = [
            "These points sample the zero level; they are NOT a mesh and carry no "
            "connectivity, normals or orientation.",
            "Only crossings between two voxels that BOTH reach the weight threshold are "
            "reported, so the surface is silent wherever the scan was thin rather than "
            "guessing across it.",
            "A voxel that reaches the weight threshold and holds exactly zero is reported "
            "once, at its own centre, with no interpolation: the level set passes through "
            "that centre. The zero of a voxel no frame observed is not that and is excluded "
            "by the same threshold, so an unobserved volume still yields nothing at all.",
            "The interpolation is linear between two STORED values, and a value clamped at "
            "the truncation distance interpolates as though the surface really were that far "
            "away, which biases a crossing towards the clamped side. It is why the truncation "
            "band wants to be two to four voxels wide: with a wide band the voxels either "
            "side of a surface are unclamped and the bias does not arise.",
            *INHERENT_LIMITATIONS,
        ]
        return ZeroLevelSamples(
            points=points,
            min_weight=min_weight,
            crossings_found=found,
            edge_crossings=edges,
            exact_zero_voxels=exact,
            limit=limit,
            limitations=tuple(limitations),
        )

    def _axes(self) -> tuple[tuple[int, int, int], ...]:
        """``(stride, axis, span)`` for x, y and z: the neighbour one step on."""
        nx, ny, nz = self._config.shape
        return ((ny * nz, 0, nx), (nz, 1, ny), (1, 2, nz))

    def _count_crossings(self, min_weight: float) -> int:
        """How many opposite-sign, sufficiently observed neighbour pairs exist."""
        if self._backend == "numpy":
            return self._count_crossings_numpy(min_weight)
        return sum(1 for _ in self._walk_crossings_python(min_weight))

    def _build_crossings(self, min_weight: float) -> tuple[Point, ...]:
        if self._backend == "numpy":
            return self._build_crossings_numpy(min_weight)
        return tuple(
            self._interpolate(flat, stride, axis)
            for flat, stride, axis in self._walk_crossings_python(min_weight)
        )

    def _count_exact_zeros(self, min_weight: float) -> int:
        """How many sufficiently observed voxels hold exactly zero."""
        if self._backend == "numpy":
            return self._count_exact_zeros_numpy(min_weight)
        return sum(1 for _ in self._walk_exact_zeros_python(min_weight))

    def _build_exact_zeros(self, min_weight: float) -> tuple[Point, ...]:
        """Their centres, each once, in flat index order."""
        if self._backend == "numpy":
            return self._build_exact_zeros_numpy(min_weight)
        _, ny, nz = self._config.shape
        plane = ny * nz
        centres: list[Point] = []
        for flat in self._walk_exact_zeros_python(min_weight):
            i, remainder = divmod(flat, plane)
            j, k = divmod(remainder, nz)
            centres.append(self._config.voxel_centre(i, j, k))
        return tuple(centres)

    def _walk_exact_zeros_python(self, min_weight: float) -> Iterator[int]:
        """Every observed exactly-zero voxel, as a flat index, in flat order.

        The weight test comes first, and it is the whole difference between the
        two zeros this field can store. A voxel no frame observed holds zero
        because nothing was ever written to it; only a voxel some frame did
        observe can hold a zero that means "the surface is here".
        """
        tsdf = self._tsdf
        weight = self._weight
        for flat in range(self._config.cell_count):
            if weight[flat] >= min_weight and tsdf[flat] == 0.0:
                yield flat

    def _walk_crossings_python(self, min_weight: float) -> Iterator[tuple[int, int, int]]:
        """Every crossing as ``(flat, stride, axis)``, x then y then z, in flat order."""
        nx, ny, nz = self._config.shape
        tsdf = self._tsdf
        weight = self._weight
        plane = ny * nz
        cells = nx * ny * nz
        for stride, axis, span in self._axes():
            if span < 2:
                continue
            for flat in range(cells):
                if axis == 0:
                    if flat // plane >= nx - 1:
                        continue
                elif axis == 1:
                    if (flat // nz) % ny >= ny - 1:
                        continue
                elif flat % nz >= nz - 1:
                    continue
                other = flat + stride
                if weight[flat] < min_weight or weight[other] < min_weight:
                    continue
                here = tsdf[flat]
                there = tsdf[other]
                if here * there >= 0.0:
                    continue
                yield flat, stride, axis

    def _interpolate(self, flat: int, stride: int, axis: int) -> Point:
        """Where the straight line between two voxel centres reaches zero."""
        _, ny, nz = self._config.shape
        i, remainder = divmod(flat, ny * nz)
        j, k = divmod(remainder, nz)
        here = self._tsdf[flat]
        there = self._tsdf[flat + stride]
        fraction = here / (here - there)
        centre = list(self._config.voxel_centre(i, j, k))
        centre[axis] += fraction * self._config.voxel_size
        return (centre[0], centre[1], centre[2])

    def _crossing_masks(
        self, min_weight: float, start: int, stop: int, axis: int, stride: int
    ) -> tuple[IndexArray, BoolArray, FloatArray, FloatArray]:
        """One chunk's flat indices, crossing mask and the two values it crossed.

        numpy only. The mask is already false wherever the voxel has no
        neighbour on this axis, so the caller never has to re-derive the volume
        boundary; the two value arrays are returned in full rather than masked
        so that the caller can select from them in the mask's own order.
        """
        import numpy as np

        nx, ny, nz = self._config.shape
        tsdf = np.frombuffer(self._tsdf, dtype=np.float32)
        weight = np.frombuffer(self._weight, dtype=np.float32)
        flat = np.arange(start, stop, dtype=np.int64)
        plane = ny * nz
        if axis == 0:
            has_neighbour = (flat // plane) < nx - 1
        elif axis == 1:
            has_neighbour = ((flat // nz) % ny) < ny - 1
        else:
            has_neighbour = (flat % nz) < nz - 1
        other = np.where(has_neighbour, flat + stride, flat)
        here = tsdf[flat].astype(np.float64)
        there = tsdf[other].astype(np.float64)
        heavy = has_neighbour & (weight[flat] >= min_weight) & (weight[other] >= min_weight)
        crossing = heavy & (here * there < 0.0)
        return flat, crossing, here, there

    def _exact_zero_mask(
        self, min_weight: float, start: int, stop: int
    ) -> tuple[IndexArray, BoolArray]:
        """One chunk's flat indices and its observed-exactly-zero mask. numpy only.

        The weight test is half the mask for the same reason it is the first
        test on the interpreted path: an unobserved voxel holds a zero that is
        the absence of evidence, and only a voxel some frame observed can hold
        one that is the level set.
        """
        import numpy as np

        tsdf = np.frombuffer(self._tsdf, dtype=np.float32)[start:stop]
        weight = np.frombuffer(self._weight, dtype=np.float32)[start:stop]
        flat = np.arange(start, stop, dtype=np.int64)
        observed_zero = (weight >= min_weight) & (tsdf == 0.0)
        return flat, observed_zero

    def _count_exact_zeros_numpy(self, min_weight: float) -> int:
        import numpy as np

        cells = self._config.cell_count
        found = 0
        for start in range(0, cells, CHUNK_VOXELS):
            stop = min(start + CHUNK_VOXELS, cells)
            _, observed_zero = self._exact_zero_mask(min_weight, start, stop)
            found += int(np.count_nonzero(observed_zero))
        return found

    def _build_exact_zeros_numpy(self, min_weight: float) -> tuple[Point, ...]:
        import numpy as np

        config = self._config
        _, ny, nz = config.shape
        cells = config.cell_count
        voxel = config.voxel_size
        origin = (
            config.bounds.min_x + 0.5 * voxel,
            config.bounds.min_y + 0.5 * voxel,
            config.bounds.min_z + 0.5 * voxel,
        )
        plane = ny * nz
        points: list[Point] = []
        for start in range(0, cells, CHUNK_VOXELS):
            stop = min(start + CHUNK_VOXELS, cells)
            flat, observed_zero = self._exact_zero_mask(min_weight, start, stop)
            if not observed_zero.any():
                continue
            chosen = flat[observed_zero]
            i, remainder = np.divmod(chosen, plane)
            j, k = np.divmod(remainder, nz)
            px = origin[0] + i * voxel
            py = origin[1] + j * voxel
            pz = origin[2] + k * voxel
            points.extend(zip(px.tolist(), py.tolist(), pz.tolist(), strict=True))
        return tuple(points)

    def _count_crossings_numpy(self, min_weight: float) -> int:
        import numpy as np

        cells = self._config.cell_count
        found = 0
        for stride, axis, span in self._axes():
            if span < 2:
                continue
            for start in range(0, cells, CHUNK_VOXELS):
                stop = min(start + CHUNK_VOXELS, cells)
                _, crossing, _here, _there = self._crossing_masks(
                    min_weight, start, stop, axis, stride
                )
                found += int(np.count_nonzero(crossing))
        return found

    def _build_crossings_numpy(self, min_weight: float) -> tuple[Point, ...]:
        import numpy as np

        config = self._config
        _, ny, nz = config.shape
        cells = config.cell_count
        voxel = config.voxel_size
        origin = (
            config.bounds.min_x + 0.5 * voxel,
            config.bounds.min_y + 0.5 * voxel,
            config.bounds.min_z + 0.5 * voxel,
        )
        plane = ny * nz
        points: list[Point] = []
        for stride, axis, span in self._axes():
            if span < 2:
                continue
            for start in range(0, cells, CHUNK_VOXELS):
                stop = min(start + CHUNK_VOXELS, cells)
                flat, crossing, here, there = self._crossing_masks(
                    min_weight, start, stop, axis, stride
                )
                if not crossing.any():
                    continue
                chosen = flat[crossing]
                i, remainder = np.divmod(chosen, plane)
                j, k = np.divmod(remainder, nz)
                a = here[crossing]
                fraction = a / (a - there[crossing])
                px = origin[0] + i * voxel
                py = origin[1] + j * voxel
                pz = origin[2] + k * voxel
                step = fraction * voxel
                if axis == 0:
                    px = px + step
                elif axis == 1:
                    py = py + step
                else:
                    pz = pz + step
                points.extend(zip(px.tolist(), py.tolist(), pz.tolist(), strict=True))
        return tuple(points)
