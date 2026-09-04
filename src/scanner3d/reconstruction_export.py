"""One registered sweep, written out as geometry another program can open.

What this is for
----------------
Registration answers "where was the scanner for each frame". That answer lived
only in memory: the window drew a cloud from it and then threw it away, so the
four minutes of a capture ended with a folder of raw frames and a score, and
nothing anybody could open in a mesh editor. This module is the other half -
the registered points, a surface fitted to them, and a machine-readable record
of exactly what was produced - written into a fresh folder beside the frames it
came from.

Nothing here registers or scores anything new. It applies poses that were
already accepted, fuses the frames those poses place, calls
:mod:`scanner3d.pointcloud` and :mod:`scanner3d.mesh` to encode what comes out,
and is otherwise about integrity: what is on disk when this returns, and what
is on disk when it does not.

The final surface is FUSED, not filled
--------------------------------------
The mesh this writes is the zero level set of a truncated signed-distance field
that :mod:`scanner3d.tsdf` fuses from the ORIGINAL Y16 frames at the poses the
registration accepted, extracted by :mod:`scanner3d.tsdf_surface`. The frames
are the input, not the decimated clouds: the clouds were thinned to make
registration affordable, and meshing the thinned copy would throw away the
evidence the scanner actually returned at exactly the stage that decides what
the object looks like.

That replaces what this used to write, which was :func:`scanner3d.mesh.reconstruct`'s
occupancy boundary - a solid made of every cell any sample landed in, flood
filled from the grid border and marched over a padded grid. That surface is a
shrink-wrap: it CONTAINS the samples rather than passing through them, so it
reads about a voxel large in every direction, and its padded grid always closes,
so an object seen from one side came back with a back wall nobody observed. The
fused field has neither defect - it interpolates the crossing between voxel
centres, and it stops where the evidence stops - and this module never falls
back to the occupancy path. If the field cannot be built, or holds no crossing,
the export is REFUSED. A mesh that quietly changed method between runs would
make two exports of one capture incomparable without saying so.

The two artifacts therefore come from two different things, deliberately: the
registered cloud is still the union of the placed clouds, sample for sample,
because that is the raw evidence and fusing it would be an opinion; the mesh is
the fused field, because a surface is an opinion either way and this is the
better-founded one. The record says which is which.

Whole or absent, never half
---------------------------
Every file is written into a private staging directory that is hidden, created
by the process alone, and named so that nothing mistakes it for a result. Only
when all four files exist, and each one has been read back and matched against
the digest the record claims for it, is the directory RENAMED into place. A
rename of a directory over the empty directory this reserved for it is one
atomic operation, so an export is either entirely there or entirely not.

That matters more here than for a single file. A folder holding a point cloud,
a mesh and a manifest is read as a unit by whoever opens it, and a manifest
naming a mesh that is not there - or that is there but truncated - is worse than
no export at all, because it is evidence that is wrong rather than evidence that
is missing. So the record is checked against the bytes, not against the
intention: every artifact is re-read from the staging directory and hashed again
before anything is published.

A cancelled or failed run removes its staging directory and the empty name it
reserved. Neither leaves anything that looks like a finished export.

Never on top of an earlier result
---------------------------------
The output name is reserved with an exclusive ``mkdir``, which fails on an
existing directory AND on a symlink of that name rather than following it out of
the capture folder. Names are tried in order - ``reconstruction-0001``,
``-0002`` and so on - so a second reconstruction of the same capture is a second
folder and the first one is untouched. This is the same collision policy
:func:`scanner3d.config.allocate_capture_dir` applies to captures, for the same
reason: two runs' results mixed in one directory destroys both as evidence.

The upright export frame, and why it is a decision and not a fix
----------------------------------------------------------------
Depth coordinates come out of :mod:`scanner3d.depth_metric` in the camera's
optical frame: +x right along a sensor row, +y DOWN a sensor column, +z away
along the optical axis. That is the right frame for a camera and the wrong one
for a viewer, every one of which draws +y up - which is why a scan of an object
that was the right way up in front of the scanner opens upside down everywhere
else.

So this module applies exactly one rotation, :data:`UPRIGHT_FROM_OPTICAL`, a
half turn about the x axis. It is a proper rotation - determinant +1, so the
result is right-handed and no distance, angle or scale changes anywhere - and it
is applied ONCE to each artifact, at the seam between "what was measured" and
"what is shown and shipped". Raw frames, the deprojection, the poses, the fusion
and the registration scoring never see it: the field is built in the model frame
the poses are in, and the extracted mesh is turned upright afterwards, by
:func:`upright_mesh`, which moves the vertices and leaves the faces exactly as
the extraction wound them. Determinant +1 is the whole reason it may: a proper
rotation preserves orientation, so reversing the winding to "compensate" would
turn the surface inside out.

It is a PRESENTATION decision made by this repository, and the record says so in
those words. Nothing about the scanner, the file format or the factory
intrinsics declares an up direction; what the record states is that this
repository decided which way up to write the file, and which axis of the optical
frame each exported axis came from, so a consumer can undo it exactly.
"""

from __future__ import annotations

import contextlib
import errno as errno_mod
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from scanner3d import __version__
from scanner3d.depth_metric import (
    DepthConversionNotAllowed,
    ensure_frame_length,
    ensure_intrinsics_usable,
    ensure_scale_usable,
)
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Rigid
from scanner3d.mesh import (
    Mesh,
    MeshExport,
    MeshMethod,
    MeshProvenance,
    write_obj,
    write_ply_mesh,
)
from scanner3d.pointcloud import (
    CAMERA_OPTICAL_FRAME,
    MAX_POINTS,
    CalibrationProvenance,
    CloudProvenance,
    CoordinateConvention,
    EncodedCloud,
    PlyFormat,
    sha256_of,
    write_ply,
)
from scanner3d.preparation import DepthBand
from scanner3d.reconstruction_readiness import (
    ReconstructionReadiness,
    ReconstructionRefused,
    evaluate_readiness,
    fusion_evidence_sample,
    surface_evidence,
)
from scanner3d.tsdf import (
    AUTO_BACKEND,
    DEFAULT_MAX_WEIGHT,
    FIELD_LIMITATIONS,
    MAX_VOXELS,
    TsdfConfig,
    TsdfNotAllowed,
    TsdfVolume,
    VolumeBounds,
    WeightPolicy,
)
from scanner3d.tsdf_surface import TsdfSurfaceNotAllowed, extract_surface

# The one depth mode this repository captures is DEFINED there, and
# scanner3d.depth_metric - which this module already depends on for the gates
# above - reads it from the same place. A second copy of 512000 here would be a
# number free to drift from the capture that produces the bytes.
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from scanner3d.depth_metric import DepthUnitAssumption
    from scanner3d.geometry import Point
    from scanner3d.intrinsics import IntrinsicsRecord
    from scanner3d.pipeline import SweepReconstruction
    from scanner3d.tsdf import TsdfEvidence
    from scanner3d.tsdf_surface import SurfaceEvidence

#: Every record this module writes stamps this.
SCHEMA_VERSION: Final = 1

# --- the presentation frame ---------------------------------------------------

#: A half turn about the x axis, and the only transform this module applies.
#:
#: ``(x, y, z) -> (x, -y, -z)``. Determinant +1, so it is a rotation and not a
#: mirror: right-handed geometry stays right-handed, and no length, angle or
#: volume changes. See the module docstring for why it is applied at all.
UPRIGHT_FROM_OPTICAL: Final = Rigid.of(
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), (0.0, 0.0, 0.0)
)

#: What the exported files are in, stated in the record the PLY writer takes.
#: Every axis names what it is in the optical frame, so the rotation above can be
#: read back off the description alone.
UPRIGHT_FRAME: Final = CoordinateConvention(
    name="scanner3d_upright_presentation_frame",
    x_axis=(
        "right along a sensor row, in the direction of increasing column - the depth optical "
        "frame's +x, unchanged"
    ),
    y_axis=(
        "up: opposite the direction of increasing sensor row, so the top of the object as the "
        "scanner saw it is the top of this file - the depth optical frame's -y"
    ),
    z_axis=(
        "back towards the depth camera, opposite the optical axis - the depth optical frame's "
        "-z, so a point further from the scanner has a smaller z here"
    ),
    origin="the depth camera's projection centre as the factory intrinsics place it, unmoved",
    handedness="right-handed",
)

#: The mapping between the two frames, for a consumer that wants to undo it
#: rather than read prose. Machine-readable half of :data:`UPRIGHT_FRAME`.
AXIS_MAPPING: Final[Mapping[str, str]] = {
    "x": "+x of the depth camera optical frame",
    "y": "-y of the depth camera optical frame",
    "z": "-z of the depth camera optical frame",
}

#: Said in the file and in the record, because a consumer cannot tell a frame a
#: device declared from one a program chose, and this one was chosen here.
PRESENTATION_CAVEAT: Final = (
    "coordinate convention: these coordinates are NOT the frame the scanner or the deprojection "
    "produced. scanner3d applied one half turn about the x axis - (x, y, z) -> (x, -y, -z), a "
    "proper rotation of determinant +1 that changes no length, angle or scale - so that the "
    "object opens the right way up in viewers that draw +y as up. Neither the scanner, the "
    "factory intrinsics nor the PLY and OBJ formats declare an up direction; this one is a "
    "presentation decision made by this repository, stated here so it can be undone exactly"
)

#: How the surface was produced, for :attr:`~scanner3d.mesh.MeshProvenance.method`.
#: Names both halves and the rotation, so the one line a viewer shows says which
#: construction this is without anybody reading the caveats.
MESH_METHOD: Final = (
    "scanner3d.tsdf: projective truncated signed-distance fusion of the original Y16 frames at "
    "the accepted camera-to-model poses, running weighted mean; scanner3d.tsdf_surface: "
    "marching cubes on the zero level set of that field; then one half turn about x, applied to "
    "the vertices alone"
)

# --- the fused field ----------------------------------------------------------

#: Truncation, in voxels. Three, because the crossing is found BETWEEN
#: neighbouring voxel centres and a band narrower than two leaves both
#: neighbours of a crossing clamped or unobserved - the surface would disappear
#: rather than coarsen. Three keeps a voxel of margin on that and still carves a
#: band thin enough that two surfaces a few voxels apart do not fuse into one.
TRUNCATION_VOXELS: Final = 3.0

#: Voxels of empty field left outside the measured extent on every side. Larger
#: than :data:`TRUNCATION_VOXELS` on purpose: a voxel exactly on the extent of
#: the cloud has no neighbour on its far side to cross zero against, so a box cut
#: flush to the samples loses the outermost shell of the very surface it was
#: built to recover, and a box cut to exactly the truncation band leaves the
#: outermost crossing with nowhere to interpolate into. Four also guarantees
#: every axis holds at least nine voxels, so a cube always exists to march.
MARGIN_VOXELS: Final = 4

#: The accepted depth interval, in the depth scale's own unestablished unit.
#: Observations outside it take no part in the fusion.
#:
#: The SAME window the two deprojecting call sites in this repository already
#: keep - ``60.0 < z < 300.0`` in :func:`scanner3d.gui._cloud_from_bytes` and in
#: the CLI's own cloud path - so the field is fused from the same band of
#: samples the registered cloud is written from. Naming it a second time here
#: rather than importing one of theirs is deliberate: those two filter POINTS
#: after deprojection and this gates an OBSERVATION during projection, and a
#: single constant would imply the two are the same decision. Stated rather than
#: derived from the capture so that two exports of one capture cannot differ by
#: a range that drifted.
TSDF_MIN_DEPTH: Final = 60.0
TSDF_MAX_DEPTH: Final = 300.0

#: What one accepted observation of one voxel is worth.
#: :attr:`~scanner3d.tsdf.WeightPolicy.UNIFORM`, so the stored value is the plain
#: arithmetic mean of the observations: the variance-minimising estimator when
#: the observations are equally uncertain, and - the reason it is the choice for
#: the artifact an operator is asked to trust - the one whose behaviour is
#: readable when a result is wrong. The alternative down-weights the occluded
#: band behind a surface, which is a better estimator and a worse explanation,
#: and it also lets a voxel that exactly one frame observed accumulate less than
#: unit weight, which the gate below would then read as "not observed".
TSDF_WEIGHT_POLICY: Final = WeightPolicy.UNIFORM

#: The cap on one voxel's accumulated weight, and the module default. Past it
#: the running mean becomes an exponential forget rather than an all-time
#: average, which is what keeps a long capture responsive to the frames arriving
#: now.
TSDF_MAX_WEIGHT: Final = DEFAULT_MAX_WEIGHT

#: The evidence gate the extraction applies to all eight corners of a cube.
#: Exactly one frame's worth of weight under :data:`TSDF_WEIGHT_POLICY`, which
#: makes it the weakest gate that still means "a frame actually observed this":
#: it refuses to interpolate across a voxel nothing saw, and it refuses nothing
#: else. Anything higher would be this module deciding how many views an
#: operator's capture should have had.
TSDF_MIN_WEIGHT: Final = 1.0

# --- resolution ---------------------------------------------------------------

#: Each coarsening step multiplies the voxel size by this. An eighth of an
#: octave: fine enough that the chosen resolution lands close to the cell budget
#: rather than well under it, coarse enough that the walk below is a handful of
#: multiplications rather than a search.
COARSENING_RATIO: Final = 2.0 ** (1.0 / 8.0)

#: The most coarsening steps the walk may take before giving up and refusing the
#: export. 2**(200/8) is a factor of about 33 million, so reaching this bound
#: means the extent or the budget is not a shape this can mesh, not that the
#: walk was too short.
MAX_COARSENING_STEPS: Final = 200

#: A degenerate axis - a cloud that is flat, or a line - is treated as this
#: fraction of the longest axis when the first voxel estimate is computed. The
#: estimate only has to be close; the walk below is what makes the answer fit.
DEGENERATE_SPAN_FRACTION: Final = 1e-3

# --- what is written ----------------------------------------------------------

#: The output directory names, tried in order. Zero-padded so a shell lists them
#: in the order they were made.
OUTPUT_PREFIX: Final = "reconstruction-"
OUTPUT_ATTEMPTS: Final = 9999

#: The staging directory's prefix. Leading dot so it is hidden, and the word
#: "partial" so that one left behind by a killed process is not mistaken for a
#: result by whoever finds it.
STAGING_PREFIX: Final = ".reconstruction-partial-"

CLOUD_NAME: Final = "registered_cloud.ply"
MESH_PLY_NAME: Final = "mesh.ply"
MESH_OBJ_NAME: Final = "mesh.obj"
RECORD_NAME: Final = "reconstruction.json"

#: Stated as a field and in prose in every record this writes.
ACCURACY_DISCLAIMER: Final = (
    "no dimensional accuracy is claimed. The coordinates are raw depth samples through the "
    "factory intrinsics with an assumed unit, placed by a registration this repository scored "
    "but did not validate against a reference object, and the surface is the boundary of a "
    "voxel grid fitted to them. No extent, radius, thickness or volume read off these files is "
    "a measured length"
)


class ReconstructionExportNotAllowed(ScannerError):
    """An export was requested that this module refuses to attempt.

    Always raised before the output name is reserved and before the staging
    directory exists, so a refused export leaves the capture folder exactly as
    it found it.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class NoGeometryOffered(ReconstructionExportNotAllowed):
    """There were no frames, or every frame was empty, so there is nothing to write."""


class GeometryNotExportable(ReconstructionExportNotAllowed):
    """The registered geometry cannot be meshed or encoded as it stands.

    A cloud with more points than the writers' ceiling, and a cloud whose extent
    is a single coordinate, both end here: they are outcomes of the capture, and
    an export that quietly dropped points or invented a resolution to get past
    one would be reporting a scan that was never taken.
    """


class FrameCountMismatch(ReconstructionExportNotAllowed):
    """Raw frames, clouds and poses did not pair up one to one.

    Refused rather than zipped to the shortest. Pairing by position after
    truncating would fuse frames at other frames' poses, which is not a partial
    scan but a confidently wrong one, and it would publish it with a manifest
    that counted the frames it dropped.
    """


class RegistrationMismatch(ReconstructionExportNotAllowed):
    """The registration record does not describe the pose sequence being fused.

    An export takes the poses TWICE: once as the explicit sequence that places
    the clouds and the frames, and once inside the
    :class:`~scanner3d.pipeline.SweepReconstruction` whose score, limitations
    and frame count are serialised into the manifest. Nothing forces a caller to
    pass the same ones, and if they differ then the geometry on disk was built
    from one answer while the record vouches for another - a file that is wrong
    in the one way a reader cannot detect, because both halves look internally
    consistent.

    So they are compared element by element, before a byte of field is allocated
    and before a name is reserved, and any disagreement is this refusal rather
    than an export. Compared for EXACT equality: both sequences come from one
    registration in every call site this repository has, so a difference is a
    caller pairing up the wrong objects, never arithmetic drift.
    """


class CaptureInputUnusable(ReconstructionExportNotAllowed):
    """A raw frame, the intrinsics or the depth scale cannot be fused.

    Raised through the same :mod:`scanner3d.depth_metric` gates the production
    deprojection uses, and always before a field is allocated or a name is
    reserved. The underlying refusal travels as ``__cause__``: this module's
    contract is that every refusal it makes is a
    :class:`ReconstructionExportNotAllowed`, and a caller that wants to know
    which of the three inputs was wrong reads the cause or the message.
    """


class TsdfSurfaceUnavailable(ReconstructionExportNotAllowed):
    """The fused field or its surface could not be built, so nothing was written.

    A field past a ceiling, a backend that is not installed, poses and bounds
    that do not agree about where the object is, a field with no zero crossing
    in it - all of them end here, and NONE of them falls back to the occupancy
    reconstruction this module used to write. A quieter export that silently
    changed method would make two exports of one capture incomparable while
    looking identical from outside.
    """


class ArtifactVerificationFailed(ReconstructionExportNotAllowed):
    """A staged file did not match the digest or the length its record claims.

    Nothing is published on this path. The record is checked against the BYTES
    on disk rather than against what the encoder returned, because the whole
    value of a manifest is that a consumer can trust it without re-deriving it.
    """


class OutputDirectoryUnavailable(ReconstructionExportNotAllowed):
    """No fresh output name could be created inside the capture directory."""


# --- the registered points ----------------------------------------------------


def upright_pose(pose: Rigid) -> Rigid:
    """``pose``, then the presentation rotation. The ONE place the two combine.

    Callers that need the same points for something other than a file - the
    window's own 3D view - compose through here rather than rotating the result
    afterwards, so the geometry on screen and the geometry in the file are the
    same arithmetic and cannot drift apart.
    """
    return UPRIGHT_FROM_OPTICAL.compose(pose)


def registered_points(clouds: Sequence[Sequence[Point]], poses: Sequence[Rigid]) -> Iterator[Point]:
    """Every point of every frame, placed by its pose, in the upright frame.

    Streamed frame by frame, in the order the frames were given and the order
    the points sit in each frame, so two exports of one reconstruction are
    byte-identical files. Nothing is sampled, decimated or averaged here: this
    is the output the operator is asked to trust, and the live preview's bounds
    have no business in it.

    The pairing is checked before anything is yielded rather than inside the
    generator, so a caller that got the two lists out of step is told so when it
    asks rather than part-way through consuming an answer.
    """
    if len(clouds) != len(poses):
        raise FrameCountMismatch(
            f"{len(clouds)} clouds and {len(poses)} poses cannot be paired; a registered point "
            "set needs exactly one pose per frame"
        )
    return _placed(clouds, poses)


def _placed(clouds: Sequence[Sequence[Point]], poses: Sequence[Rigid]) -> Iterator[Point]:
    for cloud, pose in zip(clouds, poses, strict=True):
        yield from upright_pose(pose).apply_all(cloud)


def model_points(clouds: Sequence[Sequence[Point]], poses: Sequence[Rigid]) -> Iterator[Point]:
    """Every point of every frame, placed by its pose, in the MODEL frame.

    The same points :func:`registered_points` yields, one rotation earlier. This
    is the frame the poses are in and therefore the frame the field is fused in:
    :meth:`scanner3d.tsdf.TsdfVolume.integrate` inverts each pose to carry a
    voxel back into its camera, and a voxel measured in the presentation frame
    would be carried to the wrong place by exactly a half turn - a scan that
    still looks like a scan.

    Streamed, and checked before anything is yielded, for the reasons
    :func:`registered_points` gives about both.
    """
    if len(clouds) != len(poses):
        raise FrameCountMismatch(
            f"{len(clouds)} clouds and {len(poses)} poses cannot be paired; a model-frame point "
            "set needs exactly one pose per frame"
        )
    return _model_placed(clouds, poses)


def _model_placed(clouds: Sequence[Sequence[Point]], poses: Sequence[Rigid]) -> Iterator[Point]:
    for cloud, pose in zip(clouds, poses, strict=True):
        yield from pose.apply_all(cloud)


def upright_mesh(mesh: Mesh) -> Mesh:
    """``mesh`` in the presentation frame: vertices turned, winding untouched.

    :data:`UPRIGHT_FROM_OPTICAL` has determinant +1, so it preserves
    orientation: a triangle wound anticlockwise seen from outside is still wound
    anticlockwise seen from outside after it. Reversing the faces to
    "compensate" - which is right for a mirror and wrong for a rotation - would
    turn every normal in the file inward, and nothing downstream that draws a
    surface can tell that from a surface that was extracted inside out.

    Applied exactly ONCE, here, after extraction. The field, the poses and the
    frames stay in the model frame.
    """
    return replace(mesh, vertices=tuple(UPRIGHT_FROM_OPTICAL.apply_all(mesh.vertices)))


# --- resolution ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TsdfResolution:
    """The voxel size one export fused at, and how it was arrived at.

    Kept as a record rather than a bare float because "0.51" answers nothing on
    its own: what makes a resolution defensible is the extent it was derived
    from, the voxel budget it had to fit, and the fact that arriving at it
    allocated no field and attempted no fusion.

    :attr:`minimum` and :attr:`maximum` are the MEASURED extent of the placed
    clouds, before the margin. :meth:`bounds` is what the field actually covers.
    """

    voxel_size: float
    truncation: float
    margin: float
    voxel_budget: int
    predicted_voxels: int
    grid_shape: tuple[int, int, int]
    span: tuple[float, float, float]
    minimum: Point
    maximum: Point
    coarsening_steps: int

    @property
    def bounds(self) -> VolumeBounds:
        """The model-frame box the field covers: the measured extent plus margin."""
        return VolumeBounds.of(
            tuple(value - self.margin for value in self.minimum),  # type: ignore[arg-type]
            tuple(value + self.margin for value in self.maximum),  # type: ignore[arg-type]
        )

    @property
    def summary(self) -> str:
        nx, ny, nz = self.grid_shape
        return (
            f"voxel {self.voxel_size!r} over a measured span of {self.span[0]:.1f} x "
            f"{self.span[1]:.1f} x {self.span[2]:.1f} grown by {self.margin!r} on every side, "
            f"a {nx}x{ny}x{nz} field of {self.predicted_voxels} voxels against a "
            f"{self.voxel_budget}-voxel budget, after {self.coarsening_steps} coarsening step(s)"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "voxel_size": self.voxel_size,
            "truncation": self.truncation,
            "truncation_in_voxels": TRUNCATION_VOXELS,
            "margin": self.margin,
            "margin_in_voxels": MARGIN_VOXELS,
            "voxel_budget": self.voxel_budget,
            "module_max_voxels": MAX_VOXELS,
            "grid_shape": list(self.grid_shape),
            "voxels": self.predicted_voxels,
            "measured_span": list(self.span),
            "measured_minimum": list(self.minimum),
            "measured_maximum": list(self.maximum),
            "measured_in": "the model frame, from the clouds placed by their ORIGINAL poses",
            "coarsening_steps": self.coarsening_steps,
            "how": (
                "derived from the placed extent alone: a first voxel size that would fill the "
                "voxel budget exactly, then multiplied by 2**(1/8) until the margined field it "
                "implies fits inside the budget. Arithmetic only - no field is allocated and no "
                "fusion is attempted while choosing, so nothing here is a retry - and the shape "
                "is computed by the same expression scanner3d.tsdf.TsdfConfig.shape uses, so "
                "the prediction and the allocation cannot disagree"
            ),
        }


def _margined_shape(
    minimum: Point, maximum: Point, voxel: float
) -> tuple[tuple[int, int, int], float]:
    """``(shape, margin)`` for a candidate voxel size. Allocates nothing.

    Deliberately the SAME expression as :attr:`scanner3d.tsdf.TsdfConfig.shape` -
    ``max(1, ceil(extent / voxel))`` over the margined extent, on the same
    floats - rather than the closed form ``ceil(span / voxel) + 2 * margin``.
    The two agree in exact arithmetic and can differ by one cell per axis in
    binary floating point, and the direction that matters is the one where a
    prediction that cleared the budget allocates a field that does not.
    """
    margin = MARGIN_VOXELS * voxel
    counts = [
        max(1, math.ceil(((high + margin) - (low - margin)) / voxel))
        for low, high in zip(minimum, maximum, strict=True)
    ]
    return (counts[0], counts[1], counts[2]), margin


def choose_tsdf_resolution(
    points: Iterable[Point],
    *,
    voxel_budget: int = MAX_VOXELS,
) -> TsdfResolution:
    """The finest voxel size the budget allows for this MODEL-frame extent.

    Deterministic, and pure arithmetic: the same points and the same budget give
    the same answer, no field is allocated while deciding, and no fusion is
    attempted and discarded. The first estimate is the voxel size that would
    fill the budget exactly if the field needed no margin; the walk after it
    multiplies by :data:`COARSENING_RATIO` until the margined field really does
    fit. Both the estimate and the number of steps are reported, because a
    resolution nobody can re-derive is a number in a file rather than a
    parameter.

    ``points`` is streamed once, so a cloud is not held twice to be measured.

    ``voxel_budget`` may be lowered by a caller and never raised past
    :data:`scanner3d.tsdf.MAX_VOXELS`, matching every other bound in this
    repository: a ceiling that an argument can widen is not a ceiling.
    """
    if not 0 < voxel_budget <= MAX_VOXELS:
        raise GeometryNotExportable(
            f"voxel_budget {voxel_budget} is outside (0, {MAX_VOXELS}]; the TSDF module's "
            "ceiling is never widened at run time"
        )
    low = [math.inf, math.inf, math.inf]
    high = [-math.inf, -math.inf, -math.inf]
    seen = 0
    for point in points:
        for axis in range(3):
            value = point[axis]
            low[axis] = min(low[axis], value)
            high[axis] = max(high[axis], value)
        seen += 1
    if seen == 0:
        raise NoGeometryOffered("no registered points were offered, so there is no extent")
    minimum: Point = (low[0], low[1], low[2])
    maximum: Point = (high[0], high[1], high[2])
    span: tuple[float, float, float] = (
        high[0] - low[0],
        high[1] - low[1],
        high[2] - low[2],
    )
    if not all(math.isfinite(extent) for extent in span):
        raise GeometryNotExportable(
            f"the placed extent {span} is not finite, so no voxel size describes it"
        )
    longest = max(span)
    if longest <= 0.0:
        raise GeometryNotExportable(
            "every registered point sits at the same coordinate, so the cloud has no extent to "
            "fuse into; this is an outcome of the capture, not a resolution to be chosen around"
        )
    effective = [max(extent, longest * DEGENERATE_SPAN_FRACTION) for extent in span]
    voxel = (effective[0] * effective[1] * effective[2] / voxel_budget) ** (1.0 / 3.0)
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise GeometryNotExportable(
            f"no usable voxel size follows from a span of {span} and a {voxel_budget}-voxel budget"
        )
    steps = 0
    shape, margin = _margined_shape(minimum, maximum, voxel)
    while shape[0] * shape[1] * shape[2] > voxel_budget and steps < MAX_COARSENING_STEPS:
        voxel *= COARSENING_RATIO
        steps += 1
        shape, margin = _margined_shape(minimum, maximum, voxel)
    voxels = shape[0] * shape[1] * shape[2]
    if voxels > voxel_budget:
        raise GeometryNotExportable(
            f"a span of {span} still needs {voxels} voxels after {steps} coarsening steps, past "
            f"the {voxel_budget}-voxel budget; nothing was allocated"
        )
    return TsdfResolution(
        voxel_size=voxel,
        truncation=TRUNCATION_VOXELS * voxel,
        margin=margin,
        voxel_budget=voxel_budget,
        predicted_voxels=voxels,
        grid_shape=shape,
        span=span,
        minimum=minimum,
        maximum=maximum,
        coarsening_steps=steps,
    )


# --- provenance ---------------------------------------------------------------


def registered_provenance(
    *,
    unit: DepthUnitAssumption,
    depth_scale: float,
    calibration_source: str,
    frames_dir: Path,
    frames: int,
    points: int,
    points_per_frame_target: int | None,
    per_frame_filter: str | None = None,
    registration_limitations: Sequence[str] = (),
) -> CloudProvenance:
    """Everything the registered PLY cannot say about itself.

    Built here rather than at the call site for the reason
    :func:`scanner3d.cloud_export.cloud_provenance_for` gives about its own
    seam: a caller assembling one of these by hand can drop a caveat without
    noticing, and the caveats are the part of this file nobody can re-derive.

    The registration's own limitations travel verbatim. They are the strongest
    statement anyone has about whether these poses are right, and summarising
    them here would be this module deciding which of another module's findings
    a reader needs.
    """
    from scanner3d.cloud_export import units_label
    from scanner3d.depth_scale import UNIT_DISCLAIMER

    per_frame = (
        "every point of every frame as it was loaded"
        if points_per_frame_target is None
        else (
            f"each frame was decimated to about {points_per_frame_target} points before "
            "registration, and this file carries those points and no others; the raw frames in "
            "the capture folder still hold every sample the scanner returned"
        )
    )
    # Two separate sentences about two separate things, because a reader
    # deciding whether to trust a thin patch of this cloud needs to know which
    # of them thinned it: decimation drops samples with no regard for what they
    # are, and a rejection filter drops them for a reason that can be wrong.
    rejection = (
        "no per-frame rejection was applied: every point that survived the deprojection and "
        "the decimation is in this file, stray samples included"
        if per_frame_filter is None
        else per_frame_filter
    )
    limitations = [
        PRESENTATION_CAVEAT,
        (
            f"this cloud is {points} points from {frames} frames read out of {frames_dir}, each "
            "placed by the pose the registration accepted for it. The frames are UNION-ed, not "
            "fused: no averaging across frames and no duplicate removal was applied, so "
            "overlapping frames leave several samples of one surface about as far apart as the "
            "registration error between them"
        ),
        f"point selection: {per_frame}",
        f"per-frame rejection: {rejection}",
        ACCURACY_DISCLAIMER,
        *(f"registration: {limitation}" for limitation in registration_limitations),
    ]
    return CloudProvenance(
        units=units_label(unit, depth_scale),
        convention=UPRIGHT_FRAME,
        calibration=CalibrationProvenance(
            source=calibration_source,
            depth_scale=depth_scale,
            depth_scale_unit=UNIT_DISCLAIMER,
        ),
        limitations=tuple(limitations),
    )


# --- the result ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconstructionExport:
    """What one export produced: a directory, three artifacts and the record.

    Every path named here is a path that existed in full when this was returned,
    because the directory is only published after every file in it has been read
    back and matched against its digest.
    """

    directory: Path
    cloud: EncodedCloud
    mesh_ply: MeshExport
    mesh_obj: MeshExport
    resolution: TsdfResolution
    record: Mapping[str, object]
    #: The surface the two mesh files encode, kept rather than re-read.
    #:
    #: Here so a caller that wants to SHOW what it just exported does not have
    #: to parse a PLY back to do it. It is the same object both encoders were
    #: given - the extracted TSDF mesh, already turned upright - so a window
    #: drawing this and a viewer opening the file are looking at one surface and
    #: not at two that could differ.
    surface: Mesh
    #: What the fusion did, as :meth:`scanner3d.tsdf.TsdfVolume.evidence` measured
    #: it: the config it ran under, the backend, every voxel observation and why
    #: each ignored one was ignored, and the weights the field ended up holding.
    #: An immutable snapshot, so it cannot drift from the file it describes.
    integration: TsdfEvidence
    #: What the extraction examined, refused and produced, from the walk that
    #: built the mesh rather than inferred from the mesh afterwards. Immutable
    #: for the same reason.
    extraction: SurfaceEvidence

    @property
    def point_count(self) -> int:
        return self.cloud.point_count

    @property
    def vertex_count(self) -> int:
        return self.mesh_ply.vertex_count

    @property
    def face_count(self) -> int:
        return self.mesh_ply.face_count

    @property
    def watertight(self) -> bool:
        return self.mesh_ply.watertight

    @property
    def paths(self) -> tuple[Path, ...]:
        """The four files, in the order the record lists them."""
        return (
            self.directory / CLOUD_NAME,
            self.directory / MESH_PLY_NAME,
            self.directory / MESH_OBJ_NAME,
            self.directory / RECORD_NAME,
        )

    @property
    def summary(self) -> str:
        """One line naming the folder and the counts, for a status bar."""
        closed = "closed" if self.watertight else "not closed"
        return (
            f"{self.directory}: {self.point_count} points, {self.vertex_count} vertices, "
            f"{self.face_count} faces ({closed})"
        )


# --- writing ------------------------------------------------------------------


def _reserve_output_dir(frames_dir: Path) -> Path:
    """Create and return a fresh output directory inside ``frames_dir``.

    ``mkdir`` and nothing else, because an exclusive create is the only test of
    a free name that cannot be raced: it fails on an existing directory, on an
    existing file, and on a SYMLINK of that name rather than following it
    somewhere this was never asked to write. A taken name is stepped over, never
    adopted - a prior reconstruction is evidence, and a second run that wrote
    into it would destroy both.
    """
    for attempt in range(1, OUTPUT_ATTEMPTS + 1):
        candidate = frames_dir / f"{OUTPUT_PREFIX}{attempt:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        except OSError as exc:
            raise OutputDirectoryUnavailable(
                f"cannot create {candidate}: {exc}. Nothing was exported."
            ) from exc
        return candidate
    raise OutputDirectoryUnavailable(
        f"every one of {OUTPUT_ATTEMPTS} names under {frames_dir} starting {OUTPUT_PREFIX} is "
        "already taken, so nothing was exported. An earlier reconstruction is never written "
        "into or over."
    )


def _write_record(path: Path, payload: bytes) -> None:
    """Create ``path`` exclusively and write all of ``payload``, or leave nothing.

    The same discipline as :func:`scanner3d.pointcloud.write_ply`'s own writer -
    ``O_EXCL | O_NOFOLLOW``, a loop that resumes from what the kernel actually
    took, an fsync before this claims the file exists, and an unlink of anything
    partial. A record is small enough that one ``os.write`` almost always
    finishes it, and "almost always" is not a guarantee a manifest can be built
    on: a truncated JSON file is not a shorter record, it is an unparseable one.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    written = 0
    try:
        while written < len(payload):
            just_written = os.write(fd, payload[written:])
            if just_written <= 0:
                raise OSError(
                    errno_mod.EIO,
                    f"write to {path} made no progress after {written} of {len(payload)} bytes",
                )
            written += just_written
        os.fsync(fd)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    finally:
        os.close(fd)


def _verify(staging: Path, artifacts: Sequence[Mapping[str, object]]) -> None:
    """Re-read every staged artifact and match it against what the record claims.

    The point of the manifest is that a consumer can trust it without redoing
    the work, so it is checked against the BYTES rather than against the
    encoder's return value. A mismatch here means the file on disk is not the
    file the record describes, which is exactly the case that must never be
    published.
    """
    for artifact in artifacts:
        name = str(artifact["file"])
        expected_bytes = int(str(artifact["byte_count"]))
        expected_digest = str(artifact["sha256"])
        path = staging / name
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ArtifactVerificationFailed(
                f"{name} is named by the record and could not be read back: {exc}"
            ) from exc
        if len(payload) != expected_bytes:
            raise ArtifactVerificationFailed(
                f"{name} is {len(payload)} bytes on disk and the record claims "
                f"{expected_bytes}; nothing was published"
            )
        digest = sha256_of(payload)
        if digest != expected_digest:
            raise ArtifactVerificationFailed(
                f"{name} hashes to {digest} and the record claims {expected_digest}; nothing "
                "was published"
            )


def _artifact(
    name: str, role: str, directory: Path, *, byte_count: int, sha256: str
) -> dict[str, object]:
    """One entry of the manifest's artifact list.

    The length and the digest come from the encoder that produced the bytes, and
    :func:`_verify` then checks both against the file that was actually written.
    """
    return {
        "file": name,
        "role": role,
        "path": str(directory / name),
        "byte_count": byte_count,
        "sha256": sha256,
        "verified_after_write": True,
    }


def _json_safe(value: object, replaced: list[str], where: str = "") -> object:
    """``value`` with every non-finite float replaced by ``null``, paths recorded.

    JSON has no infinity and no NaN. Python's encoder writes ``Infinity``
    anyway, which is not JSON and which a consumer outside Python refuses, so a
    record that reached one would be unreadable exactly where it mattered most.

    The one field that can genuinely produce an infinity here is the
    registration's ``improvement_factor``, whose own contract already uses
    ``null`` for "this could not be measured" - so null is the right word rather
    than a substitution. Every replacement is listed by its path in the record,
    because a value silently changed on the way into a file is precisely the
    kind of thing this repository writes records to avoid.
    """
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        replaced.append(f"{where or '<root>'} ({value!r})")
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item, replaced, f"{where}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item, replaced, f"{where}[{index}]") for index, item in enumerate(value)]
    return value


def _ensure_registration_describes(
    poses: Sequence[Rigid], registration: SweepReconstruction
) -> None:
    """Refuse unless ``registration`` is a record OF ``poses``, pose by pose.

    Three claims, in the order that makes the message most useful: the frame
    count the manifest publishes, the length of the pose sequence the
    registration carries, and then every pose in it against the sequence that is
    actually about to be fused. The first difference is named by index, because
    "the record disagrees" over a hundred-frame sweep is a question about which
    frame and an export that would not answer it sends its operator through the
    whole capture by hand.

    Raises :class:`RegistrationMismatch` and touches nothing.
    """
    if registration.frames != len(poses):
        raise RegistrationMismatch(
            f"the registration record says {registration.frames} frames and {len(poses)} poses "
            "were offered to fuse; the record published beside this geometry would then count "
            "frames that were never placed. Nothing was exported"
        )
    if len(registration.poses) != len(poses):
        raise RegistrationMismatch(
            f"the registration carries {len(registration.poses)} poses and {len(poses)} were "
            "offered to fuse; one of the two is not the sequence this export would place, and "
            "the record cannot vouch for geometry it does not describe. Nothing was exported"
        )
    for index, (offered, recorded) in enumerate(zip(poses, registration.poses, strict=True)):
        if offered != recorded:
            raise RegistrationMismatch(
                f"pose {index} offered to fuse is not the pose the registration recorded for "
                f"that frame ({offered!r} against {recorded!r}); the surface would be built "
                "from one set of poses and published with a record of another. Nothing was "
                "exported"
            )


def export_reconstruction(
    frames_dir: Path,
    clouds: Sequence[Sequence[Point]],
    poses: Sequence[Rigid],
    *,
    raw_frames: Sequence[bytes],
    intrinsics: IntrinsicsRecord,
    registration: SweepReconstruction,
    unit: DepthUnitAssumption,
    depth_scale: float,
    calibration_source: str,
    points_per_frame_target: int | None = None,
    per_frame_filter: str | None = None,
    preparation: Mapping[str, object] | None = None,
    voxel_budget: int = MAX_VOXELS,
    max_points: int = MAX_POINTS,
    tsdf_backend: str = AUTO_BACKEND,
    on_progress: Callable[[str], None] | None = None,
    depth_band: DepthBand | None = None,
) -> ReconstructionExport:
    """Write one registered sweep into a fresh folder inside ``frames_dir``.

    ``depth_band`` is the interval the registration clouds were clipped to;
    when given, the field accepts exactly that interval of each frame's depths,
    so what was fused is what was registered. ``None`` keeps the sensor's own
    :data:`TSDF_MIN_DEPTH` to :data:`TSDF_MAX_DEPTH`.

    Produces a registered point-cloud PLY, one fused surface as a PLY mesh and
    as an OBJ, and a JSON record naming all three with their lengths and
    digests, the registration result, the fusion and extraction evidence, the
    parameters, the axis mapping and every accumulated limitation.

    ``raw_frames`` are the full-resolution ``Y16 `` frames to FUSE, one per
    cloud and pose, in the same order - not the decimated clouds. They are what
    the field is fused from; ``clouds`` are the placed samples the registered
    cloud is written from, and ``intrinsics`` and ``depth_scale`` are the exact
    calibration the frames were deprojected under. Each pose maps ITS FRAME'S
    CAMERA coordinates into the model frame, the direction
    :func:`registered_points` applies;
    :meth:`scanner3d.tsdf.TsdfVolume.integrate` inverts it.

    A caller may hand over frames it DERIVED in memory from the captured ones -
    :func:`scanner3d.preparation.mask_support_band` produces exactly that, a
    frame with a proved support band set to the sensor's own no-return value -
    provided it describes what it did in ``preparation``. This function writes
    nothing back to the capture either way: ``frames_dir`` is read for its
    frames and written to only by creating one new directory inside it.

    ``preparation`` is the structured, schema-versioned record of how those
    clouds and frames were produced, published under ``input.preparation``. It
    is what lets a reader tell the registration cloud's filtering from the
    fusion input's masking, which are different filters on different artifacts;
    ``per_frame_filter`` is the same account in prose.

    ``registration`` must be the record OF ``poses``: its frame count, the
    length of its own pose sequence and every pose in it are checked against the
    explicit sequence before anything is allocated, so the fusion, the placed
    cloud and the manifest cannot describe three different answers.

    ``voxel_budget`` and ``max_points`` may be LOWERED by a caller and never
    raised past the TSDF and cloud modules' own ceilings, which is the same rule
    every bound in this repository follows.

    ``on_progress`` is called between stages with one line each, and BETWEEN
    FRAMES during the fusion - the one stage whose length grows with the
    capture. It is also the cancellation point: a caller whose callback RAISES -
    which is how the window's Stop reaches a running job - unwinds through the
    cleanup below, so a cancelled export leaves no staging directory and no
    reserved name.

    Every refusal - counts that do not pair, a registration that is not the
    record of these poses, a frame of the wrong length, unusable intrinsics or
    scale, no geometry, a cloud past the writers' ceiling, an extent with no
    size, a field past a ceiling or a backend that is not installed, a field
    with no zero crossing - happens before the output name is reserved, and the
    last three of them happen before a byte of field is allocated. A refused
    export leaves the capture folder untouched, and none of them falls back to
    an occupancy mesh.
    """
    report = on_progress if on_progress is not None else _silent
    if not 0 < max_points <= MAX_POINTS:
        raise GeometryNotExportable(
            f"max_points {max_points} is outside (0, {MAX_POINTS}]; the cloud writer's ceiling "
            "is never widened at run time"
        )
    # Counts first, and all three against each other rather than pairwise
    # against the clouds: a raw frame list one short of the poses would
    # otherwise be discovered by ``zip`` inside the fusion, after the registered
    # cloud had been built and measured.
    if not len(raw_frames) == len(clouds) == len(poses):
        raise FrameCountMismatch(
            f"{len(raw_frames)} raw frames, {len(clouds)} clouds and {len(poses)} poses cannot "
            "be paired; an export needs exactly one raw frame and one pose per cloud, and "
            "truncating to the shortest would fuse frames at other frames' poses"
        )
    # Then that the record and the geometry are about the SAME poses. ``poses``
    # drives the fusion and the registered cloud; ``registration`` is what the
    # manifest publishes about them. Two arguments carrying one answer is a
    # convenience for the caller and a hazard for the reader, so it is checked
    # here rather than trusted - still before the extent is measured, before a
    # field exists and before a name is reserved.
    _ensure_registration_describes(poses, registration)
    # Then the calibration and every frame length, through the same
    # scanner3d.depth_metric gates the production deprojection uses - before the
    # extent is measured, before a field exists and before a name is reserved.
    try:
        ensure_intrinsics_usable(intrinsics)
        ensure_scale_usable(depth_scale)
        for index, frame in enumerate(raw_frames):
            try:
                ensure_frame_length(frame)
            except DepthConversionNotAllowed as exc:
                raise CaptureInputUnusable(f"raw frame {index}: {exc}") from exc
    except DepthConversionNotAllowed as exc:
        raise CaptureInputUnusable(
            f"the capture inputs cannot be fused: {exc}. Nothing was exported"
        ) from exc

    points = tuple(registered_points(clouds, poses))
    if not points:
        raise NoGeometryOffered(
            f"{len(clouds)} frames were offered and none of them carried a point, so there is "
            "nothing to export"
        )
    if len(points) > max_points:
        raise GeometryNotExportable(
            f"{len(points)} registered points is past the {max_points}-point ceiling the cloud "
            "writer enforces; nothing was written. Capture fewer frames, or decimate them "
            "further before registering"
        )
    report(f"  {len(points)} registered points from {len(clouds)} frames, in the upright frame")

    # The field is measured and built in the MODEL frame - the frame the poses
    # are in - and turned upright only after the surface has been extracted.
    resolution = choose_tsdf_resolution(model_points(clouds, poses), voxel_budget=voxel_budget)

    # BEFORE the fusion, not after it. Every gate that can be answered without a
    # volume is answered here - the capture's own tracking authority, whether
    # these poses describe a scan at all, and whether the occupied surface was
    # ever seen from more than one direction - and a capture that fails any of
    # them is refused now rather than after several minutes of projecting
    # full-resolution frames into a field nobody may keep. All of them are
    # evaluated whichever fails, so the refusal names everything that is wrong.
    #
    # This is also the reason it lives HERE rather than in each caller. This
    # function is the direct API: it takes a capture directory and a list of
    # poses, and it used to believe both. A caller that never asked the capture
    # reader anything - because it was handed clouds and poses from somewhere
    # else - now asks it through this, so there is no entry point left that can
    # publish a reconstruction of a capture whose tracking loss was never
    # resolved.
    report("  checking reconstruction readiness")
    readiness = evaluate_readiness(frames_dir, clouds, poses)
    if not readiness.ready:
        raise ReconstructionRefused(readiness)
    report(f"  readiness: {len(clouds)} frames pass every gate that precedes fusion")

    report(f"  fusing at {resolution.summary}")
    try:
        config = TsdfConfig.of(
            bounds=resolution.bounds,
            voxel_size=resolution.voxel_size,
            truncation=resolution.truncation,
            min_depth=TSDF_MIN_DEPTH if depth_band is None else depth_band.near,
            max_depth=TSDF_MAX_DEPTH if depth_band is None else depth_band.far,
            weight_policy=TSDF_WEIGHT_POLICY,
            max_weight=TSDF_MAX_WEIGHT,
            max_voxels=voxel_budget,
        )
        volume = TsdfVolume(config, backend=tsdf_backend)
    except TsdfNotAllowed as exc:
        raise TsdfSurfaceUnavailable(
            f"no field could be built for this capture: {exc}. Nothing was exported, and the "
            "occupancy reconstruction was NOT attempted in its place"
        ) from exc
    report(f"  integrating {len(raw_frames)} raw frames on the {volume.backend} backend")
    # ONE FRAME PER CALL, and a progress line after each. Fusing is the longest
    # stage of an export and the only one whose cost grows with the capture - a
    # hundred frames is a hundred full-resolution projections - so handing the
    # whole sequence to a single ``integrate`` made the window's Stop
    # unreachable for the entire fusion: the callback fired once, before the
    # first projection, and the next time it could fire the field was finished.
    # The arithmetic is identical either way. ``integrate`` accumulates frame by
    # frame in the order it is given, and its own accounting - ``frames_offered``
    # and ``frames_integrated`` - is a running total across calls, so splitting
    # the sequence changes the field, the evidence and the surface not at all.
    #
    # The line comes AFTER the frame it reports, which is what makes it useful
    # for cancelling: a callback that raises on it stops a fusion that has
    # genuinely done that much work, rather than one that has done none.
    for index, (frame, pose) in enumerate(zip(raw_frames, poses, strict=True), start=1):
        try:
            volume.integrate([frame], [pose], intrinsics=intrinsics, depth_scale=depth_scale)
        except TsdfNotAllowed as exc:
            raise TsdfSurfaceUnavailable(
                f"raw frame {index - 1} could not be fused at its pose: {exc}. Nothing was exported"
            ) from exc
        report(
            f"  fused frame {index}/{len(raw_frames)}, "
            f"{volume.frames_integrated} of them changed the field"
        )
    integration = volume.evidence()
    report(
        f"  {integration.frames_integrated}/{integration.frames_offered} frames changed the "
        f"field, {integration.weights.observed_voxels} voxels observed"
    )
    try:
        extracted = extract_surface(volume, min_weight=TSDF_MIN_WEIGHT)
    except (TsdfNotAllowed, TsdfSurfaceNotAllowed) as exc:
        raise TsdfSurfaceUnavailable(
            f"the fused field holds no surface to export: {exc}. Reported as a refusal rather "
            "than published as an export with no mesh in it, and the occupancy reconstruction "
            "was NOT attempted in its place"
        ) from exc
    extraction = extracted.evidence
    surface = upright_mesh(extracted.mesh)
    report(f"  {surface.vertex_count} vertices, {surface.face_count} faces")

    # The post-fusion gate, and the last thing that happens before a name is
    # reserved. The surface is compared, in both directions, against the points
    # the VOLUME was fused from - a bounded deterministic sample of the same
    # full-resolution ``raw_frames``, deprojected under the same intrinsics and
    # depth scale, kept only inside the same inclusive accepted depth interval,
    # and placed by the same poses through the same ``upright_pose`` the mesh
    # and the cloud are written in.
    #
    # NOT ``points``, which is the registered cloud. That set is the DECIMATED
    # per-frame samples the registration was fitted to; the fusion never saw
    # it. Comparing against it measures whether the surface agrees with a
    # thinned rehearsal of the evidence, so a defect that lives in the frames
    # but misses the decimation - most of every frame, by construction - passes
    # unseen, and the sentence "the points it was fused from" is then simply
    # not true of what was compared.
    #
    # Nothing is re-fused here. ``fusion_evidence_sample`` projects pixels to
    # points; it builds no field and allocates no voxel, and it is bounded at
    # ``fusion_sample_frames x fusion_pixels_per_frame`` candidates.
    #
    # Refusing here means no output name has been reserved, no staging
    # directory exists, and nothing has been written anywhere: the cleanup a
    # later failure would have to perform is a cleanup this refusal never needs
    # to perform, because there is nothing yet to clean. Every earlier failure
    # is carried into the result, so a capture that was going to fail two gates
    # is told about both.
    report("  sampling the full-resolution frames the field was fused from")
    fused_evidence = fusion_evidence_sample(
        raw_frames,
        [upright_pose(pose) for pose in poses],
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
    )
    report(
        f"  {len(fused_evidence.points)} evidence points from "
        f"{fused_evidence.frames_sampled}/{fused_evidence.frames_offered} fused frames"
    )
    report("  checking the fused surface against the points it was fused from")
    agreement, surface_failures = surface_evidence(
        surface.vertices,
        fused_evidence,
        voxel_size=resolution.voxel_size,
        truncation=resolution.truncation,
    )
    readiness = readiness.with_surface(agreement, surface_failures)
    if not readiness.ready:
        raise ReconstructionRefused(readiness)

    cloud_provenance = registered_provenance(
        unit=unit,
        depth_scale=depth_scale,
        calibration_source=calibration_source,
        frames_dir=frames_dir,
        frames=len(clouds),
        points=len(points),
        points_per_frame_target=points_per_frame_target,
        per_frame_filter=per_frame_filter,
        registration_limitations=tuple(registration.limitations),
    )
    mesh_provenance = MeshProvenance(
        cloud=cloud_provenance,
        method=MESH_METHOD,
        voxel_size=resolution.voxel_size,
        # Deliberately absent. An iso level is an occupancy parameter - how many
        # of a node's eight cells must be solid - and there is no such quantity
        # here; the level is zero because that is where a signed distance
        # changes sign. Stating one would describe a reconstruction that did not
        # happen.
        iso_level=None,
        method_family=MeshMethod.TSDF,
        limitations=(
            f"fusion resolution: {resolution.summary}. It was derived from the placed extent "
            "and the voxel budget, not chosen for this object, and a different voxel size gives "
            "a different surface",
            f"this surface was fused from {len(raw_frames)} full-resolution depth frames "
            "placed by the poses the registration accepted, and then rotated once into the "
            "presentation frame by a proper rotation that moved the vertices and left the "
            "winding alone. Whether any sample was masked out of those frames before fusion - "
            "a proved support surface is - is in this export's record under "
            "input.preparation.full_resolution_fusion; the captured files were not modified",
        ),
    )

    directory = _reserve_output_dir(frames_dir)
    # ``staging`` is created INSIDE the block that cleans up, not before it.
    # Making the staging directory can fail on its own - a full disk, a
    # read-only mount, no file descriptors left - and every one of those used to
    # happen after the output name had been reserved and before anything would
    # remove it, leaving an empty ``reconstruction-NNNN`` in the capture folder
    # that this module's own contract says a failed run never leaves.
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(dir=frames_dir, prefix=STAGING_PREFIX))
        # ``mkdtemp`` always makes 0o700, which is right for something private
        # and half-written and wrong for a result. Renaming a directory carries
        # the SOURCE's permissions and discards the destination's, so without
        # this the published export would be reachable by fewer people than the
        # capture directory it sits inside, for no reason anybody chose. The
        # reserved name was made by a plain ``mkdir``, so its mode is what this
        # account's umask gives every other directory this repository creates,
        # including :func:`scanner3d.config.allocate_capture_dir`'s own.
        #
        # Suppressed rather than raised: a filesystem that will not take a mode -
        # a vfat external drive is the ordinary case - would otherwise turn an
        # export that is entirely correct into a failed one over a permission
        # bit. Being reachable is a convenience; being whole is the contract.
        with contextlib.suppress(OSError):
            staging.chmod(stat.S_IMODE(directory.stat().st_mode))
        report(f"  writing {CLOUD_NAME}")
        cloud = write_ply(
            staging / CLOUD_NAME,
            points,
            provenance=cloud_provenance,
            ply_format=PlyFormat.BINARY_LITTLE_ENDIAN,
            max_points=max_points,
        )
        report(f"  writing {MESH_PLY_NAME}")
        mesh_ply = write_ply_mesh(staging / MESH_PLY_NAME, surface, provenance=mesh_provenance)
        report(f"  writing {MESH_OBJ_NAME}")
        mesh_obj = write_obj(staging / MESH_OBJ_NAME, surface, provenance=mesh_provenance)

        # The records name where the files are GOING, not where they are being
        # staged: a manifest published inside the final directory that pointed
        # at a temporary name would be wrong the instant it became readable.
        cloud = replace(cloud, path=str(directory / CLOUD_NAME))
        mesh_ply = replace(mesh_ply, path=str(directory / MESH_PLY_NAME))
        mesh_obj = replace(mesh_obj, path=str(directory / MESH_OBJ_NAME))

        artifacts = [
            _artifact(
                CLOUD_NAME,
                "registered point cloud",
                directory,
                byte_count=cloud.byte_count,
                sha256=cloud.sha256,
            ),
            _artifact(
                MESH_PLY_NAME,
                "triangle mesh, PLY",
                directory,
                byte_count=mesh_ply.byte_count,
                sha256=mesh_ply.sha256,
            ),
            _artifact(
                MESH_OBJ_NAME,
                "triangle mesh, OBJ",
                directory,
                byte_count=mesh_obj.byte_count,
                sha256=mesh_obj.sha256,
            ),
        ]
        record = _record(
            artifacts=artifacts,
            frames_dir=frames_dir,
            directory=directory,
            clouds=len(clouds),
            raw_frames=len(raw_frames),
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            points_per_frame_target=points_per_frame_target,
            per_frame_filter=per_frame_filter,
            preparation=preparation,
            registration=registration,
            resolution=resolution,
            integration=integration,
            extraction=extraction,
            surface_bounds=surface.bounds,
            readiness=readiness,
            cloud=cloud,
            mesh_ply=mesh_ply,
            mesh_obj=mesh_obj,
        )
        payload = json.dumps(record, indent=2, allow_nan=False).encode("ascii")
        _write_record(staging / RECORD_NAME, payload)
        _verify(staging, artifacts)
        _verify(
            staging,
            [
                {
                    "file": RECORD_NAME,
                    "byte_count": len(payload),
                    "sha256": sha256_of(payload),
                }
            ],
        )
        report(f"  publishing {directory.name}")
        # The whole export becomes visible under its final name in one step.
        # ``directory`` is the empty directory reserved above, which is what
        # makes this rename legal and what stops it from ever landing on
        # somebody else's result: a directory with anything in it refuses.
        staging.rename(directory)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        with contextlib.suppress(OSError):
            directory.rmdir()
        raise
    return ReconstructionExport(
        directory=directory,
        cloud=cloud,
        mesh_ply=mesh_ply,
        mesh_obj=mesh_obj,
        resolution=resolution,
        record=record,
        surface=surface,
        integration=integration,
        extraction=extraction,
    )


def _silent(_message: str) -> None:
    """The default progress callback: says nothing, cancels nothing."""


def _field_record(integration: TsdfEvidence) -> dict[str, object]:
    """The fusion evidence, with the field's caveats rather than the sample bag's.

    :meth:`scanner3d.tsdf.TsdfVolume.evidence` attaches that module's full
    :data:`~scanner3d.tsdf.INHERENT_LIMITATIONS`, which includes one caveat that
    is about :meth:`~scanner3d.tsdf.TsdfVolume.zero_level_samples` - an
    unconnected bag of points with no topology, no orientation and no
    watertightness. That caveat is FALSE of what this export wrote, and shipping
    it in the manifest of a connected mesh would tell a reader the opposite of
    what is in their hands. So the field-only split that module maintains for
    exactly this case is used instead, and the substitution is stated rather
    than performed silently. Every measured number is untouched.
    """
    field = integration.as_dict()
    field["limitations"] = list(FIELD_LIMITATIONS)
    field["limitations_note"] = (
        "scanner3d.tsdf.FIELD_LIMITATIONS: every caveat that is about the FIELD. That module's "
        "zero-level-sample caveat is deliberately absent because no sample bag was produced "
        "here - the field was read by scanner3d.tsdf_surface into a connected mesh, whose own "
        "caveats are in this record's extraction block and limitations list"
    )
    return field


def _record(
    *,
    artifacts: Sequence[Mapping[str, object]],
    frames_dir: Path,
    directory: Path,
    clouds: int,
    raw_frames: int,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
    points_per_frame_target: int | None,
    per_frame_filter: str | None,
    preparation: Mapping[str, object] | None,
    registration: SweepReconstruction,
    resolution: TsdfResolution,
    integration: TsdfEvidence,
    extraction: SurfaceEvidence,
    surface_bounds: tuple[Point, Point] | None,
    readiness: ReconstructionReadiness,
    cloud: EncodedCloud,
    mesh_ply: MeshExport,
    mesh_obj: MeshExport,
) -> dict[str, object]:
    """The JSON document written beside the geometry.

    Everything a consumer needs that the three geometry files cannot carry
    between them, and nothing that duplicates their contents: the artifacts and
    their digests, what was registered, what was meshed and at what resolution,
    which axis of the optical frame each exported axis is, and every limitation
    the two provenance records accumulated.
    """
    cloud_record = cloud.as_dict()
    ply_record = mesh_ply.as_dict()
    obj_record = mesh_obj.as_dict()
    replaced: list[str] = []
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "producer": f"scanner3d {__version__}",
        "input": {
            "frames_directory": str(frames_dir),
            "frames": clouds,
            "raw_frames_fused": raw_frames,
            "raw_frame_bytes": EXPECTED_FRAME_BYTES,
            "points_per_frame_target": points_per_frame_target,
            "per_frame_filter": per_frame_filter,
            # The same account as ``per_frame_filter``, structured and version
            # stamped, so a consumer can branch on it rather than parse a
            # sentence - and so that the registration cloud's filtering and the
            # fusion input's masking stay legible as the two separate things
            # they are.
            "preparation": None if preparation is None else dict(preparation),
            "depth_intrinsics": intrinsics.as_dict(),
            "depth_scale": depth_scale,
            "calibration_note": (
                "the intrinsics and depth scale recorded here are the exact values the fusion "
                "projected with; they were supplied by the caller and nothing here estimated, "
                "refined or checked them beyond the finiteness the arithmetic needs"
            ),
            "note": (
                "the captured frames in this directory are the input to this export and were "
                "not modified, moved or removed by it. The registered cloud is written from "
                "the decimated clouds; the mesh is fused from full-resolution frames. Those "
                "fused frames may be DERIVED in memory from the captured ones - see "
                "input.preparation.full_resolution_fusion, which says whether any sample was "
                "masked out of them and why. Nothing was written back to a capture either way"
            ),
        },
        "output_directory": str(directory),
        "artifacts": [dict(artifact) for artifact in artifacts],
        "coordinate_convention": {
            "exported_frame": UPRIGHT_FRAME.as_dict(),
            "source_frame": CAMERA_OPTICAL_FRAME.as_dict(),
            "axis_mapping": dict(AXIS_MAPPING),
            "rotation_applied": [list(row) for row in UPRIGHT_FROM_OPTICAL.rotation],
            "applied_by": (
                "scanner3d, as a presentation decision - see the caveat in this record's "
                "limitations. Nothing about the scanner, the intrinsics or the file formats "
                "declares an up direction"
            ),
            "registration_frame_unchanged": True,
        },
        "registration": registration.as_dict(),
        # What was checked before this export was allowed to exist, the policy
        # values it was checked against and the evidence measured under them.
        # A published record always carries a READY readiness - a refused one
        # publishes nothing at all - so this is the evidence for the decision
        # that let these files be written, not a warning beside them.
        "readiness": readiness.as_dict(),
        "point_cloud": cloud_record,
        "mesh": {
            "method": MESH_METHOD,
            "method_family": str(MeshMethod.TSDF),
            "occupancy_fallback_used": False,
            "resolution": resolution.as_dict(),
            "vertex_count": mesh_ply.vertex_count,
            "face_count": mesh_ply.face_count,
            "watertight": {
                "measured": mesh_ply.watertight,
                "how": (
                    "every undirected edge of every face was counted; a closed surface has each "
                    "of them used by exactly two faces"
                ),
                "means": (
                    "this surface stops where the evidence stops, so an open mesh is where the "
                    "capture stopped seeing the object and a closed one is a capture that "
                    "surrounded it. Nothing padded the field to make it close"
                ),
                "edges": mesh_ply.census.as_dict(),
            },
            "bounds": (
                None
                if surface_bounds is None
                else {"min": list(surface_bounds[0]), "max": list(surface_bounds[1])}
            ),
            "tsdf": {
                "raw_frames_fused": raw_frames,
                "raw_frame_bytes": EXPECTED_FRAME_BYTES,
                "depth_intrinsics": intrinsics.as_dict(),
                "depth_scale": depth_scale,
                "pose_direction": (
                    "each pose maps ITS FRAME'S CAMERA coordinates into the model frame - the "
                    "direction scanner3d.reconstruction_export.registered_points applies. The "
                    "fusion inverts it to carry each voxel back into that camera; a caller that "
                    "passed the inverse would get a scan that is wrong in a way that still looks "
                    "like a scan"
                ),
                "frame_of_fusion": (
                    "the model frame the poses are in, NOT the upright presentation frame. The "
                    "half turn was applied once, to the extracted vertices, after extraction"
                ),
                "upright_rotation_applied_after_extraction": True,
                "faces_reversed_by_the_rotation": False,
                "min_weight": extraction.min_weight,
                "integration": _field_record(integration),
                "extraction": extraction.as_dict(),
                "limitations": [
                    ACCURACY_DISCLAIMER,
                    "poses are consumed exactly as given. Nothing in this export estimated, "
                    "refined, repaired, scored or corrected one, and a wrong pose is fused as "
                    "faithfully as a right one into a confidently wrong surface",
                    "no dimensional accuracy is claimed for this mesh beyond whatever the "
                    "supplied intrinsics and depth scale already carried, and the depth scale's "
                    "physical unit remains unestablished on this device, so a distance of 1.0 "
                    "here is one unit of an unnamed quantity",
                ],
            },
            "ply": ply_record,
            "obj": obj_record,
            "geometry_is_identical_in_both_files": True,
        },
        "dimensional_accuracy_claimed": False,
        "dimensional_accuracy_disclaimer": ACCURACY_DISCLAIMER,
        "limitations": list(mesh_ply.limitations),
    }
    safe = _json_safe(document, replaced)
    if not isinstance(safe, dict):  # pragma: no cover - a dict in is a dict out
        raise ArtifactVerificationFailed("the record did not survive being made JSON-safe")
    safe["non_finite_values_replaced_with_null"] = replaced
    return safe
