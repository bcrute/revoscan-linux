"""The raw-points versus raw-TSDF-surface oracle: what each arm must say, and what is measured.

This is the contract half of a third deterministic corpus, beside
:mod:`tests.quality_harness` and :mod:`tests.depth_completeness`. It exists to
answer one question with evidence instead of by eye: **why is the fused surface
lumpier and less complete than the measured dots?**

The method is a controlled comparison. One exact-truth multi-view scene is
rendered once. A fixture then declares one *perturbation* of it - and only one,
except in the deliberately combined case - and the same raw frames and the same
poses are handed to two arms:

* ``raw_points``, the unchanged measured point set, deprojected through the
  production path and placed by the same poses the fusion is given;
* ``fused:<cell>``, the unchanged :mod:`scanner3d.tsdf` field integrated from
  those same frames and poses at one predeclared parameter cell, meshed by the
  unchanged :func:`scanner3d.tsdf_surface.extract_surface`;
* ``fused_at_appliance_resolution``, the same two modules at the resolution the
  SHIPPED export policy chooses for this scene - see
  :func:`appliance_cell` - because a conclusion about the mesh an owner
  actually gets cannot be drawn from a coarser proxy of it.

Every metric is computed **here**, from the arm's output and the scene's exact
truth, and never taken from the arm. What an arm is allowed to say about itself
is a small closed vocabulary of *claims*, every one of which is bound to
something this module counts for itself, so a candidate that flatters its own
work record is refused rather than published beside it.

Two references, and they are not the same thing
-----------------------------------------------
A point can be exactly on the object and still be somewhere no camera ray
landed. Conflating those two facts is how an earlier round of this corpus
reported an extraction defect that was really a property of its reference, so
they are now separate measurements with separate names:

* **analytic accuracy** (:func:`analytic_distance`) is the exact unsigned
  distance from a point to the boundary of the union of the scene's declared
  shapes. It is closed-form, independent of where the sensor sampled, and
  independent of the depth quantisation - a valid mesh vertex interpolated
  BETWEEN two camera rays scores zero here, which is correct;
* **measurement support** (the ``support`` measurement) is proximity to the
  ideal returns the five views actually produced. It says whether geometry sits
  where something was measured. It is NOT truth accuracy and this module never
  calls it that.

The symmetric coverage of the analytic reference uses a dense exact sampling of
the same surface - :func:`analytic_reference` - at a supersample factor the
FIXTURE owns, computed by intersecting sub-pixel camera rays with the shapes in
closed form rather than by reading a rendered frame back.

Fusion, or extraction?
----------------------
A fused arm publishes an ``ablation`` measurement that separates the two stages
it is made of. :meth:`scanner3d.tsdf.TsdfVolume.zero_level_samples` is where the
FIELD's own evidence puts the zero level, with no marching cubes anywhere near
it; the arm's points are what the extractor emitted from that same field. Both
are scored against the same analytic truth, so the difference between them is
what extraction added and everything they share is what fusion did.

Vertices are not the surface
----------------------------
That ablation is necessary and it is not sufficient, and the reason is a
property of marching cubes rather than of this corpus: an extracted vertex sits
on the same field-edge zero crossing that
:meth:`~scanner3d.tsdf.TsdfVolume.zero_level_samples` reports, so comparing
those two point lists can only ever find small disagreements. A triangle that
spans a protected gap, or that seals a region nothing measured, does it with
all three of its vertices on perfectly ordinary zero crossings - and a
vertex-only reader sees nothing at all.

So the emitted surface is measured as a surface, twice:

* ``surface_samples`` scores a fixed, strictly interior barycentric lattice
  inside EVERY emitted triangle - see :func:`barycentric_lattice` - against the
  same analytic truth and the same protected volume. The order and the sample
  ceiling are owned by the fixture, so the work is bounded before any result
  exists and nothing is thinned afterwards to fit;
* ``topology`` counts what the faces are, not where they are:
  :func:`scanner3d.mesh.edge_census` for manifoldness and closure, the shells
  the face graph breaks into, and - the number the complaint is really about -
  how many of those shells are CLOSED while no view measured anything near
  them. Each of those is then scored a second time against the analytic
  surface, because unsupported is not false: a sealed shell on a face no view
  could see is unsupported measurement and correct geometry at once. The
  artifact publishes the two groups separately - the shells that are off the
  object, which is false surface, and the shells that are on it, which is the
  cost any support-only pruning rule would pay - and never collapses them back
  into one number.

The vertex evidence stays. It is read beside these two and never instead of
them, and neither vertex agreement nor a component count is allowed to clear
extraction as a cause on its own.

Nothing in this module routes anything into the product. It reads
:mod:`scanner3d.tsdf`, :mod:`scanner3d.tsdf_surface`, :mod:`scanner3d.quality`
and :mod:`scanner3d.depth_metric` and changes none of them. Task 032's result -
that re-fusing temporally completed samples adds no surface coverage once their
raw supporting views are fused - is preserved by construction: no completion of
any kind is offered to a volume here, and the only input a fused arm may
integrate is the case's own raw measured frames.

What this corpus cannot say
---------------------------
It is synthetic. :mod:`scanner3d.simulate` is not a sensor model, so a cause
this corpus isolates is a cause *of the arithmetic*, demonstrated on exact
truth. It is necessary evidence for a change and never sufficient; physical
evidence is a separate, later gate and no number here is a physical length.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import scanner3d.quality as quality
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.mesh import edge_census
from scanner3d.reconstruction_export import (
    TRUNCATION_VOXELS,
    TSDF_MIN_WEIGHT,
    TSDF_WEIGHT_POLICY,
    choose_tsdf_resolution,
)
from scanner3d.simulate import Box as SimBox
from scanner3d.simulate import Plane, Shape, Sphere
from scanner3d.tsdf import (
    TsdfConfig,
    TsdfVolume,
    VolumeBounds,
    WeightPolicy,
)
from scanner3d.tsdf_surface import NoSurfaceFound, extract_surface
from tests.depth_completeness import (
    MAX_CODE,
    Window,
    cloud_of,
    codes_of,
)

#: Reused rather than re-derived. Every denominator, every window read and
#: every synthetic frame in this corpus is the depth-completeness benchmark's
#: own primitive, so there is one authority for what a window is and how its
#: codes become a real 512000-byte frame.
__all__ = [
    "APPLIANCE_CELL_ID",
    "ARMS",
    "CANDIDATE_FORMS",
    "CLAIMS",
    "PERTURBATIONS",
    "REFUSALS",
    "AppliancePolicy",
    "ArmResult",
    "CandidateRefused",
    "GridCell",
    "Perturbation",
    "Regions",
    "Scene",
    "ScoringUnusable",
    "ViewEvidence",
    "WorkClaim",
    "analytic_distance",
    "analytic_reference",
    "appliance_cell",
    "barycentric_lattice",
    "candidate",
    "component_sizes",
    "measure",
    "score_surface",
    "surface_samples",
    "surface_topology",
    "triangle_interior_samples",
    "validate",
]

SCHEMA_VERSION: Final = 1

#: A sample that never returned. The same zero every reader in this repository
#: treats as "no measurement".
NO_RETURN: Final = 0

#: The unit prose on every length this module measures.
DEPTH_UNIT: Final = (
    "synthetic depth units: raw code x depth scale, exact by construction. No physical length"
)
FRACTION_UNIT: Final = quality.FRACTION_UNIT
COUNT_UNIT: Final = "points"
VOXEL_UNIT: Final = "voxels"
BYTE_UNIT: Final = "bytes"

#: Where a dimension's truth came from. Constructed, not measured.
TRUTH_SOURCE: Final = (
    "the exact truth frames rendered by scanner3d.simulate before any perturbation, deprojected "
    "through the production path and placed by the scene's exact poses. Synthetic and exact by "
    "construction; no instrument and no physical unit"
)

#: What the ANALYTIC reference is, in one sentence a reader can check. Closed
#: form, so it does not know where the sensor sampled and cannot be biased by
#: the ray pitch, the depth quantisation or any decimation applied afterwards.
ANALYTIC_SOURCE: Final = (
    "the exact unsigned distance to the boundary of the union of the scene's declared shapes, "
    "in closed form. Independent of where any camera ray landed, of the raw-code quantisation "
    "and of every lattice this module decimates onto. Synthetic and exact by construction; no "
    "instrument and no physical unit"
)

#: What the SUPPORT reference is, and - said out loud, because an earlier round
#: of this corpus reported it as accuracy - what it is not.
SUPPORT_SOURCE: Final = (
    "proximity to the ideal returns the scene's own views produced, decimated onto the "
    "fixture's lattice. It measures whether geometry sits where something was MEASURED and is "
    "not a truth-accuracy measure: a point exactly on the object, between two camera rays or on "
    "a face no view could see, is correct geometry and unsupported measurement at the same time"
)

#: What the triangle-interior samples are, and the defect they exist to catch.
SURFACE_SAMPLE_SOURCE: Final = (
    "a fixed, strictly interior barycentric lattice inside every emitted triangle - one centroid "
    "per sub-triangle of a regular subdivision at the order the fixture owns, which is n**2 "
    "EQUAL-AREA samples per face - under a sample ceiling the fixture owns. Equal area is what "
    "lets a fraction of a face's samples be read as a fraction of its area. Marching-cubes "
    "vertices lie "
    "on the same field-edge zero crossings the pre-extraction samples do, so a vertex-only "
    "comparison cannot see a triangle that spans a protected gap or seals an unsupported region "
    "while all three of its vertices stay outside it. What a person looking at the mesh sees is "
    "the interior of the emitted surface, and this is that interior"
)

#: The two arms a case compares. Closed: an arm this module cannot re-derive
#: is an arm whose numbers nobody can check.
RAW_POINTS: Final = "raw_points"
FUSED: Final = "fused"
ARMS: Final[tuple[str, ...]] = (RAW_POINTS, FUSED)

#: Every perturbation label a fixture may declare, and nothing else. The label
#: is not decoration: :func:`check_perturbation_label` binds it to the fields
#: that are actually non-zero, so a case cannot be filed under "pose error"
#: while quietly carrying depth noise.
EXACT: Final = "exact"
DEPTH_NOISE: Final = "depth-noise"
DEPTH_OUTLIERS: Final = "depth-outliers"
NO_RETURN_BANDS: Final = "no-return-bands"
POSE_TRANSLATION: Final = "pose-translation"
POSE_ROTATION: Final = "pose-rotation"
COMBINED: Final = "combined"
PERTURBATIONS: Final[tuple[str, ...]] = (
    EXACT,
    DEPTH_NOISE,
    DEPTH_OUTLIERS,
    NO_RETURN_BANDS,
    POSE_TRANSLATION,
    POSE_ROTATION,
    COMBINED,
)

#: The perturbation families, for the attribution the harness publishes. A
#: label belongs to exactly one, and ``combined`` deliberately belongs to
#: neither: it is the interaction case, and calling it "depth" or "pose" would
#: pre-decide the thing the corpus exists to measure.
DEPTH_FAMILY: Final[tuple[str, ...]] = (DEPTH_NOISE, DEPTH_OUTLIERS, NO_RETURN_BANDS)
POSE_FAMILY: Final[tuple[str, ...]] = (POSE_TRANSLATION, POSE_ROTATION)

#: What an arm may claim about itself. Deliberately forgeable, deliberately
#: tiny, and every one of them bound in :func:`validate` to something this
#: module computes. There is no claim for a threshold, a score or a verdict,
#: because a candidate that could author its own bar would not be measured by
#: this corpus at all.
CLAIMS: Final[tuple[str, ...]] = (
    "point_count",
    "frames_fused",
    "observed_voxels",
    "surface_vertices",
    "surface_faces",
    "work_units",
)

#: Every way an arm's own account of itself can fail to hold. Closed, ordered,
#: and published in the artifact, so a control that must be refused names the
#: exact refusal it has to earn rather than "something went wrong".
REFUSALS: Final[tuple[str, ...]] = (
    "arm_unknown",
    "grid_cell_unknown",
    "views_omitted",
    "pose_substituted",
    "raw_frame_edited",
    "points_not_measured",
    "surface_not_derived",
    "point_not_finite",
    "points_outside_volume",
    "sample_count_mismatch",
    "claim_unknown",
    "claim_not_supported",
    "work_budget_exceeded",
)

#: The candidate forms a fixture may name. ``fused`` takes one argument, the
#: id of a grid cell the CORPUS declared; every other form takes none.
CANDIDATE_FORMS: Final[tuple[str, ...]] = (
    "raw_points",
    "fused",
    "fused_at_appliance_resolution",
    "fused_from_truth_frames",
    "fused_dropping_a_view",
    "fused_with_true_poses",
    "fused_with_edited_surface",
    "fused_with_bridging_triangle",
    "raw_points_dropping_hard_samples",
    "fused_overclaiming_work",
    "fused_claiming_the_unmeasurable",
)

#: The backend the grid arms run on. The pure-Python references, so the answer
#: does not depend on numpy being installed and two runs on two machines agree.
BACKEND: Final = "python"

#: The id of the derived cell the appliance-resolution arm fuses at. It is NOT
#: a member of the corpus grid: its voxel size is not a number a fixture may
#: choose, it is whatever the shipped policy computes for this scene.
APPLIANCE_CELL_ID: Final = "appliance"

#: The backend that arm runs on. The resolution the shipped policy picks needs
#: about two million voxels, which is thirty times what
#: :data:`scanner3d.tsdf.PYTHON_BACKEND_MAX_VOXELS` allows, so this one arm is
#: accelerated and its numbers are therefore machine-dependent in their last
#: bits in a way the pure-Python corpus is not. That is the trade the review
#: allowed, and it is stated in the artifact rather than left to be discovered.
APPLIANCE_BACKEND: Final = "numpy"

#: How many vertices the bridging-triangle control may consider NEAR EACH
#: CORNER of the protected volume. Two corners rather than one centre, because
#: a gap is spanned from one side of it to the other and the geometry nearest
#: the middle of an empty box is all on whichever side happens to be closest.
#: A ceiling on the search rather than a target: at most twice this many
#: vertices, and therefore a fixed number of triples, whatever the mesh's size.
_BRIDGE_SHORTLIST: Final = 16


class CandidateRefused(ScannerError):
    """An arm whose own account of itself does not hold, and the reason why."""

    def __init__(self, refusal: str, detail: str) -> None:
        if refusal not in REFUSALS:
            raise ScoringUnusable(
                f"{refusal!r} is not one of this oracle's refusals {list(REFUSALS)}; a refusal "
                "nobody declared cannot be expected by a fixture"
            )
        super().__init__(f"{refusal}: {detail}")
        self.refusal = refusal
        self.detail = detail


class ScoringUnusable(ScannerError):
    """The scene or the request itself is unusable, and nothing was measured."""


# --- the perturbation, and the label it has to deserve --------------------------------


@dataclass(frozen=True, slots=True)
class Perturbation:
    """Exactly what a case did to the exact scene, as data.

    Applied by this module, never by a candidate, so both arms of a case are
    given byte-identical frames and identical poses. Splitting the depth fields
    from the pose fields is the whole design: a case that changes one of them
    is comparable with its control, and the report can say which one moved the
    metric.
    """

    label: str
    #: Bounded, zero-mean, integer depth noise: each returned sample moves by a
    #: draw uniform on the symmetric integer range [-n, +n].
    depth_noise_codes: int = 0
    #: A localized outlier on every ``outlier_period``-th returned sample, of
    #: ``outlier_codes`` magnitude and alternating sign, so an outlier field is
    #: a tail rather than a bias.
    outlier_period: int = 0
    outlier_codes: int = 0
    #: Structured no-return: ``band_width`` of every ``band_period`` columns
    #: never return, and the band shifts by ``band_shift`` columns per frame so
    #: the loss is complementary across views rather than a permanent blind
    #: stripe.
    band_period: int = 0
    band_width: int = 0
    band_shift: int = 0
    #: A constant pose error added to every view's estimate, and a per-view
    #: draw on top of it. The MEASUREMENT is untouched: the frames are exact
    #: and only the pose the fusion is given is wrong, which is what a
    #: tracking error actually is.
    translation_bias: Point = (0.0, 0.0, 0.0)
    translation_jitter: float = 0.0
    rotation_bias_degrees: float = 0.0
    rotation_jitter_degrees: float = 0.0
    rotation_axis: Point = (0.0, 1.0, 0.0)

    @property
    def touches_depth(self) -> bool:
        return bool(
            self.depth_noise_codes
            or (self.outlier_period and self.outlier_codes)
            or (self.band_period and self.band_width)
        )

    @property
    def touches_pose(self) -> bool:
        return bool(
            any(value != 0.0 for value in self.translation_bias)
            or self.translation_jitter
            or self.rotation_bias_degrees
            or self.rotation_jitter_degrees
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "depth_noise_codes": self.depth_noise_codes,
            "outlier_period": self.outlier_period,
            "outlier_codes": self.outlier_codes,
            "band_period": self.band_period,
            "band_width": self.band_width,
            "band_shift": self.band_shift,
            "translation_bias": list(self.translation_bias),
            "translation_jitter": self.translation_jitter,
            "rotation_bias_degrees": self.rotation_bias_degrees,
            "rotation_jitter_degrees": self.rotation_jitter_degrees,
            "rotation_axis": list(self.rotation_axis),
            "touches_depth": self.touches_depth,
            "touches_pose": self.touches_pose,
        }


def check_perturbation_label(perturbation: Perturbation) -> None:
    """Refuse a perturbation whose label is not what it actually does.

    The forged-label control. A corpus whose cases are named after their causes
    is only evidence if the names are load-bearing, so the label is bound to
    the fields: ``exact`` must change nothing, a depth label must change depth
    and nothing else, a pose label must change pose and nothing else, and
    ``combined`` must change both. Anything else is a case filed under the
    wrong cause, which is worse than no case at all.
    """
    label = perturbation.label
    if label not in PERTURBATIONS:
        raise ScoringUnusable(
            f"perturbation label {label!r} is not one of {list(PERTURBATIONS)}; this oracle "
            "attributes results by label and cannot attribute one it does not define"
        )
    depth = perturbation.touches_depth
    pose = perturbation.touches_pose
    if label == EXACT and (depth or pose):
        raise ScoringUnusable(
            "a perturbation called 'exact' changes the depth or the poses. The exact case is "
            "every other case's control, and a control that is itself perturbed measures nothing"
        )
    if label in DEPTH_FAMILY and (not depth or pose):
        raise ScoringUnusable(
            f"perturbation {label!r} is a depth perturbation but touches_depth={depth} and "
            f"touches_pose={pose}. A depth case that also moves the poses cannot be compared "
            "with the exact control on one variable"
        )
    if label in POSE_FAMILY and (depth or not pose):
        raise ScoringUnusable(
            f"perturbation {label!r} is a pose perturbation but touches_depth={depth} and "
            f"touches_pose={pose}. A pose case that also edits the depth cannot be compared "
            "with the exact control on one variable"
        )
    if label == COMBINED and not (depth and pose):
        raise ScoringUnusable(
            f"perturbation 'combined' touches_depth={depth} and touches_pose={pose}. The "
            "combined case exists to show an interaction, and one that moves a single family "
            "is a duplicate of that family's own case under another name"
        )
    if label in DEPTH_FAMILY:
        expected = {
            DEPTH_NOISE: bool(perturbation.depth_noise_codes),
            DEPTH_OUTLIERS: bool(perturbation.outlier_period and perturbation.outlier_codes),
            NO_RETURN_BANDS: bool(perturbation.band_period and perturbation.band_width),
        }
        if not expected[label]:
            raise ScoringUnusable(
                f"perturbation {label!r} does not carry the field its own name describes"
            )
        for other, active in expected.items():
            if other != label and active:
                raise ScoringUnusable(
                    f"perturbation {label!r} also carries the {other!r} field. One case varies "
                    "one thing, or the attribution beside it is a guess"
                )


def draws(seed: str, count: int) -> tuple[float, ...]:
    """``count`` reproducible draws in [0, 1), from a seed and nothing else.

    A digest chain rather than :mod:`random`: the values depend on the seed
    text alone, so a run on another machine, another CPython build or another
    day produces the same perturbation. No wall clock, no process state, no
    global generator anyone else could have advanced.
    """
    if count < 0:
        raise ScoringUnusable(f"cannot draw {count} values")
    out: list[float] = []
    block = seed.encode("utf-8")
    while len(out) < count:
        block = hashlib.sha256(block).digest()
        for offset in range(0, 32, 8):
            if len(out) == count:
                break
            out.append(int.from_bytes(block[offset : offset + 8], "big") / float(1 << 64))
    return tuple(out)


def _symmetric(draw: float, magnitude: int) -> int:
    """One integer on the symmetric range [-magnitude, +magnitude], zero-mean.

    ``2 * magnitude + 1`` equally likely values whose mean is exactly zero, so
    "bounded zero-mean depth noise" is a property of the construction rather
    than of how a sample happened to land.
    """
    span = 2 * magnitude + 1
    return min(int(draw * span), span - 1) - magnitude


def perturb_codes(
    truth: Sequence[int], perturbation: Perturbation, *, window: Window, view: int, seed: str
) -> tuple[int, ...]:
    """The measured codes one view actually returns under ``perturbation``.

    Applied to the exact truth codes in a fixed order - bands, then noise, then
    outliers - so the result is a pure function of the truth, the perturbation,
    the view index and the seed. A dropped band is a real no-return: the sample
    is zero, not a guess, and every arm sees the same zero.

    The band walks by ``band_shift`` columns per view, so structured loss is
    COMPLEMENTARY across the views rather than one permanent blind stripe. A
    stripe every view misses is a hole no multi-view method can fill, and a
    case built on one would measure the scene's construction instead of the
    fusion's behaviour.
    """
    if len(truth) != window.sample_count:
        raise ScoringUnusable(
            f"{len(truth)} code(s) for a window of {window.sample_count} sample(s)"
        )
    dropped: set[int] = set()
    if perturbation.band_period and perturbation.band_width:
        for index in range(len(truth)):
            column = index % window.width
            phase = (column + perturbation.band_shift * view) % perturbation.band_period
            if phase < perturbation.band_width:
                dropped.add(index)

    returned = [index for index, code in enumerate(truth) if code != NO_RETURN]
    noise = draws(f"{seed}:noise", len(returned)) if perturbation.depth_noise_codes else ()
    out = list(truth)
    for position, index in enumerate(returned):
        if index in dropped:
            out[index] = NO_RETURN
            continue
        code = truth[index]
        if perturbation.depth_noise_codes:
            code += _symmetric(noise[position], perturbation.depth_noise_codes)
        if perturbation.outlier_period and position % perturbation.outlier_period == 0:
            sign = 1 if (position // perturbation.outlier_period) % 2 == 0 else -1
            code += sign * perturbation.outlier_codes
        out[index] = min(MAX_CODE, max(1, code))
    return tuple(out)


def perturb_pose(pose: Rigid, perturbation: Perturbation, *, seed: str) -> Rigid:
    """The pose the fusion is GIVEN, which is the true pose plus a stated error.

    The error is applied in world coordinates on the outside of the true pose,
    so it is exactly the failure a tracker makes: the camera really was where
    the truth says, and the reconstruction believes otherwise.
    """
    if not perturbation.touches_pose:
        return pose
    values = draws(f"{seed}:pose", 4)
    shift = tuple(
        bias + perturbation.translation_jitter * (2.0 * values[axis] - 1.0)
        for axis, bias in enumerate(perturbation.translation_bias)
    )
    degrees = perturbation.rotation_bias_degrees + perturbation.rotation_jitter_degrees * (
        2.0 * values[3] - 1.0
    )
    error = rotation_about_axis(perturbation.rotation_axis, math.radians(degrees))
    return Rigid.of(error.rotation, (shift[0], shift[1], shift[2])).compose(pose)


# --- the predeclared parameter grid ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridCell:
    """One predeclared fusion/extraction parameter cell.

    Owned by the CORPUS manifest, not by a case and never by a candidate: every
    case runs every cell, so a result cannot be produced by searching for a
    parameter that flatters it and reporting only that one.
    """

    cell_id: str
    voxel_size: float
    truncation_voxels: float
    min_weight: float
    weight_policy: str

    @property
    def truncation(self) -> float:
        return self.voxel_size * self.truncation_voxels

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "voxel_size": self.voxel_size,
            "truncation_voxels": self.truncation_voxels,
            "truncation": self.truncation,
            "min_weight": self.min_weight,
            "weight_policy": self.weight_policy,
        }


@dataclass(frozen=True, slots=True)
class VolumeSpec:
    """The fixed box every cell of a case fuses into, and the accepted depths.

    Fixed rather than derived from the data. A volume fitted to whatever the
    arm produced would shrink around a result that lost the object, and the
    coverage denominator would shrink with it.
    """

    minimum: Point
    maximum: Point
    min_depth: float
    max_depth: float

    def config(self, cell: GridCell) -> TsdfConfig:
        return TsdfConfig.of(
            bounds=VolumeBounds.of(self.minimum, self.maximum),
            voxel_size=cell.voxel_size,
            truncation=cell.truncation,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            weight_policy=WeightPolicy(cell.weight_policy),
        )

    def contains(self, point: Point, *, margin: float) -> bool:
        return all(
            self.minimum[axis] - margin <= point[axis] <= self.maximum[axis] + margin
            for axis in range(3)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
        }


@dataclass(frozen=True, slots=True)
class AppliancePolicy:
    """The SHIPPED resolution policy one arm of every case is bound to.

    Not a parameter cell. The corpus grid is a set of voxel sizes somebody
    chose; this is the arithmetic ``scanner3d.reconstruction_export`` performs
    on the scene's own placed extent, at the voxel budget the Reconstruct
    action passes it, with the truncation ratio, minimum weight and weight
    policy that module ships. The manifest states every one of them and owns
    none of them: :func:`check_appliance_policy` binds the budget to
    :data:`scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET` and the other three to
    :mod:`scanner3d.reconstruction_export`, so a policy that drifts away from
    the shipped one is a refusal rather than a stale comparison. The budget is
    bound because it is what sets the resolution: a manifest free to write any
    number there could name a voxel size no owner ever gets and still call the
    arm "the appliance".

    The voxel size itself is deliberately absent: it is derived per scene and
    published as a result, never declared.
    """

    policy: str
    voxel_budget: int
    truncation_voxels: float
    min_weight: float
    weight_policy: str
    backend: str
    work_budget: int

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "voxel_budget": self.voxel_budget,
            "truncation_voxels": self.truncation_voxels,
            "min_weight": self.min_weight,
            "weight_policy": self.weight_policy,
            "backend": self.backend,
            "work_budget": self.work_budget,
        }


def check_appliance_policy(policy: AppliancePolicy) -> None:
    """Refuse an appliance policy that is not the one the product ships today.

    The whole value of this arm is that it is the appliance's own arithmetic.
    A declared constant that has drifted from
    :mod:`scanner3d.reconstruction_export` would make it a fourth parameter
    cell wearing the appliance's name, so each one is compared with the
    imported value and nothing is guessed.

    The budget is bound the same way and to its own authority. It is not a
    constant :mod:`scanner3d.reconstruction_export` owns: it is the number the
    desktop Reconstruct action passes in, so it is compared with
    :data:`scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET` itself. Without that
    binding, "the appliance resolution" would rest on a number a fixture
    happened to write down, and the whole point of this arm is that the
    resolution is the shipped policy's own arithmetic rather than a fourth
    voxel size somebody chose. The import is local because it is one integer:
    the corpus, its builder and the read-only replay have no other reason to
    pull the desktop window - and everything it imports - into memory.
    """
    from scanner3d.gui import RECONSTRUCT_VOXEL_BUDGET

    for name, declared, shipped in (
        ("voxel_budget", policy.voxel_budget, RECONSTRUCT_VOXEL_BUDGET),
        ("truncation_voxels", policy.truncation_voxels, TRUNCATION_VOXELS),
        ("min_weight", policy.min_weight, TSDF_MIN_WEIGHT),
        ("weight_policy", policy.weight_policy, str(TSDF_WEIGHT_POLICY)),
    ):
        if declared != shipped:
            raise ScoringUnusable(
                f"the appliance policy declares {name}={declared!r} and "
                f"the product ships {shipped!r}. This arm exists to be the appliance's own "
                "arithmetic, and one that has drifted from it is a parameter cell under a name "
                "that claims otherwise"
            )
    if policy.backend not in (BACKEND, APPLIANCE_BACKEND):
        raise ScoringUnusable(
            f"the appliance policy names backend {policy.backend!r}; this oracle runs "
            f"{[BACKEND, APPLIANCE_BACKEND]}"
        )
    if policy.voxel_budget <= 0 or policy.work_budget <= 0:
        raise ScoringUnusable(
            f"the appliance policy states voxel_budget={policy.voxel_budget} and "
            f"work_budget={policy.work_budget}; both are ceilings and both must be positive"
        )


# --- what a case names about its own geometry -----------------------------------------


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned region of the scene, named by the fixture."""

    minimum: Point
    maximum: Point

    def contains(self, point: Point) -> bool:
        return all(self.minimum[axis] <= point[axis] <= self.maximum[axis] for axis in range(3))

    def as_dict(self) -> dict[str, object]:
        return {"minimum": list(self.minimum), "maximum": list(self.maximum)}


@dataclass(frozen=True, slots=True)
class Regions:
    """The parts of the scene a case names, and what each one is for.

    Every one of these is chosen by the fixture and stated before any result
    exists. A region picked after the fact would be a threshold picked after
    the fact wearing a different hat.
    """

    #: The backing plane, for flatness. Truth there really is one plane.
    plane: Box
    #: Free space beside the depth step. Truth has NOTHING here, so a point in
    #: it is a surface bridging an edge that is really there.
    protected: Box
    #: A band across the depth step. Truth points here are the edge itself, and
    #: their coverage is how much of the edge survived.
    edge_band: Box
    #: The thin pillar. Small enough that a coarse voxel erases it, which is
    #: exactly the way a "smoother" surface can be worse.
    thin_feature: Box
    #: The object without its backing plane, for named dimensions.
    object_region: Box
    #: How close a point must be to count as covering a truth point.
    coverage_tolerance: float
    #: The pitch the exact truth model is decimated to, so the symmetric
    #: agreement below stays inside scanner3d.quality's stated work bound.
    truth_pitch: float
    #: Sub-rays per pixel, per axis, for the DENSE ANALYTIC reference: the
    #: window is intersected with the shapes in closed form at this many
    #: sub-samples across and down each pixel. It sets the density of the
    #: reference the analytic coverage is taken over, and nothing else - the
    #: analytic DISTANCES need no reference points at all. Owned by the fixture
    #: and stated before any result, because a reference somebody could thin
    #: after seeing a coverage number is not a reference.
    analytic_supersample: int
    #: Sub-triangles per edge for the TRIANGLE-INTERIOR lattice: an emitted
    #: triangle is sampled at the centroid of EVERY sub-triangle of its regular
    #: subdivision at this order - upright and inverted - which is
    #: ``order ** 2`` equal-area, strictly interior points per face. Equal area
    #: is what lets a sampled fraction be read as a fraction of the face's
    #: area. Owned by the fixture.
    triangle_sample_order: int
    #: The most interior samples one arm's surface may be scored at. Also owned
    #: by the fixture, and a ceiling rather than a target: an arm whose mesh
    #: needs more is refused as unscoreable, because a measurement that thinned
    #: itself to fit would be choosing which part of the surface to look at
    #: after seeing it.
    max_triangle_samples: int
    #: Points within this distance of each other are one component.
    linkage_radius: float
    #: A component with at most this many points is an isolated lump.
    isolated_max_points: int
    #: How far outside the fused volume a point may sit before it is refused.
    bounds_margin: float

    def as_dict(self) -> dict[str, object]:
        return {
            "plane": self.plane.as_dict(),
            "protected": self.protected.as_dict(),
            "edge_band": self.edge_band.as_dict(),
            "thin_feature": self.thin_feature.as_dict(),
            "object_region": self.object_region.as_dict(),
            "coverage_tolerance": self.coverage_tolerance,
            "truth_pitch": self.truth_pitch,
            "analytic_supersample": self.analytic_supersample,
            "triangle_sample_order": self.triangle_sample_order,
            "max_triangle_samples": self.max_triangle_samples,
            "linkage_radius": self.linkage_radius,
            "isolated_max_points": self.isolated_max_points,
            "bounds_margin": self.bounds_margin,
        }


# --- the scene ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewEvidence:
    """One view: its exact truth, its perturbed measurement, and two poses.

    Two poses because the difference between them IS the pose perturbation.
    ``true_pose`` is where the camera was and is used only to build truth;
    ``fused_pose`` is what a reconstruction is told and is the only pose an arm
    may use. A candidate that reaches for ``true_pose`` is refused.
    """

    index: int
    truth: bytes
    raw: bytes
    true_pose: Rigid
    fused_pose: Rigid

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def truth_sha256(self) -> str:
        return hashlib.sha256(self.truth).hexdigest()


@dataclass(frozen=True, slots=True)
class Scene:
    """One case of the corpus: exact truth, one perturbation, one parameter grid."""

    case_id: str
    window: Window
    intrinsics: IntrinsicsRecord
    depth_scale: float
    #: The shapes the truth IS, carried so the analytic reference can be taken
    #: against the object itself rather than against a rendering of it.
    shapes: tuple[Shape, ...]
    perturbation: Perturbation
    volume: VolumeSpec
    regions: Regions
    views: tuple[ViewEvidence, ...]
    grid: tuple[GridCell, ...]
    appliance: AppliancePolicy
    work_budget: int
    #: The case whose perturbation is ``exact`` and whose scene is otherwise
    #: identical. ``None`` only for the exact case itself.
    control_case: str | None

    def __post_init__(self) -> None:
        if len(self.views) < 3:
            raise ScoringUnusable(
                f"{self.case_id} has {len(self.views)} view(s). A multi-view oracle needs at "
                "least three, or 'the fusion disagrees with the dots' has no second opinion"
            )
        for position, view in enumerate(self.views):
            if view.index != position:
                raise ScoringUnusable(f"{self.case_id} view {position} calls itself {view.index}")
        if not self.grid:
            raise ScoringUnusable(f"{self.case_id} was given no parameter grid")
        if not self.shapes:
            raise ScoringUnusable(
                f"{self.case_id} names no shapes, so it has no analytic truth to be measured "
                "against and only a rendering of one"
            )
        check_shapes_are_separable(self.shapes)
        if any(cell.cell_id == APPLIANCE_CELL_ID for cell in self.grid):
            raise ScoringUnusable(
                f"{self.case_id} declares a grid cell called {APPLIANCE_CELL_ID!r}. That id "
                "belongs to the derived shipped-policy cell, and a fixture that claimed it "
                "could publish a voxel size of its own choosing under the appliance's name"
            )
        check_appliance_policy(self.appliance)
        check_perturbation_label(self.perturbation)
        if (self.perturbation.label == EXACT) != (self.control_case is None):
            raise ScoringUnusable(
                f"{self.case_id} is {self.perturbation.label!r} and names control "
                f"{self.control_case!r}. Exactly the exact case is its own control, and every "
                "perturbed case must name one"
            )
        if self.work_budget <= 0:
            raise ScoringUnusable(f"{self.case_id} states a work budget of {self.work_budget}")

    def cell(self, cell_id: str) -> GridCell:
        if cell_id == APPLIANCE_CELL_ID:
            return appliance_cell(self)
        for cell in self.grid:
            if cell.cell_id == cell_id:
                return cell
        raise CandidateRefused(
            "grid_cell_unknown",
            f"{self.case_id} was asked for cell {cell_id!r}; the corpus declares "
            f"{[cell.cell_id for cell in self.grid]} and derives {APPLIANCE_CELL_ID!r}",
        )

    def work_budget_for(self, cell_id: str | None) -> int:
        """The ceiling that binds this cell, which is not one number.

        The appliance cell is two orders of magnitude finer than the grid and
        would fail the corpus ceiling by construction, so it carries its own -
        declared in the same manifest, before any result, and published beside
        the spend it bounds.
        """
        if cell_id == APPLIANCE_CELL_ID:
            return self.appliance.work_budget
        return self.work_budget

    def raw_frames(self) -> tuple[bytes, ...]:
        return tuple(view.raw for view in self.views)

    def fused_poses(self) -> tuple[Rigid, ...]:
        return tuple(view.fused_pose for view in self.views)

    def raw_codes(self, index: int) -> tuple[int, ...]:
        return codes_of(self.views[index].raw, self.window)

    def truth_codes(self, index: int) -> tuple[int, ...]:
        return codes_of(self.views[index].truth, self.window)


# --- what an arm returns --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkClaim:
    """What an arm says it spent, declared BY the arm and checked here.

    Every field is bound to something :func:`validate` recomputes. The budget
    is the CASE's, published beside the spend so "bounded" is a stated fact
    rather than a wall-clock measurement that would differ on every machine.
    """

    claims: Mapping[str, int]
    work_budget: int

    def as_dict(self) -> dict[str, object]:
        return {
            "claims": {name: int(value) for name, value in sorted(self.claims.items())},
            "work_budget": self.work_budget,
        }


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm of one case: the geometry it produced and what it says it did."""

    name: str
    arm: str
    cell_id: str | None
    #: The view indices it used, in order, and the poses it placed them with.
    views_used: tuple[int, ...]
    poses_used: tuple[Rigid, ...]
    #: The SHA-256 of each frame it integrated or deprojected, in view order.
    input_sha256: tuple[str, ...]
    #: The output-facing geometry: measured points, or surface vertices.
    points: tuple[Point, ...]
    work: WorkClaim
    #: The emitted triangles, indexing :attr:`points`. Empty for the raw arm,
    #: which is a cloud and has no faces at all. Carried because a surface is
    #: not its vertices: the topology and the triangle interiors are measured
    #: from exactly these, and :func:`validate` binds them to what the
    #: extractor really produced.
    faces: tuple[tuple[int, int, int], ...] = ()
    #: Fusion bookkeeping this module re-derives for a fused arm, and ``None``
    #: for an arm that never allocated a volume.
    field_evidence: FieldRecord | None = None


#: An arm is a pure function of the scene it is given.
Candidate = Callable[[Scene], ArmResult]


# --- deriving the two arms, once, so a forgery has something to fail against ----------


@dataclass(frozen=True, slots=True)
class FieldRecord:
    """What one fusion actually did, re-derived here and never taken from an arm.

    A record rather than a mapping so every number has a declared type: an
    accounting field that could arrive as a string is an accounting field a
    reader has to guess about.
    """

    cells: int
    owned_bytes: int
    frames_offered: int
    frames_integrated: int
    observed_voxels: int
    unobserved_voxels: int
    surface_extracted: bool
    surface_vertices: int
    surface_faces: int
    #: The emitted triangles themselves, not merely how many there are. A face
    #: count cannot distinguish a mesh from one with the same number of
    #: differently connected triangles.
    faces: tuple[tuple[int, int, int], ...]
    #: Which backend produced this field. Published because one arm - the
    #: shipped-policy one - needs an accelerated one to exist at all.
    backend: str
    #: Where the FIELD's own evidence puts the zero level, with no marching
    #: cubes anywhere near it. This is the fusion half of the ablation: the
    #: extractor's vertices are scored against the same analytic truth as
    #: these points, so what they do not share is what extraction added.
    zero_level: tuple[Point, ...]
    edge_crossings: int
    exact_zero_voxels: int
    #: One voxel visit per integrated view: what the projective sweep costs,
    #: and exactly what a coarser grid buys. Deterministic, so it is comparable
    #: between machines in a way a wall clock is not.
    work_units: int


_FUSION_CACHE: dict[tuple[object, ...], tuple[tuple[Point, ...], FieldRecord]] = {}


def _fusion_key(
    scene: Scene, cell: GridCell, views: Sequence[int], backend: str
) -> tuple[object, ...]:
    """Everything the fused field is a function of, and nothing it is not.

    A memo key is a claim that two calls MUST produce the same field. The
    earlier key named the case, the cell id, the voxel size, the backend and
    the raw frame digests, which left the fused POSES, the intrinsics, the
    depth scale, the volume box and the cell's own truncation, minimum weight
    and weight policy outside it. Two fusions of the same frames at the same
    voxel size under different poses are different fields, so that key could
    hand one caller the other one's surface and its weights - which is exactly
    the identity :mod:`tests.shell_discriminant` binds its observable evidence
    to. Every reconstruction input is in the key now.
    """
    return (
        scene.case_id,
        cell.cell_id,
        cell.voxel_size,
        cell.truncation,
        cell.min_weight,
        cell.weight_policy,
        backend,
        tuple(views),
        tuple(scene.views[index].raw_sha256 for index in views),
        tuple(
            (scene.views[index].fused_pose.rotation, scene.views[index].fused_pose.translation)
            for index in views
        ),
        tuple(sorted(scene.intrinsics.values.items())),
        scene.depth_scale,
        (scene.window.u0, scene.window.v0, scene.window.width, scene.window.height),
        scene.volume.minimum,
        scene.volume.maximum,
        scene.volume.min_depth,
        scene.volume.max_depth,
    )


def fuse(
    scene: Scene,
    cell: GridCell,
    *,
    views: Sequence[int] | None = None,
    backend: str = BACKEND,
) -> tuple[tuple[Point, ...], FieldRecord]:
    """Integrate the scene's raw frames at ``cell`` and extract the surface.

    Memoised on the case, the cell, the view indices and the digests of the
    frames being integrated. The memo cannot launder a forgery: a forged arm
    that declares the same inputs is compared against exactly this result, and
    one that declares different inputs gets a different key.

    The field's own zero level is taken here too, from the same volume, before
    it is dropped. It costs one more bounded pass and it is the only way to say
    which of fusion and extraction a surface error belongs to.
    """
    indices = tuple(range(len(scene.views))) if views is None else tuple(views)
    key = _fusion_key(scene, cell, indices, backend)
    cached = _FUSION_CACHE.get(key)
    if cached is not None:
        return cached
    _, vertices, record = fuse_keeping_volume(scene, cell, views=indices, backend=backend)
    return vertices, record


def fuse_keeping_volume(
    scene: Scene,
    cell: GridCell,
    *,
    views: Sequence[int] | None = None,
    backend: str = BACKEND,
) -> tuple[TsdfVolume, tuple[Point, ...], FieldRecord]:
    """The same fusion, with the VOLUME handed back rather than dropped.

    :func:`fuse` is the arms' entry point and drops the field as soon as it has
    taken its bookkeeping, because a memo of whole volumes would be the largest
    thing in this process. Some evidence, though, is about the field itself
    rather than about the surface: :mod:`tests.shell_discriminant` reads the
    accumulated weight under each shell, and reading it out of a SECOND fusion
    would be evidence about a different field than the one the surface came
    from.

    So the work lives here and :func:`fuse` delegates to it. The result is
    memoised exactly as before - same key, same value - and the volume itself
    is not, so a caller that wants the field pays for it and nothing else in
    this corpus changes.
    """
    indices = tuple(range(len(scene.views))) if views is None else tuple(views)
    key = _fusion_key(scene, cell, indices, backend)
    volume = TsdfVolume(scene.volume.config(cell), backend=backend)
    volume.integrate(
        [scene.views[index].raw for index in indices],
        [scene.views[index].fused_pose for index in indices],
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )
    evidence = volume.evidence()
    vertices: tuple[Point, ...] = ()
    faces: tuple[tuple[int, int, int], ...] = ()
    extracted = True
    try:
        surface = extract_surface(volume, min_weight=cell.min_weight, backend=backend)
    except NoSurfaceFound:
        extracted = False
    else:
        vertices = tuple(surface.mesh.vertices)
        faces = tuple(surface.mesh.faces)
    zero = volume.zero_level_samples(min_weight=cell.min_weight)
    config = volume.config
    record = FieldRecord(
        cells=config.cell_count,
        owned_bytes=config.owned_bytes,
        frames_offered=evidence.frames_offered,
        frames_integrated=evidence.frames_integrated,
        observed_voxels=evidence.weights.observed_voxels,
        unobserved_voxels=evidence.weights.unobserved_voxels,
        surface_extracted=extracted,
        surface_vertices=len(vertices),
        surface_faces=len(faces),
        faces=faces,
        backend=volume.backend,
        zero_level=zero.points,
        edge_crossings=zero.edge_crossings,
        exact_zero_voxels=zero.exact_zero_voxels,
        work_units=config.cell_count * len(indices),
    )
    _FUSION_CACHE[key] = (vertices, record)
    return volume, vertices, record


def measured_points(scene: Scene, *, views: Sequence[int] | None = None) -> tuple[Point, ...]:
    """The unchanged measured dots: every returned sample, placed by the fused poses.

    Clipped to the case's fixed volume, and to nothing else. That box is the
    observed space both arms share: a TSDF cannot produce geometry outside the
    volume it was allocated, so comparing an unclipped cloud against a clipped
    surface would credit the dots for a region the fusion was never asked
    about. The clip is a property of the SCENE - stated in the fixture before
    any result exists - and never of a candidate, which is why the raw arm's
    own sample count is still gated: a candidate that drops a further sample of
    its own is refused by :func:`validate`.
    """
    indices = tuple(range(len(scene.views))) if views is None else tuple(views)
    out: list[Point] = []
    for index in indices:
        out.extend(
            point
            for point in cloud_of(
                scene.raw_codes(index),
                scene.window,
                intrinsics=scene.intrinsics,
                depth_scale=scene.depth_scale,
                pose=scene.views[index].fused_pose,
            )
            if scene.volume.contains(point, margin=0.0)
        )
    return tuple(out)


def truth_surface(scene: Scene) -> tuple[Point, ...]:
    """Every exact truth sample, placed by the EXACT poses and clipped to the volume.

    Undecimated. This is what the REGIONAL measures are taken against - named
    extents, the edge band, the thin feature, the protected volume - because a
    decimated reference would report an extent short by up to one lattice cell
    and a coverage denominator that had quietly lost its extremes. Only the
    symmetric agreement uses the decimated model below, and only because
    scanner3d.quality states a work bound that a full reference would exceed.
    """
    out: list[Point] = []
    for view in scene.views:
        out.extend(
            point
            for point in cloud_of(
                scene.truth_codes(view.index),
                scene.window,
                intrinsics=scene.intrinsics,
                depth_scale=scene.depth_scale,
                pose=view.true_pose,
            )
            if scene.volume.contains(point, margin=0.0)
        )
    return tuple(out)


def truth_points(scene: Scene) -> tuple[Point, ...]:
    """The exact truth surface, placed by the EXACT poses and decimated once.

    Decimated on a fixed lattice at the fixture's own pitch, keeping the first
    point that lands in each cell in a fixed traversal order, so the reference
    is a pure function of the scene. Decimation is what keeps the symmetric
    agreement inside :data:`scanner3d.quality.MAX_NEAREST_WORK` without anybody
    choosing which truth to leave out.
    """
    pitch = scene.regions.truth_pitch
    seen: dict[tuple[int, int, int], Point] = {}
    for view in scene.views:
        for point in cloud_of(
            scene.truth_codes(view.index),
            scene.window,
            intrinsics=scene.intrinsics,
            depth_scale=scene.depth_scale,
            pose=view.true_pose,
        ):
            if not scene.volume.contains(point, margin=0.0):
                continue
            key = (
                math.floor(point[0] / pitch),
                math.floor(point[1] / pitch),
                math.floor(point[2] / pitch),
            )
            if key not in seen:
                seen[key] = point
    return tuple(seen[key] for key in sorted(seen))


def decimate(points: Sequence[Point], pitch: float) -> tuple[Point, ...]:
    """The same fixed-lattice decimation, applied to an arm's own points.

    Both sides of every comparison go through it, at the same pitch, so a
    denser arm is not penalised for density and a sparser one is not flattered
    by it: what is compared is where the surface IS, and the count each arm
    actually produced is published separately and gated separately.
    """
    seen: dict[tuple[int, int, int], Point] = {}
    for point in points:
        key = (
            math.floor(point[0] / pitch),
            math.floor(point[1] / pitch),
            math.floor(point[2] / pitch),
        )
        if key not in seen:
            seen[key] = point
    return tuple(seen[key] for key in sorted(seen))


# --- the analytic truth, which does not know where the sensor sampled ------------------


def _plane_distance(point: Point, shape: Plane) -> float:
    normal = shape.normal
    length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if length <= 0.0:  # pragma: no cover - scanner3d.simulate refuses one at construction
        raise ScoringUnusable("a plane with no normal has no surface to measure against")
    return abs(sum(normal[axis] * point[axis] for axis in range(3)) - shape.offset) / length


def _sphere_distance(point: Point, shape: Sphere) -> float:
    centre = shape.centre
    radius = math.sqrt(sum((point[axis] - centre[axis]) ** 2 for axis in range(3)))
    return abs(radius - shape.radius)


def _box_distance(point: Point, shape: SimBox) -> float:
    """Exact unsigned distance to an oriented box's SURFACE, inside and out.

    Outside, it is the length of the per-axis overshoot, which is the standard
    closed form. Inside, it is the smallest distance to a face, which is the
    smallest remaining half-extent. Both are exact; neither is a bound.
    """
    local = shape.pose.inverse().apply(point)
    over = [abs(local[axis]) - shape.half_extents[axis] for axis in range(3)]
    if all(value <= 0.0 for value in over):
        return -max(over)
    return math.sqrt(sum(max(value, 0.0) ** 2 for value in over))


def analytic_distance(shapes: Sequence[Shape], point: Point) -> float:
    """The exact unsigned distance from ``point`` to the scene's own surface.

    Closed form, over the union of the declared shapes. Every shape in this
    corpus is pairwise disjoint from every other - the plane stands four units
    behind an object whose parts do not touch - so the distance to the boundary
    of the union is exactly the smallest distance to any one boundary, with no
    approximation and no marching. :func:`check_shapes_are_separable` refuses a
    scene where that stops being true rather than quietly reporting a bound.

    This is the reference that says whether geometry is CORRECT. It knows
    nothing about where a camera ray landed, which is the whole reason it
    exists: a mesh vertex interpolated between two rays is on the object, and a
    reference built out of the rays themselves cannot say so.
    """
    best = math.inf
    for shape in shapes:
        if isinstance(shape, Plane):
            distance = _plane_distance(point, shape)
        elif isinstance(shape, Sphere):
            distance = _sphere_distance(point, shape)
        elif isinstance(shape, SimBox):
            distance = _box_distance(point, shape)
        else:
            raise ScoringUnusable(
                f"no closed-form distance is implemented for {type(shape).__name__}; this "
                "oracle measures against analytic truth and will not approximate one"
            )
        best = min(best, distance)
    if not math.isfinite(best):  # pragma: no cover - Scene refuses an empty shape list
        raise ScoringUnusable("a scene with no shapes has no analytic surface")
    return best


def _box_axes(shape: SimBox) -> tuple[Point, Point, Point]:
    """The box's three unit axes in world coordinates: the columns of its rotation."""
    rotation = shape.pose.rotation
    return tuple(  # type: ignore[return-value]
        (rotation[0][axis], rotation[1][axis], rotation[2][axis]) for axis in range(3)
    )


def _dot(left: Point, right: Point) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _separated(first: Shape, second: Shape) -> bool:
    """Whether two shapes are strictly disjoint. Exact for the three kinds here."""
    if isinstance(second, Plane) and not isinstance(first, Plane):
        first, second = second, first
    if isinstance(first, Plane):
        if isinstance(second, Plane):
            # Two planes are either parallel and apart, or they cross. Neither
            # arises in this corpus, and guessing is not a check.
            return False
        normal = first.normal
        length = math.sqrt(_dot(normal, normal))
        unit: Point = (normal[0] / length, normal[1] / length, normal[2] / length)
        offset = first.offset / length
        if isinstance(second, Sphere):
            return abs(_dot(unit, second.centre) - offset) > second.radius
        assert isinstance(second, SimBox)
        centre = second.pose.apply((0.0, 0.0, 0.0))
        reach = math.fsum(
            second.half_extents[axis] * abs(_dot(unit, axis_vector))
            for axis, axis_vector in enumerate(_box_axes(second))
        )
        return abs(_dot(unit, centre) - offset) > reach
    if isinstance(first, Sphere) and isinstance(second, Sphere):
        gap = math.sqrt(sum((first.centre[i] - second.centre[i]) ** 2 for i in range(3)))
        return gap > first.radius + second.radius
    if isinstance(second, Sphere):
        first, second = second, first
    if isinstance(first, Sphere):
        assert isinstance(second, SimBox)
        local = second.pose.inverse().apply(first.centre)
        over = [abs(local[axis]) - second.half_extents[axis] for axis in range(3)]
        if all(value <= 0.0 for value in over):
            return False
        return math.sqrt(sum(max(value, 0.0) ** 2 for value in over)) > first.radius
    assert isinstance(first, SimBox)
    assert isinstance(second, SimBox)
    # The separating-axis theorem for two oriented boxes: fifteen candidate
    # axes, and one that separates them is a proof that they do not touch.
    first_axes, second_axes = _box_axes(first), _box_axes(second)
    delta_world = tuple(
        second.pose.apply((0.0, 0.0, 0.0))[axis] - first.pose.apply((0.0, 0.0, 0.0))[axis]
        for axis in range(3)
    )
    delta: Point = (delta_world[0], delta_world[1], delta_world[2])
    candidates: list[Point] = [*first_axes, *second_axes]
    for left in first_axes:
        for right in second_axes:
            cross: Point = (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
            if _dot(cross, cross) > 1e-18:
                candidates.append(cross)
    for axis_vector in candidates:
        reach = math.fsum(
            first.half_extents[index] * abs(_dot(axis_vector, vector))
            for index, vector in enumerate(first_axes)
        ) + math.fsum(
            second.half_extents[index] * abs(_dot(axis_vector, vector))
            for index, vector in enumerate(second_axes)
        )
        if abs(_dot(axis_vector, delta)) > reach:
            return True
    return False


def check_shapes_are_separable(shapes: Sequence[Shape]) -> None:
    """Refuse a scene whose shapes touch, because the union distance would be wrong.

    ``min`` over the individual boundary distances is the distance to the
    boundary of the union only while no shape's surface is buried inside
    another shape. On disjoint shapes it is exact everywhere, inside and out,
    so the check is that every pair is strictly disjoint - exactly, by the
    closed forms above and the separating-axis theorem, never by a bound.

    A scene that fails this is not a scene this module will score. Reporting a
    buried face as though a camera could have seen it would credit an arm for
    geometry inside another solid.
    """
    for index, shape in enumerate(shapes):
        for other_index in range(index + 1, len(shapes)):
            if not _separated(shape, shapes[other_index]):
                raise ScoringUnusable(
                    f"shapes {index} and {other_index} are not strictly disjoint. The analytic "
                    "distance in this module is the minimum over separable shapes, and on "
                    "overlapping ones it would report a surface buried inside another solid as "
                    "if it were on the outside of the object"
                )


_ANALYTIC_CACHE: dict[tuple[object, ...], tuple[Point, ...]] = {}


def analytic_reference(scene: Scene) -> tuple[Point, ...]:
    """A dense EXACT sampling of the scene's surface over the observed region.

    Each pixel of the window is split into ``analytic_supersample`` sub-rays per
    axis and every sub-ray is intersected with the shapes in CLOSED FORM, at the
    scene's exact poses. Two differences from :func:`truth_points`, and they
    are the point of the whole function:

    * the position is the real intersection, not a raw code rounded to the
      nearest depth step and not a lattice cell's first occupant; and
    * the sampling is ``supersample ** 2`` times denser than the sensor's, so
      the spacing between reference points is a small fraction of the coverage
      tolerance rather than comparable with it.

    It is used for the analytic COVERAGE - how much of the observed surface an
    arm actually reproduced. The analytic ACCURACY needs no reference points at
    all, because :func:`analytic_distance` is closed form.

    Bounded by construction: the window, the view count and the supersample
    factor are all declared in the fixture, and the cost is their product.
    """
    supersample = scene.regions.analytic_supersample
    if supersample < 1:
        raise ScoringUnusable(
            f"{scene.case_id} states analytic_supersample={supersample}; a reference sampled "
            "less than once per pixel is thinner than the sensor it is supposed to be denser "
            "than"
        )
    key = (scene.case_id, supersample, tuple(view.truth_sha256 for view in scene.views))
    cached = _ANALYTIC_CACHE.get(key)
    if cached is not None:
        return cached
    values = scene.intrinsics.values
    fx, fy = values["fx"], values["fy"]
    cx, cy = values["cx"], values["cy"]
    window = scene.window
    out: list[Point] = []
    for view in scene.views:
        placed = [shape.transformed(view.true_pose.inverse()) for shape in scene.shapes]
        for row in range(window.height):
            for sub_v in range(supersample):
                v = window.v0 + row + (sub_v + 0.5) / supersample - 0.5
                dy = (v - cy) / fy
                for column in range(window.width):
                    for sub_u in range(supersample):
                        u = window.u0 + column + (sub_u + 0.5) / supersample - 0.5
                        direction = ((u - cx) / fx, dy, 1.0)
                        nearest: float | None = None
                        for shape in placed:
                            hit = shape.intersect(direction)
                            if hit is not None and (nearest is None or hit < nearest):
                                nearest = hit
                        if nearest is None:
                            continue
                        camera = (
                            nearest * direction[0],
                            nearest * direction[1],
                            nearest * direction[2],
                        )
                        point = view.true_pose.apply(camera)
                        if scene.volume.contains(point, margin=0.0):
                            out.append(point)
    reference = tuple(out)
    if not reference:
        raise ScoringUnusable(
            f"{scene.case_id} produced no analytic reference points inside its own volume"
        )
    _ANALYTIC_CACHE[key] = reference
    return reference


# --- the shipped resolution policy, applied to this scene -------------------------------


_APPLIANCE_CACHE: dict[tuple[object, ...], tuple[GridCell, dict[str, object]]] = {}


def _appliance(scene: Scene) -> tuple[GridCell, dict[str, object]]:
    key = (
        scene.case_id,
        scene.appliance.as_dict()["voxel_budget"],
        tuple(view.raw_sha256 for view in scene.views),
    )
    cached = _APPLIANCE_CACHE.get(key)
    if cached is not None:
        return cached
    policy = scene.appliance
    resolution = choose_tsdf_resolution(measured_points(scene), voxel_budget=policy.voxel_budget)
    cell = GridCell(
        cell_id=APPLIANCE_CELL_ID,
        voxel_size=resolution.voxel_size,
        truncation_voxels=policy.truncation_voxels,
        min_weight=policy.min_weight,
        weight_policy=policy.weight_policy,
    )
    record: dict[str, object] = {
        **policy.as_dict(),
        "derived": resolution.as_dict(),
        "derived_from": (
            "this case's own measured points, placed by the poses a reconstruction is given "
            "and clipped to the case's fixed volume - the same cloud the raw-points arm "
            "publishes"
        ),
        "cell": cell.as_dict(),
    }
    _APPLIANCE_CACHE[key] = (cell, record)
    return cell, record


def appliance_cell(scene: Scene) -> GridCell:
    """The cell the SHIPPED policy chooses for this scene. Derived, never declared."""
    return _appliance(scene)[0]


def appliance_resolution(scene: Scene) -> dict[str, object]:
    """The whole derivation, for the artifact: policy, budget, and what it produced."""
    return _appliance(scene)[1]


def backend_for(scene: Scene, cell_id: str | None) -> str:
    """Which backend a cell runs on, and it is not a candidate's choice.

    Every declared grid cell runs on the pure-Python reference so two machines
    agree. The derived shipped-policy cell cannot: at about two million voxels
    it is thirty times :data:`scanner3d.tsdf.PYTHON_BACKEND_MAX_VOXELS`, and a
    corpus that refused to run it would be refusing to measure the only
    resolution an owner actually gets.
    """
    if cell_id == APPLIANCE_CELL_ID:
        return scene.appliance.backend
    return BACKEND


# --- refusing an arm whose account of itself does not hold ----------------------------


def validate(scene: Scene, result: ArmResult) -> None:
    """Refuse a result this oracle cannot stand behind. Fail closed.

    Every refusal here is a way one arm could look better than it is: fusing
    the truth instead of the measurement, quietly dropping the view that hurt,
    using the poses it was not given, editing the surface it claims came out of
    the field, or declaring a spend nothing supports.
    """
    if result.arm not in ARMS:
        raise CandidateRefused(
            "arm_unknown",
            f"{result.name} calls itself arm {result.arm!r}; this oracle re-derives "
            f"{list(ARMS)} and cannot check one it cannot reproduce",
        )
    expected_views = tuple(range(len(scene.views)))
    if result.views_used != expected_views:
        raise CandidateRefused(
            "views_omitted",
            f"{result.name} used views {list(result.views_used)} of {list(expected_views)}. "
            "Both arms are given the same frames or the comparison is between two questions",
        )
    fused_poses = scene.fused_poses()
    if len(result.poses_used) != len(fused_poses):
        raise CandidateRefused(
            "views_omitted",
            f"{result.name} placed {len(result.poses_used)} view(s) of {len(fused_poses)}",
        )
    for index, (used, given) in enumerate(zip(result.poses_used, fused_poses, strict=True)):
        if used.as_dict() != given.as_dict():
            raise CandidateRefused(
                "pose_substituted",
                f"{result.name} placed view {index} with a pose that is not the one the case "
                "gives a reconstruction. The pose error IS the perturbation, and an arm that "
                "reaches around it is measuring a different case",
            )
    expected_digests = tuple(view.raw_sha256 for view in scene.views)
    if result.input_sha256 != expected_digests:
        raise CandidateRefused(
            "raw_frame_edited",
            f"{result.name} declares input digests that are not the case's raw measured "
            "frames. The measurement is the evidence; an arm may not improve it before "
            "consuming it",
        )

    for index, point in enumerate(result.points):
        if not all(math.isfinite(value) for value in point):
            raise CandidateRefused(
                "point_not_finite",
                f"{result.name} point {index} is {point!r}, which could not be written to "
                "strict JSON or compared to anything",
            )
        if not scene.volume.contains(point, margin=scene.regions.bounds_margin):
            raise CandidateRefused(
                "points_outside_volume",
                f"{result.name} point {index} at {point!r} is outside the case's fixed volume "
                f"by more than {scene.regions.bounds_margin}. Observed space is bounded by the "
                "volume every arm shares, and geometry outside it was never measured",
            )

    if result.arm == RAW_POINTS:
        if result.cell_id is not None:
            raise CandidateRefused(
                "grid_cell_unknown",
                f"{result.name} is the raw-points arm and names grid cell {result.cell_id!r}. "
                "The dots are not fused and no parameter cell produced them",
            )
        expected_points = measured_points(scene)
        if len(result.points) != len(expected_points):
            raise CandidateRefused(
                "sample_count_mismatch",
                f"{result.name} offers {len(result.points)} measured point(s) where the case's "
                f"own frames deproject to {len(expected_points)}. A raw arm that keeps only the "
                "easy samples has changed its own denominator",
            )
        if result.points != expected_points:
            raise CandidateRefused(
                "points_not_measured",
                f"{result.name} offers points that are not this case's deprojected measurements",
            )
        if result.faces:
            raise CandidateRefused(
                "surface_not_derived",
                f"{result.name} is the raw-points arm and offers {len(result.faces)} face(s). "
                "The dots are a cloud; nothing meshed them, and a triangle nobody extracted is "
                "geometry an arm brought with it",
            )
        derived: FieldRecord | None = None
    else:
        if result.cell_id is None:
            raise CandidateRefused(
                "grid_cell_unknown",
                f"{result.name} is a fused arm and names no parameter cell",
            )
        cell = scene.cell(result.cell_id)
        vertices, derived = fuse(scene, cell, backend=backend_for(scene, result.cell_id))
        if result.points != vertices:
            raise CandidateRefused(
                "surface_not_derived",
                f"{result.name} offers {len(result.points)} vertex/vertices that are not the "
                f"ones {cell.cell_id} actually extracts from this case's own frames "
                f"({len(vertices)}). A surface is what the field produced, not what an arm "
                "would like it to have produced",
            )
        if result.faces != derived.faces:
            raise CandidateRefused(
                "surface_not_derived",
                f"{result.name} offers {len(result.faces)} triangle(s) where {cell.cell_id} "
                f"emits {len(derived.faces)}, or emits the same number differently connected. "
                "This is the forgery a vertex-only reader cannot see: every vertex can be "
                "exactly where the extractor put it while the surface between them spans a gap "
                "the field never crossed",
            )

    _validate_claims(scene, result, derived)


def _validate_claims(scene: Scene, result: ArmResult, derived: FieldRecord | None) -> None:
    """Bind every claim an arm made to a number this module computed."""
    claims = result.work.claims
    unknown = sorted(set(claims) - set(CLAIMS))
    if unknown:
        raise CandidateRefused(
            "claim_unknown",
            f"{result.name} claims {unknown}, and this oracle checks only {list(CLAIMS)}. A "
            "claim nobody can check is a threshold an arm wrote for itself",
        )
    supported: dict[str, int] = {
        "point_count": len(result.points),
        "frames_fused": len(result.views_used),
    }
    if derived is not None:
        supported["observed_voxels"] = derived.observed_voxels
        supported["surface_vertices"] = derived.surface_vertices
        supported["surface_faces"] = derived.surface_faces
        supported["work_units"] = derived.work_units
    for name, stated in claims.items():
        if name not in supported:
            raise CandidateRefused(
                "claim_not_supported",
                f"{result.name} claims {name}={stated} on the {result.arm} arm, which does not "
                "produce that quantity at all",
            )
        if int(stated) != supported[name]:
            raise CandidateRefused(
                "claim_not_supported",
                f"{result.name} claims {name}={stated}; this oracle counted "
                f"{supported[name]} from its own re-derivation",
            )
    budget = scene.work_budget_for(result.cell_id)
    if result.work.work_budget != budget:
        raise CandidateRefused(
            "claim_not_supported",
            f"{result.name} publishes work budget {result.work.work_budget} where the case "
            f"owns {budget} for cell {result.cell_id!r}. The ceiling belongs to the fixture",
        )
    spent = supported.get("work_units", 0)
    if spent > budget:
        raise CandidateRefused(
            "work_budget_exceeded",
            f"{result.name} spends {spent} voxel-view unit(s) against the case's declared "
            f"ceiling of {budget}. The ceiling is owned by the fixture and was set "
            "before any result existed",
        )


# --- the measurements ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldEvidence:
    """What the field and the extractor actually did, and what it cost.

    Present only for a fused arm. The raw-points arm has no field, and its
    gates are recorded as inapplicable rather than as passing: "there was no
    volume" and "the volume was fine" are different facts.
    """

    name: str
    cell_id: str
    voxel_size: float
    truncation: float
    truncation_in_voxels: float
    min_weight: float
    weight_policy: str
    cells: int
    observed_voxels: int
    unobserved_voxels: int
    owned_bytes: int
    frames_offered: int
    frames_integrated: int
    surface_extracted: bool
    surface_vertices: int
    surface_faces: int
    backend: str
    zero_level_points: int
    edge_crossings: int
    exact_zero_voxels: int
    work_units: int
    work_budget: int

    @property
    def kind(self) -> str:
        return "surface_oracle_field"

    @property
    def within_budget(self) -> bool:
        return self.work_units <= self.work_budget

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "cell_id": self.cell_id,
            "voxel_size": self.voxel_size,
            "truncation": self.truncation,
            "truncation_in_voxels": self.truncation_in_voxels,
            "min_weight": self.min_weight,
            "weight_policy": self.weight_policy,
            "cells": self.cells,
            "observed_voxels": self.observed_voxels,
            "unobserved_voxels": self.unobserved_voxels,
            "owned_bytes": self.owned_bytes,
            "frames_offered": self.frames_offered,
            "frames_integrated": self.frames_integrated,
            "surface_extracted": self.surface_extracted,
            "surface_vertices": self.surface_vertices,
            "surface_faces": self.surface_faces,
            "backend": self.backend,
            "zero_level_points": self.zero_level_points,
            "edge_crossings": self.edge_crossings,
            "exact_zero_voxels": self.exact_zero_voxels,
            "work_units": self.work_units,
            "work_budget": self.work_budget,
            "within_budget": self.within_budget,
            "unit": VOXEL_UNIT,
        }


@dataclass(frozen=True, slots=True)
class GeometryEvidence:
    """The output-facing point set: how much there is and where it is allowed to be.

    ``point_count`` beside the agreement is what stops "fewer points" reading
    as "a better surface": an arm that kept a tenth of the geometry and put all
    of it in the right place is visible here as exactly that.
    """

    name: str
    arm: str
    point_count: int
    compared_points: int
    truth_points: int
    outside_volume: int
    extent_x: float
    extent_y: float
    extent_z: float

    @property
    def kind(self) -> str:
        return "surface_oracle_geometry"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "arm": self.arm,
            "point_count": self.point_count,
            "compared_points": self.compared_points,
            "truth_points": self.truth_points,
            "outside_volume": self.outside_volume,
            "extent_x": self.extent_x,
            "extent_y": self.extent_y,
            "extent_z": self.extent_z,
            "unit": DEPTH_UNIT,
        }


@dataclass(frozen=True, slots=True)
class EdgeEvidence:
    """Bridging and erosion at the hard depth step, and the thin feature's survival.

    Three numbers that move in opposite directions on purpose. Smoothing the
    step bridges it (``bridged_points`` rises); a truncation or minimum weight
    that refuses to bridge erodes it (``edge_coverage`` falls); and a voxel
    coarse enough to do neither erases the pillar (``thin_feature_coverage``
    falls). No single parameter makes all three better, which is why they are
    published together.
    """

    name: str
    bridged_points: int
    protected_truth_points: int
    edge_truth_points: int
    edge_covered_points: int
    edge_coverage: float
    thin_feature_truth_points: int
    thin_feature_covered_points: int
    thin_feature_coverage: float

    @property
    def kind(self) -> str:
        return "surface_oracle_edge"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "bridged_points": self.bridged_points,
            "protected_truth_points": self.protected_truth_points,
            "edge_truth_points": self.edge_truth_points,
            "edge_covered_points": self.edge_covered_points,
            "edge_coverage": self.edge_coverage,
            "thin_feature_truth_points": self.thin_feature_truth_points,
            "thin_feature_covered_points": self.thin_feature_covered_points,
            "thin_feature_coverage": self.thin_feature_coverage,
            "unit": FRACTION_UNIT,
        }


@dataclass(frozen=True, slots=True)
class ComponentEvidence:
    """Connected components of the output geometry, and the lumps among them.

    This is the measurement the user's complaint is actually about. A surface
    that has broken into a main body plus a scatter of small blobs reads, to a
    person looking at it, as "lumpy" - and every distance summary above can
    stay respectable while it happens, because the lumps are near the surface
    they broke off.
    """

    name: str
    linkage_radius: float
    component_count: int
    largest_component_points: int
    largest_component_fraction: float
    isolated_components: int
    isolated_points: int
    isolated_fraction: float

    @property
    def kind(self) -> str:
        return "surface_oracle_components"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "linkage_radius": self.linkage_radius,
            "component_count": self.component_count,
            "largest_component_points": self.largest_component_points,
            "largest_component_fraction": self.largest_component_fraction,
            "isolated_components": self.isolated_components,
            "isolated_points": self.isolated_points,
            "isolated_fraction": self.isolated_fraction,
            "unit": FRACTION_UNIT,
        }


def _within(points: Sequence[Point], box: Box) -> tuple[Point, ...]:
    return tuple(point for point in points if box.contains(point))


@dataclass(frozen=True, slots=True)
class AnalyticEvidence:
    """Geometry against the object ITSELF, in closed form. Independent truth.

    This is the measurement the earlier rounds of this corpus did not have. It
    is taken against :func:`analytic_distance` - the exact distance to the
    boundary of the union of the scene's declared shapes - so it does not know
    where a camera ray landed and cannot be biased by the ray pitch, the raw
    code quantisation or the lattice anything was decimated onto.

    Read it BESIDE the ``support`` measurement, never instead of it. A point on
    a face no view could see is accurate here and unsupported there, and both
    of those are true at once.
    """

    name: str
    arm: str
    #: The arm's own points, all of them and undecimated: the distance is
    #: closed form and needs no neighbour search, so nothing is thinned to
    #: afford it.
    scored_points: int
    distances: quality.Distribution
    within_tolerance_points: int
    within_tolerance_fraction: float
    #: The dense exact sampling of the same surface, used for coverage only.
    reference_points: int
    reference_covered_points: int
    reference_coverage_fraction: float
    coverage_tolerance: float
    supersample: int

    @property
    def kind(self) -> str:
        return "surface_oracle_analytic"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "arm": self.arm,
            "scored_points": self.scored_points,
            "distances": self.distances.as_dict(),
            "within_tolerance_points": self.within_tolerance_points,
            "within_tolerance_fraction": self.within_tolerance_fraction,
            "reference_points": self.reference_points,
            "reference_covered_points": self.reference_covered_points,
            "reference_coverage_fraction": self.reference_coverage_fraction,
            "coverage_tolerance": self.coverage_tolerance,
            "supersample": self.supersample,
            "truth_source": ANALYTIC_SOURCE,
            "unit": DEPTH_UNIT,
        }


@dataclass(frozen=True, slots=True)
class AblationEvidence:
    """Fusion, or extraction? The same field, scored twice against the same truth.

    ``field_*`` is :meth:`scanner3d.tsdf.TsdfVolume.zero_level_samples` - where
    the FIELD's own evidence puts the zero level, with no marching cubes
    anywhere near it. ``surface_*`` is what the extractor emitted from that same
    field at the same minimum weight. Both are measured against the analytic
    truth by the same code, so:

    * what they SHARE is fusion: discretisation, weighting, pose and depth;
    * what the surface has and the field does not is EXTRACTION: the case
      table, the interpolation and the topology it closes.

    The deltas are published rather than a verdict. A ``surface`` that is worse
    than its own field is extraction adding error; one that matches it says the
    extractor reproduced the field it was given, which is a finding and not a
    pass.
    """

    name: str
    cell_id: str
    min_weight: float
    field_points: int
    surface_points: int
    field_distances: quality.Distribution
    surface_distances: quality.Distribution
    extraction_p95_delta: float
    extraction_worst_delta: float
    field_within_tolerance_fraction: float
    surface_within_tolerance_fraction: float
    extraction_within_tolerance_delta: float
    field_points_in_protected: int
    surface_points_in_protected: int
    #: Vertices with no zero-level sample of the same field within the coverage
    #: tolerance: geometry the extractor put somewhere the field's own evidence
    #: did not. This is extraction acting alone, counted rather than argued.
    surface_points_without_field_support: int
    surface_points_without_field_support_fraction: float

    @property
    def kind(self) -> str:
        return "surface_oracle_ablation"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "cell_id": self.cell_id,
            "min_weight": self.min_weight,
            "field_points": self.field_points,
            "surface_points": self.surface_points,
            "field_distances": self.field_distances.as_dict(),
            "surface_distances": self.surface_distances.as_dict(),
            "extraction_p95_delta": self.extraction_p95_delta,
            "extraction_worst_delta": self.extraction_worst_delta,
            "field_within_tolerance_fraction": self.field_within_tolerance_fraction,
            "surface_within_tolerance_fraction": self.surface_within_tolerance_fraction,
            "extraction_within_tolerance_delta": self.extraction_within_tolerance_delta,
            "field_points_in_protected": self.field_points_in_protected,
            "surface_points_in_protected": self.surface_points_in_protected,
            "surface_points_without_field_support": self.surface_points_without_field_support,
            "surface_points_without_field_support_fraction": (
                self.surface_points_without_field_support_fraction
            ),
            "measured_against": ANALYTIC_SOURCE,
            "unit": DEPTH_UNIT,
        }


@dataclass(frozen=True, slots=True)
class SurfaceSampleEvidence:
    """The emitted surface measured AS A SURFACE, inside the triangles.

    Every other geometric measurement in this module scores a point list, and
    for a fused arm that point list is the extractor's vertices. Those vertices
    sit on the same field-edge zero crossings
    :meth:`scanner3d.tsdf.TsdfVolume.zero_level_samples` reports, so no
    comparison between them can find the defect a person actually sees: a
    triangle whose three corners are all in sensible places and whose INTERIOR
    crosses a gap the scene really has.

    ``bridging_faces_with_every_vertex_outside`` is that defect, counted. It is
    the number this measurement exists for, and it is published beside
    ``edge.bridged_points`` - the vertex-only count - rather than instead of it,
    because the two answer different questions and the difference between them
    is exactly what the vertex evidence was blind to.
    """

    name: str
    arm: str
    faces: int
    samples_per_face: int
    sample_order: int
    sampled_points: int
    sample_ceiling: int
    #: Analytic distance at every interior sample. Closed form, so a valid
    #: triangle interpolated between two camera rays scores near zero here and
    #: a triangle spanning free space does not.
    distances: quality.Distribution
    within_tolerance_points: int
    within_tolerance_fraction: float
    surface_area: float
    protected_samples: int
    #: The area of emitted surface standing in free space, inferred from the
    #: samples that landed there. The inference is only available because the
    #: lattice is equal-area - see :func:`barycentric_lattice` - so a face's
    #: sampled fraction is an unbiased estimate of its area fraction.
    protected_area: float
    bridging_faces: int
    bridging_faces_with_every_vertex_outside: int
    #: Interior samples with no ideal return within the coverage tolerance.
    #: MEASUREMENT SUPPORT, not accuracy - see :data:`SUPPORT_SOURCE`.
    unsupported_samples: int
    unsupported_fraction: float
    support_tolerance: float

    @property
    def kind(self) -> str:
        return "surface_oracle_surface_samples"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "arm": self.arm,
            "faces": self.faces,
            "samples_per_face": self.samples_per_face,
            "sample_order": self.sample_order,
            "sampled_points": self.sampled_points,
            "sample_ceiling": self.sample_ceiling,
            "distances": self.distances.as_dict(),
            "within_tolerance_points": self.within_tolerance_points,
            "within_tolerance_fraction": self.within_tolerance_fraction,
            "surface_area": self.surface_area,
            "protected_samples": self.protected_samples,
            "protected_area": self.protected_area,
            "bridging_faces": self.bridging_faces,
            "bridging_faces_with_every_vertex_outside": (
                self.bridging_faces_with_every_vertex_outside
            ),
            "unsupported_samples": self.unsupported_samples,
            "unsupported_fraction": self.unsupported_fraction,
            "support_tolerance": self.support_tolerance,
            "sampled_from": SURFACE_SAMPLE_SOURCE,
            "measured_against": ANALYTIC_SOURCE,
            "unit": DEPTH_UNIT,
        }


@dataclass(frozen=True, slots=True)
class TopologyEvidence:
    """What the faces ARE, rather than where they are. Connectivity, not distance.

    Built on :func:`scanner3d.mesh.edge_census`, which is the product's own
    counting rather than a second implementation of it, plus the shells the
    face graph breaks into.

    ``unsupported_closed_shells`` is the second defect vertex evidence cannot
    see: a shell every edge of which is used twice - a sealed, watertight blob -
    standing where no view measured anything. Every one of its vertices is an
    ordinary zero crossing of the field, every one of its distances can be
    unremarkable, and it is still a solid the scan invented.
    ``components``, which links the arm's POINTS by radius, cannot tell such a
    shell from a legitimately separate body; this can, because it knows which
    triangles close.

    And an unsupported shell is still not, by itself, a false one. Support and
    truth were separated at the top of this module for a reason, and collapsing
    them again here would undo it: a sealed shell standing on a face no view
    could see, or between two camera rays, is unsupported MEASUREMENT and
    correct GEOMETRY at the same time. So each unsupported closed shell is
    scored a second time against the analytic surface - at its own vertices and
    inside its own triangles - and published in two disjoint groups:

    * ``unsupported_inaccurate_closed_shells`` is false surface. Nothing
      measured near it AND it is not on the object. This is the number a
      pruning change would be entitled to remove;
    * ``unsupported_accurate_closed_shells`` is correct surface a
      support-only rule would delete. Its face count is the COST of such a
      rule, measured on the same inputs, and it is published beside the benefit
      rather than left for a reader to assume it is zero.

    The two are disjoint and sum to ``unsupported_closed_shells``, so no shell
    is counted in both and none is dropped.
    """

    name: str
    arm: str
    vertices: int
    referenced_vertices: int
    faces: int
    degenerate_faces: int
    edge_count: int
    boundary_edge_count: int
    shared_edge_count: int
    nonmanifold_edge_count: int
    reversed_edge_count: int
    closed: bool
    consistently_oriented: bool
    euler_characteristic: int
    shell_count: int
    largest_shell_faces: int
    largest_shell_fraction: float
    closed_shells: int
    unsupported_closed_shells: int
    unsupported_closed_shell_faces: int
    #: Unsupported AND off the object: false surface, and the whole of what a
    #: pruning rule could honestly claim to remove.
    unsupported_inaccurate_closed_shells: int
    unsupported_inaccurate_closed_shell_faces: int
    #: The worst analytic distance any of those shells reaches, over its
    #: vertices and its sampled triangle interiors together. ``0.0`` when there
    #: are none.
    unsupported_inaccurate_worst_distance: float
    #: Unsupported and ON the object: correct geometry nothing happened to
    #: measure. A support-only prune would delete every one of these faces.
    unsupported_accurate_closed_shells: int
    unsupported_accurate_closed_shell_faces: int
    unsupported_accurate_worst_distance: float
    #: The bar the split is made at, and the reference it is made against.
    accuracy_tolerance: float
    support_tolerance: float

    @property
    def kind(self) -> str:
        return "surface_oracle_topology"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "arm": self.arm,
            "vertices": self.vertices,
            "referenced_vertices": self.referenced_vertices,
            "faces": self.faces,
            "degenerate_faces": self.degenerate_faces,
            "edge_count": self.edge_count,
            "boundary_edge_count": self.boundary_edge_count,
            "shared_edge_count": self.shared_edge_count,
            "nonmanifold_edge_count": self.nonmanifold_edge_count,
            "reversed_edge_count": self.reversed_edge_count,
            "closed": self.closed,
            "consistently_oriented": self.consistently_oriented,
            "euler_characteristic": self.euler_characteristic,
            "shell_count": self.shell_count,
            "largest_shell_faces": self.largest_shell_faces,
            "largest_shell_fraction": self.largest_shell_fraction,
            "closed_shells": self.closed_shells,
            "unsupported_closed_shells": self.unsupported_closed_shells,
            "unsupported_closed_shell_faces": self.unsupported_closed_shell_faces,
            "unsupported_inaccurate_closed_shells": self.unsupported_inaccurate_closed_shells,
            "unsupported_inaccurate_closed_shell_faces": (
                self.unsupported_inaccurate_closed_shell_faces
            ),
            "unsupported_inaccurate_worst_distance": self.unsupported_inaccurate_worst_distance,
            "unsupported_accurate_closed_shells": self.unsupported_accurate_closed_shells,
            "unsupported_accurate_closed_shell_faces": (
                self.unsupported_accurate_closed_shell_faces
            ),
            "unsupported_accurate_worst_distance": self.unsupported_accurate_worst_distance,
            "accuracy_tolerance": self.accuracy_tolerance,
            "support_tolerance": self.support_tolerance,
            "support_source": SUPPORT_SOURCE,
            "shell_accuracy_source": ANALYTIC_SOURCE,
            "unit": COUNT_UNIT,
        }


def barycentric_lattice(order: int) -> tuple[tuple[float, float, float], ...]:
    """The fixed, strictly interior pattern one triangle is sampled at.

    Subdivide the triangle regularly into ``order ** 2`` sub-triangles and take
    the centroid of EVERY one of them - the ``order * (order + 1) / 2`` upright
    ones and the ``order * (order - 1) / 2`` inverted ones - which is
    ``order ** 2`` points. Every barycentric coordinate is strictly positive
    and they sum to one, so no sample is ever a vertex or an edge point: the
    corners are already scored by every other measurement here, and a lattice
    that included them would be re-reporting them under a new name.

    The inverted sub-triangles are not decoration. A regular subdivision at
    order ``n`` partitions the triangle into ``n ** 2`` sub-triangles of
    IDENTICAL area, so one sample per sub-triangle makes the sampled fraction
    of a face an equal-area estimate of the fraction of that face's AREA, which
    is the inference :func:`surface_samples` makes when it reports
    ``protected_area``. The upright half alone covers only
    ``(order + 1) / (2 * order)`` of the area - two thirds at order three - and
    leaves the inverted sub-triangles entirely unsampled, so an area inferred
    from it would be an inference the sampler cannot make.

    Deterministic and closed form: no random barycentric draw, no area
    weighting that would sample a big triangle more finely than a small one,
    and no dependence on the order the extractor happened to emit faces in.
    """
    if order < 1:
        raise ScoringUnusable(
            f"a triangle sample order of {order} takes no sample inside any triangle at all; "
            "the order is the fixture's and must be at least one"
        )
    out: list[tuple[float, float, float]] = []
    for i in range(order):
        for j in range(order - i):
            u = (i + 1.0 / 3.0) / order
            v = (j + 1.0 / 3.0) / order
            out.append((u, v, 1.0 - u - v))
        for j in range(order - i - 1):
            u = (i + 2.0 / 3.0) / order
            v = (j + 2.0 / 3.0) / order
            out.append((u, v, 1.0 - u - v))
    return tuple(out)


def _triangle_area(a: Point, b: Point, c: Point) -> float:
    """The exact area of one triangle: half the magnitude of the edge cross product."""
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def triangle_interior_samples(
    vertices: Sequence[Point],
    faces: Sequence[tuple[int, int, int]],
    *,
    order: int,
    ceiling: int,
) -> tuple[tuple[int, Point], ...]:
    """``(face index, point)`` for every interior sample, in face then lattice order.

    Bounded before anything is built: the count is ``len(faces)`` times the
    lattice size, and an arm whose mesh needs more than ``ceiling`` samples is
    refused as unscoreable rather than sampled more thinly. A measurement that
    quietly reduced its own density on the meshes that produced the most
    triangles would be choosing what to look at after seeing it.
    """
    lattice = barycentric_lattice(order)
    needed = len(faces) * len(lattice)
    if needed > ceiling:
        raise ScoringUnusable(
            f"scoring {len(faces)} triangle(s) at order {order} needs {needed} interior "
            f"sample(s) against the fixture's ceiling of {ceiling}. The ceiling is owned by the "
            "case and was stated before any result existed"
        )
    out: list[tuple[int, Point]] = []
    for index, (first, second, third) in enumerate(faces):
        a, b, c = vertices[first], vertices[second], vertices[third]
        for u, v, w in lattice:
            out.append(
                (
                    index,
                    (
                        u * a[0] + v * b[0] + w * c[0],
                        u * a[1] + v * b[1] + w * c[1],
                        u * a[2] + v * b[2] + w * c[2],
                    ),
                )
            )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class SurfaceScoring:
    """One scoring pass over an arm's mesh, shared by both surface measurements.

    :func:`surface_samples` reports the distribution of these distances;
    :func:`surface_topology` asks which shell each of them belongs to. Taking
    them once means the two cannot disagree about where the surface is or how
    far from the object it is, and the work the fixture's ceiling bounds is
    paid once rather than three times.
    """

    #: ``(face index, point)`` for every interior sample, in face then lattice
    #: order.
    samples: tuple[tuple[int, Point], ...]
    #: The analytic distance at each of those samples, in the same order.
    sample_distances: tuple[float, ...]
    #: The analytic distance at each of the arm's vertices, in its own order.
    vertex_distances: tuple[float, ...]


def score_surface(scene: Scene, result: ArmResult) -> SurfaceScoring:
    """Sample an arm's mesh once, at the fixture's own order and under its own ceiling."""
    samples = triangle_interior_samples(
        result.points,
        result.faces,
        order=scene.regions.triangle_sample_order,
        ceiling=scene.regions.max_triangle_samples,
    )
    return SurfaceScoring(
        samples=samples,
        sample_distances=tuple(analytic_distance(scene.shapes, point) for _, point in samples),
        vertex_distances=tuple(analytic_distance(scene.shapes, point) for point in result.points),
    )


def surface_samples(
    scene: Scene,
    result: ArmResult,
    *,
    support: Sequence[Point],
    scoring: SurfaceScoring | None = None,
) -> SurfaceSampleEvidence:
    """Score the inside of every emitted triangle against analytic truth and the gap."""
    regions = scene.regions
    lattice = len(barycentric_lattice(regions.triangle_sample_order))
    scoring = score_surface(scene, result) if scoring is None else scoring
    samples = scoring.samples
    points = [point for _, point in samples]
    distances = scoring.sample_distances
    tolerance = regions.coverage_tolerance
    within = sum(1 for distance in distances if distance <= tolerance)
    supported = _covered_mask(points, support, tolerance)

    protected_per_face: dict[int, int] = {}
    for face_index, point in samples:
        if regions.protected.contains(point):
            protected_per_face[face_index] = protected_per_face.get(face_index, 0) + 1

    area = 0.0
    protected_area = 0.0
    vertex_free = 0
    for index, (first, second, third) in enumerate(result.faces):
        a, b, c = result.points[first], result.points[second], result.points[third]
        face_area = _triangle_area(a, b, c)
        area += face_area
        inside = protected_per_face.get(index, 0)
        if inside:
            protected_area += face_area * inside / lattice
            if not any(regions.protected.contains(corner) for corner in (a, b, c)):
                vertex_free += 1

    return SurfaceSampleEvidence(
        name="surface_samples",
        arm=result.arm,
        faces=len(result.faces),
        samples_per_face=lattice,
        sample_order=regions.triangle_sample_order,
        sampled_points=len(points),
        sample_ceiling=regions.max_triangle_samples,
        distances=quality._distribution(
            distances, unit=DEPTH_UNIT, what=f"{result.name} triangle-interior distances"
        ),
        within_tolerance_points=within,
        within_tolerance_fraction=within / len(points),
        surface_area=area,
        protected_samples=sum(protected_per_face.values()),
        protected_area=protected_area,
        bridging_faces=len(protected_per_face),
        bridging_faces_with_every_vertex_outside=vertex_free,
        unsupported_samples=len(points) - sum(supported),
        unsupported_fraction=(len(points) - sum(supported)) / len(points),
        support_tolerance=tolerance,
    )


def _shells(faces: Sequence[tuple[int, int, int]]) -> list[list[int]]:
    """Face indices grouped into connected shells, largest first, then lowest index.

    Two faces are in one shell when they share an undirected edge. Union-find
    over a deterministic traversal, so the same mesh always yields the same
    grouping in the same order.
    """
    parent = list(range(len(faces)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owner: dict[tuple[int, int], int] = {}
    for index, (a, b, c) in enumerate(faces):
        for start, end in ((a, b), (b, c), (c, a)):
            key = (start, end) if start < end else (end, start)
            other = owner.get(key)
            if other is None:
                owner[key] = index
                continue
            left, right = find(index), find(other)
            if left != right:
                parent[max(left, right)] = min(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(faces)):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda members: (-len(members), members[0]))


def surface_topology(
    scene: Scene,
    result: ArmResult,
    *,
    support: Sequence[Point],
    scoring: SurfaceScoring | None = None,
) -> TopologyEvidence:
    """Manifoldness, closure, shells, and how many of those shells nothing measured."""
    faces = result.faces
    census = edge_census(faces)
    degenerate = sum(
        1
        for a, b, c in faces
        if len({a, b, c}) < 3
        or _triangle_area(result.points[a], result.points[b], result.points[c]) <= 0.0
    )
    referenced = {index for face in faces for index in face}
    tolerance = scene.regions.coverage_tolerance
    supported = _covered_mask(result.points, support, tolerance)
    # The second reference, and the one the support mask cannot stand in for.
    # A shell is scored where it actually is - at its corners AND inside its
    # triangles - so a sealed blob whose vertices happen to sit on the object
    # while its interior does not is not read as correct geometry.
    scoring = score_surface(scene, result) if scoring is None else scoring
    vertex_distance = scoring.vertex_distances
    face_distance = [0.0] * len(faces)
    for (face_index, _), distance in zip(scoring.samples, scoring.sample_distances, strict=True):
        if distance > face_distance[face_index]:
            face_distance[face_index] = distance
    shells = _shells(faces)
    closed_shells = 0
    unsupported_shells = 0
    unsupported_faces = 0
    false_shells = 0
    false_faces = 0
    false_worst = 0.0
    correct_shells = 0
    correct_faces = 0
    correct_worst = 0.0
    for members in shells:
        shell_faces = [faces[index] for index in members]
        if not edge_census(shell_faces).closed:
            continue
        closed_shells += 1
        corners = {index for face in shell_faces for index in face}
        if any(supported[index] for index in corners):
            continue
        unsupported_shells += 1
        unsupported_faces += len(members)
        worst = max(
            max(vertex_distance[index] for index in corners),
            max(face_distance[index] for index in members),
        )
        if worst > tolerance:
            false_shells += 1
            false_faces += len(members)
            false_worst = max(false_worst, worst)
        else:
            correct_shells += 1
            correct_faces += len(members)
            correct_worst = max(correct_worst, worst)
    return TopologyEvidence(
        name="topology",
        arm=result.arm,
        vertices=len(result.points),
        referenced_vertices=len(referenced),
        faces=len(faces),
        degenerate_faces=degenerate,
        edge_count=census.edge_count,
        boundary_edge_count=census.boundary_edge_count,
        shared_edge_count=census.shared_edge_count,
        nonmanifold_edge_count=census.nonmanifold_edge_count,
        reversed_edge_count=census.reversed_edge_count,
        closed=census.closed,
        consistently_oriented=census.consistently_oriented,
        euler_characteristic=len(referenced) - census.edge_count + len(faces),
        shell_count=len(shells),
        largest_shell_faces=len(shells[0]) if shells else 0,
        largest_shell_fraction=(len(shells[0]) / len(faces)) if faces else 0.0,
        closed_shells=closed_shells,
        unsupported_closed_shells=unsupported_shells,
        unsupported_closed_shell_faces=unsupported_faces,
        unsupported_inaccurate_closed_shells=false_shells,
        unsupported_inaccurate_closed_shell_faces=false_faces,
        unsupported_inaccurate_worst_distance=false_worst,
        unsupported_accurate_closed_shells=correct_shells,
        unsupported_accurate_closed_shell_faces=correct_faces,
        unsupported_accurate_worst_distance=correct_worst,
        accuracy_tolerance=tolerance,
        support_tolerance=tolerance,
    )


def _bucket(points: Sequence[Point], cell: float) -> dict[tuple[int, int, int], list[int]]:
    """Point indices on a uniform lattice of side ``cell``. Deterministic."""
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(points):
        key = (
            math.floor(point[0] / cell),
            math.floor(point[1] / cell),
            math.floor(point[2] / cell),
        )
        grid.setdefault(key, []).append(index)
    return grid


def _covered_mask(
    queries: Sequence[Point], targets: Sequence[Point], tolerance: float
) -> tuple[bool, ...]:
    """Which queries have a target within ``tolerance``, in query order.

    Exact, not approximate. The lattice cell is the tolerance, so every point
    that could be within it lies in one of the 27 neighbouring cells and none
    is missed; the grid changes the cost, never the answer.

    A mask rather than a count because two callers need different things from
    the same search: a coverage fraction wants the total, and the topology
    below wants to know WHICH vertices of a shell nothing measured near.
    """
    if not targets:
        return tuple(False for _ in queries)
    grid = _bucket(targets, tolerance)
    limit = tolerance * tolerance
    hits: list[bool] = []
    for query in queries:
        base = (
            math.floor(query[0] / tolerance),
            math.floor(query[1] / tolerance),
            math.floor(query[2] / tolerance),
        )
        found = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for index in grid.get((base[0] + di, base[1] + dj, base[2] + dk), ()):
                        target = targets[index]
                        dx = query[0] - target[0]
                        dy = query[1] - target[1]
                        dz = query[2] - target[2]
                        if dx * dx + dy * dy + dz * dz <= limit:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        hits.append(found)
    return tuple(hits)


def _covered(queries: Sequence[Point], targets: Sequence[Point], tolerance: float) -> int:
    """How many queries have a target within ``tolerance``. The mask above, summed."""
    return sum(_covered_mask(queries, targets, tolerance))


def component_sizes(points: Sequence[Point], radius: float) -> list[int]:
    """Component sizes, largest first, by fixed-radius linkage on a uniform grid.

    The grid cell is the linkage radius, so only the 27 neighbouring cells can
    hold a link and the cost stays linear in the population for the densities
    this corpus produces. Union-find over a deterministic traversal order, so
    the same points always produce the same components.
    """
    if not points:
        return []
    buckets = _bucket(points, radius)
    parent = list(range(len(points)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    limit = radius * radius
    for key in sorted(buckets):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    other = buckets.get((key[0] + di, key[1] + dj, key[2] + dk))
                    if other is None:
                        continue
                    for left in buckets[key]:
                        for right in other:
                            if right <= left:
                                continue
                            a, b = points[left], points[right]
                            dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
                            if dx * dx + dy * dy + dz * dz <= limit:
                                union(left, right)
    sizes: dict[int, int] = {}
    for index in range(len(points)):
        root = find(index)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


def _extents(points: Sequence[Point]) -> dict[str, float]:
    if not points:
        return {"extent_x": 0.0, "extent_y": 0.0, "extent_z": 0.0}
    return {
        f"extent_{axis}": max(point[index] for point in points)
        - min(point[index] for point in points)
        for index, axis in enumerate("xyz")
    }


def measure(scene: Scene, result: ArmResult) -> tuple[quality.QualityMeasurement, ...]:
    """Everything this oracle says about one arm, computed here and nowhere else."""
    regions = scene.regions
    truth = truth_points(scene)
    full_truth = truth_surface(scene)
    compared = decimate(result.points, regions.truth_pitch)

    measurements: list[quality.QualityMeasurement] = []

    outside = sum(1 for point in result.points if not scene.volume.contains(point, margin=0.0))
    measurements.append(
        GeometryEvidence(
            name="geometry",
            arm=result.arm,
            point_count=len(result.points),
            compared_points=len(compared),
            truth_points=len(truth),
            outside_volume=outside,
            **_extents(result.points),
        )
    )

    if compared:
        measurements.append(
            quality.cloud_agreement(
                compared,
                truth,
                name="support",
                coverage_tolerance=regions.coverage_tolerance,
                unit=DEPTH_UNIT,
            )
        )

    # The truth reference that does not know where the sensor sampled. Every
    # point is scored, undecimated, because the distance is closed form and
    # needs no neighbour search to afford.
    if result.points:
        reference = analytic_reference(scene)
        analytic = [analytic_distance(scene.shapes, point) for point in result.points]
        near = sum(1 for distance in analytic if distance <= regions.coverage_tolerance)
        covered = _covered(reference, result.points, regions.coverage_tolerance)
        measurements.append(
            AnalyticEvidence(
                name="analytic",
                arm=result.arm,
                scored_points=len(analytic),
                distances=quality._distribution(
                    analytic, unit=DEPTH_UNIT, what=f"{result.name} analytic distances"
                ),
                within_tolerance_points=near,
                within_tolerance_fraction=near / len(analytic),
                reference_points=len(reference),
                reference_covered_points=covered,
                reference_coverage_fraction=covered / len(reference),
                coverage_tolerance=regions.coverage_tolerance,
                supersample=regions.analytic_supersample,
            )
        )

    plane_points = _within(result.points, regions.plane)
    if len(plane_points) >= 3:
        measurements.append(
            quality.plane_roughness(
                plane_points,
                name="plane",
                region="the backing plane this scene really is flat on",
                unit=DEPTH_UNIT,
            )
        )

    object_points = _within(result.points, regions.object_region)
    truth_object = _within(full_truth, regions.object_region)
    if object_points and truth_object:
        measurements.append(
            quality.dimension_errors(
                _extents(object_points),
                _extents(truth_object),
                name="dimensions",
                truth_source=TRUTH_SOURCE,
                unit=DEPTH_UNIT,
            )
        )

    # The queries are truth points inside the named band; the targets are the
    # arm's WHOLE point set rather than its points inside the same band, so a
    # truth point one tolerance from the band's own face is not scored as
    # uncovered by an arm that covered it from just outside.
    edge_truth = _within(full_truth, regions.edge_band)
    thin_truth = _within(full_truth, regions.thin_feature)
    edge_hits = _covered(edge_truth, result.points, regions.coverage_tolerance)
    thin_hits = _covered(thin_truth, result.points, regions.coverage_tolerance)
    measurements.append(
        EdgeEvidence(
            name="edge",
            bridged_points=len(_within(result.points, regions.protected)),
            protected_truth_points=len(_within(full_truth, regions.protected)),
            edge_truth_points=len(edge_truth),
            edge_covered_points=edge_hits,
            edge_coverage=(edge_hits / len(edge_truth)) if edge_truth else 0.0,
            thin_feature_truth_points=len(thin_truth),
            thin_feature_covered_points=thin_hits,
            thin_feature_coverage=(thin_hits / len(thin_truth)) if thin_truth else 0.0,
        )
    )

    sizes = component_sizes(compared, regions.linkage_radius)
    total = sum(sizes)
    isolated = [size for size in sizes if size <= regions.isolated_max_points]
    measurements.append(
        ComponentEvidence(
            name="components",
            linkage_radius=regions.linkage_radius,
            component_count=len(sizes),
            largest_component_points=sizes[0] if sizes else 0,
            largest_component_fraction=(sizes[0] / total) if total else 0.0,
            isolated_components=len(isolated),
            isolated_points=sum(isolated),
            isolated_fraction=(sum(isolated) / total) if total else 0.0,
        )
    )

    # The surface itself, which is not its vertices. Both of these are absent
    # for the raw arm - the dots have no faces - and a gate that reads them is
    # recorded as inapplicable there rather than as passing.
    if result.faces:
        scoring = score_surface(scene, result)
        measurements.append(surface_samples(scene, result, support=truth, scoring=scoring))
        measurements.append(surface_topology(scene, result, support=truth, scoring=scoring))

    derived = result.field_evidence
    if result.arm == FUSED and derived is not None:
        cell = scene.cell(str(result.cell_id))
        measurements.append(
            FieldEvidence(
                name="field",
                cell_id=cell.cell_id,
                voxel_size=cell.voxel_size,
                truncation=cell.truncation,
                truncation_in_voxels=cell.truncation_voxels,
                min_weight=cell.min_weight,
                weight_policy=cell.weight_policy,
                cells=derived.cells,
                observed_voxels=derived.observed_voxels,
                unobserved_voxels=derived.unobserved_voxels,
                owned_bytes=derived.owned_bytes,
                frames_offered=derived.frames_offered,
                frames_integrated=derived.frames_integrated,
                surface_extracted=derived.surface_extracted,
                surface_vertices=derived.surface_vertices,
                surface_faces=derived.surface_faces,
                backend=derived.backend,
                zero_level_points=len(derived.zero_level),
                edge_crossings=derived.edge_crossings,
                exact_zero_voxels=derived.exact_zero_voxels,
                work_units=derived.work_units,
                work_budget=scene.work_budget_for(result.cell_id),
            )
        )
        # Fusion, or extraction? The field's own zero level and the extractor's
        # vertices, against the same analytic truth, by the same code.
        field_distances = [analytic_distance(scene.shapes, point) for point in derived.zero_level]
        surface_distances = [analytic_distance(scene.shapes, point) for point in result.points]
        if field_distances and surface_distances:
            field_summary = quality._distribution(
                field_distances, unit=DEPTH_UNIT, what=f"{result.name} field zero-level distances"
            )
            surface_summary = quality._distribution(
                surface_distances, unit=DEPTH_UNIT, what=f"{result.name} surface distances"
            )
            tolerance = regions.coverage_tolerance
            field_near = sum(1 for value in field_distances if value <= tolerance) / len(
                field_distances
            )
            surface_near = sum(1 for value in surface_distances if value <= tolerance) / len(
                surface_distances
            )
            without = len(result.points) - _covered(result.points, derived.zero_level, tolerance)
            measurements.append(
                AblationEvidence(
                    name="ablation",
                    cell_id=cell.cell_id,
                    min_weight=cell.min_weight,
                    field_points=len(derived.zero_level),
                    surface_points=len(result.points),
                    field_distances=field_summary,
                    surface_distances=surface_summary,
                    extraction_p95_delta=surface_summary.p95 - field_summary.p95,
                    extraction_worst_delta=surface_summary.worst - field_summary.worst,
                    field_within_tolerance_fraction=field_near,
                    surface_within_tolerance_fraction=surface_near,
                    extraction_within_tolerance_delta=surface_near - field_near,
                    field_points_in_protected=len(_within(derived.zero_level, regions.protected)),
                    surface_points_in_protected=len(_within(result.points, regions.protected)),
                    surface_points_without_field_support=without,
                    surface_points_without_field_support_fraction=(without / len(result.points)),
                )
            )
    return tuple(measurements)


# --- the honest arms -------------------------------------------------------------------


def _digests(scene: Scene) -> tuple[str, ...]:
    return tuple(view.raw_sha256 for view in scene.views)


def raw_points(scene: Scene) -> ArmResult:
    """The measured dots, unchanged: the arm the user says already looks right."""
    points = measured_points(scene)
    return ArmResult(
        name=RAW_POINTS,
        arm=RAW_POINTS,
        cell_id=None,
        views_used=tuple(range(len(scene.views))),
        poses_used=scene.fused_poses(),
        input_sha256=_digests(scene),
        points=points,
        work=WorkClaim(
            claims={"point_count": len(points), "frames_fused": len(scene.views)},
            work_budget=scene.work_budget,
        ),
    )


def fused(scene: Scene, cell_id: str) -> ArmResult:
    """The unchanged raw TSDF and the unchanged extractor, at one cell.

    ``cell_id`` is a cell the CORPUS declared, or - for
    :func:`fused_at_appliance_resolution` - the derived shipped-policy cell,
    which is why the backend is looked up rather than assumed.
    """
    cell = scene.cell(cell_id)
    vertices, derived = fuse(scene, cell, backend=backend_for(scene, cell_id))
    return ArmResult(
        name=f"{FUSED}:{cell_id}",
        arm=FUSED,
        cell_id=cell_id,
        views_used=tuple(range(len(scene.views))),
        poses_used=scene.fused_poses(),
        input_sha256=_digests(scene),
        points=vertices,
        work=WorkClaim(
            claims={
                "point_count": len(vertices),
                "frames_fused": len(scene.views),
                "observed_voxels": derived.observed_voxels,
                "surface_vertices": derived.surface_vertices,
                "surface_faces": derived.surface_faces,
                "work_units": derived.work_units,
            },
            work_budget=scene.work_budget_for(cell_id),
        ),
        faces=derived.faces,
        field_evidence=derived,
    )


def fused_at_appliance_resolution(scene: Scene) -> ArmResult:
    """The same two modules at the resolution the SHIPPED policy picks for this scene.

    Not a parameter cell. :func:`appliance_cell` runs
    :func:`scanner3d.reconstruction_export.choose_tsdf_resolution` on this
    case's own measured points at the voxel budget the Reconstruct action
    passes it, with the truncation ratio, minimum weight and weight policy that
    module ships and :func:`check_appliance_policy` has already compared with
    the imported constants.

    It exists because the corpus grid's finest cell is 0.10 and the shipped
    policy chooses roughly a third of that on this scene. A conclusion about
    the mesh an owner actually gets cannot be drawn from a surface three times
    coarser than the one they are looking at, so this arm is the one bound to
    the appliance and every other cell is an envelope around it.
    """
    result = fused(scene, APPLIANCE_CELL_ID)
    return ArmResult(
        name="fused_at_appliance_resolution",
        arm=result.arm,
        cell_id=result.cell_id,
        views_used=result.views_used,
        poses_used=result.poses_used,
        input_sha256=result.input_sha256,
        points=result.points,
        work=result.work,
        faces=result.faces,
        field_evidence=result.field_evidence,
    )


# --- the corruption controls ------------------------------------------------------------


def fused_from_truth_frames(scene: Scene, cell_id: str) -> ArmResult:
    """Fuse the EXACT truth frames and present the result as the measured one.

    The impossible-improvement control. Its surface really is better than any
    honest cell's, and that is the point: an oracle that judged only the
    numbers would report it as the fix. The declared input digests are the
    truth frames', so the refusal names what actually happened.
    """
    cell = scene.cell(cell_id)
    volume = TsdfVolume(scene.volume.config(cell), backend=BACKEND)
    volume.integrate(
        [view.truth for view in scene.views],
        list(scene.fused_poses()),
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )
    try:
        surface = extract_surface(volume, min_weight=cell.min_weight, backend=BACKEND)
    except NoSurfaceFound:  # pragma: no cover - exact truth always meshes here
        vertices: tuple[Point, ...] = ()
    else:
        vertices = tuple(surface.mesh.vertices)
    return ArmResult(
        name="fused_from_truth_frames",
        arm=FUSED,
        cell_id=cell_id,
        views_used=tuple(range(len(scene.views))),
        poses_used=scene.fused_poses(),
        input_sha256=tuple(view.truth_sha256 for view in scene.views),
        points=vertices,
        work=WorkClaim(claims={}, work_budget=scene.work_budget),
    )


def fused_dropping_a_view(scene: Scene, cell_id: str) -> ArmResult:
    """Fuse every view but the last, which is the cheapest way to look cleaner.

    Dropping the view that disagrees removes the disagreement. It also removes
    what that view alone saw, and the omission is only visible if somebody
    checks the view list - which is what the refusal is for.
    """
    cell = scene.cell(cell_id)
    kept = tuple(range(len(scene.views) - 1))
    vertices, _ = fuse(scene, cell, views=kept)
    return ArmResult(
        name="fused_dropping_a_view",
        arm=FUSED,
        cell_id=cell_id,
        views_used=kept,
        poses_used=tuple(scene.views[index].fused_pose for index in kept),
        input_sha256=_digests(scene),
        points=vertices,
        work=WorkClaim(claims={}, work_budget=scene.work_budget),
    )


def fused_with_true_poses(scene: Scene, cell_id: str) -> ArmResult:
    """Fuse with the poses the scene kept back, which no reconstruction has.

    On a pose-perturbed case this produces a visibly better surface from the
    same frames. It is the counterfactual the report wants and NOT a candidate
    result, because a pipeline cannot obtain it; the harness publishes the same
    counterfactual itself, as attribution, rather than as an arm that passed.
    """
    cell = scene.cell(cell_id)
    volume = TsdfVolume(scene.volume.config(cell), backend=BACKEND)
    volume.integrate(
        list(scene.raw_frames()),
        [view.true_pose for view in scene.views],
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )
    try:
        surface = extract_surface(volume, min_weight=cell.min_weight, backend=BACKEND)
    except NoSurfaceFound:  # pragma: no cover - the object always meshes here
        vertices: tuple[Point, ...] = ()
    else:
        vertices = tuple(surface.mesh.vertices)
    return ArmResult(
        name="fused_with_true_poses",
        arm=FUSED,
        cell_id=cell_id,
        views_used=tuple(range(len(scene.views))),
        poses_used=tuple(view.true_pose for view in scene.views),
        input_sha256=_digests(scene),
        points=vertices,
        work=WorkClaim(claims={}, work_budget=scene.work_budget),
    )


def fused_with_edited_surface(scene: Scene, cell_id: str) -> ArmResult:
    """Fuse honestly, then quietly delete the vertices that sit in free space.

    The most plausible forgery in the whole corpus, because deleting the
    bridged points is exactly what somebody would do by hand after looking at
    the mesh, and every distance metric improves when they do.

    When the honest surface put NOTHING in the protected volume - which is the
    ordinary outcome on the exact control - that deletion would remove nothing
    and this control would quietly become a copy of the honest arm. A control
    that can turn into a no-op tests nothing, so the fallback edit is the
    blandest one available: drop the last vertex the extractor emitted. It is
    still an arm publishing geometry that is not what the field produced, which
    is the only thing ``surface_not_derived`` is about.
    """
    cell = scene.cell(cell_id)
    vertices, derived = fuse(scene, cell)
    if not vertices:  # pragma: no cover - every case in this corpus extracts a surface
        raise ScoringUnusable(
            f"{cell.cell_id} extracted no surface on this case, so there is nothing for the "
            "edited-surface control to edit and no forgery for the reader to refuse"
        )
    kept = tuple(point for point in vertices if not scene.regions.protected.contains(point))
    if len(kept) == len(vertices):
        kept = vertices[:-1]
    return ArmResult(
        name="fused_with_edited_surface",
        arm=FUSED,
        cell_id=cell_id,
        views_used=tuple(range(len(scene.views))),
        poses_used=scene.fused_poses(),
        input_sha256=_digests(scene),
        points=kept,
        work=WorkClaim(claims={}, work_budget=scene.work_budget),
        field_evidence=derived,
    )


def fused_with_bridging_triangle(scene: Scene, cell_id: str) -> ArmResult:
    """Fuse honestly, keep every vertex exactly where it is, and add one triangle.

    This is the forgery a VERTEX-ONLY reader cannot see, and it is the reason
    round three of this corpus measures triangle interiors at all. The point
    list is byte-identical to the honest arm's, so ``geometry``, ``support``,
    ``analytic``, ``components`` and ``edge.bridged_points`` all agree with it
    to the last bit; what changed is one face, spanning the protected volume
    the scene declares empty.

    The added face is found by a BOUNDED deterministic search - at most
    :data:`_BRIDGE_SHORTLIST` vertices nearest the protected volume's centre,
    in a fixed order, and the first triple whose interior enters that volume
    while none of its three corners does. A control that could quietly become a
    no-op tests nothing, so a case where no such triple exists is refused as
    unscoreable rather than published as a passing control.
    """
    cell = scene.cell(cell_id)
    vertices, derived = fuse(scene, cell)
    protected = scene.regions.protected
    shortlist: list[int] = []
    for corner in (protected.minimum, protected.maximum):
        shortlist.extend(
            sorted(
                range(len(vertices)),
                key=lambda index, corner=corner: (  # type: ignore[misc]
                    sum((vertices[index][axis] - corner[axis]) ** 2 for axis in range(3)),
                    index,
                ),
            )[:_BRIDGE_SHORTLIST]
        )
    outside = sorted({index for index in shortlist if not protected.contains(vertices[index])})
    lattice = barycentric_lattice(scene.regions.triangle_sample_order)
    added: tuple[int, int, int] | None = None
    for first in range(len(outside)):
        for second in range(first + 1, len(outside)):
            for third in range(second + 1, len(outside)):
                face = (outside[first], outside[second], outside[third])
                a, b, c = (vertices[index] for index in face)
                if any(
                    protected.contains(
                        (
                            u * a[0] + v * b[0] + w * c[0],
                            u * a[1] + v * b[1] + w * c[1],
                            u * a[2] + v * b[2] + w * c[2],
                        )
                    )
                    for u, v, w in lattice
                ):
                    added = face
                    break
            if added is not None:
                break
        if added is not None:
            break
    if added is None:
        raise ScoringUnusable(
            f"no triangle over the {len(outside)} vertex/vertices nearest {cell.cell_id}'s "
            "protected volume has an interior sample inside it while all three corners stay "
            "outside. This control has nothing to forge on this case, and a control that "
            "becomes a copy of the honest arm tests nothing"
        )
    return ArmResult(
        name="fused_with_bridging_triangle",
        arm=FUSED,
        cell_id=cell_id,
        views_used=tuple(range(len(scene.views))),
        poses_used=scene.fused_poses(),
        input_sha256=_digests(scene),
        points=vertices,
        work=WorkClaim(claims={}, work_budget=scene.work_budget),
        faces=(*derived.faces, added),
        field_evidence=derived,
    )


def raw_points_dropping_hard_samples(scene: Scene) -> ArmResult:
    """The dots, minus every measurement near the depth step.

    The raw arm's own way of cheating: the edge is where the dots are noisiest,
    and an arm that drops those samples reports a cleaner cloud over a smaller
    denominator.
    """
    points = tuple(
        point for point in measured_points(scene) if not scene.regions.edge_band.contains(point)
    )
    return ArmResult(
        name="raw_points_dropping_hard_samples",
        arm=RAW_POINTS,
        cell_id=None,
        views_used=tuple(range(len(scene.views))),
        poses_used=scene.fused_poses(),
        input_sha256=_digests(scene),
        points=points,
        work=WorkClaim(claims={}, work_budget=scene.work_budget),
    )


def fused_overclaiming_work(scene: Scene, cell_id: str) -> ArmResult:
    """An honest surface with a work record that flatters the parameter choice.

    Halving the declared spend is how a coarse grid could be sold as a cheap
    one. The surface is real; the accounting is not.
    """
    honest = fused(scene, cell_id)
    claims = dict(honest.work.claims)
    claims["work_units"] = claims["work_units"] // 2
    return ArmResult(
        name="fused_overclaiming_work",
        arm=FUSED,
        cell_id=cell_id,
        views_used=honest.views_used,
        poses_used=honest.poses_used,
        input_sha256=honest.input_sha256,
        points=honest.points,
        work=WorkClaim(claims=claims, work_budget=scene.work_budget),
        faces=honest.faces,
        field_evidence=honest.field_evidence,
    )


def fused_claiming_the_unmeasurable(scene: Scene, cell_id: str) -> ArmResult:
    """An honest surface that declares a bar of its own beside it.

    "surface_accuracy_target" is not a quantity this oracle computes, and an
    arm that could publish one would be grading its own work. The refusal is
    the whole point of the closed claim vocabulary.
    """
    honest = fused(scene, cell_id)
    claims = dict(honest.work.claims)
    claims["surface_accuracy_target"] = 1
    return ArmResult(
        name="fused_claiming_the_unmeasurable",
        arm=FUSED,
        cell_id=cell_id,
        views_used=honest.views_used,
        poses_used=honest.poses_used,
        input_sha256=honest.input_sha256,
        points=honest.points,
        work=WorkClaim(claims=claims, work_budget=scene.work_budget),
        faces=honest.faces,
        field_evidence=honest.field_evidence,
    )


def candidate(form: str, argument: str | None) -> Candidate:
    """The arm one fixture name runs, or a refusal to build one nobody wrote."""
    if form not in CANDIDATE_FORMS:
        raise ScoringUnusable(
            f"candidate form {form!r} is not one of {list(CANDIDATE_FORMS)}; a fixture cannot "
            "name behaviour nobody wrote"
        )
    if form == "raw_points":
        if argument is not None:
            raise ScoringUnusable("the raw-points arm takes no parameter cell")
        return raw_points
    if form == "raw_points_dropping_hard_samples":
        if argument is not None:
            raise ScoringUnusable(f"{form} takes no parameter cell")
        return raw_points_dropping_hard_samples
    if form == "fused_at_appliance_resolution":
        if argument is not None:
            raise ScoringUnusable(
                f"{form} takes no parameter cell: its voxel size is derived from the shipped "
                "policy and is not a number a fixture may choose"
            )
        return fused_at_appliance_resolution
    if argument is None:
        raise ScoringUnusable(f"candidate form {form!r} needs the id of a grid cell")
    cell_id = argument
    builders: Mapping[str, Callable[[Scene, str], ArmResult]] = {
        "fused": fused,
        "fused_from_truth_frames": fused_from_truth_frames,
        "fused_dropping_a_view": fused_dropping_a_view,
        "fused_with_true_poses": fused_with_true_poses,
        "fused_with_edited_surface": fused_with_edited_surface,
        "fused_with_bridging_triangle": fused_with_bridging_triangle,
        "fused_overclaiming_work": fused_overclaiming_work,
        "fused_claiming_the_unmeasurable": fused_claiming_the_unmeasurable,
    }
    builder = builders[form]
    return lambda scene: builder(scene, cell_id)
