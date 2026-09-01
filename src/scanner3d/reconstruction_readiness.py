"""One strict, inspectable answer to "may this reconstruction be published?".

Why this exists, observed
-------------------------
On 2026-08-28 a capture kept writing frames for two minutes after live
alignment had stopped placing any of them (see :doc:`../docs/tracking-loss`).
The folder afterwards looked exactly like a healthy one.
:mod:`scanner3d.capture_sequence` now refuses that folder outright, and this
module is the other half of the same argument: a reconstruction can also be
*structurally* fine and still be worthless, and the shapes that produce are
all things nobody looking at the result can see.

Four of them, and each has its own instrument here because one number cannot
say which one happened:

* **The capture never resolved a tracking loss.** Authority for that stays
  entirely with :func:`scanner3d.capture_sequence.read_capture_sequence`; this
  module only makes sure a caller that entered through a *direct* API - the
  export function takes a capture directory and a list of poses, and used to
  believe both - still reaches it.
* **The poses do not describe a scan.** An all-identical sequence, a sequence
  that jitters below the measurement noise floor, or one that is stuck between
  a couple of viewpoints, all register perfectly against themselves and produce
  a plausible-looking cloud of one view.
* **The occupied surface was seen from one angle.** A high point count is not
  evidence: a million points from one direction still know nothing about the
  other side, and the shape that comes out of it is a guess.
* **The extracted surface disagrees with the points it was fused from.** A
  displaced, floating or grossly lumpy TSDF surface is topologically fine and
  opens in any viewer. "The points it was fused from" is meant literally here:
  the comparison is against a bounded sample of the FULL-RESOLUTION frames the
  volume integrated (:func:`fusion_evidence_sample`), not against the decimated
  clouds the registration was fitted to - a defect that lives in the frames but
  misses the decimation is invisible to the second and is most of every frame.

What this is NOT
----------------
Everything measured here is **internal consistency**. Not one number in this
module is a dimensional accuracy result, and nothing here has seen an
independent measurement of the object. A reconstruction this module calls ready
can still be the wrong size, the wrong shape, or a beautiful model of the wrong
thing - see :doc:`../docs/quality-harness` for what would and would not catch
that, and :data:`LIMITATIONS` for the same sentence in the record itself.

Nor is it a watertightness requirement. A physical scan of one side of an
object legitimately produces an open surface with real boundaries, and the
surface gate below is written so that such a surface passes: it asks whether
the surface sits on the evidence and whether it abandons it, never whether it
closes.

Scale
-----
Every threshold that has a length in it is expressed as a multiple of something
already calibrated in this repository - :data:`scanner3d.pipeline.NOISE_FLOOR`
and :data:`scanner3d.pipeline.TRAVEL_MIN_MULTIPLE` for motion, the placed
geometry's own bounding diagonal for coverage, and the TSDF's own declared
truncation for the surface. There is no fixed world coordinate magic number
here, so a capture at a different scale is judged by the same policy rather than
by an accident of units - and nothing a caller tunes for SPEED enters a
decision: the coverage cell deliberately does not follow the fusion voxel size,
because a smaller voxel budget must not be able to change what counts as
adequate coverage.

Cost
----
Bounded and CPU-only; no new dependency, no GPU, nothing retained past the
call.

* tracking: one ``stat`` per run. A capture with no loss record costs exactly
  that and never opens the authority at all.
* poses: a fixed probe of :attr:`ReadinessPolicy.probe_points` points is taken
  once from the first non-empty cloud. Travel and excursion are one pass over
  ``poses x probe``. Viewpoint counting is greedy against at most
  :attr:`ReadinessPolicy.max_viewpoints_examined` representatives, so the worst
  case is ``poses x max_viewpoints_examined x probe`` distances and never
  all-pairs over the poses.
* coverage: at most :attr:`ReadinessPolicy.coverage_max_frames` frames, chosen
  by uniform stride over the sequence, at most
  :attr:`ReadinessPolicy.coverage_points_per_frame` points each.
* fusion evidence: at most :attr:`ReadinessPolicy.fusion_sample_frames`
  frames are opened, by uniform stride, and at most
  :attr:`ReadinessPolicy.fusion_pixels_per_frame` pixels of each are examined,
  by uniform stride over the raster. One frame's bytes are the largest thing
  held at a time and no field is built: this deprojects pixels, it does not
  re-fuse anything.
* surface: both clouds are voxel-thinned to one point per comparison cell,
  and each direction asks at most :attr:`ReadinessPolicy.surface_sample_points`
  of them. The search is a voxel hash over 27 cells, so the whole pass is
  linear in the number of questions asked and never all-pairs.

Resolution
----------
The surface gate sees a displacement of about two truncations and not one of
half a truncation, because its comparison cell is the truncation and its
tolerance is that cell's diagonal. It is a GROSS consistency check by
construction and is not a fine metrology instrument; the numbers it records
say which tolerance was used, so nobody has to infer it.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scanner3d.coverage import DEFAULT_MIN_SPREAD, CoverageNotAllowed, coverage_map
from scanner3d.depth_metric import (
    NO_RETURN_SAMPLE,
    PIXELS_PER_FRAME,
    ensure_frame_length,
    ensure_intrinsics_usable,
    ensure_scale_usable,
)
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.pipeline import NOISE_FLOOR, TRAVEL_MIN_MULTIPLE, frame_travel
from scanner3d.v4l2_capture import FRAME_WIDTH

#: Bumped when the serialized shape below changes in any way a reader could
#: misinterpret. The reader demands this exact value and never guesses at an
#: earlier or a later one.
SCHEMA_VERSION: Final = 2

#: So a record found loose is identifiable as this and not as one of the other
#: schema-1 objects this repository writes.
KIND: Final = "scanner3d.reconstruction-readiness"

#: Longest detail string kept in the record, so one refusal cannot carry an
#: unbounded exception message into a JSON file somebody has to read.
DETAIL_LIMIT: Final = 400
TRUNCATION_MARK: Final = " [truncated]"


# -- reason vocabulary ------------------------------------------------------
#
# Plain strings rather than an enum, for the reason
# :data:`scanner3d.relocalize.Refusals` gives: these are written into durable
# records and read back by things that are not this process.

#: The capture holds a tracking loss its own authority does not consider closed.
TRACKING_LOST: Final = "tracking_lost"
#: Something loss-record-shaped is in the capture and cannot be read as one.
TRACKING_LOSS_UNREADABLE: Final = "tracking_loss_record_unusable"
#: Too few genuinely different viewpoints among the accepted poses.
POSE_VIEWPOINTS_INSUFFICIENT: Final = "pose_viewpoints_insufficient"
#: The whole trajectory stays inside a volume too small to be a scan.
POSE_EXCURSION_INSUFFICIENT: Final = "pose_excursion_insufficient"
#: Almost every adjacent step moved the surface less than the noise floor.
POSE_MOTION_BELOW_NOISE: Final = "pose_motion_below_noise"
#: There was nothing to measure motion with at all.
POSE_MOTION_NOT_MEASURABLE: Final = "pose_motion_not_measurable"
#: Too little of the occupied surface was seen repeatedly and from a spread.
COVERAGE_UNSUPPORTED: Final = "coverage_unsupported"
#: Coverage could not be measured from what was offered.
COVERAGE_NOT_MEASURABLE: Final = "coverage_not_measurable"
#: The extracted surface does not sit on the points it was fused from.
SURFACE_UNSUPPORTED: Final = "surface_unsupported_by_evidence"
#: The extracted surface abandons most of the points it was fused from.
SURFACE_ABANDONS_EVIDENCE: Final = "surface_abandons_evidence"
#: Surface agreement could not be measured from what was offered.
SURFACE_NOT_MEASURABLE: Final = "surface_agreement_not_measurable"

#: What the operator should actually DO about each reason. One sentence, in the
#: imperative, and the same words in the window and on the command line - the
#: front ends render this mapping rather than each writing its own advice.
ADVICE: Final[Mapping[str, str]] = {
    TRACKING_LOST: (
        "the scan stopped knowing where the object is and never found it again. Rescan the "
        "object; the frames before the loss are kept, but nothing relates the ones after it to "
        "them"
    ),
    TRACKING_LOSS_UNREADABLE: (
        "this capture holds a tracking-loss record that cannot be read, so what it says cannot "
        "be obeyed. Keep the folder and rescan; do not delete the record"
    ),
    POSE_VIEWPOINTS_INSUFFICIENT: (
        "move the scanner around the object. These frames were all taken from what amounts to "
        "the same place, so nothing here knows the shape of the other side"
    ),
    POSE_EXCURSION_INSUFFICIENT: (
        "acquire real viewpoint motion. The whole sweep stayed inside a volume smaller than one "
        "useful scanning step, so it is a repeated view rather than a scan"
    ),
    POSE_MOTION_BELOW_NOISE: (
        "acquire real viewpoint motion. Almost nothing moved between frames by more than the "
        "scanner's own measurement noise, so there is no parallax to reconstruct from"
    ),
    POSE_MOTION_NOT_MEASURABLE: (
        "rescan. There were no usable points to measure viewpoint motion against, so no claim "
        "about this trajectory can be made either way"
    ),
    COVERAGE_UNSUPPORTED: (
        "add overlapping views from different angles. Most of the surface that was seen at all "
        "was seen from one direction, and a high point count from one angle is not evidence of "
        "shape"
    ),
    COVERAGE_NOT_MEASURABLE: (
        "rescan. Coverage could not be measured from these frames and poses, so the surface "
        "support behind this model is unknown rather than adequate"
    ),
    SURFACE_UNSUPPORTED: (
        "rescan. The fused surface does not sit on the points it was built from - a displaced, "
        "floating or lumpy surface - so it is not a picture of this capture"
    ),
    SURFACE_ABANDONS_EVIDENCE: (
        "rescan, more slowly and with more overlap. The fused surface leaves most of the "
        "registered points unexplained, so it is a fragment presented as a model"
    ),
    SURFACE_NOT_MEASURABLE: (
        "rescan. Agreement between the fused surface and the registered points could not be "
        "measured, so the surface is unchecked rather than checked and sound"
    ),
}

#: Every reason this module can produce, in the order it reports them.
REASONS: Final[tuple[str, ...]] = tuple(ADVICE)

#: Stated in every record, because a reader who has one of these in front of
#: them is entitled to know what it does not say.
LIMITATIONS: Final[tuple[str, ...]] = (
    "every measurement here is INTERNAL CONSISTENCY. Nothing in this record was compared "
    "against an independent measurement of the object, so a ready result is not a dimensional "
    "accuracy claim and never establishes scale",
    "watertightness is NOT required and is not measured here. A scan of one side of an object "
    "legitimately produces an open surface with real boundaries and passes",
    "the pose gates judge the trajectory that was ACCEPTED. A pose the registration got wrong "
    "and was confident about is a pose this module measures as ordinary motion",
    "coverage is measured over OCCUPIED cells only. Empty space nobody looked at is not "
    "counted as an unsupported surface, and this therefore says nothing about what was missed "
    "entirely",
)


class ReadinessNotMeasurable(ScannerError):
    """A readiness evaluation was asked for that cannot be performed at all.

    Reserved for a caller error - a pose count that does not match the clouds,
    a policy value that is not a number - and never for the scan being bad,
    which is what the failures inside a returned result are for.
    """

    status = DoctorStatus.INTERNAL_ERROR


class ReadinessRecordUnusable(ScannerError):
    """A JSON document offered as a readiness record is not one.

    A record that cannot be read is never read as a record that passed; that
    fallback is exactly the failure this whole module exists to prevent.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class ReconstructionRefused(ScannerError):
    """Publication was refused because at least one readiness gate failed.

    Carries the whole :class:`ReconstructionReadiness` on :attr:`readiness`, so
    a front end can render every failure and its advice rather than only the
    sentence this exception happens to stringify to.
    """

    status = DoctorStatus.UNSUPPORTED_MODE

    def __init__(self, readiness: ReconstructionReadiness) -> None:
        super().__init__(readiness.refusal_text())
        self.readiness = readiness


def _bounded(text: str) -> str:
    """One detail string, marked if it had to be cut."""
    if len(text) <= DETAIL_LIMIT:
        return text
    return text[:DETAIL_LIMIT] + TRUNCATION_MARK


def _json_number(value: float) -> float | None:
    """``value``, or ``None`` where JSON has no word for it.

    The convention :func:`scanner3d.relocalize._json_number` established, kept
    identical so a reader does not have to learn a second one.
    """
    return value if math.isfinite(value) else None


# -- policy -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReadinessPolicy:
    """Every value a decision here depends on, in one inspectable object.

    Serialized with the result rather than only documented, so a record from
    six months ago states the policy it was judged under instead of leaving a
    reader to assume it was today's.
    """

    #: Per-point measurement noise, in the capture's depth units. Not chosen
    #: here: it is :data:`scanner3d.pipeline.NOISE_FLOOR`, the same constant
    #: every gate in the tracking stack is a multiple of.
    noise_floor: float = NOISE_FLOOR
    #: Two views closer together than this many noise floors are the same
    #: viewpoint. :data:`scanner3d.pipeline.TRAVEL_MIN_MULTIPLE` is the smallest
    #: per-frame surface travel the tracker calls useful, so a step below it is
    #: below the resolution at which "a different view" means anything.
    viewpoint_separation_multiple: float = TRAVEL_MIN_MULTIPLE
    #: How many genuinely different viewpoints a sweep must contain. Three,
    #: because it is also the minimum frame count a sweep is allowed to have,
    #: and two views of an object establish nothing about the volume between
    #: them.
    min_distinct_viewpoints: int = 3
    #: Counting stops here. A sequence with this many distinct viewpoints is
    #: unambiguously diverse and the exact number stops mattering; the cap is
    #: what keeps viewpoint counting linear in the pose count.
    max_viewpoints_examined: int = 16
    #: How many points of the first non-empty cloud stand in for the object
    #: when motion is measured. A fixed deterministic stride, not a sample.
    probe_points: int = 128
    #: How many adjacent steps must have moved the surface by more than the
    #: noise floor. Deliberately a small absolute count rather than a fraction,
    #: so that an ordinary pause - which is a legitimate thing to do mid-scan -
    #: costs nothing.
    min_moving_steps: int = 2
    #: The whole trajectory must move the probe surface by at least this many
    #: noise floors, measured as the extent of each probe point's track rather
    #: than the distance between the first and last pose - so an orbit that
    #: returns to where it started is not mistaken for a scan that never left.
    #: Six is two useful scanning steps.
    min_excursion_multiple: float = 6.0
    #: How many coverage cells span the placed geometry's longest diagonal. The
    #: cell is therefore derived from the OBJECT and nothing else - deliberately
    #: not from the fusion voxel size, which is a cost setting: a smaller voxel
    #: budget must not be able to change what counts as adequate coverage.
    coverage_cells_per_span: int = 32
    #: And a floor on that cell, as a multiple of the clouds' own estimated
    #: point spacing. A cell finer than the sampling measures the SAMPLING: a
    #: cell holding one point can only ever have been seen once, so a thin
    #: cloud would fail coverage for being thin rather than for being
    #: one-sided. Two, so that a cell holds several points of every frame that
    #: saw it.
    coverage_spacing_multiple: float = 2.0
    #: How many distinct frames must see an occupied cell for it to count as
    #: supported.
    coverage_min_observations: int = 2
    #: And how far apart, in radians, the two most divergent of those frames
    #: must have been looking. :data:`scanner3d.coverage.DEFAULT_MIN_SPREAD`,
    #: reused rather than re-chosen.
    coverage_min_spread: float = DEFAULT_MIN_SPREAD
    #: The share of OCCUPIED cells that must be supported.
    coverage_min_supported_fraction: float = 0.5
    #: Ceiling on frames offered to the coverage map, taken by uniform stride
    #: over the sequence so the angular spread of the sweep is preserved.
    coverage_max_frames: int = 64
    #: And on points per frame, likewise by stride.
    coverage_points_per_frame: int = 2000
    #: How far a surface vertex may sit from the nearest registered point and
    #: still count as supported, as a multiple of the TSDF's own declared
    #: truncation. A signed-distance surface cannot carry information from
    #: further away than that, so it is the natural unit here.
    surface_tolerance_truncations: float = 1.0
    #: Floor on that tolerance, as a multiple of the noise floor, so a very
    #: fine voxel size cannot demand agreement finer than the scanner measures.
    surface_tolerance_noise_multiple: float = 2.0
    #: The share of surface vertices that must be within tolerance of a
    #: registered point. High: a surface that does not sit on its own evidence
    #: is not a surface of this capture.
    surface_min_supported_fraction: float = 0.9
    #: The share of registered points that must be within tolerance of the
    #: surface. Lower, deliberately: fusion legitimately drops thinly observed
    #: points below its weight threshold, and an open surface is allowed.
    surface_min_evidence_explained: float = 0.6
    #: Ceiling on how many points either direction of the surface comparison
    #: asks about, and on the thinned evidence cloud the comparison cell
    #: responds to. Reached by VOXEL downsampling for the cloud that is being
    #: searched and by a stride for the points doing the searching - never the
    #: other way round. A stride thins a target without preserving its spacing,
    #: so two clouds lying exactly on each other come out of it a random half
    #: spacing apart and a perfect surface measures as a poor one.
    surface_sample_points: int = 20000
    #: The comparison cell, as a fraction of the base tolerance.
    surface_cell_fraction: float = 1.0
    #: How many times it may double before the measurement is abandoned. Each
    #: doubling shrinks a cloud by roughly eight, so this is far more headroom
    #: than any capture this repository can produce needs.
    surface_max_coarsening: int = 16
    #: How many of the FULL-RESOLUTION frames the volume integrated are sampled
    #: for the surface gate's evidence, taken by uniform stride over the
    #: sequence so the angular spread of the sweep is preserved. The same
    #: ceiling and the same reasoning as :attr:`coverage_max_frames`.
    fusion_sample_frames: int = 64
    #: And how many pixels of each of those frames are examined, by uniform
    #: stride over the raster. This is a CANDIDATE count, not a point count: a
    #: candidate that carries no return, or a distance outside the fusion's own
    #: accepted interval, is skipped exactly as the fusion skipped it, so the
    #: points that come out are at most this many and usually far fewer.
    #: 2048 of 256000 samples is one pixel in 125 - dense enough that a defect
    #: covering a few percent of a frame lands in it many times over, and small
    #: enough that the whole sample is bounded at
    #: ``fusion_sample_frames * fusion_pixels_per_frame`` points.
    fusion_pixels_per_frame: int = 2048

    def __post_init__(self) -> None:
        for name in (
            "noise_floor",
            "viewpoint_separation_multiple",
            "min_excursion_multiple",
            "coverage_min_spread",
            "coverage_spacing_multiple",
            "coverage_min_supported_fraction",
            "surface_tolerance_truncations",
            "surface_tolerance_noise_multiple",
            "surface_min_supported_fraction",
            "surface_min_evidence_explained",
            "surface_cell_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ReadinessNotMeasurable(f"policy {name!r} is {value!r}, not a number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ReadinessNotMeasurable(
                    f"policy {name!r} is {value!r}; it must be a finite value of at least zero"
                )
        for name in (
            "min_distinct_viewpoints",
            "max_viewpoints_examined",
            "probe_points",
            "min_moving_steps",
            "coverage_min_observations",
            "coverage_cells_per_span",
            "coverage_max_frames",
            "coverage_points_per_frame",
            "surface_sample_points",
            "surface_max_coarsening",
            "fusion_sample_frames",
            "fusion_pixels_per_frame",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ReadinessNotMeasurable(f"policy {name!r} is {value!r}, not an integer")
            if value < 1:
                raise ReadinessNotMeasurable(f"policy {name!r} is {value!r}; it must be at least 1")
        self._check_domains()
        self._check_relationships()

    def _check_domains(self) -> None:
        """The RANGE of each value, not only its sign.

        A finite non-negative number can still be a policy no evaluation could
        ever have been judged under. A required fraction of 1.5 is a gate
        nothing passes, an angular spread of 4 radians is one no two view
        directions can span, and a scale of zero collapses the unit the values
        beside it are expressed in. Each of those is a policy that is broken
        rather than strict, so it is refused where it is built instead of
        silently refusing every reconstruction that meets it.
        """
        for name in (
            "coverage_min_supported_fraction",
            "surface_min_supported_fraction",
            "surface_min_evidence_explained",
        ):
            value = float(getattr(self, name))
            if value > 1.0:
                raise ReadinessNotMeasurable(
                    f"policy {name!r} is {value!r}; it is a required SHARE and must lie in "
                    "[0, 1]. A policy demanding more than all of something is a gate no "
                    "reconstruction can pass"
                )
        if self.coverage_min_spread > math.pi:
            raise ReadinessNotMeasurable(
                f"policy 'coverage_min_spread' is {self.coverage_min_spread!r} radians; the "
                f"angle between two view directions is at most pi ({math.pi!r}), so no cell "
                "could ever meet this"
            )
        # Scales, as opposed to thresholds. A threshold of zero is a lax policy
        # and a legitimate thing to state; a SCALE of zero is a policy with no
        # unit in it at all.
        for name in ("noise_floor", "viewpoint_separation_multiple", "surface_cell_fraction"):
            value = float(getattr(self, name))
            if value == 0.0:
                raise ReadinessNotMeasurable(
                    f"policy {name!r} is {value!r}; it is the SCALE the values beside it are "
                    "measured in and a scale of zero measures nothing"
                )

    def _check_relationships(self) -> None:
        """Pairs of values that make each other's gate impossible by construction.

        Individually in range and jointly unreachable is still unreachable: a
        ceiling below the minimum it is a ceiling on describes a gate that
        cannot be passed rather than one that is hard to pass.
        """
        if self.max_viewpoints_examined < self.min_distinct_viewpoints:
            raise ReadinessNotMeasurable(
                f"policy stops counting viewpoints at {self.max_viewpoints_examined} and "
                f"requires {self.min_distinct_viewpoints}; counting that stops below the "
                "requirement can never reach it"
            )
        if self.coverage_max_frames < self.coverage_min_observations:
            raise ReadinessNotMeasurable(
                f"policy offers coverage at most {self.coverage_max_frames} frame(s) and "
                f"requires each supported cell to have been seen by "
                f"{self.coverage_min_observations}; no cell could be seen by more frames than "
                "were offered"
            )
        no_truncation_term = float(self.surface_tolerance_truncations) == 0.0
        no_noise_term = float(self.surface_tolerance_noise_multiple) == 0.0
        if no_truncation_term and no_noise_term:
            raise ReadinessNotMeasurable(
                "policy derives the surface tolerance from a zero multiple of the truncation "
                "AND a zero multiple of the noise floor, which is a tolerance of zero; no "
                "extracted surface lies exactly on its own evidence"
            )

    @property
    def viewpoint_separation(self) -> float:
        """The distance below which two placements are the same viewpoint."""
        return self.noise_floor * self.viewpoint_separation_multiple

    @property
    def min_excursion(self) -> float:
        """The smallest whole-trajectory extent this policy calls a scan."""
        return self.noise_floor * self.min_excursion_multiple

    def surface_tolerance(self, *, truncation: float) -> float:
        """How far a surface vertex may sit from the evidence, for this volume."""
        return max(
            self.surface_tolerance_truncations * truncation,
            self.surface_tolerance_noise_multiple * self.noise_floor,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "noise_floor": _json_number(self.noise_floor),
            "viewpoint_separation_multiple": _json_number(self.viewpoint_separation_multiple),
            "viewpoint_separation": _json_number(self.viewpoint_separation),
            "min_distinct_viewpoints": self.min_distinct_viewpoints,
            "max_viewpoints_examined": self.max_viewpoints_examined,
            "probe_points": self.probe_points,
            "min_moving_steps": self.min_moving_steps,
            "min_excursion_multiple": _json_number(self.min_excursion_multiple),
            "min_excursion": _json_number(self.min_excursion),
            "coverage_cells_per_span": self.coverage_cells_per_span,
            "coverage_spacing_multiple": _json_number(self.coverage_spacing_multiple),
            "coverage_min_observations": self.coverage_min_observations,
            "coverage_min_spread_radians": _json_number(self.coverage_min_spread),
            "coverage_min_supported_fraction": _json_number(self.coverage_min_supported_fraction),
            "coverage_max_frames": self.coverage_max_frames,
            "coverage_points_per_frame": self.coverage_points_per_frame,
            "surface_tolerance_truncations": _json_number(self.surface_tolerance_truncations),
            "surface_tolerance_noise_multiple": _json_number(self.surface_tolerance_noise_multiple),
            "surface_min_supported_fraction": _json_number(self.surface_min_supported_fraction),
            "surface_min_evidence_explained": _json_number(self.surface_min_evidence_explained),
            "surface_sample_points": self.surface_sample_points,
            "surface_cell_fraction": _json_number(self.surface_cell_fraction),
            "surface_max_coarsening": self.surface_max_coarsening,
            "fusion_sample_frames": self.fusion_sample_frames,
            "fusion_pixels_per_frame": self.fusion_pixels_per_frame,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object], *, what: str = "policy") -> ReadinessPolicy:
        _exact_keys(
            record,
            (
                "noise_floor",
                "viewpoint_separation_multiple",
                "viewpoint_separation",
                "min_distinct_viewpoints",
                "max_viewpoints_examined",
                "probe_points",
                "min_moving_steps",
                "min_excursion_multiple",
                "min_excursion",
                "coverage_cells_per_span",
                "coverage_spacing_multiple",
                "coverage_min_observations",
                "coverage_min_spread_radians",
                "coverage_min_supported_fraction",
                "coverage_max_frames",
                "coverage_points_per_frame",
                "surface_tolerance_truncations",
                "surface_tolerance_noise_multiple",
                "surface_min_supported_fraction",
                "surface_min_evidence_explained",
                "surface_sample_points",
                "surface_cell_fraction",
                "surface_max_coarsening",
                "fusion_sample_frames",
                "fusion_pixels_per_frame",
            ),
            what,
        )
        policy = cls(
            noise_floor=_number(record, "noise_floor", what),
            viewpoint_separation_multiple=_number(record, "viewpoint_separation_multiple", what),
            min_distinct_viewpoints=_integer(record, "min_distinct_viewpoints", what),
            max_viewpoints_examined=_integer(record, "max_viewpoints_examined", what),
            probe_points=_integer(record, "probe_points", what),
            min_moving_steps=_integer(record, "min_moving_steps", what),
            min_excursion_multiple=_number(record, "min_excursion_multiple", what),
            coverage_cells_per_span=_integer(record, "coverage_cells_per_span", what),
            coverage_spacing_multiple=_number(record, "coverage_spacing_multiple", what),
            coverage_min_observations=_integer(record, "coverage_min_observations", what),
            coverage_min_spread=_number(record, "coverage_min_spread_radians", what),
            coverage_min_supported_fraction=_number(
                record, "coverage_min_supported_fraction", what
            ),
            coverage_max_frames=_integer(record, "coverage_max_frames", what),
            coverage_points_per_frame=_integer(record, "coverage_points_per_frame", what),
            surface_tolerance_truncations=_number(record, "surface_tolerance_truncations", what),
            surface_tolerance_noise_multiple=_number(
                record, "surface_tolerance_noise_multiple", what
            ),
            surface_min_supported_fraction=_number(record, "surface_min_supported_fraction", what),
            surface_min_evidence_explained=_number(record, "surface_min_evidence_explained", what),
            surface_sample_points=_integer(record, "surface_sample_points", what),
            surface_cell_fraction=_number(record, "surface_cell_fraction", what),
            surface_max_coarsening=_integer(record, "surface_max_coarsening", what),
            fusion_sample_frames=_integer(record, "fusion_sample_frames", what),
            fusion_pixels_per_frame=_integer(record, "fusion_pixels_per_frame", what),
        )
        # The two derived values are written out for a reader's convenience and
        # are therefore checked, not trusted: a record whose stated separation
        # does not follow from its stated multiple is internally inconsistent.
        _consistent(
            _number(record, "viewpoint_separation", what),
            policy.viewpoint_separation,
            what=f"{what} 'viewpoint_separation'",
        )
        _consistent(
            _number(record, "min_excursion", what),
            policy.min_excursion,
            what=f"{what} 'min_excursion'",
        )
        return policy


#: The policy this version publishes. Callers may pass their own; nothing in
#: this repository does, and a record states whichever was used.
DEFAULT_POLICY: Final = ReadinessPolicy()


# -- strict reading helpers -------------------------------------------------


def _exact_keys(record: Mapping[str, object], allowed: Sequence[str], what: str) -> None:
    """Refuse a record that is missing a field or carries an unknown one.

    Both directions, deliberately. A missing field is the ordinary defect; an
    unknown one is the interesting one, because it is how a record written by
    a different, richer schema gets read as though this one understood all of
    it.
    """
    present = set(record)
    expected = set(allowed)
    missing = sorted(expected - present)
    unknown = sorted(present - expected)
    if missing:
        raise ReadinessRecordUnusable(f"{what} is missing {missing}")
    if unknown:
        raise ReadinessRecordUnusable(
            f"{what} carries unknown field(s) {unknown}; this reader understands exactly "
            f"schema version {SCHEMA_VERSION} and never ignores what it does not know"
        )


def _mapping(record: Mapping[str, object], key: str, what: str) -> Mapping[str, object]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise ReadinessRecordUnusable(f"{what}: {key!r} is {type(value).__name__}, not a mapping")
    return value


def _boolean(record: Mapping[str, object], key: str, what: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ReadinessRecordUnusable(f"{what}: {key!r} is {value!r}, not a boolean")
    return value


def _integer(record: Mapping[str, object], key: str, what: str) -> int:
    value = record.get(key)
    # ``bool`` is an ``int`` in Python and JSON has no such rule. A record
    # saying ``"observations": true`` must be refused, not counted as one.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReadinessRecordUnusable(f"{what}: {key!r} is {value!r}, not an integer")
    return value


def _number(record: Mapping[str, object], key: str, what: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReadinessRecordUnusable(f"{what}: {key!r} is {value!r}, not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ReadinessRecordUnusable(f"{what}: {key!r} is {value!r}, which is not finite")
    return number


def _text(record: Mapping[str, object], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ReadinessRecordUnusable(f"{what}: {key!r} is {value!r}, not a string")
    if len(value) > DETAIL_LIMIT + len(TRUNCATION_MARK):
        raise ReadinessRecordUnusable(f"{what}: {key!r} is longer than this schema allows")
    return value


#: How far a stated derived value may differ from the recomputed one before the
#: record is called inconsistent. Wide enough for the last bit of a float that
#: made a round trip through JSON text, narrow enough that a wrong number is
#: never accepted as a rounding difference.
DERIVED_TOLERANCE: Final = 1e-9

#: Floor on the agreement downsampling cell, so a degenerate truncation cannot
#: ask :func:`scanner3d.registration.voxel_downsample` for a cell of zero.
MIN_AGREEMENT_CELL: Final = 1e-6


def _consistent(stated: float, derived: float, *, what: str) -> None:
    if not math.isclose(stated, derived, rel_tol=DERIVED_TOLERANCE, abs_tol=DERIVED_TOLERANCE):
        raise ReadinessRecordUnusable(
            f"{what} states {stated!r} but its own fields give {derived!r}; a record that "
            "contradicts itself is refused rather than believed"
        )


def _fraction(count: int, total: int) -> float:
    """``count / total``, and 0.0 rather than an error when nothing was seen."""
    return count / total if total else 0.0


# -- evidence ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrackingEvidence:
    """What the capture's own authority says about tracking loss."""

    #: How many ``tracking-loss.json`` records this capture holds, across the
    #: root sweep and every segment.
    records_found: int
    #: How many of them the authority accepted as closed by a validated
    #: recovery segment. A recovered episode is HISTORY, not a refusal.
    recovered_episodes: int
    #: True when at least one loss is not closed, or when the capture holding
    #: one could not be read at all.
    unresolved: bool
    #: The authority's own words, bounded. Empty when there was nothing to say.
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "records_found": self.records_found,
            "recovered_episodes": self.recovered_episodes,
            "unresolved": self.unresolved,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object], *, what: str) -> TrackingEvidence:
        _exact_keys(record, ("records_found", "recovered_episodes", "unresolved", "detail"), what)
        found = _integer(record, "records_found", what)
        recovered = _integer(record, "recovered_episodes", what)
        if found < 0 or recovered < 0 or recovered > found:
            raise ReadinessRecordUnusable(
                f"{what} claims {recovered} recovered episode(s) out of {found} record(s)"
            )
        # And the two STATES the authority can leave behind, which are not
        # every combination of those two counts.
        #
        # :func:`tracking_evidence` reaches "resolved" only by handing the
        # whole capture to the capture reader, which refuses any loss a
        # validated recovery segment does not close - so a resolved record has
        # closed every loss it found. It reaches "unresolved" only by returning
        # before that answer exists or without one at all, so
        # an unresolved or unreadable record has closed none. A record between
        # the two is claiming a completed recovery state its own verdict says
        # was never reached.
        unresolved = _boolean(record, "unresolved", what)
        if unresolved and recovered:
            raise ReadinessRecordUnusable(
                f"{what} states an unresolved or unreadable tracking state beside {recovered} "
                "recovered episode(s); the capture reader is what validates a recovery, and a "
                "record that never got an answer out of it has validated none"
            )
        if not unresolved and recovered != found:
            raise ReadinessRecordUnusable(
                f"{what} states a resolved tracking state with {recovered} of {found} loss "
                "record(s) recovered; the capture reader refuses any loss no validated "
                "recovery segment closed, so a resolved capture has closed all of them"
            )
        return cls(
            records_found=found,
            recovered_episodes=recovered,
            unresolved=unresolved,
            detail=_text(record, "detail", what),
        )


@dataclass(frozen=True, slots=True)
class PoseEvidence:
    """What the accepted trajectory measures, in the depth units of the capture."""

    #: How many poses were judged.
    poses: int
    #: And how many adjacent steps that gives - always ``poses - 1`` when there
    #: is anything to measure.
    adjacent_steps: int
    #: Steps whose surface travel exceeded the noise floor.
    moving_steps: int
    #: And the ones that did not. The two always sum to :attr:`adjacent_steps`.
    stationary_steps: int
    #: The distribution of that travel, so a reader can see whether a scan was
    #: slow or stopped without re-deriving it.
    travel_median: float
    travel_worst: float
    #: How many genuinely different viewpoints the sweep visited, counted
    #: greedily and capped.
    distinct_viewpoints: int
    #: True when counting stopped at the cap rather than at the end of the
    #: sequence, so the number above is a floor and not the total.
    viewpoints_capped: bool
    #: The extent of the probe surface's travel over the whole sweep.
    excursion: float
    #: How many points stood in for the object while all of this was measured.
    probe_points: int

    def as_dict(self) -> dict[str, object]:
        return {
            "poses": self.poses,
            "adjacent_steps": self.adjacent_steps,
            "moving_steps": self.moving_steps,
            "stationary_steps": self.stationary_steps,
            "travel_median": _json_number(self.travel_median),
            "travel_worst": _json_number(self.travel_worst),
            "distinct_viewpoints": self.distinct_viewpoints,
            "viewpoints_capped": self.viewpoints_capped,
            "excursion": _json_number(self.excursion),
            "probe_points": self.probe_points,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object], *, what: str) -> PoseEvidence:
        _exact_keys(
            record,
            (
                "poses",
                "adjacent_steps",
                "moving_steps",
                "stationary_steps",
                "travel_median",
                "travel_worst",
                "distinct_viewpoints",
                "viewpoints_capped",
                "excursion",
                "probe_points",
            ),
            what,
        )
        steps = _integer(record, "adjacent_steps", what)
        moving = _integer(record, "moving_steps", what)
        stationary = _integer(record, "stationary_steps", what)
        if min(steps, moving, stationary) < 0 or moving + stationary != steps:
            raise ReadinessRecordUnusable(
                f"{what} states {moving} moving and {stationary} stationary steps, which is not "
                f"the {steps} adjacent steps it also states"
            )
        count = _integer(record, "poses", what)
        viewpoints = _integer(record, "distinct_viewpoints", what)
        probe = _integer(record, "probe_points", what)
        if min(count, viewpoints, probe) < 0:
            raise ReadinessRecordUnusable(
                f"{what} states a negative pose, viewpoint or probe count "
                f"({count}, {viewpoints}, {probe})"
            )
        if steps != max(0, count - 1):
            raise ReadinessRecordUnusable(
                f"{what} states {steps} adjacent step(s) for {count} pose(s); a sequence of n "
                "poses has exactly n-1 adjacent steps"
            )
        if viewpoints > count:
            raise ReadinessRecordUnusable(
                f"{what} claims {viewpoints} distinct viewpoint(s) among {count} pose(s)"
            )
        median = _number(record, "travel_median", what)
        worst = _number(record, "travel_worst", what)
        excursion = _number(record, "excursion", what)
        if min(median, worst, excursion) < 0.0:
            raise ReadinessRecordUnusable(
                f"{what} states a negative travel or excursion "
                f"({median!r}, {worst!r}, {excursion!r}); these are distances"
            )
        if median > worst:
            raise ReadinessRecordUnusable(
                f"{what} states a median travel of {median!r} above its own worst of {worst!r}"
            )
        capped = _boolean(record, "viewpoints_capped", what)
        # The two SHAPES :func:`pose_evidence` emits. With no probe points or
        # fewer than two poses it measured nothing, and the shape it returns
        # says so in every field: no moving step, no travel, no excursion, one
        # viewpoint if there is a pose at all and no cap. A record carrying
        # measurements beside "nothing to measure against" is describing a
        # measurement that did not happen - and the reason set derived below
        # was derived from that same unmeasurable branch.
        if probe == 0 or count < 2:
            for name, value in (
                ("moving_steps", moving),
                ("travel_median", median),
                ("travel_worst", worst),
                ("excursion", excursion),
            ):
                if value != 0:
                    raise ReadinessRecordUnusable(
                        f"{what} states {name!r} of {value!r} with {probe} probe point(s) and "
                        f"{count} pose(s); there was nothing here to measure it against"
                    )
            if viewpoints != min(count, 1) or capped:
                raise ReadinessRecordUnusable(
                    f"{what} states {viewpoints} distinct viewpoint(s) (capped={capped}) with "
                    f"{probe} probe point(s) and {count} pose(s); an unmeasurable trajectory "
                    f"reports exactly {min(count, 1)} and never a cap"
                )
        elif viewpoints < 1:
            raise ReadinessRecordUnusable(
                f"{what} states {viewpoints} distinct viewpoint(s) among {count} placed pose(s); "
                "a placement is always a viewpoint of itself"
            )
        return cls(
            poses=count,
            adjacent_steps=steps,
            moving_steps=moving,
            stationary_steps=stationary,
            travel_median=median,
            travel_worst=worst,
            distinct_viewpoints=viewpoints,
            viewpoints_capped=capped,
            excursion=excursion,
            probe_points=probe,
        )


@dataclass(frozen=True, slots=True)
class CoverageEvidence:
    """How much of the OCCUPIED surface was seen repeatedly and from a spread."""

    #: False when :func:`scanner3d.coverage.coverage_map` refused the inputs;
    #: every count below is then zero and :attr:`detail` says why.
    measured: bool
    frames_offered: int
    frames_used: int
    points_used: int
    cell_size: float
    #: Occupied cells. Empty space is not among them and never counts against
    #: a scan.
    total_cells: int
    #: Of those, the ones meeting both the observation and the spread policy.
    supported_cells: int
    supported_fraction: float
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "measured": self.measured,
            "frames_offered": self.frames_offered,
            "frames_used": self.frames_used,
            "points_used": self.points_used,
            "cell_size": _json_number(self.cell_size),
            "total_cells": self.total_cells,
            "supported_cells": self.supported_cells,
            "supported_fraction": _json_number(self.supported_fraction),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object], *, what: str) -> CoverageEvidence:
        _exact_keys(
            record,
            (
                "measured",
                "frames_offered",
                "frames_used",
                "points_used",
                "cell_size",
                "total_cells",
                "supported_cells",
                "supported_fraction",
                "detail",
            ),
            what,
        )
        total = _integer(record, "total_cells", what)
        supported = _integer(record, "supported_cells", what)
        if min(total, supported) < 0 or supported > total:
            raise ReadinessRecordUnusable(
                f"{what} claims {supported} supported of {total} occupied cells"
            )
        used = _integer(record, "frames_used", what)
        offered = _integer(record, "frames_offered", what)
        if min(used, offered) < 0 or used > offered:
            raise ReadinessRecordUnusable(f"{what} claims {used} frames used of {offered} offered")
        points_used = _integer(record, "points_used", what)
        cell_size = _number(record, "cell_size", what)
        if points_used < 0 or cell_size < 0.0:
            raise ReadinessRecordUnusable(
                f"{what} states {points_used} point(s) at a cell size of {cell_size!r}; neither "
                "a point count nor a cell size is ever negative"
            )
        _consistent(
            _number(record, "supported_fraction", what),
            _fraction(supported, total),
            what=f"{what} 'supported_fraction'",
        )
        measured = _boolean(record, "measured", what)
        # "Not measured" is exactly the branch where
        # :func:`scanner3d.coverage.coverage_map` REFUSED the inputs, so there
        # is no map behind it: no occupied cell was found and none was
        # supported. The frame and point counts stay, because those describe
        # what was offered to it rather than what came back. A record claiming
        # a populated map beside its own "this could not be measured" is
        # carrying evidence from a call that did not return one.
        if not measured and (total or supported):
            raise ReadinessRecordUnusable(
                f"{what} states that coverage could not be measured beside {supported} "
                f"supported of {total} occupied cell(s); a refused coverage map produces no "
                "cells at all"
            )
        return cls(
            measured=measured,
            frames_offered=offered,
            frames_used=used,
            points_used=points_used,
            cell_size=cell_size,
            total_cells=total,
            supported_cells=supported,
            supported_fraction=_fraction(supported, total),
            detail=_text(record, "detail", what),
        )


@dataclass(frozen=True, slots=True)
class SurfaceAgreement:
    """Symmetric agreement between the extracted surface and its own evidence.

    Explicitly INTERNAL CONSISTENCY. Both directions are always here because
    either one alone passes a reconstruction the other fails: surface-to-
    evidence is blind to a surface that abandoned half the scan, and
    evidence-to-surface is blind to geometry that should not exist at all.

    The evidence is a bounded sample of the FULL-RESOLUTION frames the volume
    integrated - see :class:`FusionEvidence` - and the three source counts
    below record which frames and how much of them, so a reader can tell what
    the fractions above are fractions OF.
    """

    #: False when the agreement could not be measured; the numbers below are
    #: then zero and :attr:`detail` says why.
    measured: bool
    voxel_size: float
    truncation: float
    #: The distance within which a point counts as supported, derived from the
    #: two values above and the noise floor by
    #: :meth:`ReadinessPolicy.surface_tolerance`, then widened by the sampling
    #: cell below so the measurement is not measuring its own downsampling.
    tolerance: float
    #: The voxel cell both clouds were downsampled at before comparison.
    sample_cell: float
    surface_points: int
    evidence_points: int
    #: How many full-resolution frames the volume was offered. The evidence
    #: below is a sample of THOSE frames - the ones fusion actually consumed -
    #: and not of the decimated clouds the registration was fitted to. The
    #: three counts are here so a reader can see the bound the sample was taken
    #: under rather than infer it: a record that examined 64 of 176 frames at
    #: 2048 candidate pixels each says so.
    source_frames_offered: int
    #: How many of them were sampled, by uniform stride.
    source_frames_sampled: int
    #: How many candidate pixels were examined across those frames. Always at
    #: least :attr:`evidence_points` before downsampling, because a candidate
    #: with no return, or outside the fusion's own accepted depth interval, is
    #: skipped exactly as the fusion skipped it.
    source_samples_examined: int
    #: How many points that produced, before the comparison downsampling.
    source_points: int
    #: Share of surface vertices within :attr:`tolerance` of a registered point.
    surface_supported_fraction: float
    #: Share of registered points within :attr:`tolerance` of the surface.
    evidence_explained_fraction: float
    surface_distance_median: float
    surface_distance_p95: float
    evidence_distance_median: float
    evidence_distance_p95: float
    #: Distances are exact below this value and clamped AT it above, because
    #: the search that produced them is exact only inside :attr:`tolerance`. A
    #: recorded distance equal to this means "this far or further", never
    #: "exactly this far", and in particular does not say whether the point was
    #: counted: support is INCLUSIVE at the tolerance, so a point at exactly
    #: this distance is supported.
    distance_clamp: float
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "measured": self.measured,
            "voxel_size": _json_number(self.voxel_size),
            "truncation": _json_number(self.truncation),
            "tolerance": _json_number(self.tolerance),
            "sample_cell": _json_number(self.sample_cell),
            "surface_points": self.surface_points,
            "evidence_points": self.evidence_points,
            "source_frames_offered": self.source_frames_offered,
            "source_frames_sampled": self.source_frames_sampled,
            "source_samples_examined": self.source_samples_examined,
            "source_points": self.source_points,
            "surface_supported_fraction": _json_number(self.surface_supported_fraction),
            "evidence_explained_fraction": _json_number(self.evidence_explained_fraction),
            "surface_distance_median": _json_number(self.surface_distance_median),
            "surface_distance_p95": _json_number(self.surface_distance_p95),
            "evidence_distance_median": _json_number(self.evidence_distance_median),
            "evidence_distance_p95": _json_number(self.evidence_distance_p95),
            "distance_clamp": _json_number(self.distance_clamp),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object], *, what: str) -> SurfaceAgreement:
        _exact_keys(
            record,
            (
                "measured",
                "voxel_size",
                "truncation",
                "tolerance",
                "sample_cell",
                "surface_points",
                "evidence_points",
                "source_frames_offered",
                "source_frames_sampled",
                "source_samples_examined",
                "source_points",
                "surface_supported_fraction",
                "evidence_explained_fraction",
                "surface_distance_median",
                "surface_distance_p95",
                "evidence_distance_median",
                "evidence_distance_p95",
                "distance_clamp",
                "detail",
            ),
            what,
        )
        surface_fraction = _number(record, "surface_supported_fraction", what)
        evidence_fraction = _number(record, "evidence_explained_fraction", what)
        for name, value in (
            ("surface_supported_fraction", surface_fraction),
            ("evidence_explained_fraction", evidence_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ReadinessRecordUnusable(f"{what}: {name!r} is {value!r}, not a fraction")
        # Domains, so a record cannot state a negative count, a negative length
        # or a sample bigger than the population it was drawn from. None of
        # these is reachable from a real evaluation, which is exactly why a
        # record carrying one is not a real evaluation.
        frames_offered = _integer(record, "source_frames_offered", what)
        frames_sampled = _integer(record, "source_frames_sampled", what)
        samples_examined = _integer(record, "source_samples_examined", what)
        source_points = _integer(record, "source_points", what)
        surface_points = _integer(record, "surface_points", what)
        evidence_points = _integer(record, "evidence_points", what)
        for name, count in (
            ("surface_points", surface_points),
            ("evidence_points", evidence_points),
            ("source_frames_offered", frames_offered),
            ("source_frames_sampled", frames_sampled),
            ("source_samples_examined", samples_examined),
            ("source_points", source_points),
        ):
            if count < 0:
                raise ReadinessRecordUnusable(f"{what}: {name!r} is {count!r}, not a count")
        if frames_sampled > frames_offered:
            raise ReadinessRecordUnusable(
                f"{what} claims {frames_sampled} of {frames_offered} full-resolution frame(s) "
                "were sampled"
            )
        if source_points > samples_examined:
            raise ReadinessRecordUnusable(
                f"{what} claims {source_points} evidence point(s) from {samples_examined} "
                "examined sample(s); a sample cannot yield more points than it examined"
            )
        # The comparison populations are DOWNSAMPLED from the deprojected
        # evidence, never added to it: the pipeline is examined candidates ->
        # source points -> one point per comparison cell -> the strided
        # queries this field counts. Each step can only shrink, so a record
        # whose thinned cloud outnumbers the cloud it was thinned from has one
        # of the two counts invented.
        if evidence_points > source_points:
            raise ReadinessRecordUnusable(
                f"{what} claims {evidence_points} downsampled evidence point(s) from "
                f"{source_points} deprojected point(s); downsampling never produces more "
                "points than it was given"
            )
        for name in (
            "voxel_size",
            "truncation",
            "tolerance",
            "sample_cell",
            "surface_distance_median",
            "surface_distance_p95",
            "evidence_distance_median",
            "evidence_distance_p95",
            "distance_clamp",
        ):
            length = _number(record, name, what)
            if length < 0.0:
                raise ReadinessRecordUnusable(
                    f"{what}: {name!r} is {length!r}; a length here is never negative"
                )
        for median, p95, direction in (
            (
                _number(record, "surface_distance_median", what),
                _number(record, "surface_distance_p95", what),
                "surface",
            ),
            (
                _number(record, "evidence_distance_median", what),
                _number(record, "evidence_distance_p95", what),
                "evidence",
            ),
        ):
            clamp = _number(record, "distance_clamp", what)
            if median > p95 or p95 > clamp:
                raise ReadinessRecordUnusable(
                    f"{what} states a {direction} distance median {median!r}, p95 {p95!r} and "
                    f"clamp {clamp!r}, which cannot all be true of one distribution"
                )
        # The clamp is not a separate policy value: the search is exact only
        # inside the tolerance it was given, so the two are one number written
        # twice, and a record where they differ describes a search that was
        # never run.
        _consistent(
            _number(record, "distance_clamp", what),
            _number(record, "tolerance", what),
            what=f"{what} 'distance_clamp'",
        )
        # And the two SHAPES :func:`surface_evidence` emits.
        measured = _boolean(record, "measured", what)
        if measured:
            # It measured, so both directions had somebody to ask and
            # something to ask about. Zero of either is precisely the
            # condition under which it does NOT measure.
            for name, count in (
                ("surface_points", surface_points),
                ("evidence_points", evidence_points),
                ("source_points", source_points),
                ("source_frames_sampled", frames_sampled),
            ):
                if count == 0:
                    raise ReadinessRecordUnusable(
                        f"{what} states a measured agreement with {name!r} of zero; a direction "
                        "with no query or no target population is what 'not measured' means"
                    )
        else:
            if surface_points and evidence_points:
                raise ReadinessRecordUnusable(
                    f"{what} states that agreement could not be measured beside "
                    f"{surface_points} surface and {evidence_points} evidence point(s); with "
                    "both populations present the comparison is what happens"
                )
            for name in (
                "surface_supported_fraction",
                "evidence_explained_fraction",
                "surface_distance_median",
                "surface_distance_p95",
                "evidence_distance_median",
                "evidence_distance_p95",
            ):
                value = _number(record, name, what)
                if value != 0.0:
                    raise ReadinessRecordUnusable(
                        f"{what} states {name!r} of {value!r} beside its own 'this could not be "
                        "measured'; an unmeasured agreement reports no fraction and no distance"
                    )
        return cls(
            measured=measured,
            voxel_size=_number(record, "voxel_size", what),
            truncation=_number(record, "truncation", what),
            tolerance=_number(record, "tolerance", what),
            sample_cell=_number(record, "sample_cell", what),
            surface_points=surface_points,
            evidence_points=evidence_points,
            source_frames_offered=frames_offered,
            source_frames_sampled=frames_sampled,
            source_samples_examined=samples_examined,
            source_points=source_points,
            surface_supported_fraction=surface_fraction,
            evidence_explained_fraction=evidence_fraction,
            surface_distance_median=_number(record, "surface_distance_median", what),
            surface_distance_p95=_number(record, "surface_distance_p95", what),
            evidence_distance_median=_number(record, "evidence_distance_median", what),
            evidence_distance_p95=_number(record, "evidence_distance_p95", what),
            distance_clamp=_number(record, "distance_clamp", what),
            detail=_text(record, "detail", what),
        )


@dataclass(frozen=True, slots=True)
class ReadinessFailure:
    """One failed gate: what, why, and what to do about it."""

    reason: str
    detail: str
    advice: str

    def as_dict(self) -> dict[str, object]:
        return {"reason": self.reason, "detail": self.detail, "advice": self.advice}

    @classmethod
    def from_dict(cls, record: Mapping[str, object], *, what: str) -> ReadinessFailure:
        _exact_keys(record, ("reason", "detail", "advice"), what)
        reason = _text(record, "reason", what)
        if reason not in ADVICE:
            raise ReadinessRecordUnusable(
                f"{what} states reason {reason!r}, which is not one this version publishes"
            )
        advice = _text(record, "advice", what)
        if advice != ADVICE[reason]:
            raise ReadinessRecordUnusable(
                f"{what} carries advice that is not this version's advice for {reason!r}"
            )
        return cls(reason=reason, detail=_text(record, "detail", what), advice=advice)


def _failure(reason: str, detail: str) -> ReadinessFailure:
    return ReadinessFailure(reason=reason, detail=_bounded(detail), advice=ADVICE[reason])


#: The tracking gate's two reasons. They answer one question - is there an
#: unresolved loss in this capture - and differ only in whether the authority
#: could read the record at all, so a record must carry exactly one of them or
#: neither, and which one is not derivable from the evidence.
TRACKING_REASONS: Final[tuple[str, ...]] = (TRACKING_LOST, TRACKING_LOSS_UNREADABLE)


def _required_reasons(
    policy: ReadinessPolicy,
    poses: PoseEvidence,
    coverage: CoverageEvidence,
    surface: SurfaceAgreement | None,
) -> tuple[str, ...]:
    """Exactly the non-tracking reasons this evidence obliges, in report order.

    Derived from the record's own evidence under the record's own policy, which
    is the point: a reader that parses the failure list without recomputing it
    accepts a record claiming adequate coverage beside a coverage refusal, or a
    record with a coverage refusal missing beside inadequate coverage, and
    either one is a forged verdict that reads as a real one.

    Tracking is deliberately not here - see :data:`TRACKING_REASONS`.
    """
    required: list[str] = []
    # The pose gate answers ONE of two shapes. With nothing to measure against
    # it reports only that, and the three thresholds below were never
    # evaluated; a record naming one of them anyway is describing a measurement
    # that did not happen.
    if poses.probe_points == 0 or poses.poses < 2:
        required.append(POSE_MOTION_NOT_MEASURABLE)
    else:
        if poses.distinct_viewpoints < policy.min_distinct_viewpoints:
            required.append(POSE_VIEWPOINTS_INSUFFICIENT)
        if poses.excursion < policy.min_excursion:
            required.append(POSE_EXCURSION_INSUFFICIENT)
        if poses.moving_steps < policy.min_moving_steps:
            required.append(POSE_MOTION_BELOW_NOISE)
    if not coverage.measured:
        required.append(COVERAGE_NOT_MEASURABLE)
    elif coverage.supported_fraction < policy.coverage_min_supported_fraction:
        required.append(COVERAGE_UNSUPPORTED)
    # ``None`` is an honest pre-fusion answer and obliges nothing: the volume
    # does not exist, so the surface gate has not run and has no verdict to be
    # missing.
    if surface is not None:
        if not surface.measured:
            required.append(SURFACE_NOT_MEASURABLE)
        else:
            if surface.surface_supported_fraction < policy.surface_min_supported_fraction:
                required.append(SURFACE_UNSUPPORTED)
            if surface.evidence_explained_fraction < policy.surface_min_evidence_explained:
                required.append(SURFACE_ABANDONS_EVIDENCE)
    return tuple(sorted(required, key=REASONS.index))


def _validate_decision(
    stated: Sequence[str],
    *,
    policy: ReadinessPolicy,
    tracking: TrackingEvidence,
    poses: PoseEvidence,
    coverage: CoverageEvidence,
    surface: SurfaceAgreement | None,
    what: str,
) -> None:
    """Read the failure list and the evidence as ONE statement, or refuse both.

    Three separate things have to hold, and a record failing any of them is a
    record whose decision is not the decision its own numbers support:

    * the reasons are in the canonical order this schema publishes, so two
      records of one verdict cannot differ in shape;
    * the tracking half carries exactly one refusal when the capture's own
      authority left a loss unresolved, and none when it did not - a validated
      recovery is history and obliges nothing;
    * every other reason present is one the evidence obliges, and every reason
      the evidence obliges is present.

    Then :func:`_validate_sampling` reads the counts against the bounds the
    same policy put on them, which is the other half of the same idea: a record
    is refused for contradicting its own policy as readily as for contradicting
    its own evidence.
    """
    order = [REASONS.index(reason) for reason in stated]
    if order != sorted(order):
        raise ReadinessRecordUnusable(
            f"{what} lists its reasons as {list(stated)}, which is not the canonical order "
            f"{[reason for reason in REASONS if reason in set(stated)]} this schema publishes"
        )
    tracking_stated = [reason for reason in stated if reason in TRACKING_REASONS]
    if tracking.unresolved and len(tracking_stated) != 1:
        raise ReadinessRecordUnusable(
            f"{what} states an unresolved tracking loss beside {len(tracking_stated)} tracking "
            "refusal(s); a capture whose own authority did not close a loss is refused, and a "
            "record saying otherwise is the exact failure these gates exist to prevent"
        )
    if not tracking.unresolved and tracking_stated:
        raise ReadinessRecordUnusable(
            f"{what} states {tracking_stated} while its tracking evidence is resolved"
        )
    required = _required_reasons(policy, poses, coverage, surface)
    present = tuple(reason for reason in stated if reason not in TRACKING_REASONS)
    if present != required:
        missing = [reason for reason in required if reason not in present]
        spurious = [reason for reason in present if reason not in required]
        raise ReadinessRecordUnusable(
            f"{what} states {list(present)} where its own evidence and policy give "
            f"{list(required)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unsupported {spurious}" if spurious else "")
        )
    _validate_sampling(policy=policy, poses=poses, coverage=coverage, surface=surface, what=what)


def _validate_sampling(
    *,
    policy: ReadinessPolicy,
    poses: PoseEvidence,
    coverage: CoverageEvidence,
    surface: SurfaceAgreement | None,
    what: str,
) -> None:
    """Every count against the bound the record's own policy put on it.

    Each gate here is bounded on purpose - see this module's Cost section - and
    the policy that bounded it travels in the same record. So a count above its
    own stated ceiling is not an unusually large scan; it is a count no run
    under that policy produced.
    """
    if poses.distinct_viewpoints > policy.max_viewpoints_examined:
        raise ReadinessRecordUnusable(
            f"{what} claims {poses.distinct_viewpoints} distinct viewpoint(s) where its own "
            f"policy stops counting at {policy.max_viewpoints_examined}"
        )
    # Counting stops AT the cap, so a record that says it stopped there and
    # names a different number is describing two different runs.
    if poses.viewpoints_capped and poses.distinct_viewpoints != policy.max_viewpoints_examined:
        raise ReadinessRecordUnusable(
            f"{what} states that viewpoint counting hit its cap at "
            f"{poses.distinct_viewpoints} viewpoint(s) where its own policy caps counting at "
            f"{policy.max_viewpoints_examined}"
        )
    if poses.probe_points > policy.probe_points:
        raise ReadinessRecordUnusable(
            f"{what} claims {poses.probe_points} probe point(s) where its own policy takes at "
            f"most {policy.probe_points}"
        )
    if coverage.frames_used > policy.coverage_max_frames:
        raise ReadinessRecordUnusable(
            f"{what} claims {coverage.frames_used} coverage frame(s) where its own policy "
            f"offers at most {policy.coverage_max_frames}"
        )
    if coverage.points_used > coverage.frames_used * policy.coverage_points_per_frame:
        raise ReadinessRecordUnusable(
            f"{what} claims {coverage.points_used} coverage point(s) from "
            f"{coverage.frames_used} frame(s) where its own policy takes at most "
            f"{policy.coverage_points_per_frame} of each"
        )
    if surface is None:
        return
    if surface.source_frames_sampled > policy.fusion_sample_frames:
        raise ReadinessRecordUnusable(
            f"{what} claims {surface.source_frames_sampled} sampled fusion frame(s) where its "
            f"own policy samples at most {policy.fusion_sample_frames}"
        )
    examined_ceiling = surface.source_frames_sampled * policy.fusion_pixels_per_frame
    if surface.source_samples_examined > examined_ceiling:
        raise ReadinessRecordUnusable(
            f"{what} claims {surface.source_samples_examined} candidate sample(s) examined "
            f"across {surface.source_frames_sampled} frame(s) where its own policy examines at "
            f"most {policy.fusion_pixels_per_frame} pixel(s) of each"
        )
    for name, count in (
        ("surface_points", surface.surface_points),
        ("evidence_points", surface.evidence_points),
    ):
        if count > policy.surface_sample_points:
            raise ReadinessRecordUnusable(
                f"{what} claims {count} compared {name!r} where its own policy asks at most "
                f"{policy.surface_sample_points} of either cloud"
            )


# -- the result -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconstructionReadiness:
    """Whether this reconstruction may be published, and everything behind it.

    :attr:`ready` is DERIVED - it is exactly "no gate failed" - so a result
    that says it is ready and carries failures cannot be constructed, and one
    read back from JSON claiming that combination is refused.
    """

    policy: ReadinessPolicy
    tracking: TrackingEvidence
    poses: PoseEvidence
    coverage: CoverageEvidence
    #: ``None`` until fusion has happened. A pre-fusion result is a complete,
    #: honest answer about everything knowable before the volume exists.
    surface: SurfaceAgreement | None
    failures: tuple[ReadinessFailure, ...]

    @property
    def ready(self) -> bool:
        """True only when every applicable gate passed."""
        return not self.failures

    @property
    def reasons(self) -> tuple[str, ...]:
        """Every failed gate's reason code, in report order."""
        return tuple(failure.reason for failure in self.failures)

    def with_surface(
        self, surface: SurfaceAgreement, failures: Sequence[ReadinessFailure]
    ) -> ReconstructionReadiness:
        """This result plus the post-fusion gate, keeping every earlier failure.

        Additive by construction: the pre-fusion failures are carried through
        rather than replaced, so the record a refused export leaves behind
        names everything that was wrong and not only the last thing measured.
        """
        return ReconstructionReadiness(
            policy=self.policy,
            tracking=self.tracking,
            poses=self.poses,
            coverage=self.coverage,
            surface=surface,
            failures=(*self.failures, *failures),
        )

    def refusal_text(self) -> str:
        """The whole refusal, with the most actionable sentence LAST.

        Last because that is the line the window's failure dialog shows; the
        report and the terminal get all of it.
        """
        if self.ready:
            return "reconstruction readiness: every applicable gate passed"
        head = (
            f"RECONSTRUCTION REFUSED. {len(self.failures)} readiness "
            f"gate{'s' if len(self.failures) != 1 else ''} failed, so nothing was published."
        )
        lines = [head]
        for failure in self.failures:
            lines.append(f"  - {failure.reason}: {failure.detail}")
        lines.append("")
        lines.append(f"What to do: {self.failures[0].advice}.")
        return "\n".join(lines)

    def summary(self) -> str:
        """One line, for a status bar or a log."""
        if self.ready:
            return "readiness: ready, every applicable gate passed"
        return f"readiness: REFUSED ({', '.join(self.reasons)})"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "ready": self.ready,
            "failures": [failure.as_dict() for failure in self.failures],
            "policy": self.policy.as_dict(),
            "tracking": self.tracking.as_dict(),
            "poses": self.poses.as_dict(),
            "coverage": self.coverage.as_dict(),
            "surface": None if self.surface is None else self.surface.as_dict(),
            "limitations": list(LIMITATIONS),
        }

    @classmethod
    def from_dict(
        cls, record: Mapping[str, object], *, what: str = "readiness record"
    ) -> ReconstructionReadiness:
        """Read a record back, or refuse it. Never a partial or guessed result."""
        if not isinstance(record, Mapping):
            raise ReadinessRecordUnusable(f"{what} is not a JSON object")
        _exact_keys(
            record,
            (
                "kind",
                "schema_version",
                "ready",
                "failures",
                "policy",
                "tracking",
                "poses",
                "coverage",
                "surface",
                "limitations",
            ),
            what,
        )
        kind = _text(record, "kind", what)
        if kind != KIND:
            raise ReadinessRecordUnusable(f"{what} states kind {kind!r}, not {KIND!r}")
        version = record.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool) or version != SCHEMA_VERSION:
            raise ReadinessRecordUnusable(
                f"{what} states schema version {version!r}; this reader understands exactly "
                f"version {SCHEMA_VERSION} and does not guess at an earlier or a later one"
            )
        raw_failures = record.get("failures")
        if not isinstance(raw_failures, list):
            raise ReadinessRecordUnusable(f"{what}: 'failures' is not a list")
        parsed: list[ReadinessFailure] = []
        for index, entry in enumerate(raw_failures):
            if not isinstance(entry, Mapping):
                raise ReadinessRecordUnusable(f"{what} failure {index} is not a JSON object")
            parsed.append(ReadinessFailure.from_dict(entry, what=f"{what} failure {index}"))
        failures = tuple(parsed)
        seen: set[str] = set()
        for failure in failures:
            if failure.reason in seen:
                raise ReadinessRecordUnusable(
                    f"{what} states reason {failure.reason!r} more than once"
                )
            seen.add(failure.reason)
        # The decision is derived here exactly as it is derived in memory, and
        # the stated one has to agree. A record that says it is ready while
        # naming failures is the one shape this whole module exists to refuse.
        stated_ready = _boolean(record, "ready", what)
        if stated_ready != (not failures):
            raise ReadinessRecordUnusable(
                f"{what} states ready={stated_ready} beside {len(failures)} failure(s); a "
                "record whose decision contradicts its own evidence is refused"
            )
        limitations = record.get("limitations")
        if limitations != list(LIMITATIONS):
            raise ReadinessRecordUnusable(
                f"{what} does not carry this version's limitations; they are the record's own "
                "statement of what it does not establish and are not optional"
            )
        surface_record = record.get("surface")
        surface: SurfaceAgreement | None = None
        if surface_record is not None:
            if not isinstance(surface_record, Mapping):
                raise ReadinessRecordUnusable(f"{what} 'surface' is not a JSON object")
            surface = SurfaceAgreement.from_dict(surface_record, what=f"{what} 'surface'")
        policy = ReadinessPolicy.from_dict(
            _mapping(record, "policy", what), what=f"{what} 'policy'"
        )
        tracking = TrackingEvidence.from_dict(
            _mapping(record, "tracking", what), what=f"{what} 'tracking'"
        )
        poses = PoseEvidence.from_dict(_mapping(record, "poses", what), what=f"{what} 'poses'")
        coverage = CoverageEvidence.from_dict(
            _mapping(record, "coverage", what), what=f"{what} 'coverage'"
        )
        # And LAST, the whole thing as one statement. Every check above reads
        # one field or one object; this one reads the decision against the
        # evidence and the policy that were supposed to have produced it, which
        # is the only check that can catch a record whose every part is
        # individually well formed and whose verdict is nonetheless invented.
        _validate_decision(
            [failure.reason for failure in failures],
            policy=policy,
            tracking=tracking,
            poses=poses,
            coverage=coverage,
            surface=surface,
            what=what,
        )
        return cls(
            policy=policy,
            tracking=tracking,
            poses=poses,
            coverage=coverage,
            surface=surface,
            failures=failures,
        )


# -- gate 1: the capture's own tracking authority ---------------------------


def tracking_evidence(frames_dir: Path) -> tuple[TrackingEvidence, tuple[ReadinessFailure, ...]]:
    """What the capture reader says about this folder's tracking losses.

    The whole point is that this is NOT a second opinion.
    :func:`scanner3d.capture_sequence.read_capture_sequence` owns every rule
    about what closes a loss - the byte binding, the excluded-frame digests,
    the segment ordering - and none of it is duplicated, weakened or
    reimplemented here. What this adds is *reach*: an entry point that was
    handed a capture directory and a list of poses, and never asked the reader
    anything, now asks it.

    The cheap path is preserved exactly. A capture with no loss record costs
    one ``stat`` per run and the authority is never opened at all, which is
    what keeps this affordable on the export path.
    """
    from scanner3d.capture_sequence import (
        SEGMENT_GLOB,
        CaptureSequenceUnreadable,
        TrackingLossRecordUnusable,
        read_capture_sequence,
        read_tracking_loss,
    )

    runs: list[Path] = [frames_dir]
    try:
        runs.extend(sorted(path for path in frames_dir.glob(SEGMENT_GLOB) if path.is_dir()))
    except OSError as exc:
        return (
            TrackingEvidence(0, 0, unresolved=True, detail=_bounded(str(exc))),
            (_failure(TRACKING_LOSS_UNREADABLE, f"{frames_dir} could not be listed: {exc}"),),
        )
    found = 0
    for run in runs:
        try:
            if read_tracking_loss(run) is not None:
                found += 1
        except TrackingLossRecordUnusable as exc:
            evidence = TrackingEvidence(found, 0, unresolved=True, detail=_bounded(str(exc)))
            return evidence, (_failure(TRACKING_LOSS_UNREADABLE, str(exc)),)
    if found == 0:
        return TrackingEvidence(0, 0, unresolved=False, detail=""), ()
    # Something in here says the scan lost the object. From this point the
    # authority decides, and its refusal is quoted rather than paraphrased.
    try:
        sequence = read_capture_sequence(frames_dir)
    except CaptureSequenceUnreadable as exc:
        evidence = TrackingEvidence(found, 0, unresolved=True, detail=_bounded(str(exc)))
        return evidence, (_failure(TRACKING_LOST, str(exc)),)
    recovered = len(sequence.recoveries)
    detail = (
        f"{recovered} tracking-loss episode(s) in this capture were closed by a recovery segment "
        f"the capture reader validated; {len(sequence.excluded)} frame(s) are on disk as the "
        "evidence for those losses and are not part of the scan"
    )
    return TrackingEvidence(found, recovered, unresolved=False, detail=_bounded(detail)), ()


# -- gate 2: does this trajectory describe a scan? --------------------------


def _stride(count: int, limit: int) -> int:
    """The smallest stride that brings ``count`` items within ``limit``."""
    return 1 if count <= limit else -(-count // limit)


def _strided(items: Sequence[Point], limit: int) -> tuple[Point, ...]:
    """A deterministic subsample. Uniform stride, input order, never random."""
    return tuple(items[:: _stride(len(items), limit)])[:limit]


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _placement_distance(a: Sequence[Point], b: Sequence[Point]) -> float:
    """How far the probe surface moved between two placements of it."""
    return _median([math.dist(left, right) for left, right in zip(a, b, strict=True)])


def pose_evidence(
    clouds: Sequence[Sequence[Point]],
    poses: Sequence[Rigid],
    *,
    policy: ReadinessPolicy = DEFAULT_POLICY,
) -> tuple[PoseEvidence, tuple[ReadinessFailure, ...]]:
    """Measure the trajectory against the calibrated motion vocabulary.

    Three separate questions, because a manufactured pose sequence fails them
    in different combinations and one number could not say which:

    * how far the surface moved between ADJACENT frames, which is exactly
      :func:`scanner3d.pipeline.frame_travel` - the same quantity the live
      tracker gates on - judged against
      :data:`scanner3d.pipeline.NOISE_FLOOR`;
    * how far it moved over the WHOLE sweep, measured as the extent of each
      probe point's track rather than the distance between the first and last
      pose, so an orbit that comes back to its starting view is not mistaken
      for a scan that never moved;
    * how many genuinely DIFFERENT viewpoints were visited, counted greedily
      against a separation of one useful scanning step, so a sequence that is
      stuck between a couple of places fails even when its endpoints are far
      apart.

    No object identity and no exact float equality is used anywhere: two poses
    are the same viewpoint when the surface they place agrees to within the
    calibrated separation, which is what makes a "substantially repeated"
    sequence fail rather than only a byte-identical one.
    """
    if len(clouds) != len(poses):
        raise ReadinessNotMeasurable(
            f"{len(clouds)} cloud(s) were offered with {len(poses)} pose(s); readiness cannot "
            "be measured for a trajectory that does not describe the frames"
        )
    probe: tuple[Point, ...] = ()
    for cloud in clouds:
        if cloud:
            probe = _strided(cloud, policy.probe_points)
            break
    if not probe or len(poses) < 2:
        detail = (
            f"{len(poses)} pose(s) and {sum(1 for cloud in clouds if cloud)} non-empty cloud(s): "
            "there is nothing here to measure viewpoint motion against"
        )
        evidence = PoseEvidence(
            poses=len(poses),
            adjacent_steps=max(0, len(poses) - 1),
            moving_steps=0,
            stationary_steps=max(0, len(poses) - 1),
            travel_median=0.0,
            travel_worst=0.0,
            distinct_viewpoints=min(len(poses), 1),
            viewpoints_capped=False,
            excursion=0.0,
            probe_points=len(probe),
        )
        return evidence, (_failure(POSE_MOTION_NOT_MEASURABLE, detail),)

    # One placement of the probe per pose. Everything below reads these; the
    # cost is poses x probe points, once.
    placements = [tuple(pose.apply_all(probe)) for pose in poses]

    # Adjacent travel, through the shared definition rather than a second one.
    travels = [
        frame_travel(probe, poses[index].inverse().compose(poses[index + 1]))
        for index in range(len(poses) - 1)
    ]
    moving = sum(1 for travel in travels if travel >= policy.noise_floor)

    # Excursion: the extent of each probe point's own track, then the median of
    # those. Linear, and insensitive to where the sweep happens to end.
    extents: list[float] = []
    for column in range(len(probe)):
        track = [placement[column] for placement in placements]
        low = tuple(min(point[axis] for point in track) for axis in range(3))
        high = tuple(max(point[axis] for point in track) for axis in range(3))
        extents.append(math.dist(low, high))
    excursion = _median(extents)

    # Distinct viewpoints, greedily and capped.
    separation = policy.viewpoint_separation
    representatives: list[tuple[Point, ...]] = []
    capped = False
    for placement in placements:
        if any(_placement_distance(placement, chosen) < separation for chosen in representatives):
            continue
        if len(representatives) >= policy.max_viewpoints_examined:
            capped = True
            break
        representatives.append(placement)

    evidence = PoseEvidence(
        poses=len(poses),
        adjacent_steps=len(travels),
        moving_steps=moving,
        stationary_steps=len(travels) - moving,
        travel_median=_median(travels),
        travel_worst=max(travels, default=0.0),
        distinct_viewpoints=len(representatives),
        viewpoints_capped=capped,
        excursion=excursion,
        probe_points=len(probe),
    )

    failures: list[ReadinessFailure] = []
    # Boundary policy, stated once and the same for all three: the measured
    # value PASSES when it is at least the required one.
    if evidence.distinct_viewpoints < policy.min_distinct_viewpoints:
        failures.append(
            _failure(
                POSE_VIEWPOINTS_INSUFFICIENT,
                f"{evidence.distinct_viewpoints} distinct viewpoint(s) among {len(poses)} poses, "
                f"at a separation of {separation:.3f}; policy requires at least "
                f"{policy.min_distinct_viewpoints}",
            )
        )
    if excursion < policy.min_excursion:
        failures.append(
            _failure(
                POSE_EXCURSION_INSUFFICIENT,
                f"the probe surface moved {excursion:.3f} over the whole sweep; policy requires "
                f"at least {policy.min_excursion:.3f} "
                f"({policy.min_excursion_multiple:g} x the {policy.noise_floor:g} noise floor)",
            )
        )
    if moving < policy.min_moving_steps:
        failures.append(
            _failure(
                POSE_MOTION_BELOW_NOISE,
                f"{moving} of {len(travels)} adjacent step(s) moved the surface by more than the "
                f"{policy.noise_floor:g} noise floor (median travel {evidence.travel_median:.3f}, "
                f"worst {evidence.travel_worst:.3f}); policy requires at least "
                f"{policy.min_moving_steps}",
            )
        )
    return evidence, tuple(failures)


# -- gate 3: was the occupied surface actually seen? ------------------------


def _span_cell(points: Sequence[Point], *, per_frame: int, policy: ReadinessPolicy) -> float:
    """The coverage cell for this object, and never finer than its sampling.

    The object's bounding diagonal divided by
    :attr:`ReadinessPolicy.coverage_cells_per_span`, floored at
    :attr:`ReadinessPolicy.coverage_spacing_multiple` times the spacing a frame
    of ``per_frame`` points has over that diagonal. On any real capture the
    first term wins - twenty thousand points across a span makes a spacing far
    below a thirty-second of it - and the floor is what stops a deliberately
    thin cloud from failing a coverage gate it was never dense enough to be
    asked about.

    Zero when there is nothing placed, which
    :func:`scanner3d.coverage.coverage_map` refuses as an unusable cell size -
    the honest answer for a cloud with no extent, rather than a grid invented
    for it.
    """
    if not points:
        return 0.0
    low = tuple(min(point[axis] for point in points) for axis in range(3))
    high = tuple(max(point[axis] for point in points) for axis in range(3))
    span = math.dist(low, high)
    spacing = span / math.sqrt(per_frame) if per_frame > 0 else 0.0
    return max(span / policy.coverage_cells_per_span, policy.coverage_spacing_multiple * spacing)


def coverage_evidence(
    clouds: Sequence[Sequence[Point]],
    poses: Sequence[Rigid],
    *,
    policy: ReadinessPolicy = DEFAULT_POLICY,
) -> tuple[CoverageEvidence, tuple[ReadinessFailure, ...]]:
    """Occupied-cell coverage, through :func:`scanner3d.coverage.coverage_map`.

    Occupied cells only, which is the whole reason that function is used rather
    than a grid of the bounding box: unseen empty volume is not a surface
    defect and must never manufacture a refusal. What this adds is the policy -
    how many frames must have seen a cell and how far apart they must have been
    looking - and the gate on the resulting fraction.

    A million points from one angle is exactly what this refuses: the point
    count does not enter the decision at all, only how many distinct frames saw
    each occupied cell and the widest angle between their view directions.

    The cell comes from the placed geometry's own bounding diagonal, divided by
    :attr:`ReadinessPolicy.coverage_cells_per_span`. Scale-aware without being
    tied to anything a caller tunes for speed: an object twice the size is
    judged at twice the cell, and lowering a voxel budget to make an export
    finish sooner cannot change whether its coverage is adequate.
    """
    if len(clouds) != len(poses):
        raise ReadinessNotMeasurable(
            f"{len(clouds)} cloud(s) were offered with {len(poses)} pose(s)"
        )
    stride = _stride(len(clouds), policy.coverage_max_frames)
    used_clouds = [_strided(cloud, policy.coverage_points_per_frame) for cloud in clouds[::stride]]
    used_poses = list(poses[::stride])
    points_used = sum(len(cloud) for cloud in used_clouds)
    placed = [
        pose.apply(point)
        for cloud, pose in zip(used_clouds, used_poses, strict=True)
        for point in cloud
    ]
    per_frame = max((len(cloud) for cloud in used_clouds), default=0)
    cell = _span_cell(placed, per_frame=per_frame, policy=policy)
    try:
        report = coverage_map(
            used_clouds,
            used_poses,
            cell=cell,
            min_observations=policy.coverage_min_observations,
            min_spread=policy.coverage_min_spread,
        )
    except CoverageNotAllowed as exc:
        evidence = CoverageEvidence(
            measured=False,
            frames_offered=len(clouds),
            frames_used=len(used_clouds),
            points_used=points_used,
            cell_size=cell,
            total_cells=0,
            supported_cells=0,
            supported_fraction=0.0,
            detail=_bounded(str(exc)),
        )
        return evidence, (_failure(COVERAGE_NOT_MEASURABLE, str(exc)),)
    supported = report.well_covered_cells
    total = report.total_cells
    fraction = _fraction(supported, total)
    detail = (
        f"{supported} of {total} occupied cell(s) of size {cell:.3f} were seen by at least "
        f"{policy.coverage_min_observations} frame(s) spanning at least "
        f"{math.degrees(policy.coverage_min_spread):.1f} degrees of view direction"
    )
    evidence = CoverageEvidence(
        measured=True,
        frames_offered=len(clouds),
        frames_used=len(used_clouds),
        points_used=points_used,
        cell_size=cell,
        total_cells=total,
        supported_cells=supported,
        supported_fraction=fraction,
        detail=_bounded(detail),
    )
    if fraction < policy.coverage_min_supported_fraction:
        return evidence, (
            _failure(
                COVERAGE_UNSUPPORTED,
                f"{detail}; that is {fraction:.3f} of the occupied surface and policy requires "
                f"at least {policy.coverage_min_supported_fraction:.3f}",
            ),
        )
    return evidence, ()


# -- gate 4: does the surface agree with the points it came from? -----------


@dataclass(frozen=True, slots=True)
class FusionEvidence:
    """A bounded, deterministic sample of the frames a volume actually fused.

    This exists because the obvious evidence cloud is the wrong one. An export
    holds two point sets: the DECIMATED, filtered clouds the registration was
    fitted to, and the FULL-RESOLUTION frames the volume integrated. Only the
    second is what the surface came from, and comparing a surface against the
    first answers a question nobody asked - a defect that lives in the frames
    but not in the registration sample passes unseen, and the record's own
    prose about "the points it was fused from" is then simply untrue.

    So this carries the sample and the bound it was taken under together: a
    cloud with no counts beside it cannot say how much of the evidence it
    stands for.
    """

    #: How many full-resolution frames the volume was offered.
    frames_offered: int
    #: How many of them this sample was drawn from, by uniform stride.
    frames_sampled: int
    #: How many candidate pixels were examined across those frames.
    samples_examined: int
    #: The placed points, in whatever frame the poses that placed them were in.
    points: tuple[Point, ...]


def fusion_evidence_sample(
    raw_frames: Sequence[bytes],
    placements: Sequence[Rigid],
    *,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    policy: ReadinessPolicy = DEFAULT_POLICY,
) -> FusionEvidence:
    """Deproject a bounded sample of the frames a fusion consumed, and place it.

    The arithmetic is deliberately the fusion's own, not a second version of
    it. ``intrinsics`` and ``depth_scale`` are the export's exact values, run
    through the same :mod:`scanner3d.depth_metric` gates the production
    deprojection uses; a zero sample is skipped as "no depth returned", which
    is also how a support mask reaches this function - a masked sample is a
    zero in the frame that was handed over, so a proved support surface is
    already absent here and this module never has to know that it was masked;
    and ``min_depth``/``max_depth`` are the volume's own INCLUSIVE accepted
    interval, so a sample the fusion refused to integrate is not held against
    the surface it never contributed to.

    ``placements`` are the poses ALREADY composed with whatever presentation
    rotation the surface will be compared in - the caller passes
    ``upright_pose(pose)`` because that is the frame the mesh is written in.
    Composed by the caller rather than here so this module never grows a second
    opinion about which way up a reconstruction is.

    NOTHING IS RE-FUSED. This projects pixels to points; it builds no field,
    allocates no voxel and touches no volume.

    Bounds, exactly. At most :attr:`ReadinessPolicy.fusion_sample_frames`
    frames are opened, chosen by uniform stride over the sequence so the
    angular spread is preserved; at most
    :attr:`ReadinessPolicy.fusion_pixels_per_frame` pixels of each are
    examined, by uniform stride over the raster; so the work is at most the
    product of those two, the returned cloud is at most that many points, and
    one frame's bytes are the largest thing held at a time. Nothing is retained
    past the call.
    """
    if len(raw_frames) != len(placements):
        raise ReadinessNotMeasurable(
            f"{len(raw_frames)} raw frame(s) were offered with {len(placements)} placement(s); "
            "the fusion evidence cannot be sampled for frames that do not have poses"
        )
    fx, fy, cx, cy = ensure_intrinsics_usable(intrinsics)
    scale = ensure_scale_usable(depth_scale)
    frame_stride = _stride(len(raw_frames), policy.fusion_sample_frames)
    chosen = list(range(0, len(raw_frames), frame_stride))[: policy.fusion_sample_frames]
    pixel_stride = _stride(PIXELS_PER_FRAME, policy.fusion_pixels_per_frame)
    points: list[Point] = []
    examined = 0
    for index in chosen:
        raw = raw_frames[index]
        ensure_frame_length(raw)
        pose = placements[index]
        for offset in range(0, PIXELS_PER_FRAME, pixel_stride):
            examined += 1
            code: int = struct.unpack_from("<H", raw, offset * 2)[0]
            if code == NO_RETURN_SAMPLE:
                continue
            distance = code * scale
            if not math.isfinite(distance) or distance < min_depth or distance > max_depth:
                continue
            row, column = divmod(offset, FRAME_WIDTH)
            x = (column - cx) * distance / fx
            y = (row - cy) * distance / fy
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            points.append(pose.apply((x, y, distance)))
    return FusionEvidence(
        frames_offered=len(raw_frames),
        frames_sampled=len(chosen),
        samples_examined=examined,
        points=tuple(points),
    )


def _cell_key(point: Point, cell: float) -> tuple[int, int, int]:
    return (
        math.floor(point[0] / cell),
        math.floor(point[1] / cell),
        math.floor(point[2] / cell),
    )


#: Every neighbour of a cell, itself included. A target within one cell size of
#: a query is guaranteed to be in one of these, which is what makes the
#: directed test below EXACT inside its tolerance rather than approximate.
_NEIGHBOURHOOD: Final[tuple[tuple[int, int, int], ...]] = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)


def _directed_support(
    queries: Sequence[Point], targets: Sequence[Point], *, tolerance: float
) -> tuple[int, tuple[float, ...]]:
    """How many queries have a target within ``tolerance``, and how far each is.

    "Within" is INCLUSIVE - ``distance <= tolerance`` - which is the boundary
    :func:`scanner3d.quality.cloud_agreement` already publishes for exactly
    this measurement. The two had disagreed on the single exact case: a query
    and a target separated by precisely the tolerance counted as one there and
    as none here, so the same pair of clouds got two answers depending on which
    of the repository's instruments asked. There is now one answer.

    A voxel hash at cell size ``tolerance``, searched over the 27-cell
    neighbourhood. That combination is not an approximation, and it stays exact
    at the inclusive boundary: a target at a Euclidean distance of at most
    ``tolerance`` differs from the query by at most ``tolerance`` on every
    axis, so ``floor((q + d) / tolerance) - floor(q / tolerance)`` is in
    ``{-1, 0, 1}`` for every arrangement of the two relative to the cell walls -
    including the one where the target lands exactly on a wall, which is where
    an exclusive test and an inclusive one differ. The count returned is
    therefore the same count an all-pairs search would return.

    Why not :func:`scanner3d.quality.cloud_agreement`, which already measures
    exactly this in both directions: its search is all-pairs and its work ceiling
    is :data:`scanner3d.quality.MAX_NEAREST_WORK`. Meeting that ceiling on a
    fused surface means thinning both clouds until their own spacing is far
    coarser than the truncation, and a tolerance widened to cover that spacing
    stops being able to see the displacement this gate exists to catch. The
    hash keeps full resolution at the truncation scale for linear work.
    :mod:`tests.test_reconstruction_readiness` asserts the two agree, on
    fixtures small enough for the all-pairs search to be affordable, so this is
    a cheaper route to the repository's own measurement and not a second,
    weaker one.

    Distances are exact below ``tolerance`` and clamped AT it above: the search
    is not entitled to claim a distance it did not look far enough to see. A
    recorded distance equal to ``tolerance`` therefore means "this far or
    further", and says nothing about whether the query was counted - which is
    why the count is carried by its own flag rather than re-derived from the
    clamped distance.

    Work bound. Both clouds are voxel-downsampled at a cell no larger than
    ``tolerance`` before this is called, so an index cell of side ``tolerance``
    holds at most ``ceil(tolerance / sample cell) ** 3`` points - 27 at the
    coarsest sampling this module uses - and the whole pass is at most
    ``27 * 27`` distance computations per query. Nothing is retained past the
    call.
    """
    index: dict[tuple[int, int, int], list[Point]] = {}
    for target in targets:
        index.setdefault(_cell_key(target, tolerance), []).append(target)
    within = 0
    distances: list[float] = []
    for query in queries:
        base = _cell_key(query, tolerance)
        best = tolerance
        found = False
        for offset in _NEIGHBOURHOOD:
            bucket = index.get((base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]))
            if bucket is None:
                continue
            for target in bucket:
                distance = math.dist(query, target)
                if distance <= tolerance:
                    found = True
                if distance < best:
                    best = distance
        if found:
            within += 1
        distances.append(best)
    return within, tuple(distances)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, the convention :mod:`scanner3d.quality` uses."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[rank]


def _agreement_samples(
    surface_points: Sequence[Point],
    evidence_points: Sequence[Point],
    *,
    base: float,
    policy: ReadinessPolicy,
) -> tuple[tuple[Point, ...], tuple[Point, ...], float]:
    """Both clouds at one comparison cell, and the cell that was used.

    :func:`scanner3d.registration.voxel_downsample` does the thinning: one
    centroid per occupied cell, emitted in first-seen order, so the same input
    always produces the same output and the same decision.

    The cell starts at :attr:`ReadinessPolicy.surface_cell_fraction` of the
    base tolerance and is coarsened - by doubling, up to
    :attr:`ReadinessPolicy.surface_max_coarsening` times - only while the
    EVIDENCE cloud is above the ceiling. That restriction is the whole point of
    this function. The cell sets the tolerance the decision uses, and a cell
    driven by whichever cloud is bigger would let a lumpy surface inflate its
    own point count, inflate the cell, inflate the tolerance, and pass because
    it was lumpy. The registered points are evidence a bad surface cannot
    manufacture, so they are what the cell may respond to.

    In practice it rarely responds at all: the base tolerance comes from the
    fusion truncation, which comes from the placed extent and the voxel budget,
    so a bigger object gets a bigger cell and the thinned counts stay roughly
    where they are whatever the object's size.
    """
    from scanner3d.registration import voxel_downsample

    cell = max(base * policy.surface_cell_fraction, MIN_AGREEMENT_CELL)
    evidence = voxel_downsample(evidence_points, cell)
    for _ in range(policy.surface_max_coarsening):
        if len(evidence) <= policy.surface_sample_points:
            break
        cell *= 2.0
        evidence = voxel_downsample(evidence_points, cell)
    return voxel_downsample(surface_points, cell), evidence, cell


def _unmeasured(
    *,
    voxel_size: float,
    truncation: float,
    tolerance: float,
    cell: float,
    surface_points: int,
    evidence_points: int,
    source: FusionEvidence,
    detail: str,
) -> SurfaceAgreement:
    """The shape of "this could not be measured", with the reason in it."""
    return SurfaceAgreement(
        measured=False,
        voxel_size=voxel_size,
        truncation=truncation,
        tolerance=tolerance,
        sample_cell=cell,
        surface_points=surface_points,
        evidence_points=evidence_points,
        source_frames_offered=source.frames_offered,
        source_frames_sampled=source.frames_sampled,
        source_samples_examined=source.samples_examined,
        source_points=len(source.points),
        surface_supported_fraction=0.0,
        evidence_explained_fraction=0.0,
        surface_distance_median=0.0,
        surface_distance_p95=0.0,
        evidence_distance_median=0.0,
        evidence_distance_p95=0.0,
        distance_clamp=tolerance,
        detail=_bounded(detail),
    )


def surface_evidence(
    surface_points: Sequence[Point],
    evidence: FusionEvidence,
    *,
    voxel_size: float,
    truncation: float,
    policy: ReadinessPolicy = DEFAULT_POLICY,
) -> tuple[SurfaceAgreement, tuple[ReadinessFailure, ...]]:
    """Symmetric agreement between an extracted surface and its own evidence.

    ``evidence`` is a :class:`FusionEvidence` and not a bare cloud on purpose.
    The one thing this gate can get wrong without anybody noticing is being
    handed the WRONG points - the decimated registration sample rather than the
    full-resolution frames the volume integrated - and a type that carries the
    frames it was drawn from is a type a caller cannot satisfy with the other
    cloud by accident.

    Both directions, always, and both gated - see :class:`SurfaceAgreement` for
    why either alone is not enough. The tolerance comes from the volume's own
    declared truncation and the calibrated noise floor, so a finer or coarser
    fusion is judged on its own terms rather than against a number somebody
    typed in world coordinates.

    This is INTERNAL CONSISTENCY and is labelled as such everywhere it appears.
    It cannot tell a correct surface from a correct-looking one at the wrong
    scale, because both clouds live in the same coordinate system and a
    uniformly scaled world is perfectly consistent in it.

    An open surface passes. Nothing here asks for a closed mesh, counts
    boundary edges, or treats a real boundary as a defect; the questions are
    only whether the surface sits on the evidence and whether it abandons it -
    and the evidence a real one-sided scan is compared against is the one-sided
    set of points its own frames produced, not an imagined whole object.
    """
    base = policy.surface_tolerance(truncation=truncation)
    surface_sample, evidence_sample, cell = _agreement_samples(
        surface_points, evidence.points, base=base, policy=policy
    )
    # The tolerance the DECISION uses, widened by exactly as much as the
    # downsampling above could have moved the two clouds apart. A kept point
    # sits within half a cell diagonal of whatever it stood in for, on each
    # side, so two surfaces that coincide exactly can still measure a full cell
    # diagonal apart. Deriving that rather than assuming the sampling is free
    # is what keeps a perfect surface from failing its own gate.
    tolerance = max(base, cell * math.sqrt(3.0))
    if not surface_sample or not evidence_sample:
        detail = (
            f"{len(surface_sample)} surface point(s) and {len(evidence_sample)} evidence "
            "point(s): there is nothing to compare"
        )
        agreement = _unmeasured(
            voxel_size=voxel_size,
            truncation=truncation,
            tolerance=tolerance,
            cell=cell,
            surface_points=len(surface_sample),
            evidence_points=len(evidence_sample),
            source=evidence,
            detail=detail,
        )
        return agreement, (_failure(SURFACE_NOT_MEASURABLE, detail),)

    # Only the QUERY side of each direction is thinned to the work ceiling.
    # Striding a target would leave gaps in the very thing the query is being
    # measured against, so a cloud lying exactly on it would measure a random
    # half-spacing away; striding a query only asks the same question of fewer
    # points. The two directions therefore share one index-scale sampling and
    # differ only in how many of each cloud were asked.
    surface_queries = _strided(surface_sample, policy.surface_sample_points)
    evidence_queries = _strided(evidence_sample, policy.surface_sample_points)
    supported, surface_distances = _directed_support(
        surface_queries, evidence_sample, tolerance=tolerance
    )
    explained, evidence_distances = _directed_support(
        evidence_queries, surface_sample, tolerance=tolerance
    )
    supported_fraction = _fraction(supported, len(surface_queries))
    explained_fraction = _fraction(explained, len(evidence_queries))
    agreement = SurfaceAgreement(
        measured=True,
        voxel_size=voxel_size,
        truncation=truncation,
        tolerance=tolerance,
        sample_cell=cell,
        surface_points=len(surface_queries),
        evidence_points=len(evidence_queries),
        source_frames_offered=evidence.frames_offered,
        source_frames_sampled=evidence.frames_sampled,
        source_samples_examined=evidence.samples_examined,
        source_points=len(evidence.points),
        surface_supported_fraction=supported_fraction,
        evidence_explained_fraction=explained_fraction,
        surface_distance_median=_median(surface_distances),
        surface_distance_p95=_percentile(surface_distances, 0.95),
        evidence_distance_median=_median(evidence_distances),
        evidence_distance_p95=_percentile(evidence_distances, 0.95),
        distance_clamp=tolerance,
        detail=_bounded(
            f"tolerance {tolerance:.4f} = max({policy.surface_tolerance_truncations:g} x "
            f"truncation {truncation:.4f}, {policy.surface_tolerance_noise_multiple:g} x noise "
            f"floor {policy.noise_floor:g}, one {cell:.4f} sampling cell diagonal). Evidence: "
            f"{len(evidence.points)} point(s) deprojected from {evidence.frames_sampled} of "
            f"{evidence.frames_offered} FULL-RESOLUTION fused frame(s), "
            f"{evidence.samples_examined} candidate sample(s) examined. INTERNAL "
            "CONSISTENCY ONLY: both clouds are in the reconstruction's own coordinate system, "
            "so this establishes no dimensional accuracy and sees no uniform scale error"
        ),
    )
    failures: list[ReadinessFailure] = []
    if supported_fraction < policy.surface_min_supported_fraction:
        failures.append(
            _failure(
                SURFACE_UNSUPPORTED,
                f"{supported_fraction:.3f} of the fused surface lies within {tolerance:.4f} of a "
                f"registered point (median {agreement.surface_distance_median:.4f}, p95 "
                f"{agreement.surface_distance_p95:.4f}); policy requires at least "
                f"{policy.surface_min_supported_fraction:.3f}",
            )
        )
    if explained_fraction < policy.surface_min_evidence_explained:
        failures.append(
            _failure(
                SURFACE_ABANDONS_EVIDENCE,
                f"{explained_fraction:.3f} of the registered points lie within {tolerance:.4f} "
                f"of the fused surface (median {agreement.evidence_distance_median:.4f}, p95 "
                f"{agreement.evidence_distance_p95:.4f}); policy requires at least "
                f"{policy.surface_min_evidence_explained:.3f}",
            )
        )
    return agreement, tuple(failures)


# -- the whole pre-fusion answer, in one call -------------------------------


def evaluate_readiness(
    frames_dir: Path,
    clouds: Sequence[Sequence[Point]],
    poses: Sequence[Rigid],
    *,
    policy: ReadinessPolicy = DEFAULT_POLICY,
) -> ReconstructionReadiness:
    """Every gate that can be answered before the volume exists.

    All of them are evaluated, always: a capture that failed the tracking gate
    is still measured for pose diversity and coverage, so the record a refusal
    leaves behind names everything that was wrong rather than the first thing
    that was.

    ``frames_dir`` is REQUIRED and is not optional for a reason. Omitting the
    capture was the bypass this module closes: an entry point that starts from
    a folder of frames must consult that folder's own tracking authority, and
    an API that let a caller skip it by passing nothing would be the same
    silent failure with a different shape.
    """
    tracking, tracking_failures = tracking_evidence(frames_dir)
    poses_evidence, pose_failures = pose_evidence(clouds, poses, policy=policy)
    coverage, coverage_failures = coverage_evidence(clouds, poses, policy=policy)
    return ReconstructionReadiness(
        policy=policy,
        tracking=tracking,
        poses=poses_evidence,
        coverage=coverage,
        surface=None,
        failures=(*tracking_failures, *pose_failures, *coverage_failures),
    )
