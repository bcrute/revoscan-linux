"""Replay one real capture through raw, weighted-raw and temporally weighted fusion, read-only.

Deliberately NOT a ``test_`` module: pytest does not collect it. It is the
harness a handoff runs when the physical numbers are wanted, in the role
:mod:`tests.surface_oracle_replay` plays for the raw-points-versus-surface
comparison and :mod:`tests.readiness_replay` plays for the readiness gates. It
reuses that first module's read-only machinery directly -
:func:`tests.surface_oracle_replay.fingerprint`,
:func:`tests.surface_oracle_replay.verify_unchanged`,
:func:`tests.surface_oracle_replay.bind_identity`,
:func:`tests.surface_oracle_replay.return_coverage` and its component-proxy
constants - rather than copying it, so the physical replays share one proof of
read-only-ness rather than several that could drift apart.

Run it with::

    .venv/bin/python -m tests.temporal_weighted_replay \\
        --capture /home/ben/scans/scan/20260827-152842 --limit 12 --backend numpy \\
        > /tmp/temporal-weighted-upper-bound.json

``--capture`` is REQUIRED and has no default. This replay measures exactly the
capture it is pointed at, and a default would let a run publish numbers about a
capture nobody asked about. The report is written to **stdout only**; this
module opens no path for writing at all, so the caller decides where the
artifact lands and nothing is left behind when it does not redirect.

STRICTLY READ-ONLY, and it proves it the way its sibling does: size,
modification time and SHA-256 of every file in the capture are recorded before
and after the replay, and the report carries both plus whether they agree. It
calls ``Path.is_dir``, ``Path.rglob``, ``Path.glob``, ``Path.stat`` and
``Path.read_bytes`` on the capture and nothing else. It never opens a path for
writing, creates no directory and no temporary file, exports nothing, opens no
device node, makes no network call and touches no scanner control.

Three arms, one field, one set of bytes
---------------------------------------
The question is narrow: on ONE capture whose raw baseline is already recorded,
what does the accepted temporal and weighted path do when pose quality is set
to its most permissive value under one fixed nearest-eight source rule? Three
arms answer it, and every one of them receives the same original raw bytes, the
same registration poses, the same intrinsics, the same depth scale, the same
:class:`scanner3d.tsdf.TsdfConfig` and the same backend. This is not a search
over other legal eight-frame source subsets and therefore is not a global
upper bound on the method or capture:

* **raw** - :meth:`scanner3d.tsdf.TsdfVolume.integrate` over the raw frames.
  The baseline, and the same call a production export makes;
* **weighted_raw** - :meth:`scanner3d.tsdf.TsdfVolume.integrate_weighted` over
  :func:`scanner3d.tsdf_input.weighted_raw_frame` of those same bytes. The
  IDENTITY CONTROL: every measured sample weighs
  :data:`scanner3d.tsdf_input.MEASURED_WEIGHT` and every no-return weighs
  nothing, so this field must be byte-for-byte the raw field. It is established
  BEFORE the completion arm is looked at, because a weighted seam that moved
  the field by itself would make every later difference uninterpretable, and
  this replay stops there rather than attributing that difference to temporal
  evidence;
* **candidate** - :func:`scanner3d.depth_temporal.complete_temporally` over
  each whole 640x400 raw frame, adapted with
  :func:`scanner3d.tsdf_input.weighted_temporal_frame` and fused the same way.

The source set is fixed, and it was fixed first
-----------------------------------------------
:func:`scanner3d.depth_temporal.complete_temporally` accepts at most
:data:`scanner3d.depth_temporal.MAX_SUPPORT_FRAMES` source frames. So each
target uses exactly the :data:`SOURCE_FRAMES` other frames nearest to it in
capture-index distance, ties broken by the lower index, offered in ascending
index order - :func:`nearest_sources`, which is a pure function of the target
index and the frame count and cannot see an outcome. The rule was read off that
ceiling and written down before any temporal result existed, and nothing here
narrows, widens, reorders or re-picks it afterwards. No accepted sample, no
frame and no result is post-selected either: all of the selected frames are
completed, all of them are fused, and a refusal is reported rather than skipped.

The one deliberate epistemic limit
-----------------------------------
:func:`scanner3d.pipeline.reconstruct_sweep` publishes accepted poses and no
calibrated scalar that could honestly populate
:attr:`scanner3d.depth_temporal.PoseRecord.quality`. This replay therefore runs
exactly one arm, named :data:`UPPER_BOUND_ARM`, at
:data:`POSE_QUALITY` = 1.0. Temporal confidence is monotone in pose quality, so
that arm is a **pose-quality upper bound under the fixed nearest-eight source
rule and otherwise fixed inputs** and nothing else. It does not measure the
global most the method could admit because other legal source subsets are not
tested. It is not a calibrated physical confidence, it does not authorize
product routing, and no pose-quality mapping is invented or tuned here. Every
artifact this module writes says so in
``temporal.pose_quality_calibrated``, in the arm's own name and in
:data:`LIMITATIONS`.

What this replay may conclude, and what it may not
---------------------------------------------------
The decision is mechanical, declared in :data:`DECISION_RULE` before the run and
applied in that order:

* zero accepted temporal samples, or a field and surface the candidate left
  unchanged, is a **negative same-input result** - not a failure of the run;
* worse observable support or lump proxies **rejects** the candidate;
* anything else is an **uncalibrated pose-quality-upper-bound candidate under
  the fixed source rule** and no more.

No outcome claims dimensional accuracy, edge preservation, calibrated
roughness, pose accuracy or material physical improvement: this capture has no
independent reference and none of those can be measured without one. The
surface agreement published here is INTERNAL CONSISTENCY against the capture's
own measured returns - measurement support, which is a different question from
accuracy and is kept separate from it everywhere in the report. And no outcome
routes the candidate anywhere: no export, GUI, preview, tracker, capture path
or readiness policy reads this module.

One frame, from the bounds to the fusion
-----------------------------------------
Every coordinate here is in the MODEL frame, exactly as
:mod:`tests.surface_oracle_replay` documents and for the same reason; that
module's :data:`~tests.surface_oracle_replay.MODEL_FRAME` note is imported and
republished rather than restated. The presentation half turn
:data:`scanner3d.reconstruction_export.UPRIGHT_FROM_OPTICAL` is applied to
NOTHING, on any arm.

A refusal is a result
---------------------
If the capture cannot be read, cannot be registered, fails a readiness gate,
fails the identity control, or is refused by a temporal work ceiling, this
replay STOPS THERE and reports the refusal. It does not fuse anyway, does not
offer fewer source frames to get under a ceiling, and does not complete a
smaller patch to make a number appear.

Exit codes
----------
* **0** - the replay ran to completion, or stopped at one of its own declared
  boundaries, and the capture was proved unchanged.
* **1** - the replay failed part-way with an unexpected error. The capture was
  still fingerprinted afterwards and the read-only result is still reported.
* **2** - the capture could not be read or bound at all; nothing was measured.
* **3** - the before/after fingerprints disagree. This replay changed a capture
  it promised only to read, which dominates every other code above.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scanner3d import depth_temporal, tsdf_input
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_completion import DepthPatch
from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.preparation import RECONSTRUCT_POINTS, prepare_frame
from scanner3d.reconstruction_export import (
    TSDF_MAX_DEPTH,
    TSDF_MAX_WEIGHT,
    TSDF_MIN_DEPTH,
    TSDF_MIN_WEIGHT,
    TSDF_WEIGHT_POLICY,
    choose_tsdf_resolution,
    model_points,
)
from scanner3d.reconstruction_readiness import (
    DEFAULT_POLICY,
    FusionEvidence,
    coverage_evidence,
    fusion_evidence_sample,
    pose_evidence,
    surface_evidence,
)
from scanner3d.simulate import FACTORY_INTRINSICS
from scanner3d.tsdf import TsdfConfig, TsdfVolume
from scanner3d.tsdf_surface import NoSurfaceFound, extract_surface
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH
from tests.surface_oracle import component_sizes
from tests.surface_oracle_replay import (
    ISOLATED_MAX_POINTS,
    LINKAGE_RADIUS_VOXELS,
    MODEL_FRAME,
    ReplayRefused,
    bind_identity,
    fingerprint,
    return_coverage,
    verify_unchanged,
)

#: The registration backend, and the one the accepted task-037 baseline used on
#: this capture. Named so the trajectory this replay measures is the trajectory
#: that baseline recorded rather than a second one fitted differently.
REGISTRATION_BACKEND: Final = "auto"

#: The compute backend every FIELD in this replay is built and extracted on.
#: One value for all three arms, because a difference between two arms that ran
#: on two backends is a difference between the backends.
DEFAULT_BACKEND: Final = "numpy"

#: How many name-sorted frames the replay takes when the caller does not say.
#: Twelve: the exact prefix of `20260827-152842` that task 037 established a
#: valid raw baseline on, so the raw arm here is comparable with that record.
DEFAULT_LIMIT: Final = 12

#: The desktop Reconstruct action's voxel budget, ``128 ** 3``. Written as the
#: literal rather than imported from :data:`scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET`
#: because importing the GUI would pull a Qt dependency into a harness that has
#: no window, exactly as :mod:`tests.build_surface_oracle_fixtures` writes it;
#: the name it comes from is stated here and republished in every report.
VOXEL_BUDGET: Final = 128 * 128 * 128
VOXEL_BUDGET_SOURCE: Final = "scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET, 128 ** 3"

#: The most source frames one temporal completion may rest on, imported rather
#: than restated: the source rule below is derived FROM this ceiling and would
#: be a different rule if the ceiling moved.
SOURCE_FRAMES: Final = depth_temporal.MAX_SUPPORT_FRAMES

#: The shipped temporal policy, unchanged. Nothing here tunes a threshold.
TEMPORAL_POLICY: Final = depth_temporal.DEFAULT_POLICY

#: The only temporal backend that exists. Named explicitly rather than left to
#: ``"auto"`` so the report cannot imply a completion ran on the field backend.
TEMPORAL_BACKEND: Final = depth_temporal.PYTHON_BACKEND

#: The exact source rule, published in every report beside the sets it produced.
SOURCE_RULE: Final = (
    "each target uses exactly the eight other frames nearest to it in capture-index distance, "
    "ties broken by the lower index, offered in ascending index order. Eight is "
    "scanner3d.depth_temporal.MAX_SUPPORT_FRAMES, the API's own ceiling on how many distinct "
    "other frames one reconstruction may rest on; the rule was read off that ceiling and fixed "
    "before any temporal outcome existed. It is a pure function of the target index and the "
    "frame count and cannot see a result, and it is neither narrowed, widened nor reordered "
    "afterwards"
)

#: What the tracker's own word for these poses is. ``reconstruct_sweep``
#: publishes a pose or it raises; a pose it returned is one it placed.
POSE_STATUS: Final = "reconstruct_sweep placed this frame"

#: The pose quality the one declared arm runs at. See :data:`UPPER_BOUND_ARM`.
POSE_QUALITY: Final = 1.0

#: The single arm this replay runs, named in the report so no reader can take
#: its confidences for calibrated ones.
UPPER_BOUND_ARM: Final = "uncalibrated_pose_quality_upper_bound"

UPPER_BOUND_MEANING: Final = (
    "scanner3d.pipeline.reconstruct_sweep publishes accepted poses and no calibrated scalar "
    f"that could honestly populate scanner3d.depth_temporal.PoseRecord.quality, so this arm "
    f"runs at quality {POSE_QUALITY} and is labelled for exactly that reason. Temporal "
    "confidence is monotone in pose quality, so this is a pose-quality upper bound under the "
    "fixed nearest-eight source rule and otherwise fixed frames, poses, policy and fusion "
    "inputs. It is not a global upper bound on the method or capture because other legal "
    "eight-frame source subsets are not tested. It is not a calibrated physical confidence: "
    "it cannot authorize product routing, and no pose-quality mapping is invented, fitted or "
    "tuned anywhere in this module"
)

#: Every arm this replay runs, in the order it runs them.
ARM_RAW: Final = "raw"
ARM_WEIGHTED_RAW: Final = "weighted_raw"
ARM_CANDIDATE: Final = "candidate"

#: Where the replay stopped, when it stopped.
STOPPED_FRAME_COUNT: Final = "frame_count"
STOPPED_REGISTRATION: Final = "registration"
STOPPED_READINESS: Final = "readiness"
STOPPED_IDENTITY_CONTROL: Final = "identity_control"
STOPPED_TEMPORAL: Final = "temporal"

#: The mechanical outcomes, and the order they are tested in.
OUTCOME_COMPARATOR_INVALID: Final = "comparator_invalid"
OUTCOME_NEGATIVE: Final = "negative_same_input"
OUTCOME_NOT_COMPARABLE: Final = "not_comparable"
OUTCOME_REJECTED: Final = "rejected_worse_observable_proxies"
OUTCOME_CANDIDATE: Final = "uncalibrated_upper_bound_candidate"

DECISION_RULE: Final[tuple[str, ...]] = (
    f"{OUTCOME_COMPARATOR_INVALID}: the weighted-raw identity control did not reproduce the raw "
    "field. The weighted seam moved the field by itself, so no later difference is attributable "
    "to temporal evidence and the candidate arm is not run at all.",
    f"{OUTCOME_NEGATIVE}: no temporal sample was accepted anywhere, or the candidate field and "
    "its surface are identical to the raw ones. Same input, same answer - a negative result "
    "about this capture and not a failure of the run.",
    f"{OUTCOME_NOT_COMPARABLE}: one arm produced a surface or a measured agreement the other "
    "did not, so the observable proxies have nothing to compare. No verdict is invented.",
    f"{OUTCOME_REJECTED}: an observable support or lump proxy got worse - a lower supported or "
    "explained fraction, or more components or a larger isolated-point share. Worse observable "
    "geometry rejects the candidate whatever field evidence it added.",
    f"{OUTCOME_CANDIDATE}: the candidate changed the field and no observable proxy got worse. "
    "This is only an UNCALIBRATED POSE-QUALITY UPPER BOUND under the fixed nearest-eight source "
    "rule and otherwise fixed inputs. It is not proof that extracted surface geometry was "
    "added, and no accuracy, improvement or route into the product follows from it.",
)

#: The observable proxies the decision compares, which section of an arm's
#: surface record each is read out of, and which direction is WORSE. Declared
#: here, before the run, and neither extended nor dropped after an outcome.
PROXIES: Final[tuple[tuple[str, str, str], ...]] = (
    ("surface_supported_fraction", "measured_support", "lower"),
    ("evidence_explained_fraction", "measured_support", "lower"),
    ("component_count", "components", "higher"),
    ("isolated_fraction", "components", "higher"),
)

#: Every claim this replay is structurally incapable of making, published as
#: machine-readable falses beside the prose so a reader never has to trust the
#: prose alone.
CLAIMS: Final[dict[str, bool]] = {
    "dimensional_accuracy_claimed": False,
    "edge_preservation_claimed": False,
    "calibrated_roughness_claimed": False,
    "pose_accuracy_claimed": False,
    "physical_improvement_claimed": False,
    "pose_quality_calibrated": False,
    "routed_into_product": False,
}

LIMITATIONS: Final[tuple[str, ...]] = (
    "read-only. No file in the capture was written, renamed, deleted or created, and the "
    "before/after fingerprints in this report are the evidence for that. This module opens no "
    "path for writing at all: the report goes to stdout and the caller redirects it.",
    "there is no truth model for a physical capture. No coverage of a truth surface, surface "
    "accuracy, dimension error, edge coverage, thin-feature survival or calibrated roughness is "
    "measured or claimed anywhere in this report.",
    "the surface agreement here is INTERNAL CONSISTENCY against this capture's own measured "
    "returns, and it is measurement SUPPORT rather than accuracy. Both clouds live in the same "
    "coordinate system, so a uniformly wrong scale is perfectly consistent in it.",
    "no number here has a physical unit. Every length is in the capture's own depth units, "
    "whose physical meaning this repository has not established.",
    "the poses the readiness gates judge, the poses every arm is fused at, and the poses the "
    "temporal completion carries evidence through were all fitted by THIS run. A pose the "
    "registration got wrong and was confident about is measured here as ordinary motion, and no "
    "pose accuracy is measured or claimed.",
    f"the temporal arm runs at pose quality {POSE_QUALITY} and is named "
    f"{UPPER_BOUND_ARM!r} for that reason. {UPPER_BOUND_MEANING}",
    "every coordinate here is in the model frame the poses were fitted in. The presentation "
    "half turn scanner3d.reconstruction_export.UPRIGHT_FROM_OPTICAL is applied to nothing on "
    "any arm, so a number here is not directly comparable with an exported PLY's coordinates.",
    "the depth window every arm projects through is the appliance's own TSDF_MIN_DEPTH and "
    "TSDF_MAX_DEPTH, imported rather than restated. A return outside that band is not measured "
    "here, exactly as it would not be exported.",
    "component counts depend on the linkage radius, which is stated in voxels rather than "
    "chosen to produce a number, and is imported from tests.surface_oracle_replay so the two "
    "physical replays cannot drift apart.",
    "nothing here is routed into the product. No export, live preview, GUI, capture, tracker, "
    "readiness policy, temporal-completion default or weighted-input default reads this module, "
    "and no threshold in the repository is changed by anything it prints.",
)


class ComparatorInvalid(ScannerError):
    """The identity control did not reproduce the raw field, so nothing is comparable."""


# --- the fixed source rule ------------------------------------------------------------------


def nearest_sources(target: int, frames: int, *, limit: int = SOURCE_FRAMES) -> tuple[int, ...]:
    """The ``limit`` other frames nearest ``target`` in index, ascending. See :data:`SOURCE_RULE`.

    A pure function of two integers. It cannot see a raw frame, a pose, a
    confidence or an accepted sample, which is what makes the declared rule
    checkable rather than merely asserted: there is nothing in scope here for
    an outcome to influence.
    """
    if frames < 2:
        raise ReplayRefused(f"{frames} frame(s) offer no other frame to any target")
    if not 0 <= target < frames:
        raise ReplayRefused(f"frame {target} is not one of the {frames} frames offered")
    if limit < 1:
        raise ReplayRefused(f"a source set of {limit} frame(s) is not a source set")
    available = frames - 1
    if available < limit:
        raise ReplayRefused(
            f"{frames} frame(s) offer only {available} other frame(s) to target {target}; "
            f"the fixed source rule requires exactly {limit}"
        )
    others = [index for index in range(frames) if index != target]
    ranked = sorted(others, key=lambda index: (abs(index - target), index))
    return tuple(sorted(ranked[:limit]))


# --- observable proxies ---------------------------------------------------------------------


def components(points: Sequence[Point], radius: float) -> dict[str, object]:
    """Connected-component and isolated-lump proxies, on the shared linkage rule."""
    sizes = component_sizes(points, radius)
    total = sum(sizes)
    isolated = [size for size in sizes if size <= ISOLATED_MAX_POINTS]
    return {
        "linkage_radius": radius,
        "component_count": len(sizes),
        "largest_component_points": sizes[0] if sizes else 0,
        "largest_component_fraction": (sizes[0] / total) if total else 0.0,
        "isolated_components": len(isolated),
        "isolated_points": sum(isolated),
        "isolated_fraction": (sum(isolated) / total) if total else 0.0,
    }


def field_delta(base: TsdfVolume, other: TsdfVolume) -> dict[str, object]:
    """Exactly how two fields of the same shape differ, voxel by voxel.

    ``identical`` is decided on the buffers themselves rather than inferred
    from the counts below, because that is the claim the identity control has
    to make: not "close", but the same bytes.
    """
    base_values, base_weights = base.values, base.weights
    other_values, other_weights = other.values, other.weights
    if len(base_values) != len(other_values) or len(base_weights) != len(other_weights):
        raise ComparatorInvalid(
            f"a {len(base_values)}-voxel field cannot be compared with a {len(other_values)}-"
            "voxel one; the two arms were not given the same field definition"
        )
    same_values = bytes(base_values) == bytes(other_values)
    same_weights = bytes(base_weights) == bytes(other_weights)
    record: dict[str, object] = {
        "voxels": len(base_values),
        "values_identical": same_values,
        "weights_identical": same_weights,
        "identical": same_values and same_weights,
    }
    if same_values and same_weights:
        # Nothing moved, so every count below is zero by construction and the
        # two-million-voxel walk that would prove it again is not spent.
        record.update(
            {
                "value_changed_voxels": 0,
                "weight_changed_voxels": 0,
                "observed_gained": 0,
                "observed_lost": 0,
                "max_abs_value_delta": 0.0,
                "max_abs_weight_delta": 0.0,
                "total_weight_delta": 0.0,
            }
        )
        return record

    value_changed = 0
    weight_changed = 0
    gained = 0
    lost = 0
    max_value = 0.0
    max_weight = 0.0
    for index in range(len(base_values)):
        before_weight = base_weights[index]
        after_weight = other_weights[index]
        if before_weight != after_weight:
            weight_changed += 1
            max_weight = max(max_weight, abs(after_weight - before_weight))
        if before_weight <= 0.0 < after_weight:
            gained += 1
        elif after_weight <= 0.0 < before_weight:
            lost += 1
        before_value = base_values[index]
        after_value = other_values[index]
        if before_value != after_value:
            value_changed += 1
            max_value = max(max_value, abs(after_value - before_value))
    record.update(
        {
            "value_changed_voxels": value_changed,
            "weight_changed_voxels": weight_changed,
            "observed_gained": gained,
            "observed_lost": lost,
            "max_abs_value_delta": max_value,
            "max_abs_weight_delta": max_weight,
            "total_weight_delta": math.fsum(
                other_weights[index] - base_weights[index] for index in range(len(base_weights))
            ),
        }
    )
    return record


@dataclass(frozen=True, slots=True)
class ArmSurface:
    """One arm's extracted surface: the published record, and the counts, typed.

    The counts are carried beside the record rather than read back out of it,
    so a delta between two arms is arithmetic on integers this module holds and
    not on values recovered from a document it also writes.
    """

    record: dict[str, object]
    vertices: tuple[Point, ...]
    faces: int
    extracted: bool

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)


def _surface_record(
    volume: TsdfVolume,
    *,
    backend: str,
    voxel_size: float,
    truncation: float,
    evidence: FusionEvidence,
) -> ArmSurface:
    """One arm's extracted surface, its lump proxies and its measured-support agreement."""
    record: dict[str, object] = {}
    try:
        surface = extract_surface(volume, min_weight=TSDF_MIN_WEIGHT, backend=backend)
    except (NoSurfaceFound, ScannerError) as error:
        record["extracted"] = False
        record["extraction_refused"] = str(error)
        record["vertices"] = 0
        record["faces"] = 0
        record["components"] = components((), voxel_size * LINKAGE_RADIUS_VOXELS)
        record["measured_support"] = {
            "measured": False,
            "why": "no surface was extracted, so there is nothing to compare with the evidence",
        }
        return ArmSurface(record=record, vertices=(), faces=0, extracted=False)
    vertices = tuple(surface.mesh.vertices)
    agreement, failures = surface_evidence(
        vertices, evidence, voxel_size=voxel_size, truncation=truncation
    )
    record["extracted"] = True
    record["vertices"] = len(vertices)
    record["faces"] = surface.mesh.face_count
    record["extraction"] = surface.evidence.as_dict()
    record["components"] = components(vertices, voxel_size * LINKAGE_RADIUS_VOXELS)
    record["measured_support"] = {
        **agreement.as_dict(),
        "failures": [failure.as_dict() for failure in failures],
        "is": (
            "internal consistency against this capture's own MEASURED returns, which is "
            "measurement support and not accuracy. The evidence cloud is deprojected from the "
            "raw bytes alone and is identical for every arm, so a reconstructed sample can "
            "never explain itself here"
        ),
    }
    return ArmSurface(
        record=record,
        vertices=vertices,
        faces=surface.mesh.face_count,
        extracted=True,
    )


# --- the temporal arm -----------------------------------------------------------------------


def _confidence_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"samples": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "samples": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": math.fsum(values) / len(values),
    }


def _support_summary(result: depth_temporal.TemporalPatch) -> dict[str, object]:
    """How many other frames, and which, actually carried each accepted sample."""
    counts: list[int] = []
    named: set[int] = set()
    for index in result.temporally_reconstructed_indices():
        support = result.support_at(index)
        counts.append(len(support))
        named.update(frame for frame, _ in support)
    return {
        "support_samples": result.work.support_samples,
        "per_sample_minimum": min(counts) if counts else None,
        "per_sample_maximum": max(counts) if counts else None,
        "per_sample_mean": (math.fsum(counts) / len(counts)) if counts else None,
        "distinct_source_frames_used": sorted(named),
    }


def _viewpoint_summary(
    target: depth_temporal.TemporalFrame, sources: Sequence[depth_temporal.TemporalFrame]
) -> list[dict[str, object]]:
    """Each offered source's separation from the target camera: baseline and degrees.

    Measured with the module's own :func:`~scanner3d.depth_temporal.temporal_transform`
    and :func:`~scanner3d.depth_temporal.viewpoint_separation` rather than a
    second version of the same arithmetic, so what is published is what the
    policy's viewpoint gate actually sees.
    """
    identity = Rigid.identity()
    summary: list[dict[str, object]] = []
    for source in sources:
        into_target = depth_temporal.temporal_transform(target.pose, source.pose)
        baseline, degrees = depth_temporal.viewpoint_separation(identity, into_target)
        summary.append({"source": source.index, "baseline": baseline, "degrees": degrees})
    return summary


def _temporal_frames(
    raws: Sequence[bytes], poses: Sequence[Rigid]
) -> tuple[tuple[depth_temporal.TemporalFrame, ...], depth_temporal.CameraModel]:
    """The whole-frame patches, poses and camera the completion is given.

    ``DepthPatch.from_frame`` over the WHOLE 640x400 frame at origin ``(0, 0)``:
    no window is cut, so no sample is selected out of a frame before the module
    that judges it has seen it. The camera's four numbers come from the factory
    record through :func:`scanner3d.depth_metric.ensure_intrinsics_usable` - the
    same gate the production deprojection passes them through - so the fusion
    and the completion cannot be looking at two different cameras.
    """
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    camera = depth_temporal.CameraModel(fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=DEPTH_SCALE_METRES)
    frames = tuple(
        depth_temporal.TemporalFrame(
            index=index,
            patch=DepthPatch.from_frame(raw, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT),
            u0=0,
            v0=0,
            pose=depth_temporal.PoseRecord(
                transform=pose, accepted=True, status=POSE_STATUS, quality=POSE_QUALITY
            ),
        )
        for index, (raw, pose) in enumerate(zip(raws, poses, strict=True))
    )
    return frames, camera


@dataclass(frozen=True, slots=True)
class TemporalArm:
    """Every selected frame's completion, its adapted weighted frame, and its record.

    All three sequences are the same length and in the same order as the frames
    the replay selected, so there is no place for a frame or a result to be
    dropped between the completion and the fusion.
    """

    records: tuple[dict[str, object], ...]
    results: tuple[depth_temporal.TemporalPatch, ...]
    weighted: tuple[tsdf_input.WeightedDepthFrame, ...]

    @property
    def accepted(self) -> int:
        return sum(result.work.accepted for result in self.results)


def complete_every_frame(
    raws: Sequence[bytes], poses: Sequence[Rigid], names: Sequence[str]
) -> TemporalArm:
    """Complete EVERY selected frame under the fixed source rule, and adapt each result.

    Every frame, in order, with no skipping and no post-selection: a frame the
    completion refuses raises out of here so the caller can report the refusal
    as the result, rather than being dropped so the remaining ones can be fused.
    """
    frames, camera = _temporal_frames(raws, poses)
    records: list[dict[str, object]] = []
    results: list[depth_temporal.TemporalPatch] = []
    weighted: list[tsdf_input.WeightedDepthFrame] = []
    for target in frames:
        chosen = nearest_sources(target.index, len(frames))
        sources = [frames[index] for index in chosen]
        eligible = sum(len(source.measured_indices()) for source in sources)
        holes = len(target.patch.hole_indices())
        started = time.perf_counter()
        result = depth_temporal.complete_temporally(
            target,
            sources,
            camera=camera,
            policy=TEMPORAL_POLICY,
            backend=TEMPORAL_BACKEND,
        )
        elapsed = time.perf_counter() - started
        adapted = tsdf_input.weighted_temporal_frame(raws[target.index], result, origin=(0, 0))
        confidences = [confidence for _, _, confidence in depth_temporal.temporal_samples(result)]
        records.append(
            {
                "index": target.index,
                "file": names[target.index],
                "sources": list(chosen),
                "source_rule": SOURCE_RULE,
                "eligible_samples": eligible,
                "holes": holes,
                "declared_work_budget": TEMPORAL_POLICY.worst_case_work(eligible, holes),
                "work_ceiling": depth_temporal.MAX_WORK_UNITS,
                "result": result.as_dict(),
                "confidence": _confidence_summary(confidences),
                "support": _support_summary(result),
                "viewpoints": _viewpoint_summary(target, sources),
                "weighted_input": adapted.summary.as_dict(),
                "weighted_adapter": adapted.adapter,
                "weighted_owned_bytes": adapted.owned_bytes,
                "elapsed_s": elapsed,
            }
        )
        results.append(result)
        weighted.append(adapted)
    return TemporalArm(records=tuple(records), results=tuple(results), weighted=tuple(weighted))


def _temporal_totals(arm: TemporalArm) -> dict[str, object]:
    """The whole arm in one place: accepted, rejected, work and every refusal reason.

    Summed from the accepted result objects themselves rather than read back out
    of the records this module also writes, so the totals and the per-frame
    numbers cannot disagree about what happened.
    """
    rejections = dict.fromkeys(depth_temporal.REJECTION_REASONS, 0)
    observation_rejections = dict.fromkeys(depth_temporal.OBSERVATION_REJECTIONS, 0)
    accepted = 0
    rejected = 0
    holes = 0
    observations = 0
    candidates = 0
    support_samples = 0
    work_units = 0
    work_budget = 0
    for result in arm.results:
        work = result.work
        accepted += work.accepted
        rejected += work.rejected
        holes += work.holes
        observations += work.observations
        candidates += work.candidates
        support_samples += work.support_samples
        work_units += work.work_units
        work_budget += work.work_budget
        for reason in depth_temporal.REJECTION_REASONS:
            rejections[reason] += work.rejections[reason]
        for reason in depth_temporal.OBSERVATION_REJECTIONS:
            observation_rejections[reason] += work.observation_rejections[reason]
    return {
        "frames_completed": len(arm.results),
        "holes_examined": holes,
        "accepted": accepted,
        "rejected": rejected,
        "observations": observations,
        "candidates": candidates,
        "support_samples": support_samples,
        "rejections": rejections,
        "observation_rejections": observation_rejections,
        "work_units": work_units,
        "work_budget": work_budget,
        "work_ceiling": depth_temporal.MAX_WORK_UNITS,
        "temporally_reconstructed_samples": sum(
            frame.summary.temporally_reconstructed for frame in arm.weighted
        ),
        "measured_samples": sum(frame.summary.measured for frame in arm.weighted),
        "spatially_excluded_samples": sum(
            frame.summary.spatially_excluded for frame in arm.weighted
        ),
        "missing_samples": sum(frame.summary.missing for frame in arm.weighted),
        "offered_weight": math.fsum(frame.summary.total_weight for frame in arm.weighted),
    }


# --- the decision ---------------------------------------------------------------------------


def _proxy_value(record: Mapping[str, object], key: str, section: str) -> float | None:
    """One declared proxy off one arm's surface record, or ``None`` when it has none."""
    block = record.get(section)
    if not isinstance(block, dict):
        return None
    if section == "measured_support" and not block.get("measured"):
        return None
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def decide(
    *,
    identical_control: bool,
    accepted_samples: int,
    field_identical: bool,
    surface_identical: bool,
    raw_surface: Mapping[str, object],
    candidate_surface: Mapping[str, object],
) -> dict[str, object]:
    """The declared rule in :data:`DECISION_RULE`, applied in that order and nowhere else."""
    verdict: dict[str, object] = {
        "arm": UPPER_BOUND_ARM,
        "rule": list(DECISION_RULE),
        "accepted_temporal_samples": accepted_samples,
        **CLAIMS,
    }
    if not identical_control:
        verdict["outcome"] = OUTCOME_COMPARATOR_INVALID
        verdict["because"] = (
            "the weighted-raw identity control did not reproduce the raw field byte for byte, "
            "so a difference in the candidate arm cannot be attributed to temporal evidence"
        )
        return verdict

    if accepted_samples == 0 or (field_identical and surface_identical):
        verdict["outcome"] = OUTCOME_NEGATIVE
        verdict["because"] = (
            "no temporal sample was accepted anywhere"
            if accepted_samples == 0
            else "the candidate field and its surface are identical to the raw ones"
        ) + (
            ". Same input, same answer: this is a negative same-input result about this capture "
            "and not a failure of the run"
        )
        return verdict

    observed: dict[str, object] = {}
    worse: list[str] = []
    comparable = True
    for key, section, direction in PROXIES:
        before = _proxy_value(raw_surface, key, section)
        after = _proxy_value(candidate_surface, key, section)
        observed[key] = {"raw": before, "candidate": after, "worse_when": direction}
        if before is None or after is None:
            comparable = False
            continue
        if (after < before) if direction == "lower" else (after > before):
            worse.append(key)
    verdict["proxies"] = observed
    if not comparable:
        verdict["outcome"] = OUTCOME_NOT_COMPARABLE
        verdict["because"] = (
            "one arm produced a surface or a measured agreement the other did not, so the "
            "observable proxies have nothing to compare and no verdict is invented"
        )
        return verdict

    verdict["worse_proxies"] = worse
    if worse:
        verdict["outcome"] = OUTCOME_REJECTED
        verdict["because"] = (
            f"{', '.join(worse)} got worse under the candidate. Worse observable geometry "
            "rejects the candidate whatever field evidence it added"
        )
        return verdict
    verdict["outcome"] = OUTCOME_CANDIDATE
    verdict["because"] = (
        "the candidate changed the field and no observable support or lump proxy got worse. "
        f"{UPPER_BOUND_MEANING}"
    )
    return verdict


# --- the replay -----------------------------------------------------------------------------


def replay(
    capture: Path,
    *,
    limit: int | None,
    backend: str,
    voxel_budget: int = VOXEL_BUDGET,
) -> dict[str, object]:
    """Everything this capture can support saying, and where it stopped if it stopped.

    ``voxel_budget`` defaults to the desktop Reconstruct action's own
    :data:`VOXEL_BUDGET` and exists as an argument so a test can fuse a bounded
    field in seconds rather than minutes. The chooser refuses any value above
    :data:`scanner3d.tsdf.MAX_VOXELS`, so this argument can narrow the field and
    can never widen it.
    """
    identity, paths = bind_identity(capture, limit=limit)
    report: dict[str, object] = {
        "kind": "scanner3d.temporal-weighted-upper-bound-physical-replay",
        "schema_version": 1,
        "identity": identity,
        "backend": backend,
        "registration_backend": REGISTRATION_BACKEND,
        "temporal_backend": TEMPORAL_BACKEND,
        "readiness_policy": DEFAULT_POLICY.as_dict(),
        "temporal_policy": TEMPORAL_POLICY.as_dict(),
        "source_rule": SOURCE_RULE,
        "source_frame_ceiling": SOURCE_FRAMES,
        "voxel_budget": voxel_budget,
        "voxel_budget_source": VOXEL_BUDGET_SOURCE,
        "arms": [ARM_RAW, ARM_WEIGHTED_RAW, ARM_CANDIDATE],
        "upper_bound_arm": UPPER_BOUND_ARM,
        "upper_bound_meaning": UPPER_BOUND_MEANING,
        "decision_rule": list(DECISION_RULE),
        "frame": MODEL_FRAME,
        "limitations": list(LIMITATIONS),
        **CLAIMS,
    }
    minimum_frames = SOURCE_FRAMES + 1
    if len(paths) < minimum_frames:
        report["stopped_at"] = STOPPED_FRAME_COUNT
        report["stopped_because"] = (
            f"{len(paths)} usable frames; the fixed source rule needs at least {minimum_frames} "
            f"so every target receives exactly {SOURCE_FRAMES} other frames"
        )
        return report

    raws = [path.read_bytes() for path in paths]
    report["returns"] = {
        "per_frame": [
            {"file": path.name, **return_coverage(raw)}
            for path, raw in zip(paths, raws, strict=True)
        ]
    }

    started = time.perf_counter()
    clouds = [prepare_frame(raw, target_points=RECONSTRUCT_POINTS).points for raw in raws]
    prepared_s = time.perf_counter() - started

    from scanner3d.pipeline import reconstruct_sweep

    started = time.perf_counter()
    try:
        result = reconstruct_sweep(clouds, backend=REGISTRATION_BACKEND)
    except ScannerError as error:
        report["stopped_at"] = STOPPED_REGISTRATION
        report["stopped_because"] = str(error)
        report["timing"] = {"prepare_s": prepared_s}
        return report
    registered_s = time.perf_counter() - started

    poses, pose_failures = pose_evidence(clouds, result.poses)
    coverage, coverage_failures = coverage_evidence(clouds, result.poses)
    reasons = [failure.reason for failure in (*pose_failures, *coverage_failures)]
    report["registration"] = {
        "frames": result.frames,
        "revisits_found": result.revisits_found,
        "optimised": result.optimised,
    }
    report["poses"] = {
        "evidence": poses.as_dict(),
        "failures": [failure.as_dict() for failure in pose_failures],
    }
    report["coverage"] = {
        "evidence": coverage.as_dict(),
        "failures": [failure.as_dict() for failure in coverage_failures],
    }
    report["readiness_reasons"] = reasons
    report["ready_before_fusion"] = not reasons
    if reasons:
        # The refusal IS the result, exactly as tests.surface_oracle_replay
        # treats it: the boundary has already said a reconstruction of this
        # capture would not be trustworthy, and an arm that reached past it
        # would be answering a question the boundary refused.
        report["stopped_at"] = STOPPED_READINESS
        report["stopped_because"] = (
            "the shipped readiness policy refuses this capture, so no field was built and no "
            "arm was run. The refusal is the finding"
        )
        report["timing"] = {"prepare_s": prepared_s, "register_s": registered_s}
        return report

    # ONE field definition, derived once, shared by all three arms. These are
    # the poses the registration fitted, in the model frame; ``model_points``
    # places the clouds by them to measure the extent and the SAME list is what
    # every integrate and integrate_weighted call below is given. That is the
    # contract those methods document - each inverts a pose to carry a voxel
    # back into its camera - and it is what production export_reconstruction
    # does: measure with ``model_points``, integrate the raw poses, turn the
    # extracted MESH upright afterwards. A field bounded here and fused at
    # presentation-rotated poses is not a scan drawn the wrong way up; it is
    # every voxel projected half a turn away from the raw evidence.
    fusion_poses = list(result.poses)
    placed = list(model_points(clouds, fusion_poses))
    resolution = choose_tsdf_resolution(placed, voxel_budget=voxel_budget)
    report["resolution"] = resolution.as_dict()
    report["raw_points"] = {
        "point_count": len(placed),
        "components": components(placed, resolution.voxel_size * LINKAGE_RADIUS_VOXELS),
        "note": (
            "the measured dots, placed by the poses this run fitted and clipped to nothing. No "
            "truth-referenced number is published for them because there is no truth"
        ),
    }

    config = TsdfConfig.of(
        bounds=resolution.bounds,
        voxel_size=resolution.voxel_size,
        truncation=resolution.truncation,
        min_depth=TSDF_MIN_DEPTH,
        max_depth=TSDF_MAX_DEPTH,
        weight_policy=TSDF_WEIGHT_POLICY,
        max_weight=TSDF_MAX_WEIGHT,
        max_voxels=voxel_budget,
    )
    report["field"] = {
        "config": config.as_dict(),
        "shared_by": [ARM_RAW, ARM_WEIGHTED_RAW, ARM_CANDIDATE],
        "extraction_min_weight": TSDF_MIN_WEIGHT,
        "from": (
            "scanner3d.reconstruction_export's own bounds, truncation, depth window, weight "
            "policy, weight cap and extraction minimum, imported rather than restated, at the "
            f"desktop Reconstruct voxel budget ({VOXEL_BUDGET_SOURCE})"
        ),
    }

    # The evidence every arm's surface is compared against: a bounded
    # deterministic sample of the RAW frames, deprojected under the same
    # intrinsics, scale and accepted depth window the fusion used, and placed by
    # the same model-frame poses. Raw bytes only - a reconstructed sample never
    # enters this cloud, so it can never explain itself.
    evidence = fusion_evidence_sample(
        raws,
        fusion_poses,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        min_depth=TSDF_MIN_DEPTH,
        max_depth=TSDF_MAX_DEPTH,
    )
    report["measured_evidence"] = {
        "frames_offered": evidence.frames_offered,
        "frames_sampled": evidence.frames_sampled,
        "samples_examined": evidence.samples_examined,
        "points": len(evidence.points),
        "is": (
            "deprojected from the raw bytes alone, inside the fusion's own inclusive depth "
            "window, and shared unchanged by every arm"
        ),
    }

    started = time.perf_counter()
    raw_volume = TsdfVolume(config, backend=backend)
    raw_volume.integrate(
        raws, fusion_poses, intrinsics=FACTORY_INTRINSICS, depth_scale=DEPTH_SCALE_METRES
    )
    raw_fused_s = time.perf_counter() - started
    raw_surface = _surface_record(
        raw_volume,
        backend=backend,
        voxel_size=resolution.voxel_size,
        truncation=resolution.truncation,
        evidence=evidence,
    )
    report["raw_arm"] = {
        "arm": ARM_RAW,
        "integration": raw_volume.evidence(elapsed_seconds=raw_fused_s).as_dict(),
        "surface": raw_surface.record,
    }

    # The identity control, and it is established BEFORE the completion arm is
    # run at all. weighted_raw_frame weighs every measured sample exactly one
    # and every no-return exactly nothing, so this field must be the raw field's
    # own bytes; if it is not, the weighted seam moved the field by itself and
    # no later difference is attributable to temporal evidence.
    started = time.perf_counter()
    control_volume = TsdfVolume(config, backend=backend)
    control_volume.integrate_weighted(
        [tsdf_input.weighted_raw_frame(raw) for raw in raws],
        fusion_poses,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    control_fused_s = time.perf_counter() - started
    control_surface = _surface_record(
        control_volume,
        backend=backend,
        voxel_size=resolution.voxel_size,
        truncation=resolution.truncation,
        evidence=evidence,
    )
    control_delta = field_delta(raw_volume, control_volume)
    same_control_surface = (
        control_surface.vertices == raw_surface.vertices
        and control_surface.faces == raw_surface.faces
        and control_surface.extracted == raw_surface.extracted
    )
    identical_control = bool(control_delta["identical"]) and same_control_surface
    report["weighted_raw_arm"] = {
        "arm": ARM_WEIGHTED_RAW,
        "integration": control_volume.evidence(elapsed_seconds=control_fused_s).as_dict(),
        "weighted_integration": control_volume.weighted_evidence().as_dict(),
        "surface": control_surface.record,
        "field_delta_against_raw": control_delta,
        "surface_identical_to_raw": same_control_surface,
        "reproduces_raw": identical_control,
        "is": (
            "the identity control. Every measured sample weighs MEASURED_WEIGHT and every "
            "no-return weighs nothing, so this field is the raw field or the weighted seam "
            "itself created a difference"
        ),
    }
    if not identical_control:
        report["stopped_at"] = STOPPED_IDENTITY_CONTROL
        report["stopped_because"] = (
            "the weighted-raw identity control did not reproduce the raw field and surface, so "
            "the comparator is invalid and the candidate arm was not run"
        )
        report["decision"] = decide(
            identical_control=False,
            accepted_samples=0,
            field_identical=bool(control_delta["identical"]),
            surface_identical=same_control_surface,
            raw_surface=raw_surface.record,
            candidate_surface=control_surface.record,
        )
        report["timing"] = {
            "prepare_s": prepared_s,
            "register_s": registered_s,
            "raw_fuse_s": raw_fused_s,
            "control_fuse_s": control_fused_s,
        }
        return report

    started = time.perf_counter()
    try:
        arm = complete_every_frame(raws, fusion_poses, [path.name for path in paths])
    except depth_temporal.TemporalNotAllowed as error:
        # A ceiling refusal is a result. Nothing here retries with fewer source
        # frames or a smaller patch to make the number appear.
        report["stopped_at"] = STOPPED_TEMPORAL
        report["stopped_because"] = f"{type(error).__name__}: {error}"
        report["timing"] = {
            "prepare_s": prepared_s,
            "register_s": registered_s,
            "raw_fuse_s": raw_fused_s,
            "control_fuse_s": control_fused_s,
            "temporal_s": time.perf_counter() - started,
        }
        return report
    temporal_s = time.perf_counter() - started
    totals = _temporal_totals(arm)
    report["temporal"] = {
        "arm": UPPER_BOUND_ARM,
        "meaning": UPPER_BOUND_MEANING,
        "pose_quality": POSE_QUALITY,
        "pose_status": POSE_STATUS,
        "policy": TEMPORAL_POLICY.as_dict(),
        "backend": TEMPORAL_BACKEND,
        "source_rule": SOURCE_RULE,
        "totals": totals,
        "per_frame": list(arm.records),
        **CLAIMS,
    }

    started = time.perf_counter()
    candidate_volume = TsdfVolume(config, backend=backend)
    candidate_volume.integrate_weighted(
        list(arm.weighted),
        fusion_poses,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    candidate_fused_s = time.perf_counter() - started
    candidate_surface = _surface_record(
        candidate_volume,
        backend=backend,
        voxel_size=resolution.voxel_size,
        truncation=resolution.truncation,
        evidence=evidence,
    )
    candidate_delta = field_delta(raw_volume, candidate_volume)
    same_candidate_surface = (
        candidate_surface.vertices == raw_surface.vertices
        and candidate_surface.faces == raw_surface.faces
        and candidate_surface.extracted == raw_surface.extracted
    )
    report["candidate_arm"] = {
        "arm": ARM_CANDIDATE,
        "integration": candidate_volume.evidence(elapsed_seconds=candidate_fused_s).as_dict(),
        "weighted_integration": candidate_volume.weighted_evidence().as_dict(),
        "surface": candidate_surface.record,
        "field_delta_against_raw": candidate_delta,
        "surface_identical_to_raw": same_candidate_surface,
        "surface_delta_against_raw": {
            "vertices": candidate_surface.vertex_count - raw_surface.vertex_count,
            "faces": candidate_surface.faces - raw_surface.faces,
        },
        "attributable_to": (
            "accepted temporal samples carried from the same other RAW views of this capture, "
            "and nothing else: the arm was given the same bytes, poses, intrinsics, scale, "
            "field and backend as the raw arm, and the identity control above proves the "
            "weighted seam contributes no difference of its own"
        ),
    }
    report["decision"] = decide(
        identical_control=True,
        accepted_samples=arm.accepted,
        field_identical=bool(candidate_delta["identical"]),
        surface_identical=same_candidate_surface,
        raw_surface=raw_surface.record,
        candidate_surface=candidate_surface.record,
    )
    report["timing"] = {
        "prepare_s": prepared_s,
        "register_s": registered_s,
        "raw_fuse_s": raw_fused_s,
        "control_fuse_s": control_fused_s,
        "temporal_s": temporal_s,
        "candidate_fuse_s": candidate_fused_s,
    }
    return report


def _failed_report(*, stopped_at: str, error: BaseException) -> dict[str, object]:
    """The document a run that ended in an error still publishes.

    The read-only proof is attached to a report, so a run that failed part-way
    needs a report to attach it to.
    """
    return {
        "kind": "scanner3d.temporal-weighted-upper-bound-physical-replay",
        "schema_version": 1,
        "source_rule": SOURCE_RULE,
        "upper_bound_arm": UPPER_BOUND_ARM,
        "upper_bound_meaning": UPPER_BOUND_MEANING,
        "decision_rule": list(DECISION_RULE),
        "limitations": list(LIMITATIONS),
        "stopped_at": stopped_at,
        "stopped_because": f"{type(error).__name__}: {error}",
        **CLAIMS,
    }


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.temporal_weighted_replay",
        description=(
            "Read-only matched raw versus temporally weighted replay of one capture. Writes "
            "its report to stdout only; redirect it where you want it kept."
        ),
    )
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="the capture to replay. Required: this module has no default capture",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)

    before = fingerprint(args.capture) if args.capture.is_dir() else {}
    # From here on there is a baseline, so EVERY exit path takes the after
    # fingerprint, compares it and reports it. A run that blew up half-way is
    # exactly the run whose read-only evidence a reader most needs.
    exit_code = 0
    try:
        report = replay(args.capture, limit=args.limit, backend=args.backend)
    except ReplayRefused as error:
        print(f"the replay could not be run: {error}", file=sys.stderr)
        report = _failed_report(stopped_at="refused", error=error)
        exit_code = 2
    except ScannerError as error:
        print(f"the replay failed: {type(error).__name__}: {error}", file=sys.stderr)
        report = _failed_report(stopped_at="error", error=error)
        exit_code = 1
    except Exception as error:
        print(f"the replay failed: {type(error).__name__}: {error}", file=sys.stderr)
        report = _failed_report(stopped_at="unexpected_error", error=error)
        exit_code = 1

    after = fingerprint(args.capture) if args.capture.is_dir() else {}
    read_only = verify_unchanged(before, after)
    report["read_only"] = read_only
    try:
        # Strict: sorted, indented, and NOT-A-NUMBER free. A report that cannot
        # be serialised under those rules is a failure of this module rather
        # than something to publish with infinities in it, and the read-only
        # proof is still delivered, because that is the part a reader most
        # needs from a run that went wrong.
        text = json.dumps(report, indent=2, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        print(f"the report could not be serialised: {error}", file=sys.stderr)
        print(
            json.dumps(
                {
                    **_failed_report(stopped_at="unpublishable_report", error=error),
                    "read_only": read_only,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 3 if not read_only["unchanged"] else 1
    print(text)
    if not read_only["unchanged"]:
        return 3
    return exit_code


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
