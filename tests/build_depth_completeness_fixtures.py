"""Regenerate the deterministic depth-completeness fixture corpus.

The corpus itself - ``tests/fixtures/depth_completeness/cases/`` - is the
evidence, and :mod:`tests.depth_completeness_harness` reads it off disk and
never calls anything here. This module exists so a reviewer can see exactly
which scene, which dropout masks and which reasoning produced the committed
numbers.

It is deliberately NOT a test and is not collected by pytest: it writes files.
Run it explicitly, from the repository root::

    .venv/bin/python -m tests.build_depth_completeness_fixtures

and then read the diff. A run that changes a fixture is changing evidence.

What a case file contains, and what it does not
-----------------------------------------------
A completeness case is a family of 512000-byte raw depth frames. Written out as
JSON that is a quarter of a million integers per frame, which nobody reviews.
So each case commits:

* the exact scene - shapes, intrinsics, depth scale, window, and each frame's
  pose as an axis, an angle and a centre of rotation;
* the exact dropout masks, by shape and by named seed;
* the SHA-256 and returned-sample count of every truth frame and every masked
  input frame those produce.

The harness re-renders from the scene and refuses the case if the frames do not
hash to what the fixture says. A change to the renderer, to a mask rule or to
the scene therefore shows up as a refusal rather than as a quietly different
benchmark, and the frames stay immutable evidence without being unreadable
committed data.

Where the limits come from
--------------------------
Never from what a candidate happened to score. Every limit is one of three
things, and each gate's ``context`` says which:

1. **An acceptance target**, stated by this benchmark for the completion tasks
   that follow: recover :data:`ACCEPTANCE_FRACTION` of the surface the scene
   really has.
2. **A non-inferiority bar**: the value the unchanged baseline itself reaches,
   plus a stated margin in raw codes. These are the gates that say a completion
   may not make a plane rougher, an object smaller or a surface further from
   truth than not completing at all did.
3. **An absolute tolerance in raw codes**, for error against held-out truth and
   for the geometry that is exact by construction - a depth step is where the
   truth says it is, a slot is open, and empty space stays empty.

Everything here is pure CPU arithmetic on committed constants - no scanner, no
capture, no network, no clock, no module random source - so two runs of this
module produce byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import scanner3d.quality as quality
import tests.depth_completeness as contract
import tests.depth_completeness_harness as harness
from scanner3d.simulate import render as simulate_render

#: Where the corpus lives, relative to this file.
CORPUS = Path(__file__).parent / "fixtures" / "depth_completeness" / "cases"

FIXTURE_SCHEMA_VERSION = harness.FIXTURE_SCHEMA_VERSION

#: The synthetic camera. A centred principal point and round focal lengths,
#: declared synthetic: this is NOT the unit's factory record, and no number
#: here describes the attached scanner.
INTRINSICS: Final[dict[str, Any]] = {
    "fx": 400.0,
    "fy": 400.0,
    "cx": 320.0,
    "cy": 200.0,
    "width": 640,
    "height": 400,
}

#: A power of two, so ``code x scale`` is exact in binary floating point and a
#: round trip through the uint16 frame loses nothing beyond the quantisation
#: this benchmark measures on purpose.
DEPTH_SCALE: Final = 0.0078125

#: One raw code, in depth units. Every absolute tolerance below is a whole
#: number of these.
CODE: Final = DEPTH_SCALE

#: The acceptance target the completion tasks must reach: this fraction of the
#: surface the scene really has must end up measured or supported-reconstructed.
#: Not reached by the baseline, and not reached by any control here honestly.
ACCEPTANCE_FRACTION: Final = 0.95

#: Non-inferiority margins, in raw codes, added to the baseline's own value.
ROUGHNESS_MARGIN_CODES: Final = 4
DIMENSION_MARGIN_CODES: Final = 8
LANDMARK_MARGIN_CODES: Final = 8
SURFACE_MARGIN_CODES: Final = 8

#: Absolute tolerances on error against held-out truth, in raw codes.
ERROR_P95_CODES: Final = 4
ERROR_WORST_CODES: Final = 16

#: Non-inferiority margins for the tracking replay: how much worse than the
#: baseline's own placement a completion may leave the real tracker's poses.
#:
#: A sixteenth of the 4-degree rotation between adjacent frames in the only
#: case that replays tracking. Stated as a fraction of the scene's own motion
#: rather than picked to suit a candidate: a completion that shifts a placement
#: by a sixteenth of the step between two frames has changed where the object
#: is, and one that does not has not.
TRACKING_ROTATION_MARGIN_DEGREES: Final = 4.0 / 16.0
TRACKING_TRANSLATION_MARGIN_CODES: Final = 8

#: How far a completed depth step may sit from where the truth puts it.
STEP_COLUMN_TOLERANCE: Final = 1.0

#: The one place any invented sample is tolerated, and why. A temporal carry
#: through exact poses lands a real surface sample one pixel outside the
#: silhouette when the projection rounds, and the multi-view case would
#: otherwise refuse the only honest temporal control it has. The single-view
#: curved case gates false fill at exactly zero.
MULTIVIEW_FALSE_FILL_FRACTION: Final = 0.10

#: Decimal places every written limit is rounded to.
LIMIT_DECIMALS: Final = 6

#: How far, in raw codes, an independently posed view's measurement of a target
#: sample may sit from that sample's own truth and still be called a view of it.
#: Used only by :func:`_temporal_opportunity`, which states what the SCENE makes
#: available to a temporal method - never what any candidate achieved.
TEMPORAL_VIEW_TOLERANCE_CODES: Final = 8

#: The one gate whose applicability follows from whether a candidate reports a
#: run at all, rather than from the scene's regions. Named once here so the
#: rule that fills it in and the gate that states it cannot drift apart.
WORK_GATE: Final = "completion_work.within_budget"

#: The same, for the pose-compensated stage. Only a candidate that RAN that
#: stage reports what it cost, so every other candidate leaves this gate
#: unevaluated - stated in the fixture rather than discovered by the harness.
TEMPORAL_WORK_GATE: Final = "temporal_work.within_budget"

#: The candidates that run the temporal primitive and are therefore the only
#: ones that can report a temporal run. Named here so the fixture rule and the
#: adapter in tests/depth_completeness.py cannot drift apart.
TEMPORAL_CANDIDATES: Final[tuple[str, ...]] = (
    "temporal_completion",
    "spatial_then_temporal_completion",
)

#: The gates that judge what a confirmation was worth. They exist only where a
#: run was offered an accepted spatial estimate to confirm, which is one
#: candidate: a block of zeros beside a run that was never offered one would
#: read as a stage that checked and found nothing.
CONFIRMATION_GATES: Final[tuple[str, ...]] = (
    "confirmation.confirmed_false",
    "confirmation.confirmed_on_surface",
)
CONFIRMING_CANDIDATE: Final = "spatial_then_temporal_completion"

UNIT = contract.DEPTH_UNIT
FRACTION = quality.FRACTION_UNIT
COUNT = contract.COUNT_UNIT
COLUMNS = contract.COLUMN_UNIT


#: Limits are rounded so a reviewer can read them, and always in the direction
#: that cannot flip the baseline's verdict on its own bar: a "must be at least"
#: limit rounds DOWN and a "must be at most" limit rounds UP. Rounding the
#: other way would fail the baseline against a value it actually reached, which
#: is a fixture bug that looks exactly like a regression.
def _floor(value: float) -> float:
    return math.floor(value * 10**LIMIT_DECIMALS) / 10**LIMIT_DECIMALS


def _ceil(value: float) -> float:
    return math.ceil(value * 10**LIMIT_DECIMALS) / 10**LIMIT_DECIMALS


# --- the scenes ------------------------------------------------------------------------


def _plane(normal: Sequence[float], offset: float) -> dict[str, object]:
    return {"kind": "plane", "normal": [float(value) for value in normal], "offset": offset}


def _box(half_extents: Sequence[float], translation: Sequence[float]) -> dict[str, object]:
    return {
        "kind": "box",
        "half_extents": [float(value) for value in half_extents],
        "translation": [float(value) for value in translation],
    }


def _sphere(centre: Sequence[float], radius: float) -> dict[str, object]:
    return {"kind": "sphere", "centre": [float(value) for value in centre], "radius": radius}


def _still() -> dict[str, object]:
    """The pose of a single-view case: no rotation at all, stated explicitly."""
    return {"axis": [0.0, 1.0, 0.0], "degrees": 0.0, "through": [0.0, 0.0, 100.0]}


def _shifted(dx: float) -> dict[str, object]:
    """A camera displaced laterally, with no rotation at all.

    The pose spec's ``translation`` moves the SCENE, so the frame's own world
    pose - what :mod:`scanner3d.tracking` and this benchmark call a pose - is
    its inverse. A scene shift of ``+dx`` therefore moves the image content
    right by ``dx * fx / depth`` pixels, which is the one displacement a
    reviewer can check on paper, and it is what the motion case needs to state
    an exact source pixel against an exact target pixel.
    """
    return {
        "axis": [0.0, 1.0, 0.0],
        "degrees": 0.0,
        "through": [0.0, 0.0, 100.0],
        "translation": [dx, 0.0, 0.0],
    }


def _turned(degrees: float, through: Sequence[float]) -> dict[str, object]:
    return {
        "axis": [0.0, 1.0, 0.0],
        "degrees": degrees,
        "through": [float(value) for value in through],
    }


def _window(u0: int, v0: int, width: int, height: int) -> dict[str, object]:
    return {
        "u0": u0,
        "v0": v0,
        "width": width,
        "height": height,
        "sample_count": width * height,
    }


def _regions(**overrides: object) -> dict[str, object]:
    regions: dict[str, object] = {
        "plane": False,
        "foreground_below": None,
        "slot_columns": [],
        "surface_tolerance": None,
        "dimensions": False,
        "landmarks": False,
        "fusion": None,
        "weighted_fusion": None,
        "tracking": False,
    }
    regions.update(overrides)
    return regions


# --- materialising a scene, through the harness's own reader ----------------------------


def _materialise(document: Mapping[str, object]) -> contract.Scene:
    """Fill in each frame's digests and counts, then read the case back.

    The frames are rendered through exactly the code the harness renders with,
    and the finished document is then handed to
    :func:`tests.depth_completeness_harness.build_scene`, so a fixture this
    module writes is one the harness has already accepted - digests, counts and
    all.
    """
    return _fill_frame_identities(document)


def _fill_frame_identities(document: Mapping[str, object]) -> contract.Scene:
    """Render once with the checks disabled, and write what came out."""
    scene_spec = document["scene"]
    assert isinstance(scene_spec, dict)
    frames = scene_spec["frames"]
    assert isinstance(frames, list)
    window = harness._window(scene_spec["window"], "generated window")
    intrinsics = harness._intrinsics(scene_spec["intrinsics"], "generated intrinsics")
    shape_specs = scene_spec["shapes"]
    assert isinstance(shape_specs, list)
    shapes = [harness._shape(shape, "generated shape") for shape in shape_specs]
    for index, frame in enumerate(frames):
        assert isinstance(frame, dict)
        pose = harness._render_pose(frame["pose"], "generated pose")
        rendered = simulate_render(
            shapes, intrinsics=intrinsics, depth_scale=DEPTH_SCALE, pose=pose
        )
        truth_codes = contract.codes_of(rendered.raw, window)
        removed: set[int] = set()
        for position, mask in enumerate(frame["masks"]):
            removed |= harness.mask_indices(
                mask,
                window=window,
                truth=truth_codes,
                seed=f"{scene_spec['seed']}:frame{index}:mask{position}",
                what="generated mask",
            )
        input_codes = [0 if sample in removed else code for sample, code in enumerate(truth_codes)]
        frame["truth_sha256"] = hashlib.sha256(
            contract.frame_of_codes(truth_codes, window)
        ).hexdigest()
        frame["input_sha256"] = hashlib.sha256(
            contract.frame_of_codes(input_codes, window)
        ).hexdigest()
        frame["truth_returned"] = sum(1 for code in truth_codes if code != 0)
        frame["input_returned"] = sum(1 for code in input_codes if code != 0)
    return harness.build_scene(document, "generated case")


def _baseline_of(scene: contract.Scene) -> dict[str, Mapping[str, object]]:
    """What the unchanged pipeline measures on this scene. The bar to not fall below."""
    measurements = contract.measure(scene, contract.baseline(scene))
    return {measurement.name: measurement.as_dict() for measurement in measurements}


def _at(record: Mapping[str, object], *path: str) -> float:
    current: object = record
    for step in path:
        assert isinstance(current, Mapping)
        current = current[step]
    assert isinstance(current, int | float)
    return float(current)


# --- the gates -------------------------------------------------------------------------


def _gate(
    name: str,
    measurement: str,
    path: Sequence[str],
    comparison: str,
    limit: float | bool,
    unit: str,
    context: str,
) -> dict[str, object]:
    gate: dict[str, object] = {
        "name": name,
        "measurement": measurement,
        "path": list(path),
        "comparison": comparison,
        "limit": limit,
        "context": context,
    }
    if comparison != "is":
        gate["unit"] = unit
    return gate


def _isolated_dropout(scene: contract.Scene) -> int:
    """Dropped samples whose four neighbours all came back. A scene fact, not a policy.

    The strongest statement this corpus can make about where single-frame
    completion is *owed* an answer: a sample the sensor lost with measurements
    on every side of it, on a scene whose truth surface is locally planar, is a
    hole one frame determines. Derived from the rendered truth and the masks -
    never from what any candidate managed - so the bar cannot drift towards
    whatever an implementation happens to reach.

    It is deliberately not applied to the curved case. There, "surrounded by
    measurements" does not mean the surface between them is locally determined:
    two samples either side of a silhouette's shoulder bracket a surface that
    is turning away from the camera, and a bar demanding they be interpolated
    would be demanding an error.
    """
    total = 0
    for index in range(len(scene.frames)):
        truth = scene.truth_codes(index)
        given = scene.input_codes(index)
        window = scene.window
        for position, code in enumerate(given):
            if code != 0 or truth[position] == 0:
                continue
            row, column = divmod(position, window.width)
            surrounded = True
            for step_column, step_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                near_column, near_row = column + step_column, row + step_row
                if not (0 <= near_column < window.width and 0 <= near_row < window.height):
                    continue
                if given[near_row * window.width + near_column] == 0:
                    surrounded = False
                    break
            total += int(surrounded)
    return total


def _local_recovery_gate(scene: contract.Scene) -> list[dict[str, object]]:
    """The strict recovery bar, where one frame's own evidence settles the answer."""
    owed = _isolated_dropout(scene)
    if owed <= 0:
        return []
    return [
        _gate(
            "coverage.local_recovery",
            "coverage",
            ("reconstructed_on_surface",),
            ">=",
            float(owed),
            COUNT,
            f"a strict gain, stated by this scene and not by any candidate: {owed} sample(s) of "
            "this frame were dropped with a returned measurement on all four sides of them, on "
            "geometry that is locally planar, and a completion must recover at least that many "
            "onto surface that is really there. Beside coverage.measured_fraction, which pins "
            "the measured population, this is a strict supported-coverage gain over the "
            "unchanged baseline - which recovers none of them",
        )
    ]


def _on_silhouette(scene: contract.Scene, index: int, position: int) -> bool:
    """Whether this scene puts a sample on its own declared foreground boundary.

    Read off the rendered truth and the case's own ``foreground_below`` code -
    no completion policy, no tolerance and no threshold borrowed from a product
    module. A sample whose eight truth neighbours include surface on BOTH sides
    of that line is a sample the scene itself says sits on the silhouette.

    It exists because this corpus makes two demands at once. The edge
    measurements forbid putting depth on the wrong side of a step or in the
    empty band across it, and :func:`_temporal_opportunity` would otherwise owe
    recovery of samples that sit exactly there. A bar that requires a method to
    answer where the same corpus requires it not to guess is not a bar, so the
    opportunity is stated over the samples away from the boundary and the
    boundary ones are named separately. A case that declares no foreground
    split has no silhouette and nothing is excluded.
    """
    split = scene.regions.foreground_below
    if split is None:
        return False
    window = scene.window
    truth = scene.truth_codes(index)
    u, v = window.pixel_of(position)
    near = far = False
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            at = window.index_of(u + du, v + dv)
            if at is None or truth[at] == 0:
                continue
            if truth[at] < split:
                near = True
            else:
                far = True
    return near and far


def _temporal_opportunity(scene: contract.Scene) -> int:
    """Target samples another posed view of this scene actually measured.

    A property of the scene, derived forwards from the rendered truth and the
    exact poses and never from what any candidate managed: take each dropped
    sample whose truth is real surface, put it where the truth says it is, look
    at where that point lands in every OTHER frame, and count the frames whose
    own raw input returned a measurement of that same point there. A sample at
    least two other frames measured is one a temporal method is owed; a sample
    no other frame saw is not, and the scene says which is which.

    Deliberately strict about "there": the point has to land on the sample the
    other frame actually measured, at the depth that frame measured it at. A
    neighbouring sample of the same surface is a different point, and counting
    it would make this bar a statement about how smooth the scene is.

    Samples the scene puts on its own declared foreground boundary are left out
    - see :func:`_on_silhouette` for why a corpus that gates guessing at a step
    may not also owe recovery there. They are excluded from the bar, not from
    the scene: a method is free to recover them, and the product one does
    recover most of them.
    """
    window = scene.window
    values = scene.intrinsics.values
    fx, fy, cx, cy = values["fx"], values["fy"], values["cx"], values["cy"]
    scale = scene.depth_scale
    owed = 0
    for index, frame in enumerate(scene.frames):
        truth = scene.truth_codes(index)
        given = scene.input_codes(index)
        for position, code in enumerate(given):
            if code != 0 or truth[position] == 0:
                continue
            if _on_silhouette(scene, index, position):
                continue
            u, v = window.pixel_of(position)
            depth = truth[position] * scale
            world = frame.pose.apply(((u - cx) * depth / fx, (v - cy) * depth / fy, depth))
            seen = 0
            for other in range(len(scene.frames)):
                if other == index:
                    continue
                moved = scene.frames[other].pose.inverse().apply(world)
                if moved[2] <= 0.0:
                    continue
                at = window.index_of(
                    round(cx + moved[0] * fx / moved[2]), round(cy + moved[1] * fy / moved[2])
                )
                if at is None:
                    continue
                measured = scene.input_codes(other)[at]
                if measured == 0:
                    continue
                if abs(measured - moved[2] / scale) > TEMPORAL_VIEW_TOLERANCE_CODES:
                    continue
                # And the correspondence has to be mutual. Rounding is not
                # symmetric: a target sample can land inside another frame's
                # measured sample while that sample's own centre lands in a
                # different target sample, and a bar built on the one-way test
                # would owe a method a sample no observation of that other
                # frame actually points at.
                back_u, back_v = window.pixel_of(at)
                back_depth = measured * scale
                returned = frame.pose.inverse().apply(
                    scene.frames[other].pose.apply(
                        (
                            (back_u - cx) * back_depth / fx,
                            (back_v - cy) * back_depth / fy,
                            back_depth,
                        )
                    )
                )
                if returned[2] <= 0.0:
                    continue
                home = window.index_of(
                    round(cx + returned[0] * fx / returned[2]),
                    round(cy + returned[1] * fy / returned[2]),
                )
                if home == position:
                    seen += 1
            owed += int(seen >= 2)
    return owed


def _temporal_recovery_gate(scene: contract.Scene) -> list[dict[str, object]]:
    """The strict gain a temporal method owes where other views actually looked."""
    owed = _temporal_opportunity(scene)
    if owed <= 0:
        return []
    return [
        _gate(
            "coverage.temporal_recovery",
            "coverage",
            ("temporally_reconstructed_on_surface",),
            ">=",
            float(owed),
            COUNT,
            f"a strict gain, stated by this scene and not by any candidate: {owed} dropped "
            "sample(s) of these frames, away from any foreground boundary the scene declares, "
            "were measured by at least two OTHER posed frames, at the sample and the depth the "
            "truth puts them at, with the correspondence holding in both directions. A "
            "pose-compensated method must put at least that many back, as "
            "temporally reconstructed samples standing on surface the truth really has. Only "
            "recovered depth pays this gate: a refusal, a contradicted spatial estimate and a "
            "single-frame fill are each evidence about a sample rather than a sample put back. "
            "The unchanged baseline and the single-frame spatial primitive recover none of them",
        )
    ]


def _temporal_work_gate() -> list[dict[str, object]]:
    """Bounded projection work, stated rather than timed."""
    return [
        _gate(
            "temporal_work.within_budget",
            "temporal_work",
            ("within_budget",),
            "is",
            True,
            quality.BOOLEAN_UNIT,
            "the projection and comparison work a temporal completion spent must be inside the "
            "budget derivable from its input before it ran - one projection per eligible source "
            "sample into its published footprint, and one bounded comparison per hole. A bound "
            "by construction rather than a wall-clock assertion, which would measure the "
            "machine. Candidates that report no temporal run leave this gate inapplicable",
        )
    ]


def _confirmation_gates(*, owed: int) -> list[dict[str, object]]:
    """What a confirmation is worth: never wrong, and not decorative either."""
    gates = [
        _gate(
            "confirmation.confirmed_false",
            "confirmation",
            ("confirmed_false",),
            "<=",
            0.0,
            COUNT,
            "absolute: zero. A spatial estimate standing in space this scene really leaves "
            "empty must never be CONFIRMED by independent posed evidence. This is the failure a "
            "temporal stage is most likely to produce and the hardest to see: the estimate was "
            "already there, the confirmation only blesses it, and the sample walks away with a "
            "stronger provenance label than one frame could ever have given it",
        )
    ]
    if owed > 0:
        gates.append(
            _gate(
                "confirmation.confirmed_on_surface",
                "confirmation",
                ("confirmed_on_surface",),
                ">=",
                float(owed),
                COUNT,
                f"a strict bar stated by this scene: {owed} of its spatial estimate(s) sit on "
                "real surface that at least two other posed frames measured, so independent "
                "evidence exists for them and a stage that confirms nothing has not checked. "
                "Beside confirmation.confirmed_false, which refuses the opposite failure",
            )
        )
    return gates


def _confidence_gate() -> list[dict[str, object]]:
    """Confidence has to be a claim that can be wrong, not a decoration."""
    return [
        _gate(
            "confidence.high_confidence_worst",
            "confidence",
            ("high_confidence_worst",),
            "<=",
            _ceil(ERROR_WORST_CODES * CODE),
            UNIT,
            f"absolute: {ERROR_WORST_CODES} raw codes, the same worst-sample tolerance the "
            "held-out error gate uses, applied only to reconstructions the candidate itself "
            f"published at confidence {contract.CONFIDENCE_THRESHOLD} or above. A completion is "
            "free to be unsure; what it may not do is be sure and wrong. Lowering its own "
            "confidence is the only way to leave this gate, and that lowering is visible in the "
            "same measurement",
        )
    ]


def _work_gate() -> list[dict[str, object]]:
    """Bounded work, stated rather than timed."""
    return [
        _gate(
            "completion_work.within_budget",
            "completion_work",
            ("within_budget",),
            "is",
            True,
            quality.BOOLEAN_UNIT,
            "the work a completion spent must be inside the budget derivable from its input "
            "before it ran - one neighbourhood scan per hole at its published radius. A bound "
            "by construction rather than a wall-clock assertion, which would measure the "
            "machine. Candidates that report no work record leave this gate inapplicable, and "
            "each case declares exactly which ones those are",
        )
    ]


def _coverage_gates(baseline: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    coverage = baseline["coverage"]
    surface_fraction = _at(coverage, "truth_surface") / _at(coverage, "domain_samples")
    return [
        _gate(
            "coverage.supported_fraction",
            "coverage",
            ("supported_fraction",),
            ">=",
            _floor(surface_fraction * ACCEPTANCE_FRACTION),
            FRACTION,
            "the acceptance target: this fraction of the window must end up measured, or "
            f"reconstructed onto surface that is really there. {ACCEPTANCE_FRACTION} of the "
            "surface this scene actually has; the unchanged baseline does not reach it, which "
            "is the whole reason this thread exists",
        ),
        _gate(
            "coverage.measured_fraction",
            "coverage",
            ("measured_fraction",),
            ">=",
            _floor(_at(coverage, "measured_fraction")),
            FRACTION,
            "non-inferiority: the raw measurements the sensor did return must all still be "
            "there and still be called measurements. A completion that discards awkward "
            "samples and interpolates over them is not a completion",
        ),
    ]


def _error_gates(*, worst: bool = True) -> list[dict[str, object]]:
    gates = [
        _gate(
            "reconstructed_depth.p95",
            "reconstructed_depth",
            ("absolute", "p95"),
            "<=",
            _ceil(ERROR_P95_CODES * CODE),
            UNIT,
            f"absolute: {ERROR_P95_CODES} raw codes. Error is scored ONLY on samples the "
            "candidate reconstructed where truth was held out, so copying the input cannot "
            "earn a good score here, and the coverage gates stop a candidate improving this "
            "by reconstructing only the easy holes",
        )
    ]
    if worst:
        gates.append(
            _gate(
                "reconstructed_depth.worst",
                "reconstructed_depth",
                ("absolute", "worst"),
                "<=",
                _ceil(ERROR_WORST_CODES * CODE),
                UNIT,
                f"absolute: {ERROR_WORST_CODES} raw codes. A single sample placed on the wrong "
                "surface is the failure a percentile hides",
            )
        )
    return gates


def _plane_gates(baseline: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    plane = baseline["plane"]
    margin = ROUGHNESS_MARGIN_CODES * CODE
    return [
        _gate(
            "plane.residual.p95",
            "plane",
            ("residuals", "p95"),
            "<=",
            _ceil(_at(plane, "residuals", "p95") + margin),
            UNIT,
            f"non-inferiority: the baseline's own roughness plus {ROUGHNESS_MARGIN_CODES} raw "
            "codes. A surface known to be flat may not come out of a completion lumpier than "
            "it went in",
        ),
        _gate(
            "plane.residual.worst",
            "plane",
            ("residuals", "worst"),
            "<=",
            _ceil(_at(plane, "residuals", "worst") + margin),
            UNIT,
            f"non-inferiority: the baseline's own worst residual plus "
            f"{ROUGHNESS_MARGIN_CODES} raw codes, for a local bump a percentile would hide",
        ),
    ]


def _shape_gates(
    baseline: Mapping[str, Mapping[str, object]], *, dimensions: bool, landmarks: bool
) -> list[dict[str, object]]:
    gates: list[dict[str, object]] = []
    if dimensions:
        gates.append(
            _gate(
                "dimensions.worst",
                "dimensions",
                ("absolute", "worst"),
                "<=",
                _ceil(
                    _at(baseline["dimensions"], "absolute", "worst") + DIMENSION_MARGIN_CODES * CODE
                ),
                UNIT,
                f"non-inferiority: the baseline's own worst extent error plus "
                f"{DIMENSION_MARGIN_CODES} raw codes. Smoothing an object smaller is the "
                "classic way a denser result looks better and measures worse",
            )
        )
    if landmarks:
        gates.append(
            _gate(
                "landmarks.worst",
                "landmarks",
                ("offsets", "worst"),
                "<=",
                _ceil(
                    _at(baseline["landmarks"], "offsets", "worst") + LANDMARK_MARGIN_CODES * CODE
                ),
                UNIT,
                f"non-inferiority: the baseline's own worst landmark offset plus "
                f"{LANDMARK_MARGIN_CODES} raw codes. A feature must not move",
            )
        )
    return gates


def _surface_gates(baseline: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    surface = baseline["surface"]
    margin = SURFACE_MARGIN_CODES * CODE
    return [
        _gate(
            "surface.observed_to_reference.p95",
            "surface",
            ("observed_to_reference", "distances", "p95"),
            "<=",
            _ceil(_at(surface, "observed_to_reference", "distances", "p95") + margin),
            UNIT,
            f"non-inferiority: the baseline's own agreement plus {SURFACE_MARGIN_CODES} raw "
            "codes, measured observed-to-reference, which is the direction that notices "
            "geometry that is there and should not be",
        ),
        _gate(
            "surface.reference_to_observed.p95",
            "surface",
            ("reference_to_observed", "distances", "p95"),
            "<=",
            _ceil(_at(surface, "reference_to_observed", "distances", "p95") + margin),
            UNIT,
            f"non-inferiority: the baseline's own agreement plus {SURFACE_MARGIN_CODES} raw "
            "codes, measured reference-to-observed, which is the direction that notices "
            "surface that is missing. Both directions, because neither alone is agreement",
        ),
    ]


def _false_fill_gate(
    baseline: Mapping[str, Mapping[str, object]], *, fraction: float | None
) -> list[dict[str, object]]:
    coverage = baseline["coverage"]
    if not _at(coverage, "truth_empty"):
        return []
    if fraction is None:
        return [
            _gate(
                "coverage.false_fill",
                "coverage",
                ("false_fill",),
                "<=",
                0.0,
                COUNT,
                "absolute: zero. Space this scene really leaves empty must stay empty. This is "
                "the gate a dense fill fails while its aggregate coverage improves",
            )
        ]
    return [
        _gate(
            "coverage.false_fill_fraction",
            "coverage",
            ("false_fill_fraction",),
            "<=",
            _ceil(fraction),
            FRACTION,
            "a stated allowance, and the only one in this corpus: reprojecting a real sample "
            "through exact poses rounds it one pixel outside the silhouette. The curved "
            "single-view case gates false fill at exactly zero and is where invention is "
            "refused outright",
        )
    ]


def _confident_false_fill_gate() -> list[dict[str, object]]:
    """Scoped to the aperture-ambiguity case, and nowhere else.

    Every other case in this corpus already gates all false fill at exactly
    zero (see ``_false_fill_gate`` with ``fraction=None``), so a separate bar
    on *confident* false fill there would be vacuous: false fill of any
    confidence is already refused, so nothing new could ever be caught by it.

    The aperture-ambiguity case is different on purpose. It pairs one sample
    of genuinely empty space with one held-out sample of real surface whose
    input evidence - the 9x9 neighbourhood a spatial completion scores - is
    byte-identical between the two. No deterministic single-frame method can
    refuse the empty one without also refusing the real one, so this corpus
    does not require both to be answered correctly; it requires whichever
    side loses to say so honestly. Reconstructing the empty sample and being
    unsure about it is the ambiguity showing up where it belongs. Reconstructing
    it and reporting high confidence is not the ambiguity - it is a claim the
    input could never have supported, because the same bytes would have
    produced the same confidence over real surface. That is the one thing
    this case refuses outright, which is why this gate exists only here.
    """
    return [
        _gate(
            "confidence.high_confidence_false_fill",
            "confidence",
            ("high_confidence_false_fill",),
            "<=",
            0.0,
            COUNT,
            "absolute: zero, and scoped to this case alone. Elsewhere false fill of any "
            "confidence already fails coverage.false_fill, which makes a confidence-only bar "
            "vacuous. Here the aperture and the held-out dropout score identical input bytes, "
            "so a completion may be unsure about the aperture without failing this gate - that "
            "is the disclosed ambiguity - but it may not be sure and wrong, because nothing in "
            "one frame could have told it apart from the real surface a few pixels away",
        )
    ]


def _edge_gates() -> list[dict[str, object]]:
    return [
        _gate(
            "edge.bridging",
            "edge",
            ("bridging",),
            "<=",
            0.0,
            COUNT,
            "absolute: zero. A reconstructed sample in the empty band between the foreground "
            "and the background is a flying pixel, whatever produced it",
        ),
        _gate(
            "edge.misplaced",
            "edge",
            ("misplaced",),
            "<=",
            0.0,
            COUNT,
            "absolute: zero. A reconstructed sample on the wrong side of the step has been "
            "put on a surface it is not on",
        ),
        _gate(
            "edge_step_column.worst",
            "edge_step_column",
            ("absolute", "worst"),
            "<=",
            STEP_COLUMN_TOLERANCE,
            COLUMNS,
            f"absolute: {STEP_COLUMN_TOLERANCE} column. The step belongs where the truth puts "
            "it. The baseline fails this gate on purpose - dropout along the edge moves its "
            "apparent boundary inward, which is exactly the defect completion should repair",
        ),
    ]


def _slot_gates() -> list[dict[str, object]]:
    return [
        _gate(
            "slot.open",
            "slot",
            ("open",),
            "is",
            True,
            quality.BOOLEAN_UNIT,
            "absolute: a gap that is really there is still there. Filling a slot with "
            "foreground depth is the most convincing-looking way to gain coverage and the "
            "least recoverable error in the result",
        ),
        _gate(
            "slot.closed_samples",
            "slot",
            ("closed_samples",),
            "<=",
            0.0,
            COUNT,
            "absolute: zero. Beside the boolean, so the report says how far the slot was "
            "closed and not only that it was",
        ),
    ]


def _fusion_gates(baseline: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        _gate(
            "fusion.surface_extracted",
            "fusion",
            ("surface_extracted",),
            "is",
            True,
            quality.BOOLEAN_UNIT,
            "the unchanged TSDF and its unchanged surface extraction must still produce a "
            "surface from these frames; a completion that breaks fusion has not helped",
        ),
        _gate(
            "fusion_surface.observed_to_reference.p95",
            "fusion_surface",
            ("observed_to_reference", "distances", "p95"),
            "<=",
            _ceil(
                _at(baseline["fusion_surface"], "observed_to_reference", "distances", "p95")
                + SURFACE_MARGIN_CODES * CODE
            ),
            UNIT,
            f"non-inferiority: the baseline's own fused-surface agreement plus "
            f"{SURFACE_MARGIN_CODES} raw codes. Measured on the surface the real "
            "scanner3d.tsdf_surface extractor pulled out of the real scanner3d.tsdf field",
        ),
    ]


def _weighted_fusion_gates(
    baseline: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """What the accepted weighted input contract must not cost. No gain is gated.

    There was a gain gate here, over ``weighted_fusion.voxel_updates``, and it
    is gone deliberately rather than because it failed. Voxel updates are
    accounting: they say a sample reached the field, not that the field came
    out better. On this corpus the accepted temporal result reaches 225 more
    voxel observations and produces the byte-identical observed-voxel count,
    the same 333 surface vertices and the same truth-referenced coverage as the
    raw multi-view baseline, because every one of those observations lands on a
    voxel the supporting views already saw. Gating the update count would have
    dressed that redundancy up as an improvement.

    So what is gated is non-inferiority, and it is now stated against the raw
    multi-view baseline directly: ``observed_voxel_gain``,
    ``surface_vertex_gain`` and ``surface_coverage_gain`` are that baseline
    subtracted from this field's own output measures, and none of them may go
    negative. Accepted contribution may not fall, the surface must still
    extract, its agreement with truth may not get worse by more than the same
    margin every other surface gate in this corpus uses, and an unconfirmed
    single-frame estimate must contribute exactly nothing.

    The corpus keeps measuring the gain it does not gate. If a later scene ever
    does produce one, these three fields are where it will show, and it will
    show against the same frames rather than against a baseline with the
    supporting views withheld.
    """
    return [
        _gate(
            "weighted_fusion.observed_voxel_gain",
            "weighted_fusion",
            ("observed_voxel_gain",),
            ">=",
            0.0,
            COUNT,
            "non-inferiority against the SAME raw multi-view field - same frames, same poses, "
            "same volume, same extractor. This corpus measures exactly zero here: what the "
            "accepted temporal result recovers, the supporting views already put in this "
            "volume. No gain is claimed and none is gated; what is gated is that none is lost",
        ),
        _gate(
            "weighted_fusion.surface_vertex_gain",
            "weighted_fusion",
            ("surface_vertex_gain",),
            ">=",
            0.0,
            COUNT,
            "non-inferiority: the surface the unchanged extractor pulls out of the weighted "
            "field may not have fewer vertices than the one it pulls out of the same raw "
            "multi-view field. Measured at zero on this corpus",
        ),
        _gate(
            "weighted_fusion.surface_coverage_gain",
            "weighted_fusion",
            ("surface_coverage_gain",),
            ">=",
            0.0,
            FRACTION,
            "non-inferiority on the output-facing measure: the fraction of the truth model "
            "covered by the extracted surface, minus the same fraction for the raw multi-view "
            "baseline. This is where a real geometric gain would appear, and on this corpus it "
            "is exactly zero",
        ),
        _gate(
            "weighted_fusion.contribution",
            "weighted_fusion",
            ("contribution",),
            ">=",
            _floor(_at(baseline["weighted_fusion"], "contribution")),
            COUNT,
            "non-inferiority: the total fusion weight the baseline's own measurements put into "
            "this volume. A weighting that let a reconstruction displace evidence rather than "
            "add to it would lower this",
        ),
        _gate(
            "weighted_fusion.observed_voxels",
            "weighted_fusion",
            ("observed_voxels",),
            ">=",
            _floor(_at(baseline["weighted_fusion"], "observed_voxels")),
            COUNT,
            "non-inferiority: the baseline's own observed space. This corpus's recovered samples "
            "fall on voxels other views already saw, so no gain is claimed here and none is "
            "gated - what is gated is that none is lost",
        ),
        _gate(
            "weighted_fusion.surface_extracted",
            "weighted_fusion",
            ("surface_extracted",),
            "is",
            True,
            quality.BOOLEAN_UNIT,
            "the unchanged scanner3d.tsdf_surface extractor must still pull a surface out of the "
            "weighted field; a weighting that breaks extraction has not helped",
        ),
        _gate(
            "weighted_fusion_surface.observed_to_reference.p95",
            "weighted_fusion_surface",
            ("observed_to_reference", "distances", "p95"),
            "<=",
            _ceil(
                _at(
                    baseline["weighted_fusion_surface"], "observed_to_reference", "distances", "p95"
                )
                + SURFACE_MARGIN_CODES * CODE
            ),
            UNIT,
            f"non-inferiority: the baseline's own weighted-fusion surface agreement plus "
            f"{SURFACE_MARGIN_CODES} raw codes. Measured on the surface the real "
            "scanner3d.tsdf_surface extractor pulled out of the real weighted field",
        ),
        *_excluded_contribution_gate(),
    ]


def _excluded_contribution_gate() -> list[dict[str, object]]:
    """The absolute bar: an unconfirmed single-frame estimate adds exactly nothing.

    Zero, and not a small number. The aperture-ambiguity case established that
    one frame cannot tell a small true gap in a surface from a small dropout on
    it, so a spatial estimate produced under that ambiguity is a hypothesis. A
    hypothesis that can add any weight at all to a voxel is invented geometry
    arriving with a discount, which is worse than either fusing it or refusing
    it, because it is refusable-looking.
    """
    return [
        _gate(
            "weighted_fusion.spatially_excluded_contribution",
            "weighted_fusion",
            ("spatially_excluded_contribution",),
            "<=",
            0.0,
            COUNT,
            "absolute: every accepted single-frame spatial estimate that no independent view "
            "confirmed contributes exactly zero to the field. Not a low weight - zero, which "
            "scanner3d.tsdf treats as absence",
        ),
    ]


def _weighted_exclusion_gates(
    baseline: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """What weighted fusion must do on a case whose spatial estimates are unconfirmed.

    The counterpart of :func:`_weighted_fusion_gates`, for the case where there
    is no second view and therefore nothing to confirm anything. Nothing is
    gained here and nothing may be: the field a single-frame completion
    produces has to be the field its measurements alone produce, byte for byte,
    and its observed space may not exceed the baseline's by one voxel.
    """
    return [
        _gate(
            "weighted_fusion.observed_voxel_gain",
            "weighted_fusion",
            ("observed_voxel_gain",),
            "<=",
            0.0,
            COUNT,
            "absolute: not one voxel more observed than the same raw field observes. Space does "
            "not become observed by being filled in an image, and this is the case where "
            "filling it is indistinguishable from recovering it",
        ),
        _gate(
            "weighted_fusion.matches_measured_only",
            "weighted_fusion",
            ("matches_measured_only",),
            "is",
            True,
            quality.BOOLEAN_UNIT,
            "the weighted field must be BYTE-IDENTICAL - values and accumulated weights both - "
            "to the field this scene's raw measurements alone produce. A single-frame estimate "
            "nothing independent confirmed changed the picture and may not change the geometry",
        ),
        _gate(
            "weighted_fusion.observed_voxels",
            "weighted_fusion",
            ("observed_voxels",),
            "<=",
            _ceil(_at(baseline["weighted_fusion"], "observed_voxels")),
            COUNT,
            "absolute: the baseline's own observed space, and not one voxel more. Space does not "
            "become observed by being filled in an image, and this is the case where filling it "
            "is indistinguishable from recovering it",
        ),
        *_excluded_contribution_gate(),
    ]


def _tracking_gates(baseline: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    """Non-inferiority bars over the measured-only replay, and over nothing else.

    Every gate here names 'tracking' or 'tracking_pose': the tracker over each
    frame's own returned samples, which is the only route this project's
    tracker has and the one the completion stages deliberately stay out of.

    The counterfactual 'tracking_completed_cloud' records are published beside
    them and carry NO gate. That is deliberate rather than an omission: they
    measure what would happen if a completion did enter the tracker - it
    degrades placement on this case - and a bar over a route the product does
    not take would read as an acceptance condition for taking it.
    """
    accepted = int(_at(baseline["tracking"], "frames_accepted"))
    poses = baseline["tracking_pose"]
    return [
        _gate(
            "tracking_pose.rotation.worst",
            "tracking_pose",
            ("rotation", "worst"),
            "<=",
            _ceil(_at(poses, "rotation", "worst") + TRACKING_ROTATION_MARGIN_DEGREES),
            quality.DEGREES_UNIT,
            "non-inferiority: the baseline's own worst placement error plus "
            f"{TRACKING_ROTATION_MARGIN_DEGREES} degrees - a sixteenth of the rotation between "
            "adjacent frames in this scene - measured by "
            "scanner3d.quality.pose_sequence_error against the scene's exact poses. Where the "
            "frames were placed, not merely whether they were",
        ),
        _gate(
            "tracking_pose.translation.worst",
            "tracking_pose",
            ("translation", "worst"),
            "<=",
            _ceil(_at(poses, "translation", "worst") + TRACKING_TRANSLATION_MARGIN_CODES * CODE),
            UNIT,
            f"non-inferiority: the baseline's own worst placement offset plus "
            f"{TRACKING_TRANSLATION_MARGIN_CODES} raw codes, against the scene's exact poses",
        ),
        _gate(
            "tracking.accepted_all",
            "tracking",
            ("accepted_all",),
            "is",
            True,
            quality.BOOLEAN_UNIT,
            "the real tracker placed every frame of this scene from its measured samples "
            "alone, and a completion may not cost that",
        ),
        _gate(
            "tracking.frames_accepted",
            "tracking",
            ("frames_accepted",),
            ">=",
            float(accepted),
            COUNT,
            f"non-inferiority: the baseline placed {accepted} frame(s). This task records that "
            "baseline; it does not claim any candidate improved tracking",
        ),
    ]


# --- candidates and their declared expectations ------------------------------------------


def _candidate(
    name: str,
    what: str,
    outcome: str,
    *,
    failed: Sequence[str] = (),
    inapplicable: Sequence[str] = (),
    refusal: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "what": what,
        "expectation": {
            "outcome": outcome,
            "failed_gates": sorted(failed),
            "inapplicable_gates": sorted(inapplicable),
            "refusal": refusal,
        },
    }


#: The nine corruption controls, with the exact refusal each must earn. Every
#: one of them is a way a completion could look better than it is, and a run in
#: which any of them is scored at all is a run whose oracle has stopped working.
CORRUPTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "forged_measured_states",
        "reports every reconstructed sample as a camera measurement",
        "measured_value_changed",
    ),
    (
        "mutated_raw_frame",
        "returns raw bytes that are not the ones it was given",
        "raw_frame_mutated",
    ),
    (
        "unsupported_fill",
        "reconstructs without naming any support",
        "unsupported_reconstruction",
    ),
    (
        "forged_confidence",
        "claims a confidence outside the fraction it is declared to be",
        "confidence_out_of_range",
    ),
    (
        "missing_carrying_depth",
        "declares a sample missing and hands out a depth for it anyway",
        "missing_carries_depth",
    ),
    ("unknown_provenance_state", "invents a fifth provenance state", "unknown_state"),
    (
        "truncated_window",
        "returns fewer samples than the window, dropping the hardest",
        "sample_count_mismatch",
    ),
    (
        "phantom_support",
        "names support at a pixel that returned no measurement",
        "support_not_measured",
    ),
    (
        "temporal_support_from_own_frame",
        "calls a same-frame fill a temporal reconstruction",
        "temporal_support_not_another_frame",
    ),
)


def _corruption_candidates(gates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    every = sorted(str(gate["name"]) for gate in gates)
    return [
        _candidate(name, what, "refused", inapplicable=every, refusal=refusal)
        for name, what, refusal in CORRUPTIONS
    ]


def _case(
    *,
    case_id: str,
    what: str,
    why: str,
    notes: Sequence[str],
    shapes: Sequence[Mapping[str, object]],
    window: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
    regions: Mapping[str, object],
) -> tuple[dict[str, object], contract.Scene]:
    """The half of a case that can be built before its baseline is known."""
    document: dict[str, object] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": harness.CASE_KIND,
        "case_id": case_id,
        "what": what,
        "why": why,
        "notes": list(notes),
        "scene": {
            "intrinsics": dict(INTRINSICS),
            "depth_scale": DEPTH_SCALE,
            "window": dict(window),
            "shapes": [dict(shape) for shape in shapes],
            "seed": f"depth-completeness-028:{case_id}",
            "frames": [dict(frame) for frame in frames],
        },
        "regions": dict(regions),
        "gates": [],
        "candidates": [],
    }
    return document, _materialise(document)


# --- the seven cases ---------------------------------------------------------------------

PLANE_WINDOW = _window(300, 188, 40, 24)
SPHERE_WINDOW = _window(300, 182, 40, 36)
STEP_WINDOW = _window(310, 188, 40, 24)
SLOT_WINDOW = _window(300, 188, 40, 24)
MULTIVIEW_WINDOW = _window(310, 190, 20, 16)

#: The code below which a sample belongs to the foreground object in the two
#: cases that have one. Between the box front face and the backing plane, and
#: stated in the fixture rather than discovered, so the harness cannot choose a
#: threshold that flatters a candidate.
FOREGROUND_BELOW: Final = 15000

#: The window columns the slot occupies. Read off the rendered truth: the two
#: boxes leave exactly these three columns showing the backing plane.
SLOT_COLUMNS: Final = (19, 20, 21)

STEP_SHAPES = (
    _box((4.0, 6.0, 4.0), (0.0, 0.0, 100.0)),
    _plane((0.0, 0.0, 1.0), 140.0),
)
SLOT_SHAPES = (
    _box((4.0, 6.0, 4.0), (-4.36, 0.0, 100.0)),
    _box((4.0, 6.0, 4.0), (4.36, 0.0, 100.0)),
    _plane((0.0, 0.0, 1.0), 140.0),
)
MULTIVIEW_SHAPES = (
    _box((2.0, 3.0, 2.0), (0.0, 0.0, 100.0)),
    _sphere((1.0, -2.0, 97.0), 1.2),
)

SPECKLE = {"kind": "speckle", "fraction": 0.12}
LIGHT_SPECKLE = {"kind": "speckle", "fraction": 0.08}

APERTURE_WINDOW = _window(300, 188, 40, 24)

#: Four thin coplanar slabs, front faces all at z = 100.0, tiling the window
#: except a single window sample - genuinely empty space, nothing behind it.
#: The slabs are thin enough (half-depth 0.02) that ray divergence at the
#: window's edges cannot re-enter a neighbouring slab and hide the aperture.
APERTURE_BIG: Final = 40.0
APERTURE_SLAB_HALF_DEPTH: Final = 0.02
APERTURE_SLAB_CENTRE_Z: Final = 100.02

#: The one-pixel gap in x and y, at window column 29, row 12. Read off the
#: rendered truth, the way SLOT_COLUMNS is read off the slot case's boxes.
APERTURE_X: Final = (2.125, 2.375)
APERTURE_Y: Final = (-0.125, 0.125)

#: The volume the aperture case fuses into, weighted. It is cut around the one
#: constant-depth plane this scene has - the window's own extent at z = 100,
#: opened by two truncation distances on every side - so the field contains the
#: surface the case is about and nothing else.
APERTURE_FUSION: Final[dict[str, object]] = {
    "minimum": [-5.5, -3.5, 98.5],
    "maximum": [5.5, 3.5, 101.5],
    "voxel_size": 0.25,
    "truncation": 0.75,
    "min_depth": 50.0,
    "max_depth": 200.0,
    "min_weight": 1.0,
    "coverage_tolerance": 0.5,
}

APERTURE_SHAPES = (
    _box(
        (APERTURE_BIG, APERTURE_BIG, APERTURE_SLAB_HALF_DEPTH),
        (APERTURE_X[0] - APERTURE_BIG, 0.0, APERTURE_SLAB_CENTRE_Z),
    ),  # covers x <= 2.125
    _box(
        (APERTURE_BIG, APERTURE_BIG, APERTURE_SLAB_HALF_DEPTH),
        (APERTURE_X[1] + APERTURE_BIG, 0.0, APERTURE_SLAB_CENTRE_Z),
    ),  # covers x >= 2.375
    _box(
        (APERTURE_BIG, APERTURE_BIG, APERTURE_SLAB_HALF_DEPTH),
        (0.0, APERTURE_Y[0] - APERTURE_BIG, APERTURE_SLAB_CENTRE_Z),
    ),  # covers y <= -0.125
    _box(
        (APERTURE_BIG, APERTURE_BIG, APERTURE_SLAB_HALF_DEPTH),
        (0.0, APERTURE_Y[1] + APERTURE_BIG, APERTURE_SLAB_CENTRE_Z),
    ),  # covers y >= 0.125
)

#: The pose-compensated motion case. The box front face of ``STEP_SHAPES`` sits
#: at z = 96, so one image pixel there is ``96 / fx`` world units and a scene
#: shift of five of them moves the silhouette exactly five columns. Five rather
#: than one because the strip masks have to be disjoint IN OBJECT SPACE: a
#: sample dropped in one frame is only recoverable from another frame that did
#: not drop the same part of the object, and a one-pixel motion makes those two
#: statements almost the same one.
MOTION_WINDOW = STEP_WINDOW
MOTION_PIXEL: Final = 96.0 / 400.0
MOTION_STEP_PIXELS: Final = 5

#: The strip each frame loses, in that frame's own columns. Frame 0's strip
#: straddles its own silhouette, at column 26 of the rendered truth - that is
#: where a same-pixel carry from a moved frame lands background depth on
#: foreground surface. The other two are placed away from the object columns
#: frame 0 is missing, so the three losses are complementary rather than the
#: same part of the box three times.
MOTION_STRIPS: Final[tuple[tuple[int, int], ...]] = ((24, 28), (4, 8), (30, 34))

#: The confirmation case. Four coplanar slabs with one genuinely empty sample -
#: the same aperture construction the single-view ambiguity case uses - plus one
#: small pillar standing in front of them, one sample wide, whose front face is
#: four units nearer than the plane behind it.
CONFIRMATION_WINDOW = APERTURE_WINDOW
CONFIRMATION_PILLAR_HALF: Final = 0.13
CONFIRMATION_PILLAR_CENTRE: Final = (-2.64, -1.44, 98.0)
CONFIRMATION_SHAPES = (
    *APERTURE_SHAPES,
    _box((CONFIRMATION_PILLAR_HALF, CONFIRMATION_PILLAR_HALF, 2.0), CONFIRMATION_PILLAR_CENTRE),
)

#: Where the pillar and the held-out plane sample land in the window, read off
#: the rendered truth the way ``SLOT_COLUMNS`` and the aperture are.
CONFIRMATION_PILLAR_PIXEL: Final = (9, 6)
CONFIRMATION_PLANE_PIXEL: Final = (20, 18)

#: How far the two other views are displaced. One pixel at the plane's depth,
#: one each way, which is enough for two genuinely distinct viewpoints and small
#: enough that the pillar and the plane behind it both land within half a pixel
#: of the samples they belong to.
CONFIRMATION_SHIFT: Final = 100.0 / 400.0

#: The paired dropout: a blob of this radius removes exactly its centre
#: sample and nothing else (mask_indices keeps hypot(c-column, r-row) <=
#: radius). Placed far from the aperture, on the same constant-depth plane,
#: so its 9x9 input neighbourhood - a single zero surrounded by code 12800 -
#: is byte-identical to the aperture's.
APERTURE_DROPOUT_COLUMN: Final = 9
APERTURE_DROPOUT_ROW: Final = 12
APERTURE_DROPOUT_RADIUS: Final = 0.5


def _fronto_complete() -> dict[str, object]:
    document, scene = _case(
        case_id="plane-fronto-complete",
        what="A fronto-parallel plane with nothing missing at all.",
        why=(
            "The control that says this benchmark measures the candidates and not its own "
            "noise. On complete input an identity completion must reproduce every measured "
            "value and every downstream metric exactly, and a candidate that discards "
            "measurements to smooth the result must still be refused."
        ),
        notes=[
            "No dropout mask. Every sample of the window is a returned measurement.",
            "The reconstructed-depth gates do not apply to a candidate that reconstructs "
            "nothing; the fixture declares that rather than letting a gate be skipped "
            "silently.",
        ],
        shapes=[_plane((0.0, 0.0, 1.0), 100.0)],
        window=PLANE_WINDOW,
        frames=[{"index": 0, "pose": _still(), "masks": []}],
        regions=_regions(plane=True, dimensions=True, landmarks=True, surface_tolerance=0.1),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_plane_gates(baseline),
        *_shape_gates(baseline, dimensions=True, landmarks=True),
        *_surface_gates(baseline),
        *_work_gate(),
    ]
    errors = ["reconstructed_depth.p95", "reconstructed_depth.worst"]
    work = ["completion_work.within_budget"]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline over input that is already complete",
            "pass",
            inapplicable=errors + work,
        ),
        _candidate(
            "identity",
            "a completion that completes nothing, over complete input",
            "pass",
            inapplicable=errors + work,
        ),
        _candidate(
            "spatial_completion",
            "the product primitive over input with no holes in it: it faces nothing, "
            "reconstructs nothing, and every downstream metric is the baseline's exactly",
            "pass",
            inapplicable=errors,
        ),
        _candidate(
            "discarded_measurements",
            "with nothing missing there is nothing to interpolate, so this candidate only "
            "discards measurements - and every downstream metric notices",
            "fail",
            failed=[
                "coverage.measured_fraction",
                "coverage.supported_fraction",
                "dimensions.worst",
                "landmarks.worst",
                "surface.reference_to_observed.p95",
            ],
            inapplicable=errors + work,
        ),
        _candidate(
            "mutated_raw_frame",
            "returns raw bytes that are not the ones it was given",
            "refused",
            inapplicable=sorted(str(gate["name"]) for gate in gates),
            refusal="raw_frame_mutated",
        ),
    ]
    return document


def _fronto_masked() -> dict[str, object]:
    document, scene = _case(
        case_id="plane-fronto",
        what="A fronto-parallel plane with speckle, a connected hole and a strip collapse.",
        why=(
            "The case where a dense fill is RIGHT. A constant-depth surface is the one "
            "geometry nearest-neighbour completion reproduces exactly, and an oracle that "
            "refused it would be refusing density rather than invention. The same corpus "
            "refuses the same candidate on every other geometry."
        ),
        notes=[
            "Three distinct dropout shapes, not one percentage: isolated speckle, one "
            "connected interior hole, and a one-sided strip collapse patterned after the "
            "column-wise loss seen in the recorded capture.",
            "The strip is where interpolation has support on one side only, which is the "
            "shape that most often produces a plausible and wrong surface.",
        ],
        shapes=[_plane((0.0, 0.0, 1.0), 100.0)],
        window=PLANE_WINDOW,
        frames=[
            {
                "index": 0,
                "pose": _still(),
                "masks": [
                    SPECKLE,
                    {"kind": "blob", "column": 10, "row": 10, "radius": 3.5},
                    {"kind": "strip", "first_column": 30, "last_column": 33},
                ],
            }
        ],
        regions=_regions(plane=True, dimensions=True, landmarks=True, surface_tolerance=0.1),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_plane_gates(baseline),
        *_shape_gates(baseline, dimensions=True, landmarks=True),
        *_surface_gates(baseline),
        *_local_recovery_gate(scene),
        *_confidence_gate(),
        *_work_gate(),
    ]
    errors = ["reconstructed_depth.p95", "reconstructed_depth.worst"]
    confidence = ["confidence.high_confidence_worst"]
    work = ["completion_work.within_budget"]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline: what returned is measured, the rest stays missing",
            "fail",
            failed=["coverage.supported_fraction", "coverage.local_recovery"],
            inapplicable=errors + confidence + work,
        ),
        _candidate(
            "dense_nearest",
            "fills every hole with its nearest returned sample, which on a constant-depth "
            "plane is the right answer",
            "pass",
            inapplicable=work,
        ),
        _candidate(
            "broad_interpolation",
            "averages the nearest measured sample in each direction",
            "pass",
            inapplicable=work,
        ),
        _candidate(
            "spatial_completion",
            "the product primitive: it recovers the isolated dropout and the interior hole "
            "from local evidence, and refuses the four-column strip collapse, whose span is "
            "further than any support inside this frame reaches",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
        _candidate(
            "easy_samples_only",
            "fills only holes with three close neighbours and leaves the hard ones missing",
            "fail",
            failed=["coverage.supported_fraction"],
            inapplicable=work,
        ),
        _candidate(
            "discarded_measurements",
            "drops the extreme returned samples and interpolates over them",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "coverage.measured_fraction",
                "landmarks.worst",
            ],
            inapplicable=work,
        ),
        _candidate(
            "spatial_completion_forged_work",
            "reports one refusal as an acceptance and changes nothing else",
            "refused",
            inapplicable=sorted(str(gate["name"]) for gate in gates),
            refusal="work_counts_inconsistent",
        ),
        _candidate(
            "spatial_completion_recursive_support",
            "cites one of its own reconstructions as the evidence for another",
            "refused",
            inapplicable=sorted(str(gate["name"]) for gate in gates),
            refusal="support_not_measured",
        ),
        *_corruption_candidates(gates),
    ]
    return document


def _oblique() -> dict[str, object]:
    document, scene = _case(
        case_id="plane-oblique",
        what="An oblique plane with the same three dropout shapes.",
        why=(
            "The same dense fills that are exactly right on a fronto-parallel plane are "
            "wrong here, and wrong in the way that matters: the surface comes out lumpier "
            "than the one that went in. Tilt is the whole difference between the two cases."
        ),
        notes=[
            "The truth is quantised to whole raw codes, so the plane's own residuals are "
            "not zero. The roughness gates are the baseline's own value plus a margin, "
            "which is what makes 'lumpier than before' the thing being measured.",
        ],
        shapes=[_plane((0.3, 0.15, 1.0), 105.0)],
        window=PLANE_WINDOW,
        frames=[
            {
                "index": 0,
                "pose": _still(),
                "masks": [
                    SPECKLE,
                    {"kind": "blob", "column": 10, "row": 10, "radius": 3.5},
                    {"kind": "strip", "first_column": 30, "last_column": 33},
                ],
            }
        ],
        regions=_regions(plane=True, dimensions=True, landmarks=True),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_plane_gates(baseline),
        *_shape_gates(baseline, dimensions=True, landmarks=True),
        *_local_recovery_gate(scene),
        *_confidence_gate(),
        *_work_gate(),
    ]
    errors = ["reconstructed_depth.p95", "reconstructed_depth.worst"]
    confidence = ["confidence.high_confidence_worst"]
    work = ["completion_work.within_budget"]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline",
            "fail",
            failed=["coverage.supported_fraction", "coverage.local_recovery"],
            inapplicable=errors + confidence + work,
        ),
        _candidate(
            "dense_nearest",
            "fills every hole with its nearest returned sample, which on a tilted surface "
            "is a staircase",
            "fail",
            failed=[
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
                "plane.residual.p95",
                "plane.residual.worst",
            ],
            inapplicable=work,
        ),
        _candidate(
            "broad_interpolation",
            "averages the nearest measured sample in each direction",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
                "plane.residual.p95",
                "plane.residual.worst",
            ],
            inapplicable=work,
        ),
        _candidate(
            "spatial_completion",
            "the product primitive on a tilted surface: the same holes it recovered on the "
            "fronto-parallel plane, and this time the answer is a different depth at every "
            "one of them. Copying the nearest code cannot do this and a local fit can",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
        _candidate(
            "easy_samples_only",
            "fills only the easy holes: accurate where it acts, and short of the target",
            "fail",
            failed=["coverage.supported_fraction"],
            inapplicable=work,
        ),
        _candidate(
            "discarded_measurements",
            "drops the extreme returned samples and interpolates over them",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "coverage.measured_fraction",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
                "plane.residual.p95",
                "plane.residual.worst",
                "dimensions.worst",
                "landmarks.worst",
            ],
            inapplicable=work,
        ),
        _candidate(
            "spatial_completion_recursive_support",
            "cites one of its own reconstructions as the evidence for another",
            "refused",
            inapplicable=sorted(str(gate["name"]) for gate in gates),
            refusal="support_not_measured",
        ),
    ]
    return document


def _curved() -> dict[str, object]:
    document, scene = _case(
        case_id="curved-surface",
        what="A sphere against empty space, with speckle and a connected hole.",
        why=(
            "Two failures at once that a plane cannot show: a smooth curved surface a "
            "staircase fill roughens, and real empty space around the silhouette that a "
            "dense fill paints with whatever was nearest. False fill is gated at exactly "
            "zero here."
        ),
        notes=[
            "Most of this window is empty: the sphere subtends only part of it, and every "
            "sample outside the silhouette is truth zero - nothing there to measure, and "
            "nothing there to reconstruct.",
            "Missing coverage and empty space are different facts and are counted against "
            "different denominators. A completion is not asked to fill the background.",
        ],
        shapes=[_sphere((0.0, 0.0, 150.0), 6.0)],
        window=SPHERE_WINDOW,
        frames=[
            {
                "index": 0,
                "pose": _still(),
                "masks": [SPECKLE, {"kind": "blob", "column": 18, "row": 18, "radius": 3.5}],
            }
        ],
        regions=_regions(surface_tolerance=0.1, dimensions=True, landmarks=True),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_false_fill_gate(baseline, fraction=None),
        *_error_gates(),
        *_shape_gates(baseline, dimensions=True, landmarks=True),
        *_surface_gates(baseline),
        *_confidence_gate(),
        *_work_gate(),
    ]
    errors = ["reconstructed_depth.p95", "reconstructed_depth.worst"]
    confidence = ["confidence.high_confidence_worst"]
    work = ["completion_work.within_budget"]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline: no invented geometry, and short of the target",
            "fail",
            failed=["coverage.supported_fraction"],
            inapplicable=errors + confidence + work,
        ),
        _candidate(
            "dense_nearest",
            "fills every hole, including the empty background, with its nearest sample",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "coverage.false_fill",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
                "dimensions.worst",
                "landmarks.worst",
                "surface.observed_to_reference.p95",
            ],
        ),
        _candidate(
            "broad_interpolation",
            "averages across the silhouette and into the background",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "coverage.false_fill",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
                "landmarks.worst",
                "surface.observed_to_reference.p95",
            ],
        ),
        _candidate(
            "easy_samples_only",
            "fills only the easy holes: no invention, and still short of the target",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "coverage.supported_fraction",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
            ],
            inapplicable=work,
        ),
        _candidate(
            "spatial_completion",
            "the product primitive against a silhouette: the background is one connected "
            "region far larger than any local evidence spans and is refused whole, and on the "
            "sphere itself the plane it would fit misses its own support, so this frame is "
            "returned unchanged. Conservative refusal, stated as the failure it is rather "
            "than as a success",
            "fail",
            failed=["coverage.supported_fraction"],
            inapplicable=errors + confidence,
        ),
    ]
    return document


def _step() -> dict[str, object]:
    document, scene = _case(
        case_id="depth-step",
        what="A box in front of a backing plane, with dropout hugging the depth edge.",
        why=(
            "The failure this whole thread exists to prevent. Dropout is worst exactly "
            "where the geometry is most discontinuous, so a completion is most tempted to "
            "interpolate precisely where interpolation is meaningless."
        ),
        notes=[
            "The edge-band mask removes the samples on both sides of the boundary, which is "
            "why the BASELINE fails the step-column gate: with its edge samples gone, the "
            "apparent step is displaced inward. That is a real defect and the one a "
            "completion is supposed to repair - not an accident of the fixture.",
            "'Bridging' counts reconstructed samples in the empty band between the two "
            "surfaces; 'misplaced' counts samples put on the wrong side of the step.",
        ],
        shapes=list(STEP_SHAPES),
        window=STEP_WINDOW,
        frames=[
            {
                "index": 0,
                "pose": _still(),
                "masks": [
                    LIGHT_SPECKLE,
                    {"kind": "edge_band", "foreground_below": FOREGROUND_BELOW, "width": 3},
                ],
            }
        ],
        regions=_regions(foreground_below=FOREGROUND_BELOW, dimensions=True),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_edge_gates(),
        *_shape_gates(baseline, dimensions=True, landmarks=False),
        *_local_recovery_gate(scene),
        *_confidence_gate(),
        *_work_gate(),
    ]
    errors = ["reconstructed_depth.p95", "reconstructed_depth.worst"]
    confidence = ["confidence.high_confidence_worst"]
    work = ["completion_work.within_budget"]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline, whose own edge is displaced by the missing samples",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "coverage.local_recovery",
                "edge_step_column.worst",
            ],
            inapplicable=errors + confidence + work,
        ),
        _candidate(
            "dense_nearest",
            "fills from the nearest sample, which lands on the wrong surface near the edge",
            "fail",
            failed=["reconstructed_depth.worst", "edge.misplaced"],
        ),
        _candidate(
            "broad_interpolation",
            "averages across the step, which is the flying-pixel geometry itself",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
                "edge.bridging",
                "edge.misplaced",
                "edge_step_column.worst",
            ],
        ),
        _candidate(
            "foreground_bias",
            "prefers the nearer surface among nearby samples, which walks the step outward",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "reconstructed_depth.worst",
                "edge.misplaced",
            ],
        ),
        _candidate(
            "spatial_completion",
            "the product primitive at a depth edge: it recovers the isolated speckle away "
            "from the boundary and refuses the whole edge band, which is one connected region "
            "spanning the frame. Nothing is bridged and nothing is misplaced, and the step "
            "the missing samples displaced is still displaced - the honest remaining failure, "
            "and one another view has to fix",
            "fail",
            failed=["coverage.supported_fraction", "edge_step_column.worst"],
        ),
        _candidate(
            "easy_samples_only",
            "fills only the easy holes and leaves the edge alone",
            "fail",
            failed=["coverage.supported_fraction", "edge_step_column.worst"],
            inapplicable=work,
        ),
        _candidate(
            "forged_measured_states",
            "reports every reconstructed sample as a camera measurement",
            "refused",
            inapplicable=sorted(str(gate["name"]) for gate in gates),
            refusal="measured_value_changed",
        ),
    ]
    return document


def _slot() -> dict[str, object]:
    document, scene = _case(
        case_id="narrow-slot",
        what="Two boxes with a three-column gap showing the backing plane, and the gap dropped.",
        why=(
            "A slot is the geometry a completion is most likely to destroy and least likely "
            "to be caught destroying: closing it makes the result denser, smoother and "
            "more plausible-looking, and the object is now the wrong shape."
        ),
        notes=[
            "The strip mask removes exactly the slot columns, so every candidate faces the "
            "same question: is there something between the two boxes, or not?",
            "Reconstructing the BACKGROUND inside the slot is allowed and is the right "
            "answer. What is refused is putting foreground depth there.",
        ],
        shapes=list(SLOT_SHAPES),
        window=SLOT_WINDOW,
        frames=[
            {
                "index": 0,
                "pose": _still(),
                "masks": [
                    LIGHT_SPECKLE,
                    {
                        "kind": "strip",
                        "first_column": SLOT_COLUMNS[0],
                        "last_column": SLOT_COLUMNS[-1],
                    },
                ],
            }
        ],
        regions=_regions(foreground_below=FOREGROUND_BELOW, slot_columns=list(SLOT_COLUMNS)),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_slot_gates(),
        *_local_recovery_gate(scene),
        *_confidence_gate(),
        *_work_gate(),
    ]
    errors = ["reconstructed_depth.p95", "reconstructed_depth.worst"]
    confidence = ["confidence.high_confidence_worst"]
    closed = [
        "slot.open",
        "slot.closed_samples",
        "reconstructed_depth.p95",
        "reconstructed_depth.worst",
        "confidence.high_confidence_worst",
    ]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline: the slot stays a hole rather than becoming a lie",
            "fail",
            failed=["coverage.supported_fraction", "coverage.local_recovery"],
            inapplicable=errors + confidence,
        ),
        _candidate(
            "dense_nearest",
            "fills the slot with the nearest box sample and closes it",
            "fail",
            failed=closed,
        ),
        _candidate(
            "broad_interpolation",
            "averages the two box faces across the slot and closes it",
            "fail",
            failed=closed,
        ),
        _candidate(
            "foreground_bias",
            "prefers the nearer surface, which is the box, and closes the slot",
            "fail",
            failed=closed,
        ),
        _candidate(
            "spatial_completion",
            "the product primitive beside a real gap: the two box faces either side of the "
            "slot are at the same depth, so nothing local distinguishes the gap from dropout "
            "on one surface. The slot is refused because it spans the whole frame, further "
            "than any support inside it reaches, and the isolated speckle on the boxes is "
            "recovered",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
        _candidate(
            "easy_samples_only",
            "has no close support inside the slot and leaves it open",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
    ]
    return document


def _multiview() -> dict[str, object]:
    fusion = {
        "minimum": [-3.5, -4.5, 95.0],
        "maximum": [3.5, 4.5, 103.0],
        "voxel_size": 0.25,
        "truncation": 0.75,
        "min_depth": 50.0,
        "max_depth": 200.0,
        "min_weight": 1.0,
        "coverage_tolerance": 0.5,
    }
    document, scene = _case(
        case_id="multiview-object",
        what="An asymmetric object from three exact poses, with complementary intermittent loss.",
        why=(
            "Temporal completion's whole claim is that another frame saw what this one "
            "missed. That is true only after the pose between them is applied, and this "
            "case is where carrying the same pixel across motion is proved to fail while "
            "carrying it through the exact poses does not."
        ),
        notes=[
            "The complementary masks drop a different third of the columns in each frame, "
            "so the union of the three frames sees more than any one of them - the "
            "opportunity a temporal method exists to take.",
            "One block of columns is dropped in EVERY frame. No temporal method can recover "
            "it, which is why the exact-pose control still misses the acceptance target and "
            "is declared to fail it rather than being called a success.",
            "The poses are the scene's own, exact. A real capture does not have them for "
            "free, so what this case measures is available evidence and not achievable "
            "performance.",
            "There are two tracking replays here and they are not the same evidence. The gated "
            "one - 'tracking' and 'tracking_pose' - offers the tracker each frame's own "
            "measured samples and nothing a completion added, which is the only route this "
            "project's tracker has; every candidate that preserved its measurements therefore "
            "reproduces the baseline's record exactly, and the strict reader refuses a report "
            "in which two candidates disagree about it.",
            "'tracking_completed_cloud' and 'tracking_completed_cloud_pose' are the "
            "counterfactual: the same tracker over the candidate's COMPLETED cloud. They carry "
            "no gate and are not an acceptance route. They are published because the answer is "
            "unfavourable and is the measured reason this task keeps completion output out of "
            "the tracker: dense in-frame fill, naive same-pixel carry and the real "
            "pose-compensated stage all move where that tracker places frame 2, the last of "
            "them by 1.09 to 2.38 degrees even though every one of its reconstructions is "
            "within a single raw code of the truth. "
            "tests/test_depth_completeness.py asserts that degradation directly.",
            "The weighted-fusion records carry this task's NEGATIVE result, and it is the "
            "measurement rather than an admission. Fusing the accepted temporal result through "
            "the provenance-weighted contract puts 225 more voxel observations into the volume "
            "than the raw baseline does - 5953 against 5728 - and produces the same 3099 "
            "observed voxels, the same 333 surface vertices and the same truth-referenced "
            "coverage of 0.9647630619684082. The gain is exactly zero on every output-facing "
            "measure. Withhold the supporting views and the same reconstructions look "
            "load-bearing: fusing the single frame carrying the most reconstructions on its own "
            "observes 1949 voxels raw and 2174 weighted. "
            "That is the whole finding - what the temporal stage recovers is what the support "
            "frames measured, so once those frames are themselves fused there is nothing left "
            "for it to add. No gain is gated here; the three '*_gain' fields are gated for "
            "non-inferiority and publish the zero.",
            "The worst-case error gate is deliberately absent here: reprojecting a real "
            "sample through exact poses rounds it across the silhouette by one pixel, and a "
            "worst-sample gate would refuse the honest control for a rounding artefact. The "
            "single-view cases gate the worst sample.",
        ],
        shapes=list(MULTIVIEW_SHAPES),
        window=MULTIVIEW_WINDOW,
        frames=[
            {
                "index": index,
                "pose": _turned(degrees, (0.0, 0.0, 100.0)),
                "masks": [
                    {"kind": "complementary", "period": 3, "phase": index},
                    {"kind": "never_return", "first_column": 2, "last_column": 3},
                ],
            }
            for index, degrees in enumerate((-4.0, 0.0, 4.0))
        ],
        regions=_regions(
            fusion=fusion,
            # The SAME volume, fused a second time through the accepted
            # provenance-weighted input contract. Same bounds, same voxels,
            # same truncation and the same uniform weight policy, so the two
            # records differ in exactly one thing: whether a sample's
            # provenance was allowed to change what it was worth.
            weighted_fusion=fusion,
            tracking=True,
            dimensions=True,
        ),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_false_fill_gate(baseline, fraction=MULTIVIEW_FALSE_FILL_FRACTION),
        *_error_gates(worst=False),
        *_shape_gates(baseline, dimensions=True, landmarks=False),
        *_fusion_gates(baseline),
        *_tracking_gates(baseline),
        *_temporal_recovery_gate(scene),
        *_weighted_fusion_gates(baseline),
        *_work_gate(),
        *_temporal_work_gate(),
    ]
    weighted = [
        "weighted_fusion.contribution",
        "weighted_fusion.matches_measured_only",
        "weighted_fusion.observed_voxels",
        "weighted_fusion.observed_voxel_gain",
        "weighted_fusion.spatially_excluded_contribution",
        "weighted_fusion.surface_coverage_gain",
        "weighted_fusion.surface_extracted",
        "weighted_fusion.surface_vertex_gain",
        "weighted_fusion_surface.observed_to_reference.p95",
    ]
    weighted = [name for name in weighted if any(gate["name"] == name for gate in gates)]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline over three frames of complementary loss. It is also the "
            "raw route into weighted fusion and the reference every weighted gate is stated "
            "against, so its weighted field is that reference and every weighted "
            "non-inferiority bar is met exactly. It still misses the coverage the corpus asks "
            "for, which is what this case is about",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "coverage.temporal_recovery",
            ],
            inapplicable=["reconstructed_depth.p95"],
        ),
        _candidate(
            "dense_nearest",
            "fills every hole in every frame from within that frame. The measured-only tracking "
            "record is the baseline's, because the tracker never sees what it invented; the "
            "counterfactual completed-cloud replay is where the invention moves the poses",
            "fail",
            failed=[
                "coverage.false_fill_fraction",
                "coverage.temporal_recovery",
                "reconstructed_depth.p95",
                "fusion_surface.observed_to_reference.p95",
            ],
            inapplicable=weighted,
        ),
        _candidate(
            "naive_temporal_carry",
            "carries the other frames' samples at the SAME pixel, ignoring the motion. It puts "
            "depth on the wrong surface, which the error and fused-surface gates catch, and in "
            "the counterfactual replay it degrades placement further than the dense in-frame "
            "fill does",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "reconstructed_depth.p95",
                "fusion_surface.observed_to_reference.p95",
            ],
            inapplicable=weighted,
        ),
        _candidate(
            "posed_temporal_carry",
            "carries the other frames' samples THROUGH the exact poses: accurate, and still "
            "short of the acceptance target because the never-return columns are in no frame. "
            "It carries no accepted product result, so it has no route into the weighted "
            "volume at all - which is the point: a self-declared provenance is not evidence",
            "fail",
            failed=["coverage.supported_fraction"],
            inapplicable=weighted,
        ),
        _candidate(
            "spatial_completion",
            "the product primitive on each frame independently. Every hole here is a whole "
            "column, dropped top to bottom, and no support inside one frame spans it - so it "
            "recovers nothing and returns each frame unchanged. That is the point of running "
            "it here: the tracking, pose, fused-surface, dimension and false-fill records are "
            "the baseline's exactly, so single-frame completion is shown to cost the "
            "multi-view pipeline nothing. Recovering what another view saw belongs to the "
            "temporal task and is not claimed here. Its weighted field is byte-identical to "
            "the baseline's for the same reason, so it reaches no more of the volume and "
            "changes no output measure at all",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "coverage.temporal_recovery",
            ],
            inapplicable=["reconstructed_depth.p95"],
        ),
        _candidate(
            "temporal_completion",
            "the product primitive on the three posed frames. It recovers exactly the samples "
            "another posed view really measured - every one of them within a single raw code "
            "of the truth - and invents nothing. Its measured-only tracking record is the "
            "baseline's exactly, which is the acceptance condition: the tracker consumes "
            "measured geometry only and this stage does not feed it. The counterfactual "
            "completed-cloud replay is where the cost of feeding it shows, and it is why "
            "temporal output enters no tracker in this task. It reaches a fused volume only "
            "through the provenance-weighted route beside it, at a weight derived from its own "
            "accepted confidence and never as a measurement",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
        _candidate(
            "spatial_then_temporal_completion",
            "the two product stages in order. Single-frame completion recovers nothing here - "
            "every hole is a whole column - so this is the temporal result with an empty "
            "spatial stage in front of it, and it behaves identically",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
        _candidate(
            "mutated_raw_frame",
            "returns raw bytes that are not the ones it was given",
            "refused",
            inapplicable=sorted(str(gate["name"]) for gate in gates),
            refusal="raw_frame_mutated",
        ),
        *_temporal_corruption_candidates(gates, confirming=False),
    ]
    return document


def _aperture_ambiguity() -> dict[str, object]:
    document, scene = _case(
        case_id="aperture-ambiguity",
        what=(
            "A single genuine aperture and a single held-out dropout on the same plane, "
            "placed so their scored input neighbourhoods are byte-identical. This is not a "
            "component-size tuning limit like the slot or the strip: a small true gap in a "
            "surface and a small sensor dropout on that same surface are, at this size and "
            "with no other evidence, the same input bytes. One frame cannot know which it is "
            "looking at, and this case exists to make that failure land on a gate instead of "
            "disappearing into 'component too long to close'."
        ),
        why=(
            "The other single-view cases refuse invention because the geometry they hide is "
            "bigger than any local support: a slot spanning the frame, a strip collapse wider "
            "than the scored neighbourhood. That is a real and useful bar, but it lets a "
            "candidate pass by being conservative at scale rather than by being right about "
            "ambiguity. Shrink the hidden geometry to the same size as an ordinary dropout and "
            "the excuse of span is gone: the aperture here sits inside one scored "
            "neighbourhood, exactly like the dropout six columns away, and nothing about the "
            "input distinguishes a hole in the sensor's return from a hole in the world. A "
            "deterministic single-frame method is not asked to resolve that - it structurally "
            "cannot - but it is asked to say which side of the ambiguity it lost, in gates a "
            "report cannot omit."
        ),
        notes=[
            "Four thin coplanar slabs, front faces all at z = 100.0, tile the window except "
            "one sample at column 29, row 12: genuine empty space, nothing behind it, truth "
            "code 0. That is the aperture.",
            "A blob mask of radius 0.5 at column 9, row 12 removes exactly its centre sample "
            "from the same plane - an ordinary isolated dropout, on real surface.",
            "No speckle mask is used in this case, unlike every other masked case in this "
            "corpus. Speckle would put additional zeros inside the two scored neighbourhoods "
            "and destroy the byte-identity the case depends on.",
            "The 9x9 window centred on the aperture and the 9x9 window centred on the dropout "
            "are byte-identical in the input frame: a single zero at the centre, code 12800 "
            "everywhere else. tests/test_depth_completeness.py asserts this directly rather "
            "than trusting the geometry comment.",
            "This is an epistemic boundary, not a size tuning knob. Recovering the dropout "
            "and false-filling the aperture are the same act performed on indistinguishable "
            "input; nothing here is evidence that a bigger neighbourhood radius would do "
            "better; a bigger radius only relocates where the identical pair can be built. "
            "A spatial-only estimate produced under this ambiguity must not be treated as "
            "tracking or fused geometry until independent temporal or physical evidence - a "
            "second view, a second frame, a return that disagrees - confirms it.",
        ],
        shapes=list(APERTURE_SHAPES),
        window=APERTURE_WINDOW,
        frames=[
            {
                "index": 0,
                "pose": _still(),
                "masks": [
                    {
                        "kind": "blob",
                        "column": APERTURE_DROPOUT_COLUMN,
                        "row": APERTURE_DROPOUT_ROW,
                        "radius": APERTURE_DROPOUT_RADIUS,
                    }
                ],
            }
        ],
        regions=_regions(weighted_fusion=APERTURE_FUSION),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_false_fill_gate(baseline, fraction=None),
        *_local_recovery_gate(scene),
        *_confidence_gate(),
        *_confident_false_fill_gate(),
        *_weighted_exclusion_gates(baseline),
        *_work_gate(),
    ]
    errors = ["reconstructed_depth.p95", "reconstructed_depth.worst"]
    confidence = ["confidence.high_confidence_worst"]
    confident_false_fill = ["confidence.high_confidence_false_fill"]
    work = ["completion_work.within_budget"]
    weighted = [
        "weighted_fusion.matches_measured_only",
        "weighted_fusion.observed_voxel_gain",
        "weighted_fusion.observed_voxels",
        "weighted_fusion.spatially_excluded_contribution",
    ]
    document["gates"] = gates
    document["candidates"] = [
        _candidate(
            "baseline",
            "the unchanged pipeline: the aperture and the dropout both stay missing, and "
            "the ambiguity between them never has to be resolved. One dropped sample out of "
            "960 still clears the acceptance fraction, so the only bar this fails is the "
            "strict local-recovery gain the scene itself owes",
            "fail",
            failed=["coverage.local_recovery"],
            inapplicable=errors + confidence + confident_false_fill + work,
        ),
        _candidate(
            "dense_nearest",
            "fills every hole with its nearest returned sample, which on this constant-depth "
            "plane recovers the dropout onto real surface and fills the aperture with the "
            "same confident, wrong answer: both holes get code 12800 at confidence 1.0",
            "fail",
            failed=["coverage.false_fill", "confidence.high_confidence_false_fill"],
            inapplicable=weighted,
        ),
        _candidate(
            "broad_interpolation",
            "averages the nearest measured sample in each direction, which does the same on "
            "both holes as dense_nearest does",
            "fail",
            failed=["coverage.false_fill", "confidence.high_confidence_false_fill"],
            inapplicable=weighted,
        ),
        _candidate(
            "foreground_bias",
            "there is no foreground here to prefer, so it behaves like dense_nearest on both holes",
            "fail",
            failed=["coverage.false_fill", "confidence.high_confidence_false_fill"],
            inapplicable=weighted,
        ),
        _candidate(
            "spatial_completion",
            "the product primitive's scored neighbourhood around the aperture is "
            "byte-identical to the one around the dropout - a single zero surrounded by a "
            "constant depth - so it cannot tell a small true gap from a small sensor "
            "dropout on the same plane. This is not a tuning limit on component size; it is "
            "what one frame cannot know. It reconstructs both onto the constant plane depth "
            "at confidence 0.73: the dropout correctly, and the aperture as a confident, "
            "invented surface where none exists",
            "fail",
            failed=["coverage.false_fill", "confidence.high_confidence_false_fill"],
        ),
        _candidate(
            "easy_samples_only",
            "both holes have four close neighbours, so it fills both the same way "
            "dense_nearest does",
            "fail",
            failed=["coverage.false_fill", "confidence.high_confidence_false_fill"],
            inapplicable=weighted,
        ),
    ]
    return document


#: How many of this scene's spatial estimates sit on real surface that at least
#: two other posed views measured. One: the held-out plane sample. Read off the
#: scene the way the slot columns and the aperture pixel are.
CONFIRMATION_OWED: Final = 1

#: Every candidate the two temporal cases run, with the exact expectation each
#: one has to meet. Written out rather than generated, because a corpus that
#: derives its own expectations from what a candidate did is not an oracle.
#: The four ways a temporal result can be dressed up, run on both temporal
#: cases. Each one changes exactly one fact about an otherwise valid product
#: result, and each must be refused rather than scored.
TEMPORAL_CORRUPTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "temporal_completion_forged_work",
        "reports a temporal run that refused one hole fewer than it did",
        "work_counts_inconsistent",
    ),
    (
        "temporal_completion_self_support",
        "cites the target frame's own measured sample as temporal evidence about itself",
        "temporal_support_not_another_frame",
    ),
    (
        "temporal_completion_unmeasured_support",
        "cites another frame's NO-RETURN sample as temporal evidence",
        "support_not_measured",
    ),
    (
        "temporal_completion_forged_confirmation",
        "calls a spatial estimate no independent evidence ever reached CONFIRMED, and leaves it "
        "spatial",
        "confirmation_inconsistent",
    ),
)


def _temporal_corruption_candidates(
    gates: Sequence[Mapping[str, object]], *, confirming: bool = True
) -> list[dict[str, object]]:
    """The temporal corruption controls, refused against every gate of a case.

    ``confirming`` says whether the case gives the single-frame stage anything
    to estimate. Where it does not, the forged-confirmation control has no
    spatial estimate to relabel and is left out rather than declared against a
    result it could not build.
    """
    every = sorted(str(gate["name"]) for gate in gates)
    return [
        _candidate(name, what, "refused", inapplicable=every, refusal=refusal)
        for name, what, refusal in TEMPORAL_CORRUPTIONS
        if confirming or name != "temporal_completion_forged_confirmation"
    ]


def _motion_candidates(gates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """What each candidate does to a scene whose camera actually moved.

    The pair this case exists for is ``naive_temporal_carry`` and the product's
    own ``temporal_completion``. Both are given the same three raw frames.
    The naive one carries other frames' samples at the SAME pixel and ends up
    denser than the product does - and puts backing-plane depth on the box's
    front face, which the edge, step-column and error gates all say out loud.
    The product carries the same raw samples through the exact poses, recovers
    fewer samples, and gets every one of them right.
    """
    empty = ["confidence.high_confidence_worst", "reconstructed_depth.p95"]
    return [
        _candidate(
            "baseline",
            "the unchanged pipeline: three strips and the speckle stay holes, and the box's "
            "step column cannot be located in frame 0 because its strip removed it",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "coverage.temporal_recovery",
                "edge_step_column.worst",
            ],
            inapplicable=[*empty, "reconstructed_depth.worst"],
        ),
        _candidate(
            "dense_nearest",
            "fills every hole from the nearest sample in the same frame, which puts box depth "
            "across the silhouette and settles none of the complementary evidence",
            "fail",
            failed=[
                "coverage.temporal_recovery",
                "edge.misplaced",
                "reconstructed_depth.worst",
            ],
        ),
        _candidate(
            "broad_interpolation",
            "averages across the strip, bridging the step into the empty band between the two "
            "surfaces",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "coverage.temporal_recovery",
                "edge.bridging",
                "edge.misplaced",
                "edge_step_column.worst",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
            ],
        ),
        _candidate(
            "naive_temporal_carry",
            "carries the other frames' samples at the SAME pixel. It is the densest result in "
            "this case and it is wrong about which surface a sample is on: at five pixels of "
            "motion the box silhouette has moved, so backing-plane codes land on box front "
            "face. Denser, and refused by the error, edge and step-column gates",
            "fail",
            failed=[
                "confidence.high_confidence_worst",
                "edge.misplaced",
                "edge_step_column.worst",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
            ],
        ),
        _candidate(
            "posed_temporal_carry",
            "carries the same raw samples THROUGH the exact poses, with no visibility rule and "
            "no agreement test. Accurate in the middle of a surface and still wrong at the "
            "worst sample, where a projection rounds across the silhouette - which is what the "
            "product primitive's footprint and edge rules exist to refuse",
            "fail",
            failed=["confidence.high_confidence_worst", "reconstructed_depth.worst"],
        ),
        _candidate(
            "spatial_completion",
            "the accepted single-frame primitive. It recovers isolated speckle and cannot "
            "cross a five-column strip, so it settles none of the complementary evidence and "
            "still cannot place the step column in frame 0",
            "fail",
            failed=[
                "coverage.supported_fraction",
                "coverage.temporal_recovery",
                "edge_step_column.worst",
            ],
        ),
        _candidate(
            "temporal_completion",
            "the product primitive on the raw frames and the exact poses. It recovers the "
            "complementary evidence the scene contains, restores the step column the strip "
            "removed, and passes every error, edge, dimension, confidence and work gate. It "
            "remains below the supported-coverage target because the speckle holes that are "
            "in every frame are in no other view either",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
        _candidate(
            "spatial_then_temporal_completion",
            "the two product stages in the order a pipeline would run them. Independent posed "
            "evidence confirms most of the single-frame estimates and never observes the rest, "
            "which keep their spatial provenance; the coverage is the union and every "
            "downstream gate still holds",
            "fail",
            failed=["coverage.supported_fraction"],
        ),
        *_corruption_candidates(gates),
        *_temporal_corruption_candidates(gates),
    ]


def _confirmation_candidates(gates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """What each candidate does to three spatial estimates that look identical.

    ``spatial_then_temporal_completion`` is the candidate this case is for. It
    is the only one offered an accepted spatial estimate to check, and it
    reaches all three verdicts: the plane sample is CONFIRMED, the pillar
    sample is CONTRADICTED and republished as the independent measurement, and
    the apertures are NOT OBSERVED and stay explicitly spatial - which is why
    it still fails the false-fill gates the single-frame primitive fails. That
    failure is the accepted aperture ambiguity being disclosed rather than
    laundered by a second stage.
    """
    empty = [
        "confidence.high_confidence_false_fill",
        "confidence.high_confidence_worst",
        "reconstructed_depth.p95",
        "reconstructed_depth.worst",
    ]
    false_fill = [
        "confidence.high_confidence_false_fill",
        "confidence.high_confidence_worst",
        "coverage.false_fill",
        "coverage.temporal_recovery",
        "reconstructed_depth.p95",
        "reconstructed_depth.worst",
    ]
    return [
        _candidate(
            "baseline",
            "the unchanged pipeline: the two held-out samples stay holes and nothing is "
            "invented in the aperture",
            "fail",
            failed=["coverage.temporal_recovery"],
            inapplicable=empty,
        ),
        _candidate(
            "dense_nearest",
            "fills every hole from the nearest sample in the same frame, aperture included",
            "fail",
            failed=false_fill,
        ),
        _candidate(
            "broad_interpolation",
            "averages the ring of plane samples around each hole, which closes the aperture "
            "and calls the pillar plane",
            "fail",
            failed=false_fill,
        ),
        _candidate(
            "naive_temporal_carry",
            "carries the other frames' samples at the same pixel. At one pixel of motion the "
            "same-pixel answer is close enough to settle the two real dropouts, and it also "
            "fills the aperture in every frame from the other frames' plane samples - which is "
            "the failure this corpus gates absolutely",
            "fail",
            failed=[
                "confidence.high_confidence_false_fill",
                "confidence.high_confidence_worst",
                "coverage.false_fill",
                "reconstructed_depth.p95",
                "reconstructed_depth.worst",
            ],
        ),
        _candidate(
            "posed_temporal_carry",
            "carries the same raw samples through the exact poses. No sample of any other "
            "frame lies on the aperture ray, so it recovers both dropouts and leaves the "
            "aperture empty. The positive control for what this scene makes available",
            "pass",
        ),
        _candidate(
            "spatial_completion",
            "the accepted single-frame primitive, which cannot distinguish the aperture from a "
            "dropout and cannot know the pillar is not the plane. Its exact accepted failure, "
            "carried into this case unchanged",
            "fail",
            failed=false_fill,
        ),
        _candidate(
            "temporal_completion",
            "the product primitive with no spatial estimate offered. It recovers both held-out "
            "samples from independent posed evidence, invents nothing in the aperture, and "
            "passes every gate this case states",
            "pass",
        ),
        _candidate(
            "spatial_then_temporal_completion",
            "the product primitive offered the accepted spatial estimate. One estimate is "
            "confirmed, one is contradicted and replaced by the independent measurement, and "
            "the three aperture estimates are never observed and stay spatial - so the "
            "single-frame false fill is still there, still gated, and still labelled spatial",
            "fail",
            failed=["confidence.high_confidence_false_fill", "coverage.false_fill"],
        ),
        *_corruption_candidates(gates),
        *_temporal_corruption_candidates(gates),
    ]


def _temporal_motion() -> dict[str, object]:
    document, scene = _case(
        case_id="temporal-motion",
        what=(
            "One box in front of a backing plane, seen from three laterally displaced "
            "cameras, with a complementary strip of columns missing from each."
        ),
        why=(
            "This is the case that separates 'use the other frame' from 'use the other "
            "frame THROUGH the pose between them'. The camera moves exactly five pixels "
            "at the box's own depth between frames, so the silhouette sits at column 26 in "
            "frame 0, 31 in frame 1 and 36 in frame 2. Frame 0's missing strip straddles "
            "its silhouette. Carrying the other frames' samples at the SAME pixel therefore "
            "lands backing-plane depth - 17920 raw codes - on samples whose truth is the "
            "box front face at 12288, which is a 5,632-code error on the wrong surface. "
            "Carrying the same raw samples through the exact poses lands them on the "
            "samples they belong to, exactly. Nothing here is an aggregate score: the two "
            "answers differ by which surface a sample is on."
        ),
        notes=[
            "The pose spec's translation moves the SCENE, so a frame's own world pose - "
            "the camera-to-model transform this benchmark scores with - is its inverse. "
            "The temporal transform from a source frame into the target camera is the "
            "target's inverse composed with the source's, and composing those the other "
            "way round is invisible on a stationary pair; here it moves every sample ten "
            "columns the wrong way.",
            "The three strips are disjoint in OBJECT space, not merely in image columns. "
            "A sample dropped in one frame is only recoverable from a frame that did not "
            "drop the same part of the box, and at a five-pixel motion the image column "
            "and the object column are five apart.",
            "Only frame 0's strip straddles a silhouette. The silhouette is one object "
            "feature, so three strips that each straddle it in their own frame would be "
            "the same object columns three times and nothing would be complementary. The "
            "edge measurements score frame 0, which is the frame that has the step.",
            "Speckle is applied to every frame as well, so some holes are isolated and "
            "recoverable from one frame's own evidence and some are not. The single-frame "
            "spatial primitive is run here for exactly that comparison.",
        ],
        shapes=list(STEP_SHAPES),
        window=MOTION_WINDOW,
        frames=[
            {
                "index": index,
                "pose": _shifted(index * MOTION_STEP_PIXELS * MOTION_PIXEL),
                "masks": [
                    LIGHT_SPECKLE,
                    {"kind": "strip", "first_column": first, "last_column": last},
                ],
            }
            for index, (first, last) in enumerate(MOTION_STRIPS)
        ],
        regions=_regions(foreground_below=FOREGROUND_BELOW, dimensions=True),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_edge_gates(),
        *_shape_gates(baseline, dimensions=True, landmarks=False),
        *_temporal_recovery_gate(scene),
        *_confidence_gate(),
        *_work_gate(),
        *_temporal_work_gate(),
    ]
    document["gates"] = gates
    document["candidates"] = _motion_candidates(gates)
    return document


def _temporal_confirmation() -> dict[str, object]:
    document, scene = _case(
        case_id="temporal-confirmation",
        what=(
            "One genuinely empty aperture, one held-out plane sample and one held-out "
            "sample of a pillar standing in front of the plane, from three posed views."
        ),
        why=(
            "A temporal stage has to do three different things to three spatial estimates "
            "that look identical from inside one frame, and this scene contains one of "
            "each. The plane sample was really the plane, two other views measured it, and "
            "the estimate is CONFIRMED. The pillar sample was really four units nearer than "
            "the plane its neighbours are on, so the single-frame estimate is confidently "
            "wrong and the two other views CONTRADICT it - and the contradiction is "
            "published rather than averaged, because a value halfway between 12288 and "
            "12800 is evidence of neither. The aperture is empty space, no other view "
            "measured anything on that ray, and the estimate is NOT OBSERVED: it stays "
            "explicitly spatial, and the single-frame ambiguity the corpus already records "
            "is neither erased nor blessed by having passed through a second stage."
        ),
        notes=[
            "The aperture is the same four-slab construction as the single-view ambiguity "
            "case: a one-sample gap with nothing behind it. It moves one column between "
            "views, and no sample of any other frame lies on the target's aperture ray, "
            "which is why no independent evidence for it exists to confirm it with.",
            "The pillar is one sample wide at the window's scale, so its held-out sample "
            "has a full ring of plane samples around it and no single-frame method can "
            "know it is not plane. That is what makes the contradiction load-bearing: the "
            "spatial estimate here is not careless, it is the best one frame supports.",
            "The two other views are displaced one pixel at the plane's depth, one each "
            "way. That is two genuinely distinct viewpoints under the primitive's own "
            "baseline rule, and small enough that the pillar - which is nearer, and so "
            "moves further per unit of camera shift - still lands within half a pixel of "
            "the sample it belongs to.",
            "Frames 1 and 2 carry no dropout mask. Their only holes are their own aperture "
            "samples, which have no truth behind them, so a candidate that fills them has "
            "invented geometry in protected empty space in every frame rather than in one.",
        ],
        shapes=list(CONFIRMATION_SHAPES),
        window=CONFIRMATION_WINDOW,
        frames=[
            {
                "index": 0,
                "pose": _shifted(0.0),
                "masks": [
                    {
                        "kind": "blob",
                        "column": CONFIRMATION_PILLAR_PIXEL[0],
                        "row": CONFIRMATION_PILLAR_PIXEL[1],
                        "radius": APERTURE_DROPOUT_RADIUS,
                    },
                    {
                        "kind": "blob",
                        "column": CONFIRMATION_PLANE_PIXEL[0],
                        "row": CONFIRMATION_PLANE_PIXEL[1],
                        "radius": APERTURE_DROPOUT_RADIUS,
                    },
                ],
            },
            {"index": 1, "pose": _shifted(-CONFIRMATION_SHIFT), "masks": []},
            {"index": 2, "pose": _shifted(CONFIRMATION_SHIFT), "masks": []},
        ],
        regions=_regions(),
    )
    baseline = _baseline_of(scene)
    gates = [
        *_coverage_gates(baseline),
        *_error_gates(),
        *_false_fill_gate(baseline, fraction=None),
        *_temporal_recovery_gate(scene),
        *_confirmation_gates(owed=CONFIRMATION_OWED),
        *_confidence_gate(),
        *_confident_false_fill_gate(),
        *_work_gate(),
        *_temporal_work_gate(),
    ]
    document["gates"] = gates
    document["candidates"] = _confirmation_candidates(gates)
    return document


# --- writing the corpus -------------------------------------------------------------------


def _mark_work_inapplicable(document: Mapping[str, object]) -> None:
    """Declare the work gate inapplicable for every candidate that reports no work.

    A rule rather than a hand-written list, because it is one: only a candidate
    that HAS a run reports what the run cost, and the controls in this corpus
    are hand-written fills with nothing to report. Written into the fixture all
    the same - a gate that quietly did not apply is exactly the shape of a
    candidate dodging one, and this corpus states which gates each candidate
    leaves unevaluated rather than letting the harness discover it.
    """
    gates = document["gates"]
    assert isinstance(gates, list)
    if not any(str(gate["name"]) == WORK_GATE for gate in gates):
        return
    candidates = document["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        expectation = candidate["expectation"]
        assert isinstance(expectation, dict)
        if expectation["outcome"] == "refused":
            continue
        if str(candidate["name"]).startswith("spatial_completion"):
            continue
        gaps = expectation["inapplicable_gates"]
        assert isinstance(gaps, list)
        if WORK_GATE not in gaps:
            expectation["inapplicable_gates"] = sorted([*gaps, WORK_GATE])


def _mark_temporal_gates_inapplicable(document: Mapping[str, object]) -> None:
    """Declare the temporal gates inapplicable for every candidate that runs no stage.

    The same rule as :func:`_mark_work_inapplicable` and for the same reason.
    Only a candidate that ran the pose-compensated stage reports what it cost,
    and only the one that was offered an accepted spatial estimate has a
    confirmation to be judged on. The hand-written fills in this corpus have
    neither, and the fixture says so rather than letting the harness discover
    it - a gate that quietly did not apply is exactly the shape of a candidate
    dodging one.
    """
    gates = document["gates"]
    assert isinstance(gates, list)
    present = {str(gate["name"]) for gate in gates}
    candidates = document["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        expectation = candidate["expectation"]
        assert isinstance(expectation, dict)
        if expectation["outcome"] == "refused":
            continue
        name = str(candidate["name"])
        unevaluated = set()
        if TEMPORAL_WORK_GATE in present and name not in TEMPORAL_CANDIDATES:
            unevaluated.add(TEMPORAL_WORK_GATE)
        if name != CONFIRMING_CANDIDATE:
            unevaluated |= present & set(CONFIRMATION_GATES)
        if not unevaluated:
            continue
        gaps = expectation["inapplicable_gates"]
        assert isinstance(gaps, list)
        expectation["inapplicable_gates"] = sorted(set(gaps) | unevaluated)


def build() -> dict[str, dict[str, object]]:
    """Every case, by the file name it is written to."""
    cases = [
        _fronto_complete(),
        _fronto_masked(),
        _oblique(),
        _curved(),
        _step(),
        _slot(),
        _multiview(),
        _aperture_ambiguity(),
        _temporal_motion(),
        _temporal_confirmation(),
    ]
    for case in cases:
        _mark_work_inapplicable(case)
        _mark_temporal_gates_inapplicable(case)
    return {f"{case['case_id']}.json": case for case in cases}


def serialise(document: object) -> str:
    """One deterministic rendering: sorted keys, strict JSON, trailing newline."""
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"


def manifest(cases: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": harness.CORPUS_KIND,
        "what": (
            "Deterministic synthetic depth-completeness scenes, dropout masks, gates and "
            "candidate expectations, read by tests/depth_completeness_harness.py."
        ),
        "why": (
            "So that a depth completion cannot be called an improvement for being denser. "
            "Each case states one bar and runs many candidates against it, including "
            "deliberately bad fills whose exact failures are the evidence that the bar is "
            "load-bearing."
        ),
        "units": (
            "Depth is raw code times depth scale, in the constructed scene's own units, and "
            "is exact by construction. Nothing here is a physical measurement and no case "
            "makes a dimensional-accuracy or scanner-performance claim."
        ),
        "generated_by": "tests/build_depth_completeness_fixtures.py",
        "files": [
            {
                "name": name,
                "case_id": case["case_id"],
                "bytes": len(serialise(case).encode("utf-8")),
                "sha256": hashlib.sha256(serialise(case).encode("utf-8")).hexdigest(),
            }
            for name, case in sorted(cases.items())
        ],
    }


def write(root: Path = CORPUS) -> list[Path]:
    """Write the corpus and its manifest, and return every path written."""
    cases = build()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, case in sorted(cases.items()):
        path = root / name
        path.write_text(serialise(case), encoding="utf-8")
        written.append(path)
    index = root / "manifest.json"
    index.write_text(serialise(manifest(cases)), encoding="utf-8")
    written.append(index)
    return written


def main() -> int:
    for path in write():
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - a developer entry point
    raise SystemExit(main())
