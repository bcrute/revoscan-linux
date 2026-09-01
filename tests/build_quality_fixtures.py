"""Regenerate the deterministic quality-harness fixture corpus.

The corpus itself - ``tests/fixtures/quality/cases/`` - is the evidence, and
:mod:`tests.quality_harness` reads it off disk and never calls anything here.
This module exists so that a reviewer can see exactly which geometry produced
those numbers and can reproduce them, rather than having to trust a wall of
committed coordinates.

It is deliberately NOT a test and is not collected by pytest: it writes files.
Run it explicitly, from the repository root::

    .venv/bin/python -m tests.build_quality_fixtures

and then read the diff. A run that changes a fixture is changing evidence, and
the diff is where that decision is visible.

Everything here is pure CPU arithmetic on committed constants - no scanner, no
capture, no network, no clock, no random source - so two runs of this module
produce byte-identical files. Coordinates are rounded to
:data:`COORDINATE_DECIMALS` before they are written, so the corpus is short
enough to read and the value a reader sees is exactly the value the harness
measures.

The readiness records are not hand-written. They are produced by running the
real :func:`scanner3d.reconstruction_readiness.evaluate_readiness` and
:func:`~scanner3d.reconstruction_readiness.surface_evidence` over synthetic
sweeps and freezing what they returned, because a hand-written record is a
record whose strict reader has never been proved to accept it.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

import scanner3d.quality as quality
from scanner3d.geometry import Point, Rigid, normalize, rotation_about_axis
from scanner3d.reconstruction_readiness import (
    DEFAULT_POLICY,
    REASONS,
    FusionEvidence,
    ReconstructionReadiness,
    evaluate_readiness,
    surface_evidence,
)

#: Where the corpus lives, relative to this file.
CORPUS = Path(__file__).parent / "fixtures" / "quality" / "cases"

#: Decimal places every written coordinate is rounded to.
#:
#: Six, because the tightest tolerance any case here states is 1e-3 of the
#: object's own size - three orders above the rounding - and because a
#: coordinate a reviewer cannot read is a coordinate nobody checks.
COORDINATE_DECIMALS = 6

#: The schema the corpus and every case file in it declare.
FIXTURE_SCHEMA_VERSION = 1

#: What every synthetic length in this corpus is in.
SYNTHETIC_UNIT = "synthetic units of the constructed truth, exact by construction"

#: Where the numbers in a synthetic case came from, written into every
#: truth-bearing measurement. Not an instrument, and it does not pretend to be.
SYNTHETIC_SOURCE = (
    "constructed exactly by tests/build_quality_fixtures.py; this is synthetic truth and makes "
    "no claim about any physical object"
)


def _rounded(point: Point) -> list[float]:
    return [round(value, COORDINATE_DECIMALS) for value in point]


# --- the synthetic geometry ---------------------------------------------------------


def lobed_surface(count: int = 120) -> list[Point]:
    """An ellipsoid with one off-axis lobe, sampled on a Fibonacci lattice.

    The same shape idiom ``tests/test_quality.py`` and ``tests/test_relocalize.py``
    use: the lobe is what makes one placement of this object better than every
    other, so a metric that trips on it is tripping on geometry rather than on
    an accidental symmetry.
    """
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * index
        x, z = math.cos(theta) * radius, math.sin(theta) * radius
        bulge = 1.0 + 0.35 * max(0.0, x * 0.6 + z * 0.8) ** 3
        points.append((x * 20.0 * bulge, y * 14.0 * bulge, z * 17.0 * bulge))
    return points


#: Side of the square panel every flatness case is measured on.
PANEL_SIDE = 60.0

#: Peak displacement of the deliberate ripple, in the same units - half a
#: percent of the panel it sits on, which is invisible on a rendered model.
LUMP_AMPLITUDE = 0.30

#: The panel is tilted off every coordinate axis so a plane fit has to do real
#: work rather than passing by quietly regressing z on x and y.
_PANEL_POSE = rotation_about_axis(normalize((0.21, 0.87, 0.44)), 0.7, through=(3.0, -2.0, 5.0))


def flat_panel(*, steps: int = 12, lump: float = 0.0) -> list[Point]:
    """A square planar patch, optionally given a gentle two-axis ripple.

    The ripple is applied along the panel's own normal BEFORE the panel is
    tilted into place, so ``lump`` is exactly the peak displacement from the
    plane. Both axes are sampled over exactly one whole period with the
    endpoint dropped, so the ripple has zero mean and zero correlation with
    either in-plane axis - which is what keeps the fitted plane where the
    clean panel's is instead of letting the fit tilt toward the bumps and
    report less roughness than there is.
    """
    points: list[Point] = []
    for row in range(steps):
        for column in range(steps):
            u = -PANEL_SIDE / 2.0 + PANEL_SIDE * row / steps
            v = -PANEL_SIDE / 2.0 + PANEL_SIDE * column / steps
            w = (
                lump
                * math.sin(2.0 * math.pi * row / steps)
                * math.cos(2.0 * math.pi * column / steps)
            )
            points.append(_PANEL_POSE.apply((u, v, w)))
    return points


#: The named lengths and landmarks a scale case is judged by, in the same
#: synthetic units. Truth first; the "measured" side of each case is derived
#: from it so the error is exactly the number the case name states.
TRUTH_DIMENSIONS: dict[str, float] = {
    "height": 28.0,
    "length": 40.0,
    "width": 34.0,
}
TRUTH_LANDMARKS: dict[str, Point] = {
    "lobe apex": (13.2, 0.0, 17.6),
    "north pole": (0.0, 14.0, 0.0),
    "south pole": (0.0, -14.0, 0.0),
}

#: The uniform scale error the scale case carries: two percent, which leaves
#: every rotation and every plane fit in the reconstruction untouched.
SCALE_ERROR = 1.02


# --- the readiness sweeps -----------------------------------------------------------

#: Radius of the synthetic object the readiness sweeps orbit, and the standoff
#: they orbit it at, both in the capture's depth units - the units
#: ``scanner3d.pipeline.NOISE_FLOOR`` is expressed in, so a policy stated as a
#: multiple of that noise floor means here what it means in a capture.
READINESS_RADIUS = 40.0
READINESS_DISTANCE = 150.0

#: Points in the world the readiness sweeps see, and how many of them a frame
#: keeps. Small enough that the whole corpus regenerates in under a second and
#: large enough that the coverage cell is judged on view directions rather
#: than on being thin.
READINESS_WORLD_POINTS = 4000
READINESS_FRAME_POINTS = 800
READINESS_FRAMES = 12

#: The fusion resolution the surface cases are judged at. A real export
#: derives these from the placed extent and the voxel budget; stating them
#: here keeps both surface fixtures judged at one known tolerance.
READINESS_VOXEL = 0.34
READINESS_TRUNCATION = 1.02


def readiness_sphere(count: int, radius: float = READINESS_RADIUS) -> list[Point]:
    """A deterministic near-uniform sampling of a sphere, in world coordinates."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points: list[Point] = []
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        angle = index * golden
        points.append(
            (radius * ring * math.cos(angle), radius * y, radius * ring * math.sin(angle))
        )
    return points


READINESS_WORLD = readiness_sphere(READINESS_WORLD_POINTS)


def readiness_camera(theta: float) -> Rigid:
    """The camera-to-world pose of a camera on the orbit at angle ``theta``."""
    rotation = rotation_about_axis((0.0, 1.0, 0.0), theta)
    return Rigid.of(rotation.rotation, rotation.apply((0.0, 0.0, -READINESS_DISTANCE)))


def readiness_seen_from(pose: Rigid) -> list[Point]:
    """The front-facing half of the sphere, in that camera's own coordinates.

    Front-facing rather than the whole shell, because a camera cannot see
    through an object: a fixture that pretended otherwise would give every
    cell the same observation count whatever the trajectory did, which is the
    distinction the coverage gate exists to make.
    """
    inverse = pose.inverse()
    eye = pose.translation
    visible = [
        point
        for point in READINESS_WORLD
        if sum(point[axis] * (eye[axis] - point[axis]) for axis in range(3)) > 0.0
    ]
    step = max(1, len(visible) // READINESS_FRAME_POINTS)
    return [inverse.apply(point) for point in visible[::step]]


def readiness_sweep(angles: Sequence[float]) -> tuple[list[list[Point]], list[Rigid]]:
    poses = [readiness_camera(angle) for angle in angles]
    return [readiness_seen_from(pose) for pose in poses], poses


def fusion_evidence_of(points: Sequence[Point]) -> FusionEvidence:
    """A ``FusionEvidence`` for a cloud whose frames are not the subject here.

    The frame counts are the SMALLEST a real sampler could have produced this
    many points under the default policy, because these records go through a
    strict reader that refuses a shape no evaluation could have emitted.
    """
    frames = max(1, -(-len(points) // DEFAULT_POLICY.fusion_pixels_per_frame))
    return FusionEvidence(
        frames_offered=frames,
        frames_sampled=frames,
        samples_examined=len(points),
        points=tuple(points),
    )


def _readiness(
    angles: Sequence[float],
    *,
    surface: Sequence[Point] | None = None,
) -> ReconstructionReadiness:
    """Run the real evaluator over a synthetic sweep and return its verdict.

    ``frames_dir`` is required by the evaluator and is not optional: the
    capture's own tracking authority is consulted on the folder. An empty
    temporary folder is a capture with no tracking-loss record in it, which is
    the one honest way to say "tracking is not what this fixture is about"
    without bypassing the gate.
    """
    clouds, poses = readiness_sweep(angles)
    with tempfile.TemporaryDirectory() as directory:
        readiness = evaluate_readiness(Path(directory), clouds, poses)
    if surface is None:
        return readiness
    agreement, failures = surface_evidence(
        surface,
        fusion_evidence_of(readiness_sphere(READINESS_WORLD_POINTS)),
        voxel_size=READINESS_VOXEL,
        truncation=READINESS_TRUNCATION,
    )
    return readiness.with_surface(agreement, failures)


def _orbit_angles(count: int = READINESS_FRAMES, span: float = 2.0 * math.pi) -> list[float]:
    return [index * span / count for index in range(count)]


# --- case construction --------------------------------------------------------------


def _cloud_case(
    *,
    case_id: str,
    what: str,
    why: str,
    observed: Sequence[Point],
    reference: Sequence[Point],
    coverage_tolerance: float,
    gates: list[dict[str, object]],
    outcome: str,
    failed_gates: Sequence[str],
    notes: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": "scanner3d.quality-harness-case",
        "case_id": case_id,
        "what": what,
        "why": why,
        "expectation": {"outcome": outcome, "failed_gates": sorted(failed_gates)},
        "notes": list(notes),
        "measurements": [
            {
                "kind": "cloud_agreement",
                "name": "cloud",
                "unit": SYNTHETIC_UNIT,
                "coverage_tolerance": coverage_tolerance,
                "observed": [_rounded(point) for point in observed],
                "reference": [_rounded(point) for point in reference],
            }
        ],
        "gates": gates,
    }


def _cloud_gates(
    *,
    forward_p95: float,
    forward_coverage: float,
    reverse_worst: float,
    reverse_coverage: float,
) -> list[dict[str, object]]:
    """The four gates every cloud case in this corpus states, in one order.

    Both directions, always, and coverage beside distance in each - see
    ``docs/quality-harness.md``: observed-to-reference is blind to a missing
    half and reference-to-observed is blind to geometry that should not exist,
    so a case stating only one of them passes a reconstruction the other
    fails. Stating all four in every cloud case is also what makes the
    expected failure SET load-bearing: a case that failed for the wrong reason
    would name different gates here.
    """
    return [
        {
            "name": "cloud.observed_to_reference.p95",
            "measurement": "cloud",
            "path": ["observed_to_reference", "distances", "p95"],
            "comparison": "<=",
            "limit": forward_p95,
            "unit": SYNTHETIC_UNIT,
            "context": (
                "reconstructed surface sits away from the truth it should lie on; look for a "
                "displaced or spurious region"
            ),
        },
        {
            "name": "cloud.observed_to_reference.coverage",
            "measurement": "cloud",
            "path": ["observed_to_reference", "coverage_fraction"],
            "comparison": ">=",
            "limit": forward_coverage,
            "unit": quality.FRACTION_UNIT,
            "context": (
                "part of the reconstruction has no truth near it at all; that geometry was invented"
            ),
        },
        {
            "name": "cloud.reference_to_observed.worst",
            "measurement": "cloud",
            "path": ["reference_to_observed", "distances", "worst"],
            "comparison": "<=",
            "limit": reverse_worst,
            "unit": SYNTHETIC_UNIT,
            "context": "some of the truth surface has no reconstruction near it; capture that side",
        },
        {
            "name": "cloud.reference_to_observed.coverage",
            "measurement": "cloud",
            "path": ["reference_to_observed", "coverage_fraction"],
            "comparison": ">=",
            "limit": reverse_coverage,
            "unit": quality.FRACTION_UNIT,
            "context": "the reconstruction is missing part of the object; capture the missing side",
        },
    ]


def _plane_gates(*, p95: float, worst: float) -> list[dict[str, object]]:
    return [
        {
            "name": "panel.residual.p95",
            "measurement": "panel",
            "path": ["residuals", "p95"],
            "comparison": "<=",
            "limit": p95,
            "unit": SYNTHETIC_UNIT,
            "context": (
                "a surface declared flat is coming out bumpy; check the fusion and the poses"
            ),
        },
        {
            "name": "panel.residual.worst",
            "measurement": "panel",
            "path": ["residuals", "worst"],
            "comparison": "<=",
            "limit": worst,
            "unit": SYNTHETIC_UNIT,
            "context": "one part of a flat panel is far off its plane; look for a local defect",
        },
    ]


def _readiness_gates(
    *,
    ready: bool,
    reasons: Sequence[str],
    absent: Sequence[str],
) -> list[dict[str, object]]:
    """``ready``, then every reason code this record must and must not carry.

    The absent codes are as load-bearing as the present ones. A record refused
    for the wrong reason is a different failure from the one the fixture was
    written for, and a case that only asserted "refused" would keep passing
    through it.
    """
    gates: list[dict[str, object]] = [
        {
            "name": "readiness.ready",
            "measurement": "readiness",
            "path": ["ready"],
            "comparison": "is",
            "limit": ready,
            "context": (
                "the real readiness reader's decision on this record is not the one the fixture "
                "records; the refusal rules or the record changed"
            ),
        }
    ]
    for reason in sorted(reasons):
        gates.append(
            {
                "name": f"readiness.reason.{reason}",
                "measurement": "readiness",
                "path": ["reason_present", reason],
                "comparison": "is",
                "limit": True,
                "context": f"this record must still be refused for {reason}",
            }
        )
    for reason in sorted(absent):
        gates.append(
            {
                "name": f"readiness.not_reason.{reason}",
                "measurement": "readiness",
                "path": ["reason_present", reason],
                "comparison": "is",
                "limit": False,
                "context": (
                    f"{reason} fired on a record it does not belong to; the refusal is coming "
                    "from somewhere other than the evidence this fixture is about"
                ),
            }
        )
    return gates


def _readiness_case(
    *,
    case_id: str,
    what: str,
    why: str,
    readiness: ReconstructionReadiness,
    absent: Sequence[str],
    notes: Sequence[str] = (),
) -> dict[str, object]:
    gates = _readiness_gates(ready=readiness.ready, reasons=readiness.reasons, absent=absent)
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": "scanner3d.quality-harness-case",
        "case_id": case_id,
        "what": what,
        "why": why,
        # Every readiness gate above is written from the verdict the real
        # evaluator produced, so the expected outcome is a pass: the fixture
        # asserts that today's reader still says what it said when the record
        # was frozen. The known-BAD half of these cases is the record itself.
        "expectation": {"outcome": "pass", "failed_gates": []},
        "notes": list(notes),
        "measurements": [
            {
                "kind": "readiness_record",
                "name": "readiness",
                "record": readiness.as_dict(),
            }
        ],
        "gates": gates,
    }


def build() -> dict[str, dict[str, object]]:
    """Every case in the corpus, by file name. Pure; writes nothing."""
    reference = lobed_surface()
    # A real but tiny reconstruction error: every point nudged along a fixed
    # direction by a thousandth of the object's own size. Small enough to
    # clear every gate, large enough that the clean case is not measuring
    # identity.
    nudge = 0.02
    clean = [(x + nudge, y - nudge, z + nudge) for x, y, z in reference]
    # The cap is removed by taking points from the SAME rounded list, so every
    # surviving point is byte-for-byte a point of the truth and the
    # observed-to-reference direction is exactly zero.
    kept = [point for point in reference if point[1] <= 6.0]
    spurious = [
        *reference,
        *[(x * 0.35 + 26.0, y * 0.35 + 22.0, z * 0.35) for x, y, z in reference],
    ]

    cases: dict[str, dict[str, object]] = {}

    cases["clean-surface.json"] = _cloud_case(
        case_id="clean-surface",
        what="A reconstruction of the lobed reference surface with a uniform 0.02-unit offset.",
        why=(
            "The negative control. A harness that only ever ran on broken input would report "
            "'all gates passed' for the rest of its life and nobody would find out until a scan "
            "was wrong."
        ),
        observed=clean,
        reference=reference,
        coverage_tolerance=0.25,
        gates=_cloud_gates(
            forward_p95=0.1,
            forward_coverage=0.999,
            reverse_worst=0.1,
            reverse_coverage=0.999,
        ),
        outcome="pass",
        failed_gates=(),
    )

    cases["clean-panel.json"] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": "scanner3d.quality-harness-case",
        "case_id": "clean-panel",
        "what": f"A flat {PANEL_SIDE:g}-unit square panel, tilted off every coordinate axis.",
        "why": (
            "The control for the roughness instrument. It is the same panel the lumpy case "
            "uses, without the ripple, so the two differ in exactly one thing."
        ),
        "expectation": {"outcome": "pass", "failed_gates": []},
        "notes": [
            "A plane fit produces no length and a shear maps planes to planes, so this case "
            "says the region is flat and says nothing about its size or placement.",
        ],
        "measurements": [
            {
                "kind": "plane_roughness",
                "name": "panel",
                "region": "the synthetic tilted reference panel",
                "unit": SYNTHETIC_UNIT,
                "points": [_rounded(point) for point in flat_panel()],
            }
        ],
        "gates": _plane_gates(p95=0.001, worst=0.001),
    }

    cases["missing-surface.json"] = _cloud_case(
        case_id="missing-surface",
        what="The reference surface with its whole upper cap absent.",
        why=(
            "Every surviving point is byte-for-byte a point of the truth, so "
            "observed-to-reference is exactly zero with coverage exactly 1.0. A harness that "
            "reported one direction would pass this."
        ),
        observed=kept,
        reference=reference,
        coverage_tolerance=0.25,
        gates=_cloud_gates(
            forward_p95=0.1,
            forward_coverage=0.999,
            reverse_worst=0.1,
            reverse_coverage=0.999,
        ),
        outcome="fail",
        failed_gates=(
            "cloud.reference_to_observed.worst",
            "cloud.reference_to_observed.coverage",
        ),
        notes=[
            "The two forward gates are expected to PASS here, and the expectation names the "
            "exact failing set, so this case cannot be satisfied by failing everything.",
        ],
    )

    cases["spurious-geometry.json"] = _cloud_case(
        case_id="spurious-geometry",
        what="The whole reference surface, plus a shrunken copy of it floating beside the object.",
        why=(
            "The mirror image of the missing-surface case: nothing of the truth is absent, so "
            "the reverse direction is perfect, and the invented lobe is visible only "
            "observed-to-reference."
        ),
        observed=spurious,
        reference=reference,
        coverage_tolerance=0.25,
        gates=_cloud_gates(
            forward_p95=0.1,
            forward_coverage=0.999,
            reverse_worst=0.1,
            reverse_coverage=0.999,
        ),
        outcome="fail",
        failed_gates=(
            "cloud.observed_to_reference.p95",
            "cloud.observed_to_reference.coverage",
        ),
        notes=[
            "The two reverse gates are expected to PASS, which is what distinguishes this "
            "failure from the missing-surface one.",
        ],
    )

    cases["lumpy-panel.json"] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": "scanner3d.quality-harness-case",
        "case_id": "lumpy-panel",
        "what": (
            f"The same panel with a {LUMP_AMPLITUDE:g}-unit two-axis ripple - half a percent of "
            f"the {PANEL_SIDE:g}-unit panel it sits on."
        ),
        "why": (
            "Half a percent is invisible on a rendered model and is diluted to nothing by any "
            "whole-cloud distance summary. A plane fit on a region the caller says is flat is "
            "what sees it."
        ),
        "expectation": {
            "outcome": "fail",
            "failed_gates": ["panel.residual.p95", "panel.residual.worst"],
        },
        "notes": [],
        "measurements": [
            {
                "kind": "plane_roughness",
                "name": "panel",
                "region": "the synthetic tilted reference panel",
                "unit": SYNTHETIC_UNIT,
                "points": [_rounded(point) for point in flat_panel(lump=LUMP_AMPLITUDE)],
            }
        ],
        "gates": _plane_gates(p95=0.001, worst=0.001),
    }

    cases["uniform-scale.json"] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": "scanner3d.quality-harness-case",
        "case_id": "uniform-scale",
        "what": (
            f"Named lengths and landmarks of a reconstruction {(SCALE_ERROR - 1.0) * 100:g}% "
            "larger than the truth."
        ),
        "why": (
            "A uniform scale is invisible to held-out closure, to plane roughness and to every "
            "other check computed inside the reconstruction's own coordinate system, because a "
            "larger world is perfectly self-consistent in it. Sparse named truth from outside "
            "that system is what catches it."
        ),
        "expectation": {
            "outcome": "fail",
            "failed_gates": ["dimensions.absolute.worst", "landmarks.offsets.worst"],
        },
        "notes": [
            "This case deliberately runs no cloud_agreement. A dense reference surface in the "
            "same frame would see the scale too; the point being made is that the sparse named "
            "check catches it when no such surface exists, which is the situation this "
            "repository is actually in.",
        ],
        "measurements": [
            {
                "kind": "dimension_errors",
                "name": "dimensions",
                "unit": SYNTHETIC_UNIT,
                "truth_source": SYNTHETIC_SOURCE,
                "measured": {
                    name: round(value * SCALE_ERROR, COORDINATE_DECIMALS)
                    for name, value in TRUTH_DIMENSIONS.items()
                },
                "truth": dict(TRUTH_DIMENSIONS),
            },
            {
                "kind": "landmark_offsets",
                "name": "landmarks",
                "unit": SYNTHETIC_UNIT,
                "truth_source": SYNTHETIC_SOURCE,
                "measured": {
                    name: [round(value * SCALE_ERROR, COORDINATE_DECIMALS) for value in point]
                    for name, point in TRUTH_LANDMARKS.items()
                },
                "truth": {name: _rounded(point) for name, point in TRUTH_LANDMARKS.items()},
            },
        ],
        "gates": [
            {
                "name": "dimensions.absolute.worst",
                "measurement": "dimensions",
                "path": ["absolute", "worst"],
                "comparison": "<=",
                "limit": 0.1,
                "unit": SYNTHETIC_UNIT,
                "context": "a named length is wrong; suspect the depth scale before the geometry",
            },
            {
                "name": "dimensions.relative.worst",
                "measurement": "dimensions",
                "path": ["relative", "worst"],
                "comparison": "<=",
                "limit": 0.05,
                "unit": quality.FRACTION_UNIT,
                "context": (
                    "stated beside the absolute gate on purpose: at two percent this one still "
                    "passes, so a case cannot be satisfied by failing every gate it names"
                ),
            },
            {
                "name": "landmarks.offsets.worst",
                "measurement": "landmarks",
                "path": ["offsets", "worst"],
                "comparison": "<=",
                "limit": 0.1,
                "unit": SYNTHETIC_UNIT,
                "context": "a named feature is not where it belongs; check scale and placement",
            },
        ],
    }

    # -- the readiness boundary ------------------------------------------------------

    every_reason = tuple(sorted(REASONS))
    valid = _readiness(_orbit_angles(), surface=readiness_sphere(6000))
    cases["readiness-valid-orbit.json"] = _readiness_case(
        case_id="readiness-valid-orbit",
        what="A full twelve-view orbit whose extracted surface sits on its own fused evidence.",
        why=(
            "The control for the readiness boundary: a record the real reader accepts, so a "
            "refusal that started firing on everything would be caught here rather than being "
            "mistaken for strictness."
        ),
        readiness=valid,
        absent=every_reason,
        notes=[
            "Produced by running the real evaluate_readiness() and surface_evidence() and "
            "freezing the record they returned.",
        ],
    )

    stationary = _readiness([0.0] * READINESS_FRAMES)
    cases["readiness-poses-stationary.json"] = _readiness_case(
        case_id="readiness-poses-stationary",
        what="Twelve identical poses: a sequence that never moved.",
        why=(
            "Known-invalid pose evidence. Every pose gate must fire, by its own reason code, "
            "and the record must still be readable so the refusal names all of them."
        ),
        readiness=stationary,
        absent=tuple(reason for reason in every_reason if reason not in set(stationary.reasons)),
    )

    narrow = _readiness([index * 0.02 for index in range(READINESS_FRAMES)])
    cases["readiness-coverage-one-sided.json"] = _readiness_case(
        case_id="readiness-coverage-one-sided",
        what="Twelve views spread over a fifth of a radian: real motion, one side of the object.",
        why=(
            "Known-invalid occupied-coverage evidence, and deliberately a case that passes the "
            "pose gates: coverage_unsupported has to be able to fire alone."
        ),
        readiness=narrow,
        absent=tuple(reason for reason in every_reason if reason not in set(narrow.reasons)),
    )

    displaced = _readiness(
        _orbit_angles(),
        surface=[(x + 3.0, y, z) for x, y, z in readiness_sphere(6000)],
    )
    cases["readiness-surface-displaced.json"] = _readiness_case(
        case_id="readiness-surface-displaced",
        what="The accepted orbit above, with an extracted surface displaced three units.",
        why=(
            "Known-invalid surface evidence. The pre-fusion gates all pass, so this proves the "
            "post-fusion gate is what refused it."
        ),
        readiness=displaced,
        absent=tuple(reason for reason in every_reason if reason not in set(displaced.reasons)),
    )
    return cases


def serialise(document: object) -> str:
    """One deterministic rendering: sorted keys, strict JSON, trailing newline."""
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"


def manifest(cases: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "kind": "scanner3d.quality-harness-corpus",
        "what": (
            "Deterministic synthetic reconstruction-quality cases and frozen readiness records, "
            "read by tests/quality_harness.py."
        ),
        "why": (
            "Known geometry, portable as data rather than as Python helpers, so a change that "
            "makes a reconstruction worse fails a named gate in a named case instead of being "
            "noticed on a scan."
        ),
        "units": (
            "Every synthetic length is in the constructed truth's own units and is exact by "
            "construction. Nothing here is a physical measurement and no case makes a "
            "dimensional-accuracy claim. The readiness records are in the capture's depth units."
        ),
        "generated_by": "tests/build_quality_fixtures.py",
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
