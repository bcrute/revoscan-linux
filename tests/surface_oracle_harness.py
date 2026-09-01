"""Run the raw-points versus raw-TSDF-surface corpus, and publish what it found.

One command::

    .venv/bin/python -m tests.surface_oracle_harness --out surface-oracle-report.json

Exit codes, and nothing wraps them:

* **0** - every candidate of every case did exactly what its fixture declares,
  and the cross-case attribution held.
* **1** - the corpus ran and something did not. An unexpected pass, a missing
  expected failure, a corruption control that was not refused, a malformed
  fixture, an attribution that does not reconcile, or an exception raised while
  scoring.
* **2** - the corpus could not be run at all: no corpus, a manifest that is not
  a manifest, a case file that is not where the manifest says, a case whose
  control is not its control, a perturbation nobody wrote a case for, a grid
  cell a case did not run, or an output path a report may not be written to.

Cases are independent and every one is attempted, so a failing artifact
preserves all of them rather than the first.

Why a third command
-------------------
:mod:`tests.quality_harness` measures reviewable geometry declared as JSON, and
:mod:`tests.depth_completeness_harness` measures what a *completion* did to
per-sample provenance. Neither can answer this thread's question, which is
about two whole reconstructions of the same frames: the measured dots and the
fused surface. That comparison needs a case to run the same scene through many
predeclared parameter cells and to be compared with a DIFFERENT case - its
exact-truth control - which is a shape neither existing reader has a schema
for. So this is a third corpus with its own manifest, its own artifact and its
own strict reader, built out of the other two harnesses' primitives rather than
beside a copy of them: the path safety, the strict JSON, the gate factories and
the publication path are all imported from :mod:`tests.quality_harness`, and
the window, the frame construction and the deprojection are imported from
:mod:`tests.depth_completeness`.

It is deliberately NOT added to `.github/workflows/ci.yml`. The active task
forbids touching CI, and the two existing named oracle jobs keep running
exactly as they did; this command is run by hand and its artifact is the
evidence a handoff carries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import scanner3d.quality as quality
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.simulate import Box as SimBox
from scanner3d.simulate import Plane, Shape, Sphere, intrinsics_of
from scanner3d.simulate import render as render_frame
from tests import depth_completeness as completeness
from tests import surface_oracle as contract
from tests.quality_harness import (
    FixtureUnusable,
    HarnessRefused,
    ReportUnusable,
    clear_destination,
    publish,
)
from tests.quality_harness import (
    _as_number as as_number,
)

CORPUS: Final = Path(__file__).parent / "fixtures" / "surface_oracle" / "cases"
MANIFEST_NAME: Final = "manifest.json"

FIXTURE_SCHEMA_VERSION: Final = 1
CORPUS_KIND: Final = "scanner3d.surface-oracle-corpus"
CASE_KIND: Final = "scanner3d.surface-oracle-case"

REPORT_SCHEMA_VERSION: Final = 1
REPORT_KIND: Final = "scanner3d.surface-oracle-report"

FIXTURE_NAME: Final = re.compile(r"^[a-z0-9][a-z0-9-]*\.json$")
PATH_STEP: Final = re.compile(r"^[a-z][a-z0-9_]*$")
CELL_ID: Final = re.compile(r"^[a-z0-9][a-z0-9-]*$")
CANDIDATE_NAME: Final = re.compile(r"^([a-z][a-z0-9_]*)(?::([a-z0-9][a-z0-9-]*))?$")

OUTCOMES: Final[tuple[str, ...]] = ("pass", "fail", "refused")
STATUSES: Final[tuple[str, ...]] = ("pass", "fail", "malformed")

#: The one arm bound to the resolution the shipped export policy chooses. Named
#: here because three different places have to agree on it: the attribution,
#: the headline comparison and the fixture check that every case runs it.
APPLIANCE_ARM: Final = "fused_at_appliance_resolution"

COMPARISONS: Final[Mapping[str, quality.Comparison]] = {
    "<=": quality.Comparison.AT_MOST,
    ">=": quality.Comparison.AT_LEAST,
    "is": quality.Comparison.IS,
}

#: Kinds owned by scanner3d.quality, and the seven this oracle adds.
QUALITY_KINDS: Final[tuple[str, ...]] = (
    "cloud_agreement",
    "plane_roughness",
    "dimension_errors",
)
ORACLE_KINDS: Final[tuple[str, ...]] = (
    "surface_oracle_geometry",
    "surface_oracle_edge",
    "surface_oracle_components",
    "surface_oracle_field",
    "surface_oracle_analytic",
    "surface_oracle_ablation",
    "surface_oracle_surface_samples",
    "surface_oracle_topology",
)
MEASUREMENT_KINDS: Final[tuple[str, ...]] = (*QUALITY_KINDS, *ORACLE_KINDS)

#: How the recommended cell of a case is chosen. Stated HERE, in code, and
#: repeated verbatim in the manifest, which :func:`load_corpus` checks: the
#: corpus records the rule it was reviewed with, and the rule cannot be
#: rewritten by editing a fixture after the results are in.
#:
#: It ranks on the ANALYTIC coverage - how much of a dense exact sampling of
#: the object itself the arm reproduced - rather than on proximity to the
#: sensor's own returns. Round two of this corpus ranked on the latter and
#: reported a reference artefact as an extraction defect; a rule that scores a
#: surface against the rays that measured it cannot tell a vertex correctly
#: interpolated between two rays from one that is wrong.
SELECTION_RULE: Final = (
    "the recommended cell is the one with the highest analytic.reference_coverage_fraction over "
    "the dense exact sampling of the scene's own shapes. Ties break on the lower "
    "analytic.distances.p95, then the lower field.work_units, then the lexicographically first "
    "cell_id. It is applied to EVERY declared cell after all of them have run; a case that did "
    "not run a declared cell is refused before any cell is chosen. The derived "
    "shipped-policy arm is reported beside the winner and is never eligible to be it, because "
    "its resolution is not a choice the grid offers."
)

#: Every metric the cross-case attribution follows, as (measurement, path).
#: Fixed here and published in the artifact, so a perturbation cannot be
#: attributed on whichever number happened to move.
ATTRIBUTED_METRICS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("analytic", ("distances", "p95")),
    ("analytic", ("distances", "worst")),
    ("analytic", ("within_tolerance_fraction",)),
    ("analytic", ("reference_coverage_fraction",)),
    ("support", ("reference_to_observed", "coverage_fraction")),
    ("support", ("observed_to_reference", "coverage_fraction")),
    ("support", ("observed_to_reference", "distances", "p95")),
    ("plane", ("residuals", "p95")),
    ("edge", ("bridged_points",)),
    ("edge", ("edge_coverage",)),
    ("edge", ("thin_feature_coverage",)),
    ("dimensions", ("absolute", "worst")),
    ("components", ("component_count",)),
    ("components", ("isolated_components",)),
    ("components", ("largest_component_fraction",)),
    ("geometry", ("point_count",)),
    ("surface_samples", ("distances", "p95")),
    ("surface_samples", ("within_tolerance_fraction",)),
    ("surface_samples", ("bridging_faces",)),
    ("surface_samples", ("bridging_faces_with_every_vertex_outside",)),
    ("surface_samples", ("unsupported_fraction",)),
    ("surface_samples", ("protected_area",)),
    ("topology", ("shell_count",)),
    ("topology", ("unsupported_closed_shells",)),
    ("topology", ("unsupported_inaccurate_closed_shells",)),
    ("topology", ("nonmanifold_edge_count",)),
    ("topology", ("boundary_edge_count",)),
    ("ablation", ("extraction_p95_delta",)),
    ("ablation", ("surface_points_without_field_support_fraction",)),
)

#: Which direction is worse, per attributed metric. Declared before any result
#: exists, so "this perturbation degraded the surface" is a reading of the
#: metric's own meaning rather than of the sign that suited the conclusion.
WORSE_WHEN_HIGHER: Final[frozenset[str]] = frozenset(
    {
        "analytic.distances.p95",
        "analytic.distances.worst",
        "support.observed_to_reference.distances.p95",
        "plane.residuals.p95",
        "edge.bridged_points",
        "components.component_count",
        "components.isolated_components",
        "dimensions.absolute.worst",
        "surface_samples.distances.p95",
        "surface_samples.bridging_faces",
        "surface_samples.bridging_faces_with_every_vertex_outside",
        "surface_samples.unsupported_fraction",
        "surface_samples.protected_area",
        "topology.shell_count",
        "topology.unsupported_closed_shells",
        "topology.unsupported_inaccurate_closed_shells",
        "topology.nonmanifold_edge_count",
        "topology.boundary_edge_count",
        "ablation.extraction_p95_delta",
        "ablation.surface_points_without_field_support_fraction",
    }
)
WORSE_WHEN_LOWER: Final[frozenset[str]] = frozenset(
    {
        "analytic.within_tolerance_fraction",
        "analytic.reference_coverage_fraction",
        "support.observed_to_reference.coverage_fraction",
        "support.reference_to_observed.coverage_fraction",
        "edge.edge_coverage",
        "edge.thin_feature_coverage",
        "components.largest_component_fraction",
        "geometry.point_count",
        "surface_samples.within_tolerance_fraction",
    }
)

#: Every refusal the corruption controls exist to earn, somewhere in the
#: corpus. Not every control is detectable on every case - fusing the truth
#: frames is indistinguishable from fusing the measured ones when the case has
#: no depth perturbation, and substituting the true poses is indistinguishable
#: when the case has no pose perturbation - so the requirement is corpus-wide
#: rather than per-case. :func:`_check_controls` refuses a corpus that stopped
#: expecting one of them, which is what a quietly deleted bad control looks
#: like from the outside.
CONTROL_REFUSALS: Final[frozenset[str]] = frozenset(
    {
        "raw_frame_edited",
        "views_omitted",
        "pose_substituted",
        "surface_not_derived",
        "sample_count_mismatch",
        "claim_not_supported",
        "claim_unknown",
    }
)

LIMITATIONS: Final[tuple[str, ...]] = (
    "Synthetic truth. Every scene is rendered by scanner3d.simulate, which is not a sensor "
    "model: no speckle physics, multipath, subsurface scattering or baseline shadowing. A cause "
    "isolated here is a cause of the arithmetic on exact truth, not a measurement of the device.",
    "The perturbations are constructed, not fitted to a capture. Their magnitudes are stated in "
    "raw codes and in the scene's own units and are not claims about what the MINI does.",
    "Depth is in synthetic units of raw code times depth scale. No physical length, and no "
    "dimensional-accuracy claim, is made anywhere in this artifact.",
    "Pose perturbation moves only the pose a reconstruction is GIVEN. The frames stay exact, so "
    "this isolates pose error from the depth error a real tracker would also be reacting to.",
    "Both arms are clipped to the case's fixed volume, which is the observed space they share. "
    "Neither arm is credited for geometry outside it.",
    "The truth model and every arm are decimated onto the same fixed lattice before the "
    "symmetric agreement, so that comparison is about where the surface is and not about how "
    "many points carry it. Point counts are published and gated separately.",
    "Nothing here is routed into the product. No export, live preview, GUI, capture, tracker or "
    "readiness policy reads this module, and task 032's finding that completed samples add no "
    "fused coverage is preserved: no completion of any kind is offered to a volume here.",
    "Two references, and they are not interchangeable. The analytic measurement is the exact "
    "closed-form distance to the object; the support measurement is proximity to the ideal "
    "returns the five views produced. A point on a face no view could see is accurate and "
    "unsupported at the same time, and neither number is called the other's name anywhere.",
    "The shipped-policy arm runs on the numpy backend because the resolution it derives needs "
    "about thirty times the pure-Python voxel ceiling. Its last bits are therefore "
    "machine-dependent in a way the rest of this corpus is not. Every declared grid cell runs "
    "on the pure-Python reference.",
    "The triangle interiors are sampled on a fixed barycentric lattice at the order the fixture "
    "owns, under a sample ceiling the fixture owns. A defect entirely between two adjacent "
    "samples of one triangle is not seen; the order is stated in the artifact so a reader can "
    "judge how fine that is against the voxel size beside it.",
    "The topology evidence counts what the faces are, not whether the object should have been "
    "closed. A closed shell is not itself a defect, and neither is an unsupported one: a sealed "
    "shell on a face no view could see is unsupported measurement and correct geometry at the "
    "same time. Only a shell that is unsupported AND off the object is gated, and the "
    "unsupported-but-accurate shells are published beside it as the measured cost any "
    "support-only pruning rule would pay.",
    "The accurate/inaccurate split of an unsupported shell is made at the fixture's coverage "
    "tolerance over the shell's vertices and its sampled triangle interiors. A shell whose "
    "error is entirely between two adjacent interior samples and away from every corner would "
    "be read as accurate; the sample order and the tolerance are both published so a reader can "
    "judge that against the voxel size beside them.",
)


# --- strict reading helpers, in the accepted harnesses' vocabulary ---------------------


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FixtureUnusable(f"{what} is {type(value).__name__}, not an object")
    for key in value:
        if not isinstance(key, str):
            raise FixtureUnusable(f"{what} has a non-string key {key!r}")
    return value


def _exact_keys(record: Mapping[str, object], keys: Sequence[str], what: str) -> None:
    present = set(record)
    expected = set(keys)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        raise FixtureUnusable(
            f"{what} must state exactly {sorted(expected)}; missing {missing}, unexpected {extra}"
        )


def _text(record: Mapping[str, object], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, not non-empty text")
    return value


def _bool(record: Mapping[str, object], key: str, what: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, not a boolean")
    return value


def _number(record: Mapping[str, object], key: str, what: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, not a number")
    if not math.isfinite(float(value)):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, which is not finite")
    return float(value)


def _integer(record: Mapping[str, object], key: str, what: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, not an integer")
    return value


def _strings(record: Mapping[str, object], key: str, what: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise FixtureUnusable(f"{what} {key!r} is not a list of strings")
    return tuple(str(item) for item in value)


def _point(record: Mapping[str, object], key: str, what: str) -> Point:
    value = record.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise FixtureUnusable(f"{what} {key!r} is not a three-number point")
    out: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise FixtureUnusable(f"{what} {key!r}[{index}] is {item!r}, not a number")
        if not math.isfinite(float(item)):
            raise FixtureUnusable(f"{what} {key!r}[{index}] is not finite")
        out.append(float(item))
    return (out[0], out[1], out[2])


def _digest_text(record: Mapping[str, object], key: str, what: str) -> str:
    value = _text(record, key, what)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, not a lower-case sha256 digest")
    return value


def _loads(text: str, what: str, *, error: type[ScannerError] = FixtureUnusable) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise error(f"{what} is not JSON: {exc}") from exc


# --- building a scene from its fixture -------------------------------------------------

#: Rendering the same shapes at the same pose twice is the same frame, and the
#: whole corpus shares one object and one set of views. The cache makes the run
#: affordable and changes nothing about the answer: it is keyed on the exact
#: text of the shapes, the pose and the intrinsics.
_RENDER_CACHE: dict[str, bytes] = {}


def _shape(spec: Mapping[str, object], what: str) -> Shape:
    """One shape, built through :mod:`scanner3d.simulate` and never by hand."""
    kind = _text(spec, "kind", what)
    if kind == "plane":
        _exact_keys(spec, ("kind", "normal", "offset"), what)
        return Plane(normal=_point(spec, "normal", what), offset=_number(spec, "offset", what))
    if kind == "sphere":
        _exact_keys(spec, ("kind", "centre", "radius"), what)
        return Sphere(centre=_point(spec, "centre", what), radius=_number(spec, "radius", what))
    if kind == "box":
        _exact_keys(spec, ("kind", "half_extents", "translation"), what)
        return SimBox(
            half_extents=_point(spec, "half_extents", what),
            pose=Rigid.of(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                _point(spec, "translation", what),
            ),
        )
    raise FixtureUnusable(f"{what} states shape kind {kind!r}, which this harness cannot build")


def _render_pose(spec: Mapping[str, object], what: str) -> Rigid:
    """A view stated as an exact axis, angle and centre of rotation.

    Constructive rather than nine written-out numbers, exactly as the
    depth-completeness corpus states its poses: the pose behind a frame is then
    reproducible to the last bit from something a reviewer can read.
    """
    _exact_keys(spec, ("axis", "degrees", "through"), what)
    return rotation_about_axis(
        _point(spec, "axis", what),
        math.radians(_number(spec, "degrees", what)),
        through=_point(spec, "through", what),
    )


def _intrinsics(spec: Mapping[str, object], what: str) -> IntrinsicsRecord:
    _exact_keys(spec, ("fx", "fy", "cx", "cy", "width", "height"), what)
    return intrinsics_of(
        fx=_number(spec, "fx", what),
        fy=_number(spec, "fy", what),
        cx=_number(spec, "cx", what),
        cy=_number(spec, "cy", what),
        width=_integer(spec, "width", what),
        height=_integer(spec, "height", what),
    )


def _window(spec: Mapping[str, object], what: str) -> completeness.Window:
    _exact_keys(spec, ("u0", "v0", "width", "height", "sample_count"), what)
    window = completeness.Window(
        u0=_integer(spec, "u0", what),
        v0=_integer(spec, "v0", what),
        width=_integer(spec, "width", what),
        height=_integer(spec, "height", what),
    )
    stated = _integer(spec, "sample_count", what)
    if stated != window.sample_count:
        raise FixtureUnusable(
            f"{what} states {stated} sample(s) for a window that holds {window.sample_count}"
        )
    return window


def _box(spec: object, what: str) -> contract.Box:
    record = _mapping(spec, what)
    _exact_keys(record, ("minimum", "maximum"), what)
    minimum = _point(record, "minimum", what)
    maximum = _point(record, "maximum", what)
    if any(minimum[axis] >= maximum[axis] for axis in range(3)):
        raise FixtureUnusable(f"{what} is not a strictly ordered box: {minimum} to {maximum}")
    return contract.Box(minimum=minimum, maximum=maximum)


def _regions(spec: Mapping[str, object], what: str) -> contract.Regions:
    _exact_keys(
        spec,
        (
            "plane",
            "protected",
            "edge_band",
            "thin_feature",
            "object_region",
            "coverage_tolerance",
            "truth_pitch",
            "analytic_supersample",
            "triangle_sample_order",
            "max_triangle_samples",
            "linkage_radius",
            "isolated_max_points",
            "bounds_margin",
        ),
        what,
    )
    regions = contract.Regions(
        plane=_box(spec.get("plane"), f"{what} 'plane'"),
        protected=_box(spec.get("protected"), f"{what} 'protected'"),
        edge_band=_box(spec.get("edge_band"), f"{what} 'edge_band'"),
        thin_feature=_box(spec.get("thin_feature"), f"{what} 'thin_feature'"),
        object_region=_box(spec.get("object_region"), f"{what} 'object_region'"),
        coverage_tolerance=_number(spec, "coverage_tolerance", what),
        truth_pitch=_number(spec, "truth_pitch", what),
        analytic_supersample=_integer(spec, "analytic_supersample", what),
        triangle_sample_order=_integer(spec, "triangle_sample_order", what),
        max_triangle_samples=_integer(spec, "max_triangle_samples", what),
        linkage_radius=_number(spec, "linkage_radius", what),
        isolated_max_points=_integer(spec, "isolated_max_points", what),
        bounds_margin=_number(spec, "bounds_margin", what),
    )
    for name, value in (
        ("coverage_tolerance", regions.coverage_tolerance),
        ("truth_pitch", regions.truth_pitch),
        ("linkage_radius", regions.linkage_radius),
    ):
        if value <= 0.0:
            raise FixtureUnusable(f"{what} {name} is {value!r}; it must be positive")
    if regions.bounds_margin < 0.0:
        raise FixtureUnusable(f"{what} bounds_margin is {regions.bounds_margin!r}")
    if regions.isolated_max_points < 1:
        raise FixtureUnusable(f"{what} isolated_max_points is {regions.isolated_max_points!r}")
    if regions.analytic_supersample < 1:
        raise FixtureUnusable(
            f"{what} analytic_supersample is {regions.analytic_supersample!r}; a reference "
            "sampled less than once per pixel is thinner than the sensor it exists to be denser "
            "than"
        )
    if regions.triangle_sample_order < 1:
        raise FixtureUnusable(
            f"{what} triangle_sample_order is {regions.triangle_sample_order!r}; an order below "
            "one takes no sample inside any triangle, which is the whole measurement"
        )
    if regions.max_triangle_samples < 1:
        raise FixtureUnusable(
            f"{what} max_triangle_samples is {regions.max_triangle_samples!r}; the ceiling on "
            "the interior sampling is the fixture's and must admit at least one sample"
        )
    return regions


def _volume(spec: Mapping[str, object], what: str) -> contract.VolumeSpec:
    _exact_keys(spec, ("minimum", "maximum", "min_depth", "max_depth"), what)
    volume = contract.VolumeSpec(
        minimum=_point(spec, "minimum", what),
        maximum=_point(spec, "maximum", what),
        min_depth=_number(spec, "min_depth", what),
        max_depth=_number(spec, "max_depth", what),
    )
    if any(volume.minimum[axis] >= volume.maximum[axis] for axis in range(3)):
        raise FixtureUnusable(f"{what} is not a strictly ordered volume")
    return volume


def _perturbation(spec: Mapping[str, object], what: str) -> contract.Perturbation:
    _exact_keys(
        spec,
        (
            "label",
            "depth_noise_codes",
            "outlier_period",
            "outlier_codes",
            "band_period",
            "band_width",
            "band_shift",
            "translation_bias",
            "translation_jitter",
            "rotation_bias_degrees",
            "rotation_jitter_degrees",
            "rotation_axis",
        ),
        what,
    )
    perturbation = contract.Perturbation(
        label=_text(spec, "label", what),
        depth_noise_codes=_integer(spec, "depth_noise_codes", what),
        outlier_period=_integer(spec, "outlier_period", what),
        outlier_codes=_integer(spec, "outlier_codes", what),
        band_period=_integer(spec, "band_period", what),
        band_width=_integer(spec, "band_width", what),
        band_shift=_integer(spec, "band_shift", what),
        translation_bias=_point(spec, "translation_bias", what),
        translation_jitter=_number(spec, "translation_jitter", what),
        rotation_bias_degrees=_number(spec, "rotation_bias_degrees", what),
        rotation_jitter_degrees=_number(spec, "rotation_jitter_degrees", what),
        rotation_axis=_point(spec, "rotation_axis", what),
    )
    try:
        contract.check_perturbation_label(perturbation)
    except contract.ScoringUnusable as error:
        raise FixtureUnusable(f"{what}: {error}") from error
    return perturbation


def _render(
    shapes: Sequence[Shape], *, intrinsics: IntrinsicsRecord, scale: float, pose: Rigid
) -> bytes:
    key = json.dumps(
        {
            "shapes": [shape.as_dict() for shape in shapes],
            "pose": pose.as_dict(),
            "intrinsics": {name: intrinsics.values[name] for name in ("fx", "fy", "cx", "cy")},
            "depth_scale": scale,
        },
        sort_keys=True,
    )
    cached = _RENDER_CACHE.get(key)
    if cached is None:
        cached = render_frame(shapes, intrinsics=intrinsics, depth_scale=scale, pose=pose).raw
        _RENDER_CACHE[key] = cached
    return cached


def build_scene(
    document: Mapping[str, object],
    *,
    grid: Sequence[contract.GridCell],
    work_budget: int,
    appliance: contract.AppliancePolicy,
    what: str,
) -> contract.Scene:
    """Materialise a case's views from its declared scene and perturbation.

    The frames are not committed as data - a family of 512000-byte frames is
    not reviewable JSON. What is committed is the object, the views, the
    perturbation, the seed and the SHA-256 of every truth and measured frame
    they produce, and this refuses the case if what it renders does not hash to
    what the fixture says. A change to the renderer, the perturbation rule or
    the object shows up here as a refusal rather than as a quietly different
    benchmark - which is also the control against an ALTERED TRUTH: editing a
    truth frame changes its digest, and nothing downstream ever sees it.
    """
    scene_spec = _mapping(document.get("scene"), f"{what} 'scene'")
    _exact_keys(
        scene_spec,
        ("intrinsics", "depth_scale", "window", "shapes", "seed", "volume", "views"),
        f"{what} 'scene'",
    )
    intrinsics = _intrinsics(
        _mapping(scene_spec.get("intrinsics"), f"{what} 'intrinsics'"), f"{what} 'intrinsics'"
    )
    depth_scale = _number(scene_spec, "depth_scale", f"{what} 'scene'")
    if depth_scale <= 0.0:
        raise FixtureUnusable(f"{what} states a depth scale of {depth_scale!r}")
    window = _window(_mapping(scene_spec.get("window"), f"{what} 'window'"), f"{what} 'window'")
    volume = _volume(_mapping(scene_spec.get("volume"), f"{what} 'volume'"), f"{what} 'volume'")
    shape_specs = scene_spec.get("shapes")
    if not isinstance(shape_specs, list) or not shape_specs:
        raise FixtureUnusable(f"{what} names no shapes")
    shapes = [
        _shape(_mapping(spec, f"{what} shape {index}"), f"{what} shape {index}")
        for index, spec in enumerate(shape_specs)
    ]
    seed = _text(scene_spec, "seed", f"{what} 'scene'")
    perturbation = _perturbation(
        _mapping(document.get("perturbation"), f"{what} 'perturbation'"),
        f"{what} 'perturbation'",
    )

    view_specs = scene_spec.get("views")
    if not isinstance(view_specs, list) or not view_specs:
        raise FixtureUnusable(f"{what} names no views")

    views: list[contract.ViewEvidence] = []
    for index, spec in enumerate(view_specs):
        where = f"{what} view {index}"
        record = _mapping(spec, where)
        _exact_keys(
            record,
            ("index", "pose", "truth_sha256", "raw_sha256", "truth_returned", "raw_returned"),
            where,
        )
        if _integer(record, "index", where) != index:
            raise FixtureUnusable(f"{where} states an index that is not its position")
        render_pose = _render_pose(
            _mapping(record.get("pose"), f"{where} 'pose'"), f"{where} 'pose'"
        )
        rendered = _render(shapes, intrinsics=intrinsics, scale=depth_scale, pose=render_pose)
        truth_codes = completeness.codes_of(rendered, window)
        truth_frame = completeness.frame_of_codes(truth_codes, window)
        raw_codes = contract.perturb_codes(
            truth_codes, perturbation, window=window, view=index, seed=f"{seed}:view{index}"
        )
        raw_frame = completeness.frame_of_codes(raw_codes, window)
        _expect_digest(truth_frame, record, "truth_sha256", where)
        _expect_digest(raw_frame, record, "raw_sha256", where)
        _expect_count(truth_codes, record, "truth_returned", where)
        _expect_count(raw_codes, record, "raw_returned", where)
        true_pose = render_pose.inverse()
        views.append(
            contract.ViewEvidence(
                index=index,
                truth=truth_frame,
                raw=raw_frame,
                true_pose=true_pose,
                fused_pose=contract.perturb_pose(
                    true_pose, perturbation, seed=f"{seed}:view{index}"
                ),
            )
        )

    control = document.get("control_case")
    if control is not None and not isinstance(control, str):
        raise FixtureUnusable(f"{what} 'control_case' is {control!r}, not text or null")
    try:
        return contract.Scene(
            case_id=_text(document, "case_id", what),
            window=window,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            shapes=tuple(shapes),
            perturbation=perturbation,
            volume=volume,
            regions=_regions(
                _mapping(document.get("regions"), f"{what} 'regions'"), f"{what} 'regions'"
            ),
            views=tuple(views),
            grid=tuple(grid),
            appliance=appliance,
            work_budget=work_budget,
            control_case=control,
        )
    except contract.ScoringUnusable as error:
        raise FixtureUnusable(f"{what}: {error}") from error


def _expect_digest(frame: bytes, record: Mapping[str, object], key: str, where: str) -> None:
    stated = _digest_text(record, key, where)
    actual = hashlib.sha256(frame).hexdigest()
    if actual != stated:
        raise FixtureUnusable(
            f"{where} {key} is {stated}; the scene and perturbation it declares render {actual}. "
            "A frame that is not the frame the fixture was reviewed with is not evidence"
        )


def _expect_count(codes: Sequence[int], record: Mapping[str, object], key: str, where: str) -> None:
    stated = _integer(record, key, where)
    actual = sum(1 for code in codes if code != contract.NO_RETURN)
    if stated != actual:
        raise FixtureUnusable(f"{where} {key} is {stated}; the frame returns {actual}")


# --- the corpus ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FixtureFile:
    """One case file, with the identity and the label the manifest records for it."""

    name: str
    sha256: str
    bytes_: int
    case_id: str
    perturbation: str


@dataclass(frozen=True, slots=True)
class Corpus:
    """A manifest that has been checked against the directory beside it."""

    root: Path
    manifest_sha256: str
    files: tuple[FixtureFile, ...]
    grid: tuple[contract.GridCell, ...]
    work_budget: int
    appliance: contract.AppliancePolicy
    selection_rule: str


def _safe_child(root: Path, name: str) -> Path:
    """The file ``name`` inside ``root``, or a refusal. The corpora next door's rules."""
    if not FIXTURE_NAME.match(name):
        raise HarnessRefused(
            f"{name!r} is not a fixture file name; a case is a plain lower-case .json file "
            "directly inside the corpus"
        )
    path = root / name
    if path.is_symlink():
        raise HarnessRefused(
            f"{name!r} is a symbolic link. A fixture is evidence and must be the bytes in the "
            "corpus, not a pointer to bytes somewhere else"
        )
    resolved = path.resolve()
    if resolved.parent != root.resolve():
        raise HarnessRefused(f"{name!r} resolves to {resolved}, which is outside {root}")
    if not resolved.is_file():
        raise HarnessRefused(f"{name!r} is named by the manifest but is not a file in {root}")
    return path


def _grid_cell(spec: object, what: str) -> contract.GridCell:
    record = _mapping(spec, what)
    _exact_keys(
        record, ("cell_id", "voxel_size", "truncation_voxels", "min_weight", "weight_policy"), what
    )
    cell_id = _text(record, "cell_id", what)
    if not CELL_ID.match(cell_id):
        raise FixtureUnusable(f"{what} cell id {cell_id!r} is not a plain lower-case name")
    cell = contract.GridCell(
        cell_id=cell_id,
        voxel_size=_number(record, "voxel_size", what),
        truncation_voxels=_number(record, "truncation_voxels", what),
        min_weight=_number(record, "min_weight", what),
        weight_policy=_text(record, "weight_policy", what),
    )
    if cell.voxel_size <= 0.0 or cell.truncation_voxels < 1.0 or cell.min_weight <= 0.0:
        raise FixtureUnusable(
            f"{what} states voxel_size={cell.voxel_size!r}, truncation_voxels="
            f"{cell.truncation_voxels!r}, min_weight={cell.min_weight!r}; a cell scanner3d.tsdf "
            "would refuse is not a cell this corpus can compare"
        )
    return cell


def _appliance_policy(spec: object, what: str) -> contract.AppliancePolicy:
    """The shipped-resolution policy, read from the manifest and checked against the product.

    The voxel size is deliberately not a field here. It is derived per scene by
    :func:`tests.surface_oracle.appliance_cell` from the case's own measured
    points, so a manifest cannot state one; what a manifest owns is the BUDGET
    the Reconstruct action passes and the constants
    :mod:`scanner3d.reconstruction_export` ships, and
    :func:`tests.surface_oracle.check_appliance_policy` refuses a declaration
    that has drifted from those.
    """
    record = _mapping(spec, what)
    _exact_keys(
        record,
        (
            "policy",
            "voxel_budget",
            "truncation_voxels",
            "min_weight",
            "weight_policy",
            "backend",
            "work_budget",
        ),
        what,
    )
    policy = contract.AppliancePolicy(
        policy=_text(record, "policy", what),
        voxel_budget=_integer(record, "voxel_budget", what),
        truncation_voxels=_number(record, "truncation_voxels", what),
        min_weight=_number(record, "min_weight", what),
        weight_policy=_text(record, "weight_policy", what),
        backend=_text(record, "backend", what),
        work_budget=_integer(record, "work_budget", what),
    )
    try:
        contract.check_appliance_policy(policy)
    except contract.ScoringUnusable as error:
        raise FixtureUnusable(f"{what}: {error}") from error
    return policy


def load_corpus(root: Path = CORPUS) -> Corpus:
    """Read and check the manifest against what is on disk, both ways.

    Three things beyond the sibling corpora's checks, and each of them is a
    corruption control:

    * the parameter GRID and the work ceiling live here, in the manifest, and
      every case runs every cell. A grid a case could choose from would be a
      grid a result could be searched for;
    * the manifest records the perturbation label of every case, and every
      label :mod:`tests.surface_oracle` defines must be present. A corpus that
      quietly dropped its bad control would otherwise report only good news;
    * the selection rule text must equal :data:`SELECTION_RULE` exactly, so the
      recommendation the artifact publishes is the one that was reviewed.

    The manifest also owns the APPLIANCE policy - the voxel budget the shipped
    Reconstruct action passes and the fusion constants
    :mod:`scanner3d.reconstruction_export` exports - and
    :func:`_appliance_policy` refuses a declaration that has drifted from those
    imported constants. Without it, every conclusion here would be about a
    surface roughly three times coarser than the one an owner actually gets.
    """
    if not root.is_dir():
        raise HarnessRefused(f"{root} is not a directory; there is no corpus to run")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise HarnessRefused(f"{manifest_path} is not a regular file")
    manifest_bytes = manifest_path.read_bytes()
    try:
        document = _mapping(
            _loads(manifest_bytes.decode("utf-8"), str(manifest_path)), str(manifest_path)
        )
        _exact_keys(
            document,
            (
                "schema_version",
                "kind",
                "what",
                "why",
                "units",
                "generated_by",
                "grid",
                "work_budget",
                "appliance",
                "selection_rule",
                "files",
            ),
            str(manifest_path),
        )
        version = document.get("schema_version")
        if version != FIXTURE_SCHEMA_VERSION or isinstance(version, bool):
            raise FixtureUnusable(
                f"{manifest_path} states schema version {version!r}; this harness understands "
                f"exactly version {FIXTURE_SCHEMA_VERSION} and does not guess at another"
            )
        if _text(document, "kind", str(manifest_path)) != CORPUS_KIND:
            raise FixtureUnusable(f"{manifest_path} is not a {CORPUS_KIND}")
        if _text(document, "selection_rule", str(manifest_path)) != SELECTION_RULE:
            raise FixtureUnusable(
                f"{manifest_path} states a selection rule that is not the one "
                "tests/surface_oracle_harness.py applies. The rule that chooses a recommended "
                "cell is code, and a corpus that records a different one is describing a run "
                "nobody made"
            )
        work_budget = _integer(document, "work_budget", str(manifest_path))
        if work_budget <= 0:
            raise FixtureUnusable(f"{manifest_path} states a work budget of {work_budget}")
        appliance = _appliance_policy(document.get("appliance"), f"{manifest_path} 'appliance'")
        grid_specs = document.get("grid")
        if not isinstance(grid_specs, list) or len(grid_specs) < 2:
            raise FixtureUnusable(
                f"{manifest_path} states fewer than two grid cells. A one-cell grid is a single "
                "parameter choice presented as an envelope"
            )
        grid = tuple(
            _grid_cell(spec, f"{manifest_path} grid {index}")
            for index, spec in enumerate(grid_specs)
        )
        cell_ids = [cell.cell_id for cell in grid]
        if len(set(cell_ids)) != len(cell_ids):
            raise FixtureUnusable(f"{manifest_path} declares a grid cell id twice")

        entries = document.get("files")
        if not isinstance(entries, list) or not entries:
            raise FixtureUnusable(f"{manifest_path} names no files")
        files: list[FixtureFile] = []
        for index, entry in enumerate(entries):
            what = f"{manifest_path} entry {index}"
            record = _mapping(entry, what)
            _exact_keys(record, ("name", "case_id", "perturbation", "bytes", "sha256"), what)
            size = _integer(record, "bytes", what)
            if size <= 0:
                raise FixtureUnusable(f"{what} bytes {size!r} is not a positive integer")
            label = _text(record, "perturbation", what)
            if label not in contract.PERTURBATIONS:
                raise FixtureUnusable(
                    f"{what} files the case under perturbation {label!r}, which is not one of "
                    f"{list(contract.PERTURBATIONS)}"
                )
            files.append(
                FixtureFile(
                    name=_text(record, "name", what),
                    sha256=_digest_text(record, "sha256", what),
                    bytes_=size,
                    case_id=_text(record, "case_id", what),
                    perturbation=label,
                )
            )
    except FixtureUnusable as error:
        raise HarnessRefused(f"the corpus manifest is unusable: {error}") from error

    named = [entry.name for entry in files]
    duplicates = sorted({name for name in named if named.count(name) > 1})
    if duplicates:
        raise HarnessRefused(f"{manifest_path} names {duplicates} more than once")
    identifiers = [entry.case_id for entry in files]
    repeated = sorted({case for case in identifiers if identifiers.count(case) > 1})
    if repeated:
        raise HarnessRefused(
            f"{manifest_path} gives case id(s) {repeated} to more than one file; two cases with "
            "one name cannot both be reported"
        )
    labels = [entry.perturbation for entry in files]
    missing = sorted(set(contract.PERTURBATIONS) - set(labels))
    if missing:
        raise HarnessRefused(
            f"{manifest_path} has no case for perturbation(s) {missing}. Every perturbation this "
            "oracle defines has to be present, or the corpus reports on the causes somebody "
            "chose to keep rather than on the causes there are"
        )
    if labels.count(contract.EXACT) != 1:
        raise HarnessRefused(
            f"{manifest_path} declares {labels.count(contract.EXACT)} exact case(s); the corpus "
            "has exactly one control, or 'compared with the control' means nothing"
        )
    for entry in files:
        _safe_child(root, entry.name)
    on_disk = sorted(
        path.name for path in root.iterdir() if path.name != MANIFEST_NAME and path.is_file()
    )
    unlisted = sorted(set(on_disk) - set(named))
    if unlisted:
        raise HarnessRefused(
            f"{root} holds {unlisted}, which the manifest does not name. A case file nobody "
            "listed is a case nobody reviewed"
        )
    return Corpus(
        root=root,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        files=tuple(sorted(files, key=lambda entry: entry.name)),
        grid=grid,
        work_budget=work_budget,
        appliance=appliance,
        selection_rule=SELECTION_RULE,
    )


# --- one case ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One bar, and where the value it judges must come from."""

    name: str
    measurement: str
    path: tuple[str, ...]
    comparison: quality.Comparison
    limit: float | bool
    unit: str
    context: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "measurement": self.measurement,
            "path": list(self.path),
            "comparison": str(self.comparison),
            "limit": self.limit,
            "unit": self.unit,
            "context": self.context,
        }


@dataclass(frozen=True, slots=True)
class Expectation:
    """What a fixture declares one candidate must do."""

    outcome: str
    failed_gates: tuple[str, ...]
    inapplicable_gates: tuple[str, ...]
    refusal: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "failed_gates": sorted(self.failed_gates),
            "inapplicable_gates": sorted(self.inapplicable_gates),
            "refusal": self.refusal,
        }


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One arm a case runs, and why it is in the case."""

    name: str
    form: str
    cell_id: str | None
    what: str
    expectation: Expectation


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """A fixture that has been read but not yet run."""

    case_id: str
    what: str
    why: str
    notes: tuple[str, ...]
    scene: contract.Scene
    gates: tuple[GateSpec, ...]
    candidates: tuple[CandidateSpec, ...]


def _gate_spec(spec: Mapping[str, object], what: str) -> GateSpec:
    comparison_name = _text(spec, "comparison", what)
    if comparison_name not in COMPARISONS:
        raise FixtureUnusable(
            f"{what} compares with {comparison_name!r}; this harness states "
            f"{sorted(COMPARISONS)} and refuses one it cannot evaluate"
        )
    comparison = COMPARISONS[comparison_name]
    if comparison is quality.Comparison.IS:
        _exact_keys(spec, ("name", "measurement", "path", "comparison", "limit", "context"), what)
        limit: float | bool = _bool(spec, "limit", what)
        unit = quality.BOOLEAN_UNIT
    else:
        _exact_keys(
            spec, ("name", "measurement", "path", "comparison", "limit", "unit", "context"), what
        )
        limit = _number(spec, "limit", what)
        unit = _text(spec, "unit", what)
    path = _strings(spec, "path", what)
    if not path:
        raise FixtureUnusable(f"{what} states an empty path")
    for step in path:
        if not PATH_STEP.match(step):
            raise FixtureUnusable(
                f"{what} walks {step!r}, which is not a plain measurement field name. A gate "
                "reads a field of the measurement it names and nothing else"
            )
    return GateSpec(
        name=_text(spec, "name", what),
        measurement=_text(spec, "measurement", what),
        path=path,
        comparison=comparison,
        limit=limit,
        unit=unit,
        context=_text(spec, "context", what),
    )


def _expectation(spec: Mapping[str, object], gates: Sequence[GateSpec], what: str) -> Expectation:
    _exact_keys(spec, ("outcome", "failed_gates", "inapplicable_gates", "refusal"), what)
    outcome = _text(spec, "outcome", what)
    if outcome not in OUTCOMES:
        raise FixtureUnusable(f"{what} expects outcome {outcome!r}; it must be one of {OUTCOMES}")
    failed = _strings(spec, "failed_gates", what)
    inapplicable = _strings(spec, "inapplicable_gates", what)
    for names, label in ((failed, "failed_gates"), (inapplicable, "inapplicable_gates")):
        if len(set(names)) != len(names):
            raise FixtureUnusable(f"{what} names a gate twice in {label!r}")
        unknown = sorted(set(names) - {gate.name for gate in gates})
        if unknown:
            raise FixtureUnusable(
                f"{what} names gate(s) {unknown} in {label!r}, and the case states no such gate"
            )
    overlap = sorted(set(failed) & set(inapplicable))
    if overlap:
        raise FixtureUnusable(f"{what} expects gate(s) {overlap} to both fail and not apply")
    refusal = spec.get("refusal")
    if refusal is not None and (not isinstance(refusal, str) or refusal not in contract.REFUSALS):
        raise FixtureUnusable(
            f"{what} expects refusal {refusal!r}; the contract states {list(contract.REFUSALS)}"
        )
    if (outcome == "refused") != (refusal is not None):
        raise FixtureUnusable(
            f"{what} expects {outcome!r} beside refusal {refusal!r}. A refused candidate names "
            "the exact refusal it must earn, and a scored one names none"
        )
    if outcome == "refused" and failed:
        raise FixtureUnusable(
            f"{what} expects a refusal and names failing gate(s) {list(failed)}; a refused "
            "candidate is never scored, so no gate of its own could have failed"
        )
    if (outcome == "fail") != bool(failed):
        raise FixtureUnusable(
            f"{what} expects {outcome!r} beside {len(failed)} named failing gate(s). A failing "
            "candidate must say which gates fail, and a passing one must name none"
        )
    return Expectation(
        outcome=outcome,
        failed_gates=failed,
        inapplicable_gates=inapplicable,
        refusal=None if refusal is None else str(refusal),
    )


def _candidate_spec(
    record: Mapping[str, object],
    gates: Sequence[GateSpec],
    grid: Sequence[contract.GridCell],
    where: str,
) -> CandidateSpec:
    _exact_keys(record, ("name", "what", "expectation"), where)
    name = _text(record, "name", where)
    match = CANDIDATE_NAME.match(name)
    if match is None:
        raise FixtureUnusable(
            f"{where} names candidate {name!r}, which is not 'form' or 'form:cell'"
        )
    form, cell_id = match.group(1), match.group(2)
    if form not in contract.CANDIDATE_FORMS:
        raise FixtureUnusable(
            f"{where} runs candidate form {form!r}, which tests/surface_oracle.py does not "
            "define. A fixture cannot name behaviour nobody wrote"
        )
    if cell_id is not None and cell_id not in {cell.cell_id for cell in grid}:
        raise FixtureUnusable(
            f"{where} runs cell {cell_id!r}, which the corpus manifest does not declare. The "
            "grid belongs to the corpus"
        )
    return CandidateSpec(
        name=name,
        form=form,
        cell_id=cell_id,
        what=_text(record, "what", where),
        expectation=_expectation(
            _mapping(record.get("expectation"), f"{where} 'expectation'"),
            gates,
            f"{where} 'expectation'",
        ),
    )


def load_case(path: Path, *, entry: FixtureFile, corpus: Corpus) -> CaseSpec:
    """Read one fixture and materialise its scene. Raises :class:`FixtureUnusable` only."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry.sha256 or len(raw) != entry.bytes_:
        raise FixtureUnusable(
            f"{path.name} is {len(raw)} byte(s) with sha256 {digest}; the manifest records "
            f"{entry.bytes_} byte(s) with sha256 {entry.sha256}. A fixture that has been edited "
            "is no longer the evidence that was reviewed"
        )
    what = path.name
    document = _mapping(_loads(raw.decode("utf-8"), what), what)
    _exact_keys(
        document,
        (
            "schema_version",
            "kind",
            "case_id",
            "what",
            "why",
            "notes",
            "control_case",
            "perturbation",
            "scene",
            "regions",
            "gates",
            "candidates",
        ),
        what,
    )
    version = document.get("schema_version")
    if version != FIXTURE_SCHEMA_VERSION or isinstance(version, bool):
        raise FixtureUnusable(
            f"{what} states schema version {version!r}; this harness understands exactly version "
            f"{FIXTURE_SCHEMA_VERSION}"
        )
    if _text(document, "kind", what) != CASE_KIND:
        raise FixtureUnusable(f"{what} is not a {CASE_KIND}")
    case_id = _text(document, "case_id", what)
    if case_id != entry.case_id:
        raise FixtureUnusable(
            f"{what} calls itself {case_id!r} and the manifest calls it {entry.case_id!r}"
        )
    label = _text(_mapping(document.get("perturbation"), what), "label", what)
    if label != entry.perturbation:
        raise FixtureUnusable(
            f"{what} declares perturbation {label!r}; the manifest files it under "
            f"{entry.perturbation!r}. The label is how a result is attributed to a cause"
        )

    gate_specs = document.get("gates")
    if not isinstance(gate_specs, list) or not gate_specs:
        raise FixtureUnusable(
            f"{what} states no gates. A case that applies no gate passes trivially"
        )
    gates: list[GateSpec] = []
    seen: set[str] = set()
    for index, spec in enumerate(gate_specs):
        gate = _gate_spec(_mapping(spec, f"{what} gate {index}"), f"{what} gate {index}")
        if gate.name in seen:
            raise FixtureUnusable(f"{what} states two gates called {gate.name!r}")
        seen.add(gate.name)
        gates.append(gate)

    candidate_specs = document.get("candidates")
    if not isinstance(candidate_specs, list) or not candidate_specs:
        raise FixtureUnusable(f"{what} runs no candidates")
    candidates: list[CandidateSpec] = []
    named: set[str] = set()
    for index, spec in enumerate(candidate_specs):
        where = f"{what} candidate {index}"
        candidate = _candidate_spec(_mapping(spec, where), gates, corpus.grid, where)
        if candidate.name in named:
            raise FixtureUnusable(f"{what} runs candidate {candidate.name!r} twice")
        named.add(candidate.name)
        candidates.append(candidate)

    # The cherry-picking control, at the only place it can be enforced: the
    # fixture. Every declared cell is run and reported, so a case cannot show
    # the parameter that flattered it and leave the rest unmeasured.
    required = {f"fused:{cell.cell_id}" for cell in corpus.grid}
    absent = sorted(required - named)
    if absent:
        raise FixtureUnusable(
            f"{what} does not run {absent}. Every case runs every declared grid cell; a case "
            "that runs a subset is reporting a search, not an envelope"
        )
    if "raw_points" not in named:
        raise FixtureUnusable(
            f"{what} does not run 'raw_points'. The measured dots are the comparison this whole "
            "corpus exists to make, and a case without them measures fusion against nothing"
        )
    if "fused_at_appliance_resolution" not in named:
        raise FixtureUnusable(
            f"{what} does not run 'fused_at_appliance_resolution'. The declared grid is coarser "
            "than the resolution the shipped export policy chooses for this scene, and a "
            "conclusion about the mesh an owner actually gets cannot be drawn from a proxy of it"
        )
    return CaseSpec(
        case_id=case_id,
        what=_text(document, "what", what),
        why=_text(document, "why", what),
        notes=_strings(document, "notes", what),
        scene=build_scene(
            document,
            grid=corpus.grid,
            work_budget=corpus.work_budget,
            appliance=corpus.appliance,
            what=what,
        ),
        gates=tuple(gates),
        candidates=tuple(candidates),
    )


# --- running one candidate ---------------------------------------------------------------


def _resolve(value: object, path: Sequence[str], what: str) -> object:
    current = value
    for step in path:
        if not isinstance(current, Mapping) or step not in current:
            raise FixtureUnusable(f"{what}: no {step!r} at {list(path)}")
        current = current[step]
    return current


@dataclass(frozen=True, slots=True)
class GateBinding:
    """Which measurement field a gate actually read."""

    gate: str
    measurement: str
    path: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"gate": self.gate, "measurement": self.measurement, "path": list(self.path)}


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """What one arm did, beside what its fixture said it must do."""

    name: str
    what: str
    expectation: Expectation
    outcome: str
    failed_gates: tuple[str, ...]
    inapplicable_gates: tuple[str, ...]
    refusal: str | None
    report: quality.CaseReport | None
    bindings: tuple[GateBinding, ...]
    error: str | None


def _mismatch_lines(expected: Expectation, observed: CandidateOutcome) -> list[str]:
    lines: list[str] = []
    if observed.outcome != expected.outcome:
        lines.append(f"expected outcome {expected.outcome!r}, observed {observed.outcome!r}")
    if observed.refusal != expected.refusal:
        lines.append(f"expected refusal {expected.refusal!r}, observed {observed.refusal!r}")
    if sorted(observed.failed_gates) != sorted(expected.failed_gates):
        lines.append(
            f"expected failing gate(s) {sorted(expected.failed_gates)}, observed "
            f"{sorted(observed.failed_gates)}"
        )
    if sorted(observed.inapplicable_gates) != sorted(expected.inapplicable_gates):
        lines.append(
            f"expected inapplicable gate(s) {sorted(expected.inapplicable_gates)}, observed "
            f"{sorted(observed.inapplicable_gates)}"
        )
    if observed.error is not None:
        lines.append(f"raised {observed.error}")
    return lines


def run_candidate(case: CaseSpec, spec: CandidateSpec) -> CandidateOutcome:
    """Run one arm against the case's own gates and report what happened.

    An arm :mod:`tests.surface_oracle` refuses is recorded as refused with its
    exact refusal code and is NOT scored: a metric computed from a result whose
    account of itself does not hold would be a number about nothing.
    """
    inapplicable_all = tuple(gate.name for gate in case.gates)
    try:
        arm = contract.candidate(spec.form, spec.cell_id)(case.scene)
        contract.validate(case.scene, arm)
        measurements = contract.measure(case.scene, arm)
    except contract.CandidateRefused as refusal:
        return CandidateOutcome(
            name=spec.name,
            what=spec.what,
            expectation=spec.expectation,
            outcome="refused",
            failed_gates=(),
            inapplicable_gates=inapplicable_all,
            refusal=refusal.refusal,
            report=None,
            bindings=(),
            error=None,
        )
    except Exception as error:
        return CandidateOutcome(
            name=spec.name,
            what=spec.what,
            expectation=spec.expectation,
            outcome="fail",
            failed_gates=(),
            inapplicable_gates=inapplicable_all,
            refusal=None,
            report=None,
            bindings=(),
            error=f"{type(error).__name__}: {error}",
        )

    by_name = {measurement.name: measurement for measurement in measurements}
    gates: list[quality.Gate] = []
    bindings: list[GateBinding] = []
    inapplicable: list[str] = []
    for gate in case.gates:
        measurement = by_name.get(gate.measurement)
        if measurement is None:
            inapplicable.append(gate.name)
            continue
        measured = _resolve(
            measurement.as_dict(), gate.path, f"{case.case_id}:{spec.name} gate {gate.name!r}"
        )
        bindings.append(GateBinding(gate=gate.name, measurement=gate.measurement, path=gate.path))
        if gate.comparison is quality.Comparison.IS:
            if not isinstance(measured, bool):
                raise FixtureUnusable(
                    f"{case.case_id} gate {gate.name!r} compares {measured!r} with 'is', but it "
                    "is not a boolean"
                )
            gates.append(
                quality.must_be(gate.name, measured, bool(gate.limit), context=gate.context)
            )
            continue
        value = as_number(measured, f"{case.case_id} gate {gate.name!r} measured value")
        limit = float(gate.limit)
        if gate.comparison is quality.Comparison.AT_MOST:
            gates.append(
                quality.at_most(gate.name, value, limit, unit=gate.unit, context=gate.context)
            )
        else:
            gates.append(
                quality.at_least(gate.name, value, limit, unit=gate.unit, context=gate.context)
            )
    report = quality.CaseReport(
        name=f"{case.case_id}:{spec.name}",
        measurements=measurements,
        gates=tuple(gates),
        notes=(spec.what,),
    )
    return CandidateOutcome(
        name=spec.name,
        what=spec.what,
        expectation=spec.expectation,
        outcome="pass" if report.passed else "fail",
        failed_gates=tuple(gate.name for gate in report.failures),
        inapplicable_gates=tuple(inapplicable),
        refusal=None,
        report=report,
        bindings=tuple(bindings),
        error=None,
    )


def _candidate_entry(outcome: CandidateOutcome) -> dict[str, object]:
    mismatches = _mismatch_lines(outcome.expectation, outcome)
    return {
        "candidate": outcome.name,
        "what": outcome.what,
        "status": "pass" if not mismatches else "fail",
        "expected": outcome.expectation.as_dict(),
        "observed": {
            "outcome": outcome.outcome,
            "failed_gates": sorted(outcome.failed_gates),
            "inapplicable_gates": sorted(outcome.inapplicable_gates),
            "refusal": outcome.refusal,
        },
        "mismatches": mismatches,
        "error": outcome.error,
        "quality": None if outcome.report is None else outcome.report.as_dict(),
        "gate_bindings": [binding.as_dict() for binding in outcome.bindings],
    }


def _scene_entry(scene: contract.Scene) -> dict[str, object]:
    return {
        "window": scene.window.as_dict(),
        "depth_scale": scene.depth_scale,
        "intrinsics": {key: scene.intrinsics.values[key] for key in ("fx", "fy", "cx", "cy")},
        "shapes": [shape.as_dict() for shape in scene.shapes],
        "perturbation": scene.perturbation.as_dict(),
        "volume": scene.volume.as_dict(),
        "regions": scene.regions.as_dict(),
        # The shipped policy, the budget it was given and the voxel size it
        # derived for THIS scene. Published rather than described so a reader
        # can re-run choose_tsdf_resolution on the same points and get the same
        # number, and so a coarse proxy can never be mistaken for the appliance.
        "appliance": contract.appliance_resolution(scene),
        "control_case": scene.control_case,
        "work_budget": scene.work_budget,
        "views": [
            {
                "index": view.index,
                "truth_sha256": view.truth_sha256,
                "raw_sha256": view.raw_sha256,
                "truth_returned": sum(
                    1 for code in scene.truth_codes(view.index) if code != contract.NO_RETURN
                ),
                "raw_returned": sum(
                    1 for code in scene.raw_codes(view.index) if code != contract.NO_RETURN
                ),
                "true_pose": view.true_pose.as_dict(),
                "fused_pose": view.fused_pose.as_dict(),
                "pose_error_degrees": math.degrees(
                    view.true_pose.inverse().compose(view.fused_pose).rotation_angle
                ),
                "pose_error_translation": view.true_pose.inverse()
                .compose(view.fused_pose)
                .translation_magnitude,
            }
            for view in scene.views
        ],
    }


# --- the numbers a case is summarised by --------------------------------------------------


def _metric_key(measurement: str, path: Sequence[str]) -> str:
    return ".".join([measurement, *path])


def _metric_of(entry: Mapping[str, object], measurement: str, path: Sequence[str]) -> float | None:
    """One attributed number out of a published candidate entry, or ``None``.

    ``None`` rather than a substituted zero: a candidate that was refused, or
    one whose case could not measure a region, has no value here, and inventing
    one would put a delta in the attribution that nothing measured.
    """
    report = entry.get("quality")
    if not isinstance(report, Mapping):
        return None
    measurements = report.get("measurements")
    if not isinstance(measurements, list):
        return None
    for record in measurements:
        if not isinstance(record, Mapping) or record.get("name") != measurement:
            continue
        current: object = record
        for step in path:
            if not isinstance(current, Mapping) or step not in current:
                return None
            current = current[step]
        if isinstance(current, bool) or not isinstance(current, int | float):
            return None
        return float(current)
    return None


def _summary(entry: Mapping[str, object]) -> dict[str, float]:
    """Every attributed metric of one candidate, by its dotted name."""
    out: dict[str, float] = {}
    for measurement, path in ATTRIBUTED_METRICS:
        value = _metric_of(entry, measurement, path)
        if value is not None:
            out[_metric_key(measurement, path)] = value
    return out


def _candidates_by_name(case: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    entries = case.get("candidates")
    if not isinstance(entries, list):
        return {}
    return {
        str(entry["candidate"]): entry
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("candidate"), str)
    }


def _recommended_cell(case: Mapping[str, object], grid: Sequence[contract.GridCell]) -> str | None:
    """Apply :data:`SELECTION_RULE` over every declared cell, or answer ``None``.

    ``None`` when a declared cell produced no coverage number at all, which is
    a case that could not be summarised rather than a case whose best cell is
    the one that still worked.
    """
    entries = _candidates_by_name(case)
    ranked: list[tuple[float, float, float, str]] = []
    for cell in grid:
        entry = entries.get(f"fused:{cell.cell_id}")
        if entry is None:
            return None
        coverage = _metric_of(entry, "analytic", ("reference_coverage_fraction",))
        p95 = _metric_of(entry, "analytic", ("distances", "p95"))
        work = _metric_of(entry, "field", ("work_units",))
        if coverage is None or p95 is None or work is None:
            return None
        ranked.append((-coverage, p95, work, cell.cell_id))
    if not ranked:
        return None
    return min(ranked)[3]


def _attribution(
    cases: Sequence[Mapping[str, object]], grid: Sequence[contract.GridCell]
) -> list[dict[str, object]]:
    """What each perturbation did to each arm, measured against its own control.

    One entry per perturbed case. Each entry compares the SAME candidate name
    on the perturbed case and on its exact control, so the only difference
    between the two numbers is the perturbation. The deltas are arithmetic and
    are re-checked by :func:`validate_report`; the ``dominant`` field applies
    :data:`WORSE_WHEN_HIGHER` / :data:`WORSE_WHEN_LOWER`, which were declared
    before any case ran.
    """
    by_id = {str(case["case_id"]): case for case in cases if isinstance(case.get("case_id"), str)}
    out: list[dict[str, object]] = []
    for case in cases:
        scene = case.get("scene")
        if not isinstance(scene, Mapping):
            continue
        control_id = scene.get("control_case")
        if not isinstance(control_id, str):
            continue
        control = by_id.get(control_id)
        if control is None:
            continue
        perturbation = scene.get("perturbation")
        label = (
            str(perturbation["label"])
            if isinstance(perturbation, Mapping) and isinstance(perturbation.get("label"), str)
            else "unknown"
        )
        family = (
            "depth"
            if label in contract.DEPTH_FAMILY
            else "pose"
            if label in contract.POSE_FAMILY
            else "interaction"
        )
        perturbed_entries = _candidates_by_name(case)
        control_entries = _candidates_by_name(control)
        arms: list[dict[str, object]] = []
        for name in sorted(set(perturbed_entries) & set(control_entries)):
            if not (name in ("raw_points", APPLIANCE_ARM) or name.startswith("fused:")):
                continue
            here = _summary(perturbed_entries[name])
            there = _summary(control_entries[name])
            metrics = [
                {
                    "metric": key,
                    "control": there[key],
                    "perturbed": here[key],
                    "delta": here[key] - there[key],
                    "degraded": (
                        here[key] > there[key]
                        if key in WORSE_WHEN_HIGHER
                        else here[key] < there[key]
                    ),
                }
                for key in sorted(set(here) & set(there))
                if key in WORSE_WHEN_HIGHER or key in WORSE_WHEN_LOWER
            ]
            arms.append(
                {
                    "candidate": name,
                    "metrics": metrics,
                    "degraded_metrics": sum(1 for record in metrics if record["degraded"]),
                }
            )
        out.append(
            {
                "case_id": str(case["case_id"]),
                "control_case": control_id,
                "perturbation": label,
                "family": family,
                "recommended_cell": _recommended_cell(case, grid),
                "arms": arms,
            }
        )
    return out


def _raw_versus_fused(
    cases: Sequence[Mapping[str, object]], grid: Sequence[contract.GridCell]
) -> list[dict[str, object]]:
    """The headline of the whole corpus, per case: the dots against the best cell.

    Published for every case including the exact control, and never reduced to
    one verdict. If the dots win on surface accuracy while the surface wins on
    nothing, that is the measurement and it is stated as such.

    Two fused arms, not one. ``recommended_cell`` is the best of the declared
    grid and ``appliance`` is the resolution the shipped policy actually
    chooses for this scene, which on this corpus is several times finer. A
    reader who only had the first would be reading a proxy of the mesh an owner
    gets rather than the mesh itself.
    """
    out: list[dict[str, object]] = []
    for case in cases:
        entries = _candidates_by_name(case)
        raw = entries.get("raw_points")
        cell_id = _recommended_cell(case, grid)
        if raw is None or cell_id is None:
            continue
        fused = entries.get(f"fused:{cell_id}")
        if fused is None:  # pragma: no cover - _recommended_cell already required it
            continue
        raw_summary = _summary(raw)
        record: dict[str, object] = {
            "case_id": str(case.get("case_id")),
            "recommended_cell": cell_id,
        }
        for label, entry in (("grid", fused), ("appliance", entries.get(APPLIANCE_ARM))):
            if entry is None:
                record[label] = None
                continue
            fused_summary = _summary(entry)
            comparisons = [
                {
                    "metric": key,
                    "raw_points": raw_summary[key],
                    "fused": fused_summary[key],
                    "delta": fused_summary[key] - raw_summary[key],
                    "fusion_better": (
                        fused_summary[key] < raw_summary[key]
                        if key in WORSE_WHEN_HIGHER
                        else fused_summary[key] > raw_summary[key]
                    ),
                }
                for key in sorted(set(raw_summary) & set(fused_summary))
                if key in WORSE_WHEN_HIGHER or key in WORSE_WHEN_LOWER
            ]
            record[label] = {
                "candidate": (f"fused:{cell_id}" if label == "grid" else APPLIANCE_ARM),
                "comparisons": comparisons,
                "fusion_better_on": sum(
                    1 for comparison in comparisons if comparison["fusion_better"]
                ),
                "metrics_compared": len(comparisons),
            }
        out.append(record)
    return out


def _case_entry(
    entry: FixtureFile, case: CaseSpec | None, error: str | None, grid: Sequence[contract.GridCell]
) -> dict[str, object]:
    fixture = {
        "name": entry.name,
        "sha256": entry.sha256,
        "bytes": entry.bytes_,
        "perturbation": entry.perturbation,
    }
    if case is None:
        return {
            "case_id": entry.case_id,
            "fixture": fixture,
            "status": "malformed",
            "what": None,
            "why": None,
            "notes": [],
            "scene": None,
            "gates": [],
            "candidates": [],
            "recommended_cell": None,
            "error": error,
        }
    candidates = [_candidate_entry(run_candidate(case, spec)) for spec in case.candidates]
    failed = [item for item in candidates if item["status"] != "pass"]
    record: dict[str, object] = {
        "case_id": case.case_id,
        "fixture": fixture,
        "status": "pass" if not failed else "fail",
        "what": case.what,
        "why": case.why,
        "notes": list(case.notes),
        "scene": _scene_entry(case.scene),
        "gates": [gate.as_dict() for gate in case.gates],
        "candidates": candidates,
        "recommended_cell": None,
        "error": None,
    }
    record["recommended_cell"] = _recommended_cell(record, grid)
    return record


def _check_controls(cases: Sequence[CaseSpec], corpus: Corpus) -> None:
    """Refuse a corpus whose controls are not controls. The swapped-baseline check.

    A perturbed case is only comparable with its control if the two are the
    same scene: same object, same views, same window, same volume, same
    regions. The strongest available evidence for that is the TRUTH of every
    view, which is rendered before any perturbation - so the check is that the
    control's truth digests are the perturbed case's truth digests, view for
    view. A case pointed at the wrong baseline, at another perturbed case, or
    at a scene that differs by one shape fails here, before any number is
    published.
    """
    expected_refusals = {
        candidate.expectation.refusal
        for case in cases
        for candidate in case.candidates
        if candidate.expectation.refusal is not None
    }
    unexercised = sorted(CONTROL_REFUSALS - expected_refusals)
    if unexercised:
        raise HarnessRefused(
            f"no case in this corpus expects the refusal(s) {unexercised}. Every corruption "
            "control this harness owns has to be earned somewhere, or the strict reader is "
            "being graded on the attacks that were left in"
        )
    unknown = sorted(expected_refusals - set(contract.REFUSALS))
    if unknown:
        raise HarnessRefused(
            f"a case expects the refusal(s) {unknown}, which the contract does not define. A "
            "fixture cannot name a refusal nobody wrote"
        )

    by_id = {case.case_id: case for case in cases}
    for case in cases:
        control_id = case.scene.control_case
        if control_id is None:
            continue
        control = by_id.get(control_id)
        if control is None:
            raise HarnessRefused(
                f"case {case.case_id!r} names control {control_id!r}, which the corpus does not "
                "contain. A perturbation compared against nothing measures nothing"
            )
        if control.scene.perturbation.label != contract.EXACT:
            raise HarnessRefused(
                f"case {case.case_id!r} names {control_id!r} as its control, and that case is "
                f"perturbed ({control.scene.perturbation.label!r}). A control is the exact case, "
                "or the difference between them is two causes at once"
            )
        here = [view.truth_sha256 for view in case.scene.views]
        there = [view.truth_sha256 for view in control.scene.views]
        if here != there:
            raise HarnessRefused(
                f"case {case.case_id!r} and its control {control_id!r} do not render the same "
                "truth. They are different scenes, and every delta between them would be the "
                "difference between two objects rather than the effect of one perturbation"
            )
        if case.scene.volume.as_dict() != control.scene.volume.as_dict():
            raise HarnessRefused(
                f"case {case.case_id!r} and its control {control_id!r} fuse into different "
                "volumes; observed space has to be the same box on both sides of a comparison"
            )
        if case.scene.regions.as_dict() != control.scene.regions.as_dict():
            raise HarnessRefused(
                f"case {case.case_id!r} and its control {control_id!r} name different regions; "
                "a plane, an edge or a protected volume that moved between them would make "
                "every regional delta a comparison of two different questions"
            )


def run(root: Path = CORPUS) -> dict[str, object]:
    """Run every arm of every case, attribute the results, and return the artifact.

    Cases are independent and every one is attempted: an exception out of a
    fixture or a measurement is recorded against its own case and the next case
    still runs, so the artifact preserves all failures rather than the first.
    ``BaseException`` is deliberately not caught - a KeyboardInterrupt is the
    operator ending the run, not a case failing.
    """
    corpus = load_corpus(root)
    loaded: list[tuple[FixtureFile, CaseSpec | None, str | None]] = []
    for fixture in corpus.files:
        path = _safe_child(root, fixture.name)
        try:
            loaded.append((fixture, load_case(path, entry=fixture, corpus=corpus), None))
        except Exception as error:
            loaded.append((fixture, None, f"{type(error).__name__}: {error}"))

    _check_controls([case for _, case, _ in loaded if case is not None], corpus)

    entries: list[dict[str, object]] = []
    for fixture, case, reason in loaded:
        try:
            entries.append(_case_entry(fixture, case, reason, corpus.grid))
        except Exception as failure:  # pragma: no cover - a gate path that will not resolve
            entries.append(
                _case_entry(fixture, None, f"{type(failure).__name__}: {failure}", corpus.grid)
            )

    statuses = [str(entry["status"]) for entry in entries]
    candidate_entries: list[Mapping[str, object]] = []
    for entry in entries:
        reported = entry["candidates"]
        assert isinstance(reported, list)
        candidate_entries.extend(item for item in reported if isinstance(item, Mapping))
    attribution = _attribution(entries, corpus.grid)
    totals = {
        "cases": len(entries),
        "passed": statuses.count("pass"),
        "failed": statuses.count("fail"),
        "malformed": statuses.count("malformed"),
        "candidates": len(candidate_entries),
        "candidates_passed": sum(
            1 for candidate in candidate_entries if candidate["status"] == "pass"
        ),
        "attributed_cases": len(attribution),
    }
    document: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "corpus": {
            "kind": CORPUS_KIND,
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "manifest_sha256": corpus.manifest_sha256,
            "case_count": len(corpus.files),
            "work_budget": corpus.work_budget,
            "appliance": corpus.appliance.as_dict(),
            "selection_rule": corpus.selection_rule,
            "grid": [cell.as_dict() for cell in corpus.grid],
        },
        "arms": list(contract.ARMS),
        "perturbations": list(contract.PERTURBATIONS),
        "refusals": list(contract.REFUSALS),
        "claims": list(contract.CLAIMS),
        "attributed_metrics": [
            _metric_key(measurement, path) for measurement, path in ATTRIBUTED_METRICS
        ],
        "worse_when_higher": sorted(WORSE_WHEN_HIGHER),
        "worse_when_lower": sorted(WORSE_WHEN_LOWER),
        "cases": entries,
        "attribution": attribution,
        "raw_versus_fused": _raw_versus_fused(entries, corpus.grid),
        "totals": totals,
        "decision": "pass" if totals["passed"] == totals["cases"] and entries else "fail",
        "limitations": list(LIMITATIONS),
    }
    validate_report(document)
    return document


# --- reading a published artifact back, strictly -------------------------------------------


def loads_report(text: str, *, what: str = "surface-oracle report") -> Mapping[str, object]:
    """Read an artifact from its TEXT, strictly, and validate it.

    From the text, not from an already-decoded mapping: a duplicated key is one
    mapping by the time an ordinary decoder is done with it, and no validator
    over that mapping could tell that the file had two answers in it.
    """

    def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        seen: dict[str, object] = {}
        for key, value in pairs:
            if key in seen:
                raise ReportUnusable(f"{what} states {key!r} twice in one object")
            seen[key] = value
        return seen

    try:
        document = json.loads(text, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as exc:
        raise ReportUnusable(f"{what} is not JSON: {exc}") from exc
    return validate_report(document, what=what)


def read_report(path: Path, *, what: str | None = None) -> Mapping[str, object]:
    return loads_report(path.read_text(encoding="utf-8"), what=what or str(path))


def _report_mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportUnusable(f"{what} is {type(value).__name__}, not an object")
    return value


def _report_list(record: Mapping[str, object], key: str, what: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ReportUnusable(f"{what} {key!r} is not a list")
    return value


def _report_text(record: Mapping[str, object], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not text")
    return value


def _report_number(record: Mapping[str, object], key: str, what: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not a number")
    if not math.isfinite(float(value)):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, which is not finite")
    return float(value)


def _report_int(record: Mapping[str, object], key: str, what: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not an integer")
    return value


def _no_nonfinite(value: object, what: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportUnusable(f"{what} carries the non-finite value {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _no_nonfinite(item, f"{what}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _no_nonfinite(item, f"{what}[{index}]")


_TOLERANCE: Final = 1e-9


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TOLERANCE * max(1.0, abs(left), abs(right))


def _validate_measurement(value: object, where: str) -> tuple[str, Mapping[str, object]]:
    """One measurement, of a kind this artifact declares, with its name."""
    record = _report_mapping(value, where)
    kind = _report_text(record, "kind", where)
    if kind not in MEASUREMENT_KINDS:
        raise ReportUnusable(
            f"{where} is of kind {kind!r}; this artifact carries {list(MEASUREMENT_KINDS)} and a "
            "reader that guessed at another would be reading a schema nobody wrote"
        )
    name = _report_text(record, "name", where)
    if kind == "surface_oracle_geometry":
        _validate_geometry(record, where)
    elif kind == "surface_oracle_edge":
        _validate_edge(record, where)
    elif kind == "surface_oracle_components":
        _validate_components(record, where)
    elif kind == "surface_oracle_field":
        _validate_field(record, where)
    elif kind == "surface_oracle_analytic":
        _validate_analytic(record, where)
    elif kind == "surface_oracle_ablation":
        _validate_ablation(record, where)
    elif kind == "surface_oracle_surface_samples":
        _validate_surface_samples(record, where)
    elif kind == "surface_oracle_topology":
        _validate_topology(record, where)
    elif kind == "cloud_agreement":
        _validate_agreement(record, where)
    return name, record


def _fraction(record: Mapping[str, object], key: str, where: str) -> float:
    value = _report_number(record, key, where)
    if not 0.0 <= value <= 1.0:
        raise ReportUnusable(
            f"{where} {key!r} is {value!r}, outside [0, 1]. A fraction of a fixed denominator "
            "cannot leave the interval, and one that has is an impossible improvement"
        )
    return value


def _count(record: Mapping[str, object], key: str, where: str) -> int:
    value = _report_int(record, key, where)
    if value < 0:
        raise ReportUnusable(f"{where} {key!r} is {value!r}, and a count is never negative")
    return value


def _validate_geometry(record: Mapping[str, object], where: str) -> None:
    point_count = _count(record, "point_count", where)
    compared = _count(record, "compared_points", where)
    truth = _count(record, "truth_points", where)
    outside = _count(record, "outside_volume", where)
    if compared > point_count:
        raise ReportUnusable(
            f"{where} compares {compared} point(s) out of {point_count}. Decimation removes "
            "points and never invents them"
        )
    if outside > point_count:
        raise ReportUnusable(f"{where} puts {outside} of {point_count} point(s) outside")
    if truth <= 0:
        raise ReportUnusable(f"{where} states {truth} truth point(s); there is nothing to score")
    arm = _report_text(record, "arm", where)
    if arm not in contract.ARMS:
        raise ReportUnusable(
            f"{where} names arm {arm!r}, which is not one of {list(contract.ARMS)}"
        )
    for key in ("extent_x", "extent_y", "extent_z"):
        if _report_number(record, key, where) < 0.0:
            raise ReportUnusable(f"{where} {key!r} is negative")


def _validate_edge(record: Mapping[str, object], where: str) -> None:
    for covered, total, fraction in (
        ("edge_covered_points", "edge_truth_points", "edge_coverage"),
        ("thin_feature_covered_points", "thin_feature_truth_points", "thin_feature_coverage"),
    ):
        hits = _count(record, covered, where)
        population = _count(record, total, where)
        value = _fraction(record, fraction, where)
        if hits > population:
            raise ReportUnusable(
                f"{where} covers {hits} of {population} truth point(s) in {covered!r}; a "
                "coverage over its own denominator is an impossible improvement"
            )
        expected = (hits / population) if population else 0.0
        if not _close(expected, value):
            raise ReportUnusable(
                f"{where} states {fraction}={value!r}, and {hits}/{population} is {expected!r}"
            )
    _count(record, "bridged_points", where)
    _count(record, "protected_truth_points", where)
    if _count(record, "protected_truth_points", where) != 0:
        raise ReportUnusable(
            f"{where} finds truth inside the protected volume. That region is declared empty, "
            "and a bridging count measured against a region that is not empty measures nothing"
        )


def _validate_components(record: Mapping[str, object], where: str) -> None:
    total_components = _count(record, "component_count", where)
    largest = _count(record, "largest_component_points", where)
    isolated = _count(record, "isolated_components", where)
    isolated_points = _count(record, "isolated_points", where)
    if isolated > total_components:
        raise ReportUnusable(
            f"{where} counts {isolated} isolated of {total_components} component(s)"
        )
    if total_components == 0 and largest != 0:
        raise ReportUnusable(f"{where} has no components and a largest of {largest}")
    if isolated_points < isolated:
        raise ReportUnusable(
            f"{where} puts {isolated_points} point(s) in {isolated} isolated component(s); a "
            "component holds at least one point"
        )
    _fraction(record, "largest_component_fraction", where)
    _fraction(record, "isolated_fraction", where)
    if _report_number(record, "linkage_radius", where) <= 0.0:
        raise ReportUnusable(f"{where} states a non-positive linkage radius")


def _validate_field(record: Mapping[str, object], where: str) -> None:
    cells = _count(record, "cells", where)
    observed = _count(record, "observed_voxels", where)
    unobserved = _count(record, "unobserved_voxels", where)
    if observed + unobserved != cells:
        raise ReportUnusable(
            f"{where} accounts for {observed} + {unobserved} voxel(s) of {cells}. Every voxel is "
            "observed or it is not"
        )
    offered = _count(record, "frames_offered", where)
    integrated = _count(record, "frames_integrated", where)
    if integrated > offered:
        raise ReportUnusable(f"{where} integrated {integrated} of {offered} frame(s) offered")
    vertices = _count(record, "surface_vertices", where)
    _count(record, "surface_faces", where)
    extracted = record.get("surface_extracted")
    if not isinstance(extracted, bool):
        raise ReportUnusable(f"{where} 'surface_extracted' is not a boolean")
    if not extracted and vertices:
        raise ReportUnusable(f"{where} extracted no surface and reports {vertices} vertex/vertices")
    work = _count(record, "work_units", where)
    budget = _count(record, "work_budget", where)
    within = record.get("within_budget")
    if not isinstance(within, bool):
        raise ReportUnusable(f"{where} 'within_budget' is not a boolean")
    if within != (work <= budget):
        raise ReportUnusable(
            f"{where} states within_budget={within} for {work} unit(s) against a {budget} "
            "ceiling. The ceiling is the fixture's, and the verdict is arithmetic"
        )
    if _report_number(record, "truncation", where) < _report_number(record, "voxel_size", where):
        raise ReportUnusable(
            f"{where} truncates below one voxel, which scanner3d.tsdf refuses to configure"
        )
    if _report_text(record, "weight_policy", where) not in {"uniform", "ramped_behind"}:
        raise ReportUnusable(f"{where} names a weight policy scanner3d.tsdf does not define")


def _validate_agreement(record: Mapping[str, object], where: str) -> None:
    for direction in ("observed_to_reference", "reference_to_observed"):
        side = _report_mapping(record.get(direction), f"{where}.{direction}")
        _fraction(side, "coverage_fraction", f"{where}.{direction}")
        distances = _report_mapping(side.get("distances"), f"{where}.{direction}.distances")
        median = _report_number(distances, "median", f"{where}.{direction}.distances")
        p95 = _report_number(distances, "p95", f"{where}.{direction}.distances")
        worst = _report_number(distances, "worst", f"{where}.{direction}.distances")
        if not 0.0 <= median <= p95 <= worst:
            raise ReportUnusable(
                f"{where}.{direction} is not ordered: median {median}, p95 {p95}, worst {worst}"
            )


def _validate_distances(record: Mapping[str, object], key: str, where: str) -> int:
    """One distance summary, ordered and self-consistent. Returns its own count."""
    block = _report_mapping(record.get(key), f"{where}.{key}")
    count = _count(block, "count", f"{where}.{key}")
    median = _report_number(block, "median", f"{where}.{key}")
    rms = _report_number(block, "rms", f"{where}.{key}")
    p95 = _report_number(block, "p95", f"{where}.{key}")
    worst = _report_number(block, "worst", f"{where}.{key}")
    if not 0.0 <= median <= p95 <= worst:
        raise ReportUnusable(
            f"{where}.{key} is not ordered: median {median}, p95 {p95}, worst {worst}"
        )
    if not 0.0 <= rms <= worst:
        raise ReportUnusable(f"{where}.{key} states an rms of {rms} above its worst of {worst}")
    return count


def _validate_analytic(record: Mapping[str, object], where: str) -> None:
    """Closed-form accuracy and dense-reference coverage, re-checked as arithmetic."""
    scored = _count(record, "scored_points", where)
    if _validate_distances(record, "distances", where) != scored:
        raise ReportUnusable(
            f"{where} summarises a different number of distances than the {scored} point(s) it "
            "says it scored. The analytic distance is closed form and every point has one"
        )
    near = _count(record, "within_tolerance_points", where)
    fraction = _fraction(record, "within_tolerance_fraction", where)
    if near > scored:
        raise ReportUnusable(f"{where} puts {near} of {scored} point(s) within tolerance")
    if not _close(fraction, (near / scored) if scored else 0.0):
        raise ReportUnusable(f"{where} within_tolerance_fraction is not {near}/{scored}")
    reference = _count(record, "reference_points", where)
    covered = _count(record, "reference_covered_points", where)
    coverage = _fraction(record, "reference_coverage_fraction", where)
    if reference <= 0:
        raise ReportUnusable(
            f"{where} states {reference} analytic reference point(s); a coverage over an empty "
            "reference measures nothing"
        )
    if covered > reference:
        raise ReportUnusable(f"{where} covers {covered} of {reference} reference point(s)")
    if not _close(coverage, covered / reference):
        raise ReportUnusable(f"{where} reference_coverage_fraction is not {covered}/{reference}")
    if _report_int(record, "supersample", where) < 1:
        raise ReportUnusable(f"{where} states a supersample below one")
    if _report_text(record, "truth_source", where) != contract.ANALYTIC_SOURCE:
        raise ReportUnusable(
            f"{where} states a truth source that is not the closed-form one this oracle "
            "measures against. A reference described as something else is a different question"
        )


def _validate_ablation(record: Mapping[str, object], where: str) -> None:
    """Fusion against extraction, with both deltas redone from the two summaries."""
    field = _count(record, "field_points", where)
    surface = _count(record, "surface_points", where)
    if _validate_distances(record, "field_distances", where) != field:
        raise ReportUnusable(
            f"{where} summarises a different count than its {field} field point(s)"
        )
    if _validate_distances(record, "surface_distances", where) != surface:
        raise ReportUnusable(
            f"{where} summarises a different count than its {surface} surface point(s)"
        )
    field_block = _report_mapping(record.get("field_distances"), f"{where}.field_distances")
    surface_block = _report_mapping(record.get("surface_distances"), f"{where}.surface_distances")
    for key, left, right in (
        ("extraction_p95_delta", "p95", "p95"),
        ("extraction_worst_delta", "worst", "worst"),
    ):
        expected = _report_number(surface_block, left, where) - _report_number(
            field_block, right, where
        )
        if not _close(_report_number(record, key, where), expected):
            raise ReportUnusable(
                f"{where} states {key}={record.get(key)!r}; its own two summaries differ by "
                f"{expected!r}. The ablation is subtraction and is redone here"
            )
    field_near = _fraction(record, "field_within_tolerance_fraction", where)
    surface_near = _fraction(record, "surface_within_tolerance_fraction", where)
    if not _close(
        _report_number(record, "extraction_within_tolerance_delta", where),
        surface_near - field_near,
    ):
        raise ReportUnusable(
            f"{where} extraction_within_tolerance_delta is not its own subtraction"
        )
    without = _count(record, "surface_points_without_field_support", where)
    if without > surface:
        raise ReportUnusable(
            f"{where} finds {without} of {surface} surface point(s) unsupported by the field"
        )
    if not _close(
        _fraction(record, "surface_points_without_field_support_fraction", where),
        (without / surface) if surface else 0.0,
    ):
        raise ReportUnusable(
            f"{where} surface_points_without_field_support_fraction is not its own"
        )
    _count(record, "field_points_in_protected", where)
    _count(record, "surface_points_in_protected", where)
    if _report_text(record, "measured_against", where) != contract.ANALYTIC_SOURCE:
        raise ReportUnusable(
            f"{where} does not measure both stages against the closed-form truth. An ablation "
            "whose two halves are scored differently is a comparison of two references"
        )


def _validate_surface_samples(record: Mapping[str, object], where: str) -> None:
    """The triangle interiors: bounded, complete, and consistent with its own faces."""
    faces = _count(record, "faces", where)
    per_face = _count(record, "samples_per_face", where)
    sampled = _count(record, "sampled_points", where)
    ceiling = _count(record, "sample_ceiling", where)
    order = _report_int(record, "sample_order", where)
    if order < 1 or per_face != order * order:
        raise ReportUnusable(
            f"{where} states order {order} and {per_face} sample(s) per face; the interior "
            "lattice of order n is one centroid per sub-triangle of a regular subdivision, "
            "which is exactly n**2 equal-area points, and a reader does not guess at another "
            "pattern. A lattice with fewer of them could not carry the area claim beside it"
        )
    if sampled != faces * per_face:
        raise ReportUnusable(
            f"{where} sampled {sampled} point(s) over {faces} face(s) at {per_face} each. Every "
            "emitted triangle is sampled, or the measurement chose which of the surface to look "
            "at after seeing it"
        )
    if sampled > ceiling:
        raise ReportUnusable(
            f"{where} sampled {sampled} point(s) against its own ceiling of {ceiling}"
        )
    if _validate_distances(record, "distances", where) != sampled:
        raise ReportUnusable(f"{where} summarises a different count than its {sampled} sample(s)")
    near = _count(record, "within_tolerance_points", where)
    if near > sampled:
        raise ReportUnusable(f"{where} puts {near} of {sampled} sample(s) within tolerance")
    if not _close(
        _fraction(record, "within_tolerance_fraction", where), (near / sampled) if sampled else 0.0
    ):
        raise ReportUnusable(f"{where} within_tolerance_fraction is not {near}/{sampled}")
    unsupported = _count(record, "unsupported_samples", where)
    if unsupported > sampled:
        raise ReportUnusable(f"{where} finds {unsupported} of {sampled} sample(s) unsupported")
    if not _close(
        _fraction(record, "unsupported_fraction", where),
        (unsupported / sampled) if sampled else 0.0,
    ):
        raise ReportUnusable(f"{where} unsupported_fraction is not {unsupported}/{sampled}")
    protected = _count(record, "protected_samples", where)
    bridging = _count(record, "bridging_faces", where)
    hidden = _count(record, "bridging_faces_with_every_vertex_outside", where)
    if protected > sampled:
        raise ReportUnusable(f"{where} puts {protected} of {sampled} sample(s) in free space")
    if bridging > faces or bridging > protected:
        raise ReportUnusable(
            f"{where} counts {bridging} bridging face(s) over {faces} face(s) and {protected} "
            "protected sample(s); a bridging face holds at least one of those samples"
        )
    if hidden > bridging:
        raise ReportUnusable(
            f"{where} says {hidden} of {bridging} bridging face(s) have every vertex outside the "
            "protected volume; that set is a subset of the bridging faces"
        )
    area = _report_number(record, "surface_area", where)
    protected_area = _report_number(record, "protected_area", where)
    if area < 0.0 or protected_area < 0.0 or protected_area > area:
        raise ReportUnusable(
            f"{where} puts {protected_area} of {area} unit(s) of surface area in free space"
        )
    if (protected_area > 0.0) != (protected > 0):
        raise ReportUnusable(
            f"{where} reports {protected} protected sample(s) beside {protected_area} of area "
            "there; the area is those samples' own faces weighted by them"
        )
    if _report_text(record, "sampled_from", where) != contract.SURFACE_SAMPLE_SOURCE:
        raise ReportUnusable(f"{where} describes a sampling this oracle does not perform")


def _validate_topology(record: Mapping[str, object], where: str) -> None:
    """The edge census and the shells, re-checked against each other."""
    vertices = _count(record, "vertices", where)
    referenced = _count(record, "referenced_vertices", where)
    faces = _count(record, "faces", where)
    if referenced > vertices:
        raise ReportUnusable(
            f"{where} says {referenced} of {vertices} vertex/vertices are used by a face"
        )
    if _count(record, "degenerate_faces", where) > faces:
        raise ReportUnusable(f"{where} counts more degenerate faces than faces")
    edges = _count(record, "edge_count", where)
    boundary = _count(record, "boundary_edge_count", where)
    shared = _count(record, "shared_edge_count", where)
    nonmanifold = _count(record, "nonmanifold_edge_count", where)
    reversed_edges = _count(record, "reversed_edge_count", where)
    if boundary + shared + nonmanifold != edges:
        raise ReportUnusable(
            f"{where} accounts for {boundary} + {shared} + {nonmanifold} edge(s) of {edges}. An "
            "edge is used once, twice, or more than twice"
        )
    if reversed_edges > shared:
        raise ReportUnusable(f"{where} counts {reversed_edges} reversed of {shared} shared edge(s)")
    for key, expected in (
        ("closed", edges > 0 and boundary == 0 and nonmanifold == 0),
        ("consistently_oriented", nonmanifold == 0 and reversed_edges == shared),
    ):
        value = record.get(key)
        if not isinstance(value, bool):
            raise ReportUnusable(f"{where} {key!r} is not a boolean")
        if value != expected:
            raise ReportUnusable(
                f"{where} states {key}={value} where its own edge counts give {expected}. Both "
                "are definitions over the counts beside them, not opinions about the mesh"
            )
    if _report_int(record, "euler_characteristic", where) != referenced - edges + faces:
        raise ReportUnusable(f"{where} states an Euler characteristic that is not V - E + F")
    shells = _count(record, "shell_count", where)
    largest = _count(record, "largest_shell_faces", where)
    closed_shells = _count(record, "closed_shells", where)
    unsupported = _count(record, "unsupported_closed_shells", where)
    unsupported_faces = _count(record, "unsupported_closed_shell_faces", where)
    if largest > faces or shells > faces:
        raise ReportUnusable(
            f"{where} counts {shells} shell(s) and {largest} face(s) in the largest"
        )
    if closed_shells > shells or unsupported > closed_shells:
        raise ReportUnusable(
            f"{where} counts {unsupported} unsupported of {closed_shells} closed of {shells} "
            "shell(s); each set is a subset of the one before it"
        )
    if unsupported_faces > faces or (unsupported_faces == 0) != (unsupported == 0):
        raise ReportUnusable(
            f"{where} puts {unsupported_faces} face(s) in {unsupported} unsupported closed "
            "shell(s); a shell holds at least one face"
        )
    # Support and truth were separated on purpose, and this is where a reader
    # checks that they stayed separated. An unsupported shell is one or the
    # other of these and never both, and the two groups account for all of it.
    tolerance = _report_number(record, "accuracy_tolerance", where)
    if tolerance <= 0.0:
        raise ReportUnusable(
            f"{where} splits its unsupported shells at an accuracy tolerance of {tolerance}; "
            "the bar is the fixture's coverage tolerance and is positive"
        )
    false_shells = _count(record, "unsupported_inaccurate_closed_shells", where)
    false_faces = _count(record, "unsupported_inaccurate_closed_shell_faces", where)
    correct_shells = _count(record, "unsupported_accurate_closed_shells", where)
    correct_faces = _count(record, "unsupported_accurate_closed_shell_faces", where)
    if false_shells + correct_shells != unsupported:
        raise ReportUnusable(
            f"{where} splits {unsupported} unsupported closed shell(s) into {false_shells} "
            f"inaccurate and {correct_shells} accurate. The two are disjoint and exhaustive: a "
            "shell nothing measured is either on the object or it is not"
        )
    if false_faces + correct_faces != unsupported_faces:
        raise ReportUnusable(
            f"{where} splits {unsupported_faces} unsupported closed shell face(s) into "
            f"{false_faces} and {correct_faces}. Every face of an unsupported shell belongs to "
            "whichever group its shell does"
        )
    for count, count_faces, key in (
        (false_shells, false_faces, "unsupported_inaccurate"),
        (correct_shells, correct_faces, "unsupported_accurate"),
    ):
        if (count == 0) != (count_faces == 0):
            raise ReportUnusable(
                f"{where} puts {count_faces} face(s) in {count} {key} shell(s); a shell holds "
                "at least one face"
            )
        worst = _report_number(record, f"{key}_worst_distance", where)
        if worst < 0.0:
            raise ReportUnusable(f"{where} states a negative {key}_worst_distance of {worst}")
        if count == 0 and worst != 0.0:
            raise ReportUnusable(
                f"{where} reports no {key} shell and a worst distance of {worst} over them"
            )
    false_worst = _report_number(record, "unsupported_inaccurate_worst_distance", where)
    correct_worst = _report_number(record, "unsupported_accurate_worst_distance", where)
    if false_shells and false_worst <= tolerance:
        raise ReportUnusable(
            f"{where} calls {false_shells} shell(s) inaccurate whose worst analytic distance is "
            f"{false_worst}, inside its own {tolerance} tolerance. That is a shell nothing "
            "measured standing on the object, which is the case this split exists to protect"
        )
    if correct_shells and correct_worst > tolerance:
        raise ReportUnusable(
            f"{where} calls {correct_shells} shell(s) accurate whose worst analytic distance is "
            f"{correct_worst}, outside its own {tolerance} tolerance"
        )
    if not _close(
        _fraction(record, "largest_shell_fraction", where), (largest / faces) if faces else 0.0
    ):
        raise ReportUnusable(f"{where} largest_shell_fraction is not {largest}/{faces}")
    if _report_text(record, "support_source", where) != contract.SUPPORT_SOURCE:
        raise ReportUnusable(
            f"{where} describes its support reference as something other than proximity to the "
            "ideal returns. An unsupported shell is unsupported by MEASUREMENT, not by truth"
        )
    if _report_text(record, "shell_accuracy_source", where) != contract.ANALYTIC_SOURCE:
        raise ReportUnusable(
            f"{where} describes the reference its accurate/inaccurate split is made against as "
            "something other than the closed-form distance to the object. Splitting support by "
            "support would not be a split at all"
        )


def _validate_candidate(
    value: object, gates: Mapping[str, Mapping[str, object]], where: str
) -> tuple[str, str]:
    record = _report_mapping(value, where)
    name = _report_text(record, "candidate", where)
    status = _report_text(record, "status", where)
    if status not in ("pass", "fail"):
        raise ReportUnusable(f"{where} status {status!r} is not 'pass' or 'fail'")
    observed = _report_mapping(record.get("observed"), f"{where}.observed")
    outcome = _report_text(observed, "outcome", f"{where}.observed")
    if outcome not in OUTCOMES:
        raise ReportUnusable(f"{where} observed outcome {outcome!r} is not one of {OUTCOMES}")
    refusal = observed.get("refusal")
    if refusal is not None and (not isinstance(refusal, str) or refusal not in contract.REFUSALS):
        raise ReportUnusable(f"{where} observed refusal {refusal!r} is not a declared refusal")
    if (outcome == "refused") != (refusal is not None):
        raise ReportUnusable(f"{where} observed {outcome!r} beside refusal {refusal!r}")
    mismatches = _report_list(record, "mismatches", where)
    if (status == "pass") != (not mismatches):
        raise ReportUnusable(
            f"{where} calls itself {status!r} with {len(mismatches)} mismatch(es); the status IS "
            "whether the observation matched the declaration"
        )
    report = record.get("quality")
    if outcome == "refused":
        if report is not None:
            raise ReportUnusable(
                f"{where} was refused and still carries measurements. A refused arm is not "
                "scored, and publishing its numbers beside a refusal is how one gets quoted"
            )
        return name, outcome
    if report is None:
        # An arm that RAISED while being scored. The harness records the
        # exception against its own candidate and keeps every other case, so the
        # artifact has to be able to carry that: a reader that refused the whole
        # file here would turn one unscoreable arm into no evidence at all.
        if record.get("error") is None:
            raise ReportUnusable(
                f"{where} published no measurements and no error. An arm is refused, scored, or "
                "raised, and the third of those says what it raised"
            )
        if status == "pass":
            raise ReportUnusable(f"{where} raised while scored and still calls itself a pass")
        return name, outcome
    quality_record = _report_mapping(report, f"{where}.quality")
    measurements = _report_list(quality_record, "measurements", f"{where}.quality")
    names: list[str] = []
    for index, measurement in enumerate(measurements):
        measurement_name, _ = _validate_measurement(
            measurement, f"{where}.quality.measurements[{index}]"
        )
        names.append(measurement_name)
    if len(set(names)) != len(names):
        raise ReportUnusable(f"{where} carries two measurements with one name")
    for index, binding in enumerate(_report_list(record, "gate_bindings", where)):
        bound = _report_mapping(binding, f"{where}.gate_bindings[{index}]")
        gate_name = _report_text(bound, "gate", f"{where}.gate_bindings[{index}]")
        if gate_name not in gates:
            raise ReportUnusable(
                f"{where} binds gate {gate_name!r}, which its case does not declare. A gate a "
                "candidate brought with it is a bar it wrote for itself"
            )
        if _report_text(bound, "measurement", f"{where}.gate_bindings[{index}]") not in names:
            raise ReportUnusable(
                f"{where} binds gate {gate_name!r} to a measurement it did not publish"
            )
    return name, outcome


def _validate_case(value: object, grid: Sequence[Mapping[str, object]], what: str) -> str:
    case = _report_mapping(value, what)
    case_id = _report_text(case, "case_id", what)
    where = f"{what} case {case_id!r}"
    status = _report_text(case, "status", where)
    if status not in STATUSES:
        raise ReportUnusable(f"{where} status {status!r} is not one of {STATUSES}")
    gates: dict[str, Mapping[str, object]] = {}
    for index, gate in enumerate(_report_list(case, "gates", where)):
        record = _report_mapping(gate, f"{where}.gates[{index}]")
        gates[_report_text(record, "name", f"{where}.gates[{index}]")] = record
    if status == "malformed":
        if case.get("candidates") != [] or case.get("error") is None:
            raise ReportUnusable(
                f"{where} is malformed and still reports candidates or no error. A fixture "
                "nobody could read has no results"
            )
        return case_id
    if not gates:
        raise ReportUnusable(f"{where} applies no gates at all")
    seen: dict[str, str] = {}
    for index, candidate in enumerate(_report_list(case, "candidates", where)):
        name, outcome = _validate_candidate(candidate, gates, f"{where}.candidates[{index}]")
        if name in seen:
            raise ReportUnusable(f"{where} reports candidate {name!r} twice")
        seen[name] = outcome
    if "raw_points" not in seen:
        raise ReportUnusable(f"{where} published no raw-points arm to compare fusion against")
    if APPLIANCE_ARM not in seen:
        raise ReportUnusable(
            f"{where} published no {APPLIANCE_ARM!r} arm. The declared grid is coarser than the "
            "resolution the shipped policy chooses, and an artifact without that arm cannot "
            "support a claim about the mesh an owner actually gets"
        )
    missing = sorted(
        f"fused:{cell['cell_id']}" for cell in grid if f"fused:{cell['cell_id']}" not in seen
    )
    if missing:
        raise ReportUnusable(
            f"{where} publishes no result for {missing}. Every case runs every declared grid "
            "cell, and an artifact missing one is a search reported as an envelope"
        )
    recommended = case.get("recommended_cell")
    if recommended is not None and recommended not in {str(cell["cell_id"]) for cell in grid}:
        raise ReportUnusable(f"{where} recommends {recommended!r}, which is not a declared cell")
    return case_id


def _validate_attribution(
    document: Mapping[str, object], cases: Sequence[Mapping[str, object]], what: str
) -> None:
    """Re-check every published delta against the two cases it came from.

    The arithmetic is redone from the artifact's own numbers, so an attribution
    edited after the fact - a delta made smaller, a degraded flag flipped, a
    control swapped for a friendlier case - does not survive being read back.
    """
    by_id = {str(case["case_id"]): case for case in cases if isinstance(case.get("case_id"), str)}
    attributed: set[str] = set()
    for index, entry in enumerate(_report_list(document, "attribution", what)):
        where = f"{what}.attribution[{index}]"
        record = _report_mapping(entry, where)
        case_id = _report_text(record, "case_id", where)
        control_id = _report_text(record, "control_case", where)
        if case_id == control_id:
            raise ReportUnusable(f"{where} names itself as its own control")
        attributed.add(case_id)
        case = by_id.get(case_id)
        control = by_id.get(control_id)
        if case is None or control is None:
            raise ReportUnusable(f"{where} attributes a case the artifact does not carry")
        control_scene = _report_mapping(control.get("scene"), f"{where} control scene")
        control_perturbation = _report_mapping(
            control_scene.get("perturbation"), f"{where} control perturbation"
        )
        if _report_text(control_perturbation, "label", where) != contract.EXACT:
            raise ReportUnusable(
                f"{where} uses a perturbed case as a control. The baseline is the exact case"
            )
        here = _candidates_by_name(case)
        there = _candidates_by_name(control)
        for position, arm in enumerate(_report_list(record, "arms", where)):
            arm_record = _report_mapping(arm, f"{where}.arms[{position}]")
            candidate_name = _report_text(arm_record, "candidate", f"{where}.arms[{position}]")
            if candidate_name not in here or candidate_name not in there:
                raise ReportUnusable(
                    f"{where}.arms[{position}] compares {candidate_name!r}, which is not "
                    "published on both sides"
                )
            perturbed_summary = _summary(here[candidate_name])
            control_summary = _summary(there[candidate_name])
            for slot, metric in enumerate(
                _report_list(arm_record, "metrics", f"{where}.arms[{position}]")
            ):
                spot = f"{where}.arms[{position}].metrics[{slot}]"
                metric_record = _report_mapping(metric, spot)
                key = _report_text(metric_record, "metric", spot)
                if key not in WORSE_WHEN_HIGHER and key not in WORSE_WHEN_LOWER:
                    raise ReportUnusable(
                        f"{spot} attributes {key!r}, which this harness declares no direction "
                        "for. A metric without a stated worse direction cannot say anything "
                        "degraded"
                    )
                if key not in perturbed_summary or key not in control_summary:
                    raise ReportUnusable(f"{spot} attributes {key!r}, which was not measured")
                perturbed = _report_number(metric_record, "perturbed", spot)
                baseline = _report_number(metric_record, "control", spot)
                delta = _report_number(metric_record, "delta", spot)
                if not _close(perturbed, perturbed_summary[key]) or not _close(
                    baseline, control_summary[key]
                ):
                    raise ReportUnusable(
                        f"{spot} states numbers that are not the ones the two cases published"
                    )
                if not _close(delta, perturbed - baseline):
                    raise ReportUnusable(
                        f"{spot} states delta {delta!r} for {perturbed!r} minus {baseline!r}"
                    )
                degraded = metric_record.get("degraded")
                if not isinstance(degraded, bool):
                    raise ReportUnusable(f"{spot} 'degraded' is not a boolean")
                expected = (
                    perturbed > baseline if key in WORSE_WHEN_HIGHER else perturbed < baseline
                )
                if degraded != expected:
                    raise ReportUnusable(
                        f"{spot} calls a move from {baseline!r} to {perturbed!r} "
                        f"degraded={degraded}, and the declared direction says {expected}"
                    )
    for case in cases:
        scene = case.get("scene")
        if not isinstance(scene, Mapping):
            continue
        if isinstance(scene.get("control_case"), str) and str(case["case_id"]) not in attributed:
            raise ReportUnusable(
                f"{what} publishes case {case['case_id']!r} with a control and no attribution "
                "for it. A perturbed case whose comparison went missing is a cause nobody "
                "reported"
            )


def validate_report(
    document: object, *, what: str = "surface-oracle report"
) -> Mapping[str, object]:
    """Everything this artifact must be true of itself before anyone reads it."""
    record = _report_mapping(document, what)
    _no_nonfinite(record, what)
    if record.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ReportUnusable(
            f"{what} states schema version {record.get('schema_version')!r}; this reader "
            f"understands exactly {REPORT_SCHEMA_VERSION}"
        )
    if _report_text(record, "kind", what) != REPORT_KIND:
        raise ReportUnusable(f"{what} is not a {REPORT_KIND}")
    for key, expected in (
        ("arms", list(contract.ARMS)),
        ("perturbations", list(contract.PERTURBATIONS)),
        ("refusals", list(contract.REFUSALS)),
        ("claims", list(contract.CLAIMS)),
        ("worse_when_higher", sorted(WORSE_WHEN_HIGHER)),
        ("worse_when_lower", sorted(WORSE_WHEN_LOWER)),
    ):
        if _report_list(record, key, what) != expected:
            raise ReportUnusable(
                f"{what} states {key!r} that is not this module's own vocabulary. A reader that "
                "accepted a widened vocabulary would accept a result it has no rules for"
            )
    corpus = _report_mapping(record.get("corpus"), f"{what}.corpus")
    if _report_text(corpus, "selection_rule", f"{what}.corpus") != SELECTION_RULE:
        raise ReportUnusable(f"{what} records a selection rule this module does not apply")
    grid = [
        _report_mapping(cell, f"{what}.corpus.grid[{index}]")
        for index, cell in enumerate(_report_list(corpus, "grid", f"{what}.corpus"))
    ]
    if len(grid) < 2:
        raise ReportUnusable(f"{what} records fewer than two grid cells")
    budget = _report_int(corpus, "work_budget", f"{what}.corpus")
    if budget <= 0:
        raise ReportUnusable(f"{what} records a work budget of {budget}")

    cases = _report_list(record, "cases", what)
    identifiers = [_validate_case(case, grid, what) for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ReportUnusable(f"{what} reports a case twice")
    case_records = [case for case in cases if isinstance(case, Mapping)]
    labels = {
        str(_report_mapping(case["scene"], what)["perturbation"]["label"])  # type: ignore[index]
        for case in case_records
        if isinstance(case.get("scene"), Mapping)
    }
    missing = sorted(set(contract.PERTURBATIONS) - labels)
    if missing and not any(str(case.get("status")) == "malformed" for case in case_records):
        raise ReportUnusable(
            f"{what} carries no case for perturbation(s) {missing}, and no malformed case to "
            "explain the absence. A corpus missing its bad controls reports only good news"
        )
    _validate_attribution(record, case_records, what)

    totals = _report_mapping(record.get("totals"), f"{what}.totals")
    statuses = [str(case.get("status")) for case in case_records]
    for key, expected_count in (
        ("cases", len(case_records)),
        ("passed", statuses.count("pass")),
        ("failed", statuses.count("fail")),
        ("malformed", statuses.count("malformed")),
    ):
        if _report_int(totals, key, f"{what}.totals") != expected_count:
            raise ReportUnusable(f"{what}.totals {key!r} does not count its own cases")
    decision = _report_text(record, "decision", what)
    expected_decision = (
        "pass" if case_records and statuses.count("pass") == len(case_records) else "fail"
    )
    if decision != expected_decision:
        raise ReportUnusable(
            f"{what} decides {decision!r} over {statuses.count('pass')} of {len(case_records)} "
            f"passing case(s); the decision is arithmetic and is {expected_decision!r}"
        )
    if not _report_list(record, "limitations", what):
        raise ReportUnusable(f"{what} states no limitations")
    return record


# --- rendering and the command ---------------------------------------------------------


def render(document: Mapping[str, object]) -> str:
    """The whole artifact as text, deterministically, for a log nobody downloads."""
    lines: list[str] = []
    corpus = document["corpus"]
    assert isinstance(corpus, Mapping)
    lines.append(
        f"surface oracle: manifest {corpus['manifest_sha256']} over {corpus['case_count']} case(s)"
    )
    appliance = corpus["appliance"]
    assert isinstance(appliance, Mapping)
    lines.append(
        f"  appliance policy {appliance['policy']}: voxel budget {appliance['voxel_budget']}, "
        f"truncation {appliance['truncation_voxels']} voxels, min weight "
        f"{appliance['min_weight']}, policy {appliance['weight_policy']}, backend "
        f"{appliance['backend']}"
    )
    grid = corpus["grid"]
    assert isinstance(grid, list)
    for cell in grid:
        assert isinstance(cell, Mapping)
        lines.append(
            f"  cell {cell['cell_id']}: voxel {cell['voxel_size']}, truncation "
            f"{cell['truncation']} ({cell['truncation_voxels']} voxels), min weight "
            f"{cell['min_weight']}, policy {cell['weight_policy']}"
        )
    cases = document["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, Mapping)
        lines.append(
            f"[{str(case['status']).upper()}] {case['case_id']} "
            f"({case['fixture']['perturbation'] if isinstance(case['fixture'], Mapping) else '?'})"
            f" recommended cell {case['recommended_cell']}"
        )
        if case["error"]:
            lines.append(f"  error: {case['error']}")
        scene = case["scene"]
        if isinstance(scene, Mapping) and isinstance(scene.get("appliance"), Mapping):
            appliance_record = scene["appliance"]
            assert isinstance(appliance_record, Mapping)
            derived = appliance_record["derived"]
            assert isinstance(derived, Mapping)
            lines.append(
                f"  shipped policy derives voxel {derived['voxel_size']} for this scene, against "
                f"a finest declared cell of {min(float(cell['voxel_size']) for cell in grid)}"
            )
        candidates = case["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert isinstance(candidate, Mapping)
            observed = candidate["observed"]
            assert isinstance(observed, Mapping)
            lines.append(
                f"  [{str(candidate['status']).upper()}] {candidate['candidate']}: "
                f"{observed['outcome']}"
                + (f" ({observed['refusal']})" if observed["refusal"] else "")
            )
            mismatches = candidate["mismatches"]
            assert isinstance(mismatches, list)
            for line in mismatches:
                lines.append(f"    mismatch: {line}")
            report = candidate["quality"]
            if isinstance(report, Mapping):
                gates = report["gates"]
                assert isinstance(gates, list)
                for gate in gates:
                    assert isinstance(gate, Mapping)
                    verdict = "PASS" if gate["passed"] else "FAIL"
                    lines.append(
                        f"    {verdict} {gate['name']}: {gate['measured']} "
                        f"{gate['comparison']} {gate['limit']} ({gate['unit']})"
                    )
    attribution = document["attribution"]
    assert isinstance(attribution, list)
    for entry in attribution:
        assert isinstance(entry, Mapping)
        lines.append(
            f"attribution {entry['case_id']} ({entry['perturbation']}, {entry['family']}) "
            f"against {entry['control_case']}"
        )
        arms = entry["arms"]
        assert isinstance(arms, list)
        for arm in arms:
            assert isinstance(arm, Mapping)
            metrics = arm["metrics"]
            assert isinstance(metrics, list)
            lines.append(
                f"  {arm['candidate']}: {arm['degraded_metrics']} of {len(metrics)} metric(s) "
                "degraded"
            )
            for metric in metrics:
                assert isinstance(metric, Mapping)
                mark = "WORSE" if metric["degraded"] else "     "
                lines.append(
                    f"    {mark} {metric['metric']}: {metric['control']} -> "
                    f"{metric['perturbed']} (delta {metric['delta']})"
                )
    headline = document["raw_versus_fused"]
    assert isinstance(headline, list)
    for entry in headline:
        assert isinstance(entry, Mapping)
        for label in ("grid", "appliance"):
            block = entry[label]
            if not isinstance(block, Mapping):
                lines.append(f"raw versus fused {entry['case_id']} ({label}): not published")
                continue
            lines.append(
                f"raw versus fused {entry['case_id']} ({label}, {block['candidate']}): fusion "
                f"better on {block['fusion_better_on']} of {block['metrics_compared']} metric(s)"
            )
            comparisons = block["comparisons"]
            assert isinstance(comparisons, list)
            for comparison in comparisons:
                assert isinstance(comparison, Mapping)
                mark = "FUSED" if comparison["fusion_better"] else "DOTS "
                lines.append(
                    f"    {mark} {comparison['metric']}: dots {comparison['raw_points']} vs "
                    f"fused {comparison['fused']}"
                )
    totals = document["totals"]
    assert isinstance(totals, Mapping)
    lines.append(
        f"totals: {totals['passed']}/{totals['cases']} case(s), "
        f"{totals['candidates_passed']}/{totals['candidates']} candidate(s), decision "
        f"{document['decision']}"
    )
    limitations = document["limitations"]
    assert isinstance(limitations, list)
    for limitation in limitations:
        lines.append(f"limitation: {limitation}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.surface_oracle_harness",
        description=(
            "Compare the measured points with the raw TSDF surface over exact synthetic truth, "
            "one perturbation at a time, and publish the artifact."
        ),
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="the corpus directory")
    parser.add_argument("--out", type=Path, default=None, help="where to write the artifact")
    parser.add_argument(
        "--quiet", action="store_true", help="publish the artifact without rendering it"
    )
    args = parser.parse_args(argv)

    if args.out is not None:
        # Cleared up front, before the corpus is even loaded: a requested
        # destination must never be left holding an EARLIER report after a run
        # that could not produce one. publish() clears it again itself, which
        # is the write that matters; this one is what turns an unwritable path
        # into exit 2 before any work is done.
        try:
            clear_destination(args.out)
        except ScannerError as error:
            print(f"cannot publish to {args.out}: {error}", file=sys.stderr)
            return 2
    try:
        document = run(args.corpus)
    except (HarnessRefused, ReportUnusable) as error:
        print(f"the corpus could not be run: {error}", file=sys.stderr)
        return 2
    if args.out is not None:
        try:
            publish(document, args.out)
        except ScannerError as error:
            print(f"the artifact could not be published: {error}", file=sys.stderr)
            return 2
    if not args.quiet:
        print(render(document))
    return 0 if document["decision"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover - the command's own entry point
    raise SystemExit(main())
