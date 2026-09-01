"""Replay a real capture through :mod:`scanner3d.quality`, and report what it shows.

Deliberately NOT a ``test_`` module: pytest does not collect it. Every number
below depends on the capture handed to it and on the machine that ran the
alignment - a symmetric cloud agreement or an ICP residual measured against
``tests/fixtures/turntable/`` is not a bound that would mean anything against a
different capture, and asserting one in the suite would produce a test that
fails on the next capture and tells nobody why. This is a harness that is run
when a number is wanted, and its JSON is pasted into whatever handoff wanted
it - the same role :mod:`tests.tsdf_benchmark` plays for fusion cost.

Run it with::

    .venv/bin/python -m tests.quality_replay
    .venv/bin/python -m tests.quality_replay --capture tests/fixtures/turntable
    .venv/bin/python -m tests.quality_replay --register --frames 3
    .venv/bin/python -m tests.quality_replay \\
        --plane-roi 0,50,0,50,150,220 --plane-name "turntable bed"
    .venv/bin/python -m tests.quality_replay --truth truth.json --max-dimension-error 2.0

The capture directory is opened read-only. Every file this harness touches -
the manifest, every ``.bin`` frame, every ``bridge.json`` it finds - is read
with ``Path.read_bytes()``/``Path.read_text()`` and nothing here ever opens a
path in that directory for writing. Nothing here opens a device node, touches
USB, or starts a capture.

What this replays, and what it refuses to claim
------------------------------------------------
The default capture, ``tests/fixtures/turntable/``, is 24 depth frames of a
model truck bed on a turntable, one revolution at roughly 15-degree steps (see
its ``manifest.json``). That is TWENTY-FOUR DIFFERENT VIEWS of a moving object,
not twenty-four repeats of the same view, so:

* Two consecutive frames disagree because the object turned between them, not
  because either frame is wrong. Their raw nearest-neighbour distance is not
  reconstruction error and this harness never reports it as one. The only
  cloud-agreement metric offered is behind ``--register``: it first aligns a
  consecutive pair with the repository's own ICP
  (:func:`scanner3d.registration.icp`) and reports :func:`scanner3d.quality.cloud_agreement`
  of the ALIGNED pair, labelled "agreement after alignment, not accuracy
  against truth" everywhere it appears. Without ``--register`` the case
  carries a ``not measured`` note instead, because there is no honest
  unaligned comparison to make.
* The manifest's ``target.object`` field says "1/18 scale model truck bed,
  roughly 110 x 95 x 24 mm". That sentence is prose from whoever wrote the
  manifest, not a measurement with an instrument behind it, and the depth
  unit it would be expressed in has never been established either
  (:data:`scanner3d.depth_metric.UNVALIDATED_UNIT`). This harness REFUSES to
  turn it into a dimension gate or a truth value anywhere - see
  :data:`MANIFEST_PROSE_IS_NOT_TRUTH` - and reports that refusal as a note
  whenever dimension or landmark truth was not otherwise supplied.
* Named dimensions and landmarks are compared to truth only from an explicit
  ``--truth FILE``. This repository has no automated landmark or named-feature
  detector, so the file must state BOTH the value the caller trusts as
  "measured" and the truth value; see :data:`TRUTH_SCHEMA` for the exact
  shape. Guessing a "measured" value from the cloud here would be exactly the
  kind of invented number this module exists to refuse.
* Plane roughness is measured only for a region the caller names explicitly
  with ``--plane-roi`` and ``--plane-name``. No region is ever chosen for the
  caller.

Gates are opt-in and named after the CLI flag that supplies their limit
(``--max-plane-p95``, ``--max-dimension-error``, ``--max-landmark-offset``,
``--max-agreement-p95``, ``--max-icp-rmse``, ``--require-integrity``,
``--require-bridge-pass``). A flag that is not passed adds no gate; the case's
own notes say so, so a reader cannot mistake "no gate was asked for" for
"the gate passed".

Wall-clock timing is reported ONLY under the top-level ``"timing"`` key of this
tool's own JSON, never inside anything :mod:`scanner3d.quality` serialises.
Timing is evidence about this machine and this run, never a gate and never
part of a deterministic report - see that module's own docstring for why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from scanner3d.depth_metric import UNVALIDATED_UNIT, DepthConversion, deproject_depth_frame
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point
from scanner3d.quality import (
    COORDINATE_UNIT,
    BridgeAudit,
    CaseReport,
    CloudAgreement,
    DimensionErrors,
    Gate,
    LandmarkOffsets,
    PlaneRoughness,
    QualityMeasurement,
    SuiteReport,
    at_most,
    audit_bridge_record,
    cloud_agreement,
    dimension_errors,
    landmark_offsets,
    must_be,
    plane_roughness,
)
from scanner3d.registration import icp
from scanner3d.simulate import FACTORY_INTRINSICS
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES

#: Where a caller finds the shipped fixture if ``--capture`` is not given.
DEFAULT_CAPTURE: Final = Path(__file__).resolve().parent / "fixtures" / "turntable"

#: This capture's own device record, copied from :mod:`tests.tsdf_benchmark`
#: rather than re-derived: both harnesses read frames from the same MINI unit,
#: and a second, independently chosen pair of numbers here would be a second
#: place they could quietly drift apart. See that module's ``CAPTURE_CAMERA``.
INTRINSICS: Final = FACTORY_INTRINSICS
DEPTH_SCALE: Final = 0.004999999888241291

#: Depth clipped to this range before anything downstream sees it, again
#: copied from :mod:`tests.tsdf_benchmark`'s own real-capture path: the
#: manifest states a roughly 250 mm standoff, and depth outside a wide band
#: around that is the sensor's own near/far noise, not the object.
NEAR_CLIP: Final = 60.0
FAR_CLIP: Final = 300.0

#: Every point cloud stride divides a frame's included points by, by default.
#: A frame from this fixture carries 67,000-107,000 non-zero samples
#: (``manifest.json``); this repository's :mod:`scanner3d.quality` refuses a
#: cloud over :data:`scanner3d.quality.MAX_POINTS` (100,000) and refuses a pair
#: whose product of sizes exceeds :data:`scanner3d.quality.MAX_NEAREST_WORK`
#: (25,000,000). Stride 32 brings a frame down to roughly 2,000-3,300 points,
#: comfortably under both bounds for BOTH per-frame statistics and a pairwise
#: comparison - (3,300)^2 is about 10.9 million, well inside the search-work
#: ceiling. It is recorded in every report that uses it rather than assumed.
DEFAULT_STRIDE: Final = 32

#: How many consecutive frames :func:`main` reads by default when a symmetric
#: cloud-agreement pass is requested. Kept small: each additional frame adds
#: one more (bounded) ICP alignment and one more (bounded) cloud comparison,
#: and this harness has no use for a sweep across the whole fixture.
DEFAULT_FRAME_COUNT: Final = 3

#: What this harness refuses to do with the manifest's own prose, quoted in
#: full wherever a truth-dependent case has nothing else to report. Named as a
#: constant, not inlined, so the refusal reads identically everywhere it
#: appears rather than being re-worded per call site.
MANIFEST_PROSE_IS_NOT_TRUTH: Final = (
    "the capture manifest's target.object field ('1/18 scale model truck bed, roughly 110 x "
    "95 x 24 mm') is prose from whoever wrote the manifest, not a measurement taken with an "
    "instrument, and the physical unit a depth sample carries has never been established "
    "either. This harness will not turn that sentence into a dimension, a landmark position "
    "or a gate limit, under any flag"
)

#: The label every cloud-agreement measurement this harness reports carries,
#: because "agreement" and "accuracy" are two different claims and a reader
#: skimming a report should not have to re-derive which one a number is.
AGREEMENT_LABEL: Final = (
    "agreement after alignment, not accuracy against truth: the two frames are different "
    "turntable views of a moving object, aligned here by ICP against each other, never "
    "against an independent reference"
)

#: The shape a ``--truth`` file must have. This repository has no automated
#: named-dimension or named-landmark extractor, so the file states BOTH sides
#: of every comparison rather than only the truth: a "measured" value the
#: caller already trusts, and the "truth" value it is judged against. Inventing
#: the "measured" side from the cloud here - guessing which points are the
#: named corner - is exactly the kind of unearned number this harness exists
#: to refuse; see :data:`MANIFEST_PROSE_IS_NOT_TRUTH` for the sibling refusal.
#:
#: .. code-block:: json
#:
#:     {
#:       "truth_source": "digital calipers, 2026-08-28, Ben",
#:       "dimensions": {
#:         "bed_length": {"measured": 21.6, "truth": 22.0}
#:       },
#:       "landmarks": {
#:         "front_left_corner": {
#:           "measured": [1.0, 2.0, 3.0],
#:           "truth": [1.1, 2.0, 3.2]
#:         }
#:       }
#:     }
#:
#: Either ``dimensions`` or ``landmarks`` may be omitted or empty; a file with
#: neither is refused with a clear error rather than silently measuring
#: nothing.
TRUTH_SCHEMA: Final = "see the module docstring of tests/quality_replay.py"


def sha256_of(data: bytes) -> str:
    """One hash implementation for this harness, so no call site can drift."""
    return hashlib.sha256(data).hexdigest()


def _read_manifest(directory: Path) -> Mapping[str, object] | None:
    """The capture's own ``manifest.json``, or ``None`` if it has none.

    Read-only: ``Path.read_text`` never truncates or creates. A capture
    directory without a manifest is not an error - it simply has no recorded
    sha256 to check frames against, which :func:`frame_inventory` says plainly.
    """
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{path}: top level is {type(manifest).__name__}, not a JSON object")
    return manifest


def _manifest_sha256_by_name(manifest: Mapping[str, object] | None) -> dict[str, str]:
    """Every frame name the manifest names, mapped to its recorded sha256."""
    if manifest is None:
        return {}
    frames = manifest.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, str | bytes):
        return {}
    by_name: dict[str, str] = {}
    for entry in frames:
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            sha = entry.get("sha256")
            if isinstance(sha, str):
                by_name[entry["name"]] = sha
    return by_name


def discover_frames(directory: Path) -> list[Path]:
    """Every ``*.bin`` frame the capture directory holds, sorted by name."""
    return sorted(directory.glob("*.bin"))


def frame_inventory(
    directory: Path, manifest: Mapping[str, object] | None
) -> list[dict[str, object]]:
    """Byte count and sha256 of every frame on disk, checked against the manifest.

    Cheap and always available: this needs nothing but the bytes already on
    disk, so it is reported unconditionally rather than gated behind any flag.
    A frame whose name the manifest does not mention reports
    ``matches_manifest: None`` - "the manifest says nothing about this frame" -
    which is a different fact from a mismatch and is never conflated with one.
    """
    recorded = _manifest_sha256_by_name(manifest)
    rows: list[dict[str, object]] = []
    for path in discover_frames(directory):
        data = path.read_bytes()
        digest = sha256_of(data)
        expected = recorded.get(path.stem)
        rows.append(
            {
                "name": path.stem,
                "bytes": len(data),
                "expected_bytes": EXPECTED_FRAME_BYTES,
                "bytes_as_expected": len(data) == EXPECTED_FRAME_BYTES,
                "sha256": digest,
                "manifest_sha256": expected,
                "matches_manifest": None if expected is None else digest == expected,
            }
        )
    return rows


def deproject_frame(raw: bytes) -> DepthConversion:
    """Deproject one raw frame through this fixture's own device record.

    ``UNVALIDATED_UNIT`` on purpose: nothing in this repository has
    established what a converted distance measures, and reaching for a
    different unit assumption here would misstate the frame's own history
    rather than this harness's.
    """
    return deproject_depth_frame(
        raw, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE, unit=UNVALIDATED_UNIT
    )


def clipped_cloud(conversion: DepthConversion, *, stride: int) -> tuple[Point, ...]:
    """The deprojected cloud, depth-clipped and then strided, in that order.

    Clipping first so the stride subsamples the object rather than diluting it
    with the near/far noise :data:`NEAR_CLIP`/:data:`FAR_CLIP` exist to drop;
    striding second because it is the cheap, explicit bound this harness states
    in every report it appears in.
    """
    clipped = [
        (point.x, point.y, point.z) for point in conversion.points if NEAR_CLIP < point.z < FAR_CLIP
    ]
    return tuple(clipped[::stride])


def _extent(points: Sequence[Point]) -> dict[str, list[float]] | None:
    if not points:
        return None
    return {
        "x": [min(p[0] for p in points), max(p[0] for p in points)],
        "y": [min(p[1] for p in points), max(p[1] for p in points)],
        "z": [min(p[2] for p in points), max(p[2] for p in points)],
    }


def cloud_statistics(
    frames: Sequence[tuple[str, bytes]], *, stride: int
) -> tuple[list[dict[str, object]], dict[str, tuple[Point, ...]]]:
    """Per-frame point counts and extents, and the strided clouds behind them.

    Returns the report rows AND the clouds, so callers needing the clouds for
    a later step (cloud agreement) do not deproject the same bytes twice.
    """
    rows: list[dict[str, object]] = []
    clouds: dict[str, tuple[Point, ...]] = {}
    for name, raw in frames:
        conversion = deproject_frame(raw)
        cloud = clipped_cloud(conversion, stride=stride)
        clouds[name] = cloud
        rows.append(
            {
                "name": name,
                "included_by_deprojection": conversion.included,
                "excluded_by_deprojection": conversion.excluded,
                "accounting_holds": conversion.accounting_holds,
                "after_depth_clip_and_stride": len(cloud),
                "near_clip": NEAR_CLIP,
                "far_clip": FAR_CLIP,
                "stride": stride,
                "extent": _extent(cloud),
                "unit": COORDINATE_UNIT,
            }
        )
    return rows, clouds


def _agreement_case(
    frames: Sequence[str],
    clouds: Mapping[str, tuple[Point, ...]],
    *,
    register: bool,
    coverage_tolerance: float,
    max_agreement_p95: float | None,
    max_icp_rmse: float | None,
) -> CaseReport:
    """Consecutive-pair cloud agreement, honest about what it can and cannot see.

    Never runs an unaligned comparison: consecutive turntable frames are
    different views, so their raw disagreement would be motion, not error. See
    the module docstring's "different views" paragraph.
    """
    if len(frames) < 2:
        return CaseReport(
            name="cloud_agreement",
            measurements=(),
            gates=(),
            notes=(
                f"not measured - only {len(frames)} frame(s) were read; at least 2 are needed "
                "to form a consecutive pair",
            ),
        )
    if not register:
        return CaseReport(
            name="cloud_agreement",
            measurements=(),
            gates=(),
            notes=(
                "not measured - consecutive turntable frames are different views of a moving "
                "object (see manifest.json's 'rotation' block), so their unaligned disagreement "
                "is motion between views, not reconstruction error. Pass --register to align "
                "each consecutive pair with scanner3d.registration.icp first and measure "
                f"agreement of the ALIGNED pair instead: {AGREEMENT_LABEL}",
            ),
        )

    measurements: list[QualityMeasurement] = []
    gates: list[Gate] = []
    notes: list[str] = [AGREEMENT_LABEL]
    worst_p95 = -1.0
    worst_rmse = -1.0
    for index in range(len(frames) - 1):
        source_name, target_name = frames[index], frames[index + 1]
        source, target = clouds[source_name], clouds[target_name]
        pair = f"{source_name}->{target_name}"
        fit = icp(source, target, backend="python")
        aligned: tuple[Point, ...] = tuple(fit.transform.apply_all(source))
        agreement: CloudAgreement = cloud_agreement(
            aligned,
            target,
            name=f"{pair} (icp-aligned)",
            coverage_tolerance=coverage_tolerance,
            unit=COORDINATE_UNIT,
            backend="python",
        )
        measurements.append(agreement)
        notes.append(
            f"{pair}: icp rmse {fit.rmse:.4f}, converged={fit.converged}, "
            f"matched_fraction={fit.matched_fraction:.3f}, used_fraction={fit.used_fraction:.3f}"
        )
        worst_p95 = max(
            worst_p95,
            agreement.observed_to_reference.distances.p95,
            agreement.reference_to_observed.distances.p95,
        )
        worst_rmse = max(worst_rmse, fit.rmse)

    if max_agreement_p95 is not None:
        gates.append(
            at_most(
                "worst directed p95 agreement distance across aligned pairs",
                worst_p95,
                max_agreement_p95,
                unit=COORDINATE_UNIT,
                context=(
                    "a large post-alignment residual means either the alignment failed or the "
                    "surface genuinely disagrees between views; check the icp diagnostics in "
                    "this case's notes before assuming either one"
                ),
            )
        )
    if max_icp_rmse is not None:
        gates.append(
            at_most(
                "worst icp rmse across aligned pairs",
                worst_rmse,
                max_icp_rmse,
                unit=COORDINATE_UNIT,
                context="a high rmse means the alignment itself did not converge to a good fit",
            )
        )
    return CaseReport(
        name="cloud_agreement",
        measurements=tuple(measurements),
        gates=tuple(gates),
        notes=tuple(notes),
    )


def _plane_case(
    plane_roi: tuple[float, float, float, float, float, float] | None,
    plane_name: str | None,
    frames: Sequence[tuple[str, bytes]],
    *,
    max_plane_p95: float | None,
) -> CaseReport:
    """Flatness of a caller-named region of the FIRST frame's full cloud.

    Fitted against the first frame's un-strided, depth-clipped cloud rather
    than the downsampled statistics cloud: a plane fit wants every point it can
    get inside the region, and the region a caller states is small enough that
    the full cloud stays well under scanner3d.quality.MAX_POINTS on its own.
    """
    if plane_roi is None or plane_name is None:
        return CaseReport(
            name="plane_roughness",
            measurements=(),
            gates=(),
            notes=("plane roughness: not measured - no planar region was selected",),
        )
    if not frames:
        return CaseReport(
            name="plane_roughness",
            measurements=(),
            gates=(),
            notes=("not measured - no frames were read to select the region from",),
        )
    xmin, xmax, ymin, ymax, zmin, zmax = plane_roi
    _, raw = frames[0]
    conversion = deproject_frame(raw)
    region = tuple(
        (point.x, point.y, point.z)
        for point in conversion.points
        if xmin <= point.x <= xmax and ymin <= point.y <= ymax and zmin <= point.z <= zmax
    )
    roughness: PlaneRoughness = plane_roughness(
        region,
        name=plane_name,
        region=(
            f"points of frame {frames[0][0]!r} with x in [{xmin}, {xmax}], "
            f"y in [{ymin}, {ymax}], z in [{zmin}, {zmax}] (caller-selected roi)"
        ),
        unit=COORDINATE_UNIT,
    )
    gates: list[Gate] = []
    if max_plane_p95 is not None:
        gates.append(
            at_most(
                f"{plane_name} p95 residual",
                roughness.residuals.p95,
                max_plane_p95,
                unit=COORDINATE_UNIT,
                context="a high p95 residual against a region the caller states is flat means "
                "the reconstructed surface is not actually flat there",
            )
        )
    return CaseReport(name="plane_roughness", measurements=(roughness,), gates=tuple(gates))


def _load_truth(path: Path) -> Mapping[str, object]:
    """Parse and structurally check a ``--truth`` file. See :data:`TRUTH_SCHEMA`."""
    truth = json.loads(path.read_text())
    if not isinstance(truth, Mapping):
        raise ValueError(f"{path}: top level is {type(truth).__name__}, not a JSON object")
    if not isinstance(truth.get("truth_source"), str) or not truth["truth_source"].strip():
        raise ValueError(f"{path}: 'truth_source' must be a non-empty string naming the origin")
    dimensions = truth.get("dimensions", {})
    landmarks = truth.get("landmarks", {})
    if not isinstance(dimensions, Mapping) or not isinstance(landmarks, Mapping):
        raise ValueError(f"{path}: 'dimensions' and 'landmarks' must both be JSON objects")
    if not dimensions and not landmarks:
        raise ValueError(f"{path}: neither 'dimensions' nor 'landmarks' names anything to check")
    return truth


def _entry_pair(entries: Mapping[str, object], name: str, *, what: str) -> tuple[object, object]:
    entry = entries[name]
    if not isinstance(entry, Mapping) or "measured" not in entry or "truth" not in entry:
        raise ValueError(
            f"{what} entry {name!r} must be an object with both 'measured' and 'truth'; "
            "this harness has no automated extractor and will not guess the 'measured' side"
        )
    return entry["measured"], entry["truth"]


def _as_float(value: object, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} is {value!r}, not a number")
    return float(value)


def _as_point(value: object, *, what: str) -> Point:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 3:
        raise ValueError(f"{what} is {value!r}, not a 3-element [x, y, z] array")
    x, y, z = (_as_float(component, what=f"{what} component") for component in value)
    return x, y, z


def _named_dimension_pairs(
    entries: Mapping[str, object], *, what: str
) -> tuple[dict[str, float], dict[str, float]]:
    measured: dict[str, float] = {}
    truth: dict[str, float] = {}
    for name in entries:
        raw_measured, raw_truth = _entry_pair(entries, name, what=what)
        measured[name] = _as_float(raw_measured, what=f"{what} {name!r} 'measured'")
        truth[name] = _as_float(raw_truth, what=f"{what} {name!r} 'truth'")
    return measured, truth


def _named_landmark_pairs(
    entries: Mapping[str, object], *, what: str
) -> tuple[dict[str, Point], dict[str, Point]]:
    measured: dict[str, Point] = {}
    truth: dict[str, Point] = {}
    for name in entries:
        raw_measured, raw_truth = _entry_pair(entries, name, what=what)
        measured[name] = _as_point(raw_measured, what=f"{what} {name!r} 'measured'")
        truth[name] = _as_point(raw_truth, what=f"{what} {name!r} 'truth'")
    return measured, truth


def _dimension_and_landmark_case(
    truth_path: Path | None,
    *,
    max_dimension_error: float | None,
    max_landmark_offset: float | None,
) -> CaseReport:
    if truth_path is None:
        return CaseReport(
            name="dimension_and_landmark",
            measurements=(),
            gates=(),
            notes=(
                "landmark and dimension error: not measured - no independently measured truth "
                f"was supplied. {MANIFEST_PROSE_IS_NOT_TRUTH}",
            ),
        )
    truth = _load_truth(truth_path)
    truth_source = str(truth["truth_source"])
    dimensions_raw = truth.get("dimensions") or {}
    landmarks_raw = truth.get("landmarks") or {}
    measurements: list[QualityMeasurement] = []
    gates: list[Gate] = []
    notes: list[str] = [f"truth read from {truth_path}", MANIFEST_PROSE_IS_NOT_TRUTH]

    if dimensions_raw:
        assert isinstance(dimensions_raw, Mapping)
        measured_dims, expected_dims = _named_dimension_pairs(dimensions_raw, what="a dimension")
        dims: DimensionErrors = dimension_errors(
            measured_dims,
            expected_dims,
            name="dimensions",
            truth_source=truth_source,
            unit=COORDINATE_UNIT,
        )
        measurements.append(dims)
        if max_dimension_error is not None:
            gates.append(
                at_most(
                    "p95 absolute dimension error",
                    dims.absolute.p95,
                    max_dimension_error,
                    unit=COORDINATE_UNIT,
                    context="a large dimension error against caller-supplied truth means the "
                    "reconstruction is the wrong size, not merely noisy",
                )
            )
    else:
        notes.append("dimensions: not measured - the truth file named none")

    if landmarks_raw:
        assert isinstance(landmarks_raw, Mapping)
        measured_marks, expected_marks = _named_landmark_pairs(landmarks_raw, what="a landmark")
        marks: LandmarkOffsets = landmark_offsets(
            measured_marks,
            expected_marks,
            name="landmarks",
            truth_source=truth_source,
            unit=COORDINATE_UNIT,
        )
        measurements.append(marks)
        if max_landmark_offset is not None:
            gates.append(
                at_most(
                    "p95 landmark offset",
                    marks.offsets.p95,
                    max_landmark_offset,
                    unit=COORDINATE_UNIT,
                    context="a large landmark offset against caller-supplied truth means a "
                    "named feature is not where it belongs",
                )
            )
    else:
        notes.append("landmarks: not measured - the truth file named none")

    return CaseReport(
        name="dimension_and_landmark",
        measurements=tuple(measurements),
        gates=tuple(gates),
        notes=tuple(notes),
    )


def find_bridge_records(directory: Path) -> list[Path]:
    """Every ``bridge.json`` in ``directory`` itself or one level of subdirectory.

    Matches the layout :mod:`scanner3d.capture_sequence` documents: a paused
    capture publishes ``bridge.json`` inside each ``segment-NNNN`` directory
    directly under the capture root. Deeper nesting is not searched - a bridge
    record does not live anywhere else in a capture this repository writes.
    """
    found: list[Path] = []
    root_candidate = directory / "bridge.json"
    if root_candidate.is_file():
        found.append(root_candidate)
    for child in sorted(directory.iterdir()):
        if child.is_dir():
            candidate = child / "bridge.json"
            if candidate.is_file():
                found.append(candidate)
    return found


def _bridge_case(directory: Path, *, require_pass: bool) -> CaseReport:
    paths = find_bridge_records(directory)
    if not paths:
        return CaseReport(
            name="bridge_audit",
            measurements=(),
            gates=(),
            notes=(f"not measured - no bridge.json found under {directory} or one level below",),
        )
    measurements: list[QualityMeasurement] = []
    gates: list[Gate] = []
    for path in paths:
        raw = path.read_bytes()
        record = json.loads(raw)
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}: top level is {type(record).__name__}, not a JSON object")
        audit: BridgeAudit = audit_bridge_record(
            record,
            name=str(path.relative_to(directory)),
            source=str(path),
            source_sha256=sha256_of(raw),
        )
        measurements.append(audit)
        if require_pass:
            gates.append(
                must_be(
                    f"{path.relative_to(directory)} accepted under current policy",
                    audit.accepted_under_current_policy,
                    True,
                    context="the record's own recorded acceptance is re-judged against TODAY's "
                    "relocalization policy; a false here means a decision made earlier would be "
                    "refused if it were made again",
                )
            )
    return CaseReport(name="bridge_audit", measurements=tuple(measurements), gates=tuple(gates))


def _frame_integrity_case(
    rows: Sequence[Mapping[str, object]], *, require_integrity: bool
) -> CaseReport:
    checked = [row for row in rows if row["matches_manifest"] is not None]
    mismatched = [row for row in checked if row["matches_manifest"] is False]
    notes = [
        f"{len(checked)} of {len(rows)} frame(s) on disk had a manifest sha256 to check against "
        f"({len(mismatched)} mismatch(es)); the remaining {len(rows) - len(checked)} are not "
        "named by the manifest at all, which the manifest field 'matches_manifest: None' says "
        "for each of them individually"
    ]
    gates: list[Gate] = []
    if require_integrity:
        gates.append(
            must_be(
                "every frame on disk matches the manifest's recorded sha256",
                not mismatched and len(checked) == len(rows),
                True,
                context="a mismatch means either the manifest or the frame on disk was changed "
                "after the other was written; find out which before trusting anything derived "
                "from these frames",
            )
        )
    return CaseReport(
        name="frame_integrity", measurements=(), gates=tuple(gates), notes=tuple(notes)
    )


def _parse_roi(text: str) -> tuple[float, float, float, float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"--plane-roi needs 6 comma-separated numbers (xmin,xmax,ymin,ymax,zmin,zmax), "
            f"got {len(parts)}"
        )
    try:
        xmin, xmax, ymin, ymax, zmin, zmax = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--plane-roi: {exc}") from exc
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        raise argparse.ArgumentTypeError(
            "--plane-roi needs min < max on every axis "
            f"(got x:[{xmin},{xmax}] y:[{ymin},{ymax}] z:[{zmin},{zmax}])"
        )
    return xmin, xmax, ymin, ymax, zmin, zmax


def run(
    capture: Path,
    *,
    stride: int,
    frame_count: int,
    register: bool,
    coverage_tolerance: float,
    plane_roi: tuple[float, float, float, float, float, float] | None,
    plane_name: str | None,
    truth_path: Path | None,
    max_plane_p95: float | None,
    max_dimension_error: float | None,
    max_landmark_offset: float | None,
    max_agreement_p95: float | None,
    max_icp_rmse: float | None,
    require_integrity: bool,
    require_bridge_pass: bool,
) -> tuple[dict[str, object], dict[str, float]]:
    """Do the replay. Returns the JSON-able report and a separate timing dict.

    Timing is returned apart from the report on purpose - see the module
    docstring's closing paragraph - so :func:`main` can put it under its own
    top-level key and nothing here is tempted to fold it into a case.
    """
    if not capture.is_dir():
        raise ValueError(
            f"{capture} is not a directory. Point --capture at a capture root; this harness "
            "reads one, it never creates one"
        )
    timing: dict[str, float] = {}
    overall_started = time.perf_counter()

    manifest = _read_manifest(capture)
    inventory_started = time.perf_counter()
    inventory = frame_inventory(capture, manifest)
    timing["frame_inventory_seconds"] = time.perf_counter() - inventory_started

    all_paths = discover_frames(capture)
    selected = all_paths[: max(0, frame_count)]
    read = [(path.stem, path.read_bytes()) for path in selected]

    # A frame the inventory above has already measured as the wrong size cannot
    # be deprojected - deproject_depth_frame refuses it, correctly - and letting
    # that refusal escape would replace the whole report with a traceback over
    # one bad file. The integrity case is where a wrong-sized frame is reported
    # and, under --require-integrity, gated; here it is simply set aside and
    # named, so every other frame is still measured and the reason is in the
    # report rather than on stderr.
    frames: list[tuple[str, bytes]] = []
    set_aside: list[dict[str, object]] = []
    for name, raw in read:
        if len(raw) == EXPECTED_FRAME_BYTES:
            frames.append((name, raw))
            continue
        set_aside.append(
            {
                "name": name,
                "bytes": len(raw),
                "expected_bytes": EXPECTED_FRAME_BYTES,
                "reason": "byte count is not the frame size this device records, so the frame "
                "cannot be deprojected. See frame_integrity for the gate on this",
            }
        )

    stats_started = time.perf_counter()
    stats_rows, clouds = cloud_statistics(frames, stride=stride)
    timing["cloud_statistics_seconds"] = time.perf_counter() - stats_started

    agreement_started = time.perf_counter()
    agreement_case = _agreement_case(
        [name for name, _ in frames],
        clouds,
        register=register,
        coverage_tolerance=coverage_tolerance,
        max_agreement_p95=max_agreement_p95,
        max_icp_rmse=max_icp_rmse,
    )
    timing["cloud_agreement_seconds"] = time.perf_counter() - agreement_started

    plane_started = time.perf_counter()
    plane_case = _plane_case(plane_roi, plane_name, frames, max_plane_p95=max_plane_p95)
    timing["plane_roughness_seconds"] = time.perf_counter() - plane_started

    truth_started = time.perf_counter()
    truth_case = _dimension_and_landmark_case(
        truth_path,
        max_dimension_error=max_dimension_error,
        max_landmark_offset=max_landmark_offset,
    )
    timing["dimension_and_landmark_seconds"] = time.perf_counter() - truth_started

    bridge_started = time.perf_counter()
    bridge_case = _bridge_case(capture, require_pass=require_bridge_pass)
    timing["bridge_audit_seconds"] = time.perf_counter() - bridge_started

    integrity_case = _frame_integrity_case(inventory, require_integrity=require_integrity)

    suite = SuiteReport(
        name=f"quality_replay: {capture}",
        cases=(integrity_case, agreement_case, plane_case, truth_case, bridge_case),
    )
    timing["total_seconds"] = time.perf_counter() - overall_started

    report: dict[str, object] = {
        "capture": str(capture),
        "manifest_present": manifest is not None,
        "frames_on_disk": len(all_paths),
        "frames_read": len(read),
        "frames_measured": len(frames),
        "frames_set_aside": set_aside,
        "frame_inventory": inventory,
        "cloud_statistics": {
            "stride": stride,
            "near_clip": NEAR_CLIP,
            "far_clip": FAR_CLIP,
            "frames": stats_rows,
        },
        "suite": suite.as_dict(),
    }
    return report, timing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=DEFAULT_CAPTURE,
        help="a capture directory to read (read-only)",
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="cloud downsample stride"
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAME_COUNT, dest="frame_count", help="frames to read"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="align each consecutive frame pair with ICP before measuring cloud agreement",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=1.0,
        help="distance within which a point counts as 'covered' for cloud agreement",
    )
    parser.add_argument(
        "--plane-roi", type=_parse_roi, default=None, metavar="XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX"
    )
    parser.add_argument("--plane-name", type=str, default=None)
    parser.add_argument("--truth", type=Path, default=None, dest="truth_path", help=TRUTH_SCHEMA)
    parser.add_argument("--max-plane-p95", type=float, default=None)
    parser.add_argument("--max-dimension-error", type=float, default=None)
    parser.add_argument("--max-landmark-offset", type=float, default=None)
    parser.add_argument("--max-agreement-p95", type=float, default=None)
    parser.add_argument("--max-icp-rmse", type=float, default=None)
    parser.add_argument("--require-integrity", action="store_true")
    parser.add_argument("--require-bridge-pass", action="store_true")
    args = parser.parse_args(argv)

    if (args.plane_roi is None) != (args.plane_name is None):
        parser.error("--plane-roi and --plane-name must be given together")

    # Every refusal below is a statement about the INPUT - a directory that is
    # not one, a manifest or --truth file that is not the shape it must be, a
    # bridge.json that was never a decision. Each already carries a sentence
    # saying what is wrong; a traceback would bury that sentence in a stack the
    # reader did not ask for. Exit 2, distinct from the 1 a failed gate returns,
    # so a caller can tell "the harness could not run" from "the capture failed
    # a gate".
    try:
        report, timing = run(
            args.capture,
            stride=args.stride,
            frame_count=args.frame_count,
            register=args.register,
            coverage_tolerance=args.coverage_tolerance,
            plane_roi=args.plane_roi,
            plane_name=args.plane_name,
            truth_path=args.truth_path,
            max_plane_p95=args.max_plane_p95,
            max_dimension_error=args.max_dimension_error,
            max_landmark_offset=args.max_landmark_offset,
            max_agreement_p95=args.max_agreement_p95,
            max_icp_rmse=args.max_icp_rmse,
            require_integrity=args.require_integrity,
            require_bridge_pass=args.require_bridge_pass,
        )
    except (ValueError, OSError, ScannerError) as exc:
        sys.stderr.write(f"quality_replay: {exc}\n")
        return 2
    report["timing"] = timing
    # Serialised whole before a byte reaches stdout, and strictly: allow_nan is
    # False for the same reason scanner3d.quality.to_json sets it - Python's
    # default emits NaN/Infinity, which are not JSON - and building the string
    # first means a violation refuses the document rather than truncating a
    # half-written one on stdout.
    try:
        document = json.dumps(report, indent=2, sort_keys=False, allow_nan=False)
    except ValueError as exc:  # pragma: no cover - every measured value is checked
        sys.stderr.write(f"quality_replay: the report is not strict JSON: {exc}\n")
        return 2
    sys.stdout.write(document + "\n")
    return 0 if report["suite"]["passed"] else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
