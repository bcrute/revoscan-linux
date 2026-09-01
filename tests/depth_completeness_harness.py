"""The depth-completeness benchmark, as one command.

``tests/depth_completeness.py`` owns the contract: what a completion must say
about every sample, and what is measured about what it said.
``tests/test_depth_completeness.py`` proves those rules in Python. This is the
other half: the scenes live on disk as data
(``tests/fixtures/depth_completeness/cases/``), one command runs every
candidate of every case through the real
:mod:`scanner3d.quality`, :mod:`scanner3d.tsdf` and :mod:`scanner3d.tracking`
authorities, and the run leaves ONE strict JSON artifact naming every case,
every candidate and every expected-versus-observed gate::

    .venv/bin/python -m tests.depth_completeness_harness --out depth-completeness-report.json

Exit codes, kept apart so a caller can tell them apart:

* ``0`` - every candidate of every case produced exactly the outcome its
  fixture declares.
* ``1`` - the harness ran and at least one did not. An unexpected pass, a
  missing expected failure, an extra failure, a corruption control that was
  NOT refused, a malformed fixture and an exception raised while scoring are
  all this.
* ``2`` - the harness could not run at all: the corpus is missing, its manifest
  is not the shape a manifest must be, a case file is not where the manifest
  says, the output path is not one a report may be written to, or the artifact
  could not be published.

Why this is a second command beside ``tests.quality_harness``
------------------------------------------------------------
Deliberately, and not because one authority was inconvenient. Both harnesses
measure through the same :mod:`scanner3d.quality` primitives, use the same
gate factories, and are published through that module's own path-safety and
serialisation code, which this one imports rather than repeats. What differs
is the evidence:

* a quality-harness case is a point cloud or a frozen readiness record, and its
  fixture IS the geometry. A completeness case is a 512000-byte raw depth
  frame family plus per-sample dropout masks and exact poses - hundreds of
  thousands of integers, which is not reviewable as committed JSON. These
  fixtures therefore commit the *scene* and the SHA-256 of the frames it
  renders, and the harness refuses any case whose bytes do not hash to what
  the fixture says.
* a quality-harness case declares one expected outcome. A completeness case
  runs many CANDIDATES against one set of gates, because the whole point is
  that the same bar refuses a dense invented fill and admits a supported one.

Folding either shape into the other would have meant a second schema version
of the accepted quality-harness artifact and a case kind its reader has no
schema for. The two commands stay separate; the measurement authority does
not.

Nothing here touches a scanner, a capture, a display server, a GPU, the network
or the clock. The compute backends are the pure-Python references, the corpus
is committed data, and two runs over an unchanged corpus produce byte-identical
JSON - which is why no duration and no machine identity appears anywhere in the
artifact.

This is a CORRECTNESS oracle over synthetic truth. It proves that a completion
did not gain coverage by inventing geometry. It makes no claim about the
physical scanner - see ``docs/depth-completeness-benchmark.md`` for exactly
what it does and does not prove.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import scanner3d.depth_completion as depth_completion
import scanner3d.depth_temporal as depth_temporal
import scanner3d.quality as quality
import scanner3d.tsdf_input as tsdf_input
from scanner3d.depth_completion import REJECTION_REASONS
from scanner3d.errors import ScannerError
from scanner3d.geometry import SCHEMA_VERSION as GEOMETRY_SCHEMA_VERSION
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.simulate import Box, Plane, Shape, Sphere, intrinsics_of
from scanner3d.simulate import render as render_frame
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH
from tests import depth_completeness as contract
from tests import quality_harness as base

# The strict-JSON reading, path-safety and serialisation primitives are the
# accepted quality harness's. Importing them keeps ONE authority for how a
# fixture is read and how an artifact is published; a second copy of these
# rules would be a second set of rules to keep true.
from tests.quality_harness import (
    FixtureUnusable,
    HarnessRefused,
    ReportUnusable,
    clear_destination,
    publish,
)

#: Where the corpus lives, and the manifest that names its contents.
CORPUS: Final = Path(__file__).parent / "fixtures" / "depth_completeness" / "cases"
MANIFEST_NAME: Final = "manifest.json"

#: The unit every depth-derived measurement here is in, as the contract states it.
DEPTH_UNIT_TEXT: Final = contract.DEPTH_UNIT

FIXTURE_SCHEMA_VERSION: Final = 1
CORPUS_KIND: Final = "scanner3d.depth-completeness-corpus"
CASE_KIND: Final = "scanner3d.depth-completeness-case"

REPORT_SCHEMA_VERSION: Final = 1
REPORT_KIND: Final = "scanner3d.depth-completeness-report"

FIXTURE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*\.json$")
PATH_STEP = re.compile(r"^[a-z][a-z0-9_]*$")

#: How a candidate's outcome may be declared. "refused" is not a failure of a
#: gate: it is a result whose own account of itself is unreadable, and the
#: corruption controls exist to produce exactly it.
OUTCOMES: Final[tuple[str, ...]] = ("pass", "fail", "refused")

#: What a case status can be. "malformed" stays apart from "fail" for the same
#: reason it does in the quality harness: a fixture nobody could read is a
#: different fact from a candidate that behaved unexpectedly.
STATUSES: Final[tuple[str, ...]] = ("pass", "fail", "malformed")

COMPARISONS: Final[Mapping[str, quality.Comparison]] = {
    "<=": quality.Comparison.AT_MOST,
    ">=": quality.Comparison.AT_LEAST,
    "is": quality.Comparison.IS,
}

#: Every measurement kind a case may carry. Four belong to the quality
#: authority and are validated by its own schema in the quality harness; the
#: rest are this benchmark's own.
QUALITY_KINDS: Final[tuple[str, ...]] = (
    "cloud_agreement",
    "plane_roughness",
    "dimension_errors",
    "landmark_offsets",
)
#: The quality authority's own pose measurement. Its type and schema version
#: belong to scanner3d.quality; the quality harness has no schema for it
#: because no case there makes one, so the schema below is this harness's.
POSE_KIND: Final = "pose_sequence_error"

COMPLETENESS_KINDS: Final[tuple[str, ...]] = (
    "depth_completeness_coverage",
    "depth_completeness_reconstructed_error",
    "depth_completeness_confidence",
    "depth_completeness_work",
    "depth_completeness_temporal_work",
    "depth_completeness_confirmation",
    "depth_completeness_edge",
    "depth_completeness_slot",
    "depth_completeness_fusion",
    "depth_completeness_weighted_fusion",
    "depth_completeness_tracking",
)
MEASUREMENT_KINDS: Final[tuple[str, ...]] = (
    *QUALITY_KINDS,
    POSE_KIND,
    *COMPLETENESS_KINDS,
)

#: What this artifact does not say. Written into every report.
LIMITATIONS: Final[tuple[str, ...]] = (
    "Synthetic truth. Every scene is rendered by scanner3d.simulate, which is not a sensor "
    "model: no speckle physics, multipath, subsurface scattering or baseline shadowing. A "
    "candidate that passes here has not been shown to work on the device.",
    "The dropout masks are patterned after shapes seen in a recorded capture. They are not "
    "that capture, and no fraction here is a measurement of the scanner.",
    "Depth is in synthetic units of raw code times depth scale. No physical length, and no "
    "dimensional-accuracy claim, is made anywhere in this artifact.",
    "The temporal control is given the scene's exact poses. It measures how much complementary "
    "evidence posed frames contain, not that any algorithm can recover it from a real capture.",
    "The tracking record is a baseline. This benchmark records what the unchanged tracker did "
    "with each frame's own measured samples; it does not establish that any candidate improved "
    "tracking.",
    "There are two tracking records per candidate and they answer different questions. "
    "'tracking' and 'tracking_pose' are the intended dataflow - the tracker over measured "
    "geometry only, which is the sole route this project's tracker has - and the tracking gates "
    "are stated over them alone. 'tracking_completed_cloud' and 'tracking_completed_cloud_pose' "
    "are a counterfactual replay over the candidate's completed cloud, carry no gate, and are "
    "not evidence that any completion may be fed to a tracker.",
    "The pose-compensated stage is given exact accepted poses, which is what makes it an offline "
    "post-tracking stage here. Its output enters no tracker, no mesh, no export and no live "
    "preview, and that boundary is measured rather than asserted: on the multi-view case the "
    "counterfactual replay shows that feeding its reconstructions to the real tracker moves "
    "where that tracker places a frame past the bar the baseline sets, even though every one of "
    "those reconstructions is within a single raw code of the truth.",
    "The 'weighted_fusion' records are the one place a reconstruction reaches a fused volume, "
    "and they reach an offline volume this artifact builds for the comparison. No export, live "
    "preview or product path fuses anything but raw frames: scanner3d.reconstruction_export is "
    "unchanged and still integrates the original raw bytes. A fusion weight is evidence about "
    "PROVENANCE and not about accuracy, and no weighting here has been shown to improve a scan "
    "of a real object on any device.",
    "The weighted-fusion result on this corpus is NEGATIVE and no gain is gated. Against the "
    "same raw multi-view field - same frames, same poses, same volume, same extractor - the "
    "accepted temporal result changes the observed-voxel count, the extracted surface and the "
    "truth-referenced coverage by exactly zero, while making 225 more voxel updates. The "
    "'withheld_support_*' fields show the same reconstructions adding 225 observed voxels when "
    "the supporting views are withheld, which is why the redundancy reading is the honest one: "
    "what the temporal stage recovers is what those views measured, and a scan fuses them too. "
    "Nothing here supports routing completed samples into an export that already fuses every "
    "raw frame.",
    "Passing every gate here is necessary and not sufficient. A completion also has to be "
    "judged on physical evidence, which this artifact does not contain.",
)


# --- strict reading, on top of the quality harness's primitives -----------------------


def _mapping(
    value: object, what: str, *, error: type[ScannerError] = FixtureUnusable
) -> Mapping[str, object]:
    return base._mapping(value, what, error=error)


def _exact_keys(
    record: Mapping[str, object],
    keys: Sequence[str],
    what: str,
    *,
    error: type[ScannerError] = FixtureUnusable,
) -> None:
    base._exact_keys(record, keys, what, error=error)


def _text(record: Mapping[str, object], key: str, what: str) -> str:
    return base._text(record, key, what)


def _bool(record: Mapping[str, object], key: str, what: str) -> bool:
    return base._bool(record, key, what)


def _number(record: Mapping[str, object], key: str, what: str) -> float:
    return base._number(record, key, what)


def _strings(record: Mapping[str, object], key: str, what: str) -> tuple[str, ...]:
    return base._strings(record, key, what)


def _loads(text: str, what: str, *, error: type[ScannerError] = FixtureUnusable) -> object:
    return base._loads(text, what, error=error)


def _integer(record: Mapping[str, object], key: str, what: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, which is not a whole number")
    return value


def _point(record: Mapping[str, object], key: str, what: str) -> Point:
    value = record.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise FixtureUnusable(f"{what} {key!r} is not three coordinates")
    coordinates = tuple(base._as_number(item, f"{what} {key!r}") for item in value)
    return (coordinates[0], coordinates[1], coordinates[2])


def _digest_text(record: Mapping[str, object], key: str, what: str) -> str:
    value = _text(record, key, what)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, which is not a sha256 digest")
    return value


# --- the deterministic masks ----------------------------------------------------------


def pseudo_random(seed: str, count: int) -> tuple[float, ...]:
    """A fixed stream in [0, 1), from a named seed. No module RNG anywhere.

    SHA-256 of ``seed:block`` rather than :mod:`random`, so the corpus does not
    depend on a generator implementation that is free to change between Python
    releases. The seed is written into the fixture and the frames it produces
    are committed by digest, so the identity of a mask is checkable and not
    merely reproducible in principle.
    """
    values: list[float] = []
    block = 0
    while len(values) < count:
        digest = hashlib.sha256(f"{seed}:{block}".encode()).digest()
        values.extend(
            int.from_bytes(digest[offset : offset + 4], "big") / 4294967296.0
            for offset in range(0, 32, 4)
        )
        block += 1
    return tuple(values[:count])


MASK_KINDS: Final[tuple[str, ...]] = (
    "speckle",
    "blob",
    "edge_band",
    "strip",
    "never_return",
    "complementary",
)


def mask_indices(
    spec: Mapping[str, object],
    *,
    window: contract.Window,
    truth: Sequence[int],
    seed: str,
    what: str,
) -> set[int]:
    """The window samples one mask removes. Six shapes, none of them random.

    Each shape is a distinct failure the recorded capture actually shows -
    isolated speckle, a connected interior hole, dropout hugging a depth edge,
    a one-sided strip collapse, a region that never returns at all, and
    complementary intermittent loss across posed frames. One overall dropout
    percentage would hide every one of these differences behind one number.
    """
    kind = _text(spec, "kind", what)
    if kind not in MASK_KINDS:
        raise FixtureUnusable(f"{what} states mask kind {kind!r}, which this harness has none of")
    if kind == "speckle":
        _exact_keys(spec, ("kind", "fraction"), what)
        fraction = _number(spec, "fraction", what)
        if not 0.0 < fraction < 1.0:
            raise FixtureUnusable(f"{what} states a speckle fraction of {fraction!r}")
        stream = pseudo_random(f"{seed}:speckle", window.sample_count)
        return {index for index, value in enumerate(stream) if value < fraction}
    if kind == "blob":
        _exact_keys(spec, ("kind", "column", "row", "radius"), what)
        column = _integer(spec, "column", what)
        row = _integer(spec, "row", what)
        radius = _number(spec, "radius", what)
        if radius <= 0.0:
            raise FixtureUnusable(f"{what} states a blob radius of {radius!r}")
        chosen: set[int] = set()
        for index in range(window.sample_count):
            r, c = divmod(index, window.width)
            if math.hypot(c - column, r - row) <= radius:
                chosen.add(index)
        return chosen
    if kind == "edge_band":
        _exact_keys(spec, ("kind", "foreground_below", "width"), what)
        threshold = _integer(spec, "foreground_below", what)
        width = _integer(spec, "width", what)
        if width <= 0:
            raise FixtureUnusable(f"{what} states an edge band {width} column(s) wide")
        chosen = set()
        for _v, indices in window.rows():
            columns = [
                column
                for column, index in enumerate(indices)
                if truth[index] != 0 and truth[index] <= threshold
            ]
            if not columns:
                continue
            boundary = max(columns)
            for column, index in enumerate(indices):
                if abs(column - boundary) < width:
                    chosen.add(index)
        return chosen
    if kind in ("strip", "never_return"):
        _exact_keys(spec, ("kind", "first_column", "last_column"), what)
        first = _integer(spec, "first_column", what)
        last = _integer(spec, "last_column", what)
        if not 0 <= first <= last < window.width:
            raise FixtureUnusable(f"{what} states columns {first}..{last} of {window.width}")
        chosen = set()
        for _v, indices in window.rows():
            chosen.update(indices[first : last + 1])
        return chosen
    _exact_keys(spec, ("kind", "period", "phase"), what)
    period = _integer(spec, "period", what)
    phase = _integer(spec, "phase", what)
    if period < 2:
        raise FixtureUnusable(f"{what} states a complementary period of {period}")
    chosen = set()
    for _v, indices in window.rows():
        for column, index in enumerate(indices):
            if (column + phase) % period == 0:
                chosen.add(index)
    return chosen


# --- the scene a fixture declares -----------------------------------------------------


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
        translation = _point(spec, "translation", what)
        return Box(
            half_extents=_point(spec, "half_extents", what),
            pose=Rigid.of(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                translation,
            ),
        )
    raise FixtureUnusable(f"{what} states shape kind {kind!r}, which this harness cannot build")


def _render_pose(spec: Mapping[str, object], what: str) -> Rigid:
    """A pose stated as an exact axis, angle, centre of rotation and shift.

    Constructive rather than a matrix of nine written-out numbers: the pose
    that produced a frame is then reproducible to the last bit from something
    a reviewer can read.

    ``translation`` is optional and additive: the pose is the stated rotation,
    then displaced by that exact vector in world coordinates. It exists because
    a pure lateral camera shift is the one motion whose image displacement can
    be worked out on paper - ``shift * fx / depth`` pixels - and a case that
    wants to assert an exact source pixel against an exact target pixel needs
    that. A pose spec without the key is exactly what it was before, so every
    case written before this one is unchanged.
    """
    keys = set(spec)
    if keys == {"axis", "degrees", "through"}:
        _exact_keys(spec, ("axis", "degrees", "through"), what)
        shift = (0.0, 0.0, 0.0)
    else:
        _exact_keys(spec, ("axis", "degrees", "through", "translation"), what)
        shift = _point(spec, "translation", what)
    degrees = _number(spec, "degrees", what)
    turned = rotation_about_axis(
        _point(spec, "axis", what),
        math.radians(degrees),
        through=_point(spec, "through", what),
    )
    if shift == (0.0, 0.0, 0.0):
        return turned
    return Rigid.of(
        turned.rotation,
        (
            turned.translation[0] + shift[0],
            turned.translation[1] + shift[1],
            turned.translation[2] + shift[2],
        ),
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


def _window(spec: Mapping[str, object], what: str) -> contract.Window:
    _exact_keys(spec, ("u0", "v0", "width", "height", "sample_count"), what)
    window = contract.Window(
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


def _fusion(spec: object, what: str) -> contract.FusionSpec | None:
    if spec is None:
        return None
    record = _mapping(spec, what)
    _exact_keys(
        record,
        (
            "minimum",
            "maximum",
            "voxel_size",
            "truncation",
            "min_depth",
            "max_depth",
            "min_weight",
            "coverage_tolerance",
        ),
        what,
    )
    return contract.FusionSpec(
        minimum=_point(record, "minimum", what),
        maximum=_point(record, "maximum", what),
        voxel_size=_number(record, "voxel_size", what),
        truncation=_number(record, "truncation", what),
        min_depth=_number(record, "min_depth", what),
        max_depth=_number(record, "max_depth", what),
        min_weight=_number(record, "min_weight", what),
        coverage_tolerance=_number(record, "coverage_tolerance", what),
    )


def _regions(spec: Mapping[str, object], what: str) -> contract.Regions:
    _exact_keys(
        spec,
        (
            "plane",
            "foreground_below",
            "slot_columns",
            "surface_tolerance",
            "dimensions",
            "landmarks",
            "fusion",
            "weighted_fusion",
            "tracking",
        ),
        what,
    )
    foreground = spec.get("foreground_below")
    tolerance = spec.get("surface_tolerance")
    columns = spec.get("slot_columns")
    if not isinstance(columns, list) or any(
        isinstance(column, bool) or not isinstance(column, int) for column in columns
    ):
        raise FixtureUnusable(f"{what} 'slot_columns' is not a list of column numbers")
    return contract.Regions(
        plane=_bool(spec, "plane", what),
        foreground_below=(None if foreground is None else _integer(spec, "foreground_below", what)),
        slot_columns=tuple(columns),
        surface_tolerance=(None if tolerance is None else _number(spec, "surface_tolerance", what)),
        dimensions=_bool(spec, "dimensions", what),
        landmarks=_bool(spec, "landmarks", what),
        fusion=_fusion(spec.get("fusion"), f"{what} 'fusion'"),
        weighted_fusion=_fusion(spec.get("weighted_fusion"), f"{what} 'weighted_fusion'"),
        tracking=_bool(spec, "tracking", what),
    )


def build_scene(document: Mapping[str, object], what: str) -> contract.Scene:
    """Materialise a case's frames from its declared scene, and check them.

    The frames are not committed as data - a family of 512000-byte frames is
    not reviewable JSON. What is committed is the scene, the masks, the seed
    and the SHA-256 of every truth and input frame they produce, and this
    function refuses the case if what it renders does not hash to what the
    fixture says. A change to the renderer, the mask rule or the scene shows
    up here as a refusal rather than as a quietly different benchmark.
    """
    scene_spec = _mapping(document.get("scene"), f"{what} 'scene'")
    _exact_keys(
        scene_spec,
        ("intrinsics", "depth_scale", "window", "shapes", "seed", "frames"),
        f"{what} 'scene'",
    )
    intrinsics = _intrinsics(
        _mapping(scene_spec.get("intrinsics"), f"{what} 'intrinsics'"), f"{what} 'intrinsics'"
    )
    depth_scale = _number(scene_spec, "depth_scale", f"{what} 'scene'")
    if not depth_scale > 0.0:
        raise FixtureUnusable(f"{what} states a depth scale of {depth_scale!r}")
    window = _window(_mapping(scene_spec.get("window"), f"{what} 'window'"), f"{what} 'window'")
    shape_specs = scene_spec.get("shapes")
    if not isinstance(shape_specs, list) or not shape_specs:
        raise FixtureUnusable(f"{what} names no shapes")
    shapes = [
        _shape(_mapping(spec, f"{what} shape {index}"), f"{what} shape {index}")
        for index, spec in enumerate(shape_specs)
    ]
    seed = _text(scene_spec, "seed", f"{what} 'scene'")
    frame_specs = scene_spec.get("frames")
    if not isinstance(frame_specs, list) or not frame_specs:
        raise FixtureUnusable(f"{what} names no frames")

    frames: list[contract.FrameEvidence] = []
    for index, spec in enumerate(frame_specs):
        where = f"{what} frame {index}"
        record = _mapping(spec, where)
        _exact_keys(
            record,
            (
                "index",
                "pose",
                "masks",
                "truth_sha256",
                "input_sha256",
                "truth_returned",
                "input_returned",
            ),
            where,
        )
        if _integer(record, "index", where) != index:
            raise FixtureUnusable(f"{where} states an index that is not its position")
        pose_spec = _mapping(record.get("pose"), f"{where} 'pose'")
        render_pose = _render_pose(pose_spec, f"{where} 'pose'")
        rendered = render_frame(
            shapes, intrinsics=intrinsics, depth_scale=depth_scale, pose=render_pose
        )
        truth_codes = contract.codes_of(rendered.raw, window)
        truth_frame = contract.frame_of_codes(truth_codes, window)
        mask_specs = record.get("masks")
        if not isinstance(mask_specs, list):
            raise FixtureUnusable(f"{where} 'masks' is not a list")
        removed: set[int] = set()
        for position, mask in enumerate(mask_specs):
            removed |= mask_indices(
                _mapping(mask, f"{where} mask {position}"),
                window=window,
                truth=truth_codes,
                seed=f"{seed}:frame{index}:mask{position}",
                what=f"{where} mask {position}",
            )
        input_codes = [
            0 if position in removed else code for position, code in enumerate(truth_codes)
        ]
        input_frame = contract.frame_of_codes(input_codes, window)
        _expect_digest(truth_frame, record, "truth_sha256", where)
        _expect_digest(input_frame, record, "input_sha256", where)
        _expect_count(truth_codes, record, "truth_returned", where)
        _expect_count(input_codes, record, "input_returned", where)
        frames.append(
            contract.FrameEvidence(
                index=index,
                truth=truth_frame,
                raw=input_frame,
                pose=render_pose.inverse(),
            )
        )
    return contract.Scene(
        case_id=_text(document, "case_id", what),
        window=window,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        frames=tuple(frames),
        regions=_regions(
            _mapping(document.get("regions"), f"{what} 'regions'"), f"{what} 'regions'"
        ),
    )


def _expect_digest(frame: bytes, record: Mapping[str, object], key: str, where: str) -> None:
    stated = _digest_text(record, key, where)
    actual = hashlib.sha256(frame).hexdigest()
    if actual != stated:
        raise FixtureUnusable(
            f"{where} {key} is {stated}; the scene it declares renders {actual}. A frame that is "
            "not the frame the fixture was reviewed with is not evidence"
        )


def _expect_count(codes: Sequence[int], record: Mapping[str, object], key: str, where: str) -> None:
    stated = _integer(record, key, where)
    actual = sum(1 for code in codes if code != 0)
    if actual != stated:
        raise FixtureUnusable(f"{where} {key} is {stated} and the frame returns {actual}")


# --- the corpus -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FixtureFile:
    """One case file, with the identity the manifest records for it."""

    name: str
    sha256: str
    bytes_: int
    case_id: str


@dataclass(frozen=True, slots=True)
class Corpus:
    """A manifest that has been checked against the directory beside it."""

    root: Path
    manifest_sha256: str
    files: tuple[FixtureFile, ...]


def _safe_child(root: Path, name: str) -> Path:
    """The file ``name`` inside ``root``, or a refusal. Same rules as the corpus next door."""
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


def load_corpus(root: Path = CORPUS) -> Corpus:
    """Read and check the manifest against what is actually on disk, both ways."""
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
            ("schema_version", "kind", "what", "why", "units", "generated_by", "files"),
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
        entries = document.get("files")
        if not isinstance(entries, list) or not entries:
            raise FixtureUnusable(f"{manifest_path} names no files")
        files: list[FixtureFile] = []
        for index, entry in enumerate(entries):
            what = f"{manifest_path} entry {index}"
            record = _mapping(entry, what)
            _exact_keys(record, ("name", "case_id", "bytes", "sha256"), what)
            size = _integer(record, "bytes", what)
            if size <= 0:
                raise FixtureUnusable(f"{what} bytes {size!r} is not a positive integer")
            files.append(
                FixtureFile(
                    name=_text(record, "name", what),
                    sha256=_digest_text(record, "sha256", what),
                    bytes_=size,
                    case_id=_text(record, "case_id", what),
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
    )


# --- one case, and the candidates it runs ---------------------------------------------


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
    """One candidate a case runs, and why it is in the case."""

    name: str
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
                f"{what} names gate(s) {unknown} in {label!r}, and the case states no such gate. "
                "An expectation naming a gate that does not exist can never be satisfied"
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
            "candidate must say which gates fail, and a passing one must name none - otherwise "
            "'something failed' would count as the expected failure"
        )
    return Expectation(
        outcome=outcome,
        failed_gates=failed,
        inapplicable_gates=inapplicable,
        refusal=None if refusal is None else str(refusal),
    )


def load_case(path: Path, *, entry: FixtureFile) -> CaseSpec:
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
    gate_specs = document.get("gates")
    if not isinstance(gate_specs, list) or not gate_specs:
        raise FixtureUnusable(
            f"{what} states no gates. A case that applies no gate passes trivially and is not "
            "evidence of anything"
        )
    gates: list[GateSpec] = []
    seen: set[str] = set()
    for index, spec in enumerate(gate_specs):
        gate = _gate_spec(_mapping(spec, f"{what} gate {index}"), f"{what} gate {index}")
        if gate.name in seen:
            raise FixtureUnusable(
                f"{what} states two gates called {gate.name!r}; an expectation naming one of "
                "them could not say which"
            )
        seen.add(gate.name)
        gates.append(gate)

    candidate_specs = document.get("candidates")
    if not isinstance(candidate_specs, list) or not candidate_specs:
        raise FixtureUnusable(f"{what} runs no candidates")
    candidates: list[CandidateSpec] = []
    named: set[str] = set()
    for index, spec in enumerate(candidate_specs):
        where = f"{what} candidate {index}"
        record = _mapping(spec, where)
        _exact_keys(record, ("name", "what", "expectation"), where)
        name = _text(record, "name", where)
        if name not in contract.CANDIDATES:
            raise FixtureUnusable(
                f"{where} runs candidate {name!r}, which tests/depth_completeness.py does not "
                "define. A fixture cannot name behaviour nobody wrote"
            )
        if name in named:
            raise FixtureUnusable(f"{what} runs candidate {name!r} twice")
        named.add(name)
        candidates.append(
            CandidateSpec(
                name=name,
                what=_text(record, "what", where),
                expectation=_expectation(
                    _mapping(record.get("expectation"), f"{where} 'expectation'"),
                    gates,
                    f"{where} 'expectation'",
                ),
            )
        )
    return CaseSpec(
        case_id=case_id,
        what=_text(document, "what", what),
        why=_text(document, "why", what),
        notes=_strings(document, "notes", what),
        scene=build_scene(document, what),
        gates=tuple(gates),
        candidates=tuple(candidates),
    )


# --- running one candidate ------------------------------------------------------------


def _resolve(value: object, path: Sequence[str], what: str) -> object:
    """Walk a gate's declared path into a measurement's own serialised form."""
    current = value
    for step in path:
        if not isinstance(current, Mapping) or step not in current:
            raise FixtureUnusable(f"{what}: no {step!r} at {list(path)}")
        current = current[step]
    return current


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """What one candidate did, beside what its fixture said it must do."""

    name: str
    what: str
    expectation: Expectation
    outcome: str
    failed_gates: tuple[str, ...]
    inapplicable_gates: tuple[str, ...]
    refusal: str | None
    report: quality.CaseReport | None
    bindings: tuple[base.GateBinding, ...]
    error: str | None


def _mismatch_lines(expected: Expectation, observed: CandidateOutcome) -> list[str]:
    """Every way an observed candidate differs from its declaration.

    All of them, not the first. Derived from expected-versus-observed by the
    same function the harness writes with and the reader checks with, so an
    artifact carrying invented findings - or none where the two disagree - is
    refused rather than believed.
    """
    lines = base._mismatch_lines(
        expected.outcome,
        expected.failed_gates,
        observed.outcome,
        observed.failed_gates,
    )
    if expected.refusal != observed.refusal:
        lines.append(f"expected refusal {expected.refusal!r} and observed {observed.refusal!r}")
    expected_gaps, observed_gaps = (
        set(expected.inapplicable_gates),
        set(observed.inapplicable_gates),
    )
    lines.extend(
        f"gate {name!r} was expected not to apply and was evaluated"
        for name in sorted(expected_gaps - observed_gaps)
    )
    lines.extend(
        f"gate {name!r} did not apply and was expected to be evaluated"
        for name in sorted(observed_gaps - expected_gaps)
    )
    if observed.error is not None:
        lines.append(f"scoring raised: {observed.error}")
    return lines


def run_candidate(case: CaseSpec, spec: CandidateSpec) -> CandidateOutcome:
    """Run one candidate against the case's own gates and report what happened.

    A candidate that :mod:`tests.depth_completeness` refuses is recorded as
    refused with its exact refusal code and is NOT scored: a metric computed
    from a result whose provenance does not hold would be a number about
    nothing. Any other exception is recorded against this candidate alone, and
    the case's other candidates still run.
    """
    function = contract.CANDIDATES[spec.name]
    try:
        result = function(case.scene)
        measurements = contract.measure(case.scene, result)
    except contract.CandidateRefused as refusal:
        return CandidateOutcome(
            name=spec.name,
            what=spec.what,
            expectation=spec.expectation,
            outcome="refused",
            failed_gates=(),
            inapplicable_gates=tuple(gate.name for gate in case.gates),
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
            inapplicable_gates=tuple(gate.name for gate in case.gates),
            refusal=None,
            report=None,
            bindings=(),
            error=f"{type(error).__name__}: {error}",
        )

    by_name = {measurement.name: measurement for measurement in measurements}
    gates: list[quality.Gate] = []
    bindings: list[base.GateBinding] = []
    inapplicable: list[str] = []
    for gate in case.gates:
        measurement = by_name.get(gate.measurement)
        if measurement is None:
            # Not a silent skip: the fixture declares exactly which gates a
            # candidate leaves inapplicable, and a candidate that dodges one
            # by measuring nothing still has to meet the coverage gates.
            inapplicable.append(gate.name)
            continue
        measured = _resolve(
            measurement.as_dict(), gate.path, f"{case.case_id}:{spec.name} gate {gate.name!r}"
        )
        bindings.append(
            base.GateBinding(gate=gate.name, measurement=gate.measurement, path=gate.path)
        )
        if gate.comparison is quality.Comparison.IS:
            if not isinstance(measured, bool):
                raise FixtureUnusable(
                    f"{case.case_id} gate {gate.name!r} compares {measured!r} with 'is', but it "
                    "is not a boolean. An ordering belongs under <= or >="
                )
            gates.append(
                quality.must_be(gate.name, measured, bool(gate.limit), context=gate.context)
            )
            continue
        value = base._as_number(measured, f"{case.case_id} gate {gate.name!r} measured value")
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
        "regions": scene.regions.as_dict(),
        "frames": [
            {
                "index": frame.index,
                "truth_sha256": hashlib.sha256(frame.truth).hexdigest(),
                "input_sha256": hashlib.sha256(frame.raw).hexdigest(),
                "truth_returned": sum(1 for code in scene.truth_codes(frame.index) if code != 0),
                "input_returned": sum(1 for code in scene.input_codes(frame.index) if code != 0),
                "pose": frame.pose.as_dict(),
            }
            for frame in scene.frames
        ],
    }


def _case_entry(entry: FixtureFile, case: CaseSpec | None, error: str | None) -> dict[str, object]:
    fixture = {"name": entry.name, "sha256": entry.sha256, "bytes": entry.bytes_}
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
            "error": error,
        }
    candidates = [_candidate_entry(run_candidate(case, spec)) for spec in case.candidates]
    failed = [entry_ for entry_ in candidates if entry_["status"] != "pass"]
    return {
        "case_id": case.case_id,
        "fixture": fixture,
        "status": "pass" if not failed else "fail",
        "what": case.what,
        "why": case.why,
        "notes": list(case.notes),
        "scene": _scene_entry(case.scene),
        "gates": [gate.as_dict() for gate in case.gates],
        "candidates": candidates,
        "error": None,
    }


def run(root: Path = CORPUS) -> dict[str, object]:
    """Run every candidate of every case and return the artifact as a mapping.

    Cases are independent and every one of them is attempted: an exception out
    of a fixture or a measurement is recorded against its own case and the next
    case still runs, so the artifact preserves all failures rather than the
    first. ``BaseException`` is deliberately not caught - a KeyboardInterrupt
    is the operator ending the run, not a case failing.
    """
    corpus = load_corpus(root)
    entries: list[dict[str, object]] = []
    for fixture in corpus.files:
        path = _safe_child(root, fixture.name)
        try:
            # Reading the case AND running it, inside one guard: a gate whose
            # declared path does not resolve is a broken fixture like any
            # other, and it belongs against its own case rather than ending the
            # run before the later cases have been attempted.
            entry = _case_entry(fixture, load_case(path, entry=fixture), None)
        except Exception as error:
            entry = _case_entry(fixture, None, f"{type(error).__name__}: {error}")
        entries.append(entry)

    statuses = [str(entry["status"]) for entry in entries]
    candidate_entries: list[Mapping[str, object]] = []
    for entry in entries:
        reported = entry["candidates"]
        assert isinstance(reported, list)
        candidate_entries.extend(
            candidate for candidate in reported if isinstance(candidate, Mapping)
        )
    totals = {
        "cases": len(entries),
        "passed": statuses.count("pass"),
        "failed": statuses.count("fail"),
        "malformed": statuses.count("malformed"),
        "candidates": len(candidate_entries),
        "candidates_passed": sum(
            1 for candidate in candidate_entries if candidate["status"] == "pass"
        ),
    }
    document: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "corpus": {
            "kind": CORPUS_KIND,
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "manifest_sha256": corpus.manifest_sha256,
            "case_count": len(corpus.files),
        },
        "provenance_states": list(contract.STATES),
        "refusals": list(contract.REFUSALS),
        "cases": entries,
        "totals": totals,
        "decision": "pass" if totals["passed"] == totals["cases"] and entries else "fail",
        "limitations": list(LIMITATIONS),
    }
    validate_report(document)
    return document


# --- reading a published artifact back, strictly ---------------------------------------


def loads_report(text: str, *, what: str = "depth-completeness report") -> Mapping[str, object]:
    """Read an artifact from its TEXT, strictly, and validate it.

    From the text, not from an already-decoded mapping: ``{"decision": "fail",
    "decision": "pass"}`` is one mapping by the time an ordinary decoder is
    done with it, and no validator over that mapping can tell that the file
    said both. ``NaN`` and ``Infinity`` are refused for the same reason.
    """
    return validate_report(_loads(text, what, error=ReportUnusable), what=what)


def read_report(path: Path, *, what: str | None = None) -> Mapping[str, object]:
    """Read an artifact from disk through :func:`loads_report`."""
    where = what or str(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReportUnusable(f"{where} could not be read: {error}") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportUnusable(f"{where} is not UTF-8 text: {error}") from error
    return loads_report(text, what=where)


def _list_field(record: Mapping[str, object], key: str, where: str) -> list[object]:
    """A list-valued field, which may be empty.

    Beside the quality harness's own reader rather than instead of it: that one
    refuses an empty list, which is right for a case that must carry gates and
    wrong for a candidate that was refused before any gate could be evaluated.
    """
    value = record.get(key)
    if not isinstance(value, list):
        raise ReportUnusable(f"{where} {key!r} is {value!r}, which is not a list")
    return value


def _count_field(record: Mapping[str, object], key: str, where: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportUnusable(f"{where} {key!r} is {value!r}, which is not a count")
    return value


def _fraction_matches(
    stated: float, numerator: int, denominator: int, where: str, key: str
) -> None:
    expected = 0.0 if denominator == 0 else numerator / denominator
    if not base._close(stated, expected):
        raise ReportUnusable(
            f"{where} states {key} {stated!r}; {numerator} of {denominator} is {expected!r}. A "
            "fraction adjusted away from its own counts is not a measurement"
        )


def _validate_coverage(record: Mapping[str, object], where: str, scene: _SceneView) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "domain_samples",
            "truth_surface",
            "truth_empty",
            "input_measured",
            "input_dropout",
            "measured",
            "spatially_reconstructed",
            "temporally_reconstructed",
            "reconstructed",
            "missing",
            "reconstructed_on_surface",
            "false_fill",
            "temporally_reconstructed_on_surface",
            "supported",
            "measured_fraction",
            "reconstructed_fraction",
            "missing_fraction",
            "supported_fraction",
            "false_fill_fraction",
            "dropout_recovered_fraction",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    domain = _count_field(record, "domain_samples", where)
    if domain <= 0:
        raise ReportUnusable(f"{where} scores {domain} sample(s)")
    counts = {
        key: _count_field(record, key, where)
        for key in (
            "truth_surface",
            "truth_empty",
            "input_measured",
            "input_dropout",
            "measured",
            "spatially_reconstructed",
            "temporally_reconstructed",
            "reconstructed",
            "missing",
            "reconstructed_on_surface",
            "false_fill",
            "temporally_reconstructed_on_surface",
            "supported",
        )
    }
    states = (
        counts["measured"]
        + counts["spatially_reconstructed"]
        + counts["temporally_reconstructed"]
        + counts["missing"]
    )
    if states != domain:
        raise ReportUnusable(
            f"{where} accounts for {states} sample(s) of {domain}; every sample of the window is "
            "measured, reconstructed or missing, and the denominator is fixed"
        )
    if counts["truth_surface"] + counts["truth_empty"] != domain:
        raise ReportUnusable(f"{where} splits its truth into more or less than the window")
    if counts["reconstructed"] != (
        counts["spatially_reconstructed"] + counts["temporally_reconstructed"]
    ):
        raise ReportUnusable(f"{where} totals its reconstructions to something else")
    if counts["reconstructed_on_surface"] + counts["false_fill"] != counts["reconstructed"]:
        raise ReportUnusable(
            f"{where} reconstructs {counts['reconstructed']} sample(s) and places "
            f"{counts['reconstructed_on_surface']} on surface beside {counts['false_fill']} in "
            "empty space; every reconstruction is one or the other"
        )
    if counts["supported"] != counts["measured"] + counts["reconstructed_on_surface"]:
        raise ReportUnusable(f"{where} states supported coverage that is not its own counts")
    # The temporal share of the on-surface recovery is bound from both sides, so
    # it cannot be moved on its own. It is a subset of the temporal
    # reconstructions and a subset of the on-surface ones, and everything the
    # spatial reconstructions cannot account for has to be in it. Shrinking it
    # to dodge a recovery gate therefore forces a matching lie about a count
    # this schema already binds to the scene.
    temporal_on_surface = counts["temporally_reconstructed_on_surface"]
    for key in ("temporally_reconstructed", "reconstructed_on_surface"):
        if temporal_on_surface > counts[key]:
            raise ReportUnusable(
                f"{where} places {temporal_on_surface} temporal reconstruction(s) on real "
                f"surface against a {key} of {counts[key]}; that population is a subset of this "
                "one and cannot outnumber it"
            )
    if temporal_on_surface < counts["reconstructed_on_surface"] - counts["spatially_reconstructed"]:
        raise ReportUnusable(
            f"{where} places {counts['reconstructed_on_surface']} reconstruction(s) on real "
            f"surface with only {counts['spatially_reconstructed']} spatial reconstruction(s) in "
            f"the result, and calls {temporal_on_surface} of them temporal; every on-surface "
            "sample the spatial ones cannot account for is a temporal one"
        )
    if counts["input_dropout"] > counts["truth_surface"]:
        raise ReportUnusable(f"{where} loses more samples than its truth ever had")
    # The populations above are mutually exclusive partitions of two fixed
    # sets: the input the candidate was given, and the truth behind it. Adding
    # up is not enough on its own - a count can stay internally consistent and
    # still claim more measurements than were ever handed over, or place more
    # surface than the scene has. These are the inequalities that follow.
    if counts["input_measured"] + counts["input_dropout"] != counts["truth_surface"]:
        raise ReportUnusable(
            f"{where} was given {counts['input_measured']} measurement(s) and lost "
            f"{counts['input_dropout']} of a truth surface of {counts['truth_surface']}; every "
            "sample of that surface either survived its dropout mask or did not"
        )
    if counts["measured"] > counts["input_measured"]:
        raise ReportUnusable(
            f"{where} calls {counts['measured']} sample(s) measured out of "
            f"{counts['input_measured']} the input returned. A measurement is a raw sample that "
            "came back, and a candidate cannot report more of them than it was given"
        )
    for key, limit in (
        ("reconstructed_on_surface", "truth_surface"),
        ("false_fill", "truth_empty"),
        ("supported", "truth_surface"),
    ):
        if counts[key] > counts[limit]:
            raise ReportUnusable(
                f"{where} states {key} {counts[key]} against a {limit} of {counts[limit]}; these "
                "populations are samples of that set and cannot outnumber it"
            )
    for key, numerator, denominator in (
        ("measured_fraction", counts["measured"], domain),
        ("reconstructed_fraction", counts["reconstructed"], domain),
        ("missing_fraction", counts["missing"], domain),
        ("supported_fraction", counts["supported"], domain),
        ("false_fill_fraction", counts["false_fill"], counts["truth_empty"]),
        (
            "dropout_recovered_fraction",
            counts["reconstructed_on_surface"],
            counts["input_dropout"],
        ),
    ):
        _fraction_matches(
            base._report_number(record, key, where), numerator, denominator, where, key
        )
    # And now against the scene the report publishes beside it. Every one of
    # these is stated twice in the artifact, once as geometry and once as a
    # measurement, and a candidate is free to edit neither: the domain is the
    # window times the frames, the truth population is what the scene's frames
    # return, and the dropout is exactly what the masks took away.
    if domain != scene.domain_samples:
        raise ReportUnusable(
            f"{where} scores {domain} sample(s) beside a scene of {scene.frame_count} frame(s) of "
            f"{scene.sample_count}; the denominator is the window times the frames and never "
            "what a candidate chose to look at"
        )
    for key, published, what in (
        ("truth_surface", scene.truth_returned, "return before their dropout mask"),
        ("input_measured", scene.input_returned, "return after their dropout mask"),
        ("input_dropout", scene.dropout, "lose to their dropout mask"),
    ):
        if counts[key] != published:
            raise ReportUnusable(
                f"{where} states {key} {counts[key]} beside a scene whose frames {what} "
                f"{published}. The same fact stated twice has to be the same fact"
            )


def _validate_reconstructed_error(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "entry_count",
            "dropout_samples",
            "reconstructed_on_surface",
            "attempted_fraction",
            "absolute",
            "relative",
            "unit",
            "truth_source",
        ),
        where,
        error=ReportUnusable,
    )
    entries = _count_field(record, "entry_count", where)
    if entries <= 0:
        raise ReportUnusable(f"{where} summarises {entries} error(s)")
    dropout = _count_field(record, "dropout_samples", where)
    on_surface = _count_field(record, "reconstructed_on_surface", where)
    if entries != on_surface:
        raise ReportUnusable(
            f"{where} scores {entries} sample(s) beside {on_surface} reconstructed onto real "
            "surface; the error population is exactly the reconstructions with truth behind them"
        )
    _fraction_matches(
        base._report_number(record, "attempted_fraction", where),
        entries,
        dropout,
        where,
        "attempted_fraction",
    )
    unit = base._report_text(record, "unit", where)
    base._report_text(record, "truth_source", where)
    base._validate_distribution(
        record.get("absolute"), f"{where} 'absolute'", count=entries, unit=unit
    )
    relative = record.get("relative")
    if relative is not None:
        base._validate_distribution(
            relative, f"{where} 'relative'", count=entries, unit=quality.FRACTION_UNIT
        )


def _validate_edge(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "rows_scored",
            "band_low",
            "band_high",
            "region_samples",
            "reconstructed_in_region",
            "bridging",
            "misplaced",
            "bridging_fraction",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    rows = _count_field(record, "rows_scored", where)
    if rows <= 0:
        raise ReportUnusable(f"{where} scores {rows} row(s) of a depth step")
    low = base._report_number(record, "band_low", where)
    high = base._report_number(record, "band_high", where)
    if not high > low:
        raise ReportUnusable(f"{where} states an empty band between {low!r} and {high!r}")
    region = _count_field(record, "region_samples", where)
    reconstructed = _count_field(record, "reconstructed_in_region", where)
    if reconstructed > region:
        raise ReportUnusable(f"{where} reconstructs more samples than its region holds")
    for key in ("bridging", "misplaced"):
        if _count_field(record, key, where) > reconstructed:
            raise ReportUnusable(
                f"{where} counts more {key} sample(s) than it reconstructed at all"
            )
    _fraction_matches(
        base._report_number(record, "bridging_fraction", where),
        _count_field(record, "bridging", where),
        region,
        where,
        "bridging_fraction",
    )
    base._report_text(record, "unit", where)


def _validate_slot(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "slot_samples",
            "slot_reconstructed",
            "closed_samples",
            "open",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    samples = _count_field(record, "slot_samples", where)
    if samples <= 0:
        raise ReportUnusable(f"{where} states a slot of {samples} sample(s)")
    reconstructed = _count_field(record, "slot_reconstructed", where)
    closed = _count_field(record, "closed_samples", where)
    if reconstructed > samples or closed > reconstructed:
        raise ReportUnusable(f"{where} counts more closed or reconstructed samples than it has")
    if base._report_bool(record, "open", where) != (closed == 0):
        raise ReportUnusable(
            f"{where} calls the slot open beside {closed} sample(s) filled with foreground"
        )
    base._report_text(record, "unit", where)


def _validate_fusion(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "frames_offered",
            "frames_integrated",
            "observed_voxels",
            "unobserved_voxels",
            "surface_extracted",
            "surface_vertices",
            "surface_faces",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    offered = _count_field(record, "frames_offered", where)
    integrated = _count_field(record, "frames_integrated", where)
    if integrated > offered:
        raise ReportUnusable(f"{where} integrated more frames than it was offered")
    observed = _count_field(record, "observed_voxels", where)
    _count_field(record, "unobserved_voxels", where)
    vertices = _count_field(record, "surface_vertices", where)
    faces = _count_field(record, "surface_faces", where)
    extracted = base._report_bool(record, "surface_extracted", where)
    if not extracted and (vertices or faces):
        raise ReportUnusable(f"{where} extracted no surface and reports {vertices} vertices")
    if extracted and not vertices:
        raise ReportUnusable(f"{where} extracted a surface with no vertices")
    if vertices and not observed:
        raise ReportUnusable(
            f"{where} carries a surface with no observed voxel behind it; unobserved space does "
            "not become surface"
        )
    base._report_text(record, "unit", where)


def _validate_weighted_fusion(record: Mapping[str, object], where: str) -> None:
    """What the provenance-weighted TSDF did, read strictly and bound to itself.

    Every arithmetic relation this record can be forged through is recomputed
    here from the other fields of the same record, so a report cannot publish a
    gain it did not make:

    * the four provenance counts partition the samples the frames offered, and
      that total is the frames times the samples in one 640x400 frame;
    * a measurement weighs one, so the offered weight is at least the measured
      count and strictly less than the measured count plus the reconstructed
      one - which is what "strictly below a measurement, and never zero" means
      arithmetically;
    * a field with excluded samples and no reconstructions cannot differ from
      the measured-only field, so ``matches_measured_only`` is bound rather
      than believed;
    * the contribution cannot exceed the voxel updates, because no single
      accepted voxel observation can be worth more than one;
    * an unconfirmed spatial estimate contributes exactly zero, absolutely.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "adapter",
            "frames_offered",
            "frames_integrated",
            "measured_samples",
            "temporally_reconstructed_samples",
            "spatially_excluded_samples",
            "missing_samples",
            "offered_weight",
            "voxel_updates",
            "zero_weight_observations",
            "contribution",
            "spatially_excluded_contribution",
            "observed_voxels",
            "unobserved_voxels",
            "surface_extracted",
            "surface_vertices",
            "surface_faces",
            "matches_measured_only",
            "measured_only_frames",
            "measured_only_observed_voxels",
            "measured_only_surface_vertices",
            "measured_only_surface_coverage",
            "surface_coverage",
            "observed_voxel_gain",
            "surface_vertex_gain",
            "surface_coverage_gain",
            "withheld_support_frames",
            "withheld_support_observed_voxels",
            "withheld_support_weighted_observed_voxels",
            "unweighted_observed_voxels",
            "unweighted_surface_vertices",
            "refusals",
            "unit",
            "counterfactual_is",
            "comparison_is",
        ),
        where,
        error=ReportUnusable,
    )
    adapter = base._report_text(record, "adapter", where)
    for name in adapter.split("+"):
        if name not in tsdf_input.ADAPTERS:
            raise ReportUnusable(
                f"{where} names adapter {name!r}; the accepted weighted input contract states "
                f"{list(tsdf_input.ADAPTERS)}, and a producer nobody implemented is a provenance "
                "nobody checked"
            )
    offered = _count_field(record, "frames_offered", where)
    integrated = _count_field(record, "frames_integrated", where)
    if integrated > offered:
        raise ReportUnusable(f"{where} integrated more frames than it was offered")
    measured = _count_field(record, "measured_samples", where)
    temporal = _count_field(record, "temporally_reconstructed_samples", where)
    excluded = _count_field(record, "spatially_excluded_samples", where)
    missing = _count_field(record, "missing_samples", where)
    if measured + temporal + excluded + missing != offered * tsdf_input.FRAME_SAMPLES:
        raise ReportUnusable(
            f"{where} accounts for {measured + temporal + excluded + missing} sample(s) beside "
            f"{offered} frame(s) of {tsdf_input.FRAME_SAMPLES}; the four provenances partition "
            "the whole frame and a result that is not the whole frame cannot be counted against "
            "it"
        )
    weight = base._report_number(record, "offered_weight", where)
    if weight < float(measured):
        raise ReportUnusable(
            f"{where} offers total weight {weight!r} beside {measured} measurement(s), each of "
            f"which weighs {tsdf_input.MEASURED_WEIGHT}. A measurement cannot be discounted here"
        )
    if temporal and weight <= float(measured):
        raise ReportUnusable(
            f"{where} carries {temporal} temporal reconstruction(s) and offers no more weight "
            "than its measurements alone; an accepted reconstruction weighs strictly more than "
            "nothing"
        )
    if temporal and weight >= float(measured + temporal):
        raise ReportUnusable(
            f"{where} offers total weight {weight!r} beside {measured} measurement(s) and "
            f"{temporal} reconstruction(s); a reconstruction weighs strictly less than the "
            "measurement beside it, so the total is strictly below their count"
        )
    updates = _count_field(record, "voxel_updates", where)
    _count_field(record, "zero_weight_observations", where)
    contribution = base._report_number(record, "contribution", where)
    if contribution > float(updates):
        raise ReportUnusable(
            f"{where} reports a contribution of {contribution!r} over {updates} accepted voxel "
            "observation(s); none of them can be worth more than one"
        )
    if updates and contribution <= 0.0:
        raise ReportUnusable(f"{where} updated {updates} voxel(s) at no weight at all")
    excluded_contribution = base._report_number(record, "spatially_excluded_contribution", where)
    if excluded_contribution != 0.0:
        raise ReportUnusable(
            f"{where} gives unconfirmed single-frame estimates a contribution of "
            f"{excluded_contribution!r}. It is exactly zero in this contract: a hypothesis one "
            "frame cannot tell from a sensor dropout is not fused at a discount"
        )
    observed = _count_field(record, "observed_voxels", where)
    _count_field(record, "unobserved_voxels", where)
    vertices = _count_field(record, "surface_vertices", where)
    faces = _count_field(record, "surface_faces", where)
    extracted = base._report_bool(record, "surface_extracted", where)
    if not extracted and (vertices or faces):
        raise ReportUnusable(f"{where} extracted no surface and reports {vertices} vertices")
    if extracted and not vertices:
        raise ReportUnusable(f"{where} extracted a surface with no vertices")
    if vertices and not observed:
        raise ReportUnusable(
            f"{where} carries a surface with no observed voxel behind it; unobserved space does "
            "not become surface"
        )
    identical = base._report_bool(record, "matches_measured_only", where)
    if temporal and identical:
        raise ReportUnusable(
            f"{where} carries {temporal} temporal reconstruction(s) at positive weight and says "
            "the field is byte-identical to the measured-only one; weight that changes nothing "
            "was not applied"
        )
    if not temporal and not identical:
        raise ReportUnusable(
            f"{where} offered nothing but measurements and excluded estimates and says the field "
            "differs from the measured-only one. Zero weight is absence, so it cannot"
        )
    counterfactual = _count_field(record, "unweighted_observed_voxels", where)
    _count_field(record, "unweighted_surface_vertices", where)
    if excluded and counterfactual < observed:
        raise ReportUnusable(
            f"{where} excluded {excluded} single-frame estimate(s) and reports that fusing them "
            "at uniform weight would observe FEWER voxels; the counterfactual is the same "
            "evidence plus those estimates"
        )
    if not excluded and not temporal and counterfactual != observed:
        raise ReportUnusable(
            f"{where} offered nothing but measurements, and its uniform counterfactual observes "
            f"{counterfactual} voxel(s) beside {observed}. With no reconstruction to weight, the "
            "two are the same fusion of the same codes"
        )
    _validate_weighted_comparison(
        record, where, offered=offered, observed=observed, vertices=vertices, temporal=temporal
    )
    _count_field(record, "refusals", where)
    base._report_text(record, "unit", where)
    base._report_text(record, "counterfactual_is", where)
    base._report_text(record, "comparison_is", where)


def _signed_count(record: Mapping[str, object], key: str, where: str) -> int:
    """A whole number that is allowed to be negative, because a gain can be a loss."""
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportUnusable(f"{where} {key!r} is {value!r}, which is not a whole number")
    return value


def _validate_weighted_comparison(
    record: Mapping[str, object],
    where: str,
    *,
    offered: int,
    observed: int,
    vertices: int,
    temporal: int,
) -> None:
    """The output-facing comparison, bound so it cannot be flattered.

    This is the part of the record that could carry a geometric claim, so every
    way of manufacturing one is closed here:

    * the baseline has to be EVERY frame this record was offered.
      ``measured_only_frames`` is the count it was actually given, and a report
      that fused the weighted target frame against a baseline missing the
      support frames those reconstructions came out of is refused here rather
      than read as a gain;
    * each ``*_gain`` is recomputed from the two numbers it is the difference
      of, so a gain cannot be stated beside measurements that do not produce
      it;
    * a coverage is a fraction of the truth model, and a field that extracted
      no surface covers none of it;
    * with no reconstruction at positive weight the weighted field IS the
      measured-only field, so all three gains must be exactly zero;
    * ``withheld_support_*`` is the comparison this corpus is forbidden to
      claim a gain from. It is bound to be a subset - never observing more of
      the volume than the full baseline - so it cannot be quietly published as
      the baseline itself.
    """
    baseline_frames = _count_field(record, "measured_only_frames", where)
    if baseline_frames != offered:
        raise ReportUnusable(
            f"{where} compares against a raw baseline of {baseline_frames} frame(s) beside the "
            f"{offered} frame(s) it was offered. The baseline is the same frames at the same "
            "poses in the same volume, support included; a baseline with the supporting views "
            "withheld would show a gain that is only the support those views already measured"
        )
    baseline_observed = _count_field(record, "measured_only_observed_voxels", where)
    baseline_vertices = _count_field(record, "measured_only_surface_vertices", where)
    baseline_coverage = base._report_number(record, "measured_only_surface_coverage", where)
    coverage = base._report_number(record, "surface_coverage", where)
    for name, fraction in (
        ("measured_only_surface_coverage", baseline_coverage),
        ("surface_coverage", coverage),
    ):
        if not 0.0 <= fraction <= 1.0:
            raise ReportUnusable(
                f"{where} states {name} {fraction!r}, which is not a fraction of the truth model"
            )
    if not vertices and coverage != 0.0:
        raise ReportUnusable(
            f"{where} extracted no surface and reports covering {coverage!r} of the truth model"
        )
    if not baseline_vertices and baseline_coverage != 0.0:
        raise ReportUnusable(
            f"{where} extracted no baseline surface and reports it covering {baseline_coverage!r}"
        )
    for name, stated, expected in (
        (
            "observed_voxel_gain",
            _signed_count(record, "observed_voxel_gain", where),
            observed - baseline_observed,
        ),
        (
            "surface_vertex_gain",
            _signed_count(record, "surface_vertex_gain", where),
            vertices - baseline_vertices,
        ),
    ):
        if stated != expected:
            raise ReportUnusable(
                f"{where} states {name} {stated} and its own two measurements differ by "
                f"{expected}. A gain is the subtraction, never a number beside it"
            )
    coverage_gain = base._report_number(record, "surface_coverage_gain", where)
    if coverage_gain != coverage - baseline_coverage:
        raise ReportUnusable(
            f"{where} states surface_coverage_gain {coverage_gain!r} beside a coverage of "
            f"{coverage!r} over a baseline of {baseline_coverage!r}, whose difference is "
            f"{coverage - baseline_coverage!r}"
        )
    identical = base._report_bool(record, "matches_measured_only", where)
    if not temporal and (
        record["observed_voxel_gain"] or record["surface_vertex_gain"] or coverage_gain != 0.0
    ):
        raise ReportUnusable(
            f"{where} offered no reconstruction at positive weight and reports a gain over the "
            "measured-only field. Zero weight is absence, so there is nothing there to gain with"
        )
    if identical and (
        record["observed_voxel_gain"] or record["surface_vertex_gain"] or coverage_gain != 0.0
    ):
        raise ReportUnusable(
            f"{where} says its field is byte-identical to the measured-only one and reports a "
            "gain over it"
        )
    withheld_frames = base._positive_count(record, "withheld_support_frames", where)
    if withheld_frames > offered:
        raise ReportUnusable(
            f"{where} withholds support from {withheld_frames} frame(s) of {offered}"
        )
    withheld_raw = _count_field(record, "withheld_support_observed_voxels", where)
    withheld_weighted = _count_field(record, "withheld_support_weighted_observed_voxels", where)
    if withheld_raw > baseline_observed:
        raise ReportUnusable(
            f"{where} fuses {withheld_frames} of {offered} frame(s) raw and observes "
            f"{withheld_raw} voxel(s) beside the full baseline's {baseline_observed}. Fusing "
            "fewer frames cannot observe more of the volume, so this is not that subset"
        )
    if temporal and withheld_weighted < withheld_raw:
        raise ReportUnusable(
            f"{where} carries {temporal} reconstruction(s) and reports that weighting them into "
            f"the same withheld-support volume observes {withheld_weighted} voxel(s) beside "
            f"{withheld_raw} raw. Positive weight does not remove observations"
        )


def _validate_tracking(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "frames_offered",
            "frames_accepted",
            "accepted_all",
            "origins",
            "failure_types",
            "refusals",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    offered = _count_field(record, "frames_offered", where)
    accepted = _count_field(record, "frames_accepted", where)
    if accepted > offered:
        raise ReportUnusable(f"{where} accepted more frames than it was offered")
    if base._report_bool(record, "accepted_all", where) != (accepted == offered):
        raise ReportUnusable(f"{where} states accepted_all beside {accepted} of {offered}")
    origins = base._report_strings(record.get("origins"), f"{where} 'origins'")
    if len(origins) != offered:
        raise ReportUnusable(f"{where} names {len(origins)} origin(s) for {offered} frame(s)")
    failures = base._report_strings(record.get("failure_types"), f"{where} 'failure_types'")
    if len(failures) != offered - accepted:
        raise ReportUnusable(
            f"{where} names {len(failures)} failure type(s) for {offered - accepted} refusal(s)"
        )
    base._report_strings(record.get("refusals"), f"{where} 'refusals'")
    base._report_text(record, "unit", where)


def _validate_pose_sequence(record: Mapping[str, object], where: str) -> None:
    """One :class:`scanner3d.quality.PoseSequenceError`, held to its own construction.

    ``frames`` excludes the gauge frame by construction, so the two summaries
    describe exactly ``pose_count - 1`` frames. A sequence trimmed to drop the
    frame it placed worst breaks that count rather than improving the numbers.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "pose_count",
            "gauge_frame_index",
            "gauge",
            "frames",
            "rotation",
            "translation",
        ),
        where,
        error=ReportUnusable,
    )
    poses = _count_field(record, "pose_count", where)
    if poses < 2:
        raise ReportUnusable(f"{where} compares {poses} pose(s), which is not a sequence")
    gauge_index = _count_field(record, "gauge_frame_index", where)
    if gauge_index != 0:
        raise ReportUnusable(f"{where} removes its gauge at frame {gauge_index}")
    if base._report_text(record, "gauge", where) != quality.GAUGE_STATEMENT:
        raise ReportUnusable(
            f"{where} states a gauge that is not the one scanner3d.quality removes"
        )
    frames = _list_field(record, "frames", where)
    if len(frames) != poses - 1:
        raise ReportUnusable(
            f"{where} summarises {len(frames)} frame(s) of a {poses}-pose sequence; the gauge "
            "frame is excluded and every other frame is present"
        )
    for index, frame in enumerate(frames):
        at = f"{where} frame {index}"
        entry = _mapping(frame, at, error=ReportUnusable)
        _exact_keys(
            entry, ("index", "rotation_degrees", "translation_error"), at, error=ReportUnusable
        )
        if _count_field(entry, "index", at) != index + 1:
            raise ReportUnusable(f"{at} states an index that is not its place in the sequence")
        for key in ("rotation_degrees", "translation_error"):
            if base._report_number(entry, key, at) < 0.0:
                raise ReportUnusable(f"{at} {key} is negative, and it is a magnitude")
    base._validate_distribution(
        record.get("rotation"),
        f"{where} 'rotation'",
        count=len(frames),
        unit=quality.DEGREES_UNIT,
    )
    base._validate_distribution(
        record.get("translation"), f"{where} 'translation'", count=len(frames), unit=DEPTH_UNIT_TEXT
    )


def _validate_confidence(record: Mapping[str, object], where: str) -> None:
    """One :class:`tests.depth_completeness.ConfidenceEvidence`, held to itself.

    Two different populations are being described in one record - every
    reconstruction, and the confident subset of it - and the arithmetic that
    ties them together is what stops a distribution being reported without the
    samples it was worst on. The five bins partition the whole population, the
    confident counts partition themselves, and the means are held inside the
    fraction they are declared to be. Whether the record agrees with COVERAGE
    is checked separately, where coverage is in scope.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "reconstructed",
            "on_surface",
            "false_fill",
            "minimum",
            "maximum",
            "mean",
            "on_surface_mean",
            "false_fill_mean",
            "surface_advantage",
            "threshold",
            "high_confidence",
            "high_confidence_on_surface",
            "high_confidence_false_fill",
            "high_confidence_worst",
            "bins",
            "unit",
            "error_unit",
            "truth_source",
        ),
        where,
        error=ReportUnusable,
    )
    reconstructed = _count_field(record, "reconstructed", where)
    if reconstructed <= 0:
        raise ReportUnusable(
            f"{where} describes the confidence of {reconstructed} reconstruction(s); a "
            "distribution over nothing is not one, and a candidate that reconstructed nothing "
            "makes no confidence measurement at all"
        )
    on_surface = _count_field(record, "on_surface", where)
    false_fill = _count_field(record, "false_fill", where)
    if on_surface + false_fill != reconstructed:
        raise ReportUnusable(
            f"{where} places {on_surface} reconstruction(s) on surface beside {false_fill} in "
            f"empty space, of {reconstructed}; every one of them is one or the other"
        )
    threshold = base._report_number(record, "threshold", where)
    if not base._close(threshold, contract.CONFIDENCE_THRESHOLD):
        raise ReportUnusable(
            f"{where} states a confidence threshold of {threshold!r}; the oracle fixes it at "
            f"{contract.CONFIDENCE_THRESHOLD} so that no candidate can move the line its own "
            "accuracy is judged at"
        )
    confident = _count_field(record, "high_confidence", where)
    confident_surface = _count_field(record, "high_confidence_on_surface", where)
    confident_empty = _count_field(record, "high_confidence_false_fill", where)
    if confident_surface + confident_empty != confident:
        raise ReportUnusable(f"{where} splits its confident population into something else")
    if confident > reconstructed:
        raise ReportUnusable(
            f"{where} calls {confident} of {reconstructed} reconstruction(s) confident"
        )
    if confident_surface > on_surface or confident_empty > false_fill:
        raise ReportUnusable(
            f"{where} states a confident subset larger than the population it is drawn from"
        )
    for key in ("minimum", "maximum", "mean", "on_surface_mean", "false_fill_mean", "threshold"):
        value = base._report_number(record, key, where)
        if not 0.0 <= value <= 1.0:
            raise ReportUnusable(
                f"{where} states {key} {value!r}; every confidence here is a fraction of 1"
            )
    low = base._report_number(record, "minimum", where)
    high = base._report_number(record, "maximum", where)
    mean = base._report_number(record, "mean", where)
    if not low <= mean <= high:
        raise ReportUnusable(
            f"{where} states a mean of {mean!r} outside the {low!r}..{high!r} it also states"
        )
    surface_mean = base._report_number(record, "on_surface_mean", where)
    empty_mean = base._report_number(record, "false_fill_mean", where)
    advantage = base._report_number(record, "surface_advantage", where)
    expected = 0.0 if not on_surface or not false_fill else surface_mean - empty_mean
    if not base._close(advantage, expected):
        raise ReportUnusable(
            f"{where} states a surface advantage of {advantage!r}; the two means beside it give "
            f"{expected!r}, and an advantage over a population that does not exist is not one"
        )
    for key, population in (("on_surface_mean", on_surface), ("false_fill_mean", false_fill)):
        if population == 0 and base._report_number(record, key, where) != 0.0:
            raise ReportUnusable(f"{where} averages {key} over an empty population")
    worst = base._report_number(record, "high_confidence_worst", where)
    if worst < 0.0:
        raise ReportUnusable(f"{where} states a worst error of {worst!r}, and it is a magnitude")
    if confident_surface == 0 and worst != 0.0:
        raise ReportUnusable(
            f"{where} states a worst confident error of {worst!r} with no confident "
            "reconstruction on surface to have made it"
        )
    bins = _list_field(record, "bins", where)
    if len(bins) != len(contract.CONFIDENCE_BIN_EDGES) - 1:
        raise ReportUnusable(
            f"{where} carries {len(bins)} bin(s); the histogram's edges are fixed at "
            f"{list(contract.CONFIDENCE_BIN_EDGES)} and a histogram missing a bin is a "
            "population missing its worst samples"
        )
    total = surface_total = 0
    for index, item in enumerate(bins):
        at = f"{where} bin {index}"
        entry = _mapping(item, at, error=ReportUnusable)
        _exact_keys(
            entry,
            ("low", "high", "count", "on_surface", "false_fill"),
            at,
            error=ReportUnusable,
        )
        for key, edge in (
            ("low", contract.CONFIDENCE_BIN_EDGES[index]),
            ("high", contract.CONFIDENCE_BIN_EDGES[index + 1]),
        ):
            if not base._close(base._report_number(entry, key, at), edge):
                raise ReportUnusable(
                    f"{at} states {key} {entry.get(key)!r} where the fixed edge is {edge!r}"
                )
        count = _count_field(entry, "count", at)
        surface = _count_field(entry, "on_surface", at)
        empty = _count_field(entry, "false_fill", at)
        if surface + empty != count:
            raise ReportUnusable(f"{at} splits {count} sample(s) into {surface} and {empty}")
        total += count
        surface_total += surface
    if total != reconstructed or surface_total != on_surface:
        raise ReportUnusable(
            f"{where} bins {total} reconstruction(s) of {reconstructed}, {surface_total} of them "
            f"on surface of {on_surface}; the bins partition the whole population and dropping "
            "the one a candidate did worst in is not an improvement"
        )
    _bind_confident_subset_to_bins(record, bins, threshold, where)
    for key in ("unit", "error_unit", "truth_source"):
        base._report_text(record, key, where)


def _bind_confident_subset_to_bins(
    record: Mapping[str, object], bins: Sequence[object], threshold: float, where: str
) -> None:
    """The confident counts, bounded by the bins that lie wholly one side of the line.

    The confident subset is the one number in this block a gate is read from, so
    it is the one a candidate has a reason to restate. It is not recomputable
    from the artifact - the individual confidences are not published, and a
    thousand of them per candidate is a report nobody reads - but it is bounded:
    every reconstruction in a bin entirely at or above the threshold is
    confident, and every one in a bin entirely below it is not. Shrinking the
    confident population to escape its accuracy gate has to break one of those.
    """
    below = [_mapping(item, where, error=ReportUnusable) for item in bins]
    at_least = at_least_surface = 0
    at_most_excluded = at_most_excluded_surface = 0
    for entry in below:
        low = base._report_number(entry, "low", where)
        high = base._report_number(entry, "high", where)
        count = _count_field(entry, "count", where)
        surface = _count_field(entry, "on_surface", where)
        if low >= threshold:
            at_least += count
            at_least_surface += surface
        elif high <= threshold:
            at_most_excluded += count
            at_most_excluded_surface += surface
    reconstructed = _count_field(record, "reconstructed", where)
    on_surface = _count_field(record, "on_surface", where)
    for key, floor, ceiling in (
        ("high_confidence", at_least, reconstructed - at_most_excluded),
        (
            "high_confidence_on_surface",
            at_least_surface,
            on_surface - at_most_excluded_surface,
        ),
    ):
        stated = _count_field(record, key, where)
        if not floor <= stated <= ceiling:
            raise ReportUnusable(
                f"{where} states {key} {stated} beside bins that put it between {floor} and "
                f"{ceiling}; a reconstruction in a bin wholly at or above the threshold is "
                "confident, and one in a bin wholly below it is not"
            )


def _validate_work(record: Mapping[str, object], where: str) -> None:
    """One :class:`tests.depth_completeness.WorkEvidence`, held to itself.

    The work record is the only measurement here that describes the RUN rather
    than the output, so this schema checks the arithmetic that is internal to
    it - the reasons add up to the refusals, the refusals and the acceptances
    add up to the holes, the spend is inside a budget the policy determines -
    and leaves the bindings to the output for
    :func:`_bind_optional_measurements`, where coverage is in scope. The policy
    is validated too: a published policy is what makes an accepted sample
    reproducible, and one stating an unbounded radius or a support requirement
    its own radius cannot hold reproduces nothing.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "backend",
            "policy",
            "holes",
            "attempted",
            "accepted",
            "rejected",
            "rejections",
            "support_samples",
            "work_units",
            "work_budget",
            "within_budget",
            "accepted_fraction",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    base._report_text(record, "backend", where)
    base._report_text(record, "unit", where)
    holes = _count_field(record, "holes", where)
    attempted = _count_field(record, "attempted", where)
    accepted = _count_field(record, "accepted", where)
    rejected = _count_field(record, "rejected", where)
    support = _count_field(record, "support_samples", where)
    units = _count_field(record, "work_units", where)
    budget = _count_field(record, "work_budget", where)
    if attempted != accepted + rejected:
        raise ReportUnusable(
            f"{where} attempted {attempted} hole(s) and accounts for {accepted + rejected}"
        )
    if attempted != holes:
        raise ReportUnusable(
            f"{where} attempted {attempted} of {holes} hole(s); a hole nobody looked at is a "
            "refusal with no reason, and a completion is judged on the ones it declined too"
        )
    reasons = _mapping(record.get("rejections"), f"{where} 'rejections'", error=ReportUnusable)
    if sorted(reasons) != sorted(REJECTION_REASONS):
        raise ReportUnusable(
            f"{where} gives reasons {sorted(reasons)}; the vocabulary is fixed at "
            f"{sorted(REJECTION_REASONS)} and a run reporting only the reasons that fired hides "
            "the shape of its own conservatism"
        )
    given = 0
    for reason in sorted(reasons):
        given += _count_field(reasons, reason, f"{where} 'rejections'")
    if given != rejected:
        raise ReportUnusable(f"{where} rejected {rejected} hole(s) and gives reasons for {given}")
    if units > budget:
        raise ReportUnusable(
            f"{where} spent {units} work unit(s) against a budget of {budget}; the budget is "
            "derivable from the input before the run and is not advisory"
        )
    if base._report_bool(record, "within_budget", where) != (units <= budget):
        raise ReportUnusable(f"{where} states within_budget beside {units} of {budget}")
    _fraction_matches(
        base._report_number(record, "accepted_fraction", where),
        accepted,
        holes,
        where,
        "accepted_fraction",
    )
    policy = _mapping(record.get("policy"), f"{where} 'policy'", error=ReportUnusable)
    _validate_completion_policy(policy, f"{where} 'policy'")
    lower = accepted * _count_field(policy, "min_support", f"{where} 'policy'")
    upper = accepted * _count_field(policy, "max_support", f"{where} 'policy'")
    if not lower <= support <= upper:
        raise ReportUnusable(
            f"{where} names {support} support sample(s) for {accepted} reconstruction(s), and "
            f"the policy it publishes allows {lower}..{upper}"
        )
    radius = _count_field(policy, "radius", f"{where} 'policy'")
    if budget != holes * (2 * radius + 1) ** 2:
        raise ReportUnusable(
            f"{where} states a budget of {budget} for {holes} hole(s) at radius {radius}; the "
            "budget is one neighbourhood scan per hole and is not the candidate's to state"
        )


def _validate_completion_policy(record: Mapping[str, object], where: str) -> None:
    """The published completion policy, rebuilt through the product's own type.

    Rebuilt rather than pattern-matched: :class:`scanner3d.depth_completion.CompletionPolicy`
    already refuses an unbounded radius, a non-finite tolerance, a support
    requirement larger than its own radius can hold and a boolean where a count
    belongs, and re-stating those rules here would be a second set of rules to
    keep true. What this adds is that the artifact's policy is exactly a policy
    that type would accept.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "radius",
            "min_support",
            "max_support",
            "min_spread",
            "cluster_gap_codes",
            "max_support_span_codes",
            "max_residual_codes",
            "max_overshoot_codes",
            "max_component_samples",
            "max_component_span",
            "min_confidence",
            "min_code",
            "max_code",
            "passes",
        ),
        where,
        error=ReportUnusable,
    )
    version = record.get("schema_version")
    if version != depth_completion.SCHEMA_VERSION or isinstance(version, bool):
        raise ReportUnusable(
            f"{where} states completion schema version {version!r}; a policy this harness wrote "
            f"states {depth_completion.SCHEMA_VERSION}"
        )
    fields = {key: value for key, value in record.items() if key != "schema_version"}
    try:
        depth_completion.CompletionPolicy(**fields)  # type: ignore[arg-type]
    except (ScannerError, TypeError) as error:
        raise ReportUnusable(
            f"{where} is not a policy scanner3d.depth_completion would accept: {error}"
        ) from error


def _validate_temporal_work(record: Mapping[str, object], where: str) -> None:
    """One pose-compensated run's own account of itself, held to its own arithmetic.

    Beside :func:`_validate_work` rather than folded into it: the two stages
    refuse for different reasons, spend different work against different
    budgets, and one combined record would hide which of them declined what.
    Everything here is checked against the record's own published policy, which
    is rebuilt through :class:`scanner3d.depth_temporal.TemporalPolicy` so that
    the bounds a run says it was held to are exactly bounds that type accepts.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "backend",
            "policy",
            "source_frames",
            "eligible_samples",
            "observations",
            "observation_rejections",
            "candidates",
            "holes",
            "attempted",
            "accepted",
            "rejected",
            "rejections",
            "confirmed",
            "contradicted",
            "not_observed",
            "support_samples",
            "work_units",
            "work_budget",
            "within_budget",
            "accepted_fraction",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    backend = base._report_text(record, "backend", where)
    if backend != depth_temporal.PYTHON_BACKEND:
        raise ReportUnusable(
            f"{where} names backend {backend!r}; {depth_temporal.PYTHON_BACKEND!r} is the only "
            "implementation that module has and so the only one that can have produced a run"
        )
    base._report_text(record, "unit", where)
    counts = {
        key: _count_field(record, key, where)
        for key in (
            "source_frames",
            "eligible_samples",
            "observations",
            "candidates",
            "holes",
            "attempted",
            "accepted",
            "rejected",
            "confirmed",
            "contradicted",
            "not_observed",
            "support_samples",
            "work_units",
            "work_budget",
        )
    }
    if counts["attempted"] != counts["accepted"] + counts["rejected"]:
        raise ReportUnusable(
            f"{where} attempted {counts['attempted']} hole(s) and accounts for "
            f"{counts['accepted'] + counts['rejected']}"
        )
    if counts["attempted"] != counts["holes"]:
        raise ReportUnusable(
            f"{where} attempted {counts['attempted']} of {counts['holes']} hole(s); a hole nobody "
            "looked at is a refusal with no reason"
        )
    # The candidate count is bounded by the observations *and the footprint*,
    # which is checked below where the record's own policy has been rebuilt. A
    # bare "candidates <= observations" is false: one projection is deposited
    # into every target sample of its footprint that it is close enough to, and
    # a point on an exact half-pixel boundary legitimately supports both sides
    # of it.
    if counts["observations"] > counts["eligible_samples"]:
        raise ReportUnusable(
            f"{where} projected {counts['observations']} observation(s) from "
            f"{counts['eligible_samples']} eligible source sample(s); every projection is one "
            "original raw measurement of one other frame, carried once, and a run cannot have "
            "more evidence than the frames returned"
        )
    if counts["confirmed"] + counts["contradicted"] > counts["accepted"]:
        raise ReportUnusable(
            f"{where} confirmed {counts['confirmed']} and contradicted "
            f"{counts['contradicted']} spatial estimate(s) beside {counts['accepted']} temporal "
            "reconstruction(s); either verdict is only reached where independent evidence "
            "settled the sample, so each of them is one of those reconstructions"
        )
    for named, vocabulary in (
        ("rejections", depth_temporal.REJECTION_REASONS),
        ("observation_rejections", depth_temporal.OBSERVATION_REJECTIONS),
    ):
        reasons = _mapping(record.get(named), f"{where} {named!r}", error=ReportUnusable)
        if sorted(reasons) != sorted(vocabulary):
            raise ReportUnusable(
                f"{where} gives {named} {sorted(reasons)}; the vocabulary is fixed at "
                f"{sorted(vocabulary)} and a run reporting only the reasons that fired hides the "
                "shape of its own conservatism"
            )
        given = 0
        for reason in sorted(reasons):
            given += _count_field(reasons, reason, f"{where} {named!r}")
        if named == "rejections" and given != counts["rejected"]:
            raise ReportUnusable(
                f"{where} rejected {counts['rejected']} hole(s) and gives reasons for {given}"
            )
        # Deliberately no "kept plus discarded equals projected" rule here. One
        # projection is deposited into up to a whole footprint of target
        # samples and is judged once per sample it landed in, so those three
        # numbers are counts of different things and an equality between them
        # would refuse a legitimate run at any footprint wider than one cell.
        # What IS structural is bounded below.
    if counts["work_units"] > counts["work_budget"]:
        raise ReportUnusable(
            f"{where} spent {counts['work_units']} work unit(s) against a budget of "
            f"{counts['work_budget']}; the budget is derivable from the input before the run and "
            "is not advisory"
        )
    if base._report_bool(record, "within_budget", where) != (
        counts["work_units"] <= counts["work_budget"]
    ):
        raise ReportUnusable(
            f"{where} states within_budget beside {counts['work_units']} of {counts['work_budget']}"
        )
    _fraction_matches(
        base._report_number(record, "accepted_fraction", where),
        counts["accepted"],
        counts["holes"],
        where,
        "accepted_fraction",
    )
    policy = _mapping(record.get("policy"), f"{where} 'policy'", error=ReportUnusable)
    _validate_temporal_policy(policy, f"{where} 'policy'")
    lower = counts["accepted"] * _count_field(policy, "min_support_frames", f"{where} 'policy'")
    upper = counts["accepted"] * _count_field(policy, "max_support_frames", f"{where} 'policy'")
    if not lower <= counts["support_samples"] <= upper:
        raise ReportUnusable(
            f"{where} names {counts['support_samples']} support sample(s) for "
            f"{counts['accepted']} reconstruction(s), and the policy it publishes allows "
            f"{lower}..{upper}"
        )
    rebuilt = depth_temporal.TemporalPolicy(
        **{key: value for key, value in policy.items() if key != "schema_version"}  # type: ignore[arg-type]
    )
    deposits = counts["observations"] * rebuilt.footprint_cells
    if counts["candidates"] > deposits:
        raise ReportUnusable(
            f"{where} deposited {counts['candidates']} candidate(s) from "
            f"{counts['observations']} projected observation(s), past the {deposits} a footprint "
            f"of {rebuilt.footprint_radius} allows. A candidate is a projection surviving at one "
            "target sample, and one projection may reach every sample of its footprint"
        )
    budget = rebuilt.worst_case_work(counts["eligible_samples"], counts["holes"])
    if counts["work_budget"] != budget:
        raise ReportUnusable(
            f"{where} states a budget of {counts['work_budget']} for "
            f"{counts['eligible_samples']} eligible sample(s) and {counts['holes']} hole(s), "
            f"which its own policy derives {budget} for. The budget is the input and the policy, "
            "not a number the run chose"
        )


def _validate_temporal_policy(record: Mapping[str, object], where: str) -> None:
    """The published temporal policy, rebuilt through the product's own type.

    Rebuilt rather than pattern-matched, for the same reason the completion
    policy is: :class:`scanner3d.depth_temporal.TemporalPolicy` already refuses
    an unreachable support requirement, a non-finite tolerance, a footprint
    past its own ceiling and a boolean where a count belongs, and re-stating
    those rules here would be a second set of rules to keep true.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "min_support_frames",
            "max_support_frames",
            "min_distinct_viewpoints",
            "min_baseline",
            "min_rotation_degrees",
            "footprint_radius",
            "max_reprojection_distance",
            "max_candidates_per_target",
            "max_depth_disagreement_codes",
            "occlusion_margin_codes",
            "cluster_gap_codes",
            "confirmation_tolerance_codes",
            "min_confidence",
            "min_code",
            "max_code",
        ),
        where,
        error=ReportUnusable,
    )
    version = record.get("schema_version")
    if version != depth_temporal.SCHEMA_VERSION or isinstance(version, bool):
        raise ReportUnusable(
            f"{where} states temporal schema version {version!r}; a policy this harness wrote "
            f"states {depth_temporal.SCHEMA_VERSION}"
        )
    fields = {key: value for key, value in record.items() if key != "schema_version"}
    try:
        depth_temporal.TemporalPolicy(**fields)  # type: ignore[arg-type]
    except (ScannerError, TypeError) as error:
        raise ReportUnusable(
            f"{where} is not a policy scanner3d.depth_temporal would accept: {error}"
        ) from error


def _validate_confirmation(record: Mapping[str, object], where: str) -> None:
    """What each confirmation was worth against held-out truth, held to its own totals.

    The three verdicts partition the spatial estimates a run was offered, and
    each one's on-surface and in-empty-space halves are subsets of it. The
    single number this schema exists to keep honest is ``confirmed_false``: it
    cannot be hidden by shrinking ``confirmed``, because the two halves have to
    add up to it.
    """
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "spatial_estimates",
            "confirmed",
            "confirmed_on_surface",
            "confirmed_false",
            "contradicted",
            "contradicted_on_surface",
            "not_observed",
            "not_observed_false",
            "confirmed_fraction",
            "unit",
            "truth_source",
        ),
        where,
        error=ReportUnusable,
    )
    base._report_text(record, "unit", where)
    base._report_text(record, "truth_source", where)
    counts = {
        key: _count_field(record, key, where)
        for key in (
            "spatial_estimates",
            "confirmed",
            "confirmed_on_surface",
            "confirmed_false",
            "contradicted",
            "contradicted_on_surface",
            "not_observed",
            "not_observed_false",
        )
    }
    verdicts = counts["confirmed"] + counts["contradicted"] + counts["not_observed"]
    if verdicts != counts["spatial_estimates"]:
        raise ReportUnusable(
            f"{where} was offered {counts['spatial_estimates']} spatial estimate(s) and reaches "
            f"{verdicts} verdict(s); every estimate is confirmed, contradicted or never observed"
        )
    if counts["confirmed_on_surface"] + counts["confirmed_false"] != counts["confirmed"]:
        raise ReportUnusable(
            f"{where} confirms {counts['confirmed']} estimate(s) and places "
            f"{counts['confirmed_on_surface']} on surface beside {counts['confirmed_false']} in "
            "empty space; every confirmation is one or the other"
        )
    for part, whole in (
        ("contradicted_on_surface", "contradicted"),
        ("not_observed_false", "not_observed"),
    ):
        if counts[part] > counts[whole]:
            raise ReportUnusable(
                f"{where} states {part} {counts[part]} against a {whole} of {counts[whole]}; that "
                "population is a subset of this one"
            )
    _fraction_matches(
        base._report_number(record, "confirmed_fraction", where),
        counts["confirmed"],
        counts["spatial_estimates"],
        where,
        "confirmed_fraction",
    )


#: Every completeness kind whose schema needs nothing but the record itself.
#: Coverage is dispatched apart from this table rather than left out of it by
#: accident: it is the one measurement bound to the published scene, so it
#: takes an argument the others do not.
_COMPLETENESS_SCHEMAS: Final[Mapping[str, Callable[[Mapping[str, object], str], None]]] = {
    POSE_KIND: _validate_pose_sequence,
    "depth_completeness_reconstructed_error": _validate_reconstructed_error,
    "depth_completeness_confidence": _validate_confidence,
    "depth_completeness_work": _validate_work,
    "depth_completeness_temporal_work": _validate_temporal_work,
    "depth_completeness_confirmation": _validate_confirmation,
    "depth_completeness_edge": _validate_edge,
    "depth_completeness_slot": _validate_slot,
    "depth_completeness_fusion": _validate_fusion,
    "depth_completeness_weighted_fusion": _validate_weighted_fusion,
    "depth_completeness_tracking": _validate_tracking,
}

#: A kind this reader has no schema for is refused, so a kind that gained a
#: schema table entry nobody wrote is caught here rather than at read time.
assert set(_COMPLETENESS_SCHEMAS) | {"depth_completeness_coverage"} == {
    *COMPLETENESS_KINDS,
    POSE_KIND,
}


def _validate_measurement(
    value: object, where: str, scene: _SceneView
) -> tuple[str, Mapping[str, object]]:
    """One embedded measurement, held to the exact schema of its own kind.

    The four quality kinds are validated by the quality harness's own schemas -
    the same code that holds the accepted artifact next door to them - and this
    benchmark's own kinds by the schemas above. A kind with no schema is
    refused rather than passed through: a measurement nobody checked is a
    measurement a gate must not be read from.
    """
    record = _mapping(value, where, error=ReportUnusable)
    kind = base._report_text(record, "kind", where)
    if kind not in MEASUREMENT_KINDS:
        raise ReportUnusable(
            f"{where} states measurement kind {kind!r}; this schema states "
            f"{list(MEASUREMENT_KINDS)} and refuses one it has no schema for"
        )
    expected_version = quality.SCHEMA_VERSION if kind in QUALITY_KINDS else contract.SCHEMA_VERSION
    version = record.get("schema_version")
    if version != expected_version or isinstance(version, bool):
        raise ReportUnusable(
            f"{where} states schema version {version!r}; a {kind!r} measurement this harness "
            f"wrote states {expected_version}"
        )
    if kind in QUALITY_KINDS:
        quality_schema = base._MEASUREMENT_SCHEMAS[kind]
        assert callable(quality_schema)
        quality_schema(record, where)
    elif kind == "depth_completeness_coverage":
        _validate_coverage(record, where, scene)
    else:
        _COMPLETENESS_SCHEMAS[kind](record, where)
    return base._report_text(record, "name", where), record


def _outcome_block(value: object, where: str) -> tuple[str, list[str], list[str], str | None]:
    """One expected or observed block, checked for coherence with itself."""
    block = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        block,
        ("outcome", "failed_gates", "inapplicable_gates", "refusal"),
        where,
        error=ReportUnusable,
    )
    outcome = base._report_text(block, "outcome", where)
    if outcome not in OUTCOMES:
        raise ReportUnusable(f"{where} states outcome {outcome!r}")
    failed = base._report_strings(block.get("failed_gates"), f"{where} 'failed_gates'")
    inapplicable = base._report_strings(
        block.get("inapplicable_gates"), f"{where} 'inapplicable_gates'"
    )
    base._sorted_distinct(failed, f"{where} 'failed_gates'")
    base._sorted_distinct(inapplicable, f"{where} 'inapplicable_gates'")
    overlap = sorted(set(failed) & set(inapplicable))
    if overlap:
        raise ReportUnusable(f"{where} states gate(s) {overlap} as both failing and inapplicable")
    refusal = block.get("refusal")
    if refusal is not None and (not isinstance(refusal, str) or refusal not in contract.REFUSALS):
        raise ReportUnusable(f"{where} states refusal {refusal!r}")
    if (outcome == "refused") != (refusal is not None):
        raise ReportUnusable(f"{where} states outcome {outcome!r} beside refusal {refusal!r}")
    if outcome != "fail" and failed:
        raise ReportUnusable(f"{where} states outcome {outcome!r} beside failing gates {failed}")
    return outcome, list(failed), list(inapplicable), None if refusal is None else str(refusal)


def _validate_candidate(
    value: object,
    declared: Sequence[Mapping[str, object]],
    scene: _SceneView,
    case_id: str,
    what: str,
) -> str:
    where = f"{what} candidate"
    entry = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        entry,
        (
            "candidate",
            "what",
            "status",
            "expected",
            "observed",
            "mismatches",
            "error",
            "quality",
            "gate_bindings",
        ),
        where,
        error=ReportUnusable,
    )
    name = base._report_text(entry, "candidate", where)
    where = f"{what} candidate {name!r}"
    if name not in contract.CANDIDATES:
        raise ReportUnusable(f"{where} is not a candidate this benchmark defines")
    base._report_text(entry, "what", where)
    status = base._report_text(entry, "status", where)
    if status not in ("pass", "fail"):
        raise ReportUnusable(f"{where} states status {status!r}")
    expected = _outcome_block(entry.get("expected"), f"{where} 'expected'")
    observed = _outcome_block(entry.get("observed"), f"{where} 'observed'")
    declared_names = {str(gate["name"]) for gate in declared}
    for names, label in (
        (expected[1] + expected[2], "expected"),
        (observed[1] + observed[2], "observed"),
    ):
        unknown = sorted(set(names) - declared_names)
        if unknown:
            raise ReportUnusable(f"{where} {label} names gate(s) {unknown} the case does not state")
    error = entry.get("error")
    if error is not None and not isinstance(error, str):
        raise ReportUnusable(f"{where} states error {error!r}")

    report = entry.get("quality")
    bindings = _list_field(entry, "gate_bindings", where)
    if observed[0] == "refused" or error is not None:
        if report is not None or bindings:
            raise ReportUnusable(
                f"{where} was not scored and still carries measurements or bindings; a result "
                "whose provenance does not hold is never scored"
            )
        if observed[0] == "refused" and sorted(observed[2]) != sorted(declared_names):
            raise ReportUnusable(
                f"{where} was refused and does not record every gate as inapplicable"
            )
    else:
        view = _validate_quality(report, scene, f"{case_id}:{name}", name, f"{where} 'quality'")
        if view.failing != sorted(observed[1]):
            raise ReportUnusable(
                f"{where} observes failing gates {sorted(observed[1])} beside a report whose "
                f"gates actually failing are {view.failing}"
            )
        if (observed[0] == "pass") != view.passed:
            raise ReportUnusable(f"{where} observes {observed[0]!r} beside a report that did not")
        evaluated = {str(gate["name"]) for gate in view.gates}
        if evaluated | set(observed[2]) != declared_names or evaluated & set(observed[2]):
            raise ReportUnusable(
                f"{where} evaluated {sorted(evaluated)} and records {sorted(observed[2])} as "
                f"inapplicable; together these must be exactly the case's gates "
                f"{sorted(declared_names)}"
            )
        _validate_bindings(bindings, view, declared, where)

    lines = base._mismatch_lines(expected[0], expected[1], observed[0], observed[1])
    if expected[3] != observed[3]:
        lines.append(f"expected refusal {expected[3]!r} and observed {observed[3]!r}")
    lines.extend(
        f"gate {gate!r} was expected not to apply and was evaluated"
        for gate in sorted(set(expected[2]) - set(observed[2]))
    )
    lines.extend(
        f"gate {gate!r} did not apply and was expected to be evaluated"
        for gate in sorted(set(observed[2]) - set(expected[2]))
    )
    if error is not None:
        lines.append(f"scoring raised: {error}")
    stated = entry.get("mismatches")
    if stated != lines:
        raise ReportUnusable(
            f"{where} lists mismatches {stated!r}; expected-versus-observed produces {lines!r}"
        )
    if status != ("pass" if not lines else "fail"):
        raise ReportUnusable(f"{where} states status {status!r} beside {len(lines)} mismatch(es)")
    return name


@dataclass(frozen=True, slots=True)
class _QualityView:
    """What a validated embedded quality report yields to its caller."""

    passed: bool
    failing: list[str]
    measurements: Mapping[str, Mapping[str, object]]
    gates: tuple[Mapping[str, object], ...]


def _validate_quality(
    value: object, scene: _SceneView, case_name: str, candidate: str, where: str
) -> _QualityView:
    """The embedded :class:`~scanner3d.quality.CaseReport`, checked against itself.

    ``candidate`` is the closed field the report already declared itself under
    (``_validate_candidate`` has already checked it against
    :data:`contract.CANDIDATES`); it is threaded down to
    :func:`_bind_measurements_to_scene` so the presence of algorithm-specific
    measurements can be derived from who the candidate is, rather than parsed
    back out of ``case_name`` or taken on the report's word.
    """
    report = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        report,
        (
            "schema_version",
            "case",
            "passed",
            "gate_count",
            "measurements",
            "gates",
            "failures",
            "notes",
        ),
        where,
        error=ReportUnusable,
    )
    version = report.get("schema_version")
    if version != quality.SCHEMA_VERSION or isinstance(version, bool):
        raise ReportUnusable(f"{where} states quality schema version {version!r}")
    if report.get("case") != case_name:
        raise ReportUnusable(
            f"{where} calls itself {report.get('case')!r} inside the report of {case_name!r}"
        )
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ReportUnusable(f"{where} carries no measurements")
    by_name: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(measurements):
        name, measurement = _validate_measurement(item, f"{where} measurement {index}", scene)
        if name in by_name:
            raise ReportUnusable(f"{where} reports measurement {name!r} twice")
        by_name[name] = measurement
    base._no_nonfinite(measurements, f"{where} 'measurements'")
    _bind_measurements_to_scene(by_name, scene, candidate, where)
    gates = report.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ReportUnusable(f"{where} carries no gates")
    names: set[str] = set()
    failing: list[str] = []
    for index, item in enumerate(gates):
        name, verdict = base._validate_gate(item, f"{where} gate {index}")
        if name in names:
            raise ReportUnusable(f"{where} reports gate {name!r} twice")
        names.add(name)
        if not verdict:
            failing.append(name)
    stated_count = report.get("gate_count")
    if not isinstance(stated_count, int) or isinstance(stated_count, bool):
        raise ReportUnusable(f"{where} states gate_count {stated_count!r}, which is not a count")
    if stated_count != len(gates):
        raise ReportUnusable(f"{where} states {stated_count} gate(s) and carries {len(gates)}")
    if report.get("failures") != [gate for gate in gates if not gate["passed"]]:
        raise ReportUnusable(
            f"{where} lists failures that are not exactly the gates it reports as failing"
        )
    passed = report.get("passed")
    if not isinstance(passed, bool):
        raise ReportUnusable(f"{where} states passed {passed!r}, which is not a boolean")
    if passed != (not failing):
        raise ReportUnusable(
            f"{where} states passed {passed!r} beside {len(failing)} failing gate(s) {failing}"
        )
    notes = report.get("notes")
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ReportUnusable(f"{where} notes is not a list of strings")
    return _QualityView(
        passed=passed,
        failing=sorted(failing),
        measurements=by_name,
        gates=tuple(
            _mapping(gate, f"{where} gate {index}", error=ReportUnusable)
            for index, gate in enumerate(gates)
        ),
    )


def _bind_measurements_to_scene(
    by_name: Mapping[str, Mapping[str, object]], scene: _SceneView, candidate: str, where: str
) -> None:
    """Which measurements were made, and the counts two of them state twice.

    Which measurements exist is not the candidate's to choose: the scene names
    the regions, and the regions decide exactly what is measured. A result that
    quietly stopped measuring the slot, the fusion or the tracking it was
    scored over would otherwise carry no trace of having done so.

    The reconstructed-error block then has to agree with coverage about its own
    population. That block is where a candidate's error distribution lives, and
    the cheapest way to improve a distribution is to summarise fewer of the
    hardest samples - which shows up here as a population smaller than the
    reconstructions coverage counted.
    """
    missing = sorted(scene.required - set(by_name))
    if missing:
        raise ReportUnusable(
            f"{where} makes none of the measurement(s) {missing} the regions of its own scene "
            "name; what is measured is decided by the scene and not by the candidate"
        )
    extra = sorted(set(by_name) - scene.required - scene.optional)
    if extra:
        raise ReportUnusable(
            f"{where} carries measurement(s) {extra} that the regions of its own scene do not name"
        )
    coverage = by_name["coverage"]
    _bind_optional_measurements(by_name, coverage, scene, candidate, where)
    for name, measurement in by_name.items():
        if measurement.get("kind") != "depth_completeness_reconstructed_error":
            continue
        for key, against in (
            ("dropout_samples", "input_dropout"),
            ("reconstructed_on_surface", "reconstructed_on_surface"),
        ):
            if measurement[key] != coverage[against]:
                raise ReportUnusable(
                    f"{where} measurement {name!r} states {key} {measurement[key]!r} beside "
                    f"coverage that states {against} {coverage[against]!r}. The error population "
                    "and its denominator are coverage's counts, and dropping the hardest samples "
                    "from one and not the other is not an improvement"
                )


#: The candidates :mod:`tests.depth_completeness` runs the product's spatial
#: completion primitive for, i.e. the ones :func:`tests.depth_completeness._adapt`
#: attaches a :class:`~tests.depth_completeness.WorkRecord` to - mirroring the
#: names whose functions call ``spatial_completion`` (directly, or by calling
#: it and then corrupting one field of its result). This is a closed fact
#: about the candidate vocabulary, not a re-tuned threshold, so it is named
#: once here rather than parsed back out of a case name or trusted from the
#: report. ``tests/test_depth_completeness_harness.py`` runs every candidate
#: in :data:`contract.CANDIDATES` and asserts this constant is exactly the set
#: that comes back with a non-``None`` ``work``, so the two sets cannot drift
#: apart silently.
CANDIDATES_WITH_COMPLETION_WORK: Final[frozenset[str]] = frozenset(
    {"spatial_completion", "spatial_completion_forged_work", "spatial_completion_recursive_support"}
)

#: The same closed fact for the pose-compensated stage: the candidates whose
#: functions run :func:`scanner3d.depth_temporal.complete_temporally` and so
#: always come back with a ``TemporalWorkRecord``. Deleting that block to hide
#: what a run refused therefore fails on candidate identity, before a single
#: gate or count is read.
CANDIDATES_WITH_TEMPORAL_WORK: Final[frozenset[str]] = frozenset(
    {
        "temporal_completion",
        "spatial_then_temporal_completion",
        "temporal_completion_forged_work",
        "temporal_completion_self_support",
        "temporal_completion_unmeasured_support",
        "temporal_completion_forged_confirmation",
    }
)

#: And the candidates that are offered an accepted spatial estimate to check,
#: which are the only ones that publish confirmation states and therefore the
#: only ones a ``confirmation`` measurement can come from. This is what makes
#: ``confirmed_false`` undeletable: the block's presence follows from the
#: candidate's name, not from whether a case declared a gate over it.
CANDIDATES_WITH_CONFIRMATIONS: Final[frozenset[str]] = frozenset(
    {"spatial_then_temporal_completion", "temporal_completion_forged_confirmation"}
)

#: And the candidates that carry an ACCEPTED product result, which is the only
#: route into the provenance-weighted volume. Closed, and closed on purpose:
#: weighted fusion derives every provenance and every weight from a validated
#: :class:`~scanner3d.depth_completion.CompletedPatch` or
#: :class:`~scanner3d.depth_temporal.TemporalPatch`, so a candidate that
#: declares states and confidences of its own has nothing to offer it. That is
#: what stops a dense fill from reaching the field by calling its invention
#: low-confidence: self-reported confidence is not evidence, and a candidate
#: with no accepted result publishes no weighted-fusion measurement at all.
#: ``baseline`` is here because raw measurement IS an accepted provenance -
#: it is the reference the weighted gates are stated against.
CANDIDATES_WITH_WEIGHTED_FUSION: Final[frozenset[str]] = frozenset(
    {
        "baseline",
        "spatial_completion",
        "temporal_completion",
        "spatial_then_temporal_completion",
    }
)


def _bind_optional_measurements(
    by_name: Mapping[str, Mapping[str, object]],
    coverage: Mapping[str, object],
    scene: _SceneView,
    candidate: str,
    where: str,
) -> None:
    """The optional blocks, present exactly when the evidence beside them says so.

    The scene's regions say which measurements a case makes at all, but three
    of them are legitimately absent for some candidates - there was nothing
    reconstructed to score, no surface came out of the volume, the tracker did
    not place every frame - and "legitimately absent" is precisely the hole a
    result could drop its worst evidence through. Whether each of those blocks
    exists is derivable from the measurements published next to it, so it is
    derived here rather than taken on the report's word.

    ``reconstructed_depth`` is the whole-population form of the hard-sample
    omission attack: coverage says how many reconstructions landed on real
    surface, and every one of them has held-out truth behind it, so a nonzero
    count is exactly the condition under which
    :func:`tests.depth_completeness.measure` computes an error distribution.
    Deleting the block, its gates and its bindings together leaves coverage
    still saying the population was not empty.

    ``completion_work`` is not derived from coverage at all: whether a run
    reports what its own completion cost is a fact about which candidate ran,
    not about what its output looks like. A case can gate on that evidence and
    then move the gate to inapplicable for a candidate that never made the
    block, so relying on gate mismatches to catch its deletion trusts exactly
    the bookkeeping an attacker who deletes the block also has to rewrite.
    :data:`CANDIDATES_WITH_COMPLETION_WORK` is the closed answer instead: the
    candidate identity was already checked against
    :data:`tests.depth_completeness.CANDIDATES` before this function runs, so
    whether *this* run owes a work record is settled the moment the candidate
    is known, before a single measurement is read.

    The frame counts go the same way. Fusion and tracking were both offered the
    scene's frames, and a pose sequence exists only when the tracker placed all
    of them, so all three counts are the published frame count and none of them
    is the candidate's to state differently.
    """
    derived: list[tuple[str, bool, str]] = [
        (
            "reconstructed_depth",
            _count_field(coverage, "reconstructed_on_surface", f"{where} 'coverage'") > 0,
            "coverage places reconstructions on real surface, and every one of them has "
            "held-out truth behind it to be scored against",
        ),
        (
            "confidence",
            _count_field(coverage, "reconstructed", f"{where} 'coverage'") > 0,
            "coverage counts reconstructions, and every one of them was published at some "
            "confidence whose worth is what that measurement describes",
        ),
    ]
    fusion = by_name.get("fusion")
    if fusion is not None:
        at = f"{where} 'fusion'"
        derived.append(
            (
                "fusion_surface",
                base._report_bool(fusion, "surface_extracted", at)
                and _count_field(fusion, "surface_vertices", at) > 0,
                "fusion extracted a surface with vertices, which is what there is to compare "
                "against the truth model",
            )
        )
    weighted = by_name.get("weighted_fusion")
    if weighted is not None:
        at = f"{where} 'weighted_fusion'"
        derived.append(
            (
                "weighted_fusion_surface",
                base._report_bool(weighted, "surface_extracted", at)
                and _count_field(weighted, "surface_vertices", at) > 0,
                "weighted fusion extracted a surface with vertices, which is what there is to "
                "compare against the truth model",
            )
        )
    for replay, pose_name in (
        ("tracking", "tracking_pose"),
        ("tracking_completed_cloud", "tracking_completed_cloud_pose"),
    ):
        tracking = by_name.get(replay)
        if tracking is None:
            continue
        at = f"{where} {replay!r}"
        derived.append(
            (
                pose_name,
                base._report_bool(tracking, "accepted_all", at)
                and _count_field(tracking, "frames_accepted", at) >= 2,
                f"{replay} accepted every frame it was offered and there are at least two of "
                "them, so there is a pose sequence to compare",
            )
        )
    for name, expected, why in derived:
        if expected and name not in by_name:
            raise ReportUnusable(
                f"{where} makes no {name!r} measurement, and {why}. A measurement its own "
                "evidence says exists is not optional, and removing the population that scored "
                "worst is not an improvement"
            )
        if not expected and name in by_name:
            raise ReportUnusable(
                f"{where} carries a {name!r} measurement where its own evidence has nothing for "
                f"it to measure: {why} is not what the measurements beside it state"
            )
    for name in ("fusion", "weighted_fusion", "tracking", "tracking_completed_cloud"):
        record = by_name.get(name)
        if record is None:
            continue
        at = f"{where} {name!r}"
        offered = _count_field(record, "frames_offered", at)
        if offered != scene.frame_count:
            raise ReportUnusable(
                f"{at} was offered {offered} frame(s) beside a scene of {scene.frame_count}; "
                "both were handed exactly the frames the scene publishes"
            )
    confidence = by_name.get("confidence")
    if confidence is not None:
        at = f"{where} 'confidence'"
        for key, against in (
            ("reconstructed", "reconstructed"),
            ("on_surface", "reconstructed_on_surface"),
            ("false_fill", "false_fill"),
        ):
            if _count_field(confidence, key, at) != _count_field(
                coverage, against, f"{where} 'coverage'"
            ):
                raise ReportUnusable(
                    f"{at} describes {confidence[key]!r} sample(s) as {key} beside coverage that "
                    f"states {against} {coverage[against]!r}. The confidence distribution covers "
                    "every reconstruction the result made, and summarising fewer of them is how "
                    "an ordering is improved without improving"
                )
    work = by_name.get("completion_work")
    runs_completion_primitive = candidate in CANDIDATES_WITH_COMPLETION_WORK
    if runs_completion_primitive and work is None:
        raise ReportUnusable(
            f"{where} is scored for candidate {candidate!r}, which runs the product's spatial "
            "completion primitive and therefore always produces a WorkRecord of its own run; a "
            "successfully scored result for this candidate that carries no 'completion_work' "
            "measurement has had that block, its gate and its binding removed rather than never "
            "having run the work it reports on"
        )
    if not runs_completion_primitive and work is not None:
        raise ReportUnusable(
            f"{where} is scored for candidate {candidate!r}, which does not run the product's "
            "spatial completion primitive and so has nothing to report a WorkRecord about; a "
            "'completion_work' measurement beside this candidate did not come from a run that "
            "produced one"
        )
    for block, population, what in (
        ("temporal_work", CANDIDATES_WITH_TEMPORAL_WORK, "the pose-compensated temporal stage"),
        (
            "confirmation",
            CANDIDATES_WITH_CONFIRMATIONS,
            "that stage with an accepted spatial estimate to check",
        ),
        (
            "weighted_fusion",
            CANDIDATES_WITH_WEIGHTED_FUSION,
            "a run that carries an accepted product result for the weighted TSDF input to "
            "derive its provenance from",
        ),
    ):
        if block == "weighted_fusion" and "weighted_fusion" not in scene.optional:
            # The case does not fuse a weighted volume at all, so no candidate
            # owes the block. The region check above has already refused a
            # report that carries one anyway.
            continue
        present = by_name.get(block) is not None
        runs = candidate in population
        if runs and not present:
            raise ReportUnusable(
                f"{where} is scored for candidate {candidate!r}, which runs {what} and therefore "
                f"always produces a {block!r} measurement; a successfully scored result for this "
                "candidate that carries none has had that block, its gates and its bindings "
                "removed rather than never having run what it reports on"
            )
        if present and not runs:
            raise ReportUnusable(
                f"{where} is scored for candidate {candidate!r}, which does not run {what} and "
                f"so has nothing to report a {block!r} measurement about"
            )
    confirmation = by_name.get("confirmation")
    if confirmation is not None:
        at = f"{where} 'confirmation'"
        estimates = _count_field(confirmation, "spatial_estimates", at)
        confirmed = _count_field(confirmation, "confirmed", at)
        contradicted = _count_field(confirmation, "contradicted", at)
        spatial = _count_field(coverage, "spatially_reconstructed", f"{where} 'coverage'")
        temporal = _count_field(coverage, "temporally_reconstructed", f"{where} 'coverage'")
        on_surface = _count_field(confirmation, "confirmed_on_surface", at) + _count_field(
            confirmation, "contradicted_on_surface", at
        )
        # An estimate is either still spatial in the output, or it is one of the
        # temporal reconstructions the same run published. Both verdicts that
        # settle a sample therefore have to fit inside the temporal count, and
        # the estimates left over have to fit inside the spatial one - so
        # shrinking either verdict to dodge a confirmation gate contradicts
        # coverage, which is bound to the scene.
        if confirmed + contradicted > temporal:
            raise ReportUnusable(
                f"{at} settles {confirmed + contradicted} spatial estimate(s) with independent "
                f"evidence beside coverage that counts {temporal} temporal reconstruction(s); a "
                "settled estimate carries the provenance of the evidence that settled it"
            )
        if estimates - confirmed - contradicted > spatial:
            raise ReportUnusable(
                f"{at} leaves {estimates - confirmed - contradicted} spatial estimate(s) "
                f"unobserved beside coverage that counts {spatial} spatially reconstructed "
                "sample(s); an estimate nothing independent reached is still exactly that"
            )
        if on_surface > _count_field(
            coverage, "temporally_reconstructed_on_surface", f"{where} 'coverage'"
        ):
            raise ReportUnusable(
                f"{at} places {on_surface} settled estimate(s) on real surface beside coverage "
                "that puts fewer temporal reconstructions there"
            )
    temporal_work = by_name.get("temporal_work")
    if temporal_work is not None:
        at = f"{where} 'temporal_work'"
        domain = _count_field(coverage, "domain_samples", f"{where} 'coverage'")
        returned = _count_field(coverage, "input_measured", f"{where} 'coverage'")
        holes = _count_field(temporal_work, "holes", at)
        if holes != domain - returned:
            raise ReportUnusable(
                f"{at} faced {holes} hole(s) beside a domain of {domain} that returned "
                f"{returned} measurement(s); which samples were holes is the input's fact"
            )
        if _count_field(temporal_work, "accepted", at) != _count_field(
            coverage, "temporally_reconstructed", f"{where} 'coverage'"
        ):
            raise ReportUnusable(
                f"{at} accepted a different number of hole(s) than coverage carries temporal "
                "reconstructions"
            )
        sources = scene.frame_count * (scene.frame_count - 1)
        if _count_field(temporal_work, "source_frames", at) != sources:
            raise ReportUnusable(
                f"{at} names {temporal_work['source_frames']!r} source frame(s) beside a scene "
                f"of {scene.frame_count} frame(s), which offers {sources}: every frame is "
                "completed from every OTHER frame, and no frame is ever its own evidence"
            )
    if work is not None:
        at = f"{where} 'completion_work'"
        domain = _count_field(coverage, "domain_samples", f"{where} 'coverage'")
        returned = _count_field(coverage, "input_measured", f"{where} 'coverage'")
        holes = _count_field(work, "holes", at)
        if holes != domain - returned:
            raise ReportUnusable(
                f"{at} faced {holes} hole(s) beside a domain of {domain} that returned "
                f"{returned} measurement(s). A hole is a sample the sensor did not return, and "
                "how many there were is the input's fact and not the candidate's"
            )
        accepted = _count_field(work, "accepted", at)
        if accepted != _count_field(coverage, "reconstructed", f"{where} 'coverage'"):
            raise ReportUnusable(
                f"{at} accepted {accepted} hole(s) beside coverage that counts "
                f"{coverage['reconstructed']!r} reconstruction(s); a run that says it refused "
                "more than its own output shows is claiming a conservatism it did not practise"
            )
    for pose_name in ("tracking_pose", "tracking_completed_cloud_pose"):
        pose = by_name.get(pose_name)
        if pose is None:
            continue
        at = f"{where} {pose_name!r}"
        poses = _count_field(pose, "pose_count", at)
        if poses != scene.frame_count:
            raise ReportUnusable(
                f"{at} compares {poses} pose(s) beside a scene of {scene.frame_count} frame(s); "
                "the sequence exists only because every one of them was placed"
            )


def _validate_bindings(
    bindings: Sequence[object],
    view: _QualityView,
    declared: Sequence[Mapping[str, object]],
    where: str,
) -> None:
    """Every evaluated gate's value, resolved from the measurement it came from.

    A gate records its value and nothing about where the value came from, so a
    report carrying only gates cannot tell a measurement that was changed from
    one that was not. Resolving the binding is what closes that: the number in
    the gate must still be the number in the measurement beside it, at the
    exact path the case declared.
    """
    by_gate = {str(gate["name"]): gate for gate in view.gates}
    declared_paths = {str(gate["name"]): gate for gate in declared}
    seen: set[str] = set()
    for index, item in enumerate(bindings):
        at = f"{where} binding {index}"
        record = _mapping(item, at, error=ReportUnusable)
        _exact_keys(record, ("gate", "measurement", "path"), at, error=ReportUnusable)
        name = base._report_text(record, "gate", f"{where} binding {index}")
        if name in seen:
            raise ReportUnusable(f"{where} binds gate {name!r} twice")
        seen.add(name)
        gate = by_gate.get(name)
        if gate is None:
            raise ReportUnusable(f"{where} binds gate {name!r}, which it did not evaluate")
        source = base._report_text(record, "measurement", f"{where} binding {index}")
        path = base._report_strings(record.get("path"), f"{where} binding {index} 'path'")
        if not path or any(not PATH_STEP.match(step) for step in path):
            raise ReportUnusable(f"{where} binding {index} walks {list(path)}")
        stated = declared_paths[name]
        declared_path = stated["path"]
        assert isinstance(declared_path, list)
        if source != stated["measurement"] or list(path) != declared_path:
            raise ReportUnusable(
                f"{where} binds gate {name!r} to {source!r} at {list(path)}, and the case "
                f"declares {stated['measurement']!r} at {stated['path']!r}"
            )
        # The bar itself, not only where the value came from. The case declares
        # a gate and the candidate was evaluated against one; a declaration
        # edited after the run - a limit loosened, a comparison flipped, a unit
        # or a context rewritten - is not the bar anything was held to, and the
        # report carries both halves.
        for key in ("comparison", "unit", "context"):
            if gate.get(key) != stated.get(key):
                raise ReportUnusable(
                    f"{where} evaluated gate {name!r} with {key} {gate.get(key)!r} and its case "
                    f"declares {stated.get(key)!r}. A declaration edited after the run is not the "
                    "bar the candidate met"
                )
        declared_limit, evaluated_limit = stated.get("limit"), gate.get("limit")
        if isinstance(declared_limit, bool) or isinstance(evaluated_limit, bool):
            same = declared_limit is evaluated_limit
        else:
            same = base._close(
                base._report_finite(evaluated_limit, f"{at} evaluated 'limit'"),
                base._report_finite(declared_limit, f"{at} declared 'limit'"),
            )
        if not same:
            raise ReportUnusable(
                f"{where} evaluated gate {name!r} against limit {evaluated_limit!r} and its case "
                f"declares {declared_limit!r}"
            )
        measurement = view.measurements.get(source)
        if measurement is None:
            raise ReportUnusable(
                f"{where} binds gate {name!r} to measurement {source!r}, which is not reported"
            )
        try:
            value = _resolve(measurement, path, f"{where} binding {index}")
        except FixtureUnusable as error:
            raise ReportUnusable(f"{where} binding {index} does not resolve: {error}") from error
        measured = gate["measured"]
        if isinstance(measured, bool) or isinstance(value, bool):
            if measured is not value:
                raise ReportUnusable(
                    f"{where} gate {name!r} judged {measured!r} and its measurement says {value!r}"
                )
            continue
        if not isinstance(value, int | float) or not base._close(float(measured), float(value)):  # type: ignore[arg-type]
            raise ReportUnusable(
                f"{where} gate {name!r} judged {measured!r} and its measurement says {value!r}. A "
                "measurement changed while its gate was left alone is not evidence"
            )
    if seen != set(by_gate):
        raise ReportUnusable(
            f"{where} evaluated {sorted(by_gate)} and binds {sorted(seen)}; every evaluated gate "
            "names where its value came from"
        )


@dataclass(frozen=True, slots=True)
class _SceneView:
    """What a validated published scene yields to the candidates beside it.

    The scene in this artifact is not decoration. Every coverage denominator,
    the truth population and the dropout population are all derivable from it,
    and the regions decide which measurements exist at all - so a report whose
    scene and whose measurements disagree is refused rather than read from
    whichever half suits it.
    """

    frame_count: int
    sample_count: int
    truth_returned: int
    input_returned: int
    #: Measurement names the regions require, and the ones they permit. The
    #: optional ones are the measurements the contract may legitimately not
    #: make: no reconstruction to score, no surface extracted, no poses.
    required: frozenset[str]
    optional: frozenset[str]

    @property
    def domain_samples(self) -> int:
        return self.frame_count * self.sample_count

    @property
    def dropout(self) -> int:
        return self.truth_returned - self.input_returned


def _validate_pose(value: object, where: str) -> None:
    """One frame's exact pose, held to :mod:`scanner3d.geometry`'s own schema.

    Checked for actually being a rigid transform, and both derived magnitudes
    recomputed from the matrix and the vector that produced them. The poses are
    what make the temporal evidence here mean anything: a report free to
    publish ``"not-a-pose"``, a reflection, or a rotation angle that is not its
    own matrix's could claim posed frames it never had.
    """
    pose = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        pose,
        (
            "schema_version",
            "rotation",
            "translation",
            "rotation_angle_radians",
            "translation_magnitude",
        ),
        where,
        error=ReportUnusable,
    )
    version = pose.get("schema_version")
    if version != GEOMETRY_SCHEMA_VERSION or isinstance(version, bool):
        raise ReportUnusable(
            f"{where} states geometry schema version {version!r}; a pose this harness wrote "
            f"states {GEOMETRY_SCHEMA_VERSION}"
        )
    rows = pose.get("rotation")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ReportUnusable(f"{where} 'rotation' is {rows!r}, which is not three rows")
    rotation = tuple(
        base._report_point(row, f"{where} 'rotation' row {index}") for index, row in enumerate(rows)
    )
    translation = base._report_point(pose.get("translation"), f"{where} 'translation'")
    try:
        rigid = Rigid.of(rotation, translation)  # type: ignore[arg-type]
    except ScannerError as error:
        raise ReportUnusable(
            f"{where} does not state a rigid transform: {error}. Something that is not a rotation "
            "beside a translation never placed a frame"
        ) from error
    for key, derived in (
        ("rotation_angle_radians", rigid.rotation_angle),
        ("translation_magnitude", rigid.translation_magnitude),
    ):
        stated = base._report_number(pose, key, where)
        if not base._close(stated, derived):
            raise ReportUnusable(
                f"{where} states {key} {stated!r}; the rotation and translation beside it give "
                f"{derived!r}"
            )


def _validate_fusion_volume(value: object, where: str) -> None:
    """The TSDF volume a case declares, checked for being a volume.

    Bounded and positive throughout: a report may publish the configuration a
    surface was extracted with, and a volume that spans nothing or a depth
    range that is empty is not one any surface came out of.
    """
    volume = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        volume,
        (
            "minimum",
            "maximum",
            "voxel_size",
            "truncation",
            "min_depth",
            "max_depth",
            "min_weight",
            "coverage_tolerance",
        ),
        where,
        error=ReportUnusable,
    )
    minimum = base._report_point(volume.get("minimum"), f"{where} 'minimum'")
    maximum = base._report_point(volume.get("maximum"), f"{where} 'maximum'")
    if any(high <= low for low, high in zip(minimum, maximum, strict=True)):
        raise ReportUnusable(
            f"{where} spans {list(minimum)} to {list(maximum)}, which encloses nothing"
        )
    for key in (
        "voxel_size",
        "truncation",
        "min_depth",
        "max_depth",
        "min_weight",
        "coverage_tolerance",
    ):
        if base._report_number(volume, key, where) <= 0.0:
            raise ReportUnusable(f"{where} states {key} {volume.get(key)!r}, which is not positive")
    if base._report_number(volume, "max_depth", where) <= base._report_number(
        volume, "min_depth", where
    ):
        raise ReportUnusable(f"{where} states a depth range no measurement falls inside")


def _validate_regions(
    value: object, width: int, where: str
) -> tuple[frozenset[str], frozenset[str]]:
    """The regions a case names, typed - and what they mean must be measured.

    Every flag here decides a measurement in :func:`tests.depth_completeness.measure`,
    so the second half of this function is the same decision made again from
    the published scene. It is what makes a region flag load-bearing rather
    than a label: a report that changed one has to have the measurements the
    changed one implies.
    """
    regions = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        regions,
        (
            "plane",
            "foreground_below",
            "slot_columns",
            "surface_tolerance",
            "dimensions",
            "landmarks",
            "fusion",
            "weighted_fusion",
            "tracking",
        ),
        where,
        error=ReportUnusable,
    )
    plane = base._report_bool(regions, "plane", where)
    dimensions = base._report_bool(regions, "dimensions", where)
    landmarks = base._report_bool(regions, "landmarks", where)
    tracking = base._report_bool(regions, "tracking", where)
    foreground = regions.get("foreground_below")
    if foreground is not None:
        code = _count_field(regions, "foreground_below", where)
        if not 0 < code <= contract.MAX_CODE:
            raise ReportUnusable(
                f"{where} calls every sample at or below code {code} foreground; a raw code that "
                f"is a return is 1 to {contract.MAX_CODE}"
            )
    tolerance = regions.get("surface_tolerance")
    if tolerance is not None and base._report_number(regions, "surface_tolerance", where) <= 0.0:
        raise ReportUnusable(
            f"{where} states a surface tolerance of {tolerance!r}; a tolerance of nothing admits "
            "no point at all"
        )
    columns = regions.get("slot_columns")
    if not isinstance(columns, list):
        raise ReportUnusable(f"{where} 'slot_columns' is {columns!r}, which is not a list")
    seen: set[int] = set()
    for index, column in enumerate(columns):
        if isinstance(column, bool) or not isinstance(column, int) or not 0 <= column < width:
            raise ReportUnusable(
                f"{where} 'slot_columns'[{index}] is {column!r}, which is not a column of a "
                f"{width}-wide window"
            )
        if column in seen:
            raise ReportUnusable(f"{where} names slot column {column} more than once")
        seen.add(column)
    fusion = regions.get("fusion")
    if fusion is not None:
        _validate_fusion_volume(fusion, f"{where} 'fusion'")
    weighted_fusion = regions.get("weighted_fusion")
    if weighted_fusion is not None:
        _validate_fusion_volume(weighted_fusion, f"{where} 'weighted_fusion'")

    required = {"coverage"}
    # Three blocks no region decides. "reconstructed_depth" and "confidence"
    # exist exactly when there is a population for them, which coverage states
    # and _bind_optional_measurements derives from the measurements beside it.
    # "completion_work" is different again: it exists exactly when the closed
    # candidate identity says the run called the product's spatial completion
    # primitive (CANDIDATES_WITH_COMPLETION_WORK in the harness), and
    # _bind_optional_measurements requires or forbids the block on that fact
    # directly - not on whatever gates a case happens to declare over it, which
    # a report could move to "inapplicable" beside the deleted block.
    # "temporal_work" and "confirmation" go the same way as "completion_work",
    # and for the same reason: whether a run reports what the pose-compensated
    # stage cost, and whether it reached a verdict on an accepted spatial
    # estimate, are facts about WHICH candidate ran rather than about how its
    # output looks. CANDIDATES_WITH_TEMPORAL_WORK and CANDIDATES_WITH_CONFIRMATIONS
    # are the closed answers, and _bind_optional_measurements requires or
    # forbids each block on that identity alone.
    optional = {
        "reconstructed_depth",
        "confidence",
        "completion_work",
        "temporal_work",
        "confirmation",
    }
    if plane:
        required.add("plane")
    if tolerance is not None:
        required.add("surface")
    if dimensions:
        required.add("dimensions")
    if landmarks:
        required.add("landmarks")
    if foreground is not None and not columns:
        required.update(("edge", "edge_step_column"))
    if columns:
        required.add("slot")
    if fusion is not None:
        required.add("fusion")
        optional.add("fusion_surface")
    if weighted_fusion is not None:
        # Optional rather than required, and not because it may be skipped.
        # A weighted-fusion record exists exactly when the candidate carries an
        # ACCEPTED product result to derive its provenance from, which is a
        # fact about which candidate ran - so it is enforced in
        # _bind_optional_measurements against the closed candidate identity,
        # the same way completion_work and temporal_work are, rather than
        # against gates a report could move to "inapplicable".
        optional.update(("weighted_fusion", "weighted_fusion_surface"))
    if tracking:
        # Both routes, always: the intended measured-only replay the gates are
        # stated over, and the counterfactual completed-cloud replay beside it.
        # A candidate cannot publish one and omit the other, so the favourable
        # record cannot be shown without the unfavourable one.
        required.update(("tracking", "tracking_completed_cloud"))
        optional.update(("tracking_pose", "tracking_completed_cloud_pose"))
    return frozenset(required), frozenset(optional)


def _validate_scene(value: object, where: str) -> _SceneView:
    scene = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        scene,
        ("window", "depth_scale", "intrinsics", "regions", "frames"),
        where,
        error=ReportUnusable,
    )
    at_window = f"{where} 'window'"
    window = _mapping(scene.get("window"), at_window, error=ReportUnusable)
    _exact_keys(
        window,
        ("u0", "v0", "width", "height", "sample_count"),
        at_window,
        error=ReportUnusable,
    )
    u0 = _count_field(window, "u0", at_window)
    v0 = _count_field(window, "v0", at_window)
    width = base._positive_count(window, "width", at_window)
    height = base._positive_count(window, "height", at_window)
    if u0 + width > FRAME_WIDTH or v0 + height > FRAME_HEIGHT:
        raise ReportUnusable(
            f"{at_window} states a {width}x{height} window at ({u0}, {v0}), which leaves the "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} frame it is a window of"
        )
    samples = _count_field(window, "sample_count", at_window)
    if samples != width * height:
        raise ReportUnusable(f"{where} 'window' states a sample count that is not its own area")
    if base._report_number(scene, "depth_scale", where) <= 0.0:
        raise ReportUnusable(f"{where} states a depth scale that is not positive")
    at_intrinsics = f"{where} 'intrinsics'"
    intrinsics = _mapping(scene.get("intrinsics"), at_intrinsics, error=ReportUnusable)
    _exact_keys(intrinsics, ("fx", "fy", "cx", "cy"), at_intrinsics, error=ReportUnusable)
    for key in ("fx", "fy"):
        if base._report_number(intrinsics, key, at_intrinsics) <= 0.0:
            raise ReportUnusable(
                f"{at_intrinsics} states {key} {intrinsics.get(key)!r}; a focal length that is "
                "not positive deprojects nothing"
            )
    for key in ("cx", "cy"):
        base._report_number(intrinsics, key, at_intrinsics)
    required, optional = _validate_regions(scene.get("regions"), width, f"{where} 'regions'")
    frames = scene.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReportUnusable(f"{where} carries no frames")
    truth_total = input_total = 0
    for index, item in enumerate(frames):
        at = f"{where} frame {index}"
        frame = _mapping(item, at, error=ReportUnusable)
        _exact_keys(
            frame,
            (
                "index",
                "truth_sha256",
                "input_sha256",
                "truth_returned",
                "input_returned",
                "pose",
            ),
            at,
            error=ReportUnusable,
        )
        if _count_field(frame, "index", at) != index:
            raise ReportUnusable(f"{at} states an index that is not its position")
        for key in ("truth_sha256", "input_sha256"):
            digest = base._report_text(frame, key, at)
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ReportUnusable(f"{at} {key} is not a sha256 digest")
        truth = _count_field(frame, "truth_returned", at)
        returned = _count_field(frame, "input_returned", at)
        if returned > truth:
            raise ReportUnusable(
                f"{at} returns more samples after its dropout mask than the truth ever held"
            )
        if truth > samples:
            raise ReportUnusable(f"{at} returns more samples than the window holds")
        _validate_pose(frame.get("pose"), f"{at} 'pose'")
        truth_total += truth
        input_total += returned
    return _SceneView(
        frame_count=len(frames),
        sample_count=samples,
        truth_returned=truth_total,
        input_returned=input_total,
        required=required,
        optional=optional,
    )


#: The measured-only tracking records, which are the case's and not the
#: candidate's. Closed here so the rule below cannot be widened by a report.
MEASURED_ONLY_TRACKING: Final[tuple[str, ...]] = ("tracking", "tracking_pose")


def _bind_measured_only_tracking(candidates: Sequence[object], where: str) -> None:
    """One measured-only tracking record per case, byte-identical across candidates.

    This is the acceptance condition itself, checked rather than described.
    The tracker in this project consumes measured geometry only; no completion
    stage feeds it. So the measured-only replay of a case is a property of the
    case's own frames, and every candidate that preserved its measurements -
    which all of them are required to - must reproduce it exactly, gates and
    pose errors included.

    Two things follow, and both are load-bearing. A candidate cannot be shown
    as tracking-non-inferior by a replay that quietly differed from the
    baseline's; and a candidate that damaged a measured sample cannot hide it
    here, because the cloud is built from the candidate's own output and a
    changed measurement moves this record.

    The counterfactual completed-cloud replay is deliberately NOT held to this:
    it is expected to differ per candidate, which is the whole reason it is
    published.
    """
    first: dict[str, tuple[str, Mapping[str, object]]] = {}
    for item in candidates:
        entry = _mapping(item, where, error=ReportUnusable)
        report = entry.get("quality")
        if not isinstance(report, Mapping):
            continue
        candidate = base._report_text(entry, "candidate", where)
        measurements = report.get("measurements")
        if not isinstance(measurements, list):
            continue
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                continue
            name = measurement.get("name")
            if not isinstance(name, str) or name not in MEASURED_ONLY_TRACKING:
                continue
            recorded = first.get(name)
            if recorded is None:
                first[name] = (candidate, measurement)
            elif measurement != recorded[1]:
                raise ReportUnusable(
                    f"{where} candidate {candidate!r} publishes a {name!r} record that differs "
                    f"from {recorded[0]!r}'s. The tracker here is offered measured geometry "
                    "only and no completion enters it, so this record is the case's fact: two "
                    "candidates disagreeing about it means one of them changed a measurement it "
                    "was given, or the replay was not the measured-only one"
                )


def _validate_case(value: object, what: str) -> tuple[str, str]:
    case = _mapping(value, what, error=ReportUnusable)
    _exact_keys(
        case,
        (
            "case_id",
            "fixture",
            "status",
            "what",
            "why",
            "notes",
            "scene",
            "gates",
            "candidates",
            "error",
        ),
        what,
        error=ReportUnusable,
    )
    case_id = base._report_text(case, "case_id", what)
    where = f"{what} case {case_id!r}"
    fixture = _mapping(case.get("fixture"), f"{where} 'fixture'", error=ReportUnusable)
    _exact_keys(fixture, ("name", "sha256", "bytes"), f"{where} 'fixture'", error=ReportUnusable)
    if not re.fullmatch(r"[0-9a-f]{64}", base._report_text(fixture, "sha256", where)):
        raise ReportUnusable(f"{where} fixture sha256 is not a digest")
    base._positive_count(fixture, "bytes", f"{where} 'fixture'")
    status = base._report_text(case, "status", where)
    if status not in STATUSES:
        raise ReportUnusable(f"{where} states status {status!r}")
    error = case.get("error")
    candidates = _list_field(case, "candidates", where)
    gates = _list_field(case, "gates", where)
    if status == "malformed":
        if not isinstance(error, str) or candidates or gates or case.get("scene") is not None:
            raise ReportUnusable(
                f"{where} is malformed and must carry its error and nothing it could not read"
            )
        return case_id, status
    if error is not None:
        raise ReportUnusable(f"{where} ran and carries a fixture error")
    base._report_text(case, "what", where)
    base._report_text(case, "why", where)
    base._report_strings(case.get("notes"), f"{where} 'notes'")
    scene = _validate_scene(case.get("scene"), f"{where} 'scene'")
    if not gates:
        raise ReportUnusable(f"{where} states no gates")
    declared: list[Mapping[str, object]] = []
    names: set[str] = set()
    for index, item in enumerate(gates):
        gate = _mapping(item, f"{where} gate {index}", error=ReportUnusable)
        _exact_keys(
            gate,
            ("name", "measurement", "path", "comparison", "limit", "unit", "context"),
            f"{where} gate {index}",
            error=ReportUnusable,
        )
        at = f"{where} gate {index}"
        name = base._report_text(gate, "name", at)
        if name in names:
            raise ReportUnusable(f"{where} states gate {name!r} twice")
        names.add(name)
        comparison = base._report_text(gate, "comparison", at)
        if comparison not in COMPARISONS:
            raise ReportUnusable(f"{where} gate {name!r} states an unknown comparison")
        path = base._report_strings(gate.get("path"), f"{at} 'path'")
        if not path or any(not PATH_STEP.match(step) for step in path):
            raise ReportUnusable(f"{where} gate {name!r} walks {list(path)}")
        base._report_text(gate, "measurement", at)
        base._report_text(gate, "context", at)
        # The declaration is a bar, and a bar is a comparison against a value
        # of a kind. A limit that is a string, a boolean under an ordering or a
        # number under 'is' is not a bar anything could have been held to, and
        # this is where the report says what the case asked for.
        unit = base._report_text(gate, "unit", at)
        limit = gate.get("limit")
        if COMPARISONS[comparison] is quality.Comparison.IS:
            if not isinstance(limit, bool):
                raise ReportUnusable(
                    f"{where} gate {name!r} compares to {limit!r} with 'is', and that is not a "
                    "boolean"
                )
            if unit != quality.BOOLEAN_UNIT:
                raise ReportUnusable(
                    f"{where} gate {name!r} compares with 'is' in {unit!r}; a boolean is in "
                    f"{quality.BOOLEAN_UNIT!r}"
                )
        else:
            base._report_finite(limit, f"{at} 'limit'")
        declared.append(gate)
    if not candidates:
        raise ReportUnusable(f"{where} runs no candidates")
    seen: set[str] = set()
    statuses: list[str] = []
    for item in candidates:
        name = _validate_candidate(item, declared, scene, case_id, where)
        if name in seen:
            raise ReportUnusable(f"{where} runs candidate {name!r} twice")
        seen.add(name)
        entry = _mapping(item, where, error=ReportUnusable)
        statuses.append(str(entry["status"]))
    _bind_measured_only_tracking(candidates, where)
    recomputed = "pass" if all(state == "pass" for state in statuses) else "fail"
    if status != recomputed:
        raise ReportUnusable(f"{where} states {status!r} beside candidates that are {statuses}")
    return case_id, status


def validate_report(
    document: object, *, what: str = "depth-completeness report"
) -> Mapping[str, object]:
    """Refuse an artifact that is not exactly this schema. Fail closed.

    Every verdict, count and mismatch is recomputed from the evidence beside
    it rather than read and believed, so a report edited to say it passed does
    not survive being read back.
    """
    report = _mapping(document, what, error=ReportUnusable)
    _exact_keys(
        report,
        (
            "schema_version",
            "kind",
            "corpus",
            "provenance_states",
            "refusals",
            "cases",
            "totals",
            "decision",
            "limitations",
        ),
        what,
        error=ReportUnusable,
    )
    version = report.get("schema_version")
    if version != REPORT_SCHEMA_VERSION or isinstance(version, bool):
        raise ReportUnusable(
            f"{what} states schema version {version!r}; this reader understands exactly "
            f"{REPORT_SCHEMA_VERSION}"
        )
    if base._report_text(report, "kind", what) != REPORT_KIND:
        raise ReportUnusable(f"{what} is not a {REPORT_KIND}")
    corpus = _mapping(report.get("corpus"), f"{what} 'corpus'", error=ReportUnusable)
    _exact_keys(
        corpus,
        ("kind", "schema_version", "manifest_sha256", "case_count"),
        f"{what} 'corpus'",
        error=ReportUnusable,
    )
    if base._report_text(corpus, "kind", f"{what} 'corpus'") != CORPUS_KIND:
        raise ReportUnusable(f"{what} names a corpus that is not a {CORPUS_KIND}")
    if corpus.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ReportUnusable(f"{what} names a corpus of another schema version")
    if not re.fullmatch(
        r"[0-9a-f]{64}", base._report_text(corpus, "manifest_sha256", f"{what} 'corpus'")
    ):
        raise ReportUnusable(f"{what} 'corpus' manifest_sha256 is not a digest")
    if _list_field(report, "provenance_states", what) != list(contract.STATES):
        raise ReportUnusable(
            f"{what} states provenance states that are not the contract's {list(contract.STATES)}"
        )
    if _list_field(report, "refusals", what) != list(contract.REFUSALS):
        raise ReportUnusable(f"{what} states refusals that are not the contract's")
    if _list_field(report, "limitations", what) != list(LIMITATIONS):
        raise ReportUnusable(
            f"{what} states limitations that are not the ones this harness writes. A report that "
            "dropped a limitation would be read as claiming more than the run establishes"
        )
    cases = base._entries(report, "cases", what)
    if not cases:
        raise ReportUnusable(f"{what} reports no cases")
    if base._positive_count(corpus, "case_count", f"{what} 'corpus'") != len(cases):
        raise ReportUnusable(f"{what} names a corpus of a different size than it reports")
    identifiers: list[str] = []
    statuses: list[str] = []
    for index, case in enumerate(cases):
        case_id, status = _validate_case(case, f"{what} entry {index}")
        identifiers.append(case_id)
        statuses.append(status)
    repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if repeated:
        raise ReportUnusable(f"{what} reports case(s) {repeated} more than once")
    totals = _mapping(report.get("totals"), f"{what} 'totals'", error=ReportUnusable)
    _exact_keys(
        totals,
        ("cases", "passed", "failed", "malformed", "candidates", "candidates_passed"),
        f"{what} 'totals'",
        error=ReportUnusable,
    )
    candidate_entries: list[Mapping[str, object]] = []
    for case in cases:
        reported = _list_field(_mapping(case, what, error=ReportUnusable), "candidates", what)
        candidate_entries.extend(
            candidate for candidate in reported if isinstance(candidate, Mapping)
        )
    expected_totals = {
        "cases": len(cases),
        "passed": statuses.count("pass"),
        "failed": statuses.count("fail"),
        "malformed": statuses.count("malformed"),
        "candidates": len(candidate_entries),
        "candidates_passed": sum(
            1 for candidate in candidate_entries if candidate.get("status") == "pass"
        ),
    }
    for key, value in expected_totals.items():
        if _count_field(totals, key, f"{what} 'totals'") != value:
            raise ReportUnusable(
                f"{what} 'totals' states {key} {totals.get(key)!r} beside {value} in its own cases"
            )
    decision = base._report_text(report, "decision", what)
    recomputed = "pass" if expected_totals["passed"] == expected_totals["cases"] else "fail"
    if decision != recomputed:
        raise ReportUnusable(
            f"{what} decides {decision!r} beside {expected_totals['passed']} of "
            f"{expected_totals['cases']} case(s) that passed"
        )
    base._no_nonfinite(report, what)
    return report


# --- the command ----------------------------------------------------------------------


def render(document: Mapping[str, object]) -> str:
    """The whole report as text, for a log. Deterministic.

    Every candidate of every case, not only the mismatched ones: the ones that
    behaved are what say which failure mode was ruled out.
    """
    cases = document["cases"]
    assert isinstance(cases, list)
    totals = document["totals"]
    assert isinstance(totals, Mapping)
    decision = str(document["decision"]).upper()
    lines = [
        f"[{decision}] depth-completeness benchmark: {totals['passed']} of {totals['cases']} "
        f"case(s) and {totals['candidates_passed']} of {totals['candidates']} candidate(s) "
        f"produced exactly the outcome their fixture declares "
        f"({totals['failed']} mismatched, {totals['malformed']} malformed)"
    ]
    for entry in cases:
        assert isinstance(entry, Mapping)
        status = str(entry["status"]).upper()
        lines.append(f"[{status}] {entry['case_id']} ({entry['fixture']['name']})")
        if entry["status"] == "malformed":
            lines.append(f"  fixture unusable: {entry['error']}")
            continue
        candidates = entry["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert isinstance(candidate, Mapping)
            observed = candidate["observed"]
            expected = candidate["expected"]
            assert isinstance(observed, Mapping) and isinstance(expected, Mapping)
            mark = "ok " if candidate["status"] == "pass" else "BAD"
            lines.append(
                f"  {mark} {candidate['candidate']}: expected {expected['outcome']} failing "
                f"{list(expected['failed_gates'])}; observed {observed['outcome']} failing "
                f"{list(observed['failed_gates'])}"
                + (f" refused {observed['refusal']}" if observed["refusal"] else "")
            )
            report = candidate["quality"]
            if isinstance(report, Mapping):
                gates = report["gates"]
                assert isinstance(gates, list)
                for gate in gates:
                    assert isinstance(gate, Mapping)
                    verdict = "PASS" if gate["passed"] else "FAIL"
                    lines.append(
                        f"      {verdict} {gate['name']}: {gate['measured']} "
                        f"{gate['comparison']} {gate['limit']} ({gate['unit']})"
                    )
            mismatches = candidate["mismatches"]
            assert isinstance(mismatches, list)
            lines.extend(f"      MISMATCH: {line}" for line in mismatches)
    lines.append("")
    lines.extend(f"limitation: {line}" for line in LIMITATIONS)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tests.depth_completeness_harness",
        description=(
            "Run the deterministic depth-completeness fixture corpus through the real quality, "
            "TSDF and tracking authorities, and write one strict JSON report."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS,
        help="the fixture corpus to run (default: tests/fixtures/depth_completeness/cases)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to publish the strict JSON report, written atomically",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the rendered report; the exit code and --out still say everything",
    )
    arguments = parser.parse_args(argv)

    # Before the corpus is even opened, so a run that refuses at exit 2 never
    # leaves the previous run's report where the next reader would take it for
    # this answer.
    if arguments.out is not None:
        try:
            clear_destination(arguments.out)
        except HarnessRefused as error:
            print(
                f"the depth-completeness harness could not use that output path: {error}",
                file=sys.stderr,
            )
            return 2
    try:
        document = run(arguments.corpus)
    except HarnessRefused as error:
        print(f"the depth-completeness harness could not run: {error}", file=sys.stderr)
        return 2
    if arguments.out is not None:
        try:
            publish(document, arguments.out)
        except (HarnessRefused, ValueError) as error:
            print(
                f"the depth-completeness harness could not publish its report: {error}",
                file=sys.stderr,
            )
            return 2
    if not arguments.quiet:
        print(render(document))
    return 0 if document["decision"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover - the command line entry point
    raise SystemExit(main())
