"""The depth-completeness contract: provenance, scoring, and the candidates.

What this is for
----------------
The next tasks in this thread implement depth completion - filling samples the
sensor did not return, spatially within a frame and temporally across posed
frames. This module is the oracle those tasks must satisfy, written BEFORE
them, so that a denser or prettier result cannot be called an improvement when
it bridges a depth edge, closes a slot, moves a dimension, roughens a plane,
invents geometry in empty space, or quietly relabels a reconstruction as a
measurement.

It measures. It does not complete anything, and nothing in
:mod:`scanner3d` imports it. The candidates below exist only as controls: the
unchanged baseline, an identity, and a set of deliberately bad fills whose
exact failures are what prove the scoring is load-bearing rather than
decorative.

Where the numbers come from
---------------------------
Nothing here invents a measurement authority:

* :mod:`scanner3d.simulate` renders the truth frames, in the same
  parametrisation :mod:`scanner3d.depth_metric` deprojects with.
* :mod:`scanner3d.depth_metric` decodes every frame and owns the rule that a
  raw sample of zero is *no measurement* and never a point at the origin.
* :mod:`scanner3d.quality` owns every distribution, gate, plane fit, symmetric
  cloud agreement, named dimension and landmark reported here.
* :mod:`scanner3d.tsdf` and :mod:`scanner3d.tsdf_surface` fuse and extract the
  multi-view surface, unchanged, at their own public API.
* :mod:`scanner3d.tracking` places the frames for the tracking-neutrality
  record.

The only arithmetic this module owns is bookkeeping: which sample is which,
against which denominator, and what a completion said about where its value
came from.

Units
-----
Every scene here is synthetic. Depth is ``raw code x depth_scale`` in the
scene's own constructed units, which measure nothing physical; the recorded
scanner captures are not read by anything in this file.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import scanner3d.depth_completion as depth_completion
import scanner3d.depth_temporal as depth_temporal
import scanner3d.quality as quality
import scanner3d.tsdf_input as tsdf_input
from scanner3d.depth_metric import (
    NO_RETURN_SAMPLE,
    DepthConversion,
    DepthPoint,
    DepthUnitAssumption,
    deproject_depth_frame,
)
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.tracking import IncrementalTracker, TrackingPolicy
from scanner3d.tsdf import TsdfConfig, TsdfVolume, VolumeBounds, WeightPolicy
from scanner3d.tsdf_surface import NoSurfaceFound, extract_surface
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

#: The schema of every fixture and every measurement this module defines.
SCHEMA_VERSION: Final = 1

#: What a completion says about one sample. Exactly these four, and the reader
#: refuses anything else: a state nobody defined is a claim nobody checked.
MEASURED: Final = "measured"
SPATIALLY_RECONSTRUCTED: Final = "spatially_reconstructed"
TEMPORALLY_RECONSTRUCTED: Final = "temporally_reconstructed"
MISSING: Final = "missing"
STATES: Final[tuple[str, ...]] = (
    MEASURED,
    SPATIALLY_RECONSTRUCTED,
    TEMPORALLY_RECONSTRUCTED,
    MISSING,
)
RECONSTRUCTED_STATES: Final[tuple[str, ...]] = (
    SPATIALLY_RECONSTRUCTED,
    TEMPORALLY_RECONSTRUCTED,
)

#: Where a reconstructed sample says its evidence came from. One measured
#: sample of one frame, named exactly, so "supported" can be checked rather
#: than believed.
SUPPORT_NAME = re.compile(r"^frame:(\d+):pixel:(\d+),(\d+)$")

#: Every refusal a candidate result can earn. These are not gate failures: a
#: result that earns one of them was never scored at all, because its own
#: account of itself is not readable as evidence.
REFUSALS: Final[tuple[str, ...]] = (
    "frame_set_mismatch",
    "raw_frame_mutated",
    "sample_count_mismatch",
    "unknown_state",
    "confidence_out_of_range",
    "measured_value_changed",
    "missing_carries_depth",
    "reconstructed_without_depth",
    "unsupported_reconstruction",
    "support_not_measured",
    "support_without_reconstruction",
    "temporal_support_not_another_frame",
    "spatial_support_not_this_frame",
    "depth_code_out_of_range",
    "work_counts_inconsistent",
    "unknown_confirmation",
    "confirmation_inconsistent",
    "accepted_result_mismatch",
)

#: What independent posed evidence said about a spatial estimate at one sample.
#: Closed, and stated here rather than imported, so the oracle's vocabulary is
#: the oracle's: a completion that invents a fifth confirmation state is
#: refused rather than described.
NO_SPATIAL_ESTIMATE: Final = "no_spatial_estimate"
CONFIRMED: Final = "confirmed"
CONTRADICTED: Final = "contradicted"
NOT_OBSERVED: Final = "not_observed"
CONFIRMATIONS: Final[tuple[str, ...]] = (
    NO_SPATIAL_ESTIMATE,
    CONFIRMED,
    CONTRADICTED,
    NOT_OBSERVED,
)

#: The confidence at or above which a reconstruction is a CLAIM rather than a
#: hedge. Fixed here, in the oracle, so no candidate can move the line it is
#: judged at: everything at or above it is scored for accuracy against
#: held-out truth, and a completion that wants a better score there has to
#: lower its own confidence rather than the bar.
CONFIDENCE_THRESHOLD: Final = 0.5

#: The confidence histogram's fixed edges. Five equal bins over the whole
#: fraction, published with their populations, so a distribution cannot be
#: improved by dropping the bin it did worst in - the bins have to add up to
#: the reconstructions coverage counted.
CONFIDENCE_BIN_EDGES: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: The unit prose written into every measurement made from a depth code.
DEPTH_UNIT: Final = "synthetic depth units: raw code x depth scale, exact by construction"
FRACTION_UNIT: Final = quality.FRACTION_UNIT
COUNT_UNIT: Final = "samples"
COLUMN_UNIT: Final = "window columns"

#: Where the truth in a dimension or landmark measurement came from. Not an
#: instrument, and it does not pretend to be one.
TRUTH_SOURCE: Final = (
    "the truth frame rendered by scanner3d.simulate before any dropout mask was applied; "
    "synthetic, exact by construction, and no physical measurement"
)

#: What the depth unit is, for scanner3d.depth_metric. Declined deliberately:
#: nothing here measures a physical distance.
SYNTHETIC_UNIT: Final = DepthUnitAssumption(
    label="synthetic depth-completeness unit",
    stated_by="tests/depth_completeness.py",
    basis=(
        "the constructed scene's own unit. It is exact for this benchmark and measures nothing "
        "physical; no scanner capture is read here"
    ),
)

#: The tracking policy the neutrality replay uses. Adjacent-only and fixed, so
#: the record is a property of the frames rather than of a keyframe history
#: that a later candidate could change for unrelated reasons.
TRACKING_POLICY: Final = TrackingPolicy(
    use_keyframes=False,
    use_model=False,
    recovery_enabled=False,
)

#: The ICP scale the tracking replay runs at. The synthetic objects here are a
#: few units across, so the module defaults - tuned for the scanner's own
#: coordinate magnitudes - would voxel-downsample a whole scene into a handful
#: of points. Fixed here so the replay is a property of the frames.
TRACKING_VOXEL_SIZE: Final = 0.25
TRACKING_MAX_CORRESPONDENCE: Final = 1.5

#: A pixel no scored window contains, so support naming it is support that was
#: never measured. Read by the phantom-support control.
_PHANTOM_PIXEL: Final = (0, 0)

#: The highest raw code a frame can carry.
MAX_CODE: Final = 0xFFFF


class CandidateRefused(ScannerError):
    """A candidate result that is not readable as evidence, and its reason."""

    def __init__(self, refusal: str, detail: str) -> None:
        if refusal not in REFUSALS:
            raise ValueError(f"{refusal!r} is not one of the declared refusals {list(REFUSALS)}")
        super().__init__(f"{refusal}: {detail}")
        self.refusal = refusal
        self.detail = detail


class ScoringUnusable(ScannerError):
    """A case whose declared regions cannot be scored as declared."""


# --- the sampling window ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Window:
    """The rectangle of one 640x400 frame a case is scored over.

    A window rather than the whole frame for one reason that is not
    convenience: every denominator in this benchmark is the number of samples
    in it. A metric whose denominator is "whatever the candidate looked at"
    can be improved by looking at less, and no gate built on one is evidence.
    """

    u0: int
    v0: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ScoringUnusable(f"a window spans nothing: {self.width}x{self.height}")
        if self.u0 < 0 or self.v0 < 0:
            raise ScoringUnusable(f"a window starts outside the frame at ({self.u0}, {self.v0})")
        if self.u0 + self.width > FRAME_WIDTH or self.v0 + self.height > FRAME_HEIGHT:
            raise ScoringUnusable(
                f"a {self.width}x{self.height} window at ({self.u0}, {self.v0}) leaves the "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT} frame"
            )

    @property
    def sample_count(self) -> int:
        return self.width * self.height

    def pixel_of(self, index: int) -> tuple[int, int]:
        """The (u, v) of one window index, in frame coordinates."""
        row, column = divmod(index, self.width)
        return self.u0 + column, self.v0 + row

    def index_of(self, u: int, v: int) -> int | None:
        """The window index of a frame pixel, or ``None`` if it is outside."""
        if not (self.u0 <= u < self.u0 + self.width and self.v0 <= v < self.v0 + self.height):
            return None
        return (v - self.v0) * self.width + (u - self.u0)

    def rows(self) -> Iterator[tuple[int, tuple[int, ...]]]:
        """Each window row, as its frame ``v`` and the indices along it."""
        for row in range(self.height):
            base = row * self.width
            yield self.v0 + row, tuple(range(base, base + self.width))

    def as_dict(self) -> dict[str, object]:
        return {
            "u0": self.u0,
            "v0": self.v0,
            "width": self.width,
            "height": self.height,
            "sample_count": self.sample_count,
        }


def codes_of(raw: bytes, window: Window) -> tuple[int, ...]:
    """The raw codes of one window, row-major. Reads bytes, decides nothing."""
    if len(raw) != EXPECTED_FRAME_BYTES:
        raise ScoringUnusable(f"a frame is {EXPECTED_FRAME_BYTES} bytes, not {len(raw)}")
    out: list[int] = []
    for row in range(window.height):
        offset = ((window.v0 + row) * FRAME_WIDTH + window.u0) * 2
        out.extend(struct.unpack_from(f"<{window.width}H", raw, offset))
    return tuple(out)


def frame_of_codes(codes: Sequence[int], window: Window) -> bytes:
    """One real 512000-byte frame carrying ``codes`` in ``window`` and zero elsewhere.

    Zero elsewhere is not padding: it is
    :data:`scanner3d.depth_metric.NO_RETURN_SAMPLE`, so everything outside the
    window is *no measurement* to every reader downstream, exactly as an
    unreturned sample is.
    """
    if len(codes) != window.sample_count:
        raise ScoringUnusable(
            f"{len(codes)} code(s) for a window of {window.sample_count} sample(s)"
        )
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    for index, code in enumerate(codes):
        if code == NO_RETURN_SAMPLE:
            continue
        if not 0 <= code <= MAX_CODE:
            raise ScoringUnusable(f"raw code {code!r} is not a uint16 sample")
        u, v = window.pixel_of(index)
        struct.pack_into("<H", buffer, (v * FRAME_WIDTH + u) * 2, code)
    return bytes(buffer)


def deproject(raw: bytes, *, intrinsics: IntrinsicsRecord, depth_scale: float) -> DepthConversion:
    """Deproject through the production path, with zero left as no measurement."""
    return deproject_depth_frame(
        raw, intrinsics=intrinsics, depth_scale=depth_scale, unit=SYNTHETIC_UNIT
    )


def cloud_of(
    codes: Sequence[int],
    window: Window,
    *,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
    pose: Rigid | None = None,
) -> tuple[Point, ...]:
    """The points of a window's codes, through the real deprojection."""
    conversion = deproject(
        frame_of_codes(codes, window), intrinsics=intrinsics, depth_scale=depth_scale
    )
    points = tuple((point.x, point.y, point.z) for point in conversion.points)
    if pose is None:
        return points
    return tuple(pose.apply(point) for point in points)


def depth_points_of(
    codes: Sequence[int], window: Window, *, intrinsics: IntrinsicsRecord, depth_scale: float
) -> tuple[DepthPoint, ...]:
    """The deprojected samples, each still carrying the pixel it came from."""
    conversion = deproject(
        frame_of_codes(codes, window), intrinsics=intrinsics, depth_scale=depth_scale
    )
    return conversion.points


# --- what a case is, and what a candidate is given ------------------------------------


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    """One frame of a case: its truth, the masked input, and its exact pose."""

    index: int
    truth: bytes
    raw: bytes
    pose: Rigid

    def as_dict(self) -> dict[str, object]:
        return {"index": self.index, "pose": self.pose.as_dict()}


@dataclass(frozen=True, slots=True)
class Regions:
    """The parts of a window a case names, and what each one is for."""

    #: Measure flatness over the window. The scene must be one plane.
    plane: bool = False
    #: The code at or below which a sample is foreground, for the step scene.
    foreground_below: int | None = None
    #: Window columns that must stay open: truth there is background, and a
    #: completion that fills them with foreground has closed a real gap.
    slot_columns: tuple[int, ...] = ()
    #: Coverage tolerance for symmetric cloud agreement, when the case makes it.
    surface_tolerance: float | None = None
    #: Measure named extents and landmarks against the truth cloud.
    dimensions: bool = False
    landmarks: bool = False
    #: Fuse the frames and compare the extracted surface against truth.
    fusion: FusionSpec | None = None
    #: Fuse the frames AGAIN through the provenance-weighted input contract,
    #: for candidates that carry an accepted product result to derive it from.
    #: A separate volume, and deliberately not the same field: the record
    #: above is what uniform fusion does with a candidate's completed codes -
    #: which is the counterfactual this one exists to be compared against - and
    #: collapsing the two would leave no way to show that the weighting is
    #: load-bearing rather than decorative.
    weighted_fusion: FusionSpec | None = None
    #: Replay the tracker over the frames.
    tracking: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "plane": self.plane,
            "foreground_below": self.foreground_below,
            "slot_columns": list(self.slot_columns),
            "surface_tolerance": self.surface_tolerance,
            "dimensions": self.dimensions,
            "landmarks": self.landmarks,
            "fusion": None if self.fusion is None else self.fusion.as_dict(),
            "weighted_fusion": (
                None if self.weighted_fusion is None else self.weighted_fusion.as_dict()
            ),
            "tracking": self.tracking,
        }


@dataclass(frozen=True, slots=True)
class FusionSpec:
    """The exact volume the multi-view cases fuse into. Fixed, not derived."""

    minimum: Point
    maximum: Point
    voxel_size: float
    truncation: float
    min_depth: float
    max_depth: float
    min_weight: float
    coverage_tolerance: float

    def config(self) -> TsdfConfig:
        return TsdfConfig.of(
            bounds=VolumeBounds.of(self.minimum, self.maximum),
            voxel_size=self.voxel_size,
            truncation=self.truncation,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            weight_policy=WeightPolicy.UNIFORM,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "voxel_size": self.voxel_size,
            "truncation": self.truncation,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "min_weight": self.min_weight,
            "coverage_tolerance": self.coverage_tolerance,
        }


@dataclass(frozen=True, slots=True)
class Scene:
    """One completeness case: its frames, its window and what it measures."""

    case_id: str
    window: Window
    intrinsics: IntrinsicsRecord
    depth_scale: float
    frames: tuple[FrameEvidence, ...]
    regions: Regions

    def __post_init__(self) -> None:
        if not self.frames:
            raise ScoringUnusable(f"{self.case_id} has no frames")
        for position, frame in enumerate(self.frames):
            if frame.index != position:
                raise ScoringUnusable(f"{self.case_id} frame {position} calls itself {frame.index}")

    def truth_codes(self, index: int) -> tuple[int, ...]:
        return codes_of(self.frames[index].truth, self.window)

    def input_codes(self, index: int) -> tuple[int, ...]:
        return codes_of(self.frames[index].raw, self.window)

    def inputs(self) -> tuple[bytes, ...]:
        return tuple(frame.raw for frame in self.frames)

    def poses(self) -> tuple[Rigid, ...]:
        return tuple(frame.pose for frame in self.frames)


# --- what a candidate returns ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompletedFrame:
    """One frame a candidate produced, with per-sample provenance.

    ``confirmations`` is empty for every candidate that has no independent
    evidence to confirm anything with, and one entry per sample for the ones
    that do. It is deliberately separate from ``states``: what a sample IS and
    what independent evidence SAID about a proposal for it are two facts, and a
    completion that collapses them can relabel an unconfirmed spatial guess as
    a temporal measurement by changing one field.
    """

    index: int
    raw: bytes
    codes: tuple[int, ...]
    states: tuple[str, ...]
    confidence: tuple[float, ...]
    support: Mapping[int, tuple[str, ...]]
    confirmations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkRecord:
    """What a candidate says its own completion cost, and what it refused.

    Declared BY the candidate, deliberately as plain unchecked data: this is
    the shape a completion is free to lie about, and :func:`validate` is where
    the lie is caught. Every field is bound to something the oracle counts for
    itself - the holes the input actually had, the reconstructions the result
    actually carries, the support it actually named - so a work record that
    flatters its own conservatism refuses the whole result rather than being
    published beside it.

    Optional: the controls in this module recover nothing and declare none,
    and a case that gates on work evidence records those gates as inapplicable
    for them rather than pretending they were met.
    """

    backend: str
    policy: Mapping[str, object]
    holes: int
    attempted: int
    accepted: int
    rejected: int
    rejections: Mapping[str, int]
    support_samples: int
    work_units: int
    work_budget: int


@dataclass(frozen=True, slots=True)
class TemporalWorkRecord:
    """What a candidate says its own POSE-COMPENSATED run cost, and what it refused.

    Separate from :class:`WorkRecord` on purpose, and not an extension of it.
    A spatial run and a temporal run refuse for different reasons, spend
    different work and rest on different evidence, and folding the two into one
    record would need one rejection vocabulary covering both - at which point a
    run could report a refusal it structurally cannot make and nothing would
    notice. Declared BY the candidate as plain unchecked data, and bound by
    :func:`validate` to what this module counts for itself.
    """

    backend: str
    policy: Mapping[str, object]
    source_frames: int
    eligible_samples: int
    observations: int
    observation_rejections: Mapping[str, int]
    candidates: int
    holes: int
    attempted: int
    accepted: int
    rejected: int
    rejections: Mapping[str, int]
    confirmed: int
    contradicted: int
    not_observed: int
    support_samples: int
    work_units: int
    work_budget: int


#: What a candidate may hand over as the ACCEPTED product result behind one
#: frame. ``None`` means "this frame is the raw measurement and nothing else",
#: which is what the unchanged baseline offers weighted fusion.
AcceptedFrame = depth_completion.CompletedPatch | depth_temporal.TemporalPatch | None


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """A whole candidate: one completed frame per input frame."""

    name: str
    frames: tuple[CompletedFrame, ...]
    #: What producing those frames cost, when the candidate reports it at all.
    work: WorkRecord | None = None
    #: The same, for the pose-compensated temporal run, when there was one.
    temporal: TemporalWorkRecord | None = None
    #: The accepted product results this candidate's frames really came out of,
    #: one per frame, or ``None`` for a candidate that has none.
    #:
    #: This is the ONLY route into weighted fusion, and that is the point. A
    #: weighted frame's provenance is derived from an accepted
    #: :class:`~scanner3d.depth_completion.CompletedPatch` or
    #: :class:`~scanner3d.depth_temporal.TemporalPatch` - each of which has
    #: already bound itself to its own immutable raw input and re-derived its
    #: own run - so a candidate cannot reach a weighted volume by declaring
    #: states and confidences of its own. ``dense_nearest`` fills the aperture
    #: at confidence 1.0 and ``broad_interpolation`` could just as easily fill
    #: it at 0.1; neither has an accepted result to offer, so neither has any
    #: weight to be low. Self-reported confidence is not evidence here.
    accepted: tuple[AcceptedFrame, ...] | None = None


#: A candidate is a pure function of the scene it is given.
Candidate = Callable[[Scene], CandidateResult]


def validate(scene: Scene, result: CandidateResult) -> None:
    """Refuse a result whose own account of itself does not hold. Fail closed.

    Every refusal here is a way a completion could look better than it is:
    changing a raw frame, calling a reconstruction a measurement, carrying a
    depth on a sample it declared missing, reconstructing from support it
    cannot name, naming support that was itself never measured, or returning
    fewer samples than the window so the hard ones leave the denominator.
    """
    window = scene.window
    if len(result.frames) != len(scene.frames):
        raise CandidateRefused(
            "frame_set_mismatch",
            f"{result.name} returned {len(result.frames)} frame(s) for {len(scene.frames)}",
        )
    measured_inputs = [set(_measured_indices(frame.raw, window)) for frame in scene.frames]
    for position, (given, produced) in enumerate(zip(scene.frames, result.frames, strict=True)):
        where = f"{result.name} frame {position}"
        if produced.index != given.index:
            raise CandidateRefused(
                "frame_set_mismatch", f"{where} calls itself frame {produced.index}"
            )
        if produced.raw != given.raw:
            raise CandidateRefused(
                "raw_frame_mutated",
                f"{where} returned raw bytes that are not the ones it was given; the raw "
                "measurement is the evidence and a completion may not edit it",
            )
        counts = (len(produced.codes), len(produced.states), len(produced.confidence))
        if counts != (window.sample_count,) * 3:
            raise CandidateRefused(
                "sample_count_mismatch",
                f"{where} carries {counts} value(s) for a window of {window.sample_count} "
                "sample(s); every metric here is denominator-bound and a result may not drop "
                "the samples it did worst on",
            )
        given_codes = codes_of(given.raw, window)
        for index, (code, state, confidence) in enumerate(
            zip(produced.codes, produced.states, produced.confidence, strict=True)
        ):
            _validate_sample(
                where=where,
                index=index,
                window=window,
                code=code,
                state=state,
                confidence=confidence,
                raw_code=given_codes[index],
                support=produced.support.get(index, ()),
                frame_index=position,
                measured_inputs=measured_inputs,
            )
        unknown = sorted(set(produced.support) - set(range(window.sample_count)))
        if unknown:
            raise CandidateRefused(
                "sample_count_mismatch",
                f"{where} names support for sample(s) {unknown[:4]} outside its own window",
            )
        _validate_confirmations(where, window, produced)
    _validate_work(scene, result)
    _validate_temporal_work(scene, result)
    _validate_accepted(scene, result)


def _validate_accepted(scene: Scene, result: CandidateResult) -> None:
    """Bind a declared accepted product result to this candidate's own frames.

    A candidate that carries accepted results is claiming that its published
    codes, states and confidences ARE those results. That claim is the one
    thing standing between the weighted volume and a self-reported provenance,
    so it is checked against the accepted result rather than believed:

    * the accepted result was derived from THIS frame's raw window codes, so a
      result about other bytes cannot be attached to these;
    * every published state is the state the accepted result carries there,
      renamed and never widened - so a sample the product left spatial cannot
      be published as temporal, and one it left missing cannot acquire a
      depth;
    * every published code and confidence is the accepted result's own.

    A candidate with no accepted result declares ``None`` and publishes no
    weighted-fusion evidence at all. That is not a lenient path: it is the
    absence of the only path, and a case that gates on weighted fusion records
    those gates as inapplicable for such a candidate rather than letting it
    past them.
    """
    accepted = result.accepted
    if accepted is None:
        return
    window = scene.window
    if len(accepted) != len(result.frames):
        raise CandidateRefused(
            "accepted_result_mismatch",
            f"{result.name} carries {len(accepted)} accepted result(s) for "
            f"{len(result.frames)} frame(s)",
        )
    for position, (produced, source) in enumerate(zip(result.frames, accepted, strict=True)):
        where = f"{result.name} frame {position}"
        given = codes_of(scene.frames[position].raw, window)
        if source is None:
            # The raw route. It claims nothing beyond the measurement, so what
            # is checked is that it published nothing beyond it either.
            for index, (code, state) in enumerate(
                zip(produced.codes, produced.states, strict=True)
            ):
                expected = MEASURED if given[index] != NO_RETURN_SAMPLE else MISSING
                if state != expected or code != given[index]:
                    raise CandidateRefused(
                        "accepted_result_mismatch",
                        f"{where} offers raw measurement as its accepted result and publishes "
                        f"{state!r} at code {code} where the raw frame has {given[index]}",
                    )
            continue
        if isinstance(source, depth_completion.CompletedPatch):
            patch, states = source.source, _PRODUCT_STATES
        else:
            patch, states = source.target.patch, _TEMPORAL_STATES
        if (source.width, source.height) != (window.width, window.height):
            raise CandidateRefused(
                "accepted_result_mismatch",
                f"{where} carries a {source.width}x{source.height} accepted result for a "
                f"{window.width}x{window.height} window",
            )
        if patch.codes != given:
            raise CandidateRefused(
                "accepted_result_mismatch",
                f"{where} carries an accepted result derived from raw codes that are not this "
                "frame's; a result about other bytes says nothing about these",
            )
        for index, name in enumerate(source.states()):
            if states[name] != produced.states[index]:
                raise CandidateRefused(
                    "accepted_result_mismatch",
                    f"{where} publishes {produced.states[index]!r} at sample {index} and the "
                    f"accepted product result it carries says {states[name]!r}",
                )
            if source.codes[index] != produced.codes[index] or float(
                source.confidence[index]
            ) != float(produced.confidence[index]):
                raise CandidateRefused(
                    "accepted_result_mismatch",
                    f"{where} publishes code {produced.codes[index]} at confidence "
                    f"{produced.confidence[index]!r} for sample {index}, and the accepted "
                    f"product result says {source.codes[index]} at "
                    f"{float(source.confidence[index])!r}",
                )


def _validate_confirmations(where: str, window: Window, produced: CompletedFrame) -> None:
    """What a candidate says independent evidence concluded, against what it published.

    A confirmation is a claim about a sample that some OTHER stage proposed, so
    it can only be checked against the state the same result carries. Three
    rules close the ways it could be decorative. A sample whose spatial estimate
    independent evidence CONFIRMED carries temporal provenance, because it now
    rests on that evidence. A sample independent evidence CONTRADICTED carries
    it too, because a contradiction is only reachable when that evidence settled
    the sample itself, and what is published there is the independent value
    rather than the overruled estimate. And a sample independent evidence NEVER
    OBSERVED keeps its spatial provenance explicitly, because nothing has
    vouched for it and a temporal label would say something has.
    """
    if not produced.confirmations:
        return
    if len(produced.confirmations) != window.sample_count:
        raise CandidateRefused(
            "sample_count_mismatch",
            f"{where} carries {len(produced.confirmations)} confirmation(s) for a window of "
            f"{window.sample_count} sample(s)",
        )
    for index, confirmation in enumerate(produced.confirmations):
        u, v = window.pixel_of(index)
        at = f"{where} sample ({u}, {v})"
        if confirmation not in CONFIRMATIONS:
            raise CandidateRefused(
                "unknown_confirmation",
                f"{at} states {confirmation!r}, which is not a confirmation state",
            )
        state = produced.states[index]
        if confirmation in (CONFIRMED, CONTRADICTED) and state != TEMPORALLY_RECONSTRUCTED:
            raise CandidateRefused(
                "confirmation_inconsistent",
                f"{at} says independent evidence {confirmation!r} a spatial estimate and calls "
                f"the sample {state!r}; either verdict is only reachable when that evidence "
                "settled the sample, so the sample rests on it and carries the provenance it "
                "gives",
            )
        if confirmation == NOT_OBSERVED and state != SPATIALLY_RECONSTRUCTED:
            raise CandidateRefused(
                "confirmation_inconsistent",
                f"{at} says a spatial estimate was never observed and calls the sample "
                f"{state!r}; an estimate no independent evidence reached keeps its spatial "
                "provenance rather than being quietly relabelled",
            )
        if confirmation == NO_SPATIAL_ESTIMATE and state == SPATIALLY_RECONSTRUCTED:
            raise CandidateRefused(
                "confirmation_inconsistent",
                f"{at} is spatially reconstructed and says no spatial estimate was made there",
            )


def _validate_temporal_work(scene: Scene, result: CandidateResult) -> None:
    """Hold a declared temporal work record to what this module counted for itself.

    The same rule as :func:`_validate_work` and for the same reason: every
    field of a record about the RUN is bound here to something the output
    determines. The holes are the input's own no-return samples, the accepted
    count is the temporal reconstructions the result carries, the support total
    is the other-frame support it named, and the three confirmation counts are
    the confirmation states it published. A record that flatters its own
    conservatism refuses the whole result.
    """
    work = result.temporal
    if work is None:
        return
    where = f"{result.name} temporal work"
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
    ):
        value = getattr(work, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CandidateRefused(
                "work_counts_inconsistent", f"{where} states {key} {value!r}, which is not a count"
            )
    for named in (work.rejections, work.observation_rejections):
        for reason, value in named.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CandidateRefused(
                    "work_counts_inconsistent",
                    f"{where} states {value!r} rejection(s) for {reason!r}",
                )
    holes = 0
    reconstructed = 0
    named_support = 0
    confirmations: dict[str, int] = dict.fromkeys(CONFIRMATIONS, 0)
    for index, frame in enumerate(result.frames):
        given = scene.input_codes(index)
        holes += sum(1 for code in given if code == NO_RETURN_SAMPLE)
        for position, state in enumerate(frame.states):
            if state == TEMPORALLY_RECONSTRUCTED:
                reconstructed += 1
                named_support += len(frame.support.get(position, ()))
        for confirmation in frame.confirmations:
            confirmations[confirmation] += 1
    if work.holes != holes or work.attempted != holes:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} faced {work.holes} hole(s) and attempted {work.attempted}, beside input "
            f"frames that returned no measurement at {holes}",
        )
    if work.attempted != work.accepted + work.rejected:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} attempted {work.attempted} hole(s) and accounts for "
            f"{work.accepted + work.rejected}",
        )
    if sum(work.rejections.values()) != work.rejected:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} rejected {work.rejected} hole(s) and gives reasons for "
            f"{sum(work.rejections.values())}",
        )
    if work.accepted != reconstructed:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} accepted {work.accepted} hole(s) and the result carries {reconstructed} "
            "temporal reconstruction(s)",
        )
    if work.support_samples != named_support:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} used {work.support_samples} support sample(s) and the result names "
            f"{named_support}",
        )
    for key, counted in (
        ("confirmed", confirmations[CONFIRMED]),
        ("contradicted", confirmations[CONTRADICTED]),
        ("not_observed", confirmations[NOT_OBSERVED]),
    ):
        if getattr(work, key) != counted:
            raise CandidateRefused(
                "work_counts_inconsistent",
                f"{where} states {getattr(work, key)} {key} spatial estimate(s) and the result "
                f"publishes {counted}",
            )
    radius = work.policy.get("footprint_radius")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} publishes footprint_radius {radius!r}, which is not a number of samples",
        )
    deposits = work.observations * (2 * radius + 1) ** 2
    if work.candidates > deposits:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} kept {work.candidates} candidate(s) from {work.observations} projected "
            f"observation(s), past the {deposits} a footprint of {radius} allows. One projection "
            "is deposited into every target sample of its footprint it is close enough to, so a "
            "candidate count above the observation count is legitimate and only the footprint "
            "bounds it",
        )
    if work.work_units > work.work_budget:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} spent {work.work_units} unit(s) against a budget of {work.work_budget}; "
            "the budget is derivable from the input before the run and is not advisory",
        )


def _validate_work(scene: Scene, result: CandidateResult) -> None:
    """Hold a declared work record to what this module counted for itself.

    A work record is the only part of a candidate's account that is about the
    RUN rather than about the output, so nothing downstream can recompute it -
    which is exactly why every field of it is bound here to something that can
    be. The holes are the input's own no-return samples, the accepted count is
    the reconstructions the result carries, the support total is the support it
    named, and the budget is the policy's own bound on a run of that size.
    A record that disagrees with any of them refuses the result: a completion
    that misreports what it refused is reporting a conservatism it did not
    practise.
    """
    work = result.work
    if work is None:
        return
    where = f"{result.name} work"
    counts = {
        "holes": work.holes,
        "attempted": work.attempted,
        "accepted": work.accepted,
        "rejected": work.rejected,
        "support_samples": work.support_samples,
        "work_units": work.work_units,
        "work_budget": work.work_budget,
    }
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CandidateRefused(
                "work_counts_inconsistent", f"{where} states {key} {value!r}, which is not a count"
            )
    for reason, value in work.rejections.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CandidateRefused(
                "work_counts_inconsistent",
                f"{where} states {value!r} rejection(s) for {reason!r}",
            )
    holes = 0
    reconstructed = 0
    named_support = 0
    for index, frame in enumerate(result.frames):
        given = scene.input_codes(index)
        holes += sum(1 for code in given if code == NO_RETURN_SAMPLE)
        for position, state in enumerate(frame.states):
            if state not in RECONSTRUCTED_STATES:
                continue
            reconstructed += 1
            named_support += len(frame.support.get(position, ()))
    if work.holes != holes:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} says it faced {work.holes} hole(s) and its input returned no measurement "
            f"at {holes}",
        )
    if work.attempted != work.accepted + work.rejected:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} attempted {work.attempted} hole(s) and accounts for "
            f"{work.accepted + work.rejected}",
        )
    if work.attempted != work.holes:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} attempted {work.attempted} of {work.holes} hole(s); a hole nobody looked "
            "at is a refusal with no reason",
        )
    if sum(work.rejections.values()) != work.rejected:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} rejected {work.rejected} hole(s) and gives reasons for "
            f"{sum(work.rejections.values())}",
        )
    if work.accepted != reconstructed:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} accepted {work.accepted} hole(s) and the result carries {reconstructed} "
            "reconstruction(s)",
        )
    if work.support_samples != named_support:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} used {work.support_samples} support sample(s) and the result names "
            f"{named_support}",
        )
    if work.work_units > work.work_budget:
        raise CandidateRefused(
            "work_counts_inconsistent",
            f"{where} spent {work.work_units} unit(s) against a budget of {work.work_budget}; "
            "the budget is derivable from the input before the run and is not advisory",
        )


def _validate_sample(
    *,
    where: str,
    index: int,
    window: Window,
    code: int,
    state: str,
    confidence: float,
    raw_code: int,
    support: Sequence[str],
    frame_index: int,
    measured_inputs: Sequence[set[int]],
) -> None:
    u, v = window.pixel_of(index)
    at = f"{where} sample ({u}, {v})"
    if state not in STATES:
        raise CandidateRefused("unknown_state", f"{at} states {state!r}, which is not a state")
    if isinstance(confidence, bool) or not isinstance(confidence, float | int):
        raise CandidateRefused("confidence_out_of_range", f"{at} confidence {confidence!r}")
    if not math.isfinite(confidence) or not 0.0 <= float(confidence) <= 1.0:
        raise CandidateRefused(
            "confidence_out_of_range",
            f"{at} carries confidence {confidence!r}; confidence is a finite fraction of 1",
        )
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= MAX_CODE:
        raise CandidateRefused("depth_code_out_of_range", f"{at} carries depth code {code!r}")
    if state == MEASURED:
        if raw_code == NO_RETURN_SAMPLE:
            raise CandidateRefused(
                "measured_value_changed",
                f"{at} is called a measurement and its raw sample was zero, which is no "
                "measurement at all",
            )
        if code != raw_code:
            raise CandidateRefused(
                "measured_value_changed",
                f"{at} reports {code} where the raw sample says {raw_code}",
            )
    if state == MISSING and code != NO_RETURN_SAMPLE:
        raise CandidateRefused(
            "missing_carries_depth", f"{at} is declared missing and carries depth {code}"
        )
    if state in RECONSTRUCTED_STATES and code == NO_RETURN_SAMPLE:
        raise CandidateRefused(
            "reconstructed_without_depth", f"{at} is reconstructed and carries no depth"
        )
    if state not in RECONSTRUCTED_STATES:
        if support:
            raise CandidateRefused(
                "support_without_reconstruction",
                f"{at} is {state!r} and names support {list(support)}; support belongs to a "
                "reconstruction",
            )
        return
    if not support:
        raise CandidateRefused(
            "unsupported_reconstruction",
            f"{at} is reconstructed and names no support. A value with no named evidence is "
            "invention, whatever its confidence says",
        )
    for name in support:
        match = SUPPORT_NAME.match(name) if isinstance(name, str) else None
        if match is None:
            raise CandidateRefused(
                "unsupported_reconstruction", f"{at} names support {name!r}, which is not a sample"
            )
        source_frame, source_u, source_v = (int(part) for part in match.groups())
        if source_frame >= len(measured_inputs):
            raise CandidateRefused(
                "unsupported_reconstruction",
                f"{at} names support in frame {source_frame}, which the case does not have",
            )
        if state == SPATIALLY_RECONSTRUCTED and source_frame != frame_index:
            raise CandidateRefused(
                "spatial_support_not_this_frame",
                f"{at} is spatially reconstructed and names frame {source_frame}",
            )
        if state == TEMPORALLY_RECONSTRUCTED and source_frame == frame_index:
            raise CandidateRefused(
                "temporal_support_not_another_frame",
                f"{at} is temporally reconstructed and names its own frame",
            )
        source_index = window.index_of(source_u, source_v)
        if source_index is None or source_index not in measured_inputs[source_frame]:
            raise CandidateRefused(
                "support_not_measured",
                f"{at} names support ({source_u}, {source_v}) of frame {source_frame}, which "
                "returned no measurement there",
            )


def _measured_indices(raw: bytes, window: Window) -> tuple[int, ...]:
    codes = codes_of(raw, window)
    return tuple(index for index, code in enumerate(codes) if code != NO_RETURN_SAMPLE)


def support_name(frame: int, u: int, v: int) -> str:
    """The one spelling of a support reference."""
    return f"frame:{frame}:pixel:{u},{v}"


# --- the measurements -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageEvidence:
    """Where every sample of the scored domain ended up, against fixed denominators.

    The denominators are the whole point. ``domain_samples`` is the window
    times the frames and never anything else, so a candidate that returns
    fewer samples, or that declares its hard ones missing, moves a fraction
    DOWN rather than out of the arithmetic. Measured, reconstructed and
    missing coverage are reported separately, because a single "coverage"
    number is exactly the number a dense invented fill improves.
    """

    name: str
    domain_samples: int
    truth_surface: int
    truth_empty: int
    input_measured: int
    input_dropout: int
    measured: int
    spatially_reconstructed: int
    temporally_reconstructed: int
    missing: int
    reconstructed_on_surface: int
    false_fill: int
    #: The subset of :attr:`reconstructed_on_surface` that carries temporal
    #: provenance: samples the input did not return, which another posed frame's
    #: raw measurements put back, on surface the truth really has. This is the
    #: only count that a pose-compensated stage's coverage gain may be read
    #: from. A refusal, a contradiction and a spatial estimate are all evidence
    #: and none of them is a recovered sample.
    temporally_reconstructed_on_surface: int = 0
    unit: str = FRACTION_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_coverage"

    @property
    def reconstructed(self) -> int:
        return self.spatially_reconstructed + self.temporally_reconstructed

    @property
    def supported(self) -> int:
        """Measured samples plus reconstructions that land on real surface."""
        return self.measured + self.reconstructed_on_surface

    def _over_domain(self, count: int) -> float:
        return count / self.domain_samples

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "domain_samples": self.domain_samples,
            "truth_surface": self.truth_surface,
            "truth_empty": self.truth_empty,
            "input_measured": self.input_measured,
            "input_dropout": self.input_dropout,
            "measured": self.measured,
            "spatially_reconstructed": self.spatially_reconstructed,
            "temporally_reconstructed": self.temporally_reconstructed,
            "reconstructed": self.reconstructed,
            "missing": self.missing,
            "reconstructed_on_surface": self.reconstructed_on_surface,
            "false_fill": self.false_fill,
            "temporally_reconstructed_on_surface": self.temporally_reconstructed_on_surface,
            "supported": self.supported,
            "measured_fraction": self._over_domain(self.measured),
            "reconstructed_fraction": self._over_domain(self.reconstructed),
            "missing_fraction": self._over_domain(self.missing),
            "supported_fraction": self._over_domain(self.supported),
            "false_fill_fraction": (
                0.0 if self.truth_empty == 0 else self.false_fill / self.truth_empty
            ),
            "dropout_recovered_fraction": (
                0.0
                if self.input_dropout == 0
                else self.reconstructed_on_surface / self.input_dropout
            ),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class EdgeEvidence:
    """What a completion did at a foreground/background depth discontinuity.

    ``bridging`` counts reconstructed samples whose depth lands in the empty
    band BETWEEN the two surfaces - the flying-pixel signature of an
    interpolation that crossed the step. ``misplaced`` counts reconstructed
    samples put on the wrong side of it. Both are counts over a fixed
    denominator, and both are zero for a completion that left the edge alone.
    """

    name: str
    rows_scored: int
    band_low: float
    band_high: float
    region_samples: int
    reconstructed_in_region: int
    bridging: int
    misplaced: int
    unit: str = COUNT_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_edge"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "rows_scored": self.rows_scored,
            "band_low": self.band_low,
            "band_high": self.band_high,
            "region_samples": self.region_samples,
            "reconstructed_in_region": self.reconstructed_in_region,
            "bridging": self.bridging,
            "misplaced": self.misplaced,
            "bridging_fraction": (
                0.0 if self.region_samples == 0 else self.bridging / self.region_samples
            ),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class SlotEvidence:
    """Whether a gap that is really there is still there.

    A completion may reconstruct inside the slot - the background is real
    surface and recovering it is the point. What it may not do is put
    FOREGROUND depth there, which closes the gap and is the single most
    convincing-looking way to gain coverage.
    """

    name: str
    slot_samples: int
    slot_reconstructed: int
    closed_samples: int
    unit: str = COUNT_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_slot"

    @property
    def open(self) -> bool:
        return self.closed_samples == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "slot_samples": self.slot_samples,
            "slot_reconstructed": self.slot_reconstructed,
            "closed_samples": self.closed_samples,
            "open": self.open,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class FusionEvidence:
    """What the unchanged TSDF did with the completed frames.

    ``unobserved_voxels`` is carried because it is the invariant a completion
    must not quietly break: a voxel nothing saw keeps zero weight, and space
    does not become observed by being filled in an image.
    """

    name: str
    frames_offered: int
    frames_integrated: int
    observed_voxels: int
    unobserved_voxels: int
    surface_extracted: bool
    surface_vertices: int
    surface_faces: int
    unit: str = COUNT_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_fusion"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "frames_offered": self.frames_offered,
            "frames_integrated": self.frames_integrated,
            "observed_voxels": self.observed_voxels,
            "unobserved_voxels": self.unobserved_voxels,
            "surface_extracted": self.surface_extracted,
            "surface_vertices": self.surface_vertices,
            "surface_faces": self.surface_faces,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class WeightedFusionEvidence:
    """What the SAME TSDF did with the same frames once provenance carried a weight.

    Published beside :class:`FusionEvidence` and never instead of it, because
    the two answer different questions and the pair is the evidence:

    * ``fusion`` is what uniform fusion does with the candidate's completed
      codes. Every sample it published - measurement, confirmed reconstruction
      and unconfirmed single-frame guess alike - weighs exactly one. That is
      the counterfactual, and this record carries its two headline numbers
      under ``unweighted_*`` so the comparison survives into a case that does
      not gate the uniform record at all.
    * ``weighted_fusion`` is what the accepted weighted input contract does
      with the same frames: a measurement at full weight, an accepted temporal
      reconstruction at a strictly lower weight derived from its own
      confidence, and an unconfirmed spatial estimate or a missing sample at
      exactly zero.

    ``matches_measured_only`` is the invariant that makes "zero is absence"
    checkable rather than asserted. It is ``True`` when this field is
    BYTE-IDENTICAL - values and accumulated weights both - to the field the
    same frames' raw measurements alone produce. A completion that only ever
    proposed unconfirmed spatial estimates must leave it ``True``: it changed
    the picture and it did not change the geometry.

    ``spatially_excluded_contribution`` is carried for the same reason
    :class:`FusionEvidence` carries ``unobserved_voxels``: it is the number a
    later relaxation would move, and a report in which it is not zero is a
    report about a different contract.

    The output-facing comparison, and why it is the one that counts
    ---------------------------------------------------------------
    ``measured_only_*`` is the SAME raw multi-view field the
    ``matches_measured_only`` invariant is taken against - the same frames, the
    same poses, the same volume, the same extraction - measured for what it
    puts out rather than only compared byte for byte. ``observed_voxel_gain``,
    ``surface_vertex_gain`` and ``surface_coverage_gain`` are the weighted
    field's output measures minus those, and they are the only numbers in this
    record that could be a geometric gain. ``voxel_updates`` and
    ``contribution`` are accounting: they say evidence reached the field, not
    that the field came out better, and this corpus's own numbers are why the
    distinction is kept - see the negative result below.

    ``measured_only_frames`` exists so a reader can refuse the one way this
    comparison could be flattered: fusing the weighted target frame against a
    baseline that was not given the support frames. It is the frame count the
    baseline was actually given, and the strict reader requires it to be every
    frame this record was offered.

    ``withheld_support_*`` is that flattering comparison, computed on purpose
    and named for what it is. It fuses ONE frame alone - the one carrying the
    most temporal reconstructions - raw and then weighted, so the support
    frames those reconstructions came out of are absent from both. It is the
    measurement behind this task's negative result: the reconstructions do
    carry geometry a lone frame lacks, and that geometry is already in the
    multi-view baseline, because it came out of the very frames that baseline
    fuses. It carries no gate and is not evidence of any improvement.
    """

    name: str
    adapter: str
    frames_offered: int
    frames_integrated: int
    measured_samples: int
    temporally_reconstructed_samples: int
    spatially_excluded_samples: int
    missing_samples: int
    offered_weight: float
    voxel_updates: int
    zero_weight_observations: int
    contribution: float
    spatially_excluded_contribution: float
    observed_voxels: int
    unobserved_voxels: int
    surface_extracted: bool
    surface_vertices: int
    surface_faces: int
    matches_measured_only: bool
    measured_only_frames: int
    measured_only_observed_voxels: int
    measured_only_surface_vertices: int
    measured_only_surface_coverage: float
    surface_coverage: float
    observed_voxel_gain: int
    surface_vertex_gain: int
    surface_coverage_gain: float
    withheld_support_frames: int
    withheld_support_observed_voxels: int
    withheld_support_weighted_observed_voxels: int
    unweighted_observed_voxels: int
    unweighted_surface_vertices: int
    refusals: int
    unit: str = COUNT_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_weighted_fusion"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "adapter": self.adapter,
            "frames_offered": self.frames_offered,
            "frames_integrated": self.frames_integrated,
            "measured_samples": self.measured_samples,
            "temporally_reconstructed_samples": self.temporally_reconstructed_samples,
            "spatially_excluded_samples": self.spatially_excluded_samples,
            "missing_samples": self.missing_samples,
            "offered_weight": self.offered_weight,
            "voxel_updates": self.voxel_updates,
            "zero_weight_observations": self.zero_weight_observations,
            "contribution": self.contribution,
            "spatially_excluded_contribution": self.spatially_excluded_contribution,
            "observed_voxels": self.observed_voxels,
            "unobserved_voxels": self.unobserved_voxels,
            "surface_extracted": self.surface_extracted,
            "surface_vertices": self.surface_vertices,
            "surface_faces": self.surface_faces,
            "matches_measured_only": self.matches_measured_only,
            "measured_only_frames": self.measured_only_frames,
            "measured_only_observed_voxels": self.measured_only_observed_voxels,
            "measured_only_surface_vertices": self.measured_only_surface_vertices,
            "measured_only_surface_coverage": self.measured_only_surface_coverage,
            "surface_coverage": self.surface_coverage,
            "observed_voxel_gain": self.observed_voxel_gain,
            "surface_vertex_gain": self.surface_vertex_gain,
            "surface_coverage_gain": self.surface_coverage_gain,
            "withheld_support_frames": self.withheld_support_frames,
            "withheld_support_observed_voxels": self.withheld_support_observed_voxels,
            "withheld_support_weighted_observed_voxels": (
                self.withheld_support_weighted_observed_voxels
            ),
            "unweighted_observed_voxels": self.unweighted_observed_voxels,
            "unweighted_surface_vertices": self.unweighted_surface_vertices,
            "refusals": self.refusals,
            "unit": self.unit,
            "counterfactual_is": (
                "unweighted_* are the same completed codes fused at uniform weight, which is "
                "what ignoring provenance would do with this candidate"
            ),
            "comparison_is": (
                "measured_only_* is the same frames, poses, volume and extraction with the raw "
                "measurements alone - measured_only_frames says how many frames that baseline "
                "was given, and it is every frame this record was offered. The three *_gain "
                "fields are this field's output measures minus that baseline's and are the only "
                "geometric claim available here; voxel_updates and contribution are accounting "
                "and are not one. withheld_support_* fuses ONLY the frames carrying an accepted "
                "completion result and is published to show what withholding the support frames "
                "would make this look like - it is not evidence of an improvement"
            ),
        }


@dataclass(frozen=True, slots=True)
class TrackingEvidence:
    """What the real tracker did with these exact clouds, and nothing more.

    Published twice per candidate under two names that must not be confused.
    ``tracking`` is the intended measured-only route - the tracker over each
    frame's own returned samples, which is the only route this project's
    tracker has - and it is what the tracking gates are stated over.
    ``tracking_completed_cloud`` is the counterfactual replay over the
    candidate's completed cloud, published because a completion fed to a
    tracker can degrade it and that fact is the measurement behind this task's
    no-integration boundary. See :func:`_tracking`.
    """

    name: str
    frames_offered: int
    frames_accepted: int
    origins: tuple[str, ...]
    failure_types: tuple[str, ...]
    refusals: tuple[str, ...]
    unit: str = COUNT_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_tracking"

    @property
    def accepted_all(self) -> bool:
        return self.frames_accepted == self.frames_offered

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "frames_offered": self.frames_offered,
            "frames_accepted": self.frames_accepted,
            "accepted_all": self.accepted_all,
            "origins": list(self.origins),
            "failure_types": list(self.failure_types),
            "refusals": list(self.refusals),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceEvidence:
    """What a candidate's own confidence was worth, against held-out truth.

    Computed HERE, from the result and the truth, and never taken from the
    candidate: the numbers below are what its confidence values actually
    predicted, not what it says they mean. Three facts are carried, and each
    one closes a different way a confidence could be decorative:

    * the whole distribution, as five fixed bins whose populations add up to
      the reconstructions coverage counted - so a candidate cannot improve its
      ordering by omitting the bin it did worst in;
    * what happened at and above :data:`CONFIDENCE_THRESHOLD`, including the
      worst error any confident reconstruction actually made - so confidence
      is a claim that can be wrong rather than a label;
    * the mean confidence on real surface beside the mean in empty space - so
      a completion that is most sure exactly where it invented geometry says
      so in its own artifact.

    This is not a camera confidence and nothing here relabels a
    reconstruction. It measures a completion's self-assessment, which is a
    different thing from the sensor's.
    """

    name: str
    reconstructed: int
    on_surface: int
    false_fill: int
    minimum: float
    maximum: float
    mean: float
    on_surface_mean: float
    false_fill_mean: float
    surface_advantage: float
    threshold: float
    high_confidence: int
    high_confidence_on_surface: int
    high_confidence_false_fill: int
    high_confidence_worst: float
    bins: tuple[Mapping[str, object], ...]
    unit: str = FRACTION_UNIT
    error_unit: str = DEPTH_UNIT
    truth_source: str = TRUTH_SOURCE

    @property
    def kind(self) -> str:
        return "depth_completeness_confidence"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "reconstructed": self.reconstructed,
            "on_surface": self.on_surface,
            "false_fill": self.false_fill,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "on_surface_mean": self.on_surface_mean,
            "false_fill_mean": self.false_fill_mean,
            "surface_advantage": self.surface_advantage,
            "threshold": self.threshold,
            "high_confidence": self.high_confidence,
            "high_confidence_on_surface": self.high_confidence_on_surface,
            "high_confidence_false_fill": self.high_confidence_false_fill,
            "high_confidence_worst": self.high_confidence_worst,
            "bins": [dict(entry) for entry in self.bins],
            "unit": self.unit,
            "error_unit": self.error_unit,
            "truth_source": self.truth_source,
        }


@dataclass(frozen=True, slots=True)
class WorkEvidence:
    """What a candidate says its own run cost, after :func:`validate` bound it.

    Reported so conservatism is inspectable. Recovering little because the
    frame was genuinely ambiguous and recovering little because nothing was
    attempted produce the same output and are entirely different facts, and
    the rejection reasons are the only place that difference is visible. The
    policy is published beside them so an accepted sample is reproducible from
    its named support and these numbers alone.
    """

    name: str
    backend: str
    policy: Mapping[str, object]
    holes: int
    attempted: int
    accepted: int
    rejected: int
    rejections: Mapping[str, int]
    support_samples: int
    work_units: int
    work_budget: int
    unit: str = COUNT_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_work"

    @property
    def within_budget(self) -> bool:
        return self.work_units <= self.work_budget

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "backend": self.backend,
            "policy": dict(self.policy),
            "holes": self.holes,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejections": {key: self.rejections[key] for key in sorted(self.rejections)},
            "support_samples": self.support_samples,
            "work_units": self.work_units,
            "work_budget": self.work_budget,
            "within_budget": self.within_budget,
            "accepted_fraction": (0.0 if self.holes == 0 else self.accepted / self.holes),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class TemporalWorkEvidence:
    """What a candidate says its pose-compensated run cost, after :func:`validate`.

    Beside the spatial :class:`WorkEvidence` rather than replacing it, because
    a candidate that runs both stages spent both, and one combined total would
    hide which of them refused what. The two rejection vocabularies are carried
    whole - every reason at zero when it never fired - so the shape of a run's
    conservatism is stable across cases rather than being whichever keys
    happened to be non-empty.
    """

    name: str
    backend: str
    policy: Mapping[str, object]
    source_frames: int
    eligible_samples: int
    observations: int
    observation_rejections: Mapping[str, int]
    candidates: int
    holes: int
    attempted: int
    accepted: int
    rejected: int
    rejections: Mapping[str, int]
    confirmed: int
    contradicted: int
    not_observed: int
    support_samples: int
    work_units: int
    work_budget: int
    unit: str = COUNT_UNIT

    @property
    def kind(self) -> str:
        return "depth_completeness_temporal_work"

    @property
    def within_budget(self) -> bool:
        return self.work_units <= self.work_budget

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "backend": self.backend,
            "policy": dict(self.policy),
            "source_frames": self.source_frames,
            "eligible_samples": self.eligible_samples,
            "observations": self.observations,
            "observation_rejections": {
                key: self.observation_rejections[key] for key in sorted(self.observation_rejections)
            },
            "candidates": self.candidates,
            "holes": self.holes,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejections": {key: self.rejections[key] for key in sorted(self.rejections)},
            "confirmed": self.confirmed,
            "contradicted": self.contradicted,
            "not_observed": self.not_observed,
            "support_samples": self.support_samples,
            "work_units": self.work_units,
            "work_budget": self.work_budget,
            "within_budget": self.within_budget,
            "accepted_fraction": (0.0 if self.holes == 0 else self.accepted / self.holes),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ConfirmationEvidence:
    """What each confirmation was worth, against held-out truth.

    Computed HERE, from the result and the truth, and never taken from the
    candidate. The one number this measurement exists for is
    ``confirmed_false``: a spatial estimate standing in space the scene really
    leaves empty, which independent posed evidence nevertheless CONFIRMED. That
    is the specific failure a temporal stage is most likely to produce and the
    hardest to see - the estimate was already there, the confirmation only
    blessed it, and the result gets a stronger provenance label for a sample
    that is not on any surface.

    ``not_observed_false`` is its counterpart and is not a failure: a spatial
    estimate in empty space that independent evidence declined to confirm is
    the ambiguity being disclosed rather than laundered.
    """

    name: str
    spatial_estimates: int
    confirmed: int
    confirmed_on_surface: int
    confirmed_false: int
    contradicted: int
    contradicted_on_surface: int
    not_observed: int
    not_observed_false: int
    unit: str = COUNT_UNIT
    truth_source: str = TRUTH_SOURCE

    @property
    def kind(self) -> str:
        return "depth_completeness_confirmation"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "spatial_estimates": self.spatial_estimates,
            "confirmed": self.confirmed,
            "confirmed_on_surface": self.confirmed_on_surface,
            "confirmed_false": self.confirmed_false,
            "contradicted": self.contradicted,
            "contradicted_on_surface": self.contradicted_on_surface,
            "not_observed": self.not_observed,
            "not_observed_false": self.not_observed_false,
            "confirmed_fraction": (
                0.0 if self.spatial_estimates == 0 else self.confirmed / self.spatial_estimates
            ),
            "unit": self.unit,
            "truth_source": self.truth_source,
        }


# --- scoring one candidate ------------------------------------------------------------


def _depth(code: int, depth_scale: float) -> float:
    return code * depth_scale


def _cloud_from(frame: CompletedFrame, scene: Scene) -> tuple[Point, ...]:
    return cloud_of(
        frame.codes,
        scene.window,
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )


def _measured_cloud_from(frame: CompletedFrame, scene: Scene) -> tuple[Point, ...]:
    """The frame's own returned samples, and nothing a completion put there.

    Built from the candidate's own output rather than from the scene's input,
    so a candidate that damaged a measurement it was told to preserve moves
    this cloud - while every candidate that preserved them produces exactly the
    baseline's. Reconstructions are dropped back to no-return, which is what
    the tracker sees in the intended dataflow: this project's tracker consumes
    measured geometry only.
    """
    codes = tuple(
        code if state == MEASURED else NO_RETURN_SAMPLE
        for code, state in zip(frame.codes, frame.states, strict=True)
    )
    return cloud_of(
        codes,
        scene.window,
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )


def _truth_cloud(scene: Scene, index: int, *, pose: Rigid | None = None) -> tuple[Point, ...]:
    return cloud_of(
        scene.truth_codes(index),
        scene.window,
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
        pose=pose,
    )


def _extents(points: Sequence[Point]) -> dict[str, float]:
    """Three named extents of a cloud, by one rule applied to both sides.

    Not a shape descriptor. It is the smallest measurement that moves when a
    completion smooths an object inward, which is the failure a denser result
    is most likely to hide.
    """
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    return {
        "extent_x": max(xs) - min(xs),
        "extent_y": max(ys) - min(ys),
        "extent_z": max(zs) - min(zs),
    }


def _landmarks(points: Sequence[Point]) -> dict[str, Point]:
    """Four extreme points, chosen by a total order so ties cannot wobble."""
    ordered = sorted(points)
    return {
        "leftmost": min(ordered, key=lambda p: (p[0], p[1], p[2])),
        "rightmost": max(ordered, key=lambda p: (p[0], p[1], p[2])),
        "nearest": min(ordered, key=lambda p: (p[2], p[0], p[1])),
        "farthest": max(ordered, key=lambda p: (p[2], p[0], p[1])),
    }


def _coverage(scene: Scene, result: CandidateResult) -> CoverageEvidence:
    window = scene.window
    domain = window.sample_count * len(scene.frames)
    truth_surface = truth_empty = input_measured = input_dropout = 0
    measured = spatial = temporal = missing = 0
    on_surface = false_fill = temporal_on_surface = 0
    for index, frame in enumerate(result.frames):
        truth = scene.truth_codes(index)
        given = scene.input_codes(index)
        for position, state in enumerate(frame.states):
            has_truth = truth[position] != NO_RETURN_SAMPLE
            truth_surface += int(has_truth)
            truth_empty += int(not has_truth)
            returned = given[position] != NO_RETURN_SAMPLE
            input_measured += int(returned)
            input_dropout += int(has_truth and not returned)
            if state == MEASURED:
                measured += 1
            elif state == MISSING:
                missing += 1
            else:
                spatial += int(state == SPATIALLY_RECONSTRUCTED)
                temporal += int(state == TEMPORALLY_RECONSTRUCTED)
                on_surface += int(has_truth)
                false_fill += int(not has_truth)
                temporal_on_surface += int(has_truth and state == TEMPORALLY_RECONSTRUCTED)
    return CoverageEvidence(
        name="coverage",
        domain_samples=domain,
        truth_surface=truth_surface,
        truth_empty=truth_empty,
        input_measured=input_measured,
        input_dropout=input_dropout,
        measured=measured,
        spatially_reconstructed=spatial,
        temporally_reconstructed=temporal,
        missing=missing,
        reconstructed_on_surface=on_surface,
        false_fill=false_fill,
        temporally_reconstructed_on_surface=temporal_on_surface,
    )


@dataclass(frozen=True, slots=True)
class ReconstructedError:
    """Error on held-out truth, summarised by the authority that computed it.

    The per-sample entries are deliberately not carried into the artifact:
    there are up to a thousand of them per candidate and a report nobody can
    read is a report nobody checks. What is carried is what
    :func:`scanner3d.quality.dimension_errors` returned - its own
    distributions, unedited - beside the counts that say which population they
    summarise, so a distribution cannot be read without its denominator.
    """

    name: str
    entry_count: int
    dropout_samples: int
    reconstructed_on_surface: int
    absolute: Mapping[str, object]
    relative: Mapping[str, object] | None
    unit: str
    truth_source: str = TRUTH_SOURCE

    @property
    def kind(self) -> str:
        return "depth_completeness_reconstructed_error"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "entry_count": self.entry_count,
            "dropout_samples": self.dropout_samples,
            "reconstructed_on_surface": self.reconstructed_on_surface,
            "attempted_fraction": (
                0.0 if self.dropout_samples == 0 else self.entry_count / self.dropout_samples
            ),
            "absolute": dict(self.absolute),
            "relative": None if self.relative is None else dict(self.relative),
            "unit": self.unit,
            "truth_source": self.truth_source,
        }


def _reconstructed_depth(
    scene: Scene, result: CandidateResult, coverage: CoverageEvidence
) -> ReconstructedError | None:
    """Error on held-out truth, and only there.

    Measured samples are excluded deliberately: they are the input, and
    scoring a completion on values it copied would let a result that
    reconstructed nothing report a superb error. Samples reconstructed into
    empty space are excluded too - they have no truth to be near, and they are
    counted as false fill instead, where they cannot be averaged away.
    """
    measured: dict[str, float] = {}
    truth_values: dict[str, float] = {}
    for index, frame in enumerate(result.frames):
        truth = scene.truth_codes(index)
        for position, state in enumerate(frame.states):
            if state not in RECONSTRUCTED_STATES or truth[position] == NO_RETURN_SAMPLE:
                continue
            u, v = scene.window.pixel_of(position)
            key = f"f{index}:{u},{v}"
            measured[key] = _depth(frame.codes[position], scene.depth_scale)
            truth_values[key] = _depth(truth[position], scene.depth_scale)
    if not truth_values:
        return None
    errors = quality.dimension_errors(
        measured,
        truth_values,
        name="reconstructed_depth",
        truth_source=TRUTH_SOURCE,
        unit=DEPTH_UNIT,
    )
    return ReconstructedError(
        name="reconstructed_depth",
        entry_count=len(errors.entries),
        dropout_samples=coverage.input_dropout,
        reconstructed_on_surface=coverage.reconstructed_on_surface,
        absolute=errors.absolute.as_dict(),
        relative=None if errors.relative is None else errors.relative.as_dict(),
        unit=DEPTH_UNIT,
    )


def _edge(scene: Scene, result: CandidateResult) -> tuple[EdgeEvidence, quality.DimensionErrors]:
    window = scene.window
    threshold = scene.regions.foreground_below
    if threshold is None:
        raise ScoringUnusable(f"{scene.case_id} scores an edge without naming its foreground")
    truth = scene.truth_codes(0)
    frame = result.frames[0]
    foreground_depths = [
        _depth(code, scene.depth_scale)
        for code in truth
        if code != NO_RETURN_SAMPLE and code <= threshold
    ]
    background_depths = [
        _depth(code, scene.depth_scale)
        for code in truth
        if code != NO_RETURN_SAMPLE and code > threshold
    ]
    if not foreground_depths or not background_depths:
        raise ScoringUnusable(f"{scene.case_id} has no depth step to score")
    band_low = max(foreground_depths)
    band_high = min(background_depths)
    if not band_high > band_low:
        raise ScoringUnusable(
            f"{scene.case_id} states a foreground code of {threshold}, which does not separate "
            "its two surfaces"
        )
    margin = scene.depth_scale
    bridging = misplaced = reconstructed = 0
    for position, state in enumerate(frame.states):
        if state not in RECONSTRUCTED_STATES:
            continue
        reconstructed += 1
        depth = _depth(frame.codes[position], scene.depth_scale)
        if band_low + margin < depth < band_high - margin:
            bridging += 1
        truth_code = truth[position]
        if truth_code == NO_RETURN_SAMPLE:
            continue
        truth_is_foreground = truth_code <= threshold
        placed_foreground = depth <= band_low + margin
        if truth_is_foreground != placed_foreground:
            misplaced += 1

    measured_columns: dict[str, float] = {}
    truth_columns: dict[str, float] = {}
    for v, indices in window.rows():
        row_truth = [truth[index] for index in indices]
        foreground = [
            column
            for column, code in enumerate(row_truth)
            if code != NO_RETURN_SAMPLE and code <= threshold
        ]
        background = [
            column
            for column, code in enumerate(row_truth)
            if code != NO_RETURN_SAMPLE and code > threshold
        ]
        if not foreground or not background:
            continue
        key = f"row:{v}"
        truth_columns[key] = float(max(foreground) + 1)
        row_codes = [frame.codes[index] for index in indices]
        row_states = [frame.states[index] for index in indices]
        placed = [
            column
            for column, (code, state) in enumerate(zip(row_codes, row_states, strict=True))
            if state != MISSING and code != NO_RETURN_SAMPLE and code <= threshold
        ]
        measured_columns[key] = float(max(placed) + 1) if placed else 0.0
    if not truth_columns:
        raise ScoringUnusable(f"{scene.case_id} has no row where the step is visible")
    displacement = quality.dimension_errors(
        measured_columns,
        truth_columns,
        name="edge_step_column",
        truth_source=TRUTH_SOURCE,
        unit=COLUMN_UNIT,
    )
    return (
        EdgeEvidence(
            name="edge",
            rows_scored=len(truth_columns),
            band_low=band_low,
            band_high=band_high,
            region_samples=window.sample_count,
            reconstructed_in_region=reconstructed,
            bridging=bridging,
            misplaced=misplaced,
        ),
        displacement,
    )


def _slot(scene: Scene, result: CandidateResult) -> SlotEvidence:
    threshold = scene.regions.foreground_below
    if threshold is None:
        raise ScoringUnusable(f"{scene.case_id} scores a slot without naming its foreground")
    window = scene.window
    frame = result.frames[0]
    slot_samples = reconstructed = closed = 0
    for _v, indices in window.rows():
        for column in scene.regions.slot_columns:
            if not 0 <= column < window.width:
                raise ScoringUnusable(f"{scene.case_id} names slot column {column}")
            position = indices[column]
            slot_samples += 1
            state = frame.states[position]
            if state not in RECONSTRUCTED_STATES:
                continue
            reconstructed += 1
            if frame.codes[position] <= threshold:
                closed += 1
    return SlotEvidence(
        name="slot",
        slot_samples=slot_samples,
        slot_reconstructed=reconstructed,
        closed_samples=closed,
    )


def _fusion(
    scene: Scene, result: CandidateResult
) -> tuple[FusionEvidence, quality.CloudAgreement | None]:
    spec = scene.regions.fusion
    if spec is None:
        raise ScoringUnusable(f"{scene.case_id} scores fusion without a volume")
    volume = TsdfVolume(spec.config(), backend="python")
    frames = [frame_of_codes(frame.codes, scene.window) for frame in result.frames]
    volume.integrate(
        frames,
        scene.poses(),
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )
    evidence = volume.evidence()
    extracted = True
    vertices: tuple[Point, ...] = ()
    faces = 0
    try:
        surface = extract_surface(volume, min_weight=spec.min_weight, backend="python")
    except NoSurfaceFound:
        extracted = False
    else:
        vertices = surface.mesh.vertices
        faces = surface.mesh.face_count
    fusion = FusionEvidence(
        name="fusion",
        frames_offered=evidence.frames_offered,
        frames_integrated=evidence.frames_integrated,
        observed_voxels=evidence.weights.observed_voxels,
        unobserved_voxels=evidence.weights.unobserved_voxels,
        surface_extracted=extracted,
        surface_vertices=len(vertices),
        surface_faces=faces,
    )
    if not vertices:
        return fusion, None
    truth_model: list[Point] = []
    for index, frame in enumerate(scene.frames):
        truth_model.extend(_truth_cloud(scene, index, pose=frame.pose))
    agreement = quality.cloud_agreement(
        vertices,
        truth_model,
        name="fusion_surface",
        coverage_tolerance=spec.coverage_tolerance,
        unit=DEPTH_UNIT,
    )
    return fusion, agreement


def _weighted_frames(
    scene: Scene, result: CandidateResult
) -> tuple[tsdf_input.WeightedDepthFrame, ...]:
    """Every frame of this candidate as accepted weighted TSDF input.

    The raw frame handed to each adapter is the case's own window codes in a
    real 512000-byte frame - the exact bytes :func:`_fusion` fuses for the
    baseline - so the adapter's binding of the accepted result to those bytes
    is checked against the same evidence the rest of this module scores.
    """
    accepted = result.accepted
    if accepted is None:
        raise ScoringUnusable(f"{result.name} has no accepted result to weight")
    window = scene.window
    origin = (window.u0, window.v0)
    frames: list[tsdf_input.WeightedDepthFrame] = []
    for index, source in enumerate(accepted):
        raw = frame_of_codes(scene.input_codes(index), window)
        if source is None:
            frames.append(tsdf_input.weighted_raw_frame(raw))
        elif isinstance(source, depth_completion.CompletedPatch):
            frames.append(tsdf_input.weighted_completed_frame(raw, source, origin=origin))
        else:
            frames.append(tsdf_input.weighted_temporal_frame(raw, source, origin=origin))
    return tuple(frames)


def _fuse_raw(
    scene: Scene,
    spec: FusionSpec,
    frames: Sequence[bytes],
    poses: Sequence[Rigid] | None = None,
) -> TsdfVolume:
    """One volume of ``spec``, with ``frames`` folded in through the raw API.

    ``poses`` defaults to the scene's own, which is every frame. A caller that
    passes fewer frames has to pass their poses with them, so a subset cannot
    be fused against the wrong placements.
    """
    volume = TsdfVolume(spec.config(), backend="python")
    volume.integrate(
        list(frames),
        list(scene.poses() if poses is None else poses),
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )
    return volume


def _surface_of(volume: TsdfVolume, spec: FusionSpec) -> tuple[Point, ...]:
    """The unchanged extractor's surface, or no vertices at all."""
    try:
        surface = extract_surface(volume, min_weight=spec.min_weight, backend="python")
    except NoSurfaceFound:
        return ()
    return surface.mesh.vertices


def _surface_coverage(
    vertices: Sequence[Point], truth: Sequence[Point], spec: FusionSpec, *, name: str
) -> tuple[quality.CloudAgreement | None, float]:
    """Truth-referenced coverage of one extracted surface, and the record behind it.

    The output-facing measure: the fraction of the truth model that has a
    surface vertex within the case's own coverage tolerance. A field that
    extracted nothing covers nothing, which is ``0.0`` rather than an absent
    number, because the comparison against another field has to exist either
    way.
    """
    if not vertices:
        return None, 0.0
    agreement = quality.cloud_agreement(
        vertices,
        truth,
        name=name,
        coverage_tolerance=spec.coverage_tolerance,
        unit=DEPTH_UNIT,
    )
    return agreement, agreement.reference_to_observed.coverage_fraction


def _weighted_fusion(
    scene: Scene, result: CandidateResult
) -> tuple[WeightedFusionEvidence, quality.CloudAgreement | None]:
    """Fuse this candidate through the accepted weighted input contract.

    Five volumes of the SAME configuration, which is what makes the record
    readable:

    1. the weighted one, through
       :meth:`scanner3d.tsdf.TsdfVolume.integrate_weighted`;
    2. a measured-only control, the same frames' raw bytes through the same
       weighted entry point. It is what ``matches_measured_only`` compares
       against byte for byte AND the raw multi-view baseline every ``*_gain``
       field is taken against - the same frames, the same poses, the same
       volume, the same extractor, the same truth model;
    3. the uniform counterfactual, the candidate's own completed codes through
       the unchanged raw API - provenance ignored entirely;
    4. and 5. the withheld-support pair: one frame alone, raw and weighted.
       That is the comparison this task is forbidden to claim a gain from,
       computed so the report can show what it would have looked like and why
       it is not the baseline.

    Nothing here decides anything. The provenance and every weight come out of
    the accepted product results the candidate carries, through
    :mod:`scanner3d.tsdf_input`, which derives the weight from the accepted
    confidence and nothing else.
    """
    spec = scene.regions.weighted_fusion
    if spec is None:
        raise ScoringUnusable(f"{scene.case_id} scores weighted fusion without a volume")
    weighted = _weighted_frames(scene, result)
    adapters = {frame.adapter for frame in weighted}
    poses = scene.poses()
    volume = TsdfVolume(spec.config(), backend="python")
    volume.integrate_weighted(
        list(weighted),
        poses,
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )
    evidence = volume.weighted_evidence()
    weights = volume.evidence().weights

    # The raw multi-view baseline. EVERY frame of the scene, at the scene's own
    # poses, in the same volume - the support frames the reconstructions came
    # out of included, which is what makes the comparison below honest rather
    # than flattering.
    raw_frames = [
        frame_of_codes(scene.input_codes(index), scene.window) for index in range(len(scene.frames))
    ]
    measured_only = TsdfVolume(spec.config(), backend="python")
    measured_only.integrate_weighted(
        [tsdf_input.weighted_raw_frame(raw) for raw in raw_frames],
        poses,
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )
    identical = bytes(volume.values) == bytes(measured_only.values) and bytes(
        volume.weights
    ) == bytes(measured_only.weights)

    counterfactual = _fuse_raw(
        scene,
        spec,
        [frame_of_codes(frame.codes, scene.window) for frame in result.frames],
    )

    # The comparison this task may NOT claim a gain from, computed anyway. One
    # frame is fused alone - raw, then weighted - so the support frames its
    # reconstructions were derived from are absent. That is exactly the
    # withholding the round-three review forbids as a route to a gain, and
    # publishing it is how this corpus shows what the gain would have been made
    # of.
    # The one frame carrying the most temporal reconstructions, alone. Ties go
    # to the lowest index, so the choice is the scene's and not a run's.
    target = max(
        range(len(weighted)),
        key=lambda index: (weighted[index].summary.temporally_reconstructed, -index),
    )
    carried = [target]
    withheld_poses = [poses[index] for index in carried]
    withheld_raw = _fuse_raw(scene, spec, [raw_frames[index] for index in carried], withheld_poses)
    withheld_weighted = TsdfVolume(spec.config(), backend="python")
    withheld_weighted.integrate_weighted(
        [weighted[index] for index in carried],
        withheld_poses,
        intrinsics=scene.intrinsics,
        depth_scale=scene.depth_scale,
    )

    truth_model: list[Point] = []
    for index, frame in enumerate(scene.frames):
        truth_model.extend(_truth_cloud(scene, index, pose=frame.pose))

    vertices = _surface_of(volume, spec)
    extracted = bool(vertices)
    faces = 0
    if extracted:
        surface = extract_surface(volume, min_weight=spec.min_weight, backend="python")
        faces = surface.mesh.face_count
    agreement, coverage = _surface_coverage(
        vertices, truth_model, spec, name="weighted_fusion_surface"
    )
    baseline_vertices = _surface_of(measured_only, spec)
    _, baseline_coverage = _surface_coverage(
        baseline_vertices, truth_model, spec, name="measured_only_surface"
    )
    counterfactual_vertices = len(_surface_of(counterfactual, spec))

    inputs = evidence.inputs
    record = WeightedFusionEvidence(
        name="weighted_fusion",
        adapter="+".join(sorted(adapters)),
        frames_offered=evidence.frames_offered,
        frames_integrated=evidence.frames_integrated,
        measured_samples=inputs.measured,
        temporally_reconstructed_samples=inputs.temporally_reconstructed,
        spatially_excluded_samples=inputs.spatially_excluded,
        missing_samples=inputs.missing,
        offered_weight=inputs.offered_weight,
        voxel_updates=evidence.voxel_updates,
        zero_weight_observations=evidence.zero_weight_observations,
        contribution=evidence.contribution,
        # Read out of the accepted evidence rather than assumed. The number is
        # zero because scanner3d.tsdf accumulated zero against that provenance
        # over this run, not because this line says so: a weighting that let an
        # unconfirmed single-frame estimate reach a voxel would land here, and
        # the case's absolute gate on it would fail.
        spatially_excluded_contribution=evidence.contribution_by_provenance["spatially_excluded"],
        observed_voxels=weights.observed_voxels,
        unobserved_voxels=weights.unobserved_voxels,
        surface_extracted=extracted,
        surface_vertices=len(vertices),
        surface_faces=faces,
        matches_measured_only=identical,
        measured_only_frames=len(raw_frames),
        measured_only_observed_voxels=measured_only.evidence().weights.observed_voxels,
        measured_only_surface_vertices=len(baseline_vertices),
        measured_only_surface_coverage=baseline_coverage,
        surface_coverage=coverage,
        observed_voxel_gain=(
            weights.observed_voxels - measured_only.evidence().weights.observed_voxels
        ),
        surface_vertex_gain=len(vertices) - len(baseline_vertices),
        surface_coverage_gain=coverage - baseline_coverage,
        withheld_support_frames=len(carried),
        withheld_support_observed_voxels=withheld_raw.evidence().weights.observed_voxels,
        withheld_support_weighted_observed_voxels=(
            withheld_weighted.evidence().weights.observed_voxels
        ),
        unweighted_observed_voxels=counterfactual.evidence().weights.observed_voxels,
        unweighted_surface_vertices=counterfactual_vertices,
        refusals=evidence.refusals,
    )
    return record, agreement


def _replay_tracking(
    scene: Scene,
    clouds: Sequence[tuple[Point, ...]],
    *,
    name: str,
    pose_name: str,
) -> tuple[TrackingEvidence, quality.PoseSequenceError | None]:
    """Replay the real tracker over one set of clouds, and compare what it placed.

    Two records, because a candidate can degrade tracking in two different
    ways. The counts and refusals say whether the frames were placed at all;
    :func:`scanner3d.quality.pose_sequence_error` says whether the poses it
    placed them at are still the scene's own. The pose comparison exists only
    when every frame was accepted - there is no pose to compare for a frame the
    tracker refused, and reporting a partial sequence as a pose error would
    flatter the run that lost frames.
    """
    tracker = IncrementalTracker(
        backend="python",
        policy=TRACKING_POLICY,
        max_correspondence_distance=TRACKING_MAX_CORRESPONDENCE,
        voxel_size=TRACKING_VOXEL_SIZE,
    )
    accepted = 0
    origins: list[str] = []
    failures: list[str] = []
    refusals: list[str] = []
    for cloud in clouds:
        decision = tracker.offer(cloud)
        if decision.accepted:
            accepted += 1
            origins.append(str(decision.origin))
        else:
            origins.append("refused")
            failures.append(str(decision.failure_type))
            refusals.extend(decision.refusals)
    evidence = TrackingEvidence(
        name=name,
        frames_offered=len(clouds),
        frames_accepted=accepted,
        origins=tuple(origins),
        failure_types=tuple(failures),
        refusals=tuple(sorted(set(refusals))),
    )
    if accepted != len(clouds) or accepted < 2:
        return evidence, None
    return evidence, quality.pose_sequence_error(
        tracker.poses,
        scene.poses(),
        name=pose_name,
        unit=DEPTH_UNIT,
    )


def _tracking(
    scene: Scene, result: CandidateResult
) -> tuple[
    TrackingEvidence,
    quality.PoseSequenceError | None,
    TrackingEvidence,
    quality.PoseSequenceError | None,
]:
    """The intended measured-only tracking route, and the counterfactual beside it.

    Two replays, and they are not interchangeable.

    ``tracking`` / ``tracking_pose`` is the **intended dataflow of this
    project**: the tracker is offered each frame's own returned samples and
    nothing a completion added. Neither the spatial stage nor the temporal one
    feeds it, so this record is the same record for every candidate that
    preserved its measurements - which is exactly the acceptance condition the
    tracking gates state, and a candidate that quietly damaged a measured
    sample would move it.

    ``tracking_completed_cloud`` / ``tracking_completed_cloud_pose`` is a
    **counterfactual**: the same tracker over the candidate's completed cloud,
    reconstructions and all. It answers a question this task does NOT act on -
    what would happen if the completion did enter the tracker - and it is
    published precisely because the answer is unfavourable and would otherwise
    be invisible. It carries no acceptance gate: it is the measurement behind
    the no-integration boundary and not a bar any candidate is held to.
    """
    measured, poses = _replay_tracking(
        scene,
        [_measured_cloud_from(frame, scene) for frame in result.frames],
        name="tracking",
        pose_name="tracking_pose",
    )
    completed, completed_poses = _replay_tracking(
        scene,
        [_cloud_from(frame, scene) for frame in result.frames],
        name="tracking_completed_cloud",
        pose_name="tracking_completed_cloud_pose",
    )
    return measured, poses, completed, completed_poses


def _confidence(
    scene: Scene, result: CandidateResult, coverage: CoverageEvidence
) -> ConfidenceEvidence | None:
    """What a candidate's confidence values were worth, bin by bin.

    Made for every candidate that reconstructed anything, and for no other:
    there is no distribution to describe when nothing was reconstructed, and a
    block of zeros would read as a calibrated completion rather than as an
    absent one. The bins partition exactly the reconstructions coverage
    counted, so the two are checkable against each other downstream.
    """
    values: list[float] = []
    on_surface: list[float] = []
    false_fill: list[float] = []
    confident_errors: list[float] = []
    confident_on_surface = confident_false_fill = 0
    bin_counts = [0] * (len(CONFIDENCE_BIN_EDGES) - 1)
    bin_on_surface = [0] * (len(CONFIDENCE_BIN_EDGES) - 1)
    for index, frame in enumerate(result.frames):
        truth = scene.truth_codes(index)
        for position, state in enumerate(frame.states):
            if state not in RECONSTRUCTED_STATES:
                continue
            value = float(frame.confidence[position])
            values.append(value)
            has_truth = truth[position] != NO_RETURN_SAMPLE
            (on_surface if has_truth else false_fill).append(value)
            bucket = _confidence_bin(value)
            bin_counts[bucket] += 1
            bin_on_surface[bucket] += int(has_truth)
            if value < CONFIDENCE_THRESHOLD:
                continue
            if has_truth:
                confident_on_surface += 1
                confident_errors.append(
                    abs(
                        _depth(frame.codes[position], scene.depth_scale)
                        - _depth(truth[position], scene.depth_scale)
                    )
                )
            else:
                confident_false_fill += 1
    if not values:
        return None
    surface_mean = sum(on_surface) / len(on_surface) if on_surface else 0.0
    empty_mean = sum(false_fill) / len(false_fill) if false_fill else 0.0
    return ConfidenceEvidence(
        name="confidence",
        reconstructed=len(values),
        on_surface=len(on_surface),
        false_fill=len(false_fill),
        minimum=min(values),
        maximum=max(values),
        mean=sum(values) / len(values),
        on_surface_mean=surface_mean,
        false_fill_mean=empty_mean,
        # Zero, not a negative number, when there was nothing invented to
        # compare against: an advantage over an empty population is not an
        # advantage, and reporting one would let a candidate earn credit for
        # a comparison it never made.
        surface_advantage=0.0 if not false_fill or not on_surface else surface_mean - empty_mean,
        threshold=CONFIDENCE_THRESHOLD,
        high_confidence=confident_on_surface + confident_false_fill,
        high_confidence_on_surface=confident_on_surface,
        high_confidence_false_fill=confident_false_fill,
        high_confidence_worst=max(confident_errors) if confident_errors else 0.0,
        bins=tuple(
            {
                "low": CONFIDENCE_BIN_EDGES[position],
                "high": CONFIDENCE_BIN_EDGES[position + 1],
                "count": bin_counts[position],
                "on_surface": bin_on_surface[position],
                "false_fill": bin_counts[position] - bin_on_surface[position],
            }
            for position in range(len(bin_counts))
        ),
    )


def _confidence_bin(value: float) -> int:
    """Which fixed bin one confidence falls in. The top edge closes the last bin."""
    for position in range(len(CONFIDENCE_BIN_EDGES) - 1):
        if value < CONFIDENCE_BIN_EDGES[position + 1]:
            return position
    return len(CONFIDENCE_BIN_EDGES) - 2


def _work(result: CandidateResult) -> WorkEvidence | None:
    """The candidate's own work record, published after :func:`validate` bound it."""
    work = result.work
    if work is None:
        return None
    return WorkEvidence(
        name="completion_work",
        backend=work.backend,
        policy=dict(work.policy),
        holes=work.holes,
        attempted=work.attempted,
        accepted=work.accepted,
        rejected=work.rejected,
        rejections=dict(work.rejections),
        support_samples=work.support_samples,
        work_units=work.work_units,
        work_budget=work.work_budget,
    )


def _temporal_work(result: CandidateResult) -> TemporalWorkEvidence | None:
    """The candidate's own temporal work record, published after :func:`validate`."""
    work = result.temporal
    if work is None:
        return None
    return TemporalWorkEvidence(
        name="temporal_work",
        backend=work.backend,
        policy=dict(work.policy),
        source_frames=work.source_frames,
        eligible_samples=work.eligible_samples,
        observations=work.observations,
        observation_rejections=dict(work.observation_rejections),
        candidates=work.candidates,
        holes=work.holes,
        attempted=work.attempted,
        accepted=work.accepted,
        rejected=work.rejected,
        rejections=dict(work.rejections),
        confirmed=work.confirmed,
        contradicted=work.contradicted,
        not_observed=work.not_observed,
        support_samples=work.support_samples,
        work_units=work.work_units,
        work_budget=work.work_budget,
    )


def _confirmation(scene: Scene, result: CandidateResult) -> ConfirmationEvidence | None:
    """Every confirmation the candidate published, scored against the truth frames.

    Made only for a candidate that published confirmations at all, and for no
    other: a block of zeros beside a candidate with nothing to confirm would
    read as a stage that checked and found nothing rather than as one that
    never ran.
    """
    if not any(frame.confirmations for frame in result.frames):
        return None
    estimates = confirmed = confirmed_on = confirmed_false = 0
    contradicted = contradicted_on = not_observed = not_observed_false = 0
    for index, frame in enumerate(result.frames):
        truth = scene.truth_codes(index)
        for position, confirmation in enumerate(frame.confirmations):
            if confirmation == NO_SPATIAL_ESTIMATE:
                continue
            estimates += 1
            has_truth = truth[position] != NO_RETURN_SAMPLE
            if confirmation == CONFIRMED:
                confirmed += 1
                confirmed_on += int(has_truth)
                confirmed_false += int(not has_truth)
            elif confirmation == CONTRADICTED:
                contradicted += 1
                contradicted_on += int(has_truth)
            else:
                not_observed += 1
                not_observed_false += int(not has_truth)
    return ConfirmationEvidence(
        name="confirmation",
        spatial_estimates=estimates,
        confirmed=confirmed,
        confirmed_on_surface=confirmed_on,
        confirmed_false=confirmed_false,
        contradicted=contradicted,
        contradicted_on_surface=contradicted_on,
        not_observed=not_observed,
        not_observed_false=not_observed_false,
    )


def measure(scene: Scene, result: CandidateResult) -> tuple[quality.QualityMeasurement, ...]:
    """Every measurement one case makes of one candidate, in a fixed order.

    :func:`validate` runs first and raises: a result whose provenance does not
    hold is not scored at all, because a metric computed from an unreadable
    account of the data would be a number about nothing.
    """
    validate(scene, result)
    coverage = _coverage(scene, result)
    measurements: list[quality.QualityMeasurement] = [coverage]
    error = _reconstructed_depth(scene, result, coverage)
    if error is not None:
        measurements.append(error)
    confidence = _confidence(scene, result, coverage)
    if confidence is not None:
        measurements.append(confidence)
    work = _work(result)
    if work is not None:
        measurements.append(work)
    temporal = _temporal_work(result)
    if temporal is not None:
        measurements.append(temporal)
    confirmation = _confirmation(scene, result)
    if confirmation is not None:
        measurements.append(confirmation)
    regions = scene.regions
    completed = _cloud_from(result.frames[0], scene)
    truth = _truth_cloud(scene, 0)
    if regions.plane:
        measurements.append(
            quality.plane_roughness(
                completed,
                name="plane",
                region=f"the whole scored window of {scene.case_id}, which is one flat surface",
                unit=DEPTH_UNIT,
            )
        )
    if regions.surface_tolerance is not None:
        measurements.append(
            quality.cloud_agreement(
                completed,
                truth,
                name="surface",
                coverage_tolerance=regions.surface_tolerance,
                unit=DEPTH_UNIT,
            )
        )
    if regions.dimensions:
        measurements.append(
            quality.dimension_errors(
                _extents(completed),
                _extents(truth),
                name="dimensions",
                truth_source=TRUTH_SOURCE,
                unit=DEPTH_UNIT,
            )
        )
    if regions.landmarks:
        measurements.append(
            quality.landmark_offsets(
                _landmarks(completed),
                _landmarks(truth),
                name="landmarks",
                truth_source=TRUTH_SOURCE,
                unit=DEPTH_UNIT,
            )
        )
    if regions.foreground_below is not None and not regions.slot_columns:
        edge, displacement = _edge(scene, result)
        measurements.extend((edge, displacement))
    if regions.slot_columns:
        measurements.append(_slot(scene, result))
    if regions.fusion is not None:
        fusion, agreement = _fusion(scene, result)
        measurements.append(fusion)
        if agreement is not None:
            measurements.append(agreement)
    if regions.weighted_fusion is not None and result.accepted is not None:
        weighted, weighted_agreement = _weighted_fusion(scene, result)
        measurements.append(weighted)
        if weighted_agreement is not None:
            measurements.append(weighted_agreement)
    if regions.tracking:
        tracking, poses, counterfactual, counterfactual_poses = _tracking(scene, result)
        measurements.append(tracking)
        if poses is not None:
            measurements.append(poses)
        measurements.append(counterfactual)
        if counterfactual_poses is not None:
            measurements.append(counterfactual_poses)
    return tuple(measurements)


# --- the candidates, which are controls and not algorithms -----------------------------


def _start(
    scene: Scene, index: int
) -> tuple[list[int], list[str], list[float], dict[int, tuple[str, ...]]]:
    """The unchanged input of one frame, as a result under construction."""
    codes = list(scene.input_codes(index))
    states = [MEASURED if code != NO_RETURN_SAMPLE else MISSING for code in codes]
    confidence = [1.0 if code != NO_RETURN_SAMPLE else 0.0 for code in codes]
    return codes, states, confidence, {}


def _frame(
    scene: Scene,
    index: int,
    codes: Sequence[int],
    states: Sequence[str],
    confidence: Sequence[float],
    support: Mapping[int, tuple[str, ...]],
    *,
    raw: bytes | None = None,
    confirmations: Sequence[str] = (),
) -> CompletedFrame:
    return CompletedFrame(
        index=index,
        raw=scene.frames[index].raw if raw is None else raw,
        codes=tuple(codes),
        states=tuple(states),
        confidence=tuple(confidence),
        support=dict(support),
        confirmations=tuple(confirmations),
    )


def _measured_positions(scene: Scene, index: int) -> list[tuple[int, int, int]]:
    """Every returned sample of one frame, as ``(window index, u, v)``."""
    codes = scene.input_codes(index)
    out: list[tuple[int, int, int]] = []
    for position, code in enumerate(codes):
        if code == NO_RETURN_SAMPLE:
            continue
        u, v = scene.window.pixel_of(position)
        out.append((position, u, v))
    return out


def baseline(scene: Scene) -> CandidateResult:
    """The unchanged pipeline: what returned is measured, the rest is missing.

    This is the state of the world this thread starts from. It invents
    nothing, and its missing coverage stays visible as missing rather than
    being scored as a success.
    """
    frames = []
    for index in range(len(scene.frames)):
        codes, states, confidence, support = _start(scene, index)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    # The raw route into weighted fusion, and the reference the weighted gates
    # are stated against. It carries no completion result because it made none:
    # every sample is either the measurement at full weight or a no-return at
    # none, which is exactly the field the unchanged integrate() produces.
    return CandidateResult(
        name="baseline", frames=tuple(frames), accepted=(None,) * len(scene.frames)
    )


def identity(scene: Scene) -> CandidateResult:
    """A completion that completes nothing, run on complete input.

    Every measured value must come back exactly, and every downstream metric
    with it. A benchmark in which this is not exactly the baseline is a
    benchmark measuring its own noise.
    """
    return CandidateResult(name="identity", frames=baseline(scene).frames)


def dense_nearest(scene: Scene) -> CandidateResult:
    """Fill every hole with its nearest returned sample. Dense, and wrong.

    The canonical bad candidate: aggregate coverage improves everywhere, and
    it closes slots, crosses depth steps and paints empty space with whatever
    was nearest.
    """
    frames = []
    for index in range(len(scene.frames)):
        codes, states, confidence, support = _start(scene, index)
        measured = _measured_positions(scene, index)
        for position, state in enumerate(states):
            if state != MISSING or not measured:
                continue
            u, v = scene.window.pixel_of(position)
            best = min(
                measured, key=lambda item: ((item[1] - u) ** 2 + (item[2] - v) ** 2, item[0])
            )
            codes[position] = codes[best[0]]
            states[position] = SPATIALLY_RECONSTRUCTED
            distance = math.hypot(best[1] - u, best[2] - v)
            confidence[position] = 1.0 / (1.0 + distance)
            support[position] = (support_name(index, best[1], best[2]),)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    return CandidateResult(name="dense_nearest", frames=tuple(frames))


def _axis_neighbours(
    scene: Scene, codes: Sequence[int], states: Sequence[str], position: int
) -> list[tuple[int, int, int]]:
    """The nearest originally-measured sample along each of the four axes."""
    window = scene.window
    row, column = divmod(position, window.width)
    found: list[tuple[int, int, int]] = []
    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        step = 1
        while True:
            c, r = column + du * step, row + dv * step
            if not (0 <= c < window.width and 0 <= r < window.height):
                break
            index = r * window.width + c
            if states[index] == MEASURED:
                u, v = window.pixel_of(index)
                found.append((codes[index], u, v))
                break
            step += 1
    return found


def broad_interpolation(scene: Scene) -> CandidateResult:
    """Average the nearest measured sample in each direction. Smooth, and wrong.

    A gentler bad candidate than :func:`dense_nearest`, and a more persuasive
    one: the result looks like a surface. Averaging across a depth step puts
    samples in the empty band between the two surfaces, which is exactly the
    flying-pixel geometry the edge gates count.
    """
    frames = []
    for index in range(len(scene.frames)):
        codes, states, confidence, support = _start(scene, index)
        original = list(states)
        for position, state in enumerate(original):
            if state != MISSING:
                continue
            neighbours = _axis_neighbours(scene, codes, original, position)
            if len(neighbours) < 2:
                continue
            codes[position] = round(sum(code for code, _u, _v in neighbours) / len(neighbours))
            states[position] = SPATIALLY_RECONSTRUCTED
            confidence[position] = len(neighbours) / 4.0
            support[position] = tuple(support_name(index, u, v) for _c, u, v in neighbours)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    return CandidateResult(name="broad_interpolation", frames=tuple(frames))


def naive_temporal_carry(scene: Scene) -> CandidateResult:
    """Carry the other frame's sample at the SAME pixel. Ignores the motion.

    The control the temporal task exists to rule out. It is what "use the
    previous frame" means if the pose between them is not applied, and under
    real motion it fills holes with the depth of whatever else was at that
    pixel.
    """
    frames = []
    for index in range(len(scene.frames)):
        codes, states, confidence, support = _start(scene, index)
        others = [other for other in range(len(scene.frames)) if other != index]
        for other in others:
            source = scene.input_codes(other)
            for position, state in enumerate(states):
                if state != MISSING or source[position] == NO_RETURN_SAMPLE:
                    continue
                u, v = scene.window.pixel_of(position)
                codes[position] = source[position]
                states[position] = TEMPORALLY_RECONSTRUCTED
                confidence[position] = 0.5
                support[position] = (support_name(other, u, v),)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    return CandidateResult(name="naive_temporal_carry", frames=tuple(frames))


def posed_temporal_carry(scene: Scene) -> CandidateResult:
    """Carry the other frame's samples THROUGH the exact poses.

    The positive control for temporal opportunity, and nothing more than that.
    It uses the case's exact known poses, which a real capture does not have
    for free, so what it measures is how much complementary evidence the posed
    frames contain - not that any implemented algorithm can reach it.
    """
    window = scene.window
    values = scene.intrinsics.values
    fx, fy = values["fx"], values["fy"]
    cx, cy = values["cx"], values["cy"]
    frames = []
    for index in range(len(scene.frames)):
        codes, states, confidence, support = _start(scene, index)
        to_camera = scene.frames[index].pose.inverse()
        for other in range(len(scene.frames)):
            if other == index:
                continue
            source_pose = scene.frames[other].pose
            for point in depth_points_of(
                scene.input_codes(other),
                window,
                intrinsics=scene.intrinsics,
                depth_scale=scene.depth_scale,
            ):
                moved = to_camera.apply(source_pose.apply((point.x, point.y, point.z)))
                if moved[2] <= 0.0:
                    continue
                u = round(cx + moved[0] * fx / moved[2])
                v = round(cy + moved[1] * fy / moved[2])
                position = window.index_of(u, v)
                if position is None or states[position] != MISSING:
                    continue
                code = round(moved[2] / scene.depth_scale)
                if not 0 < code <= MAX_CODE:
                    continue
                codes[position] = code
                states[position] = TEMPORALLY_RECONSTRUCTED
                confidence[position] = 0.75
                support[position] = (support_name(other, point.u, point.v),)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    return CandidateResult(name="posed_temporal_carry", frames=tuple(frames))


def foreground_bias(scene: Scene) -> CandidateResult:
    """Fill each hole from the NEAREST SURFACE among its nearest returned samples.

    A heuristic that sounds careful - prefer the closer surface, it is the one
    the object is made of - and is the specific way a completion moves a depth
    step outward and closes a real gap. Every value it writes was measured
    somewhere nearby, so nothing about its provenance is false; what is wrong
    is where it put it.
    """
    frames = []
    for index in range(len(scene.frames)):
        codes, states, confidence, support = _start(scene, index)
        measured = _measured_positions(scene, index)
        for position, state in enumerate(states):
            if state != MISSING or not measured:
                continue
            u, v = scene.window.pixel_of(position)
            nearby = sorted(
                measured, key=lambda item: ((item[1] - u) ** 2 + (item[2] - v) ** 2, item[0])
            )[:8]
            chosen = min(nearby, key=lambda item: (codes[item[0]], item[0]))
            codes[position] = codes[chosen[0]]
            states[position] = SPATIALLY_RECONSTRUCTED
            confidence[position] = 0.6
            support[position] = (support_name(index, chosen[1], chosen[2]),)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    return CandidateResult(name="foreground_bias", frames=tuple(frames))


def easy_samples_only(scene: Scene) -> CandidateResult:
    """Fill only the easy holes and leave the hard ones missing.

    Legal, and exactly the shape of a result that reports a superb error by
    never attempting anything difficult. Every metric here is bound to the
    whole window, so what this improves in error it loses in supported
    coverage.
    """
    window = scene.window
    frames = []
    for index in range(len(scene.frames)):
        codes, states, confidence, support = _start(scene, index)
        original = list(states)
        for position, state in enumerate(original):
            if state != MISSING:
                continue
            neighbours = _axis_neighbours(scene, codes, original, position)
            here_u, here_v = window.pixel_of(position)
            close = [
                (code, u, v) for code, u, v in neighbours if abs(u - here_u) + abs(v - here_v) <= 1
            ]
            if len(close) < 3:
                continue
            codes[position] = round(sum(code for code, _u, _v in close) / len(close))
            states[position] = SPATIALLY_RECONSTRUCTED
            confidence[position] = 0.9
            support[position] = tuple(support_name(index, u, v) for _c, u, v in close)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    return CandidateResult(name="easy_samples_only", frames=tuple(frames))


def discarded_measurements(scene: Scene) -> CandidateResult:
    """Drop the extreme returned samples, then fill densely.

    How a completion makes an object smaller and smoother while looking
    better: throw away the measurements at the edges of the extent and
    interpolate what remains. The measured-coverage floor and the named
    dimensions are what refuse it.
    """
    result = dense_nearest(scene)
    frames = []
    for index, frame in enumerate(result.frames):
        codes = list(frame.codes)
        states = list(frame.states)
        confidence = list(frame.confidence)
        support = dict(frame.support)
        returned = [
            (codes[position], position)
            for position, state in enumerate(states)
            if state == MEASURED
        ]
        nearest = sorted(returned)[: max(1, len(returned) // 10)]
        for _code, position in nearest:
            codes[position] = NO_RETURN_SAMPLE
            states[position] = MISSING
            confidence[position] = 0.0
            support.pop(position, None)
        frames.append(_frame(scene, index, codes, states, confidence, support))
    return CandidateResult(name="discarded_measurements", frames=tuple(frames))


# --- the product primitive, run through a thin adapter --------------------------------


#: The policy :func:`spatial_completion` runs the product primitive at. The
#: module's own default, named here rather than re-tuned: a benchmark that
#: picked its own bounds would be scoring a configuration nothing ships.
COMPLETION_POLICY: Final = depth_completion.CompletionPolicy()

#: How this adapter names the product's three states, in the vocabulary this
#: module scores. The product has no temporal state, so there is nothing to
#: map onto :data:`TEMPORALLY_RECONSTRUCTED` and a spatial reconstruction can
#: never be reported as one.
_PRODUCT_STATES: Final[Mapping[str, str]] = {
    "measured": MEASURED,
    "spatially_reconstructed": SPATIALLY_RECONSTRUCTED,
    "missing": MISSING,
}


def _complete_frame(
    scene: Scene, index: int, policy: depth_completion.CompletionPolicy
) -> depth_completion.CompletedPatch:
    """One frame through :func:`scanner3d.depth_completion.complete_patch`.

    The patch is built from the case's own window over the immutable raw frame,
    so the product sees exactly the codes the benchmark scores and nothing
    about the case, the truth or the other frames.
    """
    window = scene.window
    patch = depth_completion.DepthPatch.of(window.width, window.height, scene.input_codes(index))
    return depth_completion.complete_patch(patch, policy=policy, backend="python")


def _adapt(
    scene: Scene,
    completed: Sequence[depth_completion.CompletedPatch],
    *,
    name: str,
) -> CandidateResult:
    """The product's own result, restated in this module's provenance vocabulary.

    A translation and nothing else: no value is recomputed, no state is
    widened, and the support indices the product named are spelled as the
    frame-and-pixel references this module can check against the raw input.
    """
    window = scene.window
    frames: list[CompletedFrame] = []
    holes = attempted = accepted = rejected = support_samples = 0
    work_units = work_budget = 0
    rejections = dict.fromkeys(depth_completion.REJECTION_REASONS, 0)
    for index, patch in enumerate(completed):
        support = {
            position: tuple(support_name(index, *window.pixel_of(source)) for source in sources)
            for position, sources in sorted(patch.support.items())
        }
        frames.append(
            _frame(
                scene,
                index,
                patch.codes,
                [_PRODUCT_STATES[state] for state in patch.states()],
                patch.confidence,
                support,
            )
        )
        work = patch.work
        holes += work.holes
        attempted += work.attempted
        accepted += work.accepted
        rejected += work.rejected
        support_samples += work.support_samples
        work_units += work.work_units
        work_budget += work.work_budget
        for reason, count in work.rejections.items():
            rejections[reason] += count
    return CandidateResult(
        name=name,
        frames=tuple(frames),
        accepted=tuple(completed),
        work=WorkRecord(
            backend=depth_completion.PYTHON_BACKEND,
            policy=COMPLETION_POLICY.as_dict(),
            holes=holes,
            attempted=attempted,
            accepted=accepted,
            rejected=rejected,
            rejections=rejections,
            support_samples=support_samples,
            work_units=work_units,
            work_budget=work_budget,
        ),
    )


def spatial_completion(scene: Scene) -> CandidateResult:
    """:mod:`scanner3d.depth_completion`, run on each frame of the case alone.

    The one candidate here that is product code rather than a control. Each
    frame is completed from itself, with no access to the truth, to the other
    frames, or to the poses between them - so what this scores is what a single
    raw frame supports, and every case that needs another view is expected to
    remain short rather than to be filled from somewhere it could not know.
    """
    completed = [
        _complete_frame(scene, index, COMPLETION_POLICY) for index in range(len(scene.frames))
    ]
    return _adapt(scene, completed, name="spatial_completion")


# --- the temporal product primitive, run through a thin test-side adapter --------------


#: The policy the temporal candidates run the product primitive at. The
#: module's own default, named here rather than re-tuned.
TEMPORAL_POLICY: Final = depth_temporal.TemporalPolicy()

#: How the temporal adapter names the product's four states in this module's
#: vocabulary. Every one of them exists on both sides, so the translation is a
#: renaming and never a widening.
_TEMPORAL_STATES: Final[Mapping[str, str]] = {
    "measured": MEASURED,
    "spatially_reconstructed": SPATIALLY_RECONSTRUCTED,
    "temporally_reconstructed": TEMPORALLY_RECONSTRUCTED,
    "missing": MISSING,
}

#: What the benchmark tells the product about the poses it is handed. They are
#: the scene's own exact poses, which a real capture does not have for free, and
#: the status says so rather than borrowing a tracker's word for it.
POSE_STATUS: Final = "depth-completeness scene pose, exact by construction"


def _camera(scene: Scene) -> depth_temporal.CameraModel:
    """The scene's intrinsics and scale, as the product's own camera type."""
    values = scene.intrinsics.values
    return depth_temporal.CameraModel(
        fx=values["fx"],
        fy=values["fy"],
        cx=values["cx"],
        cy=values["cy"],
        depth_scale=scene.depth_scale,
    )


def _temporal_frames(scene: Scene) -> tuple[depth_temporal.TemporalFrame, ...]:
    """Every frame of the case, as the product's immutable posed raw evidence."""
    window = scene.window
    return tuple(
        depth_temporal.TemporalFrame(
            index=index,
            patch=depth_completion.DepthPatch.of(
                window.width, window.height, scene.input_codes(index)
            ),
            u0=window.u0,
            v0=window.v0,
            pose=depth_temporal.PoseRecord(
                transform=scene.frames[index].pose,
                accepted=True,
                status=POSE_STATUS,
                quality=1.0,
            ),
        )
        for index in range(len(scene.frames))
    )


def _adapt_temporal(
    scene: Scene,
    completed: Sequence[depth_temporal.TemporalPatch],
    *,
    name: str,
) -> CandidateResult:
    """The product's own temporal result, restated in this module's vocabulary.

    A translation and nothing else: no value is recomputed, no state is
    widened, and the support the product named is spelled as the frame-and-pixel
    references this module can check against the raw inputs.

    One thing is deliberately narrowed rather than translated. The product names
    both the other-frame raw samples a CONFIRMED sample rests on and the
    same-frame spatial support it confirms; this module's support vocabulary is
    one flat list per sample and refuses a temporal reconstruction that names
    its own frame - a control that exists for a good reason and is not being
    relaxed for this adapter. So the support list carries the independent
    other-frame evidence, which is what makes the sample temporal, and the
    spatial side of a confirmation is carried by the confirmation states and
    counted in :class:`ConfirmationEvidence`.

    Only the temporal work record is published, even by the candidate that runs
    both stages. :class:`WorkRecord` binds its accepted count to the
    reconstructions the result carries, and after confirmation some of those
    carry temporal provenance - so publishing the spatial record beside this
    output would need that binding loosened, and it is a control this task does
    not get to relax for its own convenience. What the spatial stage did is
    scored on its own, by the ``spatial_completion`` candidate, in the same case.
    """
    window = scene.window
    frames: list[CompletedFrame] = []
    source_frames = eligible = observations = candidates = 0
    holes = attempted = accepted = rejected = support_samples = 0
    confirmed = contradicted = not_observed = 0
    work_units = work_budget = 0
    rejections = dict.fromkeys(depth_temporal.REJECTION_REASONS, 0)
    observation_rejections = dict.fromkeys(depth_temporal.OBSERVATION_REJECTIONS, 0)
    for index, patch in enumerate(completed):
        support = {
            position: tuple(
                support_name(frame, *window.pixel_of(sample)) for frame, sample in references
            )
            for position, references in sorted(patch.temporal_support.items())
        }
        # A spatial estimate the temporal stage carried through unchanged still
        # rests on its own frame's measured samples, and this module refuses a
        # reconstruction that names none. So the spatial support travels with
        # it, spelled in the same frame it came from - which is exactly what
        # this module requires of a spatially reconstructed sample.
        for position, sources in sorted(patch.spatial_support.items()):
            if position in support:
                continue
            support[position] = tuple(
                support_name(index, *window.pixel_of(sample)) for sample in sources
            )
        frames.append(
            _frame(
                scene,
                index,
                patch.codes,
                [_TEMPORAL_STATES[state] for state in patch.states()],
                patch.confidence,
                support,
                # Published only by a run that was given a spatial estimate to
                # confirm. A block of "no_spatial_estimate" beside a run that
                # was never offered one would read as a stage that checked and
                # found nothing rather than as one that had nothing to check.
                confirmations=() if patch.spatial is None else patch.confirmations(),
            )
        )
        work = patch.work
        source_frames += work.source_frames
        eligible += work.eligible_samples
        observations += work.observations
        candidates += work.candidates
        holes += work.holes
        attempted += work.attempted
        accepted += work.accepted
        rejected += work.rejected
        confirmed += work.confirmed
        contradicted += work.contradicted
        not_observed += work.not_observed
        support_samples += work.support_samples
        work_units += work.work_units
        work_budget += work.work_budget
        for reason, count in work.rejections.items():
            rejections[reason] += count
        for reason, count in work.observation_rejections.items():
            observation_rejections[reason] += count
    return CandidateResult(
        name=name,
        frames=tuple(frames),
        accepted=tuple(completed),
        temporal=TemporalWorkRecord(
            backend=depth_temporal.PYTHON_BACKEND,
            policy=TEMPORAL_POLICY.as_dict(),
            source_frames=source_frames,
            eligible_samples=eligible,
            observations=observations,
            observation_rejections=observation_rejections,
            candidates=candidates,
            holes=holes,
            attempted=attempted,
            accepted=accepted,
            rejected=rejected,
            rejections=rejections,
            confirmed=confirmed,
            contradicted=contradicted,
            not_observed=not_observed,
            support_samples=support_samples,
            work_units=work_units,
            work_budget=work_budget,
        ),
    )


def _run_temporal(
    scene: Scene, *, spatial: Sequence[depth_completion.CompletedPatch] | None
) -> tuple[depth_temporal.TemporalPatch, ...]:
    """Every frame of the case completed from the OTHER frames of the case.

    Each frame in turn is the target and every other frame is a source, so the
    result is one completed frame per input frame - what this benchmark scores -
    and no frame is ever its own evidence.
    """
    frames = _temporal_frames(scene)
    camera = _camera(scene)
    return tuple(
        depth_temporal.complete_temporally(
            target,
            [frame for frame in frames if frame.index != target.index],
            camera=camera,
            policy=TEMPORAL_POLICY,
            spatial=None if spatial is None else spatial[target.index],
            backend="python",
        )
        for target in frames
    )


def temporal_completion(scene: Scene) -> CandidateResult:
    """:mod:`scanner3d.depth_temporal`, on the raw frames and the exact poses.

    Product code, not a control. It sees the other frames' ORIGINAL raw samples
    and the case's accepted poses, and nothing else: no truth, no spatial
    estimate, and no sample of the frame it is completing. What it recovers is
    what independent posed measurements actually agree about.
    """
    return _adapt_temporal(scene, _run_temporal(scene, spatial=None), name="temporal_completion")


def spatial_then_temporal_completion(scene: Scene) -> CandidateResult:
    """The accepted spatial result, offered to the temporal stage for confirmation.

    The two product primitives in the order the pipeline would run them. Every
    spatial estimate is either confirmed by independent posed raw measurements -
    and then carries their provenance - contradicted by them, or never observed
    by them, in which case it stays exactly the spatial estimate it was. Nothing
    is averaged and nothing is relabelled, which is the whole point of running
    them this way round rather than merging their outputs.
    """
    spatial = [
        _complete_frame(scene, index, COMPLETION_POLICY) for index in range(len(scene.frames))
    ]
    return _adapt_temporal(
        scene, _run_temporal(scene, spatial=spatial), name="spatial_then_temporal_completion"
    )


# --- the corruption controls, every one of which must be refused ----------------------


@dataclass(slots=True)
class _Parts:
    """One frame of a result, opened up so a control can corrupt exactly one thing."""

    raw: bytes
    codes: list[int]
    states: list[str]
    confidence: list[float]
    support: dict[int, tuple[str, ...]]


def _edited(
    result: CandidateResult, name: str, edit: Callable[[int, _Parts], None]
) -> CandidateResult:
    """One candidate, with one deliberate corruption applied per frame."""
    frames = []
    for index, frame in enumerate(result.frames):
        parts = _Parts(
            raw=frame.raw,
            codes=list(frame.codes),
            states=list(frame.states),
            confidence=list(frame.confidence),
            support=dict(frame.support),
        )
        edit(index, parts)
        frames.append(
            CompletedFrame(
                index=frame.index,
                raw=parts.raw,
                codes=tuple(parts.codes),
                states=tuple(parts.states),
                confidence=tuple(parts.confidence),
                support=dict(parts.support),
            )
        )
    return CandidateResult(name=name, frames=tuple(frames))


def _first_reconstructed(parts: _Parts) -> int:
    for position, state in enumerate(parts.states):
        if state in RECONSTRUCTED_STATES:
            return position
    raise ScoringUnusable("this control needs a case where the fill reconstructs something")


def forged_measured_states(scene: Scene) -> CandidateResult:
    """Report reconstructed samples as camera measurements."""

    def edit(_index: int, parts: _Parts) -> None:
        for position, state in enumerate(parts.states):
            if state in RECONSTRUCTED_STATES:
                parts.states[position] = MEASURED

    return _edited(dense_nearest(scene), "forged_measured_states", edit)


def mutated_raw_frame(scene: Scene) -> CandidateResult:
    """Return raw bytes that are not the ones the case supplied."""

    def edit(_index: int, parts: _Parts) -> None:
        raw = bytearray(parts.raw)
        struct.pack_into("<H", raw, 0, 4321)
        parts.raw = bytes(raw)

    return _edited(baseline(scene), "mutated_raw_frame", edit)


def unsupported_fill(scene: Scene) -> CandidateResult:
    """Reconstruct without naming any support at all."""

    def edit(_index: int, parts: _Parts) -> None:
        parts.support = {}

    return _edited(dense_nearest(scene), "unsupported_fill", edit)


def forged_confidence(scene: Scene) -> CandidateResult:
    """Claim a confidence outside the fraction it is declared to be."""

    def edit(_index: int, parts: _Parts) -> None:
        parts.confidence[_first_reconstructed(parts)] = 1.5

    return _edited(dense_nearest(scene), "forged_confidence", edit)


def missing_carrying_depth(scene: Scene) -> CandidateResult:
    """Declare a sample missing and hand out a depth for it anyway."""

    def edit(_index: int, parts: _Parts) -> None:
        for position, state in enumerate(parts.states):
            if state == MISSING:
                parts.codes[position] = 12345
                return

    return _edited(baseline(scene), "missing_carrying_depth", edit)


def unknown_provenance_state(scene: Scene) -> CandidateResult:
    """Invent a fifth provenance state."""

    def edit(_index: int, parts: _Parts) -> None:
        parts.states[_first_reconstructed(parts)] = "guessed"

    return _edited(dense_nearest(scene), "unknown_provenance_state", edit)


def truncated_window(scene: Scene) -> CandidateResult:
    """Return fewer samples than the window, dropping the hardest.

    The denominator attack in its bluntest form. Every metric here counts
    against the window, so a result that is not the window cannot be scored.
    """

    def edit(_index: int, parts: _Parts) -> None:
        parts.codes = parts.codes[:-1]
        parts.states = parts.states[:-1]
        parts.confidence = parts.confidence[:-1]

    return _edited(dense_nearest(scene), "truncated_window", edit)


def phantom_support(scene: Scene) -> CandidateResult:
    """Name support at a pixel that returned no measurement."""

    def edit(index: int, parts: _Parts) -> None:
        u, v = _PHANTOM_PIXEL
        parts.support[_first_reconstructed(parts)] = (support_name(index, u, v),)

    return _edited(dense_nearest(scene), "phantom_support", edit)


def temporal_support_from_own_frame(scene: Scene) -> CandidateResult:
    """Call a same-frame fill a temporal reconstruction."""

    def edit(_index: int, parts: _Parts) -> None:
        parts.states[_first_reconstructed(parts)] = TEMPORALLY_RECONSTRUCTED

    return _edited(dense_nearest(scene), "temporal_support_from_own_frame", edit)


#: The order :func:`spatial_completion_forged_work` takes a refusal from, so
#: the control corrupts one named reason rather than whichever the dictionary
#: happened to yield first.
REJECTION_ORDER: Final[tuple[str, ...]] = tuple(depth_completion.REJECTION_REASONS)


def spatial_completion_forged_work(scene: Scene) -> CandidateResult:
    """Report a completion that refused less than it did.

    The work record is the only part of a result that is about the run rather
    than about the output, so it is the one a candidate could restate freely.
    Moving one refusal into the accepted column - and nothing else - leaves
    every frame untouched and every gate reachable, and is refused because the
    accepted count is bound to the reconstructions the result actually carries.
    """
    result = spatial_completion(scene)
    work = result.work
    assert work is not None
    rejections = dict(work.rejections)
    for reason in REJECTION_ORDER:
        if rejections.get(reason):
            rejections[reason] -= 1
            break
    else:  # pragma: no cover - every case in this corpus refuses something
        raise ScoringUnusable("this control needs a case where the completion refuses a hole")
    return CandidateResult(
        name="spatial_completion_forged_work",
        frames=result.frames,
        work=WorkRecord(
            backend=work.backend,
            policy=work.policy,
            holes=work.holes,
            attempted=work.attempted,
            accepted=work.accepted + 1,
            rejected=work.rejected - 1,
            rejections=rejections,
            support_samples=work.support_samples,
            work_units=work.work_units,
            work_budget=work.work_budget,
        ),
    )


def spatial_completion_recursive_support(scene: Scene) -> CandidateResult:
    """Cite one reconstruction as the evidence for another.

    The failure a spatial method is structurally prone to: once a hole has a
    value, the hole beside it has a neighbour, and a fill can walk across a
    frame one sample at a time with every step "supported". Naming a
    reconstructed sample as support is refused because support is checked
    against the RAW input, where that sample returned nothing at all.
    """
    result = spatial_completion(scene)
    frames = list(result.frames)
    for index, frame in enumerate(frames):
        reconstructed = [
            position for position, state in enumerate(frame.states) if state in RECONSTRUCTED_STATES
        ]
        if len(reconstructed) < 2:
            continue
        borrowed, lender = reconstructed[0], reconstructed[1]
        support = dict(frame.support)
        support[borrowed] = (support_name(index, *scene.window.pixel_of(lender)),)
        frames[index] = CompletedFrame(
            index=frame.index,
            raw=frame.raw,
            codes=frame.codes,
            states=frame.states,
            confidence=frame.confidence,
            support=support,
        )
        break
    else:  # pragma: no cover - every case running this control recovers two samples
        raise ScoringUnusable("this control needs a case where the completion recovers two holes")
    return CandidateResult(
        name="spatial_completion_recursive_support",
        frames=tuple(frames),
        work=result.work,
    )


def _replace_frame(frame: CompletedFrame, **changes: object) -> CompletedFrame:
    """One frame with exactly the named fields changed. Controls edit one thing."""
    fields: dict[str, object] = {
        "index": frame.index,
        "raw": frame.raw,
        "codes": frame.codes,
        "states": frame.states,
        "confidence": frame.confidence,
        "support": dict(frame.support),
        "confirmations": frame.confirmations,
    }
    fields.update(changes)
    return CompletedFrame(**fields)  # type: ignore[arg-type]


def _first_temporal(frames: Sequence[CompletedFrame]) -> tuple[int, int]:
    """The first frame and sample a temporal run reconstructed, in index order.

    Across the frames rather than inside one: a case can recover nothing in its
    first frame and a great deal in its second, and a control that gave up on
    frame 0 would be reporting the frame ordering rather than the run.
    """
    for index, frame in enumerate(frames):
        for position, state in enumerate(frame.states):
            if state == TEMPORALLY_RECONSTRUCTED:
                return index, position
    raise ScoringUnusable("this control needs a case where the temporal stage recovers a sample")


def temporal_completion_forged_work(scene: Scene) -> CandidateResult:
    """Report a temporal run that refused less than it did.

    The temporal work record is about the RUN rather than the output, so it is
    the part a candidate could restate freely. Moving one refusal into the
    accepted column leaves every frame untouched and every gate reachable, and
    is refused because the accepted count is bound to the temporal
    reconstructions the result actually carries.
    """
    result = temporal_completion(scene)
    work = result.temporal
    assert work is not None
    rejections = dict(work.rejections)
    for reason in depth_temporal.REJECTION_REASONS:
        if rejections.get(reason):
            rejections[reason] -= 1
            break
    else:  # pragma: no cover - every case in this corpus refuses something
        raise ScoringUnusable("this control needs a case where the temporal stage refuses a hole")
    return CandidateResult(
        name="temporal_completion_forged_work",
        frames=result.frames,
        work=result.work,
        temporal=TemporalWorkRecord(
            backend=work.backend,
            policy=work.policy,
            source_frames=work.source_frames,
            eligible_samples=work.eligible_samples,
            observations=work.observations,
            observation_rejections=work.observation_rejections,
            candidates=work.candidates,
            holes=work.holes,
            attempted=work.attempted,
            accepted=work.accepted + 1,
            rejected=work.rejected - 1,
            rejections=rejections,
            confirmed=work.confirmed,
            contradicted=work.contradicted,
            not_observed=work.not_observed,
            support_samples=work.support_samples,
            work_units=work.work_units,
            work_budget=work.work_budget,
        ),
    )


def temporal_completion_self_support(scene: Scene) -> CandidateResult:
    """Cite the target frame's own sample as temporal evidence about itself.

    The failure a temporal method is structurally prone to: a frame is already
    in memory, and a same-frame neighbour is the easiest thing to reach for. It
    is refused because support for a temporal reconstruction has to be in
    another frame, whatever the state says.
    """
    result = temporal_completion(scene)
    frames = list(result.frames)
    index, position = _first_temporal(frames)
    measured = _measured_positions(scene, index)
    if not measured:  # pragma: no cover - every case here returns something
        raise ScoringUnusable("this control needs a frame that returned a measurement")
    _at, u, v = measured[0]
    support = dict(frames[index].support)
    support[position] = (support_name(index, u, v),)
    frames[index] = _replace_frame(frames[index], support=support)
    return CandidateResult(
        name="temporal_completion_self_support",
        frames=tuple(frames),
        work=result.work,
        temporal=result.temporal,
    )


def temporal_completion_unmeasured_support(scene: Scene) -> CandidateResult:
    """Cite another frame's NO-RETURN sample as temporal evidence.

    Another frame is not enough: the sample named in it has to be one that
    frame's sensor actually returned. A reconstruction of that other frame,
    spatial or temporal, returned nothing in the raw bytes and is refused here
    exactly as a phantom pixel is.
    """
    result = temporal_completion(scene)
    frames = list(result.frames)
    index, position = _first_temporal(frames)
    other = next((i for i in range(len(scene.frames)) if i != index), None)
    if other is None:  # pragma: no cover - this control needs two frames
        raise ScoringUnusable("this control needs a case with more than one frame")
    codes = scene.input_codes(other)
    hole = next((at for at, code in enumerate(codes) if code == NO_RETURN_SAMPLE), None)
    if hole is None:  # pragma: no cover - every temporal case has dropout
        raise ScoringUnusable("this control needs another frame that lost a sample")
    support = dict(frames[index].support)
    support[position] = (support_name(other, *scene.window.pixel_of(hole)),)
    frames[index] = _replace_frame(frames[index], support=support)
    return CandidateResult(
        name="temporal_completion_unmeasured_support",
        frames=tuple(frames),
        work=result.work,
        temporal=result.temporal,
    )


def temporal_completion_forged_confirmation(scene: Scene) -> CandidateResult:
    """Call a never-observed spatial estimate confirmed, and leave it spatial.

    The specific way an unsupported spatial hypothesis is made to look safe by
    passing through a temporal stage: nothing about the sample changes except
    the word for what independent evidence said about it. It is refused because
    a confirmed estimate rests on that evidence and therefore carries its
    provenance, and this one does not.

    NOT OBSERVED specifically, and not merely "unconfirmed". A contradicted
    estimate is one independent evidence did settle - it published its own
    value there - so its sample already carries temporal provenance and this
    edit would be invisible in the states. That relabelling is refused by
    :mod:`scanner3d.depth_temporal` itself, against the accepted spatial codes
    a result carries and this module deliberately does not, and is proved in
    ``tests/test_depth_temporal.py``.
    """
    result = spatial_then_temporal_completion(scene)
    frames = list(result.frames)
    for index, frame in enumerate(frames):
        position = next(
            (
                at
                for at, confirmation in enumerate(frame.confirmations)
                if confirmation == NOT_OBSERVED
            ),
            None,
        )
        if position is None:
            continue
        confirmations = list(frame.confirmations)
        confirmations[position] = CONFIRMED
        frames[index] = _replace_frame(frame, confirmations=tuple(confirmations))
        break
    else:  # pragma: no cover - every case running this control leaves one unconfirmed
        raise ScoringUnusable("this control needs a case with an unconfirmed spatial estimate")
    return CandidateResult(
        name="temporal_completion_forged_confirmation",
        frames=tuple(frames),
        work=result.work,
        temporal=result.temporal,
    )


#: Every candidate this benchmark knows how to run, by the name a fixture uses.
CANDIDATES: Final[Mapping[str, Candidate]] = {
    "baseline": baseline,
    "identity": identity,
    "dense_nearest": dense_nearest,
    "broad_interpolation": broad_interpolation,
    "foreground_bias": foreground_bias,
    "naive_temporal_carry": naive_temporal_carry,
    "posed_temporal_carry": posed_temporal_carry,
    "easy_samples_only": easy_samples_only,
    "discarded_measurements": discarded_measurements,
    "forged_measured_states": forged_measured_states,
    "mutated_raw_frame": mutated_raw_frame,
    "unsupported_fill": unsupported_fill,
    "forged_confidence": forged_confidence,
    "missing_carrying_depth": missing_carrying_depth,
    "unknown_provenance_state": unknown_provenance_state,
    "truncated_window": truncated_window,
    "phantom_support": phantom_support,
    "temporal_support_from_own_frame": temporal_support_from_own_frame,
    "spatial_completion": spatial_completion,
    "spatial_completion_forged_work": spatial_completion_forged_work,
    "spatial_completion_recursive_support": spatial_completion_recursive_support,
    "temporal_completion": temporal_completion,
    "spatial_then_temporal_completion": spatial_then_temporal_completion,
    "temporal_completion_forged_work": temporal_completion_forged_work,
    "temporal_completion_self_support": temporal_completion_self_support,
    "temporal_completion_unmeasured_support": temporal_completion_unmeasured_support,
    "temporal_completion_forged_confirmation": temporal_completion_forged_confirmation,
}
