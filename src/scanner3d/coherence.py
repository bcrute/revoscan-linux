"""Whether a registered sweep agrees with itself, judged by the frames alone.

Why this exists
---------------
Every score this pipeline reported on 2026-09-02 said the sweep was fine: the
trajectory stepped smoothly, loop closure converged, the held-out closure
error was 1.3 degrees and the target gate read 1.1. Frames one turntable
revolution apart - whose depth images are identical to 0.15 - had poses that
disagreed by up to 240. Closure is scored on the revisit pairs the optimiser
was handed, so it cannot see a frame no constraint reaches, and a CAD-side
nearest-point gate is satisfied by a dense smear of misplaced copies: a
MIRRORED reference fitted that cloud exactly as well as the real one. The one
thing none of them could fake was the images themselves.

What it measures
----------------
1. :func:`image_period` - the lag at which a frame's depth image best matches
   a later one, by median absolute depth difference over co-valid pixels. On a
   turntable that lag is one revolution, and the match is better than the
   frame's own neighbour. A sweep that never comes back around has no such lag
   and reports None: this check does not APPLY to it, which is a different
   fact from passing it.
2. :func:`period_agreement` - carry the revisit frame's points through its
   pose and back through the sampled frame's pose, project them, and compare
   with the depth the sampled frame actually recorded there. Two identical
   images given consistent poses land on each other; the share of points
   within :data:`DEFAULT_TOLERANCE` is the agreement, and the WEAKEST sampled
   pair is the verdict, because one bent stretch of trajectory is enough to
   smear the model.
3. :func:`camera_orbit` - a circle fitted to the pose translations. A
   consistent turntable registration puts the camera centres on a circle about
   the spindle; its normal and centre are the axis, which a placement search
   can be seeded from without a support plane or a CAD.

All of it is pure Python over the same deprojection arithmetic
:mod:`scanner3d.depth_metric` uses, subsampled so a 190-frame sweep is judged
in seconds. Nothing here reads a device or writes a file. Distances are in the
deprojection's own scale, the one :func:`scanner3d.preparation.preparation_unit`
stakes.
"""

from __future__ import annotations

import math
import statistics
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from scanner3d.depth_metric import (
    NO_RETURN_SAMPLE,
    SAMPLES_FORMAT,
    ensure_frame_length,
    ensure_intrinsics_usable,
    ensure_scale_usable,
)
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid, cross, determinant, dot, norm, normalize, subtract
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.preparation import DepthBand, ensure_depth_band
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: A revisit's point must land within this of the depth the sampled frame
#: recorded to count as agreeing. Measured, a consistent registration reads
#: 90-99% of points within it and a bent one 17-47%.
DEFAULT_TOLERANCE: Final = 1.5

#: The weakest sampled revisit pair must reach this share within tolerance.
#: Halfway between what the two kinds of registration above read.
DEFAULT_REQUIRED_AGREEMENT: Final = 0.75

#: Frames closer than this are neighbours, not revisits.
DEFAULT_MIN_LAG: Final = 20

#: Pixel stride when comparing images for the period: 8 keeps 4000 samples of
#: a 640x400 frame, plenty for a median and cheap in pure Python.
PERIOD_STRIDE: Final = 8

#: Pixel stride for the revisit projection: 6 keeps about 7000 candidates.
AGREEMENT_STRIDE: Final = 6

#: Fewer co-valid pixels than this and two frames cannot be compared.
MIN_COVALID: Final = 200

#: Frames a period is sampled at, spread over the first third of the sweep,
#: which is where a 1.5-turn capture's revisits are.
PERIOD_SAMPLES: Final = 5

#: The final consistency gate needs more angular coverage than the inexpensive
#: period finder. Five image positions plus four median-lag projections falsely
#: passed a real 2,359-frame trajectory whose actual revisits agreed by only 31%.
COHERENCE_IMAGE_SAMPLES: Final = 25

#: Revisit pairs the agreement is measured on, spread over the frames that
#: have one.
AGREEMENT_SAMPLES: Final = 4

#: A circle needs this many camera centres before its fit means anything.
MIN_ORBIT_POSES: Final = 12


class CoherenceNotAllowed(ScannerError):
    """Inputs this module refuses to judge, raised before any frame is read."""

    status = DoctorStatus.UNSUPPORTED_MODE


# --- the image period -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeriodSample:
    """One sampled frame's best later match, and whether it counts as a revisit."""

    frame: int
    best_frame: int
    lag: int
    match: float
    adjacent: float | None

    @property
    def credible(self) -> bool:
        """A revisit matches at least as well as the frame's own neighbour does."""
        return self.adjacent is not None and self.match < self.adjacent

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "best_frame": self.best_frame,
            "lag": self.lag,
            "median_depth_difference": self.match,
            "adjacent_median_depth_difference": self.adjacent,
            "credible": self.credible,
        }


@dataclass(frozen=True, slots=True)
class PeriodEstimate:
    """The image period of a sweep, and the samples it was read from."""

    period: int
    samples: tuple[PeriodSample, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "period_frames": self.period,
            "samples": [sample.as_dict() for sample in self.samples],
        }


def _grid(raw: bytes, stride: int) -> list[int]:
    """One frame's codes on a coarse pixel grid, no-return kept as zero."""
    codes: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, ensure_frame_length(raw))
    return [
        codes[row * FRAME_WIDTH + column]
        for row in range(0, FRAME_HEIGHT, stride)
        for column in range(0, FRAME_WIDTH, stride)
    ]


def _median_difference(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Median |a - b| over pixels both frames returned, or None when too few did."""
    differences = [abs(x - y) for x, y in zip(a, b, strict=True) if x and y]
    if len(differences) < MIN_COVALID:
        return None
    return float(statistics.median(differences))


def _spread(count: int, upto: int) -> list[int]:
    """``count`` distinct indices spread evenly over ``0..upto`` inclusive."""
    if upto < 0:
        return []
    if count <= 1 or upto == 0:
        return [0]
    return sorted({round(index * upto / (count - 1)) for index in range(count)})


def image_period(
    frames: Sequence[bytes],
    *,
    depth_scale: float,
    min_lag: int = DEFAULT_MIN_LAG,
    samples: int = PERIOD_SAMPLES,
    stride: int = PERIOD_STRIDE,
) -> PeriodEstimate | None:
    """The lag, in frames, at which the sweep's images repeat - or None.

    For each sampled frame in the first third of the sweep, every later frame
    at least ``min_lag`` away is compared by median absolute depth difference
    over the pixels both returned. The best one is a revisit only if it
    matches better than the sampled frame's own neighbour; a sweep that never
    comes back around always has a best match, and it is always worse than
    the neighbour. The period is the median credible lag, and it needs at
    least two credible samples, or half of them, whichever is more.
    """
    scale = ensure_scale_usable(depth_scale)
    if min_lag < 1:
        raise CoherenceNotAllowed(f"min_lag must be at least 1, got {min_lag}")
    if len(frames) < min_lag + 2:
        return None
    grids = [_grid(raw, stride) for raw in frames]
    records: list[PeriodSample] = []
    for frame in _spread(samples, len(frames) // 3):
        adjacent = _median_difference(grids[frame], grids[frame + 1])
        best: tuple[int, float] | None = None
        for later in range(frame + min_lag, len(frames)):
            score = _median_difference(grids[frame], grids[later])
            if score is not None and (best is None or score < best[1]):
                best = (later, score)
        if best is None:
            continue
        records.append(
            PeriodSample(
                frame=frame,
                best_frame=best[0],
                lag=best[0] - frame,
                match=best[1] * scale,
                adjacent=None if adjacent is None else adjacent * scale,
            )
        )
    credible = [record.lag for record in records if record.credible]
    if len(credible) < max(2, math.ceil(len(records) / 2)):
        return None
    return PeriodEstimate(period=round(statistics.median(credible)), samples=tuple(records))


# --- agreement across one period ------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeriodPair:
    """One revisit pair: the sampled frame, the frame one period later, and how they agree."""

    frame: int
    revisit: int
    points: int
    median_offset: float
    within_tolerance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "revisit": self.revisit,
            "points": self.points,
            "median_offset": self.median_offset,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True, slots=True)
class CoherenceReport:
    """Whether frames one image period apart land on each other through their poses.

    ``coherent`` is None when the sweep has no image period - a handheld pass
    or under one turn - because a check that could not run is not a check
    that passed, and a report that said ``True`` there would be believed.
    """

    period: int | None
    tolerance: float
    required_agreement: float
    pairs: tuple[PeriodPair, ...]
    estimate: PeriodEstimate | None = None
    pair_selection: str = "fixed_lag"
    requested_image_samples: int | None = None
    uncheckable_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def agreement(self) -> float | None:
        """The weakest sampled pair's share within tolerance."""
        if not self.pairs:
            return None
        return min(pair.within_tolerance for pair in self.pairs)

    @property
    def coherent(self) -> bool | None:
        agreement = self.agreement
        if agreement is not None and agreement < self.required_agreement:
            return False
        if agreement is None or self.uncheckable_pairs:
            return None
        return True

    def statement(self) -> str:
        """One sentence for a progress line, saying which of the three outcomes this is."""
        if self.period is None:
            return (
                "not checkable: no sampled frame's depth image matched a later one better "
                "than its own neighbour, so this sweep has no revisit by image similarity "
                "(a handheld pass, or under one turn)"
            )
        if not self.pairs:
            return (
                f"not checkable: the images repeat every {self.period} frames but no sampled "
                f"revisit pair had {MIN_COVALID} projectable points inside the depth band"
            )
        if self.coherent is None:
            return (
                f"not checkable: {len(self.uncheckable_pairs)} image-matched revisit pairs "
                f"had fewer than {MIN_COVALID} projectable points; other measured pairs "
                "do not establish a consistency pass for them"
            )
        weakest = min(self.pairs, key=lambda pair: pair.within_tolerance)
        detail = (
            f"weakest sampled pair (frames {weakest.frame} and {weakest.revisit}) "
            f"{weakest.within_tolerance:.0%} of points within {self.tolerance:g}, "
            f"median offset {weakest.median_offset:+.2f}; required {self.required_agreement:.0%}"
        )
        subject = (
            "image-matched revisit frames"
            if self.pair_selection == "actual_image_match"
            else f"frames one image period ({self.period} frames) apart"
        )
        if self.coherent:
            return f"coherent: {subject} agree; {detail}"
        return (
            f"INCOHERENT: {subject} have the same image but poses that disagree; "
            f"{detail}. Every number measured on this "
            "registration is measured on a smear of misplaced frames"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "period_frames": self.period,
            "pair_selection": self.pair_selection,
            "requested_image_samples": self.requested_image_samples,
            "uncheckable_pairs": [list(pair) for pair in self.uncheckable_pairs],
            "tolerance": self.tolerance,
            "required_agreement": self.required_agreement,
            "agreement": self.agreement,
            "coherent": self.coherent,
            "pairs": [pair.as_dict() for pair in self.pairs],
            "period_estimate": None if self.estimate is None else self.estimate.as_dict(),
            "statement": self.statement(),
        }


def _best_revisit(frames: Sequence[bytes], frame: int, period: int) -> int:
    """The frame nearest one period after ``frame`` whose image matches it best.

    The period is an integer and the turntable's is not: 126.5 frames a turn
    puts the revisit half a frame off at lag 126 or 127, and half a frame is
    1.4 degrees, over a millimetre at the target's edge - enough to read a
    coherent registration as 68% within tolerance on one pair and 99% on the
    next. The neighbour whose image matches best is the one in phase.
    """
    reference = _grid(frames[frame], PERIOD_STRIDE)
    best, best_score = frame + period, None
    for candidate in (frame + period - 1, frame + period, frame + period + 1):
        if not frame < candidate < len(frames):
            continue
        score = _median_difference(reference, _grid(frames[candidate], PERIOD_STRIDE))
        if score is not None and (best_score is None or score < best_score):
            best, best_score = candidate, score
    return best


def period_agreement(
    frames: Sequence[bytes],
    poses: Sequence[Rigid],
    *,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
    period: int,
    depth_band: DepthBand | None = None,
    samples: int = AGREEMENT_SAMPLES,
    stride: int = AGREEMENT_STRIDE,
    tolerance: float = DEFAULT_TOLERANCE,
    required_agreement: float = DEFAULT_REQUIRED_AGREEMENT,
    estimate: PeriodEstimate | None = None,
) -> CoherenceReport:
    """How well frames ``period`` apart land on each other through ``poses``.

    Each pose maps its frame's camera coordinates into the model frame, the
    direction :func:`scanner3d.pipeline.reconstruct_sweep` returns. For each
    sampled frame ``k``, the points of its revisit - the frame nearest
    ``k + period`` whose image matches frame ``k`` best, see
    :func:`_best_revisit` - that lie inside the depth band are carried into
    the model and back into frame ``k``'s camera, then projected with the
    same nearest-pixel rule the fusion uses. The vote is the depth frame
    ``k`` recorded at that pixel minus the point's depth. A pixel frame ``k``
    returned nothing at is not counted either way.
    """
    if len(frames) != len(poses):
        raise CoherenceNotAllowed(
            f"{len(frames)} frames were offered with {len(poses)} poses; nothing was judged"
        )
    if not 1 <= period < len(frames):
        raise CoherenceNotAllowed(
            f"a period of {period} frames cannot be checked on {len(frames)} frames"
        )
    if not (0.0 < required_agreement <= 1.0) or not math.isfinite(tolerance) or tolerance <= 0:
        raise CoherenceNotAllowed(
            f"tolerance {tolerance!r} and required agreement {required_agreement!r} must be "
            "positive, with the agreement a fraction"
        )
    fx, fy, cx, cy = ensure_intrinsics_usable(intrinsics)
    scale = ensure_scale_usable(depth_scale)
    band = ensure_depth_band(depth_band)
    pairs: list[PeriodPair] = []
    for frame in _spread(samples, len(frames) - period - 1):
        revisit = _best_revisit(frames, frame, period)
        seen: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, ensure_frame_length(frames[frame]))
        later: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, ensure_frame_length(frames[revisit]))
        to_model = poses[revisit]
        to_camera = poses[frame].inverse()
        votes: list[float] = []
        for row in range(0, FRAME_HEIGHT, stride):
            for column in range(0, FRAME_WIDTH, stride):
                code = later[row * FRAME_WIDTH + column]
                if code == NO_RETURN_SAMPLE:
                    continue
                depth = code * scale
                if not band.contains(depth):
                    continue
                point = ((column - cx) * depth / fx, (row - cy) * depth / fy, depth)
                x, y, z = to_camera.apply(to_model.apply(point))
                if z <= 0.0:
                    continue
                u = math.floor(fx * x / z + cx + 0.5)
                v = math.floor(fy * y / z + cy + 0.5)
                if not (0 <= u < FRAME_WIDTH and 0 <= v < FRAME_HEIGHT):
                    continue
                recorded = seen[v * FRAME_WIDTH + u]
                if recorded == NO_RETURN_SAMPLE:
                    continue
                votes.append(recorded * scale - z)
        if len(votes) < MIN_COVALID:
            continue
        within = sum(1 for vote in votes if abs(vote) < tolerance) / len(votes)
        pairs.append(
            PeriodPair(
                frame=frame,
                revisit=revisit,
                points=len(votes),
                median_offset=float(statistics.median(votes)),
                within_tolerance=within,
            )
        )
    return CoherenceReport(
        period=period,
        tolerance=tolerance,
        required_agreement=required_agreement,
        pairs=tuple(pairs),
        estimate=estimate,
    )


def check_coherence(
    frames: Sequence[bytes],
    poses: Sequence[Rigid],
    *,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
    depth_band: DepthBand | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    required_agreement: float = DEFAULT_REQUIRED_AGREEMENT,
) -> CoherenceReport:
    """Project every credible actual image match, with broader angular sampling.

    A sweep with no image period gets a report whose ``coherent`` is None and
    whose statement says so. The median lag remains a descriptive estimate,
    never a substitute for the actual pairs that established the revisits.
    """
    if len(frames) != len(poses):
        raise CoherenceNotAllowed(
            f"{len(frames)} frames were offered with {len(poses)} poses; nothing was judged"
        )
    estimate = image_period(frames, depth_scale=depth_scale, samples=COHERENCE_IMAGE_SAMPLES)
    if estimate is None:
        return CoherenceReport(
            period=None,
            tolerance=tolerance,
            required_agreement=required_agreement,
            pairs=(),
            pair_selection="actual_image_match",
            requested_image_samples=COHERENCE_IMAGE_SAMPLES,
        )
    pairs: list[PeriodPair] = []
    uncheckable: list[tuple[int, int]] = []
    for sample in estimate.samples:
        if not sample.credible:
            continue
        first, later = sample.frame, sample.best_frame
        checked = period_agreement(
            (frames[first], frames[later]),
            (poses[first], poses[later]),
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            period=1,
            depth_band=depth_band,
            samples=1,
            tolerance=tolerance,
            required_agreement=required_agreement,
        )
        if not checked.pairs:
            uncheckable.append((first, later))
        pairs.extend(replace(pair, frame=first, revisit=later) for pair in checked.pairs)
    return CoherenceReport(
        period=estimate.period,
        tolerance=tolerance,
        required_agreement=required_agreement,
        pairs=tuple(pairs),
        estimate=estimate,
        pair_selection="actual_image_match",
        requested_image_samples=COHERENCE_IMAGE_SAMPLES,
        uncheckable_pairs=tuple(uncheckable),
    )


# --- the camera orbit -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrbitFit:
    """A circle through the camera centres: the turntable's axis, read from the poses.

    ``axis`` is a unit vector along the spindle with no particular sign;
    ``centre`` is the point on it in the plane of the camera centres, which
    for a scanner looking down at a turntable is ABOVE the turntable top, not
    on it. A caller that needs "up" orients the axis by where the scene is.
    """

    axis: Point
    centre: Point
    radius: float
    residual_median: float
    residual_p90: float
    out_of_plane_rms: float
    poses: int

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": list(self.axis),
            "centre": list(self.centre),
            "radius": self.radius,
            "residual_median": self.residual_median,
            "residual_p90": self.residual_p90,
            "out_of_plane_rms": self.out_of_plane_rms,
            "poses": self.poses,
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def camera_orbit(poses: Sequence[Rigid], *, min_poses: int = MIN_ORBIT_POSES) -> OrbitFit | None:
    """Fit a circle to the pose translations, or None when they do not describe one.

    The plane normal is the consistent mean of chord cross products a quarter
    of the sequence apart, which is exact for a circle and indifferent to
    where along it the sweep started; the circle in that plane is the
    algebraic least-squares fit, solved by Cramer's rule so this stays pure
    Python. None for too few poses, centres that do not span a plane, or a
    fit with no real radius - never a circle of radius zero.
    """
    centres = [pose.translation for pose in poses]
    count = len(centres)
    if count < min_poses:
        return None
    mean = (
        sum(c[0] for c in centres) / count,
        sum(c[1] for c in centres) / count,
        sum(c[2] for c in centres) / count,
    )
    offsets = [subtract(c, mean) for c in centres]
    quarter = max(1, count // 4)
    total: Point = (0.0, 0.0, 0.0)
    for index, a in enumerate(offsets):
        b = offsets[(index + quarter) % count]
        chord = cross(a, b)
        if dot(chord, total) < 0.0:
            chord = (-chord[0], -chord[1], -chord[2])
        total = (total[0] + chord[0], total[1] + chord[1], total[2] + chord[2])
    if norm(total) < 1e-9:
        return None
    axis = normalize(total)
    seed = max(offsets, key=norm)
    in_plane = subtract(
        seed, (axis[0] * dot(seed, axis), axis[1] * dot(seed, axis), axis[2] * dot(seed, axis))
    )
    if norm(in_plane) < 1e-9:
        return None
    e1 = normalize(in_plane)
    e2 = cross(axis, e1)
    planar = [(dot(o, e1), dot(o, e2)) for o in offsets]
    heights = [dot(o, axis) for o in offsets]
    # Kasa fit: x^2 + y^2 = 2 a x + 2 b y + c, linear in (a, b, c).
    sxx = sum(x * x for x, _ in planar)
    sxy = sum(x * y for x, y in planar)
    syy = sum(y * y for _, y in planar)
    sx = sum(x for x, _ in planar)
    sy = sum(y for _, y in planar)
    rhs = [x * x + y * y for x, y in planar]
    matrix = (
        (2.0 * sxx, 2.0 * sxy, sx),
        (2.0 * sxy, 2.0 * syy, sy),
        (2.0 * sx, 2.0 * sy, float(count)),
    )
    vector = (
        sum(r * x for r, (x, _) in zip(rhs, planar, strict=True)),
        sum(r * y for r, (_, y) in zip(rhs, planar, strict=True)),
        sum(rhs),
    )
    det = determinant(matrix)
    if abs(det) < 1e-12:
        return None

    def solve(column: int) -> float:
        replaced = tuple(
            tuple(vector[row] if col == column else matrix[row][col] for col in range(3))
            for row in range(3)
        )
        return determinant(replaced) / det  # type: ignore[arg-type]

    alpha, beta, gamma = solve(0), solve(1), solve(2)
    squared = gamma + alpha * alpha + beta * beta
    if not math.isfinite(squared) or squared <= 0.0:
        return None
    radius = math.sqrt(squared)
    residuals = [abs(math.hypot(x - alpha, y - beta) - radius) for x, y in planar]
    centre = (
        mean[0] + alpha * e1[0] + beta * e2[0],
        mean[1] + alpha * e1[1] + beta * e2[1],
        mean[2] + alpha * e1[2] + beta * e2[2],
    )
    return OrbitFit(
        axis=axis,
        centre=centre,
        radius=radius,
        residual_median=float(statistics.median(residuals)),
        residual_p90=_percentile(residuals, 0.9),
        out_of_plane_rms=math.sqrt(sum(h * h for h in heights) / count),
        poses=count,
    )
