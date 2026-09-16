"""Whether a frame's depth VALUES describe a scene, and what to say when they do not.

The gap this closes, observed
------------------------------
Between 2026-08-31 and 2026-09-01 four consecutive captures failed. Something
had entered the scanner's view about 60 mm away - fixed within a capture,
turning with the turntable - while the object sat at about 180 mm. Those points
were roughly 4600 per frame, they sat 100 mm off the object, and they dominated
the cloud's covariance badly enough to drive its principal-extent ratio to 1.05.
That made a deliberately asymmetric target look rotationally symmetric,
:func:`~scanner3d.multiview.coarse_align` could no longer rank orientations
(margins of 1.00-1.11x against the 1.15x it requires), the tracker's references
disagreed, and frames were refused.

Every part of that was in the first frame's bytes, and nothing said so:

- :mod:`scanner3d.doctor` checks presence, identity, topology, roles, modes and
  permissions - the device, never a frame's content.
- :mod:`scanner3d.return_health` counts returns and locates them by quadrant. It
  is the right instrument for "the object slid off the sensor" and it caught
  nothing here, correctly: the return COUNT was normal. It was the values that
  were wrong.
- :mod:`scanner3d.reconstruction_readiness` is strict and thorough and runs
  after the capture is over.

So the operator was told "return health: usable - returns 36.5% of the frame",
"ROTATION - GOOD", and a precise paragraph about an ICP disagreement. All three
true. None of them says *there is something in your view at 60 mm*.

This module is the sibling of :mod:`~scanner3d.return_health`: that one asks
WHERE the returns are, this one asks WHETHER THE DISTANCES MAKE SENSE, and both
answer for one frame at a time while a capture is still salvageable.

What it looks for
-----------------
Three shapes, each one a thing that was measured by hand on the failing
captures and each one cheap enough to run on every frame:

* **A detached cluster.** The depth histogram splits into a near group and a far
  group with an empty span between them. An object and its background are
  usually continuous; a 59 mm void with 4.6% of the frame stranded on the near
  side of it is a second thing in the room. See :func:`depth_profile`.
* **Returns outside the working envelope.** Distances the sensor was never
  characterised at, counted and located rather than silently clipped away by
  :data:`~scanner3d.preparation.DEPTH_CLIP_NEAR`.
* **A region that will not move.** Held across frames by
  :class:`StaticRegionTracker`: on a capture where the scene is supposed to be
  turning, a patch whose depth does not change is fixed to the camera.

What it refuses to do
---------------------
* **It never names a cause.** A near cluster can be a clamp, a cable, an
  operator's hand, the projector's overexposed near field or a correlation
  failure on a dim far surface. The bytes do not say which, so the advice names
  what to look at and never what it is.
* **It never decides anything.** Exactly like :mod:`~scanner3d.return_health`:
  advisory only. It accepts and refuses no frame, moves no tracking gate, clips
  nothing, and changes no reconstruction threshold. The only thing it may do is
  appear on screen and in the log.
* **It implies no unit.** Every distance here is raw code times depth scale, in
  whatever that produces. The envelope bounds are stated in the same scale and
  are a characterisation of this unit, not a specification.

Deterministic and allocation-bounded: the histogram is a fixed number of bins
however many samples arrive, and :class:`StaticRegionTracker` holds at most
:data:`STATIC_HISTORY_FRAMES` coarse grids however long a capture runs.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.depth_metric import NO_RETURN_SAMPLE
from scanner3d.errors import DoctorStatus, ScannerError

#: Every dict this module emits stamps this.
SCHEMA_VERSION: Final = 1


# --- refusals ---------------------------------------------------------------


class DepthPlausibilityError(ScannerError):
    """Base for every refusal here. Raised before a sample is looked at."""

    status = DoctorStatus.UNSUPPORTED_MODE


class PlausibilityFrameShapeMismatch(DepthPlausibilityError):
    """``samples`` was not exactly ``width * height`` long, or the shape was not positive.

    Refused rather than reshaped, for the same reason
    :class:`~scanner3d.return_health.FrameShapeMismatch` is: a sequence of
    another length is a frame of another shape, and guessing which would put
    every region boundary somewhere else while still producing plausible
    numbers.
    """


class PlausibilityScaleUnusable(DepthPlausibilityError):
    """The depth scale offered cannot turn a raw code into a distance."""


class PlausibilityEnvelopeUnusable(DepthPlausibilityError):
    """The working envelope offered is not an interval a return can fall inside."""


# --- the envelope this unit was characterised at ----------------------------

#: The distance interval this scanner's returns were measured over, in the scale
#: ``raw * depth_scale`` produces.
#:
#: From docs/capture-checklist.md's working envelope, itself the 1st and 99th
#: percentile of the depth codes in the nine earliest committed frames: about 84
#: to about 290. That provenance is thin and is stated here rather than hidden,
#: because it is the weakest number in this module: it describes what one set of
#: nine frames happened to contain, not a measured limit of the sensor.
#:
#: Deliberately NOT the same numbers as
#: :data:`~scanner3d.preparation.DEPTH_CLIP_NEAR` and ``DEPTH_CLIP_FAR``, which
#: are 60 and 300. Those decide what is thrown away before a cloud is built;
#: these decide what is worth REMARKING ON. A return between 60 and 84 is kept
#: by the clip and is odd enough to mention, and keeping the two apart is what
#: lets this module report on the clip's own behaviour instead of inheriting it.
ENVELOPE_NEAR: Final = 84.0
ENVELOPE_FAR: Final = 290.0

#: Bins in the depth histogram. Fixed, so the cost is a property of this module
#: rather than of the frame: 128 bins over the clip interval is about 1.9 units
#: per bin at the current bounds, which is roughly three noise floors - fine
#: enough to see a gap that matters and coarse enough not to invent one out of
#: sensor noise.
HISTOGRAM_BINS: Final = 128

#: How wide an empty span in the histogram has to be before the returns either
#: side of it are called two separate things, as a multiple of
#: :data:`~scanner3d.pipeline.NOISE_FLOOR`.
#:
#: Twenty, so about 13 units at the measured noise floor. A real object and the
#: surface behind it are usually continuous or separated by a few units; the gap
#: measured on the failing captures was 59. This is set well above the former
#: and well below the latter on purpose - it is a detector for "a second thing
#: in the room", not a shape descriptor.
GAP_NOISE_MULTIPLE: Final = 20.0

#: Smallest share of a frame's returns that has to be stranded on the near side
#: of a gap before the split is worth reporting.
#:
#: A hundredth. Below that a "cluster" is a handful of stray samples, which
#: every frame has and which nothing downstream cares about; the failing
#: captures stranded 4.6%.
MIN_CLUSTER_FRACTION: Final = 0.01

#: Largest share it may hold and still be an intruder rather than the subject.
#:
#: A half, and this bound exists because the first version of this module did
#: not have it and was wrong. Without it, an object standing well clear of a
#: distant wall - a completely ordinary scene - splits into two groups and the
#: NEAR one gets reported, which is the object itself. The thing being scanned
#: is the thing most of the returns come from; an intruder is by definition a
#: minority of the view. Two comparable groups are two answers, and this says
#: nothing rather than guessing which is which.
MAX_DETACHED_FRACTION: Final = 0.5


# --- what a frame's distances look like -------------------------------------


@dataclass(frozen=True, slots=True)
class DepthCluster:
    """One contiguous group of returns, separated from the others by a gap."""

    #: Nearest and furthest return in this group, in the depth scale's units.
    near: float
    far: float
    #: Returns in this group, and their share of the frame's returns.
    samples: int
    fraction: float

    @property
    def extent(self) -> float:
        """How deep this group is, front to back."""
        return self.far - self.near

    def as_dict(self) -> dict[str, object]:
        return {
            "near": self.near,
            "far": self.far,
            "extent": self.extent,
            "samples": self.samples,
            "fraction": self.fraction,
        }


@dataclass(frozen=True, slots=True)
class DepthProfile:
    """Every distance measurement one frame supports, and nothing derived from more.

    ``clusters`` is ordered near to far and is never empty for a frame with any
    return at all. A healthy frame has exactly one.
    """

    samples: int
    returned: int
    #: Returns nearer than :data:`ENVELOPE_NEAR` and further than
    #: :data:`ENVELOPE_FAR`. Counted, never removed.
    below_envelope: int
    above_envelope: int
    clusters: tuple[DepthCluster, ...]
    #: The widest empty span between consecutive clusters, or ``0.0`` for a
    #: frame with fewer than two.
    widest_gap: float

    @property
    def yield_fraction(self) -> float:
        return 0.0 if self.samples == 0 else self.returned / self.samples

    @property
    def out_of_envelope(self) -> int:
        return self.below_envelope + self.above_envelope

    @property
    def out_of_envelope_fraction(self) -> float:
        return 0.0 if self.returned == 0 else self.out_of_envelope / self.returned

    @property
    def detached(self) -> DepthCluster | None:
        """The near cluster of a split frame, when it is a minority of the view.

        Two conditions, and both are load-bearing. **Near**, because the failure
        this exists for puts a second object between the scanner and the
        subject; a far group is ordinarily the room behind it, which is not
        news. **A minority**, because the subject is whatever most of the
        returns came from - without that bound an object standing clear of a
        distant wall reports itself as an intruder. See
        :data:`MAX_DETACHED_FRACTION`.
        """
        if len(self.clusters) < 2:
            return None
        first = self.clusters[0]
        if not MIN_CLUSTER_FRACTION <= first.fraction < MAX_DETACHED_FRACTION:
            return None
        return first

    @property
    def dominant(self) -> DepthCluster | None:
        """The cluster most of the returns came from - the subject, when there is one.

        Distinct from ``clusters[0]``, which is merely the NEAREST group and on
        a real frame is routinely a handful of stray near returns: measured on
        ``20260901-201341`` frame 0, the nearest group held 14 of 77,764
        returns while the target sat in the fourth. Anything describing "where
        the returns are" wants this one; anything describing "what is in front
        of the subject" wants :attr:`detached`.
        """
        if not self.clusters:
            return None
        return max(self.clusters, key=lambda cluster: cluster.samples)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "samples": self.samples,
            "returned": self.returned,
            "yield_fraction": self.yield_fraction,
            "below_envelope": self.below_envelope,
            "above_envelope": self.above_envelope,
            "out_of_envelope": self.out_of_envelope,
            "out_of_envelope_fraction": self.out_of_envelope_fraction,
            "envelope_near": ENVELOPE_NEAR,
            "envelope_far": ENVELOPE_FAR,
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "widest_gap": self.widest_gap,
            "detached": None if self.detached is None else self.detached.as_dict(),
            "dominant": None if self.dominant is None else self.dominant.as_dict(),
        }


def _ensure_scale(depth_scale: float) -> float:
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise PlausibilityScaleUnusable(
            f"depth_scale {depth_scale!r} cannot turn a raw code into a distance; it must be "
            "a finite positive number"
        )
    return depth_scale


def _ensure_envelope(near: float, far: float) -> tuple[float, float]:
    if not math.isfinite(near) or not math.isfinite(far):
        raise PlausibilityEnvelopeUnusable(
            f"the working envelope ({near!r}, {far!r}) is not a pair of finite distances"
        )
    if near < 0.0 or far <= near:
        raise PlausibilityEnvelopeUnusable(
            f"the working envelope ({near}, {far}) is not an interval a return can fall inside"
        )
    return near, far


def depth_profile(
    samples: Sequence[int],
    *,
    depth_scale: float,
    envelope: tuple[float, float] = (ENVELOPE_NEAR, ENVELOPE_FAR),
    bins: int = HISTOGRAM_BINS,
    gap_multiple: float = GAP_NOISE_MULTIPLE,
) -> DepthProfile:
    """Split one frame's returns into clusters and count what sits outside the envelope.

    ``samples`` are raw codes exactly as the sensor returned them;
    :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE` is excluded as an absence
    and never as a distance, which is the one rule
    :mod:`~scanner3d.return_health` and this module share without qualification.

    The clustering is a fixed-bin histogram walked once: consecutive occupied
    bins belong to the same group, and a run of empty bins wider than
    ``gap_multiple`` noise floors ends one. That is deliberately the cheapest
    thing that can see the observed failure rather than the best available
    clustering - it runs on every frame between a driver handing a buffer over
    and getting it back, and a k-means over a quarter of a million samples does
    not belong there.
    """
    from scanner3d.pipeline import NOISE_FLOOR

    scale = _ensure_scale(depth_scale)
    near_bound, far_bound = _ensure_envelope(*envelope)
    if bins < 2:
        raise PlausibilityEnvelopeUnusable(f"a histogram needs at least 2 bins, got {bins}")

    total = len(samples)
    try:
        import numpy as np
    except ImportError:
        numeric = None
        distances = [code * scale for code in samples if code != NO_RETURN_SAMPLE]
    else:
        codes = np.asarray(samples)
        numeric = codes[codes != NO_RETURN_SAMPLE] * scale
        distances = []
    returned = len(distances) if numeric is None else len(numeric)
    if returned == 0:
        return DepthProfile(
            samples=total,
            returned=0,
            below_envelope=0,
            above_envelope=0,
            clusters=(),
            widest_gap=0.0,
        )

    if numeric is None:
        below = sum(1 for d in distances if d < near_bound)
        above = sum(1 for d in distances if d > far_bound)
        low, high = min(distances), max(distances)
    else:
        below = int(np.count_nonzero(numeric < near_bound))
        above = int(np.count_nonzero(numeric > far_bound))
        low, high = float(numeric.min()), float(numeric.max())
    span = high - low
    if span <= 0.0:
        # Every return at one distance. One cluster of zero extent, and no gap
        # to look for - a real state for a frame of a flat wall, and one the
        # binning below would divide by zero on.
        return DepthProfile(
            samples=total,
            returned=returned,
            below_envelope=below,
            above_envelope=above,
            clusters=(DepthCluster(near=low, far=high, samples=returned, fraction=1.0),),
            widest_gap=0.0,
        )

    width = span / bins
    if numeric is None:
        counts = [0] * bins
        for d in distances:
            index = int((d - low) / width)
            counts[min(index, bins - 1)] += 1
    else:
        indices = np.minimum(((numeric - low) / width).astype(np.int64), bins - 1)
        counts = np.bincount(indices, minlength=bins).tolist()

    gate = gap_multiple * NOISE_FLOOR
    groups: list[list[int]] = []
    run: list[int] = []
    empty = 0
    for index, count in enumerate(counts):
        if count:
            if run and empty * width > gate:
                groups.append(run)
                run = []
            run.append(index)
            empty = 0
        elif run:
            empty += 1
    if run:
        groups.append(run)

    clusters: list[DepthCluster] = []
    for group in groups:
        first, last = group[0], group[-1]
        if numeric is None:
            held = [d for d in distances if low + first * width <= d <= low + (last + 1) * width]
        else:
            held = numeric[
                (numeric >= low + first * width) & (numeric <= low + (last + 1) * width)
            ].tolist()
        if not held:  # pragma: no cover - a bin with a count always holds a sample
            continue
        clusters.append(
            DepthCluster(
                near=min(held),
                far=max(held),
                samples=len(held),
                fraction=len(held) / returned,
            )
        )

    widest = 0.0
    for a, b in itertools.pairwise(clusters):
        widest = max(widest, b.near - a.far)

    return DepthProfile(
        samples=total,
        returned=returned,
        below_envelope=below,
        above_envelope=above,
        clusters=tuple(clusters),
        widest_gap=widest,
    )


# --- where in the frame a cluster's returns sit ------------------------------

#: Names for the nine parts of the view, by which third a centroid falls in.
#: Three by three because it is the coarsest split that still distinguishes the
#: places an operator can actually look - an edge, a corner, the middle - and
#: because a finer grid would imply a precision the centroid of a scattered
#: cluster does not have.
FRAME_THIRDS: Final = (
    ("upper left", "top", "upper right"),
    ("left", "the middle", "right"),
    ("lower left", "bottom", "lower right"),
)


@dataclass(frozen=True, slots=True)
class ClusterLocation:
    """Where one cluster's returns sit in the frame, in pixels and in words.

    Separate from :class:`DepthCluster` because the histogram that finds a
    cluster is a one-dimensional question about distances and needs no frame
    shape at all. Localising is a second pass that does, and keeping the two
    apart is what lets the profile stay usable on a bare sequence of samples.
    """

    left: int
    right: int
    top: int
    bottom: int
    samples: int
    #: One of :data:`FRAME_THIRDS`, from where the returns' centroid falls.
    where: str

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
            "samples": self.samples,
            "where": self.where,
        }


def locate_cluster(
    samples: Sequence[int],
    width: int,
    height: int,
    cluster: DepthCluster,
    *,
    depth_scale: float,
) -> ClusterLocation | None:
    """Where the returns making up ``cluster`` sit in the frame.

    ``None`` when none of them do, which is only reachable if the cluster came
    from a different frame than the samples - stated rather than assumed.

    The bounding box is the honest summary of a set of scattered pixels and the
    words come from the CENTROID rather than the box, because a box spanning
    most of the frame with its mass in one corner is a thing in that corner, and
    an operator told "the whole view" would look nowhere.
    """
    scale = _ensure_scale(depth_scale)
    if width <= 0 or height <= 0 or len(samples) != width * height:
        raise PlausibilityFrameShapeMismatch(
            f"{len(samples)} samples is not a {width}x{height} frame; a sequence of another "
            "length is a frame of another shape, and reshaping it would put the box somewhere "
            "else while still producing plausible numbers"
        )

    left, right = width, -1
    top, bottom = height, -1
    total = 0
    sum_column = sum_row = 0
    for index, code in enumerate(samples):
        if code == NO_RETURN_SAMPLE:
            continue
        distance = code * scale
        if not cluster.near <= distance <= cluster.far:
            continue
        row, column = divmod(index, width)
        left, right = min(left, column), max(right, column)
        top, bottom = min(top, row), max(bottom, row)
        sum_column += column
        sum_row += row
        total += 1

    if total == 0:
        return None
    band_row = min(2, (sum_row // total) * 3 // height)
    band_column = min(2, (sum_column // total) * 3 // width)
    return ClusterLocation(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        samples=total,
        where=FRAME_THIRDS[band_row][band_column],
    )


# --- a region that will not move --------------------------------------------

#: Coarse cells across and down the frame for the static-region test. Twelve by
#: eight is 96 cells: coarse enough that one cell holds thousands of samples and
#: its median is stable, fine enough to localise an intruder to a part of the
#: view an operator can look at.
STATIC_COLUMNS: Final = 12
STATIC_ROWS: Final = 8

#: Frames of coarse grids held. Ten, which at the measured 4.5 fps and the
#: default decimation is several seconds of scanning - long enough for a turning
#: scene to have moved and short enough that the memory is a constant.
STATIC_HISTORY_FRAMES: Final = 10

#: How little a cell's median may move across the whole history before it counts
#: as static, as a multiple of :data:`~scanner3d.pipeline.NOISE_FLOOR`.
STATIC_NOISE_MULTIPLE: Final = 2.0

#: How much of the frame has to be moving before a static cell means anything.
#:
#: A capture that is genuinely still - the operator framing a shot, a paused
#: turntable - has every cell static, and reporting that as an intruder would
#: fire on the commonest harmless state there is. So the test only speaks when
#: most of the view IS changing and a minority of it is not.
MIN_MOVING_CELLS: Final = 0.5


@dataclass(frozen=True, slots=True)
class StaticRegion:
    """Cells whose depth did not change while the rest of the view did."""

    #: Cell coordinates, column then row, from the top-left of the frame.
    cells: tuple[tuple[int, int], ...]
    #: Cells that held a usable median across the whole history.
    compared: int
    #: Cells that moved by more than the gate.
    moving: int
    #: Median distance of the static cells, or ``None`` when there are none.
    distance: float | None

    @property
    def found(self) -> bool:
        return bool(self.cells)

    @property
    def moving_fraction(self) -> float:
        return 0.0 if self.compared == 0 else self.moving / self.compared

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "found": self.found,
            "cells": [list(cell) for cell in self.cells],
            "compared": self.compared,
            "moving": self.moving,
            "moving_fraction": self.moving_fraction,
            "distance": self.distance,
            "columns": STATIC_COLUMNS,
            "rows": STATIC_ROWS,
        }


class StaticRegionTracker:
    """Bounded, resettable memory of where the frame is and is not changing.

    A capture where the object turns and the camera is fixed - the arrangement
    the whole appliance is built around - should have almost nothing in the view
    holding still. A patch that does is fixed relative to the camera, and that
    is worth one sentence on screen whatever it turns out to be.

    Holds at most :data:`STATIC_HISTORY_FRAMES` grids of
    ``STATIC_COLUMNS * STATIC_ROWS`` floats, however long a capture runs.
    """

    def __init__(self) -> None:
        self._history: deque[list[float | None]] = deque(maxlen=STATIC_HISTORY_FRAMES)

    def reset(self) -> None:
        """Forget everything. Called at a capture boundary, never mid-capture."""
        self._history.clear()

    @property
    def frames(self) -> int:
        return len(self._history)

    def observe(
        self,
        samples: Sequence[int],
        width: int,
        height: int,
        *,
        depth_scale: float,
    ) -> StaticRegion:
        """Fold one frame in and report what has not moved across the history.

        Returns an empty :class:`StaticRegion` until the history is full: a
        verdict from two frames of a slow scene would fire on every capture's
        first second, which is exactly when an operator is least able to act on
        it and most likely to stop believing the indicator.
        """
        from scanner3d.pipeline import NOISE_FLOOR

        scale = _ensure_scale(depth_scale)
        if width <= 0 or height <= 0 or len(samples) != width * height:
            raise PlausibilityFrameShapeMismatch(
                f"{len(samples)} samples is not a {width}x{height} frame; a sequence of another "
                "length is a frame of another shape, and reshaping it would put every cell "
                "boundary somewhere else while still producing plausible numbers"
            )

        self._history.append(_cell_medians(samples, width, height, scale))
        if len(self._history) < STATIC_HISTORY_FRAMES:
            return StaticRegion(cells=(), compared=0, moving=0, distance=None)

        gate = STATIC_NOISE_MULTIPLE * NOISE_FLOOR
        static: list[tuple[int, int]] = []
        held: list[float] = []
        compared = moving = 0
        for cell in range(STATIC_COLUMNS * STATIC_ROWS):
            # Built with a loop rather than a comprehension so the None check
            # narrows: a cell is empty in some frames and not others, and the
            # spread below is only meaningful over frames that all had it.
            seen: list[float] = []
            for grid in self._history:
                value = grid[cell]
                if value is not None:
                    seen.append(value)
            if len(seen) < STATIC_HISTORY_FRAMES:
                continue
            compared += 1
            spread = max(seen) - min(seen)
            if spread > gate:
                moving += 1
            else:
                static.append((cell % STATIC_COLUMNS, cell // STATIC_COLUMNS))
                held.append(seen[-1])

        if compared == 0 or moving / compared < MIN_MOVING_CELLS:
            # Not enough of the view is changing for "this part is not" to mean
            # anything. See MIN_MOVING_CELLS.
            return StaticRegion(cells=(), compared=compared, moving=moving, distance=None)

        held.sort()
        return StaticRegion(
            cells=tuple(static),
            compared=compared,
            moving=moving,
            distance=held[len(held) // 2] if held else None,
        )


def _cell_medians(
    samples: Sequence[int], width: int, height: int, scale: float
) -> list[float | None]:
    """One coarse grid of per-cell median distances, or ``None`` for an empty cell."""
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        values = np.asarray(samples).reshape(height, width)
        medians: list[float | None] = []
        for row in range(STATIC_ROWS):
            start_y = (row * height + STATIC_ROWS - 1) // STATIC_ROWS
            end_y = ((row + 1) * height + STATIC_ROWS - 1) // STATIC_ROWS
            for column in range(STATIC_COLUMNS):
                start_x = (column * width + STATIC_COLUMNS - 1) // STATIC_COLUMNS
                end_x = ((column + 1) * width + STATIC_COLUMNS - 1) // STATIC_COLUMNS
                cell_values = values[start_y:end_y, start_x:end_x]
                distances = cell_values[cell_values != NO_RETURN_SAMPLE] * scale
                middle = len(distances) // 2
                medians.append(
                    None if not len(distances) else float(np.partition(distances, middle)[middle])
                )
        return medians

    buckets: list[list[float]] = [[] for _ in range(STATIC_COLUMNS * STATIC_ROWS)]
    for index, code in enumerate(samples):
        if code == NO_RETURN_SAMPLE:
            continue
        row, column = divmod(index, width)
        cell = (row * STATIC_ROWS // height) * STATIC_COLUMNS + (column * STATIC_COLUMNS // width)
        buckets[cell].append(code * scale)
    grid: list[float | None] = []
    for bucket in buckets:
        if not bucket:
            grid.append(None)
            continue
        bucket.sort()
        grid.append(bucket[len(bucket) // 2])
    return grid


# --- what to say ------------------------------------------------------------

PLAUSIBLE: Final = "plausible"
DETACHED_CLUSTER: Final = "detached cluster"
OUTSIDE_ENVELOPE: Final = "outside envelope"
STATIC_REGION: Final = "static region"
NO_RETURNS: Final = "no returns"

#: Verdicts an operator should act on. :data:`PLAUSIBLE` is the only one absent.
WARNING_VERDICTS: Final = (DETACHED_CLUSTER, OUTSIDE_ENVELOPE, STATIC_REGION, NO_RETURNS)

#: Share of returns outside the envelope before it is worth a sentence on its
#: own. Two hundredths: the failing captures ran 2.3% and 4.7%, and every clean
#: capture on record ran 0.0%.
ENVELOPE_WARN_FRACTION: Final = 0.02

#: Share of a detached cluster that has to sit below the envelope before the
#: advice stops looking for an intruder and says "you are too close" instead.
#:
#: Measured, and the reason it exists is a wrong sentence this module said. On
#: 2026-09-01 it correctly detected a near cluster and told the owner "something
#: is in the scanner's view much closer than the object" - and there was
#: nothing. A colour frame settled it: the near returns were the target's own
#: ramp, and the whole scene ran from 1 to 232 with 46.5% of it inside the
#: sensor's near limit. The scanner was simply too close. Both cases produce a
#: near cluster, and what tells them apart is how much of that cluster is
#: unmeasurable: a foreign object sits in front of a subject that is itself in
#: range, while a subject too close drags a large share of the frame under the
#: near bound with it.
#:
#: Half. Above it the message names the distance rather than hunting for a
#: thing, because at that point most of what the cluster contains was never
#: measurable in the first place.
TOO_CLOSE_BELOW_ENVELOPE: Final = 0.5


@dataclass(frozen=True, slots=True)
class Plausibility:
    """One deterministic verdict over one frame, and the sentence for it."""

    verdict: str
    detail: str
    advice: str

    @property
    def warning(self) -> bool:
        return self.verdict in WARNING_VERDICTS

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict,
            "detail": self.detail,
            "advice": self.advice,
            "warning": self.warning,
        }


def plausibility(
    profile: DepthProfile,
    static: StaticRegion | None = None,
    location: ClusterLocation | None = None,
) -> Plausibility:
    """Turn a profile and an optional static-region reading into one verdict.

    The order is by how specific the finding is, not by how bad it is: a
    detached cluster names a place to look, "outside the envelope" names a
    range, and a static region names a part of the view. An operator given the
    most specific true sentence can act; one given the most alarming one cannot.

    Every sentence names what to LOOK AT and never what the thing is. The bytes
    do not carry a cause and this module does not invent one - see the module
    docstring.
    """
    if profile.returned == 0:
        return Plausibility(
            verdict=NO_RETURNS,
            detail="the frame carried no depth at all",
            advice="Nothing is in range. Check the scanner is pointed at the object.",
        )

    detached = profile.detached
    if detached is not None:
        where = "" if location is None else f", in the {location.where} of the view"
        detail = (
            f"{detached.samples} returns ({detached.fraction:.1%}) sit at "
            f"{detached.near:.0f}-{detached.far:.0f}{where}, {profile.widest_gap:.0f} clear "
            "of everything else in the frame"
        )
        # Which sentence is right depends on how much of the near group was
        # measurable at all. See TOO_CLOSE_BELOW_ENVELOPE for the capture that
        # made this a distinction rather than one message.
        unmeasurable = profile.below_envelope / detached.samples if detached.samples else 0.0
        if unmeasurable >= TOO_CLOSE_BELOW_ENVELOPE:
            return Plausibility(
                verdict=DETACHED_CLUSTER,
                detail=detail,
                advice=(
                    f"Most of that is nearer than the {ENVELOPE_NEAR:.0f} this scanner was "
                    "characterised down to, so it was never measurable. Move the scanner back "
                    "until the whole object sits inside the working range."
                ),
            )
        look = (
            "including at the edges of the view"
            if location is None
            else f"and look at the {location.where} of the view first"
        )
        return Plausibility(
            verdict=DETACHED_CLUSTER,
            detail=detail,
            advice=(
                "Something is in the scanner's view much closer than the object. Check what is "
                f"between the scanner and the target, {look}."
            ),
        )

    if profile.out_of_envelope_fraction >= ENVELOPE_WARN_FRACTION:
        return Plausibility(
            verdict=OUTSIDE_ENVELOPE,
            detail=(
                f"{profile.out_of_envelope} returns ({profile.out_of_envelope_fraction:.1%}) "
                f"are outside the {ENVELOPE_NEAR:.0f}-{ENVELOPE_FAR:.0f} range this scanner "
                f"was characterised over - {profile.below_envelope} nearer, "
                f"{profile.above_envelope} further"
            ),
            advice=(
                "Move the object into the working range, or find what is sitting outside it. "
                "Distances this far out were never measured on this unit."
            ),
        )

    if static is not None and static.found:
        columns = sorted({cell[0] for cell in static.cells})
        rows = sorted({cell[1] for cell in static.cells})
        where = (
            f"columns {columns[0]}-{columns[-1]} of {STATIC_COLUMNS}, "
            f"rows {rows[0]}-{rows[-1]} of {STATIC_ROWS}"
        )
        distance = "" if static.distance is None else f" at about {static.distance:.0f}"
        return Plausibility(
            verdict=STATIC_REGION,
            detail=(
                f"{len(static.cells)} of {static.compared} regions have not changed depth"
                f"{distance} while {static.moving_fraction:.0%} of the view has - {where}"
            ),
            advice=(
                "Part of the view is not moving with the scene, so it is fixed to the scanner. "
                "Look at that part of the field of view."
            ),
        )

    # The dominant group, not the nearest one. Pairing the TOTAL return count
    # with ``clusters[0]``'s range - which this did until 2026-09-02 - reads as
    # "77764 returns in one group at 1-14" on a frame whose subject is at
    # 161-274 and whose nearest group holds 14 of those returns. That sentence
    # sends an operator to look at the wrong distance, which is the one thing
    # every message in this module exists to get right.
    main = profile.dominant
    if main is None:  # unreachable: `returned > 0` was established above
        return Plausibility(verdict=PLAUSIBLE, detail=f"{profile.returned} returns", advice="")
    others = len(profile.clusters) - 1
    if others == 0:
        detail = f"{profile.returned} returns in one group at {main.near:.0f}-{main.far:.0f}"
    else:
        detail = (
            f"{main.samples} of {profile.returned} returns at "
            f"{main.near:.0f}-{main.far:.0f}, the rest in "
            f"{others} smaller group{'' if others == 1 else 's'}"
        )
    return Plausibility(verdict=PLAUSIBLE, detail=detail, advice="")


LIMITATIONS: Final = (
    "Advisory only. Nothing here accepts or refuses a frame, moves a tracking gate, clips a "
    "sample or changes a reconstruction threshold; the only thing it may do is appear on "
    "screen and in the log.",
    "No cause is ever named. A near cluster can be an object, a cable, a hand, the projector's "
    "overexposed near field or a correlation failure on a dim far surface, and the bytes do "
    "not say which. Every sentence names what to look at instead.",
    "The working envelope is the 1st and 99th percentile of the depth codes in nine early "
    "frames, which describes what those frames contained rather than a measured limit of the "
    "sensor. It is the weakest number in this module and the one most worth replacing with a "
    "swept characterisation.",
    "The static-region test assumes the scene is supposed to be moving. It stays silent unless "
    "most of the view is changing, so it says nothing during framing, during a pause, or on a "
    "capture where the scanner and the object are both still.",
    "Clustering is a fixed-bin histogram walked once, chosen because it runs between a driver "
    "handing a buffer over and getting it back. It can merge two things at similar distances "
    "and it says nothing about where in the frame a cluster sits - the static-region test is "
    "the only part of this module that localises anything.",
    "No physical unit is implied. Every distance is raw code times depth scale.",
)
