"""Registration of a turntable sweep as one axis and one rate.

Every frame of a turntable sweep is the same rigid object turned about ONE
fixed axis, by an angle that grows with time. That is five unknowns for the
whole sweep - the axis direction (two), where it stands (two) and the rate -
where the frame-by-frame pipeline in :mod:`scanner3d.pipeline` has six per
frame and nothing but the frames' overlap to tie them together. On a subject
with one large plane and little else, that chain wanders off the circle by
millimetres and closure cannot pull it back (``docs/target-gate.md``, the
2026-09-04 evening entry); here it cannot wander, because there is no chain.

Each unknown is taken from the measurement that determines it best, and taken
DIRECTLY, because a residual minimisation cannot see two of them:

* **the axis direction** from the rotation axes of direct alignments between
  frames about a sixth of a turn apart. A 2.8-degree step carries its whole
  rotation error into the axis direction; a 60-degree rotation divides it by
  twenty, and frames 60 degrees apart still overlap by two thirds. Wider pairs
  get worse again as the overlap shrinks.
* **the centre** from the same alignments, linearly: a rotation R about a point
  c carries the translation t = (I - R) c, which is well conditioned at 60
  degrees and hopeless at 2.8.
* **the rate** from frames one revolution apart. They see the same image, so
  whatever rotation is left between them after the model's own prediction is
  the model's rate error over a whole turn, and the true period is fractional
  (126.7 frames, not 127).

Two rounds, each seeded from the previous model, converge to a hundredth of a
degree. Seeding comes from a short chain of adjacent alignments over about a
quarter turn: its camera positions lie on an arc whose circle gives the axis
to half a degree and the centre to a few millimetres, which is all the direct
estimators need to find their correspondences.

Angles come from the frame index when a capture's frames are evenly spaced,
and from the frames' delivery times when they are not - the desktop
application's held spacing changes the frame interval mid-capture, and a
frame index is then no measure of angle at all.

The result is judged against the CAD target through ``tools/target_gate.py``,
not by the image-period coherence check: frames one period apart agree by
construction here, so that check only confirms the rate. On the 2026-09-04
CAD capture this registration scored 0.51 mm RMS, 23 of 24 surfaces and
orthogonality within 0.73 degrees, against the pipeline's 0.63 mm and 1.45
degrees; the white model the pipeline smeared by 13-22 mm registers to 0.10 mm
between neighbours.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.capture_sequence import frame_times
from scanner3d.coherence import PeriodEstimate, camera_orbit, image_period
from scanner3d.compute import AUTO_BACKEND, ComputeBackend, select_backend
from scanner3d.errors import ScannerError
from scanner3d.geometry import (
    Matrix3,
    Point,
    Rigid,
    add,
    cross,
    dot,
    norm,
    normalize,
    scaled,
    subtract,
)
from scanner3d.icp_plane import IcpPlaneGeometryUnusable, icp_point_to_plane
from scanner3d.pipeline import SweepReconstruction

#: Points per frame for the seed chain and the agreement report. Enough to
#: align neighbouring frames well; the direct estimators use the full clouds.
SEED_POINTS: Final = 700

#: The pairs the axis and centre are taken from are this many degrees apart.
PAIR_DEGREES: Final = 60.0

#: Every this-many-th frame starts a pair. Two gives ~85 pairs on a
#: 190-frame sweep, which averages the pair axes' half-degree scatter down
#: to well under a tenth.
PAIR_STRIDE: Final = 2

#: Correspondence reach, in millimetres, for the seed chain, the pairs and the
#: period pairs. The seed and the pairs start from a model good to a few
#: millimetres; the period pairs from one good to a fraction of one.
CHAIN_REACH_MM: Final = 8.0
PAIR_REACH_MM: Final = 8.0
PERIOD_REACH_MM: Final = 6.0

#: The share of pairs, farthest in axis direction from the mean, left out of
#: the axis and centre averages.
TRIM_FRACTION: Final = 0.2

#: Rounds of direct estimation: as many as it takes for the centre to settle
#: within CONVERGED_MM, up to this many. A seed 13 mm off settles in three.
MAX_ROUNDS: Final = 8
CONVERGED_MM: Final = 0.5

#: Neighbours used for the normals of the agreement report.
NORMAL_NEIGHBOURS: Final = 10

#: Period pairs sampled for the rate, spread over the sweep.
PERIOD_SAMPLES: Final = 12

#: How close, in frame intervals, the nearest frame must be to exactly one
#: revolution after another for the two to count as a period pair.
PERIOD_TOLERANCE: Final = 1.0

#: Fewest usable pairs a round may rest on.
MIN_PAIRS: Final = 8

#: Gate, in millimetres, for the agreement report's nearest-neighbour search.
AGREEMENT_GATE_MM: Final = 6.0

#: Frames a seed chain aligns: about a quarter turn, never fewer than this.
MIN_CHAIN: Final = 12

Progress = Callable[[str], None] | None

__all__ = [
    "TurntableNotDetected",
    "TurntableRegistration",
    "TurntableUnusable",
    "frame_angles",
    "frame_times",
    "register_turntable",
]


class TurntableNotDetected(ScannerError):
    """The frames do not repeat: this is not a full turn of a turntable."""


class TurntableUnusable(ScannerError):
    """The frames repeat but the model could not be fitted to them."""


# --- small dense linear algebra, on tuples -----------------------------------------


def rotation_about(axis: Point, degrees: float) -> Matrix3:
    """The rotation matrix turning by ``degrees`` about the unit vector ``axis``."""
    x, y, z = normalize(axis)
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    k = 1.0 - c
    return (
        (c + x * x * k, x * y * k - z * s, x * z * k + y * s),
        (y * x * k + z * s, c + y * y * k, y * z * k - x * s),
        (z * x * k - y * s, z * y * k + x * s, c + z * z * k),
    )


def axis_angle(matrix: Matrix3) -> tuple[Point, float]:
    """The unit axis and the angle, in degrees within [0, 180], of a rotation matrix.

    The axis of a rotation by (almost) nothing is not defined; it is returned
    as +z with the angle it has, and callers weight by the angle.
    """
    v = (
        matrix[2][1] - matrix[1][2],
        matrix[0][2] - matrix[2][0],
        matrix[1][0] - matrix[0][1],
    )
    sine = norm(v) / 2.0
    cosine = (matrix[0][0] + matrix[1][1] + matrix[2][2] - 1.0) / 2.0
    angle = math.degrees(math.atan2(sine, cosine))
    if sine < 1e-12:
        return (0.0, 0.0, 1.0), angle
    return scaled(v, 1.0 / (2.0 * sine)), angle


def _apply(matrix: Matrix3, point: Point) -> Point:
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def _basis(axis: Point) -> tuple[Point, Point]:
    """Two unit vectors perpendicular to ``axis`` and to each other."""
    helper: Point = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = normalize(cross(axis, helper))
    return e1, cross(axis, e1)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of nothing")
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _unwrap(angles: Sequence[float]) -> list[float]:
    """Angles in degrees made continuous, so consecutive differences are steps."""
    out = [angles[0]] if angles else []
    for angle in angles[1:]:
        step = angle - out[-1]
        while step > 180.0:
            step -= 360.0
        while step < -180.0:
            step += 360.0
        out.append(out[-1] + step)
    return out


# --- the model ------------------------------------------------------------------------


@dataclass(frozen=True)
class TurntableModel:
    """One axis, one rate: every frame's pose follows from its angle.

    ``rate`` is the OBJECT's turn in degrees per unit of ``tau``, and ``tau``
    is each frame's position in frame intervals - the frame index for an
    evenly spaced capture, the delivery time divided by the median interval
    otherwise. Frame k's points p map into frame 0's coordinates as R p + t
    with R the rotation by ``-rate * tau[k]`` about the axis through
    ``centre``: what the object did is undone, so every frame lands on the
    same object.
    """

    axis: Point
    centre: Point
    rate: float
    tau: tuple[float, ...]

    def pose(self, k: int) -> Rigid:
        rotation = rotation_about(self.axis, -self.rate * self.tau[k])
        translation = subtract(self.centre, _apply(rotation, self.centre))
        return Rigid.of(rotation, translation)

    def relative(self, k: int, j: int) -> Rigid:
        """Frame j's points into frame k's coordinates."""
        return self.pose(k).inverse().compose(self.pose(j))


# --- the result -------------------------------------------------------------------


@dataclass(frozen=True)
class LagAgreement:
    """How well frames ``lag`` intervals apart land on each other under the model."""

    lag: int
    overlap: float
    median_mm: float | None

    def as_dict(self) -> dict[str, object]:
        return {"lag": self.lag, "overlap": self.overlap, "median_mm": self.median_mm}


@dataclass(frozen=True)
class TurntableRegistration:
    """A registered turntable sweep and the evidence it rests on."""

    poses: tuple[Rigid, ...]
    axis: Point
    centre: Point
    rate_degrees: float
    period_frames: int
    times_used: bool
    pairs_used: int
    axis_scatter_degrees: float
    centre_shift_mm: float
    period_leftover_degrees: float | None
    agreement: tuple[LagAgreement, ...]
    backend: str
    #: Whether the last round moved the centre by less than CONVERGED_MM. A
    #: model returned because the round budget ran out is still a model, and
    #: is still exported; it is not called settled, because it was measured
    #: not to be, and a capped fit once read as a 3 mm undersize.
    converged: bool = True
    rounds: int = 0

    @property
    def frames(self) -> int:
        return len(self.poses)

    @property
    def frames_per_turn(self) -> float:
        return 360.0 / abs(self.rate_degrees)

    def statement(self) -> str:
        """One sentence a report can carry: what was fitted, and how well it holds."""
        by_lag = {a.lag: a for a in self.agreement}
        first = by_lag.get(1)
        neighbours = (
            f"neighbouring frames land {first.median_mm:.2f} mm apart"
            if first is not None and first.median_mm is not None
            else "neighbouring frames were not compared"
        )
        leftover = (
            f"{self.period_leftover_degrees:+.2f} degrees of rotation left over one revolution"
            if self.period_leftover_degrees is not None
            else "no frames one revolution apart"
        )
        source = "delivery times" if self.times_used else "frame order"
        settled = (
            f"settled in {self.rounds} round(s)"
            if self.converged
            else (
                f"NOT converged: the centre was still moving {self.centre_shift_mm:.2f} mm "
                f"when the {self.rounds}-round budget ran out"
            )
        )
        return (
            f"turntable model: one axis, {self.frames_per_turn:.2f} frames per turn from "
            f"{source}, from {self.pairs_used} pairs {PAIR_DEGREES:.0f} degrees apart "
            f"(axis scatter {self.axis_scatter_degrees:.2f} degrees, centre moved "
            f"{self.centre_shift_mm:.2f} mm in the last round, {settled}); {neighbours}, "
            f"{leftover}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "registration": "turntable",
            "frames": self.frames,
            "axis": list(self.axis),
            "centre_mm": list(self.centre),
            "rate_degrees_per_interval": self.rate_degrees,
            "frames_per_turn": self.frames_per_turn,
            "period_frames": self.period_frames,
            "angles_from": "delivery times" if self.times_used else "frame order",
            "pairs_used": self.pairs_used,
            "axis_scatter_degrees": self.axis_scatter_degrees,
            "centre_shift_mm": self.centre_shift_mm,
            "converged": self.converged,
            "rounds": self.rounds,
            "period_leftover_degrees": self.period_leftover_degrees,
            "agreement": [a.as_dict() for a in self.agreement],
            "backend": self.backend,
        }

    def as_reconstruction(self) -> SweepReconstruction:
        """The shape the export and its record take: the poses, and what is NOT claimed.

        No closure score, because there is no chain to close; the model's own
        evidence goes into the limitations, where the record's reader will
        find it beside the frame count.
        """
        return SweepReconstruction(
            poses=self.poses,
            frames=self.frames,
            revisits_found=self.pairs_used,
            chained_score=None,
            optimised_score=None,
            optimised=False,
            converged=self.converged,
            limitations=(self.statement(),),
        )


# --- timing ---------------------------------------------------------------------------


def frame_angles(times: Sequence[float] | None, frames: int) -> tuple[float, ...]:
    """Each frame's position in frame intervals: the index, or the time in median intervals.

    Times whose spacing never varies by more than a tenth of an interval are
    treated as the index they amount to, so that an evenly spaced capture
    registers identically whether or not its times are known.
    """
    if frames <= 0:
        return ()
    if times is None or len(times) != frames or frames < 3:
        return tuple(float(k) for k in range(frames))
    gaps = [times[k + 1] - times[k] for k in range(frames - 1)]
    spacing = _median(gaps)
    if spacing <= 0.0:
        return tuple(float(k) for k in range(frames))
    tau = tuple((t - times[0]) / spacing for t in times)
    if all(abs(tau[k] - k) < 0.1 for k in range(frames)):
        return tuple(float(k) for k in range(frames))
    return tau


def detect_period(raws: Sequence[bytes], *, depth_scale: float) -> PeriodEstimate | None:
    """The image period of a capture, or None when its frames never repeat."""
    return image_period(raws, depth_scale=depth_scale)


# --- the fit ----------------------------------------------------------------------


def _thin(cloud: Sequence[Point], count: int) -> list[Point]:
    """Every n-th point, so that two runs on one capture see the same points."""
    if len(cloud) <= count:
        return list(cloud)
    stride = len(cloud) / count
    return [cloud[int(i * stride)] for i in range(count)]


def _align(
    source: Sequence[Point],
    target: Sequence[Point],
    *,
    initial: Rigid | None,
    reach: float,
    backend: str | ComputeBackend,
) -> Rigid | None:
    """Source onto target by point-to-plane ICP, or None when the pair cannot be fitted."""
    try:
        if initial is None:
            fit = icp_point_to_plane(
                source, target, max_correspondence_distance=reach, backend=backend
            )
        else:
            fit = icp_point_to_plane(
                source, target, initial=initial, max_correspondence_distance=reach, backend=backend
            )
    except IcpPlaneGeometryUnusable:
        return None
    return fit.transform


def _seed(
    sub: Sequence[Sequence[Point]],
    tau: Sequence[float],
    period: int,
    backend: str | ComputeBackend,
    report: Callable[[str], None],
) -> TurntableModel:
    """A first model from a chain of adjacent alignments over about a quarter turn."""
    frames = len(sub)
    length = min(frames - 1, max(MIN_CHAIN, period // 4 + 2))
    chain = [Rigid.identity()]
    for k in range(length):
        step = _align(sub[k + 1], sub[k], initial=None, reach=CHAIN_REACH_MM, backend=backend)
        if step is None:
            raise TurntableUnusable(
                f"frames {k} and {k + 1} could not be aligned to seed the model; a turntable "
                "sweep needs neighbouring frames that overlap on more than one flat panel"
            )
        chain.append(chain[-1].compose(step))
    orbit = camera_orbit(chain)
    if orbit is None:
        raise TurntableUnusable(
            f"the camera positions of the first {length + 1} frames do not lie on a circle; "
            "the seed chain could not find the turntable's axis"
        )
    axis = normalize(orbit.axis)
    centre = orbit.centre
    e1, e2 = _basis(axis)
    angles = []
    for pose in chain:
        offset = subtract(pose.translation, centre)
        angles.append(math.degrees(math.atan2(dot(offset, e2), dot(offset, e1))))
    unwrapped = _unwrap(angles)
    steps = [
        (unwrapped[k + 1] - unwrapped[k]) / max(tau[k + 1] - tau[k], 1e-9) for k in range(length)
    ]
    camera_step = _median(steps)
    if abs(camera_step) < 0.05:
        raise TurntableUnusable("the seed chain shows no rotation between frames")
    # The camera moves about the axis opposite to the object; the model's rate
    # is the object's.
    rate = -camera_step
    report(
        f"  seeded from a chain of {length} adjacent alignments: orbit radius "
        f"{orbit.radius:.0f} mm, {abs(rate):.3f} degrees per frame interval"
    )
    return TurntableModel(axis=axis, centre=centre, rate=rate, tau=tuple(tau))


@dataclass(frozen=True)
class _Round:
    model: TurntableModel
    pairs: int
    scatter: float
    centre_shift: float
    leftover: float | None


def _pairs_round(
    full: Sequence[Sequence[Point]],
    model: TurntableModel,
    backend: str | ComputeBackend,
    *,
    first: bool,
) -> tuple[TurntableModel, int, float, float]:
    """Axis and centre from pairs about PAIR_DEGREES apart; in the first round the rate too.

    The pairs' own rotation angles give the rate to a few tenths of a percent,
    which is what the first round needs to get the period pairs within reach
    of their prediction; after that the period pairs know it far better, and
    the model keeps theirs.
    """
    frames = len(full)
    tau = model.tau
    lag = max(2, round(PAIR_DEGREES / abs(model.rate)))
    rows: list[tuple[int, Rigid, Point, float]] = []
    for k in range(0, frames - lag, PAIR_STRIDE):
        fit = _align(
            full[k + lag],
            full[k],
            initial=model.relative(k, k + lag),
            reach=PAIR_REACH_MM,
            backend=backend,
        )
        if fit is None:
            continue
        axis, angle = axis_angle(fit.rotation)
        if angle < 1.0:
            continue
        if dot(axis, model.axis) < 0.0:
            axis = scaled(axis, -1.0)
        rows.append((k, fit, axis, angle))
    if len(rows) < MIN_PAIRS:
        raise TurntableUnusable(
            f"only {len(rows)} pairs of frames {lag} intervals apart could be aligned; "
            f"the model needs at least {MIN_PAIRS}"
        )

    def weighted_mean(selected: Sequence[tuple[int, Rigid, Point, float]]) -> Point:
        total: Point = (0.0, 0.0, 0.0)
        for _, _, axis, angle in selected:
            total = add(total, scaled(axis, angle))
        return normalize(total)

    mean = weighted_mean(rows)
    deviations = [
        math.degrees(math.acos(max(-1.0, min(1.0, dot(axis, mean))))) for _, _, axis, _ in rows
    ]
    cut = sorted(deviations)[max(0, math.ceil(len(rows) * (1.0 - TRIM_FRACTION)) - 1)]
    kept = [row for row, dev in zip(rows, deviations, strict=True) if dev <= cut]
    axis = weighted_mean(kept)
    scatter = _median(
        [math.degrees(math.acos(max(-1.0, min(1.0, dot(a, axis))))) for _, _, a, _ in kept]
    )
    # The centre: (I - R) c = t for every kept pair, solved for the two
    # components of c across the axis. Normal equations, 2 by 2.
    e1, e2 = _basis(axis)
    ata = [[0.0, 0.0], [0.0, 0.0]]
    atb = [0.0, 0.0]
    for _, fit, _, _ in kept:
        rotation = fit.rotation
        columns = []
        for basis_vector in (e1, e2):
            turned = _apply(rotation, basis_vector)
            columns.append(subtract(basis_vector, turned))  # (I - R) e
        old_turned = _apply(rotation, model.centre)
        residual = subtract(
            fit.translation, subtract(model.centre, old_turned)
        )  # t - (I - R) c_old
        for i in range(2):
            for j in range(2):
                ata[i][j] += dot(columns[i], columns[j])
            atb[i] += dot(columns[i], residual)
    det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
    if abs(det) < 1e-9:
        raise TurntableUnusable("the pairs leave the axis position undetermined")
    u = (atb[0] * ata[1][1] - ata[0][1] * atb[1]) / det
    v = (ata[0][0] * atb[1] - ata[1][0] * atb[0]) / det
    centre = add(model.centre, add(scaled(e1, u), scaled(e2, v)))
    rate = model.rate
    if first:
        per_interval = [angle / max(tau[k + lag] - tau[k], 1e-9) for k, _, _, angle in kept]
        rate = math.copysign(_median(per_interval), model.rate)
    return (
        TurntableModel(axis=axis, centre=centre, rate=rate, tau=tau),
        len(kept),
        scatter,
        math.hypot(u, v),
    )


def _period_pairs(tau: Sequence[float], rate: float) -> list[tuple[int, int]]:
    """Frames one revolution apart by angle, from the frame times, spread over the sweep."""
    frames = len(tau)
    turn = 360.0 / abs(rate)
    pairs = []
    candidates = [k for k in range(frames) if tau[k] + turn <= tau[-1] + PERIOD_TOLERANCE]
    if not candidates:
        return []
    stride = max(1, len(candidates) // PERIOD_SAMPLES)
    for k in candidates[::stride]:
        wanted = tau[k] + turn
        j = min(range(k + 1, frames), key=lambda i: abs(tau[i] - wanted), default=None)
        if j is not None and abs(tau[j] - wanted) <= PERIOD_TOLERANCE:
            pairs.append((k, j))
    return pairs


def _rate_round(
    full: Sequence[Sequence[Point]],
    model: TurntableModel,
    backend: str | ComputeBackend,
) -> tuple[TurntableModel, float | None]:
    """The rate from frames one revolution apart: the rotation the model leaves between them.

    The model brings frame j back onto frame k by turning it through
    ``-rate * span``; aligning the two directly from that prediction finds the
    discrepancy D = T_fit T_pred^-1, a small rotation about the axis. With the
    signed angle of D taken about the axis, D = R(axis, (rate - true) * span),
    so the true rate is ``rate - leftover / span``. The sign follows from the
    model's own conventions and was checked on the CAD capture, where the
    corrected rate lands frames one revolution apart on each other to a
    tenth of a millimetre and the other sign doubles their disagreement.
    """
    pairs = _period_pairs(model.tau, model.rate)
    leftovers: list[tuple[float, float]] = []
    for k, j in pairs:
        predicted = model.relative(k, j)
        fit = _align(full[j], full[k], initial=predicted, reach=PERIOD_REACH_MM, backend=backend)
        if fit is None:
            continue
        discrepancy = fit.compose(predicted.inverse())
        axis, angle = axis_angle(discrepancy.rotation)
        leftovers.append((math.copysign(angle, dot(axis, model.axis)), model.tau[j] - model.tau[k]))
    if not leftovers:
        return model, None
    leftover = _median([left for left, _ in leftovers])
    span = _median([s for _, s in leftovers])
    rate = model.rate - leftover / span
    return TurntableModel(model.axis, model.centre, rate, model.tau), leftover


def _normals(compute: ComputeBackend, cloud: Sequence[Point]) -> list[Point]:
    """A unit normal per point from the plane through its nearest neighbours.

    The smallest-eigenvector of each neighbourhood's covariance, by Jacobi
    rotations on a 3 by 3 symmetric matrix: a dozen sweeps for a dozen digits,
    and no library. Orientation is not resolved; the agreement report takes
    absolute distances.
    """
    if len(cloud) <= NORMAL_NEIGHBOURS:
        return [(0.0, 0.0, 1.0) for _ in cloud]
    neighbours = compute.k_nearest_neighbors(cloud, NORMAL_NEIGHBOURS)
    out: list[Point] = []
    for index, around in enumerate(neighbours):
        points = [cloud[index]] + [cloud[i] for i in around]
        n = len(points)
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        cz = sum(p[2] for p in points) / n
        m = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for p in points:
            d = (p[0] - cx, p[1] - cy, p[2] - cz)
            for r in range(3):
                for c in range(3):
                    m[r][c] += d[r] * d[c]
        out.append(_smallest_eigenvector(m))
    return out


def _smallest_eigenvector(m: list[list[float]]) -> Point:
    """The unit eigenvector of a symmetric 3x3 matrix with the smallest eigenvalue."""
    v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    a = [row[:] for row in m]
    for _ in range(12):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off < 1e-15:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-18:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            sign = 1.0 if theta >= 0.0 else -1.0
            tan = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
            cos = 1.0 / math.sqrt(tan * tan + 1.0)
            sin = tan * cos
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = cos * akp - sin * akq
                a[k][q] = sin * akp + cos * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k] = cos * apk - sin * aqk
                a[q][k] = sin * apk + cos * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = cos * vkp - sin * vkq
                v[k][q] = sin * vkp + cos * vkq
    smallest = min(range(3), key=lambda i: a[i][i])
    return normalize((v[0][smallest], v[1][smallest], v[2][smallest]))


def _plane_agreement(
    compute: ComputeBackend,
    queries: Sequence[Point],
    target: Sequence[Point],
    target_normals: Sequence[Point],
) -> tuple[float, float | None]:
    """Share of queries with a neighbour inside the gate, and their median plane distance."""
    if not queries:
        return 0.0, None
    hits = compute.nearest_neighbors(queries, target, gate=AGREEMENT_GATE_MM)
    # the backend hands back the neighbour's coordinates; its normal is looked
    # up by position, so the target must be the list the normals were made on
    where = {point: i for i, point in enumerate(target)}
    distances = []
    for query, hit in zip(queries, hits, strict=True):
        if hit is None:
            continue
        i = where.get(hit[1])
        if i is None:
            continue
        distances.append(abs(dot(subtract(query, hit[1]), target_normals[i])))
    overlap = len(distances) / len(queries)
    return overlap, (_median(distances) if distances else None)


def _agreement(
    full: Sequence[Sequence[Point]],
    sub: Sequence[Sequence[Point]],
    model: TurntableModel,
    compute: ComputeBackend,
) -> tuple[LagAgreement, ...]:
    """Point-to-plane agreement at a few lags: the model's own report card.

    Point-to-plane rather than point-to-point, because at the clouds' sampling
    density the nearest POINT on a perfectly registered surface is a
    millimetre away, and the distance to the surface itself is what the
    registration answers for.
    """
    frames = len(full)
    quarter = max(2, round(90.0 / abs(model.rate)))
    pair = max(2, round(PAIR_DEGREES / abs(model.rate)))
    lag_pairs: list[tuple[int, list[tuple[int, int]]]] = []
    for lag in sorted({1, pair, quarter}):
        if lag < frames:
            step = max(1, (frames - lag) // 12)
            lag_pairs.append((lag, [(k, k + lag) for k in range(0, frames - lag, step)]))
    period = _period_pairs(model.tau, model.rate)
    if period:
        lag_pairs.append((round(360.0 / abs(model.rate)), period))
    normals: dict[int, list[Point]] = {}
    worlds: dict[int, list[Point]] = {}

    def world(j: int) -> tuple[list[Point], list[Point]]:
        if j not in worlds:
            worlds[j] = list(model.pose(j).apply_all(full[j]))
            normals[j] = _normals(compute, worlds[j])
        return worlds[j], normals[j]

    out = []
    for lag, pairs in lag_pairs:
        overlaps = []
        medians = []
        for k, j in pairs:
            queries = list(model.pose(k).apply_all(sub[k]))
            target, target_normals = world(j)
            overlap, median = _plane_agreement(compute, queries, target, target_normals)
            overlaps.append(overlap)
            if median is not None:
                medians.append(median)
        out.append(
            LagAgreement(
                lag=lag,
                overlap=sum(overlaps) / len(overlaps) if overlaps else 0.0,
                median_mm=_median(medians) if medians else None,
            )
        )
    return tuple(out)


def register_turntable(
    clouds: Sequence[Sequence[Point]],
    *,
    raws: Sequence[bytes] | None = None,
    depth_scale: float | None = None,
    period: int | None = None,
    times: Sequence[float] | None = None,
    backend: str | ComputeBackend = AUTO_BACKEND,
    on_progress: Progress = None,
) -> TurntableRegistration:
    """Register a turntable sweep: one axis, one rate, every frame's pose from its angle.

    ``clouds`` are the prepared registration clouds, one per frame, in the
    camera's frame. The image period is measured from ``raws`` (with
    ``depth_scale``) unless ``period`` is given; a capture whose frames never
    repeat raises :class:`TurntableNotDetected`, and a caller that cannot
    prove a turntable falls back to :func:`scanner3d.pipeline.reconstruct_sweep`.
    ``times`` are the frames' delivery times in seconds (see
    :func:`frame_times`); without them the frames are taken as evenly spaced.

    Raises :class:`TurntableUnusable` when the frames repeat but too few pairs
    align, or the seed chain finds no circle: the frames are then not a rigid
    object on a turntable, or overlap on one flat panel only.
    """

    def report(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    frames = len(clouds)
    if period is None:
        if raws is None or depth_scale is None:
            raise TurntableNotDetected("no frames to measure the image period on")
        estimate = detect_period(raws, depth_scale=depth_scale)
        if estimate is None:
            raise TurntableNotDetected(
                f"the {frames} frames do not repeat: less than one turn of a turntable, or "
                "not a turntable at all"
            )
        period = estimate.period
    if frames < MIN_CHAIN + 2:
        raise TurntableUnusable(f"a turntable sweep needs more than {MIN_CHAIN + 1} frames")
    selection = select_backend(backend if isinstance(backend, str) else backend.name)
    compute = backend if not isinstance(backend, str) else selection.backend
    tau = frame_angles(times, frames)
    times_used = any(abs(tau[k] - k) >= 0.1 for k in range(frames))
    report(
        f"turntable: image period {period} frames, angles from "
        f"{'delivery times' if times_used else 'frame order'}, {selection.resolved} backend"
    )
    full = [list(cloud) for cloud in clouds]
    sub = [_thin(cloud, SEED_POINTS) for cloud in full]
    model = _seed(sub, tau, period, backend, report)
    pairs = 0
    scatter = 0.0
    shift = 0.0
    leftover: float | None = None
    converged = False
    rounds = 0
    for round_number in range(1, MAX_ROUNDS + 1):
        rounds = round_number
        model, pairs, scatter, shift = _pairs_round(full, model, backend, first=round_number == 1)
        model, leftover = _rate_round(full, model, backend)
        report(
            f"  round {round_number}: {pairs} pairs {PAIR_DEGREES:.0f} degrees apart, axis "
            f"scatter {scatter:.2f} degrees, centre moved {shift:.2f} mm, "
            + (
                f"{leftover:+.3f} degrees left over one revolution, "
                if leftover is not None
                else "no frames one revolution apart, "
            )
            + f"{360.0 / abs(model.rate):.2f} frames per turn"
        )
        if shift < CONVERGED_MM:
            converged = True
            break
    if not converged:
        report(
            f"  not converged: the centre still moved {shift:.2f} mm in round {rounds} "
            f"(limit {CONVERGED_MM} mm); the model is returned and says so"
        )
    agreement = _agreement(full, sub, model, compute)
    for entry in agreement:
        report(
            f"  frames {entry.lag} apart: overlap {entry.overlap:.2f}, median "
            + (f"{entry.median_mm:.2f} mm" if entry.median_mm is not None else "n/a")
        )
    return TurntableRegistration(
        poses=tuple(model.pose(k) for k in range(frames)),
        axis=model.axis,
        centre=model.centre,
        rate_degrees=model.rate,
        period_frames=period,
        times_used=times_used,
        pairs_used=pairs,
        axis_scatter_degrees=scatter,
        centre_shift_mm=shift,
        period_leftover_degrees=leftover,
        agreement=agreement,
        backend=selection.resolved,
        converged=converged,
        rounds=rounds,
    )
