"""Align scans taken from different scanner positions, where no shared axis exists.

Why this is a different problem
-------------------------------
:mod:`~scanner3d.registration` aligns a turntable sweep, and it works because
the turntable *constrains* the answer: one fixed axis, a known angular step, and
therefore a starting point already in the right neighbourhood. That constraint
is doing nearly all the work. Measured on a real model truck bed, plain
point-to-point ICP between frames 15-20 degrees apart converges to within about
one degree of the identity and reports ``converged=True`` while being completely
wrong.

Move the scanner - to reach the top of an object, or its underside - and the
constraint is gone. The new frames share no axis with the old ones, and the
transform between them is a general six-degree-of-freedom rotation and
translation that may be very large. Handing that to ICP directly reproduces
exactly the failure above: it slides into a local minimum near wherever it
started and announces success.

So this module does the part ICP cannot: it finds a *starting point*. Refinement
is still ICP's job, and this module hands off to it.

The strategy, and why it is this one
------------------------------------
Align whole fused clouds, not individual frames. A single depth frame of a
featureless panel carries almost no constraint; a fused sweep carries the whole
object. So the intended flow is:

1. register each sweep on its own turntable axis (:mod:`registration`),
2. fuse each sweep to one cloud (:mod:`fusion`),
3. align the fused clouds to each other - this module,
4. fuse the aligned result.

For the coarse step, candidate orientations come from two sources. The principal
axes of each cloud give four candidates (eigenvector signs are arbitrary, and
only the sign combinations with determinant +1 are rotations). Those are cheap
and often right. But principal axes are meaningless when a cloud's eigenvalues
are close together - a roughly cubic or rotationally symmetric object has no
distinguishable principal frame - so the octahedral rotation group is also
applied on top, giving broad coverage without a full SO(3) search.

Every candidate is scored on heavily decimated clouds, because the point of the
coarse stage is ranking, not precision.

What this does NOT do
---------------------
It does not decide whether an alignment is *correct*. It reports the residual
and the overlap it achieved and refuses when the best candidate is not clearly
better than the runner-up, because an alignment that ties with a wrong one is
not evidence of anything. A confident-looking wrong transform is the specific
failure this module exists to prevent, so it would be perverse to produce one.

No physical unit is implied anywhere. Distances are in whatever the depth scale
produced.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import (
    Matrix3,
    Point,
    Rigid,
    centroid,
    dot,
    ensure_rotation,
    spanned_dimensions,
    subtract,
)

SCHEMA_VERSION: Final = 1

#: Fewest points a cloud may have and still be worth aligning. Below this the
#: principal frame is noise and the residual means nothing.
MIN_ALIGN_POINTS: Final = 50

#: Points kept for candidate SCORING. The coarse stage ranks orientations; it
#: does not need precision, and scoring is quadratic in this number.
COARSE_SAMPLE: Final = 400

#: Cell size for the scoring grid, as a fraction of the cloud's extent. Coarse
#: enough that a roughly-right candidate scores well, fine enough to separate it
#: from a wrong one.
SCORE_CELL_FRACTION: Final = 0.04

#: How much better the best candidate must be than the runner-up to be reported
#: as unambiguous. A tie means the geometry does not distinguish them - common
#: for symmetric objects - and saying so is more useful than picking one.
AMBIGUITY_MARGIN: Final = 1.15

#: The origin :func:`coarse_align` gives a candidate that came from its caller's
#: ``priors`` rather than from the two clouds' own geometry.
PRIOR_ORIGIN: Final = "prior"

#: What one source point that found NO neighbour costs the coarse score, in
#: :data:`SCORE_CELL_FRACTION` grid cells.
#:
#: The score is a TRUNCATED mean nearest-neighbour distance: every sampled
#: source point contributes either the distance to its nearest target point or
#: this penalty, whichever is smaller, and the mean is taken over ALL of them
#: rather than over the ones that found a partner.
#:
#: Excluding the misses instead - which this module used to do - measures only
#: the geometry a candidate happened to land on, so a transform that puts nine
#: tenths of the frame off the reference and the remaining tenth tidily on it
#: outscores one that puts the whole frame roughly where it belongs. Measured on
#: this repository's incompatible-minima tracking fixture, that is not a
#: theoretical worry: the right placement and a genuinely different one scored
#: within 1.05x of each other on the matched-only mean while covering the
#: reference 0.47 against 0.34, so the shared stage called a ranking ambiguous
#: that the geometry distinguishes clearly.
#:
#: TRUNCATED rather than charged in full, because a partial view legitimately
#: has points the other view never saw and an unbounded penalty would rank on
#: how much of the frame overlaps at all rather than on how well it fits. A
#: bounded miss cost is the standard robust form: it says a miss is worth no
#: more than the worst match, and no less.
#:
#: The value is derived, not chosen. ``2 * sqrt(3)`` cells is the furthest apart
#: two points can be and still be found by :func:`_score`'s own neighbour
#: search, which scans the 3x3x3 block of cells around a moved point: a partner
#: in a diagonally adjacent cell can sit up to two cells away along each axis.
#: So no matched point can ever cost more than a miss, which is what makes the
#: truncated mean monotone in coverage - adding overlap can only improve a
#: score, and losing it can only worsen one.
SCORE_MISS_CELLS: Final = 2.0 * math.sqrt(3.0)

#: How far apart, in :data:`SCORE_CELL_FRACTION` grid cells, two candidates must
#: place the source before :func:`coarse_align` treats them as different
#: placements rather than one placement spelled twice.
#:
#: Two, because that is the reach of the scoring pass's OWN neighbour search:
#: :func:`_score` looks in the 3x3x3 block of cells around each moved point, so
#: a partner up to two cells away in an axis can be found and two placements
#: closer than that match largely the same points and score largely the same.
#: A margin between two such candidates is arithmetic noise, and reporting it as
#: an ambiguity calls a shape ambiguous for agreeing with itself. Measured on
#: this repository's asymmetric tracking fixture: the caller's prior placed the
#: source 3.40 from the best generated orientation against a 4.88 reach - one
#: placement - and the score margin between the two was 1.01, while the best
#: genuinely different orientation sat 22.5 away and 1.17 behind.
SCORE_NEIGHBOUR_REACH: Final = 2.0

#: Eigenvalue ratio below which a cloud's principal axes are treated as
#: degenerate: two comparable extents leave the frame undetermined about the
#: axis between them.
PRINCIPAL_DEGENERACY_RATIO: Final = 1.25


class MultiviewNotAllowed(ScannerError):
    """An alignment was requested that this module refuses to attempt."""

    status = DoctorStatus.UNSUPPORTED_MODE


class CloudTooSmall(MultiviewNotAllowed):
    """A cloud has too few points for its principal frame to mean anything."""


class CloudDegenerate(MultiviewNotAllowed):
    """A cloud does not span three dimensions, so no orientation is determined."""


@dataclass(frozen=True, slots=True)
class AlignmentCandidate:
    """One coarse orientation and how well it scored. Not an answer on its own."""

    transform: Rigid
    score: float
    matched: int
    origin: str

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "matched": self.matched,
            "origin": self.origin,
            "transform": self.transform.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class CoarseAlignment:
    """The ranked coarse candidates, and whether the best is clearly best."""

    candidates: tuple[AlignmentCandidate, ...]
    #: Whether the best candidate is clearly better than the best GENUINELY
    #: DIFFERENT one, and by how much.
    #:
    #: The runner-up used for this margin is not simply ``candidates[1]``. Two
    #: candidates that place the source within this scoring pass's own search
    #: reach of each other are one placement spelled twice - they match nearly
    #: the same points and therefore score nearly the same - and a margin
    #: between them measures rounding, not the geometry. Reporting such a pair
    #: as an ambiguity would call a shape ambiguous for agreeing with itself.
    #: So the runner-up is the best candidate that actually puts the source
    #: SOMEWHERE ELSE - see :data:`SCORE_NEIGHBOUR_REACH` - and
    #: :attr:`rival_index` says which one it was.
    unambiguous: bool
    margin: float
    source_principal_degenerate: bool
    target_principal_degenerate: bool
    limitations: tuple[str, ...]
    #: Index in :attr:`candidates` of the genuinely different runner-up
    #: :attr:`margin` was measured against, or ``None`` when every other
    #: candidate placed the source in the same place as the best one.
    rival_index: int | None = None
    #: How many candidates ranked between the best and that runner-up, all of
    #: them near-duplicates of the best. Zero on an ordinary ranking.
    duplicates_skipped: int = 0
    #: How many of :attr:`candidates` came from ``priors`` rather than from the
    #: generated orientation set.
    priors_ranked: int = 0

    @property
    def best(self) -> AlignmentCandidate | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidates": [c.as_dict() for c in self.candidates[:5]],
            "considered": len(self.candidates),
            "unambiguous": self.unambiguous,
            "margin": self.margin,
            "source_principal_degenerate": self.source_principal_degenerate,
            "target_principal_degenerate": self.target_principal_degenerate,
            "rival_index": self.rival_index,
            "duplicates_skipped": self.duplicates_skipped,
            "priors_ranked": self.priors_ranked,
            "limitations": list(self.limitations),
        }


def _covariance(points: Sequence[Point], centre: Point) -> list[list[float]]:
    xx = xy = xz = yy = yz = zz = 0.0
    for p in points:
        a, b, c = p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]
        xx += a * a
        xy += a * b
        xz += a * c
        yy += b * b
        yz += b * c
        zz += c * c
    return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]


def _jacobi(matrix: list[list[float]], sweeps: int = 64) -> tuple[list[float], list[list[float]]]:
    """Symmetric eigen-decomposition by cyclic Jacobi rotations.

    The same method :mod:`geometry` uses, for the same reason: unconditionally
    stable on symmetric matrices and needing no external library.
    """
    size = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(sweeps):
        off = math.fsum(a[i][j] ** 2 for i in range(size) for j in range(i + 1, size))
        if off < 1e-30:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if abs(a[p][q]) < 1e-30:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(size):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(size)], v


def principal_frame(points: Sequence[Point]) -> tuple[Matrix3, bool]:
    """The cloud's principal axes as a rotation, and whether they are degenerate.

    Columns are the eigenvectors of the centred covariance, ordered by
    descending eigenvalue. The second return value is ``True`` when two
    eigenvalues are within :data:`PRINCIPAL_DEGENERACY_RATIO` of each other,
    which means the frame is *not* determined by the geometry and any candidate
    derived from it is a guess rather than an estimate.
    """
    if len(points) < MIN_ALIGN_POINTS:
        raise CloudTooSmall(
            f"{len(points)} points is below the {MIN_ALIGN_POINTS} needed for a "
            "principal frame to carry any information"
        )
    if spanned_dimensions(points) < 3:
        raise CloudDegenerate(
            "the cloud does not span three dimensions, so no orientation is determined"
        )
    centre = centroid(points)
    values, vectors = _jacobi(_covariance(points, centre))
    order = sorted(range(3), key=lambda i: values[i], reverse=True)
    ordered = [values[i] for i in order]
    degenerate = any(
        ordered[i + 1] > 0.0 and ordered[i] / ordered[i + 1] < PRINCIPAL_DEGENERACY_RATIO
        for i in range(2)
    )
    columns = [[vectors[r][order[c]] for c in range(3)] for r in range(3)]
    matrix: Matrix3 = (
        (columns[0][0], columns[0][1], columns[0][2]),
        (columns[1][0], columns[1][1], columns[1][2]),
        (columns[2][0], columns[2][1], columns[2][2]),
    )
    # Force a right-handed frame: an eigenvector triple can come back as a
    # reflection, and geometry.Rigid would rightly refuse it.
    if _det(matrix) < 0.0:
        matrix = (
            (-matrix[0][0], matrix[0][1], matrix[0][2]),
            (-matrix[1][0], matrix[1][1], matrix[1][2]),
            (-matrix[2][0], matrix[2][1], matrix[2][2]),
        )
    return ensure_rotation(matrix), degenerate


def _det(m: Matrix3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _transpose(m: Matrix3) -> Matrix3:
    return ((m[0][0], m[1][0], m[2][0]), (m[0][1], m[1][1], m[2][1]), (m[0][2], m[1][2], m[2][2]))


def _multiply(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(  # type: ignore[return-value]
        tuple(math.fsum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def octahedral_rotations() -> tuple[Matrix3, ...]:
    """The 24 rotations mapping a cube onto itself.

    Applied on top of the principal-axis candidates so that a cloud whose
    principal frame is degenerate - a near-cubic or rotationally symmetric
    object, where the axes carry no information - is still covered. Cheaper and
    better conditioned than sampling SO(3) blindly.
    """
    found: list[Matrix3] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            rows = []
            for i in range(3):
                row = [0.0, 0.0, 0.0]
                row[perm[i]] = signs[i]
                rows.append(tuple(row))
            candidate: Matrix3 = (rows[0], rows[1], rows[2])  # type: ignore[assignment]
            if abs(_det(candidate) - 1.0) < 1e-9:
                found.append(candidate)
    return tuple(found)


def _decimate(points: Sequence[Point], target: int) -> tuple[Point, ...]:
    if len(points) <= target:
        return tuple(points)
    step = len(points) / target
    return tuple(points[int(i * step)] for i in range(target))


def _score(
    moved: Sequence[Point], grid: dict[tuple[int, int, int], list[Point]], cell: float
) -> tuple[float, int]:
    """Truncated mean nearest-neighbour distance, and how many points matched.

    Every point in ``moved`` contributes: the distance to its nearest point in
    ``grid`` if it found one, and :data:`SCORE_MISS_CELLS` cells if it did not.
    The mean is over ALL of them, so a candidate that leaves most of the source
    off the target is ranked as what it is rather than judged on the fraction of
    itself it managed to place. See :data:`SCORE_MISS_CELLS` for why the miss
    cost is bounded and why that particular bound.

    The matched COUNT is still returned, because the score alone cannot say
    whether a mediocre number came from a poor fit over the whole frame or a
    good fit over a little of it, and a caller deciding anything more than a
    ranking needs to know which.
    """
    if not moved:
        return math.inf, 0
    miss = SCORE_MISS_CELLS * cell
    total = 0.0
    matched = 0
    for p in moved:
        key = (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))
        best = math.inf
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
                        if d < best:
                            best = d
        if best < math.inf:
            total += min(math.sqrt(best), miss)
            matched += 1
        else:
            total += miss
    if matched == 0:
        return math.inf, 0
    return total / len(moved), matched


def _extent(points: Sequence[Point]) -> float:
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return max(hi[i] - lo[i] for i in range(3))


def _distinct_rival(
    candidates: Sequence[AlignmentCandidate], sample: Sequence[Point], reach: float
) -> int | None:
    """Index of the best candidate that places ``sample`` somewhere else.

    Somewhere else meaning: further, on average over the same decimated sample
    the scores were computed from, than ``reach``. Averaged rather than taken at
    the worst point because the score being defended is itself a mean, and one
    corner of a cloud swinging past the reach while the rest of it stays put is
    not two placements.

    ``None`` when no candidate does, which says the search found one placement
    and several spellings of it.
    """
    if len(candidates) < 2 or not sample:
        return None
    placed = [candidates[0].transform.apply(point) for point in sample]
    for index in range(1, len(candidates)):
        moved = candidates[index].transform.apply_all(sample)
        total = 0.0
        for a, b in zip(placed, moved, strict=True):
            total += math.dist(a, b)
        if total / len(placed) > reach:
            return index
    return None


def _transform_key(transform: Rigid) -> tuple[int, ...]:
    """A hashable identity for one transform, rounded so float noise ties."""
    return tuple(round(v * 1e6) for row in transform.rotation for v in row) + tuple(
        round(v * 1e6) for v in transform.translation
    )


def coarse_align(
    source: Sequence[Point],
    target: Sequence[Point],
    *,
    sample: int = COARSE_SAMPLE,
    include_octahedral: bool = True,
    priors: Sequence[Rigid] = (),
) -> CoarseAlignment:
    """Rank candidate orientations taking ``source`` onto ``target``.

    Returns candidates, best first, and says whether the best is *clearly* best.
    This is a starting point for ICP, not an alignment: the scores come from
    heavily decimated clouds and a coarse grid, which is all that ranking needs.

    ``priors`` are complete transforms a caller already has a reason to believe -
    a pose an earlier frame was accepted at, say - offered to this ONE ranking
    rather than compared against its result afterwards. They are scored by
    exactly the same decimated pass on exactly the same grid as the generated
    orientations and carry the origin ``"prior"``, so a prior that fits the
    target better than every generated orientation wins the ranking on the same
    measurement they lose it on, and :attr:`~CoarseAlignment.margin` is a margin
    over everything the caller offered rather than over a subset of it. A prior
    is not privileged: it earns its rank or it does not, and passing one adds no
    search - it costs one more score of ``sample`` points.
    """
    src_frame, src_degenerate = principal_frame(source)
    dst_frame, dst_degenerate = principal_frame(target)
    src_centre, dst_centre = centroid(source), centroid(target)

    small_src = _decimate(source, sample)
    small_dst = _decimate(target, sample)
    cell = max(_extent(target) * SCORE_CELL_FRACTION, 1e-6)
    grid: dict[tuple[int, int, int], list[Point]] = {}
    for q in small_dst:
        grid.setdefault((int(q[0] // cell), int(q[1] // cell), int(q[2] // cell)), []).append(q)

    # The composition order matters and is easy to get backwards. The sign and
    # axis ambiguity of an eigenvector triple lives in the PRINCIPAL frame, so
    # the extra rotation has to sit between the two frames:
    #
    #     candidate = dst_frame @ extra @ src_frame^T
    #
    # Post-multiplying the already-composed transform instead applies the extra
    # rotation in the source's original coordinates, which is a different and
    # wrong thing. Written the wrong way round it recovers a small turn exactly
    # and fails completely on a flip - measured 140 degrees of error on a
    # 180-degree flip - because a small turn does not need the sign correction
    # and a flip is nothing but one.
    src_t = _transpose(src_frame)
    bases: list[tuple[Matrix3, str]] = [(_multiply(dst_frame, src_t), "principal")]
    if include_octahedral:
        for extra in octahedral_rotations():
            bases.append((_multiply(_multiply(dst_frame, extra), src_t), "principal+octahedral"))

    # Keyed on the whole transform rather than on its rotation. Every generated
    # basis derives its translation from the same two centroids, so for those
    # the two keys are the same test; a PRIOR is a complete transform and can
    # share a rotation with a generated orientation while placing the source
    # somewhere else entirely, and deduplicating that away would silently drop
    # the candidate the caller had a reason to offer.
    seen: set[tuple[int, ...]] = set()
    candidates: list[AlignmentCandidate] = []
    offered: list[tuple[Rigid, str]] = []
    for rotation, origin in bases:
        try:
            spin = Rigid.of(rotation, (0.0, 0.0, 0.0))
        except ScannerError:
            continue
        moved_centre = spin.apply(src_centre)
        offered.append((Rigid.of(rotation, subtract(dst_centre, moved_centre)), origin))
    offered.extend((prior, PRIOR_ORIGIN) for prior in priors)
    for transform, origin in offered:
        key = _transform_key(transform)
        if key in seen:
            continue
        seen.add(key)
        score, matched = _score(tuple(transform.apply_all(small_src)), grid, cell)
        if matched:
            candidates.append(AlignmentCandidate(transform, score, matched, origin))

    candidates.sort(key=lambda c: (c.score, -c.matched))
    rival_index = _distinct_rival(candidates, small_src, cell * SCORE_NEIGHBOUR_REACH)
    duplicates_skipped = 0 if rival_index is None else rival_index - 1
    margin = math.inf
    if rival_index is not None and candidates[0].score > 0.0:
        margin = candidates[rival_index].score / candidates[0].score
    unambiguous = bool(candidates) and margin >= AMBIGUITY_MARGIN

    limitations = [
        "A coarse orientation is a starting point for ICP, not an alignment. The scores "
        "come from decimated clouds on a coarse grid and are for ranking only.",
        "The score is a TRUNCATED mean: a source point that found no neighbour costs "
        f"{SCORE_MISS_CELLS:.2f} grid cells rather than nothing and rather than everything, so "
        "unmatched geometry lowers a candidate's rank without letting a barely-overlapping "
        "placement be scored on the fraction of itself it managed to place. Because the cost is "
        "bounded, a score close to it means very little of the source landed on the target at "
        "all, and the matched count says how little.",
    ]
    if src_degenerate or dst_degenerate:
        limitations.append(
            "At least one cloud has near-equal principal extents, so its principal frame "
            "is not determined by its geometry. The principal candidate is a guess there; "
            "the octahedral set is what is actually doing the work."
        )
    ranked_priors = sum(1 for candidate in candidates if candidate.origin == PRIOR_ORIGIN)
    if ranked_priors:
        limitations.append(
            f"{ranked_priors} of these candidates came from the caller as priors rather than "
            "from the two clouds' geometry. They are scored, ranked and judged exactly as the "
            "generated orientations are; being offered by the caller buys a candidate nothing."
        )
    if duplicates_skipped:
        limitations.append(
            f"{duplicates_skipped} candidate(s) ranked between the best and the runner-up this "
            "margin was measured against, and every one of them placed the source within this "
            "scoring pass's own search reach of the best - the same placement spelled more than "
            "once, not an alternative to it."
        )
    if rival_index is None:
        limitations.append(
            "Every candidate placed the source in the same place as the best one, so there was "
            "no genuinely different placement to measure a margin against. The margin is "
            "infinite because nothing disagreed, which is not the same evidence as one "
            "candidate beating a real alternative."
        )
    if not unambiguous:
        limitations.append(
            f"The best candidate is only {margin:.2f}x better than the best genuinely different "
            f"runner-up, under the "
            f"{AMBIGUITY_MARGIN} margin. The geometry does not clearly distinguish them - "
            "common for symmetric objects - and picking one anyway would manufacture "
            "confidence that is not there."
        )
    return CoarseAlignment(
        candidates=tuple(candidates),
        unambiguous=unambiguous,
        margin=margin,
        source_principal_degenerate=src_degenerate,
        target_principal_degenerate=dst_degenerate,
        limitations=tuple(limitations),
        rival_index=rival_index,
        duplicates_skipped=duplicates_skipped,
        priors_ranked=ranked_priors,
    )


def overlap_fraction(
    source: Sequence[Point], target: Sequence[Point], transform: Rigid, *, tolerance: float
) -> float:
    """Fraction of ``source`` landing within ``tolerance`` of some target point.

    Reported alongside any residual, because a small residual over 5% of the
    cloud says nothing. Overlap is what tells a reader whether the residual is
    describing the object or a corner of it.
    """
    if not source or not target:
        return 0.0
    cell = max(tolerance, 1e-6)
    grid: dict[tuple[int, int, int], list[Point]] = {}
    for q in target:
        grid.setdefault((int(q[0] // cell), int(q[1] // cell), int(q[2] // cell)), []).append(q)
    within = 0
    limit = tolerance * tolerance
    for p in transform.apply_all(source):
        key = (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))
        hit = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if dot(subtract(p, q), subtract(p, q)) <= limit:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        within += hit
    return within / len(source)


@dataclass(frozen=True, slots=True)
class DriftPair:
    """Two frames that see the same surface, and how far apart they place it."""

    first: int
    second: int
    index_gap: int
    residual: float
    overlap: float
    compared: int

    def as_dict(self) -> dict[str, object]:
        return {
            "first": self.first,
            "second": self.second,
            "index_gap": self.index_gap,
            "residual": self.residual,
            "overlap": self.overlap,
            "compared": self.compared,
        }


@dataclass(frozen=True, slots=True)
class ModelDrift:
    """How well a registered model agrees with itself, measured geometrically.

    This replaces loop closure as the quality metric, and the reason is a
    limitation of loop closure rather than a preference. Closure composes every
    frame-to-frame transform and compares the result against the identity, which
    silently assumes the capture covered exactly one revolution. It therefore
    needs the angular step to be known precisely: on real data, three
    independent estimates of one turntable's rate disagreed by 10%, and the
    reported closure swung from 5 to 23 degrees purely on which was assumed. The
    number was describing the assumption, not the scan.

    Worse, closure cannot exist at all for a handheld capture, which is how this
    scanner is mainly meant to be used. There are no revolutions and no step.

    So drift is measured the way a reader would actually check: find frames that
    look at the same surface but are far apart in capture order, and see whether
    the model puts that surface in the same place twice. If registration drifted,
    they disagree. No angles are involved, nothing is assumed about the path the
    camera took, and it works identically for a turntable and a handheld sweep.
    """

    pairs: tuple[DriftPair, ...]
    frames: int
    min_index_gap: int
    tolerance: float
    limitations: tuple[str, ...]

    @property
    def overlapping(self) -> tuple[DriftPair, ...]:
        return tuple(p for p in self.pairs if p.compared > 0)

    @property
    def median_residual(self) -> float | None:
        found = sorted(p.residual for p in self.overlapping)
        return found[len(found) // 2] if found else None

    @property
    def worst(self) -> DriftPair | None:
        found = self.overlapping
        return max(found, key=lambda p: p.residual) if found else None

    def as_dict(self) -> dict[str, object]:
        worst = self.worst
        return {
            "schema_version": SCHEMA_VERSION,
            "frames": self.frames,
            "min_index_gap": self.min_index_gap,
            "tolerance": self.tolerance,
            "pairs_examined": len(self.pairs),
            "pairs_overlapping": len(self.overlapping),
            "median_residual": self.median_residual,
            "worst": worst.as_dict() if worst else None,
            "limitations": list(self.limitations),
        }


def model_drift(
    clouds: Sequence[Sequence[Point]],
    transforms: Sequence[Rigid],
    *,
    tolerance: float,
    min_index_gap: int = 4,
    sample: int = COARSE_SAMPLE,
) -> ModelDrift:
    """Measure a registered model's self-agreement, without reference to angles.

    Pairs of frames separated by at least ``min_index_gap`` in capture order are
    compared where they overlap in space. Adjacent frames are excluded on
    purpose: consecutive frames were aligned *to each other*, so their agreement
    is what registration optimised and says nothing about accumulated error.
    Distant pairs were never compared during registration, so their agreement is
    independent evidence.

    ``tolerance`` sets what counts as the same surface. It should be a few times
    the expected point spacing: too tight and nothing matches, too loose and
    everything does.

    A pair with ``compared == 0`` simply never saw common ground and carries no
    information - it is reported, not silently dropped, so a reader can see how
    much of the model was actually checked.
    """
    if len(clouds) != len(transforms):
        raise MultiviewNotAllowed(
            f"{len(clouds)} clouds against {len(transforms)} transforms: one transform per cloud"
        )
    if len(clouds) < 2:
        raise MultiviewNotAllowed("drift needs at least two frames to compare")
    if not tolerance > 0.0:
        raise MultiviewNotAllowed(f"tolerance must be positive, got {tolerance}")

    placed = [
        _decimate(tuple(t.apply_all(c)), sample) for c, t in zip(clouds, transforms, strict=True)
    ]
    cell = tolerance
    grids: list[dict[tuple[int, int, int], list[Point]]] = []
    for pts in placed:
        grid: dict[tuple[int, int, int], list[Point]] = {}
        for q in pts:
            grid.setdefault((int(q[0] // cell), int(q[1] // cell), int(q[2] // cell)), []).append(q)
        grids.append(grid)

    limit = tolerance * tolerance
    pairs: list[DriftPair] = []
    for i in range(len(placed)):
        for j in range(i + min_index_gap, len(placed)):
            total = 0.0
            matched = 0
            for p in placed[i]:
                key = (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))
                best = math.inf
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            for q in grids[j].get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                                d = dot(subtract(p, q), subtract(p, q))
                                if d < best:
                                    best = d
                if best <= limit:
                    total += math.sqrt(best)
                    matched += 1
            residual = (total / matched) if matched else math.inf
            pairs.append(
                DriftPair(
                    first=i,
                    second=j,
                    index_gap=j - i,
                    residual=residual,
                    overlap=matched / len(placed[i]) if placed[i] else 0.0,
                    compared=matched,
                )
            )

    limitations = [
        "Drift is agreement between frames that were never aligned to each other. It is "
        "evidence of consistency, not of correctness: a model can be self-consistent and "
        "still be the wrong shape if the calibration is wrong.",
        "Only pairs that overlap in space carry information. The count of overlapping "
        "pairs says how much of the model was actually checked.",
        "The residual is bounded above by the tolerance by construction, since points "
        "further apart than that are not treated as the same surface. A residual near the "
        "tolerance means the pair barely agreed, not that it agreed to that precision.",
    ]
    return ModelDrift(
        pairs=tuple(pairs),
        frames=len(clouds),
        min_index_gap=min_index_gap,
        tolerance=tolerance,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True, slots=True)
class ModelRefinement:
    """Poses refined against the accumulated model rather than against neighbours.

    Why this exists
    ---------------
    :func:`~scanner3d.registration.register_turntable` CHAINS: frame 2 is
    aligned to frame 1, frame 3 to frame 2, and each pairwise error compounds
    down the chain. More frames therefore means a longer chain and *more*
    accumulated error, which is the opposite of what anyone reasonably expects
    from adding data. Measured on a real sweep, a second lap took the pose error
    from 1.07 to 1.99 degrees, and self-agreement between frames degraded
    steadily with how far apart in capture order they were.

    Refining against the model breaks the chain. Every frame is aligned to
    everything else seen, so one frame's error is corrected by the consensus of
    the rest instead of being handed to its successor. More frames become more
    constraint, which is the behaviour a user is entitled to expect.

    What it does NOT do
    -------------------
    This is not bundle adjustment. Poses are refined one at a time against a
    model built from the current estimate, not solved jointly, so it can settle
    into a consistent-but-wrong configuration if the starting point is badly
    wrong. It improves a reasonable estimate; it does not rescue a broken one.

    It also cannot make a self-consistent model *correct*. Agreement is not
    accuracy, and a systematic calibration error survives this untouched.

    What the measurements actually show
    -----------------------------------
    Eight runs across two synthetic scenes with known ground truth, refining
    both a poor starting chain (unconstrained 6-DOF chaining) and a good one
    (chaining already pinned to the turntable axis):

    * From a POOR chain it helped every time - 1.849 to 1.374 degrees on one
      scene, 0.317 to 0.255 on the other.
    * From a GOOD chain it was inconsistent: 1.375 to 1.219 degrees on one
      scene, but 0.170 to 0.184 - worse - on the other.
    * ``residual_improved`` was True in ALL EIGHT runs, including the one where
      the true pose error got worse.

    That last point is the one that matters and it is why this class reports a
    residual rather than a quality score. The residual improves whether or not
    the scan does, so it cannot be used to decide whether to keep the result. An
    earlier version of this docstring claimed refinement roughly doubles pose
    error; that came from a comparison that computed the two rotations in
    opposite senses, and it was wrong.
    """

    transforms: tuple[Rigid, ...]
    residual_history: tuple[float, ...]
    iterations_run: int
    converged: bool
    residual_improved: bool
    anchor: int
    limitations: tuple[str, ...]

    @property
    def initial_residual(self) -> float | None:
        return self.residual_history[0] if self.residual_history else None

    @property
    def final_residual(self) -> float | None:
        return self.residual_history[-1] if self.residual_history else None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frames": len(self.transforms),
            "iterations_run": self.iterations_run,
            "converged": self.converged,
            "residual_improved": self.residual_improved,
            "anchor": self.anchor,
            "initial_residual": self.initial_residual,
            "final_residual": self.final_residual,
            "residual_history": list(self.residual_history),
            "limitations": list(self.limitations),
        }


def _model_index(
    clouds: Sequence[Sequence[Point]],
    transforms: Sequence[Rigid],
    *,
    skip: int | None,
    cell: float,
) -> dict[tuple[int, int, int], list[Point]]:
    """Spatial index of every frame placed in world coordinates.

    ``skip`` excludes one frame, and excluding the frame being aligned matters:
    a model containing it would partly match the frame against itself and pull
    the answer toward the pose it already has, which is the error being
    corrected.
    """
    grid: dict[tuple[int, int, int], list[Point]] = {}
    for index, (cloud, transform) in enumerate(zip(clouds, transforms, strict=True)):
        if index == skip:
            continue
        for p in transform.apply_all(cloud):
            grid.setdefault((int(p[0] // cell), int(p[1] // cell), int(p[2] // cell)), []).append(p)
    return grid


def _nearest_in(
    grid: dict[tuple[int, int, int], list[Point]], p: Point, cell: float, limit: float
) -> Point | None:
    key = (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))
    best: Point | None = None
    best_d = limit * limit
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for q in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                    d = dot(subtract(p, q), subtract(p, q))
                    if d < best_d:
                        best_d, best = d, q
    return best


def _model_residual(
    clouds: Sequence[Sequence[Point]],
    samples: Sequence[Sequence[Point]],
    poses: Sequence[Rigid],
    tolerance: float,
) -> float:
    """Mean distance from each sampled point to the model built from every OTHER frame.

    Each frame is excluded from the model it is measured against, so this is
    genuine cross-frame agreement rather than points matching themselves.
    """
    total, count = 0.0, 0
    for index, pts in enumerate(samples):
        grid = _model_index(clouds, poses, skip=index, cell=tolerance)
        for p in poses[index].apply_all(pts):
            q = _nearest_in(grid, p, tolerance, tolerance)
            if q is not None:
                total += math.sqrt(dot(subtract(p, q), subtract(p, q)))
                count += 1
    return total / count if count else math.inf


def refine_against_model(
    clouds: Sequence[Sequence[Point]],
    transforms: Sequence[Rigid],
    *,
    tolerance: float,
    iterations: int = 4,
    anchor: int = 0,
    sample: int = 300,
    min_improvement: float = 0.01,
) -> ModelRefinement:
    """Re-align every frame to the model, repeatedly, instead of to its neighbour.

    Returns refined poses and the residual after each pass. If refinement made
    things WORSE, the original transforms are returned unchanged and
    ``improved`` is False. An optimiser that can quietly degrade a scan is worse
    than none, so the outcome is measured and recorded rather than assumed.

    ``anchor`` names the frame held fixed. Without it the whole model is free to
    drift bodily through space, because aligning everything to everything else
    constrains relative poses but pins no origin.
    """
    from scanner3d.geometry import rigid_from_correspondences

    if len(clouds) != len(transforms):
        raise MultiviewNotAllowed(
            f"{len(clouds)} clouds against {len(transforms)} transforms: one each"
        )
    if len(clouds) < 3:
        raise MultiviewNotAllowed("refining against a model needs at least three frames")
    if not 0 <= anchor < len(clouds):
        raise MultiviewNotAllowed(f"anchor {anchor} is not a frame index")
    if iterations < 1:
        raise MultiviewNotAllowed(f"iterations must be at least 1, got {iterations}")
    if not tolerance > 0.0:
        raise MultiviewNotAllowed(f"tolerance must be positive, got {tolerance}")

    current = list(transforms)
    samples = [_decimate(tuple(c), sample) for c in clouds]

    history = [_model_residual(clouds, samples, current, tolerance)]
    converged = False
    for _ in range(iterations):
        for index, pts in enumerate(samples):
            grid = _model_index(clouds, current, skip=index, cell=tolerance)
            placed = tuple(current[index].apply_all(pts))
            src: list[Point] = []
            dst: list[Point] = []
            for p in placed:
                q = _nearest_in(grid, p, tolerance, tolerance)
                if q is not None:
                    src.append(p)
                    dst.append(q)
            if len(src) < 3 or spanned_dimensions(src) < 2:
                continue
            try:
                correction = rigid_from_correspondences(src, dst)
            except ScannerError:
                continue
            current[index] = correction.compose(current[index])
        # Re-anchor, so the model cannot wander bodily while agreeing with itself.
        drifted = current[anchor].compose(transforms[anchor].inverse())
        back = drifted.inverse()
        current = [back.compose(t) for t in current]
        history.append(_model_residual(clouds, samples, current, tolerance))
        if history[-2] - history[-1] < min_improvement * history[-2]:
            converged = True
            break

    residual_improved = history[-1] < history[0]
    limitations = [
        "Not bundle adjustment: poses are refined one at a time against a model built "
        "from the current estimate, not solved jointly. It improves a reasonable starting "
        "point and cannot rescue a badly wrong one.",
        "MEASURED WARNING: a falling residual does NOT mean a better scan. Across eight runs "
        "on two synthetic scenes with known ground truth, `residual_improved` was True EVERY "
        "time, while the true pose error improved in six and got worse in two. Refining a poor "
        "starting chain helped consistently (1.849 -> 1.374 degrees, and 0.317 -> 0.255); "
        "refining an already-good chain was a coin toss (1.375 -> 1.219, but 0.170 -> 0.184). "
        "Free six-degree-of-freedom refinement against a self-consistency objective is "
        "under-constrained, so poses can agree with each other better while collectively "
        "drifting from the truth. `residual_improved` names what it measures and is not a "
        "claim about accuracy; it cannot distinguish the six good cases from the two bad ones.",
        "Self-consistency is not accuracy. A systematic calibration error is untouched by "
        "this and survives into a model that agrees with itself perfectly.",
        "The frame being aligned is excluded from the model it aligns to, so no frame "
        "holds itself in place - but every other frame's current error is still part of "
        "what it aligns against.",
    ]
    if not residual_improved:
        limitations.append(
            f"Refinement did NOT improve the residual ({history[0]:.4f} -> {history[-1]:.4f}), "
            "so the original transforms were returned unchanged."
        )
    return ModelRefinement(
        transforms=tuple(current if residual_improved else transforms),
        residual_history=tuple(history),
        iterations_run=len(history) - 1,
        converged=converged,
        residual_improved=residual_improved,
        anchor=anchor,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True, slots=True)
class TurntableRefinement:
    """Poses refined against the model with the turntable's constraint enforced.

    Why this exists
    ---------------
    :func:`refine_against_model` gives every frame six free parameters and lets
    it settle wherever the residual is lowest. Measured against known truth on a
    synthetic sweep, that made the residual fall 40% while the true pose error
    DOUBLED. The optimiser was not lying about the residual; the residual was
    simply not the thing anyone cares about, and six degrees of freedom per
    frame is enough rope for a scan to agree with itself in the wrong place.

    On a turntable the object has exactly one degree of freedom: the angle it
    has turned through. Fixing the axis and refining one angle per frame keeps
    the correction inside the family of motions the hardware can actually
    perform. The wrong answers that unconstrained refinement found are not
    merely disfavoured, they are unreachable - no combination of per-frame
    angles about a fixed axis can express them.

    That is the whole idea: not a better optimiser, a smaller search space.

    What it does NOT do
    -------------------
    It cannot fix an object that moved on the turntable, a mis-estimated axis,
    or a handheld sweep. Every one of those violates the constraint that makes
    this safe, and the constraint then works against the answer instead of for
    it. Handheld capture needs a full bundle adjustment, which is a different
    and larger job.

    Like any self-agreement method it also cannot make a consistent model
    *correct*. A systematic depth-calibration error survives untouched.
    """

    transforms: tuple[Rigid, ...]
    corrections: tuple[float, ...]
    residual_history: tuple[float, ...]
    iterations_run: int
    converged: bool
    residual_improved: bool
    anchor: int
    limitations: tuple[str, ...]

    @property
    def initial_residual(self) -> float | None:
        return self.residual_history[0] if self.residual_history else None

    @property
    def final_residual(self) -> float | None:
        return self.residual_history[-1] if self.residual_history else None

    @property
    def worst_correction(self) -> float:
        """Largest angle any single frame was moved, in radians.

        A large value is a warning rather than a success: the constraint keeps
        corrections physically possible, but a frame that had to move a long way
        was badly placed to begin with, and the starting estimate is suspect.
        """
        return max((abs(c) for c in self.corrections), default=0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frames": len(self.transforms),
            "iterations_run": self.iterations_run,
            "converged": self.converged,
            "residual_improved": self.residual_improved,
            "anchor": self.anchor,
            "initial_residual": self.initial_residual,
            "final_residual": self.final_residual,
            "residual_history": list(self.residual_history),
            "corrections_deg": [math.degrees(c) for c in self.corrections],
            "worst_correction_deg": math.degrees(self.worst_correction),
            "limitations": list(self.limitations),
        }


def refine_on_turntable(
    clouds: Sequence[Sequence[Point]],
    transforms: Sequence[Rigid],
    *,
    direction: Point,
    through: Point,
    tolerance: float,
    iterations: int = 4,
    anchor: int = 0,
    sample: int = 300,
    min_improvement: float = 0.01,
) -> TurntableRefinement:
    """Refine each frame's turntable angle against the model of every other frame.

    Unlike :func:`refine_against_model` this cannot translate a frame, tilt it,
    or move it off the turntable's circle. The only thing it may change is how
    far that frame has rotated about ``direction``, which is the only thing the
    turntable ever changed. Each correction is solved in closed form by
    :func:`~scanner3d.registration.optimal_angle_about_axis` rather than
    searched.

    ``anchor`` names the frame held at exactly its input pose. Angles about a
    shared axis are only meaningful relative to something, and without an anchor
    the whole model may rotate bodily while every frame agrees with its
    neighbours.

    Returns the refined poses and the per-frame angle corrections. As with the
    unconstrained version, if the residual got worse the input transforms are
    returned unchanged.
    """
    from scanner3d.geometry import rotation_about_axis
    from scanner3d.registration import optimal_angle_about_axis

    if len(clouds) != len(transforms):
        raise MultiviewNotAllowed(
            f"{len(clouds)} clouds against {len(transforms)} transforms: one each"
        )
    if len(clouds) < 3:
        raise MultiviewNotAllowed("refining on a turntable needs at least three frames")
    if not 0 <= anchor < len(clouds):
        raise MultiviewNotAllowed(f"anchor {anchor} is not a frame index")
    if iterations < 1:
        raise MultiviewNotAllowed(f"iterations must be at least 1, got {iterations}")
    if not tolerance > 0.0:
        raise MultiviewNotAllowed(f"tolerance must be positive, got {tolerance}")
    if not math.isfinite(dot(direction, direction)) or dot(direction, direction) <= 0.0:
        raise MultiviewNotAllowed("axis direction must be a finite non-zero vector")

    samples = [_decimate(tuple(c), sample) for c in clouds]
    deltas = [0.0] * len(clouds)

    def placed(index: int, angles: Sequence[float]) -> Rigid:
        """The pose of ``index`` given the accumulated angle corrections.

        Rebuilt from the ORIGINAL transform every time rather than accumulated
        in place, so repeated passes cannot let floating-point drift leak a
        seventh degree of freedom back in.
        """
        return rotation_about_axis(direction, angles[index], through=through).compose(
            transforms[index]
        )

    def poses_for(angles: Sequence[float]) -> list[Rigid]:
        return [placed(i, angles) for i in range(len(clouds))]

    history = [_model_residual(clouds, samples, poses_for(deltas), tolerance)]
    converged = False
    for _ in range(iterations):
        for index, pts in enumerate(samples):
            current = poses_for(deltas)
            grid = _model_index(clouds, current, skip=index, cell=tolerance)
            src: list[Point] = []
            dst: list[Point] = []
            for p in current[index].apply_all(pts):
                q = _nearest_in(grid, p, tolerance, tolerance)
                if q is not None:
                    src.append(p)
                    dst.append(q)
            if len(src) < 3 or spanned_dimensions(src) < 2:
                continue
            try:
                step = optimal_angle_about_axis(src, dst, direction=direction, through=through)
            except ScannerError:
                continue
            if math.isfinite(step):
                deltas[index] += step
        # Re-anchor. Subtracting one angle from all of them is itself a rotation
        # about the same axis, so the constraint survives the re-anchoring.
        shift = deltas[anchor]
        deltas = [d - shift for d in deltas]
        history.append(_model_residual(clouds, samples, poses_for(deltas), tolerance))
        if history[-2] - history[-1] < min_improvement * history[-2]:
            converged = True
            break

    residual_improved = history[-1] < history[0]
    limitations = [
        "Assumes the object only rotated about the given axis. An object that shifted on "
        "the turntable, a mis-estimated axis, or a handheld sweep all break that assumption, "
        "and the constraint then holds the answer away from the truth instead of near it.",
        "Self-consistency is not accuracy. A systematic depth-calibration error produces a "
        "model that agrees with itself perfectly and is uniformly wrong.",
        "Poses are refined one at a time against a model built from the current estimate, "
        "not solved jointly. This is a constrained coordinate descent, not bundle adjustment.",
    ]
    if not residual_improved:
        limitations.append(
            f"Refinement did NOT improve the residual ({history[0]:.4f} -> {history[-1]:.4f}), "
            "so the original transforms were returned unchanged."
        )
    return TurntableRefinement(
        transforms=tuple(poses_for(deltas) if residual_improved else transforms),
        corrections=tuple(deltas if residual_improved else [0.0] * len(clouds)),
        residual_history=tuple(history),
        iterations_run=len(history) - 1,
        converged=converged,
        residual_improved=residual_improved,
        anchor=anchor,
        limitations=tuple(limitations),
    )
