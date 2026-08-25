"""Find frame pairs where the camera returned to a viewpoint it already occupied.

Why this exists
----------------
:mod:`scanner3d.posegraph` fixes a drifted trajectory by redistributing the
discrepancy loop-closure evidence reveals - but it cannot manufacture that
evidence itself. Without closure edges its own graph is a tree, and
:mod:`~scanner3d.posegraph`'s own honest negative case documents what a tree
does: nothing pins the drift, so there is nothing to fix. This module is
where the evidence comes from. It is the piece that, on the real capture this
project keeps citing, actually found the 65-degree error: chained ICP said
one lap of rotation had happened, and a revisit pair - the raw clouds of two
frames the camera happened to see the same surfaces from - needed only 1.4
degrees to align. That gap was the error, discovered with no reference to
angles, axes, or how many frames made up a lap.

No knowledge of capture geometry
---------------------------------
This module knows nothing about turntables, angular steps, revolution counts
or capture paths, and it must not learn any of that: the user has explicitly
ruled out turntable-specific functionality, because it would mean maintaining
two capture methods where the software should see only one. A turntable
sweep and someone slowly walking around a stationary object must look
identical to this module. Detection asks exactly one question of a pair of
frames - "have I seen this view before?" - and answers it from the frames'
own content: their depth values and, for confirmation, their raw clouds.
Nothing else about how the sequence was captured is ever consulted.

The two-stage design
----------------------
Comparing every pair of a few hundred frames with ICP is hopeless: ICP is
the expensive, high-precision tool, and most pairs are not revisits at all.
So detection is cheap-then-expensive, in two stages that trust each other for
different reasons:

1. **Proposal.** Each frame gets a crude signature - a normalised histogram
   of its returned depths, see :func:`depth_signature_from_points` - and
   candidate pairs are ranked by how close their signatures are (L1
   distance, see :func:`signature_distance`). Two frames looking at the same
   surfaces from the same viewpoint see roughly the same depths in roughly
   the same proportions; two frames looking at different parts of a scene,
   or the same part from a different angle, generally do not. This is
   deliberately crude: a false positive here costs one wasted ICP call, not
   a wrong answer, so there is no reason to spend more on it.
2. **Verification.** :func:`scanner3d.registration.icp` is run between the
   two RAW clouds, seeded from the identity - genuinely no informed guess is
   available between two frames whose relative pose is exactly the unknown
   being asked about. A real revisit needs almost no motion to align: on the
   real capture that motivated this module, verified revisits came back at
   0.05 to 1.4 degrees of rotation, nowhere near the default thresholds. That
   wide margin, not a marginal one, is what makes verification trustworthy.

Verified pairs become :class:`Revisit` records, and
:meth:`RevisitReport.pose_edges` turns them directly into
:class:`scanner3d.posegraph.PoseEdge` objects using EXACTLY the same
convention :mod:`scanner3d.posegraph` already documents for a sequential
edge: the transform ICP found relating frame ``i``'s local cloud to frame
``j``'s. A closure edge from here and a sequential edge from
:mod:`scanner3d.registration`'s own chaining are the same struct, and
:func:`~scanner3d.posegraph.optimise_poses` does not know or care which is
which - which is the entire point of building this on top of it.

Raw depth bytes or point clouds - the honest primary
--------------------------------------------------------
A per-frame signature needs only ONE number per returned sample: how far away
it is along the camera's optical axis. That number is already sitting in a
deprojected point's z-coordinate (:mod:`scanner3d.depth_metric` deprojects
with ``z = sample * depth_scale``, unchanged), so
:func:`depth_signature_from_points` reads it straight off any point cloud
given in the frame's own LOCAL (camera) coordinates - the same clouds
:func:`detect_revisits` needs anyway for Stage 2, and the same vocabulary
every other module in this pipeline already speaks. That is the PRIMARY
interface, and :func:`detect_revisits` uses it exclusively: a caller who has
clouds - from this scanner or, capture-agnostically, from anything else that
produces per-frame local point clouds - needs nothing more.

:func:`depth_signature_from_raw` exists alongside it as a genuine
optimisation, not a second primary: it reads the same z-values directly out
of this scanner's raw ``Y16 `` bytes, skipping the x and y a full deprojection
would also compute and that a signature never uses. It is for a caller who
wants to rank a large raw sequence before spending anything on deprojecting
the losers - :func:`detect_revisits` itself is not that caller, because it
needs full clouds for Stage 2 regardless and there is nothing to defer.

What this does NOT do
----------------------
It does not decide the trajectory is wrong, and it does not fix anything -
it hands :mod:`scanner3d.posegraph` evidence and stops. It cannot find a
revisit that never happened: a sweep with genuinely no repeated viewpoint -
a single arc, a straight pass along an object - correctly reports zero
verified revisits, and that silence is the intended, honest answer, not a
failure of the search. A false closure edge is worse than none - it would
drag :func:`~scanner3d.posegraph.optimise_poses` toward a confidently wrong
compromise - so this module is built to prefer reporting nothing over
reporting a wrong pair.

The Stage 2 thresholds (:data:`DEFAULT_MAX_ROTATION`,
:data:`DEFAULT_MAX_TRANSLATION`, :data:`DEFAULT_MAX_RMSE`,
:data:`DEFAULT_MIN_CORRESPONDENCES`) came from ONE real capture on this
scanner. They are wide margins on that data, not universal constants, and
every one of them is a parameter for exactly that reason - a caller with a
larger object, a noisier sensor, or a different overlap regime should not
have to accept numbers tuned to a truck bed.

No physical unit is implied anywhere, matching every other module here:
depth, rotation and translation are all in whatever the input clouds' own
coordinates measure.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.depth_metric import (
    NO_RETURN_SAMPLE,
    SAMPLES_FORMAT,
    ensure_frame_length,
    ensure_scale_usable,
)
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.posegraph import PoseEdge
from scanner3d.registration import Cloud, RegistrationNotAllowed, icp

SCHEMA_VERSION: Final = 1

#: Histogram bins for a depth signature. Verified on real data - see the
#: module docstring. Coarse on purpose: Stage 1 only has to rank pairs, not
#: measure them, and a false positive here is cheap.
DEFAULT_BINS: Final = 48

#: Fewest frames apart a pair may be proposed. Adjacent frames were aligned
#: to each other by chaining and trivially "match" - they carry no closure
#: information, and proposing them would waste every Stage 2 call on pairs
#: that tell the pose graph nothing it does not already have.
DEFAULT_MIN_INDEX_GAP: Final = 3

#: Stage 1 candidates handed to Stage 2, closest signature first. Bounds the
#: number of ICP calls, which is where all the real cost is.
DEFAULT_MAX_CANDIDATES: Final = 40

#: Verification thresholds, from one real capture - see the module
#: docstring's note on why these are parameters and not constants.
DEFAULT_MAX_ROTATION: Final = math.radians(12.0)
DEFAULT_MAX_TRANSLATION: Final = 20.0
DEFAULT_MAX_RMSE: Final = 1.6
DEFAULT_MIN_CORRESPONDENCES: Final = 400


class RevisitNotAllowed(ScannerError):
    """An input was offered that this module refuses to search or verify."""

    status = DoctorStatus.UNSUPPORTED_MODE


class SignatureUnusable(RevisitNotAllowed):
    """A depth signature could not be built, or two signatures cannot be compared."""


# --- depth signatures --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DepthSignature:
    """A normalised histogram of one frame's returned depths, and nothing else.

    Deliberately crude - see the module docstring. ``histogram`` sums to 1.0
    over the samples that were actually binned (``returned``); it is empty
    (all zeros) when nothing was.
    """

    histogram: tuple[float, ...]
    bins: int
    max_depth: float
    returned: int
    excluded: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "bins": self.bins,
            "max_depth": self.max_depth,
            "returned": self.returned,
            "excluded": self.excluded,
            "histogram": list(self.histogram),
        }


def _bin_depths(depths: Iterable[float], *, bins: int, max_depth: float) -> DepthSignature:
    """Shared binning core for both signature constructors below."""
    counts = [0] * bins
    returned = 0
    excluded = 0
    width = max_depth / bins
    for depth in depths:
        if not (0.0 < depth <= max_depth) or not math.isfinite(depth):
            excluded += 1
            continue
        index = min(bins - 1, int(depth / width))
        counts[index] += 1
        returned += 1
    histogram = (
        tuple(count / returned for count in counts) if returned else tuple(0.0 for _ in counts)
    )
    return DepthSignature(
        histogram=histogram, bins=bins, max_depth=max_depth, returned=returned, excluded=excluded
    )


def depth_signature_from_points(
    points: Sequence[Point], *, bins: int = DEFAULT_BINS, max_depth: float
) -> DepthSignature:
    """The signature of a cloud, read from each point's z-coordinate.

    REQUIRES ``points`` to be in the frame's own LOCAL (camera) coordinates,
    not a world/registered pose - z is only "depth along the optical axis"
    in the frame it was captured in. Points already carried into a common
    frame by registration would make every signature describe the SAME
    coordinate system's z-axis rather than each frame's own view, which is
    not what a viewpoint signature means. This is the primary interface -
    see the module docstring for why - and exactly what
    :func:`detect_revisits` uses internally.
    """
    if bins < 1:
        raise SignatureUnusable(f"bins must be at least 1, got {bins}")
    if not math.isfinite(max_depth) or max_depth <= 0.0:
        raise SignatureUnusable(f"max_depth must be finite and positive, got {max_depth!r}")
    return _bin_depths((p[2] for p in points), bins=bins, max_depth=max_depth)


def depth_signature_from_raw(
    raw: bytes, *, depth_scale: float, bins: int = DEFAULT_BINS, max_depth: float
) -> DepthSignature:
    """The signature of this scanner's raw ``Y16 `` bytes, without deprojecting.

    Reads exactly the samples :func:`scanner3d.depth_metric.deproject_depth_frame`
    would turn into ``z = sample * depth_scale`` - reusing that module's own
    frame-length check and scale validation so a frame this refuses is a
    frame the real deprojection would refuse too - but computes only depth,
    never x or y, which a signature has no use for. See the module docstring
    on when this earns its keep over the point-cloud path.
    """
    if bins < 1:
        raise SignatureUnusable(f"bins must be at least 1, got {bins}")
    if not math.isfinite(max_depth) or max_depth <= 0.0:
        raise SignatureUnusable(f"max_depth must be finite and positive, got {max_depth!r}")
    ensure_frame_length(raw)
    scale = ensure_scale_usable(depth_scale)
    try:
        samples: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, raw)
    except struct.error as exc:
        raise SignatureUnusable(
            f"raw frame could not be unpacked as {SAMPLES_FORMAT!r}: {exc}"
        ) from exc
    depths = (sample * scale for sample in samples if sample != NO_RETURN_SAMPLE)
    return _bin_depths(depths, bins=bins, max_depth=max_depth)


def signature_distance(a: DepthSignature, b: DepthSignature) -> float:
    """L1 distance between two normalised histograms - symmetric by construction.

    ``sum(abs(x - y))`` is unchanged by swapping ``a`` and ``b``, so this
    module never has to decide an order for a pair, and neither does a
    caller.
    """
    if a.bins != b.bins:
        raise SignatureUnusable(
            f"signatures have {a.bins} and {b.bins} bins: only signatures built with the same "
            "bin count are comparable"
        )
    if a.max_depth != b.max_depth:
        raise SignatureUnusable(
            f"signatures were built over max_depth {a.max_depth!r} and {b.max_depth!r}: their "
            "bins do not describe the same depth ranges and are not comparable"
        )
    return math.fsum(abs(x - y) for x, y in zip(a.histogram, b.histogram, strict=True))


# --- candidates and verified revisits -------------------------------------------------


@dataclass(frozen=True, slots=True)
class RevisitCandidate:
    """One pair Stage 1 proposed, ranked by how alike their signatures are."""

    i: int
    j: int
    signature_distance: float

    def as_dict(self) -> dict[str, object]:
        return {"i": self.i, "j": self.j, "signature_distance": self.signature_distance}


@dataclass(frozen=True, slots=True)
class Revisit:
    """One pair Stage 2 confirmed: what ICP found relating frame ``i`` to frame ``j``.

    ``transform`` uses the exact convention
    :class:`scanner3d.posegraph.PoseEdge` documents for an edge: it carries a
    point from frame ``i``'s local coordinates to frame ``j``'s. See
    :meth:`as_pose_edge`.
    """

    i: int
    j: int
    transform: Rigid
    rmse: float
    correspondences: int
    rotation_angle: float
    translation_magnitude: float

    def as_pose_edge(self, *, weight: float = 1.0) -> PoseEdge:
        """This revisit, ready for :func:`scanner3d.posegraph.optimise_poses`.

        No index arithmetic for the caller: ``i``, ``j`` and ``transform``
        carry over unchanged, because a revisit edge and a sequential edge
        are already the same convention.
        """
        return PoseEdge(self.i, self.j, self.transform, weight)

    def as_dict(self) -> dict[str, object]:
        return {
            "i": self.i,
            "j": self.j,
            "transform": self.transform.as_dict(),
            "rmse": self.rmse,
            "correspondences": self.correspondences,
            "rotation_angle_radians": self.rotation_angle,
            "translation_magnitude": self.translation_magnitude,
        }


@dataclass(frozen=True, slots=True)
class RevisitReport:
    """Every pair Stage 1 proposed, and which of them Stage 2 confirmed."""

    frames: int
    candidates: tuple[RevisitCandidate, ...]
    verified: tuple[Revisit, ...]
    min_index_gap: int
    max_candidates: int
    limitations: tuple[str, ...]

    @property
    def proposed(self) -> int:
        return len(self.candidates)

    @property
    def verification_rate(self) -> float:
        """Share of PROPOSED pairs that verified - zero when nothing was proposed.

        Not a share of all possible pairs: most possible pairs are never
        proposed at all, by design, and folding them into this rate would
        make an ordinary, well-behaved sweep with few genuine revisits look
        no different from one where verification is failing.
        """
        return len(self.verified) / self.proposed if self.proposed else 0.0

    def pose_edges(self, *, weight: float = 1.0) -> tuple[PoseEdge, ...]:
        """Every verified revisit as a :class:`~scanner3d.posegraph.PoseEdge`.

        The seam to :func:`scanner3d.posegraph.optimise_poses`: pass these
        alongside whatever sequential edges chaining produced, with no index
        translation in between.
        """
        return tuple(revisit.as_pose_edge(weight=weight) for revisit in self.verified)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frames": self.frames,
            "proposed": self.proposed,
            "verified": len(self.verified),
            "verification_rate": self.verification_rate,
            "min_index_gap": self.min_index_gap,
            "max_candidates": self.max_candidates,
            "candidates": [c.as_dict() for c in self.candidates],
            "revisits": [r.as_dict() for r in self.verified],
            "limitations": list(self.limitations),
        }


# --- detection ----------------------------------------------------------------------


def _ensure_cloud(points: Cloud, *, what: str) -> tuple[Point, ...]:
    """Normalise one frame's cloud to validated ``(x, y, z)`` tuples, or refuse it.

    The same finite-coordinate checks :mod:`scanner3d.registration` makes on
    its own (private) input, duplicated rather than reached into - see this
    module's siblings (:mod:`scanner3d.posegraph`, :mod:`scanner3d.icp_plane`)
    for the same established choice and the same reason.
    """
    collected: list[Point] = []
    for index, point in enumerate(points):
        values = tuple(point)
        if len(values) != 3:
            raise RevisitNotAllowed(
                f"{what}: point {index} carries {len(values)} coordinates, not the 3 an "
                "(x, y, z) triple has"
            )
        x, y, z = values
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise RevisitNotAllowed(f"{what}: point {index} is {values!r}, which is not finite")
        collected.append((float(x), float(y), float(z)))
    return tuple(collected)


def _ensure_parameters(
    *,
    frames: int,
    min_index_gap: int,
    max_candidates: int,
    bins: int,
    max_rotation: float,
    max_translation: float,
    max_rmse: float,
    min_correspondences: int,
) -> None:
    if frames < 2:
        raise RevisitNotAllowed(f"{frames} frame(s) is not a sequence to search for revisits in")
    if min_index_gap < 1:
        raise RevisitNotAllowed(f"min_index_gap must be at least 1, got {min_index_gap}")
    if min_index_gap >= frames:
        raise RevisitNotAllowed(
            f"min_index_gap {min_index_gap} is not smaller than the {frames}-frame sequence: no "
            "pair could ever satisfy it"
        )
    if max_candidates < 1:
        raise RevisitNotAllowed(f"max_candidates must be at least 1, got {max_candidates}")
    if bins < 1:
        raise RevisitNotAllowed(f"bins must be at least 1, got {bins}")
    if not math.isfinite(max_rotation) or max_rotation <= 0.0:
        raise RevisitNotAllowed(f"max_rotation must be finite and positive, got {max_rotation!r}")
    if not math.isfinite(max_translation) or max_translation <= 0.0:
        raise RevisitNotAllowed(
            f"max_translation must be finite and positive, got {max_translation!r}"
        )
    if not math.isfinite(max_rmse) or max_rmse <= 0.0:
        raise RevisitNotAllowed(f"max_rmse must be finite and positive, got {max_rmse!r}")
    if min_correspondences < 1:
        raise RevisitNotAllowed(
            f"min_correspondences must be at least 1, got {min_correspondences}"
        )


def _default_max_depth(clouds: Sequence[tuple[Point, ...]]) -> float:
    """The largest positive z seen anywhere, so no physical unit is assumed.

    A fixed absolute default would silently assume a unit this repository
    has never established for any cloud's coordinates - see the module
    docstring. Deriving the range from the data instead costs nothing: every
    frame is scanned once regardless, to build its own signature.
    """
    largest = 0.0
    for cloud in clouds:
        for point in cloud:
            z = point[2]
            if math.isfinite(z) and z > largest:
                largest = z
    if largest <= 0.0:
        raise RevisitNotAllowed(
            "no point in any cloud has a positive z-coordinate; there is no depth range to "
            "build a signature over. Either the clouds are empty of returns or they are not in "
            "each frame's own local (camera) coordinates - see depth_signature_from_points"
        )
    return largest


def detect_revisits(
    clouds: Sequence[Cloud],
    *,
    min_index_gap: int = DEFAULT_MIN_INDEX_GAP,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    bins: int = DEFAULT_BINS,
    max_depth: float | None = None,
    max_rotation: float = DEFAULT_MAX_ROTATION,
    max_translation: float = DEFAULT_MAX_TRANSLATION,
    max_rmse: float = DEFAULT_MAX_RMSE,
    min_correspondences: int = DEFAULT_MIN_CORRESPONDENCES,
) -> RevisitReport:
    """Find pairs in ``clouds`` where the camera saw the same viewpoint twice.

    ``clouds[k]`` must be frame ``k``'s own points in its own LOCAL (camera)
    coordinates - the raw, unregistered cloud a single depth frame
    deprojects to, exactly what :func:`scanner3d.registration.icp` and
    :func:`depth_signature_from_points` both need and exactly what a
    sequential chaining step already has before it is composed into a pose.
    Nothing about capture order, path or geometry is read beyond the index
    itself - see the module docstring.

    Every pair at least ``min_index_gap`` apart is ranked by
    :func:`signature_distance`; the closest ``max_candidates`` go to Stage 2,
    where :func:`scanner3d.registration.icp` (seeded from the identity - no
    informed guess is available) checks each against ``max_rotation``,
    ``max_translation``, ``max_rmse`` and ``min_correspondences``. A
    candidate ICP itself refuses (no overlap where the identity puts it, too
    few points) simply does not verify; that is an ordinary, expected
    outcome for most proposed pairs; not an error this call raises.

    ``max_depth`` bounds every signature's histogram; left as ``None`` it is
    the largest z seen in any cloud, so no absolute depth is assumed - see
    :func:`_default_max_depth`.
    """
    frames = len(clouds)
    _ensure_parameters(
        frames=frames,
        min_index_gap=min_index_gap,
        max_candidates=max_candidates,
        bins=bins,
        max_rotation=max_rotation,
        max_translation=max_translation,
        max_rmse=max_rmse,
        min_correspondences=min_correspondences,
    )
    prepared = [_ensure_cloud(cloud, what=f"frame {index}") for index, cloud in enumerate(clouds)]
    resolved_max_depth = max_depth if max_depth is not None else _default_max_depth(prepared)
    signatures = [
        depth_signature_from_points(cloud, bins=bins, max_depth=resolved_max_depth)
        for cloud in prepared
    ]

    scored: list[tuple[float, int, int]] = []
    for i in range(frames):
        for j in range(i + min_index_gap, frames):
            scored.append((signature_distance(signatures[i], signatures[j]), i, j))
    scored.sort(key=lambda entry: entry[0])
    chosen = scored[:max_candidates]
    candidates = tuple(
        RevisitCandidate(i=i, j=j, signature_distance=distance) for distance, i, j in chosen
    )

    verified: list[Revisit] = []
    for _distance, i, j in chosen:
        try:
            fit = icp(prepared[i], prepared[j])
        except RegistrationNotAllowed:
            # No overlap at the identity, or too few points to solve on -
            # an ordinary outcome for a pair that is not a real revisit.
            continue
        if not (
            fit.transform.rotation_angle < max_rotation
            and fit.transform.translation_magnitude < max_translation
            and fit.rmse < max_rmse
            and fit.correspondences > min_correspondences
        ):
            continue
        verified.append(
            Revisit(
                i=i,
                j=j,
                transform=fit.transform,
                rmse=fit.rmse,
                correspondences=fit.correspondences,
                rotation_angle=fit.transform.rotation_angle,
                translation_magnitude=fit.transform.translation_magnitude,
            )
        )

    limitations = [
        "Stage 1 is deliberately crude: a normalised depth histogram cannot distinguish every "
        "viewpoint from every other, so it is a RANKING for Stage 2 to check, not evidence on "
        "its own. Nothing here is a revisit until ICP has verified it.",
        "Verification thresholds (max_rotation, max_translation, max_rmse, "
        "min_correspondences) came from one real capture on this scanner and are wide margins "
        "on that data, not universal constants - see the module docstring.",
        "A sweep with no genuine revisit at all correctly reports zero verified pairs. That is "
        "the intended answer, not a failure of the search: a false closure edge would be worse "
        "than none, dragging the pose graph toward a confidently wrong trajectory.",
        f"only the closest {max_candidates} proposed pairs by signature distance were sent to "
        "Stage 2; a genuine revisit whose signature happened to rank worse than that many other "
        "pairs would not be found. Raising max_candidates costs one ICP call per pair.",
        "This module only says two frames saw the same viewpoint - it does not decide the "
        "trajectory is wrong or correct it. See scanner3d.posegraph.optimise_poses for that.",
    ]
    if not verified and candidates:
        limitations.append(
            f"{len(candidates)} pair(s) were proposed by signature and NONE verified - either "
            "this sequence genuinely has no revisit, or every genuine revisit's signature "
            "ranked outside max_candidates."
        )

    return RevisitReport(
        frames=frames,
        candidates=candidates,
        verified=tuple(verified),
        min_index_gap=min_index_gap,
        max_candidates=max_candidates,
        limitations=tuple(limitations),
    )
