"""Rigid-body geometry: the shared vocabulary of registration, fusion and meshing.

Why this module exists
----------------------
Registration produces a pose per frame, fusion consumes those poses, meshing
consumes the fused result, and the validation harness measures a fitted shape
against a caliper reading. All four need the same handful of primitives - a
rigid transform, a rotation about an axis, a best-fit alignment between two
point sets, and an honest way to state how far apart two poses are. Defining
them once, here, is what keeps those modules from each inventing a slightly
different convention and disagreeing at the seams.

Conventions, fixed here and not restated elsewhere
--------------------------------------------------
* A point is a plain ``(x, y, z)`` tuple of floats. Bulk point data is never
  wrapped in objects: a fused cloud can hold millions of points and per-point
  dataclass overhead is not affordable. Metadata *is* wrapped, because there is
  one of it.
* A rotation is a 3x3 matrix in row-major order, ``R[i][j]``, acting on column
  vectors: ``x' = R @ x + t``. It is validated on construction to be a proper
  rotation (orthonormal, determinant +1), so a silently-scaling or reflecting
  "rotation" cannot enter the pipeline and corrupt a scan.
* Angles are radians everywhere. Callers that want degrees convert at their own
  edge.
* Right-handed throughout, matching :data:`scanner3d.depth_metric.CAMERA_FRAME`.

What this module does NOT do
----------------------------
It makes no claim about physical units. A translation of 1.0 is one unit of
whatever the depth scale produced, and that scalar's physical meaning remains
unestablished on this device. Nothing here converts to millimetres, because
nothing in this repository has earned the right to.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: One point in space. A bare tuple on purpose - see the module docstring.
Point = tuple[float, float, float]

#: A 3x3 matrix, row-major.
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

#: How far a matrix may drift from orthonormal before it is refused. Tight
#: enough to catch a genuine defect, loose enough to survive the float64 error
#: of composing a few hundred transforms in a turntable loop.
ORTHONORMAL_TOLERANCE: Final = 1e-6

#: Correct only accumulated floating-point drift, far below the validation
#: tolerance. Repeated R @ (R.T @ R) otherwise amplifies roundoff exponentially
#: even when reexpressing a stationary pose through an unchanged reference.
ROTATION_ROUNDOFF_TRIGGER: Final = 1e-12
ROTATION_ROUNDOFF_LIMIT: Final = 1e-10

#: Below this, a vector has no reliable direction and normalising it would
#: amplify noise into a confident-looking wrong answer.
MIN_VECTOR_NORM: Final = 1e-12


class GeometryNotAllowed(ScannerError):
    """A geometric input was offered that this module refuses to accept.

    Always raised before the value can propagate. A bad transform is far more
    damaging downstream than at its source: it silently misplaces every point
    it touches, and the resulting scan looks plausible.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class NonFiniteValue(GeometryNotAllowed):
    """A coordinate or matrix entry is NaN or an infinity."""


class DegenerateVector(GeometryNotAllowed):
    """A vector is too short to carry a direction.

    Normalising it would turn floating-point dust into a confident unit vector
    pointing nowhere in particular.
    """


class NotARotation(GeometryNotAllowed):
    """A matrix is not a proper rotation: not orthonormal, or a reflection.

    A reflection (determinant -1) is the dangerous case. It composes and
    inverts like a rotation and produces a mirror-image scan that looks
    entirely reasonable until measured against a real object.
    """


class CorrespondenceUnusable(GeometryNotAllowed):
    """Two point sets cannot yield a rigid alignment.

    Either they differ in length, are too short to constrain three degrees of
    rotational freedom, or are geometrically degenerate - all points collinear,
    for instance, which leaves rotation about that line undetermined.
    """


def ensure_finite(values: Iterable[float], what: str) -> None:
    """Refuse NaN and infinity at the boundary, naming what was wrong."""
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise NonFiniteValue(f"{what}: entry {index} is {value!r}, which is not finite")


# --- vector arithmetic --------------------------------------------------------


def subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scaled(a: Point, factor: float) -> Point:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def dot(a: Point, b: Point) -> float:
    return math.fsum((a[0] * b[0], a[1] * b[1], a[2] * b[2]))


def cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Point) -> float:
    return math.sqrt(dot(a, a))


def distance(a: Point, b: Point) -> float:
    return norm(subtract(a, b))


def normalize(a: Point) -> Point:
    """A unit vector in the direction of ``a``.

    Refuses a vector too short to have a direction rather than returning
    something arbitrary: an axis derived from noise is worse than no axis,
    because everything downstream trusts it.
    """
    ensure_finite(a, "vector")
    length = norm(a)
    if length < MIN_VECTOR_NORM:
        raise DegenerateVector(
            f"vector of norm {length!r} has no reliable direction "
            f"(needs at least {MIN_VECTOR_NORM})"
        )
    return scaled(a, 1.0 / length)


def centroid(points: Sequence[Point]) -> Point:
    """The arithmetic mean, summed pairwise so a large cloud does not drift.

    ``math.fsum`` rather than ``sum``: a million points accumulated naively in
    float64 loses low bits steadily, and the centroid is the origin every
    alignment is computed about.
    """
    if not points:
        raise CorrespondenceUnusable("the centroid of an empty point set is undefined")
    count = len(points)
    return (
        math.fsum(p[0] for p in points) / count,
        math.fsum(p[1] for p in points) / count,
        math.fsum(p[2] for p in points) / count,
    )


# --- rigid transforms ---------------------------------------------------------

IDENTITY_MATRIX: Final[Matrix3] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def determinant(m: Matrix3) -> float:
    return math.fsum(
        (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
            -m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]),
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
        )
    )


def ensure_rotation(m: Matrix3, *, repair_roundoff: bool = False) -> Matrix3:
    """Refuse anything that is not a proper rotation, saying which way it failed."""
    ensure_finite((value for row in m for value in row), "rotation matrix")
    defect = 0.0
    for i in range(3):
        for j in range(i, 3):
            expected = 1.0 if i == j else 0.0
            actual = math.fsum(m[k][i] * m[k][j] for k in range(3))
            defect = max(defect, abs(actual - expected))
            if abs(actual - expected) > ORTHONORMAL_TOLERANCE:
                raise NotARotation(
                    f"columns {i} and {j} have dot product {actual!r}, expected {expected!r}: "
                    "the matrix is not orthonormal, so it scales or skews what it transforms"
                )
    det = determinant(m)
    if abs(det - 1.0) > ORTHONORMAL_TOLERANCE:
        if det < 0.0:
            raise NotARotation(
                f"determinant is {det!r}: this is a reflection, not a rotation. Applying it "
                "would produce a mirror-image scan that composes and inverts normally and is "
                "not detectable by eye."
            )
        raise NotARotation(f"determinant is {det!r}, expected 1.0")
    if repair_roundoff and ROTATION_ROUNDOFF_TRIGGER < defect <= ROTATION_ROUNDOFF_LIMIT:
        # One polar/Newton correction R (3I - R.T R) / 2. The error is
        # quadratic in the already tiny defect. Validate both before and after:
        # this must never turn an invalid scale, skew or reflection into a pose.
        gram = tuple(
            tuple(math.fsum(m[k][i] * m[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        )
        corrected = tuple(
            tuple(
                0.5 * math.fsum(m[i][k] * ((3.0 if k == j else 0.0) - gram[k][j]) for k in range(3))
                for j in range(3)
            )
            for i in range(3)
        )
        return ensure_rotation(corrected)  # type: ignore[arg-type]
    return m


@dataclass(frozen=True, slots=True)
class Rigid:
    """A rigid-body transform: ``x' = R @ x + t``.

    Validated on construction. Every way of building one goes through
    :meth:`of`, so a transform that exists is a transform that is sound.
    """

    rotation: Matrix3
    translation: Point

    @classmethod
    def of(cls, rotation: Matrix3, translation: Point, *, repair_roundoff: bool = False) -> Rigid:
        """Build a validated transform."""
        ensure_finite(translation, "translation")
        return cls(
            rotation=ensure_rotation(rotation, repair_roundoff=repair_roundoff),
            translation=translation,
        )

    @classmethod
    def identity(cls) -> Rigid:
        return cls(rotation=IDENTITY_MATRIX, translation=(0.0, 0.0, 0.0))

    def apply(self, point: Point) -> Point:
        r, t = self.rotation, self.translation
        return (
            r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + t[0],
            r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + t[1],
            r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + t[2],
        )

    def apply_all(self, points: Iterable[Point]) -> Iterator[Point]:
        """Stream transformed points, so a large cloud is never doubled in memory."""
        r, t = self.rotation, self.translation
        r00, r01, r02 = r[0]
        r10, r11, r12 = r[1]
        r20, r21, r22 = r[2]
        t0, t1, t2 = t
        for x, y, z in points:
            yield (
                r00 * x + r01 * y + r02 * z + t0,
                r10 * x + r11 * y + r12 * z + t1,
                r20 * x + r21 * y + r22 * z + t2,
            )

    def compose(self, inner: Rigid) -> Rigid:
        """``self`` applied after ``inner``.

        That is, ``self.compose(inner).apply(p)`` equals
        ``self.apply(inner.apply(p))``. Stated explicitly because composition
        order is the single easiest thing to get backwards in a registration
        chain, and getting it backwards produces a scan that is wrong in a way
        that still looks like a scan.
        """
        a, b = self.rotation, inner.rotation
        rotation = tuple(
            tuple(math.fsum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        )
        return Rigid.of(rotation, self.apply(inner.translation), repair_roundoff=True)  # type: ignore[arg-type]

    def inverse(self) -> Rigid:
        r = self.rotation
        transposed = tuple(tuple(r[j][i] for j in range(3)) for i in range(3))
        inverted = Rigid(rotation=transposed, translation=(0.0, 0.0, 0.0))  # type: ignore[arg-type]
        back = inverted.apply(self.translation)
        return Rigid.of(transposed, (-back[0], -back[1], -back[2]))  # type: ignore[arg-type]

    @property
    def rotation_angle(self) -> float:
        """The geodesic rotation magnitude in radians, in ``[0, pi]``.

        This is the honest scalar for "how much rotation is this", and it is
        what a loop-closure error reports: the angle of the residual rotation,
        not a per-entry matrix difference, which has no units anyone can
        interpret.

        Computed as ``atan2(|sin t|, cos t)`` rather than the more obvious
        ``acos((trace - 1) / 2)``, and the difference matters precisely where
        this property is used most. Near the identity ``acos`` is severely
        ill-conditioned: ``acos(1 - e) ~ sqrt(2e)``, so float64 rounding of the
        trace puts a noise floor of about ``1e-8`` radians on the result. A
        well-closed turntable loop would report that floor instead of its true
        error, and the metric would silently stop resolving exactly when the
        answer is good. The ``atan2`` form has no such floor: the off-diagonal
        differences it uses are sums of same-signed terms for small angles, so
        no catastrophic cancellation occurs, and it stays accurate at ``pi``
        too, where ``acos`` is well-conditioned but ``asin`` would not be.
        """
        r = self.rotation
        trace = math.fsum((r[0][0], r[1][1], r[2][2]))
        # The skew-symmetric part is 2 * sin(t) * axis.
        skew = (r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1])
        return math.atan2(norm(skew) / 2.0, (trace - 1.0) / 2.0)

    @property
    def translation_magnitude(self) -> float:
        return norm(self.translation)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rotation": [list(row) for row in self.rotation],
            "translation": list(self.translation),
            "rotation_angle_radians": self.rotation_angle,
            "translation_magnitude": self.translation_magnitude,
        }


def rotation_about_axis(axis: Point, angle: float, *, through: Point = (0.0, 0.0, 0.0)) -> Rigid:
    """Rotation by ``angle`` radians about a line through ``through`` along ``axis``.

    This is the turntable primitive. A turntable does not rotate about the
    origin of the camera frame - it rotates about its own spindle, which is
    some line in space that has to be estimated. Taking ``through`` explicitly
    is what lets a caller express that without composing three transforms by
    hand and getting the order wrong.

    Rodrigues' rotation formula, built directly rather than via quaternions
    because the axis and angle are exactly what a turntable gives us.
    """
    ensure_finite((angle,), "angle")
    ensure_finite(through, "axis anchor point")
    unit = normalize(axis)
    x, y, z = unit
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    rotation: Matrix3 = (
        (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s),
        (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s),
        (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c),
    )
    # Rotate about the line, not the origin: translate the anchor to the
    # origin, rotate, translate back.
    spin = Rigid.of(rotation, (0.0, 0.0, 0.0))
    moved = spin.apply(through)
    return Rigid.of(rotation, subtract(through, moved))


# --- best-fit alignment -------------------------------------------------------


#: How small a covariance eigenvalue may be, relative to the largest, before the
#: point set is treated as not spanning that direction at all. Chosen well above
#: float64 dust so genuine noise does not read as a dimension, and well below any
#: real spread so a nearly-flat but genuinely two-dimensional cloud still counts.
RANK_TOLERANCE: Final = 1e-10


def _jacobi_eigen(
    matrix: list[list[float]], *, sweeps: int = 64
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors of a symmetric matrix, by cyclic Jacobi rotations.

    Chosen over a power iteration or a characteristic-polynomial root because it
    is unconditionally stable for symmetric matrices and needs no starting
    guess - and over an SVD because this repository has no numerical library.
    Returns eigenvalues and the matching eigenvectors as columns.
    """
    size = len(matrix)
    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(sweeps):
        off_diagonal = math.fsum(a[i][j] ** 2 for i in range(size) for j in range(i + 1, size))
        if off_diagonal < 1e-30:
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
                    vkp, vkq = vectors[k][p], vectors[k][q]
                    vectors[k][p], vectors[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(size)], vectors


def _largest_eigenvector(matrix: list[list[float]]) -> list[float]:
    """The eigenvector belonging to the largest eigenvalue."""
    values, vectors = _jacobi_eigen(matrix)
    best = max(range(len(values)), key=lambda i: values[i])
    return [vectors[k][best] for k in range(len(values))]


def spanned_dimensions(points: Sequence[Point]) -> int:
    """How many independent directions a point set actually spans: 0, 1, 2 or 3.

    Computed from the eigenvalues of the centred covariance, counting those above
    :data:`RANK_TOLERANCE` of the largest. All points identical gives 0, points on
    a line give 1, points on a plane give 2.

    This exists because a rigid fit needs the answer and cannot get it from its
    own machinery - see :func:`rigid_from_correspondences`.
    """
    if not points:
        return 0
    centre = centroid(points)
    offsets = [subtract(point, centre) for point in points]
    covariance = [[math.fsum(o[i] * o[j] for o in offsets) for j in range(3)] for i in range(3)]
    values, _ = _jacobi_eigen(covariance)
    largest = max(values)
    if largest <= 0.0:
        return 0
    return sum(1 for value in values if value > largest * RANK_TOLERANCE)


def rigid_from_correspondences(source: Sequence[Point], target: Sequence[Point]) -> Rigid:
    """The rigid transform best carrying ``source`` onto ``target``, least squares.

    Horn's quaternion method. Given paired points, it returns the rotation and
    translation minimising the sum of squared residuals - the closed-form
    solution, not an iteration, so it has no convergence behaviour to tune.

    This is the inner step of ICP: match points, solve, apply, repeat. It is
    also directly useful on its own when correspondences are already known,
    which is the case for a turntable sweep.

    Rigid only. It cannot absorb a scale error, which is deliberate: if two
    clouds differ in scale, that is a calibration fault and it must surface as
    a large residual rather than being quietly fitted away.

    Degeneracy is refused rather than fitted, and the reason is worth stating.
    A set of points lying on a line leaves rotation *about that line* completely
    undetermined; a set of coincident points determines no rotation at all. In
    both cases Horn's method still returns an answer, that answer is generally
    not the transform that produced the data, and - the dangerous part - the
    residual comes back at the float64 floor. A caller doing exactly what the
    rest of this module tells them to do, namely trusting the residual, would
    get no warning whatsoever. So the rank of the point set is checked directly
    rather than inferred from the solver, which cannot see it: the Jacobi
    routine accumulates orthogonal rotations from the identity, so its
    eigenvector is always unit length no matter how degenerate the input.
    """
    if len(source) != len(target):
        raise CorrespondenceUnusable(
            f"{len(source)} source points against {len(target)} target points: "
            "a correspondence set must pair one to one"
        )
    if len(source) < 3:
        raise CorrespondenceUnusable(
            f"{len(source)} correspondences cannot constrain a rotation; at least 3 are needed"
        )
    spanned = spanned_dimensions(source)
    if spanned < 2:
        raise CorrespondenceUnusable(
            f"the source points span only {spanned} dimension(s), so the rotation is "
            f"underdetermined: "
            + (
                "every point is at the same place"
                if spanned == 0
                else "they lie on a line, and rotation about that line changes nothing"
            )
            + ". A transform fitted to this would come back with a residual at the "
            "float64 floor while being the wrong transform, so it is refused instead."
        )

    source_centre = centroid(source)
    target_centre = centroid(target)
    p = [subtract(point, source_centre) for point in source]
    q = [subtract(point, target_centre) for point in target]

    # Cross-covariance, accumulated with fsum so a large set stays exact.
    s = [[math.fsum(p[n][i] * q[n][j] for n in range(len(p))) for j in range(3)] for i in range(3)]

    # Horn's symmetric 4x4, whose principal eigenvector is the optimal rotation
    # as a quaternion (w, x, y, z).
    trace = s[0][0] + s[1][1] + s[2][2]
    n_matrix = [
        [trace, s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]],
        [s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]],
        [s[2][0] - s[0][2], s[0][1] + s[1][0], -s[0][0] + s[1][1] - s[2][2], s[1][2] + s[2][1]],
        [s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], -s[0][0] - s[1][1] + s[2][2]],
    ]
    w, x, y, z = _largest_eigenvector(n_matrix)
    length = math.sqrt(math.fsum((w * w, x * x, y * y, z * z)))
    if length < MIN_VECTOR_NORM:
        raise CorrespondenceUnusable(
            "the correspondence set is degenerate - collinear or coincident points leave the "
            "rotation undetermined, so no unique alignment exists"
        )
    w, x, y, z = w / length, x / length, y / length, z / length

    rotation: Matrix3 = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    spin = Rigid.of(rotation, (0.0, 0.0, 0.0))
    return Rigid.of(rotation, subtract(target_centre, spin.apply(source_centre)))


def alignment_rmse(source: Sequence[Point], target: Sequence[Point], transform: Rigid) -> float:
    """Root-mean-square residual after applying ``transform`` to ``source``.

    The number to report alongside any alignment. A transform without its
    residual is an assertion; with it, it is a measurement.
    """
    if len(source) != len(target):
        raise CorrespondenceUnusable(
            f"{len(source)} source points against {len(target)} target points"
        )
    if not source:
        raise CorrespondenceUnusable("the residual of an empty correspondence set is undefined")
    total = math.fsum(
        dot(subtract(moved, actual), subtract(moved, actual))
        for moved, actual in zip(transform.apply_all(source), target, strict=True)
    )
    return math.sqrt(total / len(source))


def pose_difference(a: Rigid, b: Rigid) -> Rigid:
    """The transform carrying pose ``a`` to pose ``b``.

    For loop closure: compose every step of a full turntable revolution, then
    take the difference against the identity. The result's
    :attr:`Rigid.rotation_angle` and :attr:`Rigid.translation_magnitude` are the
    two halves of the closure error, each in units a reader can interpret.
    """
    return b.compose(a.inverse())
