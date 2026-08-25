"""Levenberg-Marquardt pose-graph optimisation: a second-order alternative to relaxation.

Why this exists
----------------
:mod:`scanner3d.posegraph` fixes the same loop-closure problem this module
does, by Gauss-Seidel relaxation: repeatedly let every node move to what its
incident edges currently predict. That module's own docstring is explicit
about what that buys and what it does not: it "is not guaranteed to reach the
global optimum from an arbitrarily bad start", because first-order relaxation
has no notion of how edges COUPLE - it updates one node at a time from its
neighbours' current estimates and never asks "if I moved several nodes
together, would the total disagreement drop faster".

The vendor's own registration stack does not do this. RevoScan 6.0.3.5's
registration libraries link ``libceres.so`` - Ceres Solver, Google's
nonlinear least-squares library - alongside ``libglog``/``libgflags``, and its
global-optimisation shared object is literally named ``libRPARegisterGO.so``.
That is a second-order (Gauss-Newton family) solve over the whole graph
jointly, which is exactly what this module adds: Levenberg-Marquardt on the
SE(3) pose-graph objective, with analytic Jacobians in a proper tangent-space
parameterisation, one damped normal-equation solve per iteration.

This is an ALTERNATIVE to :func:`scanner3d.posegraph.optimise_poses`, not a
replacement for it - see :func:`optimise_poses_lm`'s docstring for how closely
the two APIs mirror each other, and ``tests/test_posegraph_lm.py`` for a
direct head-to-head on the same fixture :mod:`scanner3d.posegraph`'s own test
suite uses. Which one a caller should reach for by default is not decided
here.

SE(3), the tangent space, and why the naive series is not enough
-------------------------------------------------------------------
A rigid pose lives on the manifold SE(3), not a vector space: there is no
sound way to average two poses, or take a small step from one, by adding or
scaling their matrix entries directly (the result is not a rotation - see
:func:`scanner3d.geometry.ensure_rotation`). The standard fix is a LOCAL
linear parameterisation around the current estimate, via the group's
exponential and logarithm maps: :func:`se3_exp` turns a 6-vector tangent
``(rho, phi)`` - a translational and a rotational generator - into a genuine
``Rigid``, and :func:`se3_log` is its inverse. Every LM step solves for a
tangent-space update, then RETRACTS it back onto the manifold via
:func:`se3_exp` rather than perturbing a matrix and hoping it stays a
rotation.

Both maps have a textbook closed form built from ``sin(theta)/theta``,
``(1-cos(theta))/theta**2`` and siblings, ``theta`` the rotation angle - and
every one of those ratios is ``0/0`` at ``theta = 0``. :attr:`Rigid.rotation_angle`
already documents the analogous trap for reading an angle BACK off a matrix
(``acos`` versus ``atan2`` near the identity); this module has the same trap
in the other direction, computing a rotation FROM an angle, at every one of
several composed formulas, not just one. Every ratio here is evaluated from a
Taylor series below :data:`_SMALL_ANGLE_THRESHOLD` instead of the closed
form, so no formula in this module divides a genuinely small number by
another genuinely small number - see :func:`_so3_exp`, :func:`_so3_log`,
:func:`_so3_v`, :func:`_so3_v_inv` and :func:`_se3_q`.

The rotation LOG map has a second, unrelated conditioning trap at the OTHER
end of the range: recovering the rotation AXIS near ``theta = pi``. The
usual axis extraction divides by ``sin(theta)``, which also goes to zero
there - not because the formula is a small-angle series, but because a
rotation by exactly ``pi`` about ``axis`` and one by exactly ``pi`` about
``-axis`` are the identical rotation, so no continuous axis choice exists
through that point. :func:`_so3_log` sidesteps this: away from
``theta = 0`` it recovers the axis from the diagonal of the rotation matrix
(a division by ``1 - cos(theta)``, which is bounded away from zero
everywhere ``theta`` is not itself near zero, all the way through ``pi``)
and only uses the ``sin(theta)``-dependent skew-symmetric part to choose
between the two equally-valid global signs - see its docstring.

What this module does NOT do
------------------------------
It does not detect loop closures, decide which edges to trust, or know
anything about capture geometry - exactly the same boundary
:mod:`scanner3d.posegraph`'s module docstring draws, for the same reason: that
is the caller's job, upstream of both optimisers.

It has no robust/outlier-rejecting mode. :mod:`scanner3d.posegraph`'s
``robust=True`` (IRLS with a Cauchy kernel) has no equivalent here - every
edge is trusted exactly at its stated weight, always. A caller who needs
outlier rejection has exactly one place to get it today, and this is not it.

It does not do a sparse solve. The normal equations are a dense
``6*(n-1) x 6*(n-1)`` system (``n`` the node count, minus the anchor),
assembled and factored with ``numpy.linalg.solve`` every iteration. That is
entirely reasonable for the few hundred poses a real capture produces - see
:func:`optimise_poses_lm` - and deliberately not the sparse block-structured
solve a pose graph's mostly-local connectivity would otherwise invite; that
is a real optimisation this module leaves on the table at this scale.

It does not reach the global optimum from an arbitrarily bad starting
guess either. LM is a much better LOCAL model of the objective than
Gauss-Seidel relaxation - it uses curvature (via the Jacobian) rather than
only ever asking one node "what do your neighbours predict" - but it is
still a local, iterative method: started far enough from a solution, on an
objective with more than one basin, it can converge to the wrong one. The
starting point this module is actually asked to fix - a chain with a mild,
systematic per-step bias - is exactly the case where that risk is low; see
the head-to-head test in ``tests/test_posegraph_lm.py`` for a measurement,
not merely a claim.

It requires numpy. Unlike :mod:`scanner3d.posegraph`, which was written to
work with no numerical library at all, this module's entire reason to exist
is the joint linear solve at its core, and hand-rolling that solve in pure
Python is not a reasonable option at this scale - see
:func:`scanner3d.compute`'s module docstring for the project's numpy policy.
Like that module, this one still IMPORTS cleanly with no numpy installed;
only calling :func:`optimise_poses_lm` (or the numpy-backed helpers it calls)
raises :class:`PoseGraphLMNotAllowed` if numpy is not importable, and the
import is never assumed to have already happened at module scope - see
:func:`_probe_numpy`.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Matrix3, Point, Rigid
from scanner3d.posegraph import EdgeResidual, PoseEdge
from scanner3d.posegraph import edge_residuals as _edge_residuals

if TYPE_CHECKING:
    import numpy as np

SCHEMA_VERSION: Final = 1

#: Below this rotation angle (radians), every closed-form ratio in this module
#: ``(sin/theta, (1-cos)/theta**2, ...)`` is replaced by its Taylor series
#: instead - see the module docstring. ``1e-4`` keeps the series' own
#: truncation error (next omitted term is ``O(theta**4)``, so ``~1e-16`` here)
#: far below float64 noise, while staying far enough from zero that the
#: closed form it replaces would already have lost precision to
#: cancellation - the same reasoning
#: :data:`scanner3d.geometry.ORTHONORMAL_TOLERANCE` documents for a different
#: near-zero degeneracy.
_SMALL_ANGLE_THRESHOLD: Final = 1e-4

#: Floor under the Marquardt damping parameter, so a long run of well-fit
#: accepted steps cannot drive it to exactly zero and re-expose the system to
#: whatever near-singularity the damping exists to guard against in the
#: first place.
_MIN_DAMPING: Final = 1e-12

#: Seed for the initial damping parameter: ``tau`` times the largest diagonal
#: entry of the un-damped ``J^T J``. Standard Levenberg-Marquardt practice
#: (Madsen, Nielsen & Tingleff's tutorial figure) - large enough that the
#: first step is a cautious, gradient-descent-like one on a completely
#: unknown objective, small enough not to waste iterations if the start is
#: already good.
_INITIAL_DAMPING_TAU: Final = 1e-3

#: Multiplier applied to the damping parameter on a REJECTED step, and to
#: itself after every rejection (Nielsen's classic adaptive schedule): the
#: first rejection doubles the damping, the second quadruples it relative to
#: before, and so on - an LM run that keeps failing backs off aggressively
#: rather than retrying a barely-larger damping value one iteration at a
#: time.
_REJECT_DAMPING_GROWTH: Final = 2.0

#: A step is treated as having converged the run if it is ACCEPTED and
#: reduces the cost by less than this in absolute terms - the same role
#: :data:`scanner3d.posegraph`'s ``tolerance`` parameter plays for relaxation,
#: and deliberately named and defaulted the same way so a caller reading one
#: module's default already has a sense of the other's.
DEFAULT_TOLERANCE: Final = 1e-12

#: A run is ALSO treated as converged, independent of any step being
#: attempted, once the weighted gradient (``J^T r``, every edge's pull on
#: every free parameter) is this small in every component. This is what lets
#: a graph that already satisfies its edges converge in zero iterations
#: rather than looping until ``iterations`` is exhausted trying to accept a
#: step whose predicted improvement is genuinely zero - see
#: :func:`optimise_poses_lm`.
_GRADIENT_TOLERANCE: Final = 1e-10


class PoseGraphLMNotAllowed(ScannerError):
    """An LM pose-graph input, parameter, or environment was refused.

    Covers both malformed input - mirroring exactly what
    :exc:`scanner3d.posegraph.PoseGraphNotAllowed` refuses, so the two
    optimisers reject the same graphs for the same reasons - and numpy being
    unavailable, which :mod:`scanner3d.posegraph` has no equivalent failure
    mode for at all, since it needs no numerical library. See the module
    docstring's "It requires numpy" section.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


def _probe_numpy() -> tuple[bool, str | None]:
    """``(usable, reason-if-not)`` - the same shape and reasoning as
    :func:`scanner3d.compute._probe_numpy`: a real import, not a spec lookup,
    so a numpy that is installed but broken is reported the same way as one
    that is missing outright.
    """
    try:
        importlib.import_module("numpy")
    except ImportError as exc:
        return False, str(exc)
    return True, None


def _require_numpy() -> None:
    """Refuse with a clear reason if numpy is not usable here; otherwise do nothing.

    Called once, at the top of each PUBLIC function in this module
    (:func:`se3_exp`, :func:`se3_log`, :func:`optimise_poses_lm`), before
    anything else runs. Every private helper below numpy touches instead does
    its own local ``import numpy as np`` - unguarded, and typed by mypy as
    the real numpy stubs rather than an opaque module, which a wrapper
    returning ``import numpy`` deliberately did not; see
    :class:`scanner3d.compute._NumpyBackend` for the same split between "one
    friendly, guarded entry point" and "many cheap, unguarded local imports"
    - trusting that a private helper is never reachable except through a
    public function that already checked.
    """
    usable, reason = _probe_numpy()
    if not usable:
        raise PoseGraphLMNotAllowed(
            f"scanner3d.posegraph_lm requires numpy, which is not usable here: {reason}. "
            "Unlike scanner3d.posegraph, this module's entire purpose is the joint linear "
            "solve at its core, and there is no pure-Python fallback for it."
        )


# --- SO(3) building blocks --------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    """The 3x3 matrix such that ``skew(v) @ x == cross(v, x)``."""
    import numpy as np

    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues' formula, built from ``phi`` directly rather than an axis and
    an angle - :func:`scanner3d.geometry.rotation_about_axis` needs a nonzero
    axis (:func:`scanner3d.geometry.normalize` refuses one that is too short
    to have a direction), which is exactly the vector this function must
    still return a sound rotation for at ``phi = 0``, so it cannot be reused
    here. See the module docstring's small-angle section for the two
    coefficients this expands as a Taylor series below
    :data:`_SMALL_ANGLE_THRESHOLD`.
    """
    import numpy as np

    theta = float(np.linalg.norm(phi))
    skew_phi = _skew(phi)
    if theta < _SMALL_ANGLE_THRESHOLD:
        sin_over_theta = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        one_minus_cos_over_theta_sq = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    else:
        sin_over_theta = math.sin(theta) / theta
        one_minus_cos_over_theta_sq = (1.0 - math.cos(theta)) / theta**2
    return (  # type: ignore[no-any-return]
        np.eye(3) + sin_over_theta * skew_phi + one_minus_cos_over_theta_sq * (skew_phi @ skew_phi)
    )


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    """The rotation vector ``phi`` (axis times angle) with ``_so3_exp(phi) == rotation``.

    ``theta`` (the angle) is recovered exactly as :attr:`Rigid.rotation_angle`
    does - ``atan2`` of the skew-symmetric part against the trace, not
    ``acos``, for the identical reason that property's own docstring gives:
    ``acos`` is ill-conditioned near the identity and this is not.

    The axis is where this function earns its docstring. Below
    :data:`_SMALL_ANGLE_THRESHOLD` it comes from the skew-symmetric part
    directly, scaled by a Taylor series in ``theta`` (see the module
    docstring) rather than a division by ``sin(theta)`` - the small-angle
    trap every closed form in this module shares.

    AWAY from ``theta = 0`` - which includes ``theta`` all the way up to
    ``pi``, where the skew-symmetric part alone cannot recover an axis at
    all, since a rotation by ``pi`` about ``axis`` is identical to one by
    ``pi`` about ``-axis`` and the skew-symmetric part of both is zero - the
    axis instead comes from the DIAGONAL of ``rotation``:
    ``rotation[i][i] == cos(theta) + (1 - cos(theta)) * axis[i]**2``, a
    relation that only divides by ``1 - cos(theta)``, which is bounded away
    from zero for every ``theta`` this branch is reached for. That fixes
    ``axis`` up to an overall sign (the diagonal only ever sees
    ``axis[i]**2``); the skew-symmetric part - the same one the small-angle
    branch uses, here only for its SIGN, not scaled by a near-zero
    ``1/sin(theta)`` - breaks that last ambiguity. Near ``theta = pi``, where
    the skew-symmetric part is itself near zero, the sign it reports is
    numerically noisy, but genuinely ARBITRARY there: ``axis`` and ``-axis``
    both represent the identical rotation at ``theta = pi``, so either choice
    round-trips correctly.
    """
    import numpy as np

    trace = float(np.trace(rotation))
    skew_vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    cos_theta = (trace - 1.0) / 2.0
    theta = math.atan2(float(np.linalg.norm(skew_vector)) / 2.0, cos_theta)

    if theta < _SMALL_ANGLE_THRESHOLD:
        # theta / sin(theta), as a Taylor series so this never divides by
        # sin(theta) directly - see the module docstring.
        theta_over_sin = 1.0 + theta**2 / 6.0 + 7.0 * theta**4 / 360.0
        return skew_vector * 0.5 * theta_over_sin

    denom = 1.0 - cos_theta
    diag = np.array([max(0.0, (rotation[i, i] - cos_theta) / denom) for i in range(3)])
    largest = int(np.argmax(diag))
    axis = np.zeros(3)
    axis[largest] = math.sqrt(diag[largest])
    for other in range(3):
        if other == largest:
            continue
        axis[other] = (rotation[largest, other] + rotation[other, largest]) / (
            2.0 * denom * axis[largest]
        )
    if float(np.dot(axis, skew_vector)) < 0.0:
        axis = -axis
    norm_axis = float(np.linalg.norm(axis))
    if norm_axis > 1e-12:
        axis = axis / norm_axis
    return theta * axis


def _so3_v(phi: np.ndarray) -> np.ndarray:
    """The SE(3) exponential map's translation-coupling matrix (also SO(3)'s
    left Jacobian): ``se3_exp((rho, phi))``'s translation is ``_so3_v(phi) @ rho``.
    """
    import numpy as np

    theta = float(np.linalg.norm(phi))
    skew_phi = _skew(phi)
    if theta < _SMALL_ANGLE_THRESHOLD:
        b = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
        c = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040.0
    else:
        b = (1.0 - math.cos(theta)) / theta**2
        c = (theta - math.sin(theta)) / theta**3
    return np.eye(3) + b * skew_phi + c * (skew_phi @ skew_phi)  # type: ignore[no-any-return]


def _so3_v_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_so3_v`. ``se3_log((rotation, t))``'s ``rho`` half is
    ``_so3_v_inv(phi) @ t`` with ``phi = _so3_log(rotation)``.

    Also, evaluated at ``-phi``, this IS the (right) Jacobian inverse of
    SO(3) at ``phi`` - :func:`_se3_right_jacobian_inv` uses exactly that
    identity rather than a second, separately-derived formula, so there is
    one place this coefficient set is transcribed, not two that could drift
    out of agreement with each other.
    """
    import numpy as np

    theta = float(np.linalg.norm(phi))
    skew_phi = _skew(phi)
    if theta < _SMALL_ANGLE_THRESHOLD:
        d = 1.0 / 12.0 + theta**2 / 720.0 + theta**4 / 30240.0
    else:
        d = 1.0 / theta**2 - (1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta))
    return np.eye(3) - 0.5 * skew_phi + d * (skew_phi @ skew_phi)  # type: ignore[no-any-return]


# --- SE(3) exp/log -----------------------------------------------------------------


def se3_exp(tangent: Sequence[float]) -> Rigid:
    """The SE(3) exponential map: a 6-vector tangent to a validated :class:`Rigid`.

    ``tangent`` is ``(rho_x, rho_y, rho_z, phi_x, phi_y, phi_z)`` - a
    translational generator ``rho`` followed by a rotation vector ``phi``
    (axis times angle, the same quantity :func:`_so3_log` returns). The
    result's rotation is ``_so3_exp(phi)``; its translation is
    ``_so3_v(phi) @ rho``, NOT ``rho`` directly - the coupling is what makes
    this the genuine SE(3) exponential rather than a rotation and a
    translation exponentiated independently, and is exactly what lets a
    single tangent vector represent "move along a helix" rather than only
    "spin in place and separately translate". See the module docstring for
    why ``phi`` near zero needs care and gets it.

    Round-trips through :func:`se3_log` to float64 precision for every
    ``phi`` in ``[0, pi]`` - see ``tests/test_posegraph_lm.py``'s dedicated
    round-trip tests, including at ``phi = 0`` and ``phi`` near ``pi``, the
    two ends of the range this module's conditioning care is spent on.
    """
    _require_numpy()
    import numpy as np

    tangent_array = np.asarray(tangent, dtype=np.float64)
    if tangent_array.shape != (6,):
        raise PoseGraphLMNotAllowed(
            f"se3_exp needs a 6-vector tangent (rho, phi), got shape {tangent_array.shape}"
        )
    rho, phi = tangent_array[:3], tangent_array[3:]
    rotation = _so3_exp(phi)
    translation = _so3_v(phi) @ rho
    return Rigid.of(_to_matrix3(rotation), _to_point(translation))


def se3_log(pose: Rigid) -> tuple[float, float, float, float, float, float]:
    """The SE(3) logarithm map: the inverse of :func:`se3_exp`.

    Returns ``(rho_x, rho_y, rho_z, phi_x, phi_y, phi_z)`` such that
    ``se3_exp(se3_log(pose))`` reproduces ``pose`` to float64 precision -
    ``phi`` from :func:`_so3_log`, ``rho`` from
    ``_so3_v_inv(phi) @ pose.translation``, the exact inverse coupling
    :func:`se3_exp` applies. ``pose.rotation_angle`` (see :class:`Rigid`) is
    always ``norm(phi)`` up to float64 rounding: both are the same geodesic
    angle, computed the same conditioning-aware way.
    """
    _require_numpy()
    import numpy as np

    rotation = np.asarray(pose.rotation, dtype=np.float64)
    translation = np.asarray(pose.translation, dtype=np.float64)
    phi = _so3_log(rotation)
    rho = _so3_v_inv(phi) @ translation
    return (
        float(rho[0]),
        float(rho[1]),
        float(rho[2]),
        float(phi[0]),
        float(phi[1]),
        float(phi[2]),
    )


def _to_matrix3(rotation: np.ndarray) -> Matrix3:
    return (
        (float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])),
        (float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])),
        (float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])),
    )


def _to_point(vector: np.ndarray) -> Point:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


# --- the residual and its analytic Jacobians ---------------------------------------


def _se3_q(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """The SE(3) right Jacobian's translation-rotation coupling block, evaluated
    directly at ``(rho, phi)`` (Barfoot & Furgale 2014, "Associating
    Uncertainty with Three-Dimensional Poses" - the LEFT Jacobian's coupling
    block there, algebraically negated-and-expanded here into the RIGHT
    Jacobian's own block directly, rather than computed by negating the
    arguments to a left-Jacobian formula and re-deriving the sign changes at
    every call site).

    :func:`_se3_right_jacobian_inv`, the only caller, needs the RIGHT
    Jacobian specifically because this module's node update
    (:func:`_apply_update`) is a RIGHT/local perturbation (``pose ->
    pose.compose(se3_exp(delta))``).
    """
    import numpy as np

    theta = float(np.linalg.norm(phi))
    skew_phi = _skew(phi)
    skew_rho = _skew(rho)
    if theta < _SMALL_ANGLE_THRESHOLD:
        c1 = 1.0 / 6.0 - theta**2 / 120.0
        c2 = 1.0 / 24.0 - theta**2 / 720.0
        c3 = 1.0 / 120.0 - theta**2 / 5040.0
    else:
        c1 = (theta - math.sin(theta)) / theta**3
        c2 = (theta**2 + 2.0 * math.cos(theta) - 2.0) / (2.0 * theta**4)
        c3 = (2.0 * theta - 3.0 * math.sin(theta) + theta * math.cos(theta)) / (2.0 * theta**5)
    phi_rho = skew_phi @ skew_rho  # Phi @ P
    rho_phi = skew_rho @ skew_phi  # P @ Phi
    phi_rho_phi = phi_rho @ skew_phi  # Phi @ P @ Phi
    return (  # type: ignore[no-any-return]
        -0.5 * skew_rho
        + c1 * (phi_rho + rho_phi - phi_rho_phi)
        - c2 * (skew_phi @ phi_rho + rho_phi @ skew_phi - 3.0 * phi_rho_phi)
        + c3 * (phi_rho_phi @ skew_phi + skew_phi @ skew_phi @ rho_phi)
    )


def _se3_right_jacobian_inv(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """The 6x6 block ``J_r^{-1}`` such that ``se3_log(X.compose(se3_exp(xi)))``
    is, to first order in ``xi`` around ``xi = 0``, ``se3_log(X) + J_r^{-1} @ xi``
    - evaluated here at ``(rho, phi) = se3_log(X)``.

    Block upper-triangular, ``[[A, B], [0, A]]`` with ``A`` the SO(3) right
    Jacobian inverse (:func:`_so3_v_inv` at ``-phi``) and ``B`` built from
    :func:`_se3_q`: inverting a block upper-triangular matrix with equal
    diagonal blocks needs no general linear solve, just
    ``[[A^-1, -A^-1 B A^-1], [0, A^-1]]``, computed directly below.
    """
    import numpy as np

    right_jacobian_inv_so3 = _so3_v_inv(-phi)
    q_right = _se3_q(rho, phi)
    top_right = -right_jacobian_inv_so3 @ q_right @ right_jacobian_inv_so3
    top = np.hstack([right_jacobian_inv_so3, top_right])
    bottom = np.hstack([np.zeros((3, 3)), right_jacobian_inv_so3])
    return np.vstack([top, bottom])


def _compose(
    rotation_a: np.ndarray,
    translation_a: np.ndarray,
    rotation_b: np.ndarray,
    translation_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``a`` applied after ``b`` - the same order as :meth:`Rigid.compose`, on
    raw arrays rather than validated :class:`Rigid` objects.

    Every node's pose is re-composed with an update every LM iteration -
    :func:`_apply_update` does it once per active node, per iteration - and
    every edge's residual re-composes twice; :meth:`Rigid.of` re-validates
    orthonormality (an eigen-decomposition-free but still real cost) on every
    construction, which is the right default for a value that is CONSTRUCTED
    rarely and TRUSTED widely, but the wrong tradeoff inside an inner loop
    called thousands of times per solve. This works on the same numbers with
    none of that per-call validation; the one :class:`Rigid` that matters -
    each node's pose at the END of the solve - is still built through
    :meth:`Rigid.of`, in :func:`_extract_poses`, so a genuine defect (a
    rotation that drifted off SO(3) over many iterations) is still caught
    there rather than silently accepted throughout.
    """
    return rotation_a @ rotation_b, rotation_a @ translation_b + translation_a


def _inverse(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transposed = rotation.T
    return transposed, -(transposed @ translation)


def _edge_residual_and_jacobians(
    rotation_i: np.ndarray,
    translation_i: np.ndarray,
    rotation_j: np.ndarray,
    translation_j: np.ndarray,
    rotation_z: np.ndarray,
    translation_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tangent-space residual for one edge, and its analytic Jacobians
    with respect to a RIGHT/local perturbation of node ``i`` and node ``j``.

    The residual is ``se3_log(Z^-1 . (T_j^-1 . T_i))`` - zero exactly when
    the measured relative transform ``Z`` (mapping frame ``i``'s local
    coordinates to frame ``j``'s, :class:`scanner3d.posegraph.PoseEdge`'s own
    convention) equals the relative transform the CURRENT poses imply,
    ``T_j^-1 . T_i``. This is a different convention from
    :func:`scanner3d.posegraph.edge_residual` (which computes a WORLD-frame
    discrepancy, ``pose_difference``, between a predicted and an actual pose)
    - the two are zero on exactly the same inputs (an edge either agrees with
    the poses or it does not; see the module's derivation notes in the
    project history for the algebra), but this convention is the one whose
    Jacobian has the clean, standard closed form below. Final results are
    still reported through :func:`scanner3d.posegraph.edge_residuals`, so a
    caller comparing this module's output against
    :mod:`scanner3d.posegraph`'s sees the SAME residual metric either way -
    this internal convention never leaks into what is reported.

    Both Jacobians are ``6x6``, mapping a node's own 6-vector local
    perturbation (see :func:`_apply_update`) to the resulting change in the
    6-vector residual, to first order. Derived from two facts, both standard
    in Lie-group least squares and neither specific to this module: first,
    that ``T_j(dj)^-1 . T_i(di)``, expanded to first order in the
    perturbations ``di``, ``dj``, is ITSELF a local perturbation of the
    unperturbed relative pose ``M0 = T_j^-1 . T_i`` - a short computation
    that only needs the SO(3) conjugation identity ``R^T [v]x R == [R^T v]x``
    - and second, the standard "composing shifts the log by the inverse
    right Jacobian" identity that :func:`_se3_right_jacobian_inv` implements.
    Verified against central finite differences of the residual itself
    (:func:`se3_log`, :func:`se3_exp` and :meth:`Rigid.compose`, entirely
    independent code paths from this function) across randomised poses,
    including near ``theta = pi`` where :func:`_so3_log`'s axis extraction is
    at its most delicate - see
    ``test_edge_jacobian_matches_numeric_differences`` in
    ``tests/test_posegraph_lm.py`` for the measured agreement.
    """
    import numpy as np

    inv_rotation_j, inv_translation_j = _inverse(rotation_j, translation_j)
    relative_rotation, relative_translation = _compose(
        inv_rotation_j, inv_translation_j, rotation_i, translation_i
    )
    inv_rotation_z, inv_translation_z = _inverse(rotation_z, translation_z)
    full_rotation, full_translation = _compose(
        inv_rotation_z, inv_translation_z, relative_rotation, relative_translation
    )
    phi0 = _so3_log(full_rotation)
    rho0 = _so3_v_inv(phi0) @ full_translation
    residual = np.concatenate([rho0, phi0])

    right_jacobian_inv = _se3_right_jacobian_inv(rho0, phi0)

    # The local perturbation of M0 = T_j^-1 . T_i induced by perturbing node
    # i is the identity map (perturbing i by (rho, phi) perturbs M by exactly
    # (rho, phi) in M's own local frame - see the docstring above). See
    # tests/test_posegraph_lm.py's Jacobian-agreement test for the numeric
    # confirmation this identity block is not accidentally wrong.
    jacobian_i = right_jacobian_inv

    relative_rotation_t = relative_rotation.T
    coupling = _skew(relative_rotation_t @ relative_translation) @ relative_rotation_t
    b_j = np.zeros((6, 6))
    b_j[0:3, 0:3] = -relative_rotation_t
    b_j[0:3, 3:6] = coupling
    b_j[3:6, 3:6] = -relative_rotation_t
    jacobian_j = right_jacobian_inv @ b_j

    return residual, jacobian_i, jacobian_j


# --- LM over the whole graph --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoseGraphLMResult:
    """What Levenberg-Marquardt found, and how it got there.

    Field names deliberately mirror
    :class:`scanner3d.posegraph.PoseGraphResult` wherever the two modules
    share a concept - ``poses``, ``error_history``, ``iterations_run``,
    ``converged``, ``anchor``, ``edge_residuals``, ``limitations`` - so a
    caller comparing the two, or writing code that accepts either result,
    reads the identical attribute either way. There is no ``robust`` field
    or its siblings: this module has no robust mode - see the module
    docstring.
    """

    poses: tuple[Rigid, ...]
    error_history: tuple[float, ...]
    iterations_run: int
    converged: bool
    anchor: int
    edge_residuals: tuple[EdgeResidual, ...]
    limitations: tuple[str, ...]

    @property
    def initial_error(self) -> float | None:
        return self.error_history[0] if self.error_history else None

    @property
    def final_error(self) -> float | None:
        return self.error_history[-1] if self.error_history else None

    @property
    def worst_edge(self) -> EdgeResidual | None:
        """The edge this run agreed with least, by :attr:`EdgeResidual.combined`.

        Same definition, same caveats, as
        :attr:`scanner3d.posegraph.PoseGraphResult.worst_edge`.
        """
        return max(self.edge_residuals, key=lambda r: r.combined, default=None)

    def as_dict(self) -> dict[str, object]:
        worst = self.worst_edge
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": len(self.poses),
            "edges": len(self.edge_residuals),
            "anchor": self.anchor,
            "iterations_run": self.iterations_run,
            "converged": self.converged,
            "initial_error": self.initial_error,
            "final_error": self.final_error,
            "error_history": list(self.error_history),
            "worst_edge": worst.as_dict() if worst else None,
            "limitations": list(self.limitations),
        }


def _validate(
    nodes: Sequence[Rigid],
    edges: Sequence[PoseEdge],
    anchor: int,
    iterations: int,
    tolerance: float,
) -> None:
    """The exact checks :func:`scanner3d.posegraph.optimise_poses` makes, so the
    two optimisers refuse the same malformed input for the same reason - see
    :class:`PoseGraphLMNotAllowed`.
    """
    if len(nodes) < 2:
        raise PoseGraphLMNotAllowed(f"{len(nodes)} node(s) is not enough to have a relative pose")
    if not 0 <= anchor < len(nodes):
        raise PoseGraphLMNotAllowed(
            f"anchor {anchor} is not a node index in range(0, {len(nodes)})"
        )
    if iterations < 1:
        raise PoseGraphLMNotAllowed(f"iterations must be at least 1, got {iterations}")
    if tolerance < 0.0:
        raise PoseGraphLMNotAllowed(f"tolerance must be non-negative, got {tolerance}")
    for index, edge in enumerate(edges):
        if not 0 <= edge.i < len(nodes) or not 0 <= edge.j < len(nodes):
            raise PoseGraphLMNotAllowed(
                f"edge {index} references node {edge.i} or {edge.j}, out of range(0, {len(nodes)})"
            )
        if edge.i == edge.j:
            raise PoseGraphLMNotAllowed(f"edge {index} relates node {edge.i} to itself")
        if not edge.weight > 0.0:
            raise PoseGraphLMNotAllowed(f"edge {index} has non-positive weight {edge.weight!r}")


def _cost_and_normal_equations(
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    edges: Sequence[PoseEdge],
    param_index: dict[int, int],
    num_params: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """The weighted sum-of-squares cost, and the (dense) normal-equation
    matrices ``J^T J`` and ``J^T r`` assembled directly from every edge's own
    small Jacobian blocks - never by building the full, mostly-zero
    ``6 * len(edges) x num_params`` Jacobian and multiplying it out. Both
    give the identical ``J^T J`` and ``J^T r`` (each edge touches at most two
    node blocks, so its contribution to the product is exactly these four
    small block terms; see any Gauss-Newton/LM reference), and this is the
    dense-but-not-wasteful way to get there at the scale this module targets
    - see the module docstring's "It does not do a sparse solve" section for
    why this stops short of exploiting the graph's sparsity further.
    """
    import numpy as np

    jt_j = np.zeros((num_params, num_params))
    jt_r = np.zeros(num_params)
    cost = 0.0
    for edge in edges:
        residual, jacobian_i, jacobian_j = _edge_residual_and_jacobians(
            rotations[edge.i],
            translations[edge.i],
            rotations[edge.j],
            translations[edge.j],
            np.asarray(edge.transform.rotation, dtype=np.float64),
            np.asarray(edge.transform.translation, dtype=np.float64),
        )
        sqrt_weight = math.sqrt(edge.weight)
        residual_w = sqrt_weight * residual
        cost += float(residual_w @ residual_w)

        block_i = param_index.get(edge.i)
        block_j = param_index.get(edge.j)
        jacobian_i_w = sqrt_weight * jacobian_i
        jacobian_j_w = sqrt_weight * jacobian_j
        if block_i is not None:
            jt_j[block_i : block_i + 6, block_i : block_i + 6] += jacobian_i_w.T @ jacobian_i_w
            jt_r[block_i : block_i + 6] += jacobian_i_w.T @ residual_w
        if block_j is not None:
            jt_j[block_j : block_j + 6, block_j : block_j + 6] += jacobian_j_w.T @ jacobian_j_w
            jt_r[block_j : block_j + 6] += jacobian_j_w.T @ residual_w
        if block_i is not None and block_j is not None:
            cross = jacobian_i_w.T @ jacobian_j_w
            jt_j[block_i : block_i + 6, block_j : block_j + 6] += cross
            jt_j[block_j : block_j + 6, block_i : block_i + 6] += cross.T
    return cost, jt_j, jt_r


def _apply_update(
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    delta: np.ndarray,
    param_index: dict[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Retract every active node's pose by its slice of ``delta``: ``pose ->
    pose.compose(se3_exp(delta))`` - a RIGHT/local perturbation, expressed in
    the node's own current frame rather than the world frame. Nodes with no
    parameter block (the anchor, and any node :func:`optimise_poses_lm` found
    no edge touching) are copied through unchanged.
    """
    new_rotations = list(rotations)
    new_translations = list(translations)
    for node, block in param_index.items():
        node_delta = delta[block : block + 6]
        rho, phi = node_delta[:3], node_delta[3:]
        update_rotation = _so3_exp(phi)
        update_translation = _so3_v(phi) @ rho
        new_rotations[node], new_translations[node] = _compose(
            rotations[node], translations[node], update_rotation, update_translation
        )
    return new_rotations, new_translations


def optimise_poses_lm(
    nodes: Sequence[Rigid],
    edges: Sequence[PoseEdge],
    *,
    anchor: int = 0,
    iterations: int = 100,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PoseGraphLMResult:
    """Levenberg-Marquardt over the whole pose graph jointly, holding ``anchor`` fixed.

    Minimises the weighted sum, over every edge, of the squared SE(3)
    tangent-space residual between what the edge measured and what the
    current poses imply (see :func:`_edge_residual_and_jacobians`), by
    repeatedly: linearising every edge's residual at the current poses
    (analytic Jacobians, see above), assembling the dense damped normal
    equations ``(J^T J + lambda * diag(J^T J)) @ delta == -J^T r`` over every
    node but ``anchor``, solving them with ``numpy.linalg.solve``, and either
    accepting the step (retracting every active node's pose by its slice of
    ``delta`` - see :func:`_apply_update`) or rejecting it and raising the
    damping parameter - Levenberg-Marquardt's classic behaviour: small
    damping acts like Gauss-Newton (fast, but only trustworthy near a good
    fit); large damping acts like gradient descent (slow, but safe far from
    one). ``lambda`` is seeded at :data:`_INITIAL_DAMPING_TAU` times the
    largest diagonal entry of the initial ``J^T J``, shrinks by a
    gain-ratio-dependent factor (Nielsen's classic schedule) on every
    accepted step, and grows by :data:`_REJECT_DAMPING_GROWTH`, compounding,
    on every rejected one.

    Exactly like :func:`scanner3d.posegraph.optimise_poses`: ``anchor`` is
    never written to, at any point, so it comes back bit-for-bit identical to
    ``nodes[anchor]``, and a node with no incident edge at all is likewise
    left exactly where it started, because nothing constrains it - neither
    relaxation nor a normal-equation solve can move a parameter that never
    appears in the objective, and giving such a node its own free block would
    only make the linear system singular for no benefit. See
    :attr:`PoseGraphLMResult.limitations` for exactly which nodes, if any,
    this run left untouched for that reason.

    A run converges (``iterations_run`` less than ``iterations``,
    ``converged`` true) either the standard LM way - an ACCEPTED step whose
    cost reduction is below ``tolerance`` - or immediately, in zero
    iterations, when the weighted gradient ``J^T r`` is already smaller than
    :data:`_GRADIENT_TOLERANCE` in every component: exactly the case where
    every edge is already satisfied by the poses given, and no step would
    improve anything. See ``test_perfect_graph_converges_immediately_and_moves_nothing``
    in ``tests/test_posegraph_lm.py``.

    Dense, not sparse: the normal equations are a full
    ``6 * (active node count) x 6 * (active node count)`` system, refactored
    from scratch every iteration. That is a deliberate, stated choice for the
    scale this module targets - a few hundred poses, so at most a few
    thousand parameters - not an oversight; see the module docstring's "It
    does not do a sparse solve" section.
    """
    _require_numpy()
    import numpy as np

    _validate(nodes, edges, anchor, iterations, tolerance)

    rotations = [np.asarray(pose.rotation, dtype=np.float64) for pose in nodes]
    translations = [np.asarray(pose.translation, dtype=np.float64) for pose in nodes]

    touched: set[int] = set()
    for edge in edges:
        touched.add(edge.i)
        touched.add(edge.j)
    active_nodes = sorted(node for node in touched if node != anchor)
    param_index = {node: 6 * position for position, node in enumerate(active_nodes)}
    num_params = 6 * len(active_nodes)

    untouched = [node for node in range(len(nodes)) if node not in touched and node != anchor]

    limitations = [
        "This has no robust/outlier-rejecting mode: every edge is trusted exactly at its "
        "stated weight, always. scanner3d.posegraph's robust=True has no equivalent here.",
        "This is a dense normal-equation solve, refactored every iteration, not a sparse "
        "block solve exploiting the graph's mostly-local connectivity - a real optimisation "
        "left on the table at larger scale; see the module docstring.",
        "Levenberg-Marquardt is a local method: it is a much better local model of the "
        "objective than Gauss-Seidel relaxation (it uses curvature, not only neighbour "
        "predictions), but started far enough from a solution on a multi-basin objective it "
        "can still converge to the wrong one.",
    ]
    if untouched:
        limitations.append(
            f"node(s) {untouched} have no incident edge at all and were left exactly where "
            "they started, because nothing in the objective constrains them."
        )

    if not edges or num_params == 0:
        cost, _, _ = _cost_and_normal_equations(
            rotations, translations, edges, param_index, max(num_params, 1)
        )
        poses = tuple(nodes)
        return PoseGraphLMResult(
            poses=poses,
            error_history=(cost,),
            iterations_run=0,
            converged=True,
            anchor=anchor,
            edge_residuals=_edge_residuals(poses, edges),
            limitations=tuple(limitations),
        )

    cost, jt_j, jt_r = _cost_and_normal_equations(
        rotations, translations, edges, param_index, num_params
    )
    history = [cost]
    iterations_run = 0
    converged = False

    if float(np.max(np.abs(jt_r))) < _GRADIENT_TOLERANCE:
        converged = True
    else:
        damping = _INITIAL_DAMPING_TAU * max(float(np.max(np.diag(jt_j))), _MIN_DAMPING)
        growth = _REJECT_DAMPING_GROWTH
        for _ in range(iterations):
            diag_jt_j = np.diag(np.diag(jt_j))
            damped = jt_j + damping * diag_jt_j
            try:
                delta = np.linalg.solve(damped, -jt_r)
            except np.linalg.LinAlgError as exc:
                raise PoseGraphLMNotAllowed(
                    f"the damped normal-equation matrix is singular at damping={damping!r}: "
                    f"{exc}. This should not happen for damping > 0; it signals a degenerate "
                    "graph (e.g. every edge touching a node carries a zero-information "
                    "direction) rather than a transient numerical issue."
                ) from exc

            trial_rotations, trial_translations = _apply_update(
                rotations, translations, delta, param_index
            )
            trial_cost, trial_jt_j, trial_jt_r = _cost_and_normal_equations(
                trial_rotations, trial_translations, edges, param_index, num_params
            )
            predicted_reduction = float(delta @ (damping * diag_jt_j @ delta - jt_r))
            actual_reduction = cost - trial_cost
            iterations_run += 1

            if actual_reduction > 0.0 and predicted_reduction > 0.0:
                gain_ratio = actual_reduction / predicted_reduction
                rotations, translations = trial_rotations, trial_translations
                cost, jt_j, jt_r = trial_cost, trial_jt_j, trial_jt_r
                damping *= max(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3)
                damping = max(damping, _MIN_DAMPING)
                growth = _REJECT_DAMPING_GROWTH
                history.append(cost)
                gradient_settled = float(np.max(np.abs(jt_r))) < _GRADIENT_TOLERANCE
                if actual_reduction < tolerance or gradient_settled:
                    converged = True
                    break
            else:
                damping *= growth
                growth *= 2.0
                history.append(cost)

    poses = tuple(
        Rigid.of(_to_matrix3(rotations[index]), _to_point(translations[index]))
        for index in range(len(nodes))
    )
    return PoseGraphLMResult(
        poses=poses,
        error_history=tuple(history),
        iterations_run=iterations_run,
        converged=converged,
        anchor=anchor,
        edge_residuals=_edge_residuals(poses, edges),
        limitations=tuple(limitations),
    )
