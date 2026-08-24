"""Prove the harness recovers what it should, and refuses to certify what it cannot.

Three claims carry this file, and none of them can be settled by reading the
module.

The first is that the fits are right. Every synthetic target here is built from
a centre, a radius, a normal or a separation chosen in advance, so the fit is
checked against an answer that was known before it ran - to machine precision,
because a noiseless target has no excuse.

The second is the one this module exists for. A shear ``x += k * z`` is applied
to those same targets, and the tests assert what the shear does rather than what
a reader might assume it does: a plane stays exactly as flat, a face pair
square-on to the camera keeps its separation to the last bit, a face pair turned
across the frame loses a factor of ``sqrt(1 + k^2)``, and a sphere's fitted
diameter moves by so little that a diameter check alone passes cleanly while the
residual has become entirely structured. That last case is the whole point: the
target that would produce a confident false pass is the one the indicator has to
catch, so it is tested with a shear small enough that nothing else could.

The third is that INCONCLUSIVE is reachable and reached. A noisy sphere whose
fitted diameter happens to land on the truth still cannot decide a tolerance
finer than its own residual, and this file asserts it says so rather than
passing.

No assertion here names a physical unit, and one test enforces that over
everything the module emits.
"""

from __future__ import annotations

import json
import math
import random
import re
import struct
from pathlib import Path

import pytest

from scanner3d.cloud_export import camera_points
from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame
from scanner3d.errors import DoctorStatus
from scanner3d.intrinsics import INTRINSICS_FORMAT, parse_intrinsics
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES
from scanner3d.validation import (
    MAX_FIT_POINTS,
    MAX_NORMAL_DISAGREEMENT,
    MAX_REFINEMENT_ITERATIONS,
    MIN_CAP_HALF_ANGLE,
    MIN_PLANE_POINTS,
    MIN_SHEAR_LEVERAGE,
    MIN_SPHERE_POINTS,
    SCHEMA_VERSION,
    SHEAR_CORRELATION_THRESHOLD,
    STATIC_LIMITATIONS,
    UNIVERSAL_GAPS,
    CaliperMeasurement,
    DegenerateFit,
    FittedQuantity,
    MeasurementUnusable,
    NonFiniteInput,
    PointSetUnusable,
    QuantityMismatch,
    Tolerance,
    ToleranceDeclaredLate,
    ToleranceUnusable,
    ValidationNotAllowed,
    Verdict,
    fit_plane,
    fit_sphere,
    plane_separation,
    validate,
)

#: The shear the factory record implies if the true principal point were the
#: frame centre: ``(320 - 42.43747329711914) / 898.6829833984375``. Quoted from
#: ``docs/depth-to-pointcloud.md`` and recomputed here so the two cannot drift.
FACTORY_SHEAR = (320.0 - 42.43747329711914) / 898.6829833984375

#: A shear far too small to move a fitted diameter anywhere a tolerance would
#: notice. It is the interesting case precisely because of that.
SUBTLE_SHEAR = 0.02

#: A target whose numbers are round and whose fit is therefore exact in binary
#: floating point. These are not this unit's numbers and mean nothing physical.
CENTRE = (3.0, -2.0, 50.0)
RADIUS = 7.5


# --------------------------------------------------------------------------
# Targets built from an answer that was known before the fit ran
# --------------------------------------------------------------------------


def sphere_band(
    count: int = 600,
    *,
    centre: tuple[float, float, float] = CENTRE,
    radius: float = RADIUS,
    z_from: float = -1.0,
    z_to: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Points on an exact sphere, spread over a band of the polar cosine.

    A Fibonacci spiral rather than a random cloud, so every test in this file
    runs on the same points every time and a failure is reproducible. ``z_from``
    and ``z_to`` select a band: ``(-1, 1)`` is the whole sphere and ``(-1, 0)``
    the hemisphere nearer the camera, which is all a single view could ever see.
    """
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(count):
        cosine = z_from + (z_to - z_from) * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        angle = golden * index
        points.append(
            (
                centre[0] + radius * ring * math.cos(angle),
                centre[1] + radius * ring * math.sin(angle),
                centre[2] + radius * cosine,
            )
        )
    return points


def orthogonal_great_circles(
    count: int = 200, *, centre: tuple[float, float, float] = CENTRE, radius: float = RADIUS
) -> list[tuple[float, float, float]]:
    """Two great circles at right angles: enough to fix a sphere, blind to a shear.

    One circle has ``dx = 0`` throughout and the other ``dz = 0``, so their
    ``dx * dz`` product is identically zero and the shear indicator has no
    leverage at all - while the two together still determine the sphere exactly.
    A point set that pins the diameter and cannot see a shear is the case the
    leverage floor exists for.
    """
    points = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        points.append(
            (centre[0], centre[1] + radius * math.cos(angle), centre[2] + radius * math.sin(angle))
        )
        points.append(
            (centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle), centre[2])
        )
    return points


def face(
    normal: tuple[float, float, float],
    offset: float,
    *,
    side: int = 12,
    span: float = 5.0,
    base: tuple[float, float, float] = (0.0, 0.0, 40.0),
) -> list[tuple[float, float, float]]:
    """An exactly flat square patch with a chosen normal, at a chosen offset."""
    length = math.sqrt(sum(component * component for component in normal))
    unit = tuple(component / length for component in normal)
    across = (unit[2], 0.0, -unit[0])
    if math.isclose(across[0], 0.0) and math.isclose(across[2], 0.0):
        across = (1.0, 0.0, 0.0)
    down = (
        unit[1] * across[2] - unit[2] * across[1],
        unit[2] * across[0] - unit[0] * across[2],
        unit[0] * across[1] - unit[1] * across[0],
    )
    points = []
    for i in range(side):
        for j in range(side):
            a = -span + 2.0 * span * i / (side - 1)
            b = -span + 2.0 * span * j / (side - 1)
            points.append(
                tuple(base[k] + a * across[k] + b * down[k] + offset * unit[k] for k in range(3))
            )
    return points  # type: ignore[return-value]


def sheared(
    points: list[tuple[float, float, float]], coefficient: float
) -> list[tuple[float, float, float]]:
    """Apply ``x += coefficient * z``: what a wrong principal point does to a cloud."""
    return [(x + coefficient * z, y, z) for x, y, z in points]


def jittered(
    points: list[tuple[float, float, float]], sigma: float, *, seed: int = 7
) -> list[tuple[float, float, float]]:
    """Isotropic Gaussian noise from a fixed seed, so a failure is reproducible."""
    rng = random.Random(seed)
    return [
        (x + rng.gauss(0.0, sigma), y + rng.gauss(0.0, sigma), z + rng.gauss(0.0, sigma))
        for x, y, z in points
    ]


def tolerance_for(fit, limit: float, quantity: FittedQuantity | None = None) -> Tolerance:
    """A tolerance stamped a second before ``fit``, i.e. properly predeclared."""
    return Tolerance(
        quantity=fit.quantity if quantity is None else quantity,
        limit=limit,
        declared_by="tests/test_validation.py",
        declared_at=fit.fitted_at - 1.0,
        rationale="a bound chosen for this test before the fit it judges was computed",
    )


def caliper(value: float, uncertainty: float = 0.01, label: str = "target") -> CaliperMeasurement:
    """A stand-in ground truth. The numbers are arbitrary and mean nothing physical."""
    return CaliperMeasurement(
        label=label,
        value=value,
        uncertainty=uncertainty,
        measured_by="tests/test_validation.py",
        instrument="a stand-in for a real instrument; this reading measures nothing",
    )


# --------------------------------------------------------------------------
# The fits recover answers that were known in advance
# --------------------------------------------------------------------------


def test_a_whole_synthetic_sphere_is_recovered_to_machine_precision():
    """Noiseless points, so the fit has no excuse for any residual at all."""
    fit = fit_sphere(sphere_band())

    assert fit.radius == pytest.approx(RADIUS, abs=1e-9)
    assert fit.diameter == pytest.approx(2.0 * RADIUS, abs=1e-9)
    assert fit.centre == pytest.approx(CENTRE, abs=1e-9)
    assert fit.residual_rms < 1e-9
    assert fit.residual_max < 1e-9
    assert fit.converged
    assert fit.iterations <= MAX_REFINEMENT_ITERATIONS
    assert fit.point_count == 600


def test_a_hemisphere_is_recovered_too_because_one_view_never_sees_more():
    """A single view of a ball sees at most half of it, so half must be enough."""
    fit = fit_sphere(sphere_band(z_from=-1.0, z_to=0.0))

    assert fit.radius == pytest.approx(RADIUS, abs=1e-9)
    assert fit.centre == pytest.approx(CENTRE, abs=1e-9)
    assert fit.residual_rms < 1e-9
    assert fit.cap_half_angle >= MIN_CAP_HALF_ANGLE
    assert fit.well_conditioned
    assert fit.inconclusive_reasons == ()


def test_a_sphere_fit_reports_both_residuals_and_its_own_z_span():
    """A fitted value with no residual beside it is an assertion, not a measurement."""
    fit = fit_sphere(jittered(sphere_band(), 0.05))

    assert fit.residual_rms > 0.0
    assert fit.residual_max >= fit.residual_rms
    assert fit.fit_uncertainty == pytest.approx(2.0 * fit.residual_rms)
    assert fit.z_min < CENTRE[2] < fit.z_max
    reported = fit.as_dict()
    assert reported["schema_version"] == SCHEMA_VERSION
    residual = reported["residual"]
    assert isinstance(residual, dict)
    assert residual["rms"] == fit.residual_rms
    assert residual["max"] == fit.residual_max
    assert "signed radial residual" in str(residual["note"])


def test_a_plane_is_recovered_with_a_normal_that_does_not_depend_on_the_run():
    """An eigenvector is defined up to sign; the reported one must not wobble."""
    points = face((0.0, 0.0, 1.0), 0.0)
    fit = fit_plane(points)

    assert fit.normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    assert fit.offset == pytest.approx(40.0, abs=1e-9)
    assert fit.residual_rms < 1e-12
    assert fit.point_count == len(points)
    assert fit_plane(list(reversed(points))).normal == pytest.approx(fit.normal, abs=1e-12)


def test_the_separation_between_two_faces_is_recovered_exactly():
    """The one flat-target experiment that yields a length a caliper can judge."""
    fit = plane_separation(face((0.0, 0.0, 1.0), 0.0), face((0.0, 0.0, 1.0), 4.0))

    assert fit.separation == pytest.approx(4.0, abs=1e-9)
    assert fit.normal_disagreement == pytest.approx(0.0, abs=1e-9)
    assert fit.faces_parallel
    assert fit.residual_rms < 1e-12
    assert fit.point_count == 2 * 144


# --------------------------------------------------------------------------
# What a shear actually does - the trap, measured rather than assumed
# --------------------------------------------------------------------------


def test_a_shear_leaves_a_plane_exactly_as_flat_as_it_was():
    """The trap itself: a plane fit cannot see a principal-point error at all."""
    flat = face((0.0, 0.0, 1.0), 0.0)
    clean = fit_plane(flat)
    bent = fit_plane(sheared(flat, FACTORY_SHEAR))

    assert clean.residual_rms < 1e-12
    assert bent.residual_rms < 1e-12
    assert bent.residual_max == pytest.approx(clean.residual_max, abs=1e-12)


def test_validating_a_plane_fit_is_inconclusive_and_names_both_reasons():
    """A flat target is run and recorded as deciding nothing, never quietly passed."""
    fit = fit_plane(face((0.0, 0.0, 1.0), 0.0))
    result = validate(fit, caliper(10.0), tolerance_for(fit, 1.0))

    assert fit.fitted_value is None
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is None
    assert result.fitted_value is None
    assert result.difference is None
    assert result.ratio is None
    assert result.establishes == ()
    joined = " ".join(result.inconclusive_reasons)
    assert "no length" in joined
    assert "shear maps planes to planes" in joined


def test_a_shear_does_not_move_faces_that_are_square_on_to_the_camera():
    """A gauge block presented the natural way is exactly blind to the shear."""
    a, b = face((0.0, 0.0, 1.0), 0.0), face((0.0, 0.0, 1.0), 4.0)
    clean = plane_separation(a, b)
    bent = plane_separation(sheared(a, FACTORY_SHEAR), sheared(b, FACTORY_SHEAR))

    assert bent.separation == pytest.approx(clean.separation, abs=1e-9)
    assert bent.shear_sensitivity == pytest.approx(0.0, abs=1e-12)
    assert not bent.sensitive_to_shear
    blind = " ".join(bent.does_not_establish)
    assert "blind to a shear" in blind


def test_a_shear_shrinks_a_separation_measured_across_the_frame():
    """Turn the same target and the shear becomes visible, by a known factor.

    Under ``x += k * z`` a plane's unit normal ``n`` becomes the unnormalised
    ``(n_x, n_y, n_z - k * n_x)``, so a separation along ``n`` is divided by the
    length of that vector. For faces normal to ``x`` the factor is exactly
    ``sqrt(1 + k^2)``, and that is the number asserted here rather than a
    direction of change.
    """
    a, b = face((1.0, 0.0, 0.0), 0.0), face((1.0, 0.0, 0.0), 4.0)
    clean = plane_separation(a, b)
    bent = plane_separation(sheared(a, FACTORY_SHEAR), sheared(b, FACTORY_SHEAR))

    assert clean.separation == pytest.approx(4.0, abs=1e-9)
    assert bent.separation == pytest.approx(4.0 / math.hypot(1.0, FACTORY_SHEAR), abs=1e-9)
    assert bent.sensitive_to_shear
    assert "sqrt(1 + (k * n_x)^2)" in " ".join(bent.does_not_establish)


def test_a_subtle_shear_is_caught_by_the_indicator_and_not_by_the_diameter():
    """The false pass this module exists to prevent, demonstrated on both halves.

    A shear of 0.02 moves the fitted diameter of this sphere by well under a
    thousandth of it - far inside any tolerance an operator would declare - while
    the residual becomes almost entirely structured. A harness that reported only
    the diameter would certify this cloud.
    """
    truth = sphere_band()
    fit = fit_sphere(sheared(truth, SUBTLE_SHEAR))

    assert fit.diameter == pytest.approx(2.0 * RADIUS, rel=1e-3)
    assert abs(fit.shear_correlation) > 0.99
    assert fit.shear_correlation**2 > 0.25
    assert fit.shear_indicator_reliable
    assert fit.shear_detected
    assert fit.shear_coefficient == pytest.approx(SUBTLE_SHEAR, rel=0.1)

    tolerance = tolerance_for(fit, 0.5)
    result = validate(fit, caliper(2.0 * RADIUS), tolerance)
    assert result.difference is not None
    assert abs(result.difference) < tolerance.limit
    assert result.combined_uncertainty < tolerance.limit
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is None
    assert "shear direction" in " ".join(result.inconclusive_reasons)


def test_the_indicator_reads_back_the_coefficient_that_would_explain_it():
    """It reports a size, in the same dimensionless x-per-z a wrong cx contributes."""
    for coefficient in (0.01, 0.02, 0.05):
        fit = fit_sphere(sheared(sphere_band(), coefficient))
        assert fit.shear_coefficient == pytest.approx(coefficient, rel=0.1)
        assert fit.shear_coefficient > 0.0


@pytest.mark.parametrize(
    ("z_to", "geometry"),
    [(0.5, "a two-thirds sphere"), (0.0, "a hemisphere"), (-0.5, "a 60-degree cap")],
)
def test_the_indicator_catches_the_factory_shear_on_a_partial_sphere(z_to, geometry):
    """The case a single view actually produces, and the one that used to slip.

    ``sphere_band()`` defaults to the *whole* sphere, which no camera can see. A
    single view of a ball yields a cap, and over a cap ``dx * dz`` is nearly a
    multiple of ``dx`` - so the fitted centre absorbs almost all of it. Projecting
    the residual onto the raw shape therefore reads near zero whatever the shear
    is: before this was fixed, a 60-degree cap carrying the *full* factory shear
    reported a correlation of 0.17, was not detected, and validated as a clean
    PASS against the true diameter. The indicator must survive the target being
    a real ball rather than a mathematical one.
    """
    fit = fit_sphere(sheared(sphere_band(1200, z_from=-1.0, z_to=z_to), FACTORY_SHEAR))

    assert fit.well_conditioned, f"{geometry} should not be excused as a narrow cap"
    assert fit.shear_indicator_reliable, f"{geometry} should have leverage to see with"
    assert abs(fit.shear_correlation) > 0.9
    assert fit.shear_detected, f"the full factory shear went unseen on {geometry}"
    assert fit.shear_coefficient == pytest.approx(FACTORY_SHEAR, rel=0.15)

    result = validate(fit, caliper(2.0 * RADIUS), tolerance_for(fit, 0.5))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is not True
    assert "shear direction" in " ".join(result.inconclusive_reasons)


@pytest.mark.parametrize("z_to", [-0.7, -0.87, -0.95])
def test_a_cap_too_narrow_to_see_a_shear_says_so_instead_of_passing(z_to):
    """The other half: when the fit really cannot see, it must not read as quiet.

    Below about a 60-degree cap there is almost no shear signal the fit's own
    centre and radius have not already absorbed. The honest report is that the
    indicator is blind, not that it looked and found nothing - so the leverage
    floor has to catch this even though the raw ``dx * dz`` spread over such a
    cap is as wide as over a whole sphere. Reading that raw spread was what let a
    sheared cap through, so the floor is only worth having on the stripped one.
    """
    fit = fit_sphere(sheared(sphere_band(1200, z_from=-1.0, z_to=z_to), FACTORY_SHEAR))

    assert fit.shear_leverage < MIN_SHEAR_LEVERAGE
    assert not fit.shear_indicator_reliable
    assert any("leverage" in reason for reason in fit.inconclusive_reasons)
    result = validate(fit, caliper(2.0 * RADIUS), tolerance_for(fit, 0.5))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is not True


def test_leverage_falls_as_coverage_falls_because_the_fit_absorbs_the_rest():
    """Leverage must rank geometries by what they can actually see.

    The raw spread of ``dx * dz`` does not: over a cap it is as large as over a
    whole sphere, which is exactly why reading it raw produced a confident quiet
    indicator on a sheared cap. Ranked leverage is the property that makes the
    floor mean something.
    """
    whole = fit_sphere(sphere_band(1200)).shear_leverage
    half = fit_sphere(sphere_band(1200, z_from=-1.0, z_to=0.0)).shear_leverage
    cap = fit_sphere(sphere_band(1200, z_from=-1.0, z_to=-0.5)).shear_leverage

    assert whole > half > cap
    assert whole == pytest.approx(1.0 / math.sqrt(15.0), rel=0.1)


@pytest.mark.parametrize("z_to", [1.0, 0.0, -0.5])
def test_an_unsheared_noisy_sphere_does_not_fire_the_indicator(z_to):
    """Isotropic noise is not structure; a harness that cried shear at it is useless.

    Run over the partial spheres as well as the whole one, because a sensitivity
    fix that bought its sensitivity by firing at everything would be no better
    than the false negative it replaced.
    """
    fit = fit_sphere(jittered(sphere_band(1200, z_from=-1.0, z_to=z_to), 0.25))

    assert fit.residual_rms > 0.1
    assert abs(fit.shear_correlation) < SHEAR_CORRELATION_THRESHOLD
    assert fit.shear_indicator_reliable
    assert not fit.shear_detected


def test_a_point_set_with_no_leverage_reports_that_it_cannot_see_a_shear():
    """A quiet indicator with no leverage is silence, not evidence of absence."""
    fit = fit_sphere(orthogonal_great_circles())

    assert fit.radius == pytest.approx(RADIUS, abs=1e-9)
    assert fit.shear_leverage < MIN_SHEAR_LEVERAGE
    assert not fit.shear_indicator_reliable
    assert not fit.shear_detected
    assert fit.well_conditioned
    assert any("leverage" in reason for reason in fit.inconclusive_reasons)


def test_a_narrow_cap_is_reported_as_weakly_determined():
    """Over a small cap a bigger sphere further away fits the same points."""
    fit = fit_sphere(sphere_band(400, z_from=-1.0, z_to=-0.97))

    assert fit.residual_rms < 1e-9
    assert fit.cap_half_angle < MIN_CAP_HALF_ANGLE
    assert not fit.well_conditioned
    assert any("cap of half-angle" in reason for reason in fit.inconclusive_reasons)


def test_faces_that_are_not_parallel_do_not_have_one_separation():
    wedge = plane_separation(
        face((0.0, 0.0, 1.0), 0.0), face((math.sin(0.15), 0.0, math.cos(0.15)), 4.0)
    )

    assert wedge.normal_disagreement == pytest.approx(0.15, abs=1e-9)
    assert wedge.normal_disagreement > MAX_NORMAL_DISAGREEMENT
    assert not wedge.faces_parallel
    result = validate(wedge, caliper(4.0), tolerance_for(wedge, 0.5))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert any("disagree" in reason for reason in result.inconclusive_reasons)


# --------------------------------------------------------------------------
# The verdict rule
# --------------------------------------------------------------------------


def test_a_clean_comparison_passes_and_states_how_narrow_the_pass_is():
    fit = fit_sphere(sphere_band())
    tolerance = tolerance_for(fit, 0.05)
    result = validate(fit, caliper(2.0 * RADIUS, 0.01), tolerance)

    assert result.verdict is Verdict.PASS
    assert result.passed is True
    assert result.difference == pytest.approx(0.0, abs=1e-9)
    assert result.ratio == pytest.approx(1.0, abs=1e-9)
    assert result.difference_high is not None
    assert result.difference_high <= tolerance.limit

    established = " ".join(result.establishes)
    assert "agrees with the caliper reading" in established
    assert "spanning z from" in established
    gaps = " ".join(result.does_not_establish)
    assert "nothing about linearity" in gaps
    assert "nothing about lens distortion" in gaps
    assert "nothing about the principal point" in gaps
    assert "No physical unit" in gaps or "no physical unit" in gaps


def test_a_real_disagreement_fails_with_the_whole_band_outside_the_limit():
    fit = fit_sphere(sphere_band())
    result = validate(fit, caliper(12.0, 0.01), tolerance_for(fit, 0.05))

    assert result.verdict is Verdict.FAIL
    assert result.passed is False
    assert result.difference == pytest.approx(3.0, abs=1e-6)
    assert result.difference_low is not None
    assert result.difference_low > result.tolerance_limit
    assert result.inconclusive_reasons == ()
    assert "disagreement is real" in " ".join(result.establishes)


def test_a_band_that_straddles_the_limit_from_outside_decides_nothing():
    """A difference past the limit is not a fail while the band reaches back inside."""
    fit = fit_sphere(sphere_band())
    result = validate(fit, caliper(2.0 * RADIUS + 0.06, 0.02), tolerance_for(fit, 0.05))

    assert result.difference is not None
    assert abs(result.difference) > result.tolerance_limit
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is None
    assert result.difference_low is not None
    assert result.difference_high is not None
    assert result.difference_low <= result.tolerance_limit < result.difference_high
    assert any("straddles" in reason for reason in result.inconclusive_reasons)


def test_a_difference_inside_the_limit_is_not_enough_on_its_own_to_pass():
    """The other half of the same rule, and the half that prevents a false pass.

    The bare difference here is comfortably inside the declared tolerance, so a
    harness that compared only ``|fitted - measured|`` against the limit would
    certify this run. The band is not inside: the combined uncertainty reaches
    past the limit, so the run is equally consistent with a real disagreement
    that fails, and it asserts neither.
    """
    fit = fit_sphere(sphere_band())
    result = validate(fit, caliper(2.0 * RADIUS + 0.04, 0.02), tolerance_for(fit, 0.05))

    assert result.difference is not None
    assert abs(result.difference) < result.tolerance_limit
    assert result.combined_uncertainty < result.tolerance_limit
    assert result.difference_high is not None
    assert result.difference_high > result.tolerance_limit
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is None
    assert result.establishes == ()
    assert any("straddles" in reason for reason in result.inconclusive_reasons)


def test_a_noisy_sphere_is_inconclusive_rather_than_a_false_pass():
    """The fitted diameter lands on the truth; the experiment still cannot say so.

    This is the case a naive harness gets wrong. The noise is large enough that
    the fit's own residual swamps the declared tolerance, so the agreement is a
    coincidence the run has no power to distinguish from luck - and the honest
    report is that it decided nothing, not that it passed.
    """
    fit = fit_sphere(jittered(sphere_band(), 0.25))
    tolerance = tolerance_for(fit, 0.05)
    result = validate(fit, caliper(2.0 * RADIUS, 0.01), tolerance)

    assert result.difference is not None
    assert abs(result.difference) < tolerance.limit
    assert result.combined_uncertainty > tolerance.limit
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is None
    assert result.establishes == ()
    assert any("unreachable" in reason for reason in result.inconclusive_reasons)


def test_uncertainties_combine_in_quadrature_and_the_record_says_so():
    fit = fit_sphere(jittered(sphere_band(), 0.05))
    result = validate(fit, caliper(2.0 * RADIUS, 0.03), tolerance_for(fit, 5.0))

    assert result.fit_uncertainty == pytest.approx(2.0 * fit.residual_rms)
    assert result.caliper_uncertainty == pytest.approx(0.03)
    assert result.combined_uncertainty == pytest.approx(
        math.sqrt(result.fit_uncertainty**2 + result.caliper_uncertainty**2)
    )
    assert result.combined_uncertainty < result.fit_uncertainty + result.caliper_uncertainty
    assert "quadrature" in result.combined_uncertainty_method


def test_the_ratio_is_reported_because_it_is_what_a_scale_error_shows_up_in():
    fit = fit_sphere(sphere_band())
    result = validate(fit, caliper(RADIUS, 0.01), tolerance_for(fit, 0.05))

    assert result.ratio == pytest.approx(2.0, abs=1e-9)
    comparison = result.as_dict()["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["ratio"] == pytest.approx(2.0, abs=1e-9)
    assert "the model itself is wrong" in str(comparison["ratio_note"])


# --------------------------------------------------------------------------
# Predeclaration
# --------------------------------------------------------------------------


def test_a_tolerance_declared_after_the_fit_is_refused():
    """A bound chosen once the numbers are in describes them and tests nothing."""
    fit = fit_sphere(sphere_band())
    late = Tolerance(
        quantity=FittedQuantity.SPHERE_DIAMETER,
        limit=0.05,
        declared_by="tests/test_validation.py",
        declared_at=fit.fitted_at + 1.0,
        rationale="declared after seeing the fit, which is the thing being refused",
    )

    with pytest.raises(ToleranceDeclaredLate) as raised:
        validate(fit, caliper(2.0 * RADIUS), late)
    assert "not earlier than the result it judges" in str(raised.value)


def test_a_tolerance_declared_at_the_very_moment_of_the_fit_is_refused_too():
    """Simultaneity does not establish order, so it is not accepted as order."""
    fit = fit_sphere(sphere_band())
    simultaneous = Tolerance(
        quantity=FittedQuantity.SPHERE_DIAMETER,
        limit=0.05,
        declared_by="tests/test_validation.py",
        declared_at=fit.fitted_at,
        rationale="stamped at the same instant as the fit",
    )

    with pytest.raises(ToleranceDeclaredLate):
        validate(fit, caliper(2.0 * RADIUS), simultaneous)


def test_declare_stamps_the_clock_so_the_honest_path_is_the_short_one():
    before = Tolerance.declare(
        quantity=FittedQuantity.SPHERE_DIAMETER,
        limit=0.05,
        declared_by="tests/test_validation.py",
        rationale="declared before the target was fitted",
    )
    fit = fit_sphere(sphere_band())

    assert before.declared_at < fit.fitted_at
    assert validate(fit, caliper(2.0 * RADIUS), before).verdict is Verdict.PASS


def test_a_tolerance_cannot_be_moved_onto_a_different_quantity():
    fit = fit_sphere(sphere_band())
    for_a_plane = tolerance_for(fit, 0.05, quantity=FittedQuantity.PLANE_SEPARATION)

    with pytest.raises(QuantityMismatch) as raised:
        validate(fit, caliper(2.0 * RADIUS), for_a_plane)
    assert "plane_separation" in str(raised.value)
    assert "sphere_diameter" in str(raised.value)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_an_empty_point_set_is_refused_by_every_fit():
    for fitter in (fit_sphere, fit_plane):
        with pytest.raises(PointSetUnusable) as raised:
            fitter([])
        assert "given none" in str(raised.value)
    with pytest.raises(PointSetUnusable):
        plane_separation([], face((0.0, 0.0, 1.0), 0.0))


def test_a_point_count_that_cannot_produce_a_residual_is_refused():
    """A fit with as many points as parameters cannot fail, so it is not accepted."""
    with pytest.raises(PointSetUnusable) as raised:
        fit_sphere(sphere_band(MIN_SPHERE_POINTS - 1))
    assert f"at least {MIN_SPHERE_POINTS}" in str(raised.value)

    with pytest.raises(PointSetUnusable):
        fit_plane(face((0.0, 0.0, 1.0), 0.0, side=2)[: MIN_PLANE_POINTS - 1])

    assert fit_sphere(sphere_band(MIN_SPHERE_POINTS)).point_count == MIN_SPHERE_POINTS


def test_a_point_set_past_the_ceiling_is_refused_before_a_coordinate_is_read():
    class Unread:
        """Claims to be past the ceiling and screams if anything looks inside it."""

        def __len__(self) -> int:
            return MAX_FIT_POINTS + 1

        def __getitem__(self, index):
            raise AssertionError("the ceiling must be enforced before any point is read")

    with pytest.raises(PointSetUnusable) as raised:
        fit_sphere(Unread())  # type: ignore[arg-type]
    assert str(MAX_FIT_POINTS) in str(raised.value)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_coordinate_is_refused_before_any_moment_is_taken(bad):
    points = sphere_band(50)
    points[17] = (points[17][0], bad, points[17][2])

    with pytest.raises(NonFiniteInput) as raised:
        fit_sphere(points)
    assert "point 17" in str(raised.value)


def test_coplanar_points_do_not_determine_a_sphere():
    flat = [(float(i), float(j), 5.0) for i in range(6) for j in range(6)]

    with pytest.raises(DegenerateFit) as raised:
        fit_sphere(flat)
    assert "singular" in str(raised.value)


def test_collinear_points_do_not_determine_a_plane():
    line = [(float(i), 2.0 * i, 3.0 * i) for i in range(10)]

    with pytest.raises(DegenerateFit) as raised:
        fit_plane(line)
    assert "collinear or coincident" in str(raised.value)


def test_coincident_points_do_not_determine_a_plane():
    with pytest.raises(DegenerateFit):
        fit_plane([(1.0, 2.0, 3.0)] * 12)


@pytest.mark.parametrize("uncertainty", [0.0, -0.1, math.nan, math.inf])
def test_a_ground_truth_without_a_usable_uncertainty_cannot_be_built(uncertainty):
    """A comparison against a reference of unstated precision cannot fail honestly."""
    with pytest.raises(MeasurementUnusable) as raised:
        CaliperMeasurement(
            label="target",
            value=10.0,
            uncertainty=uncertainty,
            measured_by="tests/test_validation.py",
            instrument="a stand-in",
        )
    assert "uncertainty" in str(raised.value)


@pytest.mark.parametrize("value", [0.0, -3.0, math.nan])
def test_a_ground_truth_that_is_not_a_positive_reading_is_refused(value):
    with pytest.raises(MeasurementUnusable):
        CaliperMeasurement(
            label="target",
            value=value,
            uncertainty=0.01,
            measured_by="tests/test_validation.py",
            instrument="a stand-in",
        )


@pytest.mark.parametrize("field", ["label", "measured_by", "instrument"])
def test_a_ground_truth_with_blank_provenance_is_refused(field):
    fields = {
        "label": "target",
        "value": 10.0,
        "uncertainty": 0.01,
        "measured_by": "tests/test_validation.py",
        "instrument": "a stand-in",
    }
    fields[field] = "   "

    with pytest.raises(MeasurementUnusable) as raised:
        CaliperMeasurement(**fields)  # type: ignore[arg-type]
    assert field in str(raised.value)


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf])
def test_a_tolerance_that_cannot_decide_anything_is_refused(limit):
    with pytest.raises(ToleranceUnusable):
        Tolerance(
            quantity=FittedQuantity.SPHERE_DIAMETER,
            limit=limit,
            declared_by="tests/test_validation.py",
            declared_at=1.0,
            rationale="a bound",
        )


@pytest.mark.parametrize("field", ["declared_by", "rationale"])
def test_a_tolerance_with_no_owner_or_no_reason_is_refused(field):
    fields: dict[str, object] = {
        "quantity": FittedQuantity.SPHERE_DIAMETER,
        "limit": 0.05,
        "declared_by": "tests/test_validation.py",
        "declared_at": 1.0,
        "rationale": "a bound",
    }
    fields[field] = ""

    with pytest.raises(ToleranceUnusable) as raised:
        Tolerance(**fields)  # type: ignore[arg-type]
    assert field in str(raised.value)


@pytest.mark.parametrize("stamp", [0.0, -1.0, math.nan])
def test_a_tolerance_with_an_unusable_stamp_cannot_establish_order(stamp):
    with pytest.raises(ToleranceUnusable):
        Tolerance(
            quantity=FittedQuantity.SPHERE_DIAMETER,
            limit=0.05,
            declared_by="tests/test_validation.py",
            declared_at=stamp,
            rationale="a bound",
        )


def test_every_refusal_shares_one_base_and_one_doctor_status():
    """A caller catching the base has caught every refusal this module makes."""
    for factory in (
        lambda: fit_sphere([]),
        lambda: fit_plane([(1.0, 2.0, 3.0)] * 12),
        lambda: CaliperMeasurement("t", 1.0, 0.0, "who", "what"),
        lambda: Tolerance(FittedQuantity.SPHERE_DIAMETER, -1.0, "who", 1.0, "why"),
    ):
        with pytest.raises(ValidationNotAllowed) as raised:
            factory()
        assert raised.value.status is DoctorStatus.UNSUPPORTED_MODE


# --------------------------------------------------------------------------
# What travels with a result
# --------------------------------------------------------------------------


def test_a_result_carries_the_standing_limitations_and_the_fit_s_own():
    fit = fit_sphere(sphere_band())
    result = validate(fit, caliper(2.0 * RADIUS), tolerance_for(fit, 0.05))

    for limitation in STATIC_LIMITATIONS:
        assert limitation in result.limitations
    for limitation in fit.limitations:
        assert limitation in result.limitations
    for gap in UNIVERSAL_GAPS:
        assert gap in result.does_not_establish
    assert len(set(result.limitations)) == len(result.limitations)
    assert len(set(result.does_not_establish)) == len(result.does_not_establish)


def test_a_result_serialises_to_json_with_everything_a_reader_needs():
    fit = fit_sphere(sphere_band())
    result = validate(fit, caliper(2.0 * RADIUS), tolerance_for(fit, 0.05))
    reported = json.loads(json.dumps(result.as_dict()))

    assert reported["schema_version"] == SCHEMA_VERSION
    assert reported["verdict"] == "pass"
    assert reported["passed"] is True
    assert reported["quantity"] == "sphere_diameter"
    assert reported["uncertainty"]["method"].startswith("in quadrature")
    assert reported["fit"]["shear_indicator"]["detected"] is False
    assert reported["measurement"]["measured_by"] == "tests/test_validation.py"
    assert reported["tolerance"]["declared_by"] == "tests/test_validation.py"
    assert reported["limitations"]
    assert reported["does_not_establish"]


#: Word-bounded so that ``diameter`` and ``parameter`` are not mistaken for a
#: unit. The point of the scan is that this module must never name one.
UNIT_WORDS = re.compile(
    r"\b(mm|cm|um|nm|m|millimet\w*|centimet\w*|micromet\w*|micron\w*|metres?|meters?|"
    r"inch|inches|thou|mils?)\b",
    re.IGNORECASE,
)


def test_no_string_this_module_emits_names_a_physical_unit():
    """The defining discipline, enforced over everything a consumer can read.

    No distance this scanner produces has ever been compared against a physical
    measurement, so naming a unit anywhere - even in a caveat - would be an
    assertion the project has not earned.
    """
    fit = fit_sphere(sheared(sphere_band(), SUBTLE_SHEAR))
    plane = fit_plane(face((0.0, 0.0, 1.0), 0.0))
    pair = plane_separation(face((1.0, 0.0, 0.0), 0.0), face((1.0, 0.0, 0.0), 4.0))
    emitted = [
        json.dumps(fit.as_dict()),
        json.dumps(plane.as_dict()),
        json.dumps(pair.as_dict()),
        json.dumps(validate(fit, caliper(2.0 * RADIUS), tolerance_for(fit, 0.5)).as_dict()),
        json.dumps(validate(pair, caliper(4.0), tolerance_for(pair, 0.05)).as_dict()),
        json.dumps(validate(plane, caliper(4.0), tolerance_for(plane, 0.05)).as_dict()),
        " ".join(STATIC_LIMITATIONS),
        " ".join(UNIVERSAL_GAPS),
    ]

    for payload in emitted:
        found = UNIT_WORDS.findall(payload)
        assert found == [], f"a physical unit was named: {found}"


# --------------------------------------------------------------------------
# One real captured frame: structure only, no accuracy claim of any kind
# --------------------------------------------------------------------------

#: The nine real captures committed to this repository. They cannot be
#: regenerated without the physical scanner, which is why they are evidence
#: rather than scratch.
FIXTURE_FRAMES = Path(__file__).parent / "fixtures" / "frames"

#: The unit's own factory record, as ``docs/depth-to-pointcloud.md`` quotes it.
REAL_FX = REAL_FY = 898.6829833984375
REAL_CX = 42.43747329711914
REAL_CY = 192.8326873779297
REAL_SCALE = 0.004999999888241291

#: Every twentieth point of a real frame. The fits here are O(n) per pass and the
#: whole file has a runtime budget; a structural smoke test needs a real shape,
#: not every sample of one.
SMOKE_STRIDE = 20


def real_frame_points() -> list[tuple[float, float, float]] | None:
    """One real frame, deprojected through the real path, or ``None`` if absent."""
    if not FIXTURE_FRAMES.is_dir():
        return None
    candidates = [
        path
        for path in sorted(FIXTURE_FRAMES.glob("*.bin"))
        if path.stat().st_size == EXPECTED_FRAME_BYTES
    ]
    if not candidates:
        return None
    blob = struct.pack(
        INTRINSICS_FORMAT, 640, 400, REAL_FX, 0.0, REAL_CX, 0.0, REAL_FY, REAL_CY, 0.0, 0.0, 1.0
    )
    intrinsics = parse_intrinsics(blob)
    assert intrinsics is not None
    conversion = deproject_depth_frame(
        candidates[-1].read_bytes(),
        intrinsics=intrinsics,
        depth_scale=REAL_SCALE,
        unit=UNVALIDATED_UNIT,
    )
    return list(camera_points(conversion))[::SMOKE_STRIDE]


def test_a_real_captured_frame_fits_without_error_and_decides_nothing():
    """Structure only. Not one assertion here is about accuracy, and none can be.

    The points come from a real capture of an unknown scene - no sphere, no gauge
    block, no reference of any kind - so the only honest claims are that the
    plumbing runs on real data, that the residuals are finite numbers, and that a
    verdict over them is INCONCLUSIVE. A test that asserted a fitted dimension
    here would be inventing the ground truth this repository does not have.
    """
    points = real_frame_points()
    if points is None:
        pytest.skip(f"no real 512000-byte depth frame is committed under {FIXTURE_FRAMES}")
    assert len(points) > MIN_PLANE_POINTS

    fit = fit_plane(points)

    assert fit.point_count == len(points)
    assert math.isfinite(fit.residual_rms)
    assert math.isfinite(fit.residual_max)
    assert fit.residual_max >= fit.residual_rms >= 0.0
    assert math.isclose(sum(component**2 for component in fit.normal), 1.0, abs_tol=1e-9)
    assert fit.z_min <= fit.z_max
    assert fit.fitted_value is None

    result = validate(fit, caliper(10.0), tolerance_for(fit, 1.0))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.passed is None
    assert result.establishes == ()

    assert json.loads(json.dumps(result.as_dict()))["passed"] is None


def test_the_fits_are_deterministic_over_a_real_frame():
    """Two runs over the same points must be the same fit, bit for bit."""
    points = real_frame_points()
    if points is None:
        pytest.skip(f"no real 512000-byte depth frame is committed under {FIXTURE_FRAMES}")

    first, second = fit_plane(points), fit_plane(points)
    assert first.normal == second.normal
    assert first.offset == second.offset
    assert first.residual_rms == second.residual_rms
