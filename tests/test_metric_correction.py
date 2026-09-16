"""The per-unit metric correction: what it applies, and that it undoes the measured error."""

from __future__ import annotations

import math

import pytest

from scanner3d.depth_scale import SDK_MINI_FALLBACK_SCALE
from scanner3d.metric_correction import (
    CUE_BALL_20260911,
    FACTORY_DEPTH_SCALE,
    MetricCorrection,
    MetricCorrectionUnusable,
    calibrated_deprojection,
    calibration_source_statement,
)
from scanner3d.preparation import DepthBand, prepare_frame
from scanner3d.simulate import FACTORY_INTRINSICS, Sphere, intrinsics_of, render

RADIUS = 57.265 / 2
CENTRE = (52.0, -4.0, 209.0)


def test_the_record_applies_as_fy_and_depth_scale_only() -> None:
    correction = CUE_BALL_20260911
    assert correction.serial == "A22704522A6F00A64"
    assert correction.fx_factor == pytest.approx(1.0)
    assert correction.fy_factor == pytest.approx(0.9945 / 0.9985)
    assert correction.depth_factor == pytest.approx(1 / 0.9985)
    intrinsics, scale, applied = calibrated_deprojection()
    assert applied is correction
    assert intrinsics.values["fx"] == pytest.approx(FACTORY_INTRINSICS.values["fx"], abs=1e-3)
    assert intrinsics.values["fy"] == pytest.approx(
        FACTORY_INTRINSICS.values["fy"] * 0.9945 / 0.9985, abs=1e-3
    )
    assert intrinsics.values["cx"] == FACTORY_INTRINSICS.values["cx"]
    assert intrinsics.values["cy"] == FACTORY_INTRINSICS.values["cy"]
    assert scale == pytest.approx(FACTORY_DEPTH_SCALE / 0.9985)
    assert FACTORY_DEPTH_SCALE == SDK_MINI_FALLBACK_SCALE


def test_no_correction_is_the_factory_record_untouched() -> None:
    intrinsics, scale, applied = calibrated_deprojection(None)
    assert applied is None
    assert intrinsics is FACTORY_INTRINSICS
    assert scale == FACTORY_DEPTH_SCALE
    assert "unmodified" in calibration_source_statement(None)
    assert "metric correction for unit A22704522A6F00A64" in calibration_source_statement()


def test_implausible_factors_are_refused() -> None:
    with pytest.raises(MetricCorrectionUnusable):
        MetricCorrection("x", 1.2, 1.0, 1.0, "today", "test", ())
    with pytest.raises(MetricCorrectionUnusable):
        MetricCorrection("x", 1.0, math.nan, 1.0, "today", "test", ())


def test_as_dict_carries_measured_and_applied_factors() -> None:
    record = CUE_BALL_20260911.as_dict()
    assert record["measured_over_true"] == {"x": 0.9985, "y": 0.9945, "z": 0.9985}
    applied = record["applied"]
    assert isinstance(applied, dict)
    assert applied["fy_factor"] == pytest.approx(0.9945 / 0.9985)
    assert record["evidence"]


def _radial_error(points: list[tuple[float, float, float]]) -> float:
    """Largest distance of a prepared point from the true sphere's surface."""
    return max(
        abs(math.dist(point, CENTRE) - RADIUS) for point in points if 100.0 < point[2] < 300.0
    )


def _device_frame(correction: MetricCorrection) -> bytes:
    """A frame the way this unit would deliver it: the true sphere, with each
    camera axis reading ``measured over true`` times its real value."""
    lx, ly, lz = correction.x_over_true, correction.y_over_true, correction.z_over_true
    as_delivered = intrinsics_of(
        fx=FACTORY_INTRINSICS.values["fx"] * lx / lz,
        fy=FACTORY_INTRINSICS.values["fy"] * ly / lz,
        cx=FACTORY_INTRINSICS.values["cx"],
        cy=FACTORY_INTRINSICS.values["cy"],
    )
    return render(
        [Sphere(centre=CENTRE, radius=RADIUS)],
        intrinsics=as_delivered,
        depth_scale=FACTORY_DEPTH_SCALE / lz,
        seed=7,
    ).raw


def test_the_correction_undoes_the_measured_error_in_prepare_frame() -> None:
    raw = _device_frame(CUE_BALL_20260911)
    band = DepthBand(100, 300)
    intrinsics, scale, _ = calibrated_deprojection()
    corrected = prepare_frame(
        raw, target_points=2000, depth_band=band, intrinsics=intrinsics, depth_scale=scale
    )
    factory = prepare_frame(raw, target_points=2000, depth_band=band)
    # Quantisation to 0.005 mm codes bounds the corrected residual; the
    # uncorrected one carries the 0.55% vertical shortfall, 0.1 mm and more.
    assert _radial_error(list(corrected.points)) < 0.03
    assert _radial_error(list(factory.points)) > 0.1
