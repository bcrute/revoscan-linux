import copy
import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_normal_confidence_review")


def field(radius=5, angle=10.0):
    area = (radius * 2 + 1) ** 2
    return dict(
        radius=radius,
        normal=[math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0],
        status="usable",
        support_count=area,
        required_support=(3 * area + 3) // 4,
        perpendicular_rms=0.0,
    )


@pytest.mark.parametrize("radius", [5, 10])
@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_formula_and_no_input_mutation(module, radius, weight):
    values = field(radius, 10.0)
    original = copy.deepcopy(values)
    normal, info = module.combine_confidence((1.0, 0.0, 0.0), values, maximum_weight=weight)
    expected_weight = weight * (1 - (10 / 30) ** 2) ** 2
    assert info["weight"] == pytest.approx(expected_weight, abs=1e-14)
    expected = [
        1 - expected_weight + expected_weight * math.cos(math.radians(10)),
        expected_weight * math.sin(math.radians(10)),
        0.0,
    ]
    length = math.hypot(*expected)
    assert normal == pytest.approx([x / length for x in expected], abs=1e-14)
    assert info["mode"] == "weighted" and values == original


@pytest.mark.parametrize("radius", [5, 10])
@pytest.mark.parametrize("boundary", ["support", "rms", "angle"])
def test_zero_and_near_zero_boundaries(module, radius, boundary):
    values = field(radius, 10.0)
    if boundary == "support":
        values["support_count"] = values["required_support"]
    elif boundary == "rms":
        values["perpendicular_rms"] = 0.25
    else:
        values = field(radius, 30.000001)
    normal, info = module.combine_confidence((1.0, 0.0, 0.0), values, maximum_weight=1.0)
    assert normal == (1.0, 0.0, 0.0) and info["weight"] == 0.0 and info["mode"] == "fallback"
    if boundary == "support":
        values["support_count"] += 1
    elif boundary == "rms":
        values["perpendicular_rms"] -= 1e-7
    else:
        values = field(radius, 29.999999)
    normal, info = module.combine_confidence((1.0, 0.0, 0.0), values, maximum_weight=1.0)
    assert normal is not None and math.hypot(*normal) == pytest.approx(1.0)
    limit = 0.034 if boundary == "support" else 1e-10
    assert 0.0 <= info["weight"] <= limit


@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_missing_and_opposed_signs(module, weight):
    values = field(angle=180.0)
    assert module.combine_confidence(None, values, maximum_weight=weight)[0] is None
    normal, info = module.combine_confidence((1.0, 0.0, 0.0), values, maximum_weight=weight)
    assert normal == pytest.approx((1.0, 0.0, 0.0), abs=1e-15)
    assert info["weight"] == pytest.approx(weight)


@pytest.mark.parametrize(
    "status",
    ["border", "center-missing-or-band", "insufficient-support", "degenerate", "nonplanar"],
)
def test_unavailable_falls_back_exactly(module, status):
    values = field() | dict(status=status, normal=None)
    assert module.combine_confidence((0.6, 0.0, -0.8), values, maximum_weight=1.0)[0] == (
        0.6,
        0.0,
        -0.8,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        dict(radius=2),
        dict(radius=True),
        dict(status="unknown"),
        dict(support_count=90),
        dict(support_count=122),
        dict(support_count=100.0),
        dict(required_support=90),
        dict(perpendicular_rms=-0.1),
        dict(perpendicular_rms=0.25001),
        dict(perpendicular_rms=float("nan")),
        dict(perpendicular_rms=True),
        dict(normal=None),
        dict(normal=[2.0, 0.0, 0.0]),
        dict(status="border"),
    ],
)
def test_malformed_field(module, mutation):
    with pytest.raises(ValueError):
        module.combine_confidence((1.0, 0.0, 0.0), field() | mutation, maximum_weight=1.0)


@pytest.mark.parametrize("weight", [0.0, 0.1, True, float("nan"), 2.0])
def test_malformed_weight(module, weight):
    with pytest.raises(ValueError):
        module.combine_confidence((1.0, 0.0, 0.0), field(), maximum_weight=weight)
