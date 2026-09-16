import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_normal_hybrid_review")


@pytest.mark.parametrize("weight", [0.5, 1.0])
@pytest.mark.parametrize("organized", [None, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)])
def test_original_missing_stays_missing(module, weight, organized):
    assert module.combine(None, organized, weight=weight) == (None, "missing")


@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_fallback_exact_and_input_immutable(module, weight):
    original = [0.6, 0.0, -0.8]
    assert module.combine(original, None, weight=weight) == ((0.6, 0.0, -0.8), "fallback")
    assert original == [0.6, 0.0, -0.8]


@pytest.mark.parametrize("weight", [0.5, 1.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_opposed_signs_do_not_cancel(module, weight, sign):
    normal, mode = module.combine((1.0, 0.0, 0.0), (sign, 0.0, 0.0), weight=weight)
    assert normal == (1.0, 0.0, 0.0) and mode == ("blend" if weight == 0.5 else "organized")


def test_orthogonal_blend(module):
    normal, mode = module.combine((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), weight=0.5)
    assert normal == pytest.approx((math.sqrt(0.5), math.sqrt(0.5), 0.0)) and mode == "blend"


@pytest.mark.parametrize("weight", [0, 0.1, 0.75, 2.0, True, float("nan"), float("inf"), "1"])
def test_bad_weight(module, weight):
    with pytest.raises(ValueError):
        module.combine((1.0, 0.0, 0.0), None, weight=weight)


@pytest.mark.parametrize(
    "vector",
    [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (float("nan"), 0.0, 1.0),
        (float("inf"), 0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (True, 0.0, 0.0),
        "100",
        ("1", 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("position", [0, 1])
def test_bad_direction(module, vector, position):
    pair = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    pair[position] = vector
    with pytest.raises(ValueError):
        module.combine(*pair, weight=0.5)
