"""M9 combines two exact prior interventions; it does not create another budget."""

import importlib
import math
from itertools import pairwise
from pathlib import Path

import pytest


@pytest.fixture
def dense(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_dense_model_sampling")


@pytest.mark.parametrize("sampler", ["stride", "grid"])
@pytest.mark.parametrize("solver", ["observe-origin", "centered"])
def test_combination_capacity_is_exactly_m7(dense, sampler, solver):
    factors = dense.Factors(sampler, "pooled", "residual")
    for profile in dense.PROFILES:
        dense.validate(profile, factors, solver, None, "source-1800")
    points, policy = dense.settings("source-1800", factors, solver, None)
    assert points == 1800 and policy == dense.EXPECTED_POLICY
    before, after = dense.DEFAULT_POLICY.as_dict(), policy.as_dict()
    assert {key for key in before if before[key] != after[key]} == {
        "candidate_source_points",
        "retained_cloud_points",
    }
    assert after["model_reference_points"] == 1800


def test_dense_indices_are_exactly_the_m8_rule_where_both_are_defined(dense):
    original = importlib.import_module("scanner_model_sampling")
    for size in (0, 1, 224, 225, 226, 460, 599, 600, 601, 899, 900):
        for frames in range(1, 9):
            limit = 1800 // frames
            for position in range(frames):
                assert dense.indices(
                    size, limit, position=position, frames=frames
                ) == original.indices(size, limit, position=position, frames=frames)


def test_dense_indices_fill_but_never_exceed_the_original_model_budget(dense):
    for size in (*range(900, 1801, 53), 901, 1799, 1800):
        for frames in range(1, 9):
            limit = 1800 // frames
            actual = [dense.indices(size, limit, position=p, frames=frames) for p in range(frames)]
            assert sum(map(len, actual)) == min(size, limit) * frames <= 1800
            for selected in actual:
                assert selected == tuple(sorted(set(selected)))
                assert len(selected) == min(size, limit)
                assert all(0 <= i < size for i in selected)
                if size > limit:
                    assert set(b - a for a, b in pairwise(selected)) <= {
                        math.floor(size / limit),
                        math.ceil(size / limit),
                    }
                    assert selected[0] < math.ceil(size / limit)
                    assert size - selected[-1] <= math.ceil(size / limit)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(size=1801),
        dict(size=-1),
        dict(size=True),
        dict(limit=0),
        dict(limit=1801),
        dict(frames=0),
        dict(frames=9),
        dict(position=-1),
        dict(position=8),
    ],
)
def test_dense_integer_bounds_refuse(dense, overrides):
    with pytest.raises(ValueError, match="bounded integer"):
        dense.indices(**(dict(size=1800, limit=225, position=0, frames=8) | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(profile="guess"),
        dict(source_profile=None),
        dict(source_profile="observe-900"),
        dict(model_profile="observed-point"),
        dict(solver_profile="symmetric"),
        dict(factors=("rank", "pooled", "residual")),
        dict(factors=("grid", "native", "residual")),
        dict(factors=("grid", "pooled", "settling")),
    ],
)
def test_combination_refuses_undeclared_mixtures(dense, overrides):
    values = (
        dict(
            profile="stratified-model",
            source_profile="source-1800",
            model_profile=None,
            solver_profile="centered",
            factors=("grid", "pooled", "residual"),
        )
        | overrides
    )
    values["factors"] = dense.Factors(*values["factors"])
    with pytest.raises(ValueError):
        dense.validate(**values)
