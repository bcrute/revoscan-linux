"""G1 never uses validation to trigger or hide a failed correction."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_loop_policy")


def rows():
    return [
        dict(
            role="validation" if i % 2 else "fit",
            exclusion_reasons=[],
            before=dict(agreement=0.98),
            after=dict(agreement=0.98),
        )
        for i in range(12)
    ]


def test_activation_ignores_held_out_and_excluded_pairs(policy):
    checks = rows()
    checks[1]["before"]["agreement"] = 0.1
    checks[0]["before"]["agreement"] = 0.1
    checks[0]["exclusion_reasons"] = ["validation_proximity"]
    assert policy.activation(checks) == "keep"
    checks[2]["before"]["agreement"] = 0.899
    assert policy.activation(checks) == "correct"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, 1.1, True])
def test_bad_fit_measurement_is_insufficient_not_noop(policy, bad):
    checks = rows()
    checks[0]["before"]["agreement"] = bad
    assert policy.activation(checks) == "insufficient"
    assert "uncheckable_credible_revisit" in policy.rejection_reasons(checks, rows())


def test_validation_failure_never_changes_activation_or_gets_a_pass(policy):
    checks = rows()
    checks[1]["before"]["agreement"] = 0.8
    checks[1]["after"]["agreement"] = 0.8
    snapshot = copy.deepcopy(checks)
    assert policy.activation(checks) == "keep"
    assert "sampled_revisit_below_90_percent" in policy.rejection_reasons(checks, rows())
    assert checks == snapshot


def test_good_pair_regression_rejects_even_when_every_pair_above_target(policy):
    checks = rows()
    checks[1]["after"]["agreement"] = 0.96
    assert policy.rejection_reasons(checks, rows()) == [
        "previously_good_pair_degraded_over_one_point"
    ]


def test_local_regression_and_missing_evidence_fail_closed(policy):
    checks, local = rows(), rows()
    local[0]["after"]["agreement"] = 0.96
    assert policy.rejection_reasons(checks, local) == ["local_pair_degraded_over_one_point"]
    assert "missing_or_uncheckable_local_checks" in policy.rejection_reasons(checks, [])
    assert "fewer_than_five_held_out_pairs" in policy.rejection_reasons([], local)
    assert policy.activation(checks[:8]) == "insufficient"


def test_frozen_boundary_and_unchanged_control_pass_without_mutation(policy):
    checks = rows()
    checks[0]["before"]["agreement"] = 0.9
    checks[0]["after"]["agreement"] = 0.9
    checks[1]["after"]["agreement"] = checks[1]["before"]["agreement"] - 0.01
    assert policy.activation(checks) == "keep"
    assert policy.rejection_reasons(checks, rows()) == []
