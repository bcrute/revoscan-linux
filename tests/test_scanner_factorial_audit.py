"""Frozen original-pair guards and interactions cannot hide regressions or omissions."""

import importlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from scanner3d.geometry import Rigid


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_factorial_audit")


def fixture(audit):
    factors = audit.Factors()
    identity = Rigid.identity()
    metadata = dict(
        quality_qualification_allowed=False,
        diagnostic_factors=factors.as_dict(),
        sampling_rule=factors.sampling_rule,
        model_normal_rule=factors.model_normal_rule,
        convergence_rule=factors.convergence_rule,
    )
    case = dict(
        name="control",
        static=False,
        frames=4,
        input_frames=[{"index": i} for i in range(4)],
        depth_band={"near": 60, "far": 300},
        original_pair_no_regression_guard=True,
        required_pairs=[
            dict(
                identifier="first",
                schedule="original",
                source_indices=[0, 2],
                baseline_agreement=0.94,
            ),
            dict(
                identifier="second",
                schedule="original",
                source_indices=[1, 3],
                baseline_agreement=0.99,
            ),
        ],
    )
    source = dict(
        stride=1,
        accepted=4,
        offered=4,
        tracking_stopped=False,
        decisions=[
            dict(
                source_index=i, decision=dict(accepted=True, pose=identity.as_dict(), candidates=[])
            )
            for i in range(4)
        ],
        accepted_source_indices=list(range(4)),
        poses=[identity.as_dict() for _ in range(4)],
        model_normal_records=None,
        messages=["GPU-assisted tracking: unit-test double"],
        **metadata,
    )
    document = dict(
        status="completed",
        input_frames=deepcopy(case["input_frames"]),
        depth_band=case["depth_band"],
        runs=[source],
        **metadata,
    )
    return case, document, factors, [bytes([i]) for i in range(4)]


def scores(audit, monkeypatch, first=0.94, second=0.99):
    values = {(0, 2): first, (1, 3): second}
    monkeypatch.setattr(
        audit, "projection", lambda raw, *_a: {"agreement": values[(raw[0][0], raw[1][0])]}
    )


def test_better_minimum_cannot_hide_an_original_pair_regression(audit, monkeypatch):
    args = fixture(audit)
    scores(audit, monkeypatch, 0.96, 0.975)
    row = audit.score_cell(*args)
    assert row["weakest_agreement"] == 0.96  # Baseline minimum was only 0.94.
    assert row["maximum_loss_percentage_points"] == pytest.approx(1.5)
    assert row["verdict"] == "rejected"
    assert row["reasons"] == ["original_pair_regressed_over_one_percentage_point"]
    assert row["quality_qualification_allowed"] is False


def test_unchanged_complete_case_only_survives_engineering_screen(audit, monkeypatch):
    args = fixture(audit)
    scores(audit, monkeypatch)
    row = audit.score_cell(*args)
    assert row["verdict"] == "stage_a_case_guard_met" and not row["reasons"]
    assert row["checked_pairs"] == row["required_pairs"] == 2
    assert row["quality_qualification_allowed"] is False


def test_partial_trajectory_preserves_missing_scores_and_original_schedule(audit, monkeypatch):
    case, document, factors, raws = fixture(audit)
    source = document["runs"][0]
    source.update(offered=3, accepted=2, accepted_source_indices=[0, 1], poses=source["poses"][:2])
    source["decisions"] = source["decisions"][:3]
    source["decisions"][2]["decision"].update(accepted=False, pose=None)
    scores(audit, monkeypatch)
    row = audit.score_cell(case, document, factors, raws)
    assert row["unoffered_count"] == 1 and row["refused_source_indices"] == [2]
    assert row["required_pairs"] == 2 and row["checked_pairs"] == 0
    assert row["weakest_agreement"] is None and row["maximum_loss_percentage_points"] is None
    assert all(p["agreement"] is None and p["loss_percentage_points"] is None for p in row["pairs"])
    assert "original_pairs_missing_or_uncheckable" in row["reasons"]


@pytest.mark.parametrize("value", [None, 0.89])
def test_unprojectable_or_bad_geometry_cannot_pass(audit, monkeypatch, value):
    args = fixture(audit)
    scores(audit, monkeypatch, value)
    row = audit.score_cell(*args)
    assert row["verdict"] == "rejected" and row["checked_pairs"] == (1 if value is None else 2)


@pytest.mark.parametrize(
    "damage",
    [
        "input",
        "factor",
        "qualification",
        "run_qualification",
        "mapping",
        "pose",
        "gpu",
        "empty_pairs",
    ],
)
def test_malformed_cell_provenance_is_refused(audit, monkeypatch, damage):
    case, document, factors, raws = fixture(audit)
    source = document["runs"][0]
    if damage == "input":
        document["input_frames"][0]["index"] = 10
    elif damage == "factor":
        document["diagnostic_factors"] = {}
    elif damage == "qualification":
        document["quality_qualification_allowed"] = True
    elif damage == "run_qualification":
        source["quality_qualification_allowed"] = True
    elif damage == "mapping":
        source["accepted_source_indices"] = [0, 1, 1, 3]
    elif damage == "pose":
        source["poses"][2]["translation"] = [1, 0, 0]
    elif damage == "gpu":
        source["messages"] = []
    else:
        case["required_pairs"] = []
    scores(audit, monkeypatch)
    with pytest.raises(ValueError):
        audit.score_cell(case, document, factors, raws)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_agreement_never_becomes_a_verdict(audit, monkeypatch, value):
    args = fixture(audit)
    scores(audit, monkeypatch, value)
    with pytest.raises(ValueError, match="invalid agreement"):
        audit.score_cell(*args)


def test_failed_run_with_complete_poses_is_still_not_a_success(audit, monkeypatch):
    case, document, factors, raws = fixture(audit)
    document.update(
        status="failed", error="injected reporting failure", failed_run=document["runs"][0]
    )
    document["runs"] = []
    scores(audit, monkeypatch)
    row = audit.score_cell(case, document, factors, raws)
    assert row["checked_pairs"] == 2 and row["verdict"] == "rejected"
    assert row["reasons"] == ["replay_did_not_complete"]


def test_contrast_product_is_complete_and_zero_sum(audit):
    specs = audit.contrast_specs()
    assert len(specs) == len({s["label"] for s in specs}) == 33
    assert Counter(s["interaction_order"] for s in specs) == {1: 20, 2: 11, 3: 2}
    allowed = {c.identifier for c in audit.CONFIGURATIONS}
    for spec in specs:
        assert set(spec["coefficients"]) <= allowed
        assert len(spec["coefficients"]) == 2 ** spec["interaction_order"]
        assert sum(spec["coefficients"].values()) == 0


def interaction_fixture(audit):
    case = dict(required_pairs=[dict(identifier="fixed", source_indices=[3, 8])])
    rows = []
    for config in audit.CONFIGURATIONS:
        s = {"stride": 0, "rank": 1, "grid": 2}[config.sampling]
        n, c = int(config.model_normals == "native"), int(config.convergence == "settling")
        value = (
            0.5
            + 0.01 * s
            + 0.02 * n
            + 0.03 * c
            + 0.004 * s * n
            + 0.005 * s * c
            + 0.006 * n * c
            + 0.007 * s * n * c
        )
        assert 0 <= value <= 1
        rows.append(
            dict(
                configuration=config.identifier,
                pairs=[dict(identifier="fixed", checked=True, agreement=value)],
            )
        )
    return case, rows


def test_interactions_recover_known_couplings_in_percentage_points(audit):
    result = audit.interactions(*interaction_fixture(audit))
    assert len(result) == 33 and all(r["computable"] for r in result)
    by_label = {r["label"]: r["percentage_points"] for r in result}
    for label, expected in {
        "rank_x_native/residual": 0.4,
        "rank_x_native/settling": 1.1,
        "rank_x_native_x_settling": 0.7,
        "grid_x_native_x_settling": 1.4,
        "native_x_settling/stride": 0.6,
        "native_x_settling/rank": 1.3,
        "native_effect/grid/settling": 4.8,
        "settling_effect/rank/native": 4.8,
        "grid_effect/native/settling": 5.2,
    }.items():
        assert by_label[label] == pytest.approx(expected, abs=1e-12)


def test_one_missing_configuration_invalidates_only_its_contrasts(audit):
    case, rows = interaction_fixture(audit)
    target = next(r for r in rows if r["configuration"] == "grid-native-settling")
    target["pairs"][0].update(checked=False, agreement=None)
    result = audit.interactions(case, rows)
    missing = [r for r in result if not r["computable"]]
    assert len(missing) == 7
    assert all(r["percentage_points"] is None for r in missing)
    assert all(any(c["agreement"] is None for c in r["constituents"]) for r in missing)
    assert all(r["computable"] for r in result if r["label"].startswith("rank"))


@pytest.mark.parametrize("records", [None, [], [{"status": "failed"}]])
def test_declared_native_normals_need_completed_model_call_evidence(audit, monkeypatch, records):
    case, document, _, raws = fixture(audit)
    factors = audit.Factors("stride", "native", "residual")
    source = document["runs"][0]
    for record in (document, source):
        record.update(
            diagnostic_factors=factors.as_dict(), model_normal_rule=factors.model_normal_rule
        )
    source["decisions"][-1]["decision"]["candidates"] = [dict(origin="model")]
    source["model_normal_records"] = records
    scores(audit, monkeypatch)
    with pytest.raises(ValueError, match="Native model calls"):
        audit.score_cell(case, document, factors, raws)
    source["model_normal_records"] = [dict(status="completed")]
    row = audit.score_cell(case, document, factors, raws)
    assert row["model_calls"] == row["formed_model_candidates"] == 1


def test_pooled_mode_cannot_silently_substitute_native_normals(audit, monkeypatch):
    case, document, factors, raws = fixture(audit)
    document["runs"][0]["model_normal_records"] = []
    scores(audit, monkeypatch)
    with pytest.raises(ValueError, match="unexpectedly substituted"):
        audit.score_cell(case, document, factors, raws)
