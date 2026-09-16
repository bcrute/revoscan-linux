"""Frozen Stage B schedules retain raw indices, refusals and missing evidence."""

import copy
import importlib
from pathlib import Path

import pytest

from scanner3d.geometry import Rigid


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_stage_b_audit")


def fixture(module, role="slow_d_1609", *, refused=(), offered=100):
    factors = module.Factors("grid", "pooled", "residual")
    inputs = [dict(path=f"frame-{i}.raw", sha256=f"{i:064x}") for i in range(100)]
    indices = [i for i in range(offered) if i not in refused]
    poses = [Rigid.of(Rigid.identity().rotation, (i * 0.01, 0.0, 0.0)).as_dict() for i in indices]
    by_index = dict(zip(indices, poses, strict=True))
    decisions = [
        dict(source_index=i, decision=dict(accepted=i in by_index, pose=by_index.get(i)))
        for i in range(offered)
    ]
    run = dict(
        stride=1,
        quality_qualification_allowed=False,
        diagnostic_factors=factors.as_dict(),
        accepted_source_indices=indices,
        poses=poses,
        decisions=decisions,
        offered=offered,
        accepted=len(indices),
        tracking_stopped=False,
        messages=["GPU-assisted tracking: fixture"],
    )
    case = dict(
        role=role,
        input_frames=inputs,
        depth_band=dict(near=60.0, far=300.0),
        accepted_source_indices=list(range(100)),
    )
    document = dict(
        status="completed",
        input_frames=inputs,
        depth_band=case["depth_band"],
        diagnostic_factors=factors.as_dict(),
        quality_qualification_allowed=False,
        runs=[run],
        sampling_rule=factors.sampling_rule,
        model_normal_rule=factors.model_normal_rule,
        convergence_rule=factors.convergence_rule,
    )
    pairs = (
        []
        if role.startswith("handheld_")
        else [
            dict(
                kind="original_coherence",
                ordinal=i,
                accepted_indices=[0, i + 1],
                source_indices=[0, i + 1],
                raw_inputs=[inputs[0], inputs[i + 1]],
                original_pair=dict(within_tolerance=0.99),
            )
            for i in range(65)
        ]
    )
    roster = dict(
        role=role,
        pairs=pairs,
        original_coherence_metadata=dict(pair_selection="frozen_fixture"),
        prior_partition=dict(preserve="all original roles"),
        absence_reason=None if pairs else "No original geometry checks",
    )
    return case, roster, document, [str(i).encode() for i in range(100)]


def projector(monkeypatch, module, agreement=0.99):
    calls = []

    def project(raws, poses, band):
        calls.append((raws, poses, band))
        return dict(
            agreement=agreement,
            pairs=[dict(points=123, median_offset=0.25, within_tolerance=agreement)],
        )

    monkeypatch.setattr(module, "projection", project)
    return calls


def test_full_roster_and_projection_details_preserved(module, monkeypatch):
    args = fixture(module)
    calls = projector(monkeypatch, module)
    before = copy.deepcopy(args)
    score = module.score_case(*args)
    assert args == before and score["longer_geometry_guard_met"]
    assert score["required_pairs"] == score["checked_pairs"] == len(calls) == 65
    assert (
        score["verdict"] == "stage_b_longer_guard_met"
        and not score["quality_qualification_allowed"]
    )
    assert score["pairs"][16]["source_indices"] == [0, 17]
    assert calls[16][0] == [b"0", b"17"] and calls[16][1][1].translation == (0.17, 0.0, 0.0)
    assert score["pairs"][16]["projection"]["pairs"][0]["points"] == 123
    assert score["prior_partition"] == args[1]["prior_partition"]


@pytest.mark.parametrize("refused,offered,checked", [((17,), 100, 64), ((), 60, 59)])
def test_refused_and_unoffered_endpoints_remain_missing(
    module, monkeypatch, refused, offered, checked
):
    calls = projector(monkeypatch, module)
    score = module.score_case(*fixture(module, refused=refused, offered=offered))
    assert score["verdict"] == "rejected" and not score["frame_acceptance_guard_met"]
    assert score["required_pairs"] == 65 and score["checked_pairs"] == len(calls) == checked
    missing = [p for p in score["pairs"] if not p["checked"]]
    assert all(p["agreement"] is None and p["projection"] is None for p in missing)
    assert (
        score["refused_source_indices"] == list(refused)
        and score["unoffered_count"] == 100 - offered
    )


@pytest.mark.parametrize("role", ["handheld_pauses", "handheld_combined"])
def test_empty_handheld_schedule_never_becomes_geometry_pass(module, monkeypatch, role):
    calls = projector(monkeypatch, module)
    score = module.score_case(*fixture(module, role))
    assert score["frame_acceptance_guard_met"] and not score["longer_geometry_guard_met"]
    assert score["verdict"] == "acceptance_only_geometry_unqualified"
    assert not calls and score["weakest_agreement"] is None and score["absence_reason"]


@pytest.mark.parametrize(
    "baseline,actual,reason",
    [
        (0.89, 0.9, None),
        (0.89, 0.899, "original_pair_below_90_percent"),
        (0.99, 0.981, None),
        (0.99, 0.979, "original_good_pair_regressed_over_one_percentage_point"),
        (0.5, 0.98, None),
    ],
)
def test_original_quality_and_good_pair_regression_guards(
    module, monkeypatch, baseline, actual, reason
):
    args = fixture(module)
    for pair in args[1]["pairs"]:
        pair["original_pair"]["within_tolerance"] = baseline
    projector(monkeypatch, module, actual)
    result = module.score_case(*args)
    assert result["longer_geometry_guard_met"] is (reason is None)
    if reason:
        assert reason in result["reasons"]
    assert all(p["original_good_pair"] == (baseline >= 0.9) for p in result["pairs"])


@pytest.mark.parametrize(
    "failure",
    [
        "input-manifest",
        "band",
        "factor",
        "quality",
        "counts",
        "mapping",
        "pose-mismatch",
        "pair-count",
        "pair-id",
        "pair-raw",
        "baseline",
        "gpu-missing",
        "gpu-fallback",
    ],
)
def test_corrupted_evidence_is_not_silently_scored(module, monkeypatch, failure):
    case, roster, document, raws = copy.deepcopy(fixture(module))
    run = document["runs"][0]
    if failure == "input-manifest":
        document["input_frames"] = []
    elif failure == "band":
        document["depth_band"] = dict(near=61.0, far=300.0)
    elif failure == "factor":
        document["diagnostic_factors"] = {}
    elif failure == "quality":
        document["quality_qualification_allowed"] = True
    elif failure == "counts":
        run["offered"] = 99
    elif failure == "mapping":
        run["accepted_source_indices"] = list(reversed(run["accepted_source_indices"]))
    elif failure == "pose-mismatch":
        run["poses"] = [Rigid.identity().as_dict()] * 100
    elif failure == "pair-count":
        roster["pairs"].pop()
    elif failure == "pair-id":
        roster["pairs"][1]["ordinal"] = 0
    elif failure == "pair-raw":
        roster["pairs"][0]["raw_inputs"] = []
    elif failure == "baseline":
        roster["pairs"][0]["original_pair"]["within_tolerance"] = float("nan")
    elif failure == "gpu-missing":
        run["messages"] = []
    else:
        run["messages"].append("GPU normals failed: fixture")
    projector(monkeypatch, module)
    with pytest.raises(ValueError):
        module.score_case(case, roster, document, raws)


def test_unprojectable_pair_keeps_missing_agreement(module, monkeypatch):
    projector(monkeypatch, module, None)
    result = module.score_case(*fixture(module))
    assert result["required_pairs"] == 65 and result["checked_pairs"] == 0
    assert result["weakest_agreement"] is None and result["verdict"] == "rejected"


def test_original_prior_and_sealed_pair_metrics_are_used(module, monkeypatch):
    args = fixture(module)
    for ordinal, kind in enumerate(("prior_revisit", "prior_fixed_local", "sealed_d")):
        args[1]["pairs"][ordinal]["kind"] = kind
        args[1]["pairs"][ordinal]["original"] = dict(
            agreement=0.99, preserve="full original report"
        )
    projector(monkeypatch, module)
    result = module.score_case(*args)
    assert result["longer_geometry_guard_met"]
    assert [p["kind"] for p in result["pairs"][:3]] == [
        "prior_revisit",
        "prior_fixed_local",
        "sealed_d",
    ]
