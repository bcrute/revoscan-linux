"""Score frozen Stage B raw-source endpoints, never reselect or impute poses."""

from __future__ import annotations

import math
from typing import Any

from scanner_candidate_audit import pose, projection
from scanner_factorial_config import Factors

from scanner3d.preparation import DepthBand


def score_case(
    case: dict[str, Any], roster: dict[str, Any], document: dict[str, Any], raws: list[bytes]
) -> dict[str, Any]:
    factors = Factors("grid", "pooled", "residual")
    if (
        case["role"] != roster["role"]
        or document.get("input_frames") != case["input_frames"]
        or document.get("depth_band") != case["depth_band"]
        or document.get("diagnostic_factors") != factors.as_dict()
        or document.get("quality_qualification_allowed") is not False
        or not 100 <= len(raws) == len(case["input_frames"]) <= 3000
    ):
        raise ValueError("Stage B input manifest, band, factors or qualification metadata differs")
    for key, expected in (
        ("sampling_rule", factors.sampling_rule),
        ("model_normal_rule", factors.model_normal_rule),
        ("convergence_rule", factors.convergence_rule),
    ):
        if document.get(key) != expected:
            raise ValueError(f"Stage B {key} differs from declared factors")
    completed = document.get("status") == "completed"
    runs = document.get("runs", [])
    if len(runs) > 1 or (completed and len(runs) != 1):
        raise ValueError("Stage B requires one completed or explicitly failed run")
    run = runs[0] if runs else document.get("failed_run", {})
    if (
        run.get("stride") != 1
        or run.get("quality_qualification_allowed") is not False
        or run.get("diagnostic_factors") != factors.as_dict()
    ):
        raise ValueError("Stage B run factors, stride or qualification metadata differs")
    decisions, indices, poses = (
        run.get("decisions", []),
        run.get("accepted_source_indices", []),
        run.get("poses", []),
    )
    if [d["source_index"] for d in decisions] != list(range(len(decisions))):
        raise ValueError("Stage B offered indices must retain the original contiguous prefix")
    if indices != [d["source_index"] for d in decisions if d["decision"]["accepted"]]:
        raise ValueError("Stage B accepted index mapping differs from decisions")
    if len(set(indices)) != len(indices) or len(poses) != len(indices):
        raise ValueError("Stage B requires unique explicit accepted poses")
    positions = dict(zip(indices, (pose(p) for p in poses), strict=True))
    if any(
        d["decision"]["accepted"] and positions[d["source_index"]] != pose(d["decision"]["pose"])
        for d in decisions
    ):
        raise ValueError("Stage B trajectory differs from actual accepted decisions")
    offered = run.get("offered", run.get("offered_before_error", len(decisions)))
    accepted = run.get("accepted", run.get("accepted_before_error", len(indices)))
    if offered != len(decisions) or accepted != len(indices) or not 0 <= offered <= len(raws):
        raise ValueError("Stage B counts differ from decisions or input bounds")
    messages = run.get("messages", document.get("messages", []))
    activated = [m for m in messages if m.startswith("GPU-assisted tracking:")]
    fallback = [
        m
        for m in messages
        if m.startswith(
            ("GPU normals failed:", "GPU normals unavailable;", "GPU normals require SciPy;")
        )
    ]
    if completed and (not activated or fallback):
        raise ValueError("Stage B completed run must record GPU activation without fallback")
    row: dict[str, Any] = dict(
        role=case["role"],
        report_status=document.get("status"),
        quality_qualification_allowed=False,
        offered=offered,
        accepted=accepted,
        input_count=len(raws),
        unoffered_count=len(raws) - offered,
        pairs=[],
        reasons=[],
        refused_source_indices=[
            d["source_index"] for d in decisions if not d["decision"]["accepted"]
        ],
        gpu_activation=activated,
        gpu_fallback_messages=fallback,
        original_coherence_metadata=roster["original_coherence_metadata"],
        prior_partition=roster["prior_partition"],
        absence_reason=roster["absence_reason"],
    )
    expected_counts = dict(
        slow_d_1609=65,
        development_extended=103,
        old_adverse_2703=165,
        later_adverse_2359=147,
        handheld_pauses=0,
        handheld_combined=0,
    )
    if len(roster["pairs"]) != expected_counts.get(case["role"]):
        raise ValueError("Stage B requires its entire original per-case pair roster")
    identifiers: set[tuple[str, int]] = set()
    band = DepthBand(**case["depth_band"])
    for pair in roster["pairs"]:
        identifier = (pair["kind"], pair["ordinal"])
        if identifier in identifiers or pair["kind"] not in (
            "original_coherence",
            "prior_revisit",
            "prior_fixed_local",
            "sealed_d",
        ):
            raise ValueError("Stage B pair identifiers must be unique and original")
        identifiers.add(identifier)
        a, b = pair["source_indices"]
        if not 0 <= a < len(raws) or not 0 <= b < len(raws) or a == b:
            raise ValueError("Stage B raw endpoints are invalid")
        if pair["raw_inputs"] != [case["input_frames"][i] for i in (a, b)] or [
            case["accepted_source_indices"][i] for i in pair["accepted_indices"]
        ] != [a, b]:
            raise ValueError("Stage B original accepted-to-raw endpoint mapping differs")
        baseline = (
            pair["original_pair"]["within_tolerance"]
            if pair["kind"] == "original_coherence"
            else pair["original"]["agreement"]
        )
        if baseline is None or not math.isfinite(baseline) or not 0 <= baseline <= 1:
            raise ValueError("Stage B original pair must retain its finite agreement")
        measured = (
            projection([raws[a], raws[b]], [positions[a], positions[b]], band)
            if a in positions and b in positions
            else None
        )
        agreement = None if measured is None else measured["agreement"]
        if agreement is not None and (not math.isfinite(agreement) or not 0 <= agreement <= 1):
            raise ValueError("Stage B projection returned invalid agreement")
        row["pairs"].append(
            dict(
                kind=pair["kind"],
                ordinal=pair["ordinal"],
                source_indices=[a, b],
                original_accepted_indices=pair["accepted_indices"],
                raw_inputs=pair["raw_inputs"],
                baseline_agreement=baseline,
                original_good_pair=baseline >= 0.9,
                checked=agreement is not None,
                agreement=agreement,
                projection=measured,
                loss_percentage_points=None if agreement is None else 100 * (baseline - agreement),
                reason=None if agreement is not None else "endpoint_missing_or_unprojectable",
            )
        )
    checked = [p for p in row["pairs"] if p["checked"]]
    if not completed:
        row["reasons"].append("replay_did_not_complete")
    if run.get("tracking_stopped"):
        row["reasons"].append("terminal_tracking_loss")
    if accepted != len(raws):
        row["reasons"].append("not_all_original_inputs_accepted")
    if len(checked) != len(row["pairs"]):
        row["reasons"].append("original_pairs_missing_or_uncheckable")
    if any(p["agreement"] < 0.9 for p in checked):
        row["reasons"].append("original_pair_below_90_percent")
    if any(p["original_good_pair"] and p["loss_percentage_points"] > 1 for p in checked):
        row["reasons"].append("original_good_pair_regressed_over_one_percentage_point")
    row.update(
        required_pairs=len(row["pairs"]),
        checked_pairs=len(checked),
        weakest_agreement=min((p["agreement"] for p in checked), default=None),
        maximum_good_pair_loss_percentage_points=max(
            (p["loss_percentage_points"] for p in checked if p["original_good_pair"]), default=None
        ),
        frame_acceptance_guard_met=completed
        and accepted == len(raws)
        and not run.get("tracking_stopped"),
        longer_geometry_guard_met=bool(row["pairs"]) and not row["reasons"],
        geometry_schedule_present=bool(row["pairs"]),
        verdict="rejected"
        if row["reasons"]
        else (
            "stage_b_longer_guard_met" if row["pairs"] else "acceptance_only_geometry_unqualified"
        ),
    )
    return row
