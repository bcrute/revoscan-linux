"""Fixed-pair F1 scoring and descriptive interactions; never qualify a scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

from scanner_candidate_audit import pose, projection
from scanner_factorial_config import CONFIGURATIONS, Factors
from scanner_factorial_replay import load_static
from scanner_motion_replay import load_inputs

from scanner3d.preparation import DepthBand


def score_cell(
    case: dict[str, Any], document: dict[str, Any], factors: Factors, raws: list[bytes]
) -> dict[str, Any]:
    """Score original endpoints even when a new replay stops or loses poses."""
    if (
        document.get("input_frames") != case["input_frames"]
        or document.get("depth_band") != case["depth_band"]
        or document.get("diagnostic_factors") != factors.as_dict()
        or document.get("quality_qualification_allowed") is not False
        or len(raws) != case["frames"]
        or not case["required_pairs"]
    ):
        raise ValueError("Cell inputs, band, factors or qualification metadata differ")
    for key, expected in (
        ("sampling_rule", factors.sampling_rule),
        ("model_normal_rule", factors.model_normal_rule),
        ("convergence_rule", factors.convergence_rule),
    ):
        if document.get(key) != expected:
            raise ValueError(f"Cell {key} differs from declared factors")
    completed = document.get("status") == "completed"
    if case["static"]:
        source = document
    else:
        runs = document.get("runs", [])
        if len(runs) > 1 or (completed and len(runs) != 1):
            raise ValueError("A single complete or explicitly failed F1 run is required")
        source = runs[0] if runs else document.get("failed_run", {})
        if source.get("stride") != 1:
            raise ValueError("F1 requires stride one")
    if (
        source.get("quality_qualification_allowed") is not False
        or source.get("diagnostic_factors") != factors.as_dict()
    ):
        raise ValueError("Run-level factors or qualification metadata differ")
    decisions = source.get("decisions", [])
    indices = source.get("accepted_source_indices", [])
    poses = source.get("poses", [])
    if [d["source_index"] for d in decisions] != list(range(len(decisions))):
        raise ValueError("Offered indices must be the original contiguous prefix")
    if indices != [d["source_index"] for d in decisions if d["decision"]["accepted"]]:
        raise ValueError("Accepted pose mapping differs from decisions")
    if len(set(indices)) != len(indices) or len(poses) != len(indices):
        raise ValueError("Unique, explicit accepted poses required")
    positions = dict(zip(indices, (pose(p) for p in poses), strict=True))
    for d in decisions:
        if d["decision"]["accepted"] and positions[d["source_index"]] != pose(
            d["decision"]["pose"]
        ):
            raise ValueError("Saved trajectory differs from the actually accepted decision")
    offered = source.get("offered", source.get("offered_before_error", len(decisions)))
    accepted = source.get("accepted", source.get("accepted_before_error", len(indices)))
    if offered != len(decisions) or accepted != len(indices) or not 0 <= offered <= len(raws):
        raise ValueError("Cell counts disagree with recorded decisions and input bounds")
    messages = source.get("messages", document.get("messages", []))
    gpu = any(m.startswith("GPU-assisted tracking:") for m in messages)
    if completed and not gpu:
        raise ValueError("Completed cell has no evidence of actual GPU activation")
    normal_records = source.get("model_normal_records")
    formed_models = sum(
        c["origin"] == "model" for d in decisions for c in d["decision"].get("candidates", [])
    )
    if (
        completed
        and factors.model_normals == "native"
        and (
            not isinstance(normal_records, list)
            or len(normal_records) < formed_models
            or any(r.get("status") != "completed" for r in normal_records)
        )
    ):
        raise ValueError("Native model calls are absent, incomplete or inconsistent")
    if completed and factors.model_normals == "pooled" and normal_records is not None:
        raise ValueError("Pooled configuration unexpectedly substituted native model normals")
    row: dict[str, Any] = dict(
        case=case["name"],
        configuration=factors.identifier,
        factors=factors.as_dict(),
        quality_qualification_allowed=False,
        report_status=document.get("status"),
        source_error=document.get("error"),
        offered=offered,
        accepted=accepted,
        input_count=len(raws),
        unoffered_count=len(raws) - offered,
        refused_source_indices=[
            d["source_index"] for d in decisions if not d["decision"]["accepted"]
        ],
        tracking_stopped=source.get("tracking_stopped"),
        gpu_verified=gpu,
        model_calls=None if normal_records is None else len(normal_records),
        formed_model_candidates=formed_models,
        pairs=[],
        reasons=[],
    )
    band = DepthBand(**case["depth_band"])
    for original in case["required_pairs"]:
        a, b = original["source_indices"]
        if not 0 <= a < len(raws) or not 0 <= b < len(raws) or a == b:
            raise ValueError("Frozen pair endpoints are outside this recording")
        measured = (
            projection([raws[a], raws[b]], [positions[a], positions[b]], band)
            if a in positions and b in positions
            else None
        )
        agreement = None if measured is None else measured["agreement"]
        if agreement is not None and (not math.isfinite(agreement) or not 0 <= agreement <= 1):
            raise ValueError("Projection returned an invalid agreement")
        row["pairs"].append(
            dict(
                identifier=original["identifier"],
                schedule=original["schedule"],
                source_indices=[a, b],
                baseline_agreement=original["baseline_agreement"],
                checked=agreement is not None,
                agreement=agreement,
                projection=measured,
                loss_percentage_points=None
                if agreement is None
                else 100 * (original["baseline_agreement"] - agreement),
                reason=None if agreement is not None else "endpoint_missing_or_unprojectable",
            )
        )
    checked = [p for p in row["pairs"] if p["checked"]]
    if not completed:
        row["reasons"].append("replay_did_not_complete")
    if row["tracking_stopped"]:
        row["reasons"].append("terminal_tracking_loss")
    if accepted != len(raws):
        row["reasons"].append("not_all_original_inputs_accepted")
    if len(checked) != len(row["pairs"]):
        row["reasons"].append("original_pairs_missing_or_uncheckable")
    if any(p["agreement"] < 0.9 for p in checked):
        row["reasons"].append("original_pair_below_90_percent")
    if case["original_pair_no_regression_guard"] and any(
        p["loss_percentage_points"] > 1 for p in checked
    ):
        row["reasons"].append("original_pair_regressed_over_one_percentage_point")
    row.update(
        checked_pairs=len(checked),
        required_pairs=len(row["pairs"]),
        weakest_agreement=min((p["agreement"] for p in checked), default=None),
        maximum_loss_percentage_points=max(
            (p["loss_percentage_points"] for p in checked), default=None
        ),
        below_margin_pairs=[p["identifier"] for p in checked if p["agreement"] < 0.9],
        verdict="rejected" if row["reasons"] else "stage_a_case_guard_met",
    )
    if case["static"]:
        row["maximum_apparent_surface_motion"] = max(
            (
                d["surface_displacement_from_identity"]
                for d in decisions
                if d["decision"]["accepted"]
            ),
            default=None,
        )
    return row


def contrast_specs() -> list[dict[str, Any]]:
    """Every conditional main effect and two-/three-factor interaction in F1."""
    specs = []

    def add(
        label: str,
        order: int,
        sampling: list[tuple[str, int]],
        normals: list[tuple[str, int]],
        convergence: list[tuple[str, int]],
    ) -> None:
        terms = {
            Factors(s, n, c).identifier: sw * nw * cw
            for (s, sw), (n, nw), (c, cw) in product(sampling, normals, convergence)
        }
        specs.append(dict(label=label, interaction_order=order, coefficients=terms))

    normal_delta = [("pooled", -1), ("native", 1)]
    convergence_delta = [("residual", -1), ("settling", 1)]
    for sampling in ("stride", "rank", "grid"):
        for convergence in ("residual", "settling"):
            add(
                f"native_effect/{sampling}/{convergence}",
                1,
                [(sampling, 1)],
                normal_delta,
                [(convergence, 1)],
            )
        for normals in ("pooled", "native"):
            add(
                f"settling_effect/{sampling}/{normals}",
                1,
                [(sampling, 1)],
                [(normals, 1)],
                convergence_delta,
            )
        add(f"native_x_settling/{sampling}", 2, [(sampling, 1)], normal_delta, convergence_delta)
    for sampling in ("rank", "grid"):
        sample_delta = [("stride", -1), (sampling, 1)]
        for normals, convergence in product(("pooled", "native"), ("residual", "settling")):
            add(
                f"{sampling}_effect/{normals}/{convergence}",
                1,
                sample_delta,
                [(normals, 1)],
                [(convergence, 1)],
            )
        for convergence in ("residual", "settling"):
            add(
                f"{sampling}_x_native/{convergence}",
                2,
                sample_delta,
                normal_delta,
                [(convergence, 1)],
            )
        for normals in ("pooled", "native"):
            add(
                f"{sampling}_x_settling/{normals}",
                2,
                sample_delta,
                [(normals, 1)],
                convergence_delta,
            )
        add(f"{sampling}_x_native_x_settling", 3, sample_delta, normal_delta, convergence_delta)
    return specs


def interactions(case: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired agreement contrasts in percentage points; missing is never zero."""
    by_config = {
        row["configuration"]: {p["identifier"]: p for p in row.get("pairs", [])} for row in rows
    }
    results = []
    for original in case["required_pairs"]:
        for spec in contrast_specs():
            constituents = []
            for config, coefficient in spec["coefficients"].items():
                pair = by_config.get(config, {}).get(original["identifier"])
                value = pair["agreement"] if pair is not None and pair["checked"] else None
                constituents.append(
                    dict(configuration=config, coefficient=coefficient, agreement=value)
                )
            available = all(c["agreement"] is not None for c in constituents)
            results.append(
                dict(
                    pair=original["identifier"],
                    source_indices=original["source_indices"],
                    label=spec["label"],
                    interaction_order=spec["interaction_order"],
                    constituents=constituents,
                    computable=available,
                    percentage_points=sum(
                        c["coefficient"] * c["agreement"] * 100 for c in constituents
                    )
                    if available
                    else None,
                    reason=None if available else "missing_score_in_one_or_more_configurations",
                )
            )
    return results


def run(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Refusing to overwrite a prior factorial audit")
    plan_bytes = (root / "stage-a-plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("status") != "completed" or len(plan["cells"]) != 60:
        raise ValueError("A completed frozen 60-cell plan is required")
    execution = []
    for phase in ("pilot", "matrix"):
        ledger = json.loads((root / f"{phase}-execution.json").read_bytes())
        if (
            ledger.get("status") != "completed"
            or ledger.get("plan_sha256") != hashlib.sha256(plan_bytes).hexdigest()
        ):
            raise ValueError("Every matrix cell must finish its declared execution attempt")
        execution.extend(ledger["cells"])
    executed = {e["identifier"]: e for e in execution}
    if len(execution) != 60 or set(executed) != {c["identifier"] for c in plan["cells"]}:
        raise ValueError(
            "Execution roster differs from the twelve complete five-case configurations"
        )
    report: dict[str, Any] = dict(
        status="running",
        quality_qualification_allowed=False,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        cells=[],
        case_interactions=[],
        configurations=[],
        limitations=[
            "Original raw pairs are frozen; no alternate new match replaces one.",
            "Interactions are descriptive on reused recordings, not independent release evidence.",
            "C has only 22 endpoint-separated checks; Stage A survival never qualifies a scan.",
            "Missing scores remain missing, never imputed as zero or averaged away.",
        ],
    )
    try:
        for case in plan["cases"]:
            if case["static"]:
                raws, inputs, _ = load_static(Path(case["capture"]))
            else:
                _, raws, _, inputs = load_inputs(
                    Path(case["capture"]), case["frames"], 12, (60, 300), diagnostic_scene=True
                )
            if inputs != case["input_frames"]:
                raise ValueError("Raw inputs have changed since the frozen plan")
            case_rows = []
            for factors in CONFIGURATIONS:
                identifier = f"{case['name']}/{factors.identifier}"
                attempt = executed[identifier]
                path = Path(attempt["output"]) / "report.json"
                try:
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != attempt["report_sha256"]:
                        raise ValueError("Replay report changed after execution")
                    row = score_cell(case, json.loads(payload), factors, raws)
                    row.update(source=str(path), source_sha256=attempt["report_sha256"])
                    if attempt["status"] != "completed":
                        row["reasons"].append("process_did_not_complete_cleanly")
                        row["verdict"] = "rejected"
                except (ValueError, KeyError, OSError, TypeError) as error:
                    row = dict(
                        case=case["name"],
                        configuration=factors.identifier,
                        factors=factors.as_dict(),
                        quality_qualification_allowed=False,
                        verdict="unusable_cell_evidence",
                        pairs=[],
                        reasons=[f"{type(error).__name__}: {error}"],
                        execution=attempt,
                    )
                case_rows.append(row)
                report["cells"].append(row)
                print(
                    identifier,
                    row["verdict"],
                    row.get("weakest_agreement"),
                    row.get("maximum_loss_percentage_points"),
                    flush=True,
                )
            report["case_interactions"].append(
                dict(case=case["name"], contrasts=interactions(case, case_rows))
            )
        for factors in CONFIGURATIONS:
            cells = [c for c in report["cells"] if c["configuration"] == factors.identifier]
            assert len(cells) == 5
            passed = all(c["verdict"] == "stage_a_case_guard_met" for c in cells)
            report["configurations"].append(
                dict(
                    configuration=factors.identifier,
                    factors=factors.as_dict(),
                    stage_a_survives=passed,
                    stage_b_required=passed,
                    failed_cases=[
                        dict(case=c["case"], reasons=c["reasons"])
                        for c in cells
                        if c["verdict"] != "stage_a_case_guard_met"
                    ],
                )
            )
        report["status"] = "completed"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
