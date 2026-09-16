"""Closed offline raw-loop correction experiment; never a live scan-quality pass.

Fixed fit/validation split, existing raw/ICP gates, bounded graph, and explicit
interpolated corrections. Source poses and raw bytes are never changed.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scanner_candidate_audit import pose, projection
from scanner_moving_abba import descriptor, nearest_view
from scanner_revisit_audit import _verified_raw

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.coherence import image_period
from scanner3d.errors import ScannerError
from scanner3d.geometry import Rigid
from scanner3d.icp_plane import icp_point_to_plane
from scanner3d.pipeline import NOISE_FLOOR, frame_travel
from scanner3d.posegraph import PoseEdge
from scanner3d.posegraph_lm import optimise_poses_lm, se3_exp, se3_log
from scanner3d.preparation import DepthBand, prepare_frame
from scanner3d.relocalize import judge_local_edge
from scanner3d.tracking import DEFAULT_POLICY, convergence_tolerance


def partition_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split before fitting, with raw endpoint exclusion around validation views."""
    held = [s for i, s in enumerate(samples) if i % 2 and s["credible"]]
    protected = [s[k] for s in held for k in ("frame", "best_frame")]
    rows = []
    for i, sample in enumerate(samples):
        role = "validation" if i % 2 else "fit"
        reasons = []
        if not sample["credible"]:
            reasons.append("raw_image_match_not_credible")
        if role == "fit" and any(
            abs(sample[k] - v) <= 4 for k in ("frame", "best_frame") for v in protected
        ):
            reasons.append("within_four_frames_of_validation_endpoint")
        rows.append(dict(ordinal=i, role=role, sample=sample, exclusion_reasons=reasons))
    return rows


def fit_link(raws: list[bytes], band: DepthBand) -> tuple[Rigid | None, dict[str, Any]]:
    a, b = [np.frombuffer(raw, dtype="<u2").reshape(400, 640) for raw in raws]
    target_descriptor, source_descriptor = descriptor(a), descriptor(b)
    match = nearest_view(
        target_descriptor, source_descriptor[0][None], source_descriptor[1][None], [1]
    )
    evidence: dict[str, Any] = dict(raw_match=match, accepted=False)
    if not match["comparable"]:
        evidence["reasons"] = ["raw_descriptor_not_comparable"]
        return None, evidence
    clouds = [prepare_frame(raw, target_points=900, depth_band=band).points for raw in raws]
    evidence["prepared_points"] = [len(c) for c in clouds]
    fitted = []
    directions = []
    try:
        for source, target in ((clouds[1], clouds[0]), (clouds[0], clouds[1])):
            result = icp_point_to_plane(
                source,
                target,
                initial=Rigid.identity(),
                max_correspondence_distance=12.0,
                voxel_size=1.5,
                keep_fraction=0.8,
                backend="scipy",
                tolerance=convergence_tolerance(
                    target, fraction=DEFAULT_POLICY.convergence_noise_fraction
                ),
            )
            fitted.append(result.transform)
            directions.append(judge_local_edge(result, frame=1).as_dict())
    except (ScannerError, ValueError, ArithmeticError) as error:
        evidence.update(error=f"{type(error).__name__}: {error}", local_gates=directions)
        return None, evidence
    forward, reverse = fitted
    direct = projection(raws, [Rigid.identity(), forward], band)
    backward = projection(list(reversed(raws)), [Rigid.identity(), reverse], band)
    cycle = frame_travel(clouds[1], reverse.compose(forward))
    reasons = []
    if not all(d["accepted"] for d in directions):
        reasons.append("existing_local_edge_gate_failed")
    if any(p["agreement"] is None or p["agreement"] < 0.9 for p in (direct, backward)):
        reasons.append("measured_raw_projection_below_90_percent_or_uncheckable")
    if cycle > NOISE_FLOOR:
        reasons.append("forward_reverse_cycle_above_noise_floor")
    evidence.update(
        local_gates=directions,
        forward=forward.as_dict(),
        reverse=reverse.as_dict(),
        forward_projection=direct,
        reverse_projection=backward,
        cycle_surface_travel=cycle,
        accepted=not reasons,
        reasons=reasons,
    )
    return (None if reasons else forward), evidence


def interpolate_corrections(
    original: list[Rigid], indices: list[int], fitted: list[Rigid]
) -> list[Rigid]:
    if (
        not original
        or len(indices) != len(fitted)
        or len(indices) < 2
        or indices != sorted(set(indices))
        or indices[0] != 0
        or indices[-1] != len(original) - 1
    ):
        raise ValueError("Ordered correction nodes must cover both sequence endpoints")
    corrections = [
        new.compose(original[i].inverse()) for i, new in zip(indices, fitted, strict=True)
    ]
    result = []
    for i, old in enumerate(original):
        position = bisect.bisect_right(indices, i) - 1
        if indices[position] == i:
            result.append(fitted[position])  # Keep solved nodes exactly, including the anchor.
            continue
        fraction = (i - indices[position]) / (indices[position + 1] - indices[position])
        left, right = corrections[position : position + 2]
        relative = se3_log(left.inverse().compose(right))
        correction = left.compose(se3_exp([fraction * v for v in relative]))
        result.append(correction.compose(old))
    return result


def compare_pairs(
    raws: list[bytes],
    original: list[Rigid],
    corrected: list[Rigid],
    band: DepthBand,
    pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows = []
    for early, late in pairs:
        raw_pair = [raws[early], raws[late]]
        before = projection(raw_pair, [original[early], original[late]], band)
        after = projection(raw_pair, [corrected[early], corrected[late]], band)
        rows.append(
            dict(
                accepted_indices=[early, late],
                before=before,
                after=after,
                agreement_change=(after["agreement"] - before["agreement"])
                if before["agreement"] is not None and after["agreement"] is not None
                else None,
            )
        )
    return rows


def run(
    path: Path,
    output: Path,
    *,
    graph_metric: str = "se3",
    application_policy: str = "always",
    closure_schedule: str = "sparse",
) -> dict[str, Any]:
    if graph_metric not in ("se3", "surface", "surface-analytic"):
        raise ValueError("Only closed L1/L2/L3 experiments are supported")
    if application_policy not in ("always", "needed-only") or (
        application_policy == "needed-only" and graph_metric != "surface-analytic"
    ):
        raise ValueError("G1 needed-only requires the unchanged analytic surface solver")
    if closure_schedule not in ("sparse", "dense") or (
        closure_schedule == "dense" and application_policy != "needed-only"
    ):
        raise ValueError("G2 dense schedule requires G1 guards")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = dict(
        scope=__doc__,
        status="running",
        graph_metric=graph_metric,
        application_policy=application_policy,
        closure_schedule=closure_schedule,
        quality_qualification_allowed=False,
        closures=[],
    )
    try:
        content = path.read_bytes()
        replay = json.loads(content)
        if replay["status"] != "completed" or len(replay["runs"]) != 1:
            raise ValueError("One completed source replay required")
        source = replay["runs"][0]
        indices = source["accepted_source_indices"]
        if (
            source["stride"] != 1
            or not 100 <= len(indices) <= 3000
            or indices != sorted(set(indices))
            or len(source["poses"]) != len(indices)
        ):
            raise ValueError("100..3000 ordered unique stride-one accepted inputs required")
        inputs = replay["input_frames"]
        if len(inputs) > 3000 or any(
            type(i) is not int or not 0 <= i < len(inputs) for i in indices
        ):
            raise ValueError("Accepted indices must name bounded source inputs")
        all_raws = [_verified_raw(Path(replay["capture"]), row) for row in inputs]
        if any(len(raw) != 512000 for raw in all_raws):
            raise ValueError("All raw payloads must be complete")
        raws = [all_raws[i] for i in indices]
        original = [pose(p) for p in source["poses"]]
        band = DepthBand(**replay["depth_band"])
        report.update(
            source=str(path),
            source_sha256=hashlib.sha256(content).hexdigest(),
            input_frames=inputs,
            accepted_source_indices=indices,
            all_source_raw_hashes_verified=len(inputs),
            source_scene_validity=replay.get("source_scene_validity"),
            source_tracking_loss=replay.get("source_tracking_loss"),
        )
        estimate = image_period(raws, depth_scale=DEPTH_SCALE_METRES, samples=25)
        if estimate is None:
            report.update(status="insufficient", reason="no credible raw-image revisit set")
            return report
        samples = partition_samples([s.as_dict() for s in estimate.samples])
        report["partition"] = samples
        credible = [r for r in samples if r["sample"]["credible"]]
        pairs = [(r["sample"]["frame"], r["sample"]["best_frame"]) for r in credible]
        if application_policy == "needed-only":
            from scanner_loop_policy import activation, rejection_reasons

            baseline = compare_pairs(raws, original, original, band, pairs)
            for row, record in zip(baseline, credible, strict=True):
                row.update(
                    ordinal=record["ordinal"],
                    role=record["role"],
                    exclusion_reasons=record["exclusion_reasons"],
                )
            choice = activation(baseline)
            report.update(activation=choice, activation_checks=baseline)
            if choice == "insufficient":
                report.update(status="insufficient", reason="insufficient original fit checks")
                return report
            if choice == "keep":
                local = compare_pairs(
                    raws,
                    original,
                    original,
                    band,
                    [(i, i + 1) for i in range(0, len(raws) - 1, 20)],
                )
                reasons = rejection_reasons(baseline, local)
                report.update(
                    status="rejected" if reasons else "unchanged",
                    rejection_reasons=reasons,
                    revisit_checks=baseline,
                    fixed_local_checks=local,
                    selected_poses=source["poses"] if not reasons else None,
                    pose_choice="rejected" if reasons else "original_exact",
                    held_out_pairs=sum(r["role"] == "validation" for r in baseline),
                )
                return report
        scheduled = [
            dict(
                ordinal=r["ordinal"],
                accepted_indices=[r["sample"]["frame"], r["sample"]["best_frame"]],
            )
            for r in samples
            if r["role"] == "fit" and not r["exclusion_reasons"]
        ]
        if closure_schedule == "dense":
            from scanner_dense_loop import proposals

            dense = proposals(raws, samples)
            report["dense_proposals"] = dense
            scheduled = [
                dict(ordinal=r["query"], accepted_indices=r["accepted_indices"])
                for r in dense
                if not r["exclusion_reasons"]
            ]
        links = []
        for record in scheduled:
            if time.monotonic() - started > 900:
                raise TimeoutError("900-second offline experiment bound reached")
            early, late = record["accepted_indices"]
            transform, evidence = fit_link([raws[early], raws[late]], band)
            report["closures"].append(
                dict(ordinal=record["ordinal"], accepted_indices=[early, late], **evidence)
            )
            if transform is not None:
                links.append((early, late, transform))
            print("CLOSURE", record["ordinal"], evidence["accepted"], flush=True)
        if len(links) < 3:
            report.update(status="insufficient", reason="fewer than three admitted raw closures")
            return report
        nodes = sorted(
            {
                0,
                len(original) - 1,
                *range(0, len(original), 32),
                *(i for early, late, _ in links for i in (early, late)),
            }
        )
        node_limit = 256 if closure_schedule == "dense" else 128
        if len(nodes) > node_limit:
            raise ValueError(f"{node_limit}-node graph bound exceeded")
        lookup = {frame: i for i, frame in enumerate(nodes)}
        edges = [
            PoseEdge(
                i + 1,
                i,
                original[left].inverse().compose(original[right]),
                weight=1 / (right - left),
            )
            for i, (left, right) in enumerate(pairwise(nodes))
        ]
        edges.extend(
            PoseEdge(lookup[late], lookup[early], transform, weight=1)
            for early, late, transform in links
        )
        if graph_metric == "se3":
            solved = optimise_poses_lm(
                [original[i] for i in nodes], edges, anchor=0, iterations=100
            )
            node_poses, graph_report = list(solved.poses), solved.as_dict()
        else:
            from scanner_surface_graph import analytic_optimise, optimise

            node_clouds = [
                prepare_frame(raws[i], target_points=900, depth_band=band).points for i in nodes
            ]
            if graph_metric == "surface-analytic":
                node_poses, graph_report = analytic_optimise(
                    [original[i] for i in nodes], edges, node_clouds, max_nodes=node_limit
                )
            else:
                node_poses, graph_report = optimise(
                    [original[i] for i in nodes], edges, node_clouds
                )
        report.update(
            node_accepted_indices=nodes,
            graph=graph_report,
            graph_candidate_poses=[p.as_dict() for p in node_poses],
        )
        if not graph_report["converged"]:
            report.update(status="rejected", reason="graph did not converge within fixed bound")
            return report
        corrected = interpolate_corrections(original, nodes, node_poses)
        report["corrected_poses"] = [p.as_dict() for p in corrected]
        checks = compare_pairs(
            raws,
            original,
            corrected,
            band,
            [
                (r["sample"]["frame"], r["sample"]["best_frame"])
                for r in samples
                if r["sample"]["credible"]
            ],
        )
        for row, record in zip(
            checks, [r for r in samples if r["sample"]["credible"]], strict=True
        ):
            row.update(
                ordinal=record["ordinal"],
                role=record["role"],
                exclusion_reasons=record["exclusion_reasons"],
            )
        local = compare_pairs(
            raws, original, corrected, band, [(i, i + 1) for i in range(0, len(raws) - 1, 20)]
        )
        report.update(revisit_checks=checks, fixed_local_checks=local)
        held = [r for r in checks if r["role"] == "validation"]
        measured = [r["after"]["agreement"] for r in held if r["after"]["agreement"] is not None]
        report.update(
            held_out_pairs=len(held),
            checkable_held_out_pairs=len(measured),
            weakest_held_out_agreement=min(measured) if measured else None,
        )
        reasons = []
        if len(measured) < 5 or len(measured) != len(held):
            reasons.append("insufficient_checkable_held_out_pairs")
        if measured and min(measured) < 0.9:
            reasons.append("held_out_agreement_below_90_percent")
        if any(r["agreement_change"] is None or r["agreement_change"] < -0.01 for r in local):
            reasons.append("local_agreement_degraded_over_one_point_or_uncheckable")
        if application_policy == "needed-only":
            reasons.extend(rejection_reasons(checks, local))
            report.update(
                selected_poses=report["corrected_poses"] if not reasons else None,
                pose_choice="rejected" if reasons else "corrected_candidate",
            )
        report.update(status="rejected" if reasons else "candidate", rejection_reasons=reasons)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--graph-metric", choices=("se3", "surface", "surface-analytic"), default="se3"
    )
    parser.add_argument("--application-policy", choices=("always", "needed-only"), default="always")
    parser.add_argument("--closure-schedule", choices=("sparse", "dense"), default="sparse")
    args = parser.parse_args()
    report = run(
        args.replay,
        args.output,
        graph_metric=args.graph_metric,
        application_policy=args.application_policy,
        closure_schedule=args.closure_schedule,
    )
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "status",
                    "reason",
                    "rejection_reasons",
                    "held_out_pairs",
                    "weakest_held_out_agreement",
                    "elapsed_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
