"""Offline phase-matched moving 20/14/14/20 comparison, never a scan-quality pass.

All anchors, including poor/unmatchable views, remain in the report. No raw
alignment, stationary averaging, fitted axis or tracking-policy changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scanner_stopped_abba import STAGES, load_stage
from scipy.ndimage import binary_erosion, label

from scanner3d.cli import DEPTH_SCALE_METRES


def anchor_indices(rows: list[dict[str, Any]]) -> list[int]:
    """One fixed anchor nearest each one-second-bin centre, without view scoring."""
    window = [i for i, row in enumerate(rows) if row["in_comparison_window"]]
    if not window:
        raise ValueError("Missing declared first comparison window")
    times = np.array([rows[i]["monotonic_s"] for i in window])
    targets = times[0] + np.arange(60) + 0.5
    selected = [window[int(np.argmin(np.abs(times - target)))] for target in targets]
    if len(set(selected)) != 60:
        raise ValueError("Insufficient distinct deliveries for all 60 one-second bins")
    return selected


def descriptor(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = raw[::8, ::8].astype(np.float64) * DEPTH_SCALE_METRES
    return depth, (depth > 60) & (depth < 300)


def nearest_view(
    reference: tuple[np.ndarray, np.ndarray],
    depths: np.ndarray,
    masks: np.ndarray,
    source_indices: list[int],
) -> dict[str, Any]:
    """Old missing-aware score, plus explicit bounded diagnostic matchability."""
    depth, valid = reference
    if len(depths) != len(masks) or len(depths) != len(source_indices) or not len(depths):
        raise ValueError("Nonempty one-to-one descriptor/index evidence required")
    if depth.shape != valid.shape or depths.shape[1:] != depth.shape or masks.shape != depths.shape:
        raise ValueError("Descriptor shapes disagree")
    mutual = masks & valid
    union = masks | valid
    counts = mutual.sum(axis=(1, 2))
    union_counts = union.sum(axis=(1, 2))
    residual = np.minimum(np.abs(depths - depth), 10)
    scores = (
        (residual * mutual).sum(axis=(1, 2)) + 10 * (union & ~mutual).sum(axis=(1, 2))
    ) / np.maximum(1, union_counts)
    scores[counts == 0] = np.inf  # two empty images must never look like a perfect view
    nearest = int(np.argmin(scores))
    count = int(counts[nearest])
    iou = float(count / union_counts[nearest]) if union_counts[nearest] else None
    difference = (
        float(np.median(np.abs(depths[nearest][mutual[nearest]] - depth[mutual[nearest]])))
        if count
        else None
    )
    score = float(scores[nearest]) if np.isfinite(scores[nearest]) else None
    reasons = []
    if count < 100:
        reasons.append("fewer_than_100_mutual_descriptor_pixels")
    if score is None or score > 1.5:
        reasons.append("missing_aware_score_above_1.5")
    if iou is None or iou < 0.8:
        reasons.append("descriptor_mask_iou_below_0.8")
    if difference is None or difference > 1.5:
        reasons.append("median_mutual_depth_difference_above_1.5")
    return dict(
        source_index=source_indices[nearest],
        score=score,
        mutual_descriptor_pixels=count,
        descriptor_mask_iou=iou,
        median_mutual_depth_difference=difference,
        comparable=not reasons,
        reasons=reasons,
    )


def paired_regions(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    a = reference.astype(np.float64) * DEPTH_SCALE_METRES
    b = candidate.astype(np.float64) * DEPTH_SCALE_METRES
    va, vb = (a > 60) & (a < 300), (b > 60) & (b < 300)
    labels, count = label(va)
    if count:
        populations = np.bincount(labels.ravel())
        populations[0] = 0
        component = labels == populations.argmax()
    else:
        component = np.zeros_like(va)
    core = binary_erosion(component, iterations=3)
    result = {}
    for name, mask in dict(
        all_reference_returns=va, component=component, core=core, edge=component & ~core
    ).items():
        common = mask & vb
        errors = np.abs(b[common] - a[common])
        result[name] = dict(
            reference_pixels=int(mask.sum()),
            candidate_retained_pixels=int(common.sum()),
            return_retention=float(common.sum() / mask.sum()) if mask.any() else None,
            absolute_depth_difference_p50_p90=np.percentile(errors, [50, 90]).tolist()
            if errors.size
            else None,
        )
    return result


def verified_raw(row: dict[str, Any]) -> np.ndarray:
    path = Path(row["source_path"])
    raw = path.read_bytes()
    if path.is_symlink() or len(raw) != 512000 or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Raw source changed after initial provenance verification")
    return np.frombuffer(raw, dtype="<u2").reshape(400, 640)


def common_bin_summary(comparisons: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    control = comparisons["a2-20000"]
    result = {}
    for name in ("b1-14000", "b2-14000"):
        candidate = comparisons[name]
        if len(candidate) != 60 or len(control) != 60:
            raise ValueError("Every comparison must preserve all 60 anchor bins")
        paired = []
        for index, (a, b) in enumerate(zip(control, candidate, strict=True)):
            if a["anchor_source_index"] != b["anchor_source_index"]:
                raise ValueError("Control and candidate must use identical anchor views")
            if a["match"]["comparable"] and b["match"]["comparable"]:
                paired.append(index)
        region_stats = {}
        for region in ("all_reference_returns", "component", "core", "edge"):
            pairs = [
                (
                    control[i]["regions"][region]["return_retention"],
                    candidate[i]["regions"][region]["return_retention"],
                )
                for i in paired
            ]
            measured = [(a, b) for a, b in pairs if a is not None and b is not None]
            region_stats[region] = dict(
                measured_bins=len(measured),
                control_retention_p10_p50_p90=np.percentile(
                    [a for a, _ in measured], [10, 50, 90]
                ).tolist()
                if measured
                else None,
                candidate_retention_p10_p50_p90=np.percentile(
                    [b for _, b in measured], [10, 50, 90]
                ).tolist()
                if measured
                else None,
                candidate_minus_control_p10_p50_p90=np.percentile(
                    [b - a for a, b in measured], [10, 50, 90]
                ).tolist()
                if measured
                else None,
            )
        result[name] = dict(
            total_anchor_bins=60,
            comparable_to_control_and_candidate_bins=paired,
            omitted_from_matched_summary_bins=[i for i in range(60) if i not in paired],
            candidate_comparable_bins=sum(row["match"]["comparable"] for row in candidate),
            control_comparable_bins=sum(row["match"]["comparable"] for row in control),
            regions=region_stats,
        )
    return result


def run(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    result: dict[str, Any] = dict(
        scope=__doc__,
        quality_qualification_allowed=False,
        status="running",
        stages=[],
        comparisons={},
        limitations=[
            "Image matchability is descriptive, not independent angular or physical-motion truth.",
            "Different views, occlusion and motion-corrupted depth can all worsen match scores.",
            "A1 first-60-second anchors need not span one complete turn; unmatched bins remain.",
            "Core/edge masks can include support. No semantic full-object coverage claim.",
            "Matched-subset retention cannot compensate for omitted or unmatchable views.",
        ],
    )
    sources = {}
    anchor_raws = []
    previous_time = None
    try:
        for name, timing in STAGES:
            window, info = load_stage(root / name, timing)
            del window  # Loader validates first window; no stationary averaging here.
            if previous_time is not None and info["rows"][0]["monotonic_s"] <= previous_time:
                raise ValueError("Capture stages are not in declared chronological order")
            previous_time = info["rows"][-1]["monotonic_s"]
            result["stages"].append({**info, "name": name})
            if name == "a1-20000":
                anchors = anchor_indices(info["rows"])
                result["anchor_source_indices"] = anchors
                anchor_raws = [verified_raw(info["rows"][i]) for i in anchors]
            else:
                sources[name] = info
        reference_descriptors = [descriptor(raw) for raw in anchor_raws]
        for name in ("a2-20000", "b1-14000", "b2-14000"):
            info = sources[name]
            start = info["comparison_source_indices"][0]
            indices = list(range(start, len(info["rows"])))
            descriptors = [descriptor(verified_raw(info["rows"][i])) for i in indices]
            depths = np.stack([d for d, _ in descriptors])
            masks = np.stack([m for _, m in descriptors])
            rows: list[dict[str, Any]] = []
            result["comparisons"][name] = rows
            for bin_index, (raw, desc) in enumerate(
                zip(anchor_raws, reference_descriptors, strict=True)
            ):
                match = nearest_view(desc, depths, masks, indices)
                selected = info["rows"][match["source_index"]]
                rows.append(
                    dict(
                        bin_index=bin_index,
                        anchor_source_index=result["anchor_source_indices"][bin_index],
                        match=match,
                        selected_path=selected["source_path"],
                        selected_sha256=selected["sha256"],
                        regions=paired_regions(raw, verified_raw(selected)),
                    )
                )
            print(
                name, "comparable", sum(r["match"]["comparable"] for r in rows), "/ 60", flush=True
            )
        result["common_bin_summary"] = common_bin_summary(result["comparisons"])
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
