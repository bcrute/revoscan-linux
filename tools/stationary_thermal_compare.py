"""Fixed-pixel warm-versus-initial comparison; no alignment or thermal causality claim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from analyze_stationary_quality import analyze, frames
from scipy.ndimage import binary_erosion, label
from stationary_spatial_report import summary


def compare(initial: np.ndarray, warm: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    a, fa, ma = summary(initial)
    b, fb, mb = summary(warm)
    stable = mask & (fa >= 0.9)
    common = stable & np.isfinite(a) & np.isfinite(b)
    core = binary_erosion(mask, iterations=3)
    if not common.any() or not core.any() or not (mask & ~core).any():
        raise ValueError("Insufficient stable, interior or edge depth for a warm comparison")
    changes = b[common] - a[common]
    return dict(
        initial_stable_pixels=int(stable.sum()),
        mutually_observed_stable_pixels=int(common.sum()),
        newly_persistently_lost_pixels=int((stable & (fb < 0.1)).sum()),
        newly_persistently_lost_fraction=float(np.mean(fb[stable] < 0.1)),
        signed_depth_change_mm_p10_p50_p90=np.percentile(changes, [10, 50, 90]).tolist(),
        absolute_depth_change_mm_p50_p90_p99=np.percentile(np.abs(changes), [50, 90, 99]).tolist(),
        temporal_mad_initial_mm_p50_p90=np.percentile(ma[common], [50, 90]).tolist(),
        temporal_mad_warm_mm_p50_p90=np.percentile(mb[common], [50, 90]).tolist(),
        coverage={
            name: dict(
                initial=float(fa[region].mean()),
                warm=float(fb[region].mean()),
                warm_relative_to_initial=float(fb[region].mean() / fa[region].mean()),
            )
            for name, region in dict(overall=mask, interior=core, edge=mask & ~core).items()
        },
    )


def run(initial: Path, warm: Path, roi: tuple[int, int, int, int], output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("New output required")
    reports = [analyze(path, roi, (1, 7)) for path in (initial, warm)]
    metadata = [json.loads((path / "report.json").read_text()) for path in (initial, warm)]
    for key in ("serial", "timings", "exposures", "gains"):
        if metadata[0].get(key) != metadata[1].get(key):
            raise ValueError("Warm and initial capture labels must match")
    if not metadata[0]["sample_times"][-1][-1] < metadata[1]["sample_times"][0][0]:
        raise ValueError("Warm capture must follow the initial capture")
    reference = np.concatenate([frames(initial / f"{i}-20000") for i in (1, 7)])
    _, frequency, _ = summary(reference)
    components, _ = label(frequency >= 0.5)
    populations = np.bincount(components.ravel())
    populations[0] = 0
    mask = components == populations.argmax()
    x0, y0, x1, y1 = roi
    crop = np.zeros_like(mask)
    crop[y0:y1, x0:x1] = True
    mask &= crop
    if int(mask.sum()) != reports[0]["object_mask_pixels"]:
        raise RuntimeError("Initial reference mask did not reproduce")
    result: dict[str, Any] = dict(
        scope=(
            "Warm versus initial at fixed pixels; "
            "not alignment, absolute accuracy or proof of thermal causality"
        ),
        initial=str(initial),
        warm=str(warm),
        roi_xyxy=roi,
        mask_pixels=int(mask.sum()),
        mask_source="Initial clean references 1/7 only; never refitted to warm data",
        initial_temperatures=metadata[0]["temperatures"],
        warm_temperatures=metadata[1]["temperatures"],
        runs=[],
    )
    for i, timing in enumerate(metadata[0]["timings"]):
        arrays = [frames(path / f"{i}-{timing}") for path in (initial, warm)]
        result["runs"].append(
            dict(
                stage=i,
                timing_us=timing,
                exposure_us=metadata[0]["exposures"][i],
                gain_register=metadata[0]["gains"][i],
                initial_frame_sha256=[
                    hashlib.sha256(raw.tobytes()).hexdigest() for raw in arrays[0]
                ],
                warm_frame_sha256=[hashlib.sha256(raw.tobytes()).hexdigest() for raw in arrays[1]],
                **compare(arrays[0], arrays[1], mask),
            )
        )
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
    for row in result["runs"]:
        print({k: v for k, v in row.items() if not k.endswith("sha256")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=Path)
    parser.add_argument("warm", type=Path)
    parser.add_argument("--roi", type=int, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.initial, args.warm, tuple(args.roi), args.output)
