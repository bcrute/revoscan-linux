"""Matched-pixel return coverage and repeatability for a stationary timing A/B/A.

No scanner access. Interior AND edge coverage are reported; repeatability is
not absolute accuracy. Both baseline repetitions expose drift during the test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, label

from scanner3d.cli import DEPTH_SCALE_METRES


def frames(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("frame_*.raw"))
    if len(paths) != 40:
        raise ValueError("A comparison requires all 40 complete settled frames per setting")
    return np.stack([np.frombuffer(p.read_bytes(), dtype="<u2").reshape(400, 640) for p in paths])


def analyze(
    directory: Path,
    roi: tuple[int, int, int, int] | None = None,
    reference_stages: tuple[int, int] | None = None,
) -> dict[str, Any]:
    metadata = json.loads((directory / "report.json").read_text())
    validity_path = directory / "quality-validity.json"
    if validity_path.exists():
        validity = json.loads(validity_path.read_text())
        if validity.get("quality_qualification_allowed") is not True:
            raise ValueError("Operator validity warning: use an uncontaminated replacement block")
    if metadata["status"] != "completed" or not metadata["restoration_verified"]:
        raise ValueError("Stationary sweep did not complete and verify restoration")
    timings = metadata["timings"]
    exposures = metadata.get("exposures", timings)
    gains = metadata.get("gains")
    if (
        not 3 <= len(timings) <= 10
        or len(exposures) != len(timings)
        or (timings[0], exposures[0], timings[-1], exposures[-1]) != (20000,) * 4
        or len(metadata["sample_times"]) != len(timings)
        or (gains is not None and (len(gains) != len(timings) or gains[0] != 24 or gains[-1] != 24))
    ):
        raise ValueError("Missing bracketing baseline or inconsistent stage labels")
    for stamps in metadata["sample_times"]:
        if len(stamps) != 40 or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
            raise ValueError("Need 40 finite, strictly increasing settled timestamps per stage")
    first, last = reference_stages or (0, len(timings) - 1)
    if not 0 <= first < last < len(timings):
        raise ValueError("Invalid reference stage indices")
    if (timings[first], exposures[first], gains[first] if gains else None) != (
        timings[last],
        exposures[last],
        gains[last] if gains else None,
    ):
        raise ValueError("Reference stages must have identical timing, exposure and gain")
    initial = frames(directory / f"{first}-{timings[first]}")
    final = frames(directory / f"{last}-{timings[last]}")
    baseline = np.concatenate([initial, final])
    valid = (baseline * DEPTH_SCALE_METRES > 60) & (baseline * DEPTH_SCALE_METRES < 300)
    frequency = valid.mean(axis=0)
    regions, count = label(frequency >= 0.5)
    if not count:
        raise ValueError("No connected object in the baseline")
    populations = np.bincount(regions.ravel())
    populations[0] = 0
    mask = regions == int(np.argmax(populations))
    if roi is not None:
        x0, y0, x1, y1 = roi
        if not 0 <= x0 < x1 <= mask.shape[1] or not 0 <= y0 < y1 <= mask.shape[0]:
            raise ValueError("ROI is outside depth image")
        cropped = np.zeros_like(mask)
        cropped[y0:y1, x0:x1] = True
        mask &= cropped
    core = binary_erosion(mask, iterations=3)
    edge = mask & ~core
    stable = frequency[mask] >= 0.9
    if not stable.any() or not core.any() or not edge.any():
        raise ValueError("Baseline has insufficient stable/interior/edge coverage")

    def depths(raw: np.ndarray) -> np.ndarray:
        values = raw[:, mask].astype(np.float64) * DEPTH_SCALE_METRES
        values[(values <= 60) | (values >= 300)] = np.nan
        return values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = np.nanmedian(depths(baseline), axis=0)
        baseline_change = np.nanmedian(depths(final), axis=0) - np.nanmedian(
            depths(initial), axis=0
        )
        runs = []
        for index, timing in enumerate(metadata["timings"]):
            raw = frames(directory / f"{index}-{timing}")
            measured = depths(raw)
            present = np.isfinite(measured)
            returns = present.mean(axis=0)
            median = np.nanmedian(measured, axis=0)
            bias = median - reference
            mad = np.nanmedian(np.abs(measured - median), axis=0)
            common = stable & np.isfinite(median)
            stamps = metadata["sample_times"][index]
            runs.append(
                {
                    "timing_us": timing,
                    "exposure_us": exposures[index],
                    "gain_register": gains[index] if gains is not None else None,
                    "settled_delivery_fps": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
                    "settled_gap_ms_p50_p95_p99": np.percentile(
                        np.diff(stamps) * 1000, [50, 95, 99]
                    ).tolist(),
                    "unique_frames": len(
                        {hashlib.sha256(frame.tobytes()).hexdigest() for frame in raw}
                    ),
                    "object_return_fraction": float(present.mean()),
                    "core_return_fraction": float(present[:, core[mask]].mean()),
                    "edge_return_fraction": float(present[:, edge[mask]].mean()),
                    "persistently_lost_stable_pixels": int(
                        np.count_nonzero(stable & (returns < 0.1))
                    ),
                    "persistently_lost_stable_fraction": float(np.mean(returns[stable] < 0.1)),
                    "median_signed_bias_mm": float(np.median(bias[common])),
                    "absolute_bias_mm_p50_p90_p99": np.percentile(
                        np.abs(bias[common]), [50, 90, 99]
                    ).tolist(),
                    "temporal_mad_mm_p50_p90": np.percentile(mad[common], [50, 90]).tolist(),
                }
            )
    return {
        "scope": (
            "Stationary matched-pixel coverage/repeatability, not absolute accuracy or metrology"
        ),
        "source": str(directory),
        "roi_xyxy": roi,
        "reference_stages": [first, last],
        "segmentation_scope": (
            "Largest baseline connected region, optionally cropped; not semantic segmentation"
        ),
        "object_mask_pixels": int(mask.sum()),
        "core_pixels": int(core.sum()),
        "edge_pixels": int(edge.sum()),
        "stable_pixels": int(stable.sum()),
        "baseline_change_absolute_mm_p50_p90": np.nanpercentile(
            np.abs(baseline_change[stable]), [50, 90]
        ).tolist(),
        "runs": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--reference-stages", type=int, nargs=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite report")
    report = analyze(
        args.capture,
        tuple(args.roi) if args.roi else None,
        tuple(args.reference_stages) if args.reference_stages else None,
    )
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
