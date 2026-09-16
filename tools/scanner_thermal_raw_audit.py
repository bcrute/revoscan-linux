"""Compare every retained soak minute to one fixed initial view; no alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scanner_scene_audit import plot
from scipy.ndimage import label
from stationary_spatial_report import save_plot, summary
from stationary_thermal_compare import compare


def excerpts(path: Path) -> tuple[dict[str, Any], dict[int, list[np.ndarray]]]:
    metadata = json.loads(path.read_bytes())
    if metadata.get("requested_timing_us") != 20000:
        raise ValueError("Baseline-only comparison; no cross-setting pooling")
    entries = metadata["raw_excerpts"]
    if not 10 <= len(entries) <= 300:
        raise ValueError("Bounded 10..300 retained raw excerpts required")
    groups: dict[int, list[np.ndarray]] = {}
    seen = set()
    for entry in entries:
        raw_path = Path(entry["path"])
        if raw_path in seen or raw_path.is_symlink():
            raise ValueError("Raw excerpt names must be unique and not linked")
        seen.add(raw_path)
        payload = raw_path.read_bytes()
        if len(payload) != 512000 or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Raw excerpt hash/length mismatch")
        minute = int(entry["elapsed_s"] // 60)
        groups.setdefault(minute, []).append(np.frombuffer(payload, dtype="<u2").reshape(400, 640))
    if any(len(group) != 10 for group in groups.values()):
        raise ValueError("Each compared minute requires all ten declared excerpts")
    return metadata, groups


def fixed_masks(reference: np.ndarray) -> dict[str, np.ndarray]:
    _, frequency, _ = summary(reference)
    components, count = label(frequency >= 0.5)
    if not count:
        raise ValueError("No persistent initial depth component")
    populations = np.bincount(components.ravel())
    populations[0] = 0
    return dict(
        whole_initial_return_footprint=frequency >= 0.1,
        largest_initial_component=components == int(populations.argmax()),
    )


def run(reference_path: Path, capture_path: Path, output: Path) -> dict[str, Any]:
    initial_metadata, initial = excerpts(reference_path)
    metadata, groups = excerpts(capture_path)
    for key in ("serial", "verified", "verified_gain_register"):
        if initial_metadata.get(key) != metadata.get(key):
            raise ValueError("Initial/current readback labels disagree")
    reference = np.stack(initial[0])
    masks = fixed_masks(reference)
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        scope=__doc__,
        quality_qualification_allowed=False,
        reference=str(reference_path),
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        capture=str(capture_path),
        capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        capture_status=metadata["status"],
        raw_excerpts=metadata["raw_excerpts"],
        masks={name: int(mask.sum()) for name, mask in masks.items()},
        minutes=[],
        limitations=[
            "Nominal depth units, not calibrated CAD accuracy or thermal causality.",
            "Whole-frame raw data retained; largest connected region is not semantic segmentation.",
            "Same initial fixed pixels for all minutes and stages; no pose or bias subtraction.",
            "Ten-frame minute excerpts do not describe unretained inter-minute depth.",
        ],
    )
    selected_plots = {min(groups), sorted(groups)[len(groups) // 2], max(groups)}
    for minute, frames in sorted(groups.items()):
        stack = np.stack(frames)
        report["minutes"].append(
            dict(
                minute=minute,
                full_frame_nonzero_fraction_min_max=[
                    float(np.min(np.mean(stack > 0, axis=(1, 2)))),
                    float(np.max(np.mean(stack > 0, axis=(1, 2)))),
                ],
                regions={name: compare(reference, stack, mask) for name, mask in masks.items()},
            )
        )
        if minute in selected_plots:
            save_plot(output / f"minute-{minute:02d}-first-raw.png", plot(stack[0]))
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(output, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.capture, args.output)
