"""Offline per-frame scene-change diagnostics, not an automatic arm/quality classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, label
from stationary_spatial_report import save_plot, summary

from scanner3d.cli import DEPTH_SCALE_METRES


def read_raw(path: Path) -> tuple[np.ndarray, str]:
    payload = path.read_bytes()
    if len(payload) != 512000:
        raise ValueError(f"Incomplete 640x400 Y16 frame: {path}")
    return np.frombuffer(payload, dtype="<u2").reshape(400, 640), hashlib.sha256(
        payload
    ).hexdigest()


def measurements(raw: np.ndarray, reference: np.ndarray, frequency: np.ndarray) -> dict[str, int]:
    depth = raw.astype(np.float64) * DEPTH_SCALE_METRES
    valid = (depth > 60) & (depth < 300)
    stable = frequency >= 0.9
    outside = valid & ~binary_dilation(frequency >= 0.1, iterations=5)
    components, _ = label(outside)
    populations = np.bincount(components.ravel())[1:]
    return dict(
        nonzero_pixels=int(np.count_nonzero(raw)),
        valid_60_300_mm_pixels=int(valid.sum()),
        added_outside_reference_5px_pixels=int(outside.sum()),
        largest_added_component_pixels=int(populations.max()) if populations.size else 0,
        missing_stable_reference_pixels=int((stable & ~valid).sum()),
        closer_than_reference_5mm_pixels=int((stable & valid & (depth < reference - 5)).sum()),
        farther_than_reference_5mm_pixels=int((stable & valid & (depth > reference + 5)).sum()),
    )


def plot(raw: np.ndarray) -> np.ndarray:
    depth = raw.astype(np.float64) * DEPTH_SCALE_METRES
    # Fixed scale in every plot; retain the full frame and all nonzero depth.
    level = np.clip((depth - 60) / 240, 0, 1)
    rgb = np.stack([255 * (1 - level), 220 * (1 - np.abs(2 * level - 1)), 255 * level], axis=-1)
    rgb[raw == 0] = 0
    return rgb


def run(reference_path: Path, captures: list[Path], output: Path) -> dict[str, Any]:
    paths = sorted(reference_path.glob("frame_*.raw"))
    if not 24 <= len(paths) <= 80 or not 1 <= len(captures) <= 12:
        raise ValueError("Need 24..80 reference frames and 1..12 raw directories")
    references = [read_raw(path) for path in paths]
    reference, frequency, _ = summary(np.stack([raw for raw, _ in references]))
    if not np.any(frequency >= 0.9):
        raise ValueError("No stable reference depth")
    output.mkdir(exist_ok=False, parents=True)
    result: dict[str, Any] = dict(
        scope="Per-frame scene change; no pose alignment, semantic segmentation or quality pass",
        limitations=[
            "A stationary contaminant in the reference can be missed; "
            "an arm with no depth return is invisible.",
            "Changes can also be caused by target movement, framing, "
            "exposure/gain or other objects.",
            "Five-pixel and five-mm thresholds are inspection aids, not replacement quality gates.",
            "All nonzero depth is plotted; numerical changes use "
            "the existing 60..300 mm analysis band.",
        ],
        reference=[
            dict(path=str(path), sha256=digest)
            for path, (_, digest) in zip(paths, references, strict=True)
        ],
        captures=[],
    )
    for index, capture in enumerate(captures):
        raw_paths = sorted(capture.glob("frame_*.raw"))
        if not 1 <= len(raw_paths) <= 80:
            raise ValueError("Each raw directory must contain 1..80 frames")
        records: list[dict[str, Any]] = []
        for path in raw_paths:
            raw, digest = read_raw(path)
            records.append(
                dict(path=str(path), sha256=digest, **measurements(raw, reference, frequency))
            )
        keys = list(measurements(references[0][0], reference, frequency))
        extremes = {
            key: dict(min=min(r[key] for r in records), max=max(r[key] for r in records))
            for key in keys
        }
        selected = {0, len(records) - 1}
        for key in keys[2:]:
            selected.add(max(range(len(records)), key=lambda number: records[number][key]))
        plots = []
        for number in sorted(selected):
            raw, digest = read_raw(raw_paths[number])
            if digest != records[number]["sha256"]:
                raise RuntimeError("Input changed during analysis")
            name = f"{index:02d}-{number:04d}-depth.png"
            save_plot(output / name, plot(raw))
            plots.append(dict(plot=name, source=str(raw_paths[number])))
        result["captures"].append(
            dict(source=str(capture), extremes=extremes, frames=records, plots=plots)
        )
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            [
                dict(source=c["source"], extremes=c["extremes"], plots=c["plots"])
                for c in result["captures"]
            ],
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.capture, args.output)
