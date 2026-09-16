"""Offline spatial depth diagnostics; data plots, not reconstruction or scanner access.

The baseline-derived display scale and pixel coordinates stay fixed across
stages. NPZ retains full-resolution, unrounded measurements behind the plots.
No independent fit of candidate frames is allowed to hide a spatial depth bias.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from analyze_stationary_quality import analyze, frames
from PySide6.QtGui import QImage

from scanner3d.cli import DEPTH_SCALE_METRES


def summary(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = raw.astype(np.float64) * DEPTH_SCALE_METRES
    valid = (depth > 60) & (depth < 300)
    depth[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(depth, axis=0)
        mad = np.nanmedian(np.abs(depth - median), axis=0)
    return median, valid.mean(axis=0), mad


def save_plot(path: Path, rgb: np.ndarray) -> None:
    packed = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = packed.shape
    image = QImage(packed.data, width, height, width * 3, QImage.Format.Format_RGB888)
    if path.exists() or not image.save(str(path)):
        raise RuntimeError(f"Cannot create new plot: {path}")


def run(capture: Path, output: Path) -> None:
    # All completion, bracket and timestamp checks run before artifact creation.
    report = analyze(capture)
    metadata = json.loads((capture / "report.json").read_text())
    timing = metadata["timings"]
    output.mkdir(exist_ok=False, parents=True)
    baseline = np.concatenate(
        [frames(capture / "0-20000"), frames(capture / f"{len(timing) - 1}-20000")]
    )
    reference, frequency, baseline_mad = summary(baseline)
    low, high = np.nanpercentile(reference[frequency >= 0.9], [2, 98])
    if high <= low:
        raise ValueError("Insufficient depth variation for the reference display")
    data = dict(
        reference_mm=reference, reference_frequency=frequency, reference_mad_mm=baseline_mad
    )
    for index, window in enumerate(timing):
        median, present, mad = summary(frames(capture / f"{index}-{window}"))
        data[f"median_{index}_mm"] = median
        data[f"frequency_{index}"] = present
        data[f"mad_{index}_mm"] = mad
        signed = median - reference
        level = np.nan_to_num(np.clip((median - low) / (high - low), 0, 1))
        depth_plot = np.stack(
            [255 * (1 - level), 220 * (1 - np.abs(2 * level - 1)), 255 * level], axis=-1
        )
        depth_plot[~np.isfinite(median)] = 0
        # White=unchanged, red=farther, blue=nearer, black=missing. Saturates at 1 mm.
        bias_plot = np.full((*median.shape, 3), 255.0)
        amount = np.nan_to_num(np.clip(np.abs(signed), 0, 1)) * 255
        bias_plot[..., 0] -= np.where(signed < 0, amount, 0)
        bias_plot[..., 1] -= amount
        bias_plot[..., 2] -= np.where(signed > 0, amount, 0)
        bias_plot[~np.isfinite(signed)] = 0
        loss = np.maximum(frequency - present, 0)
        loss_plot = np.stack([255 * loss, 80 * present, 80 * present], axis=-1)
        prefix = f"{index}-{window}-{report['runs'][index]['exposure_us']}"
        save_plot(output / f"{prefix}-depth.png", depth_plot)
        save_plot(output / f"{prefix}-bias.png", bias_plot)
        save_plot(output / f"{prefix}-loss.png", loss_plot)
    with (output / "spatial.npz").open("xb") as stream:
        np.savez_compressed(stream, **data)  # type: ignore[arg-type]
    report["plot_scales"] = dict(
        depth_mm=[low, high], bias_saturation_mm=1, loss="red=lost return probability; grey=present"
    )
    report["segmentation_warning"] = (
        "Connected-region metrics are not semantic object segmentation. "
        "Plots preserve the entire scene, including support/background."
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.capture, args.output)
