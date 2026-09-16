"""Offline moving-scene framing evidence; no stationary quality or semantic pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scanner_scene_audit import plot, read_raw
from scipy.ndimage import label
from stationary_spatial_report import save_plot

from scanner3d.cli import DEPTH_SCALE_METRES


def measurements(raw: np.ndarray) -> dict[str, Any]:
    depth = raw * DEPTH_SCALE_METRES
    valid = (depth > 60) & (depth < 300)
    labels, _ = label(valid)
    populations = np.bincount(labels.ravel())
    populations[0] = 0
    if not populations.max():
        return dict(component_pixels=0, bbox_xyxy=None, border_pixels=None)
    mask = labels == int(np.argmax(populations))
    yy, xx = np.nonzero(mask)
    return dict(
        valid_pixels=int(valid.sum()),
        component_pixels=int(mask.sum()),
        bbox_xyxy=[int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        border_pixels=int(
            mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum()
        ),
        depth_mm_p1_p50_p99=np.percentile(depth[mask], [1, 50, 99]).tolist(),
    )


def run(capture: Path, output: Path) -> dict[str, Any]:
    paths = sorted(capture.glob("frame_*.raw"))
    if not 32 <= len(paths) <= 2400:
        raise ValueError("Need 32..2400 complete raw frames")
    output.mkdir(exist_ok=False, parents=True)
    rows = []
    for path in paths:
        raw, digest = read_raw(path)
        rows.append(dict(path=str(path), sha256=digest, **measurements(raw)))
    selected = {round(i * (len(paths) - 1) / 8) for i in range(9)}
    visible = [i for i, row in enumerate(rows) if row["bbox_xyxy"] is not None]
    if visible:
        for coordinate in range(4):
            selected.add(min(visible, key=lambda i: rows[i]["bbox_xyxy"][coordinate]))
            selected.add(max(visible, key=lambda i: rows[i]["bbox_xyxy"][coordinate]))
        selected.add(max(visible, key=lambda i: rows[i]["border_pixels"]))
    plots = []
    for i in sorted(selected):
        raw, digest = read_raw(paths[i])
        if digest != rows[i]["sha256"]:
            raise RuntimeError("Input changed during analysis")
        name = f"{i:04d}-depth.png"
        save_plot(output / name, plot(raw))
        plots.append(dict(plot=name, source=str(paths[i])))
    result = dict(
        scope="Moving framing aid: largest 60..300 mm component may include support. "
        "No stationary averaging, semantic segmentation, pose correction or quality pass.",
        source=str(capture),
        frames=rows,
        plots=plots,
        frames_with_component_at_border=sum(bool(r["border_pixels"]) for r in rows),
        frames_without_component=len(rows) - len(visible),
    )
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.capture, args.output)
