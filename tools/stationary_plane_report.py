"""Compare fixed planar image patches using recorded full-resolution median maps.

Candidate plane fits measure flatness only, not alignment or bias correction.
Depth shifts remain measured at unchanged pixels against the common baseline.
This is relative shape evidence: the print's own flatness is not calibrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.simulate import FACTORY_INTRINSICS


def fitted_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 50 or not np.isfinite(points).all():
        raise ValueError("Insufficient finite planar samples")
    centre = points.mean(axis=0)
    _, values, vectors = np.linalg.svd(points - centre, full_matrices=False)
    if values[1] <= 0 or values[1] / max(values[2], 1e-12) < 3:
        raise ValueError("Patch is not convincingly planar")
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    return centre, normal, (points - centre) @ normal


def run(spatial: Path, regions: Path) -> dict[str, Any]:
    labels = json.loads((spatial / "report.json").read_text())
    region_record = json.loads(regions.read_text())
    data = np.load(spatial / "spatial.npz")
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    reference = data["reference_mm"]
    yy, xx = np.indices(reference.shape)
    rays = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)
    report: dict[str, Any] = dict(source=str(spatial), region_record=region_record, regions={})
    for name, bounds in region_record["regions"].items():
        x0, y0, x1, y1 = bounds
        if not 0 <= x0 < x1 <= reference.shape[1] or not 0 <= y0 < y1 <= reference.shape[0]:
            raise ValueError("Region outside image")
        mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        mask &= data["reference_frequency"] >= 0.9
        try:
            centre, normal, residual = fitted_plane(rays[mask] * reference[mask, None])
        except ValueError as reference_problem:
            report["regions"][name] = dict(
                pixels=int(mask.sum()),
                status="invalid_reference_patch",
                error=str(reference_problem),
                runs=[],
            )
            continue
        results = []
        for index, stage in enumerate(labels["runs"]):
            median = data[f"median_{index}_mm"]
            common = mask & np.isfinite(median)
            points = rays[common] * median[common, None]
            try:
                _, fitted, error = fitted_plane(points)
            except ValueError as problem:
                results.append(
                    dict(
                        timing_us=stage["timing_us"],
                        exposure_us=stage["exposure_us"],
                        gain_register=stage.get("gain_register"),
                        status="invalid_candidate_patch",
                        error=str(problem),
                    )
                )
                continue
            fixed_error = (points - centre) @ normal
            results.append(
                dict(
                    timing_us=stage["timing_us"],
                    exposure_us=stage["exposure_us"],
                    gain_register=stage.get("gain_register"),
                    return_fraction=float(data[f"frequency_{index}"][mask].mean()),
                    flatness_abs_mm_p50_p90_p99=np.percentile(np.abs(error), [50, 90, 99]).tolist(),
                    flatness_rms_mm=float(np.sqrt(np.mean(error**2))),
                    normal_angle_from_baseline_deg=float(
                        np.degrees(np.arccos(np.clip(np.dot(normal, fitted), -1, 1)))
                    ),
                    fixed_reference_plane_signed_mm_p10_p50_p90=np.percentile(
                        fixed_error, [10, 50, 90]
                    ).tolist(),
                    unchanged_pixel_abs_depth_shift_mm_p50_p90=np.percentile(
                        np.abs(median[common] - reference[common]), [50, 90]
                    ).tolist(),
                    temporal_mad_mm_p50_p90=np.percentile(
                        data[f"mad_{index}_mm"][common], [50, 90]
                    ).tolist(),
                )
            )
        report["regions"][name] = dict(
            pixels=int(mask.sum()),
            baseline_flatness_abs_mm_p50_p90_p99=np.percentile(
                np.abs(residual), [50, 90, 99]
            ).tolist(),
            runs=results,
        )
    data.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spatial", type=Path)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite report")
    report = run(args.spatial, args.regions)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    for name, region in report["regions"].items():
        print(
            name,
            region.get("error", ""),
            [
                (
                    r["timing_us"],
                    r["exposure_us"],
                    round(r["flatness_rms_mm"], 4) if "flatness_rms_mm" in r else r["error"],
                )
                for r in region["runs"]
            ],
            flush=True,
        )
