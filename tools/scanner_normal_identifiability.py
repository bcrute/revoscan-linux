"""Offline PCA normal-identifiability diagnosis; no trajectory or normal changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scanner_pixel_grid import prepare_frame as prepare_grid
from scanner_pixel_rank import prepare_frame as prepare_rank
from scanner_revisit_audit import _verified_raw

from scanner3d.compute import select_backend
from scanner3d.icp_plane import estimate_normals
from scanner3d.preparation import DepthBand, prepare_frame
from scanner3d.registration import voxel_downsample

BINS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.000000000001)
MOVING_INDICES = (
    0,
    4,
    24,
    64,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
    352,
    384,
    448,
    512,
    560,
    624,
    627,
)


def spectrum(points: Any) -> dict[str, Any]:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 11 or not np.isfinite(p).all():
        raise ValueError("At least eleven finite 3D points required")
    indices = select_backend("scipy").backend.k_nearest_neighbors(points, 10)
    neighbors = np.column_stack((np.arange(len(p)), np.asarray(indices)))
    patches = p[neighbors]
    centered = patches - patches.mean(axis=1)[:, None, :]
    covariances = np.einsum("nki,nkj->nij", centered, centered)
    eigenvalues = np.linalg.eigvalsh(covariances)
    positive = eigenvalues[:, 2] > 0
    nonnegative = np.maximum(eigenvalues, 0)
    largest = np.where(positive, nonnegative[:, 2], 1.0)
    middle = np.where(nonnegative[:, 1] > 0, nonnegative[:, 1], 1.0)
    smallest_largest = nonnegative[:, 0] / largest
    smallest_middle = nonnegative[:, 0] / middle
    middle_largest = nonnegative[:, 1] / largest
    planarity = (nonnegative[:, 1] - nonnegative[:, 0]) / largest
    selected = positive & (smallest_largest <= 0.35)
    return dict(
        points=len(p),
        eigenvalues=eigenvalues.tolist(),
        negative_roundoff_eigenvalues=int(np.sum(eigenvalues < 0)),
        zero_extent_neighborhoods=int(np.sum(~positive)),
        existing_ratio_eligible=int(np.sum(selected)),
        near_collinear_eligible=int(np.sum(selected & (middle_largest <= 1e-9))),
        smallest_middle_above_035_eligible=int(np.sum(selected & (smallest_middle > 0.35))),
        histograms={
            key: np.histogram(value[selected], bins=BINS)[0].tolist()
            for key, value in (
                ("smallest_middle", smallest_middle),
                ("middle_largest", middle_largest),
                ("planarity", planarity),
            )
        },
        histogram_edges=BINS,
        limitation="Diagnostic eigensolver/classification, not GPU bit-equivalence or uncertainty.",
    )


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("A new evidence output is required")
    payload = plan_path.read_bytes()
    plan = json.loads(payload)
    report: dict[str, Any] = dict(
        status="running",
        source_plan=str(plan_path),
        source_plan_sha256=hashlib.sha256(payload).hexdigest(),
        quality_qualification_allowed=False,
        scanner_access=False,
        gpu_access=False,
        rows=[],
    )
    try:
        line = [(float(i), 0.0, 0.0) for i in range(31)]
        plane = [(float(x), float(y), 0.0) for x in range(7) for y in range(7)]
        report["controls"] = {
            label: dict(
                production_usable=len(estimate_normals(points, backend="scipy").usable),
                spectrum=spectrum(points),
            )
            for label, points in (("line", line), ("plane", plane))
        }
        for case in plan["cases"]:
            indices = (0, *range(16, 320, 16), 319) if case["static"] else MOVING_INDICES
            for index in indices:
                raw = _verified_raw(Path(case["capture"]), case["input_frames"][index])
                for sampling, prepare in (
                    ("stride", prepare_frame),
                    ("rank", prepare_rank),
                    ("grid", prepare_grid),
                ):
                    cloud = prepare(
                        raw, target_points=900, depth_band=DepthBand(**case["depth_band"])
                    )
                    sampled = voxel_downsample(cloud.points, 1.5)
                    row = dict(
                        case=case["name"],
                        source_index=index,
                        sampling=sampling,
                        input=case["input_frames"][index],
                        prepared_points=len(cloud.points),
                        voxel_points=len(sampled),
                        spectrum=spectrum(sampled),
                    )
                    report["rows"].append(row)
            print("NORMALS", case["name"], len(indices) * 3, flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.plan, arguments.output)
