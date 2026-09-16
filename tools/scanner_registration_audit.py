"""Offline trace of unchanged ICP stopping and local directional information."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scanner_candidate_audit import pose, projection
from scanner_revisit_audit import _verified_raw

from scanner3d import icp_plane
from scanner3d.compute import select_backend
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.pipeline import frame_travel
from scanner3d.preparation import DepthBand, prepare_frame
from scanner3d.registration import IDENTITY_POSE, kept_count, voxel_downsample
from scanner3d.tracking import DEFAULT_POLICY, convergence_tolerance


def information(moved: np.ndarray, target: np.ndarray, normals: np.ndarray) -> dict[str, Any]:
    if (
        moved.shape != target.shape
        or moved.shape != normals.shape
        or moved.ndim != 2
        or moved.shape[1] != 3
        or len(moved) < 6
    ):
        raise ValueError("At least six matched finite point-normal triples required")
    if not all(np.isfinite(a).all() for a in (moved, target, normals)):
        raise ValueError("Finite geometry required")
    centred = moved - moved.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))
    if radius <= 0:
        raise ValueError("Nonzero measured surface extent required")
    jacobian = np.concatenate([normals, np.cross(centred / radius, normals)], axis=1)
    hessian = jacobian.T @ jacobian
    values, vectors = np.linalg.eigh(hessian)
    rank = int(np.sum(values > max(float(values[-1]), 1e-30) * 1e-12))
    residuals = np.sum((moved - target) * normals, axis=1)
    variance = float(residuals @ residuals / max(1, len(moved) - 6))
    return dict(
        pairs=len(moved),
        centred_rms_radius=radius,
        eigenvalues=values.tolist(),
        rank=rank,
        condition=float(values[-1] / values[0]) if rank == 6 else None,
        weakest_direction=vectors[:, 0].tolist(),
        residual_rmse=float(np.sqrt(np.mean(residuals**2))),
        linear_weak_direction_sigma=math.sqrt(variance / float(values[0])) if rank == 6 else None,
        coordinate_order="centred translation xyz, rotation xyz times RMS radius",
        limitation="Fixed-correspondence linear diagnostic only; correlated depth and estimated "
        "normals invalidate a calibrated probability interpretation",
    )


def final_correspondences(
    source: Sequence[Point], target: Sequence[Point], transform: Rigid
) -> list[tuple[Point, Point, Point]]:
    sampled_source = voxel_downsample(source, 1.5)
    sampled_target = voxel_downsample(target, 1.5)
    field = icp_plane._cached_normals(tuple(sampled_target), 10, "scipy")
    points = [e.point for e in field.usable if e.normal is not None]
    by_point = {e.point: e.normal for e in field.usable if e.normal is not None}
    pairs, _ = icp_plane._correspond_via_backend(
        sampled_source, transform, select_backend("scipy").backend, points, by_point, gate=12.0
    )
    pairs.sort(key=lambda p: p[0])
    keep = min(len(pairs), kept_count(len(pairs), 0.8))
    return [(p[1], p[2], p[3]) for p in pairs[:keep]]


def traced_fit(
    source: Sequence[Point], target: Sequence[Point], *, initial: Rigid = IDENTITY_POSE
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    original_step = icp_plane._solve_point_to_plane

    def traced_step(matched: Any, current: Rigid, *, accelerated: bool = False) -> Rigid:
        updated = original_step(matched, current, accelerated=accelerated)
        moved = [(updated.apply(o), q, n) for o, q, n in matched]
        trace.append(
            dict(
                iteration=len(trace) + 1,
                surface_step=frame_travel(
                    tuple(current.apply_all(source)), updated.compose(current.inverse())
                ),
                residual=icp_plane._point_to_plane_rmse(moved),
            )
        )
        return updated

    record: dict[str, Any] = dict(trace=trace, initial=initial.as_dict())
    icp_plane._solve_point_to_plane = traced_step
    try:
        result = icp_plane.icp_point_to_plane(
            source,
            target,
            initial=initial,
            max_correspondence_distance=12.0,
            voxel_size=1.5,
            keep_fraction=0.8,
            backend="scipy",
            tolerance=convergence_tolerance(
                target, fraction=DEFAULT_POLICY.convergence_noise_fraction
            ),
        )
        record["result"] = result.as_dict()
    except (ScannerError, ValueError, ArithmeticError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    finally:
        icp_plane._solve_point_to_plane = original_step
    try:
        matched = final_correspondences(source, target, result.transform)
        moved = np.asarray([result.transform.apply(o) for o, _, _ in matched])
        record["information"] = information(
            moved, np.asarray([q for _, q, _ in matched]), np.asarray([n for _, _, n in matched])
        )
        next_pose = original_step(matched, result.transform, accelerated=True)
        record["one_more_surface_step"] = frame_travel(
            tuple(result.transform.apply_all(source)), next_pose.compose(result.transform.inverse())
        )
    except (ScannerError, ValueError, ArithmeticError) as error:
        record["information_error"] = f"{type(error).__name__}: {error}"
    return record


def run(path: Path, output: Path, *, handheld: bool = False) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(status="running", quality_qualification_allowed=False, pairs=[])
    try:
        loop_bytes = path.read_bytes()
        loop = json.loads(loop_bytes)
        source_path = path if handheld else Path(loop["source"])
        source_bytes = source_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if not handheld and source_hash != loop["source_sha256"]:
            raise ValueError("Source replay changed")
        replay = json.loads(source_bytes)
        indices = replay["runs"][0]["accepted_source_indices"]
        inputs = replay["input_frames"]
        if len(inputs) > 3000:
            raise ValueError("3000 input bound")
        band = DepthBand(**replay["depth_band"])
        report.update(
            source_loop=str(path),
            source_loop_sha256=hashlib.sha256(loop_bytes).hexdigest(),
            source=str(source_path),
            source_sha256=source_hash,
        )
        schedule = loop.get("partition", [])
        if handheld:
            schedule = []
            for row in replay["runs"][0]["decisions"][-8:]:
                late = row["source_index"]
                previous = [i for i in indices if i < late]
                if not previous:
                    continue
                schedule.append(
                    dict(
                        ordinal=late,
                        sample=dict(frame=previous[-1], best_frame=late, credible=True),
                        original_decision=row["decision"],
                    )
                )
            report["scope"] = (
                "Handheld adjacent identity-start diagnostics; saved reports do not retain "
                "the motion-prior initial pose, and final fitted poses are not substituted for it"
            )
        for part in schedule:
            if not part["sample"]["credible"]:
                continue
            a, b = [
                part["sample"][k] if handheld else indices[part["sample"][k]]
                for k in ("frame", "best_frame")
            ]
            raws = [_verified_raw(Path(replay["capture"]), inputs[i]) for i in (a, b)]
            clouds = [prepare_frame(raw, target_points=900, depth_band=band).points for raw in raws]
            row = dict(
                ordinal=part["ordinal"],
                source_indices=[a, b],
                inputs=[inputs[a], inputs[b]],
                raw_identity=projection(raws, [Rigid.identity()] * 2, band),
                directions=[],
            )
            if handheld:
                row["original_decision"] = part["original_decision"]
            for forward in (True, False):
                target, source = clouds if forward else list(reversed(clouds))
                result = traced_fit(source, target)
                if "result" in result:
                    result["raw_projection"] = projection(
                        raws if forward else list(reversed(raws)),
                        [Rigid.identity(), pose(result["result"]["transform"])],
                        band,
                    )
                row["directions"].append(result)
            report["pairs"].append(row)
            print(
                "PAIR",
                part["ordinal"],
                [d.get("one_more_surface_step") for d in row["directions"]],
                flush=True,
            )
        report["status"] = "completed"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_loop", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handheld", action="store_true")
    arguments = parser.parse_args()
    run(arguments.source_loop, arguments.output, handheld=arguments.handheld)
