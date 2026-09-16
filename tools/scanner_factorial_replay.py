"""One F1 interaction cell, offline only; combinations never qualify a scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Any

from scanner_candidate_audit import projection
from scanner_factorial_config import CONVERGENCE, MODEL_NORMALS, SAMPLING, Factors
from scanner_model_normals import NativeModelTracker
from scanner_motion_replay import run as run_motion
from scanner_pixel_grid import prepare_frame as prepare_grid
from scanner_pixel_rank import prepare_frame as prepare_rank
from scanner_pose_settling import pose_settling

from scanner3d.capture_sequence import TRACKING_LOSS_STREAK
from scanner3d.geometry import Rigid
from scanner3d.gpu_normals import live_normals
from scanner3d.pipeline import frame_travel
from scanner3d.preparation import DepthBand, prepare_frame
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker

STATIC_COUNT = 320


def load_static(capture: Path) -> tuple[list[bytes], list[dict[str, Any]], dict[str, Any]]:
    """Original static prefix, explicitly separate from moving settling validation."""
    paths = (capture.parent / "report.json", capture / "capture-progress.json")
    if capture.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError("Linked static evidence is not supported")
    parent_bytes, progress_bytes = (p.read_bytes() for p in paths)
    parent, progress = json.loads(parent_bytes), json.loads(progress_bytes)
    if parent.get("status") != "completed" or parent.get("restoration_verified") is not True:
        raise ValueError("Static acquisition must complete and verify restoration")
    entries = progress["frames"]
    if progress.get("capture_error") is not None or len(entries) != STATIC_COUNT:
        raise ValueError("The fixed 320-frame static acquisition must be complete")
    names = [e["path"] for e in entries]
    stamps = [e["monotonic_s"] for e in entries]
    if (
        len(set(names)) != len(names)
        or not all(math.isfinite(t) for t in stamps)
        or any(b <= a for a, b in pairwise(stamps))
    ):
        raise ValueError("Static inputs require unique paths and finite increasing timestamps")
    raws, inputs = [], []
    for entry in entries:
        path = capture / entry["path"]
        if path.parent != capture or Path(entry["path"]).is_absolute() or path.is_symlink():
            raise ValueError("Static raw input must be a direct unlinked child")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != 512000 or entry["bytes"] != len(raw) or entry["sha256"] != digest:
            raise ValueError("Static raw hash/length disagrees with its capture evidence")
        raws.append(raw)
        inputs.append(
            dict(
                path=str(path),
                sha256=digest,
                monotonic_s=entry["monotonic_s"],
                delivery_index=entry["delivery_index"],
            )
        )
    provenance = {
        str(path): hashlib.sha256(payload).hexdigest()
        for path, payload in zip(paths, (parent_bytes, progress_bytes), strict=True)
    }
    validity = capture.parent / "quality-validity.json"
    if validity.is_symlink():
        raise ValueError("Linked static scene-validity evidence is not supported")
    return (
        raws,
        inputs,
        dict(
            raw_progress=str(paths[1]),
            raw_progress_sha256=provenance[str(paths[1])],
            source_reports=provenance,
            source_scene_validity=json.loads(validity.read_bytes()) if validity.exists() else None,
        ),
    )


def static_checks(
    raws: list[bytes], positions: dict[int, Rigid], band: DepthBand
) -> list[dict[str, Any]]:
    """Keep all twenty frozen checks, including endpoints beyond an early stop."""
    if len(raws) != STATIC_COUNT:
        raise ValueError("Static checks require the full 320-frame input")
    identity = Rigid.identity()
    rows = []
    for index in (*range(16, STATIC_COUNT, 16), STATIC_COUNT - 1):
        pair = [raws[0], raws[index]]
        placed = (
            projection(pair, [positions[0], positions[index]], band)
            if 0 in positions and index in positions
            else None
        )
        checked = placed is not None and placed["agreement"] is not None
        rows.append(
            dict(
                source_index=index,
                raw_identity=projection(pair, [identity, identity], band),
                raw_placed=placed,
                checked=checked,
                reason=None if checked else "endpoint_missing_or_unprojectable",
            )
        )
    return rows


def run_static(
    capture: Path,
    output: Path,
    factors: Factors,
    *,
    diagnostic_solver_profile: str | None = None,
    diagnostic_model_profile: str | None = None,
    diagnostic_source_profile: str | None = None,
) -> None:
    preparation_target_points, policy = 900, DEFAULT_POLICY
    if diagnostic_source_profile is not None:
        from scanner_source_capacity import settings

        preparation_target_points, policy = settings(
            diagnostic_source_profile, factors, diagnostic_solver_profile, diagnostic_model_profile
        )
    if diagnostic_model_profile is not None:
        from scanner_model_representation import PROFILES as MODEL_PROFILES

        if diagnostic_model_profile not in MODEL_PROFILES or diagnostic_solver_profile not in (
            None,
            "observe-origin",
            "centered",
        ):
            raise ValueError("M4 requires closed model profile and original/centered solver")
    solver = None
    if diagnostic_solver_profile is not None:
        from scanner_solver_registry import create_experiment

        solver = create_experiment(diagnostic_solver_profile)
    raws, inputs, provenance = load_static(capture)
    output.mkdir(exist_ok=False)
    band = DepthBand(60, 300)
    prepare = {"stride": prepare_frame, "rank": prepare_rank, "grid": prepare_grid}[
        factors.sampling
    ]
    tracker: IncrementalTracker
    if diagnostic_model_profile is not None:
        from scanner_model_representation import ModelRepresentationTracker

        tracker = ModelRepresentationTracker(
            backend="scipy",
            policy=policy,
            model_profile=diagnostic_model_profile,
            normal_mode=factors.model_normals,
        )
    else:
        tracker = (
            NativeModelTracker(backend="scipy", policy=policy)
            if factors.model_normals == "native"
            else IncrementalTracker(backend="scipy", policy=policy)
        )
    report: dict[str, Any] = dict(
        status="running",
        scope=__doc__,
        quality_qualification_allowed=False,
        capture=str(capture),
        diagnostic_static=True,
        diagnostic_factors=factors.as_dict(),
        diagnostic_solver_profile=diagnostic_solver_profile,
        diagnostic_model_profile=diagnostic_model_profile,
        diagnostic_source_profile=diagnostic_source_profile,
        preparation_target_points=preparation_target_points,
        solver_records=None if solver is None else solver.records,
        configuration=factors.identifier,
        sampling_rule=factors.sampling_rule,
        model_normal_rule=factors.model_normal_rule,
        convergence_rule=factors.convergence_rule,
        input_frames=inputs,
        depth_band=band.as_dict(),
        tracking_policy=policy.as_dict(),
        messages=[],
        decisions=[],
        stationary_checks=[],
        accepted_source_indices=[],
        poses=[],
        model_normal_records=getattr(tracker, "model_normal_records", None),
        model_representation_records=getattr(tracker, "model_representation_records", None),
        limitations=[
            "All 320 original static inputs, no settling exclusion; no moving-source exemption.",
            "Identity is a fixed-camera diagnostic, not independently measured physical pose.",
            "Concurrent offline timing is not a live GUI or serial throughput qualification.",
        ],
        **provenance,
    )
    started = time.monotonic()
    positions: dict[int, Rigid] = {}
    streak = 0
    try:
        stopping = pose_settling() if factors.convergence == "settling" else nullcontext()
        with (
            live_normals(None, report["messages"].append),
            stopping,
            nullcontext() if solver is None else solver.active(),
        ):
            if not any(m.startswith("GPU-assisted tracking:") for m in report["messages"]):
                raise RuntimeError("Factorial static replay requires actual GPU activation")
            for index, raw in enumerate(raws):
                if time.monotonic() - started > 900:
                    raise TimeoutError("Static replay exceeded its unchanged 900-second bound")
                report["source_index_in_progress"] = index
                tick = time.perf_counter()
                cloud = prepare(raw, target_points=preparation_target_points, depth_band=band)
                preparation_s = time.perf_counter() - tick
                tick = time.perf_counter()
                if solver is not None:
                    solver.mark_frame(index)
                decision = tracker.offer(cloud.points)
                row: dict[str, Any] = dict(
                    source_index=index,
                    decision=decision.as_dict(),
                    preparation_s=preparation_s,
                    tracking_s=time.perf_counter() - tick,
                    prepared_points=len(cloud.points),
                    deprojected_points=cloud.deprojected_points,
                    clipped_points=cloud.clipped_points,
                )
                report["decisions"].append(row)
                if decision.accepted:
                    assert decision.pose is not None
                    positions[index] = decision.pose
                    streak = 0
                    row["surface_displacement_from_identity"] = frame_travel(
                        cloud.points, decision.pose
                    )
                    row["rotation_from_identity_rad"] = decision.pose.rotation_angle
                    report["accepted_source_indices"].append(index)
                    report["poses"].append(
                        dict(rotation=decision.pose.rotation, translation=decision.pose.translation)
                    )
                else:
                    streak += 1
                if (index + 1) % 100 == 0:
                    print("STATIC", factors.identifier, index + 1, len(positions), flush=True)
                if streak >= TRACKING_LOSS_STREAK:
                    break
        report["stationary_checks"] = static_checks(raws, positions, band)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(
            elapsed_s=time.monotonic() - started,
            offered=len(report["decisions"]),
            accepted=len(positions),
            tracking_stopped=streak >= TRACKING_LOSS_STREAK,
        )
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling", choices=SAMPLING, required=True)
    parser.add_argument("--model-normals", choices=MODEL_NORMALS, required=True)
    parser.add_argument("--convergence", choices=CONVERGENCE, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    parser.add_argument("--diagnostic-scene", action="store_true")
    args = parser.parse_args()
    factors = Factors(args.sampling, args.model_normals, args.convergence)
    if args.static:
        if args.frames != STATIC_COUNT or args.skip_complete or args.depth_band is not None:
            parser.error("Static cells require exactly 320 frames and no moving-source overrides")
        run_static(args.capture, args.output, factors)
    else:
        run_motion(
            args.capture,
            args.output,
            args.frames,
            (1,),
            skip_complete=args.skip_complete,
            depth_band=None if args.depth_band is None else tuple(args.depth_band),
            diagnostic_scene=args.diagnostic_scene,
            normals_backend="gpu",
            diagnostic_factors=factors.values,
        )


if __name__ == "__main__":
    main()
