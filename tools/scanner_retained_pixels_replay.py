"""M14 retained-pixel/normal-scale replay with paired immutable trajectory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from unittest.mock import patch

import scanner_factorial_replay as static_replay
import scanner_motion_replay as motion_replay
import scanner_source_capacity
from scanner_capacity_split import ORIGINAL_SETTINGS
from scanner_factorial_config import Factors
from scanner_normal_scale import NormalScaleExperiment
from scanner_retained_pixels import PROFILES, SOURCE_PROFILE, RetainedPixelTracker, settings

from scanner3d.tracking import IncrementalTracker, TrackingPolicy

_ACTIVE: ContextVar[bool] = ContextVar("m14_retained_pixel_replay", default=False)


def run(
    capture: Path, output: Path, factors: Factors, *, retention_profile: str,
    normal_profile: str, solver_profile: str, static: bool, frames: int,
    skip_complete: int = 0, depth_band: tuple[float, float] | None = None,
) -> None:
    target, policy = settings(retention_profile, normal_profile, factors, solver_profile)
    if output.exists() or _ACTIVE.get():
        raise ValueError("M14 requires a new output and a single nonnested adapter")
    if static and (frames != 320 or skip_complete or depth_band is not None):
        raise ValueError("M14 static cells require 320 inputs with no skip or band override")
    if not 100 <= frames <= 3000:
        raise ValueError("M14 requires 100..3000 original input frames")
    replay = static_replay if static else motion_replay
    if (
        scanner_source_capacity.settings is not ORIGINAL_SETTINGS
        or replay.IncrementalTracker is not IncrementalTracker
    ):
        raise RuntimeError("M14 requires original unpatched capacity hook and tracker")
    normal = NormalScaleExperiment(normal_profile)
    instances: list[RetainedPixelTracker] = []
    calls = 0
    report: dict[str, Any] = dict(
        scope=__doc__, status="running", quality_qualification_allowed=False,
        retention_profile=retention_profile, normal_profile=normal_profile,
        diagnostic_source_profile=SOURCE_PROFILE, diagnostic_model_profile=None,
        diagnostic_solver_profile=solver_profile, configuration=factors.identifier,
        diagnostic_factors=factors.as_dict(), capture=str(capture), requested_frames=frames,
        static=static, tracking_policy=policy.as_dict(), preparation_target_points=target,
        actual_normal_neighbours=normal.neighbours, fit_records=normal.records,
        retention_records=[], underlying_reports=[],
        limitations=["Original trajectory and hash-bound companion must be read together.",
                     "Reused recordings and concurrent timings do not establish qualification."],
    )

    def capacity(
        profile: str, actual_factors: Factors, solver: str | None, model: str | None
    ) -> tuple[int, TrackingPolicy]:
        nonlocal calls
        calls += 1
        if calls != 1 or (profile, actual_factors, solver, model) != (
            SOURCE_PROFILE, factors, solver_profile, None
        ):
            raise RuntimeError("M14 requires exactly one declared capacity call")
        return target, policy

    def tracker(**kwargs: Any) -> RetainedPixelTracker:
        if instances:
            raise RuntimeError("M14 requires exactly one tracker with stride one")
        instance = RetainedPixelTracker(retention_profile=retention_profile, **kwargs)
        instances.append(instance)
        report["retention_records"] = instance.retention_records
        return instance

    token = _ACTIVE.set(True)
    try:
        with (
            patch.object(scanner_source_capacity, "settings", capacity),
            patch.object(replay, "IncrementalTracker", tracker), normal.active()
        ):
            if static:
                static_replay.run_static(capture, output, factors,
                    diagnostic_solver_profile=solver_profile,
                    diagnostic_source_profile=SOURCE_PROFILE)
            else:
                motion_replay.run(capture, output, frames, (1,), skip_complete=skip_complete,
                    depth_band=depth_band, diagnostic_scene=True, normals_backend="gpu",
                    diagnostic_factors=factors.values, diagnostic_solver_profile=solver_profile,
                    diagnostic_source_profile=SOURCE_PROFILE)
        if calls != 1 or len(instances) != 1 or not normal.records or any(
            r["status"] not in ("completed", "refused") for r in normal.records
        ) or not report["retention_records"] or any(
            r["status"] != "completed" for r in report["retention_records"]
        ):
            raise RuntimeError("M14 actual capacity/tracker/fit/retention evidence is incomplete")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _ACTIVE.reset(token)
        report["capacity_settings_calls"] = calls
        if output.is_dir():
            for name in ("report.json", "stride-1.json"):
                path = output / name
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    report["underlying_reports"].append(dict(path=str(path), sha256=digest))
            with (output / "retention-report.json").open("x") as metadata_stream:
                json.dump(report, metadata_stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", choices=("grid-pooled-residual",), required=True)
    parser.add_argument("--mechanism", choices=("observe-origin", "centered"), required=True)
    parser.add_argument("--retention", choices=PROFILES, required=True)
    parser.add_argument("--normal-scale", choices=("observe-10", "normal-20"), required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(args.capture, args.output, Factors(*args.configuration.split("-")),
        retention_profile=args.retention, normal_profile=args.normal_scale,
        solver_profile=args.mechanism, static=args.static, frames=args.frames,
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band))


if __name__ == "__main__":
    main()
