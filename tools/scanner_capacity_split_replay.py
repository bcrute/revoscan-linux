"""M12 closed capacity-split replay with truthful hash-bound capacity evidence."""

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
from scanner_capacity_split import ORIGINAL_SETTINGS, PROFILES, settings
from scanner_factorial_config import Factors

from scanner3d.tracking import IncrementalTracker

_ACTIVE: ContextVar[bool] = ContextVar("m12_capacity_split", default=False)


def run(
    capture: Path, output: Path, factors: Factors, *, capacity_profile: str,
    solver_profile: str, static: bool, frames: int, skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
) -> None:
    preparation_target, policy = settings(capacity_profile, factors, solver_profile, None)
    if output.exists() or _ACTIVE.get():
        raise ValueError("M12 requires a new output directory and a single nonnested adapter")
    if static and (frames != 320 or skip_complete or depth_band is not None):
        raise ValueError("M12 static cells require all 320 inputs, no skip or band override")
    if not 100 <= frames <= 3000:
        raise ValueError("M12 requires 100..3000 original input frames")
    replay = static_replay if static else motion_replay
    if (
        scanner_source_capacity.settings is not ORIGINAL_SETTINGS
        or replay.IncrementalTracker is not IncrementalTracker
    ):
        raise RuntimeError("M12 requires original unpatched capacity hook and tracker")
    report: dict[str, Any] = dict(
        scope=__doc__, status="running", quality_qualification_allowed=False,
        capacity_profile=capacity_profile, diagnostic_source_profile=capacity_profile,
        diagnostic_solver_profile=solver_profile, configuration=factors.identifier,
        diagnostic_factors=factors.as_dict(), diagnostic_model_profile=None,
        capture=str(capture), requested_frames=frames, static=static,
        tracking_policy=policy.as_dict(), preparation_target_points=preparation_target,
        underlying_reports=[],
        limitations=["Original replay and hash-bound capacity report must be read together.",
                     "Repeated recordings and concurrent timings are not qualification."],
    )
    calls = 0

    def capacity(
        profile: str, actual_factors: Factors, solver: str | None, model: str | None
    ) -> tuple[int, Any]:
        nonlocal calls
        calls += 1
        if calls != 1 or (profile, actual_factors, solver, model) != (
            capacity_profile, factors, solver_profile, None
        ):
            raise RuntimeError("M12 requires exactly one declared capacity settings call")
        return settings(profile, actual_factors, solver, model)

    token = _ACTIVE.set(True)
    try:
        with patch.object(scanner_source_capacity, "settings", capacity):
            if static:
                static_replay.run_static(capture, output, factors,
                    diagnostic_solver_profile=solver_profile,
                    diagnostic_source_profile=capacity_profile)
            else:
                motion_replay.run(capture, output, frames, (1,), skip_complete=skip_complete,
                    depth_band=depth_band, diagnostic_scene=True, normals_backend="gpu",
                    diagnostic_factors=factors.values, diagnostic_solver_profile=solver_profile,
                    diagnostic_source_profile=capacity_profile)
        if calls != 1:
            raise RuntimeError("M12 declared capacity hook was not called")
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
            with (output / "capacity-split-report.json").open("x") as metadata_stream:
                json.dump(report, metadata_stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", choices=("grid-pooled-residual",), required=True)
    parser.add_argument("--mechanism", choices=("observe-origin", "centered"), required=True)
    parser.add_argument("--capacity", choices=PROFILES, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(args.capture, args.output, Factors(*args.configuration.split("-")),
        capacity_profile=args.capacity, solver_profile=args.mechanism,
        static=args.static, frames=args.frames, skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band))


if __name__ == "__main__":
    main()
