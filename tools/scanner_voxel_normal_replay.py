"""M17 composition of unchanged M11 voxel model and M13 normal-support adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import scanner_voxel_model_replay as voxel_replay
from scanner_factorial_config import Factors
from scanner_normal_scale import CAPACITIES, PROFILES, NormalScaleExperiment
from scanner_normal_scale import settings as normal_settings
from scanner_voxel_model import settings as voxel_settings

MODEL_PROFILE = "voxel-first-model"
_ACTIVE: ContextVar[bool] = ContextVar("m17_voxel_normal_adapter", default=False)


def run(
    capture: Path, output: Path, factors: Factors, *, capacity_profile: str,
    normal_profile: str, solver_profile: str, static: bool, frames: int,
    skip_complete: int = 0, depth_band: tuple[float, float] | None = None,
) -> None:
    target, policy, source = voxel_settings(
        capacity_profile, MODEL_PROFILE, factors, solver_profile)
    normal_target, normal_policy, _ = normal_settings(
        capacity_profile, normal_profile, factors, solver_profile)
    if target != normal_target or policy != normal_policy:
        raise RuntimeError("M17 component capacities disagree")
    if output.exists() or _ACTIVE.get():
        raise ValueError("M17 requires a new output directory and a single nonnested adapter")
    if static and (frames != 320 or skip_complete or depth_band is not None):
        raise ValueError("M17 static cells require all 320 inputs, no skip or band override")
    if not 100 <= frames <= 3000:
        raise ValueError("M17 requires 100..3000 original input frames")
    experiment = NormalScaleExperiment(normal_profile)
    report: dict[str, Any] = dict(scope=__doc__, status="running",
        quality_qualification_allowed=False, capacity_profile=capacity_profile,
        sampling_profile=MODEL_PROFILE, normal_profile=normal_profile,
        diagnostic_solver_profile=solver_profile, diagnostic_source_profile=source,
        diagnostic_model_profile=None, configuration=factors.identifier,
        diagnostic_factors=factors.as_dict(), preparation_target_points=target,
        tracking_policy=policy.as_dict(), capture=str(capture), requested_frames=frames,
        static=static, fit_records=experiment.records, underlying_reports=[],
        limitations=["Unchanged component code does not guarantee the combination's geometry.",
                     "Read original replay, model companion and outer normal companion together.",
                     "Concurrent development timings do not qualify live speed or reliability."])
    token = _ACTIVE.set(True)
    try:
        with experiment.active():
            voxel_replay.run(capture, output, factors, sampling_profile=MODEL_PROFILE,
                capacity_profile=capacity_profile, solver_profile=solver_profile,
                static=static, frames=frames, skip_complete=skip_complete, depth_band=depth_band)
        if not experiment.records:
            raise RuntimeError("M17 requires actual tracker fit records")
        required = {"report.json", "model-sampling-report.json"}
        if not static:
            required.add("stride-1.json")
        if not all((output / name).is_file() for name in required):
            raise RuntimeError("M17 underlying replay/model evidence is incomplete")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _ACTIVE.reset(token)
        if output.is_dir():
            for name in ("report.json", "stride-1.json", "model-sampling-report.json"):
                path = output / name
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    report["underlying_reports"].append(dict(path=str(path), sha256=digest))
            with (output / "voxel-normal-report.json").open("x") as metadata_stream:
                json.dump(report, metadata_stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", choices=("grid-pooled-residual",), required=True)
    parser.add_argument("--mechanism", choices=("observe-origin", "centered"), required=True)
    parser.add_argument("--capacity", choices=CAPACITIES, required=True)
    parser.add_argument("--normal-scale", choices=PROFILES, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(args.capture, args.output, Factors(*args.configuration.split("-")),
        capacity_profile=args.capacity, normal_profile=args.normal_scale,
        solver_profile=args.mechanism, static=args.static, frames=args.frames,
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band))


if __name__ == "__main__":
    main()
