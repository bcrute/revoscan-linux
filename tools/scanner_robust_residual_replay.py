"""M20 closed robust-residual solver factory with unchanged core replay reports."""

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
import scanner_solver_registry
from scanner_factorial_config import Factors
from scanner_robust_residual import CAPACITIES, MODES, PROFILES, RobustResidualExperiment, settings
from scanner_solver_mechanisms import SolverExperiment

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.tracking import IncrementalTracker

ORIGINAL_FACTORY = scanner_solver_registry.create_experiment
_ACTIVE: ContextVar[bool] = ContextVar("m20_robust_factory", default=False)


def run(
    capture: Path,
    output: Path,
    factors: Factors,
    *,
    kernel_profile: str,
    capacity_profile: str,
    solver_profile: str,
    static: bool,
    frames: int,
    skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
) -> None:
    target, policy, source_profile = settings(
        capacity_profile, kernel_profile, factors, solver_profile
    )
    if output.exists() or _ACTIVE.get():
        raise ValueError("M20 requires a new output directory and nonnested adapter")
    if static and (frames != 320 or skip_complete or depth_band is not None):
        raise ValueError("M20 static cells require all 320 inputs, no skip or band override")
    if not 100 <= frames <= 3000:
        raise ValueError("M20 requires 100..3000 original inputs")
    replay = static_replay if static else motion_replay
    if (
        scanner_solver_registry.create_experiment is not ORIGINAL_FACTORY
        or replay.IncrementalTracker is not IncrementalTracker
    ):
        raise RuntimeError("M20 requires the original solver and tracker factories")
    report: dict[str, Any] = dict(
        scope=__doc__,
        status="running",
        quality_qualification_allowed=False,
        kernel_profile=kernel_profile,
        capacity_profile=capacity_profile,
        diagnostic_source_profile=source_profile,
        diagnostic_model_profile=None,
        diagnostic_solver_profile=solver_profile,
        configuration=factors.identifier,
        diagnostic_factors=factors.as_dict(),
        capture=str(capture),
        requested_frames=frames,
        static=static,
        preparation_target_points=target,
        tracking_policy=policy.as_dict(),
        actual_normal_neighbours=10,
        residual_scale_floor=DEPTH_SCALE_METRES,
        residual_scale_rule="max(raw_code, median(abs(r))/0.6744897501960817)",
        underlying_reports=[],
        step_records=[],
        limitations=[
            "Original trajectory and hash-bound robust report form paired evidence.",
            "Robust step costs do not replace externally reported unweighted RMSE.",
            "Reused recordings and concurrent timings do not qualify accuracy or speed.",
        ],
    )
    calls = 0

    def factory(profile: str) -> SolverExperiment | RobustResidualExperiment:
        nonlocal calls
        calls += 1
        if calls != 1 or profile != solver_profile:
            raise RuntimeError("M20 requires exactly one declared solver-factory call")
        experiment: SolverExperiment | RobustResidualExperiment
        if kernel_profile == "observe-l2":
            original = ORIGINAL_FACTORY(profile)
            if not isinstance(original, SolverExperiment):
                raise RuntimeError("M20 observation requires the original finite solver")
            experiment = original
        else:
            experiment = RobustResidualExperiment(kernel_profile, profile)
        report["step_records"] = experiment.records
        return experiment

    token = _ACTIVE.set(True)
    try:
        with patch.object(scanner_solver_registry, "create_experiment", factory):
            if static:
                static_replay.run_static(
                    capture,
                    output,
                    factors,
                    diagnostic_solver_profile=solver_profile,
                    diagnostic_source_profile=source_profile,
                )
            else:
                motion_replay.run(
                    capture,
                    output,
                    frames,
                    (1,),
                    skip_complete=skip_complete,
                    depth_band=depth_band,
                    diagnostic_scene=True,
                    normals_backend="gpu",
                    diagnostic_factors=factors.values,
                    diagnostic_solver_profile=solver_profile,
                    diagnostic_source_profile=source_profile,
                )
        if calls != 1 or not report["step_records"]:
            raise RuntimeError("M20 requires the declared solver call and actual step records")
        required = ("report.json",) if static else ("report.json", "stride-1.json")
        if not all((output / name).is_file() for name in required):
            raise RuntimeError("M20 requires complete underlying replay reports")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _ACTIVE.reset(token)
        report["solver_factory_calls"] = calls
        if output.is_dir():
            for name in ("report.json", "stride-1.json"):
                path = output / name
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    report["underlying_reports"].append(dict(path=str(path), sha256=digest))
            with (output / "robust-residual-report.json").open("x") as stream_text:
                json.dump(report, stream_text, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", choices=("grid-pooled-residual",), required=True)
    parser.add_argument("--mechanism", choices=MODES, required=True)
    parser.add_argument("--capacity", choices=CAPACITIES, required=True)
    parser.add_argument("--kernel", choices=PROFILES, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(
        args.capture,
        args.output,
        Factors(*args.configuration.split("-")),
        kernel_profile=args.kernel,
        capacity_profile=args.capacity,
        solver_profile=args.mechanism,
        static=args.static,
        frames=args.frames,
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band),
    )


if __name__ == "__main__":
    main()
