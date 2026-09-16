"""M16 closed solver-factory adapter with untouched original replay reports."""

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
from scanner_covariance_descent import BASES, PROFILES, RotationAwareExperiment
from scanner_factorial_config import Factors
from scanner_generalized_icp import GeneralizedExperiment
from scanner_symmetric_icp import SymmetricExperiment

from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker

ORIGINAL_FACTORY = scanner_solver_registry.create_experiment
_ACTIVE: ContextVar[bool] = ContextVar("m16_covariance_factory", default=False)


def run(
    capture: Path, output: Path, factors: Factors, *, solver_profile: str,
    static: bool, frames: int, skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
) -> None:
    if factors.identifier not in BASES or solver_profile not in PROFILES:
        raise ValueError("M16 requires its closed original four-base/covariance solver roster")
    if output.exists() or _ACTIVE.get():
        raise ValueError("M16 requires a new output directory and a single nonnested adapter")
    if static and (frames != 320 or skip_complete or depth_band is not None):
        raise ValueError("M16 static cells require all 320 inputs, no skip or band override")
    if not 100 <= frames <= 3000:
        raise ValueError("M16 requires 100..3000 original input frames")
    replay = static_replay if static else motion_replay
    if (scanner_solver_registry.create_experiment is not ORIGINAL_FACTORY
        or replay.IncrementalTracker is not IncrementalTracker):
        raise RuntimeError("M16 requires the original unpatched solver factory and tracker")
    report: dict[str, Any] = dict(scope=__doc__, status="running",
        quality_qualification_allowed=False, diagnostic_solver_profile=solver_profile,
        configuration=factors.identifier, diagnostic_factors=factors.as_dict(),
        diagnostic_model_profile=None, diagnostic_source_profile=None,
        capture=str(capture), requested_frames=frames, static=static,
        tracking_policy=DEFAULT_POLICY.as_dict(), preparation_target_points=900,
        convergence_rule=factors.convergence_rule, underlying_reports=[],
        limitations=["Descent is not physical pose accuracy or full inner optimization.",
                     "Original replay and hash-bound companion must be read together.",
                     "Concurrent development timings are not live performance."])
    calls = 0

    def factory(profile: str) -> SymmetricExperiment:
        nonlocal calls
        calls += 1
        if calls != 1 or profile != solver_profile:
            raise RuntimeError("M16 requires exactly one declared solver-factory invocation")
        if profile == "generalized-covariance":
            original = ORIGINAL_FACTORY(profile)
            if not isinstance(original, GeneralizedExperiment):
                raise RuntimeError("M16 observation requires the original M5 experiment")
            return original
        return RotationAwareExperiment()

    token = _ACTIVE.set(True)
    try:
        with patch.object(scanner_solver_registry, "create_experiment", factory):
            if static:
                static_replay.run_static(capture, output, factors,
                                        diagnostic_solver_profile=solver_profile)
            else:
                motion_replay.run(capture, output, frames, (1,), skip_complete=skip_complete,
                    depth_band=depth_band, diagnostic_scene=True, normals_backend="gpu",
                    diagnostic_factors=factors.values, diagnostic_solver_profile=solver_profile)
        if calls != 1:
            raise RuntimeError("M16 declared solver-factory hook was not called")
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
            with (output / "covariance-gradient-report.json").open("x") as metadata_stream:
                json.dump(report, metadata_stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", choices=BASES, required=True)
    parser.add_argument("--mechanism", choices=PROFILES, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(args.capture, args.output, Factors(*args.configuration.split("-")),
        solver_profile=args.mechanism, static=args.static, frames=args.frames,
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band))


if __name__ == "__main__":
    main()
