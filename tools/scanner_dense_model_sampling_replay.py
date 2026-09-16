"""Run a closed M9 ordinary-model sample cell; paired evidence, never qualification."""

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
from scanner_dense_model_sampling import (
    EXPECTED_POLICY,
    PROFILES,
    DenseModelSamplingTracker,
    validate,
)
from scanner_factorial_config import Factors

from scanner3d.tracking import IncrementalTracker

_ACTIVE: ContextVar[bool] = ContextVar("m9_replay_adapter", default=False)


def run(
    capture: Path,
    output: Path,
    factors: Factors,
    *,
    sampling_profile: str,
    solver_profile: str,
    static: bool,
    frames: int,
    skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
) -> None:
    validate(sampling_profile, factors, solver_profile, None, "source-1800")
    if output.exists() or _ACTIVE.get():
        raise ValueError("M9 requires a new output directory and a single nonnested adapter")
    if static and (frames != 320 or skip_complete or depth_band is not None):
        raise ValueError("M9 static cells require all 320 inputs, no skip or band override")
    if not 100 <= frames <= 3000:
        raise ValueError("M9 requires 100..3000 original input frames")
    replay = static_replay if static else motion_replay
    if replay.IncrementalTracker is not IncrementalTracker:
        raise RuntimeError("M9 requires the original unpatched ordinary tracker factory")
    instances: list[DenseModelSamplingTracker] = []
    report: dict[str, Any] = dict(
        scope=__doc__,
        status="running",
        quality_qualification_allowed=False,
        sampling_profile=sampling_profile,
        diagnostic_solver_profile=solver_profile,
        configuration=factors.identifier,
        diagnostic_factors=factors.as_dict(),
        capture=str(capture),
        requested_frames=frames,
        static=static,
        tracking_policy=EXPECTED_POLICY.as_dict(),
        preparation_target_points=1800,
        diagnostic_source_profile="source-1800",
        diagnostic_model_profile=None,
        model_sampling_records=[],
        underlying_reports=[],
        limitations=[
            "Underlying trajectory report and this hash-bound diagnostic must be read together.",
            "Repeated development recordings and concurrent timings do not qualify "
            "geometry or speed.",
        ],
    )

    def tracker(**kwargs: Any) -> DenseModelSamplingTracker:
        if instances:
            raise RuntimeError("M9 requires exactly one tracker with stride one")
        instance = DenseModelSamplingTracker(sampling_profile=sampling_profile, **kwargs)
        instances.append(instance)
        report["model_sampling_records"] = instance.model_sampling_records
        return instance

    token = _ACTIVE.set(True)
    try:
        with patch.object(replay, "IncrementalTracker", tracker):
            if static:
                static_replay.run_static(
                    capture, output, factors, diagnostic_solver_profile=solver_profile,
                    diagnostic_source_profile="source-1800",
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
                    diagnostic_source_profile="source-1800",
                )
        if len(instances) != 1 or any(
            row["status"] != "completed" or not row["original_reference_verified"]
            for row in report["model_sampling_records"]
        ):
            raise RuntimeError("M9 tracker/model evidence is absent or incomplete")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _ACTIVE.reset(token)
        if output.is_dir():
            for name in ("report.json", "stride-1.json"):
                path = output / name
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    report["underlying_reports"].append(dict(path=str(path), sha256=digest))
            with (output / "model-sampling-report.json").open("x") as metadata_stream:
                json.dump(report, metadata_stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--configuration", choices=("stride-pooled-residual", "grid-pooled-residual"), required=True
    )
    parser.add_argument("--mechanism", choices=("observe-origin", "centered"), required=True)
    parser.add_argument("--model-sampling", choices=PROFILES, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(
        args.capture,
        args.output,
        Factors(*args.configuration.split("-")),
        sampling_profile=args.model_sampling,
        solver_profile=args.mechanism,
        static=args.static,
        frames=args.frames,
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band),
    )


if __name__ == "__main__":
    main()
