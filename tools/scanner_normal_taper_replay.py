"""M24 tapered-covariance replay, paired evidence and no automatic qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import scanner_factorial_replay as static_replay
import scanner_motion_replay as motion_replay
from scanner_factorial_config import Factors
from scanner_normal_taper import CAPACITIES, PROFILES, NormalTaperExperiment, settings

from scanner3d.tracking import IncrementalTracker

_ACTIVE: ContextVar[bool] = ContextVar("m24_replay_adapter", default=False)


def run(
    capture: Path,
    output: Path,
    factors: Factors,
    *,
    normal_profile: str,
    capacity_profile: str,
    solver_profile: str,
    static: bool,
    frames: int,
    skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
) -> None:
    preparation_target, policy, source_profile = settings(
        capacity_profile, normal_profile, factors, solver_profile
    )
    if output.exists() or _ACTIVE.get():
        raise ValueError("M24 requires a new output directory and a single nonnested adapter")
    if static:
        if frames != 320 or skip_complete != 0 or depth_band is not None:
            raise ValueError("M24 static cells require all 320 inputs, no skip or band override")
    elif skip_complete != 12 or depth_band != (60.0, 300.0):
        raise ValueError("M24 moving cells require original skip twelve and depth band 60..300")
    if not 100 <= frames <= 3000:
        raise ValueError("M24 requires 100..3000 original input frames")
    replay = static_replay if static else motion_replay
    if replay.IncrementalTracker is not IncrementalTracker:
        raise RuntimeError("M24 requires the original unpatched tracker factory")
    experiment = NormalTaperExperiment(normal_profile)
    report: dict[str, Any] = dict(
        scope=__doc__,
        status="running",
        quality_qualification_allowed=False,
        normal_profile=normal_profile,
        capacity_profile=capacity_profile,
        diagnostic_source_profile=source_profile,
        diagnostic_model_profile=None,
        diagnostic_solver_profile=solver_profile,
        configuration=factors.identifier,
        diagnostic_factors=factors.as_dict(),
        capture=str(capture),
        requested_frames=frames,
        static=static,
        tracking_policy=policy.as_dict(),
        preparation_target_points=preparation_target,
        actual_normal_neighbours=experiment.neighbours,
        normal_kernel="unweighted" if normal_profile == "observe-20" else "squared-distance-taper",
        normal_statistics_include_query=True,
        fit_records=experiment.records,
        field_records=experiment.field_records,
        underlying_reports=[],
        limitations=[
            "Original trajectory and hash-bound normal-taper report form paired evidence.",
            "Reused recordings and concurrent timings do not qualify accuracy or speed.",
        ],
    )
    token = _ACTIVE.set(True)
    try:
        with experiment.active():
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
        if not experiment.records or any(
            row["status"] not in ("completed", "refused") for row in experiment.records
        ):
            raise RuntimeError("M24 fit records are absent or incomplete")
        if not experiment.field_records or any(
            r["status"] != "completed" for r in experiment.field_records
        ):
            raise RuntimeError("M24 actual GPU field records are absent or incomplete")
        required = ("report.json",) if static else ("report.json", "stride-1.json")
        if not all((output / name).is_file() for name in required):
            raise RuntimeError("M24 underlying reports are incomplete")
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
            with (output / "normal-taper-report.json").open("x") as stream_text:
                json.dump(report, stream_text, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", choices=("grid-pooled-residual",), required=True)
    parser.add_argument("--mechanism", choices=("observe-origin", "centered"), required=True)
    parser.add_argument("--capacity", choices=CAPACITIES, required=True)
    parser.add_argument("--normal-taper", choices=PROFILES, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(
        args.capture,
        args.output,
        Factors(*args.configuration.split("-")),
        normal_profile=args.normal_taper,
        capacity_profile=args.capacity,
        solver_profile=args.mechanism,
        static=args.static,
        frames=args.frames,
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band),
    )


if __name__ == "__main__":
    main()
