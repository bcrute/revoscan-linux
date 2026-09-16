"""M25 composes agreeing-candidate preference with the unchanged M21 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import scanner_reverse_seed_replay as original
from scanner_factorial_config import Factors
from scanner_reverse_seed import CAPACITIES, settings
from scanner_reverse_seed import PROFILES as SEED_PROFILES
from scanner_seed_selection import PROFILES, SeedSelectionExperiment

from scanner3d import tracking

ORIGINAL_REPLAY = original.run


def run(
    capture: Path,
    output: Path,
    *,
    selection_profile: str,
    seed_profile: str,
    capacity_profile: str,
    solver_profile: str,
    static: bool,
    frames: int,
    skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
) -> None:
    factors = Factors("grid", "pooled", "residual")
    target, policy, source_profile = settings(
        capacity_profile, seed_profile, factors, solver_profile
    )
    experiment = SeedSelectionExperiment(selection_profile)
    if output.exists() or original.run is not ORIGINAL_REPLAY:
        raise ValueError("M25 requires a new output and the original M21 delegate")
    if static:
        if frames != 320 or skip_complete or depth_band is not None:
            raise ValueError("M25 static cells require original 320-frame framing")
    elif skip_complete != 12 or depth_band != (60.0, 300.0):
        raise ValueError("M25 moving Stage A cells require original twelve-skip 60..300 framing")
    if not 100 <= frames <= 3000:
        raise ValueError("M25 requires 100..3000 original inputs")
    report: dict[str, Any] = dict(
        scope=__doc__,
        status="running",
        quality_qualification_allowed=False,
        selection_profile=selection_profile,
        seed_profile=seed_profile,
        capacity_profile=capacity_profile,
        diagnostic_solver_profile=solver_profile,
        diagnostic_source_profile=source_profile,
        diagnostic_model_profile=None,
        tracking_policy=policy.as_dict(),
        preparation_target_points=target,
        candidate_preference=list(experiment.preference),
        preference_observations=[],
        configuration=factors.identifier,
        static=static,
        requested_frames=frames,
        capture=str(capture),
        underlying_reports=[],
        limitations=[
            "Preference cannot resolve ambiguous agreement clusters.",
            "Selected overlap still drives original keyframe admission.",
            "Reused development data and concurrent timings do not qualify live performance.",
        ],
    )
    try:
        with experiment.active():
            report["preference_observations"].append(list(tracking.PREFERENCE_ORDER))
            original.run(
                capture,
                output,
                factors,
                seed_profile=seed_profile,
                capacity_profile=capacity_profile,
                solver_profile=solver_profile,
                static=static,
                frames=frames,
                skip_complete=skip_complete,
                depth_band=depth_band,
            )
            report["preference_observations"].append(list(tracking.PREFERENCE_ORDER))
        names = (
            ("report.json", "reverse-seed-report.json")
            if static
            else ("report.json", "stride-1.json", "reverse-seed-report.json")
        )
        if not all((output / name).is_file() for name in names):
            raise RuntimeError("M25 original reports are incomplete")
        document = json.loads((output / "report.json").read_bytes())
        if (
            document.get("status") != "completed"
            or document.get("tracking_policy") != policy.as_dict()
        ):
            raise RuntimeError("M25 original report status or policy differs")
        actual_runs = [document] if static else [document, document["runs"][0]]
        for actual in actual_runs:
            expected_metadata = dict(
                tracking_policy=policy.as_dict(),
                preparation_target_points=target,
                diagnostic_factors=factors.as_dict(),
                diagnostic_source_profile=source_profile,
                diagnostic_solver_profile=solver_profile,
                diagnostic_model_profile=None,
            )
            if actual.get("quality_qualification_allowed") is not False or any(
                actual.get(key) != value for key, value in expected_metadata.items()
            ):
                raise RuntimeError("M25 original report does not preserve declared metadata")
            present = "candidate_preference" in actual
            if (not static or present) and actual.get("candidate_preference") != list(
                experiment.preference
            ):
                raise RuntimeError(
                    "M25 reported candidate preference differs from active selection"
                )
        report["original_static_preference_field_present"] = (
            "candidate_preference" in document if static else None
        )
        seed = json.loads((output / "reverse-seed-report.json").read_bytes())
        if (
            seed.get("status") != "completed"
            or seed.get("seed_profile") != seed_profile
            or seed.get("capacity_profile") != capacity_profile
            or not seed.get("fit_records")
        ):
            raise RuntimeError("M25 original seed metadata is absent or differs")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if output.is_dir():
            for name in ("report.json", "stride-1.json", "reverse-seed-report.json"):
                path = output / name
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    report["underlying_reports"].append(dict(path=str(path), sha256=digest))
            with (output / "seed-selection-report.json").open("x") as destination:
                json.dump(report, destination, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", choices=PROFILES, required=True)
    parser.add_argument("--reverse-seed", choices=SEED_PROFILES, required=True)
    parser.add_argument("--capacity", choices=CAPACITIES, required=True)
    parser.add_argument("--mechanism", choices=("observe-origin", "centered"), required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    run(
        args.capture,
        args.output,
        selection_profile=args.selection,
        seed_profile=args.reverse_seed,
        capacity_profile=args.capacity,
        solver_profile=args.mechanism,
        static=args.static,
        frames=args.frames,
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band),
    )


if __name__ == "__main__":
    main()
