"""M21 adapter with explicit historical framing; no tracking algorithm changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from unittest.mock import patch

import scanner_motion_replay as motion
import scanner_reverse_seed_replay as original
from scanner_factorial_config import Factors
from scanner_reverse_seed import CAPACITIES, PROFILES, settings

FRAMINGS = ("diagnostic-raw", "legacy-project")
ORIGINAL_MOTION = motion.run
_ACTIVE: ContextVar[bool] = ContextVar("m21_stage_b_framing", default=False)


def run(
    capture: Path,
    output: Path,
    *,
    framing: str,
    seed_profile: str,
    capacity_profile: str,
    solver_profile: str,
    frames: int,
) -> None:
    factors = Factors("grid", "pooled", "residual")
    target, policy, source = settings(capacity_profile, seed_profile, factors, solver_profile)
    if framing not in FRAMINGS or not 100 <= frames <= 3000 or output.exists() or _ACTIVE.get():
        raise ValueError("Stage B requires closed framing, 100..3000 frames and a new output")
    if motion.run is not ORIGINAL_MOTION:
        raise RuntimeError("Stage B requires the original motion runner")
    legacy = framing == "legacy-project"
    report: dict[str, Any] = dict(
        scope=__doc__,
        status="running",
        framing=framing,
        capture=str(capture),
        requested_frames=frames,
        seed_profile=seed_profile,
        capacity_profile=capacity_profile,
        diagnostic_solver_profile=solver_profile,
        configuration=factors.identifier,
        diagnostic_scene=not legacy,
        skip_complete=0 if legacy else 12,
        depth_band_override=None if legacy else [60.0, 300.0],
        preparation_target_points=target,
        tracking_policy=policy.as_dict(),
        quality_qualification_allowed=False,
        legacy_delegations=[],
        underlying_reports=[],
    )

    def legacy_call(*args: Any, **kwargs: Any) -> None:
        expected = dict(
            skip_complete=0,
            depth_band=None,
            diagnostic_scene=True,
            normals_backend="gpu",
            diagnostic_factors=factors.values,
            diagnostic_solver_profile=solver_profile,
            diagnostic_source_profile=source,
        )
        if (
            args != (capture, output, frames, (1,))
            or kwargs != expected
            or report["legacy_delegations"]
        ):
            raise RuntimeError("Stage B refuses an unexpected or repeated legacy delegation")
        report["legacy_delegations"].append(
            dict(
                original_scene=True,
                actual_scene=False,
                skip_complete=0,
                depth_band_override=None,
                changed_argument="diagnostic_scene",
            )
        )
        ORIGINAL_MOTION(*args, **(kwargs | dict(diagnostic_scene=False)))

    token = _ACTIVE.set(True)
    try:
        with patch.object(motion, "run", legacy_call) if legacy else nullcontext():
            original.run(
                capture,
                output,
                factors,
                seed_profile=seed_profile,
                capacity_profile=capacity_profile,
                solver_profile=solver_profile,
                static=False,
                frames=frames,
                skip_complete=0 if legacy else 12,
                depth_band=None if legacy else (60.0, 300.0),
            )
        if len(report["legacy_delegations"]) != int(legacy):
            raise RuntimeError("Stage B legacy route did not execute exactly once")
        required = ("report.json", "stride-1.json", "reverse-seed-report.json")
        if not all((output / name).is_file() for name in required):
            raise RuntimeError("Stage B underlying reports are incomplete")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _ACTIVE.reset(token)
        if output.is_dir():
            for name in ("report.json", "stride-1.json", "reverse-seed-report.json"):
                path = output / name
                if path.is_file():
                    report["underlying_reports"].append(
                        dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    )
            with (output / "framing-report.json").open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--framing", choices=FRAMINGS, required=True)
    parser.add_argument("--reverse-seed", choices=PROFILES, required=True)
    parser.add_argument("--capacity", choices=CAPACITIES, required=True)
    parser.add_argument("--mechanism", choices=("observe-origin", "centered"), required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    run(
        args.capture,
        args.output,
        framing=args.framing,
        seed_profile=args.reverse_seed,
        capacity_profile=args.capacity,
        solver_profile=args.mechanism,
        frames=args.frames,
    )


if __name__ == "__main__":
    main()
