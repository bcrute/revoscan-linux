"""Run one explicitly labelled offline solver-mechanism cell, with GPU normals."""

from __future__ import annotations

import argparse
from pathlib import Path

from scanner_factorial_config import CONFIGURATIONS
from scanner_factorial_replay import run_static
from scanner_model_representation import PROFILES as MODEL_PROFILES
from scanner_motion_replay import run as run_motion
from scanner_solver_registry import PROFILES
from scanner_source_capacity import PROFILES as SOURCE_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--configuration", choices=[c.identifier for c in CONFIGURATIONS], required=True
    )
    parser.add_argument("--mechanism", choices=PROFILES, required=True)
    parser.add_argument("--model-representation", choices=MODEL_PROFILES)
    parser.add_argument("--source-capacity", choices=SOURCE_PROFILES)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    args = parser.parse_args()
    factors = next(c for c in CONFIGURATIONS if c.identifier == args.configuration)
    if args.static:
        if args.frames != 320 or args.skip_complete or args.depth_band is not None:
            parser.error("Static mechanism cells require all 320 inputs and no band/skip override")
        run_static(
            args.capture,
            args.output,
            factors,
            diagnostic_solver_profile=args.mechanism,
            diagnostic_model_profile=args.model_representation,
            diagnostic_source_profile=args.source_capacity,
        )
    else:
        run_motion(
            args.capture,
            args.output,
            args.frames,
            (1,),
            skip_complete=args.skip_complete,
            depth_band=None if args.depth_band is None else tuple(args.depth_band),
            diagnostic_scene=True,
            normals_backend="gpu",
            diagnostic_factors=factors.values,
            diagnostic_solver_profile=args.mechanism,
            diagnostic_model_profile=args.model_representation,
            diagnostic_source_profile=args.source_capacity,
        )


if __name__ == "__main__":
    main()
