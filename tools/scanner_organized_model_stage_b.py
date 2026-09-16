"""M27 model-normal scope composed with unchanged M21 historical framing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import scanner_reverse_seed_stage_b as original
from scanner_factorial_config import Factors
from scanner_organized_model import PROFILES, OrganizedModelExperiment
from scanner_reverse_seed import CAPACITIES, settings
from scanner_reverse_seed import PROFILES as SEED_PROFILES

from scanner3d import tracking

ORIGINAL_REPLAY = original.run


def run(
    capture: Path,
    output: Path,
    *,
    normal_profile: str,
    radius: int,
    maximum_weight: float,
    seed_profile: str,
    capacity_profile: str,
    solver_profile: str,
    framing: str,
    frames: int,
) -> None:
    factors = Factors("grid", "pooled", "residual")
    target, policy, source_profile = settings(
        capacity_profile, seed_profile, factors, solver_profile
    )
    experiment = OrganizedModelExperiment(
        normal_profile, capacity_profile, seed_profile, radius=radius, maximum_weight=maximum_weight
    )
    if (
        output.exists()
        or solver_profile != "centered"
        or original.run is not ORIGINAL_REPLAY
        or tracking.PREFERENCE_ORDER != ("model", "adjacent", "keyframe")
    ):
        raise ValueError("M27 Stage B requires new output, centered solver and original delegate")
    if framing not in original.FRAMINGS or not 100 <= frames <= 3000:
        raise ValueError("M27 Stage B requires closed framing and 100..3000 original inputs")
    legacy = framing == "legacy-project"
    report: dict[str, Any] = dict(
        scope=__doc__,
        status="running",
        quality_qualification_allowed=False,
        normal_profile=normal_profile,
        radius=radius,
        maximum_weight=maximum_weight,
        seed_profile=seed_profile,
        capacity_profile=capacity_profile,
        diagnostic_solver_profile=solver_profile,
        diagnostic_source_profile=source_profile,
        diagnostic_model_profile=None,
        tracking_policy=policy.as_dict(),
        preparation_target_points=target,
        candidate_preference=["model", "adjacent", "keyframe"],
        preference_observations=[],
        preparation_records=experiment.preparations,
        model_records=experiment.models,
        hooks_restored=False,
        configuration=factors.identifier,
        static=False,
        framing=framing,
        diagnostic_scene=not legacy,
        skip_complete=0 if legacy else 12,
        depth_band_override=None if legacy else [60.0, 300.0],
        requested_frames=frames,
        capture=str(capture),
        underlying_reports=[],
        limitations=[
            "Only forward model-target normal directions may change; no depth or model-point edit.",
            "Original pooled PCA fallback/availability and reverse fields remain unchanged.",
            "Reused development data and concurrent timings do not qualify live performance.",
        ],
    )
    try:
        with experiment.active():
            report["preference_observations"].append(list(tracking.PREFERENCE_ORDER))
            original.run(
                capture,
                output,
                framing=framing,
                seed_profile=seed_profile,
                capacity_profile=capacity_profile,
                solver_profile=solver_profile,
                frames=frames,
            )
            report["preference_observations"].append(list(tracking.PREFERENCE_ORDER))
        names = ("report.json", "stride-1.json", "reverse-seed-report.json", "framing-report.json")
        if not all((output / name).is_file() for name in names):
            raise RuntimeError("M27 original reports are incomplete")
        document = json.loads((output / "report.json").read_bytes())
        if (
            document.get("status") != "completed"
            or document.get("tracking_policy") != policy.as_dict()
        ):
            raise RuntimeError("M27 original report status or policy differs")
        actual_runs = [document, document["runs"][0]]
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
                raise RuntimeError("M27 original report does not preserve declared metadata")
            if actual.get("candidate_preference") != ["model", "adjacent", "keyframe"]:
                raise RuntimeError(
                    "M27 reported candidate preference differs from active selection"
                )
        framing_report = json.loads((output / "framing-report.json").read_bytes())
        if (
            framing_report.get("status") != "completed"
            or framing_report.get("quality_qualification_allowed") is not False
            or any(
                framing_report.get(key) != report[key]
                for key in (
                    "framing",
                    "capture",
                    "requested_frames",
                    "seed_profile",
                    "capacity_profile",
                    "diagnostic_solver_profile",
                    "configuration",
                    "diagnostic_scene",
                    "skip_complete",
                    "depth_band_override",
                    "preparation_target_points",
                    "tracking_policy",
                )
            )
        ):
            raise RuntimeError("M27 Stage B framing differs from the original capture contract")
        links = framing_report.get("underlying_reports", [])
        expected_links = {str(output / name) for name in names if name != "framing-report.json"}
        if len(links) != 3 or {item.get("path") for item in links} != expected_links:
            raise RuntimeError("M27 Stage B framing report links are incomplete")
        for item in links:
            path = Path(item["path"])
            if path.parent != output or path.name not in names:
                raise RuntimeError("M27 Stage B framing references an unexpected report")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError("M27 Stage B framing report hash differs")
        seed = json.loads((output / "reverse-seed-report.json").read_bytes())
        if (
            seed.get("status") != "completed"
            or seed.get("seed_profile") != seed_profile
            or seed.get("capacity_profile") != capacity_profile
            or not seed.get("fit_records")
        ):
            raise RuntimeError("M27 original seed metadata is absent or differs")
        run_report = document["runs"][0]
        if (
            report["preference_observations"] != [report["candidate_preference"]] * 2
            or len(experiment.preparations) != run_report["offered"]
            or not experiment.preparations
            or not experiment.models
            or not experiment.hooks_restored
            or experiment.binding.accepted_count != run_report["accepted"]
        ):
            raise RuntimeError(
                "M27 preparation, model, admission or restoration evidence incomplete"
            )
        accepted = 0
        for record, decision in zip(experiment.preparations, run_report["decisions"], strict=True):
            index = decision["source_index"]
            admitted = decision["decision"]["accepted"]
            if (
                record["status"] != ("accepted" if admitted else "refused")
                or record["ordinal"] != index
                or record["raw_sha256"] != document["input_frames"][index]["sha256"]
                or record["points"] != decision["prepared_points"]
                or record["accepted_index"] != (accepted if admitted else None)
            ):
                raise RuntimeError(
                    "M27 prepared raw/accepted identities differ from original decisions"
                )
            accepted += int(admitted)
        if any(row["status"] != "completed" for row in experiment.models):
            raise RuntimeError("M27 model records are incomplete")
        queries = [q for row in experiment.models for q in row["queries"]]
        if not queries:
            raise RuntimeError("M27 recorded no model normal-provider queries")
        for query in queries:
            fit_index, stage_index = query["seed_fit_ordinal"], query["seed_stage_ordinal"]
            if type(fit_index) is not int or not 0 <= fit_index < len(seed["fit_records"]):
                raise RuntimeError("M27 normal query has an invalid seed fit identity")
            fit = seed["fit_records"][fit_index]
            if type(stage_index) is not int or not 0 <= stage_index < len(fit["stages"]):
                raise RuntimeError("M27 normal query has an invalid seed stage identity")
            stage = fit["stages"][stage_index]
            if (
                stage["direction"] != query["direction"]
                or stage["status"] not in ("completed", "refused")
                or query["neighbours"] != 10
                or query["usable"] != query["baseline"]["usable"]
                or (
                    (normal_profile == "observe-pooled" or query["direction"] == "reverse")
                    and query["actual_sha256"] != query["baseline"]["sha256"]
                )
            ):
                raise RuntimeError(
                    "M27 normal field cannot be reconciled with its original seed stage"
                )
        report["hooks_restored"] = experiment.hooks_restored
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["hooks_restored"] = experiment.hooks_restored
        if output.is_dir():
            for name in (
                "report.json", "stride-1.json", "reverse-seed-report.json", "framing-report.json"
            ):
                path = output / name
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    report["underlying_reports"].append(dict(path=str(path), sha256=digest))
            with (output / "organized-model-stage-b-report.json").open("x") as destination:
                json.dump(report, destination, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-profile", choices=PROFILES, required=True)
    parser.add_argument("--radius", type=int, choices=(5, 10), required=True)
    parser.add_argument("--maximum-weight", type=float, choices=(0.5, 1.0), required=True)
    parser.add_argument("--reverse-seed", choices=SEED_PROFILES, required=True)
    parser.add_argument("--capacity", choices=CAPACITIES, required=True)
    parser.add_argument("--mechanism", choices=("centered",), required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--framing", choices=original.FRAMINGS, required=True)
    args = parser.parse_args()
    run(
        args.capture,
        args.output,
        normal_profile=args.normal_profile,
        radius=args.radius,
        maximum_weight=args.maximum_weight,
        seed_profile=args.reverse_seed,
        capacity_profile=args.capacity,
        solver_profile=args.mechanism,
        framing=args.framing,
        frames=args.frames,
    )


if __name__ == "__main__":
    main()
