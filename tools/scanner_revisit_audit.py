"""Check individually image-matched revisits, without assuming a constant frame lag.

The legacy period gate can choose multiples of a revolution and constant-lag
pairs can drift in phase as GUI frame losses vary. This supplemental audit
uses the image matcher's actual best pairs, not the median of their lags.
It neither changes registration nor hides the original coherence result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.coherence import image_period, period_agreement
from scanner3d.geometry import Matrix3, Point, Rigid
from scanner3d.preparation import DepthBand
from scanner3d.simulate import FACTORY_INTRINSICS


def _verified_raw(capture: Path, row: dict[str, Any]) -> bytes:
    path = Path(row["path"])
    if path.parent != capture or path.is_symlink():
        raise ValueError("Unexpected raw input location")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Recorded raw input hash changed")
    return raw


def audit(replay: dict[str, Any], *, dense: bool = False) -> dict[str, Any]:
    qualification: dict[str, Any] = {
        key: replay[key]
        for key in (
            "diagnostic_scene",
            "start_offset",
            "normals_backend",
            "diagnostic_reference_profile",
            "tracking_policy",
            "keyframe_eviction_rule",
            "candidate_preference",
            "alignment_method",
            "alignment_keep_fraction",
            "quality_qualification_allowed",
            "source_scene_validity",
        )
        if key in replay
    }
    selection = dict(
        pair_selection="fresh_25_position_image_match" if dense else "recorded_image_match",
        image_match_samples_override=25 if dense else None,
    )
    report: dict[str, Any] = dict(scope=__doc__, runs=[], **selection, **qualification)
    capture = Path(replay["capture"])
    band = DepthBand(**replay["depth_band"])
    inputs = replay["input_frames"]
    for run in replay["runs"]:
        pairs = []
        estimate = run["coherence"]["period_estimate"]
        if dense:
            indices = run["accepted_source_indices"]
            if len(indices) > 3000 or len(set(indices)) != len(indices):
                raise ValueError("Dense audit requires at most 3000 unique accepted frames")
            if any(not isinstance(i, int) or not 0 <= i < len(inputs) for i in indices):
                raise ValueError("Invalid accepted source index")
            # Pair selection itself is evidence: verify every participating raw,
            # not only the two that happen to win the search. Never change poses.
            raws = [_verified_raw(capture, inputs[i]) for i in indices]
            estimated = image_period(raws, depth_scale=DEPTH_SCALE_METRES, samples=25)
            estimate = None if estimated is None else estimated.as_dict()
        if estimate is not None:
            for sample in estimate["samples"]:
                if not sample["credible"]:
                    continue
                indices = [sample["frame"], sample["best_frame"]]
                source = [run["accepted_source_indices"][i] for i in indices]
                poses = [
                    Rigid.of(
                        cast(Matrix3, tuple(tuple(row) for row in run["poses"][i]["rotation"])),
                        cast(Point, tuple(run["poses"][i]["translation"])),
                    )
                    for i in indices
                ]
                raws = [_verified_raw(capture, inputs[i]) for i in source]
                checked = period_agreement(
                    raws,
                    poses,
                    intrinsics=FACTORY_INTRINSICS,
                    depth_scale=DEPTH_SCALE_METRES,
                    depth_band=band,
                    period=1,
                    samples=1,
                ).as_dict()
                pairs.append(
                    dict(
                        source_indices=source,
                        image_match_mm=sample["median_depth_difference"],
                        delivery_interval_s=(
                            inputs[source[1]]["monotonic_s"] - inputs[source[0]]["monotonic_s"]
                            if all("monotonic_s" in inputs[i] for i in source)
                            else None
                        ),
                        projection=checked,
                    )
                )
        measured = [
            p["projection"]["agreement"] for p in pairs if p["projection"]["agreement"] is not None
        ]
        weakest = min(measured) if measured else None
        uncheckable = len(pairs) - len(measured)
        source_selection = run["coherence"].get("pair_selection", "fixed_lag")
        verdict = (
            False
            if weakest is not None and weakest < 0.75
            else (None if weakest is None or uncheckable else True)
        )
        report["runs"].append(
            dict(
                stride=run["stride"],
                offered=run["offered"],
                accepted=run["accepted"],
                legacy_fixed_lag_coherent=(
                    run["coherence"]["coherent"] if source_selection == "fixed_lag" else None
                ),
                source_coherence_coherent=run["coherence"]["coherent"],
                source_coherence_pair_selection=source_selection,
                audit_period_estimate=estimate,
                weakest_individually_matched_agreement=weakest,
                individually_matched_coherent=verdict,
                uncheckable_pair_count=uncheckable,
                pairs=pairs,
                **selection,
                **qualification,
            )
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dense", action="store_true", help="Recompute image matches at 25 sampled positions"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing overwrite")
    report = audit(json.loads(args.replay.read_text()), dense=args.dense)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    for run in report["runs"]:
        print(
            run["stride"],
            run["accepted"],
            run["weakest_individually_matched_agreement"],
            flush=True,
        )
