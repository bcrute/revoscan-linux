"""Offline raw-revisit audit of every saved candidate; no new pose or scan pass.

This is a counterfactual at recorded states, not a closed-loop replay of a new
selection policy. Earlier poses also contain error; identity is a same-view
diagnostic, not independently measured physical truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from scanner_revisit_audit import _verified_raw

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.coherence import period_agreement
from scanner3d.geometry import Matrix3, Point, Rigid
from scanner3d.preparation import DepthBand
from scanner3d.simulate import FACTORY_INTRINSICS


def pose(record: dict[str, Any]) -> Rigid:
    return Rigid.of(
        cast(Matrix3, tuple(tuple(row) for row in record["rotation"])),
        cast(Point, tuple(record["translation"])),
    )


def projection(raws: list[bytes], poses: list[Rigid], band: DepthBand) -> dict[str, Any]:
    return period_agreement(
        raws,
        poses,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        depth_band=band,
        period=1,
        samples=1,
    ).as_dict()


def audit(path: Path) -> dict[str, Any]:
    original = path.read_bytes()
    replay = json.loads(original)
    if replay["status"] != "completed" or len(replay["runs"]) != 1:
        raise ValueError("One completed replay run required")
    run = replay["runs"][0]
    if run["stride"] != 1 or len(replay["input_frames"]) > 3000:
        raise ValueError("At most 3000 inputs and full stride-one decisions required")
    inputs = replay["input_frames"]
    accepted = run["accepted_source_indices"]
    decisions = {row["source_index"]: row for row in run["decisions"]}
    if len(decisions) != len(run["decisions"]):
        raise ValueError("Unique offered source indices required")
    band = DepthBand(**replay["depth_band"])
    result: dict[str, Any] = dict(
        scope=__doc__,
        source=str(path),
        source_sha256=hashlib.sha256(original).hexdigest(),
        quality_qualification_allowed=False,
        pairs=[],
        source_scene_validity=replay.get("source_scene_validity"),
        source_tracking_loss=replay.get("source_tracking_loss"),
        source_pair_selection=run["coherence"].get("pair_selection", "fixed_lag"),
    )
    if result["source_pair_selection"] != "actual_image_match":
        raise ValueError("Audit requires actual-image pairs, never legacy median-lag pairs")
    for pair in run["coherence"]["pairs"]:
        early, late = pair["frame"], pair["revisit"]
        sources = [accepted[i] for i in (early, late)]
        raws = [_verified_raw(Path(replay["capture"]), inputs[i]) for i in sources]
        left = pose(run["poses"][early])
        original_projection = projection(raws, [left, pose(run["poses"][late])], band)
        if original_projection["agreement"] != pair["within_tolerance"]:
            raise ValueError("Saved-pose pair no longer reproduces its recorded agreement")
        decision = decisions[sources[1]]["decision"]
        row: dict[str, Any] = dict(
            accepted_indices=[early, late],
            source_indices=sources,
            raw_inputs=[inputs[i] for i in sources],
            selected_origin=decision["origin"],
            selected=original_projection,
            raw_identity=projection(raws, [Rigid.identity(), Rigid.identity()], band),
            keyframes=decision["keyframes"],
            candidates=[],
        )
        for candidate in decision["candidates"]:
            row["candidates"].append(
                dict(
                    origin=candidate["origin"],
                    accepted=candidate["accepted"],
                    reference_frames=candidate["reference_frames"],
                    alignment=candidate["alignment"],
                    overlap=candidate["overlap"],
                    reference_pose_includes_early_frame=decisions[sources[0]]["decision"]["frame"]
                    in candidate["reference_frames"],
                    projection=projection(raws, [left, pose(candidate["pose"])], band),
                )
            )
        result["pairs"].append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite prior evidence")
    result = audit(args.replay)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    for row in result["pairs"]:
        print(
            row["accepted_indices"],
            row["selected_origin"],
            row["selected"]["agreement"],
            [(c["origin"], c["accepted"], c["projection"]["agreement"]) for c in row["candidates"]],
            flush=True,
        )


if __name__ == "__main__":
    main()
