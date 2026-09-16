"""Raw-selected common revisit checks; no pose fitting or scan qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scanner_candidate_audit import pose, projection
from scanner_motion_replay import load_inputs

from scanner3d import coherence
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.geometry import Rigid
from scanner3d.preparation import DepthBand


def separated_ordinals(rows: list[dict[str, Any]]) -> list[int]:
    """Greedy raw-only selection; four-frame exclusion, not statistical independence."""
    used: list[int] = []
    selected = []
    for ordinal, row in enumerate(rows):
        if not row.get("credible"):
            continue
        endpoints = [row["frame"], row["best_frame"]]
        if any(abs(endpoint - previous) <= 4 for endpoint in endpoints for previous in used):
            continue
        selected.append(ordinal)
        used.extend(endpoints)
    return selected


def raw_schedule(raws: list[bytes]) -> dict[str, Any]:
    """Instrument the original matcher to retain records even when it returns None."""
    records = []
    original = coherence.PeriodSample

    def observed(*args: Any, **kwargs: Any) -> coherence.PeriodSample:
        record = original(*args, **kwargs)
        records.append(record.as_dict())
        return record

    # Single-thread offline scope. Returned objects remain the original class;
    # no scoring, tie, credibility, lag or sample-position arithmetic changes.
    with patch.object(coherence, "PeriodSample", observed):
        estimate = coherence.image_period(raws, depth_scale=DEPTH_SCALE_METRES, samples=40)
    by_frame = {row["frame"]: row for row in records}
    rows = [
        by_frame.get(
            frame,
            dict(frame=frame, credible=False, reason="no_comparable_later_frame"),
        )
        for frame in coherence._spread(40, len(raws) // 3)
    ]
    return dict(
        image_samples=40,
        period_estimate=None if estimate is None else estimate.as_dict(),
        anchors=rows,
        endpoint_separated_ordinals=separated_ordinals(rows),
        endpoint_exclusion_frames=4,
        independence_limitation="Temporal endpoint separation is not statistical independence",
    )


def evaluate(
    schedule: dict[str, Any],
    raws: list[bytes],
    poses: dict[int, Rigid],
    band: DepthBand,
    *,
    tracking_stopped: bool,
) -> dict[str, Any]:
    identity = Rigid.identity()
    rows = []
    for ordinal, sample in enumerate(schedule["anchors"]):
        row = dict(ordinal=ordinal, image_match=sample, checked=False)
        if not sample.get("credible"):
            row["reason"] = "not_a_credible_raw_revisit"
        else:
            early, late = sample["frame"], sample["best_frame"]
            pair = [raws[early], raws[late]]
            row["raw_identity"] = projection(pair, [identity, identity], band)
            if early not in poses or late not in poses:
                row["reason"] = "endpoint_not_accepted_or_not_offered"
            else:
                row["placed"] = projection(pair, [poses[early], poses[late]], band)
                row["checked"] = row["placed"]["agreement"] is not None
                if not row["checked"]:
                    row["reason"] = "insufficient_projectable_depth"
        rows.append(row)
    checked = [row for row in rows if row["checked"]]
    separated = [
        row for row in checked if row["ordinal"] in schedule["endpoint_separated_ordinals"]
    ]
    omitted = [
        row["ordinal"] for row in rows if row["image_match"].get("credible") and not row["checked"]
    ]
    weakest = min((row["placed"]["agreement"] for row in checked), default=None)
    reasons = []
    if tracking_stopped:
        reasons.append("tracking_stopped")
    if weakest is not None and weakest < 0.9:
        reasons.append("credible_pair_below_90_percent")
    if schedule["period_estimate"] is None:
        reasons.append("no_credible_image_period")
    if omitted:
        reasons.append("credible_pairs_unplaced_or_uncheckable")
    if len(separated) < 25:
        reasons.append("fewer_than_25_endpoint_separated_checkable_pairs")
    return dict(
        quality_qualification_allowed=False,
        diagnostic_verdict=(
            "rejected"
            if tracking_stopped or (weakest is not None and weakest < 0.9)
            else "inconclusive"
            if reasons
            else "observed_checks_meet_margin"
        ),
        reasons=reasons,
        checked_pairs=len(checked),
        endpoint_separated_checked_pairs=len(separated),
        uncheckable_or_unplaced_ordinals=omitted,
        weakest_agreement=weakest,
        checks=rows,
    )


def run(capture: Path, output: Path, replays: list[Path], *, frames: int = 628) -> dict[str, Any]:
    if not 1 <= len(replays) <= 3 or len(set(replays)) != len(replays):
        raise ValueError("One to three distinct replay reports required")
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        status="running", scope=__doc__, quality_qualification_allowed=False, runs=[]
    )
    try:
        _, raws, band, inputs = load_inputs(capture, frames, 12, (60, 300), diagnostic_scene=True)
        schedule = raw_schedule(raws)
        report.update(capture=str(capture), input_frames=inputs, schedule=schedule)
        for path in replays:
            payload = path.read_bytes()
            replay = json.loads(payload)
            if replay.get("status") != "completed" or len(replay.get("runs", [])) != 1:
                raise ValueError("A completed single-run replay is required, including early stops")
            if replay["input_frames"] != inputs or replay["depth_band"] != band.as_dict():
                raise ValueError("Replay inputs/band differ from the frozen raw schedule")
            source = replay["runs"][0]
            indices = source["accepted_source_indices"]
            if source["stride"] != 1 or len(indices) != len(set(indices)):
                raise ValueError("Stride-one unique accepted source indices required")
            positions = dict(zip(indices, (pose(p) for p in source["poses"]), strict=True))
            checked = evaluate(
                schedule, raws, positions, band, tracking_stopped=source["tracking_stopped"]
            )
            checked.update(
                source=str(path),
                sha256=hashlib.sha256(payload).hexdigest(),
                profile=replay["diagnostic_reference_profile"],
                offered=source["offered"],
                accepted=source["accepted"],
                original_coherence=source["coherence"],
            )
            report["runs"].append(checked)
            print(
                checked["profile"],
                checked["diagnostic_verdict"],
                checked["weakest_agreement"],
                flush=True,
            )
        report["status"] = "completed"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--frames", type=int, default=628)
    args = parser.parse_args()
    run(args.capture, args.output, args.replay, frames=args.frames)
