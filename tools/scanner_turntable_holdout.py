"""Offline fixed-axis fit on the first ~1.5 turns, tested on later unseen turns.

Not a handheld fix or a live processing mode. Does not fit held-out frames,
change the input data or build a mesh. The independent comparison uses actual
image-matched frame pairs from the prior full incremental replay audit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Any

from scanner_revisit_audit import audit

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.gpu_normals import live_normals
from scanner3d.preparation import DepthBand, prepare_frame
from scanner3d.turntable import TurntableModel, register_turntable


def run(replay_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Refusing overwrite")
    replay = json.loads(replay_path.read_text())
    capture = Path(replay["capture"])
    metadata = json.loads((capture / "scan.scanner3d").read_text())
    times = metadata["capture"]["frame_times_s"]
    if len(times) != 2703 or any(b <= a for a, b in itertools.pairwise(times)):
        raise ValueError("Expected the exact 2703-frame, timestamped capture")
    indices = list(range(0, 400, 2))
    raws = [Path(replay["input_frames"][i]["path"]).read_bytes() for i in indices]
    band = DepthBand(**replay["depth_band"])
    clouds = [prepare_frame(raw, target_points=2200, depth_band=band).points for raw in raws]
    sampled_times = [times[i] for i in indices]
    started = time.monotonic()
    messages: list[str] = []
    report: dict[str, Any] = dict(
        scope=__doc__,
        source=str(replay_path),
        training_indices=indices,
        heldout_starts_at_source_index=400,
        status="running",
    )

    def progress(message: str) -> None:
        messages.append(message)
        print(message, flush=True)
        if time.monotonic() - started > 360:
            raise TimeoutError("360-second offline fitting bound")

    try:
        with live_normals(None, progress):
            fitted = register_turntable(
                clouds,
                raws=raws,
                depth_scale=DEPTH_SCALE_METRES,
                times=sampled_times,
                backend="scipy",
                on_progress=progress,
            )
        spacing = statistics.median(b - a for a, b in itertools.pairwise(sampled_times))
        model = TurntableModel(
            fitted.axis,
            fitted.centre,
            fitted.rate_degrees,
            tuple((t - times[0]) / spacing for t in times),
        )
        if not fitted.times_used:
            raise RuntimeError("Time-based model did not activate; refusing invented extrapolation")
        for j, original in enumerate(indices):
            actual, expected = model.pose(original), fitted.poses[j]
            if actual != expected:
                raise RuntimeError("Dense time extension changed a fitted training pose")
        extended = replay["runs"][0]
        extended["poses"] = [
            dict(rotation=model.pose(i).rotation, translation=model.pose(i).translation)
            for i in range(len(times))
        ]
        report["fit"] = fitted.as_dict()
        report["angular_speed_deg_s"] = fitted.rate_degrees / spacing
        report["audit"] = audit({**replay, "runs": [extended]})
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["messages"] = messages
        report["elapsed_s"] = time.monotonic() - started
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)
    print("HELDOUT", json.dumps(report["audit"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.replay, args.output)
