"""Spatial fit disagreement and complete raw images; no new accepted scan pose."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scanner_candidate_audit import pose, projection
from scanner_registration_audit import traced_fit
from scanner_revisit_audit import _verified_raw
from scanner_scene_audit import plot
from stationary_spatial_report import save_plot

from scanner3d.geometry import Point, Rigid
from scanner3d.pipeline import frame_travel
from scanner3d.preparation import DepthBand, prepare_frame


def regions(points: list[Point]) -> dict[str, list[Point]]:
    array = np.asarray(points)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not np.isfinite(array).all()
        or np.any(array[:, 2] <= 0)
    ):
        raise ValueError("Finite positive-depth source cloud required")
    uv = array[:, :2] / array[:, 2, None]
    lower, upper = uv.min(axis=0), uv.max(axis=0)
    masks = dict(
        left=uv[:, 0] <= lower[0] + (upper[0] - lower[0]) * 2 / 3,
        right=uv[:, 0] >= lower[0] + (upper[0] - lower[0]) / 3,
        top=uv[:, 1] <= lower[1] + (upper[1] - lower[1]) * 2 / 3,
        bottom=uv[:, 1] >= lower[1] + (upper[1] - lower[1]) / 3,
    )
    return {
        name: [p for p, keep in zip(points, mask, strict=True) if keep]
        for name, mask in masks.items()
    }


def run(path: Path, output: Path, *, control: bool = False) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    result: dict[str, Any] = dict(status="running", quality_qualification_allowed=False, pairs=[])
    try:
        content = path.read_bytes()
        audit = json.loads(content)
        source = json.loads(Path(audit["source"]).read_bytes())
        band = DepthBand(**source["depth_band"])
        result.update(
            source_audit=str(path),
            source_audit_sha256=hashlib.sha256(content).hexdigest(),
            source=audit["source"],
            control=control,
        )
        saved = set()
        for row in audit["pairs"]:
            if control and row["ordinal"] % 4:
                continue
            raws = [_verified_raw(Path(source["capture"]), r) for r in row["inputs"]]
            clouds = [prepare_frame(raw, target_points=900, depth_band=band).points for raw in raws]
            for i, raw in zip(row["source_indices"], raws, strict=True):
                if i not in saved:
                    save_plot(
                        output / f"frame-{i:04d}.png",
                        plot(np.frombuffer(raw, dtype="<u2").reshape(400, 640)),
                    )
                    saved.add(i)
            record: dict[str, Any] = dict(
                source_indices=row["source_indices"],
                inputs=row["inputs"],
                prepared_points=[len(c) for c in clouds],
                fits=[],
            )
            result["pairs"].append(record)
            complete = row["directions"][0]
            if "result" not in complete:
                record["reason"] = "full_fit_failed"
                continue
            full = pose(complete["result"]["transform"])
            record["full"] = complete
            for name, subset in regions(list(clouds[1])).items():
                fitted = traced_fit(subset, clouds[0])
                evidence = dict(region=name, source_points=len(subset), trace=fitted)
                if "result" in fitted:
                    transform = pose(fitted["result"]["transform"])
                    evidence.update(
                        full_source_disagreement=frame_travel(
                            clouds[1], full.inverse().compose(transform)
                        ),
                        raw_projection=projection(raws, [Rigid.identity(), transform], band),
                    )
                record["fits"].append(evidence)
            print(
                "REGIONS",
                record["source_indices"],
                [r.get("full_source_disagreement") for r in record["fits"]],
                flush=True,
            )
        result["status"] = "completed"
        return result
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    run(args.audit, args.output, control=args.control)
