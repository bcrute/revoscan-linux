"""Offline fixed-pixel 20/14/14/20 ms comparison of completed lifecycle captures.

First 60 settled seconds only, no alignment or best-window selection. Every raw
input is checked, including unused extras. No hardware access or quality pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scanner_motion_framing import measurements as framing_measurements
from scanner_scene_audit import plot
from scanner_validation_baseline import BASELINE, SERIAL
from scipy.ndimage import binary_erosion, label
from stationary_spatial_report import save_plot, summary

from scanner3d.cli import DEPTH_SCALE_METRES

STAGES = (("a1-20000", 20000), ("b1-14000", 14000), ("b2-14000", 14000), ("a2-20000", 20000))


def settled_window(stamps: list[float]) -> list[int]:
    times = np.asarray(stamps, dtype=np.float64)
    if len(times) < 14 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Finite strictly increasing delivery timestamps required")
    start = max(12, int(np.searchsorted(times, times[0] + 1.0)))
    if start >= len(times) or times[-1] < times[start] + 60:
        raise ValueError("Incomplete first 60-second settled window")
    return np.flatnonzero((times >= times[start]) & (times < times[start] + 60)).tolist()


def load_stage(path: Path, timing: int) -> tuple[np.ndarray, dict[str, Any]]:
    validity = path / "quality-validity.json"
    if validity.exists() and json.loads(validity.read_text()).get(
        "quality_qualification_allowed"
    ) is not True:
        raise ValueError("Flagged scene requires a newly confirmed replacement block")
    report_bytes = (path / "report.json").read_bytes()
    report = json.loads(report_bytes)
    progress_bytes = (path / "raw/capture-progress.json").read_bytes()
    progress = json.loads(progress_bytes)
    if (
        report["status"] != "completed"
        or report["restoration_verified"] is not True
        or report.get("first_transport_fault") is not None
        or report["serial"] != SERIAL
        or report["mode"] != "settled-post"
        or report["motion_timing_us"] != timing
        or report["capture_frame_budget"] != 640
        or report["usb_idle"].get("topology_stable") is not True
        or report["usb_before_readback"].get("settled_for_readback") is not True
        or progress["capture_error"] is not None
    ):
        raise ValueError("Capture must complete, settle, restore and preserve device identity")
    steps = {s["name"]: s for s in report["steps"]}
    for prefix, expected in (
        ("initial", 20000),
        ("configured", timing),
        ("after_stop", timing),
        ("verified_post_stop_restore", 20000),
    ):
        state = steps[prefix + "_timing"]
        gain = steps[prefix + "_gain"]
        if (
            state["status"] != "completed"
            or gain["status"] != "completed"
            or state["result"][0] != {**BASELINE, "frame_time": expected, "exposure_time": expected}
            or gain["result"][0] != 24
        ):
            raise ValueError("Timing/gain readback differs from predeclared settings")
    captured = report["capture"]["summary"]
    records = progress["frames"]
    if (
        len(records) != 640
        or captured["kept"] != 640
        or captured["delivered"] != 640
        or captured["short_frames"]
        or captured["dropped_by_sequence"]
        or len({r["path"] for r in records}) != len(records)
        or [r["delivery_index"] for r in records] != list(range(640))
    ):
        raise ValueError("All 640 unique, consecutive, complete raw deliveries required")
    selected = settled_window([r["monotonic_s"] for r in records])
    selected_set = set(selected)
    rows = []
    window = []
    previous = previous_valid = None
    for i, record in enumerate(records):
        raw_path = path / "raw" / record["path"]
        if raw_path.parent != path / "raw" or raw_path.is_symlink():
            raise ValueError("Raw path must be an unlinked local child")
        raw_bytes = raw_path.read_bytes()
        if (
            len(raw_bytes) != 512000
            or record["bytes"] != 512000
            or hashlib.sha256(raw_bytes).hexdigest() != record["sha256"]
        ):
            raise ValueError("Raw length/hash changed")
        raw = np.frombuffer(raw_bytes, dtype="<u2").reshape(400, 640)
        depth = raw * DEPTH_SCALE_METRES
        valid = (depth > 60) & (depth < 300)
        row = dict(
            **record,
            source_path=str(raw_path),
            in_comparison_window=i in selected_set,
            raw_yield=float(np.count_nonzero(raw) / raw.size),
            in_band_yield=float(valid.mean()),
            framing=framing_measurements(raw),
            adjacent_mask_iou=None,
            adjacent_mutual_depth_change_p50_p90=None,
        )
        if previous is not None:
            assert previous_valid is not None
            mutual = valid & previous_valid
            union = valid | previous_valid
            row["adjacent_mask_iou"] = float(mutual.sum() / union.sum()) if union.any() else None
            if mutual.any():
                row["adjacent_mutual_depth_change_p50_p90"] = np.percentile(
                    np.abs(depth[mutual] - previous[mutual]), [50, 90]
                ).tolist()
        rows.append(row)
        if i in selected_set:
            window.append(raw)
        previous, previous_valid = depth, valid
    thermal = report["temperatures"]
    info = dict(
        path=str(path),
        timing_us=timing,
        gain=24,
        auto_exposure=0,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        progress_sha256=hashlib.sha256(progress_bytes).hexdigest(),
        all_raw_hashes_verified=len(records),
        all_unique_payloads=len({r["sha256"] for r in records}),
        capture_summary=captured,
        temperatures=thermal,
        peak_temperatures={k: max(t[k] for t in thermal) for k in ("cpu_c", "npu_c")},
        comparison_source_indices=selected,
        comparison_duration_s=60,
        comparison_observed_fps=(len(selected) - 1)
        / (records[selected[-1]]["monotonic_s"] - records[selected[0]]["monotonic_s"]),
        rows=rows,
    )
    return np.stack(window), info


def region_comparison(
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    a, fa, ma = baseline
    b, fb, mb = candidate
    stable = mask & (fa >= 0.9)
    common = stable & np.isfinite(a) & np.isfinite(b)
    result: dict[str, Any] = dict(
        mask_pixels=int(mask.sum()),
        stable_pixels=int(stable.sum()),
        mutual_stable_pixels=int(common.sum()),
        initial_return_fraction=float(fa[mask].mean()) if mask.any() else None,
        candidate_return_fraction=float(fb[mask].mean()) if mask.any() else None,
        persistently_lost_stable_pixels=int((stable & (fb < 0.1)).sum()),
        persistently_lost_stable_fraction=float((fb[stable] < 0.1).mean())
        if stable.any()
        else None,
        signed_depth_change_p10_p50_p90=None,
        absolute_depth_change_p50_p90_p99=None,
        initial_temporal_mad_p50_p90=None,
        candidate_temporal_mad_p50_p90=None,
        comparison_checkable=bool(common.any()),
    )
    if common.any():
        difference = b[common] - a[common]
        result.update(
            signed_depth_change_p10_p50_p90=np.percentile(difference, [10, 50, 90]).tolist(),
            absolute_depth_change_p50_p90_p99=np.percentile(
                np.abs(difference), [50, 90, 99]
            ).tolist(),
            initial_temporal_mad_p50_p90=np.percentile(ma[common], [50, 90]).tolist(),
            candidate_temporal_mad_p50_p90=np.percentile(mb[common], [50, 90]).tolist(),
        )
    return result


def regions(frequency: np.ndarray) -> dict[str, np.ndarray]:
    components, count = label(frequency >= 0.5)
    if not count:
        raise ValueError("Initial 60-second window has no frequent-return component")
    populations = np.bincount(components.ravel())
    populations[0] = 0
    component = components == populations.argmax()
    core = binary_erosion(component, iterations=3)
    masks = dict(
        full_frame=np.ones_like(component), component=component, core=core, edge=component & ~core
    )
    height, width = frequency.shape
    for y in range(4):
        for x in range(4):
            tile = np.zeros_like(component)
            tile[y * height // 4 : (y + 1) * height // 4, x * width // 4 : (x + 1) * width // 4] = (
                True
            )
            masks[f"tile_{y}_{x}"] = tile
    return masks


def run(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    result: dict[str, Any] = dict(
        scope=__doc__,
        quality_qualification_allowed=False,
        status="running",
        runs=[],
        limitations=[
            "One operator-confirmed stationary view; not handheld or moving-table qualification.",
            "Fixed pixels and nominal depth units; no fitted alignment or absolute accuracy.",
            "A1 alone defines masks. Connected components can include support/background.",
            "A2 minus A1 exposes drift, not proof that temperature caused it.",
        ],
    )
    baseline = masks = None
    previous_last_time = None
    try:
        for name, timing in STAGES:
            raw, info = load_stage(root / name, timing)
            first_time = info["rows"][0]["monotonic_s"]
            if previous_last_time is not None and first_time <= previous_last_time:
                raise ValueError("Stages must follow the declared A-B-B-A time order")
            previous_last_time = info["rows"][-1]["monotonic_s"]
            measured = summary(raw)
            if baseline is None:
                baseline = measured
                masks = regions(measured[1])
            assert masks is not None
            info["regions"] = {
                key: region_comparison(baseline, measured, mask) for key, mask in masks.items()
            }
            info["name"] = name
            selected_rows = [r for r in info["rows"] if r["in_comparison_window"]]
            info["window_raw_yield_p5_p50_p95"] = np.percentile(
                [r["raw_yield"] for r in selected_rows], [5, 50, 95]
            ).tolist()
            save_plot(output / f"{name}-first-depth.png", plot(raw[0]))
            median, frequency, mad = measured
            signed = median - baseline[0]
            amount = np.nan_to_num(np.clip(np.abs(signed), 0, 1)) * 255
            bias = np.full((*median.shape, 3), 255.0)
            bias[..., 0] -= np.where(signed < 0, amount, 0)
            bias[..., 1] -= amount
            bias[..., 2] -= np.where(signed > 0, amount, 0)
            bias[~np.isfinite(signed)] = 0
            save_plot(output / f"{name}-bias.png", bias)
            loss = np.maximum(baseline[1] - frequency, 0)
            save_plot(
                output / f"{name}-loss.png",
                np.stack([255 * loss, 80 * frequency, 80 * frequency], axis=-1),
            )
            with (output / f"{name}-spatial.npz").open("xb") as stream:
                np.savez_compressed(stream, median=median, frequency=frequency, mad=mad)
            result["runs"].append(info)
            del raw
            print(name, json.dumps(info["regions"]["component"]), flush=True)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
