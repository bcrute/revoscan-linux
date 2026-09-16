"""Bounded offline production-path ablation. Never opens a scanner or builds a mesh.

Randomized CPU/GPU normals x sync/async coverage x packed preview on/off,
three repetitions of identical saved input. It measures computation, NOT Qt
widget presentation or acquisition latency. Inclusive nested stage times must
not be added together. Async final drain is reported separately from frame work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import random
import resource
import time
from contextlib import ExitStack, nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from scanner3d import compute, gpu_normals, gui, icp_plane, tracking
from scanner3d.live_model import AsyncCoverage
from scanner3d.preparation import prepare_frame
from scanner3d.viewer import render_packed_frame


def digest_snapshot(snapshot: gui.ModelSnapshot) -> str:
    digest = hashlib.sha256()
    for field in fields(snapshot):
        value = getattr(snapshot, field.name)
        digest.update(field.name.encode())
        if isinstance(value, np.ndarray):
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


class Measurements:
    def __init__(self) -> None:
        self.samples: dict[str, list[tuple[float, float, float]]] = {}

    def wrap(self, name: str, function: Any) -> Any:
        def measured(*args: Any, **kwargs: Any) -> Any:
            wall, process, thread = time.perf_counter(), time.process_time(), time.thread_time()
            try:
                return function(*args, **kwargs)
            finally:
                self.samples.setdefault(name, []).append(
                    (
                        time.perf_counter() - wall,
                        time.process_time() - process,
                        time.thread_time() - thread,
                    )
                )

        return measured

    def report(self) -> dict[str, Any]:
        return {
            name: dict(
                calls=len(samples),
                seconds_wall_process_thread=np.sum(samples, axis=0).tolist(),
                wall_ms_p50_p95_p99=np.percentile(
                    np.array(samples)[:, 0] * 1000, [50, 95, 99]
                ).tolist(),
            )
            for name, samples in self.samples.items()
        }


def one(raws: list[bytes], mode: tuple[bool, str, bool]) -> dict[str, Any]:
    gpu, coverage_mode, preview = mode
    gc.collect()
    icp_plane._cached_normals.cache_clear()
    tracking.IncrementalTracker._pose_gap.cache_clear()
    measurements = Measurements()
    tracker = tracking.IncrementalTracker(backend="scipy")
    clouds, poses, decisions, snapshots = [], [], [], {}
    messages: list[str] = []
    report: dict[str, Any] = dict(
        gpu=gpu, coverage=coverage_mode, preview=preview, loadavg_start=os.getloadavg()
    )
    asynchronous = AsyncCoverage() if coverage_mode == "async" else None
    gc_start, gc_seconds = 0.0, 0.0

    def garbage(phase: str, info: dict[str, Any]) -> None:
        nonlocal gc_start, gc_seconds
        if phase == "start":
            gc_start = time.perf_counter()
        elif phase == "stop":
            gc_seconds += time.perf_counter() - gc_start

    def received(result: Any) -> None:
        if result is not None:
            snapshot, _ = result
            snapshots[str(snapshot.saved)] = digest_snapshot(snapshot)

    setup = time.perf_counter()
    try:
        with (
            gpu_normals.live_normals(None, messages.append) if gpu else nullcontext(),
            ExitStack() as stack,
        ):
            report["setup_seconds"] = time.perf_counter() - setup
            if gpu and not any(m.startswith("GPU-assisted tracking:") for m in messages):
                raise RuntimeError(f"Requested GPU did not activate: {messages}")
            targets: list[tuple[Any, str]] = [
                (compute._ScipyBackend, "nearest_neighbors"),
                (compute._ScipyBackend, "k_nearest_neighbors"),
                (icp_plane, "_solve_point_to_plane"),
                (icp_plane, "estimate_normals"),
                (gpu_normals.GpuNormals, "estimate_normals"),
                (gpu_normals.GpuNormals, "jacobi"),
                (tracking, "overlap_fraction"),
            ]
            for owner, name in targets:
                label = f"{owner.__name__}.{name}"
                stack.enter_context(
                    patch.object(owner, name, measurements.wrap(label, getattr(owner, name)))
                )
            prepare = measurements.wrap("preparation", prepare_frame)
            offer = measurements.wrap("tracking_total", tracker.offer)
            picture = measurements.wrap("packed_preview", render_packed_frame)
            model = measurements.wrap("sync_coverage", gui._live_snapshot)
            gc.callbacks.append(garbage)
            started, process, thread = time.perf_counter(), time.process_time(), time.thread_time()
            for index, raw in enumerate(raws, 1):
                if time.perf_counter() - started > 120:
                    raise TimeoutError("Single replay exceeded 120-second bound")
                cloud = prepare(raw, target_points=900)
                decision = offer(cloud.points)
                decisions.append(decision.as_dict())
                if decision.accepted:
                    clouds.append(cloud.points)
                    poses.append(tracker.poses[-1])
                if preview:
                    picture(raw)
                if asynchronous is not None:
                    received(asynchronous.poll())
                if len(clouds) >= 3 and (index == 3 or index % 4 == 0):
                    if asynchronous is None:
                        snapshot = model(clouds, poses, saved=index, aligned=len(clouds))
                        snapshots[str(index)] = digest_snapshot(snapshot)
                    else:
                        measurements.wrap("async_request", asynchronous.request)(
                            clouds, poses, saved=index, aligned=len(clouds)
                        )
            report["frame_loop_seconds_wall_process_thread"] = [
                time.perf_counter() - started,
                time.process_time() - process,
                time.thread_time() - thread,
            ]
            drain = time.perf_counter()
            if asynchronous is not None and asynchronous.pending is not None:
                asynchronous.pending.result(timeout=30)
                received(asynchronous.poll())
            report["async_final_drain_seconds"] = time.perf_counter() - drain
    finally:
        if garbage in gc.callbacks:
            gc.callbacks.remove(garbage)
        if asynchronous is not None:
            asynchronous.close()
    report.update(
        decisions=decisions,
        snapshots=snapshots,
        messages=messages,
        stages=measurements.report(),
        gc_wall_seconds=gc_seconds,
        process_lifetime_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        loadavg_end=os.getloadavg(),
    )
    return report


def run(reference_path: Path, output: Path, production_only: bool = False) -> None:
    reference = json.loads(reference_path.read_text())
    capture = Path(reference["capture"])
    paths = sorted(capture.glob("frame_*.raw"))
    if not 30 <= len(paths) <= 200:
        raise ValueError("Expected 30..200 saved reference frames")
    raws = [path.read_bytes() for path in paths]
    if any(len(raw) != 512000 for raw in raws):
        raise ValueError("Incomplete source frame")
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        scope=__doc__,
        reference=str(reference_path),
        capture=str(capture),
        input_sha256=[hashlib.sha256(raw).hexdigest() for raw in raws],
        runs=[],
        status="running",
    )
    # Warm imports, solver and GL compilation, but each measured replay has new
    # tracker and normal caches. The report excludes this separate 12-frame warmup.
    one(raws[:12], (True, "sync", True))
    rng = random.Random(20260907)
    try:
        for repetition in range(3):
            modes = (
                [(True, "async", True)]
                if production_only
                else list(itertools.product((False, True), ("sync", "async"), (False, True)))
            )
            rng.shuffle(modes)
            for mode in modes:
                result = one(raws, mode)
                result["repetition"] = repetition
                result["exact_reference_decisions"] = (
                    result["decisions"] == reference["runs"][0]["decisions"]
                )
                report["runs"].append(result)
                with (output / f"run-{len(report['runs']):02d}.json").open("x") as stream:
                    json.dump(result, stream, indent=2)
                print(
                    "RUN",
                    len(report["runs"]),
                    mode,
                    round(result["frame_loop_seconds_wall_process_thread"][0], 3),
                    "exact",
                    result["exact_reference_decisions"],
                    flush=True,
                )
                if not result["exact_reference_decisions"]:
                    raise RuntimeError("Ablation changed tracking decisions; stopping")
        canonical: dict[str, str] = {}
        for result in report["runs"]:
            for count, digest in result["snapshots"].items():
                if count in canonical and canonical[count] != digest:
                    raise RuntimeError("Coverage snapshot differed for identical accepted inputs")
                canonical[count] = digest
        report["exact_coverage_snapshots"] = True
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-only", action="store_true")
    args = parser.parse_args()
    run(args.reference, args.output, args.production_only)
