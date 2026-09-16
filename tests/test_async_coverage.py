"""Display worker queue/lifecycle tests: no device or GUI window is opened."""

import subprocess
import sys
from concurrent.futures import Executor, Future
from typing import Any, cast

import pytest

pytest.importorskip("PySide6")

from scanner3d.geometry import Rigid
from scanner3d.gui import LIVE_MODEL_CLOUDS, ModelSnapshot
from scanner3d.live_model import AsyncCoverage
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    alignments,
    capture_dir,
    clouds,
    live,
    models,
    start_capture,
)

__all__ = ["alignments", "capture_dir", "clouds", "live", "models", "stream_kernel", "stream_node"]


class HeldExecutor(Executor):
    def __init__(self):
        self.jobs = []
        self.shutdowns = []

    def submit(self, fn, /, *args, **kwargs):
        future: Future[Any] = Future()
        self.jobs.append((future, fn, args, kwargs))
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdowns.append((wait, cancel_futures))


def test_only_one_bounded_immutable_request_can_be_in_flight():
    executor = HeldExecutor()
    builder = AsyncCoverage(executor)
    clouds = [[(float(i), 0.0, 100.0)] for i in range(3 * LIVE_MODEL_CLOUDS)]
    poses = [Rigid.identity()] * len(clouds)
    assert builder.request(clouds, poses, saved=120, aligned=120)
    submitted = executor.jobs[0][2][0]
    assert len(submitted) == LIVE_MODEL_CLOUDS
    clouds[0].append((999.0, 0.0, 0.0))
    assert len(submitted[0]) == 1
    assert not builder.request(clouds, poses, saved=121, aligned=121)
    assert builder.poll() is None
    snapshot = cast(ModelSnapshot, object())
    executor.jobs[0][0].set_result((snapshot, 0.1))
    assert builder.poll() == (snapshot, 0.1)
    assert builder.poll() is None
    builder.close()
    assert not builder.request(clouds, poses, saved=121, aligned=121)
    builder.close()
    assert executor.shutdowns == [(False, True)]


def test_invalidated_running_result_can_never_cross_a_boundary():
    executor = HeldExecutor()
    builder = AsyncCoverage(executor)
    clouds, poses = [[(1.0, 2.0, 3.0)]], [Rigid.identity()]
    builder.request(clouds, poses, saved=1, aligned=1)
    future = executor.jobs[0][0]
    future.set_running_or_notify_cancel()
    builder.discard()
    assert not builder.request(clouds, poses, saved=2, aligned=2)
    future.set_result(("obsolete", 0.1))
    assert builder.poll() is None
    assert builder.request(clouds, poses, saved=2, aligned=2)
    builder.close()


def test_failed_job_is_surfaced_once_and_releases_its_slot():
    executor = HeldExecutor()
    builder = AsyncCoverage(executor)
    builder.request([[(1.0, 2.0, 3.0)]], [Rigid.identity()], saved=1, aligned=1)
    executor.jobs[0][0].set_exception(RuntimeError("test failure"))
    with pytest.raises(RuntimeError, match="test failure"):
        builder.poll()
    assert builder.poll() is None
    builder.close()


@pytest.mark.parametrize("ending", ["normal", "stop", "loss", "error"])
def test_capture_publishes_only_current_results_and_closes_helper(
    ending, stream_kernel, capture_dir, live, monkeypatch
):
    from scanner3d import live_model

    executor = HeldExecutor()
    builder = AsyncCoverage(executor)
    monkeypatch.setattr(live_model, "AsyncCoverage", lambda: builder)
    stream_kernel(tuple(whole_frame(i + 1, buffer_index=i % 4) for i in range(16)))
    window = CaptureWindow(frames=16)
    window._async_coverage = True
    live(failed_alignments={9, 10, 11} if ending == "loss" else set())

    def complete_pending(saved):
        if saved == (12 if ending == "loss" else 9):
            future, function, args, kwargs = executor.jobs[0]
            if ending == "error":
                future.set_exception(RuntimeError("synthetic helper failure"))
            else:
                future.set_result(function(*args, **kwargs))
            if ending == "stop":
                window.worker.cancel()

    window.worker.saved.connect(complete_pending)
    start_capture(window)
    published = [
        cast(ModelSnapshot, snapshot).aligned for (snapshot,) in window.worker.coverage.emitted
    ]
    expected = {"normal": [3, 7, 16], "stop": [3], "loss": [3], "error": [3, 9, 13]}
    assert published == expected[ending]
    assert builder.closed and executor.shutdowns == [(False, True)]


def test_spawned_snapshot_matches_reference_and_pending_close_exits():
    # Isolate Python's process-exit join too: forcibly terminating a pool
    # writer previously left the GUI process alive after its window closed.
    program = """
from scanner3d.geometry import Rigid
from scanner3d.gui import _live_snapshot
from scanner3d.live_model import AsyncCoverage
from dataclasses import fields
import numpy as np
clouds = [[(float(i), float(i % 13), 100.0) for i in range(900)]] * 3
poses = [Rigid.identity()] * 3
builder = AsyncCoverage()
builder.request(clouds, poses, saved=3, aligned=3)
snapshot, elapsed = builder.pending.result(timeout=15)
reference = _live_snapshot(clouds, poses, saved=3, aligned=3)
for field in fields(snapshot):
    actual, expected = getattr(snapshot, field.name), getattr(reference, field.name)
    if isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected
builder.poll()
builder.request(clouds, poses, saved=4, aligned=4)
builder.close()
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=25)
