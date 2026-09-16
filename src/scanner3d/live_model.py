"""One bounded display-only coverage job, independent of scanner/tracker state."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from typing import TYPE_CHECKING

from scanner3d.geometry import Point, Rigid

if TYPE_CHECKING:
    from scanner3d.gui import ModelSnapshot


def _build(
    clouds: tuple[tuple[Point, ...], ...], poses: tuple[Rigid, ...], saved: int, aligned: int
) -> tuple[ModelSnapshot, float]:
    # Child imports definitions only. No window, device, GPU context or scanner
    # command is created. This is the same picture builder as the synchronous path.
    from scanner3d.gui import _live_snapshot

    started = time.monotonic()
    snapshot = _live_snapshot(clouds, poses, saved=saved, aligned=aligned)
    return snapshot, time.monotonic() - started


class AsyncCoverage:
    """At most one in-flight immutable request; never blocks to collect a result.

    A discarded request retains its slot until finished, so invalidation cannot
    quietly build an unbounded queue. Pause/recovery boundaries may use the
    synchronous builder while that obsolete display job finishes.
    """

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor or ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        )
        self.pending: Future[tuple[ModelSnapshot, float]] | None = None
        self.stale = False
        self.closed = False

    def request(
        self, clouds: Sequence[Sequence[Point]], poses: Sequence[Rigid], *, saved: int, aligned: int
    ) -> bool:
        if self.closed or self.pending is not None:
            return False
        from scanner3d.gui import LIVE_MODEL_CLOUDS, _sample_span

        sampled_clouds, sampled_poses = _sample_span(clouds, poses, LIVE_MODEL_CLOUDS)
        self.pending = self.executor.submit(
            _build, tuple(tuple(c) for c in sampled_clouds), tuple(sampled_poses), saved, aligned
        )
        self.stale = False
        return True

    def poll(self) -> tuple[ModelSnapshot, float] | None:
        if self.pending is None or not self.pending.done():
            return None
        future, self.pending = self.pending, None
        if self.stale or self.closed:
            return None
        return future.result()

    def discard(self) -> None:
        self.stale = True
        if self.pending is not None:
            self.pending.cancel()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.discard()
        # Do not terminate a process while it may be writing a snapshot into
        # the result pipe: an incomplete message can strand the pool's reader
        # and Python's exit join. Stop never waits; this single bounded, pure
        # display task finishes in the background and its result is discarded.
        self.executor.shutdown(wait=False, cancel_futures=True)
