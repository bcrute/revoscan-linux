"""M6: closed reciprocal correspondences, with original or centered finite updates."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
from scanner_solver_mechanisms import _ACTIVE, MAX_RECORDED_STEPS, centered_step
from scipy.spatial import cKDTree

from scanner3d import icp_plane
from scanner3d.geometry import Point, Rigid

PROFILES = ("reciprocal-origin", "reciprocal-centered")
Pair = tuple[float, Point, Point, Point]


def reciprocal_pairs(
    source: Sequence[Point], current: Rigid, pairs: Sequence[Pair]
) -> tuple[list[Pair], int]:
    """Keep a forward pair only if the target's closest source is that source.

    Original source-order ties are resolved explicitly, with no distance/gate
    tolerance added. The reverse tree includes ALL source points, not just
    forward matches. Forward tuple/order/distance/normal are returned unchanged.
    """
    if not 2 <= len(source) <= icp_plane.NORMAL_CACHE_MAX_POINTS:
        raise ValueError("Reciprocal source budget is 2..2048 points")
    points = tuple(source)
    positions = {p: i for i, p in enumerate(points)}
    if len(positions) != len(points) or not np.isfinite(points).all():
        raise ValueError("Unique finite source coordinates required")
    if not pairs:
        return [], 0
    if len(pairs) > len(points) or len({pair[1] for pair in pairs}) != len(pairs):
        raise ValueError("At most one original forward pair per source point required")
    if any(p not in positions for _, p, _, _ in pairs):
        raise ValueError("Forward pair source is absent from actual source cloud")
    moved = np.asarray(list(current.apply_all(points)), dtype=float)
    targets = np.asarray([q for _, _, q, _ in pairs], dtype=float)
    if targets.shape != (len(pairs), 3) or not np.isfinite(targets).all():
        raise ValueError("Finite three-component forward targets required")
    tree = cKDTree(moved)
    distances, indices = tree.query(targets, k=2, workers=1)
    selected = indices[:, 0].copy()
    for row in np.flatnonzero(distances[:, 0] == distances[:, 1]):
        possible = tree.query_ball_point(targets[row], np.nextafter(distances[row, 0], np.inf))
        # nextafter only enumerates potential ties; exact squared distance,
        # then original index, makes the final choice without a loose cutoff.
        selected[row] = min(
            possible, key=lambda i: (float(np.sum((moved[i] - targets[row]) ** 2)), i)
        )
    kept = [
        pair for pair, index in zip(pairs, selected, strict=True) if positions[pair[1]] == index
    ]
    return kept, len(pairs) - len(kept)


class ReciprocalExperiment:
    """Only reverse consistency changes pairs; external fit/gates stay original."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("Closed reciprocal profile required")
        self.profile = profile
        self.records: list[dict[str, Any]] = []
        self.source_index: int | None = None
        self._iteration: Callable[..., icp_plane._FitPlane] | None = None
        self._step: Callable[..., Rigid] | None = None
        self._inside_fit = False

    def mark_frame(self, index: int) -> None:
        if not 0 <= index < 3000:
            raise ValueError("Source indices must be in 0..2999")
        self.source_index = index

    def _record(self, **fields: Any) -> dict[str, Any]:
        if len(self.records) >= MAX_RECORDED_STEPS:
            raise RuntimeError("Reciprocal experiment exceeded its record bound")
        row = dict(profile=self.profile, source_index=self.source_index, **fields)
        self.records.append(row)
        return row

    def step(
        self,
        matched: Sequence[tuple[Point, Point, Point]],
        current: Rigid,
        *,
        accelerated: bool = False,
    ) -> Rigid:
        if self._step is None or not self._inside_fit:
            raise RuntimeError("Reciprocal step outside its fit context")
        row = self._record(kind="step", status="running", matched=len(matched))
        try:
            if self.profile == "reciprocal-centered":
                result, details = centered_step(matched, current)
                row.update(details)
            else:
                result = self._step(matched, current, accelerated=accelerated)
            row["status"] = "completed"
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    def iterate(
        self, source: Sequence[Point], correspond: icp_plane._Correspond, **kwargs: Any
    ) -> icp_plane._FitPlane:
        if self._iteration is None or self._inside_fit:
            raise RuntimeError("Reciprocal iteration requires one active fit context")
        row = self._record(kind="fit", status="running", source_points=len(source), calls=[])
        self._inside_fit = True

        def correspond_mutual(src: Sequence[Point], transform: Rigid) -> tuple[list[Pair], int]:
            pairs, rejected = correspond(src, transform)
            eligible, removed = reciprocal_pairs(src, transform, pairs)
            row["calls"].append(
                dict(
                    forward=len(pairs),
                    reciprocal=len(eligible),
                    original_rejected=rejected,
                    reciprocal_rejected=removed,
                )
            )
            return eligible, rejected + removed

        try:
            result = self._iteration(source, correspond_mutual, **kwargs)
            row.update(
                status="completed",
                iterations=result.iterations,
                externally_reported_target_plane_rmse=result.rmse,
            )
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self._inside_fit = False

    @contextmanager
    def active(self) -> Iterator[None]:
        if _ACTIVE.get() or self._iteration is not None:
            raise RuntimeError("Solver mechanisms cannot be nested")
        token = _ACTIVE.set(True)
        self._iteration, self._step = icp_plane._iterate, icp_plane._solve_point_to_plane
        try:
            with (
                patch.object(icp_plane, "_iterate", self.iterate),
                patch.object(icp_plane, "_solve_point_to_plane", self.step),
            ):
                yield
        finally:
            self._iteration = self._step = None
            self._inside_fit = False
            _ACTIVE.reset(token)
