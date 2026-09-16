"""Closed offline M3 source-planarity/directional-balance weighting experiments."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
from scanner_normal_identifiability import spectrum
from scanner_solver_mechanisms import _ACTIVE, MAX_RECORDED_STEPS

from scanner3d import icp_plane
from scanner3d.geometry import IDENTITY_MATRIX, MIN_VECTOR_NORM, Point, Rigid, rotation_about_axis

PROFILES = ("planarity", "normal-balance", "planarity-balance")


def planarity_weights(points: Sequence[Point]) -> list[float]:
    values = np.maximum(np.asarray(spectrum(points)["eigenvalues"]), 0.0)
    largest = values[:, 2]
    usable = largest > 0
    result = np.zeros(len(values))
    result[usable] = (values[usable, 1] - values[usable, 0]) / largest[usable]
    return result.tolist()


def matched_weights(normals: Any, confidence: Any, *, balance: bool) -> np.ndarray:
    n, weights = np.asarray(normals, dtype=float), np.asarray(confidence, dtype=float).copy()
    if n.ndim != 2 or n.shape[1] != 3 or weights.shape != (len(n),):
        raise ValueError("One explicit weight per matched normal required")
    if not np.isfinite(n).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Finite normals and nonnegative weights required")
    if balance:
        try:
            leverage = np.sum(n * np.linalg.solve(n.T @ n, n.T).T, axis=1)
        except np.linalg.LinAlgError as error:
            raise icp_plane.IcpPlaneGeometryUnusable(
                "Singular matched normal directions"
            ) from error
        if not np.isfinite(leverage).all() or np.any(leverage < 0):
            raise icp_plane.IcpPlaneGeometryUnusable("Invalid directional leverage")
        weights *= leverage
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        raise icp_plane.IcpPlaneGeometryUnusable("No finite positively weighted constraints")
    return weights * (len(weights) / total)


def weighted_step(
    matched: Sequence[tuple[Point, Point, Point]],
    current: Rigid,
    confidence: Sequence[float],
    *,
    balance: bool,
) -> tuple[Rigid, dict[str, Any]]:
    values = np.asarray(matched, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) < 6:
        raise ValueError("At least six point/target/normal triples required")
    if not np.isfinite(values).all():
        raise ValueError("Finite triples required")
    source, target, normals = values[:, 0], values[:, 1], values[:, 2]
    weights = matched_weights(normals, confidence, balance=balance)
    moved = source @ np.asarray(current.rotation).T + np.asarray(current.translation)
    centre = moved.mean(axis=0)
    jacobian = np.concatenate((np.cross(moved - centre, normals), normals), axis=1)
    residuals = np.sum((moved - target) * normals, axis=1)
    products = jacobian[:, :, None] * jacobian[:, None, :] * weights[:, None, None]
    matrix = np.add.accumulate(products, axis=0)[-1]
    rhs = np.add.accumulate(-jacobian * (weights * residuals)[:, None], axis=0)[-1]
    solution = icp_plane._solve_6x6(matrix.tolist(), rhs.tolist())
    vector = (solution[0], solution[1], solution[2])
    angle = math.sqrt(sum(v * v for v in vector))
    rotation = (
        IDENTITY_MATRIX if angle < MIN_VECTOR_NORM else rotation_about_axis(vector, angle).rotation
    )
    translation = np.asarray(solution[3:]) + centre - np.asarray(rotation) @ centre
    result = Rigid.of(
        rotation, (float(translation[0]), float(translation[1]), float(translation[2]))
    )
    return result.compose(current), dict(
        matched=len(values),
        zero_weight=int(np.sum(weights == 0)),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
        weight_sum=float(weights.sum()),
        effective_weighted_count=float(weights.sum() ** 2 / (weights @ weights)),
        rotation_step_rad=angle,
        before_unweighted_target_plane_rmse=float(np.sqrt(np.mean(residuals * residuals))),
    )


class WeightedExperiment:
    """Only fit rows are weighted; the existing iteration reports raw residuals."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown closed weighting mechanism {profile!r}")
        self.profile = profile
        self.records: list[dict[str, Any]] = []
        self.source_index: int | None = None
        self._iteration: Callable[..., icp_plane._FitPlane] | None = None
        self._confidence: dict[Point, float] | None = None
        self._cache: OrderedDict[tuple[Point, ...], tuple[float, ...]] = OrderedDict()

    def mark_frame(self, index: int) -> None:
        if not 0 <= index < 3000:
            raise ValueError("Source indices must be in 0..2999")
        self.source_index = index

    def _record(self, **fields: Any) -> dict[str, Any]:
        if len(self.records) >= MAX_RECORDED_STEPS:
            raise RuntimeError("Closed weighting experiment exceeded its record bound")
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
        if self._confidence is None:
            raise RuntimeError("Weighted step outside source context")
        row = self._record(kind="step", status="running")
        try:
            result, details = weighted_step(
                matched,
                current,
                [self._confidence[p] for p, _, _ in matched],
                balance=self.profile in ("normal-balance", "planarity-balance"),
            )
            row.update(details, status="completed")
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    def iterate(
        self, source: Sequence[Point], correspond: icp_plane._Correspond, **kwargs: Any
    ) -> icp_plane._FitPlane:
        if self._iteration is None or self._confidence is not None:
            raise RuntimeError("Weighting requires a unique active iteration context")
        if not 6 <= len(source) <= 2048:
            raise ValueError("Weighted source budget is 6..2048 points")
        row = self._record(kind="fit", status="running", source_points=len(source))
        try:
            points = tuple(source)
            if self.profile == "normal-balance":
                values = (1.0,) * len(points)
            else:
                if points not in self._cache:
                    self._cache[points] = tuple(planarity_weights(points))
                    if len(self._cache) > 32:
                        self._cache.popitem(last=False)
                self._cache.move_to_end(points)
                values = self._cache[points]
            self._confidence = dict(zip(points, values, strict=True))
            row["source_zero_weights"] = sum(value == 0 for value in values)
            result = self._iteration(source, correspond, **kwargs)
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
            self._confidence = None

    @contextmanager
    def active(self) -> Iterator[None]:
        if _ACTIVE.get() or self._iteration is not None:
            raise RuntimeError("Solver mechanisms cannot be nested")
        token = _ACTIVE.set(True)
        self._iteration = icp_plane._iterate
        try:
            with (
                patch.object(icp_plane, "_iterate", self.iterate),
                patch.object(icp_plane, "_solve_point_to_plane", self.step),
            ):
                yield
        finally:
            self._iteration = self._confidence = None
            self._cache.clear()
            _ACTIVE.reset(token)
