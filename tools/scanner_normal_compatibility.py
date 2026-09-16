"""M19 sign-invariant source-normal correspondence filters, offline only."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

import numpy as np
from scanner_factorial_config import Factors
from scanner_source_capacity import settings as capacity_settings

from scanner3d import icp_plane
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.tracking import TrackingPolicy

PROFILES = ("observe-pairs", "source-normal-required", "normal-angle-45")
Pair = tuple[float, Point, Point, Point]
CAPACITIES = ("ordinary-900", "dense-1800")
ORIGINAL_ITERATE = icp_plane._iterate
_ACTIVE: ContextVar[bool] = ContextVar("m19_normal_compatibility", default=False)


def settings(
    capacity: str, profile: str, factors: Factors, solver: str
) -> tuple[int, TrackingPolicy, str]:
    if (
        capacity not in CAPACITIES
        or profile not in PROFILES
        or factors.values != ("grid", "pooled", "residual")
        or solver not in ("observe-origin", "centered")
    ):
        raise ValueError("M19 requires its closed normal-filter/capacity/grid/solver roster")
    source = "observe-900" if capacity == "ordinary-900" else "source-1800"
    target, policy = capacity_settings(source, factors, solver, None)
    return target, policy, source


def compatible_pairs(
    normals: Mapping[Point, Point | None], current: Rigid, pairs: list[Pair], profile: str
) -> tuple[list[Pair], int, dict[str, Any]]:
    """Keep original tuples/order; observation returns the actual original list."""
    if profile not in PROFILES or not 6 <= len(normals) <= 2048:
        raise ValueError("M19 requires its declared profile and bounded source normal field")
    if len(pairs) > len(normals) or len({p for _, p, _, _ in pairs}) != len(pairs):
        raise ValueError("M19 requires at most one forward pair per actual source")
    if any(point not in normals for _, point, _, _ in pairs):
        raise ValueError("M19 correspondence source is outside the normal field")
    available, compatible = [], []
    cosines = np.array([], dtype=float)
    if pairs:
        coordinates = np.asarray([(p, q, n) for _, p, q, n in pairs], dtype=float)
        distances = np.asarray([d for d, _, _, _ in pairs], dtype=float)
        if (
            coordinates.shape != (len(pairs), 3, 3)
            or not np.isfinite(coordinates).all()
            or not np.isfinite(distances).all()
            or np.any(distances < 0)
        ):
            raise ValueError("M19 requires finite original correspondences")
        target_lengths = np.linalg.norm(coordinates[:, 2], axis=1)
        if not np.isfinite(target_lengths).all() or np.any(target_lengths <= 0):
            raise ValueError("M19 requires nonzero finite target normals")
        positions = [i for i, (_, point, _, _) in enumerate(pairs) if normals[point] is not None]
        available = [pairs[i] for i in positions]
        if positions:
            values = np.asarray([normals[pairs[i][1]] for i in positions], dtype=float)
            if values.shape != (len(positions), 3):
                raise ValueError("M19 requires three-component source normals")
            lengths = np.linalg.norm(values, axis=1)
            if (
                not np.isfinite(values).all()
                or not np.isfinite(lengths).all()
                or np.any(lengths <= 0)
            ):
                raise ValueError("M19 requires nonzero finite three-component source normals")
            moved = (values / lengths[:, None]) @ np.asarray(current.rotation).T
            target_axes = coordinates[positions, 2] / target_lengths[positions, None]
            dots = np.sum(moved * target_axes, axis=1)
            square_lengths = np.sum(moved * moved, axis=1) * np.sum(
                target_axes * target_axes, axis=1
            )
            cosines = np.abs(dots) / np.sqrt(square_lengths)
            compatible = [
                pair
                for pair, keep in zip(available, 2 * dots * dots >= square_lengths, strict=True)
                if keep
            ]
    selected = (
        pairs
        if profile == "observe-pairs"
        else available
        if profile == "source-normal-required"
        else compatible
    )
    summary = dict(
        forward=len(pairs),
        source_normal_available=len(available),
        angle_compatible=len(compatible),
        source_normals_missing=len(pairs) - len(available),
        angle_incompatible=len(available) - len(compatible),
        selected=len(selected),
        absolute_cosine_percentiles=None
        if not len(cosines)
        else np.percentile(cosines, [0, 10, 50, 90, 100]).tolist(),
    )
    return selected, len(pairs) - len(selected), summary


class NormalCompatibilityExperiment:
    """A scoped correspondence-only intervention; the finite solver is untouched."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("M19 requires a declared normal-filter profile")
        self.profile = profile
        self.records: list[dict[str, Any]] = []
        self._iteration: Callable[..., icp_plane._FitPlane] | None = None
        self._inside_fit = False

    def iterate(
        self, source: Sequence[Point], correspond: icp_plane._Correspond, **kwargs: Any
    ) -> icp_plane._FitPlane:
        if self._iteration is None or not _ACTIVE.get() or self._inside_fit:
            raise RuntimeError("M19 requires one active nonnested iteration context")
        if len(self.records) >= 90000:
            raise RuntimeError("M19 exceeds the declared 90000 iterator-call bound")
        row: dict[str, Any] = dict(
            ordinal=len(self.records),
            profile=self.profile,
            status="running",
            source_points=len(source),
            calls=[],
        )
        self.records.append(row)
        self._inside_fit = True
        try:
            provider = icp_plane._NORMAL_ACCELERATOR.get()
            if (
                provider is None
                or not 6 <= len(source) <= 2048
                or kwargs.get("max_iterations") != 50
                or kwargs.get("keep_fraction") != 0.8
            ):
                raise RuntimeError("M19 requires original iteration parameters and GPU provider")
            field = provider(tuple(source), 10)
            if (
                tuple(e.point for e in field.estimates) != tuple(source)
                or field.requested_neighbours != 10
            ):
                raise ValueError("M19 source-normal field has different points or neighbors")
            normals = {e.point: e.normal for e in field.estimates}
            if len(normals) != len(source):
                raise ValueError("M19 requires unique voxelized source points")
            row["source_field"] = field.as_dict()

            def filtered(src: Sequence[Point], transform: Rigid) -> tuple[list[Pair], int]:
                if tuple(src) != tuple(source):
                    raise RuntimeError("M19 iterator changed the declared source cloud")
                pairs, rejected = correspond(src, transform)
                selected, removed, counts = compatible_pairs(
                    normals, transform, pairs, self.profile
                )
                row["calls"].append(
                    dict(
                        **counts,
                        original_rejected=rejected,
                        additional_rejected=removed,
                        current_transform=transform.as_dict(),
                    )
                )
                return selected, rejected + removed

            result = self._iteration(source, filtered, **kwargs)
            row.update(
                status="completed",
                iterations=result.iterations,
                externally_reported_target_plane_rmse=result.rmse,
                final_transform=result.transform.as_dict(),
            )
            return result
        except (ScannerError, ValueError, ArithmeticError) as error:
            row.update(status="refused", error=f"{type(error).__name__}: {error}")
            raise
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self._inside_fit = False

    @contextmanager
    def active(self) -> Iterator[None]:
        if (
            _ACTIVE.get()
            or self._iteration is not None
            or vars(icp_plane)["_iterate"] is not ORIGINAL_ITERATE
        ):
            raise RuntimeError("M19 refuses nested context or a nonoriginal ICP iterator")
        token = _ACTIVE.set(True)
        self._iteration = ORIGINAL_ITERATE
        try:
            with patch.object(icp_plane, "_iterate", self.iterate):
                yield
        finally:
            self._iteration = None
            self._inside_fit = False
            _ACTIVE.reset(token)
