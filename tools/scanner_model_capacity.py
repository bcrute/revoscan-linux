"""M22 doubles only the ordinary model budget; diagnostic GPU routing is explicit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import scanner_source_capacity as source_capacity
from scanner_factorial_config import Factors

from scanner3d import icp_plane, tracking
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.tracking import TrackingPolicy

PROFILES = ("observe-1800", "model-3600")
CAPACITIES = ("ordinary-900", "dense-1800")
ORIGINAL_ICP = icp_plane.icp_point_to_plane
ORIGINAL_SETTINGS = source_capacity.settings
_ACTIVE: ContextVar[bool] = ContextVar("m22_model_capacity_active", default=False)


def settings(
    capacity: str, profile: str, factors: Factors, solver: str
) -> tuple[int, TrackingPolicy, str]:
    if (
        capacity not in CAPACITIES
        or profile not in PROFILES
        or factors.values != ("grid", "pooled", "residual")
        or solver not in ("observe-origin", "centered")
    ):
        raise ValueError("M22 requires its closed model/capacity/grid/solver roster")
    source = "observe-900" if capacity == "ordinary-900" else "source-1800"
    target, policy = ORIGINAL_SETTINGS(source, factors, solver, None)
    if profile == "model-3600":
        policy = replace(policy, model_reference_points=3600)
    return target, policy, source


class ModelCapacityExperiment:
    """Retain the real policy/result and record provider use, including cache hits."""

    def __init__(self, capacity: str, profile: str, factors: Factors, solver: str) -> None:
        self.preparation_target, self.policy, self.source_profile = settings(
            capacity, profile, factors, solver
        )
        self.profile, self.factors, self.solver = profile, factors, solver
        self.cache_limit = 2048 if profile == "observe-1800" else 4096
        self.records: list[dict[str, Any]] = []
        self._original: Callable[..., icp_plane.IcpPlaneResult] | None = None
        self.settings_calls = 0

    def capacity_settings(
        self, source: str, factors: Factors, solver: str | None, model: str | None
    ) -> tuple[int, TrackingPolicy]:
        if not _ACTIVE.get() or (source, factors, solver, model) != (
            self.source_profile,
            self.factors,
            self.solver,
            None,
        ):
            raise RuntimeError("M22 runner requested an undeclared source policy")
        self.settings_calls += 1
        return self.preparation_target, self.policy

    def fit(
        self, source: Sequence[Point], target: Sequence[Point], **kwargs: Any
    ) -> icp_plane.IcpPlaneResult:
        if self._original is None or not _ACTIVE.get():
            raise RuntimeError("M22 requires its active nonnested context")
        if len(self.records) >= 90000:
            raise RuntimeError("M22 exceeds the declared 90000 fit-call bound")
        record: dict[str, Any] = dict(
            ordinal=len(self.records),
            profile=self.profile,
            status="running",
            source_points=len(source),
            target_points=len(target),
            normal_cache_max_points=icp_plane.NORMAL_CACHE_MAX_POINTS,
            provider_calls=[],
        )
        self.records.append(record)
        token = None
        try:
            provider = icp_plane._NORMAL_ACCELERATOR.get()
            if (
                kwargs.get("normal_neighbours", 10) != 10
                or kwargs.get("max_iterations", 50) != 50
                or kwargs.get("voxel_size") != 1.5
                or kwargs.get("keep_fraction") != 0.8
                or kwargs.get("backend") != "scipy"
                or not isinstance(kwargs.get("initial"), Rigid)
                or provider is None
                or self.cache_limit != icp_plane.NORMAL_CACHE_MAX_POINTS
                or len(source) > self.policy.candidate_source_points
                or len(target) > self.policy.model_reference_points
            ):
                raise RuntimeError("M22 requires its bounded original fit and GPU provider")

            def observed(points: tuple[Point, ...], neighbours: int) -> icp_plane.NormalField:
                call: dict[str, Any] = dict(
                    points=len(points),
                    neighbours=neighbours,
                    status="running",
                    points_sha256=hashlib.sha256(
                        json.dumps(points, separators=(",", ":"), allow_nan=False).encode()
                    ).hexdigest(),
                    above_original_cache_limit=len(points) > 2048,
                )
                record["provider_calls"].append(call)
                if len(record["provider_calls"]) != 1 or not 6 <= len(points) <= self.cache_limit:
                    raise RuntimeError("M22 requires exactly one bounded target-normal field")
                field = provider(points, neighbours)
                if (
                    neighbours != 10
                    or field.requested_neighbours != 10
                    or tuple(e.point for e in field.estimates) != points
                ):
                    raise RuntimeError("M22 provider field identity or neighbors do not match")
                call.update(status="completed", field=field.as_dict(), point_identity_exact=True)
                return field

            token = icp_plane._NORMAL_ACCELERATOR.set(observed)
            result = self._original(source, target, **kwargs)
            if (
                len(record["provider_calls"]) != 1
                or result.voxel_size != 1.5
                or result.normal_neighbours != 10
                or not 1 <= result.iterations <= 50
            ):
                raise RuntimeError("M22 fit bypassed provider or returned changed metadata")
            record.update(status="completed", result=result.as_dict())
            return result
        except (ScannerError, ValueError, ArithmeticError) as error:
            record.update(status="refused", error=f"{type(error).__name__}: {error}")
            raise
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if token is not None:
                icp_plane._NORMAL_ACCELERATOR.reset(token)
            for call in record["provider_calls"]:
                if call["status"] == "running":
                    call.update(status=record["status"], error=record.get("error"))

    @contextmanager
    def active(self) -> Iterator[None]:
        if (
            _ACTIVE.get()
            or self._original is not None
            or vars(tracking)["icp_point_to_plane"] is not ORIGINAL_ICP
            or source_capacity.settings is not ORIGINAL_SETTINGS
            or icp_plane.NORMAL_CACHE_MAX_POINTS != 2048
            or icp_plane.NORMAL_CACHE_ENTRIES != 32
        ):
            raise RuntimeError("M22 refuses nested scope or nonoriginal tracker/cache/settings")
        token = _ACTIVE.set(True)
        self._original = ORIGINAL_ICP
        try:
            with (
                patch.object(tracking, "icp_point_to_plane", self.fit),
                patch.object(source_capacity, "settings", self.capacity_settings),
                patch.object(icp_plane, "NORMAL_CACHE_MAX_POINTS", self.cache_limit),
            ):
                yield
        finally:
            self._original = None
            _ACTIVE.reset(token)
