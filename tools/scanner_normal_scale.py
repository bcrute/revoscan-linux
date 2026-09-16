"""M13 tracker ICP normal-neighborhood scale, a closed offline-only experiment."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

from scanner_factorial_config import Factors
from scanner_source_capacity import settings as capacity_settings

from scanner3d import icp_plane, tracking
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point
from scanner3d.tracking import TrackingPolicy

PROFILES = ("observe-10", "normal-20")
CAPACITIES = ("ordinary-900", "dense-1800")
ORIGINAL_ICP = icp_plane.icp_point_to_plane
_ACTIVE: ContextVar[bool] = ContextVar("m13_normal_scale_active", default=False)


def settings(
    capacity: str, profile: str, factors: Factors, solver: str
) -> tuple[int, TrackingPolicy, str]:
    if (
        capacity not in CAPACITIES or profile not in PROFILES
        or factors.values != ("grid", "pooled", "residual")
        or solver not in ("observe-origin", "centered")
    ):
        raise ValueError("M13 requires its closed capacity/normal-scale/grid/solver roster")
    source = "observe-900" if capacity == "ordinary-900" else "source-1800"
    target, policy = capacity_settings(source, factors, solver, None)
    return target, policy, source


class NormalScaleExperiment:
    """Only the ICP neighbor argument changes; every refusal remains a refusal."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("M13 requires a declared normal-scale profile")
        self.profile = profile
        self.neighbours = 10 if profile == "observe-10" else 20
        self.records: list[dict[str, Any]] = []
        self._original: Callable[..., icp_plane.IcpPlaneResult] | None = None

    def fit(
        self, source: Sequence[Point], target: Sequence[Point], **kwargs: Any
    ) -> icp_plane.IcpPlaneResult:
        if self._original is None or not _ACTIVE.get():
            raise RuntimeError("M13 requires its active nonnested context")
        if len(self.records) >= 90000:
            raise RuntimeError("M13 exceeds the declared 90000 fit-call bound")
        record: dict[str, Any] = dict(
            ordinal=len(self.records), profile=self.profile, status="running",
            source_points=len(source), target_points=len(target),
            requested_normal_neighbours=kwargs.get("normal_neighbours", 10),
            actual_normal_neighbours=self.neighbours,
        )
        self.records.append(record)
        try:
            if (
                record["requested_normal_neighbours"] != 10
                or kwargs.get("backend") != "scipy"
                or icp_plane._NORMAL_ACCELERATOR.get() is None
            ):
                raise RuntimeError("M13 requires original ten-neighbor request and GPU provider")
            effective = kwargs if self.profile == "observe-10" else (
                kwargs | dict(normal_neighbours=20)
            )
            result = self._original(source, target, **effective)
            if result.normal_neighbours != self.neighbours:
                raise RuntimeError("M13 returned ICP metadata does not name actual neighbor count")
            record.update(status="completed", result=result.as_dict())
            return result
        except (ScannerError, ValueError, ArithmeticError) as error:
            record.update(status="refused", error=f"{type(error).__name__}: {error}")
            raise
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    @contextmanager
    def active(self) -> Iterator[None]:
        if (
            _ACTIVE.get() or self._original is not None
            or vars(tracking)["icp_point_to_plane"] is not ORIGINAL_ICP
        ):
            raise RuntimeError("M13 refuses nested context or a nonoriginal tracker ICP hook")
        token = _ACTIVE.set(True)
        self._original = ORIGINAL_ICP
        try:
            with patch.object(tracking, "icp_point_to_plane", self.fit):
                yield
        finally:
            self._original = None
            _ACTIVE.reset(token)
