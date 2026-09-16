"""M18 fixed voxel pyramid around the original ICP; offline diagnostic only."""

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
from scanner3d.geometry import Point, Rigid
from scanner3d.tracking import TrackingPolicy

PROFILES = ("observe-single", "pyramid-4-2-1")
CAPACITIES = ("ordinary-900", "dense-1800")
VOXELS = (6.0, 3.0, 1.5)
ORIGINAL_ICP = icp_plane.icp_point_to_plane
_ACTIVE: ContextVar[bool] = ContextVar("m18_multiscale_active", default=False)


def settings(
    capacity: str, profile: str, factors: Factors, solver: str
) -> tuple[int, TrackingPolicy, str]:
    if (capacity not in CAPACITIES or profile not in PROFILES
        or factors.values != ("grid", "pooled", "residual")
        or solver not in ("observe-origin", "centered")):
        raise ValueError("M18 requires its closed pyramid/capacity/grid/solver roster")
    source = "observe-900" if capacity == "ordinary-900" else "source-1800"
    target, policy = capacity_settings(source, factors, solver, None)
    return target, policy, source


class MultiscaleExperiment:
    """Only the last original result may reach the original tracker gates."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("M18 requires a declared pyramid profile")
        self.profile = profile
        self.voxels = (1.5,) if profile == "observe-single" else VOXELS
        self.records: list[dict[str, Any]] = []
        self._original: Callable[..., icp_plane.IcpPlaneResult] | None = None

    def fit(
        self, source: Sequence[Point], target: Sequence[Point], **kwargs: Any
    ) -> icp_plane.IcpPlaneResult:
        if self._original is None or not _ACTIVE.get():
            raise RuntimeError("M18 requires its active nonnested context")
        if len(self.records) >= 90000:
            raise RuntimeError("M18 exceeds the declared 90000 fit-call bound")
        record: dict[str, Any] = dict(ordinal=len(self.records), profile=self.profile,
            status="running", source_points=len(source), target_points=len(target),
            voxels=list(self.voxels), levels=[], total_completed_iterations=0,
            maximum_total_iterations=50 * len(self.voxels))
        self.records.append(record)
        try:
            if (kwargs.get("normal_neighbours", 10) != 10
                or kwargs.get("max_iterations", 50) != 50
                or kwargs.get("voxel_size") != 1.5 or kwargs.get("keep_fraction") != .8
                or kwargs.get("backend") != "scipy"
                or not isinstance(kwargs.get("initial"), Rigid)
                or icp_plane._NORMAL_ACCELERATOR.get() is None):
                raise RuntimeError("M18 requires original fit parameters and a GPU provider")
            initial = kwargs["initial"]
            result: icp_plane.IcpPlaneResult | None = None
            for ordinal, voxel in enumerate(self.voxels):
                level: dict[str, Any] = dict(ordinal=ordinal, voxel_size=voxel,
                    initial=initial.as_dict(), status="running")
                record["levels"].append(level)
                effective = kwargs if self.profile == "observe-single" else (
                    kwargs | dict(voxel_size=voxel, initial=initial))
                result = self._original(source, target, **effective)
                if (result.voxel_size != voxel or result.normal_neighbours != 10
                    or not 1 <= result.iterations <= 50):
                    raise RuntimeError("M18 returned ICP metadata does not match its level")
                level.update(status="completed", result=result.as_dict())
                record["total_completed_iterations"] += result.iterations
                initial = result.transform
            assert result is not None
            record["status"] = "completed"
            return result
        except (ScannerError, ValueError, ArithmeticError) as error:
            record.update(status="refused", error=f"{type(error).__name__}: {error}")
            if record["levels"] and record["levels"][-1]["status"] == "running":
                record["levels"][-1].update(status="refused", error=record["error"])
            raise
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            if record["levels"] and record["levels"][-1]["status"] == "running":
                record["levels"][-1].update(status="failed", error=record["error"])
            raise

    @contextmanager
    def active(self) -> Iterator[None]:
        if (_ACTIVE.get() or self._original is not None
            or vars(tracking)["icp_point_to_plane"] is not ORIGINAL_ICP):
            raise RuntimeError("M18 refuses nested context or a nonoriginal tracker ICP hook")
        token = _ACTIVE.set(True)
        self._original = ORIGINAL_ICP
        try:
            with patch.object(tracking, "icp_point_to_plane", self.fit):
                yield
        finally:
            self._original = None
            _ACTIVE.reset(token)
