"""M26: one nonrecursive, valid-nine image median; original raw evidence is immutable."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

import numpy as np
import scanner_factorial_replay as static_replay
import scanner_pixel_grid as grid

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.preparation import DepthBand, PreparedCloud, ensure_depth_band

PROFILES = ("observe-raw", "valid-nine-median3")
ORIGINAL_GRID = grid.prepare_frame
_ACTIVE: ContextVar[DepthMedianExperiment | None] = ContextVar("m26_depth_median", default=None)


def filter_depth(raw: bytes, band: DepthBand) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(raw, bytes) or len(raw) != 512000 or band != DepthBand(60, 300):
        raise ValueError("M26 requires immutable 640x400 raw bytes and the original 60..300 band")
    original = np.frombuffer(raw, dtype="<u2").reshape(400, 640)
    windows = np.lib.stride_tricks.sliding_window_view(original, (3, 3))
    samples = windows.reshape(398, 638, 9)
    low, high = samples.min(axis=2), samples.max(axis=2)
    eligible = (
        (low.astype(np.float64) * DEPTH_SCALE_METRES > band.near)
        & (high.astype(np.float64) * DEPTH_SCALE_METRES < band.far)
        & ((high.astype(np.int32) - low) * DEPTH_SCALE_METRES <= 1.5)
    )
    medians = np.partition(samples, 4, axis=2)[:, :, 4]
    result = original.copy()
    center = result[1:-1, 1:-1]
    center[eligible] = medians[eligible]
    delta = result.astype(np.int32) - original
    changed = delta[delta != 0]
    original_band = (original * DEPTH_SCALE_METRES > band.near) & (
        original * DEPTH_SCALE_METRES < band.far
    )
    result_band = (result * DEPTH_SCALE_METRES > band.near) & (
        result * DEPTH_SCALE_METRES < band.far
    )
    if not np.array_equal(original == 0, result == 0) or not np.array_equal(
        original_band, result_band
    ):
        raise RuntimeError("M26 changed return or band membership")
    output = result.astype("<u2", copy=False).tobytes() if changed.size else raw
    quantiles = [0, 0.25, 0.5, 0.75, 1]
    return output, dict(
        eligible_pixels=int(np.count_nonzero(eligible)),
        changed_pixels=int(changed.size),
        positive_changes=int(np.count_nonzero(changed > 0)),
        negative_changes=int(np.count_nonzero(changed < 0)),
        signed_changed_code_quantiles=np.quantile(changed, quantiles).tolist()
        if changed.size
        else None,
        absolute_changed_code_quantiles=np.quantile(np.abs(changed), quantiles).tolist()
        if changed.size
        else None,
        quantile_probabilities=quantiles,
        nonzero_pixels=int(np.count_nonzero(original)),
        in_band_pixels=int(np.count_nonzero(original_band)),
        return_membership_exact=True,
        band_membership_exact=True,
    )


class DepthMedianExperiment:
    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("M26 requires an explicit closed depth-preparation profile")
        self.profile = profile
        self.records: list[dict[str, Any]] = []

    def prepare(
        self, raw: bytes, *, target_points: int, depth_band: DepthBand | None = None
    ) -> PreparedCloud:
        band = ensure_depth_band(depth_band)
        if (
            _ACTIVE.get() is not self
            or len(self.records) >= 3000
            or target_points not in (900, 1800)
        ):
            raise ValueError("M26 requires its active scope, 900/1800 points and <=3000 calls")
        if not isinstance(raw, bytes) or len(raw) != 512000 or band != DepthBand(60, 300):
            raise ValueError("M26 requires exact raw shape and original band")
        record: dict[str, Any] = dict(
            preparation_call=len(self.records),
            profile=self.profile,
            status="running",
            original_raw_sha256=hashlib.sha256(raw).hexdigest(),
            target_points=target_points,
            depth_band=band.as_dict(),
        )
        self.records.append(record)
        try:
            derived, stats = (
                (raw, None) if self.profile == "observe-raw" else filter_depth(raw, band)
            )
            record.update(derived_raw_sha256=hashlib.sha256(derived).hexdigest(), filtering=stats)
            cloud = ORIGINAL_GRID(derived, target_points=target_points, depth_band=band)
            record.update(
                status="completed",
                prepared_points=len(cloud.points),
                deprojected_points=cloud.deprojected_points,
                clipped_points=cloud.clipped_points,
            )
            return cloud
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    @contextmanager
    def active(self) -> Iterator[None]:
        if (
            _ACTIVE.get()
            or grid.prepare_frame is not ORIGINAL_GRID
            or vars(static_replay).get("prepare_grid") is not ORIGINAL_GRID
        ):
            raise RuntimeError("M26 requires original preparation hooks and a nonnested scope")
        token = _ACTIVE.set(self)
        replacement = self.prepare
        try:
            with (
                patch.object(grid, "prepare_frame", replacement),
                patch.object(static_replay, "prepare_grid", replacement),
            ):
                yield
                if (
                    grid.prepare_frame is not replacement
                    or vars(static_replay).get("prepare_grid") is not replacement
                ):
                    raise RuntimeError("M26 preparation hook changed during its declared scope")
        finally:
            _ACTIVE.reset(token)
