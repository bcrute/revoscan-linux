"""S3 coarse-to-fine image-grid ordering. Offline, single-thread preparation only."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import scanner_pixel_rank

from scanner3d.preparation import DepthBand, PreparedCloud
from scanner3d.v4l2_capture import FRAME_WIDTH


def ranks(indices: np.ndarray) -> np.ndarray:
    x = np.asarray(indices % FRAME_WIDTH, dtype=np.uint64)
    y = np.asarray(indices // FRAME_WIDTH, dtype=np.uint64)
    result = np.zeros(len(indices), dtype=np.uint64)
    for bit in range(10):
        result |= np.asarray(((x >> bit) & 1) << (19 - 2 * bit), dtype=np.uint64)
        result |= np.asarray(((y >> bit) & 1) << (18 - 2 * bit), dtype=np.uint64)
    return result


def prepare_frame(
    raw: bytes, *, target_points: int, depth_band: DepthBand | None = None
) -> PreparedCloud:
    # Reuse identical projection/count/filter arithmetic; the patch ends before
    # returning a cloud. Never install this in a live/multithreaded application.
    with patch.object(scanner_pixel_rank, "ranks", ranks):
        return scanner_pixel_rank.prepare_frame(
            raw, target_points=target_points, depth_band=depth_band
        )
