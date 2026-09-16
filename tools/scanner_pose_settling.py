"""Closed S1 stopping experiment. Process-local only; never imported by the app."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from unittest.mock import patch

from scanner3d import icp_plane
from scanner3d.geometry import Point, Rigid
from scanner3d.pipeline import frame_travel
from scanner3d.registration import kept_count


def iterate(
    source: Sequence[Point],
    correspond: icp_plane._Correspond,
    *,
    initial: Rigid,
    max_iterations: int,
    threshold: float,
    keep_fraction: float,
    gate: float,
    accelerated: bool = False,
) -> icp_plane._FitPlane:
    transform = initial
    rmse = math.inf
    previous: float | None = None
    correspondences = trimmed = rejected = 0
    trim_distance = gate
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        pairs, rejected = correspond(source, transform)
        if len(pairs) < 6:
            raise icp_plane.IcpPlaneCorrespondencesInsufficient(
                f"only {len(pairs)} source points found a normal-bearing partner "
                f"at iteration {iterations}"
            )
        pairs.sort(key=lambda pair: pair[0])
        keep = min(len(pairs), kept_count(len(pairs), keep_fraction))
        correspondences = keep
        trimmed = len(pairs) - keep
        trim_distance = pairs[keep - 1][0]
        matched = [(pair[1], pair[2], pair[3]) for pair in pairs[:keep]]
        before = transform
        transform = icp_plane._solve_point_to_plane(matched, transform, accelerated=accelerated)
        moved = [(transform.apply(o), q, n) for o, q, n in matched]
        rmse = icp_plane._point_to_plane_rmse(moved)
        motion = frame_travel(source, before.inverse().compose(transform))
        if previous is not None and abs(previous - rmse) <= threshold and motion <= threshold:
            converged = True
            break
        previous = rmse
    return icp_plane._FitPlane(
        transform=transform,
        rmse=rmse,
        iterations=iterations,
        converged=converged,
        correspondences=correspondences,
        trimmed=trimmed,
        rejected=rejected,
        trim_distance=trim_distance,
    )


@contextmanager
def pose_settling() -> Iterator[None]:
    with patch.object(icp_plane, "_iterate", iterate):
        yield
