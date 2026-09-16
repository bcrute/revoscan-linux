"""Offline raw-backed model-normal transport; no tracker or hardware hooks.

The pooled PCA field remains authoritative for availability and fallback.
Coordinates never stand in for raw-frame identity or forward/reverse fit role.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scanner_image_normal_review import original_pixels, representatives
from scanner_normal_confidence_review import combine_confidence
from scanner_normal_hybrid_review import _unit
from scanner_organized_plane_review import plane_directions

from scanner3d.geometry import Point, Rigid
from scanner3d.icp_plane import NormalEstimate, NormalField
from scanner3d.tracking import _sampled

Provider = Callable[[tuple[Point, ...], int], NormalField]


@dataclass(frozen=True, slots=True)
class NativePlane:
    pixel: int
    radius: int
    status: str
    normal: Point | None
    support_count: int | None
    required_support: int
    perpendicular_rms: float | None

    def __post_init__(self) -> None:
        if type(self.radius) is not int or self.radius not in (5, 10):
            raise ValueError("Only declared image radii 5/10 are supported")
        area = (2 * self.radius + 1) ** 2
        if type(self.pixel) is not int or not 0 <= self.pixel < 256000:
            raise ValueError("A native image pixel is required")
        if type(self.required_support) is not int or self.required_support != (3 * area + 3) // 4:
            raise ValueError("Original support threshold differs")
        if self.status not in (
            "usable",
            "border",
            "center-missing-or-band",
            "insufficient-support",
            "degenerate",
            "nonplanar",
        ) or (self.status == "usable") != (self.normal is not None):
            raise ValueError("Unknown or inconsistent native-plane status")
        if self.normal is not None and not isinstance(self.normal, tuple):
            raise ValueError("Native direction must be immutable")
        _unit(self.normal)
        count, rms = self.support_count, self.perpendicular_rms
        if self.status == "border":
            if count is not None or rms is not None:
                raise ValueError("Border plane cannot have estimated support or RMS")
            return
        if type(count) is not int or not 0 <= count <= area:
            raise ValueError("Invalid native image support count")
        if self.status in ("usable", "degenerate", "nonplanar") and count < self.required_support:
            raise ValueError("Plane estimate lacks required image support")
        if self.status == "insufficient-support" and count >= self.required_support:
            raise ValueError("Insufficient status contradicts image support")
        if self.status in ("usable", "nonplanar"):
            if type(rms) not in (float, int) or rms is None or not math.isfinite(rms) or rms < 0:
                raise ValueError("Finite nonnegative native RMS required")
            if (rms <= 0.25) != (self.status == "usable"):
                raise ValueError("Native RMS contradicts plane status")
        elif rms is not None:
            raise ValueError("Unavailable fit cannot report a perpendicular RMS")

    def rotated(self, pose: Rigid) -> dict[str, Any]:
        rotation = Rigid.of(pose.rotation, (0.0, 0.0, 0.0))
        return dict(
            pixel=self.pixel,
            radius=self.radius,
            status=self.status,
            normal=None if self.normal is None else rotation.apply(self.normal),
            support_count=self.support_count,
            required_support=self.required_support,
            perpendicular_rms=self.perpendicular_rms,
        )


@dataclass(frozen=True, slots=True)
class NativeField:
    """Immutable estimates constructed only from exact original deprojections."""

    raw: bytes = field(repr=False)
    points: tuple[Point, ...]
    radius: int
    raw_sha256: str = field(init=False)
    planes: tuple[NativePlane, ...] = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.points, tuple)
            or not 1 <= len(self.points) <= 1800
            or any(not isinstance(p, tuple) or len(p) != 3 for p in self.points)
        ):
            raise ValueError("An immutable retained cloud of 1..1800 measured points is required")
        pixels = original_pixels(self.raw, self.points)
        if len(set(pixels.tolist())) != len(pixels) or any(
            not 60 < p[2] < 300 for p in self.points
        ):
            raise ValueError("Retained cloud needs distinct original in-band pixels")
        if type(self.radius) is not int or self.radius not in (5, 10):
            raise ValueError("Only declared image radii 5/10 are supported")
        rows = plane_directions(self.raw, pixels, radius=self.radius)
        planes = tuple(
            NativePlane(
                row["pixel"],
                row["radius"],
                row["status"],
                None if row["normal"] is None else tuple(row["normal"]),
                row["support_count"],
                row["required_support"],
                row["perpendicular_rms"],
            )
            for row in rows
        )
        object.__setattr__(self, "raw_sha256", hashlib.sha256(self.raw).hexdigest())
        object.__setattr__(self, "planes", planes)


@dataclass(frozen=True, slots=True)
class ModelFrame:
    index: int
    native: NativeField
    pose: Rigid

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or not isinstance(self.native, NativeField)
        ):
            raise ValueError("An accepted frame identity and validated native field are required")
        if not isinstance(self.pose, Rigid):
            raise ValueError("A model-from-native rigid pose is required")
        if (
            not isinstance(self.pose.rotation, tuple)
            or any(not isinstance(row, tuple) for row in self.pose.rotation)
            or not isinstance(self.pose.translation, tuple)
        ):
            raise ValueError("Model pose components must be immutable tuples")
        Rigid.of(self.pose.rotation, self.pose.translation)


@dataclass(frozen=True, slots=True)
class Origin:
    frame_index: int
    native_index: int
    raw_sha256: str
    pixel: int


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    frames: tuple[ModelFrame, ...]
    expected: tuple[Point, ...]
    budget: int = 1800
    origins: tuple[Origin, ...] = field(init=False)
    centers: tuple[Point, ...] = field(init=False)
    selected: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple) or not 1 <= len(self.frames) <= 8:
            raise ValueError("An ordered immutable model of 1..8 frames is required")
        if len({f.index for f in self.frames}) != len(self.frames):
            raise ValueError("Model references repeat an accepted frame identity")
        if len({f.native.radius for f in self.frames}) != 1:
            raise ValueError("One global image radius must apply to all model frames")
        if type(self.budget) is not int or not 1 <= self.budget <= 1800:
            raise ValueError("Model budget must be 1..1800")
        points: list[Point] = []
        origins: list[Origin] = []
        per_frame = max(1, self.budget // len(self.frames))
        for position, frame in enumerate(self.frames):
            native = frame.native
            indices = {p: i for i, p in enumerate(native.points)}
            sampled = _sampled(native.points, per_frame, offset=position)
            points.extend(frame.pose.apply_all(sampled))
            origins.extend(
                Origin(frame.index, indices[p], native.raw_sha256, native.planes[indices[p]].pixel)
                for p in sampled
            )
        if not isinstance(self.expected, tuple) or tuple(points[: self.budget]) != self.expected:
            raise ValueError("Reconstructed model differs from actual reference coordinates/order")
        centers, selected = representatives(self.expected)
        object.__setattr__(self, "origins", tuple(origins[: self.budget]))
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "selected", selected)


def _validate_pooled(pooled: NormalField, points: tuple[Point, ...]) -> None:
    if pooled.requested_neighbours != 10 or tuple(e.point for e in pooled.estimates) != points:
        raise ValueError("Pooled PCA changed point coordinates/order/count or requested neighbours")
    for estimate in pooled.estimates:
        if type(estimate.neighbours) is not int or not 0 <= estimate.neighbours <= 10:
            raise ValueError("Malformed original PCA neighbour count")
        _unit(estimate.normal)


def blend_model(
    model: ModelGeometry, pooled: NormalField, *, maximum_weight: float
) -> tuple[NormalField, tuple[dict[str, Any], ...]]:
    """Unchanged centroids/availability with image directions transported by owner."""
    if type(maximum_weight) not in (int, float) or maximum_weight not in (0.5, 1.0):
        raise ValueError("Only declared maximum weights 0.5/1 are supported")
    _validate_pooled(pooled, model.centers)
    frames = {f.index: f for f in model.frames}
    estimates: list[NormalEstimate] = []
    records: list[dict[str, Any]] = []
    for original, selected in zip(pooled.estimates, model.selected, strict=True):
        origin = model.origins[selected]
        frame = frames[origin.frame_index]
        plane = frame.native.planes[origin.native_index]
        rotated = plane.rotated(frame.pose)
        normal, confidence = combine_confidence(
            original.normal, rotated, maximum_weight=maximum_weight
        )
        if (normal is None) != (original.normal is None):
            raise ValueError("Hybrid changed original PCA availability")
        estimates.append(
            original
            if confidence["mode"] != "weighted"
            else NormalEstimate(original.point, normal, original.neighbours)
        )
        records.append(
            dict(
                model_index=selected,
                frame_index=origin.frame_index,
                native_index=origin.native_index,
                raw_sha256=origin.raw_sha256,
                pixel=origin.pixel,
                native_normal=plane.normal,
                model_normal=rotated["normal"],
                radius=plane.radius,
                native_status=plane.status,
                image_support_count=plane.support_count,
                image_required_support=plane.required_support,
                native_perpendicular_rms=plane.perpendicular_rms,
                pca_neighbours=original.neighbours,
                confidence=confidence,
            )
        )
    return NormalField(
        tuple(estimates),
        pooled.requested_neighbours,
        (
            *pooled.limitations,
            "Offline confidence blend of pooled PCA and raw image directions; not qualified.",
            "Centroids, order and PCA availability preserved; neighbour counts still describe PCA.",
            "Separate records identify image support, native owner and rotated direction.",
        ),
    ), tuple(records)


class RoleDispatcher:
    """Role check BEFORE any hybrid-result reuse, even with identical coordinates.

    A scoped replay hook must call this directly, not wrap it in the production
    coordinate-only normal_acceleration cache. The original pooled provider may
    retain its own unchanged cache underneath this dispatcher.
    """

    def __init__(
        self,
        model: ModelGeometry,
        reverse_points: tuple[Point, ...],
        provider: Provider,
        role: Callable[[], str],
        *,
        maximum_weight: float,
        observe: bool,
    ) -> None:
        if (
            type(observe) is not bool
            or type(maximum_weight) not in (int, float)
            or maximum_weight not in (0.5, 1.0)
        ):
            raise ValueError("Explicit observation mode and declared maximum weight required")
        if not isinstance(reverse_points, tuple) or not 1 <= len(reverse_points) <= 1800:
            raise ValueError("Explicit reverse-current target points are required")
        if any(not isinstance(p, tuple) for p in reverse_points):
            raise ValueError("Reverse-current target points must be immutable tuples")
        if any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in reverse_points):
            raise ValueError("Reverse-current target must contain finite points")
        self.model, self.reverse_points, self.provider, self.role = (
            model,
            reverse_points,
            provider,
            role,
        )
        self.maximum_weight, self.observe = maximum_weight, observe
        self.result: tuple[NormalField, tuple[dict[str, Any], ...]] | None = None
        self.calls: list[dict[str, Any]] = []

    def __call__(self, points: tuple[Point, ...], neighbours: int) -> NormalField:
        direction = self.role()
        if (
            direction not in ("forward", "reverse")
            or type(neighbours) is not int
            or neighbours != 10
        ):
            raise ValueError("Explicit fit direction and unchanged ten-neighbour request required")
        expected = self.model.centers if direction == "forward" else self.reverse_points
        if points != expected or len(self.calls) >= 16:
            raise ValueError(
                "Normal request has wrong role-bound points or exceeds diagnostic bound"
            )
        self.calls.append(dict(direction=direction, points=len(points), observe=self.observe))
        if self.observe or direction == "reverse":
            original = self.provider(points, neighbours)
            _validate_pooled(original, points)
            return original
        if self.result is None:
            self.result = blend_model(
                self.model, self.provider(points, neighbours), maximum_weight=self.maximum_weight
            )
        return self.result[0]
