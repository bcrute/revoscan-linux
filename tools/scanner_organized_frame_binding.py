"""Offline prepare/offer/admission identity ledger; no automatic tracker hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from scanner_image_normal_review import original_pixels
from scanner_organized_model_normals import NativeField

from scanner3d.geometry import Point
from scanner3d.tracking import MAX_RETAINED_CLOUDS, _sampled


@dataclass(frozen=True, slots=True)
class PreparedOffer:
    ordinal: int
    raw: bytes
    points: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class RetainedRaw:
    offer: PreparedOffer
    points: tuple[Point, ...]


class FrameBinding:
    """Consume every prepared offer, but allocate identities only on admission.

    A caller must supply the actual accepted-count and complete retained-cloud
    map from one tracker. This helper neither accepts a frame nor changes the
    tracker's own sampling, memory, refusal or eviction policy.
    """

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity not in (900, 1800):
            raise ValueError("Declared retained capacity 900/1800 required")
        self.capacity = capacity
        self.prepared_count = 0
        self.accepted_count = 0
        self.tracker: object | None = None
        self.pending: PreparedOffer | None = None
        self.active: PreparedOffer | None = None
        self.retained: dict[int, RetainedRaw] = {}
        self.fields: dict[tuple[int, int], NativeField] = {}

    def prepare(self, raw: bytes, points: tuple[Point, ...]) -> PreparedOffer:
        if self.pending is not None or self.active is not None or self.prepared_count >= 10000:
            raise ValueError("Stale/reentrant preparation or diagnostic input bound exceeded")
        if (
            not isinstance(points, tuple)
            or not 1 <= len(points) <= self.capacity
            or any(not isinstance(p, tuple) for p in points)
        ):
            raise ValueError("Immutable prepared cloud exceeds declared capacity")
        pixels = original_pixels(raw, points)
        if len(set(pixels.tolist())) != len(points) or any(not 60 < p[2] < 300 for p in points):
            raise ValueError("Prepared points require distinct original in-band pixels")
        offer = PreparedOffer(self.prepared_count, raw, points)
        self.prepared_count += 1
        self.pending = offer
        return offer

    def begin(self, tracker: object, accepted_count: int, points: tuple[Point, ...]) -> None:
        if self.active is not None or self.pending is None:
            raise ValueError("An unused exact preparation is required before offering")
        if tracker is None or (self.tracker is not None and tracker is not self.tracker):
            raise ValueError("One explicit tracker identity is required")
        if type(accepted_count) is not int or accepted_count != self.accepted_count:
            raise ValueError("Accepted frame index drifted from the raw ledger")
        if not isinstance(points, tuple) or points != self.pending.points:
            raise ValueError("Actual offered points differ from the pending raw preparation")
        self.tracker = tracker
        self.active, self.pending = self.pending, None

    def finish(self, accepted_count: int, retained_clouds: Mapping[int, tuple[Point, ...]]) -> None:
        if self.active is None or type(accepted_count) is not int:
            raise ValueError("No active offer with an actual accepted count")
        if accepted_count not in (self.accepted_count, self.accepted_count + 1):
            raise ValueError("An offer must admit zero or one accepted frame")
        if len(retained_clouds) > MAX_RETAINED_CLOUDS or any(
            type(i) is not int or not 0 <= i < accepted_count for i in retained_clouds
        ):
            raise ValueError("Actual retained frame map violates original identity/memory bounds")
        updated = dict(self.retained)
        if accepted_count == self.accepted_count + 1:
            index = self.accepted_count
            expected = _sampled(self.active.points, self.capacity)
            if retained_clouds.get(index) != expected:
                raise ValueError("New accepted cloud differs from original retained sampling")
            updated[index] = RetainedRaw(self.active, expected)
        for index, points in retained_clouds.items():
            if index not in updated or updated[index].points != points:
                raise ValueError("Retained cloud has unknown, resurrected or altered raw ownership")
        # Commit only after every identity validates; a failed finish stays active.
        self.retained = {i: updated[i] for i in retained_clouds}
        self.fields = {key: value for key, value in self.fields.items() if key[0] in self.retained}
        self.accepted_count, self.active = accepted_count, None

    def native(self, index: int, radius: int, actual: tuple[Point, ...]) -> NativeField:
        if type(index) is not int or index not in self.retained:
            raise ValueError("No retained raw ownership for this accepted frame")
        retained = self.retained[index]
        if actual != retained.points:
            raise ValueError("Actual retained reference differs from raw-backed cloud")
        if type(radius) is not int or radius not in (5, 10):
            raise ValueError("Declared radius required")
        key = (index, radius)
        if key not in self.fields:
            self.fields[key] = NativeField(retained.offer.raw, actual, radius)
        return self.fields[key]

    def complete(self) -> None:
        if self.pending is not None or self.active is not None:
            raise ValueError("Unconsumed preparation or unfinished offer at closeout")
