"""M27 scoped offline raw-backed model normals; no production/GUI installation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

import scanner_factorial_replay as static_replay
import scanner_pixel_grid as grid
from scanner_factorial_config import Factors
from scanner_organized_frame_binding import FrameBinding
from scanner_organized_model_normals import ModelFrame, ModelGeometry, Provider, RoleDispatcher
from scanner_reverse_seed import ReverseSeedExperiment, settings

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Point, Rigid
from scanner3d.preparation import DepthBand, PreparedCloud
from scanner3d.registration import voxel_downsample
from scanner3d.tracking import IncrementalTracker, TrackCandidate, TrackDecision

PROFILES = ("observe-pooled", "organized-confidence")
ORIGINAL_GRID = grid.prepare_frame
ORIGINAL_OFFER = IncrementalTracker.offer
ORIGINAL_CANDIDATE = IncrementalTracker._candidate
ORIGINAL_SEED_FIT = ReverseSeedExperiment.fit
_ACTIVE: ContextVar[OrganizedModelExperiment | None] = ContextVar(
    "m27_organized_model", default=None
)
_FIT: ContextVar[tuple[ReverseSeedExperiment, int] | None] = ContextVar(
    "m27_seed_fit", default=None
)
_MODEL: ContextVar[bool] = ContextVar("m27_model_candidate", default=False)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def field_digest(field: icp_plane.NormalField) -> str:
    return digest([estimate.as_dict() for estimate in field.estimates])


class OrganizedModelExperiment:
    def __init__(
        self, profile: str, capacity: str, seed_profile: str, *, radius: int, maximum_weight: float
    ) -> None:
        if profile not in PROFILES or type(radius) is not int or radius not in (5, 10):
            raise ValueError("M27 requires a declared profile and radius")
        if type(maximum_weight) not in (int, float) or maximum_weight not in (0.5, 1.0):
            raise ValueError("M27 requires maximum weight 0.5/1")
        self.target, self.policy, self.source_profile = settings(
            capacity, seed_profile, Factors("grid", "pooled", "residual"), "centered"
        )
        self.profile, self.capacity, self.seed_profile = profile, capacity, seed_profile
        self.radius, self.maximum_weight = radius, maximum_weight
        self.binding = FrameBinding(self.target)
        self.preparations: list[dict[str, Any]] = []
        self.models: list[dict[str, Any]] = []
        self.seed_instance: ReverseSeedExperiment | None = None
        self.hooks_restored = False

    def prepare(
        self, raw: bytes, *, target_points: int, depth_band: DepthBand | None = None
    ) -> PreparedCloud:
        if (
            _ACTIVE.get() is not self
            or target_points != self.target
            or depth_band != DepthBand(60, 300)
            or len(self.preparations) >= 3000
        ):
            raise RuntimeError("M27 requires its active original-band/capacity preparation scope")
        cloud = ORIGINAL_GRID(raw, target_points=target_points, depth_band=depth_band)
        try:
            offer = self.binding.prepare(raw, cloud.points)
        except (ValueError, ArithmeticError) as error:
            raise RuntimeError("M27 raw preparation provenance failed") from error
        self.preparations.append(
            dict(
                ordinal=offer.ordinal,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                points=len(cloud.points),
                points_sha256=digest(cloud.points),
                status="prepared",
                accepted_index=None,
            )
        )
        return cloud

    def _tracker(self, tracker: IncrementalTracker) -> None:
        if (
            _ACTIVE.get() is not self
            or type(tracker) is not IncrementalTracker
            or tracker._policy != self.policy
            or tracker._backend != "scipy"
            or not tracker._point_to_plane
            or tracker._voxel_size != 1.5
            or tracker._max_correspondence_distance != 12.0
            or tracker._keep_fraction != 0.8
            or tracking.PREFERENCE_ORDER != ("model", "adjacent", "keyframe")
        ):
            raise RuntimeError(
                "M27 requires the exact original tracker, policy and candidate gates"
            )

    def offer(
        self, tracker: IncrementalTracker, cloud: Sequence[Point], **kwargs: Any
    ) -> TrackDecision:
        self._tracker(tracker)
        if kwargs.get("adjacent_fit") is not None or set(kwargs) - {"adjacent_fit"}:
            raise RuntimeError("M27 does not accept injected or unknown offer overrides")
        try:
            self.binding.begin(tracker, len(tracker._accepted), tuple(cloud))
            active = self.binding.active
            assert active is not None
            decision = ORIGINAL_OFFER(tracker, active.points, **kwargs)
            retained = {
                item.index: item.cloud for item in tracker._accepted if item.cloud is not None
            }
            self.binding.finish(len(tracker._accepted), retained)
            self.preparations[active.ordinal].update(
                status="accepted" if decision.accepted else "refused",
                accepted_index=len(tracker._accepted) - 1 if decision.accepted else None,
                retained_indices=list(retained),
                native_fields=len(self.binding.fields),
            )
            return decision
        except (ValueError, ArithmeticError) as error:
            raise RuntimeError("M27 offer/admission provenance failed") from error

    def seed_fit(
        self,
        seed: ReverseSeedExperiment,
        source: Sequence[Point],
        target: Sequence[Point],
        **kwargs: Any,
    ) -> icp_plane.IcpPlaneResult:
        if _ACTIVE.get() is not self or _FIT.get() is not None or seed.profile != self.seed_profile:
            raise RuntimeError("M27 requires one nonnested explicitly bound M21 seed fit")
        if self.seed_instance is not None and seed is not self.seed_instance:
            raise RuntimeError("M27 seed experiment identity changed")
        self.seed_instance = seed
        token = _FIT.set((seed, len(seed.records)))
        try:
            return ORIGINAL_SEED_FIT(seed, source, target, **kwargs)
        finally:
            _FIT.reset(token)

    def fit_role(self) -> str:
        binding = _FIT.get()
        if _ACTIVE.get() is not self or not _MODEL.get() or binding is None:
            raise RuntimeError("M27 normal request lacks a bound model and seed fit")
        seed, ordinal = binding
        if seed is not self.seed_instance or len(seed.records) != ordinal + 1:
            raise RuntimeError("M27 normal request points to a stale seed record")
        record = seed.records[ordinal]
        if record["status"] != "running" or not record["stages"]:
            raise RuntimeError("M27 normal request lacks an active seed stage")
        stage = record["stages"][-1]
        direction = stage["direction"]
        if stage["status"] != "running" or direction not in ("forward", "reverse"):
            raise RuntimeError("M27 normal request has an unknown or completed fit role")
        return str(direction)

    def candidate(
        self, tracker: IncrementalTracker, index: int, source: tuple[Point, ...], **kwargs: Any
    ) -> TrackCandidate | None:
        self._tracker(tracker)
        if kwargs.get("origin") != tracking.MODEL_ORIGIN:
            return ORIGINAL_CANDIDATE(tracker, index, source, **kwargs)
        if (
            _MODEL.get()
            or self.binding.active is None
            or self.binding.tracker is not tracker
            or index != self.binding.accepted_count
            or kwargs.get("into_world") != Rigid.identity()
            or len(self.models) >= 9000
        ):
            raise RuntimeError("M27 model target lacks exact nonnested offered-frame provenance")
        provider = icp_plane._NORMAL_ACCELERATOR.get()
        if provider is None:
            raise RuntimeError("M27 model target requires the original GPU provider")
        try:
            frames = []
            for accepted_index in kwargs["reference_frames"]:
                item = tracker._accepted[accepted_index]
                if item.cloud is None:
                    raise RuntimeError("M27 model references an evicted native cloud")
                native = self.binding.native(accepted_index, self.radius, item.cloud)
                frames.append(ModelFrame(accepted_index, native, item.pose))
            model = ModelGeometry(tuple(frames), tuple(kwargs["reference"]))
            reverse_points = voxel_downsample(source, 1.5)
        except (ValueError, ArithmeticError) as error:
            raise RuntimeError("M27 model reconstruction provenance failed") from error
        record: dict[str, Any] = dict(
            ordinal=len(self.models),
            offer_ordinal=self.binding.active.ordinal,
            accepted_index=index,
            status="running",
            profile=self.profile,
            reference_frames=[
                dict(index=f.index, raw_sha256=f.native.raw_sha256, pose=f.pose.as_dict())
                for f in frames
            ],
            source_sha256=digest(source),
            model_sha256=digest(model.expected),
            centers_sha256=digest(model.centers),
            origins_sha256=digest(
                [(o.frame_index, o.native_index, o.raw_sha256, o.pixel) for o in model.origins]
            ),
            selected_sha256=digest(model.selected),
            coarse_to_fine=kwargs.get("coarse_to_fine", False),
            model_points=len(model.expected),
            voxel_points=len(model.centers),
            queries=[],
        )
        self.models.append(record)
        return self._model_fit(
            tracker, index, source, kwargs, model, reverse_points, provider, record
        )

    def _model_fit(
        self,
        tracker: IncrementalTracker,
        index: int,
        source: tuple[Point, ...],
        kwargs: dict[str, Any],
        model: ModelGeometry,
        reverse_points: tuple[Point, ...],
        provider: Provider,
        record: dict[str, Any],
    ) -> TrackCandidate | None:
        baselines: dict[tuple[tuple[Point, ...], int], dict[str, Any]] = {}

        def baseline(points: tuple[Point, ...], neighbours: int) -> icp_plane.NormalField:
            original = provider(points, neighbours)
            baselines[points, neighbours] = dict(
                sha256=field_digest(original),
                points=len(original.estimates),
                usable=len(original.usable),
            )
            return original

        route = RoleDispatcher(
            model,
            reverse_points,
            baseline,
            self.fit_role,
            maximum_weight=self.maximum_weight,
            observe=self.profile == "observe-pooled",
        )

        def dispatch(points: tuple[Point, ...], neighbours: int) -> icp_plane.NormalField:
            try:
                direction = self.fit_role()
                fit_binding = _FIT.get()
                if fit_binding is None:
                    raise RuntimeError("M27 seed binding disappeared during normal dispatch")
                seed, fit_ordinal = fit_binding
                stage_ordinal = seed.records[fit_ordinal]["stages"][-1]["ordinal"]
                result = route(points, neighbours)
                original = baselines[points, neighbours]
                actual_hash = field_digest(result)
                if (
                    self.profile == "observe-pooled" or direction == "reverse"
                ) and actual_hash != original["sha256"]:
                    raise RuntimeError("M27 changed an observation or reverse normal field")
                if len(result.usable) != original["usable"]:
                    raise RuntimeError("M27 changed normal availability")
                record["queries"].append(
                    dict(
                        direction=direction,
                        seed_fit_ordinal=fit_ordinal,
                        seed_stage_ordinal=stage_ordinal,
                        points=len(points),
                        neighbours=neighbours,
                        target_sha256=digest(points),
                        baseline=original,
                        actual_sha256=actual_hash,
                        usable=len(result.usable),
                    )
                )
                return result
            except (ValueError, ArithmeticError, KeyError) as error:
                # The tracker catches these as ordinary fit refusals. Experimental
                # binding failures must instead stop the diagnostic.
                raise RuntimeError(
                    "M27 normal-provider provenance or field validation failed"
                ) from error

        model_token = _MODEL.set(True)
        normal_token = icp_plane._NORMAL_ACCELERATOR.set(dispatch)
        try:
            result = ORIGINAL_CANDIDATE(tracker, index, source, **kwargs)
            if route.result is not None:
                _, companions = route.result
                weights = [r["confidence"]["weight"] for r in companions]
                record["hybrid"] = dict(
                    companion_sha256=digest(companions),
                    modes=dict(Counter(r["confidence"]["mode"] for r in companions)),
                    weight_min=min(weights),
                    weight_max=max(weights),
                    weight_sum=sum(weights),
                )
            record.update(status="completed", formed=result is not None)
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            mutated = icp_plane._NORMAL_ACCELERATOR.get() is not dispatch
            icp_plane._NORMAL_ACCELERATOR.reset(normal_token)
            _MODEL.reset(model_token)
            if mutated:
                record.update(
                    status="failed", error="M27 model normal hook changed within its scope"
                )
                raise RuntimeError(record["error"])

    @contextmanager
    def active(self) -> Iterator[None]:
        originals = (
            grid.prepare_frame is ORIGINAL_GRID,
            vars(static_replay).get("prepare_grid") is ORIGINAL_GRID,
            IncrementalTracker.offer is ORIGINAL_OFFER,
            IncrementalTracker._candidate is ORIGINAL_CANDIDATE,
            ReverseSeedExperiment.fit is ORIGINAL_SEED_FIT,
        )
        if (
            _ACTIVE.get() is not None
            or _FIT.get() is not None
            or _MODEL.get()
            or not all(originals)
        ):
            raise RuntimeError("M27 requires original hooks and a nonnested offline scope")
        if self.preparations or self.models or self.hooks_restored:
            raise RuntimeError("M27 experiment instances cannot be reused")

        def offer(
            tracker: IncrementalTracker, cloud: Sequence[Point], **kwargs: Any
        ) -> TrackDecision:
            return self.offer(tracker, cloud, **kwargs)

        def candidate(
            tracker: IncrementalTracker, index: int, source: tuple[Point, ...], **kwargs: Any
        ) -> TrackCandidate | None:
            return self.candidate(tracker, index, source, **kwargs)

        def seed_fit(
            seed: ReverseSeedExperiment,
            source: Sequence[Point],
            target: Sequence[Point],
            **kwargs: Any,
        ) -> icp_plane.IcpPlaneResult:
            return self.seed_fit(seed, source, target, **kwargs)

        replacement = self.prepare
        token = _ACTIVE.set(self)
        try:
            with (
                patch.object(grid, "prepare_frame", replacement),
                patch.object(static_replay, "prepare_grid", replacement),
                patch.object(IncrementalTracker, "offer", offer),
                patch.object(IncrementalTracker, "_candidate", candidate),
                patch.object(ReverseSeedExperiment, "fit", seed_fit),
            ):
                yield
                self.binding.complete()
                if (
                    grid.prepare_frame is not replacement
                    or vars(static_replay).get("prepare_grid") is not replacement
                    or vars(IncrementalTracker).get("offer") is not offer
                    or vars(IncrementalTracker).get("_candidate") is not candidate
                    or vars(ReverseSeedExperiment).get("fit") is not seed_fit
                ):
                    raise RuntimeError("M27 scoped hooks changed during execution")
        finally:
            _ACTIVE.reset(token)
            self.hooks_restored = (
                grid.prepare_frame is ORIGINAL_GRID
                and vars(static_replay).get("prepare_grid") is ORIGINAL_GRID
                and IncrementalTracker.offer is ORIGINAL_OFFER
                and IncrementalTracker._candidate is ORIGINAL_CANDIDATE
                and ReverseSeedExperiment.fit is ORIGINAL_SEED_FIT
                and _FIT.get() is None
                and not _MODEL.get()
            )
