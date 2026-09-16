"""Offline motion/geometry stress on saved real frames, never device access.

Temporal stride increases between-view displacement, not physical acquisition
speed or within-pattern motion. No live mesh, fitted turntable prior or relaxed
tracking gate is introduced. Closed alignment/reference ablations are recorded
explicitly and never promoted to the application by this probe.
Explicit diagnostic-scene replay may inspect flagged framing, but never
qualifies that scene; source warnings are carried into every output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from scanner3d import tracking as tracking_module
from scanner3d.capture_sequence import TRACKING_LOSS_STREAK
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.coherence import check_coherence
from scanner3d.geometry import Point, Rigid
from scanner3d.gpu_normals import live_normals
from scanner3d.icp_plane import estimate_normals, normal_acceleration
from scanner3d.preparation import DepthBand, ensure_depth_band, prepare_frame
from scanner3d.relocalize import LocalFit
from scanner3d.return_health import ReturnHealth, ReturnHealthTracker
from scanner3d.simulate import FACTORY_INTRINSICS
from scanner3d.tracking import (
    DEFAULT_POLICY,
    IncrementalTracker,
    KeyframeDecision,
    TrackDecision,
    TrackingPolicy,
)

REFERENCE_PROFILES = (
    "default",
    "retention64",
    "no-model",
    "stable64",
    "keyframe64",
    "untrimmed",
    "point-to-point",
    "return-health",
    "pose-settling",
    "pixel-rank",
    "pixel-grid",
    "native-model-normals",
)


class ReturnHealthReferenceTracker(IncrementalTracker):
    """Offline experiment: warning clouds are adjacent-only, never archival.

    Uses the existing causal raw-yield warning, not a new alignment gate.
    Every frame is offered and every accepted pose/velocity is retained.
    A warning cloud stays available for the immediately next adjacent solve,
    then is released once superseded. Ordinary model and recovery selection
    already exclude the previous accepted frame and released clouds, so neither
    can accidentally reuse it as independent evidence. The initial anchor is
    explicitly exempt; this experiment does not invent a different origin.

    No GUI/capture code imports or enables this diagnostic subclass.
    The September 8 experiment regressed the contrasting handheld recording;
    retained for reproducibility, not recommended for live tracking.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._current_health: ReturnHealth | None = None
        self._withheld_clouds: set[int] = set()

    def offer(
        self,
        cloud: Sequence[Point],
        *,
        adjacent_fit: LocalFit | None = None,
        return_health: ReturnHealth | None = None,
    ) -> TrackDecision:
        if return_health is None:
            raise ValueError("Return-health diagnostic requires this offer's raw health")
        self._current_health = return_health
        try:
            return super().offer(cloud, adjacent_fit=adjacent_fit)
        finally:
            self._current_health = None

    def seed(self, cloud: Sequence[Point], pose: Rigid, *, anchor: bool = True) -> TrackDecision:
        raise ValueError("Return-health diagnostic does not support seeded bridges")

    def _update_keyframes(
        self, index: int, pose: Rigid, *, overlap: float, anchor: bool
    ) -> list[KeyframeDecision]:
        if self._current_health is None:
            raise ValueError("Return-health diagnostic only supports health-labelled offers")
        if not anchor and self._current_health.is_warning:
            self._withheld_clouds.add(index)
            return [
                KeyframeDecision(
                    frame=index,
                    action="retain",
                    reason="Offline return-health experiment: leave the keyframe archive "
                    "unchanged; this warning cloud is adjacent-only and is released when "
                    "superseded. Accepted pose and all alignment gates are unchanged.",
                )
            ]
        return super()._update_keyframes(index, pose, overlap=overlap, anchor=anchor)

    def _release_clouds(self) -> None:
        super()._release_clouds()
        previous_index = self._accepted[-1].index
        for index in self._withheld_clouds:
            if index != previous_index:
                self._accepted[index].cloud = None
        # At most one flag survives; no extra per-frame archive grows here.
        self._withheld_clouds.intersection_update({previous_index})


class StableReferenceTracker(IncrementalTracker):
    """Diagnostic archive: discard the newer of the closest unordered pose pair.

    Recent clouds remain in the separate bounded recency buffer and continue
    supplying adjacent/model candidates. No first-turn labels, image matches,
    fitted axis or revisit corrections enter tracking. Not a production mode.
    """

    def _evict(self) -> list[KeyframeDecision]:
        if self.policy.protected_recent_keyframes != 0:
            raise ValueError("Stable-reference diagnostic requires a separate recent buffer")
        decisions = []
        while len(self._keyframes) > self.policy.max_keyframes:
            # Evaluate each pair in one canonical direction. Computing both
            # directions and relying on exact floating-point equality can make
            # the supposed age tie-break depend on roundoff instead of age.
            _, negative_victim = min(
                (
                    self._pose_distance(self._pose_of(older), self._pose_of(newer)),
                    -newer,
                )
                for older, newer in combinations(self._keyframes, 2)
            )
            victim = -negative_victim
            self._keyframes.remove(victim)
            decisions.append(
                KeyframeDecision(
                    frame=victim,
                    action="evict",
                    reason="Diagnostic stable archive: discard the newer member of the "
                    "closest unordered pose pair; current clouds remain in the recent buffer.",
                )
            )
        return decisions


def reference_policy(profile: str) -> TrackingPolicy:
    """Closed, offline-only ablations: no quality/acceptance gate changes."""
    if profile in (
        "default",
        "untrimmed",
        "point-to-point",
        "return-health",
        "pose-settling",
        "pixel-rank",
        "pixel-grid",
        "native-model-normals",
    ):
        return DEFAULT_POLICY
    if profile in ("retention64", "keyframe64"):
        return replace(DEFAULT_POLICY, max_keyframes=64, max_retained_clouds=96)
    if profile == "no-model":
        return replace(DEFAULT_POLICY, use_model=False)
    if profile == "stable64":
        return replace(
            DEFAULT_POLICY,
            max_keyframes=64,
            max_retained_clouds=96,
            protected_recent_keyframes=0,
        )
    raise ValueError("Unknown diagnostic reference profile")


def load_inputs(
    capture: Path,
    count: int,
    skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
    *,
    diagnostic_scene: bool = False,
    start_offset: int = 0,
) -> tuple[list[Path], list[bytes], DepthBand, list[dict[str, Any]]]:
    if not 100 <= count <= 3000 or not 0 <= skip_complete <= 40:
        raise ValueError("Bounded frame count and 0..40 complete settling frames required")
    if not 0 <= start_offset <= 3000 or (start_offset and not diagnostic_scene):
        raise ValueError("A bounded starting-view offset requires diagnostic-scene mode")
    start = skip_complete + start_offset
    paths = sorted(capture.glob("frame_*.raw"))[start : start + count]
    if len(paths) != count or any(p.is_symlink() for p in paths):
        raise ValueError("Insufficient or linked source frames")
    progress_path = capture / "capture-progress.json"
    progress = None
    if progress_path.exists():
        parent = json.loads((capture.parent / "report.json").read_text())
        progress = json.loads(progress_path.read_text())
        validity_path = capture.parent / "quality-validity.json"
        if (
            validity_path.exists()
            and json.loads(validity_path.read_text()).get("quality_qualification_allowed")
            is not True
            and not diagnostic_scene
        ):
            raise ValueError("Raw motion source has an unresolved scene-validity warning")
        if (
            parent.get("status") != "completed"
            or parent.get("restoration_verified") is not True
            or progress.get("capture_error") is not None
        ):
            raise ValueError("Raw motion source must complete and verify restoration")
        if depth_band is None:
            raise ValueError("Raw motion replay requires an explicit common depth band")
        band = ensure_depth_band(DepthBand(*depth_band))
        recorded = progress["frames"]
        stamps = [r["monotonic_s"] for r in recorded]
        if (
            len({r["path"] for r in recorded}) != len(recorded)
            or not all(math.isfinite(t) for t in stamps)
            or any(b <= a for a, b in pairwise(stamps))
        ):
            raise ValueError("Unique raw names and strictly increasing finite times required")
        if skip_complete < 12 or stamps[skip_complete] - stamps[0] < 1.0:
            raise ValueError("Raw motion needs at least 12 frames and one second settling")
        evidence = {r["path"]: r for r in recorded}
    else:
        if depth_band is not None or skip_complete or diagnostic_scene:
            raise ValueError("Legacy project replay keeps its recorded band and full prefix")
        metadata = json.loads((capture / "scan.scanner3d").read_text())
        band = ensure_depth_band(DepthBand(**metadata["capture"]["depth_band"]))
        evidence = {}
    raws = [path.read_bytes() for path in paths]
    if any(len(raw) != 512000 for raw in raws):
        raise ValueError("Incomplete source frame")
    inputs = []
    for path, raw in zip(paths, raws, strict=True):
        digest = hashlib.sha256(raw).hexdigest()
        row: dict[str, Any] = dict(path=str(path), sha256=digest)
        if progress is not None:
            saved = evidence.get(path.name)
            if saved is None or saved["sha256"] != digest or saved["bytes"] != len(raw):
                raise ValueError("Raw motion source hash/length disagrees with capture evidence")
            row.update(monotonic_s=saved["monotonic_s"], delivery_index=saved["delivery_index"])
        inputs.append(row)
    return paths, raws, band, inputs


def source_tracking_loss(capture: Path) -> dict[str, Any] | None:
    """Preserve a GUI source's terminal-loss provenance even in default replay.

    A diagnostic may examine the prefix, but must never turn a failed source
    into a quality-qualified scan merely because its own process completed.
    Hash both evidence files where present; inconsistent markers remain flagged.
    """
    project_path = capture / "scan.scanner3d"
    loss_path = capture / "tracking-loss.json"
    evidence = {}
    metadata = None
    loss = None
    for path in (project_path, loss_path):
        if path.is_symlink():
            raise ValueError("Linked tracking provenance is not supported")
        if not path.exists():
            continue
        raw = path.read_bytes()
        evidence[path.name] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        if path == project_path:
            metadata = json.loads(raw)
        else:
            loss = json.loads(raw)
    marked = bool(metadata and metadata.get("capture", {}).get("tracking_lost"))
    if marked or loss_path.exists():
        return dict(project_tracking_lost=marked, tracking_loss=loss, evidence=evidence)
    return None


def run(
    capture: Path,
    output: Path,
    count: int,
    strides: tuple[int, ...],
    *,
    skip_complete: int = 0,
    depth_band: tuple[float, float] | None = None,
    diagnostic_scene: bool = False,
    start_offset: int = 0,
    normals_backend: str = "gpu",
    diagnostic_reference_profile: str = "default",
    diagnostic_factors: tuple[str, str, str] | None = None,
    diagnostic_solver_profile: str | None = None,
    diagnostic_model_profile: str | None = None,
    diagnostic_source_profile: str | None = None,
) -> None:
    if not 100 <= count <= 3000 or not strides or any(s not in (1, 2, 3, 4) for s in strides):
        raise ValueError("Bounded 100..3000 frames and strides 1..4 required")
    if normals_backend not in ("cpu", "gpu"):
        raise ValueError("Normals backend must be cpu or gpu")
    factors = None
    if diagnostic_factors is not None:
        from scanner_factorial_config import Factors

        if diagnostic_reference_profile != "default":
            raise ValueError("Explicit factors cannot be mixed with a legacy diagnostic profile")
        factors = Factors(*diagnostic_factors)
    preparation_target_points = 900
    policy = reference_policy(diagnostic_reference_profile)
    if diagnostic_source_profile is not None:
        from scanner_source_capacity import settings

        if factors is None or strides != (1,) or normals_backend != "gpu":
            raise ValueError("M7 requires explicit factors, GPU and stride one")
        preparation_target_points, policy = settings(
            diagnostic_source_profile, factors, diagnostic_solver_profile, diagnostic_model_profile
        )
    solver = None
    if diagnostic_model_profile is not None:
        from scanner_model_representation import PROFILES as MODEL_PROFILES

        if (
            diagnostic_model_profile not in MODEL_PROFILES
            or diagnostic_solver_profile not in (None, "observe-origin", "centered")
            or factors is None
            or strides != (1,)
            or normals_backend != "gpu"
        ):
            raise ValueError("M4 requires explicit factors, GPU, stride one and closed profiles")
    if diagnostic_solver_profile is not None:
        from scanner_solver_registry import create_experiment

        if factors is None or strides != (1,):
            raise ValueError("Solver mechanisms require explicit factors and stride one")
        solver = create_experiment(diagnostic_solver_profile)
    prepare = prepare_frame
    if diagnostic_reference_profile == "pixel-rank" or (
        factors is not None and factors.sampling == "rank"
    ):
        from scanner_pixel_rank import prepare_frame as prepare_ranked

        prepare = prepare_ranked
    elif diagnostic_reference_profile == "pixel-grid" or (
        factors is not None and factors.sampling == "grid"
    ):
        from scanner_pixel_grid import prepare_frame as prepare_grid

        prepare = prepare_grid
    point_to_plane = diagnostic_reference_profile != "point-to-point"
    keep_fraction = 1.0 if diagnostic_reference_profile == "untrimmed" else 0.8
    if not point_to_plane and normals_backend != "cpu":
        raise ValueError("Point-to-point diagnostic requires cpu; no GPU normals are used")
    preference = (
        (
            tracking_module.KEYFRAME_ORIGIN,
            tracking_module.MODEL_ORIGIN,
            tracking_module.ADJACENT_ORIGIN,
        )
        if diagnostic_reference_profile == "keyframe64"
        else tracking_module.PREFERENCE_ORDER
    )
    _, raws, band, inputs = load_inputs(
        capture,
        count,
        skip_complete,
        depth_band,
        diagnostic_scene=diagnostic_scene,
        start_offset=start_offset,
    )
    validity_path = capture.parent / "quality-validity.json"
    source_loss = source_tracking_loss(capture)
    qualification = dict(
        diagnostic_scene=diagnostic_scene,
        start_offset=start_offset,
        quality_qualification_allowed=(
            not diagnostic_scene
            and diagnostic_reference_profile == "default"
            and factors is None
            and source_loss is None
        ),
        source_tracking_loss=source_loss,
        reference_health_rule=(
            "existing_causal_raw_yield_warning_adjacent_only_anchor_exempt"
            if diagnostic_reference_profile == "return-health"
            else "advisory_only_no_reference_effect"
        ),
        normals_backend=normals_backend,
        model_normal_rule=(
            factors.model_normal_rule
            if factors is not None
            else "native_frame_pca_nearest_voxel_contributor"
            if diagnostic_reference_profile == "native-model-normals"
            else "pooled_target_pca"
        ),
        diagnostic_reference_profile=diagnostic_reference_profile,
        diagnostic_factors=None if factors is None else factors.as_dict(),
        diagnostic_solver_profile=diagnostic_solver_profile,
        diagnostic_model_profile=diagnostic_model_profile,
        diagnostic_source_profile=diagnostic_source_profile,
        preparation_target_points=preparation_target_points,
        solver_records=None if solver is None else solver.records,
        sampling_rule=(
            factors.sampling_rule
            if factors is not None
            else "fixed_pixel_grid_same_count"
            if diagnostic_reference_profile == "pixel-grid"
            else "fixed_pixel_rank_same_count"
            if diagnostic_reference_profile == "pixel-rank"
            else "production_compacted_stride"
        ),
        tracking_policy=policy.as_dict(),
        alignment_method="point_to_plane" if point_to_plane else "point_to_point",
        alignment_keep_fraction=keep_fraction,
        convergence_rule=(
            factors.convergence_rule
            if factors is not None
            else "residual_and_surface_motion"
            if diagnostic_reference_profile == "pose-settling"
            else "residual_change"
        ),
        candidate_preference=list(preference),
        keyframe_eviction_rule=(
            "closest_unordered_pair_discard_newer"
            if diagnostic_reference_profile == "stable64"
            else "production_redundancy_with_protected_tail"
        ),
        source_scene_validity=(
            json.loads(validity_path.read_text()) if validity_path.exists() else None
        ),
    )
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        scope=__doc__,
        capture=str(capture),
        depth_band=band.as_dict(),
        skipped_first_complete_frames=skip_complete,
        input_frames=inputs,
        runs=[],
        status="running",
        **qualification,
    )
    started = time.monotonic()
    active_run: dict[str, Any] | None = None
    try:
        for stride in strides:
            tracker_type = (
                StableReferenceTracker
                if diagnostic_reference_profile == "stable64"
                else ReturnHealthReferenceTracker
                if diagnostic_reference_profile == "return-health"
                else IncrementalTracker
            )
            if diagnostic_reference_profile == "native-model-normals" or (
                factors is not None and factors.model_normals == "native"
            ):
                from scanner_model_normals import NativeModelTracker

                tracker_type = NativeModelTracker
            tracker: IncrementalTracker
            if diagnostic_model_profile is not None:
                from scanner_model_representation import ModelRepresentationTracker

                assert factors is not None
                tracker = ModelRepresentationTracker(
                    backend="scipy",
                    policy=policy,
                    point_to_plane=point_to_plane,
                    keep_fraction=keep_fraction,
                    model_profile=diagnostic_model_profile,
                    normal_mode=factors.model_normals,
                )
            else:
                tracker = tracker_type(
                    backend="scipy",
                    policy=policy,
                    point_to_plane=point_to_plane,
                    keep_fraction=keep_fraction,
                )
            decisions: list[dict[str, Any]] = []
            accepted_raws: list[bytes] = []
            accepted_indices: list[int] = []
            poses: list[Rigid] = []
            health_tracker = ReturnHealthTracker()
            active_run = dict(
                stride=stride,
                stage="tracking",
                decisions=decisions,
                accepted_source_indices=accepted_indices,
                poses=poses,
                model_normal_records=getattr(tracker, "model_normal_records", None),
                model_representation_records=getattr(tracker, "model_representation_records", None),
            )
            streak = 0
            messages: list[str] = []
            if not point_to_plane:
                messages.append(
                    "CPU point-to-point alignment; no normal estimation or GPU dispatch."
                )
            elif normals_backend == "cpu":
                messages.append(
                    "CPU reference normals: SciPy correspondence search; no GPU dispatch."
                )
            normal_context = (
                nullcontext()
                if not point_to_plane
                else live_normals(None, messages.append)
                if normals_backend == "gpu"
                else normal_acceleration(
                    lambda points, neighbours: estimate_normals(
                        points, neighbours=neighbours, backend="scipy"
                    )
                )
            )
            # This is a process-local offline probe, not a live policy change.
            # Keep all candidate construction, gates and arbitration intact;
            # alter only the final preference within the agreed cluster.
            stopping_context: AbstractContextManager[None] = nullcontext()
            if diagnostic_reference_profile == "pose-settling" or (
                factors is not None and factors.convergence == "settling"
            ):
                from scanner_pose_settling import pose_settling

                stopping_context = pose_settling()
            with (
                patch.object(tracking_module, "PREFERENCE_ORDER", preference),
                normal_context,
                stopping_context,
                nullcontext() if solver is None else solver.active(),
            ):
                if normals_backend == "gpu" and not any(
                    m.startswith("GPU-assisted tracking:") for m in messages
                ):
                    raise RuntimeError(f"GPU not active: {messages}")
                for index in range(0, count, stride):
                    active_run["source_index_in_progress"] = index
                    if time.monotonic() - started > 900:
                        raise TimeoutError("900-second offline bound reached")
                    tick = time.perf_counter()
                    cloud = prepare(
                        raws[index], target_points=preparation_target_points, depth_band=band
                    )
                    # deprojected_points counts finite nonzero raw samples, before
                    # banding/thinning/support filtering; the calibrated fixed
                    # scale/intrinsics are validated by prepare_frame itself.
                    raw_yield = cloud.deprojected_points / (len(raws[index]) // 2)
                    health = health_tracker.update(raw_yield)
                    preparation_s = time.perf_counter() - tick
                    tick = time.perf_counter()
                    if solver is not None:
                        solver.mark_frame(index)
                    decision = (
                        cast(ReturnHealthReferenceTracker, tracker).offer(
                            cloud.points, return_health=health
                        )
                        if diagnostic_reference_profile == "return-health"
                        else tracker.offer(cloud.points)
                    )
                    decisions.append(
                        dict(
                            source_index=index,
                            decision=decision.as_dict(),
                            preparation_s=preparation_s,
                            tracking_s=time.perf_counter() - tick,
                            prepared_points=len(cloud.points),
                            deprojected_points=cloud.deprojected_points,
                            clipped_points=cloud.clipped_points,
                            raw_return_health=health.as_dict(),
                            in_band_yield=cloud.clipped_points / (len(raws[index]) // 2),
                        )
                    )
                    if decision.accepted:
                        streak = 0
                        accepted_raws.append(raws[index])
                        accepted_indices.append(index)
                        poses.append(tracker.poses[-1])
                    else:
                        streak += 1
                    if len(decisions) % 100 == 0:
                        print("REPLAY", stride, len(decisions), "accepted", len(poses), flush=True)
                    if streak >= TRACKING_LOSS_STREAK:
                        break
            active_run["stage"] = "coherence"
            coherence = check_coherence(
                accepted_raws,
                poses,
                intrinsics=FACTORY_INTRINSICS,
                depth_scale=DEPTH_SCALE_METRES,
                depth_band=band,
            ).as_dict()
            result = dict(
                stride=stride,
                offered=len(decisions),
                accepted=len(poses),
                tracking_stopped=streak >= TRACKING_LOSS_STREAK,
                decisions=decisions,
                accepted_source_indices=accepted_indices,
                poses=[dict(rotation=p.rotation, translation=p.translation) for p in poses],
                coherence=coherence,
                messages=messages,
                model_normal_records=getattr(tracker, "model_normal_records", None),
                model_representation_records=getattr(tracker, "model_representation_records", None),
                **qualification,
            )
            report["runs"].append(result)
            with (output / f"stride-{stride}.json").open("x") as stream:
                json.dump(result, stream, indent=2)
            print("COHERENCE", stride, json.dumps(coherence), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if active_run is not None:
            report["failed_run"] = {
                **active_run,
                "offered_before_error": len(active_run["decisions"]),
                "accepted_before_error": len(active_run["poses"]),
                "poses": [
                    dict(rotation=p.rotation, translation=p.translation)
                    for p in active_run["poses"]
                ],
                **qualification,
            }
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--skip-complete", type=int, default=0)
    parser.add_argument("--depth-band", type=float, nargs=2)
    parser.add_argument("--normals-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--diagnostic-reference-profile",
        choices=REFERENCE_PROFILES,
        default="default",
        help="Closed offline alignment/reference ablation; non-default never qualifies a scan",
    )
    parser.add_argument(
        "--diagnostic-scene",
        action="store_true",
        help="Inspect flagged raw-scene geometry only; always excludes quality qualification",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Diagnostic-only additional complete-frame offset after declared settling",
    )
    args = parser.parse_args()
    run(
        args.capture,
        args.output,
        args.frames,
        tuple(args.strides),
        skip_complete=args.skip_complete,
        depth_band=None if args.depth_band is None else tuple(args.depth_band),
        diagnostic_scene=args.diagnostic_scene,
        start_offset=args.start_offset,
        normals_backend=args.normals_backend,
        diagnostic_reference_profile=args.diagnostic_reference_profile,
    )
