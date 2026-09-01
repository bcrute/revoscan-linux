"""Measure what :mod:`scanner3d.tsdf` costs, on synthetic sweeps and on one real capture.

Deliberately NOT a ``test_`` module: pytest does not collect it. What it
measures - wall clock, resident memory, update counts - is a property of the
machine it runs on, so asserting bounds on it in the suite would produce a test
that fails on a slower host and tells nobody anything. It is a harness that is
run when a number is wanted, and its output is pasted into the handoff that
wanted it.

Run it with::

    .venv/bin/python -m tests.tsdf_benchmark
    .venv/bin/python -m tests.tsdf_benchmark --capture /path/to/a/capture

Every measurement is given the exact camera it is to project with, as a
:class:`FusionCamera`; nothing below reads a module-level record of its own.
That is a correction, not decoration: this harness used to deproject a real
capture for registration through the device's factory record while fusing it
through the synthetic one, so its physical counts described a camera the frames
were never taken with. The real-capture path now hands ONE instance to both, and
each measurement reports the record it used so the JSON is auditable without
reading this source.

The capture is read and never written. Nothing here opens a device node,
touches USB, starts a capture or writes inside the capture directory.

:func:`surface_sweep` measures the other half of the pipeline that consumes a
fused field: :func:`scanner3d.tsdf_surface.extract_surface`. It always runs on
a field this harness has already fused, through the synthetic camera the
sphere was rendered with, and its timing wraps the extraction call alone -
fusion is measured separately by :func:`synthetic_sweep` and is never
double-counted here.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.simulate import FACTORY_INTRINSICS, Sphere, intrinsics_of, render
from scanner3d.tsdf import TsdfConfig, TsdfVolume, VolumeBounds, VolumeTooLarge
from scanner3d.tsdf_surface import SurfaceEvidence, TsdfSurface, extract_surface

#: The same centred synthetic camera the tests use.
INTRINSICS = intrinsics_of(fx=400.0, fy=400.0, cx=320.0, cy=200.0)

#: The unit's measured depth scale, whose physical unit is unestablished.
DEPTH_SCALE = 0.004999999888241291


@dataclass(frozen=True, slots=True)
class FusionCamera:
    """The one projection a measurement consumes, carried as a single value.

    Intrinsics and depth scale travel together because they are one statement
    about how a frame's uint16 codes become metric camera-frame points, and
    because a harness that reaches for either of them separately is a harness
    that can pick up the wrong one. Every function below that projects takes
    this and reads no module-level record, so a caller that renders, registers
    or fuses through a particular camera cannot silently get a different one.
    """

    label: str
    intrinsics: IntrinsicsRecord
    depth_scale: float

    def as_dict(self) -> dict[str, object]:
        """The exact record projected with, for the report to carry.

        Reported from the record itself rather than from the label, so the JSON
        answers "which camera was this?" on its own and a reader never has to
        infer it from which constant a source line happened to name.
        """
        values = self.intrinsics.values
        return {
            "label": self.label,
            "fx": values["fx"],
            "fy": values["fy"],
            "cx": values["cx"],
            "cy": values["cy"],
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
            "depth_scale": self.depth_scale,
        }


#: The synthetic sweep's camera. Here the synthetic record IS the camera: the
#: frames are rendered through it, so fusing through it is the truthful thing.
SYNTHETIC_CAMERA: Final = FusionCamera(
    label="synthetic centred pinhole, as rendered",
    intrinsics=INTRINSICS,
    depth_scale=DEPTH_SCALE,
)

#: A real capture's camera: the unit's own factory record, which is the only
#: record its frames were ever taken with. Registration and fusion both take
#: THIS object on the real path, so neither can quietly use the other's.
CAPTURE_CAMERA: Final = FusionCamera(
    label="MINI factory intrinsics record, as read from the device",
    intrinsics=FACTORY_INTRINSICS,
    depth_scale=DEPTH_SCALE,
)

SPHERE_CENTRE: Point = (0.0, 0.0, 100.0)
SPHERE_RADIUS = 20.0
ORBIT_CENTRE: Point = (0.0, 0.0, 80.0)

#: What the capture's own CLI ``reconstruct`` command applies before registering,
#: repeated here so the poses this harness fuses with are the poses that command
#: would have produced rather than a differently prepared set.
NEAR_CLIP = 60.0
FAR_CLIP = 300.0
REGISTRATION_POINTS = 2200

#: Voxel sizes tried, finest first, for a real capture whose extent is not
#: known until its poses are. The truncation is three voxels on every rung.
VOXEL_LADDER = (0.5, 1.0, 2.0, 4.0, 8.0)


def peak_rss_bytes() -> int:
    """Peak resident set size of this process so far, in bytes.

    ``ru_maxrss`` is a high-water mark and never falls, so a delta across one
    measured section is a lower bound on what that section needed and not an
    exact figure. It is reported alongside the volume's own exactly-known
    ``owned_bytes`` for that reason.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def synthetic_views(
    views: int, *, jitter: float, camera: FusionCamera
) -> tuple[list[bytes], list[Rigid]]:
    scene = [
        rotation_about_axis((0.0, 1.0, 0.0), 2.0 * math.pi * index / views, through=ORBIT_CENTRE)
        for index in range(views)
    ]
    frames = [
        render(
            [Sphere(centre=SPHERE_CENTRE, radius=SPHERE_RADIUS)],
            intrinsics=camera.intrinsics,
            depth_scale=camera.depth_scale,
            pose=pose,
            jitter=jitter,
            seed=index,
        ).raw
        for index, pose in enumerate(scene)
    ]
    return frames, [pose.inverse() for pose in scene]


def radial_error(points: Sequence[Point]) -> dict[str, float | int]:
    """Distance from the known sphere. Synthetic only: this truth really is exact."""
    if not points:
        return {"points": 0}
    errors = [abs(math.dist(point, SPHERE_CENTRE) - SPHERE_RADIUS) for point in points]
    ordered = sorted(errors)
    return {
        "points": len(points),
        "median": ordered[len(ordered) // 2],
        "rms": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "max": ordered[-1],
    }


def measure(
    frames: Sequence[bytes],
    poses: Sequence[Rigid],
    config: TsdfConfig,
    *,
    backend: str,
    truth: bool,
    camera: FusionCamera,
) -> dict[str, object]:
    """One fusion, timed and accounted, through the camera the caller names.

    ``camera`` has no default on purpose. A default here is what let a real
    capture be registered through the factory record and fused through the
    synthetic one: the caller had said which camera it meant, and this function
    quietly did not use it.
    """
    before = peak_rss_bytes()
    volume = TsdfVolume(config, backend=backend)
    started = time.perf_counter()
    volume.integrate(
        list(frames),
        list(poses),
        intrinsics=camera.intrinsics,
        depth_scale=camera.depth_scale,
    )
    elapsed = time.perf_counter() - started
    evidence = volume.evidence(elapsed_seconds=elapsed)

    zero_started = time.perf_counter()
    samples = volume.zero_level_samples()
    zero_elapsed = time.perf_counter() - zero_started
    after = peak_rss_bytes()

    row: dict[str, object] = {
        "backend": evidence.backend,
        "fusion_camera": camera.as_dict(),
        "voxel_size": config.voxel_size,
        "shape": list(config.shape),
        "cells": config.cell_count,
        "owned_bytes": config.owned_bytes,
        "peak_rss_delta_bytes": max(0, after - before),
        "frames": evidence.frames_offered,
        "integrate_seconds": elapsed,
        "seconds_per_frame": elapsed / max(1, evidence.frames_offered),
        "voxel_updates": evidence.voxel_updates,
        "voxel_observations": evidence.voxel_observations,
        "observed_voxels": evidence.weights.observed_voxels,
        "weight_min": evidence.weights.minimum,
        "weight_max": evidence.weights.maximum,
        "zero_samples": samples.point_count,
        "zero_level_seconds": zero_elapsed,
        "ignored": {str(entry.reason): entry.count for entry in evidence.ignored},
    }
    if truth:
        row["radial_error"] = radial_error(samples.points)
    return row


def synthetic_sweep() -> list[dict[str, object]]:
    """A representative sweep: voxel size, frame count and backend, each varied."""
    rows: list[dict[str, object]] = []
    camera = SYNTHETIC_CAMERA
    frames, poses = synthetic_views(16, jitter=0.5, camera=camera)
    bounds = VolumeBounds.of((-25.0, -25.0, 75.0), (25.0, 25.0, 125.0))

    for voxel, truncation in ((2.0, 6.0), (1.0, 3.0), (0.5, 1.5), (0.25, 0.75)):
        config = TsdfConfig.of(
            bounds=bounds,
            voxel_size=voxel,
            truncation=truncation,
            min_depth=30.0,
            max_depth=200.0,
        )
        rows.append(
            {"case": f"voxel {voxel}, 16 views, numpy"}
            | measure(frames, poses, config, backend="numpy", truth=True, camera=camera)
        )

    config = TsdfConfig.of(
        bounds=bounds, voxel_size=1.0, truncation=3.0, min_depth=30.0, max_depth=200.0
    )
    for count in (1, 4, 8, 16):
        rows.append(
            {"case": f"voxel 1.0, {count} views, numpy"}
            | measure(
                frames[:count], poses[:count], config, backend="numpy", truth=True, camera=camera
            )
        )

    # The pure-Python reference, at a volume its own ceiling allows.
    coarse = TsdfConfig.of(
        bounds=VolumeBounds.of((-24.0, -24.0, 76.0), (24.0, 24.0, 124.0)),
        voxel_size=3.0,
        truncation=6.0,
        min_depth=30.0,
        max_depth=200.0,
    )
    for backend in ("numpy", "python"):
        rows.append(
            {"case": f"voxel 3.0, 8 views, {backend}"}
            | measure(frames[:8], poses[:8], coarse, backend=backend, truth=True, camera=camera)
        )
    return rows


#: The saved capture's own field, as recorded by the finest VOXEL_LADDER rung
#: `.coordination/coder/latest-report.md` reports for
#: ``/home/ben/scans/scan/20260827-152842``: voxel 1.0, truncation 3.0,
#: shape 174 x 141 x 158. Named here, not re-derived, because deriving it
#: again would mean re-registering that capture, which this task did not
#: authorise.
SAVED_CAPTURE_SHAPE: Final = (174, 141, 158)
SAVED_CAPTURE_VOXEL_SIZE = 1.0
SAVED_CAPTURE_TRUNCATION = 3.0


def _measure_extraction(
    volume: TsdfVolume, *, backend: str, min_weight: float = 1.0
) -> tuple[TsdfSurface, float, int]:
    """One surface extraction off an already-fused field, timed and RSS-accounted alone.

    Fusion happened before this is called and is deliberately excluded from
    both the clock and the RSS delta: mixing the two costs into one number
    would make it impossible to tell how much a case costs to fuse and how
    much it costs to walk afterwards, which is exactly the split
    :func:`synthetic_sweep` and this section exist to keep apart.
    """
    before = peak_rss_bytes()
    started = time.perf_counter()
    surface = extract_surface(volume, min_weight=min_weight, backend=backend)
    elapsed = time.perf_counter() - started
    after = peak_rss_bytes()
    return surface, elapsed, max(0, after - before)


def _surface_row(
    case: str,
    config: TsdfConfig,
    surface: TsdfSurface,
    *,
    elapsed: float,
    rss_delta: int,
) -> dict[str, object]:
    evidence = surface.evidence
    extent = surface.mesh.bounds
    return {
        "case": case,
        "estimate": False,
        "backend": evidence.backend,
        "voxel_size": config.voxel_size,
        "shape": list(config.shape),
        "cells": config.cell_count,
        "owned_bytes": config.owned_bytes,
        "extract_seconds": elapsed,
        "peak_rss_delta_bytes": rss_delta,
        "cubes": {
            "examined": evidence.cubes_examined,
            "eligible": evidence.cubes_eligible,
            "excluded_insufficient_weight": evidence.cubes_excluded_insufficient_weight,
            "active": evidence.cubes_active,
        },
        "vertices": evidence.vertex_count,
        "faces": evidence.face_count,
        "edges": evidence.census.as_dict(),
        "bounds": None if extent is None else {"min": list(extent[0]), "max": list(extent[1])},
        "radial_error": radial_error(surface.mesh.vertices),
    }


def saved_capture_estimate(
    config: TsdfConfig, evidence: SurfaceEvidence, extract_seconds: float
) -> dict[str, object]:
    """What is KNOWN and what is merely estimated about the saved capture's field.

    Re-running physical registration on ``/home/ben/scans/scan/20260827-152842``
    was not authorised for this task, so no surface was ever extracted from that
    capture's own field. Exactly two kinds of number can honestly appear here,
    and they are kept apart:

    * ``cubes.examined`` is EXACT. How many cubes a walk visits is decided by
      the field's shape alone - ``(nx - 1)(ny - 1)(nz - 1)`` - and that shape is
      recorded, so this is arithmetic on a known quantity rather than a
      prediction.
    * ``extract_seconds_estimate`` is a ROUGH cell-scaled estimate: the walk's
      cost per cell is what the synthetic rows measure, and the saved field has
      a known number of cells. It is still only an estimate, because the per-cell
      cost of a real field depends on how much of it clears the weight gate.

    Everything else is left unknown. Eligible, excluded-for-insufficient-weight
    and active cube counts are decided by REAL OBSERVATION COVERAGE and by the
    area over which the field crosses zero - exactly as vertex count, face count
    and the edge census are - and a synthetic sphere fused from 16 clean
    simulated views predicts none of that for a chassis scanned by a moving
    camera. Scaling them by a cell ratio would dress a number with no basis as a
    measurement, so they are ``None`` until a real field exists to measure.
    """
    capture_cells = SAVED_CAPTURE_SHAPE[0] * SAVED_CAPTURE_SHAPE[1] * SAVED_CAPTURE_SHAPE[2]
    capture_cubes = (
        (SAVED_CAPTURE_SHAPE[0] - 1) * (SAVED_CAPTURE_SHAPE[1] - 1) * (SAVED_CAPTURE_SHAPE[2] - 1)
    )
    ratio = capture_cells / config.cell_count
    return {
        "case": "saved capture 20260827-152842, voxel 1.0 (not measured)",
        "estimate": True,
        "measured": False,
        "caveat": (
            "no surface was extracted from this capture: re-running physical registration was "
            "not authorised for this task. The examined cube count below is exact because the "
            "field's shape is recorded; the extraction time is a rough cell-scaled estimate "
            "from a measured synthetic row; every coverage-dependent count is unknown"
        ),
        "source_capture": "/home/ben/scans/scan/20260827-152842",
        "source_field": {
            "shape": list(SAVED_CAPTURE_SHAPE),
            "voxel_size": SAVED_CAPTURE_VOXEL_SIZE,
            "truncation": SAVED_CAPTURE_TRUNCATION,
            "cells": capture_cells,
        },
        "scaled_from": {"cells": config.cell_count, "backend": evidence.backend},
        "cell_ratio": ratio,
        "scaling_rule": (
            "only extraction time is cell-scaled, and only roughly. Eligible, excluded and "
            "active cube counts depend on the real observation coverage of the capture, and "
            "vertices, faces and the edge census additionally on the SURFACE AREA that crosses "
            "zero; a synthetic sphere predicts neither for a real chassis scan, so all of them "
            "are left unknown rather than multiplied by a ratio that has no basis"
        ),
        "extract_seconds_estimate": extract_seconds * ratio,
        "cubes": {
            "examined": capture_cubes,
            "examined_exact": True,
            "examined_rule": "(174 - 1) * (141 - 1) * (158 - 1), from the recorded field shape",
            "eligible": None,
            "excluded_insufficient_weight": None,
            "active": None,
        },
        "vertices_estimate": None,
        "faces_estimate": None,
        "edge_census_estimate": None,
    }


def surface_sweep() -> list[dict[str, object]]:
    """Surface extraction over the same synthetic noisy sphere the fusion sweep uses.

    Each field below is fused exactly ONCE with :meth:`TsdfVolume.integrate`
    and then extracted, so the two costs never blur together. The
    125,000-cell field is also the field this harness needs at a voxel small
    enough for the pure-Python extraction path - PYTHON_SURFACE_MAX_CELLS is
    262,144 - so ONE fused volume is extracted through both backends, and the
    reference path's cost sits right beside the accelerated one for identical
    geometry rather than a different field.
    """
    rows: list[dict[str, object]] = []
    camera = SYNTHETIC_CAMERA
    frames, poses = synthetic_views(16, jitter=0.5, camera=camera)
    bounds = VolumeBounds.of((-25.0, -25.0, 75.0), (25.0, 25.0, 125.0))

    small_config = TsdfConfig.of(
        bounds=bounds, voxel_size=1.0, truncation=3.0, min_depth=30.0, max_depth=200.0
    )
    small_volume = TsdfVolume(small_config, backend="numpy")
    small_volume.integrate(
        list(frames), list(poses), intrinsics=camera.intrinsics, depth_scale=camera.depth_scale
    )
    for backend in ("numpy", "python"):
        surface, elapsed, rss_delta = _measure_extraction(small_volume, backend=backend)
        rows.append(
            _surface_row(
                f"voxel 1.0, 125000 cells, {backend}",
                small_config,
                surface,
                elapsed=elapsed,
                rss_delta=rss_delta,
            )
        )

    # 1,000,000 cells: voxel 0.5, truncation 1.5 -> 100x100x100. Past the
    # pure-Python wall-clock ceiling, so only the accelerated backend runs
    # here; this is also the field the saved-capture estimate below scales
    # its numbers from.
    large_config = TsdfConfig.of(
        bounds=bounds, voxel_size=0.5, truncation=1.5, min_depth=30.0, max_depth=200.0
    )
    large_volume = TsdfVolume(large_config, backend="numpy")
    large_volume.integrate(
        list(frames), list(poses), intrinsics=camera.intrinsics, depth_scale=camera.depth_scale
    )
    large_surface, large_elapsed, large_rss_delta = _measure_extraction(
        large_volume, backend="numpy"
    )
    rows.append(
        _surface_row(
            "voxel 0.5, 1000000 cells, numpy",
            large_config,
            large_surface,
            elapsed=large_elapsed,
            rss_delta=large_rss_delta,
        )
    )
    rows.append(saved_capture_estimate(large_config, large_surface.evidence, large_elapsed))
    return rows


def registration_clouds(
    raw_frames: Sequence[bytes], *, camera: FusionCamera
) -> list[tuple[Point, ...]]:
    """The decimated, speckle-stripped clouds the CLI's own reconstruct registers.

    Separated from :func:`real_capture` so that the camera registration sees is
    a named input of a function rather than a record read inside a loop, and so
    a test can prove this honours the camera it is handed.
    """
    from scanner3d.denoise import strip_speckle
    from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame

    clouds: list[tuple[Point, ...]] = []
    for raw in raw_frames:
        conversion = deproject_depth_frame(
            raw,
            intrinsics=camera.intrinsics,
            depth_scale=camera.depth_scale,
            unit=UNVALIDATED_UNIT,
        )
        points = [(p.x, p.y, p.z) for p in conversion.points if NEAR_CLIP < p.z < FAR_CLIP]
        step = max(1, len(points) // REGISTRATION_POINTS)
        thinned = points[::step]
        filtered = strip_speckle(thinned) if thinned else None
        clouds.append(() if filtered is None else tuple(filtered.points))
    return clouds


def real_capture(directory: Path, *, stride: int, backend: str, workers: int) -> dict[str, object]:
    """Fuse one real capture at the poses the existing pipeline returns.

    The capture is read and never written. There is no ground truth for this
    object, so nothing here reports an accuracy: the numbers are cost, counts
    and extent, plus whether the pose-graph optimisation was ACCEPTED, which is
    the only statement about pose quality this harness is entitled to make.
    """
    from scanner3d.capture_sequence import read_capture_sequence
    from scanner3d.pipeline import reconstruct_sweep

    # ONE camera for this capture, bound once and handed to both halves. The
    # frames were taken with the device's factory record, so registering or
    # fusing them through anything else measures a camera that does not exist.
    camera = CAPTURE_CAMERA

    sequence = read_capture_sequence(directory).strided(max(1, stride))
    paths = sequence.paths
    raw_frames = [path.read_bytes() for path in paths]

    # Registration gets the decimated, speckle-stripped clouds the CLI's own
    # reconstruct command gives it. Fusion below gets the FULL frames: the whole
    # point of a TSDF is that it can afford every sample once the poses exist.
    clouds = registration_clouds(raw_frames, camera=camera)

    started = time.perf_counter()
    registration = reconstruct_sweep(
        clouds, backend=backend, workers=workers, bridges=sequence.boundaries()
    )
    registration_seconds = time.perf_counter() - started
    poses = list(registration.poses)

    # The volume is derived from the registered cloud's own extent, in the
    # poses' own frame - not the presentation frame the export rotates into,
    # which is a rigid change of basis that would move the box without changing
    # a single distance in it.
    def placed() -> list[Point]:
        out: list[Point] = []
        for cloud, pose in zip(clouds, poses, strict=True):
            out.extend(pose.apply_all(cloud))
        return out

    registered = placed()
    # The extent of a real sweep is not known in advance and a drifting
    # trajectory makes it larger, so the voxel size is chosen from a fixed
    # ladder: the FINEST rung whose volume fits the module ceiling. Nothing
    # here widens that ceiling - the alternative to choosing a coarser rung is
    # a refusal, which is what the unfixed harness got on this very capture.
    bounds = VolumeBounds.around(registered, margin=2.0 * VOXEL_LADDER[0] * 3.0)
    chosen: TsdfConfig | None = None
    rejected: list[str] = []
    for voxel in VOXEL_LADDER:
        try:
            chosen = TsdfConfig.of(
                bounds=bounds,
                voxel_size=voxel,
                truncation=3.0 * voxel,
                min_depth=NEAR_CLIP,
                max_depth=FAR_CLIP,
            )
        except VolumeTooLarge as exc:
            rejected.append(f"voxel {voxel}: {exc}")
            continue
        break
    if chosen is None:
        raise SystemExit(f"no rung of {VOXEL_LADDER} fits this extent: {rejected}")
    config = chosen
    row = measure(raw_frames, poses, config, backend="numpy", truth=False, camera=camera)
    row["voxel_ladder"] = list(VOXEL_LADDER)
    row["voxel_rungs_refused"] = rejected
    return {
        "capture": str(directory),
        "frames_on_disk": len(paths),
        "stride": stride,
        "registration": {
            "seconds": registration_seconds,
            "deprojection_camera": camera.as_dict(),
            "backend": backend,
            "workers": workers,
            "frames": registration.frames,
            "revisits_found": registration.revisits_found,
            "pose_optimisation_accepted": registration.optimised,
            "converged": registration.converged,
            "limitations": list(registration.limitations),
        },
        "registered_points": len(registered),
        "bounds": bounds.as_dict(),
        "fusion": row,
        "accuracy": (
            "none claimed: this capture has no measured ground truth, so nothing here says "
            "how close the fused surface is to the object"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=None, help="a capture directory to read")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--backend", default="auto", help="registration compute backend")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args(argv)

    report: dict[str, object] = {}
    if not args.skip_synthetic:
        report["synthetic"] = synthetic_sweep()
        report["surface"] = surface_sweep()
    if args.capture is not None:
        report["real_capture"] = real_capture(
            args.capture, stride=args.stride, backend=args.backend, workers=args.workers
        )
    json.dump(report, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
