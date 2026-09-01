"""Replay a real capture through the raw-points-versus-fused-surface comparison, read-only.

Deliberately NOT a ``test_`` module: pytest does not collect it. It is the
harness a handoff runs when the physical numbers are wanted, in the role
:mod:`tests.readiness_replay` plays for the readiness gates and
:mod:`tests.quality_replay` for the quality module. The deterministic oracle is
:mod:`tests.surface_oracle_harness`; this is its physical companion and it
shares none of its authority, because a real capture has no truth.

Run it with::

    .venv/bin/python -m tests.surface_oracle_replay
    .venv/bin/python -m tests.surface_oracle_replay --capture SOME/CAPTURE
    .venv/bin/python -m tests.surface_oracle_replay --limit 8
    .venv/bin/python -m tests.surface_oracle_replay \\
        --plane-roi -60,60,-60,60,300,420 --plane-name "turntable bed"

STRICTLY READ-ONLY. It calls ``Path.is_dir``, ``Path.rglob``, ``Path.stat``,
``Path.read_bytes`` and ``Path.read_text`` on the capture and nothing else. It
never opens a path for writing, creates no directory and no temporary file,
exports nothing, and touches no device node. It also PROVES it: size,
modification time and SHA-256 of every file in the capture are recorded before
and after the replay, and the report carries both plus whether they agree - the
same :func:`tests.readiness_replay.verify_unchanged` contract.

What it may say, and what it may not
-------------------------------------
There is no truth model for a physical capture, so **every truth-referenced
measurement the synthetic corpus publishes is absent here**: no coverage of a
truth surface, no surface accuracy, no dimension error, no edge-band coverage,
no thin-feature survival. Publishing any of them would mean inventing the truth
they are measured against. What is left is what can be observed without one:

* **return coverage** - the share of raw samples the sensor returned, counted
  from the bytes;
* **pose acceptance** - what :mod:`scanner3d.reconstruction_readiness` says
  about the trajectory this run fitted, at the shipped default policy;
* **roughness proxies** - the residuals of a plane fitted to a region the
  CALLER names with ``--plane-roi``. No region is ever chosen for the caller,
  and without the flag this is reported as not measured;
* **connected components and isolated lumps** - computed identically for the
  measured cloud and for the extracted surface, which is the closest thing to
  the original "blobby" complaint that survives having no truth;
* **parameter sensitivity** - the same predeclared grid the synthetic corpus
  owns, re-expressed against this capture's own measured extent, so the shape
  of the parameter response can be compared between the two.

One frame, from the bounds to the fusion
----------------------------------------
Every coordinate this harness measures, publishes or is given is in the MODEL
frame - the frame the registration's poses are in. That is the frame
:meth:`scanner3d.tsdf.TsdfVolume.integrate` documents for the poses it takes,
because it inverts each one to carry a voxel back into the camera that might
have seen it, and it is the frame production
:func:`scanner3d.reconstruction_export.export_reconstruction` measures its
bounds in and fuses at. The presentation half turn
:data:`scanner3d.reconstruction_export.UPRIGHT_FROM_OPTICAL` is applied to
NOTHING here, because nothing here is reported in the upright display/export
frame: an export turns its extracted mesh upright after the fusion, and a pose
that reached a model-frame field already rotated would carry every voxel half a
turn away from the raw evidence that should have filled it.

A readiness refusal is a result
-------------------------------
If the capture cannot be read, cannot be registered, or fails a readiness gate,
this harness **stops there and reports the refusal**. It does not fuse anyway
and publish the numbers with a caveat: the readiness boundary exists to say
that a reconstruction of this capture would not be trustworthy, and a harness
that reached past it would be answering a question the boundary already refused.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import scanner3d.quality as quality
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point
from scanner3d.pointcloud import sha256_of
from scanner3d.preparation import RECONSTRUCT_POINTS, prepare_frame
from scanner3d.reconstruction_export import (
    TSDF_MAX_DEPTH,
    TSDF_MIN_DEPTH,
    choose_tsdf_resolution,
    model_points,
)
from scanner3d.reconstruction_readiness import (
    DEFAULT_POLICY,
    coverage_evidence,
    pose_evidence,
)
from scanner3d.tsdf import MAX_VOXELS, TsdfConfig, TsdfVolume, VolumeBounds, WeightPolicy
from scanner3d.tsdf_surface import NoSurfaceFound, extract_surface
from tests import surface_oracle as contract
from tests import surface_oracle_harness as oracle

#: The capture this harness defaults to: 24 depth frames of a model truck bed
#: on a turntable, committed in this repository and immutable. It is the same
#: capture :mod:`tests.quality_replay` defaults to, so the two harnesses can be
#: read against each other.
DEFAULT_CAPTURE: Final = Path(__file__).parent / "fixtures" / "turntable"

#: The compute backend. The pure-Python references, matching the deterministic
#: corpus, so a number here and a number there were produced by the same
#: arithmetic.
BACKEND: Final = "python"

#: The linkage radius and lump size the component measure uses, as MULTIPLES of
#: the cell's own voxel size. The synthetic corpus can state them in its scene's
#: units because it owns the scene; this capture's units have never been
#: established, so the only defensible statement is a relative one.
LINKAGE_RADIUS_VOXELS: Final = 2.0
ISOLATED_MAX_POINTS: Final = 8

#: How this replay turns the corpus's synthetic grid into one that means
#: something against a capture whose scale is unknown. The voxel RATIOS,
#: truncation ratios, minimum weights and weight policies are the corpus's own,
#: unchanged; what is substituted is the base voxel size, which comes from
#: :func:`scanner3d.reconstruction_export.choose_tsdf_resolution` on this
#: capture's measured extent - the same chooser an export would use. Declared
#: here, before the capture is read, and published in the report.
BASE_CELL_ID: Final = "v15-t3-w1-uniform"

#: The synthetic voxel size :data:`BASE_CELL_ID` uses. Every other cell's voxel
#: is scaled by its ratio against this one, so the grid keeps its shape.
BASE_VOXEL: Final = 0.15

#: The depth window every cell fuses through, in the capture's own units: the
#: appliance's own band, imported from
#: :mod:`scanner3d.reconstruction_export` rather than restated here so the two
#: cannot drift. Still stated rather than fitted to a capture, so a sample the
#: window excludes is excluded for the same reason on every capture this harness
#: is pointed at - and now excluded for the same reason a production export
#: would exclude it, which is the only defensible window for a harness whose
#: whole claim is that it fuses the way the appliance fuses.
#:
#: They replace a hand-written ``0.05``/``4.0``. That pair is a SYNTHETIC
#: scene's window - the deterministic corpus's scenes sit between 4 and 16 -
#: and it is not a window any capture this repository has recorded can be seen
#: through: a frame's raw codes times :data:`scanner3d.cli.DEPTH_SCALE_METRES`
#: put every return of ``20260827-152842`` between roughly 140 and 280. Under
#: the old pair every observation of a physical capture was discarded during
#: projection, so every field came back with zero observed voxels and no
#: surface, whatever the poses were - which is exactly why the pose defect
#: below could sit in this harness undetected.
MIN_DEPTH: Final = TSDF_MIN_DEPTH
MAX_DEPTH: Final = TSDF_MAX_DEPTH

#: The one frame every number in this report is in, published in the report
#: itself so a reader never has to infer it from the code. See the module
#: docstring for why a field measured in one frame and fused in another is not
#: a wrong-looking scan but no scan at all.
MODEL_FRAME: Final = (
    "the MODEL frame, which is the frame this run's registration poses are in. The field's "
    "bounds, the poses the fusion is given, the placed dots, the extracted vertices and any "
    "--plane-roi region are all in that single frame. The presentation half turn "
    "scanner3d.reconstruction_export.UPRIGHT_FROM_OPTICAL, which an export applies to the mesh "
    "it writes, is applied to nothing here, because nothing here is reported in the upright "
    "display/export frame"
)

LIMITATIONS: Final[tuple[str, ...]] = (
    "read-only. No file in the capture was written, renamed, deleted or created, and the "
    "before/after fingerprints in this report are the evidence for that.",
    "there is no truth model for a physical capture. No coverage of a truth surface, surface "
    "accuracy, dimension error, edge coverage or thin-feature survival is measured or claimed "
    "anywhere in this report; the deterministic corpus is the only place those exist.",
    "no number here has a physical unit. Every length is in the capture's own depth units, "
    "whose physical meaning this repository has not established, so no dimensional accuracy "
    "is claimed and none can be read out of this report.",
    "the poses the readiness gates judge, and the poses the fusion is given, were fitted by "
    "THIS run. A pose the registration got wrong and was confident about is measured here as "
    "ordinary motion.",
    "the parameter grid's voxel sizes are ratios of a base voxel this run derived from the "
    "capture's own extent. They are not the corpus's synthetic voxel sizes and the two "
    "sequences of absolute numbers are not comparable; the SHAPE of the response is.",
    "every coordinate here is in the model frame the poses were fitted in, and no geometry is "
    "reported in the upright presentation frame an export writes. A number read out of this "
    "report is therefore not directly comparable with an exported PLY's coordinates, which "
    "have had one further half turn about x applied to them.",
    "the depth window the fusion projects through is the appliance's own TSDF_MIN_DEPTH and "
    "TSDF_MAX_DEPTH, imported rather than restated. A return outside that band is not measured "
    "here, exactly as it would not be exported; the band is a stated decision of this "
    "repository and not a property of the sensor.",
    "component counts depend on the linkage radius, which is stated in voxels rather than "
    "chosen to produce a number. Two cells with different voxel sizes are therefore linked at "
    "different absolute radii, which is the only way the comparison stays scale-free.",
    "nothing here is routed into the product. No export, live preview, GUI, capture, tracker "
    "or readiness policy reads this module, and no threshold in the repository is changed by "
    "anything it prints.",
)


class ReplayRefused(ScannerError):
    """The capture could not be replayed at all, and nothing was measured."""


def fingerprint(capture: Path) -> dict[str, dict[str, object]]:
    """Size, mtime and digest of every file in the capture, before and after."""
    record: dict[str, dict[str, object]] = {}
    for path in sorted(capture.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        record[str(path.relative_to(capture))] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_of(path.read_bytes()),
        }
    return record


def verify_unchanged(
    before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Compare two fingerprints and name every file that differs."""
    changed = sorted(name for name in before if before[name] != after.get(name))
    missing = sorted(name for name in before if name not in after)
    appeared = sorted(name for name in after if name not in before)
    return {
        "files": len(before),
        "unchanged": not (changed or missing or appeared),
        "changed": changed,
        "missing": missing,
        "appeared": appeared,
    }


def _bind_by_frames(capture: Path, *, limit: int | None) -> tuple[dict[str, object], list[Path]]:
    """Identity for a capture with no manifest: the frames' own digests, in name order.

    A live capture folder written by this repository's own capture path has
    ``frame_NNNN.raw`` files and no manifest. There is nothing to check the
    bytes AGAINST in that case, so the identity published is the digest of each
    frame plus the digest of that ordered list - enough for a later reader to
    tell whether a number came from these bytes, and honestly weaker than the
    manifest-bound case, which the report says.
    """
    paths = sorted(capture.glob("frame_*.raw"))
    if not paths:
        raise ReplayRefused(
            f"{capture} has neither a manifest.json nor any frame_*.raw file. This harness "
            "binds a capture by its bytes and will not measure a folder it cannot name"
        )
    if limit is not None:
        paths = paths[:limit]
    frames = [
        {"index": index, "file": path.name, "sha256": sha256_of(path.read_bytes())}
        for index, path in enumerate(paths)
    ]
    listing = "\n".join(f"{frame['file']} {frame['sha256']}" for frame in frames)
    identity: dict[str, object] = {
        "capture": str(capture),
        "bound_by": "frame digests; this capture carries no manifest to check them against",
        "frames_sha256": sha256_of(listing.encode("utf-8")),
        "frames_named": len(sorted(capture.glob("frame_*.raw"))),
        "frames_used": len(paths),
        "frames": frames,
    }
    return identity, paths


def bind_identity(capture: Path, *, limit: int | None) -> tuple[dict[str, object], list[Path]]:
    """Read the capture's manifest, check every frame against it, and name it.

    The identity of a physical capture is the manifest's own digest plus the
    digest of every frame it names, verified here rather than trusted: a report
    that quoted a manifest without checking the bytes beside it would be
    describing a capture that may no longer be the one on disk.
    """
    if not capture.is_dir():
        raise ReplayRefused(f"{capture} is not a directory; nothing was read")
    manifest_path = capture / "manifest.json"
    if not manifest_path.is_file():
        return _bind_by_frames(capture, limit=limit)
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayRefused(f"{manifest_path} is not readable JSON: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("frames"), list):
        raise ReplayRefused(f"{manifest_path} names no 'frames' list; nothing was read")

    paths: list[Path] = []
    frames: list[dict[str, object]] = []
    for index, entry in enumerate(manifest["frames"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ReplayRefused(f"{manifest_path} frame {index} names no file")
        name = str(entry["name"])
        if "/" in name or "\\" in name or name.startswith("."):
            raise ReplayRefused(f"{manifest_path} frame {index} names {name!r}, which is a path")
        path = capture / f"{name}.bin"
        if not path.is_file():
            raise ReplayRefused(f"{manifest_path} names {path.name}, which is not in the capture")
        digest = sha256_of(path.read_bytes())
        declared = entry.get("sha256")
        if isinstance(declared, str) and declared != digest:
            raise ReplayRefused(
                f"{path.name} hashes to {digest} and the manifest says {declared}. The capture "
                "on disk is not the capture this manifest describes"
            )
        paths.append(path)
        frames.append({"index": index, "file": path.name, "sha256": digest})

    if limit is not None:
        paths = paths[:limit]
        frames = frames[:limit]
    identity: dict[str, object] = {
        "capture": str(capture),
        "bound_by": "manifest.json; every frame's digest was checked against it before use",
        "manifest_sha256": sha256_of(manifest_bytes),
        "manifest_bytes": len(manifest_bytes),
        "frames_named": len(manifest["frames"]),
        "frames_used": len(paths),
        "frames": frames,
    }
    return identity, paths


def return_coverage(raw: bytes) -> dict[str, object]:
    """The share of one frame's raw uint16 samples that carried a return.

    Counted from the bytes, in the repository's own no-return convention: code
    zero is the sensor saying it had no depth for that pixel. Nothing here
    fills, interpolates or reconstructs a missing sample.
    """
    samples = len(raw) // 2
    returned = sum(1 for offset in range(0, samples * 2, 2) if raw[offset] or raw[offset + 1])
    return {
        "samples": samples,
        "returned": returned,
        "returned_fraction": (returned / samples) if samples else 0.0,
    }


def _components(points: Sequence[Point], radius: float) -> dict[str, object]:
    sizes = contract.component_sizes(points, radius)
    total = sum(sizes)
    isolated = [size for size in sizes if size <= ISOLATED_MAX_POINTS]
    return {
        "linkage_radius": radius,
        "component_count": len(sizes),
        "largest_component_points": sizes[0] if sizes else 0,
        "largest_component_fraction": (sizes[0] / total) if total else 0.0,
        "isolated_components": len(isolated),
        "isolated_points": sum(isolated),
        "isolated_fraction": (sum(isolated) / total) if total else 0.0,
    }


def _plane(points: Sequence[Point], roi: tuple[float, ...] | None, name: str) -> dict[str, object]:
    """Plane residuals over a region the CALLER named, or a refusal to guess one."""
    if roi is None:
        return {
            "measured": False,
            "why": (
                "no --plane-roi was given. This harness never chooses a region for the caller: "
                "a roughness number over a region somebody picked because it looked flat is a "
                "number about the picking"
            ),
        }
    x0, x1, y0, y1, z0, z1 = roi
    inside = [
        point
        for point in points
        if x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and z0 <= point[2] <= z1
    ]
    if len(inside) < 3:
        return {
            "measured": False,
            "why": f"{len(inside)} point(s) fell inside the named region; a plane fit needs three",
        }
    measurement = quality.plane_roughness(
        inside,
        name="plane",
        region=name,
        unit="the capture's own depth units; no physical length is established",
    )
    return {"measured": True, "points": len(inside), "roughness": measurement.as_dict()}


def _cell_for(base: contract.GridCell, voxel: float) -> contract.GridCell:
    """One corpus cell re-expressed at this capture's own base voxel size."""
    return contract.GridCell(
        cell_id=base.cell_id,
        voxel_size=voxel * (base.voxel_size / BASE_VOXEL),
        truncation_voxels=base.truncation_voxels,
        min_weight=base.min_weight,
        weight_policy=base.weight_policy,
    )


def replay(
    capture: Path,
    *,
    limit: int | None,
    backend: str,
    plane_roi: tuple[float, ...] | None,
    plane_name: str,
    voxel_budget: int = MAX_VOXELS,
) -> dict[str, object]:
    """Everything this capture can support saying, and where it stopped if it stopped.

    ``voxel_budget`` is the ceiling handed to
    :func:`scanner3d.reconstruction_export.choose_tsdf_resolution`, and it exists
    so a test can fuse a bounded field in seconds rather than minutes. It
    defaults to the shipped :data:`scanner3d.tsdf.MAX_VOXELS` this harness has
    always used, so a physical run is unaffected, and the chooser refuses any
    value above that ceiling: this argument can narrow the field and can never
    widen it.
    """
    corpus = oracle.load_corpus()
    grid = corpus.grid
    base = next((cell for cell in grid if cell.cell_id == BASE_CELL_ID), None)
    if base is None:  # pragma: no cover - the corpus manifest declares it
        raise ReplayRefused(f"the corpus grid has no cell {BASE_CELL_ID!r} to scale the rest from")

    identity, paths = bind_identity(capture, limit=limit)
    report: dict[str, object] = {
        "kind": "scanner3d.surface-oracle-physical-replay",
        "schema_version": 1,
        "identity": identity,
        "backend": backend,
        "policy": DEFAULT_POLICY.as_dict(),
        "corpus_grid": [cell.as_dict() for cell in grid],
        # The shipped resolution policy the synthetic corpus binds one arm to.
        # Published here as CONTEXT, never as a result: this replay has no truth
        # model and, when it stops before fusion, derives no resolution at all.
        # A reader comparing the two artifacts needs to see that the declared
        # policy is the same one on both sides.
        "appliance_policy": corpus.appliance.as_dict(),
        "base_cell": BASE_CELL_ID,
        "frame": MODEL_FRAME,
        "limitations": list(LIMITATIONS),
    }
    if len(paths) < 3:
        report["stopped_at"] = "frame_count"
        report["stopped_because"] = f"{len(paths)} usable frames; a sweep needs at least three"
        return report

    raws = [path.read_bytes() for path in paths]
    report["returns"] = {
        "per_frame": [
            {"file": path.name, **return_coverage(raw)}
            for path, raw in zip(paths, raws, strict=True)
        ],
    }

    started = time.perf_counter()
    clouds = [prepare_frame(raw, target_points=RECONSTRUCT_POINTS).points for raw in raws]
    prepared_s = time.perf_counter() - started

    from scanner3d.pipeline import reconstruct_sweep

    started = time.perf_counter()
    try:
        result = reconstruct_sweep(clouds, backend="auto")
    except ScannerError as error:
        report["stopped_at"] = "registration"
        report["stopped_because"] = str(error)
        report["timing"] = {"prepare_s": round(prepared_s, 3)}
        return report
    registered_s = time.perf_counter() - started

    poses, pose_failures = pose_evidence(clouds, result.poses)
    coverage, coverage_failures = coverage_evidence(clouds, result.poses)
    reasons = [failure.reason for failure in (*pose_failures, *coverage_failures)]
    report["registration"] = {
        "frames": result.frames,
        "revisits_found": result.revisits_found,
        "optimised": result.optimised,
    }
    report["poses"] = {
        "evidence": poses.as_dict(),
        "failures": [failure.as_dict() for failure in pose_failures],
    }
    report["coverage"] = {
        "evidence": coverage.as_dict(),
        "failures": [failure.as_dict() for failure in coverage_failures],
    }
    report["readiness_reasons"] = reasons
    report["ready_before_fusion"] = not reasons
    if reasons:
        # The refusal IS the result. Fusing anyway and publishing the numbers
        # with a caveat would answer past a boundary that has already said a
        # reconstruction of this capture would not be trustworthy.
        report["stopped_at"] = "readiness"
        report["stopped_because"] = (
            "the shipped readiness policy refuses this capture, so no fusion was attempted and "
            "no surface was measured. The refusal is the finding"
        )
        report["timing"] = {
            "prepare_s": round(prepared_s, 3),
            "register_s": round(registered_s, 3),
        }
        return report

    # ONE frame, measured and fused. These are the poses the registration
    # fitted, in the model frame; ``model_points`` places the clouds by them to
    # measure the extent, and the SAME list is what ``TsdfVolume.integrate`` is
    # given below. That is the contract ``integrate`` documents - it inverts
    # each pose to carry a voxel back into its camera - and it is what
    # production ``export_reconstruction`` does: measure with ``model_points``,
    # integrate the raw poses, turn the extracted MESH upright afterwards. A
    # field bounded here and fused at the presentation-rotated poses is not a
    # scan drawn the wrong way up; it is every voxel projected half a turn away
    # from the raw evidence that should have filled it, and therefore no
    # surface at all.
    fusion_poses = list(result.poses)
    placed = list(model_points(clouds, fusion_poses))
    resolution = choose_tsdf_resolution(placed, voxel_budget=voxel_budget)
    bounds = resolution.bounds
    report["resolution"] = resolution.as_dict()

    dots = _components(placed, resolution.voxel_size * LINKAGE_RADIUS_VOXELS)
    report["raw_points"] = {
        "point_count": len(placed),
        "components": dots,
        "plane": _plane(placed, plane_roi, plane_name),
        "note": (
            "the measured dots, placed by the poses this run fitted and clipped to nothing. No "
            "truth-referenced number is published for them because there is no truth"
        ),
    }

    from scanner3d.cli import DEPTH_SCALE_METRES
    from scanner3d.simulate import FACTORY_INTRINSICS

    cells: list[dict[str, object]] = []
    started = time.perf_counter()
    for declared in grid:
        cell = _cell_for(declared, resolution.voxel_size)
        entry: dict[str, object] = {"cell": cell.as_dict(), "declared": declared.as_dict()}
        try:
            config = TsdfConfig.of(
                bounds=VolumeBounds.of(bounds.minimum, bounds.maximum),
                voxel_size=cell.voxel_size,
                truncation=cell.truncation,
                min_depth=MIN_DEPTH,
                max_depth=MAX_DEPTH,
                weight_policy=WeightPolicy(cell.weight_policy),
            )
        except ScannerError as error:
            entry["refused"] = str(error)
            cells.append(entry)
            continue
        volume = TsdfVolume(config, backend=backend)
        volume.integrate(
            raws, fusion_poses, intrinsics=FACTORY_INTRINSICS, depth_scale=DEPTH_SCALE_METRES
        )
        evidence = volume.evidence()
        vertices: tuple[Point, ...] = ()
        faces = 0
        extracted = True
        try:
            surface = extract_surface(volume, min_weight=cell.min_weight, backend=backend)
        except (NoSurfaceFound, ScannerError) as error:
            extracted = False
            entry["extraction_refused"] = str(error)
        else:
            vertices = tuple(surface.mesh.vertices)
            faces = surface.mesh.face_count
        entry["field"] = {
            "cells": config.cell_count,
            "owned_bytes": config.owned_bytes,
            "frames_offered": evidence.frames_offered,
            "frames_integrated": evidence.frames_integrated,
            "observed_voxels": evidence.weights.observed_voxels,
            "unobserved_voxels": evidence.weights.unobserved_voxels,
            "work_units": config.cell_count * len(raws),
        }
        entry["surface"] = {
            "extracted": extracted,
            "vertices": len(vertices),
            "faces": faces,
            "components": _components(vertices, cell.voxel_size * LINKAGE_RADIUS_VOXELS),
            "plane": _plane(vertices, plane_roi, plane_name),
        }
        cells.append(entry)
    fused_s = time.perf_counter() - started

    report["sensitivity"] = cells
    report["timing"] = {
        "prepare_s": round(prepared_s, 3),
        "register_s": round(registered_s, 3),
        "fuse_and_extract_s": round(fused_s, 3),
    }
    return report


def _roi(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    parts = tuple(float(piece) for piece in text.split(","))
    if len(parts) != 6:
        raise SystemExit("--plane-roi wants exactly six comma-separated numbers: x0,x1,y0,y1,z0,z1")
    return parts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Physical replay of the surface oracle.")
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--plane-roi", default=None)
    parser.add_argument("--plane-name", default="a region the caller named")
    args = parser.parse_args(list(argv) if argv is not None else None)

    before = fingerprint(args.capture) if args.capture.is_dir() else {}
    try:
        report = replay(
            args.capture,
            limit=args.limit,
            backend=args.backend,
            plane_roi=_roi(args.plane_roi),
            plane_name=args.plane_name,
        )
    except ReplayRefused as error:
        print(json.dumps({"refused": str(error)}, indent=2))
        return 2
    after = fingerprint(args.capture)
    report["read_only"] = verify_unchanged(before, after)
    print(json.dumps(report, indent=2, allow_nan=False, sort_keys=True))
    if not report["read_only"]["unchanged"]:  # type: ignore[index]
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
