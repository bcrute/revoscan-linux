"""Measure a capture of reconstruction-target-v1 against its CAD, and judge it.

Run every session:

    .venv/bin/python tools/target_gate.py /home/ben/scans/scan/<capture>
    .venv/bin/python tools/target_gate.py <capture> --json
    .venv/bin/python tools/target_gate.py <capture> --depth-band 130 275

The depth band is the interval the SUBJECT occupies. Anything static in the
room inside the default 60-300 clip registers as a second rigid body and
bends the trajectory without closure noticing; see docs/target-gate.md and
:class:`scanner3d.preparation.DepthBand`.

STRICTLY READ-ONLY on the capture. Reads raw frames, writes nothing into the
capture directory, and touches no device.

What it judges, and what it refuses to judge
--------------------------------------------
**Coherence** is checked first and refused on. Frames one turntable
revolution apart have the same depth image, so their poses must agree;
:mod:`scanner3d.coherence` reads the period from the images and projects
each revisit through both poses. A sweep whose revisits disagree is a smear
of misplaced frames, and every number below would be measured on the smear -
a CAD-side nearest-point query is satisfied by a dense enough smear whatever
its shape, which is how a MIRRORED reference once fitted one as well as the
real one. Nothing is reported for such a sweep.

**Accuracy** is gated. Over the CAD points that were actually observed, how far
is the scan from the reference? That is answerable and it is what the tolerance
file declares a bar for.

**Completeness** is reported and NOT gated, because what limits it is not yet
known. Two surfaces of the reference come back unobserved on a capture whose
accuracy is otherwise good, and no explanation for that has survived testing -
see docs/target-gate.md. Gating on a quantity whose failure mode is not
understood would turn every capture into a refusal nobody can act on, so it is
named per surface instead: the report says WHICH surfaces were not seen.

That split is the same one docs/coverage-and-subject.md argues for and the same
one the sibling mapper repository settled on: two verdicts that are never fused.

Why the CAD is the query side
-----------------------------
Distances are measured FROM each CAD point TO the nearest scan point, not the
other way. Scan content that is not the target - turntable, room - is never
queried, so it can neither inflate nor deflate the result. That removes the need
to segment the subject out of the scene, which two earlier attempts failed to do
(see docs/target-gate.md, "Method").

A nearest-neighbour distance from the reference is large for two unrelated
reasons, though - a surface measured badly, and a surface never measured - which
is exactly why the two verdicts above are reported apart. A CAD point with no
scan point inside :data:`OBSERVED_RADIUS` counts against completeness and is
excluded from accuracy, because "how far off" means nothing about a surface
nobody saw.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.spatial import cKDTree

ROOT: Final = Path(__file__).resolve().parent.parent
TARGET: Final = ROOT / "reference/targets/reconstruction_v1"


if TYPE_CHECKING:
    # For annotations only: the real imports happen inside the functions,
    # after _src() below has put the repository's src on the path.
    from scanner3d.geometry import Rigid
    from scanner3d.intrinsics import IntrinsicsRecord
    from scanner3d.preparation import DepthBand


def _src() -> None:
    """Make the package importable from a checkout without installing it."""
    path = str(ROOT / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


#: A camera orbit is a turntable's when it is at least this far from its axis
#: and this close to a circle. A handheld pass fails both, and the placement
#: then seeds from the support plane as it always did.
ORBIT_MIN_RADIUS: Final = 50.0
ORBIT_MAX_RESIDUAL: Final = 2.0

#: A CAD point with no scan point this close was not observed, and is excluded
#: from accuracy rather than counted as a large error.
#:
#: Three millimetres: above the 1.32 mm agreement a good capture achieves and
#: above its 2.51 mm p95, so an observed surface is not mistaken for a missing
#: one; well below the 4.27 mm that the nearest scan point to an unobserved face
#: was measured at, so a missing surface is not mistaken for a bad one.
OBSERVED_RADIUS: Final = 3.0

#: Correspondence distance and selection radius for each refinement pass, wide
#: to narrow.
#:
#: Coarse-to-fine, and the schedule is the point. A single wide pass converges
#: to a fit limited by its own correspondence distance: run at a flat 8 mm this
#: registration settled at 1.99 mm median and stopped improving, because at that
#: width every point on a surface can pair with a neighbour a millimetre or two
#: off and the solve has no reason to prefer the right one. Narrowing after each
#: pass removes those pairings once the model is close enough not to need them.
#:
#: The last pass at 2.0 mm is deliberately below the 3 mm observation radius:
#: the fit is then decided only by points that would count as observed, so the
#: registration and the measurement agree about which surface they are talking
#: about.
REFINE_SCHEDULE: Final = ((8.0, 5.0), (5.0, 4.0), (3.5, 3.0), (2.0, 2.5))

#: Passes at each width before moving on.
PASSES_PER_WIDTH: Final = 3

#: Share of CAD points that must land within :data:`OBSERVED_RADIUS` before any
#: per-surface number is reported at all.
#:
#: A quarter. Well under the 79% a good capture reaches, so a genuinely
#: incomplete scan is still judged and its gaps named; far above the ~0% a failed
#: registration produces, which is what this exists to catch. There is no useful
#: value in between: a placement either found the target or it did not.
MIN_OBSERVED_TO_JUDGE: Final = 0.25


def read_reference_ply() -> np.ndarray:
    """The reference PLY as a structured array, positions UNSCALED.

    Separate from :func:`load_reference` because the normals are needed by the
    orthogonality gate and the positions are needed shrinkage-adjusted, and
    parsing the header twice to get them was the alternative.
    """
    raw = (TARGET / "reconstruction-target-v1-reference.ply").read_bytes()
    head = raw.index(b"end_header\n") + len(b"end_header\n")
    count = int(
        next(
            line for line in raw[:head].split(b"\n") if line.startswith(b"element vertex")
        ).split()[-1]
    )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("sid", "<u2"),
        ]
    )
    return np.frombuffer(raw, dtype=dtype, count=count, offset=head)


def read_reference_ply_normals() -> np.ndarray:
    """Unit normals from the reference PLY, in the CAD's own frame."""
    ply = read_reference_ply()
    return np.c_[ply["nx"], ply["ny"], ply["nz"]].astype(float)


def load_reference() -> tuple[np.ndarray, np.ndarray, dict[int, str], dict]:
    """CAD points shrinkage-adjusted as the print record predeclared, plus ids."""
    manifest = json.loads((TARGET / "reconstruction-target-v1.json").read_text())
    prints = sorted((TARGET / "prints").glob("physical-print-*.json"))
    if not prints:
        raise SystemExit("no print record: cannot know this print's shrinkage")
    record = json.loads(prints[-1].read_text())
    scale = record["dense_reference_adjustment"]

    ply = read_reference_ply()
    cad = np.c_[
        ply["x"] * scale["scale_x"], ply["y"] * scale["scale_y"], ply["z"] * scale["scale_z"]
    ].astype(float)
    names = {v["id"]: k for k, v in manifest["surface_ids"].items()}
    return cad, np.array(ply["sid"]), names, record


def nearest(a: np.ndarray, b: np.ndarray, chunk: int = 400) -> np.ndarray:
    """Distance from every row of ``a`` to the closest row of ``b``.

    Exact, via a k-d tree. The brute-force form this replaced computed the full
    |a| x |b| distance matrix in chunks, which on the refinement's own inner call
    - every scan point near the model against the placed CAD - is order 10^9
    products per pass and made a single placement several minutes of the gate's
    runtime. The answer is identical; only the time is different, and it is the
    difference between a repeatability sweep over several seeds being routine
    and being unaffordable.

    ``chunk`` is retained and ignored so existing callers keep working.
    """
    del chunk
    if len(b) == 0:
        return np.full(len(a), np.inf)
    return cKDTree(b).query(a, workers=-1)[0]


#: The four outer side faces the orthogonality gate is declared over, and the
#: face every angle is measured against.
GATE_FACES: Final = ("outer_east", "outer_west", "outer_north", "outer_south")
GATE_DATUM: Final = "base_top"

#: Scan points needed before a face's plane is fitted at all. A plane through
#: too few points is dominated by whichever stray returns landed in the set,
#: and reporting an angle from it would be exactly the plausible-number-from-a-
#: broken-measurement this tool refuses elsewhere.
MIN_FACE_POINTS: Final = 60


def fit_plane(points: np.ndarray, rounds: int = 3) -> tuple[np.ndarray, float]:
    """Unit normal of a plane through ``points``, refitted without its outliers.

    Trimmed because the points reaching this are whatever fell within the
    observation radius of a CAD face, which near an edge includes returns from
    the neighbouring face, and a least-squares plane has no defence against
    them. Returns the normal and the RMS deviation of the surviving points.
    """
    keep = np.ones(len(points), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(rounds):
        q = points[keep]
        centred = q - q.mean(0)
        normal = np.linalg.svd(centred, compute_uv=True)[2][2]
        residual = (points - q.mean(0)) @ normal
        keep = np.abs(residual) < 2.0 * float(np.sqrt((residual[keep] ** 2).mean()))
        if keep.sum() < MIN_FACE_POINTS:
            keep = np.ones(len(points), dtype=bool)
            break
    q = points[keep]
    return normal, float(np.sqrt((((q - q.mean(0)) @ normal) ** 2).mean()))


def local_normals(points: np.ndarray, radius: float = 2.0, min_n: int = 10) -> np.ndarray:
    """Unit normal of a plane fitted in each point's own neighbourhood.

    Points whose neighbourhood is too small to define one get a zero vector,
    which fails every orientation test it is later used in.
    """
    tree = cKDTree(points)
    out = np.zeros_like(points)
    for i, neighbours in enumerate(tree.query_ball_point(points, radius, workers=-1)):
        if len(neighbours) < min_n:
            continue
        q = points[neighbours]
        out[i] = np.linalg.svd(q - q.mean(0), compute_uv=True)[2][2]
    return out


#: How far a scan point's own surface orientation may sit from the CAD face it
#: was attributed to before it is dropped from that face's plane fit.
#:
#: Proximity alone is not attribution to a SURFACE, and the first version of
#: this gate proved it: it handed outer_north 4391 scan points for a face
#: carrying 335 CAD points and reported 11.8-16.0 degrees against a 2 degree
#: bar. Most scan points within 3 mm of a 92 mm model on a turntable are not on
#: the face nearest them - they are turntable, background, or the adjacent face
#: across an edge - and none of those faces the way the face faces.
FACE_NORMAL_TOLERANCE_DEGREES: Final = 25.0

#: How close a scan point must be to a CAD point OF THAT FACE to join its fit.
#:
#: Against the face's own points, not the whole model. Global
#: nearest-CAD-point attribution hands a face whatever happens to be nearer to
#: it than to anything else, which on a densely sampled model is a sliver:
#: measured, outer_north's set was 20.5 x 2.05 x 1.6 mm, a rod whose two short
#: axes are within 30% of each other, so its "normal" was whichever of them the
#: noise picked. Selecting against the face's own points gave the same face a
#: set whose normal sits 2.1 degrees from the CAD's.
FACE_SELECT_RADIUS: Final = 1.5

#: In-plane extent must exceed thickness by this much before an angle is
#: reported.
#:
#: The guard that turns the sliver problem from a wrong number into an honest
#: refusal. A plane normal is only defined when the two smallest principal
#: extents are well separated; on a strip 2 mm wide and 0.8 mm thick they are
#: not, and the fit is free to rotate within that ambiguity. Measured on
#: 20260901-201341, a turntable-only sweep sees base_top at 24x and the outer
#: side faces at 2.8x - it grazes them rather than viewing them, which is a
#: statement about what a single-axis sweep can cover and not about the
#: scanner's geometry.
FACE_FLATNESS_RATIO: Final = 3.0

#: Scan points needed before a face's plane is fitted at all.
MIN_FACE_POINTS: Final = 60


def face_plane(
    points: np.ndarray,
    observed_normals: np.ndarray,
    face_cad: np.ndarray,
    face_normal: np.ndarray,
) -> dict:
    """Fit one CAD face's plane on the scan, or say why it cannot be fitted."""
    chosen = cKDTree(face_cad).query(points, workers=-1)[0] < FACE_SELECT_RADIUS
    chosen &= np.abs(observed_normals @ face_normal) > math.cos(
        math.radians(FACE_NORMAL_TOLERANCE_DEGREES)
    )
    q = points[chosen]
    if len(q) < MIN_FACE_POINTS:
        return {"scan_points": len(q), "verdict": "not_observed"}
    spread = np.linalg.svd(q - q.mean(0), compute_uv=False) / math.sqrt(len(q))
    flatness = float(spread[1] / spread[2]) if spread[2] > 0 else float("inf")
    if flatness < FACE_FLATNESS_RATIO:
        return {
            "scan_points": len(q),
            "flatness_ratio": flatness,
            "verdict": "not_resolvable",
        }
    normal = np.linalg.svd(q - q.mean(0))[2][2]
    return {
        "scan_points": len(q),
        "flatness_ratio": flatness,
        "plane_rms_mm": float(np.sqrt((((q - q.mean(0)) @ normal) ** 2).mean())),
        "normal": normal,
        "verdict": "fitted",
    }


def orthogonality(
    scan: np.ndarray,
    placed: np.ndarray,
    sid: np.ndarray,
    normals: np.ndarray,
    names: dict[int, str],
    radius: float = OBSERVED_RADIUS,
) -> list[dict]:
    """Angle between each outer side face and base_top, fitted on the SCAN.

    The first of the three gates tolerance-v1.json declares, and one this tool
    did not compute until now. It is the principal-point test: a shear
    ``x += k*z`` leaves a plane a plane but rotates its normal, so east and west
    tilt against the top while north and south do not move.

    Fitted on the registered scan, never on the CAD - the CAD is square by
    construction and would report 90 degrees whatever the scanner did.

    A face this capture only grazed is reported ``not_resolvable`` rather than
    given an angle. See FACE_FLATNESS_RATIO: the number a sliver produces is
    determined by which of two nearly equal principal axes the noise selected,
    and reporting it as an orthogonality failure would blame the scanner's
    geometry for the sweep's coverage.
    """
    by_name = {v: k for k, v in names.items()}
    distance, _ = cKDTree(placed).query(scan, workers=-1)
    points = scan[distance < radius]
    observed_normals = local_normals(points)

    def cad_face(face: str) -> tuple[np.ndarray, np.ndarray] | None:
        fid = by_name.get(face)
        if fid is None:
            return None
        mask = sid == fid
        if not mask.any():
            return None
        normal = normals[mask].mean(0)
        length = float(np.linalg.norm(normal))
        if length < 1e-9:  # opposing facets cancel: not a single plane
            return None
        return placed[mask], normal / length

    datum = cad_face(GATE_DATUM)
    if datum is None:
        return [{"face": GATE_DATUM, "verdict": "not_observed", "scan_points": 0}]
    fit = face_plane(points, observed_normals, *datum)
    if fit["verdict"] != "fitted":
        return [{"face": GATE_DATUM, **{k: v for k, v in fit.items() if k != "normal"}}]
    datum_normal = fit["normal"]
    rows: list[dict] = [
        {"face": GATE_DATUM, **{k: v for k, v in fit.items() if k != "normal"}, "verdict": "datum"}
    ]

    for face in GATE_FACES:
        found = cad_face(face)
        if found is None:
            rows.append({"face": face, "scan_points": 0, "verdict": "not_observed"})
            continue
        fit = face_plane(points, observed_normals, *found)
        row = {"face": face, **{k: v for k, v in fit.items() if k != "normal"}}
        if fit["verdict"] == "fitted":
            cosine = abs(float(np.dot(fit["normal"], datum_normal)))
            deviation = abs(math.degrees(math.acos(min(cosine, 1.0))) - 90.0)
            row["deviation_degrees"] = deviation
            row["verdict"] = "pass" if deviation <= 2.0 else "fail"
        rows.append(row)
    return rows


def fuse(
    capture: Path,
    workers: int,
    band: DepthBand | None,
    *,
    deprojection: tuple[IntrinsicsRecord, float],
    no_turntable: bool = False,
) -> tuple[np.ndarray, list[bytes], list[Rigid]]:
    """Register the capture: its points in one frame, its raw frames, its poses. READ-ONLY.

    ``band`` is the :class:`scanner3d.preparation.DepthBand` the preparation
    clips to, and ``deprojection`` the intrinsics and depth scale every point
    is made with - the calibrated ones the product uses, unless the caller
    asked for the factory record. The raw frames and the poses come back
    beside the cloud because the coherence check needs the images and the
    placement seed needs the orbit; both were thrown away before, and the
    numbers were wrong.
    """
    _src()
    from scanner3d.pipeline import reconstruct_sweep
    from scanner3d.preparation import RECONSTRUCT_POINTS, prepare_frame

    frames = sorted(capture.glob("frame_*.raw"))
    if not frames:
        raise SystemExit(f"{capture} holds no frame_*.raw")
    raws = [f.read_bytes() for f in frames]
    intrinsics, depth_scale = deprojection
    clouds = [
        prepare_frame(
            raw,
            target_points=RECONSTRUCT_POINTS,
            depth_band=band,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
        ).points
        for raw in raws
    ]
    from scanner3d.capture_sequence import frame_times
    from scanner3d.turntable import TurntableNotDetected, TurntableUnusable, register_turntable

    poses: list[Rigid] | None = None
    if not no_turntable:
        # The production path: one axis and one rate when the frames prove a
        # turn, the frame-by-frame chain otherwise - the gate measures what
        # an operator gets.
        try:
            turntable = register_turntable(
                clouds,
                raws=raws,
                depth_scale=depth_scale,
                times=frame_times(frames),
                backend="auto",
                on_progress=lambda line: print(line, file=sys.stderr),
            )
        except (TurntableNotDetected, TurntableUnusable) as exc:
            print(f"turntable model not used: {exc}", file=sys.stderr)
        else:
            print(turntable.statement(), file=sys.stderr)
            poses = list(turntable.poses)
    if poses is None:
        result = reconstruct_sweep(clouds, workers=workers, backend="auto")
        poses = list(result.poses)
    scan = np.array([p for c, t in zip(clouds, poses, strict=True) for p in t.apply_all(c)])
    return scan, raws, poses


def _frame_about(up: np.ndarray) -> np.ndarray:
    """A right-handed orthonormal frame whose third row is ``up``.

    Right-handed matters: the yaw search rotates the CAD about this frame's
    third axis and the result is mapped back through the frame's transpose,
    so a left-handed frame would hand back the CAD's MIRROR image.
    """
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])


def _plane_seed(
    scan: np.ndarray, rng: np.random.Generator, log: list[str]
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """The original seed: the scan's dominant plane fixes up, a density peak fixes the centre."""
    sample = scan[rng.choice(len(scan), min(40000, len(scan)), replace=False)]
    best = (0, None, None)
    runner_up = 0
    for _ in range(4000):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        n = n / length
        inliers = int((np.abs((sample - tri[0]) @ n) < 1.5).sum())
        if inliers > best[0]:
            runner_up = best[0]
            best = (inliers, n, tri[0])
    up = best[1]
    if up is None:
        raise SystemExit("no dominant plane: the target must stand on one")
    if np.dot(up, np.array([0.0, 0.0, -1.0])) < 0:
        up = -up
    log.append(
        f"support plane: {best[0]}/{len(sample)} inliers "
        f"(runner-up {runner_up}, margin {best[0] / max(runner_up, 1):.2f}x)"
    )
    frame = _frame_about(up)
    planar = scan @ frame.T
    height = planar[:, 2] - np.median(planar[:, 2])
    standing = planar[(height > 3.0) & (height < 60.0)]
    hist, xe, ye = np.histogram2d(standing[:, 0], standing[:, 1], bins=90)
    i, j = np.unravel_index(np.argmax(hist), hist.shape)
    cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    near = standing[(np.abs(standing[:, 0] - cx) < 70) & (np.abs(standing[:, 1] - cy) < 70)]
    log.append(f"density peak holds {len(near)} points")
    return frame, near, float(cx), float(cy), float(np.median(standing[:, 2]))


def _orbit_seed(
    scan: np.ndarray, poses: list[Rigid] | None, rng: np.random.Generator, log: list[str]
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    """Seed from the turntable axis the camera orbit gives, or None when there is none.

    A coherent turntable registration puts the camera centres on a circle
    about the spindle. Its normal is the axis, the turntable top is the
    dominant plane perpendicular to it among the points near it, and the
    target is what stands on that plane within 70 of the axis. None of that
    needs the scan's largest plane, which on a COHERENT cloud is as likely to
    be the desk under the turntable as the turntable itself.
    """
    if poses is None:
        return None
    _src()
    from scanner3d.coherence import camera_orbit

    orbit = camera_orbit(poses)
    if orbit is None:
        log.append("camera centres do not describe a circle; seeding from the support plane")
        return None
    if orbit.radius < ORBIT_MIN_RADIUS or orbit.residual_median > ORBIT_MAX_RESIDUAL:
        log.append(
            f"camera orbit (radius {orbit.radius:.1f}, circle residual median "
            f"{orbit.residual_median:.2f}) is not a turntable's; seeding from the support plane"
        )
        return None
    up = np.array(orbit.axis)
    axis_point = np.array(orbit.centre)
    relative = scan - axis_point
    height = relative @ up
    radial = np.linalg.norm(relative - np.outer(height, up), axis=1)
    near_axis = scan[radial < 110.0]
    if len(near_axis) < 1000:
        log.append(
            f"only {len(near_axis)} points within 110 of the axis; seeding from the support plane"
        )
        return None
    sample = near_axis[rng.choice(len(near_axis), min(30000, len(near_axis)), replace=False)]
    best_inliers, top = 0, None
    for _ in range(3000):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        n = n / length
        # The turntable top is perpendicular to its own spindle.
        if abs(n @ up) < 0.9:
            continue
        inliers = int((np.abs((sample - tri[0]) @ n) < 1.5).sum())
        if inliers > best_inliers:
            best_inliers, top = inliers, float(tri[0] @ up)
    if top is None:
        log.append("no plane perpendicular to the axis; seeding from the support plane")
        return None
    # The orbit centre lies in the plane of the camera centres, ABOVE the
    # turntable, so the turntable top sits at negative height along 'up'.
    # The circle's normal carries no sign of its own; flip it if it came out
    # pointing down.
    if top - float(axis_point @ up) > 0.0:
        up, top = -up, -top
    frame = _frame_about(up)
    planar = scan @ frame.T
    standing = planar[(planar[:, 2] - top > 3.0) & (planar[:, 2] - top < 60.0)]
    axis_2d = axis_point @ frame.T
    near = standing[
        (np.abs(standing[:, 0] - axis_2d[0]) < 70) & (np.abs(standing[:, 1] - axis_2d[1]) < 70)
    ]
    log.append(
        f"placement seeded from the camera orbit: radius {orbit.radius:.1f}, circle residual "
        f"median {orbit.residual_median:.2f} over {orbit.poses} poses; turntable top "
        f"{best_inliers}/{len(sample)} inliers; {len(near)} points standing on it within 70 "
        "of the axis"
    )
    return frame, near, float(axis_2d[0]), float(axis_2d[1]), float(top)


def place(
    cad: np.ndarray, scan: np.ndarray, seed: int = 5, poses: list[Rigid] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Put the CAD where the scan says the target is, then refine onto it.

    Seeded from the camera orbit when ``poses`` describe one (see
    :func:`_orbit_seed`), else from the scan's dominant plane and its density
    peak as before. The orbit seed is what makes a COHERENT cloud placeable:
    measured on the first coherent sweep this repository produced, the
    support-plane search picked between the turntable and the desk at a
    margin of 1.0x, the density peak landed on the turntable disc, and every
    seed converged 5-8 from the target with 40% of the CAD observed - while
    the same search on a smeared cloud "converged" happily at 0.96.
    """
    _src()
    from scanner3d.icp_plane import icp_point_to_plane

    rng = np.random.default_rng(seed)
    log: list[str] = []

    seeded = _orbit_seed(scan, poses, rng, log)
    if seeded is None:
        seeded = _plane_seed(scan, rng, log)
    frame, near, centre_x, centre_y, base_height = seeded

    placed, score = None, None
    for yaw in np.arange(0, 360, 3.0):
        t = math.radians(yaw)
        rot = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
        for dx, dy in ((0.0, 0.0), (10.0, 0.0), (-10.0, 0.0), (0.0, 10.0), (0.0, -10.0)):
            q = cad @ rot.T + np.array([centre_x + dx, centre_y + dy, base_height - 3.0])
            d = nearest(q[rng.choice(len(q), 900, replace=False)], near[::7])
            if score is None or float(np.median(d)) < score:
                placed, score = q, float(np.median(d))
    log.append(f"coarse placement: median {score:.2f} mm")

    for width, radius in REFINE_SCHEDULE:
        for step in range(PASSES_PER_WIDTH):
            # Against the FULL near set, not a subsample. Measured against
            # near[::5] this line read 1.99mm while the gate's own figure over
            # the whole cloud was 1.29mm, and that gap was read as the
            # refinement stalling when it was the progress metric thinning its
            # own reference. The k-d tree makes the honest version affordable.
            before = float(np.median(nearest(placed[::4], near)))
            chosen = near[nearest(near, placed[::6]) < radius]
            if len(chosen) < 800:
                log.append(f"refine @{width:g}mm: only {len(chosen)} points near model, stopping")
                break
            fit = icp_point_to_plane(
                [tuple(p) for p in placed[::3]],
                [tuple(p) for p in chosen[:: max(1, len(chosen) // 6000)]],
                max_correspondence_distance=width,
                # Fixed: the CORRESPONDENCE width is what narrows here. Shrinking the
                # downsample with it asks for a 0.5 mm grid over a 92 mm model, which
                # is 5.3M cells against registration.py's 4M ceiling - a cell per point,
                # reducing nothing.
                voxel_size=1.0,
                keep_fraction=0.85,
            )
            placed = np.array([fit.transform.apply(tuple(p)) for p in placed])
            after = float(np.median(nearest(placed[::4], near)))
            if step == 0 or abs(after - before) >= 0.02:
                log.append(
                    f"refine @{width:g}mm: {before:.2f} -> {after:.2f} mm, "
                    f"matched {fit.matched_fraction:.2f}"
                )
            if abs(after - before) < 0.02:
                break

    # Back to the caller's frame. Everything above works in the seed's frame,
    # because 'standing on the plane' and 'the density peak' only mean
    # anything there; `frame` is orthonormal, so its transpose is its inverse.
    # Returning plane-frame coordinates silently compared the model against a
    # scan in a different basis, which read as 0% of the CAD observed while the
    # refinement's own median said 1.99 mm - the two disagreeing is what caught
    # it.
    return placed @ frame, log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path, help="capture directory (read-only)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-turntable",
        action="store_true",
        help="register frame by frame even when the frames prove a full turn",
    )
    parser.add_argument(
        "--no-metric-correction",
        action="store_true",
        help="deproject with the factory intrinsics and depth scale unmodified, without "
        "this unit's measured metric correction (scanner3d.metric_correction), which the "
        "product applies by default. Run both to see what the correction is worth on the CAD",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="placement search seed; vary it to measure how much of a per-surface "
        "result is the scan and how much is where the search happened to land",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="reuse the fused cloud at this path, writing it first if absent (its poses go "
        "beside it as <cache>.poses.json). The fusion is the expensive half and is "
        "independent of --seed, so a repeatability sweep should not pay for it more than "
        "once.",
    )
    parser.add_argument(
        "--depth-band",
        nargs=2,
        type=float,
        metavar=("NEAR", "FAR"),
        help="clip preparation to this depth interval for registration, overriding "
        "depth_band under [registration] in the config file and the sensor's own 60-300. "
        "State the interval the subject occupies; see docs/target-gate.md",
    )
    args = parser.parse_args(argv)
    _src()
    from scanner3d.cli import resolve_depth_band
    from scanner3d.coherence import check_coherence
    from scanner3d.geometry import Rigid
    from scanner3d.metric_correction import calibrated_deprojection

    gates = json.loads((TARGET / "gates/tolerance-v1.json").read_text())
    tol = next(g for g in gates["gates"] if g["id"] == "planar_rms")["tolerance_mm"]
    cad, sid, names, print_record = load_reference()

    band, band_source = resolve_depth_band(args.depth_band)
    print(f"registration depth band: {band.describe()} (from {band_source})", file=sys.stderr)
    # The same deprojection the window and `scanner3d reconstruct` use, so the
    # gate measures what an operator gets; --no-metric-correction is the control.
    intrinsics, depth_scale, correction = (
        calibrated_deprojection(None) if args.no_metric_correction else calibrated_deprojection()
    )
    print(
        "deprojection: "
        + (
            correction.statement()
            if correction is not None
            else "factory intrinsics and depth scale, unmodified (--no-metric-correction)"
        ),
        file=sys.stderr,
    )

    raws: list[bytes] | None = None
    poses: list[Rigid] | None = None
    pose_cache = (
        None if args.cache is None else args.cache.with_name(args.cache.name + ".poses.json")
    )
    if args.cache is not None and args.cache.exists():
        scan = np.load(args.cache)
        print(f"fused cloud from {args.cache} ({len(scan)} points)", file=sys.stderr)
        if pose_cache is not None and pose_cache.exists():
            poses = [
                Rigid.of(tuple(map(tuple, p["rotation"])), tuple(p["translation"]))
                for p in json.loads(pose_cache.read_text())
            ]
            raws = [f.read_bytes() for f in sorted(args.capture.glob("frame_*.raw"))]
            if len(raws) != len(poses):
                print(f"{pose_cache} does not match the capture's frames; ignored", file=sys.stderr)
                raws = poses = None
        else:
            print(
                "cached cloud carries no poses: coherence is not checked and the placement is "
                "seeded from the support plane",
                file=sys.stderr,
            )
    else:
        scan, raws, poses = fuse(
            args.capture,
            args.workers,
            band,
            deprojection=(intrinsics, depth_scale),
            no_turntable=args.no_turntable,
        )
        if args.cache is not None and pose_cache is not None:
            np.save(args.cache, scan)
            pose_cache.write_text(json.dumps([p.as_dict() for p in poses]) + "\n")

    coherence = None
    if raws is not None and poses is not None:
        coherence = check_coherence(
            raws,
            poses,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            depth_band=band,
        )
        print(f"coherence: {coherence.statement()}", file=sys.stderr)
        if coherence.coherent is False:
            print(
                "\nREGISTRATION INCOHERENT: frames one turntable revolution apart have the same "
                "depth image but poses that disagree, so the fused cloud is a smear of misplaced "
                "frames and nothing measured on it describes the target. Nothing is reported. "
                "Clear the background behind the turntable, or state the subject's depth band "
                "with --depth-band; see docs/target-gate.md.",
                file=sys.stderr,
            )
            return 2

    placed, log = place(cad, scan, seed=args.seed, poses=poses)
    d = nearest(placed, scan)
    observed = d < OBSERVED_RADIUS

    # A placement that reached almost nothing is a failed registration, not a
    # scan that missed the object. Judging it would report per-surface numbers
    # for a model sitting somewhere the target is not - the exact "plausible
    # number from a broken measurement" this repository refuses everywhere else.
    if observed.mean() < MIN_OBSERVED_TO_JUDGE:
        for line in log:
            print(f"  {line}", file=sys.stderr)
        print(
            f"\nREGISTRATION FAILED: only {observed.mean():.1%} of CAD points landed within "
            f"{OBSERVED_RADIUS} mm of any scan point, under the {MIN_OBSERVED_TO_JUDGE:.0%} this "
            "tool requires before it will judge anything. Nothing is reported. The placement "
            "search is seeded from the largest plane in the scene; check the margin on the "
            "support-plane line above, and see docs/target-gate.md.",
            file=sys.stderr,
        )
        return 2

    surfaces = []
    for s in sorted({int(x) for x in sid}):
        mask = sid == s
        if mask.sum() < 30:
            continue
        ds, seen = d[mask], observed[mask]
        row = {
            "surface": names.get(s, str(s)),
            "cad_points": int(mask.sum()),
            "observed_fraction": float(seen.mean()),
            "rms_mm": None,
            "p95_mm": None,
            "accuracy_verdict": "not_observed",
        }
        if seen.sum() >= 15:
            rms = float(np.sqrt((ds[seen] ** 2).mean()))
            row["rms_mm"] = rms
            row["p95_mm"] = float(np.percentile(ds[seen], 95))
            row["accuracy_verdict"] = "pass" if rms <= tol else "fail"
        surfaces.append(row)

    judged = [r for r in surfaces if r["rms_mm"] is not None]

    # The CAD normals carried into the scan's frame. place() applies a rigid
    # transform, so the rotation is recoverable from where the points ended up,
    # and normals rotate with it.
    centred_cad = cad - cad.mean(0)
    centred_placed = placed - placed.mean(0)
    u, _, vt = np.linalg.svd(centred_placed.T @ centred_cad)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    placed_normals = ((u @ vt) @ read_reference_ply_normals().T).T

    faces = orthogonality(scan, placed, sid, placed_normals, names)
    angled = [f for f in faces if "deviation_degrees" in f]
    report = {
        "capture": str(args.capture),
        "tolerance_file": "gates/tolerance-v1.json",
        "deprojection": {
            "fx": intrinsics.values["fx"],
            "fy": intrinsics.values["fy"],
            "depth_scale": depth_scale,
            "metric_correction": None if correction is None else correction.as_dict(),
            "note": "a --cache cloud was deprojected by whichever run wrote it",
        },
        "print_record_formal_gate_ready": print_record["capture_readiness"][
            "ready_for_formal_physical_gate"
        ],
        "observed_radius_mm": OBSERVED_RADIUS,
        "orthogonality": {
            "gate_degrees": 2.0,
            "datum": GATE_DATUM,
            "faces": faces,
            "worst_deviation_degrees": (
                max(f["deviation_degrees"] for f in angled) if angled else None
            ),
            "verdict": "pass"
            if angled and all(f["verdict"] == "pass" for f in angled)
            else ("fail" if angled else "not_observed"),
        },
        "accuracy": {
            "gate_mm": tol,
            "rms_mm": float(np.sqrt((d[observed] ** 2).mean())),
            "p95_mm": float(np.percentile(d[observed], 95)),
            "surfaces_passing": sum(1 for r in judged if r["accuracy_verdict"] == "pass"),
            "surfaces_judged": len(judged),
            "verdict": "pass"
            if judged and all(r["accuracy_verdict"] == "pass" for r in judged)
            else "fail",
        },
        "completeness": {
            "gated": False,
            "why_not": (
                "What limits completeness on this rig is not yet known - see "
                "docs/target-gate.md. Gating on a quantity whose failure mode is not understood "
                "produces refusals nobody can act on, so the unobserved surfaces are named "
                "instead."
            ),
            "observed_fraction": float(observed.mean()),
            "unobserved_surfaces": [
                r["surface"] for r in surfaces if r["observed_fraction"] < 0.05
            ],
        },
        "surfaces": surfaces,
        "registration": log,
        "depth_band": {**band.as_dict(), "source": band_source},
        "coherence": None if coherence is None else coherence.as_dict(),
        "limitations": [
            "Not an accuracy claim. The print record's formal-gate readiness is reported above; "
            "while it is false, the ground truth's own uncertainty is unmeasured.",
            "Rigid registration absorbs any global scale error, so nothing here tests the depth "
            "scale.",
            "Accuracy is measured only where a surface was observed. A capture that saw little "
            "can post a good accuracy number, which is why completeness is reported beside it "
            "and never folded into it.",
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    a = report["accuracy"]
    print(f"\n{args.capture.name}: {len(scan)} scan points")
    for line in log:
        print(f"  {line}")
    print(f"\n{'surface':<20} {'CAD':>5} {'seen':>6} {'RMS':>7} {'p95':>7}   verdict")
    for r in sorted(surfaces, key=lambda r: r["observed_fraction"]):
        if r["rms_mm"] is None:
            print(
                f"{r['surface']:<20} {r['cad_points']:>5} {r['observed_fraction']:>5.0%}"
                f" {'':>7} {'':>7}   NOT OBSERVED"
            )
        else:
            print(
                f"{r['surface']:<20} {r['cad_points']:>5} {r['observed_fraction']:>5.0%}"
                f" {r['rms_mm']:>7.2f} {r['p95_mm']:>7.2f}   {r['accuracy_verdict'].upper()}"
            )
    o = report["orthogonality"]
    print(f"\n{'face':<20} {'scan pts':>8} {'plane RMS':>10} {'vs base_top':>12}   verdict")
    for f in o["faces"]:
        angle = f"{f['deviation_degrees']:.2f} deg" if "deviation_degrees" in f else ""
        rms = f"{f['plane_rms_mm']:.2f} mm" if "plane_rms_mm" in f else ""
        print(
            f"{f['face']:<20} {f['scan_points']:>8} {rms:>10} {angle:>12}   {f['verdict'].upper()}"
        )
    if o["worst_deviation_degrees"] is not None:
        print(
            f"ORTHOGONALITY  worst {o['worst_deviation_degrees']:.2f} deg against a "
            f"{o['gate_degrees']:.0f} deg gate, {len(angled)}/{len(GATE_FACES)} faces "
            f"resolvable  -> {o['verdict'].upper()}"
        )
    else:
        print(
            f"ORTHOGONALITY  not evaluable: 0/{len(GATE_FACES)} outer faces were seen as "
            "surfaces rather than slivers. A single-axis sweep grazes them; see "
            "FACE_FLATNESS_RATIO in this tool."
        )

    print(
        f"\nACCURACY  RMS {a['rms_mm']:.2f} mm, p95 {a['p95_mm']:.2f} mm, "
        f"{a['surfaces_passing']}/{a['surfaces_judged']} surfaces inside {tol} mm  "
        f"-> {a['verdict'].upper()}"
    )
    print(
        f"COMPLETENESS  {report['completeness']['observed_fraction']:.0%} of CAD points "
        f"observed (reported, not gated)"
    )
    if report["completeness"]["unobserved_surfaces"]:
        print(f"  never observed: {', '.join(report['completeness']['unobserved_surfaces'])}")
    if not report["print_record_formal_gate_ready"]:
        print(
            "\nThis is a development metric, not an accuracy claim: the print record is "
            "ready_for_formal_physical_gate = false."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
