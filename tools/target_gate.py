"""Measure a capture of reconstruction-target-v1 against its CAD, and judge it.

Run every session:

    .venv/bin/python tools/target_gate.py /home/ben/scans/scan/<capture>
    .venv/bin/python tools/target_gate.py <capture> --json

STRICTLY READ-ONLY on the capture. Reads raw frames, writes nothing into the
capture directory, and touches no device.

What it judges, and what it refuses to judge
--------------------------------------------
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
from typing import Final

import numpy as np

ROOT: Final = Path(__file__).resolve().parent.parent
TARGET: Final = ROOT / "reference/targets/reconstruction_v1"

#: A CAD point with no scan point this close was not observed, and is excluded
#: from accuracy rather than counted as a large error.
#:
#: Three millimetres: above the 1.32 mm agreement a good capture achieves and
#: above its 2.51 mm p95, so an observed surface is not mistaken for a missing
#: one; well below the 4.27 mm that the nearest scan point to an unobserved face
#: was measured at, so a missing surface is not mistaken for a bad one.
OBSERVED_RADIUS: Final = 3.0

#: Scan points this far from the placed model join the refinement. Wide enough
#: to pull in the surface either side of a 1.5 mm registration error, narrow
#: enough to exclude the turntable the model stands on.
REFINE_RADIUS: Final = 5.0

MAX_REFINEMENTS: Final = 6


def load_reference() -> tuple[np.ndarray, np.ndarray, dict[int, str], dict]:
    """CAD points shrinkage-adjusted as the print record predeclared, plus ids."""
    manifest = json.loads((TARGET / "reconstruction-target-v1.json").read_text())
    prints = sorted((TARGET / "prints").glob("physical-print-*.json"))
    if not prints:
        raise SystemExit("no print record: cannot know this print's shrinkage")
    record = json.loads(prints[-1].read_text())
    scale = record["dense_reference_adjustment"]

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
    ply = np.frombuffer(raw, dtype=dtype, count=count, offset=head)
    cad = np.c_[
        ply["x"] * scale["scale_x"], ply["y"] * scale["scale_y"], ply["z"] * scale["scale_z"]
    ].astype(float)
    names = {v["id"]: k for k, v in manifest["surface_ids"].items()}
    return cad, np.array(ply["sid"]), names, record


def nearest(a: np.ndarray, b: np.ndarray, chunk: int = 400) -> np.ndarray:
    """Distance from every row of ``a`` to the closest row of ``b``."""
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        out[i : i + chunk] = np.linalg.norm(a[i : i + chunk, None, :] - b[None, :, :], axis=2).min(
            axis=1
        )
    return out


def fuse(capture: Path, workers: int) -> np.ndarray:
    """Register the capture and return its points in one frame. READ-ONLY."""
    sys.path.insert(0, str(ROOT / "src"))
    from scanner3d.pipeline import reconstruct_sweep
    from scanner3d.preparation import RECONSTRUCT_POINTS, prepare_frame

    frames = sorted(capture.glob("frame_*.raw"))
    if not frames:
        raise SystemExit(f"{capture} holds no frame_*.raw")
    clouds = [
        prepare_frame(f.read_bytes(), target_points=RECONSTRUCT_POINTS).points for f in frames
    ]
    result = reconstruct_sweep(clouds, workers=workers, backend="auto")
    return np.array([p for c, t in zip(clouds, result.poses, strict=True) for p in t.apply_all(c)])


def place(cad: np.ndarray, scan: np.ndarray, seed: int = 5) -> tuple[np.ndarray, list[str]]:
    """Put the CAD where the scan says the target is, then refine onto it."""
    sys.path.insert(0, str(ROOT / "src"))
    from scanner3d.icp_plane import icp_point_to_plane

    rng = np.random.default_rng(seed)
    log: list[str] = []

    sample = scan[rng.choice(len(scan), min(40000, len(scan)), replace=False)]
    best = (0, None, None)
    for _ in range(1200):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        n = n / length
        inliers = int((np.abs((sample - tri[0]) @ n) < 1.5).sum())
        if inliers > best[0]:
            best = (inliers, n, tri[0])
    up = best[1]
    if up is None:
        raise SystemExit("no dominant plane: the target must stand on one")
    if np.dot(up, np.array([0.0, 0.0, -1.0])) < 0:
        up = -up
    log.append(f"support plane: {best[0]}/{len(sample)} inliers")

    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    frame = np.stack([e1, np.cross(up, e1), up])

    planar = scan @ frame.T
    height = planar[:, 2] - np.median(planar[:, 2])
    standing = planar[(height > 3.0) & (height < 60.0)]
    hist, xe, ye = np.histogram2d(standing[:, 0], standing[:, 1], bins=90)
    i, j = np.unravel_index(np.argmax(hist), hist.shape)
    cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    near = standing[(np.abs(standing[:, 0] - cx) < 70) & (np.abs(standing[:, 1] - cy) < 70)]
    log.append(f"density peak holds {len(near)} points")

    placed, score = None, None
    for yaw in np.arange(0, 360, 3.0):
        t = math.radians(yaw)
        rot = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
        q = cad @ rot.T + np.array([cx, cy, np.median(standing[:, 2]) - 3.0])
        d = nearest(q[rng.choice(len(q), 900, replace=False)], near[::7])
        if score is None or float(np.median(d)) < score:
            placed, score = q, float(np.median(d))
    log.append(f"coarse placement: median {score:.2f} mm")

    for step in range(MAX_REFINEMENTS):
        before = float(np.median(nearest(placed[::4], near[::5])))
        chosen = near[nearest(near, placed[::6]) < REFINE_RADIUS]
        if len(chosen) < 800:
            log.append(f"refine {step}: only {len(chosen)} points near the model, stopping")
            break
        fit = icp_point_to_plane(
            [tuple(p) for p in placed[::3]],
            [tuple(p) for p in chosen[:: max(1, len(chosen) // 6000)]],
            max_correspondence_distance=8.0,
            voxel_size=1.0,
            keep_fraction=0.85,
        )
        placed = np.array([fit.transform.apply(tuple(p)) for p in placed])
        after = float(np.median(nearest(placed[::4], near[::5])))
        log.append(
            f"refine {step}: {before:.2f} -> {after:.2f} mm, matched {fit.matched_fraction:.2f}"
        )
        if abs(after - before) < 0.02:
            break

    return placed, log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path, help="capture directory (read-only)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    gates = json.loads((TARGET / "gates/tolerance-v1.json").read_text())
    tol = next(g for g in gates["gates"] if g["id"] == "planar_rms")["tolerance_mm"]
    cad, sid, names, print_record = load_reference()

    scan = fuse(args.capture, args.workers)
    placed, log = place(cad, scan)
    d = nearest(placed, scan)
    observed = d < OBSERVED_RADIUS

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
    report = {
        "capture": str(args.capture),
        "tolerance_file": "gates/tolerance-v1.json",
        "print_record_formal_gate_ready": print_record["capture_readiness"][
            "ready_for_formal_physical_gate"
        ],
        "observed_radius_mm": OBSERVED_RADIUS,
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
