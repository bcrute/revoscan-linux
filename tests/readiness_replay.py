"""Replay a real capture through :mod:`scanner3d.reconstruction_readiness`, read-only.

Deliberately NOT a ``test_`` module: pytest does not collect it. It is the
harness a handoff runs when the numbers are wanted, in the role
:mod:`tests.preview_health_replay` plays for return health and
:mod:`tests.quality_replay` for the quality module.

Run it with::

    .venv/bin/python -m tests.readiness_replay
    .venv/bin/python -m tests.readiness_replay --capture SOME/CAPTURE
    .venv/bin/python -m tests.readiness_replay --stride 4 --limit 51

STRICTLY READ-ONLY. This harness calls exactly four things on the capture
directory and the files inside it: ``Path.is_dir``, ``Path.glob``, ``Path.stat``
and ``Path.read_bytes``. It never opens a path for writing, creates no
directory and no temporary file, exports nothing, touches no device node and no
other repository. It also PROVES it: every file's size, modification time and
SHA-256 is recorded before the replay and again after it, and the report carries
both plus whether they agree - the same
:func:`tests.preview_health_replay.verify_unchanged` contract.

What it can prove, and what it cannot
--------------------------------------
It can say, for one capture, exactly what each readiness gate measures and
whether it passes. Those are computed from bytes on disk by the same functions
the window, the command line and the export boundary use, at the same default
policy.

It CANNOT say the capture is accurate. Every gate here is internal consistency
(see :data:`scanner3d.reconstruction_readiness.LIMITATIONS`), and the pose and
coverage gates are measured against a registration performed HERE, so they
describe the trajectory this run accepted rather than where the scanner
physically was.

It also cannot reach the later gates on a capture the registration refuses.
That is not a limitation of the gates; it is the earlier boundary doing its
job, and this harness reports exactly where it stopped rather than registering
a prefix and presenting it as the whole capture.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

from scanner3d.capture_sequence import CaptureSequenceUnreadable, read_capture_sequence
from scanner3d.errors import ScannerError
from scanner3d.pointcloud import sha256_of
from scanner3d.preparation import RECONSTRUCT_POINTS, prepare_frame
from scanner3d.reconstruction_readiness import (
    DEFAULT_POLICY,
    coverage_evidence,
    pose_evidence,
    tracking_evidence,
)

#: The capture this task was asked to evaluate. On 2026-08-28 it lost tracking
#: at displayed frame 53 and kept writing until 176 - see
#: :doc:`../docs/tracking-loss`, which this folder is the reason for.
DEFAULT_CAPTURE: Final = Path("/home/ben/scans/scan/20260828-105933")

#: Nothing here interprets a number as a physical length.
LIMITATIONS: Final[tuple[str, ...]] = (
    "read-only. No file in the capture was written, renamed, deleted or created, and the "
    "before/after fingerprints in this report are the evidence for that",
    "the poses these gates judge were fitted by this run. A pose the registration got wrong "
    "and was confident about is measured here as ordinary motion",
    "no number here has a physical unit. Every length is in the capture's own depth units, "
    "whose physical meaning this repository has not established",
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


def replay(capture: Path, *, stride: int, limit: int | None, backend: str) -> dict[str, object]:
    """Every gate this capture can reach, and where it stopped if it stopped."""
    if not capture.is_dir():
        raise ReplayRefused(f"{capture} is not a directory; nothing was read")

    report: dict[str, object] = {
        "capture": str(capture),
        "policy": DEFAULT_POLICY.as_dict(),
        "settings": {"stride": stride, "limit": limit, "backend": backend},
        "limitations": list(LIMITATIONS),
    }

    # Gate 1 first and on its own, because it is the one gate that answers from
    # the folder alone and is therefore the one a capture the registration
    # refuses can still be judged by.
    tracking, tracking_failures = tracking_evidence(capture)
    report["tracking"] = {
        "evidence": tracking.as_dict(),
        "failures": [failure.as_dict() for failure in tracking_failures],
    }

    try:
        sequence = read_capture_sequence(capture).strided(max(1, stride))
    except CaptureSequenceUnreadable as exc:
        report["stopped_at"] = "capture_reader"
        report["stopped_because"] = str(exc)
        return report
    paths = list(sequence.paths)
    if limit is not None:
        paths = paths[:limit]
    report["frames"] = {"offered": len(sequence.paths), "used": len(paths)}
    if len(paths) < 3:
        report["stopped_at"] = "frame_count"
        report["stopped_because"] = f"{len(paths)} usable frames; a sweep needs at least three"
        return report

    started = time.perf_counter()
    clouds = [
        prepare_frame(path.read_bytes(), target_points=RECONSTRUCT_POINTS).points for path in paths
    ]
    prepared_s = time.perf_counter() - started

    # Imported here rather than at module scope: registering is the expensive
    # half of this harness and a run that stops at the reader above should not
    # have paid for importing it.
    from scanner3d.pipeline import reconstruct_sweep

    started = time.perf_counter()
    try:
        result = reconstruct_sweep(clouds, backend=backend, bridges=sequence.boundaries())
    except ScannerError as exc:
        report["stopped_at"] = "registration"
        report["stopped_because"] = str(exc)
        report["timing"] = {"prepare_s": prepared_s}
        return report
    registered_s = time.perf_counter() - started

    started = time.perf_counter()
    poses, pose_failures = pose_evidence(clouds, result.poses)
    pose_s = time.perf_counter() - started
    started = time.perf_counter()
    coverage, coverage_failures = coverage_evidence(clouds, result.poses)
    coverage_s = time.perf_counter() - started

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
    report["reasons"] = [
        failure.reason for failure in (*tracking_failures, *pose_failures, *coverage_failures)
    ]
    report["ready_before_fusion"] = not report["reasons"]
    # Fusion, extraction and the surface gate are deliberately NOT run here.
    # They are the export's work and this harness writes nothing; what the
    # surface gate would say about this capture is stated as not measured
    # rather than guessed at.
    report["surface"] = {
        "measured": False,
        "why": (
            "the surface gate needs a fused TSDF volume, which only an export produces. This "
            "harness writes nothing, so it reports the gate as not run rather than inventing "
            "a verdict for it"
        ),
    }
    report["timing"] = {
        "prepare_s": round(prepared_s, 3),
        "register_s": round(registered_s, 3),
        "pose_gate_s": round(pose_s, 3),
        "coverage_gate_s": round(coverage_s, 3),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args(argv)

    before = fingerprint(args.capture)
    try:
        report = replay(args.capture, stride=args.stride, limit=args.limit, backend=args.backend)
    except ReplayRefused as exc:
        print(json.dumps({"refused": str(exc)}, indent=2))
        return 2
    after = fingerprint(args.capture)
    report["read_only"] = verify_unchanged(before, after)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["read_only"]["unchanged"] else 3  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
