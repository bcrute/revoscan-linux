"""Replay a real capture's raw frames through :mod:`scanner3d.return_health`, read-only.

Deliberately NOT a ``test_`` module: pytest does not collect it. It is the
harness :mod:`tests.test_preview_health_report` drives and that a handoff runs
when the numbers are wanted, in the role :mod:`tests.tracking_replay` plays for
the tracker and :mod:`tests.quality_replay` for the quality module.

Run it with::

    .venv/bin/python -m tests.preview_health_replay
    .venv/bin/python -m tests.preview_health_replay --capture SOME/CAPTURE
    .venv/bin/python -m tests.preview_health_replay --frames 0,80,165

STRICTLY READ-ONLY. This harness calls exactly four things on the capture
directory and the frames inside it: ``Path.is_dir``, ``Path.glob``,
``Path.stat`` and ``Path.read_bytes``. It never calls ``open`` in a writing
mode, never creates a directory or a temporary file, and touches no capture
write path, no ``bridge.json`` publication and no device node. It also PROVES
it: every frame file's size, modification time and SHA-256 are recorded before
the replay and again after it, and the report carries both plus whether they
agree. See :func:`verify_unchanged`.

What it is for
---------------
On 2026-08-28 the owner reported that the false-coloured depth preview of
``/home/ben/scans/scan/20260828-110641`` had "stopped showing the object". It
had not. The frames themselves had stopped returning depth, and this harness is
how that is inspected without a display: for any frame it reports how much came
back, which quadrant lost it, and the footprint of what is left, and across the
whole capture it reports where the bounded return-health verdict changed.

What it can prove, and what it cannot
--------------------------------------
It can prove, for one capture, exactly how many samples were the sensor's
no-return code and where they were; those are counts of bytes on disk and are
not open to interpretation. It can show which frames the shared verdict
vocabulary would have warned on, at the same bounded window the live front ends
use.

It CANNOT say why the returns went away. Raw code 0 means the firmware reported
no depth for that pixel and nothing more: it does not distinguish a surface out
of range, out of the projector's cone, too dark, too shiny, occluded, or a
scanner pointed at nothing. The black region in that capture is raw no-return
evidence and a REMAINING optical/capture limitation, not restored geometry and
not a proved projector or exposure defect - see :data:`LIMITATIONS`. Nothing
here fills, interpolates or invents a sample, and no physical unit is implied
by any number it prints.

Exit codes
-----------
``0`` for a completed replay, ``2`` when the harness could not run (a directory
that is not one, a frame that is not this mode's shape, a frame index that is
not in the capture), and ``3`` if a capture file changed under it while it ran,
which is a statement about the machine rather than about the capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scanner3d.return_health import (
    HISTORY_FRAMES,
    SCHEMA_VERSION,
    ReturnHealthTracker,
    spatial_returns,
    spatial_summary,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

#: The capture the report was made about, and this harness's default.
DEFAULT_CAPTURE: Final = Path("/home/ben/scans/scan/20260828-110641")

#: The frames Codex measured first, and the ones the report leads with unless
#: ``--frames`` says otherwise.
DEFAULT_FRAMES: Final = (0, 80, 165)

#: Written into every report, because a reader who sees a mostly-empty frame
#: and a confident verdict beside it is exactly the reader who could take this
#: for a diagnosis.
LIMITATIONS: Final = (
    "Raw code 0 is the sensor reporting no depth for that pixel. This harness counts those "
    "samples and says where they are; it never fills, interpolates or reconstructs them.",
    "The large no-return region in this capture is raw evidence and a remaining optical/capture "
    "limitation. It is not restored geometry, and it does not prove a projector, exposure, "
    "material or firmware defect: the raw bytes cannot distinguish those.",
    "No physical unit is implied anywhere in this report. Depth codes are the sensor's own "
    "uint16 values.",
    "A verdict here is advisory, exactly as it is in the live front ends: it accepts or refuses "
    "no frame, stops no capture, and changes no reconstruction threshold.",
)

#: One frame, unpacked in one call - the same shape and format the live viewer
#: uses, derived from the same single source of truth.
SAMPLES_FORMAT: Final = f"<{FRAME_WIDTH * FRAME_HEIGHT}H"


class ReplayRefused(Exception):
    """The harness could not run against what it was given. Nothing was read."""


def frame_paths(capture: Path) -> list[Path]:
    """Every ``frame_NNNN.raw`` in ``capture``, in name order. Read-only."""
    if not capture.is_dir():
        raise ReplayRefused(f"{capture} is not a directory; nothing was read")
    paths = sorted(capture.glob("frame_*.raw"))
    if not paths:
        raise ReplayRefused(f"{capture} holds no frame_*.raw files; nothing was read")
    return paths


def fingerprint(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    """Size, modification time and SHA-256 of every frame, by file name.

    Taken before and after the replay, so "this harness did not write to the
    capture" is a measured claim rather than a promise in a docstring.
    """
    return {
        path.name: {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    }


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


def samples_of(path: Path) -> tuple[int, ...]:
    """One frame's raw samples. Refuses a buffer that is not this mode's frame."""
    raw = path.read_bytes()
    if len(raw) != EXPECTED_FRAME_BYTES:
        raise ReplayRefused(
            f"{path.name} is {len(raw)} bytes, not the {EXPECTED_FRAME_BYTES} of a "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} Y16 frame; nothing was measured from it"
        )
    return struct.unpack(SAMPLES_FORMAT, raw)


def yield_of(samples: tuple[int, ...]) -> float:
    """The returned share of one frame, counted rather than sampled."""
    return (len(samples) - samples.count(0)) / len(samples)


def frame_report(path: Path, index: int) -> dict[str, object]:
    """The full spatial record for one named frame."""
    spatial = spatial_returns(samples_of(path), FRAME_WIDTH, FRAME_HEIGHT)
    return {
        "index": index,
        "file": path.name,
        "spatial": spatial.as_dict(),
        "summary": spatial_summary(spatial),
    }


def run(
    capture: Path = DEFAULT_CAPTURE,
    *,
    frames: Sequence[int] = DEFAULT_FRAMES,
    history: int = HISTORY_FRAMES,
) -> dict[str, object]:
    """Replay ``capture`` and return the whole report as plain data.

    READ-ONLY, and the four calls named in this module's docstring are the only
    ones made against the capture: ``is_dir`` and ``glob`` in
    :func:`frame_paths`, ``stat`` and ``read_bytes`` in :func:`fingerprint` and
    :func:`samples_of`. There is no ``open`` for writing anywhere below.
    """
    paths = frame_paths(capture)
    before = fingerprint(paths)

    for index in frames:
        if not 0 <= index < len(paths):
            raise ReplayRefused(
                f"frame {index} is not in this capture, which holds {len(paths)} frames "
                f"(0-{len(paths) - 1}); nothing was measured"
            )

    selected = [frame_report(paths[index], index) for index in frames]

    # The whole sequence through the same bounded window the live front ends
    # use, so the verdicts below are the ones an operator would have been shown.
    tracker = ReturnHealthTracker(history=history)
    transitions: list[dict[str, object]] = []
    previous: str | None = None
    yields: list[float] = []
    for index, path in enumerate(paths):
        current = yield_of(samples_of(path))
        yields.append(current)
        health = tracker.update(current)
        if health.verdict != previous:
            transitions.append({"index": index, "file": path.name, **health.as_dict()})
            previous = health.verdict

    after = fingerprint(paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "capture": str(capture),
        "frames_on_disk": len(paths),
        "history_frames": history,
        "selected_frames": selected,
        "verdict_transitions": transitions,
        "yield_fractions": yields,
        "read_only": verify_unchanged(before, after),
        "limitations": list(LIMITATIONS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument(
        "--frames",
        type=str,
        default=",".join(str(index) for index in DEFAULT_FRAMES),
        help="comma-separated frame indices to report spatially",
    )
    parser.add_argument("--history", type=int, default=HISTORY_FRAMES)
    args = parser.parse_args(argv)

    try:
        indices = [int(part) for part in args.frames.split(",") if part.strip()]
    except ValueError:
        parser.error("--frames must be comma-separated integers")

    try:
        report = run(args.capture, frames=indices, history=args.history)
    except (ReplayRefused, OSError, ValueError) as exc:
        sys.stderr.write(f"preview_health_replay: {exc}\n")
        return 2

    sys.stdout.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    read_only = report["read_only"]
    assert isinstance(read_only, dict)
    return 0 if read_only["unchanged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
