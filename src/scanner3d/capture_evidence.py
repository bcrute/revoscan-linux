"""Bounded diagnostic journal, deliberately outside the reconstructable scan.

Host delivery times are not exposure timestamps. A diagnostic raw copy is not
an accepted frame, and this module never publishes a segment or a pose. Failed
or capped evidence is reported, not allowed to interrupt device cleanup.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TextIO

from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES
from scanner3d.v4l2_stream import FrameRecord

JOURNAL_NAME = "capture-evidence.jsonl"
MAX_RAW_FRAMES = 256
MAX_EVENTS = 30000
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
FOOTER_RESERVE = 4096


class CaptureEvidence:
    """Single capture-thread writer; exclusive artifacts and explicit limits."""

    def __init__(
        self, root: Path, metadata: dict[str, object], notify: Callable[[str], None]
    ) -> None:
        self.root = root
        self.directory = root / "diagnostics"
        self.notify = notify
        self.stream: TextIO | None = None
        self.failed = False
        self.limited = False
        self.closed = False
        self.events = self.bytes_written = self.deliveries = 0
        self.raw_saved = self.raw_omitted = 0
        self.retained: set[int] = set()
        try:
            self.directory.mkdir(exist_ok=True)
            self.stream = (self.directory / JOURNAL_NAME).open("x", encoding="utf-8")
            self.event(
                "start",
                schema_version=1,
                metadata=metadata,
                limits=dict(raw_frames=MAX_RAW_FRAMES, events=MAX_EVENTS, bytes=MAX_JOURNAL_BYTES),
                limitations=[
                    "Diagnostic files never authorize reconstruction or resolve tracking loss.",
                    "monotonic_s is host delivery time, not an exposure or hardware timestamp.",
                    "Complete deliveries are journaled live; short deliveries are appended only "
                    "if the stream returns its summary. An exception can leave shorts unknown.",
                    "Raw recovery evidence is capped across the entire capture; omissions count.",
                    "Device exposure/gain/temperature readbacks are unavailable in this GUI path.",
                ],
            )
        except OSError as error:
            self._failure(error)

    def _notice(self, text: str) -> None:
        # A GUI notification failure must not skip STREAMOFF/close either.
        with suppress(Exception):
            self.notify(text)

    def _failure(self, error: Exception) -> None:
        if not self.failed:
            self.failed = True
            self._notice(f"Capture diagnostics incomplete: {type(error).__name__}: {error}")

    def event(self, kind: str, **fields: object) -> None:
        if self.closed or self.failed or self.stream is None or self.limited:
            return
        try:
            line = json.dumps(dict(event=kind, **fields), allow_nan=False, separators=(",", ":"))
            size = len(line.encode("utf-8")) + 1
            if self.events >= MAX_EVENTS or self.bytes_written + size > (
                MAX_JOURNAL_BYTES - FOOTER_RESERVE
            ):
                self.limited = True
                self._notice("Capture diagnostic journal limit reached; later evidence omitted.")
                return
            self.stream.write(line + "\n")
            self.stream.flush()
            self.events += 1
            self.bytes_written += size
        except (OSError, ValueError, TypeError) as error:
            self._failure(error)

    def delivery(self, raw: bytes, record: FrameRecord, mode: str) -> None:
        self.deliveries += 1
        recovery = mode in ("recovering", "relocalizing", "lost")
        if self.closed or self.failed or self.limited:
            self.raw_omitted += int(recovery)
            return
        diagnostic_path: str | None = None
        try:
            if len(raw) != EXPECTED_FRAME_BYTES or not record.complete:
                raise ValueError("Complete delivery callback received an incomplete raw frame")
            if recovery:
                diagnostic_path = self.retain_recovery(raw, record)
            self.event(
                "delivery",
                record=record.as_dict(),
                monotonic_s=record.monotonic,
                mode=mode,
                sha256=hashlib.sha256(raw).hexdigest(),
                diagnostic_raw=diagnostic_path,
            )
        except (OSError, ValueError, TypeError) as error:
            self.raw_omitted += int(recovery)
            self._failure(error)

    def retain_recovery(self, raw: bytes, record: FrameRecord) -> str | None:
        """Also retain a delivery that *enters* relocalization after preview.

        Idempotent for retained frames; this covers the state-transition frame
        without duplicating its delivery record or ever naming it frame_*.raw.
        """
        path = self.directory / f"delivery_{record.index:06d}.raw"
        relative = str(path.relative_to(self.root))
        if record.index in self.retained:
            return relative
        if self.closed or self.failed or self.limited or self.raw_saved >= MAX_RAW_FRAMES:
            self.raw_omitted += 1
            if self.raw_omitted == 1 and not self.failed:
                self._notice("Recovery raw evidence limit reached; later raw copies omitted.")
            return None
        try:
            if len(raw) != EXPECTED_FRAME_BYTES or not record.complete:
                raise ValueError("Cannot retain an incomplete recovery frame")
            with path.open("xb") as stream:
                stream.write(raw)
            self.retained.add(record.index)
            self.raw_saved += 1
            self.event(
                "recovery_raw",
                delivery_index=record.index,
                path=relative,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
            return relative
        except (OSError, ValueError, TypeError) as error:
            self.raw_omitted += 1
            self._failure(error)
            return None

    def close(self, **outcome: object) -> None:
        if self.closed:
            return
        try:
            if self.stream is not None and not self.failed:
                footer = dict(
                    event="end",
                    events_written=self.events,
                    deliveries_seen=self.deliveries,
                    diagnostic_raw_saved=self.raw_saved,
                    diagnostic_raw_omitted=self.raw_omitted,
                    journal_limited=self.limited,
                    outcome=outcome,
                )
                # The fixed-size footer reserve remains available after the
                # journal cap. Outcome must be a small summary, not raw data.
                line = json.dumps(footer, allow_nan=False, separators=(",", ":")) + "\n"
                if len(line.encode("utf-8")) > FOOTER_RESERVE:
                    raise ValueError("Diagnostic outcome exceeds footer reserve")
                self.stream.write(line)
                self.stream.flush()
        except (OSError, ValueError, TypeError) as error:
            self._failure(error)
        finally:
            self.closed = True
            if self.stream is not None:
                try:
                    self.stream.close()
                except OSError as error:
                    self._failure(error)
