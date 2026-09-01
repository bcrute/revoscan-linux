"""Somewhere to look when the application stops responding.

Why this exists
---------------
A window that freezes tells you nothing. Launched from a desktop menu there is
no terminal, stderr goes nowhere, and the only evidence available to the person
affected is that it stopped. This project has already shipped one failure of
exactly that shape - the application would not start at all for want of a Qt
platform plugin, printed the reason to a stderr nobody could see, and from the
outside looked like nothing happening.

So: a log file, always on, in a predictable place, with timestamps. It costs
almost nothing and it is the difference between "it froze" and "it froze in
wait_for_capture_access after the preview stopped".

What gets logged
----------------
Device operations and their durations, because this application's slow parts
are all device operations and a freeze is nearly always something blocking on
one. Thread names are included: the whole point of the worker threads here is
that the UI stays responsive, so a device call logged from the GUI thread is
itself the bug.

What this does NOT do
---------------------
It does not rotate or bound the log by size. A capture session writes a few
hundred lines; this is not a service that runs for months.

It does not log frame contents or point data, only that they happened and how
long they took. A log that is expensive to write changes the thing it measures.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

#: One logger for the whole distribution. Modules take children of it.
ROOT_NAME: Final = "scanner3d"

_CONFIGURED = False


def log_path() -> Path:
    """Where the log is written. Follows the XDG state convention."""
    root = os.environ.get("XDG_STATE_HOME")
    base = Path(root) if root else Path.home() / ".local" / "state"
    return base / "scanner3d" / "scanner3d.log"


def get_logger(name: str) -> logging.Logger:
    """A child logger. Safe to call before :func:`configure`."""
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def configure(*, verbose: bool = False) -> Path | None:
    """Start writing to the log file. Idempotent; never raises.

    Returns the path being written to, or None if it could not be opened - a
    logger that cannot open its file must not stop the application it exists to
    diagnose.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return log_path()

    logger = logging.getLogger(ROOT_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler()
        path_result = None
    else:
        path_result = path
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
    logger.info("--- logging started, writing to %s ---", path_result or "stderr")
    return path_result


@contextmanager
def timed(logger: logging.Logger, what: str, *, warn_after: float = 0.5) -> Iterator[None]:
    """Log how long something took, loudly if it took long enough to be felt.

    ``warn_after`` defaults to half a second because that is roughly where a
    person stops believing an interface is responding. Anything slower than
    that on the GUI thread is a defect, and this is how it gets noticed without
    somebody having to reproduce it while watching.
    """
    started = time.monotonic()
    logger.debug("%s: start", what)
    try:
        yield
    except Exception as exc:
        logger.warning("%s: FAILED after %.3fs: %s", what, time.monotonic() - started, exc)
        raise
    elapsed = time.monotonic() - started
    if elapsed >= warn_after:
        logger.warning("%s: took %.3fs", what, elapsed)
    else:
        logger.info("%s: %.3fs", what, elapsed)
