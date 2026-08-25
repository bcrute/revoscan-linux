"""Which scanner this installation is allowed to touch, and where that is stated.

Why a module rather than a constant
-----------------------------------
Every module that speaks to the device used to carry the same hard-coded serial
number - the one unit its author owned. That worked while the project had one
user and one scanner, and it stops working the moment anyone else runs it.

The serial is not merely a default, though. Several operations here mutate
kernel driver binding, write vendor property keys, or detach ``uvcvideo`` from
an interface, and the original design refused to aim any of those at "whatever
happens to be plugged in". That refusal is worth keeping: naming the target is
what makes an invasive operation reviewable. So this module keeps the guarantee
and moves the value out of the source.

Where the answer comes from, in order
-------------------------------------
1. The ``SCANNER3D_SERIAL`` environment variable. Wins over everything, so a
   one-off run against a second unit needs no file edited.
2. ``$XDG_CONFIG_HOME/scanner3d/config.toml`` (or ``~/.config/...``), key
   ``serial`` under ``[scanner]``.
3. The attached scanner, if there is EXACTLY ONE. Convenience for the common
   case of one device on the bench.

If two or more are attached and nothing was configured, this refuses rather
than choosing. A wrong guess here is a driver detached from someone else's
hardware, and there is no reading of "be helpful" that justifies it.

What this does NOT do
---------------------
It does not verify that the configured serial is attached, or that it is a
scanner at all. Callers resolve the device themselves and fail in their own
vocabulary; this only answers "which one am I allowed to touch".

It does not write configuration. Creating files under a user's home directory
as a side effect of a library import is not this module's business.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError

#: Overrides everything else. Named after the distribution, not the module, so
#: it reads sensibly in a shell history.
SERIAL_ENV_VAR: Final = "SCANNER3D_SERIAL"

#: Relative to the XDG config directory.
CONFIG_RELATIVE_PATH: Final = Path("scanner3d") / "config.toml"

#: Stands in when nothing is configured. Deliberately a serial no device has:
#: a call that reaches hardware with this value fails naming it, which points
#: straight at the configuration that was never set. The alternative - an empty
#: string - produces "no device with serial ''", which explains nothing.
EXAMPLE_SERIAL: Final = "EXAMPLESERIAL0001"


class SerialNotConfigured(ScannerError):
    """No scanner serial was configured and the attached set does not settle it."""

    status = DoctorStatus.UNSUPPORTED_MODE


def config_path() -> Path:
    """Where configuration is read from. Not created, and need not exist."""
    root = os.environ.get("XDG_CONFIG_HOME")
    base = Path(root) if root else Path.home() / ".config"
    return base / CONFIG_RELATIVE_PATH


def configured_serial() -> str | None:
    """The serial from the environment or the config file, if either states one.

    Returns None rather than raising when nothing is configured: "not stated"
    is an ordinary condition that callers resolve differently depending on how
    invasive the operation they are about to perform is.
    """
    from_env = os.environ.get(SERIAL_ENV_VAR, "").strip()
    if from_env:
        return from_env

    path = config_path()
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SerialNotConfigured(
            f"{path} is not readable TOML: {exc}. Fix or remove it; a malformed config is "
            "not treated as an absent one, because silently ignoring it would hide the "
            "very setting the file exists to make."
        ) from exc
    scanner = parsed.get("scanner")
    if isinstance(scanner, dict):
        serial = scanner.get("serial")
        if isinstance(serial, str) and serial.strip():
            return serial.strip()
    return None


def resolve_serial(explicit: str | None = None) -> str:
    """The serial to act on, or a refusal explaining how to state one.

    ``explicit`` - a value the caller was given directly, usually from a command
    line - wins over configuration, which wins over the single attached device.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    configured = configured_serial()
    if configured:
        return configured

    from scanner3d.discovery import find_scanners

    try:
        attached = [device.serial for device in find_scanners() if device.serial]
    except ScannerError:
        attached = []
    if len(attached) == 1:
        return attached[0]
    if not attached:
        raise SerialNotConfigured(
            "No scanner serial configured and none attached. Set "
            f"{SERIAL_ENV_VAR}, or write serial under [scanner] in {config_path()}."
        )
    raise SerialNotConfigured(
        f"{len(attached)} scanners are attached ({', '.join(sorted(attached))}) and no "
        f"serial is configured. Set {SERIAL_ENV_VAR} or write serial under [scanner] in "
        f"{config_path()}. This refuses to choose because several operations here detach "
        "kernel drivers, and aiming one at the wrong device is not recoverable by retrying."
    )
