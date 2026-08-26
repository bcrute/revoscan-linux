"""Which scanner this installation may touch, where captures go, and who says so.

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

Where captures go
-----------------
The same file answers a second question, in the same order: the
``SCANNER3D_CAPTURES_DIR`` environment variable, then ``directory`` under
``[captures]``. It is the ROOT under which captures land, not a capture
directory itself - each command appends its own name (``depth-frame``,
``rgb-frame``) or a dated one (``sweep-20260825-174501``), because two sweeps
sharing a directory would overwrite each other's ``frame_0000.raw``.

Nothing is refused when it is unset. A serial has to be right or an invasive
operation aims at the wrong hardware; an output directory only decides where
local evidence lands, so each command keeps the fallback it always had and this
setting merely replaces it.

What this does NOT do
---------------------
It does not verify that the configured serial is attached, or that it is a
scanner at all. Callers resolve the device themselves and fail in their own
vocabulary; this only answers "which one am I allowed to touch".

It does not create, validate or write into the captures directory. Whether that
path exists, is writable, or is a directory at all is discovered by the command
that writes there, in its own vocabulary and at the moment it matters.

It writes configuration only when a caller explicitly asks it to - through
:func:`set_captures_dir` or :func:`clear_captures_dir`, both reached only from
an explicit command. Nothing here creates a file as a side effect of an import
or of a read, and a write preserves every other line of the file, including
comments and settings this module knows nothing about.
"""

from __future__ import annotations

import os
import re
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Final, overload

from scanner3d.errors import DoctorStatus, ScannerError

#: Overrides everything else. Named after the distribution, not the module, so
#: it reads sensibly in a shell history.
SERIAL_ENV_VAR: Final = "SCANNER3D_SERIAL"

#: Overrides the configured captures directory, for the same reason
#: SERIAL_ENV_VAR overrides the serial: one run somewhere else - a scratch
#: disk, a mounted share - should not need the file edited and edited back.
CAPTURES_DIR_ENV_VAR: Final = "SCANNER3D_CAPTURES_DIR"

#: Relative to the XDG config directory.
CONFIG_RELATIVE_PATH: Final = Path("scanner3d") / "config.toml"

#: Where the captures directory is stated. Quoted verbatim in every message
#: that tells someone how to set it by hand, so the two cannot drift apart.
CAPTURES_TABLE: Final = "captures"
CAPTURES_KEY: Final = "directory"

#: Stands in when nothing is configured. Deliberately a serial no device has:
#: a call that reaches hardware with this value fails naming it, which points
#: straight at the configuration that was never set. The alternative - an empty
#: string - produces "no device with serial ''", which explains nothing.
EXAMPLE_SERIAL: Final = "EXAMPLESERIAL0001"


class SerialNotConfigured(ScannerError):
    """No scanner serial was configured and the attached set does not settle it."""

    status = DoctorStatus.UNSUPPORTED_MODE


class ConfigFileUnreadable(ScannerError):
    """The config file exists but is not TOML this can read.

    Separate from :class:`SerialNotConfigured` because it is a different fact:
    a setting was written and cannot be honoured, rather than never stated. It
    is raised for every reader here, so one malformed file cannot answer "no
    serial configured" to one caller and "no captures directory" to another
    while the file plainly states both.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class ConfigWriteRefused(ScannerError):
    """The config file was not rewritten, and still says what it said.

    Raised when an edit cannot be made without risking the rest of the file -
    a hand-written form this cannot edit in place, or a result that does not
    read back as the value asked for. The file is left exactly as it was, and
    the caller is told to make the change by hand.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


def config_path() -> Path:
    """Where configuration is read from. Not created, and need not exist."""
    root = os.environ.get("XDG_CONFIG_HOME")
    base = Path(root) if root else Path.home() / ".config"
    return base / CONFIG_RELATIVE_PATH


def _read_config_text() -> str | None:
    """The config file's text, or None if there is no readable file.

    An unreadable or non-UTF-8 file is the same kind of fact as malformed
    TOML - a stated setting that cannot be honoured - so only a genuinely
    absent file answers None.
    """
    path = config_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ConfigFileUnreadable(
            f"{path} exists but could not be read: {exc}. Fix its permissions or remove it."
        ) from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigFileUnreadable(
            f"{path} is not readable TOML: {exc}. Fix or remove it; a malformed config is "
            "not treated as an absent one, because silently ignoring it would hide the "
            "very setting the file exists to make."
        ) from exc


def _load_config() -> dict[str, object]:
    """The parsed config file, or an empty mapping when there is no file."""
    text = _read_config_text()
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigFileUnreadable(
            f"{config_path()} is not readable TOML: {exc}. Fix or remove it; a malformed "
            "config is not treated as an absent one, because silently ignoring it would "
            "hide the very setting the file exists to make."
        ) from exc


def configured_serial() -> str | None:
    """The serial from the environment or the config file, if either states one.

    Returns None rather than raising when nothing is configured: "not stated"
    is an ordinary condition that callers resolve differently depending on how
    invasive the operation they are about to perform is.
    """
    from_env = os.environ.get(SERIAL_ENV_VAR, "").strip()
    if from_env:
        return from_env

    parsed = _load_config()
    scanner = parsed.get("scanner")
    if isinstance(scanner, dict):
        serial = scanner.get("serial")
        if isinstance(serial, str) and serial.strip():
            return serial.strip()
    return None


def stated_serial() -> str:
    """The configured serial, or :data:`EXAMPLE_SERIAL`, and never a raise.

    For the module-level ``AUTHORIZED_SERIAL`` constants, which are evaluated
    while the package is being imported. A config file too broken to parse must
    not make ``import scanner3d.cli`` fail: that takes down every command,
    including the one that exists to fix the file. The commands that ACT on a
    serial call :func:`resolve_serial` instead, which still refuses with the
    parse error rather than aiming an invasive operation at a placeholder.
    """
    try:
        return configured_serial() or EXAMPLE_SERIAL
    except ConfigFileUnreadable:
        return EXAMPLE_SERIAL


def serial_source() -> str | None:
    """Which place states the serial, in words, or None if neither does.

    The single attached device is not a source: it is what callers fall back to
    when this answers None, and reporting it here would claim a setting exists
    where none was written.
    """
    if os.environ.get(SERIAL_ENV_VAR, "").strip():
        return SERIAL_ENV_VAR
    scanner = _load_config().get("scanner")
    if isinstance(scanner, dict):
        serial = scanner.get("serial")
        if isinstance(serial, str) and serial.strip():
            return str(config_path())
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


# --- where captures go -------------------------------------------------------


def _absolute(path: Path) -> Path:
    """``path`` with ``~`` expanded and made absolute, without following links.

    Absolute because the value outlives the shell that set it: a relative path
    recorded in a config file means a different directory in every terminal,
    which is the opposite of a default. Symlinks are deliberately left intact -
    someone who points this at a symlinked scratch disk meant that path, and
    ``resolve()`` would quietly record somewhere else.
    """
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.normpath(expanded))


def configured_captures_dir() -> Path | None:
    """The captures root from the environment or the config file, if either states one.

    None means "not stated", which is not an error: every command that writes
    captures has a fallback it used before this setting existed.
    """
    from_env = os.environ.get(CAPTURES_DIR_ENV_VAR, "").strip()
    if from_env:
        return _absolute(Path(from_env))

    table = _load_config().get(CAPTURES_TABLE)
    if isinstance(table, dict):
        value = table.get(CAPTURES_KEY)
        if isinstance(value, str) and value.strip():
            return _absolute(Path(value.strip()))
    return None


def captures_dir_source() -> str | None:
    """Which of the two places states the captures directory, in words.

    Reported rather than inferred, because "why is it writing there" is the
    question a surprising default provokes, and an environment variable set in
    one shell is exactly the answer nobody thinks to check.
    """
    if os.environ.get(CAPTURES_DIR_ENV_VAR, "").strip():
        return CAPTURES_DIR_ENV_VAR
    table = _load_config().get(CAPTURES_TABLE)
    if isinstance(table, dict):
        value = table.get(CAPTURES_KEY)
        if isinstance(value, str) and value.strip():
            return str(config_path())
    return None


@overload
def captures_dir_for(name: str, fallback: Path) -> Path: ...


@overload
def captures_dir_for(name: str, fallback: None = None) -> Path | None: ...


def captures_dir_for(name: str, fallback: Path | None = None) -> Path | None:
    """``<captures>/name`` when a captures root is configured, else ``fallback``.

    Commands append a name rather than writing into the root itself: several of
    them write fixed filenames, and two commands sharing one directory would
    silently overwrite each other's evidence.
    """
    root = configured_captures_dir()
    if root is None:
        return fallback
    return root / name


def dated_capture_name(prefix: str, *, now: datetime | None = None) -> str:
    """``prefix-YYYYmmdd-HHMMSS`` in local time, for captures that repeat.

    Local rather than UTC because the only reader is the person who ran the
    capture, sorting today's attempts by the clock on their wall. Seconds are
    enough: the captures this names take minutes.
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def set_captures_dir(directory: Path) -> Path:
    """Record ``directory`` as the default captures root. Returns what was written.

    The path is stored expanded and absolute, and neither created nor checked
    here - see this module's docstring for why.
    """
    target = _absolute(directory)
    _write_setting(CAPTURES_TABLE, CAPTURES_KEY, str(target))
    return target


def clear_captures_dir() -> bool:
    """Remove the recorded default. True if the file stated one and now does not.

    Says nothing about :envvar:`SCANNER3D_CAPTURES_DIR`, which this cannot
    unset for the caller's shell - callers report that separately rather than
    letting someone think a cleared file means a cleared setting.
    """
    return _write_setting(CAPTURES_TABLE, CAPTURES_KEY, None)


#: One bare, quoted or single-quoted key being assigned. Dotted keys are
#: deliberately not matched: they state the same setting in a form this cannot
#: edit in place, and the write is refused rather than guessed at.
_ASSIGNMENT = re.compile(r"""^\s*(?P<key>[A-Za-z0-9_-]+|"[^"]*"|'[^']*')\s*=""")

#: Anything that ends the table above it, arrays of tables ([[x]]) included -
#: a boundary missed here would put the edit inside the wrong table.
_TABLE_HEADER = re.compile(r"^\s*\[\[?[^\[\]]*\]\]?\s*$")

#: Characters TOML basic strings must not carry raw. A path containing one is
#: vanishingly unlikely and perfectly legal, so it is escaped rather than
#: refused.
_ESCAPES: Final = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _toml_basic_string(value: str) -> str:
    out = "".join(_ESCAPES.get(char, char) for char in value)
    return f'"{out}"'


def _write_setting(table: str, key: str, value: str | None) -> bool:
    """Set or remove one key in the config file. True if the file changed.

    The file is edited as text rather than reserialized from the parse tree:
    round-tripping through :mod:`tomllib` would silently drop every comment and
    reorder what someone hand-wrote, and losing a user's notes to record a
    default is not a trade this gets to make. The result is parsed back before
    it is installed, so an edit that would not read as the value asked for
    leaves the original file untouched.
    """
    original = _read_config_text()
    _load_config()  # refuse to edit a file this cannot read in the first place
    updated = _edit_toml(original or "", table, key, value)
    if updated == (original or ""):
        return False

    try:
        parsed = tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigWriteRefused(
            f"editing {config_path()} would have made it unparseable ({exc}), so it was "
            f"left alone. Set {key} under [{table}] by hand."
        ) from exc
    section = parsed.get(table)
    written = section.get(key) if isinstance(section, dict) else None
    if written != value:
        raise ConfigWriteRefused(
            f"{config_path()} states {key} in a form this cannot edit in place, so it was "
            f"left alone. Set {key} under [{table}] by hand."
        )

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".new")
    scratch.write_text(updated, encoding="utf-8")
    # Replaced rather than truncated and rewritten: an interrupted write must
    # not leave a half-file where a working configuration was.
    scratch.replace(path)
    return True


def _edit_toml(text: str, table: str, key: str, value: str | None) -> str:
    """``text`` with ``key`` in ``[table]`` set to ``value``, or removed if None."""
    lines = text.splitlines(keepends=True)
    header = f"[{table}]"
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == header),
        None,
    )
    new_line = None if value is None else f"{key} = {_toml_basic_string(value)}\n"

    if start is None:
        if new_line is None:
            return text
        prefix = text if not text or text.endswith("\n") else text + "\n"
        separator = "\n" if prefix else ""
        return f"{prefix}{separator}{header}\n{new_line}"

    end = next(
        (index for index in range(start + 1, len(lines)) if _TABLE_HEADER.match(lines[index])),
        len(lines),
    )
    for index in range(start + 1, end):
        match = _ASSIGNMENT.match(lines[index])
        if match is None or match.group("key").strip("\"'") != key:
            continue
        if new_line is None:
            return "".join([*lines[:index], *lines[index + 1 :]])
        return "".join([*lines[:index], new_line, *lines[index + 1 :]])

    if new_line is None:
        return text
    return "".join([*lines[: start + 1], new_line, *lines[start + 1 :]])
