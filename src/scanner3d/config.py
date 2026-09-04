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
directory itself. Every command claims a namespace named after itself
(``depth-frame``, ``rgb-frame``, ``paired-capture``, ``paired-rgb-series``,
``sweep``, ``scan`` for the desktop application) and each EXECUTION gets a
fresh dated directory inside that namespace::

    <root>/sweep/20260825-174501
    <root>/sweep/20260825-174501-2      # a second sweep in the same second

The desktop application asks for an optional name when a scan starts, so its
leaf can be readable instead of dated, under exactly the same namespace and the
same collision policy::

    <root>/scan/chassis left side
    <root>/scan/chassis left side-2    # the same name used a second time

:func:`allocate_capture_dir` creates that directory with a bare ``mkdir``,
which the kernel makes atomic, and never reuses one that already exists. Two
processes starting in the same second, a restart inside one second, a clock
dragged backwards, and the hour a DST change repeats all therefore land in
different directories or produce a refusal - and the refusal happens before the
hardware is touched, so a capture is never spent only to have its evidence
collide with older frames. An explicitly named output directory keeps the
semantics it always had: it is the caller's directory, used as given.

Relative paths, and why the two sources differ
----------------------------------------------
A relative ``directory`` in the config file is resolved against the directory
holding ``config.toml``, not the process's working directory. A persistent
default that means a different place in every terminal is not a default. The
environment variable is resolved against the working directory instead,
because it is a deliberate one-process override typed into one shell, and
there is nothing else it could sensibly mean.

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
comments and settings this module knows nothing about. That promise is checked
rather than asserted: the edited text is parsed back and compared, key by key,
against the original parse, and anything but the one requested change leaves
the file untouched and raises :class:`ConfigWriteRefused`.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import re
import stat
import string
import tempfile
import tomllib
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, NamedTuple, overload

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

#: Where the registration depth band is stated: ``depth_band = [near, far]``
#: under ``[registration]``, in the deprojection's scale. Read by every
#: registration entry point; :class:`scanner3d.preparation.DepthBand` says why
#: the sensor's fixed clip is not enough once a room stands inside it.
REGISTRATION_TABLE: Final = "registration"
DEPTH_BAND_KEY: Final = "depth_band"

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
    a hand-written form this cannot edit in place, a result that does not read
    back as the value asked for, an edit that would disturb something other
    than the requested key, a config path that is a symbolic link, or an I/O
    failure anywhere on the write path. The file is left exactly as it was,
    every message says so in as many words, and the caller is told to make the
    change by hand.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class CaptureDirUnavailable(ScannerError):
    """No fresh capture directory could be created, so nothing was captured.

    Raised before any device is opened. Reusing a directory that already holds
    frames would mix a new capture's evidence with an older one's, and there is
    no reading of "be helpful" that makes that a better answer than refusing.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class CaptureNameInvalid(CaptureDirUnavailable):
    """A supplied capture name is not one directory component, so nothing exists.

    A subclass rather than a sibling, deliberately: every caller that already
    guards allocation with :class:`CaptureDirUnavailable` refuses safely the
    first time somebody types a name with a slash in it, instead of letting a
    new exception escape into a traceback. The two facts really are the same
    one from the caller's side - no directory was created and nothing was
    captured - and only the message differs.
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


def _absolute(path: Path, *, base: Path) -> Path:
    """``path`` with ``~`` expanded and made absolute against ``base``.

    Absolute because the value outlives the shell that set it. ``base`` is the
    caller's answer to "relative to what", and the two sources deliberately
    answer it differently - see this module's docstring. Symlinks are left
    intact: someone who points this at a symlinked scratch disk meant that
    path, and ``resolve()`` would quietly record somewhere else.
    """
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return Path(os.path.normpath(expanded))


def configured_captures_dir() -> Path | None:
    """The captures root from the environment or the config file, if either states one.

    None means "not stated", which is not an error: every command that writes
    captures has a fallback it used before this setting existed.
    """
    from_env = os.environ.get(CAPTURES_DIR_ENV_VAR, "").strip()
    if from_env:
        # Process-relative on purpose: an environment variable is a one-shell
        # override, and the shell's own directory is the only base it can mean.
        return _absolute(Path(from_env), base=Path.cwd())

    table = _load_config().get(CAPTURES_TABLE)
    if isinstance(table, dict):
        value = table.get(CAPTURES_KEY)
        if isinstance(value, str) and value.strip():
            # Relative to the file that states it, so a hand-edited relative
            # path names one directory rather than one per terminal.
            return _absolute(Path(value.strip()), base=config_path().parent)
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


def configured_depth_band() -> tuple[float, float] | None:
    """``[registration] depth_band`` as ``(near, far)``, if the file states one.

    None means "not stated", and every registration keeps the sensor's own
    interval. A stated value that is not two increasing positive finite
    numbers is refused as :class:`ConfigFileUnreadable`, for the reason
    malformed TOML is: silently ignoring it would hide the very setting the
    file exists to make, and a registration clipped to the wrong interval
    reports plausible numbers.
    """
    table = _load_config().get(REGISTRATION_TABLE)
    if not isinstance(table, dict) or DEPTH_BAND_KEY not in table:
        return None
    value = table[DEPTH_BAND_KEY]
    problem = _depth_band_problem(value)
    if problem is not None:
        raise ConfigFileUnreadable(
            f"{config_path()} states {DEPTH_BAND_KEY} = {value!r} under [{REGISTRATION_TABLE}], "
            f"which {problem}. Write it as two increasing positive numbers, for example "
            f"{DEPTH_BAND_KEY} = [130, 275], or remove it to keep the default interval."
        )
    near, far = value
    return float(near), float(far)


def _depth_band_problem(value: object) -> str | None:
    """Why a stated depth band cannot be honoured, or None when it can."""
    if not isinstance(value, list) or len(value) != 2:
        return "is not a list of two numbers"
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            return "is not made of two finite numbers"
    near, far = value
    if near <= 0 or far <= near:
        return "is not a positive, increasing interval"
    return None


def depth_band_source() -> str | None:
    """Where the registration depth band is stated, in words, or None when it is not."""
    table = _load_config().get(REGISTRATION_TABLE)
    if isinstance(table, dict) and DEPTH_BAND_KEY in table:
        return str(config_path())
    return None


@overload
def captures_dir_for(name: str, fallback: Path) -> Path: ...


@overload
def captures_dir_for(name: str, fallback: None = None) -> Path | None: ...


def captures_dir_for(name: str, fallback: Path | None = None) -> Path | None:
    """``<captures>/name`` when a captures root is configured, else ``fallback``.

    This is a command's NAMESPACE under the root, not a directory to capture
    into: commands append a name rather than writing into the root itself
    because several of them write fixed filenames, and two commands sharing one
    directory would silently overwrite each other's evidence.

    Anything about to capture calls :func:`allocate_capture_dir`, which creates
    a fresh dated directory inside this one. This is for the dry runs and
    reports that need to say where a namespace is without claiming a particular
    capture's directory, and for the one command - ``depth-frame`` and
    ``rgb-frame``'s preflight - that writes nothing at all.
    """
    root = configured_captures_dir()
    if root is None:
        return fallback
    return root / name


def capture_stamp(now: datetime | None = None) -> str:
    """``YYYYmmdd-HHMMSS`` in local time. A name, never a uniqueness claim.

    Local rather than UTC because the only reader is the person who ran the
    capture, sorting today's attempts by the clock on their wall. Second
    resolution is all a wall clock is good for here, and it is emphatically not
    unique: two runs in one second, a restart, a clock dragged backwards and the
    repeated hour of a DST change all produce the same string.
    :func:`allocate_capture_dir` is what makes the directory unique; this only
    makes it readable.
    """
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def dated_capture_name(prefix: str, *, now: datetime | None = None) -> str:
    """``prefix-YYYYmmdd-HHMMSS`` in local time, for captures that repeat."""
    return f"{prefix}-{capture_stamp(now)}"


#: How many same-stamp directories are tried before a capture is refused. Large
#: enough that no honest burst of concurrent starts reaches it, small enough
#: that a directory somebody filled with junk ends in a refusal rather than a
#: spin.
CAPTURE_DIR_ATTEMPTS: Final = 256

#: The namespace the desktop application claims under the captures root. Named
#: here rather than in ``gui`` because ``scanner3d config`` reports it too, and
#: a report that disagrees with the behaviour it describes is worse than none.
GUI_CAPTURE_COMMAND: Final = "scan"

#: The one thing a human-readable capture name may not be reduced to: a name
#: that means "somewhere else". Everything else about a name is a matter of
#: taste, and taste is the user's.
_RESERVED_CAPTURE_NAMES: Final = frozenset({".", ".."})

#: Every character that separates one path component from another on any host
#: this name could be carried to. ``/`` and ``\`` are both in here on every
#: platform, deliberately: ``os.sep``/``os.altsep`` describe the machine that
#: happens to be running, and a name is not a per-machine question. A capture
#: directory tree is copied to a Windows box, mounted over SMB, or read back by
#: a tool that splits on either, and ``parent\child`` is a path-shaped name in
#: all of those places even though POSIX would create it as one literal folder.
#: ``os.sep``/``os.altsep`` are still unioned in so a host with a third
#: separator is covered without this set having to learn about it.
_CAPTURE_NAME_SEPARATORS: Final = frozenset(
    {"/", "\\"} | {sep for sep in (os.sep, os.altsep) if sep}
)


def _control_characters(name: str) -> list[str]:
    """The Unicode ``Cc`` characters in ``name``, in the order they appear.

    Unicode General Category ``Cc`` rather than a comparison against ASCII
    space: the C1 block is control characters too, and U+0085 and U+009F are
    below neither ``" "`` nor ``"\\x7f"``. They have no business in a folder
    name whether or not a particular filesystem would technically store them.
    """
    return [character for character in name if unicodedata.category(character) == "Cc"]


def validate_capture_name(name: str) -> str:
    """The trimmed name a capture asked for, or ``""`` when it asked for none.

    ``""`` is a legitimate answer rather than a refusal: somebody who leaves
    the field blank has said "you name it", and :func:`allocate_capture_dir`
    then uses the readable local timestamp it has always used.

    A non-empty name is ONE directory component under the command's namespace,
    and this is where that is enforced. Anything that could reach outside that
    namespace - ``.``, ``..``, ``/`` or ``\\`` on any host, an absolute path, a
    NUL or any other Unicode ``Cc`` control character - is refused with a
    message aimed at the person who typed it. Ordinary human-readable names
    survive intact, internal spaces and all: the point of asking for a name is
    that ``chassis left side`` beats ``20260826-142201``, and a validator that
    mangles it into ``chassis_left_side`` has thrown away the only thing it was
    asked to keep.

    Surrounding *printable* whitespace is trimmed rather than refused, because
    it is almost always a stray keystroke or a paste, and a directory whose
    name begins with a space is a directory somebody will have to quote
    forever. A newline, a tab or any other control character is not that: it is
    refused, and it is refused BEFORE the trim, because ``str.strip()`` removes
    several control characters itself and would otherwise turn a name that is
    nothing but controls into a silent request for a dated folder.
    """
    control = _control_characters(name)
    if control:
        offending = ", ".join(sorted({repr(character) for character in control}))
        raise CaptureNameInvalid(
            f"the scan name contains characters that cannot be in a folder name "
            f"({offending}). Use ordinary text; spaces are fine, but tabs, line breaks "
            "and other control characters are not. Nothing was created and nothing was "
            "captured."
        )
    trimmed = name.strip()
    if not trimmed:
        return ""
    if trimmed in _RESERVED_CAPTURE_NAMES:
        raise CaptureNameInvalid(
            f"{trimmed!r} is not a capture name, it is a directory that already exists. "
            "Give the scan a name, or leave the name blank to have one dated for you. "
            "Nothing was created and nothing was captured."
        )
    if any(sep in trimmed for sep in _CAPTURE_NAME_SEPARATORS):
        raise CaptureNameInvalid(
            f"{trimmed!r} contains a path separator, so it names a place rather than a "
            "scan. A scan name is one folder name - '/' and '\\' are both out, on every "
            "system - kept inside the captures folder; choose where captures go with "
            "'Captures folder...' instead. Nothing was created and nothing was captured."
        )
    # Belt and braces, and cheap: whatever this platform thinks a path is, the
    # result has to be exactly one relative component of it. A name that gets
    # past the checks above and still is not - a Windows drive letter, say - is
    # refused rather than passed to mkdir to find out.
    candidate = Path(trimmed)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.parts[0] != trimmed:
        raise CaptureNameInvalid(
            f"{trimmed!r} is not a single folder name, so it is not a scan name. "
            "Nothing was created and nothing was captured."
        )
    return trimmed


def allocate_capture_dir(
    command: str,
    *,
    root: Path | None = None,
    now: datetime | None = None,
    name: str | None = None,
) -> Path | None:
    """Create and return a fresh ``<root>/<command>/<leaf>`` directory.

    The leaf is the ``name`` the caller supplied, validated by
    :func:`validate_capture_name` as one directory component, and otherwise the
    local timestamp - which is also what a name that is blank or nothing but
    whitespace means. Callers that supply no name at all get exactly the dated
    behaviour they got before names existed.

    None means no captures root is configured, which is not an error: the
    caller falls back to whatever default it had before this setting existed.

    The directory is created with a plain ``mkdir``, which fails rather than
    succeeds if the name is taken, so this never adopts a directory that
    already exists - not one an earlier capture filled, not one a concurrent
    invocation just took, not one a clock that went backwards points at again,
    and not one that happens to share a name somebody has used before. Callers
    call this BEFORE they open a device: a capture that cannot be preserved
    should never have been taken.
    """
    # First, and before anything is created: a refused name must leave the
    # filesystem exactly as it found it, including the command's namespace.
    leaf = validate_capture_name(name) if name is not None else ""

    base = configured_captures_dir() if root is None else root
    if base is None:
        return None

    parent = base / command
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CaptureDirUnavailable(
            f"cannot create {parent}: {exc}. Nothing was captured; fix the path or record a "
            "different one with 'scanner3d config --set-captures-dir DIR'."
        ) from exc

    # One collision policy, whatever the leaf is called: the first attempt asks
    # for the name itself, and every later one appends -2, -3 and so on. A named
    # scan and a dated one therefore behave identically when the name is taken,
    # which is the only way "your frames are in the folder you named" stays true
    # without ever meaning "mixed in with the last scan of that name".
    stem = leaf or capture_stamp(now)
    for attempt in range(1, CAPTURE_DIR_ATTEMPTS + 1):
        candidate = parent / (stem if attempt == 1 else f"{stem}-{attempt}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        except OSError as exc:
            raise CaptureDirUnavailable(
                f"cannot create {candidate}: {exc}. Nothing was captured."
            ) from exc
        return candidate

    raise CaptureDirUnavailable(
        f"every one of {CAPTURE_DIR_ATTEMPTS} names under {parent} starting {stem} is already "
        "taken, so nothing was captured. New frames are never written into a directory that "
        "already holds some, because mixing two captures' frames destroys both as evidence."
    )


class CapturesWrite(NamedTuple):
    """What recording a captures root actually did.

    Two facts, because they are not the same fact and reporting the first as
    though it implied the second is a lie a caller then repeats. ``changed`` is
    false when the file already said exactly this, which is the ordinary result
    of running the same command twice - nothing was rewritten, and a report
    claiming otherwise invites someone to go looking for a change that is not
    in the file's mtime or in anyone's version control.
    """

    #: The absolute path that the file now states.
    path: Path
    #: Whether that required writing the file at all.
    changed: bool


def set_captures_dir(directory: Path) -> CapturesWrite:
    """Record ``directory`` as the default captures root.

    Returns what was written and whether writing it changed the file, so a
    caller can report a repeated identical set as the no-op it is.

    A relative ``directory`` is resolved against the working directory: this
    value was typed at a shell prompt just now, so the shell's own directory is
    what it means. What lands in the file is always absolute, and is neither
    created nor checked here - see this module's docstring for why.
    """
    target = _absolute(directory, base=Path.cwd())
    changed = _write_setting(CAPTURES_TABLE, CAPTURES_KEY, str(target))
    return CapturesWrite(target, changed)


def set_depth_band(near: float, far: float) -> bool:
    """Record the registration depth band. True if the file changed.

    Refuses, before touching the file, a band the clip could not apply - the
    same rule :func:`configured_depth_band` reads by - so the file never
    states an interval that every reader would then refuse.
    """
    values: list[float] = [float(near), float(far)]
    problem = _depth_band_problem(values)
    if problem is not None:
        raise ConfigWriteRefused(
            f"a depth band of {near!r} to {far!r} {problem}; nothing was written. The near end "
            "must be positive and the far end beyond it."
        )
    return _write_setting(
        REGISTRATION_TABLE, DEPTH_BAND_KEY, values, literal=_toml_number_list(values)
    )


def clear_depth_band() -> bool:
    """Remove the recorded depth band. True if the file stated one and now does not."""
    return _write_setting(REGISTRATION_TABLE, DEPTH_BAND_KEY, None)


def _toml_number_list(values: Sequence[float]) -> str:
    """``[130, 275]``: integers as integers, anything else as a float literal."""
    rendered = [
        str(int(value)) if float(value).is_integer() else repr(float(value)) for value in values
    ]
    return "[" + ", ".join(rendered) + "]"


def clear_captures_dir() -> bool:
    """Remove the recorded default. True if the file stated one and now does not.

    Says nothing about :envvar:`SCANNER3D_CAPTURES_DIR`, which this cannot
    unset for the caller's shell - callers report that separately rather than
    letting someone think a cleared file means a cleared setting.
    """
    return _write_setting(CAPTURES_TABLE, CAPTURES_KEY, None)


#: The three forms a TOML key part may take, and the dotted key they build.
#: A carriage return is excluded alongside the newline: TOML's only line
#: terminators are LF and CRLF, so a raw CR can never be part of a key, and
#: letting one in would let a pattern anchored at end-of-content match text a
#: reader would have seen on the next line.
_BARE_KEY = r"[A-Za-z0-9_-]+"
_BASIC_KEY = r'"(?:[^"\\\r\n]|\\.)*"'
_LITERAL_KEY = r"'[^'\r\n]*'"
_KEY_PART = rf"(?:{_BARE_KEY}|{_BASIC_KEY}|{_LITERAL_KEY})"
_DOTTED_KEY = rf"{_KEY_PART}(?:[ \t]*\.[ \t]*{_KEY_PART})*"

#: One bare, quoted or single-quoted key being assigned. Dotted keys are
#: deliberately not matched: they state the same setting in a form this cannot
#: edit in place, and the write is refused rather than guessed at. The key part
#: is the same pattern a table header uses, so an escaped quoted key such as
#: ``"direc\u0074ory"`` is matched here exactly as tomllib matches it; the
#: earlier ``"[^"]*"`` stopped at the first ``\"`` inside a key.
_ASSIGNMENT = re.compile(rf"^[ \t]*(?P<key>{_KEY_PART})[ \t]*=")

#: Anything that ends the table above it, arrays of tables ([[x]]) included.
#: A boundary missed here would put the edit inside the wrong table, so this
#: accepts every valid header form: quoted and dotted keys, whitespace inside
#: the brackets, and the trailing comment a header is allowed to carry. The
#: earlier version of this pattern rejected ``[export] # comment``, which let a
#: scan started at ``[captures]`` run straight through it and delete a
#: same-named key belonging to somebody else's table.
#:
#: This is matched against a line's *content* - the line with its terminator
#: already removed by :func:`_content` - so that it never has to know whether
#: the file ends its lines with LF or CRLF. Anchoring the old version at ``$``
#: made it accept ``[captures]\n`` and reject the equally valid
#: ``[captures]\r\n``, which tomllib reads identically.
_TABLE_HEADER = re.compile(
    rf"^[ \t]*(?P<open>\[\[|\[)[ \t]*(?P<key>{_DOTTED_KEY})[ \t]*(?P<close>\]\]|\])"
    r"(?:[ \t]*#.*)?[ \t]*\Z"
)


#: A line's terminator: TOML defines exactly two, LF and CRLF, and nothing
#: else in this module may be treated as ending a line.
_TERMINATOR = re.compile(r"\r?\n\Z")


def _split_lines(text: str) -> list[str]:
    """``text`` split into lines that keep their terminators.

    :meth:`str.splitlines` is deliberately not used: it also breaks on ``\r``,
    ``\v``, ``\f``, ``\x1c`` and ``\u2028``, none of which end a line in TOML.
    A form feed inside a multi-line string would split one logical line into
    two and could move a table boundary. Splitting on LF alone leaves any
    ``\r`` attached to the line it terminates, where :func:`_content` and
    :func:`_terminator` handle it.
    """
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _terminator(line: str) -> str:
    """The LF or CRLF that ends ``line``, or an empty string if none does."""
    match = _TERMINATOR.search(line)
    return match.group(0) if match else ""


def _content(line: str) -> str:
    """``line`` without its terminator, so patterns need not know which it was."""
    return line[: len(line) - len(_terminator(line))]


def _file_newline(text: str) -> str:
    """The line ending ``text`` already uses, defaulting to LF when it has none.

    A file that says ``\r\n`` everywhere stays that way: writing a bare LF
    assignment into it would leave a config with mixed line endings that no
    one asked for, and that a later reader would have to explain.
    """
    first = text.find("\n")
    if first < 0:
        return "\n"
    return "\r\n" if first > 0 and text[first - 1] == "\r" else "\n"


#: The single-character escapes a TOML basic string may carry, decoded. The
#: ``\\u``/``\\U`` forms are handled separately because they consume digits.
_UNESCAPES: Final = {
    "b": "\b",
    "t": "\t",
    "n": "\n",
    "f": "\f",
    "r": "\r",
    '"': '"',
    "\\": "\\",
}


def _decode_basic_string(inner: str) -> str:
    """The contents of a TOML basic string with every valid escape decoded.

    Matching a key by its *text* is not the same as matching it by its value:
    ``"cap\\u0074ures"`` is the ``captures`` table as far as tomllib is
    concerned, so an editor that compares the raw escape misses the table it
    was asked to edit and reports there was nothing to change. Decoding here
    is what makes :func:`_table_header` and :func:`_ASSIGNMENT` agree with the
    parser about which key is which.

    An escape TOML does not define cannot appear in a file that already parsed,
    so the unknown-escape branch is a fallback rather than a policy: the
    backslash is kept verbatim, which can only ever fail to match a key, never
    match the wrong one.
    """
    out: list[str] = []
    index = 0
    while index < len(inner):
        char = inner[index]
        if char != "\\" or index + 1 >= len(inner):
            out.append(char)
            index += 1
            continue
        marker = inner[index + 1]
        simple = _UNESCAPES.get(marker)
        if simple is not None:
            out.append(simple)
            index += 2
            continue
        if marker in "uU":
            width = 4 if marker == "u" else 8
            digits = inner[index + 2 : index + 2 + width]
            if len(digits) == width and all(digit in string.hexdigits for digit in digits):
                code = int(digits, 16)
                if code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF:
                    out.append(chr(code))
                    index += 2 + width
                    continue
        out.append(char)
        index += 1
    return "".join(out)


def _unquote_key_part(part: str) -> str:
    """One key part as the parser sees it: quotes removed, escapes decoded.

    A literal (single-quoted) key keeps its contents verbatim - TOML defines no
    escapes there, so a backslash in one is a backslash.
    """
    if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
        inner = part[1:-1]
        if part[0] == "'":
            return inner
        return _decode_basic_string(inner)
    return part


def _key_parts(text: str) -> tuple[str, ...]:
    """A dotted key split on its real separators, ignoring dots inside quotes."""
    parts: list[str] = []
    current = ""
    index = 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            end = index + 1
            while end < len(text):
                if char == '"' and text[end] == "\\":
                    end += 2
                    continue
                if text[end] == char:
                    break
                end += 1
            current += text[index : end + 1]
            index = end + 1
            continue
        if char == ".":
            parts.append(current)
            current = ""
        else:
            current += char
        index += 1
    parts.append(current)
    return tuple(_unquote_key_part(part.strip()) for part in parts)


def _table_header(line: str) -> tuple[tuple[str, ...], bool] | None:
    """``(key path, is an array of tables)`` if ``line`` is a table header.

    None for anything else, including ``[a]]`` and ``[[a]`` - unbalanced
    brackets are not a header, and treating one as a boundary would be as wrong
    as missing a real one.

    ``line`` may carry either terminator or none at all; the terminator is
    removed before matching, so ``[captures]``, ``[captures]\n`` and
    ``[captures]\r\n`` are the one header tomllib reads them all as.
    """
    match = _TABLE_HEADER.match(_content(line))
    if match is None or len(match.group("open")) != len(match.group("close")):
        return None
    return _key_parts(match.group("key")), len(match.group("open")) == 2


def _assignment_key(line: str) -> str | None:
    """The key ``line`` assigns, decoded as tomllib would read it, or None.

    Terminator-insensitive for the same reason :func:`_table_header` is: which
    line ending a file uses says nothing about which key a line states.
    """
    match = _ASSIGNMENT.match(_content(line))
    return None if match is None else _unquote_key_part(match.group("key"))


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


def _everything_but(parsed: dict[str, object], table: str, key: str) -> dict[str, object]:
    """``parsed`` with the one key this edit is allowed to touch removed.

    The table itself is dropped when removing the key empties it, because
    appending ``[captures]`` to a file that had no such table is part of
    setting the value, not a change to anything else. Every other table is
    compared exactly as it parsed.
    """
    copied = copy.deepcopy(parsed)
    section = copied.get(table)
    if isinstance(section, dict):
        section.pop(key, None)
        if not section:
            copied.pop(table, None)
    return copied


def _write_setting(table: str, key: str, value: object, *, literal: str | None = None) -> bool:
    """Set or remove one key in the config file. True if the file changed.

    ``value`` is what the key must READ BACK as; a string is written as a TOML
    basic string, and anything else must come with ``literal``, the exact TOML
    text to write for it. ``None`` removes the key.

    The file is edited as text rather than reserialized from the parse tree:
    round-tripping through :mod:`tomllib` would silently drop every comment and
    reorder what someone hand-wrote, and losing a user's notes to record a
    default is not a trade this gets to make.

    Editing text is only safe if the result is checked, so the edited text is
    parsed back and checked twice before anything is installed: the requested
    key must read as the requested value, AND every other key in the file must
    parse identically to how it parsed before. The second check is the one that
    matters. Any misreading of the file's structure - a table boundary, a key
    inside a multi-line string, a form nobody anticipated - shows up as a
    difference somewhere this edit had no business touching, and the original
    file is left exactly as it was.
    """
    original = _read_config_text()
    before = _load_config()  # refuse to edit a file this cannot read in the first place
    if value is None:
        rendered = None
    elif literal is not None:
        rendered = literal
    elif isinstance(value, str):
        rendered = _toml_basic_string(value)
    else:
        raise ConfigWriteRefused(
            f"{key} under [{table}] cannot be written as {value!r} without its TOML text; "
            "nothing was written"
        )
    updated = _edit_toml(original or "", table, key, rendered)
    if updated == (original or ""):
        return False

    try:
        after = tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigWriteRefused(
            f"editing {config_path()} would have made it unparseable ({exc}), so it was "
            f"left alone and still says what it said. Set {key} under [{table}] by hand."
        ) from exc

    section = after.get(table)
    written = section.get(key) if isinstance(section, dict) else None
    if written != value:
        raise ConfigWriteRefused(
            f"{config_path()} states {key} in a form this cannot edit in place, so it was "
            f"left alone and still says what it said. Set {key} under [{table}] by hand."
        )
    if _everything_but(before, table, key) != _everything_but(after, table, key):
        raise ConfigWriteRefused(
            f"editing {config_path()} would have changed something other than {key} under "
            f"[{table}], so it was left alone and still says what it said. Set {key} under "
            f"[{table}] by hand."
        )

    _install(config_path(), updated)
    return True


def _preserved_mode(path: Path) -> int:
    """The mode to give the replacement file: the original's, or a private one.

    An existing file's permissions are reproduced exactly rather than
    recomputed from the process umask, which would widen a deliberately private
    ``0600`` config to whatever the shell happened to be set to. A file that
    does not exist yet is created private, because the next thing written here
    is a path into somebody's home directory.
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0o600
    except OSError as exc:
        raise ConfigWriteRefused(
            f"cannot read the permissions of {path} ({exc}), so it was not rewritten and still "
            "says what it said. Edit it by hand."
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        return 0o600
    return stat.S_IMODE(info.st_mode)


def _install(path: Path, text: str) -> None:
    """Put ``text`` at ``path`` through a securely created same-directory file.

    Same directory so the final step can be an atomic rename rather than a copy
    across filesystems; created by :mod:`tempfile` so it never has a
    predictable name another process could have made first; ``fsync``-ed before
    the rename so a power failure leaves either the old file or the new one and
    never a truncated hybrid.

    A symlinked config path is refused rather than followed OR replaced.
    Replacing it would swap somebody's link for a regular file and quietly
    strand the file it pointed at - unusually bad here, because this module's
    whole promise is that it preserves settings it does not understand.

    Every failure leaves the original file untouched, says so, and removes the
    temporary file behind it.
    """
    if path.is_symlink():
        raise ConfigWriteRefused(
            f"{path} is a symbolic link. Rewriting it would replace the link with a regular "
            "file and leave the file it points at unchanged, so nothing was written and both "
            "the link and its target still say what they said. Edit the file it points at by "
            "hand, or point the config path at a regular file."
        )

    mode = _preserved_mode(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigWriteRefused(
            f"cannot create {path.parent} ({exc}), so nothing was written and {path} is "
            "unchanged. Fix that directory, or set the value by hand."
        ) from exc

    try:
        handle, scratch_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
    except OSError as exc:
        raise ConfigWriteRefused(
            f"cannot create a temporary file beside {path} ({exc}), so nothing was written and "
            f"{path} is unchanged. Fix that directory, or set the value by hand."
        ) from exc

    scratch = Path(scratch_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        scratch.chmod(mode)
    except OSError as exc:
        _discard(scratch)
        raise ConfigWriteRefused(
            f"the replacement for {path} could not be written ({exc}), so nothing was "
            f"installed and {path} is unchanged. Set the value by hand."
        ) from exc

    try:
        scratch.replace(path)
    except OSError as exc:
        _discard(scratch)
        raise ConfigWriteRefused(
            f"the replacement for {path} was written but could not be put in place ({exc}), so "
            f"{path} is unchanged and still says what it said. Set the value by hand."
        ) from exc

    # The rename is already durable enough to never show a half-file; syncing
    # the directory only decides whether it survives a power cut, so a
    # filesystem that will not let us is not a reason to report a failed write.
    with contextlib.suppress(OSError):
        directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _discard(scratch: Path) -> None:
    """Remove a temporary file, never turning cleanup into the reported failure."""
    with contextlib.suppress(OSError):
        scratch.unlink()


def _edit_toml(text: str, table: str, key: str, rendered: str | None) -> str:
    """``text`` with ``key`` in ``[table]`` set to ``rendered``, or removed if None.

    ``rendered`` is the value's exact TOML text - already quoted for a string,
    already bracketed for an array - so this function knows nothing about
    types and cannot mis-quote one.

    The file's own line ending is preserved throughout: a replaced assignment
    keeps the terminator of the line it replaces, an inserted one takes the
    terminator of the header it goes under, and a table appended to a file
    without one uses whatever ending the rest of the file already used.
    """
    lines = _split_lines(text)
    newline = _file_newline(text)
    assignment = None if rendered is None else f"{key} = {rendered}"
    wanted = (_key_parts(table), False)
    start = next(
        (index for index, line in enumerate(lines) if _table_header(line) == wanted),
        None,
    )

    if start is None:
        if assignment is None:
            return text
        prefix = text if not text or _terminator(text) else text + newline
        separator = newline if prefix else ""
        return f"{prefix}{separator}[{table}]{newline}{assignment}{newline}"

    after_start = range(start + 1, len(lines))
    end = next(
        (index for index in after_start if _table_header(lines[index]) is not None),
        len(lines),
    )
    for index in range(start + 1, end):
        if _assignment_key(lines[index]) != key:
            continue
        if assignment is None:
            return "".join([*lines[:index], *lines[index + 1 :]])
        replaced = assignment + _terminator(lines[index])
        return "".join([*lines[:index], replaced, *lines[index + 1 :]])

    if assignment is None:
        return text
    # A header on the file's last line may have no terminator of its own; it
    # needs one before anything can follow it, and the inserted line then ends
    # the file exactly as unterminated as the file already was.
    header_end = _terminator(lines[start]) or newline
    header = _content(lines[start]) + header_end
    inserted = assignment + _terminator(lines[start])
    return "".join([*lines[:start], header, inserted, *lines[start + 1 :]])
