"""Adapter for the temporary external Revopoint 3DCamera reference probe.

This is a **reference oracle only**. The vendor SDK is not vendored into this
repository, is never downloaded at runtime, and is not the long-term transport
(see ``docs/architecture.md``). Its job for this milestone is to establish
ground truth: what the attached unit can actually deliver, and what its
calibration values are, so the owned Linux transport can later be validated
against measured numbers instead of guesses.

Every entry point here is opt-in: nothing runs unless the caller supplies an SDK
root, and every hardware interaction is wrapped in a hard wall-clock timeout
with process-group cleanup.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from scanner3d.errors import ReferenceProbeError, ReferenceProbeTimeout

#: Environment variable pointing at the SDK's ``thirdparty/3DCamera`` directory.
SDK_ROOT_ENV: Final = "REVOPOINT_SDK_ROOT"

#: Environment variable pointing at an already-built probe binary.
PROBE_BIN_ENV: Final = "SCANNER3D_REFERENCE_PROBE"

#: Files inside the SDK root whose hashes pin the oracle's identity.
PINNED_FILES: Final[tuple[str, ...]] = (
    "include/3DCamera.hpp",
    "linux/x64/lib3DCamera.so",
)

#: Manifest recording the SDK build this project's evidence was measured against.
#:
#: It lives inside the package (not at the repository root) so the same path
#: resolves from a source checkout and from an installed wheel; an installed
#: ``scanner3d doctor`` must be able to verify the oracle without the repo.
DEFAULT_MANIFEST: Final = Path(__file__).resolve().parent / "data" / "sdk-manifest.json"

#: Hard ceiling on any single hardware interaction, per the project's safety rules.
MAX_PROBE_TIMEOUT_S: Final = 15.0
DEFAULT_PROBE_TIMEOUT_S: Final = 12.0

#: Grace period given to each of the SIGTERM and SIGKILL phases of cleanup.
GROUP_GRACE_S: Final = 2.0

#: Bound on draining the probe's pipes once it has been killed.
DRAIN_TIMEOUT_S: Final = 2.0


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SdkIdentity:
    """Hashes of the pinned SDK files, plus their agreement with the manifest."""

    root: str
    files: dict[str, str | None]
    expected: dict[str, str] = field(default_factory=dict)
    manifest_path: str | None = None

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(name for name, digest in self.files.items() if digest is None)

    @property
    def mismatched(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, digest in self.files.items()
            if name in self.expected and digest is not None and digest != self.expected[name]
        )

    @property
    def verified(self) -> bool:
        """True only when every pinned file is present and matches the manifest."""
        return bool(self.expected) and not self.missing and not self.mismatched

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "manifest_path": self.manifest_path,
            "files": dict(self.files),
            "expected": dict(self.expected),
            "missing": list(self.missing),
            "mismatched": list(self.mismatched),
            "verified": self.verified,
        }


def resolve_sdk_root(explicit: Path | str | None = None) -> Path | None:
    """Resolve the SDK root from an explicit path or ``REVOPOINT_SDK_ROOT``."""
    candidate = explicit if explicit is not None else os.environ.get(SDK_ROOT_ENV)
    if not candidate:
        return None
    return Path(candidate).expanduser()


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, str]:
    """Load the pinned hashes; an absent or malformed manifest yields ``{}``."""
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    files = raw.get("files")
    if not isinstance(files, dict):
        return {}
    return {str(k): str(v) for k, v in files.items() if isinstance(v, str)}


def identify_sdk(
    sdk_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> SdkIdentity:
    """Hash the pinned SDK files and compare them against the manifest."""
    files: dict[str, str | None] = {}
    for name in PINNED_FILES:
        path = sdk_root / name
        try:
            files[name] = sha256_file(path)
        except OSError:
            files[name] = None
    expected = load_manifest(manifest_path)
    return SdkIdentity(
        root=str(sdk_root),
        files=files,
        expected=expected,
        manifest_path=str(manifest_path) if expected else None,
    )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of one bounded reference-probe invocation."""

    ran: bool
    command: list[str]
    exit_code: int | None
    timed_out: bool
    elapsed_s: float
    stdout: str
    stderr: str
    data: dict[str, Any] | None
    error: str | None
    sdk: SdkIdentity | None

    def as_dict(self) -> dict[str, object]:
        return {
            "ran": self.ran,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "elapsed_s": round(self.elapsed_s, 3),
            "error": self.error,
            "sdk": self.sdk.as_dict() if self.sdk is not None else None,
            "data": self.data,
            "stdout_tail": self.stdout[-4000:],
            "stderr_tail": self.stderr[-4000:],
        }


def _signal_group(proc: subprocess.Popen[str], pgid: int, sig: int) -> None:
    """Best-effort signal to the whole probe process group, then to the leader."""
    try:
        os.killpg(pgid, sig)
        return
    except ProcessLookupError:
        return
    except PermissionError, OSError:
        pass
    with suppress(ProcessLookupError, OSError, ValueError):
        proc.send_signal(sig)


def _terminate_group(proc: subprocess.Popen[str], *, grace_s: float = GROUP_GRACE_S) -> None:
    """Kill the probe and every process it spawned, leaving no stream running.

    The leader exiting is **not** proof the group is clear. A grandchild that
    ignores ``SIGTERM`` keeps running - and keeps the inherited stdout/stderr
    pipes open - long after the probe binary itself is gone. So the group
    ``SIGKILL`` is unconditional rather than contingent on the leader still
    being alive.

    ``pgid`` is captured before any wait: the probe is started with
    ``start_new_session=True``, so it leads its own group, and the kernel keeps
    that pid reserved while the group still has members.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError, PermissionError, OSError:
        pgid = proc.pid

    _signal_group(proc, pgid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)

    # Unconditional: covers group members that ignored SIGTERM even when the
    # leader has already exited.
    _signal_group(proc, pgid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)


def _drain(
    proc: subprocess.Popen[str], *, timeout_s: float = DRAIN_TIMEOUT_S
) -> tuple[str, str, bool]:
    """Collect the probe's output under a bound; report whether draining failed.

    A survivor holding the inherited pipes can keep ``communicate()`` blocked
    indefinitely, which would defeat the wall-clock budget the whole adapter
    exists to guarantee. When the bound is hit the pipes are closed and the
    loss is reported rather than hidden.
    """
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                with suppress(OSError):
                    pipe.close()
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=1.0)
        return "", "", True
    return stdout or "", stderr or "", False


def run_probe(
    serial: str,
    *,
    sdk_root: Path | str | None = None,
    probe_binary: Path | str | None = None,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    manifest_path: Path = DEFAULT_MANIFEST,
    require_verified_sdk: bool = True,
    json_out: Path | None = None,
) -> ProbeResult:
    """Run the external probe once, under a hard timeout, and parse its JSON.

    Raises :class:`ReferenceProbeError` for setup problems the caller must fix
    (missing SDK, unverified SDK, missing binary). A probe that starts but
    fails or times out is reported as a :class:`ProbeResult`, not an exception,
    so ``doctor`` can fold it into the report.
    """
    if timeout_s <= 0 or timeout_s > MAX_PROBE_TIMEOUT_S:
        raise ReferenceProbeError(
            f"timeout_s must be in (0, {MAX_PROBE_TIMEOUT_S}]; got {timeout_s}"
        )

    root = resolve_sdk_root(sdk_root)
    if root is None:
        raise ReferenceProbeError(
            f"no SDK root supplied; pass --sdk-root or set {SDK_ROOT_ENV}. "
            "The SDK is deliberately not vendored and is never downloaded."
        )
    if not root.is_dir():
        raise ReferenceProbeError(f"SDK root does not exist: {root}")

    identity = identify_sdk(root, manifest_path)
    if require_verified_sdk and not identity.verified:
        detail = []
        if not identity.expected:
            detail.append(f"no pinned hashes in {manifest_path}")
        if identity.missing:
            detail.append(f"missing: {', '.join(identity.missing)}")
        if identity.mismatched:
            detail.append(f"hash mismatch: {', '.join(identity.mismatched)}")
        raise ReferenceProbeError(
            "refusing to run against an unverified SDK (" + "; ".join(detail) + ")"
        )

    binary = probe_binary or os.environ.get(PROBE_BIN_ENV)
    if not binary:
        raise ReferenceProbeError(
            f"no probe binary supplied; pass --probe-binary or set {PROBE_BIN_ENV}. "
            "Build it with reference/probe/build.sh."
        )
    binary_path = Path(binary).expanduser()
    if not (binary_path.is_file() and os.access(binary_path, os.X_OK)):
        raise ReferenceProbeError(f"probe binary is not executable: {binary_path}")

    command = [
        str(binary_path),
        "--serial",
        serial,
        "--timeout-ms",
        str(int(timeout_s * 1000) - 1500),
    ]
    if json_out is not None:
        command += ["--json-out", str(json_out)]

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(root / "linux" / "x64"), env.get("LD_LIBRARY_PATH")) if p
    )

    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    timed_out = False
    drain_failed = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_group(proc)
        stdout, stderr, drain_failed = _drain(proc)
    elapsed = time.monotonic() - started

    data: dict[str, Any] | None = None
    parse_error: str | None = None
    for source in (stdout, _read_optional(json_out)):
        if not source:
            continue
        try:
            candidate = json.loads(source)
        except json.JSONDecodeError as exc:
            parse_error = f"probe output was not valid JSON: {exc}"
            continue
        if isinstance(candidate, dict):
            data = candidate
            parse_error = None
            break

    error: str | None = None
    if timed_out:
        error = f"probe exceeded its {timeout_s:.1f}s budget and was killed"
        if drain_failed:
            error += (
                "; its output pipes were still held open after the group kill, "
                "so stdout/stderr were discarded"
            )
    elif proc.returncode != 0:
        error = f"probe exited {proc.returncode}"
    elif parse_error:
        error = parse_error

    return ProbeResult(
        ran=True,
        command=command,
        exit_code=proc.returncode,
        timed_out=timed_out,
        elapsed_s=elapsed,
        stdout=stdout or "",
        stderr=stderr or "",
        data=data,
        error=error,
        sdk=identity,
    )


def _read_optional(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def leftover_probe_processes(binary_name: str = "probe") -> list[str]:
    """Return any surviving probe processes, for post-run cleanup evidence."""
    pgrep = shutil.which("pgrep")
    if pgrep is None:
        return []
    result = subprocess.run(
        [pgrep, "-af", binary_name],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


__all__ = [
    "DEFAULT_MANIFEST",
    "DRAIN_TIMEOUT_S",
    "GROUP_GRACE_S",
    "MAX_PROBE_TIMEOUT_S",
    "PROBE_BIN_ENV",
    "SDK_ROOT_ENV",
    "ProbeResult",
    "ReferenceProbeTimeout",
    "SdkIdentity",
    "identify_sdk",
    "leftover_probe_processes",
    "load_manifest",
    "resolve_sdk_root",
    "run_probe",
    "sha256_file",
]
