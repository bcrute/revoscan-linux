"""Replay a real capture toward the observable-shell discriminator, and stop at the wall.

Deliberately NOT a ``test_`` module: pytest does not collect it. It is the
physical companion to :mod:`tests.shell_discriminant` and
:mod:`tests.shell_discriminant_harness`, in the role
:mod:`tests.surface_oracle_replay` plays for :mod:`tests.surface_oracle` and
:mod:`tests.surface_oracle_harness`. It reuses that module's read-only
machinery directly - :func:`tests.surface_oracle_replay.fingerprint`,
:func:`tests.surface_oracle_replay.verify_unchanged`,
:func:`tests.surface_oracle_replay.bind_identity` and
:data:`tests.surface_oracle_replay.DEFAULT_CAPTURE` - rather than copying it,
so the two physical replays share one proof of read-only-ness rather than two
that could drift apart.

Run it with::

    .venv/bin/python -m tests.shell_discriminant_replay
    .venv/bin/python -m tests.shell_discriminant_replay --capture SOME/CAPTURE
    .venv/bin/python -m tests.shell_discriminant_replay --limit 8
    .venv/bin/python -m tests.shell_discriminant_replay --out report.json --quiet

STRICTLY READ-ONLY, and it proves it the same way its sibling does: size,
modification time and SHA-256 of every file in the capture are recorded before
and after the replay, and the report carries both plus whether they agree. It
calls ``Path.is_dir``, ``Path.rglob``, ``Path.glob``, ``Path.stat`` and
``Path.read_bytes`` on the capture and nothing else. It never opens a path in
the capture for writing. The ONLY path this module ever writes to is the one
named by ``--out``, and only through
:func:`tests.quality_harness.clear_destination` and
:func:`tests.quality_harness.publish` - the same atomic, no-stale-artifact
contract :mod:`tests.shell_discriminant_harness` itself uses. It opens no
device node, makes no network call, and never fabricates a pose or skips a
frame to make a case pass.

The central fact this module exists to publish
------------------------------------------------
:mod:`tests.shell_discriminant` studies **unsupported closed shells** of a
FUSED synthetic surface: sealed blobs nothing measured near, split into false
surface and correct surface by an analytic label that only the synthetic
corpus can compute. A shell population is, definitionally, a population of
shells found in a fused surface. This capture is one of the ones the shipped
readiness policy refuses before fusion is ever attempted - the same boundary
:mod:`tests.surface_oracle_replay` and :mod:`tests.quality_replay` refuse to
cross - so for the recorded capture this replay is pointed at:

    **no physical unsupported-closed-shell population exists.**

There is nothing to discriminate over, because there is no fused surface for a
shell to belong to. This is not a limitation this module works around: fusing
anyway, past a boundary that has already said a reconstruction of this capture
would not be trustworthy, would manufacture the very population the discussion
above is about. So when the readiness gate refuses, this replay reports the
refusal as its result, sets ``report["shell_population"]`` to ``None``, and
publishes a separate ``shell_population_available: False`` with the exact
reason. It never fuses anyway, never invents a pose the registration did not
produce, and never trims frames to manufacture a passing trajectory.

Registration and readiness only: there is no fusion arm here
--------------------------------------------------------------
This replay reproduces exactly two physical steps - registering the capture's
frames into a trajectory, and putting that trajectory to the shipped readiness
policy - and it stops there, on every path. There is no fusion, no surface
extraction and no shell population in this module, not even behind a branch
that no recorded capture has ever taken. A fusion arm will be added when, and
only when, a real capture actually reaches readiness and the arm can be driven
by the exact policy and backend the appliance uses and covered against
something real. Until then, writing one would mean shipping an untested path
whose numbers nobody has ever seen. Consequently **no physical shell,
discrimination or accuracy claim exists anywhere in this module or its
artifact**: when readiness passes, this replay says plainly that fusion is
deliberately not implemented here and that no shell population exists, rather
than reporting a success it did not earn.

That holds even in the theoretical case where a capture clears the gate, and
it would hold for a fusion arm too, because the other half of the experiment
is unreachable for physical data on principle:
:class:`tests.shell_discriminant.ShellTruth` is computed by
:func:`tests.surface_oracle.analytic_distance` against a scene's DECLARED
exact shape, and a physical capture has no declared exact shape. Analytic
truth does not exist for it.

What this module never does
----------------------------
It never computes :class:`tests.shell_discriminant.ShellTruth` for anything
here - there is no ``tests.shell_discriminant.observe`` call in this file, and
none is added. It never calls :func:`tests.shell_discriminant.select` or
applies a member of :data:`tests.shell_discriminant.RULE_GRID`: a rule was
declared over a labelled population, and there is no labelled population here
to apply one to. And it never routes anything into the product: no export, no
GUI, no capture path and no readiness policy reads this module.

Exit codes, mirroring :mod:`tests.surface_oracle_replay`
----------------------------------------------------------
* **0** - the replay ran to completion (stopping at the readiness boundary or
  at an earlier refusal, which is as far as this module goes), the capture was
  proved unchanged, and, if ``--out`` was given, the artifact was published.
* **1** - the replay started and something did not hold: it failed part-way
  with an unexpected error instead of reaching one of its own stopping points.
  The capture was still fingerprinted afterwards and the read-only result is
  still reported and published.
* **2** - the capture could not be read or bound at all, or the ``--out``
  destination could not be cleared or written to. Nothing was measured, or the
  measurement could not be delivered anywhere the caller asked for it.
* **3** - the before/after fingerprints disagree. This replay changed the
  capture it was told to only read, which this module treats as a failure of
  its own central promise regardless of anything else in the report. It
  dominates every other code above: once a baseline fingerprint exists, EVERY
  exit path - normal return, a refusal, any other error - takes the after
  fingerprint, compares it and reports it, and a changed capture is the
  finding that gets reported.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scanner3d.errors import ScannerError
from scanner3d.preparation import RECONSTRUCT_POINTS, prepare_frame
from scanner3d.reconstruction_readiness import (
    DEFAULT_POLICY,
    coverage_evidence,
    pose_evidence,
)
from tests import shell_discriminant as experiment
from tests.quality_harness import clear_destination, publish
from tests.surface_oracle_replay import (
    DEFAULT_CAPTURE,
    ReplayRefused,
    bind_identity,
    fingerprint,
    verify_unchanged,
)

#: The compute backend the registration step is asked for, and the only
#: backend this module uses at all: there is no fusion here to have one.
REGISTRATION_BACKEND: Final = "auto"

#: What this replay says it stopped at when readiness raises no reason. It is
#: not a success: it is the honest name for "the recorded captures this module
#: ships against have never got here, so no fusion arm has been written yet".
FUSION_NOT_IMPLEMENTED: Final = "fusion_not_implemented"

LIMITATIONS: Final[tuple[str, ...]] = (
    "read-only. No file in the capture was written, renamed, deleted or created, and the "
    "before/after fingerprints in this report are the evidence for that. The only path this "
    "process ever writes to is the one named by --out.",
    "no physical unsupported-closed-shell population is claimed to exist unless this report's "
    "own shell_population.available is true. A shell population requires a fused surface, and "
    "the recorded captures this replay is meant for refuse at the readiness boundary before "
    "fusion is ever attempted.",
    "no label is ever attached to a shell in this report. tests.shell_discriminant.ShellTruth "
    "is computed against a scene's declared exact shape, and a physical capture has no declared "
    "exact shape: analytic truth does not exist for it, full stop, independent of whether "
    "fusion succeeds.",
    "tests.shell_discriminant.select and tests.shell_discriminant.RULE_GRID are never invoked "
    "here. A rule is declared over a labelled population; there is no labelled population in "
    "this module for one to be applied to.",
    "this replay reproduces registration and readiness ONLY. It contains no fusion, no surface "
    "extraction and no shell-population code at all, so it makes no physical shell, "
    "discrimination or accuracy claim. A fusion arm will be written when, and only when, a real "
    "capture reaches readiness and it can be driven by the appliance's exact policy and backend "
    "and covered against something real.",
    "no number here has a physical unit. Every length is in the capture's own depth units, "
    "whose physical meaning this repository has not established.",
    "the poses judged by readiness, and the poses any fusion would be given, were fitted by "
    "THIS run. A pose the registration got wrong and was confident about is measured here as "
    "ordinary motion.",
    "nothing here is routed into the product. No export, live preview, GUI, capture, tracker "
    "or readiness policy reads this module, and no pruning code path exists in scanner3d for "
    "this module to exercise.",
)


def _unavailable(reason: str) -> dict[str, object]:
    """The shape ``shell_population`` and its sibling flags take when there is none.

    Two ways to read "there is no population" out of the artifact: the
    top-level ``shell_population`` is literally ``None`` for a caller that
    only checks that, and ``shell_population_available`` plus
    ``shell_population_unavailable_reason`` carry the same fact with room for
    the exact reason a reader would otherwise have to infer from
    ``stopped_at``.
    """
    return {
        "shell_population": None,
        "shell_population_available": False,
        "shell_population_unavailable_reason": reason,
    }


def replay(capture: Path, *, limit: int | None) -> dict[str, object]:
    """Everything this capture can support saying, and where it stopped if it stopped."""
    identity, paths = bind_identity(capture, limit=limit)
    report: dict[str, object] = {
        "kind": "scanner3d.shell-discriminant-physical-replay",
        "schema_version": 1,
        "identity": identity,
        "backend": REGISTRATION_BACKEND,
        "policy": DEFAULT_POLICY.as_dict(),
        "observable_source": experiment.OBSERVABLE_SOURCE,
        "label_source": experiment.LABEL_SOURCE,
        "limitations": list(LIMITATIONS),
    }
    if len(paths) < 3:
        report["stopped_at"] = "frame_count"
        report["stopped_because"] = f"{len(paths)} usable frames; a sweep needs at least three"
        report.update(
            _unavailable(
                "no fusion was attempted, so there is no fused surface for a shell population "
                "to be drawn from"
            )
        )
        return report

    raws = [path.read_bytes() for path in paths]
    started = time.perf_counter()
    clouds = [prepare_frame(raw, target_points=RECONSTRUCT_POINTS).points for raw in raws]
    prepared_s = time.perf_counter() - started

    from scanner3d.pipeline import reconstruct_sweep

    started = time.perf_counter()
    try:
        result = reconstruct_sweep(clouds, backend=REGISTRATION_BACKEND)
    except ScannerError as error:
        report["stopped_at"] = "registration"
        report["stopped_because"] = str(error)
        report["timing"] = {"prepare_s": round(prepared_s, 3)}
        report.update(
            _unavailable(
                "the frames could not be registered into a trajectory, so no fusion was attempted"
            )
        )
        return report
    registered_s = time.perf_counter() - started

    poses, pose_failures = pose_evidence(clouds, result.poses)
    coverage, coverage_failures = coverage_evidence(clouds, result.poses)
    reasons = [failure.reason for failure in (*pose_failures, *coverage_failures)]
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
    report["readiness_reasons"] = reasons
    report["ready_before_fusion"] = not reasons
    if reasons:
        # The refusal IS the result, exactly as tests.surface_oracle_replay
        # treats it. Fusing anyway to publish shell numbers with a caveat
        # would manufacture the population this module exists to say does not
        # exist for a refused capture.
        report["stopped_at"] = "readiness"
        report["stopped_because"] = (
            "the shipped readiness policy refuses this capture, so no fusion was attempted and "
            "no surface was measured. The refusal is the finding"
        )
        report["timing"] = {
            "prepare_s": round(prepared_s, 3),
            "register_s": round(registered_s, 3),
        }
        report.update(
            _unavailable(
                "the readiness gate refused this capture before fusion. No physical "
                "unsupported-closed-shell population exists for this run, and this replay "
                "makes no physical discrimination claim"
            )
        )
        return report

    # Readiness raised no reason. That is NOT a shell population, and this
    # module does not pretend otherwise: there is no fusion arm here to run,
    # deliberately, because no recorded capture has ever reached this line and
    # an arm nobody has ever driven would be an untested claim rather than a
    # measurement. Say exactly that, and stop.
    report["stopped_at"] = FUSION_NOT_IMPLEMENTED
    report["stopped_because"] = (
        "the readiness policy raised no reason for this capture, and this replay stops here on "
        "purpose: it reproduces registration and readiness only. Fusion is deliberately not "
        "implemented in this module because no recorded capture has ever reached readiness, so "
        "no surface was fused, no surface was extracted and nothing was measured past this point"
    )
    report["timing"] = {
        "prepare_s": round(prepared_s, 3),
        "register_s": round(registered_s, 3),
    }
    report.update(
        _unavailable(
            "fusion is deliberately not implemented in this replay, so no fused surface exists "
            "and no shell population exists. A fusion arm will be added when, and only when, a "
            "real capture reaches readiness and it can be driven by the appliance's exact policy "
            "and backend; no physical shell, discrimination or accuracy claim is made here"
        )
    )
    return report


def _failed_report(*, stopped_at: str, error: BaseException) -> dict[str, object]:
    """The document a run that ended in an error still publishes.

    The read-only proof is attached to a report, so a run that failed part-way
    needs a report to attach it to. It carries the same kind, schema version
    and limitations as a completed run, says where it stopped and why in the
    same two keys, and - like every other stopping point in this module -
    carries no shell population.
    """
    return {
        "kind": "scanner3d.shell-discriminant-physical-replay",
        "schema_version": 1,
        "backend": REGISTRATION_BACKEND,
        "policy": DEFAULT_POLICY.as_dict(),
        "observable_source": experiment.OBSERVABLE_SOURCE,
        "label_source": experiment.LABEL_SOURCE,
        "limitations": list(LIMITATIONS),
        "stopped_at": stopped_at,
        "stopped_because": f"{type(error).__name__}: {error}",
        **_unavailable(
            "the replay ended in an error before reaching one of its own stopping points, so "
            "nothing was measured and no shell population exists"
        ),
    }


def render(document: dict[str, object]) -> str:
    """A short human summary. Every fact in it is already in the artifact."""
    lines = [
        "Observable-shell discriminator - physical replay (no analytic truth, ever).",
        "",
    ]
    stopped_at = document.get("stopped_at")
    if stopped_at is not None:
        lines.append(f"stopped at: {stopped_at}")
        lines.append(f"because: {document.get('stopped_because')}")
    else:
        lines.append("stopped at: (not recorded)")
    lines.append("")
    population = document.get("shell_population")
    if population is None:
        lines.append("shell_population: none")
        lines.append(f"why: {document.get('shell_population_unavailable_reason')}")
    else:  # pragma: no cover - no path in this module produces a population
        lines.append("shell_population: present in this document, which this module never writes")
    read_only = document.get("read_only")
    if isinstance(read_only, dict):
        lines.append("")
        lines.append(f"read-only: {'confirmed' if read_only.get('unchanged') else 'VIOLATED'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.shell_discriminant_replay",
        description=(
            "Physical replay toward the observable-shell discriminator; stops at the "
            "readiness boundary and never fabricates a shell population or a label."
        ),
    )
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="where to write the artifact")
    parser.add_argument(
        "--quiet", action="store_true", help="publish the artifact without rendering it"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.out is not None:
        # Cleared before anything else runs, so a requested destination is
        # never left holding an EARLIER report after a run that could not
        # produce one - the same contract tests.shell_discriminant_harness
        # itself uses.
        try:
            clear_destination(args.out)
        except ScannerError as error:
            print(f"cannot publish to {args.out}: {error}", file=sys.stderr)
            return 2

    before = fingerprint(args.capture) if args.capture.is_dir() else {}
    # From here on there is a baseline, so EVERY exit path takes the after
    # fingerprint, compares it, reports it and lets a changed capture dominate
    # the exit code. A run that blew up half-way is exactly the run whose
    # read-only evidence a reader most needs, so no error is allowed to
    # escape this function with that evidence never taken.
    exit_code = 0
    try:
        report = replay(args.capture, limit=args.limit)
    except ReplayRefused as error:
        print(f"the replay could not be run: {error}", file=sys.stderr)
        report = _failed_report(stopped_at="refused", error=error)
        exit_code = 2
    except ScannerError as error:
        print(f"the replay failed: {type(error).__name__}: {error}", file=sys.stderr)
        report = _failed_report(stopped_at="error", error=error)
        exit_code = 1
    except Exception as error:
        print(f"the replay failed: {type(error).__name__}: {error}", file=sys.stderr)
        report = _failed_report(stopped_at="unexpected_error", error=error)
        exit_code = 1

    after = fingerprint(args.capture) if args.capture.is_dir() else {}
    read_only = verify_unchanged(before, after)
    report["read_only"] = read_only
    unchanged = bool(read_only["unchanged"])

    if args.out is not None:
        try:
            publish(report, args.out)
        except ScannerError as error:
            print(f"the artifact could not be published: {error}", file=sys.stderr)
            return 3 if not unchanged else 2

    if not args.quiet:
        print(render(report))

    if not unchanged:
        return 3
    return exit_code


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
