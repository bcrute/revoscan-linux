"""Replay a real capture through :mod:`scanner3d.tracking`, and report what it decided.

Deliberately NOT a ``test_`` module: pytest does not collect it. Every number
below depends on the capture handed to it and on the machine that ran the
alignment - a travel distance or an ICP residual measured against one physical
capture is not a bound that would mean anything against a different one, and
asserting one in the suite would produce a test that fails on the next capture
and tells nobody why. This is a harness that is run when a decision trace is
wanted, in the same role :mod:`tests.quality_replay` plays for the quality
module and :mod:`tests.tsdf_benchmark` plays for fusion cost.

Run it with::

    .venv/bin/python -m tests.tracking_replay
    .venv/bin/python -m tests.tracking_replay --capture SOME/CAPTURE
    .venv/bin/python -m tests.tracking_replay --frames 40 --json

STRICTLY READ-ONLY. This harness calls exactly four things on the capture
directory and the frames inside it: ``Path.glob``, ``Path.is_file``,
``Path.stat`` and ``Path.read_bytes``. It never calls ``open`` in a writing
mode, never creates a directory, never writes a temporary file, and it never
touches :mod:`scanner3d.capture_sequence`'s own write paths, ``bridge.json``
publication or the live capture pipeline. :func:`run` states the same promise
again immediately above the code that keeps it, because a reader who only
skims the function is exactly the reader this sentence is for.

Why this exists, and the honesty this is built around
-------------------------------------------------------
:class:`scanner3d.tracking.IncrementalTracker` exists because a single
adjacent-frame solve was not enough evidence to place a frame captured by a
scanner carried by hand around a moving chassis: individual views go weak, and
nothing corrects the chain once it has drifted (see that module's own
docstring, "Why this exists"). This harness plays a real capture's raw depth
frames through that tracker, frame by frame, and prints exactly what it did -
which candidates it formed, which agreed, which pose it chose, and where it
refused - so a reader can see the tracker's own reasoning rather than trust a
pass/fail summary standing in for it.

THE EXPECTED HONEST RESULT ON A REAL HAND-CARRIED CAPTURE IS NOT "every frame
is recovered". A capture where the operator walks the scanner past the
chassis, out of range, behind an obstruction or off the object entirely
SHOULD show the tracker refusing those frames, one after another, until the
scanner is back over surface it can place. A run of this harness that shows
zero refusals on a long real capture is not evidence the tracker is good; on
most real captures it is evidence something upstream fed it a suspiciously
easy sequence, or that the capture itself never left the object. Refusal is
not failure here - it is the tracker doing exactly the job
:mod:`scanner3d.tracking` describes: refusing to publish a pose no independent
reference actually supports, and never repeating the last accepted pose as
though tracking had continued through a gap it could not see across (see
:class:`scanner3d.tracking.TrackDecision` - a refusal appends nothing, moves
nothing, and mutates no state the next frame is judged against).

What this harness can prove, and what it cannot
--------------------------------------------------
It can show, for one capture on one machine, the sequence of candidates
:class:`~scanner3d.tracking.IncrementalTracker` formed, which of them agreed,
which origin a pose came from when one was accepted, and the exact refusal
strings when one was not - all of it read straight out of
:meth:`~scanner3d.tracking.TrackDecision.as_dict`, never re-derived. It can
show whether and where a run of consecutive refusals reaches
:data:`scanner3d.capture_sequence.TRACKING_LOSS_STREAK`, which is the same
count the live capture uses to call a stretch of frames a tracking loss - and,
once it does, it stops there, exactly as the live capture does. See "What
happens at tracking loss" below.

It CANNOT show that an accepted pose is correct. :mod:`scanner3d.tracking`'s
own docstring says as much: an accepted pose is agreement between bounded
references under named thresholds, not a measurement of truth, and every
reference here descends from earlier accepted poses. It cannot show that a
refusal was the ONLY correct decision for that frame - only that the tracker's
own gates, run honestly against this capture's evidence, did not clear it.
There is no independently surveyed pose for any frame in a real capture
directory, so this harness offers no accuracy number and adds no pass/fail
gate; see "Exit codes" below.

What this replays
------------------
Frames are discovered with ``sorted(directory.glob("frame_*.raw"))`` - the
raw-frame layout a physical capture directory uses, not the ``*.bin`` layout
:mod:`tests.quality_replay`'s shipped fixture uses. If that glob is empty this
harness falls back to ``sorted(directory.glob("*.bin"))``, so it also runs
against a ``.bin``-layout capture or fixture without a second code path to
drift from this one.

Preparation is deliberately NOT re-derived here. A raw frame is handed
straight to :func:`scanner3d.preparation.prepare_frame`, the ONE function the
live GUI's ordinary-capture path calls - ``scanner3d.gui._load_cloud`` calls
``_prepare_frame``, which calls ``prepare_frame`` with no step of its own in
between - and this harness calls it with the same target,
:data:`scanner3d.gui.PREVIEW_POINTS`, imported rather than copied so a change
to the live preview's own point budget is a change this harness picks up
automatically instead of a second literal to keep in step by hand. That one
call already does the deprojection, the depth clip, the deterministic
thinning, the speckle rejection and the support-plane suppression the live
preview does - see :mod:`scanner3d.preparation`'s own docstring for why that
is the order and why it is one function rather than two copies of the same
four steps. There is no ``--stride`` option here any more: the live path has
no such concept, a stride this harness invented on top of the shared
preparation would be a fifth step the live capture never runs, and offering
the tracker a differently-sampled cloud than the appliance offers it would
make this harness's decisions describe a program that does not exist. Every
count :func:`~scanner3d.preparation.PreparedCloud.as_dict` returns -
deprojected, clipped, thinned, speckle-removed, support-removed, kept - is
printed alongside the tracker's own decision for the same frame, so a reader
can see what suppression removed before the tracker was ever asked to place
what was left.

A frame whose byte length does not match the device's own expected frame size
is SET ASIDE with a reason, exactly as :func:`tests.quality_replay.run` sets
one aside, rather than raising and losing every other frame's report to one
bad file.

What happens at tracking loss
------------------------------
:data:`scanner3d.capture_sequence.TRACKING_LOSS_STREAK` consecutive refusals
is the live capture's own durable, TERMINAL definition of tracking loss - see
that module's "The other kind of break". The live capture does not keep
offering frames to the tracker once it has recorded one; the frames written
after it are excluded, not retried, and nothing about them is ever described
as "the tracker recovered". So once this harness's own consecutive-refusal
streak reaches that threshold, it stops offering frames to the tracker too.
No candidate is formed, no ``IncrementalTracker.offer`` call is made and no
tracker state is touched for anything after that frame - the tracker is left
exactly as the terminal refusal left it. Every later frame still on disk is
still listed in the report, because a reader counting files in the directory
needs to be able to account for every one of them, but each is marked
``"offered": false`` with a reason naming the frame the loss was reached at.
This harness therefore never reports a later "reacquisition": there is no
mechanism left running past the terminal loss that a later frame could
reactivate, and a frame that happens to look placeable is not offered the
chance to prove it, exactly as it would not be on the live path.

Determinism
-----------
:class:`~scanner3d.tracking.IncrementalTracker` reads no clock, no device and
no random number (see its own docstring), so replaying the same frames in the
same order on the same machine reaches the same decisions every time. Per-
frame and total wall-clock seconds ARE printed, because how long a real run
took is useful to know, but they are returned apart from the decision report -
under this harness's own ``"timing"`` key, exactly where
:mod:`tests.quality_replay` puts its own - so nothing about a decision is ever
timing-dependent and no reader can mistake "how long this took" for "what was
decided".

Exit codes
----------
0 - the replay ran to completion, whatever it decided about the frames. There
is no pass/fail gate to fail: this harness measures what a real capture's
tracker decisions look like, and a capture full of refusals is not this
harness failing, for the reasons above. 2 - the harness could not run at all
(bad ``--capture``, an unreadable frame, or any other ``ValueError``,
``OSError`` or :class:`~scanner3d.errors.ScannerError`); one line on stderr,
``tracking_replay: {exc}``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scanner3d.capture_sequence import TRACKING_LOSS_STREAK
from scanner3d.errors import ScannerError
from scanner3d.gui import PREVIEW_POINTS
from scanner3d.preparation import PreparedCloud, prepare_frame
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker, TrackDecision
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES

#: Where this harness looks if ``--capture`` is not given: the physical
#: capture directory named in the task that asked for this harness. Unlike
#: tests/quality_replay.py's DEFAULT_CAPTURE this is not a fixture shipped
#: with the repository - it is one machine's real scan directory, and a
#: caller without that directory must pass ``--capture`` explicitly.
DEFAULT_CAPTURE: Final = Path("/home/ben/scans/scan/20260828-105933")

#: Frames read by default when no ``--frames`` is given. Small enough that an
#: accidental bare invocation against a long real capture does not run for
#: minutes; large enough to carry the tracker past its anchor frame and into
#: at least a few ordinary decisions.
DEFAULT_FRAME_COUNT: Final = 60


#: The reason recorded on every frame this harness reads but does not offer to
#: the tracker because a terminal tracking loss was already reached. Built
#: once, from a name rather than a number baked into the sentence, so the
#: message a reader sees names the frame the loss actually happened at rather
#: than repeating a fixed string that would be a small lie for every source
#: after the first.
def _not_offered_reason(loss_source: str) -> str:
    return (
        f"the capture had already reached its durable {TRACKING_LOSS_STREAK}-refusal tracking "
        f"loss at {loss_source}; the live capture does not continue tracking past it (see "
        "scanner3d.capture_sequence's own 'other kind of break'), so neither does this"
    )


def discover_frames(directory: Path) -> list[Path]:
    """Every raw frame the capture directory holds, sorted by name.

    ``frame_*.raw`` is the layout a physical capture directory actually uses -
    this is a raw-frame harness, not a rebuild of tests/quality_replay.py's
    ``*.bin`` fixture reader. The ``*.bin`` glob is tried as a fallback, and
    only when the ``.raw`` glob is empty, so this harness also runs against a
    ``.bin``-layout capture or fixture without a second discovery function to
    keep in step with this one.
    """
    raw_frames = sorted(directory.glob("frame_*.raw"))
    if raw_frames:
        return raw_frames
    return sorted(directory.glob("*.bin"))


def _preparation_summary(prepared: PreparedCloud) -> str:
    """One compact line naming what prepare_frame did to this frame.

    Read straight off :class:`~scanner3d.preparation.PreparedCloud`, never
    recomputed, for the same reason :func:`_candidate_summary` reads a
    candidate straight off :class:`~scanner3d.tracking.TrackCandidate`: what
    this harness prints must never be able to disagree with what
    :meth:`~scanner3d.preparation.PreparedCloud.as_dict` would have said about
    the same frame.
    """
    return (
        f"prepared: deprojected={prepared.deprojected_points} "
        f"clipped={prepared.clipped_points} thinned={prepared.thinned_points} "
        f"speckle_removed={prepared.speckle_removed} "
        f"support_removed={prepared.support_removed} ({prepared.support_reason}) "
        f"kept={prepared.kept_points}"
    )


def _candidate_summary(decision: TrackDecision) -> list[dict[str, object]]:
    """The evidence :meth:`main` prints for every candidate a frame formed.

    Read straight off :class:`~scanner3d.tracking.TrackCandidate`, never
    recomputed, so what this harness prints can never disagree with what
    :meth:`~scanner3d.tracking.TrackDecision.as_dict` would have said about the
    same decision.
    """
    rows: list[dict[str, object]] = []
    for candidate in decision.candidates:
        rows.append(
            {
                "origin": candidate.origin,
                "rmse": candidate.edge.rmse,
                "converged": candidate.edge.converged,
                "matched_fraction": candidate.edge.matched_fraction,
                "used_fraction": candidate.edge.used_fraction,
                "overlap": candidate.overlap,
                "accepted": candidate.accepted,
            }
        )
    return rows


def _recovery_summary(decision: TrackDecision) -> str:
    """What the bounded coarse recovery did for this frame, or nothing at all.

    Empty for every unambiguous ordinary placement, and that emptiness is itself
    the report: :attr:`~scanner3d.tracking.TrackDecision.recovery` is ``None``
    exactly when recovery arbitration was never reached. An ordinary placement
    selected as the distinctly tighter fallback from tied clusters instead
    carries the failed recovery evidence that was considered first, so a reader
    can distinguish "no search" from "stronger evidence did not confirm a
    replacement, and the bounded ordinary fallback remained".

    Read straight off :class:`~scanner3d.tracking.RecoveryEvidence`, never
    recomputed, for the same reason :func:`_candidate_summary` reads a candidate
    straight off :class:`~scanner3d.tracking.TrackCandidate`.
    """
    recovery = decision.recovery
    if recovery is None:
        return ""
    if not recovery.attempted:
        return f"    recovery: not attempted - {recovery.refusals[0] if recovery.refusals else ''}"
    work = recovery.work
    head = (
        f"    recovery: {'RECOVERED' if recovery.accepted else 'refused'} "
        f"triggered_by={recovery.triggered_by} "
        f"partitions={[list(partition.frames) for partition in recovery.partitions]} "
        f"partition_pose_spans={[round(p.pose_span, 3) for p in recovery.partitions]} "
        f"overlap_margin={recovery.coarse_overlap_margin:.3f} "
        f"overlap_unambiguous={recovery.coarse_overlap_unambiguous} "
        f"selected_rank={recovery.selected_rank} "
        f"selected_overlap={recovery.selected_coarse_overlap:.3f} "
        f"selected_margin={recovery.selected_overlap_margin:.3f} "
        f"selected_unambiguous={recovery.selected_overlap_unambiguous} "
        f"score_margin={recovery.coarse_score_margin:.3f} "
        f"score_unambiguous={recovery.coarse_score_unambiguous} "
        f"degenerate=({recovery.coarse_source_degenerate}, "
        f"{recovery.coarse_reference_degenerate}) "
        f"best_overlap={recovery.best_coarse_overlap:.3f} "
        f"agreed_within={[round(value, 3) for value in recovery.confirmation_disagreements]}"
    )
    cost = (
        f"    recovery work: coarse_source_points={work.coarse_source_points} "
        f"refine_source_points={work.refine_source_points} "
        f"coarse_searches={work.coarse_searches} "
        f"orientations_ranked={work.coarse_orientations_ranked} "
        f"scored={work.coarse_hypotheses_scored} "
        f"refined={work.hypotheses_refined} confirmed={work.confirmations_refined} "
        f"icp_solves={work.icp_solves} "
        f"reference_frames={work.reference_frames} reference_points={work.reference_points} "
        f"coarse_reference_points={work.coarse_reference_points}"
    )
    lines = [head, cost]
    for hypothesis in recovery.hypotheses:
        lines.append(
            f"      rank {hypothesis.rank} ({hypothesis.coarse_origin}) "
            f"coarse_overlap={hypothesis.coarse_overlap:.3f} "
            f"{'ok' if hypothesis.accepted else 'refused'}"
            + (f": {hypothesis.refusals[0]}" if hypothesis.refusals else "")
        )
    if not recovery.accepted and recovery.refusals:
        lines.append(f"      refused: {recovery.refusals[0]}")
    return "\n".join(lines)


def _print_decision(
    source: str, prepared: PreparedCloud, decision: TrackDecision, elapsed: float
) -> None:
    """One line of the frame-by-frame report this harness exists to show.

    ``source`` is the FILE the frame came from and it leads the line, because
    :attr:`~scanner3d.tracking.TrackDecision.frame` is the tracker's own
    accepted-slot number and a refusal does not advance it - a refusal accepts
    nothing, so the next offer is the same slot. That is exactly right inside
    the tracker and useless to an operator holding a folder of files, who needs
    to know WHICH frame stopped being placeable. Both numbers are printed.

    Everything else on the line is read off the decision itself, so a reader
    comparing this output to the JSON form (``--json``) sees the same numbers
    either way. Cadence is printed on the same line, always at the end, so it
    reads as what it is - a wall-clock fact, never a decision input.
    """
    status = "accepted" if decision.accepted else "REFUSED"
    agreeing = ", ".join(decision.agreeing) if decision.agreeing else "none"
    passing = ", ".join(decision.passing) if decision.passing else "none"
    candidate_bits = "; ".join(
        f"{row['origin']}(rmse={row['rmse']:.4f} converged={row['converged']} "
        f"matched={row['matched_fraction']:.2f} used={row['used_fraction']:.2f} "
        f"overlap={row['overlap']:.2f} {'ok' if row['accepted'] else 'refused'})"
        for row in _candidate_summary(decision)
    )
    line = (
        f"{source}: {status} slot={decision.frame:04d} points={prepared.kept_points} "
        f"origin={decision.origin} agreeing=[{agreeing}] passing=[{passing}] "
        f"travel={decision.travel:.3f} "
        f"keyframes_after={len(decision.keyframes)} candidates=[{candidate_bits}] "
        f"({elapsed:.3f}s)"
    )
    if not decision.accepted and decision.refusals:
        line += f" | first refusal: {decision.refusals[0]}"
    print(line)
    print(f"    {_preparation_summary(prepared)}")
    recovery = _recovery_summary(decision)
    if recovery:
        print(recovery)


def run(
    capture: Path, *, frame_count: int, report_each: bool = True
) -> tuple[dict[str, object], dict[str, object]]:
    """Replay ``capture`` through a fresh :class:`~scanner3d.tracking.IncrementalTracker`.

    Returns the JSON-able report and a separate timing dict, apart from each
    other on purpose - see the module docstring's "Determinism" section - so
    :func:`main` can put timing under its own top-level key and nothing here
    is tempted to fold it into a decision.

    STRICTLY READ-ONLY: only ``Path.glob``, ``Path.is_file``, ``Path.stat`` and
    ``Path.read_bytes`` are called on ``capture`` or anything inside it. This
    function never opens a path in that directory for writing, never creates a
    directory and never starts or touches a live capture.
    """
    if not capture.is_dir():
        raise ValueError(
            f"{capture} is not a directory. Point --capture at a capture root; this harness "
            "reads one, it never creates one"
        )
    overall_started = time.perf_counter()

    all_paths = discover_frames(capture)
    selected = all_paths[: max(0, frame_count)]
    read = [(path.stem, path.read_bytes()) for path in selected]

    # A wrong-sized frame cannot be deprojected - prepare_frame refuses it,
    # correctly - and letting that refusal escape would replace the whole
    # report with a traceback over one bad file. Set aside and named instead,
    # exactly as tests/quality_replay.py's run() does for the same reason.
    frames: list[tuple[str, bytes]] = []
    set_aside: list[dict[str, object]] = []
    for name, raw in read:
        if len(raw) == EXPECTED_FRAME_BYTES:
            frames.append((name, raw))
            continue
        set_aside.append(
            {
                "name": name,
                "bytes": len(raw),
                "expected_bytes": EXPECTED_FRAME_BYTES,
                "reason": "byte count is not the frame size this device records, so the frame "
                "cannot be prepared and was not offered to the tracker.",
            }
        )

    tracker = IncrementalTracker(policy=DEFAULT_POLICY)
    entries: list[dict[str, object]] = []
    offered_decisions: list[tuple[str, TrackDecision]] = []
    frame_seconds: list[float] = []

    current_streak = 0
    longest_streak = 0
    # The source a terminal, durable tracking loss was reached at - the frame
    # whose refusal completed the TRACKING_LOSS_STREAK run - or None while the
    # tracker has not yet been offered such a run. Once set this never clears:
    # the whole point is that the live capture's own loss is TERMINAL, not a
    # state a later frame's placement could undo. See the module docstring's
    # "What happens at tracking loss".
    terminal_loss_at: str | None = None

    for name, raw in frames:
        if terminal_loss_at is not None:
            reason = _not_offered_reason(terminal_loss_at)
            entries.append({"source": name, "offered": False, "reason": reason})
            if report_each:
                print(f"{name}: NOT OFFERED - {reason}")
            continue

        frame_started = time.perf_counter()
        prepared = prepare_frame(raw, target_points=PREVIEW_POINTS)
        decision = tracker.offer(prepared.points)
        frame_seconds.append(time.perf_counter() - frame_started)

        offered_decisions.append((name, decision))
        entries.append(
            {
                "source": name,
                "offered": True,
                "preparation": prepared.as_dict(),
                **decision.as_dict(),
            }
        )
        if report_each:
            _print_decision(name, prepared, decision, frame_seconds[-1])

        if decision.accepted:
            current_streak = 0
        else:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
            if current_streak >= TRACKING_LOSS_STREAK:
                terminal_loss_at = name

    decisions = [decision for _name, decision in offered_decisions]
    accepted = [decision for decision in decisions if decision.accepted]
    refused = [decision for decision in decisions if not decision.accepted]
    origin_counts: dict[str, int] = {}
    for decision in accepted:
        assert decision.origin is not None
        origin_counts[decision.origin] = origin_counts.get(decision.origin, 0) + 1
    rescues = sum(1 for decision in accepted if decision.rescue)
    # Counted over every OFFERED decision, not only the accepted ones: a
    # recovery that ran and refused is exactly as much a fact about this capture
    # as one that placed a frame, and a report that only counted the successes
    # would make the fallback look free.
    attempted = [
        decision
        for decision in decisions
        if decision.recovery is not None and decision.recovery.attempted
    ]
    recovered = [
        decision
        for decision in attempted
        if decision.recovery is not None and decision.recovery.accepted
    ]
    declined = [
        decision
        for decision in decisions
        if decision.recovery is not None and not decision.recovery.attempted
    ]
    coarse_searches = sum(
        decision.recovery.work.coarse_searches for decision in attempted if decision.recovery
    )
    recovery_solves = [
        decision.recovery.work.icp_solves for decision in attempted if decision.recovery
    ]

    first_refused = next(((name, d) for name, d in offered_decisions if not d.accepted), None)
    not_offered_after_loss = sum(1 for entry in entries if entry["offered"] is False)

    summary: dict[str, object] = {
        "frames_read": len(read),
        "frames_set_aside": len(set_aside),
        "frames_offered_to_tracker": len(offered_decisions),
        "frames_not_offered_after_terminal_loss": not_offered_after_loss,
        "frames_accepted": len(accepted),
        "frames_refused": len(refused),
        "placements_by_origin": origin_counts,
        "model_only_rescues": rescues,
        "coarse_recovery_attempts": len(attempted),
        "coarse_recovery_placements": len(recovered),
        "coarse_recoveries_declined_before_search": len(declined),
        "coarse_searches_run": coarse_searches,
        "coarse_recovery_worst_icp_solves": max(recovery_solves) if recovery_solves else 0,
        "coarse_recovery_policy_icp_solve_ceiling": tracker.policy.max_recovery_solves,
        "coarse_recovery_refusals": [
            {
                "source": name,
                "slot": decision.frame,
                "triggered_by": decision.recovery.triggered_by,
                "refusal": decision.recovery.refusals[0] if decision.recovery.refusals else None,
            }
            for name, decision in offered_decisions
            if decision.recovery is not None
            and decision.recovery.attempted
            and not decision.recovery.accepted
        ],
        "first_refused_frame": (
            None
            if first_refused is None
            else {
                "source": first_refused[0],
                "slot": first_refused[1].frame,
                "refusals": list(first_refused[1].refusals),
            }
        ),
        "longest_consecutive_refusal_run": longest_streak,
        "tracking_loss_streak_threshold": TRACKING_LOSS_STREAK,
        "tracking_loss_reached_at_source": terminal_loss_at,
    }

    timing: dict[str, object] = {
        "offered_frames_timed": len(frame_seconds),
        "offered_frame_seconds_mean": (statistics.mean(frame_seconds) if frame_seconds else None),
        "offered_frame_seconds_median": (
            statistics.median(frame_seconds) if frame_seconds else None
        ),
        "offered_frame_seconds_worst": max(frame_seconds) if frame_seconds else None,
        "offered_frame_seconds_total": sum(frame_seconds) if frame_seconds else 0.0,
        "total_seconds": time.perf_counter() - overall_started,
    }

    report: dict[str, object] = {
        "capture": str(capture),
        "preview_points": PREVIEW_POINTS,
        "frames_on_disk": len(all_paths),
        "frames_set_aside": set_aside,
        "tracker_policy": tracker.policy.as_dict(),
        "decisions": entries,
        "summary": summary,
    }
    return report, timing


def _print_summary(summary: dict[str, object]) -> None:
    """The end-of-run report, printed apart from the per-frame lines above it.

    Named fields, not a dumped dict, so the honesty claims the module
    docstring makes - refusal is expected, tracking loss is terminal - are
    legible on a terminal and not just in ``--json`` output.
    """
    print()
    print("summary:")
    print(f"  frames read: {summary['frames_read']}")
    print(f"  frames set aside (wrong size): {summary['frames_set_aside']}")
    print(f"  frames offered to tracker: {summary['frames_offered_to_tracker']}")
    not_offered = summary["frames_not_offered_after_terminal_loss"]
    print(f"  frames not offered (after terminal tracking loss): {not_offered}")
    print(f"  frames accepted: {summary['frames_accepted']}")
    print(f"  frames refused: {summary['frames_refused']}")
    print(f"  placements by origin: {summary['placements_by_origin']}")
    print(f"  model-only rescues: {summary['model_only_rescues']}")
    print(f"  bounded coarse recovery attempts: {summary['coarse_recovery_attempts']}")
    print(f"  frames placed by a coarse recovery: {summary['coarse_recovery_placements']}")
    declined = summary["coarse_recoveries_declined_before_search"]
    print(f"  refusals the recovery declined to search (already proved a jump): {declined}")
    print(f"  coarse searches run: {summary['coarse_searches_run']}")
    print(
        f"  worst recovery ICP solves for one frame: "
        f"{summary['coarse_recovery_worst_icp_solves']} "
        f"(policy ceiling {summary['coarse_recovery_policy_icp_solve_ceiling']})"
    )
    refused_recoveries = summary["coarse_recovery_refusals"]
    assert isinstance(refused_recoveries, list)
    for entry in refused_recoveries:
        print(f"    {entry['source']} ({entry['triggered_by']}): {entry['refusal']}")
    first_refused = summary["first_refused_frame"]
    if first_refused is None:
        print("  first refused frame: none - every offered frame was accepted")
    else:
        assert isinstance(first_refused, dict)
        print(
            f"  first refused frame: {first_refused['source']} "
            f"(tracker slot {first_refused['slot']})"
        )
        for reason in first_refused["refusals"]:
            print(f"    - {reason}")
    print(f"  longest consecutive refusal run: {summary['longest_consecutive_refusal_run']}")
    threshold = summary["tracking_loss_streak_threshold"]
    reached = summary["tracking_loss_reached_at_source"]
    if reached is None:
        print(
            f"  tracking loss (>= {threshold} consecutive refusals, "
            "scanner3d.capture_sequence.TRACKING_LOSS_STREAK): not reached"
        )
    else:
        print(
            f"  tracking loss (>= {threshold} consecutive refusals, "
            f"scanner3d.capture_sequence.TRACKING_LOSS_STREAK): reached at {reached}. This is "
            "TERMINAL - no frame after it was offered to the tracker, and nothing below is a "
            "reacquisition."
        )


def _print_timing(timing: dict[str, object]) -> None:
    """Wall-clock cadence, printed apart from the summary - never a decision input."""
    print()
    print("timing (wall clock on this machine, not a decision input):")
    print(f"  offered frames timed: {timing['offered_frames_timed']}")
    for key in (
        "offered_frame_seconds_mean",
        "offered_frame_seconds_median",
        "offered_frame_seconds_worst",
        "offered_frame_seconds_total",
        "total_seconds",
    ):
        value = timing[key]
        print(f"  {key}: {value:.3f}s" if isinstance(value, float) else f"  {key}: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=DEFAULT_CAPTURE,
        help="a capture directory to read (read-only)",
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAME_COUNT, dest="frame_count", help="frames to read"
    )
    parser.add_argument(
        "--json", action="store_true", help="print one JSON document instead of a text report"
    )
    args = parser.parse_args(argv)

    # Every refusal below is a statement about the INPUT - a directory that is
    # not one, or a frame this harness could not read - each already carrying
    # a sentence saying what is wrong. Exit 2, same convention
    # tests/quality_replay.py uses, because there is no pass/fail gate here to
    # distinguish it from (see the module docstring's "Exit codes").
    try:
        report, timing = run(
            args.capture,
            frame_count=args.frame_count,
            # In JSON mode the document IS the report, and per-frame text
            # printed alongside it would make stdout something no parser can
            # read - the one thing --json exists to guarantee.
            report_each=not args.json,
        )
    except (ValueError, OSError, ScannerError) as exc:
        sys.stderr.write(f"tracking_replay: {exc}\n")
        return 2

    if args.json:
        document_payload = dict(report)
        document_payload["timing"] = timing
        # Serialised whole before a byte reaches stdout, and strictly: the
        # default json encoder emits NaN/Infinity for non-finite floats, which
        # are not valid JSON, and building the string first means a violation
        # refuses the document rather than truncating a half-written one on
        # stdout. Same convention as tests/quality_replay.py's main().
        try:
            document = json.dumps(document_payload, indent=2, sort_keys=False, allow_nan=False)
        except ValueError as exc:  # pragma: no cover - every recorded value is checked
            sys.stderr.write(f"tracking_replay: the report is not strict JSON: {exc}\n")
            return 2
        sys.stdout.write(document + "\n")
        return 0

    # The per-frame lines were already printed by run() as each decision was
    # made, so only the summary and timing remain here.
    assert isinstance(report["summary"], dict)
    _print_summary(report["summary"])
    _print_timing(timing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
