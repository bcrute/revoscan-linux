# Tracking loss: when the scan stops knowing where the object is

Status: contained, then recovered where the evidence allows. A loss is always
detected and always recorded first; the scan then tries, once and within bounds,
to find the object again by measured evidence. When it cannot, the loss stands
and the folder is refused. Nothing here is a promise that a lost scan can be
saved — a refusal is preferable to a guessed pose, and the last section says
what is still deliberately absent.

## The capture this exists because of

`/home/ben/scans/scan/20260828-105933`, on 2026-08-28.

Fifty-one frames were saved and aligned by 11:00:08. One second later, on
displayed frame 53, live alignment failed for the first time. It then failed on
every frame after that, for another two minutes, until the capture stopped at
176 frames. The early failures were a point-to-plane singularity in translation
z — the visible geometry had gone effectively planar, and a plane can slide
along itself without the fit noticing. The later ones had no usable
correspondence at all, because by then the view had moved well past the 12-unit
gate that adjacent alignment works within.

From the operator's chair none of that was visible. The depth and colour
previews kept moving. The frame counter kept climbing. The model in "What has
been scanned so far" simply stopped changing, which looks exactly like a slowly
turning object between refreshes. And the folder afterwards held 176 frames,
every one of which looked on disk precisely like a frame that had aligned.

The problem was never that alignment failed. Alignment failing on a frame is
ordinary. The problem was that 120-odd further frames were written into a scan
they are related to by *nothing at all*, and that nothing on screen or on disk
said so.

## How a loss is detected

**Three consecutive selected frames that could not be placed.** One is a bad
frame. Two is a bad moment — a hand across the view, a burst of noise. Three in
a row with nothing in between is the scan no longer knowing where the object is.

Three routes lead to the same streak, because the operator's symptom is the same
for all of them — a model that has stopped moving while the counter climbs:

- live alignment was attempted and failed (`failure_type` is the exception's
  class name, e.g. `RegistrationError`);
- the frame deprojected to no usable points, so there was nothing to align it by
  (`EmptyTrackingInput`);
- preparing that cloud raised before alignment was ever attempted
  (`UnusableTrackingInput`).

A frame the operator's **Stop** interrupts is none of them. Cancellation is not
a tracking failure and never completes a streak.

A single accepted frame anywhere in the run resets the streak to zero, so an
occasional dropout costs nothing. The threshold is
`scanner3d.capture_sequence.TRACKING_LOSS_STREAK`. Because every selected frame
is either accepted or recorded as one of the three failures above, the streak is
always a run of *numerically consecutive* frames — and the reader enforces that
shape on the record.

This is deliberately *not* the cumulative failure count the live log already
keeps. That number answers "how has this capture been going" and rises steadily
on a scan that is working fine; a capture that ended on it would end healthy
scans.

## What the capture does about it

Containment first, always. Whether a recovery is possible is decided afterwards
and never softens any of this.

- **The model freezes, and says so.** A bold line appears above the model panel:
  `TRACKING LOST after N frames`, which frame the model is frozen at, and that
  the frames after it are not a valid continuation of the scan.
- **Nothing more is written.** No frame reaches the disk, no count moves, no
  pose is placed, and no model or surface is refreshed. Frames keep arriving
  from the scanner and either go into the bounded, in-memory recovery buffer
  below or are dropped; the log says how many at the end of the run.
- **The previews keep running and Stop still works.** Both are answered at the
  ordinary rate. A frozen window would be indistinguishable from the crash this
  is preventing.
- **Pause goes dead and says why.** It is disabled and relabelled
  `Pause (tracking lost)`, and `Pause (recovering tracking)` while the scan is
  looking. Pause places new frames against the model already accepted, which is
  both what has stopped working and what a recovery is doing at that moment, so
  a control still offering it would be offering a repair nothing behind it can
  perform.

Two things happen, in this order, and they are deliberately separate:

1. **Containment.** The capture stops, immediately, before the record is even
   attempted. Stopping is always safe and always right, so nothing gates it.
2. **Reporting.** Only once `tracking-loss.json` is on disk does the capture
   state that it lost tracking.

That order matters because "this capture lost tracking" is a claim about the
disk: it sends the operator to a file and tells every later reader that the
folder explains itself. A capture that could not write the record therefore says
the opposite instead — that there is **no** record in the folder and nothing on
disk marks where these frames stop being one scan — and then fails the run with
that error rather than ending quietly.

## What is left on disk

Every frame written before the loss stays exactly where it was. Nothing is
deleted, renamed, truncated, renumbered, rewritten or moved — not by this and
not by anything downstream.

Beside them, in the run that actually lost tracking — the capture root, or the
`segment-*` directory the capture was writing into — it writes one
`tracking-loss.json`. Per run rather than always at the root, so that a long
capture which loses tracking twice records both: an existing record is never
overwritten, and a second one at the root would either be refused or would
describe frames that are not in the directory it sits in. It is written
completely or not at all, and an existing one is never overwritten. It records:

| Field | What it is |
| --- | --- |
| `kind`, `schema_version` | So a file found loose is identifiable, and so a future format is refused rather than guessed at |
| `threshold` | The consecutive-failure count that was reached |
| `last_aligned_frame_index` | The accepted predecessor: the last frame that both reached disk and aligned. `-1` if nothing in the capture ever aligned |
| `first_failed_frame_index`, `triggering_frame_index` | Where the streak began and where it reached the threshold |
| `failures` | One entry per failure, on consecutive frames: the frame, the failure type, and its message, so the refusal can be *interpreted* later and not merely obeyed. Both texts are non-blank and bounded |
| `classified_by` | That this came from scanner3d's own live alignment, and that the projector, the camera and the object reported nothing. The record must never be read later as a hardware fault it never observed |
| `limitations` | Chiefly: **no pose bridges the failed continuation.** Nothing relates the frames after the loss to the frames before it, and no transform was estimated, interpolated or guessed to stand in for one |

If that record cannot be written, the capture **fails loudly** rather than
carrying on, and it never claims a loss it could not record. An unmarked capture
that has silently stopped tracking is the 2026-08-28 failure exactly, and it is
strictly worse than a capture that ends with an error on screen.

## Recovering from it, automatically

Once — and only once — `tracking-loss.json` is on disk, the capture enters a
distinct state, `recovering_tracking`, and tries to find the object again. The
banner says so, Stop and both previews keep working at the ordinary rate, and
Pause stays disabled and relabelled.

What it does is exactly what a **Resume** after a pause does, because the
problem is the same problem: a small segment of frames has to be placed against
a model it is not a small step away from.

- The frames it collects are held in memory in the same bounded container, with
  the same frame, byte and time ceilings and the same sliding drop of the oldest
  frame. None of them is written, counted, posed or drawn.
- Each is chained to the one before it by the same judged local edge, and a link
  that fails those gates is dropped rather than chained.
- The placement comes from the same strict global relocalization against the
  accepted model, off the stream thread, with the same overlap, residual,
  matched-fraction, convergence and runner-up ambiguity margin, and the same
  attempt ceiling.

Nothing is relaxed, bypassed or duplicated to make a recovery succeed. If fewer
than `MIN_PAUSE_REFERENCE_CLOUDS` frames had aligned before the loss there is no
model to recognise the object by, and no search is attempted at all — the
capture says so at once rather than spending the budget to arrive at the same
answer.

The clock is checked *before* the work on every frame delivered after the loss,
and *before* any search result is looked at — not only after a frame has been
placed in the buffer, and not only when a search is still unfinished. Nobody
asked for an automatic recovery, so nobody is going to end one that is getting
nowhere: a run of frames that cannot be deprojected, cannot be chained or is
refused by the local gates, or a search that never comes back, ends at
`MAX_TENTATIVE_SECONDS` with the search abandoned rather than waited for,
nothing published and the loss left standing.

`MAX_TENTATIVE_SECONDS` is an acceptance boundary, not merely a watchdog for a
search that hangs. A search that does come back, and comes back *accepted*,
after the recovery was allowed to end is dropped unlooked-at: by then nobody can
say the object is still where that answer places it, so no segment is published
and no recovery is announced. An already-finished search is abandoned exactly as
readily as a running one — abandoning never waits either way, and the answer
reaches nobody.

### When it works

One ordinary `segment-*` directory appears, atomically: the held frames, the
bridge record, and inside it a `recovery` object that **binds** the segment to
the loss it closes. Only after that publication returns does the capture clear
the lost banner, add the frames to the count and the model, and carry on
scanning into the new segment. Pause becomes available again once the model
panel reports enough accepted geometry, exactly as at the start of a capture.

The window says `TRACKING RECOVERED`, names the segment, and names **how many
frames were excluded**. That last number matters: the folder now holds more
frames than the scan does, and the capture's ending — whether it ran to its
frame count or was stopped — reports the two apart: how many frames are
reconstructable, and how many further files are on disk purely as the evidence
for the loss. Excluded frames are never counted as part of the scan, and the
count the operator was last shown while it ran never silently falls.

A publication that fails does **not** report a recovery. It fails the capture,
the loss stands, and nothing was written.

| `recovery` field | What it is |
| --- | --- |
| `kind`, `schema_version` | Versioned apart from the bridge and the loss records: three different things |
| `loss_record` | The path *relative to the capture* of the record this closes, its byte count and its SHA-256. The path is compared against the one rebuilt from the run, never followed |
| `predecessor_frame_index`, `segment_start_frame_index` | Repeated from the bridge beside it, so the two can be checked against each other |
| `excluded_frames` | Every frame written while tracking was lost: name, index, byte count and SHA-256, checked against the bytes on disk |
| `limitations` | That the excluded frames are on disk and not in the sequence, and that the seam came from a relocalization rather than from aligning the pair |

### When it does not

If the search stays ambiguous, non-converged, thin, inconsistent or simply runs
out of attempts or time — including a search still running when the time runs
out — the loss is left exactly as recorded. No segment is
written, the held frames are dropped, the model stays frozen, and the window
says `TRACKING LOST and NOT recovered` and asks for Stop and a rescan. It does
not keep retrying, and it does not resume ordinary writing.

### More than one

A long capture can lose tracking more than once, including inside a segment a
previous recovery published. Each run keeps its own record and is closed by its
own recovery, so a second episode neither overwrites the first nor goes
unrecorded. Ordinary operator pauses before, between and after recovered losses
are unaffected and are never mistaken for recoveries.

## Why reconstruction refuses an unclosed loss

`scanner3d.capture_sequence.read_capture_sequence` is the single authority on
what a capture folder holds, and both reconstruction entry points — the CLI's
`scanner3d reconstruct` and the window's Reconstruct — go through it. Neither
carries a check of its own.

It reads a capture as ordered **runs**: the root sweep, then each segment. For
each run that recorded a loss it hands over the frames up to
`last_aligned_frame_index` and no further, keeps the failed frames on disk and
out of the sequence, and then requires the segment immediately after it to close
that exact loss. Closing means all of this agrees with the bytes on disk: the
named loss record's location and SHA-256, the excluded frames' indices, lengths
and digests, the predecessor and start indices, and the relocalization and
local-chain evidence a pause boundary already has to carry.

**A recovery is not valid merely because a bridge happens to follow a loss.**
The frames in between are frames no pose was ever accepted for, so a reader that
closed a loss on adjacency alone would be deciding where somebody's scan resumes
from directory order. A missing, duplicated, stale, reordered, symlinked,
mismatched, partially published or policy-inconsistent recovery refuses the
whole capture, and so does a loss that nothing closes — word for word as it did
before recovery existed.

A recovered capture is handed over with the seam expressed as an ordinary
boundary, so a reconstruction uses the recorded transform *instead of* fitting
that pair, and with capture limitations naming the exclusion and the recovery.
Registering across the seam would hand adjacent ICP two frames that are not a
small step apart, which does not fail — it returns a small residual and a
confident transform for a placement that is nonsense.

Anything loss-record-shaped that cannot be read as a loss record is refused too:
a symlink, a directory, an unparseable or wrong-kind JSON, an unknown schema
version, a record whose own numbers contradict each other or its own failure
list. Schema 1 has exactly one policy and the reader enforces all of it rather
than accepting whatever the file claims: the threshold must be the one threshold
of three, the failures must be one consecutive run of that length carrying
non-blank bounded text, the `classified_by` statement must be the one this
version publishes — a record naming the projector, the camera or the object as
the source of the classification is refused, not believed — and the
no-pose-bridge limitation must be present, because it is the entire content of
the refusal. The one fallback that must never happen is "cannot read it, so the
capture is fine" — that fallback *is* the original failure.

Captures with no loss record are completely unaffected, including legacy
captures and ones with accepted pause/relocalization segments. Reading a healthy
capture costs one `stat` that finds nothing.

## What to do with a folder that was refused

Keep it. The prefix up to `last_aligned_frame_index` is real data, taken while
the scan knew where it was, and the record beside it is the evidence for where
it stopped being one sweep. Rescan the object; the refusal is telling you that
this capture cannot be made into one model, not that the frames are worthless.

A recovered folder is worth understanding rather than tidying. It holds frames
that are deliberately not in the scan — the ones the loss record names — and
deleting them breaks the binding and refuses the capture. They are evidence, and
they cost a few files.

If the loss came early and repeatedly, the failure messages in the record are
worth reading. `singular in translation z` means the scanner was looking at
geometry that had gone effectively flat — move so that something with shape is
in view. Failures naming correspondence usually mean the view moved too far
between frames; slow down.

## What this deliberately does not do

No pose fallback. No point-to-point fallback for singular geometry. No relaxed
gate, no fabricated transform, no background or plane filter, no keyframe or
model tracker, no widened coarse search, and no change to TSDF fusion. The
recovery above reuses the pause path's gates exactly and lowers none of them —
if it could only succeed by relaxing one, it refuses instead.

Nothing here makes tracking itself more robust. A scan that keeps losing the
object will keep losing it, faster than it recovers; better tracking is a
separate piece of work. And nothing here recovers the *excluded* frames. Their
poses were never established and are not estimated afterwards, so they stay raw
evidence beside the scan rather than becoming part of it.

Above all: **a refusal is preferable to a guessed pose.** Every gate here exists
because the alternative failure is silent — a model that looks like a scan,
joined at the wrong angle, with a small residual and a confident number beside
it.
