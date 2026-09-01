# Depth return health: missing data, said out loud

## The report this answers

On 2026-08-28 the owner watched the false-coloured depth preview of
`/home/ben/scans/scan/20260828-110641` go almost entirely black and read it as
"the preview has stopped showing the object".

It had not. The frames themselves had stopped returning depth:

| frame | returned | no return |
| ----- | -------- | --------- |
| 0     | 24.02%   | 75.98%    |
| 80    | 12.36%   | 87.64%    |
| 165   | 3.60%    | 96.40%    |

Frame 165's returns are not spread thinly across the sensor either. By
quadrant, its no-return shares are 89.66% top-left, 99.94% top-right, 96.02%
bottom-left and 100.00% bottom-right, and every return it has lies between
columns 4 and 376 of 640. What is left has collapsed toward the LEFT; the
right-hand third of the sensor is empty.

Those samples are raw code `0`, which this repository reads as *the sensor
returned no depth for this pixel* and nothing else
(`scanner3d.depth_metric.NO_RETURN_SAMPLE`). They are not a distance, and they
are not the renderer dropping anything.

## What changed

**Missing data looks like missing data.** Zero is still excluded from the depth
stretch, from every statistic, from point preparation and from reconstruction —
nothing is inpainted, interpolated, smeared, dilated or carried over from the
previous pixel. What changed is only how it is DRAWN: instead of a flat black
field, which is also what a panel that has stopped drawing looks like, no-return
pixels are drawn as a coarse neutral-grey checker
(`scanner3d.viewer.DEFAULT_PALETTE`, 8-pixel blocks). Both greys are exactly
neutral and every colour on the depth ramp is strongly tinted, so no
interpolation anywhere on the ramp can produce either of them. Every valid
non-zero sample keeps precisely the colour it had before, and a custom palette
that names no alternate colour still renders a flat fill.

**Where the returns are is measured, not guessed.**
`scanner3d.return_health.spatial_returns` counts returns and no-returns for four
fixed quadrants and reports the bounding box of every non-zero sample. Boundary
ownership is exact: the split index is `size // 2`, so an odd middle row or
column belongs to the bottom or right half, and a frame with no returns has no
footprint at all (`null`) rather than a zero-sized box at the origin. The record
is schema-versioned strict JSON.

**How fast they are going is bounded and resettable.**
`scanner3d.return_health.ReturnHealthTracker` holds at most 15 recent yield
fractions — a fixed-length deque and one boolean, whatever the length of a scan
— and judges each frame against the MEDIAN of the frames before it. The
vocabulary is four verdicts: `no returns`, `severely sparse` (under 5% of the
frame), `collapsing` (at most half the recent baseline, and at least 5
percentage points of the whole frame below it) and `usable`. A collapse is held
until the yield is back to 75% of the baseline, so a yield hovering at the
boundary does not flicker on and off once per frame.

The advice names only what the bytes support and what the operator can do: that
the blank areas are pixels where the sensor returned no depth, and to slow down,
move or re-aim so more of the surface faces the scanner squarely, or go back to
where the surface was still showing. It never names a distance, a material, an
exposure or a projector as the cause, because raw code 0 does not distinguish
them.

**It decides nothing.** The verdict accepts and refuses no frame, stops no
capture, moves no tracking gate and changes no reconstruction threshold. It is
shown, and it is written into reports.

## Where it appears

One pure implementation behind all three depth front ends:

- the Qt appliance's depth panel, as a prominent amber non-modal line directly
  under the image with the advice below it, clearing itself the moment a frame
  recovers, with the ordinary numeric return/code line left exactly as it was;
- the standalone Tk window (`scanner3d.viewer.DepthViewer`), as the same two
  lines under the image;
- the local HTTP view, in `/stats.json` as `stats.spatial`, `health` and a
  rendered `health_summary`, and on the page as the same block.

Each of those is one RUN, and a run has two ends.

In the Qt appliance the history and the warning are cleared where a preview or
a capture BEGINS and where it ENDS by any route — the operator's Stop, the
stream reaching its own ceiling, normal completion and failure — so a warning
never outlives the stream that raised it. Each run also has an identity, and
every depth frame is delivered through a slot connected for one run
(`ScannerWindow._frame_slot`). A frame a stopped preview had already emitted is
therefore dropped instead of redrawing the next run's panel or becoming the
first entry in its baseline; Qt delivers those emissions through a queue, so
that crossing is an ordering fact rather than a rare race. A preview's
COMPLETION and FAILURE are queued the same way and carry the same identity
(`ScannerWindow._preview_end_slot`): a completion or failure belonging to a
preview that ended just before Stop cannot end the capture that replaced it,
clear its warning or invalidate its frame slot. A stale failure is still
logged — it is real news about the device — but it changes nothing about the
run that is live.

The Tk window clears its history when it closes. The HTTP server RETIRES a run
rather than clearing one: start and stop each install a fresh state object
(`_RunHandle`), so a poll made before a restarted run's first frame reports
`available: false` rather than answering with, and seeding its baseline from,
the previous run's frame. A request binds the current run once, at the top of
its request, and works only against that object, so a handler still in flight
when the server was stopped cannot read or seed the run that replaced it; it is
told its run ended (`available: false`, `stale_run: true`) instead.

Because that server is threaded, each request takes its snapshot of the
published frame and reaches its verdict together, under a lock that orders the
request workers against each other. Every available answer therefore describes
exactly one frame: its `frame_number`, `fps`, numeric and spatial statistics,
`health.yield_fraction`, verdict and `health_summary` all come from the same
512000 bytes. That lock is never taken by `ViewerServer.show` and the
publication lock is never held across the analysis, so a capture loop is still
never made to wait on a browser; two browsers polling at once wait for each
other instead. The history is still advanced only for a frame number strictly
greater than the last one judged, and a repeat poll is answered from the
recorded payload without measuring the frame again, so the bounded baseline can
neither go backwards nor count one frame twice.

The on-screen warning updates on every displayed frame. The LOG records only
transitions, because a line per frame at ten frames a second buries the moment
it changed.

## Cost

One `struct.unpack` per displayed frame feeds both the colour and the
statistics (`scanner3d.viewer.render_frame`), where the window previously
unpacked the frame twice. That is a guarantee about the unpack and the shared
samples, not a claim that the frame is walked once: colour and statistics still
make their own bounded passes over the tuple that one unpack produced. Colour is now evaluated once per DISTINCT code rather
than once per pixel, and the checker is answered from two precomputed 640-entry
rows. Measured on this machine over `frame_0000.raw` of the capture above:
about 68 ms for colour and statistics together, against about 68 ms for the two
separate calls it replaces — the spatial pass costs roughly 7 ms of that, and
nothing per-pixel is retained beyond the call.

## Reading the capture again

```
.venv/bin/python -m tests.preview_health_replay
.venv/bin/python -m tests.preview_health_replay --capture SOME/CAPTURE --frames 0,80,165
```

Strictly read-only: `is_dir`, `glob`, `stat` and `read_bytes` are the only calls
it makes against the capture, and it records every frame's size, mtime and
SHA-256 before and after the replay and reports whether they agree. Exit 2 if it
could not run, 3 if a file changed under it.

`tests/test_preview_health_report.py` drives that harness against the named
capture, recomputing every count from the raw bytes with its own longhand index
arithmetic, and skips where the capture is not on the machine.

## What is still not known

The large no-return region in that capture is raw evidence and a remaining
optical/capture limitation. Nothing here restores geometry, and nothing here
proves a projector, exposure, material or firmware defect — the raw bytes cannot
distinguish those. No scanner setting was touched by any of this work.
