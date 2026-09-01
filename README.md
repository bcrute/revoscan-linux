# revoscan-linux

A Linux-native toolchain for the [Revopoint MINI](https://www.revopoint3d.com/) structured-light
3D scanner (USB `2207:110c`). Revopoint ships no Linux consumer software, so this builds the
whole path — capture, registration, reconstruction and a desktop application — from the USB bus
up, with no vendor SDK at runtime.

## What works

```
scanner3d-gui                                    # desktop application
scanner3d config --set-captures-dir ~/scans      # where captures go, once
scanner3d sweep                                  # capture into ~/scans/sweep/<date>-<time>
scanner3d reconstruct FRAMES_DIR --backend scipy # register and score it
```

The window shows three things at once: the colour camera for positioning, the depth camera for
what is actually being captured, and the accumulated model — orbitable, with under-scanned
surface in soft red and the frame that has just arrived lifted towards white on top of that. A
live readout says whether your rotation speed is in the band that registers well.

**A depth frame the sensor returned nothing for no longer looks like a broken panel.** Raw code 0
means "no depth for this pixel" and is still excluded from the stretch, the statistics, point
preparation and reconstruction — nothing is filled in, interpolated or smeared. It is now DRAWN as
a coarse neutral-grey checker instead of flat black, which the depth ramp can never produce, and
every valid sample keeps exactly the colour it had. Beside the picture, a bounded return-health
line says which of `no returns`, `severely sparse`, `collapsing` or `usable` this frame is, names
the emptiest quadrant and the footprint of what is left, and asks you to slow down, re-aim for
more face-on overlap, or go back to where the surface was still showing. It never names a cause
the raw bytes cannot support, it is advisory only — it accepts no frame and stops no capture — and
its history is a fixed 15 frames that resets between runs. The same verdict reaches the standalone
Tk window and the local HTTP view. On the capture that prompted this, returns fell from 24.02% of
the frame to 3.60% and collapsed to the left of the sensor; that black region is raw evidence and
a remaining optical limitation, not a proved defect. See
[docs/return-health.md](docs/return-health.md).

The model panel draws **the points while the scan is still running**, and offers a lit surface
beside them as an experimental preview you tick on. Every accepted frame is folded into one voxel
occupancy grid — a set union, so a frame costs the frame and is then forgotten — and the grid is
marched every few alignments: measured on this machine at 0.32 s over 254k cells and 1.30 s over
the 1M cells it is allowed at most, against the ~0.7 s each frame already costs. A refresh that
expensive is then held off until the stream thread has had three times as long to itself, so the
capture keeps the thread.

The cell edge is measured from the scan's own point spacing rather than chosen, and the panel says
what it is. It is never made coarser to keep the picture affordable: an object that outgrows the
live budget **loses its surface rather than its resolution**, with the cell size, the grid it
would have needed and the budget said beside the coverage numbers, and the panel falls back to the
points. A coarser preview is not a cheaper version of the same shape — the occupancy surface
stands as much as a voxel outside its samples, so doubling the edge doubles that error and
swallows thin features.

**The points are the view the panel opens on**, and the surface checkbox is off until you ask for
it. That is a decision from the scans themselves: capture 20260827-152842 held its finest cell
edge over 116 frames and never coarsened, and the surface still read as lumpy where the dots read
as clean. An occupancy cell joins the surface when its neighbours are occupied, so what is marched
is faceted at the cell size, stands as much as a cell outside the samples, and draws a small
disagreement between two accepted poses as solid shape. The preview stays because it answers
“what shape has this become” faster than a cloud does, not because it is the more trustworthy
picture. Its cells were also placed by the poses accepted **as the scan ran**, and no pose
correction reaches back into them.

**Reconstruct no longer calls that mesher.** It registers every frame, then fuses a *truncated
signed-distance field* from the raw frames at the corrected poses and marches its zero level set,
so its surface **passes through** the samples instead of wrapping the cells they filled. Measured
against a synthetic sphere of known radius, with both methods given the same 1.04 voxel: the fused
surface sits 0.032 from the true radius where the occupancy boundary sits 0.747 — a 23x smaller
median error, and 9x smaller in RMS. It also stops where the evidence stops rather than closing
over a side nobody looked at, so an open mesh there is a report about the capture. That is
synthetic evidence about the arithmetic, not a measurement of the scanner. The live preview above
is still the occupancy mesher and still stands outside its samples; the two are no longer the same
surface, and only the exported one is fitted. Coverage — the red “still needs scanning”
colouring — is measured on the points, so it is shown there.

**Each preview now says on screen what it is, without being hovered.** Under the depth image, a
fixed line states that the picture is raw sensor depth codes, that the grey checker is *no
return* rather than a near or far reading, and that no completion, filling, smoothing or
carry-over is applied — and it names no distance unit, because the codes establish none. Under the
model, one line describes whichever picture is actually mounted in the viewer: the dots are
**measured depth returns** whose placement depends on the calibration and on this scan's pose
estimates, and whose cloud is sampled and thinned for display — one sentence that is true of the
live preview and of a completed registered reconstruction alike. The panel has **two** surfaces
and it no longer describes them with one sentence. While a capture runs, the shell is a **derived
occupancy surface, not measurement**, with its triangle count and cell edge, marched from the grid
the poses accepted during capture were folded into. Once Reconstruct has run, the same checkbox
draws the surface it exported, and the panel says what that is: a **derived signed-distance (TSDF)
surface, not measurement**, with its own triangle count and voxel size, fused from the captured
full-resolution frames at the **corrected** poses. Which one it is comes from the producer that
built the mesh — never from the note beside it, the re-framing, the checkbox or a default — and
the same value drives the line, the coverage summary and the control's own title and tooltip, so
they cannot contradict each other. A published surface that does not say which it is is refused
rather than drawn under a guess. It follows the mounted
picture rather than the checkbox, so a preference for a surface that cannot be drawn is answered
with the points and says so; it moves with every toggle, arrival, withdrawal and restoration; a
new scan never opens on the last scan's triangle count; and a GPU-to-software fallback carries it
across with the model. The three modules that can invent or weigh invented samples —
`scanner3d.depth_completion`, `scanner3d.depth_temporal` and `scanner3d.tsdf_input` — remain
offline evidence and are reached by no GUI, capture, tracker, preview, fusion or export path. See
[docs/preview-provenance.md](docs/preview-provenance.md).

Every frame is also checked for **isolated points** before it registers. A stray sample — a
flying pixel at a depth edge, a multipath return — deprojects into a coordinate as confidently as
real surface does, pulls on the fitted pose, and draws as a dot floating around the object. A
point is dropped when fewer than three others of its own frame lie in the block of grid cells
around it, on a grid sized from that frame's own measured spacing. On synthetic frames that
removes essentially every stray further than two spacings from real surface and costs 1–3% of the
real points — rims, grazing faces, the last samples before an edge — and it cannot tell the two
apart. It rejects isolation, not error: a dense cluster of wrong points survives it intact.

**Pause** lets you pick the object up mid-scan and carry on into the same capture. A turntable
reaches about two thirds of an object; the rest is whatever it is standing on. While paused the
camera views keep running so you can frame the new position, and nothing at all is recorded — no
file, no count, no pose, no model update. The button says **Resume** while the capture is
actually paused, and goes back to saying Pause the moment it is not. Pressing Resume does *not*
simply carry on: the next few frames are held in memory, out of the scan entirely, while the
scanner works out where the object has moved to by matching them against surface it has already
scanned. That decision is a global search followed by named, conservative gates — the coarse
search separating its best orientation from the runner-up, overlap, residual against the measured
noise floor, matched and used fractions, a margin over any genuinely different placement, and
agreement from every held frame rather than a flattering union score — and until all of them
clear, the held frames stay held and the panel says which gate is failing. The first of those is
a refusal nothing measured afterwards can overturn: a shape the coarse search cannot tell the
orientation of is a shape that sits several ways on the model, and refining an ambiguous start
picks one of those placements rather than distinguishing them. Move back over a part you have
already scanned — and a part that does not look like every other part of the object — pause and
reposition again, or Stop; everything written before the pause is safe either way. When the gates do clear, the held frames and a `bridge.json` recording
the transform, its evidence and every threshold appear together in one `segment-0001/` directory
in a single atomic rename, and the model updates immediately. Reconstruction — window and CLI
alike — reads that record and uses the recorded transform at the seam instead of aligning a pair
that is not a small step apart, says so in the report, and names it in `reconstruction.json`.
Striding a repositioned capture strides within each segment, so a boundary can never be strided
away. Captures with no pause in them are read, registered and exported exactly as before.

**Tracking loss** is the other kind of break: the one nobody asked for. On 2026-08-28 a capture
aligned 51 frames, failed on the 53rd, and then failed on every frame after it for two minutes —
while the previews kept moving, the counter kept climbing and the model simply stopped changing.
It wrote 120-odd more frames that looked, on disk, exactly like frames that had aligned. Three
*consecutive* selected frames that cannot be placed — alignment failed, or the frame yielded no
usable cloud, or preparing that cloud raised — now end that: the capture says `TRACKING LOST`, names the
frame the model is frozen at, disables Pause (which places new frames against the accepted model,
and that is what has broken) and tells you Stop is the safe action. A single success resets the
streak, so an isolated dropout costs nothing. Nothing further is written, counted, posed or
drawn; the previews and Stop stay live. Every frame already written stays untouched, and one
`tracking-loss.json` — written into the run that actually lost tracking — records the last aligned
frame, the failing frames, the captured failure messages, and that nothing bridges the gap:
completely or not at all, never overwritten, and never claiming the projector, camera or object
supplied the verdict. If it cannot be written the capture stops just the same but never claims the
loss it could not record: it says there is no record in the folder and fails loudly, rather than
leave an unmarked broken folder.

**Recovery from a loss is automatic, bounded and evidence-gated.** Once the record is durable the
capture enters `recovering_tracking`: it holds a few frames back in memory — none written, none
counted, none drawn — and looks for the model it had already accepted, using exactly the pause
path's strict relocalization, ambiguity margin, overlap, residual and attempt/time/size ceilings.
Nothing is relaxed to make it succeed. If it clears every gate, one `segment-*` directory appears
atomically holding the held frames, the bridge and a `recovery` record binding it to that loss by
the record's own SHA-256 and by each excluded frame's — checked against the record as it stands on
disk at that moment, never against a stale copy of it; the window says how many frames were left
out, and scanning carries on. The frames written while tracking was lost stay exactly where they
are as raw evidence and never rejoin the scan, and the capture's ending counts them apart from the
frames that are reconstructable. If the bounded search cannot place them — including a search still
running when its time is up, and one that comes back accepted only after that deadline, both
abandoned rather than waited for or applied — nothing is written and the loss simply stands. Reconstruction — window and CLI alike, through the one reader —
hands over a recovered capture with the seam marked as a boundary and the exclusions stated, and
refuses one whose loss nothing closes: presence, order or a plausible-looking bridge never close a
loss, and a refusal is better than a guessed pose. See
[docs/tracking-loss.md](docs/tracking-loss.md).

**A reconstruction that cannot be evidence is refused rather than published.** A lost capture is
only one of the ways a scan can come out looking exactly like a scan. Forty frames taken from one
place register perfectly against each other; a million points from one angle know one silhouette;
a fused surface three truncations away from the points it was built from is topologically
flawless and opens in any viewer. So the window, `scanner3d reconstruct` and the export function
itself now share one schema-versioned readiness result that checks four things — the capture's own
durable tracking state, whether the accepted poses describe a scan, whether the occupied surface
was seen repeatedly and from a spread of directions, and whether the fused surface agrees, in both
directions, with a bounded sample of the full-resolution frames the volume was actually fused from
rather than with the decimated clouds the registration was fitted to — reports *every* failed gate
with the evidence and one
actionable sentence each, and refuses before a byte of field is allocated. Legitimate scans stay
possible: an orbit that returns to its starting view, a scan the operator paused, a validated
recovery and an open one-sided surface all pass, watertightness is never required, and none of it
is an accuracy claim. A refusal leaves no cloud, no mesh, no manifest and no staging directory. See
[docs/reconstruction-readiness.md](docs/reconstruction-readiness.md).

**Every frame is placed by agreement between independent references, not by one alignment
against its neighbour.** Chaining each frame to the one before it is right for a slowly turning
object under a fixed scanner and wrong for a scanner carried round a chassis: individual views go
weak — one flat panel determines no rotation about its own normal, and point-to-plane alignment
reports that as a small residual for a transform that slid — and nothing corrects the chain, so
every pairwise error is permanent and invisible from inside it. So each ordinary frame now gets up
to three independent candidate placements: the previous accepted frame, a keyframe chosen for its
viewpoint, and a bounded sample of the accepted model in model coordinates. Each is judged on
finiteness, convergence, residual, matched and used fractions — the same gates and the same
function the relocalization's own chain uses — plus overlap with the reference it was scored
against and the distance the placement moves the surface. A pose is taken only where the survivors
AGREE; where two of them place the frame materially differently the frame is refused rather than
resolved by picking the smaller residual, and where one reference is all there is, every gate holds
on its own. Keyframes are chosen from measured motion and novelty, never from frame number, so a
scanner left pointing at one spot cannot fill the set with twelve copies of one view. Every solve is a refinement from a
stated starting pose, and the tracker keeps one constant-velocity estimate — the step between the
last two accepted frames — so that starting pose is where the scanner was *plus how it was moving*
rather than the assertion that it did not move at all. That prediction seeds the keyframe and model
solves for free, and buys the previous frame one bounded second attempt where its first produced no
placement; it widens no gate, and the identity-seeded attempt stays first so the offline
reconstruction can still pre-fit every adjacent pair across worker processes and reach the identical
chain. Measured at 30 degrees a frame, well past the speed the capture guidance already calls too
fast: all 16 frames placed within 2.32 degrees, where the identity seed alone refuses at frame 7.
Everything is
bounded — three solves a frame, four where that retry runs, twelve keyframes, eight frames and 1800
points per model reference, thirty-two clouds retained — so frame 900 costs what frame 9 did. A refused frame appends no pose
and mutates nothing; live it enters the existing three-frame loss streak with the tracker's own
account of what each reference said, and `scanner3d reconstruct` refuses the whole sweep by frame
index rather than skipping the frame or manufacturing a pose for it. Measured on a synthetic
24-frame walk around an asymmetric object, blind chaining drifts to 1.60 degrees while this holds
0.58, and the gap grows with the sweep. See [docs/tracking.md](docs/tracking.md).

**Frame spacing can be decided while the capture runs, rather than typed in before it.**
"Use 1 frame in every N" is the one capture setting whose right value is a property of how fast the
scanner happens to be moving at that moment, and neither end of a fixed value is harmless: set for a
brisk sweep it wastes a slow one, where frames a fraction of the noise floor apart measure noise
rather than motion — the *slowest* row of the measured drift table is among its worst — and set for
a slow sweep it loses a brisk one outright. Tick "hold spacing" and the number becomes a starting
point: the capture moves it between 1 and 10 to hold the measured travel inside the same band the
travel readout already shows, going finer on one over-band frame and on any frame the tracker
refused, and coarser only after three slow ones in a row. It is deliberately not a fix for tracking
loss — its whole range is your own setting down to 1, which you can also just set yourself — and it
is off by default, because a capture spaced by a number you typed is one you can reason about
afterwards. The stream selects by a countdown of deliveries rather than a modulo of the frame index,
so a spacing that changes mid-run cannot make the phase jump, and the run's record carries the span
it actually used rather than one number that was never true throughout. See
[docs/frame-spacing.md](docs/frame-spacing.md).

**The turntable is removed from every frame before it is registered — and before it is fused.**
A stationary bed is a
large, strongly planar patch that pairwise registration is happy to explain the next frame by
leaving in place — which is why an object can freeze on screen while it is plainly turning, or
fuse into a bed-shaped blob. One deterministic plane is fitted per frame and only the narrow band
of points lying in it is removed, and only when every one of a set of independent gates agrees it is a
surrounding support surface rather than a face of the object. It is not a crop, a half-space, a background
subtraction or a hand remover, it assumes no direction is up, and a large flat chassis surface is
deliberately kept whenever the evidence is ambiguous — losing a bed costs one frame, removing a
panel costs the scan. The window and `scanner3d reconstruct` prepare frames through the same
function, in the same stated order, and both report how many frames the plane was removed from,
how many points were kept, and how many frames kept their whole cloud and why.

The exported mesh is fused from the **full-resolution** frames rather than from those clouds, so
for each frame the filter applied to, the same band is also masked out of an in-memory copy of
that frame before fusion — otherwise the bed would be gone from the poses and still in the PLY.
A frame the filter withdrew on is fused exactly as it was captured, and `frame_NNNN.raw` on disk
is never written to. Nothing is claimed about a physical scan until the owner runs one. See
[docs/support-plane.md](docs/support-plane.md).

**Reconstruct + export** turns a capture into files. You pick the folder of captured frames to
read — the input — and every frame in it is registered and then written into a *new* folder
inside it: `registered_cloud.ply`, `mesh.ply`, `mesh.obj` and a `reconstruction.json` naming all
three with their byte counts and SHA-256s, the registration result, the mesh's measured
watertightness, the voxel resolution and how it was derived, the axis convention, and everything
none of it establishes. The files are staged in a hidden sibling directory, read back and matched
against their own digests, and only then published under their final name in one atomic rename —
so a cancelled or failed run leaves no folder that looks like a result, and a second
reconstruction of the same capture makes a second folder rather than touching the first. The
model reaches the screen only once the export has succeeded.

Everything shown and everything written is in one right-handed **upright** frame: the depth
camera's optical frame counts +y down the image, so scanner3d rotates it a half turn about x —
a proper rotation that changes no length or scale — before showing or exporting it, and states
that mapping in the record so a consumer can undo it exactly. The model view orbits on an
accumulated rotation rather than a clamped yaw and pitch, so it turns all the way over and round
without a pole to stop at; left-drag turns, shift+left-drag rolls, middle-drag pans, the wheel
zooms.

The model is drawn by the **graphics card**. The points and their colours are uploaded once, when
the model changes, and turning it costs a 4x4 matrix and one draw call rather than a fresh
CPU projection, depth sort and full-window raster per mouse event — measured on this host at
0.114 ms to submit a camera-only frame of 60,000 resident points, against about 8.8 ms of
GUI-thread projection, sort and canvas for the same cloud before. Repaints are paced to the
display and rapid mouse events coalesce onto the newest camera state, so a fast pointer drops
duplicate frames rather than accumulating work behind itself; nothing is lost, because every event
is applied in full and only the drawing of positions nobody could have seen is skipped. Occlusion
comes from a depth buffer instead of the sort, and every other meaning — which way up, which face
is in front, the coverage colours, the newest-frame highlight, the framing rules, the point
ceiling — is shared with the software renderer rather than restated, and held to it pixel for
pixel by the tests.

If OpenGL is missing or the context, shader or buffers fail, the viewer **falls back to the
software renderer in place**, keeping the model it was showing and the pose you dragged to,
and says why in the log and under the picture. Which renderer is live is always named there —
with the driver's own vendor and version string, so "is this actually using my card" has an
answer that is not an assumption.

All three keep moving *during* a capture, not only before one. Colour is pumped alongside the
depth stream — the RGB camera returns nothing unless depth is already streaming, so it starts on
the first complete depth delivery and stops with that stream. The depth view shows **every**
complete delivery whatever `decimate` is set to; only the saving, deprojection and alignment are
thinned, because how often a picture reaches a screen and how often a frame is written are
different questions. The accumulated model is rebuilt every four successful alignments — about
2–3 seconds at the measured kept-frame rate — from a fixed number of frames sampled across the
whole sweep rather than from every frame so far, so a rebuild late in a scan costs what an early
one did, and the saved-frame counter beside it always describes the model actually on screen.
The view frames itself only at a boundary — a new capture, a registered reconstruction — so an
update that adds a frame reads as growth instead of being rescaled back to the same apparent size
with your own pan and zoom discarded. Every one of those views crosses to the window through a
one-slot channel that keeps the newest payload and discards superseded ones, so a busy window
skips rather than falling further behind. None of this touches what is recorded: every frame is
written to disk whole, and `reconstruct` reads all of them at full fidelity.

## The number this project is built around

Registration reports **held-out closure error**: revisited viewpoints are detected geometrically,
half are given to the optimiser, and the score comes from the half it never saw.

That is not decoration. Measured on a real two-lap capture of a model truck bed, a trajectory
that was **65 degrees wrong after one revolution** simultaneously reported per-pair alignment
rmse 0.65, 100% frame overlap, and healthy self-consistency. Every internal metric said the scan
was fine. Self-consistency cannot detect drift, because a drifted trajectory is perfectly
self-consistent — so this measures against evidence the optimiser was not allowed to fit.

Fixing what that exposed took drift on the same hardware from **65 degrees per revolution to
0.5**, via four independent changes: capture density, an ICP trim fraction that was suppressing
recovered rotation to 85% of truth, point-to-plane matching, and pose-graph optimisation against
detected loop closures.

If a sweep never revisits a viewpoint, reconstruction reports **no score at all** rather than a
number it cannot back.

## What is measured, and what is not

Everything quantitative here was measured on one unit. Findings that are not documented by the
vendor and were established here:

- The depth scale is millimetres — a 58.0 mm can measured 57.705 mm at ~122.7 mm standoff
  (−0.51%). Accuracy across the range is **not** established: −0.49% to −2.26%, no correction model.
- The device **re-enumerates on the USB bus every time a depth stream closes**, disappearing for
  about 0.39 s. Anything reopening immediately sees "no such device".
- **Coming back is not the same as being usable.** Observed on 2026-08-26: a wait that checked only
  for enumeration returned in 0.003 s, the open that followed failed `EACCES` on `/dev/video0`, that
  node had been created in the same second, and inspected afterwards it was `root:video 0660` with a
  valid per-user ACL — this account is not in `video`, so that ACL is what makes it openable. No
  access snapshot was taken at the failing open, so "the ACL had not been attached yet" is the
  inference those observations support, **not** a proven timing. What is fixed does not rest on it:
  readiness is now "the resolved index-0 depth node exists AND is readable AND is writable", never
  "a scanner is enumerable" and never a fixed `/dev/videoN`.
- The **RGB camera delivers nothing on its own** — a lone capture times out — but returns frames
  reliably while a depth stream runs. The vendor's depth-first ordering is a requirement.
- The full factory calibration set is readable over the vendor's own file-read protocol, including
  the depth→RGB extrinsics that make colour-to-geometry mapping possible.

## Honest gaps

- `keyframe.py` has no test suite. Everything else here does, mutation-tested. It is validated
  against real captures (2.4× less drift on an over-sampled sweep, no change on a well-sampled
  one) but that is not the same thing.
- Revisit detection can accept false positives on near-symmetric objects: two views half a
  revolution apart have similar depth histograms, and no local threshold separates them — the
  false ones score *better* on rmse than the true ones. The pipeline contains the damage by
  returning the unoptimised chain when optimisation makes held-out error worse.
- No mesh texturing yet, though the calibration needed for it is now in hand.
- The projector cannot be turned off. Several register writes and the device's own actuator ioctl
  are all *accepted* while the emitter stays lit; this project has never had optical readback.

## Configuration

`~/.config/scanner3d/config.toml` (or `$XDG_CONFIG_HOME/scanner3d/config.toml`) holds both
settings. `scanner3d config` prints what is set, where each value came from, and where each
capture command would write right now.

The scanner serial is read from `SCANNER3D_SERIAL`, then the config file (`serial` under
`[scanner]`), then the single attached device. It refuses to guess when several are attached,
because several operations here detach kernel drivers.

`SCANNER3D_VIEWER` chooses the 3D renderer: `software` forces the CPU one (the way to find out
whether a display problem is the driver's or this application's), `opengl` demands the graphics
card and reports it prominently if it still has to fall back, and anything else — including
leaving it unset — prefers the card and falls back automatically. It never stops the application
from opening.

The captures directory is read from `SCANNER3D_CAPTURES_DIR`, then the config file (`directory`
under `[captures]`):

```bash
scanner3d config --set-captures-dir ~/scans   # recorded; every capture goes under it afterwards
scanner3d config                              # show it, and what each command would write to
scanner3d config --unset-captures-dir         # back to each command's own default
```

It is a *root*, not a capture directory. Each command claims a subdirectory named after itself,
and every execution gets a freshly created directory inside that one — dated, or named by you
when the desktop application asks:

```
~/scans/sweep/20260825-174501        # one sweep
~/scans/sweep/20260825-174501-2      # a second sweep started in the same second
~/scans/depth-frame/20260825-174630  # one depth-frame capture
~/scans/scan/20260825-175012         # a desktop-application scan you did not name
~/scans/scan/chassis left side       # one you did
~/scans/scan/chassis left side-2     # the same name, used again a week later
```

No *automatically allocated* directory is ever one that already exists, because these commands
write fixed filenames — one sweep's `frame_0000.raw` would otherwise overwrite the last one's,
and `depth-frame` would spend a capture only to fail preserving it. Each one is created with a
plain `mkdir`, which the kernel makes atomic, so two processes starting in the same second, a
restart, a clock dragged backwards and the repeated hour of a DST change all get different
directories — or a refusal, issued before the scanner is opened. This is a guarantee about the
directories the tool picks for itself; a directory you name yourself is yours (see below).

An explicit `--output-dir`, or `sweep`'s output argument, still wins and keeps its old meaning:
that directory is used exactly as given, existing or not, alongside whatever is already in it.
Nothing above applies to it — a path you typed is a path you own, and quietly capturing into a
dated subdirectory of it, or refusing because it exists, would both be the tool overruling you.
With nothing recorded, every command keeps the default it always had. The root itself is
neither created nor checked when it is recorded — the command that writes there reports on it
in its own terms. A relative path hand-written into the config file is resolved against the
directory holding `config.toml`, while `SCANNER3D_CAPTURES_DIR` is resolved against the working
directory, because a one-shell override can mean nothing else.

In the desktop application, **Captures folder…** records the same setting, and once it is set,
Start capture stops asking where to put the frames. It always asks one thing, though: a name for
the scan. Leave the box blank and the folder is dated for you, exactly like the commands above;
type something and that is the folder name, spaces and all. Either way the frames go into a
folder created there and then under `scan/`, never into the captures folder itself, and never
into a folder that already exists — a name you have used before gets `-2`, `-3` and so on.

A scan name is one folder name, not a path: `.`, `..`, and anything containing `/` or `\` are
refused with a message, and nothing is created. Both separators are refused on every system,
because a captures tree gets copied to one where the other one splits paths. So is any Unicode
control character — a tab, a line break, a NUL, or one of the invisible C1 controls a paste can
carry — and that check runs *before* surrounding spaces are trimmed, so a name made only of
control characters is a refusal rather than a silently dated folder. Ordinary surrounding
spaces are still just trimmed off, and internal spaces are kept. Cancelling the
name prompt cancels the scan before the preview stops or the scanner is touched. With no
captures folder recorded, Start capture asks for one — and what you pick is the *root*, with the
per-scan folder allocated under `scan/` inside it just as if it had been recorded.

## Requirements

- CPython **3.14** or newer
- Linux with `sysfs` mounted at `/sys` (only `uvcvideo`-exported metadata is read)
- No root, no udev rule, and no system packages are required for `doctor`

## Development setup

```bash
uv venv --python 3.14 .venv
uv pip install -e '.[dev]'
```

Or without `uv`:

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

This is the one verified path. It was checked by building a throwaway
environment from scratch and running every quality gate below in it.

**Do not use `uv sync` here.** The development tooling is declared as a
`[project.optional-dependencies]` *extra*, not a PEP 735
`[dependency-groups]` group, and this project has no dependency groups at all.
`uv sync --all-groups` therefore treats the dev tooling as unrequested and
**uninstalls** `pytest`, `ruff`, `mypy` and `coverage` from the environment
rather than installing them. If you prefer the lockfile-driven workflow, the
extra has to be named explicitly: `uv sync --extra dev`.

## Quality gates

Every gate runs offline and needs no hardware.

| Purpose | Command |
| --- | --- |
| Format | `.venv/bin/ruff format --check .` |
| Lint | `.venv/bin/ruff check .` |
| Types | `.venv/bin/mypy` |
| Tests | `.venv/bin/pytest` |
| Reconstruction quality harness | `.venv/bin/python -m tests.quality_harness --out quality-harness-report.json` |
| Depth-completeness benchmark | `.venv/bin/python -m tests.depth_completeness_harness --out depth-completeness-report.json` |
| Surface regression oracle | `.venv/bin/python -m tests.surface_oracle_harness --out surface-oracle-report.json` |

`ruff format .` rewrites files in place. The test suite is fixture-driven: it synthesizes fake
`sysfs`/`dev` trees and replays a recorded byte-for-byte configuration descriptor from the real
scanner, so it passes with no device attached and never touches hardware.

Every push and pull request runs all of these gates on GitHub-hosted Linux runners, from
`.github/workflows/ci.yml`, as five parallel jobs: **Static quality** (format, lint, types),
**Offline test suite** (pytest), and the three named CPU correctness oracles — **Reconstruction
quality harness**, **Depth-completeness benchmark** and **Surface regression oracle**. The three
oracles are separate jobs because they measure different evidence and each keeps its own report: a
geometry regression, a depth-completion regression and a fused-surface regression are different
facts, and none of them is an ordinary test failure. Ordinary pytest already runs all three
corpora for real — the surface oracle's twice, once through its harness tests' module-scoped
report fixture and again through the determinism test that reruns and compares — so the named jobs
are there for a per-oracle status and a retained report rather than to cover a gap in the suite.
All three upload their JSON **even when they fail**, under distinct artifact names
(`quality-harness-report`, `depth-completeness-report` and `surface-oracle-report`). CI reads the
repository and runs the commands above on the same CPython 3.14 and the same locked environment
(`uv sync --frozen --extra dev`); it has no scanner attached, publishes nothing outside the run,
and is CPU-only. Every backend the oracles select is a CPU implementation over committed synthetic
truth — the pure-Python references throughout, except the surface oracle's one shipped-policy arm,
whose roughly two-million-voxel field is past the pure-Python ceiling and runs on the numpy CPU
backend — so no GPU is needed and an accelerator would add a different execution path without
improving the correctness evidence. Every job is bounded: 60 minutes for the offline suite, which
measures about 29.5 minutes locally and may be slower on a hosted runner, and 20 minutes for the
surface oracle, about six times its measured 3:11.

**What pytest already runs** is `tests/`, exactly as configured in `pyproject.toml` — including
`tests/test_quality.py`, which proves the measurement primitives in Python, and
`tests/test_quality_harness.py`, which proves the harness below fails when it should, and
`tests/test_depth_completeness.py` and `tests/test_depth_completeness_harness.py`, which do the
same for the depth-completeness benchmark, and `tests/test_surface_oracle.py` and
`tests/test_surface_oracle_harness.py`, which do the same for the surface oracle. Each of those
`*_harness` modules runs its committed corpus for real, so the ordinary suite exercises all three
corpora — and the surface one twice, since its determinism test reruns it. pytest does **not**
collect
`tests/quality_harness.py`, `tests/quality_replay.py`, `tests/readiness_replay.py`,
`tests/preview_health_replay.py`, `tests/build_quality_fixtures.py`,
`tests/depth_completeness.py`, `tests/depth_completeness_harness.py`,
`tests/build_depth_completeness_fixtures.py`, `tests/surface_oracle.py`,
`tests/surface_oracle_harness.py`, `tests/build_surface_oracle_fixtures.py`,
`tests/surface_oracle_replay.py`, `tests/shell_discriminant.py`,
`tests/shell_discriminant_harness.py`, `tests/build_shell_discriminant_fixtures.py`,
`tests/shell_discriminant_replay.py` or `tests/tsdf_benchmark.py`: those are commands and
libraries, not tests.

`tests/test_shell_discriminant.py`, `tests/test_shell_discriminant_harness.py` and
`tests/test_shell_discriminant_replay.py` do for the observable-shell discriminator what those
pairs do for the other three corpora: the first holds the fast contract tests and the truth-leakage
controls and needs no corpus, the second holds sixteen corpus controls, one control over the
trusted rebuild's cache key, sixty-seven adverse derived-logic controls over a small built report,
and — after running the committed population once through a module-scoped fixture —
twenty-three adverse controls that mutate the real published artifact and hand it to the public
strict reader, and the third holds the read-only contract of the physical replay. That
fourth corpus is **not** a CI job and none was added — the workflow's five jobs are unchanged — so
it is exercised only by ordinary pytest, which it makes measurably longer; the handoff report
records the measured cost.

**What the harness adds** is the fixture corpus itself, run as a command and leaving one artifact.
It loads `tests/fixtures/quality/cases/` through strict validation, runs the real
`scanner3d.quality` measurements and the real `scanner3d.reconstruction_readiness` reader over it,
and writes one strict-JSON report naming every case, every gate and every expected-versus-observed
outcome. Exit `0` means every case produced exactly the outcome its fixture declares; `1` means at
least one did not; `2` means the corpus could not be run at all. A case whose measurement raises is
recorded as malformed and the later cases still run, so the report preserves all of them. The CI
job prints the whole report to its log and uploads the JSON **even when the harness fails**, which
is the run whose report matters.

Read a published report back with `tests.quality_harness.read_report`: it parses the text strictly
— duplicate JSON keys and `NaN`/`Infinity` included — and recomputes every verdict, count and
mismatch from the evidence beside it rather than trusting what the report says about itself. The
requested `--out` path is cleared before the run, so a failed invocation leaves no file there
rather than the previous run's report.

The artifact is at **report schema version 2**. Every embedded measurement is held to the exact
schema of its own kind, and every gate carries a `gate_bindings` entry naming the measurement and
path its value came from, which the reader resolves and compares — so a measurement changed while
its gate was left alone is refused rather than believed.

Deliberately bad fixtures are the load-bearing part: a case declared to fail names *which* gates
must fail, so an unexpected pass, a missing expected failure and an extra failure are all harness
failures. "Something failed" is not the expected failure. Two runs over an unchanged corpus produce
byte-identical JSON, so no duration and no machine identity appears in it.

This is a CPU correctness oracle. The default compute backend is the pure-Python reference and the
whole corpus takes well under a second, so a GPU runner would change how fast the answer arrives,
not what the answer is — see [docs/quality-harness.md](docs/quality-harness.md).

### Reconstruction quality

`scanner3d.quality` judges a reconstruction against geometry that is known, rather than against
whether the dots look right. It measures pose error against exact truth, symmetric
observed-to-reference *and* reference-to-observed surface agreement, flatness of a region the
caller names, named dimensions and landmarks against truth the caller supplies, and re-judges a
recorded relocalization bridge under today's policy. It owns no notion of "good": every threshold
is passed in at the call site, in the same unit as the measurement, with the action a failure
implies.

The fast tier is `tests/test_quality.py` — pure CPU, no GPU, no display server, 0.84 s measured —
together with the shipped fixture corpus the harness above runs. The physical replay tier is opt-in
and read-only, and stays non-gating until independent truth has been measured:

```bash
.venv/bin/python -m tests.quality_harness         # the shipped fixture corpus, gating in CI
.venv/bin/python -m tests.quality_replay          # the shipped turntable fixture, reported not gated
.venv/bin/python -m tests.build_quality_fixtures  # regenerate the corpus, then read the diff
```

See [docs/quality-harness.md](docs/quality-harness.md) for what each tier proves, why held-out
closure and surface agreement answer different questions, and what has to be measured physically
before the physical tier can gate anything.

### Depth completeness

`tests/depth_completeness.py` and its harness are the oracle for the depth-completion work that
follows: they implement no completion and exist so that a denser depth result cannot be called an
improvement if it bridges a depth edge, closes a real gap, moves a dimension, roughens a plane,
invents geometry in empty space, degrades tracking, or reports a reconstructed sample as a camera
measurement.

Every sample carries one of exactly four provenance states — `measured`, `spatially_reconstructed`,
`temporally_reconstructed`, `missing` — with confidence carried separately and named support
(`frame:<i>:pixel:<u>,<v>`) required for every reconstructed sample and resolved against the input
frames. Fifteen refusals reject a result whose own account of itself does not hold, and a refused
result is not scored at all. The strict reader derives which optional measurements a report must
carry rather than believing it — including whether a completion reports what its run cost, which
follows from the candidate's own closed identity, so the work block cannot be deleted along with
its gate to dodge the bar.

```bash
.venv/bin/python -m tests.depth_completeness_harness          # eight scenes, twenty-one candidates
.venv/bin/python -m tests.build_depth_completeness_fixtures   # regenerate the corpus, then read the diff
```

Eight committed scenes — a fronto-parallel plane with and without dropout, an oblique plane, a
curved surface against empty space, a foreground/background depth step, a narrow slot that must
stay open, a paired small dropout and true aperture whose local input bytes are identical, and an
asymmetric multi-view object with exact poses — are run against one set of gates
by many deliberately bad candidates, including nine corruption controls that must each earn one
exact refusal. A dense nearest-neighbour fill **passes** on the constant-depth plane, where it is
right, and fails on every other geometry: the oracle refuses invention, not density. Measurements
come from the real `scanner3d.quality`, `scanner3d.tsdf`, `scanner3d.tsdf_surface` and
`scanner3d.tracking`; the fixtures commit the scene and the SHA-256 of every frame it renders
rather than the frames themselves, and the harness refuses a case whose frames do not hash to what
its fixture says.

Everything it measures is synthetic truth. See
[docs/depth-completeness-benchmark.md](docs/depth-completeness-benchmark.md) for what the benchmark
proves, what the one recorded capture only suggests, and the acceptance gates a completion has to
meet.

### The raw-points versus fused-surface oracle

A third deterministic corpus — run by ordinary pytest, by the command below, and in CI as its own
named job — answering why the fused surface looks lumpier and less complete than the measured
dots:

```bash
.venv/bin/python -m tests.surface_oracle_harness --out surface-oracle-report.json
.venv/bin/python -m tests.build_surface_oracle_fixtures   # regenerate the corpus, then read the diff
.venv/bin/python -m tests.surface_oracle_replay           # the physical companion, strictly read-only
```

Seven exact-truth multi-view scenes — an exact control plus bounded zero-mean depth noise,
localized outliers, structured no-return bands, translational and rotational pose error, and all of
them at once — hand the **same raw frames and the same poses** to the unchanged measured point
cloud and to the unchanged `scanner3d.tsdf` field meshed by the unchanged
`scanner3d.tsdf_surface`, at each of six predeclared parameter cells **and at the resolution the
shipped Reconstruct policy actually chooses** (about 0.035 here, against a finest declared cell of
0.10). Eight corruption controls run beside them, each earning one named refusal.

Accuracy is measured against the **object**, in closed form, and never against the sensor's own
returns; proximity to those returns is published separately as measurement support and is never
called accuracy. The mesh is measured **as a surface** rather than as a point list: a fixed
equal-area barycentric lattice inside every emitted triangle, plus an edge census, shell
decomposition and
Euler characteristic — because vertices and field zero crossings sit on the same grid edges and
cannot reveal a triangle spanning a gap or a shell sealing an unmeasured region.

**The extractor is not the cause.** The fusion-versus-extraction ablation puts
`extraction_p95_delta` between −0.027 and +0.009 on every fused arm with no vertex unsupported by
its own field, and at the shipped resolution the exact control's mesh is inside tolerance of the
object at every vertex and every triangle interior. Fusion beats the dots on analytic accuracy on
both cases carrying zero-mean noise. Edge bridging is uniquely a pose failure and vanishes at the
shipped resolution. What survives there is watertight shells nothing measured — six on the
combined case, one under outliers — invisible to every distance summary and every point-linkage
lump count. Scored against the object as well, they do **not** mean the same thing: the six on the
combined case are analytically correct surface nothing happened to measure (worst 0.125 against a
0.15 tolerance), and only the outlier case's shell is false (0.303). Support and truth are
published as separate, joint evidence for exactly this reason.

The recommendation is therefore to **keep delivering the dots** and to make **no product change**.
An earlier revision's "prune connected shells no view supports" is withdrawn: on the case most like
a real capture it would delete a hundred faces of correct geometry and remove no false surface at
all. The offline experiment into whether any *observable* shell property separates the two groups
has since been run, and it is a negative result — see the next section. Nothing in this corpus is
routed into the product, and no
export, preview, GUI, capture, tracker or readiness behaviour was changed to produce it. See
[docs/surface-oracle.md](docs/surface-oracle.md) for the full tables, the withdrawn earlier
conclusion and why it was wrong, the tested parameter envelope, the physical replay's registration
refusals, and the limits of synthetic truth.

### Can an appliance tell a false sealed shell from a correct one?

No. Not on a holdout sealed before it ran, and not well enough to prune anything. A fourth
deterministic corpus, run by ordinary pytest and by the command below, asks whether a bounded rule
over evidence the appliance actually has can separate the fused surface's *false* sealed shells
from the *correct* ones the oracle above found beside them:

```bash
.venv/bin/python -m tests.shell_discriminant_harness --out shell-discriminant-report.json
.venv/bin/python -m tests.build_shell_discriminant_fixtures   # regenerate the population, then read the diff
.venv/bin/python -m tests.shell_discriminant_replay           # the physical companion, strictly read-only
```

An earlier pass of this experiment inspected its own holdout during development and then widened
the threshold grid, so its generalization claim was not one. **All nineteen of those members are
training data now**, and this section publishes only the sealed run. Eight new members — id,
family, every magnitude and the seed — were declared by the planner and written into the fixture
recipe *before any builder, fusion or harness run touched a sealed id*, together with the frozen
candidate-only digest `4fe6627e…`, the appliance selection resolution, the 0.5 material fraction
and the unchanged 40,000-decision ceiling. None of them was tuned afterwards.

The truth wall is now the derivation's own input type. `ObservableInputs` carries the ordered raw
frames, the fused poses actually used, the intrinsics, window, depth scale and bounds, the
extracted mesh, the cell parameters, the backend the arm actually fused on and the field the
surface came out of — and has nowhere to put a scene, the shapes, the ideal support, a truth frame,
a true pose, the analytic scoring, the family or the split. A pure function of that value derives
**every closed shell**; the evaluator attaches labels separately. A test makes every truth and
true-pose accessor raise and shows the derivation still produces byte-identical output. A
realization's identity is a digest over all of those inputs, so raw-only, pose-only and
backend-only changes each give a different identity while an exact clone keeps one.

27 members at 2 resolutions produced **776 closed shells**, of which **140 are unsupported** — 117
false (2,224 faces) and 23 correct (476 faces). At the shipped **appliance** resolution, which is
the only one selection and the gates may read, that is 32 training records (712 false / 316 correct
faces) and 41 sealed holdout records (624 false / 152 correct faces): both classes present, so the
holdout can measure the damage as well as the gain.

195 predeclared candidates against 140 records is **27,300 rule decisions — one bounded pass, one
per cell**, against the 40,000 ceiling. Every confusion, all 28 split × family × resolution
cross-tab cells per candidate — 8 that hold a shell and 20 published as zero cells rather than
omitted — every aggregate, the ordering, the gates and the recommendation are lookups into that one
cached matrix, and a test proves the overlapping summaries add zero further invocations. (The
earlier pass made 89,505 invocations and published 16,380; that claim is withdrawn.)

**None of the 195 candidates was eligible**: none removed at least half the training split's false
shell faces at appliance resolution while removing none of its correct ones. So no rule is
selected, no holdout gate was evaluated, no counterfactual ran, and the recommendation is
`no-pruning-keep-dots-primary`. The truth-fed upper bound the planner predeclared —
`sealed-outlier-185-p43` at appliance resolution — turned out to produce no closed shell at all
there, so it is published as **unmeasured** rather than swapped for a member that would have given
a number. As *development* evidence, a perfect truth-fed prune on the training member
`outliers-140-p17` removes 3 false shells / 100 faces, leaves both correct unsupported shells and
their **92 faces before and 92 after**, and buys **+0.2207 percentage points** of vertex accuracy
across 0.2317 % of the triangles with coverage, protected volume, bridging, edges, dimensions and
topology unchanged.

So pruning stays rejected and dots stay primary. No physical shell population exists: both named
read-only replays stop at registration before fusion with their capture fingerprints unchanged, and
no physical discrimination claim is made. These tests run in the **ordinary pytest job on the
existing public GitHub-hosted workflow**; `.github/workflows/ci.yml` is unchanged and no
self-hosted, LAN, heavy-machine or GPU routing was added. See
[docs/shell-discriminant.md](docs/shell-discriminant.md) for the sealed member table, the
observable definitions, the full candidate matrix, the strict reader, the adverse controls, the
computational cost and the one next action.

The standalone strict reader **rebuilds the committed corpus** rather than checking the report
against its own claims. An earlier pass did the latter, and a review mutated the published artifact
fifteen ways — coherently, so every derived fraction and total still agreed — and all fifteen were
accepted. `validate_report`, `loads_report` and `read_report` now take no evidence root at all:
they load the committed corpus, rebuild every member at every resolution, bind the report's corpus
block, realizations, populations, closed shells, records, work account and predeclared upper bound
to that rebuild, and *then* rebuild the typed values and one independent audit decision matrix and
recompute every derived claim. Each of those fifteen forgeries is now a test over the real
artifact, and each is refused for its own reason. Binding does not by itself close a type
confusion — `False == 0` and `True == 1.0` in Python — so a member's whole perturbation and both
surface accounts of every counterfactual and measured ceiling are typed field by field before they
are compared, with four more real-artifact controls for that attack.

### Spatial depth completion

`scanner3d.depth_completion` is the first real completion stage: it recovers no-return samples that
one raw frame supports strongly enough, and leaves the rest missing. Every returned sample comes
back byte-for-byte as `measured`; every recovered one carries a finite confidence below 1.0 and the
indices of the **original measurements** it rests on, so no reconstruction is ever evidence for
another. Holes are grouped into four-connected components first, and one larger or longer than the
policy allows is refused whole — which is what keeps a silhouette's background, a band of dropout
along a depth edge, a collapsed column and the gap between two objects out.

Every public type is bound to the raw frame it claims to preserve: `CompletedPatch` carries the
immutable `DepthPatch` it came from, and each provenance claim is validated against **that** rather
than against the result's own labels, so a forged label cannot vouch for its own evidence. Every
bound also has an absolute product ceiling — a finite integer is not a work cap — and a call whose
worst case exceeds `MAX_WORK_UNITS` is refused by arithmetic on its input before the first
neighbourhood is scanned.

Against the accepted benchmark it recovers 107 of 230 dropped samples on the fronto-parallel plane
and 105 of 225 on the oblique one, to within the truth's own one-code quantisation, with zero false
fill on every geometry the frame can settle, nothing bridged or misplaced at the depth step, the
protected slot untouched, and the multi-view case returned unchanged so tracking, fusion, poses and
dimensions are the baseline's exactly. It does **not** yet reach the benchmark's coverage target,
and each case names the exact holes that remain rather than calling conservative refusal a success.

The `aperture-ambiguity` case is where it loses, on purpose. A small dropout on a plane and a small
true hole in that plane present **byte-identical** local evidence, and this method fills both: it
recovers the dropout correctly and invents a surface across the aperture, at high confidence. That
failure is declared in the fixture rather than tuned away, because it is not a tuning limit — one
frame does not contain the evidence. Spatial-only estimates of that class must not become tracking
or fused geometry until an independent view confirms them.

Nothing in the capture, preview, tracking, fusion, mesh or export paths calls it yet, and no
physical claim is made. See [docs/depth-completion.md](docs/depth-completion.md).

### Pose-compensated temporal completion

`scanner3d.depth_temporal` is the answer to that aperture: a second measurement, taken from
somewhere else. It carries the **original raw samples of other frames** through the exact poses
tracking already accepted, projects them into the target camera, and reconstructs a sample the
target lost only when independent posed measurements from at least two genuinely different
viewpoints agree about it. Occluded, behind-camera, out-of-frame, disagreeing and
discontinuity evidence is refused with a named reason, and a same-pixel carry — "use the other
frame" without the pose between them — is proved wrong on a scene where the camera really moved:
it is the densest result in that case and it puts backing-plane depth on the box's front face.

Where the single-frame stage proposed a value, the temporal stage publishes whether independent
evidence confirmed it, contradicted it and published its own measurement instead, or never
reached it — and an estimate nothing reached stays explicitly spatial, so the aperture ambiguity
is disclosed rather than laundered.

It is offline and post-tracking by construction, because every transform it uses is built from a
pose tracking has already accepted. Nothing calls it: no capture, setting, GUI, tracker input,
TSDF, mesh, export or live preview reads a temporal reconstruction. The benchmark's gated tracking
evidence is the measured-only route the product actually takes, and the temporal stage leaves it
exactly the baseline's; beside it an ungated counterfactual replay shows that its reconstructions —
each within a single raw code of the truth — would still move where the real tracker places a
frame, which is the measured reason that boundary is there.
See [docs/depth-temporal.md](docs/depth-temporal.md).

### Provenance-weighted TSDF input

`scanner3d.tsdf_input` is what a completed frame is allowed to be *worth* to fusion.
`scanner3d.tsdf.TsdfVolume.integrate_weighted` spends it: the same projective arithmetic, with
the geometric weight multiplied by the sample's own fusion weight. A measured sample weighs
`1.0`, so a raw frame through the weighted path is byte-for-byte the raw frame integrated. An
accepted temporal reconstruction weighs `0.5 x` its own accepted confidence — strictly above
zero, strictly below a measurement, and monotone in the confidence — so two agreeing
reconstructions still cannot outvote the one measurement they disagree with. A single-frame
spatial estimate no independent view confirmed weighs exactly **zero** and offers a no-return
code, which fusion treats as absence: it cannot update a voxel and cannot turn unobserved space
into observed space. Every provenance claim is derived from an already accepted `CompletedPatch`
or `TemporalPatch`, never from a caller's labels.

On the benchmark's complementary multi-view case the measured answer is **negative, and it is
the finding**: 16 independently supported samples make 5953 weighted voxel updates against the
raw baseline's 5728, and produce the same 3099 observed voxels, the same 333 surface vertices
and the same truth-referenced surface coverage as fusing those same three raw frames. Fuse the
target frame alone and the same reconstructions add 225 observed voxels (1949 to 2174) — so the
geometry is real and is already in the multi-view baseline, because it came out of the frames
that baseline fuses. No coverage or surface gain is claimed or gated, and the recommendation is
that the export keeps fusing raw multi-view frames. On the aperture-ambiguity case
the excluded estimates contribute exactly 0.0 and the weighted field is byte-identical to the
measurements' own, while fusing the same codes at uniform weight would have observed 12 more
voxels. That is offline synthetic evidence: no physical improvement is claimed, nothing in
`scanner3d` calls it, and the reconstruction export still fuses its original raw frames.
See [docs/weighted-tsdf-input.md](docs/weighted-tsdf-input.md).

## CLI

```bash
scanner3d list                 # attached matching devices, one line each
scanner3d config               # settings, their sources, and where captures land
scanner3d config --set-captures-dir ~/scans      # record the default captures directory
scanner3d config --unset-captures-dir            # remove it again
scanner3d doctor               # human-readable diagnostic
scanner3d doctor --json        # machine-readable report on stdout
scanner3d doctor --json-out report.json
scanner3d trace capture.jsonl   # decode a recorded usbfs trace (offline)
scanner3d trace capture.jsonl --json
scanner3d firmware-version      # describe the one owned control read (opens nothing)
scanner3d firmware-version --execute --serial <serial>   # actually perform it
scanner3d firmware-version --detach-uvcvideo-from-interface-0 --execute --serial <serial>
                                # same read, but temporarily take interface 0 from uvcvideo
scanner3d inited-sequence       # describe the four-transfer /tmp/inited sequence (opens nothing)
scanner3d inited-sequence --execute-write-sequence --serial <serial>
                                # actually perform it — this WRITES to the scanner
scanner3d depth-intrinsics      # describe the bounded /data/camparam/Pl.bin read (opens nothing)
scanner3d depth-intrinsics --execute-calibration-read --serial <serial>
                                # actually perform it — this WRITES the key to the scanner
scanner3d depth-scale           # describe the bounded 0x918 depth-scale property read (opens nothing)
scanner3d depth-scale --execute-depth-scale-read --serial <serial>
                                # actually perform it — this WRITES the property id to the scanner
                                # (allow_abbrev=False: this flag must be typed in full)
scanner3d depth-frame           # describe the one bounded Y16 640x400 depth capture (opens nothing)
scanner3d depth-frame --execute-depth-frame-capture --serial <serial> --output-dir <dir>
                                # actually perform it — reads one frame over plain V4L2
scanner3d rgb-frame             # describe the one bounded MJPG 1280x800 RGB capture (opens nothing)
scanner3d rgb-frame --execute-rgb-frame-capture --serial <serial> --output-dir <dir>
                                # actually perform it — reads one frame over plain V4L2
scanner3d rgb-controls          # describe the fixed read-only RGB control inventory (opens nothing)
scanner3d rgb-controls --read-rgb-controls --serial <serial>
                                # query standard controls once; never changes a control or starts a stream
scanner3d rgb-exposure-extension # describe the fixed SDK RGB extension read (opens nothing)
scanner3d rgb-exposure-extension --execute-rgb-exposure-extension-read --serial <serial>
                                # opt in to exactly one fixed GET_CUR (scanner-027 ran it once)
scanner3d rgb-exposure-length # describe the fixed RGB extension GET_LEN (opens nothing)
scanner3d rgb-exposure-length --execute-rgb-exposure-length-read --serial <serial>
                                # scanner-030 ran one; every new physical run owes its own off finalizer
scanner3d rgb-exposure-info # describe the fixed RGB extension GET_INFO (opens nothing)
scanner3d rgb-exposure-info --execute-rgb-exposure-info-read --serial <serial>
                                # later physical task: one GET_INFO, then mandatory off finalizer
scanner3d paired-capture      # depth-first/RGB-second plan (opens nothing)
scanner3d paired-capture --json
scanner3d paired-capture --start-depth-first-rgb-second --serial <serial> --output-dir <dir>
                                # actually perform one fixed paired V4L2 attempt
scanner3d paired-rgb-series    # depth-first/RGB-series plan (opens nothing)
scanner3d paired-rgb-series --json
scanner3d paired-rgb-series --start-depth-first-rgb-series --serial <serial> --output-dir <dir>
                                # actually perform the fixed three-observation RGB diagnostic
scanner3d paired-rgb-series --read-rgb-controls-during-series
                                # request bounded read-only control telemetry (still dry by default)
```

The paired plan preserves the pinned 3DViewer ordering: resolve distinct depth
and RGB nodes by the exact serial, start depth, wait one second, start RGB, and
wait once for one depth and at most one RGB frame. Execution requires the exact
serial plus `--start-depth-first-rgb-second`, and a local `--output-dir`; it has
fixed 8-second deadlines, the existing depth/RGB memory ceilings, no retry, and
reverse cleanup. Verification is fully covered with injected/fake V4L2 syscalls;
accepted physical evidence is documented below separately from the synthetic tests. Scanner-014's raw
usbmon target records and their bounded decoder are preserved under
`tests/fixtures/` and do not claim that RGB hardware delivery works.

`paired-rgb-series` is an offline-verified diagnostic boundary. It preserves the accepted
depth-first start and one-second interval, then uses one absolute 8.0 s RGB deadline with exactly
three bounded dequeue opportunities. Every payload is written unchanged as indexed
`rgb-series/rgb-series-000.jpg` through `rgb-series-002.jpg` before any allowed requeue; the JSON
records V4L2 sequence/timestamp/flags/bytesused, hashes, strict JPEG inspection, EOI-prefix facts
and opaque trailer bytes. A complete JPEG prefix does not make a payload valid when bytes follow
EOI, and no trimmed derivative is written. The command is dry by default and requires the exact
`--start-depth-first-rgb-series` flag, the authorized serial and an explicit local output
directory. The live-capable implementation is verified only with injected/fake V4L2 boundaries.
Scanner-019 is the accepted physical three-observation run: all three payloads had four opaque
trailing bytes after their JPEG EOI, decoded nearly black, and showed no meaningful warm-up
improvement. Scanner-021 subsequently ran the exact read-only standard-control inventory once;
its measured result is recorded below.

The separately gated `--read-rgb-controls-during-series` mode is implemented and verified
offline with fake V4L2 calls. With the exact series opt-in it uses the already-open RGB
descriptor and reads gain, exposure-auto, and exposure-absolute in fixed order before the
first RGB poll and after each successfully dequeued observation. There are at most four points
and twelve attempts; ordinary errors and returned-ID mismatches remain typed records without
retries, and capture and telemetry statuses remain separate. The flag alone, a dry run, wrong
serial, and non-series capture never open anything or issue `G_CTRL`. Scanner-024 then performed
exactly one accepted physical telemetry invocation: all 4/4 points and 12/12 fixed reads
succeeded, with gain 16, exposure-auto index 3 and raw absolute exposure 131072 invariant.
No setting was changed; these raw values do not establish exposure units or the separate
selector-0x12 extension state.

`rgb-exposure-extension` is a separate, dry-by-default boundary for the MINI
SDK's selector-`0x12` wire request. Scanner-027 ran its immutable opt-in exactly
once: `a1 81`, `wValue=0x1200`, `wIndex=0x0400`, requested 60 bytes, and the
device returned 2/60 bytes (`0000`, SHA-256
`96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7`). The
typed result was `short_response` (exit 7), with no four-word decode. The same
run performed its mandatory immutable blue-light off write exactly once,
accepted 60/60, and passed release/reconnect plus stable postflight; that was
the last physical action. “Accepted” describes transfer length; it does not
claim optical darkness.

Scanner-028 reconciled this wire result offline with the hash-pinned SDK
(`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`) and the
descriptor fixture. The SDK's `PROPERTY_EXT_EXPOSURE_TIME_RGB=0x15` and
`...RANGE_RGB=0x16` are device-property IDs, not selector 0x12. Its CameraType
12 GET paths each issue one `a1 81`/`0x1200`/`0x0400`/60-byte/1000-ms transfer;
the wrapper zeroes the 60-byte buffer and treats any nonnegative libusb byte
count as wrapper success, discarding that count. The 0x15 path reads word 0;
the 0x16 path reads words 1–3 (offsets 4, 8, 12), with SDK scaling described
in `docs/architecture.md`. Entity 4's recorded `bmControls=ffa201` does not
advertise selector 0x12 (bit 17 is clear); this does not prove the device must
reject an unadvertised selector. Therefore `0000` is only the measured two-byte
reply: it is not established exposure, microseconds, enable state, success,
failure, or a valid zero. Scanner-029 now implements the offline-first
`GET_LEN` boundary in `rgb-exposure-length`. It is dry by default and its only
possible live request is `a1 85`, `wValue=0x1200`, `wIndex=0x0400`, `wLength=2`,
timeout 1000 ms, once through the established interface-0 lifecycle. It
preserves actual length, retained raw bytes, SHA-256 and typed transfer or
cleanup failures; only an exact two-byte reply is decoded little-endian as a
raw reported control length, never as support or exposure semantics. No
physical GET_LEN was run in scanner-029. Scanner-030 subsequently ran that
exact boundary once and completed its separate immutable blue-light off write
and clean restoration. Every new physical GET_LEN invocation must again issue
its own read followed by exactly one immutable off write and clean restoration;
neither result alone establishes support or semantics.

Scanner-030 then ran that exact GET_LEN boundary once on the authorized MINI.
It returned exactly 2/2 bytes, raw `0200` (little-endian raw value `2`), with
clean release/reconnect. The separate mandatory immutable blue-light-off write
was accepted 60/60 and restored the interface; three postflight samples stayed
stable. These are command/restoration facts only, not exposure or optical
readback evidence. Scanner-031 now adds the next offline-only boundary:
`rgb-exposure-info`, whose only possible live request is one `a1 86`,
`wValue=0x1200`, `wIndex=0x0400`, `wLength=1`, timeout 1000 ms. An exact
one-byte reply is shown as raw UVC GET_INFO capability bits using Linux-UAPI
syntax; that bit view does not prove selector advertisement, semantics, SET
acceptance, units or frame behavior. It remains dry by default, and a later
physical task must again end with one separate immutable off write and stable
restoration.

An earlier accidental opted-in host run is recorded only as an observed audit:
the already-created `/tmp/scanner3d-paired-capture` directory contained a
512000-byte depth artifact (SHA-256
`0503b236738b95167a4cf283e886746460078a19f17cbc670820d9baf3189ee8`) and a
16947-byte RGB JPEG (SHA-256
`cc5f702fd18746a04d3bc5dcd814764bf93907a34f032231160a4df5b7069463`), both timestamped
17:45:46. The RGB bytes had valid 1280x800 SOI/SOF/SOS/EOI structure but four
trailing bytes `95 81 7d 01`, so strict validation reported `trailing_data`.
Trimming those bytes was not performed and no derived file was written; the
hypothetical trimmed-byte hash was recorded as
`818e026092e3f342f3cc9b08c50efbc5cd7f83548cd2fde7e9446ac5c48f44ab`, not produced.
This does not establish useful image content, synchronization, or repeatability. Scanner-017
subsequently repeated the accepted paired lifecycle once; its evidence is recorded below.

The accepted scanner-017 run delivered a 22529-byte RGB payload (SHA-256
`650240568e5b97fcc8f61773035e03e2b48fb5bb940c2d73de8572f83609c07d`) with a baseline 1280x800
JPEG prefix through EOI at offset 22523 and opaque trailing bytes `54 52 d8 01`. Strict
classification remained `trailing_data`; the hypothetical 22525-byte prefix hash was
`77ad1601dc234b059e5e076380ba8cb8c2b4f64f59c4f3fdb98ec2572b549199`, and no trimmed file was
written. In-place viewing found a nearly black image with only faint features. The depth payload
was one complete 512000-byte `Y16 ` frame (SHA-256
`950c9640612dd135b104bcdff1ef7536da435ca4eb6e34f7043ee58a5cd10c6a`); no metric interpretation
is claimed.

Scanner-019 then delivered three indexed RGB payloads in one accepted physical
series run. Their JPEG prefixes all declared 1280x800, but each had four opaque
bytes after EOI, so all remained strict-invalid `trailing_data`; the images were
nearly black and no useful RGB or warm-up improvement is claimed. The
`rgb-controls` diagnostic is intentionally separate: it is inert by default and
requires the exact `--read-rgb-controls` opt-in plus the authorized serial. It
opens only the identity-resolved RGB capture node read-only and queries a fixed
whitelist of gain, auto/absolute/priority exposure, auto white balance and
white-balance temperature. It issues no control writes, menu enumeration, format
or stream operations, and does not guess the vendor RGB exposure extension.

Scanner-021 is the one accepted physical `rgb-controls` invocation. Preflight
resolved serial `EXAMPLESERIAL0001` at USB path `1-1.1`, devnum 46, RGB
interface 2 (`RGBCamEXAMPLESERIAL0001`), `uvcvideo`, V4L2 capture index 0 at
`/dev/video2`, mode `0660`, with read access and no holder. The command exited
0 with `complete`, `succeeded=true`, one close, stable post-close identity and
11 ioctls (the fixed maximum is 12); stdout and `outcome.json` were byte-
identical and JSON-equivalent. The six read-only observations were:

| Control (ID) | QUERYCTRL metadata | Flags | Current/status |
| --- | --- | --- | --- |
| gain (`0x00980913`) | `Gain`, integer, 1..16 step 1, default 7 | none | 16, supported |
| exposure_auto (`0x009a0901`) | `Auto Exposure`, menu, 0..3 step 1, default 0 | none | 3, supported |
| exposure_absolute (`0x009a0902`) | `Exposure Time, Absolute`, integer, 1..60000 step 1, default 496 | `INACTIVE` | 131072, inactive |
| exposure_auto_priority (`0x009a0903`) | `QUERYCTRL` returned `EINVAL` | — | unsupported; no value read |
| auto_white_balance (`0x0098090c`) | `White Balance, Automatic`, boolean, 0..1 step 1, default 1 | none | 1, supported |
| white_balance_temperature (`0x0098091a`) | `White Balance Temperature`, integer, 2800..7500 step 10, default 4600 | `INACTIVE` | 6500, inactive |

These are device-reported integers/menu or boolean values only; they do not
establish exposure quality, brightness, colour accuracy, safety or a setting
recommendation. Six read-only postflight samples spanning 6.257 seconds kept
both video-role bindings/nodes, permissions, devnum 46 and zero holders stable
(`depth` `/dev/video0`, RGB `/dev/video2`). The outcome SHA-256 was
`f6540078c317a5ce7cd8dc1be6f1fc23bf8f968917c2d402b96a92fbac8fc9c2`.

Scanner-024 is the accepted physical paired run: one read-only, unprivileged
invocation delivered one complete depth frame and three strict-invalid,
nearly-black RGB payloads with four opaque bytes after each JPEG EOI. Standard
gain 16, exposure-auto index 3 and absolute-exposure raw 131072 were invariant
at all four telemetry points; these raw values do not establish units or the
separate selector-`0x12` state. No setting, vendor/XU request, retry or
elevation occurred; a devnum change from 46 to 47 was observed during stable
postflight sampling without an attributed cause.

Both capture commands are dry/preflight by default and open no video node without their
`--execute-...` flag. Each is fixed at import time to exactly one profile: `depth-frame` to the
`depth` USB interface role and `Y16 ` 640x400, `rgb-frame` to the `rgb` role and `MJPG` 1280x800.
Neither accepts a format, size, fps, buffer-count, timeout, retry, node/device-path, JPEG-quality
or camera-control flag; the only caller choices are the exact serial and a local output directory.
They share one bounded V4L2/MMAP engine — one open, one `S_FMT`, one `REQBUFS`, one `STREAMON`,
one deadline-bounded poll, at most one `DQBUF`, no retry, and shielded cleanup — and report their
outcome as a status string plus a distinct exit code (for example `complete` 0, `poll_timeout` 4,
`format_mismatch` 17, `malformed_frame` 28). RGB-only capture keeps `uvcvideo` bound throughout.
An executed depth capture owes one fixed blue-light shutdown write after exhaustive V4L2 teardown;
that write temporarily takes interface 0 through the shared ownership lifecycle and then restores
the driver. A missing or failed shutdown is a failed capture, with recovery guidance; dry runs,
refusals before depth `STREAMON`, and RGB-only capture never send it.

### Depth blue-light finalization (scanner-026)

Every path that attempts depth `VIDIOC_STREAMON` records a mandatory final-state obligation. After
all owed V4L2 `STREAMOFF`/unmap/queue-release/close operations, the live depth command may issue
exactly one host-to-device `SET_CUR` through interface 0:

```
21 01 00 07 00 04 3c 00 e8 03 00 00
echo s 0x922 0 >/dev/rk_preisp       (NUL-padded to 60 bytes)
```

The payload SHA-256 is
`6030364daddcc4c9855d16da552c58161bbec5cf77ea23ace9de2f14839ffb0d`. The command is immutable,
has a one-transfer ceiling and has no optical readback: “accepted” means the transfer returned
the required 60-byte length; successful finalization additionally requires clean interface-0
driver restoration, not that an optical sensor verified darkness. If either condition fails, the
report explicitly says not to claim the blue light is off and recommends rerunning the bounded
finalizer or physically unplugging the scanner.

Required finalizer failures use write-accurate statuses: `short_write` and `overlong_write` mean
the fixed host write moved the wrong number of bytes; `ioctl_error`, `unexpected_exception`,
`interrupted`, and identity/driver-cleanup statuses retain typed phase evidence. A required result
is never successful without an exact 60-byte transfer record and clean restoration; an inert
`not_required` result is successful only when depth `STREAMON` was never attempted.

`doctor` distinguishes each failure class by both status string and exit code:

| Status | Exit | Meaning |
| --- | --- | --- |
| `ok` | 0 | Every check that ran passed |
| `device_absent` | 2 | No `2207:110c` device attached |
| `permission_denied` | 3 | Device found, but its nodes are not read-write for this user |
| `no_frames` | 4 | The reference probe ran but delivered no frame / no calibration |
| `unsupported_mode` | 5 | The device does not advertise a format this project needs |
| `unexpected_identity` | 6 | Something matched the USB id but is not the expected unit |
| `internal_error` | 7 | A diagnostic itself failed |

Checks that were deliberately skipped are reported as `not run` and never counted as passing.

Discovery reads only kernel-exported metadata: it opens no character device, needs no root, and
resolves the depth/RGB roles from each USB interface's **string descriptor**
(`DepthCam<serial>` / `RGBCam<serial>`) rather than from `/dev/videoN` numbering, which is assigned
in probe order and is not stable across replug or reboot.

## The reconstruction pipeline (entirely offline, no hardware)

Everything from here on is arithmetic over captured bytes. None of it opens a
device, and all of it runs on the nine real frames committed under
`tests/fixtures/frames/`.

| Module | What it does |
| --- | --- |
| `depth_metric` | one raw `Y16 ` frame -> 3D points, with the accounting for every excluded sample |
| `cloud_export` | joins that to the writer, carrying every caveat across |
| `pointcloud` | writes PLY, with the caveats inside the file |
| `geometry` | rigid transforms, rotation about an arbitrary axis, Horn's best-fit alignment |
| `registration` | aligns a turntable sweep and reports **loop-closure error** |
| `fusion` | merges registered clouds on a voxel grid |
| `denoise` | drops points nothing else in their own frame is near, at a neighbourhood sized from the cloud |
| `mesh` | marching cubes over the fused occupancy grid, incremental accumulation, OBJ/PLY export |
| `validation` | fits a sphere or plane and compares it to a **caliper** reading |
| `simulate` | renders synthetic frames of known shapes, for testing against ground truth |

Two of those deserve a note.

**`simulate` exists because self-consistency is not correctness.** Registration
that converges, fusion that merges and a mesh that closes can all be wrong in
the same direction with nothing in the pipeline noticing. It renders real
512000-byte frames of shapes whose geometry is known exactly, so the pipeline
can be checked against an answer it did not compute. Measured: a rendered sphere
deprojects back onto its own surface to within uint16 quantisation
(0.002492 against a 0.002500 limit), so it is the exact inverse of the
production deprojection rather than a second opinion from the same arithmetic.

**`registration` reports loop-closure error**, which is why a turntable is worth
its cost: compose every frame-to-frame transform around a full revolution and
compare against the identity. That is a self-contained accuracy measurement
needing no external ground truth. `geometry.Rigid.rotation_angle` uses
`atan2` rather than `acos` specifically so that metric does not hit a numerical
floor — measured 1.7e-16 rad against `acos`'s 6.3e-8 rad on a 36-step
revolution.

**What none of it establishes:** no distance produced by this stack has been
metrically validated. The depth scale's physical unit is unknown, so a mesh has
a correct *shape* and an unknown *size*. Closing that is
[capture-checklist.md](docs/capture-checklist.md), and it needs calipers rather
than code. One known limitation is written up separately in
[shear-detection-limits.md](docs/shear-detection-limits.md).

## The one owned vendor control request (opt-in, touches hardware)

`scanner3d firmware-version` is the only owned/native command in this project that sends a vendor
USB request to the scanner directly, and it can send exactly one thing (`doctor --reference-probe` also causes
scanner traffic, but indirectly: it runs the vendor SDK, which issues its own requests):

`scanner3d rgb-controls` is a separate standard V4L2 read-only inventory. It can issue only the
fixed `VIDIOC_QUERYCTRL`/`VIDIOC_G_CTRL` metadata/value reads described above, and remains inert
unless its exact opt-in is supplied; it never sends vendor controls or changes a setting.

```
bmRequestType=0xa1  bRequest=0x81  wValue=0x0600  wIndex=0x0300  wLength=60  timeout=3000 ms
```

That setup packet is a UVC `GET_CUR` observed from the vendor SDK on this unit. It is read-only, it
changes no device state, and its correct answer is known independently — the SDK reports the same
firmware and algorithm version strings from `queryCameras` — so a wrong answer is unmistakable.

The safety boundary is the request itself, not a policy flag. The whitelist is checked *before* any
file descriptor is opened, there is no CLI flag that can vary any field of the packet, and
`scanner3d.control` knows exactly one usbfs command number (`USBDEVFS_CONTROL`). It performs no
write, no interface claim or kernel-driver detach, no device reset, no URB submission and no video
operation, and it closes its descriptor on every path. Tests assert each of those structurally.

Without `--execute` the command opens nothing: it prints the request it would send, the
serial-resolved target node with its `rdev` snapshot, the kernel driver currently bound to the
target interface, and the VideoControl entity the request's `wIndex` names.

| Status | Exit | Meaning |
| --- | --- | --- |
| `ok` | 0 | 60 bytes returned, NUL-padded, matching the known versions |
| `not_attempted` | 0 | Dry run; no node was opened |
| `device_absent` | 2 | No device with that serial, or its usbfs node is gone |
| `permission_denied` | 3 | The usbfs node is not open-able by this user |
| `timeout` | 4 | The transfer exceeded its bounded timeout |
| `ioctl_error` | 5 | The kernel refused the transfer (e.g. `EBUSY`) |
| `unexpected_content` | 6 | 60 bytes returned, but not the known payload |
| `short_response` | 7 | Fewer than 60 bytes returned |
| `device_replaced` | 8 | The device re-enumerated between resolution and the transfer |

**Measured result of the direct read on this unit: `ioctl_error` / `EBUSY`.** usbfs implicitly
claims the interface a class request names, and `uvcvideo` owns interface 0, so the kernel refuses
the transfer before it reaches the wire. See
[docs/architecture.md](docs/architecture.md#first-owned-control-request-measured).

### Temporarily taking interface 0 (opt-in, mutates kernel driver binding)

`--detach-uvcvideo-from-interface-0` clears exactly that obstacle, and nothing else. It performs one
fixed lifecycle on interface 0 and no other interface:

```
confirm uvcvideo -> USBDEVFS_DISCONNECT -> USBDEVFS_CLAIMINTERFACE
  -> the same unchanged read -> USBDEVFS_RELEASEINTERFACE -> USBDEVFS_CONNECT
```

The driver is confirmed twice — from sysfs before anything is opened, and again from
`USBDEVFS_GETDRIVER` on the open descriptor immediately before the detach — and the command refuses
outright unless the target is the expected `2207:110c` MINI at an explicitly given `--serial` with
interface 0 bound to exactly `uvcvideo`. No option varies the interface, the driver, the setup
packet, the timeout or the order, and no phase is ever retried.

The reconnect runs in a `finally` from the moment the detach is *attempted*, and the release runs
with it whenever the claim may be held: on success, on a short or unexpected payload, on a timeout,
on any ioctl failure, on an unexpected Python exception, and on SIGINT or SIGTERM — including a
signal that surfaces from the disconnect ioctl itself or between its return and the claim, where
whether the kernel completed the detach is unknowable. Only a disconnect that failed cleanly with an
errno undoes nothing, because the kernel then took nothing away. Cleanup happens at most once, so
none of this is a retry. A cleanup that does not fully succeed forces exit code 9 — it can never be
reported as success — and prints the exact rebind or replug needed to recover.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(cleanup failed)* | 9 | Overrides every row below, success included; recovery guidance is printed |
| `refused` | 10 | Preconditions not met; nothing was opened or detached |
| `detach_failed` | 11 | `USBDEVFS_DISCONNECT` failed; nothing was taken, nothing to undo |
| `claim_failed` | 12 | `USBDEVFS_CLAIMINTERFACE` failed; the driver was reconnected |
| `interrupted` | 13 | SIGINT/SIGTERM inside the ownership window; cleanup still ran |

**Current measured result on this unit: `ok`.** The device returned the full 60 bytes,
`v2.8.7.20220610|v7.5.7.0426`, in 1.3 ms — the first vendor bytes any code in this repository has
read, and they match what the vendor SDK independently reported for this unit. The detached window
lasted 127 ms, cleanup succeeded, and the device came back unchanged apart from `/dev/video0` and
`/dev/video1` being recreated by the `uvcvideo` re-probe.

That is **one read-only request**. Nothing about streaming or calibration is implied, and the effect
of a detach on an *active* capture has not been measured. See
[docs/architecture.md](docs/architecture.md#owned-code-reads-the-vendor-channel-measured).

## The `/tmp/inited` sequence (opt-in, WRITES to the scanner)

`scanner3d inited-sequence` is the only command in this project that sends bytes *to* the scanner.
It reproduces, exactly once and in order, the four 60-byte transfers the vendor SDK was observed
performing on this unit — rows 1–4 of the byte-by-byte capture in
[docs/architecture.md](docs/architecture.md#the-vendor-channel-byte-by-byte):

```
1. SET_CUR  bmRequestType=0x21 bRequest=0x01 wValue=0x0100 wIndex=0x0400 wLength=60
            payload: 01 "/tmp/inited" 00, zero-padded to 60
2. GET_CUR  bmRequestType=0xa1 bRequest=0x81 wValue=0x0200 wIndex=0x0400 wLength=60
3. GET_CUR  the same request again
4. SET_CUR  the same packet as (1), payload: 60 zero bytes   <- protocol cleanup
```

`wIndex=0x0400` is `(entity 4 << 8) | interface 0`, so every transfer addresses the extension unit
the device's own descriptors declare, on the interface the lifecycle takes from `uvcvideo`. Nothing
here reaches interface 4 or the HID function.

**Those four transfers are the only thing this command can send.** The path is a compile-time
constant, not an argument: there is no flag for a path, an opcode, a selector, an entity, an
interface, a length, a timeout or a repetition, and the other two paths in the same capture
(`/persist/sdk-license`, `/data/sdk-license`) do not appear in the module's executable source at
all. The whole sequence is validated before a file descriptor is opened, and each transfer again
before its ioctl argument is packed. Offline tests assert the exact packed setup bytes and the exact
outbound payload bytes.

The ownership lifecycle is the same one `--detach-uvcvideo-from-interface-0` uses, shared rather
than copied, with **one extra guarantee**: the clearing write in step 4 is protocol cleanup, not an
optional fifth idea. It is owed from the instant *before* the path write is attempted — so an
interrupted or ambiguous write still owes it — attempted exactly once, shielded from SIGINT and
SIGTERM, and run *before* the driver is given back, because it needs the claim. The driver handback
then runs from a `finally` around it, so a clearing write that fails, or raises outright, can never
cost you your driver. The two undos are reported separately and neither can be presented as success.

The live mode requires an explicit `--serial` and the deliberately separate `--execute-write-sequence`
flag — not the read command's `--execute` — so a habit formed on the read-only command cannot write
to the device by accident. Without it, nothing is opened and nothing is sent.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(driver cleanup failed)* | 9 | Outranks every row below; interface 0 may still be detached |
| *(clearing write failed)* | 14 | Outranks every row below except 9; the device may still hold the path selected |
| `ok` | 0 | All four transfers matched the SDK-observed sequence exactly |
| `not_attempted` | 0 | Dry run; nothing opened, nothing sent |
| `device_absent` | 2 | No device with that serial, or its usbfs node is gone |
| `permission_denied` | 3 | The usbfs node is not open-able by this user |
| `timeout` | 4 | A transfer exceeded its bounded 3 s timeout |
| `ioctl_error` | 5 | The kernel refused a transfer |
| `unexpected_content` | 6 | 60 bytes returned, but not what the SDK saw |
| `short_response` | 7 | A transfer moved fewer than 60 bytes |
| `device_replaced` | 8 | The device re-enumerated between resolution and the sequence |
| `refused` | 10 | Preconditions not met; nothing was opened, detached or written |
| `detach_failed` | 11 | `USBDEVFS_DISCONNECT` failed; nothing was taken, nothing written |
| `claim_failed` | 12 | `USBDEVFS_CLAIMINTERFACE` failed; nothing written, driver reconnected |
| `interrupted` | 13 | SIGINT/SIGTERM inside the window; both undos still ran |

**Current measured result on this unit: `ok`.** All four transfers matched the capture byte for
byte. The path write was accepted in full; the first read returned `0c 00 00 00` followed by
`v7.5.7.0426`, NUL-padded to 60 bytes; the second read returned 60 zeros; the clearing write was
accepted in full. The detached window lasted 40 ms, both cleanups succeeded, the kernel log gained
exactly one line (`uvcvideo` re-probing), and the device came back unchanged apart from
`/dev/video0` and `/dev/video1` being recreated.

**This is the first owned write, and the first multi-transfer vendor exchange reproduced without the
SDK.** It does not mean `/tmp/inited` was read — the reply is the algorithm version string, which is
not plausibly that path's contents — and it says nothing about any other path, selector, entity or
`SET_CUR`. The licence gate is unchanged and remains inside `lib3DCamera.so`. See
[docs/architecture.md](docs/architecture.md#owned-code-writes-to-the-vendor-channel-measured).

## Depth intrinsics (opt-in, WRITES a key to the scanner)

`scanner3d depth-intrinsics` performs, exactly once, the bounded three-transfer exchange the vendor
firmware's `libuvcReadFile` serves for the factory depth-intrinsics key `/data/camparam/Pl.bin`. It
is the same recovered protocol, the same entity and the same lifecycle as `inited-sequence`, with
one byte-string changed — the key — and a bounded chunk loop in the middle:

```
1. SET_CUR  bmRequestType=0x21 bRequest=0x01 wValue=0x0100 wIndex=0x0400 wLength=60
            payload: 01 "/data/camparam/Pl.bin" 00, zero-padded to 60
2. GET_CUR  bmRequestType=0xa1 bRequest=0x81 wValue=0x0200 wIndex=0x0400 wLength=60
            reply: uint32_le chunk length, then that many bytes; repeated until a zero length
3. SET_CUR  the same packet as (1), payload: 60 zero bytes   <- protocol cleanup
```

**This is not a file reader.** The key is a compile-time constant and the payload is derived from it
at import time. No function and no flag takes a path, an opcode, a selector, an entity, an
interface, a length, a timeout, a cap, a capacity or a repetition count. The sibling calibration
keys (`Prgb`, `LC_RT`, `Distort`), the licence paths and the write-file opcode do not appear in the
module's executable source at all, and the depth-scale channel (`wValue=0x0e00`) is not reachable
from it either.

Every bound is fixed at import and never widened at runtime: the output capacity is **256 bytes**, a
run performs at most **8** result reads — so at most **10** control transfers including the key
write and the clearing write — and a chunk declaring more than **56** bytes (more than fits behind
the 4-byte header) is rejected *before* anything is copied. A chunk that would pass the capacity is
not copied at all rather than truncated, and no transfer is ever retried.

The assembled bytes are the primary evidence: their length, SHA-256 and full hex are reported
whatever they turn out to be. Only a complete blob of exactly 40 bytes is additionally parsed as the
vendor `Intrinsics` record (`short width, height` then nine x86 little-endian floats laid out as
`fx, 0, cx, 0, fy, cy, 0, 0, 1`), and the parse never alters, normalizes or substitutes a value: a
structurally divergent record comes back with the divergence recorded, as typed evidence rather than
a crash — and never as accepted calibration. Structural checks and plausibility observations are
kept apart, and only a run whose clearing write *and* driver handback both succeeded offers the
record as calibration at all.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(driver cleanup failed)* | 9 | Outranks every row below; interface 0 may still be detached |
| *(clearing write failed)* | 14 | Outranks every row below except 9; the key may still be selected |
| `ok` / `complete` | 0 | A 40-byte, structurally exact record, both undos done |
| `not_attempted` | 0 | Dry run; nothing opened, nothing sent |
| `no_content` | 15 | The first reply carried a zero length word (cause not distinguishable) |
| `cap_exhausted` | 16 | 8 reads without a zero length word; the cap was not widened |
| `capacity_exhausted` | 17 | The next chunk would pass 256 bytes; it was not copied |
| `malformed_length` | 18 | A chunk declared more than 56 bytes; nothing was copied |
| `unexpected_content` | 6 | Terminated, but not a structurally exact 40-byte record |
| `short_response` | 7 | A transfer moved fewer than 60 bytes |
| `timeout` / `ioctl_error` | 4 / 5 | A transfer timed out at 1000 ms, or the kernel refused it |
| `refused` | 10 | Preconditions not met, or a serial other than the authorized one |
| `interrupted` | 13 | SIGINT/SIGTERM inside the window; both undos still ran |

The live mode requires an explicit `--serial`, its own `--execute-calibration-read` flag (not the
other commands' `--execute` or `--execute-write-sequence`), and refuses any serial but the single
unit the owner authorized this read for.

**Current measured result on this unit (2026-08-21, one run): `ok` / `complete`.** The key write was
accepted in full; the first result read returned `28 00 00 00` followed by a 40-byte chunk; the
second returned a zero length word; the clearing write was accepted in full. Two result reads, four
control transfers in total, a 5.4 ms read inside a 37.5 ms owned window, both cleanups clean, and
the device came back with interfaces 0–3 on `uvcvideo`, interface 4 on `usbhid` and only
`/dev/video0`/`/dev/video1` recreated. The 40-byte blob
(`fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf`) parses exactly as the vendor
layout:

```
frame  640x400                       (a mode this unit's own descriptors advertise)
fx     898.6829833984375   fy  898.6829833984375
cx      42.43747329711914  cy  192.8326873779297      (pixels of that 640x400 frame)
matrix [[898.68298, 0, 42.437473], [0, 898.68298, 192.83269], [0, 0, 1]]
```

All five structurally fixed entries were exact (`0, 0, 0, 0, 1`) and every float finite, which is
what makes the row-major pinhole reading of the layout more than a guess. One plausibility
observation is flagged and is recorded rather than explained: **cx sits 277.56 px left of the
320 px frame centre**, while cy is within 7.2 px of the vertical centre. *Inference, not
measurement:* a principal point that far off-centre is consistent with the record describing a
cropped or offset sensor window rather than the centred 640×400 frame it names — but nothing here
establishes that, and the values are reported exactly as they arrived. This read also says nothing
about metric accuracy, coordinate handedness, RGB alignment or depth scale; those need their own
evidence. See
[docs/architecture.md](docs/architecture.md#owned-code-reads-the-factory-depth-intrinsics-measured).

## Depth scale (opt-in, WRITES to the scanner)

`scanner3d depth-scale` performs, exactly once, the bounded two-transfer exchange the vendor
firmware's `libuvcReadValueExtension` serves for the depth-scale property `0x918`. It shares the
same entity and the same ownership lifecycle as `depth-intrinsics`, but it is a smaller primitive —
a closed two-transfer whitelist, not a property reader — and unlike the file-read protocols above it
has no protocol-clear transfer at all:

```
1. SET_CUR  bmRequestType=0x21 bRequest=0x01 wValue=0x0e00 wIndex=0x0400 wLength=4  timeout=1000 ms
            payload: 18 09 00 00 (0x918, little-endian)
2. GET_CUR  bmRequestType=0xa1 bRequest=0x81 wValue=0x0e00 wIndex=0x0400 wLength=60 timeout=1000 ms
            reply: the vendor's fixed 60-byte reply; byte 59 is the marker, bytes 0:4 the
            signed little-endian int32 denominator
```

**There is no third transfer, and none is invented.** The vendor helper's `memset(buf, 0, 60)`
between the two calls clears only its own host-side buffer; the function issues exactly these two
transfers and no clearing write of its own, so the file-read protocols' trailing 60-zero clearing
write is deliberately not borrowed here. `protocol_cleanup` reports "not applicable" and says why.
The driver handback — release, then reconnect — still runs on every exit, exactly as the shared
lifecycle enforces it elsewhere.

**This is not a property reader.** The property id `0x918` is a compile-time constant and the
4-byte payload is packed from it at import time. No function and no flag takes a property id, a
selector, an entity, an interface, a length, a timeout, a count or a retry, and no other `0x9xx` id
appears in the module's executable source. The whitelist is checked as a whole before anything is
opened, and each transfer again immediately before its ioctl argument is packed.

The 60-byte reply is preserved in full — length, SHA-256 and hex — as the primary evidence, and the
decode is reported beside it rather than in place of it. Byte 59 is the vendor's marker, accepted
only when it is `4`; bytes 0:4 are the signed little-endian `int32` denominator the vendor divides
`1.0f` by. Only an exact 60-byte reply carrying that marker and a positive denominator yields a
scale, and that scale is `float32(1.0f / denominator)`, reported with its exact bit pattern.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(driver cleanup failed)* | 9 | Outranks every row below, success included |
| `ok` / `complete` | 0 | An exact 60-byte reply, marker 4, positive denominator, driver handback done |
| `not_attempted` | 0 | Dry run; nothing opened, nothing sent |
| `unexpected_marker` | 19 | Byte 59 was not 4; the reply is reported in full, no scale offered |
| `zero_denominator` | 20 | Bytes 0:4 were 0; the reciprocal is undefined |
| `negative_denominator` | 21 | Bytes 0:4 were negative; no scale offered |
| `non_finite` | 22 | Defensive only — unreachable through the gates above |
| `unexpected_content` | 6 | An exact 60-byte reply that failed one of the gates above |
| `short_response` | 7 | Fewer than 60 bytes returned on either transfer |
| `timeout` / `ioctl_error` | 4 / 5 | A transfer timed out at 1000 ms, or the kernel refused it |
| `refused` | 10 | Preconditions not met, or a serial other than the authorized one |
| `interrupted` | 13 | SIGINT/SIGTERM inside the window; the driver handback still ran |

The live mode requires an explicit `--serial` and its own `--execute-depth-scale-read` flag (not
any other command's execute flag), and refuses any serial but the single unit the owner authorized
this read for. The subcommand is parsed with `allow_abbrev=False`, so `--execute-depth-scale-read`
must be typed in full — argparse will not accept a truncated prefix of it, unlike most other flags
in this CLI.

**Current measured result on this unit (2026-08-21, one run): `ok` / `complete`.** The property
write was accepted in full (`18090000`); the result read returned the full 60 bytes in 17.715 ms,
SHA-256 `9873fa193adabb7daa87a77718d963d586d798838f4e6edc1a2ddac0bb337dd4`, hex
`c80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004`.
Byte 59 was `4` and bytes 0:4 decoded to the denominator `200`, giving
`vendor_depth_scale = float32(1.0f / 200) = 0.004999999888241291` (float32 bits `0x3ba3d70a`).
Exactly two control transfers, no retry, a 19.426 ms read inside a 27.880 ms owned window, and the
device came back with identical serial and devnum (no re-enumeration), identical descriptors, and
interfaces 0–3 back on `uvcvideo` and interface 4 on `usbhid`.

This value is bit-identical to the vendor SDK's compiled-in MINI fallback constant `0.005f` — an
**observation**, not a unit: **the scalar's physical unit is unknown from this evidence.** Two
transfers and one reciprocal prove what the device returned, not what it means, and nothing here
establishes millimetres, metres, accuracy, or a raw-depth conversion factor. That the SDK's authors
chose `0.005f` because it matches this device's reported value is, at most, an inference the
agreement is consistent with — never a fact this run establishes. See
[docs/architecture.md](docs/architecture.md#domain-4--calibration--depth-scale-fact-disassembled).

## Reference probe (opt-in, touches hardware)

The probe is a small standalone C++ program built against an **externally supplied** SDK. It is
never run unless you explicitly ask for it.

```bash
# Build (fails loudly if the SDK is absent; never downloads anything)
REVOPOINT_SDK_ROOT=/path/to/3DViewer/thirdparty/3DCamera reference/probe/build.sh

# Run through doctor, under a hard timeout, with SDK hashes verified
scanner3d doctor --reference-probe \
  --sdk-root /path/to/3DViewer/thirdparty/3DCamera \
  --probe-binary /tmp/3d-scanner-probe/probe \
  --probe-timeout 12 --json
```

The adapter refuses to run unless the SDK's SHA-256 hashes match
[`src/scanner3d/data/sdk-manifest.json`](src/scanner3d/data/sdk-manifest.json)
(override with `--allow-unverified-sdk`). The manifest ships inside the
`scanner3d` package itself so an installed CLI can verify the oracle without
the repository present. It caps the run at 15 s, kills the whole process group on timeout, and
leaves no stream open. See [`reference/probe/README.md`](reference/probe/README.md).

## usbfs tracer (opt-in, observation only)

[`reference/tracer/`](reference/tracer/README.md) is an `LD_PRELOAD` ioctl interposer that records
the `USBDEVFS_*` calls one process makes to **one exact** usbfs node, and `scanner3d trace` decodes
the result offline. It exists because the vendor SDK reaches this scanner entirely through
`ioctl()` on `/dev/bus/usb/BBB/DDD` using asynchronous URBs, which `strace` can only show as bare
pointers — from inside the process, the wire bytes are readable.

```bash
reference/tracer/build.sh                     # -> /tmp/3d-scanner-trace/libusbfs_trace.so

SCANNER3D_TRACE_TARGET=/dev/bus/usb/001/041 \
SCANNER3D_TRACE_OUT=/tmp/3d-scanner-trace/capture.jsonl \
LD_PRELOAD=/tmp/3d-scanner-trace/libusbfs_trace.so \
  timeout 15 /tmp/3d-scanner-probe/probe --serial EXAMPLESERIAL0001

scanner3d trace /tmp/3d-scanner-trace/capture.jsonl.<pid>
```

It observes and never alters: every hooked call is forwarded to the real libc symbol with its
arguments, return value and `errno` untouched, and the tracer issues no device operation of its
own. Capture is off unless both environment variables are set, and it fails closed — an absent
target, a target that is not a character device under `/dev/bus/usb/`, or an output path that
already exists all mean *nothing is recorded*, never *everything is recorded*. Payload bytes, line
length and total output size are all capped, and hitting a cap is written into the stream rather
than dropped silently. Output is one file per process (`<path>.<pid>`), because a traced program
may fork and exec.

`scanner3d trace` is a pure file read: no device, no SDK, no network. It validates the stream,
correlates URB submissions with their completions deterministically, and preserves unmatched,
failed and truncated events as evidence instead of discarding them.

What one bounded capture of the vendor SDK established about this scanner's control channel is in
[`docs/architecture.md`](docs/architecture.md#vendor-control-channel-measured-on-the-wire).

## Repository layout

```
src/scanner3d/
  errors.py            status vocabulary and exit codes
  usb_descriptors.py   tolerant parser for the raw USB config descriptor
  discovery.py         sysfs discovery, rooted for fixture replay
  control.py           the single whitelisted read-only USB control request
  ownership.py         the interface-0 ownership lifecycle, and that one read
  init_sequence.py     the four-transfer /tmp/inited sequence on entity 4
  intrinsics.py        the bounded /data/camparam/Pl.bin depth-intrinsics read
  depth_scale.py       the closed two-transfer 0x918 depth-scale property read
  v4l2_capture.py      the shared bounded V4L2/MMAP single-frame engine and the depth profile
  rgb_frame.py         the fixed MJPG 1280x800 RGB capture profile for that engine
  jpeg_inspect.py      dependency-free structural JPEG reader (markers only, no pixel decode)
  doctor.py            report assembly and status classification
  reference.py         opt-in external SDK probe adapter
  usbfs_trace.py       offline decoder for recorded usbfs traces
  cli.py               `scanner3d` entry point
reference/probe/       standalone C++ reference probe (source only)
reference/tracer/      LD_PRELOAD usbfs ioctl tracer (source only)
docs/                  architecture, roadmap, and measured-hardware notes
tests/                 fixture-driven test suite
```
