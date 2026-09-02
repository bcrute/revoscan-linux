# Depth plausibility: are these distances a scene?

Status: implemented, advisory, and verified against every capture on this
machine. It decides nothing and refuses nothing — the only thing it is allowed
to do is appear on screen and in the log.

## The captures this exists because of

`20260831-215029`, `20260901-094953`, `20260901-145316` and `20260901-180235`.
Four consecutive sessions failed, over two days, and the cause was the same
every time: something had entered the scanner's view about 60 units away, while
the object sat at about 180.

It was roughly 4,600 returns per frame — 4.7% of the frame at its worst. Because
those points sat 100 units off the object, they dominated the cloud's covariance
and drove its principal-extent ratio to **1.05**. That is what
`multiview.coarse_align` means when it says "near-equal principal extents, so
its principal frame is not determined by its geometry": a deliberately
asymmetric target had been made to look rotationally symmetric by 5% of the
points. Orientation ranking then came back at 1.00–1.11× against the 1.15× it
requires, the tracker's references disagreed, and frames were refused.

The tracker did the right thing. It refused rather than inventing poses, and
`20260901-145316` carries no tracking-loss record because the refusals never ran
three deep. Nothing was corrupted. **Two days were lost anyway**, because
nothing said what was wrong.

## What the operator was actually told

Three statements, all true, none of them useful:

| Panel | What it said |
| --- | --- |
| return health | `usable — returns 36.5% of the frame` |
| travel readout | `ROTATION — GOOD` |
| tracker | `2 independent references each placed this frame acceptably and then disagreed — adjacent and model (14.530 apart) — against the 5.200 gate…` |

The gap is structural rather than an oversight in any one of them:

- **`doctor.py`** checks presence, identity, topology, roles, modes and
  permissions — the device, never a frame's content.
- **`return_health.py`** counts returns and locates them by quadrant. It is the
  right instrument for "the object slid off the sensor", and it reported health
  here *correctly*: the return **count** was normal. It was the **values** that
  were wrong.
- **`reconstruction_readiness.py`** is strict and thorough and runs after the
  capture is over.

So the whole check surface asked *"is the device there and returning pixels?"*
and nothing asked *"do these distances describe a scene?"* This module is the
sibling of `return_health`: that one asks **where** the returns are, this one
asks **whether they make sense**, and both answer per frame while a capture is
still salvageable.

## What it looks for

Three shapes, each one measured by hand on the failing captures before it was
written, and each cheap enough to run on every frame.

### A detached cluster

The depth histogram splits into a near group and a far group with an empty span
between them. Pooled over `20260901-145316`:

```
p0.1    60.6      ┐ 4.6% of returns
p1.0    65.5      ┘
                  ← nothing at all between ~66 and ~150
p5.0   149.6      ┐
p50    173.3      │ the object
p99    218.3      ┘
```

An object and the surface behind it are usually continuous, or separated by a
few units. A **59-unit void** with 4.6% of the frame stranded on the near side
of it is a second thing in the room.

Two conditions, and both are load-bearing:

- **The near group.** The failure puts something *between* the scanner and the
  subject. A far group is ordinarily the room behind it, which is not news.
- **A minority of the frame.** The subject is whatever most of the returns came
  from. *The first version of this module did not have this bound and was
  wrong*: an object standing clear of a distant wall splits into two groups too,
  and without the minority rule the object reported itself as an intruder. Two
  comparable groups are two answers, and it says nothing rather than guessing.

### Returns outside the working envelope

Distances the sensor was never characterised at — counted and reported, rather
than silently thrown away by `preparation.DEPTH_CLIP_NEAR`.

The envelope here (84–290) is deliberately **not** the clip (60–300). The clip
decides what is discarded before a cloud is built; the envelope decides what is
worth remarking on. A return at 65 is kept by the clip and is odd enough to
mention, and keeping the two apart is what lets this module report on the clip's
own behaviour instead of inheriting it.

### A region that will not move

Held across ten frames on a coarse 12×8 grid. On a capture where the object
turns and the camera is fixed, a patch whose depth does not change is fixed to
the *camera*.

It stays silent unless most of the view is moving, because a capture that is
genuinely still — framing a shot, a paused turntable — has every cell static,
and firing there would trip on the commonest harmless state there is.

## What it will not do

- **It never names a cause.** A near cluster can be a clamp, a cable, a hand,
  the projector's overexposed near field, or a correlation failure on a dim far
  surface. The bytes do not say which. Every sentence names what to **look at**
  and never what the thing **is**, and a test asserts that as a word list,
  because "it never names a cause" is the kind of promise that decays one
  helpful edit at a time.
- **It never decides anything.** It accepts and refuses no frame, stops no
  capture, moves no tracking gate, clips no sample and changes no reconstruction
  threshold.
- **It implies no unit.** Every distance is raw code times depth scale.

## What it does on the real captures

Every capture on this machine, frame 0, and the first 40 frames scanned for a
warning. Nothing here is a test — those files are not in this repository — but
it is the evidence the module was tuned against.

| Capture | | Frame 0 verdict | First warning |
| --- | --- | --- | --- |
| `20260827-152842` | clean | `plausible` | never |
| `20260828-062457` | clean | `plausible` | never |
| `20260831-215029` | 0.9% intruder | `outside envelope` | **frame 0** |
| `20260901-094953` | 2.3% intruder | `detached cluster` | **frame 0** |
| `20260901-145316` | 4.7% intruder | `detached cluster` | **frame 0** |
| `20260901-180235` | dimmer room | `outside envelope` | **frame 0** |

No false positive on a clean capture, and a warning on the first frame of every
bad one — including `20260831-215029`, the session the intruder first appeared.
Had this existed then, it would have said so two days and four captures earlier:

> `DETACHED CLUSTER — 1652 returns (1.8%) sit at 2-77, in the upper left of the
> view, 75 clear of everything else in the frame`
>
> Something is in the scanner's view much closer than the object. Check what is
> between the scanner and the target, and look at the upper left of the view
> first.

That location is not decoration. Finding it by hand took a pixel map and about
fifteen separate measurements against the raw frames; `locate_cluster` reaches
the same answer — the upper left, matching columns 60–210 and rows 0–64 of the
hand analysis — in one pass, on frame 0. The words come from the returns'
**centroid** rather than their bounding box, because a box spanning most of the
frame with its mass in one corner is a thing in that corner, and an operator
told "the whole view" would look nowhere.

Localising runs only when there is something to localise. It is a second walk
over the frame, and an ordinary frame must not pay for it.

## The sentence this got wrong, and how it was caught

The first version detected the near cluster correctly and said:

> Something is in the scanner's view much closer than the object.

There was nothing. On 2026-09-01 a colour frame settled it in one look: the near
returns were the **target's own ramp**, and the scanner was simply too close.
The whole scene ran from 1 to 232 with **46.5% of it inside the sensor's near
limit**. Four days of diagnosis had gone into depth histograms when a photograph
was one command away.

Both cases produce a detached near cluster. What tells them apart is how much of
that cluster was measurable at all:

| | near group | the subject |
| --- | --- | --- |
| a foreign object | in range | in range, behind it |
| the subject too close | **mostly below the envelope** | dragged out of range with it |

So a cluster more than `TOO_CLOSE_BELOW_ENVELOPE` below the near bound stops
hunting for a thing and names the distance instead. On the same capture it now
says:

> `DETACHED CLUSTER — 1652 returns (1.8%) sit at 2-77, in the upper left of the
> view, 75 clear of everything else in the frame`
>
> Most of that is nearer than the 84 this scanner was characterised down to, so
> it was never measurable. **Move the scanner back until the whole object sits
> inside the working range.**

## Verified on the device, before and after

Measured live on 2026-09-01, driving the scanner directly. The only change
between the two runs is the standoff:

| | before | after |
| --- | --- | --- |
| scene span | 1–232 | 55–244 |
| returns below 84 | **46.5%** | **1.04%** |
| return rate | 20.5% | **30.6%** |
| verdict | `DETACHED CLUSTER` | **`PLAUSIBLE`** |
| tracking, 30 frames at decimate 1 | — | **30/30 placed, 0 refused** |
| tracking, 60 frames at decimate 3 | — | 58/60, longest refusal run 1, travel median 2.86 (**good**) |

That is the whole value of this module stated as one row: the condition it
detects was fixed by one physical adjustment, and the adjustment was the thing
nothing in the appliance had ever said out loud.

## Where it runs

`ScannerWindow._show_plausibility`, on the same slot that draws every displayed
depth frame — so it runs during framing, before a capture starts, at the rate
the scanner delivers rather than the rate decimation keeps.

It reuses the sample tuple `viewer.render_frame` already unpacked. That module
promises one `struct.unpack` per displayed frame and one shared tuple "rather
than two that could disagree"; this is a third consumer on the same tuple, not a
second unpack.

The whole call is wrapped: a preview that stopped drawing because an *advisory*
indicator raised would be strictly worse than no indicator at all.

## What this deliberately does not do

No change to `DEPTH_CLIP_NEAR`. Raising it from 60 to 84 makes the symptom
vanish completely — replaying `20260901-145316` gives 64/64 frames placed with
zero coarse recoveries, against 61/64 with three — and that is exactly why it
would be the wrong move. The clip has been 60 the whole time and the August
captures were clean. Changing it would hide a fault that was doubling every few
hours. The clip is worth revisiting on its own merits, once the hardware is
clear and once the envelope has been characterised by a swept target rather than
taken from the 1st percentile of nine frames.

No drawing on the depth preview. The message names the third of the view a
cluster sits in, but the picture beside it is not marked. Outlining the region
on the preview is the obvious next step and is not done.

And it does not make anything more reliable by itself. It converts a silent,
day-losing failure into a named condition an operator can act on in the first
second. That is all it does, and it is the thing that was missing.
