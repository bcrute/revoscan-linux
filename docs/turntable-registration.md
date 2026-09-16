# Turntable registration: one axis, one rate

`scanner3d.turntable` registers a full turn of a turntable as what it is: one
rigid object turned about one fixed axis by an angle that grows with time.
Five unknowns for the whole sweep, where `scanner3d.pipeline.reconstruct_sweep`
has six per frame and only the frames' overlap to tie them together. It is
tried first by `scanner3d reconstruct`, by the desktop application's
Reconstruct + export and by `tools/target_gate.py`, and each falls back to the
frame-by-frame chain when the frames do not prove a turn (`--no-turntable`
forces the chain).

## Why

On 2026-09-04 a white model on the turntable - one large inclined face, a row
of fine columns, small discs - registered as a smear with the chain at every
depth band (frames one revolution apart 4-6% in agreement, 8-22 mm off), and a
second sweep of it at a finer step did the same. The chain's total rotation was
right; its camera positions wandered off their circle by up to 7 mm. The CAD
target had registered coherently the same day at the default band and
incoherently at every other, so the chain was fragile there too. Single frames
of both objects constrain a rigid motion equally weakly sideways (the weakest
eigenvalue of the point-to-plane information matrix is a translation across
the image, one thirtieth of the strongest, for both), so the difference was
not the frames but luck. See `docs/target-gate.md`.

## The estimators

Each unknown comes from the measurement that determines it best, linearly,
because a residual minimisation cannot see two of them:

| unknown | from | why there |
| --- | --- | --- |
| axis direction | the rotation axes of direct point-to-plane alignments between frames 60 degrees apart, every second frame, weighted by angle, the fifth farthest from the mean left out | a 2.8 degree step carries its rotation error straight into the axis direction; 60 degrees divides it by twenty, and the frames still overlap by two thirds. Wider pairs get worse again as overlap shrinks: 119 degrees apart the axes scatter by 3.7 degrees |
| centre | the same alignments: a rotation R about c has translation t = (I - R) c, solved in the plane across the axis by 2 by 2 normal equations | well conditioned at 60 degrees and hopeless at 2.8 |
| rate | frames one revolution apart, found by time: the rotation left between them after the model's own prediction is the rate error over a whole turn | the true period is fractional (126.68 frames, not 127) and nothing else measures it |
| seed | a chain of adjacent alignments over a quarter turn; the circle through its camera positions | the axis to half a degree and the centre to a few millimetres, which is all the estimators need to find their correspondences |

Rounds repeat until the centre moves less than half a millimetre; a seed 13 mm
off settles in three or four. After the first round the rate stays the period
pairs' rather than being re-estimated from the 60-degree pairs, whose angles
run a third of a percent low.

## What is needed, measured

The CAD gate amplifies an axis error about sixfold into face non-orthogonality
(a 20 mm face smeared by a 100 mm lever), so it is a sharp instrument for the
axis. Fits of the same CAD capture, differing only in how the axis and centre
were found:

| axis and centre from | axis off by | centre off by | RMS | orthogonality worst |
| --- | --- | --- | --- | --- |
| the pipeline's own 190-pose orbit, then a free residual fit | reference | reference | 0.61 mm | 0.87 deg, pass |
| a quarter-turn chain of adjacent alignments | 0.55 deg | 3.6 mm | 1.11 mm | 3.1 deg, fail |
| a full-turn chain, plain or loop-closed | 0.6-0.7 deg | 3.2 mm | 1.11 mm | 3.4-3.5 deg, fail |
| the turntable disc's mean normal | 0.80 deg | 1.8 mm | 1.06 mm | 2.9 deg, fail |
| the 60-degree pairs' axis, centre still from residual fitting | 0.20 deg | 3.0 mm | 1.11 mm | 3.5 deg, fail |
| a centre 1.85 mm off, axis 0.22 deg off | 0.22 deg | 1.85 mm | 1.11 mm | 2.0 deg, pass |
| **the direct estimators above (this module)** | 0.09 deg | 0.6 mm | **0.50 mm** | **0.75 deg, pass** |

So the axis must be right to about two tenths of a degree and the centre to
about half a millimetre. The frames' own residuals cannot see one of the two
tilt directions at all (flat for two degrees at a quarter turn: a 2 mm
systematic floor between views 90 degrees apart swamps it), which is why a
free five-parameter fit keeps whatever axis it was seeded with, and why the
disc's normal - which precesses by a degree around the turn and is seen at one
incidence - is not good enough either.

## Results

| capture | chain (pipeline) | turntable model |
| --- | --- | --- |
| CAD target 20260904-020845, 190 frames | 0.63 mm RMS, 23/24, orthogonality 1.45 deg, base plate 0.25 mm | 0.50 mm RMS, p95 0.89, 23/24, 95% observed, orthogonality 0.75 deg, base plate 0.11 mm |
| white model 20260904-191849, 180 frames | incoherent: 5% agreement one turn apart, 22 mm | neighbours 0.10 mm, a quarter turn 0.46 mm, one turn 0.11 mm (point-to-plane medians) |
| the same model, the operator's capture 20260904-182029 at held spacing, 146 frames | incoherent: 4%, 13 mm | 84.9 frames per turn from delivery times where the frame index said 75; neighbours 0.20 mm, one turn 0.26 mm |

The agreement numbers are the module's own report card, point-to-plane rather
than point-to-point because at the clouds' sampling density the nearest point
on a perfectly registered surface is a millimetre away. The image-period
coherence check (`scanner3d.coherence`) still runs on the result, but frames
one period apart agree by construction here, so it confirms only the rate.

## Angles from time

The desktop application's held spacing changes the interval between kept
frames mid-capture, so a frame's index is no measure of its angle: the
operator's capture had frames two to six deliveries apart. The model takes
each frame's angle from its delivery time, in median intervals, whenever the
times say the spacing varied; a capture that was evenly spaced registers
identically with or without them. The times come from the project file's
`frame_times_s`, which captures now record at their end, and from the frame
files' own modification times for older captures (`scanner3d.project.capture_frame_times`).
Copying a capture without preserving file times loses the second source; the
first survives.

## Cost

About 85 dense point-to-plane alignments per round on a 190-frame sweep, three
or four rounds, plus the seed chain: 40-70 seconds on the numpy or scipy
backend. The pure-python backend is many times slower; `scanner3d reconstruct
--backend` chooses.

## What it does not do

Nothing for a hand-held sweep, which has no axis: those go to the chain, and
to `docs/tracking.md`. Nothing for a partial turn: the frames must repeat for
the model to be attempted at all, and a sweep that stops short of a revolution
is registered frame by frame. And nothing about the sensor's own systematic
errors between viewpoints, which are what set the 0.4 mm quarter-turn floor.
