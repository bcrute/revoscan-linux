# The shear indicator and single-view coverage (a fixed defect, recorded)

Status: **the defect described below was FIXED on 2026-08-24, after this
document was written.** The record is kept because the failure mode is worth
remembering and because the measurements are the reason the fix happened. Skip
to [After the fix](#after-the-fix) for current behaviour.

Measured offline against both the validation harness's own synthetic generator
and independently rendered frames from `scanner3d.simulate`.

## The claim under test

`scanner3d.validation.fit_sphere` reports a shear indicator. Its purpose is the
one stated in [capture-checklist.md](capture-checklist.md): a wrong principal
point acts as a shear `x += k·z`, shear turns a sphere into a tilted ellipsoid,
and a sphere target is therefore supposed to catch a `cx` error that a flat
target cannot.

`SphereFit` documents `shear_coefficient` as reading back "the `k` that would
explain it — the same dimensionless `x` per `z` that a wrong principal point
contributes, so it is directly comparable against `(centre_column - cx) / fx`."

On this unit that reference value is `(320 − 42.437) / 898.683 = 0.3089`.

## What was measured

Both experiments vary exactly one thing: how much of the sphere is visible.

### The harness's own generator, one variable changed

`sphere_band()` defaults to `z_from=-1, z_to=1` — the **whole sphere** — and
that default is what the coefficient tests use. Its own docstring notes that
`(-1, 0)` is "the hemisphere nearer the camera, which is all a single view could
ever see."

Injecting `k = 0.02` and changing only the band:

| Visible band | Recovered `k` | Fraction of truth |
| --- | --- | --- |
| whole sphere `(-1, +1)` | 0.019981 | **99.9%** |
| most of it `(-1, +0.5)` | 0.017903 | 89.5% |
| front hemisphere `(-1, 0)` | 0.005932 | 29.7% |
| realistic cap `(-1, −0.3)` | 0.001718 | 8.6% |

### Independently rendered frames

Rendering a sphere (r = 20 at z = 117) with `scanner3d.simulate` and
deprojecting with a deliberately wrong `cx`, so the fault magnitude is chosen
rather than assumed:

| `cx` error (px) | Injected `k` | Recovered `k` | Fraction |
| --- | --- | --- | --- |
| 1 | −0.001113 | −0.000057 | 5.2% |
| 20 | −0.022255 | −0.001160 | 5.2% |
| 100 | −0.111274 | −0.006021 | 5.4% |
| 277.56 | −0.308852 | −0.017605 | 5.7% |

The fraction is **constant across four orders of magnitude**. That rules out the
"first order in `k`" caveat in the docstring as the explanation: a first-order
model reads near 100% for small `k` and degrades for large. A constant factor is
a systematic scale error, and its size depends on the visible geometry rather
than on `k`.

## The serious part: the detector never fired (since fixed)

`shear_detected` requires `|shear_correlation| ≥ 0.35`. Measured on rendered
frames, against the *maximum possible* fault on this unit:

| `cx` error (px) | \|correlation\| | `shear_indicator_reliable` | `shear_detected` |
| --- | --- | --- | --- |
| 0 | 0.000 | True | no |
| 1 | 0.177 | True | no |
| 20 | 0.229 | True | no |
| 100 | 0.235 | True | no |
| **277.56** | **0.247** | **True** | **no** |

Correlation saturates near 0.25 and cannot reach 0.35. Confirmed against the
harness's own generator: a whole sphere reads 0.997 and fires, a perfect
hemisphere reads 0.544 and fires, a realistic cap reads 0.293 and does not.

A real camera sees **less than a hemisphere** — for r = 20 at z = 117 the
silhouette sits at about 80° from the near pole, and projection samples the limb
sparsely. So the threshold sits in a gap that excludes every single-view
capture.

**The failure mode is the dangerous one.** The harness does not report
uncertainty here. It reports `shear_indicator_reliable = True` alongside
`shear_detected = False`, which reads as "checked, and clean" — a positive
clearance — while the largest principal-point error the unit could have is
present in the data.

## A further complication, not yet explained

On a realistic cap the statistic is not monotonic in `k`, and an unsheared
noiseless cap does not read zero:

| Condition (realistic cap) | \|correlation\| |
| --- | --- |
| unsheared, no noise | **0.150** |
| unsheared, jitter 0.05 | 0.025 |
| unsheared, jitter 1.0 | 0.060 |
| sheared k = 0.01, jitter 0.1 | 0.008 |
| sheared k = 0.05, jitter 0.1 | 0.067 |
| sheared k = 0.31, jitter 0.1 | 0.262 |

A genuinely sheared cap at `k = 0.01` reads *lower* than a clean one. The 0.150
on noiseless input vanishes as soon as any noise is added, which suggests it is
an artifact of the deterministic Fibonacci sampling interacting with a narrow
band rather than a general false-positive floor — but that is a hypothesis, not
a finding, and it has not been chased down.

**Consequence:** simply lowering the threshold is not a fix. Below about
`k = 0.05` on cap data there is no clean separation to threshold against.

## What was safe to conclude before the fix

**Safe.** On dense, near-complete coverage of a sphere the indicator is
excellent — correlation 0.997 and `k` recovered to 0.1%. That is not a
theoretical case: it is what a **fused multi-view scan** provides, which is
exactly what M6 produces. The indicator is sound; it is being asked the wrong
question by a single frame.

**Not safe.** A quiet indicator on a single-view sphere means nothing. It must
not be read as evidence that `cx` is correct, and `shear_coefficient` from a
single view is a lower bound of unknown tightness, not a measurement of `k`.

## What was recommended (items 1-3 superseded by the fix)

1. **`shear_indicator_reliable` must account for angular coverage, not only
   leverage.** Leverage reads ~0.26 for both a whole sphere and a single-view
   cap, so it cannot separate them; `cap_half_angle` is already computed and
   can. A cap that cannot reach the detection threshold must report the
   indicator as *unreliable*, so a quiet reading is correctly read as "cannot
   see" rather than "nothing there". This is the safety fix and matters most.
2. **Correct the `shear_coefficient` docstring.** "Directly comparable against
   `(centre_column - cx) / fx`" is true only for near-complete coverage. As
   written it invites reading 0.018 against 0.309 and concluding the shear is
   negligible — the opposite of the truth.
3. **Test against realistic coverage.** The coefficient tests use the
   whole-sphere default. Add cases at hemisphere and cap coverage that pin the
   real behaviour rather than the favourable case.
4. **Prefer the fused cloud.** The validation harness should be pointed at a
   fused multi-view cloud, where coverage is high and the indicator works. The
   capture checklist should say so.

Item 4 is the practical route and costs nothing: the turntable sweep the
checklist already asks for produces exactly the coverage the indicator needs.

## Method note

Neither experiment depends on trusting the other. The harness's own generator
and `scanner3d.simulate` were written by different authors from different
descriptions, and they agree: coverage, not fault magnitude, controls what the
indicator can see.


## After the fix

The estimator was reworked by the module's reviewer. Re-measured the same way,
with the same rendered frames and the same deliberately injected principal-point
errors:

| `cx` error (px) | Injected `k` | Recovered `k` | Fraction | \|correlation\| | Detected? |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | −0.000001 | — | 0.001 | no (correct) |
| 1 | −0.001113 | −0.000771 | 69.3% | 0.649 | **yes** |
| 20 | −0.022255 | −0.015371 | 69.1% | 0.832 | **yes** |
| 100 | −0.111274 | −0.075590 | 67.9% | 0.834 | **yes** |
| 277.56 | −0.308852 | −0.198418 | 64.2% | 0.830 | **yes** |

**The false clearance is gone.** On the single-view cap a camera actually
produces, the indicator now fires for a principal-point error as small as **1
pixel**, and stays quiet on a clean sphere. Correlation went from a 0.247
ceiling — permanently under the 0.35 threshold — to 0.83.

**What remains.** The coefficient still under-reports, now by about a third
rather than by twenty times: it reads 64–69% of the injected `k` across the
whole range. That is a far less dangerous error, because 0.198 against a true
0.309 is plainly the same order of magnitude and still obviously non-negligible,
where 0.018 read as "nothing to see". Treat `shear_coefficient` from a single
view as a **lower bound**, not a measurement.

The advice to sweep the sphere on the turntable stands, for a different reason
than before: fused coverage makes the coefficient accurate as well as the
detection, and the sweep costs nothing extra once the turntable is set up.

## Why this is recorded rather than deleted

The original defect is the most instructive kind this codebase can produce. The
module was not wrong in an obvious way — it worked excellently on the data its
tests used, and its tests used a whole sphere because that is the natural thing
to generate. It failed only on the distribution a real camera produces, and it
failed *silently*, reporting `shear_indicator_reliable = True` beside a quiet
indicator, which reads as a clean bill of health.

What caught it was feeding a fault of known magnitude from one module into
another written independently. Neither module's own tests could have found it,
because each was self-consistent. That is the argument for
`scanner3d.simulate` existing at all.
