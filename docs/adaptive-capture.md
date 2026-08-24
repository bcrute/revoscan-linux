# Adaptive capture: closed-loop control, and where ML actually earns its place

Status: a design argument, not an implementation. Nothing here is built. It
exists to answer a question that will otherwise get answered by reflex — should
capture parameters adapt to the scene, and should that adaptation be a learned
model?

Short answers: **yes, they should adapt**, and **no, that is not a machine
learning problem** — not yet, and not at the layer where the question usually
gets asked.

## The question

Scan quality on this sensor varies enormously with the subject. Depth yield
across nine real captured frames ranges from 2.43% to 36.60% of samples
returning a distance; zeros run 63%–98% per frame. A structured-light sensor
returns nothing from surfaces that are too dark, too shiny, too far, or too
close, and the parameters that decide where that cliff sits — exposure, gain,
working volume, confidence thresholds — are currently fixed at whatever the
firmware booted with.

Adapting them to the scene is obviously worth doing. The question is what kind
of machinery that takes.

## What we can actually reach

From `docs/dsp-protocol.md`, measured on this unit. The distinction between
these two groups is the whole point, and it is easy to blur:

**Confirmed implemented** — an id bound to a name by an observed reply:

| Id | Name | Value read |
| --- | --- | --- |
| `0x9a3` | `id_msg_cam_get_gain_reg` | 24 |
| `0x9b0` | `id_msg_cam_get_frame_time` | 20000 |
| `0x9b1` | `id_msg_cam_get_exposure_time` | 20000 |
| `0x9b2` | `id_msg_cam_get_auto_expo_mode` | 0 |
| `0x9c0` | `id_msg_cam_get_value_by_id` | via buffer |

**Names recovered but NOT bound to an id by evidence:**

```
id_msg_cam_set_gain_reg          id_msg_algo_set_z_range
id_msg_cam_set_exposure_reg      id_msg_algo_set_background_threshold
id_msg_cam_set_vts_reg           id_msg_algo_set_gradient_threshold
id_msg_cam_set_hts_reg           id_msg_algo_set_resolution_t
id_msg_cam_set_fake_mode         id_msg_algo_set_output_infrared
```

So we can **read** the current exposure, gain and frame time with confidence.
We cannot yet **write** any of them, and the one attempt to find the setters by
sweeping `0x910`–`0x91f` with value `0` left depth capture timing out until a
power cycle. Those ids are where the setters live, and writing zeros into them
is exactly the wrong way to find out which is which.

**This is the real first task**, and it is unglamorous: bind each setter name to
its id, one at a time, through the ioctl mailbox, with the log ring cleared, with
a sane value rather than zero, and with the current value read back first so it
can be restored. `auto_expo_mode` reading 0 is worth noting — automatic exposure
appears to be *off*, which may be the single cheapest quality win available.

## Why a controller, not a model

Once a setter is reachable, the loop is:

```
capture -> depth histogram -> propose parameters -> apply -> capture
```

The proposal step is arithmetic on a histogram. If the returns cluster at
12 cm, set the working volume to 8–20 cm. If yield is low and the returns that
exist are saturated, reduce exposure. If yield is low and returns are dim,
increase it. This is a hundred lines of well-tested code with a clear failure
mode, it explains its own decisions, and it can be unit-tested offline against
recorded frames with no device attached.

A learned model at this layer would need training data this project does not
have, add a dependency measured in gigabytes to a codebase with **zero runtime
dependencies**, and replace an explainable decision with an unexplainable one —
in order to do arithmetic. That is a bad trade at any level of ML enthusiasm.

The zero-dependency property is worth defending explicitly. It is why the whole
stack installs and runs offline with nothing but CPython. It should be given up
deliberately, for something that earns it, not incidentally.

## Where ML genuinely earns its place

Downstream, on the reconstruction side, where the problems are genuinely
ill-posed and classical methods have known ceilings:

- **Depth completion / hole filling.** Structured light returns nothing from
  dark and specular surfaces. With 63%–98% of samples empty per frame, this is
  the largest quality gap in the pipeline, and learned inpainting is
  legitimately better than classical interpolation at it.
- **Learned denoising.** Outperforms bilateral and median filters on depth,
  particularly at range boundaries.
- **Learned descriptors for registration.** Classical ICP with FPFH is strong
  when the initial alignment is close; learned descriptors are more robust when
  it is not. Relevant at M6.
- **Neural implicit surface reconstruction.** Competitive with Poisson when
  input is sparse and noisy, which describes this sensor well.

Every one of those sits at M6 or later.

## The precondition nobody should skip

**You cannot tell whether a model helps until there is a classical baseline
with quality metrics to compare against.**

Right now this project has one frame converted to a point cloud and *no
validated distances at all* — the depth scale's physical unit remains unknown
(see `docs/depth-to-pointcloud.md`). Adding a learned component before that
means never being able to answer whether it improved anything or merely made
different mistakes, because there is no yardstick. Worse, a model can absorb a
systematic calibration error and hide it behind plausible-looking output, which
is precisely the failure this codebase's whole design is built to prevent.

Order of operations:

1. Bind the setter ids safely. Restore-first, one at a time.
2. Build the closed-loop controller. Pure functions over a histogram, tested
   offline against recorded frames.
3. Establish metric validation — a known-dimension target, a predeclared
   tolerance. This is M4 and it gates everything quantitative.
4. Measure the baseline pipeline against it.
5. *Then* evaluate learned components, one at a time, against that baseline,
   and accept the dependency cost only where the measurement justifies it.

## The honest note on motivation

A closed-loop controller driving an undocumented coprocessor API that was
reverse-engineered from scratch is a harder and more distinctive piece of
engineering than attaching a pretrained model to a pipeline. The first is
something almost nobody could do on this hardware; the second is an afternoon
with a package manager. If the project is meant to demonstrate capability, the
controller is the stronger artifact — and it is also, independently, the
correct next step.
