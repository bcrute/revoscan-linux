# The RK1608 coprocessor command protocol, measured

Status: measured live on this unit on 2026-08-24, now including ioctl-mailbox
measurements taken later the same day. Every claim below is either a reply the
coprocessor itself emitted into the kernel log, a clean negative from a batch
where other ids in the same batch *did* reply, or — for ids re-probed through
the ioctl mailbox — a reply obtained on a channel proven not to filter by
command id. Where the method broke down, that is stated rather than papered
over.

No optical or darkness claim is made anywhere in this document, with one
exception: a single visual check recorded below (the emitter remained lit
after `0x483`), reported as a negative result, not as evidence that any
command affects illumination.

## Why this exists

The Revopoint MINI's depth projector stays lit when idle. Every documented
host-side command for turning it off had been tried and reported "accepted
60/60 bytes" with no visible effect, which was read for months as "the command
works but something else re-lights it".

That reading was wrong, and this document records how that was established.

## The transport, and what "accepted" actually meant

Two doors reach the RK1608 coprocessor:

| Door | Mechanism | Status on this unit |
| --- | --- | --- |
| `/dev/v4l-subdev3` private ioctls | `_IOW('V', 196..204)`, handled by the kernel's `rk1608_ioctl` | Kernel handles them; **the DSP rejects them** |
| `/dev/rk_preisp` | two sub-doors onto the same mailbox — `ioctl(_IOW('p',11,4))` (unfiltered) and `echo s 0xNNN <v> >/dev/rk_preisp` (its parser filters ids; see "Three reply states, not two") | The DSP's real command space |

The 60/60-byte "acceptance" every earlier experiment relied on was **the USB
control transfer acknowledging that 60 bytes arrived**. It said nothing about
whether the coprocessor understood the message. The coprocessor's actual
opinion is emitted to the kernel log, which nobody had been reading:

```
rk1608 spi0.0: rk1608_disp_set_pro_current:0
rk1608 spi0.0: DSP(0): cam(0) unknown msg(0x1074)
```

The kernel driver handled `PREISP_DISP_SET_PRO_CURRENT` faithfully and
forwarded it. The Revopoint firmware answered `unknown msg`. The whole
`_IOW('V',196..204)` family is stock-Rockchip plumbing this firmware does not
implement.

## The negative control

Before trusting any return value, a command number no driver can implement was
issued: `_IOW('V', 210, 4)` = `0x400456d2`.

It returned **non-zero — rejected**. The driver therefore validates command
numbers, which means a `0` return from that node is a genuine accept and not a
switch-statement fall-through. Without this control, every "success" measured
on that node would have been uninterpretable.

## The finding that explains everything

**Every illumination command in the vendor SDK is unimplemented on this unit.**

| Command | SDK meaning | Coprocessor reply |
| --- | --- | --- |
| `0x922` | `setLaserOnOff` | `unknown msg(0x922)` |
| `0xb00` | IR LED mode | `unknown msg(0xb00)` |
| `0xb01` | IR LED enable | `unknown msg(0xb01)` |
| `0xb02` | RGB LED mode | `unknown msg(0xb02)` |
| `0xb03` | RGB LED enable | `unknown msg(0xb03)` |
| `0xb04` | laser LED enable | `unknown msg(0xb04)` |

The pinned host SDK (`lib3DCamera.so` 3.2.23) targets a **different firmware
build** than this scanner ships. Its illumination commands do not exist here.
Every negative optical result in this project's history is explained by that
one fact.

## Method: the coprocessor self-documents

An implemented command is answered by name and current value; an unimplemented
one is answered with its own id. Both go to the kernel log, so a `dmesg`
snapshot before and after a command is a complete, machine-readable ACK
channel that needs no human watching the emitter:

```
0x9a3  id_msg_cam_get_gain_reg        [24]
0x9b0  id_msg_cam_get_frame_time      [20000]
0x9b1  id_msg_cam_get_exposure_time   [20000]
0x9b2  id_msg_cam_get_auto_expo_mode  [0]
0x9c0  id_msg_cam_get_value_by_id     (generic; answers via a buffer, not the log)
```

## Three reply states, not two

A command id produces one of three outcomes, established by a control set run
with the log ring cleared before each probe so attribution is exact:

| Outcome | Example | Meaning |
| --- | --- | --- |
| **Named reply** | `0x9a3` -> `id_msg_cam_get_gain_reg [24]` | Sent, and the firmware implements it |
| **Rejected** | `0x921` -> `cam(0) unknown msg(0x921)` | Sent, firmware does not implement it |
| **Silent (text door)** | `0x700`, `0x484` -> nothing at all via `echo s ...` | The text door's parser drops the id before it reaches the coprocessor — **not** evidence of anything; both reply normally through the ioctl mailbox (below) |

The silent state is no longer mysterious, and it no longer means what an
earlier version of this document claimed. For a named or rejected id the
kernel first logs `rk1608 spi0.0: send msg len: 3, ret: 0`. For a silent id
issued through `echo s 0xNNN <v> >/dev/rk_preisp` that line is absent too —
but that turns out to be a property of the text door, not of the command id.

There are two doors onto the same `/dev/rk_preisp` mailbox:

| Door | Mechanism | Filtering |
| --- | --- | --- |
| Text | `echo s 0xNNN <v> >/dev/rk_preisp` | The kernel-side parser accepts only a subset of ids and silently drops the rest before they ever reach the coprocessor. This is what every earlier probe in this document used. |
| ioctl | `ioctl(fd, 0x4004700b, msg)`, i.e. `_IOW('p', 11, 4)`, where `msg` is a `u32[3]` of `{word_count, command_id, parameter}` | Not filtered. This is what the vendor's own `zxmain` binary uses to talk to the coprocessor. |

Re-probing through the ioctl mailbox turns up replies for ids that looked
silent through the text door. `0x700`, `0x701`, and `0x484` — all "silent"
under the old method — answer normally through ioctl:

```
0x700  exit=0   id_msg_algo_depth_t ========
0x701  exit=0   send 0 z16 depth end.[49]~ | send 1 z16 depth end.[50]~ | id_msg_algo_calibration_t ========
0x484  exit=0   zhi_proj_one_trigger ======= finish! | do_proc_r start== | (x3) | id_msg_proj_create_one_trigger_t ======== | zhi_proj_one_trigger ======= finish! | set_stop_flag stop trigger success
```

So: **silent means unreachable via the text channel, not unimplemented.** An
id that is silent under `echo s` has not been shown to be anything at all — it
must be re-probed through the ioctl mailbox before it can be called
unimplemented.

## Verified command map

2026-09-06/07 follow-up: firmware disassembly plus fresh device-side replies
verified two useful setters, **`0x910` frame time** and **`0x911` exposure**.
With both set to 20,000 → 15,000 → 10,000 → 7,000 µs, depth output rose from
approximately 4.5 → 6 → 9 → 13 fps. The original 20,000/20,000 values were
restored and verified. These are typed timing parameters, not flags to probe
with arbitrary zero writes. See [scanner-fps-investigation.md](scanner-fps-investigation.md)
for exact binary addresses, transport, A/B/A results and thermal testing.

**Confirmed implemented** — the only ids for which a named reply was actually
observed:

| Id | Name | Reply |
| --- | --- | --- |
| `0x9a3` | `id_msg_cam_get_gain_reg` | value read: 24 |
| `0x9b0` | `id_msg_cam_get_frame_time` | value read: 20000 |
| `0x9b1` | `id_msg_cam_get_exposure_time` | value read: 20000 |
| `0x9b2` | `id_msg_cam_get_auto_expo_mode` | value read: 0 |
| `0x9c0` | `id_msg_cam_get_value_by_id` | answers via a buffer, not the log |
| `0x910` | `id_msg_cam_set_frame_time` | fresh named reply, projector/sensor timing adjustment, getter verification, measured cadence change |
| `0x911` | `id_msg_cam_set_exposure_time` | fresh named reply, camera exposure adjustment and getter verification |
| `0x483` | `id_msg_proj_stop_trigger_t` | stops an active trigger (`set_stop_flag stop trigger success`); idempotent when no trigger is active — see "The projector namespace" |
| `0x484` | `id_msg_proj_create_one_trigger_t` | fires one projector trigger |
| `0x485` | `id_msg_proj_catchframe_t` | captures a fringe frame; export to `ImageL0.bmp` currently fails |
| `0x486` | `id_msg_proj_debug_t` | debug frame import/export |
| `0x700` | `id_msg_algo_depth_t` | depth pipeline command |
| `0x701` | `id_msg_algo_calibration_t` | calibration command; reply references z16 depth frames |

The camera rows above were confirmed through the text door; the projector and
`0x70x` rows were confirmed through the ioctl mailbox, after the text door
returned silence for them (see "Three reply states, not two").

**Confirmed rejected** — an explicit `unknown msg` naming the id:

- `0x921`, `0x922`, `0x9a0`–`0x9a2`, `0x9a4`, `0x9b3`–`0x9b5`, `0x9c1`–`0x9c4`
- `0xb00`–`0xb04` (the entire vendor SDK illumination set)
- the whole `_IOW('V',196..204)` subdev family, which answers `unknown msg(0x1074)`
- `0x487`, the id immediately above the projector's implemented block (see
  "The projector namespace")

**Retracted, and now partly re-confirmed.** An earlier batch sweep reported
`0x401`, `0x480`–`0x489`, `0x48d`–`0x48f`, `0x700`–`0x708`, `0x7a2` and
`0x900`–`0x904` as implemented. That classification was wrong and stays
withdrawn: it inferred "implemented" from the absence of an `unknown msg`
line, and two separate failure modes can produce that absence. One is the
log-window saturation described in "Where the method failed" below. The
other, established since, is that the text door silently drops ids its
parser does not recognize — so an id can go unanswered because the message
never left the host, not because the firmware rejected it. Absence of a
reply through the text door is not evidence of anything, in either
direction.

Re-probing through the unfiltered ioctl mailbox has since pulled several ids
out of that withdrawn range with real evidence: `0x483`–`0x486` are confirmed
implemented and `0x487` is confirmed rejected (folded into the tables above),
and `0x700`–`0x701` are confirmed implemented (folded in above too). The rest
of the withdrawn range — `0x401`, `0x480`–`0x482`, `0x488`–`0x489`,
`0x48d`–`0x48f`, `0x702`–`0x708`, `0x7a2`, `0x900`–`0x904` — remains
**unclassified** pending the same ioctl re-probe.

## Command names recovered

```
id_msg_algo_calibration_t          id_msg_algo_set_z_range
id_msg_algo_depth_t                id_msg_algo_set_resolution_t
id_msg_algo_get_resolution_t       id_msg_algo_set_output_infrared
id_msg_algo_set_image_num_t        id_msg_algo_set_save_tmp
id_msg_algo_set_background_threshold
id_msg_algo_set_gradient_threshold

id_msg_cam_get_gain_reg            id_msg_cam_set_gain_reg
id_msg_cam_get_frame_time          id_msg_cam_set_exposure_reg
id_msg_cam_get_exposure_time       id_msg_cam_set_vts_reg
id_msg_cam_get_auto_expo_mode      id_msg_cam_set_hts_reg
id_msg_cam_get_value_by_id         id_msg_cam_set_fake_mode

zhi_proj_one_trigger               id_msg_proj_stop_trigger_t
id_msg_proj_create_one_trigger_t   id_msg_proj_catchframe_t
id_msg_proj_debug_t                (projector namespace)
```

These names are real: the firmware emitted every one of them. But most were
observed during the flawed batch sweep, where they cannot be attributed to a
specific command id, and some are simply the firmware narrating its own
background work. Treat this as a vocabulary of the API's namespaces —
`id_msg_algo_*`, `id_msg_cam_*`, `id_msg_proj_*`, `zhi_proj_*` — rather than a
command table. Only the ids in the verified command map above have a name
bound to an id by evidence; the rest of this list is namespace vocabulary,
not a command table.

## The projector namespace, from the boot log

A fresh boot log names the sequence that extinguishes the emitter:

```
t=7.998   id_msg_proj_create_trigger_t ========
t=8.450   zhi_proj_one_trigger ======= finish!    (x5)
t=10.451  id_msg_proj_stop_trigger_t===
t=10.888  zhi_proj_light_off light off
```

So a third message family exists, `id_msg_proj_*`, and
`id_msg_proj_stop_trigger_t` is followed 0.44 s later by
`zhi_proj_light_off light off` — the firmware turning the projector off, on
this unit, unprompted. Also present: `zhi_proj_set_type type = 6 ZX840` and
`set project framerate=0x1f success`.

The command ids are now known. Re-probing `0x483`–`0x487` through the ioctl
mailbox produced:

```
0x483  exit=0   id_msg_proj_stop_trigger_t=== | set_stop_flag stop trigger success | zhi_proj_light_off light off
0x484  exit=0   id_msg_proj_create_one_trigger_t ========
0x485  exit=0   id_msg_proj_catchframe_t ======== | zhi_proj_one_trigger ======= finish! | do_proc_r start== | saveFringeImage1 width=640 height=400 | xfile export addr:0x60deb1b8 size:257080 to //tmp/webroot//data/ImageL0.bmp failed
0x486  exit=0   id_msg_proj_debug_t ======== | xfile import /tmp/webroot/data/imageL0.bmp ... failed | zhi_proj_one_trigger ======= finish! | send 0 z16 depth end.[51]~
0x487  exit=0   cam(0) unknown msg(0x487)
```

So the block is `0x483`–`0x486` implemented, and `0x487` is the first
unimplemented id above it. (`0x485`'s reply shows the firmware exporting
fringe images to `//tmp/webroot//data/ImageL0.bmp` — into the nginx document
root — though the export itself reported "failed", most likely a missing
directory, so the export path is not claimed to work.)

`0x483` is idempotent in a way that matters: the `zhi_proj_light_off light
off` line only appears when a trigger is actually active. The decisive test
was run — a normal depth capture, to leave the emitter in its usual lit idle
state, followed by firing `0x483` — and the emitter was visually confirmed
still lit afterward. `0x483` stops an active trigger; it does not extinguish
the standing dim idle glow. That glow is therefore driven by something other
than an active projector trigger, and what that something is remains open
(see "Open" below).

## Where the method failed, and why

The batch sweep inferred "implemented" from the **absence** of an
`unknown msg` line within a 90-line `dmesg tail`. That inference is only valid
while the log window still contains the relevant lines.

From roughly `0x910` onward, something began writing to the kernel log every
frame — `id_msg_cam_get_gain_reg` appears 34 times in the record against a
handful actually sent. The window saturated, the `unknown msg` lines fell off
the end, and **every id from `0x910` up was falsely reported as implemented**.

Those results are discarded. Only blocks where at least one id came back
unknown are retained above, because that proves the window was intact.

The fix for any future sweep is to clear the ring (`dmesg -c`) before each
batch, so the window can only contain that batch's output and absence becomes
real evidence. That was recommended in the plan and not adopted; it should have
been.

## Cost

The sweep sent value `0` to ids in `0x910`–`0x91f`, which is where the SDK's
setters live (frame time, exposure, fringe pattern, ROI). Afterwards depth
capture began timing out — no buffer within 8 seconds, against a stable ~34%
yield all session — while USB stayed healthy with all interfaces bound. The
most likely cause is a zeroed exposure or frame time, or `set_fake_mode`.

This is coprocessor runtime state. `preisp.rkl` reloads at boot, so a power
cycle restores it. No persistent change was made and nothing was flashed.

The sweep was deliberately bounded to `0x4xx`, `0x7xx`, `0x9xx`, `0xaxx` and
`0xbxx` — the blocks `zxmain` and the SDK actually use. Unmapped space was left
alone because a blind sweep there could reach a flash or update verb, which is
the one failure a power cycle would not undo.

## Open

The central question has changed shape. `0x483` (`id_msg_proj_stop_trigger_t`)
demonstrably stops an active projector trigger, but firing it against the
unit's normal lit idle state left the emitter visually unchanged — still lit.
So the standing idle glow is not an active trigger, and stopping triggers does
not touch it. What actually drives the idle glow is now the open problem.
Candidates, untested:

- a separate, still-unmapped `id_msg_proj_*` / `zhi_proj_*` command — a
  "light on" counterpart distinct from the trigger lifecycle,
- state the coprocessor enters automatically on boot or after the last
  capture, independent of any single command, or
- behavior below the coprocessor's command interface entirely.

No command has been found that turns the idle glow off. The ids withdrawn in
"Retracted" and not yet re-confirmed — `0x401`, `0x480`–`0x482`,
`0x488`–`0x489`, `0x48d`–`0x48f`, `0x702`–`0x708`, `0x7a2`, `0x900`–`0x904` —
still need the same ioctl-mailbox re-probe applied to `0x483`–`0x486` and
`0x700`–`0x701`, since silence through the text door is no longer evidence of
anything. The `0x910`–`0x9ff` range still needs a proper sweep with a cleared
log ring, and `zhi_proj_*` remains the most promising name prefix to hunt.
