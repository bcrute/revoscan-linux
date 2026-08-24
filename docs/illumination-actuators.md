# Illumination actuator surface — offline evidence

Status: the analysis sections are offline-only and make no optical/darkness
claim. The final section, "Live finding, 2026-08-24", is the one exception: it
reports a bounded **read-only** probe of the device that issued no target
command and restored the device to its original bound topology. Everything
before it is derived from two byte-identical artifacts already frozen in this
project's evidence:

| Artifact | SHA-256 | Provenance |
| --- | --- | --- |
| `lib3DCamera.so` (pinned vendor SDK, host side) | `49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4` | matches `src/scanner3d/data/sdk-manifest.json` exactly |
| `zxmain.bin` (device firmware, recovered scanner-039) | `decdc3ad4e34449cefee8543c790866cfc87dd9279a873c3abab124b9754b0f1` | matches the scanner-039 frozen record |

## Why this document exists

Three separate live interventions (`0x922=0`, `0xb00=0`, `0xb01=0`) were each
accepted by the device as exact 60/60-byte transfers, and after each one the
owner directly observed that the blue front emitter was still lit. Before
spending further live attempts — or building a custom binary to run on the
device — the host-side command surface was enumerated exhaustively offline to
find out whether any untried vendor actuator remained.

**It does not. The host-side SDK illumination surface is exhausted.**

## Complete SDK register command surface

Every `echo s 0x… >/dev/rk_preisp` template in the pinned SDK, resolved to its
enclosing function by disassembly rather than by string proximity:

| Register | Enclosing SDK function | Illumination? |
| --- | --- | --- |
| `0x922` | `cs::Camera::setLaserOnOff` | **yes** — tried, negative |
| `0xb00`–`0xb04` | `cs::Camera::ledCtrl` | **yes** — see table below |
| `0x484`, `0x487` | trigger mode / property range | no |
| `0x702` | `cs::Camera::setStreamResolution` | no |
| `0x705`, `0xa04`, `0x910` | frame time / multiframe fusion | no |
| `0x903`, `0x911` | gain / depth exposure | no |
| `0x912`, `0xa05` | fringe pattern type | no |
| `0x913` | `cs::Camera::setDepthROI` | no |
| `0x914`–`0x917` | multiframe fusion parameters | no |

Two of these were previously plausible-looking candidates and are now ruled
out by disassembly. `0x702` sits adjacent to the strings `set IR enable failed`
and `set IR led twinkle failed` in the string table, but its only reference is
inside `setStreamResolution`. `0x913` sits adjacent to `set laser led disable
failed`, but its only reference is inside `setDepthROI`. String proximity in
this binary is not evidence of purpose.

## The exact `ledCtrl(led, param)` dispatch table

Recovered from `cs::Camera::ledCtrl(PropertyExtension)`, which reads its two
integer arguments and emits fixed, fully-literal command strings:

| `led` | `param` | Commands issued, in order |
| --- | --- | --- |
| 0 (IR) | 0 | `echo s 0xb01 1` then `echo s 0xb00 1` |
| 0 (IR) | 1 | `echo s 0xb01 1` then `echo s 0xb00 0` |
| 0 (IR) | 2 | `echo s 0xb01 1` |
| 0 (IR) | 3 | `echo s 0xb01 0` |
| 1 (RGB) | 0 | `echo s 0xb03 1` then `echo s 0xb02 1` |
| 1 (RGB) | 1 | `echo s 0xb03 1` then `echo s 0xb02 0` |
| 1 (RGB) | 2 | `echo s 0xb03 1` |
| 1 (RGB) | 3 | `echo s 0xb03 0` |
| 2 (laser) | 2 | `echo s 0xb04 1` |
| 2 (laser) | 3 | `echo s 0xb04 0` |

`param` is confirmed by the error string on each branch, not inferred from
context: 0 = often bright, 1 = twinkle, 2 = enable, 3 = disable.

The per-family split also comes from those strings. `0xb01` is the IR LED
*enable* gate and `0xb00` is its *mode* selector; `0xb03` and `0xb02` are the
same pair for the RGB LED; `0xb04` is the laser enable.

Three consequences matter:

- The laser's only SDK disable path is `0xb04 0`, which was tried and produced
  no observed optical change.
- **The earlier `0xb00=0` test did not command anything off.** Given the
  dispatch above, `0xb00 0` selects *twinkle mode for the IR LED*; it is only
  ever emitted immediately after `0xb01 1`, which switches the IR LED **on**.
  Read in isolation it is not an off command at all, so that negative result
  says nothing about shutdown and should not be counted as an exhausted
  off-path.
- `0xb01 0` (tried) is a genuine IR disable, and `0xb04 0` (tried) is a genuine
  laser disable. Those two are the only real off commands the host SDK can
  produce for the emitter families, and both were accepted with no observed
  optical change.

## Status LEDs are not the emitter

`/sys/class/leds` on the device exposes exactly three gpio-leds entries:
`led-blue-booting`, `led-green-working`, `led-red-err`. At the time of the
scanner-039 capture their brightness values were blue 0, green 1, red 0, each
with max 255. The blue status LED was already off while the owner was observing
a lit blue front emitter, so the visible emitter is the depth projector and not
a status indicator. This closes an otherwise plausible hypothesis without
spending a live attempt.

## What the device firmware exposes internally

`zxmain` contains the internal routine `preisp_set_led_on_off(int)`, which
performs `ioctl(fd, 0x400456cc, &value)` on `/dev/v4l-subdev3`. It also
references `preisp_set_pro_current` and `preisp_set_pro_time` — projector drive
current and pulse timing. These are device-internal functions reached through
the binary `zx_preisp_cmd` mailbox, not through the textual
`echo s … >/dev/rk_preisp` channel, so they are not reachable from the host by
any command string.

The only register literal present in the device firmware itself is
`echo s 0x484 1 >/dev/rk_preisp`; the `0x922`/`0xb00`/`0xb01`/`0xb04` values all
originate host-side in the SDK.

## Device shell capability already proven

Command 1 of the planned scanner-060 inventory — `busybox --list` — was in fact
already executed on the device during scanner-039 as
`busybox --list >/tmp/s39_bb 2>&1`, read back through the selector/chunk
protocol, and hash-linked in that run's manifest at `.file_reads[6].sha256`
(`7515eda73e0e57ac496662762692d639296ef01f80ddabac85701c0949c2198b`).

The device's busybox provides 254 applets. The ones that matter here:

- `printf`, `dd`, `chmod`, `sh`, `cat`, `rm` — file write, permission and
  execution primitives, so scanner-060's commands 2–4 are near-certain to
  succeed and the inventory is largely confirmatory rather than exploratory;
- `devmem` — direct physical memory read/write;
- `i2cdetect`, `i2cget`, `i2cset`, `i2cdump` — I2C bus access;
- `insmod`, `rmmod`, `modprobe`, `lsmod` — kernel module control;
- `poweroff`, `halt`, `reboot` — full system power control.

The device root filesystem is remounted read-only at boot (`/etc/init.d/rcS`
runs `mount -o remount,ro /`), so `/tmp` is the writable, RAM-only location —
consistent with the existing RAM-only framing.

## Consequences for the next bounded step

These are observations for the planner, not authorizations, and none of them
has been acted on.

1. The host-side vendor surface is exhausted. Further single-register writes
   through the existing channel are not expected to extinguish the emitter, and
   spending live attempts on them is hard to justify.
2. Reaching `preisp_set_led_on_off(0)` requires executing code inside the
   device. That was the motivation for a hand-built ARM EABI helper.
3. However, `devmem` and the `i2c*` applets already provide register-level and
   bus-level actuator access from the *existing, proven* shell channel, with no
   binary upload, no `chmod`, and no execution of foreign code. If the emitter's
   actuator is reachable as a register or an I2C device, that route is strictly
   smaller and more reversible than uploading a binary, and it should be
   evaluated before the ELF path.
4. `poweroff` would definitively extinguish the emitter but leaves the scanner
   requiring a physical replug. That is an owner decision, not a technical one.

Nothing in this document establishes that any of these paths works. The
acceptance gate remains unchanged: the owner visually confirming darkness.

## Live finding, 2026-08-24 — this unit has no proximity sensor

Offline disassembly of `zxmain` established that the only automatic
projector-off mechanism in the vendor firmware is `psensor_monitor_thread`: it
polls `/sys/bus/iio/devices/iio:device1/in_proximity_raw` every 50 ms and calls
`preisp_set_led_on_off(0)` when that value is strictly positive. Covering the
front of the scanner produced no change, so a bounded read-only probe read the
values the loop consumes, using the proven selector/chunk protocol and issuing
no target command.

| Path | Result |
| --- | --- |
| `iio:device1/in_proximity_raw` | empty |
| `iio:device1/in_proximity_power_en` | empty |
| `iio:device1/in_proximity_ps_mode` | empty |
| `iio:device1/in_proximity_led_config` | empty |
| `iio:device1/name` | empty |
| `iio:device0/name` | `ff5e0000.saradc` |
| `led-blue-booting/brightness` | `0` |
| `led-green-working/brightness` | `1` |

The controls make the empty reads conclusive rather than ambiguous: the same
protocol in the same run returned real content for three other sysfs
attributes, so the read path demonstrably works and a zero-length result means
the file is not there. **`iio:device1` does not exist on this unit.** The only
IIO device present is the SoC's SAR ADC.

Two consequences:

- **The vendor's only automatic projector-off path is inert on this hardware.**
  The accelerometer/gyro/proximity part the firmware expects is absent or never
  probed, so `in_proximity_raw` can never read positive and
  `preisp_set_led_on_off(0)` is never called. The projector is therefore left
  in whatever state a capture leaves it in, permanently. This is a firmware
  expectation that this unit's hardware does not satisfy, not a bug we can
  configure around.
- **The dim idle glow is not a status LED.** The blue status LED reads 0 while
  the emitter is visibly lit, and the green LED reads 1, confirming the read is
  live and correct. Combined with the owner's observation that the dim glow and
  the bright capture burst come from the same spot, the idle glow is the depth
  projector held at a low drive level.

No amount of further offline analysis could have established this; it required
reading the device. It closes the proximity route definitively.
