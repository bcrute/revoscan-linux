# Blue-light final-state shutdown — 2026-08-23

This was a separate owner-required final-state action after scanner-026 was
accepted. It was not a scanner-026 implementation invocation.

## Preflight

- Target: `2207:110c`, serial `A22704522A6F00A64`.
- Stable location: sysfs `1-1.1`, bus 1, devnum 47, usbfs
  `/dev/bus/usb/001/047`.
- The current user had read/write usbfs access without elevation.
- Interfaces 0–3 were bound to `uvcvideo`; interface 4 was bound to `usbhid`.
- `/dev/video0` through `/dev/video3` were present.
- No holder was found on the usbfs or video nodes.

## One exact off attempt

- Exactly one immutable host-to-device `SET_CUR` was issued; there was no
  retry, loop, reset, firmware action, privilege path or other scanner command.
- Setup: `21 01 00 07 00 04 3c 00 e8 03 00 00`.
- Payload: ASCII `echo s 0x922 0 >/dev/rk_preisp`, NUL-padded to 60 bytes.
- Payload SHA-256:
  `6030364daddcc4c9855d16da552c58161bbec5cf77ea23ace9de2f14839ffb0d`.
- Result: `complete`; accepted 60/60 bytes in 1.548 ms.
- Interface release succeeded; reconnect succeeded in 20.808 ms; cleanup was
  complete; no signal was deferred; exit code was 0.

## Postflight

- Three samples over two seconds retained sysfs `1-1.1` and devnum 47.
- Interfaces 0–3 were rebound to `uvcvideo`; HID remained bound to `usbhid`.
- `/dev/video0` through `/dev/video3` were present. The interface-0 depth-node
  inodes changed from 3412/3413 to 3437/3438 during detach/rebind; RGB-node
  inodes 3368/3369 remained unchanged.
- No usbfs holder remained.

This proves acceptance of the exact off write and clean host-side restoration.
The software has no optical readback, so it does not independently prove that
an optical sensor observed the blue light to be dark.

## Offline actuator and firmware audit

The pinned SDK exposes two distinct illumination families.  The already tried
`0x922=0` and `0xb04=0` paths are its laser controls.  Its separately documented
generic LED setter builds exact command
`echo s 0xb00 0 > /dev/rk_preisp`, NUL-padded to 60 bytes (SHA-256
`3294b9341a308de8d49f3bc63882c0655aedff7efe2c35fdb5f15ffa4fa5a79a`),
and sends it through the same entity-4 UVC SET_CUR.  No prior live `0xb00`
record was found.  This is the strongest next reversible host-side candidate;
its mapping to the MINI's visible blue emitter remains unproven until the owner
observes the result.  The lower-level `0xb01=0` IR-LED disable gate is a separate
later candidate and must not be combined with the first test.

The recovered 166,412-byte `zxmain` ARM ELF independently contains a real
device-internal `preisp_set_led_on_off(int)` routine at VA `0x1f21c`.  It issues
`ioctl(fd, 0x400456cc, &value)` on `/dev/v4l-subdev3`; exact call sites pass 0
at `0x20250` and 1 at `0x20270` from a proximity-sensor control loop.  The
ordinary error/reboot path separately runs `echo off >/dev/rk_preisp`, but no
ordinary STREAMOFF or idle cleanup was proven.  A better embedded policy can
therefore use the existing internal off actuator: default off, enable only
under a bounded depth-stream lease, and force off on STREAMOFF, disconnect,
error and lease expiry.  Firmware replacement remains staged behind raw backup
and loader/maskrom recovery proof rather than a blind flash.

## Later owner observation after detached collection

After scanner-051 completed on 2026-08-23, the exact device remained devnum 50
with depth interfaces 0/1 unbound, interfaces 2/3 owned by `uvcvideo`, and
interface 4 owned by `usbhid`. The owner then directly reported that the blue
front emitter was still on. This establishes that leaving the depth UVC
function detached does not itself extinguish the emitter. It also confirms the
earlier logical-off, GPIO117, hub-port and RK1608-decrement paths have not yet
produced an owner-observed shutdown. It does not identify the actual power rail
or authorize a blind third decrement.

The owner clarified that the operational priority is avoiding needless emitter
and scanner wear, rather than an acute exposure concern: the light can be
observed indirectly without looking into it.  The desired invariant is still
strict—emitter off whenever the scanner is idle—and the next intervention
remains the isolated SDK-generic `0xb00=0` write before any firmware mutation.

## Generic LED setter first live gate

Scanner-054 invoked the reviewed generic-LED helper once at
`/tmp/scanner-053-evidence.ARQJtR` (manifest SHA-256
`24885867efd31c3ac6e479db8dd73c309af38c530425a4ceffdacf4a8a996990`).
The device was bus 1/devnum 50, rdev `189:49`. Claim succeeded, but live sysfs
represented the held interface as `driver=usbfs`; the helper's then-current
guard expected it to remain unbound and stopped.  `control_submissions` is zero,
the event head is all zeroes, and release/close/final identity/topology passed.
Thus the `0xb00=0` actuator remains physically untried.  Scanner-055 is the
small offline correction for the now-observed claimed topology.

Scanner-056 then invoked the corrected helper exactly once at
`/tmp/scanner-055-evidence.ALVrdn` (manifest SHA-256
`4a197ad6b99e77f57212393535b000c9fe345311dd5cbba697bfd01c2877f084`).
The exact SDK-generic `0xb00=0` command was submitted once and accepted 60/60
bytes. Release/close, final detached topology and identity passed. This is the
first physical test of b00 and proves command acceptance only; owner-observed
emitter state was initially pending. The owner then reported that the front
blue emitter was still on, so `0xb00=0` is now a proven negative optical result
for this unit despite exact 60/60 command acceptance.

Scanner-058 then tested the SDK's distinct lower-level IR/LED disable gate at
`/tmp/scanner-057-evidence.FmEdsN` (manifest SHA-256
`75e08b67182937cc3aef54cbd4e28f6e62b33cbe2aa2c50c32078f211271abf6`).
The exact `b01=0` command was submitted once and accepted 60/60 bytes; cleanup,
detached final topology and identity passed. Owner-observed darkness is pending.
The owner then reported that the front blue emitter was still on. Thus b01=0 is
also a proven negative optical result despite exact command acceptance. With
b00 and b01 exhausted, the next reversible layer is a RAM-only direct call to
the firmware's recovered `/dev/v4l-subdev3` off ioctl before any flash.
