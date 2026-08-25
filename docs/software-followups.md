# Software follow-ups — work to close out after the emitter investigation

A running list so the software side can be buttoned up quickly once the
on-device work settles. Recorded 2026-08-24. Nothing here is started.

## 1. Product code currently makes a claim it cannot support (highest priority)

`src/scanner3d/illumination_shutdown.py` and the finalizer wired into
`v4l2_capture.py` report, after every depth capture:

```
Blue-light shutdown (fixed one-write finalizer)
  required=True attempted=True accepted=True restored=True
  status=complete - the fixed blue-light off command was accepted
```

and `IlluminationShutdownOutcome.succeeded` gates the whole capture's exit code.
The command it sends is `echo s 0x922 0 >/dev/rk_preisp`.

We now have direct evidence that this command **does not extinguish the
emitter**. It was tried live and the owner observed the projector still lit;
the same is true of `0xb01 0` and `0xb04 0`. Two depth captures on 2026-08-24
each reported `accepted=True restored=True` while the projector remained
visibly lit afterwards.

The wording is technically defensible — it says the transfer was *accepted*,
not that the light went out — but a reader of a "Blue-light shutdown: complete"
line will reasonably infer a safety outcome that did not occur, and the run's
success depends on it. Required changes:

- Rename the reported status so it describes the transfer, not the outcome
  (for example `command_accepted`, never `complete`).
- Add an explicit, permanent limitation string to the outcome stating that the
  repository has no optical readback and that this command is **known** not to
  darken this unit's emitter.
- Reconsider whether a command proven ineffective should gate run success at
  all. Failing a capture because an ineffective command failed is noise;
  succeeding because it was accepted is worse.
- Update `docs/roadmap.md` and the `goal.md` durable constraint that requires
  the finalizer, since that constraint currently mandates a command we know is
  inert on this hardware.

## 2. Depth-capture re-enumeration is reproducible and should be modelled

Every `depth-frame` capture is followed roughly one second later by a USB
disconnect and re-enumeration (devnum 51→52 at 05:01, 52→53 at 05:05, matching
the historical 44→45, 45→46, 46→47 transitions previously recorded as
"unexplained"). It is a gadget-level reset, not a device reboot: a reboot
returns the emitter to its dark post-boot state and these did not.

The capture path should expect this rather than treat it as an anomaly, and any
future multi-capture session engine must re-resolve the device by serial
between captures instead of caching a devnum.

## 3. Phase 2 integration still open

`src/scanner3d/session.py` (the versioned session manifest and atomic artifact
store) landed with 110 tests, but the third Phase 2 exit-gate clause is
deliberately unfinished: the existing CLI capture path does not yet emit the new
manifest. That change touches hardware-facing code and needs a live run to
verify, so it was left for a separately reviewed task.

## 4. scanner-060 is still mid-review

The round-1 helper (`d591f313…`) and its test (`c2532f6c…`) are awaiting root
review. Given the finding that its `busybox --list` command was already answered
by frozen scanner-039 evidence, the planner should decide whether scanner-061
should still spend a live invocation on the inventory as written, or be
re-scoped now that the target has moved to on-device execution.

## 5. Closed routes — do not re-spend effort

- Host-side vendor illumination commands: exhausted, every
  `echo s 0x… >/dev/rk_preisp` template resolved to its enclosing SDK function.
- The proximity interlock: `iio:device1` does not exist on this unit, so
  `psensor_monitor_thread` polls a nonexistent path and can never fire.
- I2C and `devmem`: the projector is not a host-SoC peripheral; dmesg confirms
  `No GPIO consumer pwren found` for the RK1608.
- Status LEDs: `led-blue-booting` reads 0 while the emitter is lit.
- Hub per-port power: the Realtek hub advertises per-port power switching but
  does not implement it; the port reported unpowered while the emitter stayed
  lit.

Details and evidence in `docs/illumination-actuators.md`.

## 6. Backup gap — required before any actual flash

**No device firmware has been backed up.** What exists is `zxmain.bin`
(166,412 bytes, SHA-256 `decdc3ad…`), the vendor *userspace service*, recovered
in scanner-039. Not captured, and not currently locatable:

- `preisp.rkl`, the RK1608 coprocessor firmware. It is not at `/tmp`
  (scanner-039 confirmed absence there) and loads successfully at boot from an
  unknown path, version string `rkceva4k2022042615193025`.
- Bootloader, kernel, device tree, rootfs, and any MTD/MMC block image.

This does **not** block the current on-device helper work, which writes only to
`/tmp` (tmpfs, RAM) on a read-only root filesystem and changes nothing
persistent. It absolutely does block any real firmware modification, which the
goal already stages behind recovering the boot/update/recovery chain.

If a flash is ever contemplated, the prerequisite sequence is: locate the
partitions, read each out through the proven chunk protocol, verify by hash,
and confirm a maskrom or loader recovery path exists — before writing anything.
Reading a multi-megabyte image at 56 bytes per transfer is on the order of tens
of thousands of transfers, so it needs its own bounded, resumable design.

## 7. Repository baseline

Commit authority is still not granted. `main` is unborn and every source file
is untracked, so none of this work is recoverable through version control yet.
This remains the single largest provenance risk to the project.
