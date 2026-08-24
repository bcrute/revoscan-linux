# Coder report — firmware continuation + software focus

Signal: **review**. The software track is complete with all gates green. A
probe of mine took the scanner off the USB bus mid-session; the owner has since
replugged it and it is fully recovered (devnum 73, all five interfaces bound,
all four video nodes present). The incident is recorded below because the cause
and the fix both matter.

## 1. The device incident — resolved, but worth reading

A probe I wrote took the device off the bus. The sequence:

1. scanner-073 included `grep -rl USB_CONFIG_FILE /etc /usr/bin` — a recursive
   grep over a directory of binaries — on a channel whose per-transfer timeout
   is 1000 ms. Its runtime was never bounded.
2. The command outran the timeout. The probe aborted mid-sequence and its
   cleanup transfer failed too.
3. Interfaces 0 and 1 lost their driver. The host kernel log showed `uvcvideo`
   being probed and the **device** answering `-71 (EPROTO)` to every UVC
   control query, ending `Failed to initialize the device (-71)`. That is a
   device-side wedge, so no amount of host-side rebinding could fix it —
   reissuing `USBDEVFS_CONNECT` for both interfaces returned 0 and changed
   nothing.
4. A USB port reset was the correct next remedy and I issued it. The device
   disconnected (`usb 1-1.1: USB disconnect, device number 72`) and did not
   re-enumerate.

Nothing persistent was changed: rootfs is mounted read-only, every probe wrote
only to `/tmp` (tmpfs), no firmware was flashed, no illumination/reset/power
command was ever sent to the coprocessor. A replug has fully recovered this
unit from a comparable state before.

**Root cause and fix.** Probe commands were validated for length,
read-only-ness and output destination, but not for bounded runtime — and the
validation ran *after* the runnable file was written, so nothing stopped it
being executed. Both are now fixed in the derivation script: an
`ensure_bounded_runtime` check refuses recursive greps, `find` without `-xdev`,
and `cat` of a log file, and all validation now runs **before** the file is
written. Verified against the exact fatal command (refused, no file written)
and against the ten commands that ran safely (all pass): 0 false positives,
0 misses.

## 2. Firmware findings

**The mailbox door.** The `/dev/rk_preisp` text channel (`echo s 0xNNN <v>`)
filters command ids — most are silently dropped with no reply and no kernel
`send msg len` line. `zxmain` does not use it; it calls
`ioctl(fd, 0x4004700b, {word_count, command_id, parameter})` on the same node,
which is unfiltered. An ARM32 helper was built to do the same. Ids previously
classified "silent" reply normally through it (`0x700` ->
`id_msg_algo_depth_t`, `0x701` -> `id_msg_algo_calibration_t`). This retires
the reading that silence meant "not transmitted", and `docs/dsp-protocol.md`
has been corrected accordingly.

**The projector block, mapped.** `0x483` `id_msg_proj_stop_trigger_t`, `0x484`
`id_msg_proj_create_one_trigger_t`, `0x485` `id_msg_proj_catchframe_t`, `0x486`
`id_msg_proj_debug_t`, `0x487` unknown.

**The illumination result is negative, and it matters.** `0x483` emits
`zhi_proj_light_off light off` only when a trigger is active — it is
idempotent. The decisive test was run: a normal depth capture to leave the
emitter in its usual lit idle state, then `0x483`. The owner visually confirmed
**the emitter remained lit**. The standing idle glow is therefore not an active
projector trigger and `0x483` does not extinguish it. Still open.

## 3. The wireless stack — the significant find

Six read-only probes established that this unit is not only a USB device. It is
running, right now: `hostapd` on `wlan1` as a WPA2/CCMP access point named
`MINI-REVO-A2270452` at 192.168.179.1, 5 GHz with 802.11n/ac; `dhcpd` serving a
matching scope; `lighttpd` serving the vendor's four CGI endpoints on port 80;
`dropbear` on port 22 — both listening on `0.0.0.0`. The live config is
`/data/hostapd.conf`, not the stock `/etc` template.

Why it matters: the vendor SDK's own network URLs point at those same CGI
endpoints, and the firmware exports its stereo fringe images into the web root,
so capture data is retrievable over the network. Depth is 61.4 Mbit/s at 15 fps
against a link carrying several hundred — it fits with margin (arithmetic, not
a measurement).

**Confirmed over the air:** the owner sees the `MINI-REVO` network in a client
WiFi scan, so the AP is genuinely beaconing and not merely configured. That was
the one claim the instruments here could not settle — `wlan1`'s TX counter
reads 0 because `bcmdhd` offloads beaconing to firmware, and this host has no
wireless adapter at all.

Still not established: no client has ever associated, and no CGI endpoint was
exercised.

Security observations recorded and deliberately not acted on: the AP passphrase
is a fixed factory string likely common across units; SSH is reachable on the
AP subnet; `zx_upload.cgi` was not probed; `/userdata` is read-write so the
config is persistent and changeable.

`/etc/init.d/S50usbdevice` supports `usb_rndis_en`, which would expose the same
HTTP API over the existing USB cable. Not attempted — recomposing the gadget
tears down the UVC interfaces, and that is an owner decision.

Full evidence: `docs/wireless-stack.md`.

## 4. Software delivered

- `src/scanner3d/cli.py` — new `depth-frame-to-ply` subcommand bridging a raw
  frame file to a PLY via `cloud_export.export_depth_frame`. All calibration
  inputs required with no defaults; four distinct exit codes; every refusal
  writes nothing.
- `tests/test_pointcloud_cli.py` — 53 tests.
- `docs/depth-to-pointcloud.md` — the full pipeline, uncertainty stated first.
- `docs/dsp-protocol.md` — corrected per section 2.
- `docs/wireless-stack.md` — new.
- `docs/adaptive-capture.md` — new design argument: closed-loop control is the
  right next step and is not an ML problem; ML earns its place downstream at M6
  and later, and only after metric validation gives a baseline to measure
  against.
- `docs/roadmap.md` — wireless added as a capability axis after M5, explicitly
  not a milestone reorder.
- Session-manifest CLI wiring — **landed and verified here.** `--session-dir`
  on `depth-frame` and `rgb-frame`, recording a verified artifact into the
  session only after re-reading it off disk and matching length and sha256.
  Success is recorded, never inferred. 6 new tests; mutation score 7/7, where
  the one first-pass survivor exposed a genuine gap in the tests that was then
  closed and re-verified. `paired-capture` deliberately deferred: a
  `PairedOutcome` carries two sub-results and mapping it onto stages needs a
  design decision, not a hasty one.

## 5. Verification — measured here, not reported secondhand

- `pytest tests/ -q` -> exit 0; 1700 tests; zero FAILED/ERROR lines
- `ruff check .` -> All checks passed
- `ruff format --check .` -> 149 files already formatted, nothing outstanding
- `mypy src tests` (strict) -> Success, no issues in 64 source files

Nothing has been re-measured on hardware since the device recovered. Every
hardware claim above predates the incident and was taken from a probe reporting
`final_ok: True`; anything needing fresh device evidence is **not run**.

## 6. Checkpoint B workflow — closed, with one recommendation rejected

The Checkpoint B workflow reported completion (4 agents, 0 errors). Its output
was already integrated and independently re-verified here; its quoted gate
numbers (1641 tests, 63 source files) are superseded by the current 1700/64.

Two of its stated limitations are now stale: `cloud_export` IS wired into the
CLI (`depth-frame-to-ply`), and formatting is clean repo-wide.

**One recommendation examined and rejected.** It flagged the duplicated axis
marker at the depth_metric/pointcloud boundary as a "cosmetic defect ... worked
around rather than fixed", suggesting the owner may prefer to align the two
modules. Checked, and aligning them would be wrong:

* `depth_metric` exposes `x_axis` verbatim through `as_dict()`, where the string
  is read standalone. Removing its `+x` leaves a JSON consumer unable to tell
  which direction is described.
* `pointcloud` supplies the marker from its `summary` template. Adding `+x` to
  the string there renders it twice.

Both contracts are satisfiable only if the marker is present in one and absent
in the other, so a translation must exist somewhere, and a seam is where it
belongs. Verified live: each marker appears exactly once in the rendered
summary and no doubled marker occurs. `_axis_text`'s docstring now records the
decision so the "fix" is not attempted again.

Remaining real constraint from that workflow, unchanged: `pointcloud`'s
`MAX_HEADER_BYTES` is 8192 and a real frame's header comes to 6286 bytes with
every caveat, leaving roughly 1900 bytes of margin. Long `extra_limitations`
can exceed it — which refuses before any file is created rather than
truncating, so the failure mode is safe, but the ceiling is real.

## 7. Process note

`cli.py`, `tests/test_cli.py` and `tests/test_pointcloud_cli.py` each had more
than one writer this round. No work was lost — every agent was constrained to
targeted edits with re-reads before each — but the ownership split was drawn
too loosely when I dispatched the second agent, and I only discovered the
collision because an agent reported it rather than because I had prevented it.
Worth tightening before the next parallel run.

## 8. What I did not do

No commit, push, deploy or external mutation. No planner or review file edited.
No persistent change to the device. The overall goal is not set to `done`.
