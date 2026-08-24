state: review
task: firmware/illumination continuation + software focus
round: 1
activity: DSP mailbox door found and the projector command block mapped; the scanner's undocumented wireless stack discovered, measured, documented, and confirmed beaconing over the air by the owner; depth-frame-to-ply CLI and session-manifest capture wiring both landed with full gates green. A probe of mine took the device off the USB bus mid-session; the owner replugged and it is fully recovered.

## Device incident — RESOLVED by owner replug

The scanner was taken off the USB bus mid-session and required a physical
unplug/replug. The owner has since replugged it and it is FULLY RECOVERED:
devnum 73, all five interfaces bound (0-3 uvcvideo, 4 usbhid), all four video
nodes present. Recorded here because the cause and the fix both matter.

Cause, stated plainly: probe scanner-073 included
`grep -rl USB_CONFIG_FILE /etc /usr/bin`, a recursive grep over a directory of
binaries, sent down a command channel whose per-transfer timeout is 1000 ms.
The command outran the timeout, the probe aborted mid-sequence, and its cleanup
transfer failed. Interfaces 0 and 1 lost their driver; the host kernel log
showed `uvcvideo` being probed and the DEVICE answering `-71 (EPROTO)` to every
UVC control query, ending `Failed to initialize the device (-71)` — a
device-side wedge, not a host-side unbind. Reissuing USBDEVFS_CONNECT for both
interfaces returned 0 and changed nothing. A USB port reset (USBDEVFS_RESET)
was then issued as the standard remedy; the device disconnected
(`usb 1-1.1: USB disconnect, device number 72`) and did not come back.

No persistent change was made: rootfs is mounted read-only, every probe wrote
only to /tmp (tmpfs, RAM), no firmware was flashed, and a replug has fully
recovered this unit from a comparable state before. The failure is recoverable
by the owner in seconds, but only by the owner.

Root cause, FIXED: probe commands were validated for length, read-only-ness and
output destination but NOT for bounded runtime, and the validation ran AFTER the
runnable file was written, so nothing stopped it being executed. Both are now
fixed in the derivation script: `ensure_bounded_runtime` refuses recursive
greps, `find` without `-xdev`, and `cat` of a log file, and all validation runs
BEFORE the file is written. Verified against the exact fatal command (refused,
no file produced) and against the ten commands that ran safely (all pass):
0 false positives, 0 misses.

## Firmware findings this round

dsp_mailbox_door: The `/dev/rk_preisp` TEXT channel (`echo s 0xNNN <v>`) filters
command ids — most are silently dropped, producing no reply AND no kernel
`send msg len` line. zxmain does not use it; it calls
`ioctl(fd, 0x4004700b, {word_count, command_id, parameter})` — `_IOW('p',11,4)`
on the same node — which is NOT filtered. An ARM32 helper was built to do
exactly that. Ids that were "silent" reply normally through it: 0x700 ->
`id_msg_algo_depth_t`, 0x701 -> `id_msg_algo_calibration_t`. This retires the
previous reading that silence meant "not transmitted".

projector_block: 0x483 `id_msg_proj_stop_trigger_t` (replies
`set_stop_flag stop trigger success` and `zhi_proj_light_off light off` when a
trigger is active), 0x484 `id_msg_proj_create_one_trigger_t`, 0x485
`id_msg_proj_catchframe_t`, 0x486 `id_msg_proj_debug_t`, 0x487 unknown.

illumination_negative: 0x483 is IDEMPOTENT — `light off` fires only on a live
trigger. The decisive test was run: a normal depth capture to leave the emitter
in its usual lit idle state, then 0x483. The owner visually confirmed THE
EMITTER REMAINED LIT. The standing dim idle glow is therefore not an active
projector trigger, and 0x483 does not extinguish it. Still open.

## Wireless: the scanner is not only a USB device

Six read-only probes (scanner-067..072) established that this unit runs a live
Broadcom radio, not dead BSP boilerplate:

- `wlan1` is an ACCESS POINT at 192.168.179.1, SSID `MINI-REVO-A2270452`
  (the unit's own serial), currently running.
- Live config is `/data/hostapd.conf` (NOT the stock /etc template, which says
  `ssid=test`): 5 GHz `hw_mode=a`, 802.11n + 802.11ac, WPA2-PSK/CCMP,
  passphrase `Revopoint3d`, SSID broadcast.
- Running daemons: hostapd, dhcpd, lighttpd, fcgiwrap, dropbear, zxmain.
  Ports 80 and 22 LISTEN on 0.0.0.0.
- lighttpd serves /tmp/webroot with the same four CGI endpoints the vendor SDK
  contains URLs for (zx_cmd/zx_media/zx_upload/zx_bt.cgi), and the firmware's
  own fringe images (ImageL0/L1/R0/R1.bmp, 257080 B each) land there.
- Bandwidth: depth is 61.4 Mbit/s at 15 fps, 122.9 at 30 fps, against USB 2.0's
  480 Mbit/s signalling. Arithmetic only; no wireless transfer benchmarked.
- The USB gadget script supports `usb_rndis_en`, so the same HTTP API may be
  reachable over the existing cable. NOT attempted — recomposing the gadget
  tears down the UVC interfaces. Owner decision.

CONFIRMED over the air: the owner reports seeing the `MINI-REVO` network in a
client WiFi scan. The AP is genuinely beaconing, not merely configured. This
host has no wireless adapter, so that check could only be made by the owner.

STILL NOT established: no client has ever associated (`iw dev wlan1 station
dump` empty, `/proc/net/arp` empty) and no CGI endpoint was exercised.

Security observations recorded, not acted on: the AP passphrase is a fixed
factory string likely identical across units; dropbear listens on the AP subnet;
zx_upload.cgi's name suggests a firmware upload path and was not probed;
/userdata is mounted read-write so /data/hostapd.conf is persistent.

Full evidence: docs/wireless-stack.md.

## Software delivered

- `src/scanner3d/cli.py`: new `depth-frame-to-ply` subcommand bridging a raw
  frame file to a PLY through `cloud_export.export_depth_frame`. All calibration
  inputs required, no defaults — this unit's factory numbers appear nowhere in
  the module. Four distinct exit codes; every refusal writes nothing.
- `tests/test_pointcloud_cli.py`: 53 tests.
- `docs/depth-to-pointcloud.md`: the full pipeline, uncertainty first.
- `docs/dsp-protocol.md`: corrected — the earlier retraction over-corrected;
  silent ids are unreachable via the text channel, not unimplemented.
- `docs/wireless-stack.md`: new.
- Session-manifest CLI wiring: LANDED. `--session-dir` on `depth-frame` and
  `rgb-frame` (shared `_run_frame_capture`), recording a verified artifact into
  the session only after re-reading it off disk and matching length + sha256.
  Success is recorded, never inferred; a genuine capture failure is recorded
  FAILED without being reported as a recording problem. 6 new tests, mutation
  score 7/7 (6/7 first pass; one survivor on counters found a real gap in the
  tests, which was closed and re-verified).
  DEFERRED with rationale: `paired-capture`/`paired-rgb-series` were not wired.
  A `PairedOutcome` carries two sub-results and mapping it onto session stages
  needs a design decision (one stage each vs one stage with two artifacts, and
  what a partial depth-success/RGB-failure means). Deferring was the right call
  over rushing a hole into the guarantee. Good candidate for its own task.

verification (measured by the lead, not reported secondhand; final):
- `pytest tests/ -q` -> exit 0, 1700 tests, zero FAILED/ERROR lines
- `ruff check .` -> All checks passed
- `ruff format --check .` -> 149 files already formatted, nothing outstanding
- `mypy src tests` (strict) -> Success, no issues in 64 source files

not_run: no live-device verification was performed AFTER the device recovered.
Every hardware claim in this report predates the incident and was measured with
`final_ok: True`; nothing has been re-measured since the replug. Any claim
requiring fresh hardware evidence is `not run`, not passing.

live_scanner_access: YES. Six read-only probes (scanner-067..072) all restored
the device cleanly (final_ok True). A seventh (scanner-073) aborted and left the
device offline; see the device-incident section above. No illumination, firmware, flash or power command
was ever issued, and no write to the target was made in any probe.

concurrency_note: cli.py, tests/test_cli.py and tests/test_pointcloud_cli.py had
more than one writer this round. All agents were constrained to targeted edits
and re-reads; no work was lost, but the ownership split was imperfect and is
worth tightening before the next parallel run.
