state: implementing
task: commit the repo; build+test the M4-M7 pipeline (validation, registration, fusion, mesh)
round: 1
activity: repo committed (5 commits, was previously ZERO - nothing had ever been version controlled). Nine irreplaceable depth frames preserved as fixtures. scanner3d.geometry foundation written, verified and committed. Capture checklist written. M5's unreachable exit criterion withdrawn and replaced. Four pipeline modules currently being built and adversarially reviewed by a workflow.

## Commits this round

- eff68d9 Initial commit (166 files). The repository had NEVER been committed:
  zero tracked files, unborn main. Every line of code and every piece of
  measured evidence existed only as loose working-tree files.
- 6cf221e The nine real depth frames as tests/fixtures/frames/, with a
  provenance manifest. They previously lived only under a scratchpad path
  embedding a session id that exists on no other machine, so the real-frame
  tests skipped on any fresh checkout. Recomputing from the committed bytes
  reproduces the documented figures exactly: 2304000 samples, 79.4788% zeros,
  smallest non-zero code 17.
- a1234c0 scanner3d.geometry - rigid transforms, rotation about an arbitrary
  axis, Horn's closed-form alignment, pose-difference metrics.
- 7026b89 docs/capture-checklist.md - the physical half of M4/M6.
- ba0e96e M5's exit criterion withdrawn as unreachable, replaced.

## Two findings worth the planner's attention

geometry_rotation_angle: Rotation magnitude uses atan2(|sin t|, cos t), not
acos((trace-1)/2). acos is severely ill-conditioned near the identity -
acos(1-e) ~ sqrt(2e) - putting a ~1e-8 rad noise floor on exactly the
measurement loop closure exists to make. A well-closed turntable loop would
have reported the floor instead of its true error. Measured: 1.7e-16 rad
against 6.3e-8 rad on a 36-step revolution, relative error ~1e-16 maintained
from 1e-6 to 179.999 degrees.

m5_criterion_withdrawn: M5 demanded comparison against SDK-derived output. The
vendor SDK cannot connect to this unit at all, so the criterion was unreachable
by construction. Replaced with caliper ground truth plus a predeclared
tolerance, which is strictly stronger - agreement between two implementations
never established that either was right.

## Working envelope, measured from the committed frames

Depth codes returned: p1 16817, median 23398, p99 58002. At the unproven scale
that implies ~84 to ~290 unknown units, median ~117. Field of view 36.3 x 25.0
degrees, and the principal point sits 6.6% across a 640-wide frame - near the
left edge. A target 40-70 units across fills the frame usefully.

The median implying ~117 is *consistent with* the MINI's documented 100-200mm
working distance. That is a coincidence, not evidence, and it is exactly the
kind this project has been fooled by before. One caliper reading settles it.

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
