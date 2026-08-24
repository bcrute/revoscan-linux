# Roadmap

This is a milestone-level plan from the current state (scanner-001, discovery
+ diagnostics + external oracle) toward owned Linux capture, reconstruction,
and mapper-artifact export. Numbers are sequence markers, not fixed dates.

## Scanner-023 — bounded RGB-control telemetry (implemented offline)

The accepted three-observation `paired-rgb-series` path now has an explicit
`--read-rgb-controls-during-series` gate. When paired with the exact series
opt-in, it reads the fixed standard gain, exposure-auto and exposure-absolute
controls on the already-open RGB descriptor at four bounded points: once after
RGB `STREAMON` and once after each successfully dequeued observation, before
any requeue. This is at most twelve read-only `VIDIOC_G_CTRL` attempts under
the existing absolute deadline. Values, returned-ID mismatches and ordinary
errors are cumulative typed evidence; frame capture and telemetry status are
reported independently, with signal and cleanup precedence preserved.

The implementation, fake-kernel boundary tests and CLI gates are offline
verified. Telemetry has not been run against the physical scanner; no
`QUERYCTRL`, `QUERYMENU`, control write, vendor/XU request or extra wait was
added. A later physical invocation requires its own reviewed task.

## Scanner-024/025/027 — physical baseline and fixed RGB extension boundary

Scanner-024 is the accepted one-shot, unprivileged paired run. It delivered one
complete depth frame and three nearly-black RGB payloads that each retained four
opaque bytes after JPEG EOI and therefore remained strict-invalid. Four telemetry
points showed invariant raw gain 16, exposure-auto index 3 and absolute-exposure
131072; those values do not establish units or the separate SDK extension state.
No setting, vendor/XU request, retry or elevation occurred, and the observed
devnum change has no attributed cause.

Scanner-025 implemented the exact one-GET boundary offline, and scanner-027 ran
it once: `a1 81`, `wValue=0x1200`, `wIndex=0x0400`, requested 60 bytes, actual
2 bytes, raw `0000`, SHA-256
`96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7`, typed
`short_response` (exit 7), with no decode or retry. The same accepted run's
mandatory immutable blue-light off write was the final physical action: one
60/60 write, clean restoration and stable postflight. This proves transfer
acceptance and host recovery, not optical darkness.

Scanner-028 reconciles the result offline. The pinned SDK (SHA-256
`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`) uses
property IDs 0x15/0x16 but emits the same single selector-0x12 GET shape for
CameraType 12. Its 60-byte zeroed destination and wrapper return only establish
nonnegative-transfer success; actual length is discarded, while the enclosing
functions decode little-endian words only after that wrapper result. The
descriptor fixture (`2fa267c0c41d4cc53390bb7d1562d4b54fb44e8be04e0d568de8633b5d626613`)
records entity 4 bitmap `ffa201`, whose selector-0x12 bit is clear. That is an
unadvertised/unknown device behavior, not proof of rejection. The fixed
one-transfer model remains faithful, but raw `0000` is not assigned a meaning.

## Scanner-029 — fixed RGB GET_LEN boundary (implemented offline)

`rgb-exposure-length` is a dedicated dry-by-default boundary for the claimed
selector-0x12 control length. The exact opt-in is serial-gated and permits one
and only one device-to-host request: `bmRequestType=0xa1`, `bRequest=0x85`,
`wValue=0x1200`, `wIndex=0x0400`, `wLength=2`, timeout 1000 ms. It reuses the
interface-0 detach/claim/read/release/reconnect lifecycle and has no protocol
clear, retry, GET_CUR, V4L2 operation or setting path. Dry runs, missing or
wrong serials, and refusal before ownership are inert.

The offline fake-syscall boundary preserves the packed setup, requested and
actual lengths, retained raw bytes and digest, typed short/overlong/errno,
signal, exception, identity and cleanup results. Only exactly two bytes are
decoded little-endian as a raw `reported_control_length`; bytes may be
reported numerically as raw value 0, without validity, support, exposure or
unit semantics. Scanner-029 did
not access the physical scanner. Scanner-030 subsequently issued that exact
GET_LEN once, then completed the mandatory immutable blue-light off write and
clean restoration. Every new physical GET_LEN invocation must again issue its
own read, then its own immutable off write and clean restoration before
stopping; neither read result nor finalizer establishes control semantics.

## Scanner-030 — accepted physical GET_LEN and finalization

The exact scanner-029 boundary was run once against the authorized MINI. Its
single `a185001200040200e8030000` request returned 2/2 bytes, raw `0200`, and
clean interface restoration. The mandatory separate immutable `0x922=0` write
then accepted 60/60 bytes, restored the interface, and was followed by three
stable read-only postflight samples. This establishes raw command acceptance
and host restoration only; no optical readback exists.

## Scanner-031 — fixed RGB GET_INFO boundary (implemented offline)

`rgb-exposure-info` is a dedicated dry-by-default boundary for one later
capability-bit query. Its exact serial-gated opt-in permits only one
`a1 86` GET_INFO request for selector 0x12 (`wValue=0x1200`, `wIndex=0x0400`,
`wLength=1`, timeout 1000 ms), through the existing interface-0 lifecycle.
There is no retry, protocol clear, alternate UVC request, setting, V4L2, HID,
vendor or illumination path. Exact setup, raw byte, lengths, SHA-256, typed
failure, phases, cleanup and recovery evidence are retained; dry, missing or
wrong-serial paths are inert.

An exact one-byte reply is reported as raw `raw_info` and a syntactic Linux
UAPI bit view: GET, SET, DISABLED, AUTOUPDATE, ASYNCHRONOUS and reserved high
bits. This does not prove selector advertisement, semantic exposure mapping,
SET acceptance, units, valid current values or frame behavior. No physical
GET_INFO ran in scanner-031. A later physical task must run exactly one such
read, then exactly one immutable blue-light-off write and prove restoration.

## Scanner-026 — mandatory depth blue-light finalization (implemented offline)

Any command that attempts depth `VIDIOC_STREAMON` now raises a final-state
obligation immediately before that ioctl. After every owed V4L2 teardown attempt
(`STREAMOFF`, unmap, queue release and close), the live depth command performs
one and only one fixed host-to-device `SET_CUR` through interface 0, using the
accepted SDK shape `21 01 00 07 00 04 3c 00 e8 03 00 00` and the NUL-padded
payload `echo s 0x922 0 >/dev/rk_preisp`. It uses the existing detach/claim,
release/reconnect lifecycle and never sends an on/brightness/readback command.

The outcome retains the exact setup, requested/actual length, payload digest,
phase order, cleanup and recovery guidance. A successful report means only that
the fixed transfer was accepted and interface 0 was restored; the repository has
no optical readback and therefore makes no claim that a sensor observed darkness.
The finalizer reports non-exact host writes as `short_write` or `overlong_write`;
required success additionally requires an exact successful 60-byte record and
clean restoration, while only an inert not-required result may succeed without
an attempt.
Dry runs, preflight refusals, RGB-only commands and depth paths that never reach
`STREAMON` remain inert. Scanner-026 is offline-verified; it has no new physical
invocation or accepted optical evidence.

## Illumination actuator surface — exhausted host-side (offline)

Full detail is in `docs/illumination-actuators.md`. In summary: disassembly of
the pinned SDK resolved every `echo s 0x… >/dev/rk_preisp` template to its
enclosing function, and only `cs::Camera::setLaserOnOff` (`0x922`) and
`cs::Camera::ledCtrl` (`0xb00`–`0xb04`) are illumination controls. `0x702` is
`setStreamResolution` and `0x913` is `setDepthROI`, so both earlier
plausible-looking candidates are ruled out.

The recovered `ledCtrl` dispatch table shows `0xb00 0` selects IR *twinkle
mode* and is only ever emitted straight after `0xb01 1`, which turns the IR LED
**on** — so the earlier isolated `0xb00=0` test was never an off command and its
negative result should not be counted as an exhausted off-path. The genuine off
commands are `0xb01 0` (IR) and `0xb04 0` (laser); both were accepted 60/60 with
no owner-observed optical change.

Separately, the device's busybox applet list was already captured during
scanner-039 (`busybox --list >/tmp/s39_bb`, hash-linked in that manifest). It
provides 254 applets including `printf`, `dd`, `chmod`, `sh`, and notably
`devmem` and `i2cget`/`i2cset`/`i2cdetect`. Register-level and I2C-level access
is therefore reachable through the already-proven shell channel without any
binary upload, which is a strictly smaller step than the hand-built ARM ELF
route and should be evaluated first. No optical claim is made anywhere.

## Wireless — an undocumented second transport, measured and live

Full detail is in `docs/wireless-stack.md`. In summary: this unit is not only a
USB device. It runs `hostapd` on `wlan1` as a WPA2 access point named
`MINI-REVO-A2270452` at 192.168.179.1, with `dhcpd` serving leases, `lighttpd`
serving the vendor's four CGI endpoints on port 80, and `dropbear` on port 22 —
all listening on `0.0.0.0`, all running right now. The AP is 5 GHz with
802.11n/ac enabled. None of this is documented by the vendor for this model.

This matters beyond novelty. The depth stream is 61.4 Mbit/s at 15 fps against
a link that carries several hundred, so the bandwidth fits with margin. The
vendor SDK's own network URLs point at those same CGI endpoints, so the HTTP
path is a real transport rather than an accident. And the firmware already
exports its stereo fringe images into the web root, which means capture data is
retrievable over the network without touching USB at all.

Sequencing note: this is a **capability axis**, not a milestone reorder. It does
not unblock M4 (calibration) or M6 (registration), and it should not displace
them. Its natural place is after M5, as an alternative transport under the
existing session boundary rather than a parallel stack.

Open work, in dependency order:

1. ~~Confirm over-the-air beaconing.~~ **DONE, 2026-08-24** — the owner saw the
   `MINI-REVO` network in a client WiFi scan. The AP is genuinely broadcasting.
   This was the precondition for everything below, and it is satisfied.
2. **Associate a client and reach the HTTP root.** Join
   `MINI-REVO-A2270452` (WPA2, passphrase `Revopoint3d`), take a DHCP lease
   from the 192.168.179.100-110 scope, and fetch `http://192.168.179.1/`. This
   is the first step that puts a client on the device's network rather than
   only observing it, so decide the security posture (item 5) first.
3. **Exercise one CGI endpoint read-only** and record its request/response
   shape, the same way the USB vendor channel was mapped. `zx_cmd.cgi` is the
   one the vendor SDK drives; `zx_upload.cgi` should be left alone until its
   semantics are known.
4. **Evaluate the RNDIS route.** `/etc/init.d/S50usbdevice` supports
   `usb_rndis_en`, which would expose the same HTTP API over the existing
   cable. Recomposing the gadget tears down the UVC interfaces, so this is an
   owner decision and needs the kernel's gadget config read first.
5. **Decide the security posture before anything is put on a network.** The AP
   passphrase is a fixed factory string, likely identical across units; SSH is
   reachable on the AP subnet; `zx_upload.cgi` was not probed. `/userdata` is
   read-write, so the configuration is persistent and changeable — which is
   both the fix and the risk.

No optical or throughput claim is made anywhere; the bandwidth figures are
arithmetic against a nominal link rate, not measurements.

## Session boundary — versioned manifest and atomic artifact store

`scanner3d.session` implements the Phase 2 boundary that capture, processing,
the CLI, the future local service and the mapper handoff share. It carries a
schema version, explicit session and stage states, device identity, coordinate
frames with units/axes/handedness, calibration and capture-profile provenance,
per-stage counters, and checksummed artifact records.

Three properties are enforced rather than documented: success is recorded and
never inferred, so a session whose writer died reads back as `incomplete` and a
session cannot be `complete` while any stage is unfinished; raw evidence is
immutable, created `O_EXCL | O_NOFOLLOW` and never overwritten; and a partial
write leaves no artifact at all. Artifact paths are validated segment by segment
against a bucket allowlist, must be canonical, and are re-resolved so a
symlinked parent cannot redirect a write outside the session.

`scanner3d session <path>` inspects a session entirely offline, with `--verify`
to re-hash every recorded artifact and `--redact` for a shareable support
report. It opens no device and runs no SDK probe.

## Milestones

### M1 — Foundation, doctor, reference oracle (done; oracle blocked)

- **What:** `scanner3d discovery`/`doctor`/`list`, USB descriptor parsing,
  permission checks, and the hash-pinned external SDK probe as a ground-truth
  oracle.
- **Exit criterion:** `scanner3d doctor` correctly classifies device
  presence, identity, topology, roles, advertised modes, and permissions on
  this host, with the reference probe available as an opt-in check.
- **Outcome:** the discovery/doctor half met its exit criterion. The oracle
  half did not, and cannot on this unit: the SDK enumerates the scanner but
  `connect()` fails with `ERROR_NOT_SUPPORT` at an internal licence check.
  No frames or calibration values were obtained. The probe, its hash
  pinning, its timeout and its cleanup are all working and verified — the
  blocker is the vendor's authorization gate, not this repository's code.
- **Risk:** the oracle being unavailable removes the cross-check that M3 and
  M4 were expected to validate against, so those milestones must instead be
  validated against physical ground truth (measured objects) or against the
  SDK's trace log. Residual risk: discovery assumptions (single device, this
  exact USB identity) don't generalize to other units or multi-device
  setups, which is untested.

### M2 — Characterize the vendor control handshake

- **What:** Determine what the device actually needs over HID/vendor control
  requests before it will stream usable depth — e.g. by observing the
  vendor SDK's traffic (bus capture, or instrumenting/tracing the probe) and
  correlating it with the HID error vocabulary already known
  (`ERROR_HID_CHANNEL_ERROR`/`ERROR_HID_WRITE_ERROR`/`ERROR_HID_READ_ERROR`).
- **Exit criterion:** a written, reproducible description of the control
  sequence (init/mode-select/calibration-read) sufficient to attempt an
  independent implementation, backed by captured evidence, not by reading
  the SDK's public header alone.
- **Risk:** the HID protocol may be encrypted, signed, or otherwise
  obfuscated beyond passive capture.
- **Updated by the M1 result:** the vendor SDK's own trace log already gives
  a partial answer for free. It shows the connection sequence as: open
  usbfs → detach `uvcvideo` from interface 0 → claim interface 0 → open the
  HID channel on interface 4 → claim interface 4 → attempt authorization.
  Everything up to authorization succeeds on this unit. Because the SDK
  reaches HID over usbfs and never opens `/dev/hidraw*`, observing that
  traffic does **not** need a udev rule or `/dev/hidraw` access — an
  `strace`/usbmon capture of the existing probe is sufficient, and we
  already have the usbfs ACL needed to run it.
- **Partially delivered by scanner-002 (observation half).**
  `reference/tracer/` + `scanner3d trace` captured and decoded one full run of
  the failing `connect()`. Measured, not inferred: the SDK uses **only**
  asynchronous URBs on usbfs, touches exactly two endpoints (control `0x00`
  and interrupt-IN `0x85`), and sends every vendor request as a **UVC class
  `SET_CUR`/`GET_CUR` on endpoint 0** addressed to entities 3 and 4 of
  interface 0 — 60-byte payloads, `01` + a NUL-terminated path for the file
  reads. The HID interrupt endpoint carried **zero bytes**: one URB was armed
  and then cancelled. This corrects the earlier working assumption that the
  control channel ran over HID. Full byte-level evidence and the labelled
  inferences are in
  [architecture.md](architecture.md#vendor-control-channel-measured-on-the-wire).
- **What M2 still owes.** The capture stops where the SDK stops — at
  authorization. Nothing about stream start, mode select or projector control
  has been seen on the wire by any means, so the "sufficient to attempt an
  independent implementation" exit criterion is **not** met. Calibration
  readout is the one exception, and it came from a different evidence set: the
  SDK's own traffic never reached it, but owned code has since performed one
  (scanner-007's `/data/camparam/Pl.bin` read and scanner-008's `0x918`
  depth-scale read).
- **Transport half unblocked by scanner-004.** Owned code can now reach the
  vendor control channel and get a correct answer from it, by temporarily taking
  interface 0 from `uvcvideo` (see [Open questions](#open-questions)). That
  removes the transport doubt from M2's remaining work; it does not shorten the
  work itself, which is still "observe or derive the sequence the device needs".
- **Write direction unblocked by scanner-005.** The four-transfer `/tmp/inited`
  exchange on entity 4 was reproduced over that same transport, byte for byte,
  including both `SET_CUR` writes (see [Open questions](#open-questions)). So a
  *multi-transfer* observed exchange can now be replayed without the SDK, and
  the device accepts owned writes. With scanner-007's calibration read and
  scanner-008's depth-scale read, **eleven** transfers total have been
  exercised over this transport in **four** detach/claim runs: one read on
  entity 3, the four `/tmp/inited` transfers on entity 4, four more on
  entity 4 for `/data/camparam/Pl.bin`, and two more on entity 4, selector
  `0x0E00`, for the depth-scale property `0x918`.
- **Call paths recovered statically by scanner-006.** An offline pass over the
  hash-pinned `lib3DCamera.so` (unstripped; full symbols) traced all four
  domains, reaching **exact request shapes for the XU control primitives** but
  **not** for streaming, where endpoint, alternate setting, format/frame indices
  and ordering remain unrecovered — see
  [Static protocol archaeology](architecture.md#static-protocol-archaeology-sdk-call-paths-toward-the-wire).
  Facts that change M2's remaining work: (1) the MINI's vendor "extension"
  channel is an **ASCII command/register passthrough** — `setPropertyExtension(0xffff, …)`
  sends an `echo s 0x<id> <args> >/dev/rk_preisp` string as the 60-byte payload
  of a `SET_CUR` (`0x21/0x01`, `wValue=0x0700`, `wIndex=0x0400`), with the ISP
  register ids matching the `PROPERTY_TYPE_EXTENSION` enum (laser `0x922`, LED
  `0xb00`, HDR `0x914`, fringe `0xa05`, resolution `0x702`, stream `0x484/0x487`);
  (2) there is a **second, untouched channel** — a true USB HID interface
  (`bInterfaceClass==3`, magic `0x5A5A5A5A`, `SET_REPORT`/interrupt); (3) the
  licence gate is a **virtual `authorize()` precondition inside the host
  library at connect**, reading `/data|/persist/sdk-license`. For the eleven
  transfers owned code has actually issued — now including a factory
  calibration key and the depth-scale property — it is *not* a per-transfer
  device lock. Whether the device also accepts streaming, HID, the other XU
  selector `0x0700`, other `0x0E00` property ids, or other paths from an
  unlicensed host is a **hypothesis, not established** — no such operation
  has been tried. What M2 still owes is
  unchanged in kind — the bring-up *ordering* (which `echo s …` writes precede
  UVC COMMIT) is above `uvc_stream_start` and needs a capture or owned probing,
  not more static reading.

### M3 — Owned Linux transport

- **What:** Capture raw depth and RGB frames without the vendor SDK, either
  via V4L2 reads from `/dev/video0`/`/dev/video2` plus the control channel
  characterized in M2, or via a from-scratch libusb/usbfs client mirroring
  the SDK's approach.
- **Exit criterion:** code in this repository captures at least one raw
  depth frame and one raw RGB frame end-to-end with the vendor SDK absent
  from the process, and the frame content is corroborated (e.g. dimensions,
  rough sensor pattern) against an L0 probe run on the same physical unit.
- **Risk:** largest risk in the whole roadmap — if the control handshake
  from M2 turns out to depend on undocumented, unit-specific calibration
  data that can only be read (not derived), this milestone may require
  extracting that data via the SDK rather than reimplementing the handshake
  from scratch.
- **Updated by scanner-004:** the usbfs route the SDK uses is confirmed
  workable from owned code for a read — detach, claim, transfer, release,
  reconnect all succeed on this unit. That is one data point on one read-only
  request, not a working transport for streaming, and it says nothing about
  whether the driver-mediated alternative (extension-unit controls through
  `uvcvideo`) would also work; that has still not been tried.
- **Streaming mechanism recovered statically by scanner-006.** Frame delivery is
  ordinary **libuvc**: a UVC VS PROBE/COMMIT negotiation (`wValue` `0x0100`/`0x0200`,
  `(wFormatIndex, wFrameIndex)` selecting format+resolution), then
  `libusb_set_interface_alt_setting` and isochronous **or** bulk transfers. So
  the M3 route via UVC streaming is confirmed standard; the open question is
  whether the MINI needs `/dev/rk_preisp` register writes as a pre-roll, which a
  bring-up capture (experiment 1) or a V4L2 stream attempt (experiment 4a) would
  settle. Note the ownership boundary: PROBE/COMMIT and the alt-setting act on
  the **VideoStreaming** interface (1 for depth, 3 for RGB), *not* the
  VideoControl interface 0 this repo's lifecycle owns. Ranked experiments
  are in
  [architecture.md](architecture.md#static-protocol-archaeology-sdk-call-paths-toward-the-wire).
  *(Superseded in part by scanner-012: the streaming endpoint addresses,
  transfer types, packet sizes, alternate settings and format/frame indices are
  no longer unrecorded — all six were decoded from the committed descriptor and
  are pinned by tests. See the scanner-012 bullet below.)*
- **Depth frame captured by scanner-009 (measured, one run) — the positive
  outcome for the experiment-4a discriminator above.** Plain V4L2 against the
  existing `/dev/video0` — no interface detach, no XU/vendor-control transfer,
  no `echo s …` register pre-roll, no HID traffic reported — negotiated `Y16`
  640×400 (bytesperline 1280, tight; sizeimage 512000), queued 4 MMAP buffers,
  and dequeued buffer 0 after a single `poll()` that waited 1.043 s of an 8.0 s
  deadline. The 512000-byte frame (SHA-256 `0a1b667eeca5ed2ce1c64e6aab124580efc9474e5e3e13187b53944d0c296933`)
  was written to disk, and cleanup (`STREAMOFF`, four `munmap`s,
  `REQBUFS(count=0)`, close) completed inside the process. This run got a
  frame **without** any ISP register pre-roll — for this one run only; it does
  not by itself explain the earlier zero-buffer GStreamer result recorded
  above, which used a different resolution and pixel format (1280×800
  `GRAY16_LE` there vs 640×400 `Y16` here), so the two are not treated as
  confirming or contradicting each other. Full evidence, the corrected
  buffer-flag reading, and one causally unresolved post-exit device
  re-enumeration (`devnum` 41→42, 2.788 s after the frame was captured, cause
  not established) are in
  [architecture.md](architecture.md#owned-code-captures-a-raw-depth-frame-measured).
  **What this does not do:** capture RGB, repeat the run, or establish what
  the 16-bit samples mean physically. Corroboration against an L0 probe is
  unreachable while L0 remains blocked (see M1), so M3's exit criterion —
  "at least one raw depth frame and one raw RGB frame... corroborated against
  an L0 probe" — is **not yet met**. The cumulative live-attempt count for
  `scanner-009` is one.
- **Updated by scanner-010 — one RGB attempt, no RGB frame.** The same bounded
  engine was given a second fixed profile (`rgb`, `MJPG` 1280x800) and run
  **once** against this unit. It resolved the RGB node by interface string plus
  V4L2 index (`/dev/video2`), opened it, passed `QUERYCAP`, and issued one
  `VIDIOC_S_FMT`, which succeeded. The run then refused its own result: for
  `MJPG` 1280x800 `uvcvideo` reported `sizeimage` = **8192000 B**, exactly 4x
  the mode's descriptor `dwMaxVideoFrameBufferSize` of 2048000 B and past the
  2097152 B per-buffer map ceiling **then in force** — a ceiling derived from
  that descriptor, which the driver's answer showed does not predict the
  kernel's allocation. (scanner-011 has since set the RGB ceilings from the
  measured 8192000 B instead, and spent one further attempt; see the
  scanner-011 entry below.) Exit
  `format_mismatch` (17) after 0.120 s. **No `REQBUFS`, no `mmap`, no `STREAMON`, no poll, no
  `DQBUF`, zero bytes dequeued, no artifact written**, one clean `close()`, no
  surviving holder, and a 45.06 s post-exit observation window recorded no
  re-enumeration (`devnum` 42 throughout) and no new kernel messages. Full
  evidence is in
  [architecture.md](architecture.md#owned-code-attempts-a-raw-rgb-frame-measured).
  **What this does not do:** capture RGB, establish what a larger ceiling would
  yield (nothing was mapped or streamed, so what the kernel would actually have
  filled is unknown), say anything about JPEG content, colour or depth-RGB
  synchronisation, or explain the scanner-009 re-enumeration. The cumulative
  live-attempt count for `scanner-010` is one. M3's exit criterion is still
  **not met**.
- **Updated by scanner-011 — the ceiling was the wrong suspect; still no RGB
  frame.** The RGB profile's ceilings were re-derived from the `sizeimage`
  scanner-010 measured (8192000 B per mapped buffer and per raw artifact,
  32768000 B mapped in total across at most four buffers; the depth profile was
  left untouched), and the same command was run **once** more against this
  unit. Everything scanner-010 could not reach now worked: `S_FMT` returned
  `MJPG` 1280x800 `sizeimage=8192000` again, `VIDIOC_REQBUFS` allocated 4 MMAP
  buffers, all four were `QUERYBUF`'d at `length=8192000`, `mmap`ped,
  `QBUF`'d, and `VIDIOC_STREAMON` succeeded. **Then nothing arrived**: the
  single 8.0 s poll returned `revents=0x0000` after 8.007 s — no `POLLIN` and
  no `POLLERR` — so `VIDIOC_DQBUF` was never issued, zero bytes were dequeued
  and no artifact was written. Exit `poll_timeout` (4) after 8.148 s wall, well
  inside the external `timeout ... --kill-after=20s 60s` deadline, which never
  fired. Cleanup completed in full (`STREAMOFF`, four `munmap`s,
  `REQBUFS(count=0)`, `close`), and a 76 s post-exit observation window
  recorded no re-enumeration (`devnum` 42 throughout), an unchanged node and
  binding set, no holder and no new kernel messages. Full evidence is in
  [architecture.md](architecture.md#owned-code-attempts-a-raw-rgb-frame-with-the-measured-ceiling-measured).
  **What this does do:** establish that the measured ceiling is sufficient and
  that the RGB node accepts streaming, and that scanner-010's refusal was a
  too-small ceiling and nothing else. **What this does not do:** capture RGB,
  explain *why* no buffer arrived (a required vendor initialisation, an
  alt-setting or bandwidth outcome, a first-frame latency beyond 8.0 s and
  other causes are all equally unexcluded — nothing was varied and nothing was
  probed), or say anything about JPEG content, colour or depth-RGB
  synchronisation. The cumulative live-attempt count for `scanner-011` is one,
  and two across all RGB attempts. M3's exit criterion is still **not met**.
- **Silent-stream analysis by scanner-012 (offline; no live attempt spent).**
  The scanner-011 silence was explained as far as existing evidence allows,
  without touching the scanner. Three things changed:

  1. **The six descriptor values M3 was blocked on are now recorded.** RGB is
     VideoStreaming interface **3**, endpoint **`0x84` IN, bulk,
     `wMaxPacketSize` 512**, **one alternate setting (0)**, `bFormatIndex` 1 /
     `bFrameIndex` 1 (the only mode it advertises), default frame interval
     333333 (30 Hz), `bcdUVC` `0x0100` → a **26-byte** VS Probe/Commit block.
     Depth is interface 1, endpoint `0x82`, also bulk, also single-alt. Decoded
     by [`reference/analysis/uvc_stream_state.py`](../reference/analysis/README.md)
     and pinned by `tests/test_uvc_stream_state.py`.
  2. **Two explanations are eliminated and one is largely eliminated.**
     Bandwidth / alternate-setting negotiation cannot be the cause — bulk
     endpoints have no admission control, there is no alt setting to choose,
     and the peak advertised rate is 38.5% of the single-endpoint high-speed
     bulk ceiling. Node/role resolution is eliminated. Missing `S_PARM` is
     largely eliminated: uvcvideo commits an interval at `S_FMT` regardless,
     and the successful depth capture issued a strict **subset** of the RGB
     attempt's V4L2 operations.
  3. **The instrument was switched off.** uvcvideo's `trace` parameter was `0`,
     and the module carries exactly one dynamic-debug callsite — so every
     `uvc_dbg()` message about probe/commit contents, URB completion status and
     frame decoding was suppressed at the source. The zero-line kernel delta
     therefore proves only that no `dev_err`/`dev_warn` path fired; it says
     nothing about what happened on endpoint `0x84`. A first reading of
     `quirks = 4294967295` as "all quirks forced" was **wrong** and is
     corrected: that is the driver's unset sentinel, every live module
     parameter equals its compiled default, and `2207:110c` appears nowhere in
     the module's quirk tables.

  Ranked evidence table and the full reconstruction:
  [architecture.md](architecture.md#why-the-rgb-stream-is-silent-offline-reconstruction-scanner-012).
  The leading unestablished explanation is that the device needs an
  out-of-band step before RGB payload flows; the unmeasured one is simply what
  happens on the wire after COMMIT. M3's exit criterion is still **not met**,
  and scanner-012 spent no live attempt.

### M4 — Calibration model + metric depth

- **What:** Read or derive device intrinsics and depth scale; convert raw
  Y16/YUY2 sensor output to metric (millimetre) depth; establish RGB-depth
  extrinsics.
- **Exit criterion:** a captured depth frame converts to millimetre values
  that agree with the vendor SDK's calibrated output within a stated,
  measured tolerance on a known test object.
- **Risk:** calibration data may only be retrievable through the vendor
  control channel in a format that is itself unit-specific and requires
  ongoing dependence on the M2 findings holding for other units.
- **Calibration transport recovered statically by scanner-006, and it is a
  transfer shape this repo has already driven live.** Intrinsics/extrinsics/
  distortion are **fetched over the wire, not read from a local file**:
  `getIntrinsics`/`getExtrinsics`/`getDistort` build the lookup key
  `/data/camparam/{Pl,Prgb,LC_RT,Distort}.bin` and fetch it through the
  transport vtable slot `+0x58` → `cs::libuvcReadFile` — the *same* four-transfer
  protocol as
  the `/tmp/inited` sequence already exercised on this unit: SET_CUR
  `21 01 wValue=0x0100 wIndex=0x0400 wLength=60` carrying `01` + path + NUL,
  repeated GET_CUR `a1 81 wValue=0x0200` returning a `uint32_le` length plus up
  to 56 bytes until the length is zero, then a 60-zero cleanup write. Only the
  path string differs from what has already succeeded. (The one piece of file
  I/O on that path is a *write*: an optional `'|'`-named cache file the SDK
  writes the retrieved bytes into afterwards, a branch the calibration keys do
  not take.) Depth scale comes over a
  different XU control: `getDepthScale` → `readBuffer(0x918)` →
  `libuvcReadValueExtension`, which uses a **fixed** `wValue=0x0E00` and carries
  `0x918` as the four-byte payload `18 09 00 00`, then re-reads 60 bytes on the
  same selector; `depthScale = 1/int32_le(buf[0:4])` when `buf[59]==4`. This
  makes an owned calibration read the highest-value, lowest-novelty next
  experiment for M3. See
  [architecture.md](architecture.md#domain-4--calibration--depth-scale-fact-disassembled).
- **Depth intrinsics delivered by scanner-007 (measured, one run).** That read
  was run once against this unit and completed: `/data/camparam/Pl.bin` is
  served to an unlicensed host and returned a 40-byte vendor `Intrinsics`
  record — 640×400, `fx = fy = 898.6829833984375`, `cx = 42.43747329711914`,
  `cy = 192.8326873779297`, all six structural checks passing — with both the
  clearing write and the driver handback clean. Evidence in
  [architecture.md](architecture.md#owned-code-reads-the-factory-depth-intrinsics-measured).
- **Depth scale delivered by scanner-008 (measured, one run, done).** The
  two-transfer `libuvcReadValueExtension` pair for property `0x918` — the
  `wValue=0x0E00` read-by-id channel Domain 4 recovered statically — was run
  once against this unit and completed: a 4-byte SET_CUR carrying `18090000`
  was accepted in full, and the 60-byte GET_CUR reply (SHA-256
  `9873fa193adabb7daa87a77718d963d586d798838f4e6edc1a2ddac0bb337dd4`) carried
  marker `4` and denominator `200`, giving
  `vendor_depth_scale = float32(1.0f / 200) = 0.004999999888241291`
  (float32 bits `0x3ba3d70a`) — bit-identical to the SDK's compiled-in MINI
  fallback `0.005f`, an **observation**, not a unit. **The scalar's physical
  unit is unknown from this evidence**: this run proves the reciprocal the
  device returned, not what it means, and nothing here establishes
  millimetres, metres, accuracy, or a raw-depth conversion factor. Exactly two
  control transfers ran, no retry, no protocol-clear transfer (this vendor
  primitive has none), and both the driver handback and the device's
  descriptors/topology came back unchanged. Evidence in
  [architecture.md](architecture.md#domain-4--calibration--depth-scale-fact-disassembled).
  **Done for M4:** the depth intrinsics and the depth-scale property value,
  and, from M3, one raw depth frame now exists to eventually apply them to
  (scanner-009 — see M3 above). **Still open:** the sibling calibration keys
  `Prgb`/`LC_RT`/`Distort` (never requested, and nothing here says they are
  served), extrinsics, distortion, repeatable streaming, the physical unit of
  the depth-scale value, and every metric claim — the numbers are what the
  device stores, not a validated model. Neither calibration input has been
  applied to the scanner-009 frame, and that frame's own 16-bit samples carry
  no established unit, scale, or zero/invalid-sample convention — the exit
  criterion above still needs the captured frame converted and compared
  against the (unreachable) L0 oracle.

### M5 — Point cloud generation

- **What:** Deproject a calibrated depth (+ RGB) frame into a 3D point
  cloud.
- **Exit criterion:** a single-frame point cloud is produced from an owned
  capture and visually/quantitatively compared against SDK-derived output
  for the same scene.
- **Risk:** low relative to M2/M3/M4; mainly an engineering risk once
  calibrated depth exists.

### M6 — Registration/fusion

- **What:** Align and merge point clouds from multiple frames/poses into a
  single consistent scene.
- **Exit criterion:** multiple captures of a known object register within a
  stated accuracy and fuse without visible seams on a test target.
- **Risk:** structured-light noise characteristics of this specific sensor
  are unknown; registration algorithms tuned for other sensors may need
  rework.

### M7 — Mesh + export

- **What:** Surface reconstruction from fused point clouds; export to
  standard mesh formats.
- **Exit criterion:** a fused scan exports to a mesh format (e.g. PLY/OBJ)
  openly loadable in common tooling (e.g. MeshLab), produced entirely
  without vendor tooling.
- **Risk:** low/medium; mostly a matter of picking and integrating an
  existing reconstruction algorithm.

### M8 — Mapper artifacts

- **What:** Produce whatever derived outputs the sibling mapping project
  actually consumes.
- **Exit criterion:** not yet defined — depends on the sibling project's
  input contract, which has not been specified in this repository.
- **Risk:** scope is currently unknown; this milestone cannot be sized
  further until that contract exists.

## Next experiment — `RGB-1`, `READY`

Exactly one experiment is ready to run. It is the cheapest thing that
separates the surviving explanations, and it is ready **because** scanner-012
found that the kernel's own diagnostics were disabled rather than silent.

**Do not run it as part of the analysis task that specified it.** It needs an
executor turn of its own, and the owner's authorization for the one privileged
host action it performs.

- **Question.** After `VIDIOC_STREAMON` on the RGB node, does anything arrive
  on bulk endpoint `0x84`, and what did uvcvideo actually negotiate?
- **Mutation.** One **host-side module-parameter write**:
  `/sys/module/uvcvideo/parameters/trace`, from `0` to `0xFFFFFFFF`, restored
  to `0` afterwards. This is a `uvcvideo` module attribute, **not** a device
  node, not sysfs device state, not usbfs and not firmware. Nothing about the
  scanner is mutated. The mask is all-ones deliberately: the `UVC_DBG_*` bit
  meanings cannot be verified on this host (no kernel source, no docs — see
  scanner-012), so guessing a specific mask would be inventing a constant.
- **Privilege.** The two writes need root (`trace` is `0644 root:root`). The
  invoking user is in `sudo`. This is the one thing in the experiment that
  needs owner sign-off.
- **Fixed command.** The **unchanged** scanner-011 invocation, byte for byte:

  ```
  scanner3d rgb-frame --execute-rgb-frame-capture --serial A22704522A6F00A64 \
      --json --output-dir <evidence>/artifact --json-out <evidence>/result.json
  ```

  Same MJPG 1280x800, same four buffers, same 8.0 s internal budget, same
  single dequeue, **zero retries**. No code change, no new ioctl, no
  `VIDIOC_S_PARM`, no UVC control write, no second mode, no depth access.
- **Controls held fixed.** No XU/vendor/ASCII command, no HID, no usbfs, no
  detach/claim, no reset, no replug, no install, no network.
- **Resource and time bounds.** Exactly **one** run, guarded by an attempt
  marker that makes a second impossible. External deadline
  `timeout --signal=TERM --kill-after=20s 60s`. Kernel output captured with
  `dmesg --follow-new` into a file capped at **8 MiB** and **20000 lines**
  (`trace=0xFFFFFFFF` during streaming can be verbose; the cap is the bound,
  and if it is hit that fact is recorded rather than the run re-attempted).
  `dmesg` line counts recorded before and after so ring-buffer wrap is visible.
  Post-exit observation ≥ 45 s for devnum, nodes, bindings, holders and kernel
  deltas, exactly as scanner-011 did.
- **Discriminating outcomes.** Each maps to a row of the ranked evidence table:

  | observation | conclusion |
  | --- | --- |
  | probe/commit contents logged | measures the negotiated 26-byte block — settles whether `sizeimage=8192000` is the device's `dwMaxVideoFrameSize`, and what `dwMaxPayloadTransferSize` was agreed (rows 3) |
  | URB completions with **zero** bytes, or no completions at all | the device sends nothing after COMMIT → the out-of-band/vendor-step explanation (row 1) becomes the leading survivor |
  | URB completions carrying bytes, but no frame ever completed | payload/FID/decode problem inside the driver, not a silent device (row 2) |
  | `Non-zero status (…) in video completion handler` | USB-level stall or error on `0x84` (row 2) |
  | nothing logged at all after commit | the driver never submitted URBs — a different failure than any row currently ranked, and the most surprising outcome |
- **Cleanup.** `trace` restored to `0` from an `EXIT` trap that runs on every
  path including failure and signal, then **read back and recorded**; a restore
  that cannot be verified makes the run non-successful regardless of what was
  captured. The capture tool's own cleanup (`STREAMOFF`, four `munmap`s,
  `REQBUFS(0)`, `close`) is unchanged and already proven on two runs.
- **Why not something else first.** A longer poll varies one number and
  explains nothing (8.007 s is already ~240 frame periods, and depth's first
  frame took 1.043 s). A vendor register write is still `NOT READY` — it sets
  device state with no proven read-current-state and no exact restore
  (experiment 5). Direct usbfs streaming is now *specifiable* thanks to the
  decoded descriptor, but it takes the device from the kernel driver and is
  strictly more dangerous than turning on a log.

## Exit criteria for deleting the vendor SDK oracle (L0)

The vendor SDK / reference probe (`reference/probe`,
`src/scanner3d/data/sdk-manifest.json`, `scanner3d.reference`) should be
deleted only once **all** of the following
hold:

- M3 delivers raw depth and RGB frame capture through an owned transport,
  with no dependency on the vendor SDK at runtime.
- M4 delivers metric depth that has been validated against the SDK's output
  at least once, so the owned calibration path is trusted without needing
  further side-by-side comparison.
- No remaining check in `scanner3d doctor` depends on `ProbeResult` (i.e.
  the `_check_frames`/`_check_calibration` checks in `doctor.py` are
  re-implemented against the owned transport, or removed).
- CI/test fixtures that currently assume an optional reference-probe path
  are updated so the test suite passes with the SDK entirely absent from the
  host.

Until then, L0 remains opt-in (`--reference-probe`) and hash-pinned, and is
never a hard dependency for `scanner3d doctor`/`list` to run.

## Open questions

- **Can the vendor SDK be made to connect to this unit at all?** This is
  now the blocking question for L0. `connect()` fails with
  `ERROR_NOT_SUPPORT` because the SDK's licence authorizer cannot read an
  `sdk-license` blob from the camera's own storage (`/persist/sdk-license`,
  `/data/sdk-license`) — see
  [architecture.md](architecture.md#reference-oracle-result-the-vendor-sdk-cannot-connect-to-this-unit).
  Smallest experiment: obtain a valid `sdk-license` for this serial/camera
  type from Revopoint and write it to device storage via the SDK's own
  `setUserData`/`cameraSetUserData` API (a device-side write over USB, not a
  local file), then retry `connect()`. This needs a vendor-issued artifact
  we do not have, so it is a genuine external blocker rather than a task we
  can schedule. **If this cannot be resolved, L0 is dead for this unit and
  M2 must proceed from bus capture alone** — which is viable, since the
  authorization gate is inside `lib3DCamera.so`, not in the device, and the
  device itself accepted every USB operation the SDK performed.
- ~~**Can owned code talk to the vendor control channel at all?**~~ **Answered
  by scanner-004: yes, for this one read-only request, once interface 0 is not
  held by `uvcvideo`.** scanner-003 issued exactly the observed transfer
  (`bmRequestType=0xa1`, `bRequest=0x81`, `wValue=0x0600`, `wIndex=0x0300`,
  `wLength=60`, 3 s timeout) directly and the kernel refused it with `EBUSY` in
  17.6 µs, because usbfs implicitly claims the interface a class request names
  and `uvcvideo` already owned interface 0. scanner-004 repeated the identical
  request through a bounded ownership lifecycle — confirm `uvcvideo` from sysfs
  and from `USBDEVFS_GETDRIVER`, `USBDEVFS_DISCONNECT`,
  `USBDEVFS_CLAIMINTERFACE`, the one read, `USBDEVFS_RELEASEINTERFACE`,
  `USBDEVFS_CONNECT` — and the device returned the full 60 bytes in 1.3 ms:
  `v2.8.7.20220610|v7.5.7.0426`, exactly what the independent SDK oracle
  predicts. The whole detached window lasted 127 ms, cleanup succeeded, and the
  device came back to its prior state. Full evidence in
  [architecture.md](architecture.md#owned-code-reads-the-vendor-channel-measured).

  **What that settles:** interface ownership was the obstacle, and for this
  request there is no further gate behind it — no licence, no HID init, no
  authorization step was needed to read a version string.

  **What it does not settle:** one read-only `GET_CUR` on entity 3, selector
  `0x06`. The licence question below is untouched by this result.

  **Measured disruption, identical across all four detaches so far:** detaching interface
  0 tore down the depth function — `uvcvideo` claims interface 1 alongside
  interface 0, so `/dev/video0` and `/dev/video1` were destroyed and recreated
  by the re-probe (same names and device numbers here, which V4L2 does not
  guarantee in general). The RGB pair on interfaces 2/3 was untouched. One
  kernel-log line, `Found UVC 1.00 device ZXProduct (2207:110c)`; no reset, no
  re-enumeration, no replug. Nothing was streaming and no process held a video
  node, so **the effect on an active capture remains unmeasured**. If the
  process dies between detach and reattach, interface 0 could remain unbound
  until a driver reconnect/rebind is issued explicitly or the device is
  physically replugged; the implementation reconnects in a `finally` on every
  failure path from the attempted detach onwards, SIGINT and SIGTERM included
  and the ambiguous window around the disconnect ioctl with them, and reports a
  nonzero exit with explicit recovery guidance when it cannot.

- ~~**Will the device accept an owned `SET_CUR`?**~~ **Answered by scanner-005:
  yes, for the two payloads the SDK itself sent.** The four-transfer
  `/tmp/inited` exchange on entity 4 — `SET_CUR` carrying `01` + the path,
  `GET_CUR`, the same `GET_CUR` again, `SET_CUR` of 60 zero bytes — was
  reproduced once, in order, inside the same interface-0 lifecycle, and every
  transfer matched the capture byte for byte: both writes accepted in full, the
  first read returning `0c 00 00 00` + `v7.5.7.0426`, the second returning 60
  zeros. The detached window lasted 40 ms, both the clearing write and the
  driver handback succeeded, and the kernel logged one line. Full evidence in
  [architecture.md](architecture.md#owned-code-writes-to-the-vendor-channel-measured).

  **What that settles:** the write direction of the vendor channel is open to
  owned code without the SDK, without a licence and without HID init, and a
  *multi-transfer* vendor interaction — not just one isolated request — is
  reproducible. The device is stateful across those transfers in the way the
  trace implied: the first read after the path select returned a record, the
  second returned zeros.

  **What it does not settle:** two payloads on one selector of one entity, both
  of which the vendor's own software sent to this unit. It does not show that
  `/tmp/inited` was *read* — the reply is the algorithm version string, which is
  not plausibly that path's contents — nor what the leading `01` byte means, nor
  anything about any other path or selector. The SDK's authorization step needs
  more than a `SET_CUR` the device accepts: it needs a licence blob the library
  could not find, and that gate is unchanged.

  **Next falsifiable step:** reading a second `GET_CUR` selector on entity 3 or 4
  would test whether the channel is generally *readable* beyond the two
  selectors observed. Sending a path the SDK was not observed to send is a
  different kind of step — parameterising an observed command rather than
  reproducing one verbatim. It does **not** need a fresh owner prompt: the owner
  has already authorized autonomous reversible protocol and capture work on this
  unit. What it needs is a **separately reviewed task** that implements it as a
  new closed whitelist with the bounds and cleanup fixed in source, on the same
  ownership lifecycle.

- ~~**Does the vendor channel require data that is only readable, never
  derivable (e.g. factory calibration blobs)?**~~ **Answered by scanner-007 for
  the depth intrinsics, and scanner-008 for the depth scale.** The file-read
  protocol addresses blobs by ASCII path
  (`/persist/sdk-license`, `/data/sdk-license`, `/tmp/inited` observed), and
  scanner-006 recovered which paths hold calibration data from disassembly of
  `cs::Camera::getIntrinsics` and `cs::libuvcReadFile`: the SDK fetches
  intrinsics, extrinsics and distortion through this exact protocol using the
  keys `/data/camparam/{Pl,Prgb,LC_RT,Distort}.bin`. scanner-007 then sent one
  of those keys — `/data/camparam/Pl.bin` — through the already-proven
  four-transfer shape as its own separately reviewed closed whitelist, in one
  live run. The device served it to an unlicensed host and returned a complete
  40-byte vendor `Intrinsics` record, with the clearing write and the driver
  handback both clean. So the depth intrinsics are readable, not derivable, and
  they are now in hand.

  **Depth scale answered by scanner-008.** Depth scale comes over a different
  channel entirely (`libuvcReadValueExtension`, fixed `wValue=0x0E00`, id
  `0x918`), which scanner-008 sent through its own separately reviewed closed
  whitelist — the same two-transfer shape Domain 4 recovered statically, one
  live run, no protocol-clear transfer because this primitive has none. The
  device returned an exact 60-byte reply with marker `4` and denominator `200`,
  giving `float32(1.0f / 200) = 0.004999999888241291` (bits `0x3ba3d70a`),
  bit-identical to the SDK's `0.005f` MINI fallback — an observation, not a
  established unit. So the depth-scale property is readable, not derivable,
  and it too is now in hand, with the driver handback and the device's
  descriptors/topology both unchanged afterward.

  **What remains open** is narrower than the original question. The sibling
  calibration keys (`Prgb`, `LC_RT`, `Distort` and the rest) were never
  requested by owned code, so nothing says the device serves them — one key
  answering does not generalize to `/data/camparam/*`. No `0x0E00` property id
  other than `0x918` has been requested either, so the response layout for
  those ids remains unknown. And the depth-scale value's **physical unit is
  unknown from this evidence** — two transfers and one reciprocal prove what
  the device returned, not what it means. Any further read on either channel
  (a sibling calibration key, another `0x0E00` id) would need its own
  separately reviewed task on the same terms scanner-007 and scanner-008 ran
  under: a closed whitelist (fixed path or id constants, no caller-supplied
  path, opcode, selector, entity, interface, length, timeout or repetition) on
  the accepted ownership lifecycle, under the fixed bounds and outcome
  taxonomy in
  [architecture.md](architecture.md#ranked-next-experiments-each-mutation-risk-discriminating-evidence)
  (256-byte capacity, ≤8 GET_CURs, stop on a length word `> 56`, one 60-zero
  cleanup write on every exit whose failure makes the whole run non-successful
  even if the read completed, bounded-incomplete rather than any runtime
  widening). The gate is that review, not a fresh owner authorization: the
  owner's standing authorization for autonomous reversible protocol and capture
  work already covers a bounded, self-clearing read. It stops short of — and a
  new owner decision *is* required for — anything persistent, destructive,
  licence-path-related, or otherwise beyond bounded reversible work.

- ~~**Do the `wIndex` extension-unit IDs (3 and 4 on interface 0) match the
  device's own descriptors?**~~ **Answered by scanner-003: half of them do.**
  `scanner3d.usb_descriptors` now parses VideoControl entities from the
  recorded blob, which is byte-identical to the live device's sysfs
  descriptors. Entity 4 *is* the (single) extension unit per function, GUID
  `C2211A6D-787E-3C4A-AC06-7E434B676F9A`, 9 controls, `bmControls ffa201`.
  Entity 3 is an **output terminal** (`wTerminalType 0x0101`, fed by the
  processing unit), yet the SDK's version read was addressed to it and
  succeeded. Table and reasoning in
  [architecture.md](architecture.md#videocontrol-entities-read-from-the-devices-own-descriptors).

  The follow-on question this opens: **which entity actually implements the
  vendor file-read protocol, and does the device accept vendor `GET_CUR` on
  any entity ID?** Smallest experiment: none proposed here — probing entity
  IDs the SDK was never observed to address is authoring new commands, which
  should not happen before the observed read round-trips.
- **Can raw depth/RGB frames be pulled via plain V4L2 (`ioctl`/`v4l2-ctl`)
  once the HID init sequence has run once via the SDK, or does the device
  require HID traffic to be interleaved throughout the whole streaming
  session?** A plain-UVC stream open (via GStreamer, no HID/vendor-control
  traffic involved) has already been observed to open and negotiate format
  successfully but deliver zero buffers after 17+ seconds, with no kernel
  UVC or bandwidth error, at 1280×800 `GRAY16_LE` — see
  [architecture.md](architecture.md#why-uvc-discovery-alone-did-not-produce-frames).
  ~~So the open question is no longer whether plain UVC streaming works at
  all (it does not, on its own); it is whether SDK-driven HID
  initialization changes that outcome.~~ **Revised by scanner-009: plain
  V4L2, with no SDK and no HID/vendor-control traffic reported, did deliver
  one frame — at a different resolution and pixel format than the earlier
  negative result (640×400 `Y16`, not 1280×800 `GRAY16_LE`).** scanner-009's
  `depth-frame` command negotiated `Y16` 640×400 directly against the
  existing `/dev/video0`, with `uvcvideo` left bound and no interface
  detach, and dequeued a complete 512000-byte frame after a 1.043 s
  `poll()` wait — see
  [architecture.md](architecture.md#owned-code-captures-a-raw-depth-frame-measured).
  Because the resolution and pixel format differ from the earlier
  zero-buffer attempt, this one success does not establish that plain UVC
  streaming works *in general* without HID/SDK initialization, and it does
  not explain why the 1280×800 `GRAY16_LE` attempt delivered nothing — the
  two results are not treated as confirming or contradicting each other. It
  does establish that at least one 640×400 `Y16` configuration is reachable
  through plain V4L2 alone, once, with no HID or SDK involvement reported.
  **Partly narrowed by scanner-010 for RGB:** the RGB function's capture node
  opens and negotiates `MJPG` 1280x800 under plain V4L2 with no HID or SDK
  traffic, but that run never streamed — it stopped because `uvcvideo`
  answered `sizeimage=8192000`, 4x the mode's own descriptor buffer size and
  past the 2 MiB map ceiling then in force (raised to the measured 8192000 B by
  scanner-011). **Narrowed further by scanner-011 for RGB:** with that raised
  ceiling the same node allocated, mapped and queued four 8192000 B buffers and
  `VIDIOC_STREAMON` succeeded, and then no buffer became ready within 8.0 s
  (`revents=0x0000`, no `POLLERR`). So for RGB it is established that open,
  format negotiation, buffer allocation and stream start all work under plain
  V4L2 with no HID or SDK traffic, and that in the one attempt made no buffer
  arrived afterwards. **Still open:** why no buffer arrived — the run measured
  the outcome, not the mechanism, so a required vendor initialisation, an
  alt-setting or bandwidth outcome and a first-frame latency longer than 8.0 s
  are all equally unexcluded. Also still open: whether the depth capture
  repeats, whether 1280×800 (or other advertised modes) actually deliver
  buffers under plain V4L2, whether RGB delivers any frame at all under any
  condition, and whether SDK-driven HID initialization is ever required for any
  mode. Smallest experiment: repeat scanner-009's exact capture to check
  repeatability, then retry the 1280×800 `GRAY16_LE` configuration under
  plain V4L2 to see whether resolution, not HID/SDK init, was the variable
  that mattered in the earlier negative result.
- ~~**Is the uaccess ACL sufficient for `/dev/hidraw10`?**~~ **Answered, and
  the question turned out not to matter.** `/dev/hidraw10` is `0600
  root:root` with no `user:ben` ACL entry, unlike `/dev/video*` and the
  usbfs node, which both carry one. But `strings` and a full `strace` of the
  vendor probe show the SDK never opens any `/dev/hidraw*` device — it
  reaches the HID interface over usbfs. So `/dev/hidraw` access is not
  required by the reference path, and may not be required by an owned
  transport either, provided it also goes through usbfs.
- **Does the advertised default of 15 fps for depth and 30 fps for RGB hold
  under owned capture, or is the default only meaningful to the vendor
  SDK's negotiation logic?** Smallest experiment: once M3 lands, request the
  advertised default UVC mode directly via V4L2 `ioctl` and compare the
  negotiated frame interval to the SDK's.
- **Does this project need to support more than one scanner attached
  simultaneously?** Smallest experiment: none run yet; requires owner input
  on whether multi-device is in scope, since it changes how `discovery.py`
  or a future transport layer would need to disambiguate serials.
