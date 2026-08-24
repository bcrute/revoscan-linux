# 3d-scanner project context

## Purpose

Build a native, local-first Linux scanning stack for the owner's Revopoint MINI.
The immediate operational purpose is to scan the mapper chassis and components;
the engineering purpose is to learn and own the RGB-D acquisition, calibration,
registration, fusion, meshing, and export pipeline rather than depend on Revo
Scan.

The eventual handoff to the sibling `../mapper` repository is an explicit data
boundary, not a reason to couple scanner transport code into mapper.

## Executor override

- On 2026-08-22 the owner explicitly asked Codex to use smaller Codex subagents
  instead of spawning or waiting on Claude agents for the current work. While
  that instruction remains active, a bounded Codex subagent may act as the
  implementation executor and own product edits plus `.coordination/coder/`;
  the root Codex planner still owns planner/review files and independently
  reviews every handoff before assigning the next task.

## Durable boundaries

- Target host observed 2026-08-21: Linux Mint 22.3 / Ubuntu Noble, kernel
  6.14.0-37-generic, x86_64.
- Align the supported Python runtime and coordinate/uncertainty vocabulary with
  `../mapper` where practical. Mapper currently pins CPython 3.14.
- Capture and processing must work locally without an internet connection.
- Raw observations, calibration provenance, units, coordinate frames, timing,
  dropped-frame evidence, and limitations must remain explicit.
- Discover devices by identity and stable paths; never assume `/dev/video0` and
  `/dev/video2` remain fixed.
- The owner explicitly authorized autonomous scanner work on 2026-08-22,
  including disconnect/reconnect, reset, firmware flashing, undocumented device
  commands, and accepting the risk that the dedicated unit could be bricked.
  Each implementation assignment must still bound and justify the action so its
  evidence remains interpretable; this authority does not cover unrelated host,
  repository, network, purchase, deployment, or external-system actions.
- The owner additionally requires that work which can illuminate the scanner's
  blue depth laser/projector is not finished until that light has been commanded
  off. Treat the exact laser-off write as a final-state safety obligation after
  all V4L2 descriptors are closed, on success, failure, timeout, signal and
  exception. A failed off or driver-restoration attempt must be prominent in the
  outcome with recovery guidance; never silently leave the light on.
- On 2026-08-23 the owner directly observed that the blue front emitter remained
  on while the scanner was idle despite repeated accepted `0x922=0` transfers.
  USB transfer acceptance, release/reconnect success, and stable host topology
  are therefore insufficient shutdown evidence. Treat the emitter as on until
  the owner visually confirms darkness. Suspend other physical work while
  shutdown is recovered; `uvcvideo` reconnect is a candidate relight mechanism
  and must not be assumed safe.
- After scanner-051 later left depth interfaces 0/1 unbound at stable devnum 50,
  the owner again directly observed the front emitter still on. Driver-detached
  UVC is therefore also disproven as an optical shutdown mechanism. The owner
  wants a staged path toward improved embedded firmware; first recover the
  boot/update/recovery chain and exact emitter actuator so firmware work has a
  rollback boundary rather than relying on blind flashing.
- Offline SDK review found an untried, separately documented generic LED-off
  command, `echo s 0xb00 0 > /dev/rk_preisp`; the previously ineffective
  `0x922` and `0xb04` controls are the SDK's distinct laser family. Recovered
  `zxmain` also implements internal `preisp_set_led_on_off(0)` as ioctl
  `0x400456cc` on `/dev/v4l-subdev3`, but only a proximity-sensor loop was
  proven to use it. Scanner-056 tested `0xb00=0` exactly once: the device
  accepted 60/60 bytes and host cleanup passed, but the owner observed the
  front emitter still on. It is therefore a proven negative optical result and
  must not be retried. The next isolated candidate is the lower-level
  `0xb01=0` gate. Scanner-058 then tested b01 exactly once: it too was accepted
  60/60 with clean host teardown while the owner observed the front emitter
  still on. Both public SDK LED controls are therefore negative optically and
  must not be retried. The next reversible candidate is the recovered internal
  `/dev/v4l-subdev3` off ioctl, tested RAM-only before any firmware flash.
- Do not vendor or redistribute Revopoint's prebuilt SDK binary during the
  diagnostic phase. It may be used as a pinned, external reference oracle while
  an owned transport is developed.
- Do not claim hardware support or scan accuracy from fixtures alone. Hardware
  evidence must name the exact device and test conditions.

## Observed scanner

- USB identity: `2207:110c`, manufacturer `Chishine3d`, product `ZXProduct`.
- Serial: `A22704522A6F00A64`.
- Revopoint 3DCamera SDK 3.2.23 maps this serial to model `MINI` (enum 12).
- USB exposes two UVC functions plus HID: DepthCam and RGBCam.
- Depth modes include 16-bit 640x400 and 1280x800 at 5/10/15 fps.
- RGB exposes MJPEG 1280x800 at 10/15/20/30 fps.
- One earlier 1280x800 depth/GStreamer attempt opened but produced no buffers;
  scanner-009 later captured one complete 640x400 `Y16 ` frame through native
  V4L2. The different modes are not directly comparable, and neither result
  establishes a required vendor initialization sequence.
- The scanner and current Realtek hub are both enumerating on the 480 Mb/s USB
  2.0 bus. The host's 10 Gb/s root bus is empty. No kernel bandwidth error was
  observed during the bounded stream attempt.
- Follow-up topology after moving the Realtek hub upstream on 2026-08-21: the
  hub's SuperSpeed companion enumerated on Bus 002 at 5 Gb/s, while the MINI
  still enumerated through that hub's USB 2 companion on Bus 001 at 480 Mb/s.
  This proves the hub upstream is now SuperSpeed and is not limiting the MINI;
  the remaining 480 Mb/s choice is at the scanner/cable link.
- Follow-up direct test on 2026-08-21: with the hub removed, the MINI
  enumerated directly at host path `1-1`, still at 480 Mb/s with USB version
  2.00, while all four video nodes and HID returned. This establishes 480 Mb/s
  as the attached MINI/current-cable transport on a known port; an alternate
  cable would be the only remaining way to distinguish a cable limit from a
  device limit, and no evidence currently suggests that distinction matters.
- The owner identifies the scanner cable as USB 3-rated. Revopoint's MINI quick
  start guide says to use a USB 3.0-or-newer host port because USB 2.0 may not
  supply sufficient operating power; it does not state that the MINI data link
  itself negotiates SuperSpeed. Official guide:
  `https://download.revopoint3d.com/support/download/mini/mini-quickstartguide-en-v3.0-2023.7.27.pdf`.

## Reference evidence

- Official source: `https://github.com/Revopoint/3DViewer`
- Inspected commit: `516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`
- Included Linux SDK: 3DCamera 3.2.23, external diagnostic copy currently at
  `/tmp/revopoint-3dviewer.i2pZPb`.
- Observed SDK library SHA-256:
  `49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`.
- `scanner-002` wire capture: 35 asynchronous URBs; 34 control transfers on
  endpoint 0 and one zero-byte cancelled interrupt-IN transfer on `0x85`.
  Vendor control messages are UVC class `SET_CUR`/`GET_CUR` requests to entity
  IDs encoded in interface-0 `wIndex`, not payload delivery over HID.
- Observed safe read candidate: `a1 81 0600 0300 003c` returns the NUL-padded
  ASCII pair `v2.8.7.20220610|v7.5.7.0426` before authorization.
- The MINI's configuration descriptor identifies entity 3 on interface 0 as an
  output terminal and entity 4 as extension unit GUID
  `C2211A6D-787E-3C4A-AC06-7E434B676F9A`.
- The first owned direct usbfs attempt of the safe read was made exactly once.
  The kernel returned `EBUSY` before transferring bytes because `uvcvideo`
  owned interface 0. This is the first obstacle on that path, not evidence that
  no later obstacle exists.
- `scanner-004` temporarily detached/claimed interface 0 once and repeated that
  exact read. It returned the expected 60-byte version payload in 1.266 ms;
  release/reconnect succeeded, the detached window was 127.312 ms, depth nodes
  were recreated, RGB nodes were untouched, and no reset or re-enumeration
  occurred. This proves owned access only for that request.
- `scanner-005` reproduced the exact SDK-observed `/tmp/inited` sequence once.
  Both 60-byte `SET_CUR` writes were accepted, the first `GET_CUR` returned
  `0c000000` + `v7.5.7.0426` + NUL/zeros, and the second returned 60 zeros.
  Selector clear, release, and reconnect succeeded in 39.743 ms; interfaces and
  video nodes are restored. This proves only those four observed transfers.
- `scanner-006` statically traced the pinned unstripped SDK binary. Factory
  depth intrinsics are fetched with the same entity-4 file-read protocol using
  fixed key `/data/camparam/Pl.bin`; replies are 56-byte-max length-prefixed
  chunks terminated by zero, followed by a 60-zero protocol clear. Depth scale
  instead uses fixed `wValue=0x0e00`, payload `18 09 00 00`, then a 60-byte GET.
  Streaming PROBE/COMMIT and data act on VS interface 1/3, not VC interface 0;
  direct streaming is not ready until descriptor and restoration work lands.
- `scanner-007` read `/data/camparam/Pl.bin` once through a fixed bounded
  whitelist. The device returned a complete 40-byte 640x400 depth-intrinsics
  record (`fx=fy=898.6829833984375`, `cx=42.43747329711914`,
  `cy=192.8326873779297`; SHA-256
  `fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf`).
  The zero clear, release and reconnect succeeded; this proves only that key,
  not sibling calibration keys, depth scale, alignment, or metric accuracy.
- `scanner-008` read depth-scale property `0x918` once through the exact fixed
  two-transfer XU pair. The 60-byte reply (SHA-256
  `9873fa193adabb7daa87a77718d963d586d798838f4e6edc1a2ddac0bb337dd4`)
  carried marker 4 and denominator 200, yielding binary32
  `0.004999999888241291` (`0x3ba3d70a`). The physical unit remains unknown;
  release/reconnect succeeded and no protocol-clear transfer exists on this
  vendor path.
- `scanner-009` captured one complete raw depth frame through ordinary V4L2
  with `uvcvideo` left bound: `Y16 ` 640x400, 512000 bytes, SHA-256
  `0a1b667eeca5ed2ce1c64e6aab124580efc9474e5e3e13187b53944d0c296933`.
  Cleanup completed. The sample unit/scale/invalid convention remain unknown.
  A devnum 41->42 re-enumeration 2.788 seconds after the frame is one causally
  unresolved observation, not an attributed effect of the capture.
- `scanner-010` implemented the shared RGB path and ran it once. `uvcvideo`
  retained MJPG 1280x800 but reported `sizeimage=8192000`, four times the UVC
  descriptor's 2048000-byte maximum and above the fixed 2097152-byte safety
  ceiling. The command stopped before REQBUFS/mmap/STREAMON, closed cleanly,
  and observed stable devnum 42/nodes/bindings for 45.06 seconds. No RGB frame
  exists yet; a measured-ceiling attempt is the next bounded experiment.
- `scanner-011` used the exact measured 8192000-byte ceiling, mapped/queued four
  buffers and reached STREAMON once, but no buffer became ready during the sole
  8.007-second poll. It cleaned up fully and stayed stable for 60 seconds. This
  rules out the former 2 MiB refusal as the immediate blocker but does not
  distinguish timeout, standard-UVC negotiation, vendor initialization,
  bandwidth or another reason for the silent RGB function.
- `scanner-012` decoded the RGB stream offline: UVC 1.00, VS interface 3,
  endpoint `0x84` IN bulk, only alt 0, 512-byte packets, format/frame 1/1 and
  30 Hz default. The local module has no `2207:110c` quirk and all parameters
  are at compiled defaults. Crucially, `trace=0` suppressed the driver messages
  that would distinguish no URBs, empty/error completions and decode failure.
  RGB-1 is therefore a trace-enabled repeat of the unchanged bounded capture;
  it awaits explicit authority for a temporary root-owned module-param write.
- On 2026-08-22, after the owner made blue-light shutdown an explicit completion
  condition, Codex sent exactly one unprivileged laser-off command to the
  authorized MINI: class `SET_CUR` setup
  `21 01 00 07 00 04 3c 00 e8 03 00 00`, with the ASCII payload
  `echo s 0x922 0 >/dev/rk_preisp` NUL-padded to 60 bytes (SHA-256
  `6030364daddcc4c9855d16da552c58161bbec5cf77ea23ace9de2f14839ffb0d`).
  The device accepted 60/60 bytes with no retry; interface-0 release/reconnect
  succeeded and devnum 47, all four video nodes and driver bindings remained
  stable. This is strong command/cleanup evidence, but software has no optical
  readback and therefore cannot independently assert visible-light state.
- `scanner-026` now enforces that fixed off command automatically after every
  existing depth STREAMON attempt, only after all V4L2 teardown through close.
  The complete finalizer interval is termination-shielded; exact transfer
  acceptance and clean driver restoration are separate evidence, and either
  failure blocks success with unplug/recovery guidance. This layer is accepted
  from 1,285 offline tests and has not itself been invoked on the scanner.
- Immediately after scanner-026 acceptance on 2026-08-23, the owner-required
  physical final-state action sent the exact off command once more. It was
  accepted 60/60 bytes in 1.548 ms; release/reconnect passed, devnum 47 stayed
  stable, all UVC/HID bindings and video nodes returned, and no holder remained.
  This is command/restoration evidence only; optical darkness remains unread.
- `scanner-027` then spent the one accepted physical selector-0x12 read. The
  device returned only 2/60 bytes (`0000`) for setup
  `a181001200043c00e8030000`; cleanup passed, but no four-word exposure mapping
  exists. The mandatory separate off write was accepted 60/60, cleanup passed,
  and devnum 47/bindings/nodes stayed stable. Neither transfer was retried.
- `scanner-028` reconciled that short read against the pinned SDK and descriptor
  entirely offline. The SDK requests the same one 60-byte GET, zero-fills the
  destination, treats every nonnegative libusb byte count as wrapper success and
  then decodes without preserving the count. Entity 4's `ffa201` bitmap has the
  selector-0x12 bit clear. Thus the SDK could manufacture zero-valued fields
  from a short reply, while the physical `0000` itself remains two raw bytes
  with no proven exposure or status meaning.
- `scanner-029` added and accepted an offline-only fixed UVC `GET_LEN` boundary
  for selector `0x12`: `a1 85`, `wValue=0x1200`, `wIndex=0x0400`, length 2,
  timeout 1000 ms. Its separate exact opt-in is serial-gated, never retries and
  decodes only an exact two-byte reply as a raw little-endian number with no
  assigned support, exposure, unit, validity or status semantics. No physical
  scanner access occurred; one read plus the mandatory off finalizer is next.
- `scanner-030` ran that GET_LEN once physically. Setup
  `a185001200040200e8030000` returned exact `0200`, so the device-reported raw
  length is 2; this still proves no exposure semantics or SET support. The
  mandatory separate off command was accepted 60/60 as the final transfer,
  cleanup passed and devnum 47/bindings/nodes/no-holder state stayed stable.
- `scanner-031` added and accepted the offline fixed GET_INFO boundary for the
  same selector: `a1 86`, `wValue=0x1200`, `wIndex=0x0400`, length 1, timeout
  1000 ms. Its exact opt-in is serial-gated and its raw capability-bit view is
  labelled syntactic only. It did not access the scanner; one physical read
  followed by the mandatory off/restoration is next.
- `scanner-032` ran that GET_INFO once and returned exact raw `03`: Linux-UAPI
  syntax has GET+SET set and all other defined/reserved bits clear, without
  proving actual SET acceptance or exposure semantics. The separate off command
  accepted 60/60 as the final transfer; cleanup and stable devnum 47 postflight
  passed. Marker immutability was not established and is recorded as a limit.
- Safety reopening after `scanner-032`: the owner observed the blue front emitter
  still on at idle. The range-query branch is suspended. Static SDK recovery and
  one visually verified shutdown intervention take precedence over other work.
- `scanner-033` recovered and ran both pinned-SDK logical disable paths once:
  SDK-exact `0x922=0` and laser-LED `0xb04=0`. Both transfers returned 60/60;
  the owner visually observed the blue front emitter still on after each while
  interface 0 remained detached. No logical write is now accepted as shutdown.
  The owner chose removal of USB hub-port power as the idle safety invariant.
  Until that path has permission and physical verification, unplug the scanner.
- The downstream Realtek hub then accepted one exact port-1 power-clear and
  reported power/connection off; the scanner stopped responding over USB, but
  the owner observed its blue front emitter still on. This hub does not
  physically remove VBUS despite its descriptor/status. Scanner-034 was stopped
  before implementation. The owner then clarified that the hub is externally
  powered, invalidating the proposed upstream root-port cut as a scanner power
  control; scanner-035 was canceled without that write. The remaining
  software-only lead is an embedded-device shutdown command. Until it is traced
  and visually verified, only physical unplugging or dedicated power-switching
  hardware satisfies the idle invariant.
- `scanner-036` found no SDK-supported embedded shutdown: the public operation
  sends `reboot`, and no device `poweroff`/`halt` command exists. One unreferenced
  reset-state helper matches this unit's 17-character serial byte-14 `A` guard
  and sends exact `echo 0 > /sys/class/gpio/gpio117/value`. The GPIO meaning is
  unknown, so scanner-037 treats it only as a one-shot owner-authorized reset
  experiment whose result must be judged optically.
- `scanner-037` restored downstream hub port 1's data state once, causing the
  exact scanner to re-enumerate at devnum 49, then sent the serial-matched
  `gpio117=0` command exactly once. It accepted 60/60 bytes; interface 0 was
  released and left detached, with no subsequent scanner transfer. This remains
  protocol evidence only until the owner reports the front emitter visibly dark.
- `scanner-038` proved why scanner-037 could not supply a usable control: the
  running RV1126 Buildroot firmware returns no GPIO117 sysfs files and its live
  debug GPIO map contains no GPIO117 entry. The board exposes camera-like
  power-down lines GPIO55 `pwdn9282` and GPIO60 `pwdn`, three reset lines, and
  red/green/blue status-LED lines, but no projector mapping is established yet.
- Scanner-038's exact 98,304-byte live FDT (SHA-256
  `076ae86c2f334e9fbd73942b754590a165a94075f142c33834c27c9d69fbfdc5`)
  resolves GPIO55/60 as AR0144/OV9782 image-sensor power-down lines and GPIO61
  as their reset; these must not be used as projector guesses. GPIO121/125 are
  the reset/IRQ pair for `rockchip,rk1608` on SPI. No laser/projector/
  illumination property or dedicated regulator is described, so the emitter
  is probably controlled behind RK1608/`rk_preisp`; that is an inference, not
  yet a proven actuator.
- On 2026-08-23 the owner authorized physical scans while away. Every such run
  must still execute and record the strongest available final shutdown path.
  Any belief that the front emitter is dark is provisional until the owner
  returns and visually tests it.
- `scanner-039` identified PID 569 `/tmp/zxmain` as the live vendor controller
  and durable `/dev/rk_preisp` holder. Its exact 166,412-byte ARM ELF (SHA-256
  `decdc3ad4e34449cefee8543c790866cfc87dd9279a873c3abab124b9754b0f1`)
  actively issues `echo off >/dev/rk_preisp`; the UVC stream-control path uses
  the same `off`/`on` pair as a normal RK1608 power cycle.
- The matching Rockchip driver invokes physical `rk1608_power_off()` only on a
  reference-count transition from one to zero, disabling IRQ/MCLK and driving
  RK1608 wakeup/reset low. With zxmain's one persistent descriptor, one shell
  `echo off` produces `1 -> 2` on open, `2 -> 1` on the explicit command, and
  `1 -> 0` on close. Do not issue a post-off `q`: opening the misc device at
  zero invokes the complete RK1608 power-on/firmware-load path and can relight
  the projector before closing it again.
- The vendor service stop path has no illumination cleanup. Its init script
  sources `/data/zx_stop.sh` only if present, and that file is absent/empty;
  the USB stop script only tears down configfs. The successfully boot-loaded
  `preisp.rkl` image is absent from all searched live filesystem paths.
- `scanner-040` executed the exact textual `off` once. The transfer returned
  60/60 and the kernel logged `do cmd power off, count--`, but GPIO121 reset
  stayed high and the full before/after GPIO snapshots were byte-identical.
  This disproves the earlier single-reference assumption: at least one other
  shared misc/V4L2 power reference existed under unchanged/no-race semantics.
  The emitter remains presumed on.
- A second identical command is justified only as one separate conditional
  intervention. It can reach physical shutdown if scanner-040 began at count
  two; if reset remains high, stop without a third decrement. Do not signal
  PID 569 as a fallback: it owns `/dev/watchdog`, so termination can trigger an
  embedded restart and relight the scanner. GPIO121 low would be provisional
  electronic evidence only, pending the owner's direct optical check.
- `scanner-041` spent that one separately reviewed second command. It returned
  60/60 and added a second exact `do cmd power off, count--` line, but GPIO121
  again stayed high and its complete pre/post files were byte-identical.
  Interface-0 release and fixed devnum-49 host postflight passed. Under the
  no-concurrent-change algebra, scanner-040 began with count at least three and
  at least one reference remains after both commands; the exact count is still
  unknown. The emitter remains presumed on.
- Do not issue a third blind decrement or signal PID 569. First map the shared
  V4L2/misc owners and recover exact watchdog/supervisor behavior from existing
  evidence and primary source. Any later intervention requires a new bounded
  task and owner-visible darkness remains deferred until return.
- `scanner-042` accepted the offline ownership audit. The official misc and
  forwarded V4L2 `.s_power` paths share `power_count`, while ordinary video or
  subdevice opens do not prove a reference. The original count was at least
  three and the current count is at least one, but its exact value and all
  remaining owners are unknown. PID 569's persistent pre-ISP fd is one strong
  candidate; its watchdog fd and caught signals make unverified termination an
  unacceptable shortcut. Scanner-043 may recover only bounded read-only
  supervisor/watchdog/process metadata before any teardown proposal.
- `scanner-043` completed one reviewed read-only recovery at evidence root
  `/tmp/scanner-043-evidence.BWTqs7Haw8k4` (manifest SHA-256
  `3a68f22f7286bf19cf137da5adba92348e6b294830bf866bf85e67933f02242e`).
  Its 558-event chain ends with exact `clear:final_proc_stat`; claim/release,
  close, identity, and bindings all passed with no later scanner control.
- PID 569 remained `/tmp/zxmain`, start time 998, with 32 threads. fd 5 is an
  O_WRONLY `/dev/watchdog` descriptor and fd 114 is an O_RDWR `/dev/rk_preisp`
  descriptor. The recovered `/userdata/watchlog.sh` only trims two logs above
  8,192,000 bytes and does not restart or supervise zxmain. The watchdog class
  exposes only `dev`, `device`, `power`, `subsystem`, and `uevent`, so live
  metadata does not reveal nowayout, timeout, timeleft, or armed state.
- Do not signal PID 569 or issue a third blind RK1608 decrement. Offline binary
  analysis must first prove how zxmain reaches watchdog flag value 2
  (`WDIOS_DISABLECARD`), whether any normal exit path performs it, and whether
  camera/RK1608 cleanup precedes process exit. Optical darkness remains deferred
  to the owner's return and the emitter remains presumed on.
- `scanner-044` accepted the final independently reproduced offline binary map.
  The only flag stores write 1; no value-2 setter exists. The dormant state-2
  path passes raw address 1 to an `int *` watchdog ioctl, ignores its result,
  and closes. Direct import coverage contains 53 close transfers; only
  `0x2d7d4` is proven to consume the watchdog open result. No mapped process
  termination path proves successful disable-before-exit. Exact live kernel
  config, FDT reset binding, init/respawn state, and command-child fd
  inheritance are the next read-only boundary; no signal or watchdog write is
  authorized yet.
- `scanner-045` accepted the exact standalone read-only acquisition helper at
  SHA-256
  `301cfa2a5b39549aa299918b24bc463a5d624f152acd5a8ca0eb50b3245cd114`.
  Its strict offline suite proves fixed command/path closure, bounded local
  config/FDT parsing, fd metadata-only inheritance checks, final budget
  reservation, and last-stat-clear ordering.
- `scanner-046` spent that one invocation without retry at evidence root
  `/tmp/scanner-045-evidence.aG7OZ6Aq` (manifest SHA-256
  `4044e9a65da32bbbb84d8688bfc39497b47d4b9ee09f349b72f8f09d4a11b312`).
  The S99 `cat` command succeeded, then its immediate staging selector and owed
  clear attempt both timed out. All 159 events and artifacts verify; release,
  close, identity and interfaces passed. The run preserved partial process/fd/
  init evidence but reached neither direct config/FDT recovery nor the final
  stat clear. Scanner-047 moves regular-file evidence before command-generated
  listings and adds bounded post-command settling entirely offline.
- `scanner-047` accepted the corrected helper/test after root and independent
  Codex review. Helper SHA-256 is
  `f1d7aaa7255a5c187b40957a384938c35995a527002a124d13dcbb09968f9874`;
  it validates scanner-046 lineage, performs the remote stat gate plus 54 direct
  reads before 12 commands, reads rcS/S98/S99 directly, and records one bounded
  settle before every staging selector. Scanner-048 may spend one live attempt;
  optical state remains deferred to the owner's return.
- `scanner-048` correctly stopped before a live attempt because the exact scanner
  was devnum 50 rather than the frozen helper's required 49 and interfaces 0/1
  had rebound to `uvcvideo`. All immutable hashes matched; no evidence directory,
  scanner-node open, USB operation, or retry occurred. Host kernel timestamps
  show scanner-046's two endpoint timeouts, then device 49 disconnecting about
  3.4 seconds later and the same identity returning as device 50 about 14.9
  seconds after that. The host evidence proves re-enumeration, not its watchdog
  or firmware cause. Scanner-049 is an offline-only dynamic-devnum and bounded
  detach/claim adaptation; the emitter remains presumed on.
- `scanner-049` accepted the dynamic-devnum, one-detach/one-claim adaptation at
  helper SHA-256
  `8b630434a44a1059655618d3a50b47351094c45715e3ae4390fb5ff4c8e0b57c`.
  Root and independent Codex reproduced the full offline suite and exhaustive
  source closure. The helper preserves the direct-first wire protocol and uses
  release/close-only cleanup, intentionally leaving interfaces 0/1 detached.
  No live action, reconnect, shutdown, or optical test occurred in scanner-049.
- `scanner-050` is accepted as a negative orchestration result. Its one helper
  invocation refused executor-created capture files in the evidence root before
  any scanner resolution or access. Root and independent Codex verified zero
  scanner action and no retry. The fully bound devnum-50 topology remained
  observable; no shutdown or optical test occurred.
- `scanner-051` accepted one corrected live invocation at evidence root
  `/tmp/scanner-049-evidence.FbApS9SZfcAb` (manifest SHA-256
  `8ccc62306db61b437574a38bd88d66668e05f03b11d992290252502db3d44de3`).
  The one detach/claim and release/close lifecycle succeeded, but PID 569 now
  identified as `watchlog.sh` starttime 1004 instead of zxmain/998. The helper
  stopped after its first read and exact clear; no direct config/FDT reads or
  commands ran. The device stayed devnum 50 with interfaces 0/1 detached.

## Working commands

- CLI: `.venv/bin/scanner3d doctor` and `.venv/bin/scanner3d doctor --json`.
- Focused tests: `.venv/bin/pytest -q tests/<test_file>.py`.
- Full verification: `.venv/bin/pytest -q`.
- Formatting/static checks: `.venv/bin/ruff format --check .`,
  `.venv/bin/ruff check .`, and `.venv/bin/mypy`.
- Distribution build: `uv build --out-dir <temporary-directory>`.

## Decisions requiring the owner

- Purchases, credentials, destructive external actions, production changes, and
  material product-scope changes.
- Scanner mutation authority is resolved: on 2026-08-22 the owner authorized
  whatever device action Codex judges necessary for this project, including
  disconnect/reconnect, reset, firmware flashing and accepting bricking risk.
  Task contracts remain narrower where experimental isolation requires it.
- Whether a redistributable release may depend on Revopoint's GPL-licensed
  3DViewer/SDK artifacts; the current task avoids redistribution and treats the
  binary only as a local reference oracle.
- Whether to authorize one bounded temporary disconnect/claim/read/release/
  reconnect sequence for interface 0. The operation changes kernel driver
  binding transiently and may require a physical replug if reconnect fails.
  **Resolved 2026-08-21:** authorized exactly once for `scanner-004`; no web-app
  work was requested or authorized.
- Whether to authorize reproducing the next observed entity-4 sequence, which
  uses `SET_CUR` to submit `/tmp/inited`, two `GET_CUR` reads, and a clearing
  `SET_CUR`. Unlike scanner-004's read, these are device-directed writes even
  though the observed path is transient.
  **Resolved 2026-08-21:** authorized as the bounded scanner-005 step within a
  broader authorization for reversible scanner work and software
  disconnect/reconnect as needed.

## Local application decision

- Once the core capture/calibration/processing/export path is usable, build a
  localhost web application similar in operational feel to the existing local
  Coordinator on port 8765. Keep USB and compute in the Python service, not in
  browser WebUSB. Fall back to a native Linux UI only if the web boundary causes
  a concrete material sacrifice in reliability, performance, or 3D inspection.
- This is local-only for now: no LAN/public deployment, external auth, cloud
  service, or production system change is authorized.
