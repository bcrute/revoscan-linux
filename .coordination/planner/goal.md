# Overall goal

- Goal ID: `linux-revopoint-mini-scanner-v1`
- State: `active`
- Starting ref: `unborn main; no commits`
- Target branch: `main`

## Objective

Deliver a native, offline-capable Linux scanning stack for the attached
Revopoint MINI that can acquire calibrated RGB-D evidence, register and fuse
multiple views, inspect the result, and export a dimensionally explicit mesh
and point cloud suitable for scanning the mapper chassis and components.

## Completion criteria

- Reproducible setup and diagnostics work on the recorded Linux Mint 22.3 host.
- The attached MINI is discovered by stable identity, and capture starts/stops
  without an unbounded hang while reporting frame timing, drops, and errors.
- Depth/RGB frames, factory or independently measured intrinsics/extrinsics,
  depth scale, units, and coordinate conventions are preserved with provenance.
- A real multi-view object capture can be registered and fused with inspectable
  quality/uncertainty evidence rather than a success-only status.
- Point-cloud export and at least one mesh export usable by normal CAD/mesh
  tools are demonstrated on physical scanner evidence.
- The mapper handoff is documented as a stable, versioned local artifact
  boundary; scanner transport remains independent from mapper product code.
- After the core scanning/export path works, a localhost web application offers
  device health, capture control, progress, artifact browsing, and 3D result
  inspection without putting USB access in browser JavaScript. If a web UI would
  materially compromise capture reliability, performance, or inspection, use a
  native Linux UI instead and document the concrete reason.
- Offline fixtures and automated tests cover the important acquisition,
  calibration, processing, and failure paths; all hardware claims name the
  exact physical evidence that supports them.

## Durable constraints

- Follow repository instructions and `.coordination/PROJECT.md`.
- Do not broaden external-action authority through subgoals.
- Prefer small, inspectable pipeline layers over a monolithic GUI.
- Scanner disconnect/reconnect, reset, firmware changes, undocumented device
  operations, and bricking risk are owner-authorized as of 2026-08-22 when a
  bounded task judges them necessary; preserve exact inputs, effects, and
  recovery evidence wherever the device remains observable.
- Any operation which can illuminate the blue depth laser/projector carries a
  non-optional final-state obligation: after V4L2 teardown, issue and evidence
  the exact laser-off command under bounded cleanup. A run is not successful if
  laser-off or subsequent driver restoration fails.
- An accepted off-command transfer is not evidence of optical shutdown. The
  owner reported the blue front emitter still on at idle on 2026-08-23; treat it
  as on until the owner visually confirms darkness. Safety recovery preempts
  capture and RGB control work, and driver restoration must not be performed
  blindly if it can reinitialize and relight the emitter.
- The long-term native path must not silently require Revo Scan.

## Forward delivery roadmap

- The detailed dependency order, delivery checkpoints, branch gates, evidence
  requirements, and proposed bounded task queue are maintained in
  `.coordination/planner/delivery-roadmap.md`.
- `current-task.md` remains the sole active executor assignment; the roadmap does
  not itself authorize future tasks or hardware/external actions.

## Owner decisions

- Redistributable licensing/dependency policy for any vendor SDK component is
  unresolved; diagnostic use does not authorize bundling it.

## Progress

- `scanner-001` accepted: Linux discovery/doctor foundation, bounded vendor-SDK
  oracle, stable hub/direct topology evidence, and exact SDK license blocker.
- `scanner-002` accepted: bounded usbfs observer/decoder and measured wire
  evidence through the SDK authorization failure.
- `scanner-003` accepted: descriptor-backed owned control query implemented;
  the one live request was stopped by `uvcvideo` ownership with `EBUSY` before
  any bytes reached the device.
- `scanner-004` accepted: one fixed owned version read proved the interface-0
  detach/claim/transfer/release/reconnect lifecycle and exact 60-byte response.
- `scanner-005` accepted: the exact four-transfer `/tmp/inited` exchange proved
  bounded owned writes, stateful reads, protocol clearing, and driver restoration.
- `scanner-006` accepted: offline static archaeology recovered exact XU/file
  primitives and calibration keys, bounded authorization claims, and separated
  ready calibration/V4L2 experiments from NOT-READY direct-stream/HID/LED work.
- `scanner-007` accepted: one bounded owned read retrieved and preserved the
  unit's complete 40-byte factory depth intrinsics, with protocol clear and
  driver restoration both verified.
- `scanner-008` accepted: one fixed two-transfer read retrieved the device's
  depth-scale scalar (`float32(1/200)`), preserving its unknown physical unit
  and proving clean driver restoration.
- `scanner-009` accepted: one bounded native V4L2/MMAP run captured the first
  complete owned raw depth frame (`Y16 ` 640x400, 512000 bytes), with exact
  UAPI evidence, clean resource release and no vendor/HID/usbfs operation.
- `scanner-010` accepted: the shared engine and bounded RGB/JPEG path landed;
  one MJPG 1280x800 attempt safely stopped before buffers/streaming because
  `uvcvideo` reported `sizeimage=8192000`, above the descriptor-derived
  2097152-byte ceiling. Cleanup and observation were clean; no RGB frame yet.
- `scanner-011` accepted: exact measured 8192000-byte RGB bounds allowed four
  buffers and STREAMON, but the only 8-second poll timed out with no DQBUF.
  Cleanup and 60-second observation were clean; the ceiling is no longer the
  blocker, while the reason the RGB function emits no buffer is unresolved.
- `scanner-012` accepted: offline UVC/module/SDK reconstruction eliminated RGB
  role, bandwidth/alt-setting and device-quirk explanations, largely eliminated
  missing S_PARM, and found uvcvideo `trace=0` had suppressed the decisive
  Probe/Commit/URB diagnostics. RGB-1 is specified but not run.
- `scanner-013` accepted: exactly one all-bits uvcvideo-traced RGB attempt again
  timed out, but proved successful URB completion with valid header-only UVC
  payloads and an FID transition, zero image-data packets for the dumped frame,
  no JPEG/V4L2 frame, verified trace restoration, cleanup and stable device.
- `scanner-014` accepted: one bus-specific usbmon observation proved successful
  26-byte Probe/Commit negotiation and five 16384-byte endpoint `0x84` bulk-IN
  submissions, but no successful completion or payload during the full poll;
  all URBs were cancelled at cleanup with zero actual length. Host restoration
  and device stability passed; an unpadded-devnum derivative bug is preserved as
  an evidence-packaging limitation and will be corrected offline.
- `scanner-015` accepted: the 26 scanner-014 target records are frozen with a
  padding-tolerant, one-to-one usbmon/UVC decoder, and an offline-tested paired
  lifecycle/CLI now captures the vendor application's depth-first, one-second,
  RGB-second ordering with fixed bounds, exact opt-in, conservative cleanup and
  consistent exits. No live paired backend or physical RGB claim exists yet.
- `scanner-016` accepted: the shared V4L2 engine now provides a concrete paired
  depth-first/RGB-second backend with stable pre/post-open identity checks,
  independent absolute deadlines, one dequeue maximum per role, bounded artifacts,
  typed timing/sequence evidence and exhaustive signal-shielded cleanup. All 1,133
  verification cases are synthetic; the earlier accidental strict-invalid RGB
  payload remains unaccepted until a separately bounded physical reproduction.
- `scanner-017` accepted: exactly one bounded physical paired reproduction delivered
  a complete raw depth observation and a second JPEG-coded RGB payload with exactly
  four opaque bytes after EOI. The strict RGB result remains `trailing_data`, its
  image is nearly black, cleanup passed, and stable identity returned after an
  unexplained devnum 44-to-45 re-enumeration. A multi-frame diagnostic is next.
- `scanner-018` accepted: the shared paired backend now supports one offline-tested,
  fixed three-observation RGB diagnostic with indexed byte-exact preservation,
  strict JPEG plus EOI-prefix/trailer evidence, cumulative timing/counts, truthful
  partial delivery and exhaustive cleanup across injected failures and signals.
  No physical multi-frame run has yet been made.
- `scanner-019` accepted: exactly one physical three-observation run delivered
  depth plus all three RGB payloads with clean teardown. Every RGB payload had a
  decodable 1280x800 JPEG prefix and four differing opaque trailing bytes, remained
  strict-invalid, and was almost black with no meaningful warm-up improvement.
  Devnum changed 45-to-46 for an unresolved reason and was then stable.
- `scanner-020` accepted: an offline-tested, dry-by-default standard V4L2 RGB-control
  inventory can query exactly six fixed gain/exposure/white-balance controls through
  QUERYCTRL/G_CTRL only, with a 12-call ceiling, stable identity checks and exhaustive
  typed failure/cleanup evidence. It has not yet been run physically.
- `scanner-021` accepted: the inventory ran physically exactly once with 11 read-only
  ioctls and clean stable teardown. Gain reported current/max 16/16, auto exposure
  menu index 3, auto white balance enabled, two dependent values inactive, and
  auto-exposure priority unsupported. No setting was changed.
- `scanner-022` accepted: pinned offline SDK and kernel archaeology proved V4L2
  auto-exposure index 3 maps to a raw UVC mode the SDK treats as enabled, separated
  the physical UvcCamera selector-0x12 exposure extension from the standard
  selector-4/log2 fallback, and bounded the next step as an offline implementation
  of in-stream read-only telemetry before any further physical run.
- `scanner-023` accepted: the separately gated paired RGB-series telemetry path is
  offline-tested to read exactly three fixed standard controls at no more than four
  in-stream points on the existing fd, with typed partial/error evidence, preserved
  frames and unchanged deadline/cleanup. It has not yet been run physically.
- `scanner-024` accepted: one physical telemetry series delivered depth plus three
  near-black/trailing-data RGB payloads while all 12 standard-control reads stayed
  invariant at gain 16, auto-exposure index 3 and absolute-exposure raw 131072.
  Cleanup passed; devnum changed 46-to-47 and then remained stable.
- `scanner-025` accepted: one dry-by-default fixed selector-0x12 RGB exposure
  extension GET_CUR is implemented through the established interface-0 lifecycle.
  Exact 60-byte raw evidence and four u32 words are preserved; SDK-derived
  `word0/1/2*1000` scaling remains explicitly unit-unknown. All 1,231 tests pass;
  the read has not been performed physically.
- Blue-light final-state intervention accepted: after the owner's explicit
  shutdown requirement, one exact laser-off `SET_CUR` (`0x922=0`) returned 60/60
  bytes, release/reconnect passed and devnum 47 plus all video nodes remained
  stable. There is no optical readback, so only command acceptance and host-side
  restoration are claimed.
- `scanner-026` accepted: every existing depth-STREAMON command now owns one
  post-V4L2 immutable laser-off finalizer, with the entire interval shielded,
  exact command acceptance separated from driver restoration, failed shutdown
  blocking success, and RGB/dry/refused paths inert. All 1,285 offline tests
  pass; the implementation task did not access the scanner and adds no optical
  darkness evidence.
- `scanner-027` accepted: the fixed selector-0x12 GET_CUR ran once physically
  and returned a typed 2/60-byte short response (`0000`) with clean cleanup, so
  no exposure mapping was decoded. The separately required one-shot blue-light
  off command was accepted 60/60 with clean restoration and stable devnum 47;
  neither operation was retried.
- `scanner-028` accepted: offline SDK/descriptor reconciliation proved the SDK
  requests one 60-byte selector-0x12 GET but zero-fills its buffer and discards
  libusb's nonnegative actual length before decoding. Entity 4 does not advertise
  selector 0x12, and physical `0000` remains only a 2-byte raw reply with no
  assigned exposure or status meaning. The next branch is a fixed `GET_LEN`
  implementation before any further physical read.
- `scanner-029` accepted: a dry-by-default, serial-gated UVC `GET_LEN` boundary
  can issue only `a185001200040200e8030000`, retain a two-byte raw response and
  full lifecycle evidence, and assign no semantics to the decoded little-endian
  number. All 1,314 offline tests and installed artifact checks pass. The next
  bounded task is one physical read followed by the mandatory immutable off
  command and restoration; no scanner-029 physical action occurred.
- `scanner-030` accepted: that GET_LEN ran exactly once and returned exact raw
  `0200`, reporting control length 2 with no assigned semantics. The mandatory
  separate off command then succeeded 60/60 as the last device transfer; clean
  restoration and three stable postflight samples retained devnum 47, all
  bindings/nodes and no holders. The next branch is a fixed GET_INFO boundary.
- `scanner-031` accepted: a dry-by-default serial-gated GET_INFO boundary can
  issue only `a186001200040100e8030000`, retain one raw byte and render its
  Linux-UAPI bit syntax without semantic overclaim. All 1,349 offline tests and
  installed artifact checks pass. One physical query plus mandatory off is next.
- `scanner-032` accepted: one GET_INFO returned exact raw `03`, syntactically
  reporting GET+SET with other bits clear, followed by one accepted 60/60 off
  finalizer and stable devnum 47 restoration. The attempt-marker read-only
  requirement was corrected only post hoc and is retained as an evidence
  limitation. A fixed raw range-query boundary is next.
- `scanner-033` safety override active: the owner's visible front-emitter
  observation supersedes the planned range query. Recover and physically verify
  a real shutdown path before any other scanner work resumes.
- `scanner-033` investigation accepted but shutdown failed: both SDK-exact
  `0x922=0` and SDK laser-LED `0xb04=0` were accepted 60/60 while detached, and
  the owner observed the front emitter still on after each. `scanner-034` now
  implements the owner's selected idle invariant: remove USB hub-port power.
- `scanner-034` invalidated before implementation: downstream hub port 1
  reported unpowered/disconnected and stopped scanner USB response, yet the
  owner visually confirmed the emitter stayed on. The hub leaves VBUS present.
  `scanner-035` was then canceled without a root-hub write when the owner
  clarified that the hub is externally powered, so cutting its upstream host
  link cannot be relied on to remove scanner power. `scanner-036` is tracing the
  embedded Linux shutdown path offline; physical unplugging remains the only
  proven safe idle state.
- `scanner-036` accepted: the pinned SDK exposes reboot but no supported
  embedded shutdown. Its otherwise-unreferenced, serial-matched reset-state
  helper sends `gpio117=0` for this unit. Under the owner's explicit reset and
  bricking authority, scanner-037 may test that exact command once, with visible
  darkness as the only acceptance evidence.
- `scanner-037` spent that one command at devnum 49 after restoring the
  downstream data path. `gpio117=0` was accepted 60/60, interface 0 was released
  but not reconnected, and no later scanner transfer occurred. Optical state is
  awaiting the owner's direct observation; treat the emitter as on until then.
- `scanner-038` accepted: exact live FDT evidence maps GPIO55/60 to image-sensor
  power-down, GPIO61 to sensor reset, and GPIO121/125 to the RK1608 pre-ISP
  reset/IRQ. No projector GPIO/regulator node or supported SDK shutdown exists;
  the running vendor services and `rk_preisp` handler are the next boundary.
- `scanner-039` accepted: live PID 569 `/tmp/zxmain`, its exact recovered ARM
  ELF, boot logs, init scripts, kernel symbols, and matching primary Rockchip
  source prove that textual `/dev/rk_preisp` `off` is the vendor's normal
  reversible RK1608 hardware power-down primitive. A post-off `q` is forbidden
  because reopening the misc device at count zero powers RK1608 back on.
  `scanner-040` may now issue the one exact off command and assess passive
  dmesg/GPIO evidence; optical confirmation still awaits the owner.
- `scanner-040` accepted its exact one-shot execution but did not shut RK1608
  down. The 60/60 `off` transfer added one driver count-decrement log while
  GPIO121 reset remained high with byte-identical before/after snapshots.
  Cleanup passed and no subsequent scanner transfer occurred. Under unchanged
  driver-count conditions, an additional live reference existed; scanner-041
  may prepare one separately bounded second decrement. If it does not drive
  GPIO121 low, stop without a third decrement or process signal.
- `scanner-041` accepted its exact second decrement but shutdown again failed.
  A second new driver count-decrement line appeared while GPIO121 reset stayed
  high with byte-identical pre/post snapshots. Cleanup passed, devnum 49 stayed
  stable, interface 0 remained unbound, and no later scanner transfer occurred.
  The original shared count was therefore at least three under unchanged
  semantics. Scanner-042 is an offline ownership audit; no third blind
  decrement or zxmain signal is allowed.
- `scanner-042` accepted: official source and immutable local evidence prove
  only an original shared-count lower bound of three and a current lower bound
  of one. Open V4L2 descriptors are not automatically count owners, and PID
  569's watchdog plus caught signals do not prove safe cleanup. Scanner-043 is
  a separately reviewed, read-only supervisor/watchdog metadata recovery; no
  third decrement or process signal is allowed.
- `scanner-043` accepted: one hash-gated read-only run preserved a complete
  558-transfer chain with clean interface release/close and stable identity and
  bindings. PID 569 remained the same 32-thread `/tmp/zxmain` process owning
  `/dev/watchdog` and `/dev/rk_preisp`; its recovered `watchlog.sh` only trims
  logs and does not supervise or restart zxmain. Firmware sysfs exposes no
  watchdog timeout/nowayout state, so scanner-044 must recover the exact
  watchdog-disable and exit paths offline before any cooperative teardown is
  proposed. No shutdown or darkness claim exists.
- `scanner-044` accepted: independently reproduced ARM32 analysis found only
  value-1 stores to the watchdog flag, no value-2 setter, and an invalid raw
  pointer in the dormant disable ioctl path. Across 53 direct close transfers,
  only the watchdog thread's shared close is proven to consume its fd, and no
  mapped normal or termination path proves disable-before-exit. Scanner-045 now
  prepares a separately reviewed read-only exact-unit config/FDT, supervision,
  and command-child fd-inheritance probe; no process action is yet authorized.
- `scanner-045` accepted: a standalone hash-gated helper now pins the exact
  config/FDT, init/supervision, and command-child fd-inheritance acquisition.
  Root plus independent Codex review accepted helper SHA-256
  `301cfa2a5b39549aa299918b24bc463a5d624f152acd5a8ca0eb50b3245cd114`;
  it makes no shutdown or darkness claim.
- `scanner-046` accepted its one permitted live invocation without retry. The
  helper preserved 159 hash-linked transfers and useful process/fd/init facts,
  then `select:command_s99_cat` and its exactly once clear attempt timed out.
  Release/close and host identity remained stable, but the final stat clear and
  direct config/FDT phase were not reached. Scanner-047 now prepares a direct-
  first collector offline; no shutdown or optical claim exists.
- `scanner-047` accepted: root and independent Codex review froze a direct-first
  helper at SHA-256
  `f1d7aaa7255a5c187b40957a384938c35995a527002a124d13dcbb09968f9874`.
  It gates remote stat, attempts 54 direct paths before 12 settled commands, and
  removes the three init-script `cat` commands that preceded scanner-046's
  timeout. Scanner-048 may invoke it exactly once without retry or shutdown.
- `scanner-048` accepted as a zero-attempt safety stop: immutable hashes matched,
  but the exact scanner had re-enumerated from devnum 49 to 50 and rebound
  interfaces 0/1 to `uvcvideo`, so no helper or scanner node was opened. Kernel
  journal evidence places the disconnect about 3.4 seconds after scanner-046's
  failed clear and the return about 14.9 seconds later. The reset mechanism is
  unknown; scanner-049 adapts the collector offline for dynamic devnum pinning
  and the established one-shot driver detach lifecycle.
- `scanner-049` accepted: root and independent Codex review froze the dynamic-
  devnum collector at SHA-256
  `8b630434a44a1059655618d3a50b47351094c45715e3ae4390fb5ff4c8e0b57c`.
  It preserves scanner-047's direct-first wire closure, adds one exact
  GETDRIVER/detach/claim/release lifecycle for the rebound topology, deliberately
  contains no reconnect, and passed exhaustive offline signal, identity,
  cleanup and prohibited-capability proof. Scanner-050 may invoke it once;
  shutdown and optical state remain deferred to the owner's return.
- `scanner-050` accepted-negative: its exact preflight passed, but executor
  capture files made the evidence directory non-empty before the sole helper
  invocation. The helper's first guard refused it before scanner resolution,
  open, ioctl, transfer or remote command. No retry occurred; the device stayed
  at devnum 50 with the fully bound topology. Scanner-051 is a separately
  assigned corrected invocation, not a continuation of scanner-050.
- `scanner-051` accepted-partial: one corrected invocation successfully
  detached/claimed, but the first exact `/proc/569/stat` read identified
  `watchlog.sh` starttime 1004 instead of the frozen `zxmain`/998 identity. The
  helper paid its clear, released/closed, and stopped before all 54 direct paths
  and twelve commands. Devnum 50 stayed stable with interfaces 0/1 unbound.
  Process discovery must now be adapted offline; no reset, shutdown or optical
  cause is claimed.

## Current blocker

Expanded 2026-08-22: the owner authorized autonomous scanner work including
disconnect/reconnect, reset, firmware flashing, undocumented device operations,
physical replug and accepting bricking risk. License bypass, unrelated host or
external-system mutation, deployment, purchases, credentials and repository
commits remain outside that device-specific authority unless separately granted.

No owner decision currently blocks a bounded shutdown intervention. After
scanner-051 left depth interfaces 0/1 detached, the returning owner again
directly observed that the blue front emitter remained on. Detached UVC is
therefore not optical shutdown. The owner is interested in a staged replacement
for the vendor's embedded firmware, but a blind flash is not justified: recover
the boot/update/recovery chain and actual emitter actuator first. Optical
darkness remains the final acceptance gate. Active tasks must not invoke sudo,
pkexec, PolicyKit or an operation that can prompt for unavailable privilege.
Repository commit authority and redistributable dependency/licensing policy
remain unresolved.
