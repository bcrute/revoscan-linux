# Delivery roadmap — Revopoint MINI scanner application

- Overall goal: `linux-revopoint-mini-scanner-v1`
- Roadmap state: `active`
- Last updated: `2026-08-23`
- Current task: `scanner-052` offline detached-topology and dynamic-process adaptation
- Product target: offline-capable localhost web application on Linux; native
  Linux UI only if the web boundary produces a measured material limitation

## Purpose and authority

This is the dependency-ordered delivery plan from the accepted scanner-001
through scanner-012 foundation to a usable scanning application. It complements
`docs/roadmap.md`:

- `docs/roadmap.md` remains the detailed technical history and experiment record;
- this file defines the forward delivery sequence, decision gates, intermediate
  products, and evidence required to advance;
- `.coordination/planner/current-task.md` remains the only active executor
  assignment. Entries here do not authorize hardware actions, privileged writes,
  commits, installs, network access, or scope changes.

Codex assigns and reviews one bounded implementation subgoal at a time. Exact
future task IDs and contracts are written just in time because the RGB branch and
physical calibration evidence can change the safest next action.

## Definition of a working application

The full v1 is working only when an owner can, from a reproducible local Linux
installation and without Revo Scan:

1. identify the attached MINI and see actionable health and permission status;
2. start, monitor, cancel, and finish a bounded scan without an unbounded hang;
3. preserve raw depth/RGB observations, calibration, timestamps, drops, errors,
   coordinate frames, units, and provenance in a versioned scan session;
4. turn a real multi-view capture into a registered/fused point cloud and mesh;
5. inspect the result and its quality evidence locally;
6. export dimensionally explicit artifacts usable in ordinary mesh/CAD tooling;
7. hand a versioned artifact to `../mapper` without coupling scanner transport
   into mapper; and
8. repeat the workflow offline through the localhost application.

A health dashboard, a one-frame capture button, or a fixture-only reconstruction
does not by itself satisfy this definition.

## Delivery checkpoints

| Checkpoint | User-visible result | Relation to full goal |
| --- | --- | --- |
| A — foundation | Installable CLI diagnoses the attached scanner and performs bounded one-shot experiments. | Achieved by scanner-001 through scanner-012. |
| B — depth geometry slice | Repeatable depth capture produces physically validated metric depth and a single-frame PLY. | Useful interim result; does not satisfy the required RGB-D or multi-view goal. |
| C — reconstruction CLI | A bounded CLI scan session registers/fuses multiple views and exports an inspectable point cloud and mesh. | Core pipeline complete; UI still absent. |
| D — local scanner app | Local web UI controls capture/processing and browses/inspects/downloads artifacts. | Product surface complete; still needs end-to-end acceptance. |
| E — v1 acceptance | A physical known object and mapper component pass the documented offline workflow and accuracy/quality gates. | Overall goal may be marked `done`. |

The owner may explicitly choose Checkpoint B as a depth-only interim priority if
RGB remains expensive to recover. That changes ordering, not the full v1
completion criteria; removing RGB-D from v1 would be a separate scope decision.

## Current baseline

| Layer | Evidence-backed state | Principal gap |
| --- | --- | --- |
| Repository/package | Python 3.14 wheel and sdist reproduce; CLI installs and offline gates are mature. `main` is unborn and all source is untracked. | Establish a durable VCS baseline when commit authority is granted; resolve release licensing and dependency policy. |
| Discovery/diagnostics | Stable identity, interface roles, modes, topology, and permission reporting exist. | Replace vendor-oracle-only frame/calibration health with owned checks. |
| Depth transport | Complete `Y16 ` 640x400 observations were captured through V4L2 in separate bounded runs, including scanner-017's paired run. | Continuous capture, requeue, cancellation, drops, disconnect recovery, metric semantics, and re-enumeration cause. |
| RGB transport | Scanner-024 reproduced three near-black 1280x800 JPEG prefixes with four-byte trailers while all twelve in-stream standard-control reads stayed invariant. Scanner-027's selector-0x12 GET returned only 2/60 raw bytes; scanner-030 measured GET_LEN raw `0200`; scanner-032 measured GET_INFO raw `03`. Scanner-040/041 decremented RK1608 count, scanner-043 recovered process/fd metadata, scanner-044 proved no working normal watchdog-disable path, and scanner-046 preserved partial process/fd/init evidence before a command-staging timeout. Scanner-048 then stopped before invocation after the endpoint timeouts were followed by device re-enumeration and driver rebound. | Adapt the direct-first collector offline for dynamic devnum pinning and one bounded detach/claim, then recover exact watchdog config/FDT and supervision evidence; design and owner-test cooperative teardown before resuming range/exposure work. |
| Calibration | Depth intrinsics and scalar `float32(1/200)` were retrieved once. | Physical unit, invalid convention, application to pixels, suspicious coordinate interpretation, RGB intrinsics, distortion, extrinsics, and validation. |
| Reconstruction/export | Not started. | Metric point cloud, registration, fusion, meshing, standard exports, quality evidence. |
| Application | No scanner HTTP service, web assets, native GUI, or 3D viewer exists. | Service/job layer, local UI, artifact catalog/viewer, packaging and operator workflow. |

## Dependency overview

```text
RGB-1 diagnosis ──> RGB recovery ───────────────┐
                                                ├─> synchronized RGB-D sessions ─┐
session/artifact contract ─> repeatable depth ─┤                                │
                                                └─> metric calibration ──────────┤
                                                                                 v
single-frame point cloud ─> registration ─> fusion ─> mesh/export ─> mapper boundary
                                                           │
                                                           v
                                           local service ─> web UI ─> v1 validation
```

The depth geometry slice may proceed after RGB-1 even if the RGB recovery branch
remains open. The service and UI consume the same session and processing APIs as
the CLI and should not invent a second capture path.

## Phase 0 — Repository and decision safety

### Objective

Make future reviewed changes recoverable and identify decisions that can block a
redistributable application.

### Work

- Obtain explicit authority and create an initial Git baseline containing the
  accepted scanner-001 through scanner-012 worktree. No commit is currently
  authorized by the active assignment.
- Reconcile setup documentation: the project currently defines a `dev` extra,
  while one durable command says `uv sync --all-groups`; document one verified
  path.
- Keep the vendor SDK external and diagnostic-only unless the owner makes a
  separate licensing/dependency decision.
- Before adding numerical, reconstruction, or browser dependencies, record their
  licenses, offline installation story, CPython 3.14 support, artifact size, and
  fallback/exit strategy.
- Decide whether Checkpoint B should be pursued immediately if RGB remains open.

### Exit gate

- The accepted baseline is recoverable in version control, or the owner has
  explicitly chosen to continue on the untracked worktree with that risk recorded.
- No unresolved licensing question is silently embedded in a runtime dependency.

## Phase 1 — Resolve the RGB observation gap

### 1A. Run RGB-1 (`scanner-013`, current)

- Await explicit owner authorization for the two temporary privileged host writes.
- Set `uvcvideo` trace from verified `0` to `0xFFFFFFFF`, perform exactly one
  unchanged bounded RGB capture, restore/read back `0` on every exit, and retain
  capped kernel evidence.
- Do not alter scanner state, mode, timeout, V4L2 operation order, vendor/XU/HID
  state, or retry count in this experiment.

**Accepted 2026-08-22.** The attempt produced no RGB frame, but the all-bits
trace proved successful valid header-only payload completion and an FID toggle.
The driver has no successful Probe/Commit byte or generic completion trace
callsite, so absence of those lines is not evidence that the operations failed.

### 1B. Classify the observed branch

| RGB-1 evidence | Next bounded branch |
| --- | --- |
| Valid payload and completed frame appears | Preserve the RGB artifact, validate JPEG structure/content, then reproduce delivery without diagnostic tracing. |
| URBs contain bytes but no frame completes | Isolate UVC payload/FID/decode behavior; evaluate a narrowly justified kernel quirk or owned user-space UVC decoder. |
| URBs complete empty, stall, or never complete after COMMIT | Treat missing device-side pre-roll as leading; recover the exact higher-level vendor call ordering before proposing any write. |
| Probe/Commit fields are inconsistent | Reproduce the negotiated block, compare with the descriptor and accepted depth control, then change exactly one standard negotiation input. |
| No URB submission is logged | Investigate the host driver path first; direct usbfs streaming remains gated on an explicit stop/restoration design. |

Scanner-013 selects a narrower form of the header-bytes branch: valid UVC
headers and FID transition arrived, but the dumped prior frame contained one
header-only payload and zero image data. Scanner-014 therefore observes the
unchanged attempt with usbmon to recover the successful Probe/Commit blocks and
actual endpoint `0x84` completion lengths/statuses before any device command.

### 1C. RGB transport exit gate

- Owned code captures and preserves at least one structurally valid RGB frame.
- A second separately assigned/reviewed run demonstrates that the result is not
  a one-off diagnostic side effect.
- Cleanup, device identity, bindings, nodes, and trace restoration are verified.
- The exact mechanism and any remaining initialization dependency are documented.

This phase is the largest schedule uncertainty. The owner granted general device
authority on 2026-08-22, including unobserved commands, persistent scanner
mutation, firmware action, direct ownership and physical reconnect/replug with
bricking risk accepted. The roadmap itself still does not spend that authority:
each such action must be justified and bounded in `current-task.md`.

Scanner-017 reproduced paired byte delivery once under the accepted backend, but
did not pass this exit gate: its only RGB dequeue was strict `trailing_data` and
visually nearly black. Scanner-018 accepted the finite multi-frame diagnostic
offline. Scanner-019 then ran it exactly once: all three RGB observations arrived
with decodable JPEG prefixes plus four opaque bytes and clean teardown, but all
were almost black and strict-invalid. Scanner-020 therefore builds a fixed read-
only standard-control inventory before any exposure/gain hypothesis is tested.
Scanner-020 is now accepted offline; scanner-021 spends exactly one read-only
physical inventory and permits no SET, stream, retry or privilege escalation.
Scanner-021 completed cleanly. Its state does not by itself identify a safe
setting change. Scanner-022 resolved the static meaning: index 3 maps to raw UVC
mode 8, which the SDK reports as enabled, while the physical extension getter is
a separate selector-0x12 path. Scanner-023 now implements the fixed read-only
telemetry boundary offline. Scanner-023 is accepted; scanner-024 now authorizes
exactly one unprivileged physical series with that telemetry enabled and no
setting change or retry.

Scanner-024 completed that one run: all standard values were stable across the
four points while the frames remained near-black. The depth extension unit does
not advertise selector 0x12 in its descriptor bitmap, so an in-place
`UVCIOC_CTRL_QUERY` is not an evidence-backed substitute for the SDK's direct
request. Scanner-025 therefore implements the exact standalone one-transfer
usbfs read offline, reusing the accepted interface-0 restoration lifecycle.

## Phase 2 — Versioned scan-session and artifact contract

### Objective

Create the durable boundary used by capture, processing, UI, tests, and mapper.
This lands before continuous capture so evidence is not left only in `/tmp` or
spread among command-specific JSON formats.

### Work

- Define a versioned session manifest with an explicit schema version.
- Preserve device identity, USB topology snapshot, software version, capture
  profiles, negotiated formats, calibration bytes and decoded values, timestamps,
  sequence numbers, drops, errors, cleanup, units, axes/handedness, transforms,
  and source provenance.
- Represent session and stage state explicitly: planned, running, cancelled,
  failed, incomplete, and complete. Never infer success from artifact presence.
- Define stable paths for immutable raw observations, derived artifacts, logs, and
  checksums. Use atomic complete-or-fail writes and never overwrite raw evidence.
- Define compatibility behavior for unknown schema versions and partially written
  or interrupted sessions.
- Provide fixture builders/readers and a redacted manifest suitable for support
  reports.

### Exit gate

- Fixture round trips are deterministic and reject malformed, unsafe, truncated,
  path-traversing, or unsupported-version input.
- An interrupted synthetic session remains inspectable without being called
  complete.
- The command-line capture path can emit the new manifest without weakening its
  existing bounds.

## Phase 3 — Repeatable bounded acquisition

### 3A. Depth session engine

- Generalize the accepted one-dequeue backend into a bounded session lifecycle:
  open, negotiate, allocate, queue, start, poll/dequeue/requeue, stop, release.
- Bound duration, frame count, queue memory, artifact bytes, shutdown time, and
  retained logs; make cancellation explicit and idempotent.
- Record monotonic and device timestamps, buffer sequence values, measured FPS,
  missing sequences, late/short/error frames, queue pressure, and cleanup.
- Serialize ownership so two commands or browser requests cannot control one
  function concurrently.
- Detect disappearance/re-enumeration and produce recovery guidance without an
  implicit retry or device reset.
- Preserve representative raw frames and metadata according to policy; do not
  require retaining every frame to report every frame's timing/drop evidence.

### 3B. RGB session engine

- Integrate the accepted RGB delivery mechanism into the same lifecycle without
  duplicating cleanup/state logic.
- Decode MJPEG into a documented color representation while always retaining or
  hashing the original compressed observation needed for provenance.
- Treat malformed JPEG, unexpected dimensions, excessive size, and decoder failure
  as typed outcomes.

### 3C. Dual-stream timing

- Measure the relationship between depth and RGB clocks and sequence behavior.
- Define a pairing policy and report the observed skew distribution, unpaired
  frames, and drops. Do not claim hardware synchronization unless measured.
- If reliable RGB-depth pairing is impossible, preserve that limitation and keep
  geometry processing independent of color.

### Hardware exit gate

- Multiple independently started sessions complete without an unbounded hang.
- Predeclared duration/frame-count trials report delivered frames, drops, timing,
  resource bounds, cancellation latency, and cleanup rather than only success.
- A later task must choose the exact repetition count and duration before touching
  hardware; this roadmap does not spend those attempts.
- Any deliberate unplug/replug test requires separate owner authority.

## Phase 4 — Complete and validate calibration

### 4A. Retrieve remaining factory inputs

- Implement separate closed-whitelist reads for `Prgb.bin`, `LC_RT.bin`, and
  `Distort.bin`, one reviewed key/protocol expansion at a time.
- Preserve raw bytes and hashes even when a decoder rejects the content.
- Decode structure only from measured evidence; never normalize an odd value into
  a plausible one without retaining and explaining the original.

### 4B. Resolve depth semantics

- Determine endianness, invalid/zero/saturation conventions, effective image
  region, and the physical meaning of the `0.005` scalar.
- Resolve whether the unusually off-center recorded principal point reflects a
  crop, sensor coordinate convention, record interpretation, or bad calibration.
- Use known physical targets and predeclared measurement conditions because the
  vendor SDK oracle remains licence-blocked.

### 4C. Metric and RGB-D model

- Implement raw-depth-to-metric conversion with explicit units and invalid masks.
- Implement distortion handling and depth/RGB camera models.
- Define named coordinate frames and transformations, including handedness,
  matrix convention, and units.
- Align/colorize depth only after extrinsics and timing evidence are valid.

### Exit gate

- A captured frame converts to metric coordinates and agrees with known physical
  dimensions within a tolerance declared before the validation run.
- Calibration provenance is sufficient to reproduce the result from raw session
  data offline.
- Invalid samples and uncertainty/limitations are visible in both APIs and
  artifacts.

## Phase 5 — Single-frame point-cloud vertical slice

### Objective

Deliver the first dimensionally meaningful geometry artifact before attempting
multi-view reconstruction.

### Work

- Run a bounded dependency spike for numerical/point-cloud candidates on CPython
  3.14, offline installation, licensing, memory use, determinism, and packaging.
- Keep deprojection and coordinate conventions in a library layer independent of
  CLI and UI.
- Generate points from metric depth and intrinsics; exclude invalid samples with
  explicit counts/reasons.
- Add bounded filtering and optional colorization as separate, provenance-recorded
  stages; raw points remain recoverable.
- Export a standard PLY point cloud with units, frame, calibration hash, and source
  session referenced in its companion manifest.
- Add synthetic geometry fixtures with analytically known expected points.

### Exit gate — Checkpoint B

- A real owned depth frame produces an inspectable PLY without vendor tooling.
- Known distances/planes in the cloud meet the Phase 4 validation tolerance.
- The result opens in an ordinary viewer and its orientation, scale, point count,
  invalid count, and limitations are documented.

## Phase 6 — Multi-view registration

### Work

- Define the initial pose strategy from available evidence: geometry-only
  registration first, with RGB features used only when calibrated and useful.
- Implement bounded preprocessing, correspondence search, coarse alignment, and
  refinement as separable stages.
- Record initial transform, final transform, residual distribution, inlier/overlap
  measures, convergence reason, and rejection reason for every pair.
- Build and optimize a pose graph for more than two views; preserve rejected edges
  and loop-closure evidence.
- Detect tracking loss and refuse to merge low-confidence views silently.
- Provide synthetic transforms and adverse fixtures before physical validation.

### Exit gate

- A predeclared real multi-view known-object sequence registers into one consistent
  frame within a stated measured tolerance.
- Failure and weak-overlap cases produce inspectable diagnostics, not a plausible-
  looking success artifact.

## Phase 7 — Fusion, surface reconstruction, and export

### Work

- Select a fusion representation only after measuring capture density, noise,
  memory, and registration quality; record why it fits this scanner and host.
- Bound voxel/resolution choices, total memory, processing duration, cancellation,
  and intermediate storage.
- Preserve per-view contribution/coverage and expose holes, low-confidence regions,
  clipping, and discarded samples.
- Reconstruct a surface, orient normals, and make cleanup/smoothing explicit
  optional transforms rather than invisible defaults.
- Export the fused point cloud and at least one ordinary mesh format such as PLY or
  OBJ, with a companion manifest naming units, axes, transforms, calibration,
  source session, parameters, software versions, and checksums.

### Exit gate — Checkpoint C

- A physical multi-view scan exports a point cloud and mesh that open in ordinary
  tooling without vendor software.
- Known dimensions meet a tolerance declared before scanning.
- Quality evidence makes incomplete coverage, tracking loss, or uncertain geometry
  visible.

## Phase 8 — Mapper artifact boundary

### Work

- Inspect `../mapper` as a consumer, without importing scanner transport code.
- Define the smallest versioned artifact contract mapper actually needs: geometry,
  units, origin/axes, transforms, uncertainty/quality, provenance, and checksums.
- Add producer validation here and a consumer-boundary fixture or validation
  command that does not require either repository to reach into the other's
  internals.
- Specify forward/backward compatibility and rejection of unsupported versions.

### Exit gate

- A scanner-produced artifact crosses the boundary through documented local files
  and mapper can validate or consume it without knowing about USB/V4L2.

## Phase 9 — Local application service

### Architecture

- Bind loopback only. Python owns USB, capture, processing, and filesystem access;
  browser JavaScript never uses WebUSB or direct device paths.
- Expose one canonical application/service layer used by both CLI and web routes.
- Serialize hardware control and processing jobs with explicit state transitions,
  progress, cancellation, time/resource bounds, and recovery after server restart.
- Stream state/progress through a bounded event mechanism while keeping durable
  truth in session manifests.
- Serve only catalogued artifacts through identifier-based routes; never accept an
  arbitrary filesystem path from the browser.
- Enforce Host/Origin checks and CSRF protection for control actions even on
  localhost. LAN/public exposure, external authentication, and cloud services stay
  out of scope.

### Minimum service capabilities

- Device/permission/owned-capture health.
- Create/list/open scan sessions.
- Start, stop, cancel, and observe capture and processing jobs.
- Browse stage results, evidence, errors, and limitations.
- Download manifests, point clouds, meshes, and selected raw evidence.
- Recover and display incomplete sessions after restart.

### Exit gate

- API/contract tests cover success, refusal, concurrent requests, cancellation,
  process restart, malformed identifiers, path traversal, disconnect, stage
  failure, and cleanup.
- Fake-device end-to-end tests exercise the complete state machine offline.
- No route can broaden a capture profile, device command, or filesystem target
  beyond the reviewed service contract.

## Phase 10 — Local web UI and 3D inspection

### Views

- **Health:** scanner identity, roles, permissions, modes, active holders, and
  actionable remediation.
- **Capture:** explicit profile, safety bounds, start/stop/cancel, frame/timing/drop
  telemetry, and current limitations.
- **Processing:** stage progress, parameters, resource use, quality measures,
  warnings, failure, and cancellation.
- **Sessions/artifacts:** immutable raw evidence and derived versions, metadata,
  checksums, downloads, and comparison.
- **3D inspect:** point/mesh rendering, axes, units, clipping, measurement tool,
  camera reset, quality/coverage overlays, and visible uncertainty/invalid regions.

### Product constraints

- Vendor browser assets locally; the installed app must work with networking
  disabled and must not depend on a CDN.
- Select a default port that does not silently collide with the Coordinator;
  support an explicit port and ephemeral-port mode.
- Keep browser memory bounded for large artifacts through sampling, chunking,
  levels of detail, or server-derived previews while retaining full downloads.
- Never imply completion or accuracy when the underlying session/stage is partial.
- Provide keyboard-accessible controls and legible error/recovery text.

### Exit gate — Checkpoint D

- From the browser, the owner can diagnose the attached scanner, create and control
  a scan, observe progress/failure, inspect the resulting point cloud/mesh, and
  download the exact artifacts.
- Refresh/restart does not lose durable session truth or leave hardware streaming.
- The full workflow runs locally with network access disabled.

## Phase 11 — Installation and operator readiness

### Work

- Add a packaged launcher such as `scanner3d serve` using the same installed
  service/application layer as tests.
- Document and test supported Python/OS/kernel boundaries and a fresh offline
  installation path.
- Provide least-privilege video/usb access guidance and diagnostics; do not require
  running the whole web service as root.
- Bundle versioned web assets, migrations/schema compatibility, and example/local
  configuration in the distribution.
- Define logs, disk-retention policy, cleanup, backup/export, and support bundle
  behavior.
- Resolve the project license and every runtime dependency's redistribution terms
  before calling the artifact redistributable.

### Exit gate

- A fresh supported Linux account can install, run preflight, launch the local app,
  stop it cleanly, and reopen prior sessions using documented commands.
- Wheel/sdist contents, reproducibility, leak checks, and installed smoke tests pass
  with the UI and processing stack included.

## Phase 12 — Physical end-to-end acceptance

### Validation sequence

1. Record exact host, scanner serial/firmware, cable/topology, calibration hashes,
   environment, target, lighting, motion, and chosen bounds.
2. Run a known-object scan from application preflight through capture, processing,
   inspection, and export.
3. Compare declared dimensions against independent physical measurements under a
   predeclared tolerance; retain residuals and failure regions.
4. Open exported point cloud and mesh in ordinary external tooling.
5. Scan at least one mapper chassis/component and validate the mapper artifact
   boundary.
6. Exercise cancellation and one non-destructive failure/recovery case. Physical
   disconnect/replug remains separately authorized.
7. Repeat the complete workflow with network access disabled.

### Exit gate — Checkpoint E

All criteria in `.coordination/planner/goal.md` have direct evidence. Only Codex may
then set the goal to `done` and write `reviews/completion.md`.

## Proposed bounded task queue

This queue is directional rather than a promise of exact task numbering. Each row
becomes a separately reviewed assignment, and a failed gate may insert a narrower
diagnostic task before the next row.

| Order | Bounded deliverable | Depends on |
| --- | --- | --- |
| Accepted | `scanner-014`: one unchanged RGB attempt under bounded usbmon observation | Accepted scanner-013 header-only result |
| Accepted | `scanner-015`: freeze/decode RGB evidence and implement the depth-first/RGB-second experiment path offline | Accepted scanner-014 no-completion result |
| Accepted | `scanner-023`: implement fixed in-stream RGB standard-control telemetry offline | Accepted scanner-022 archaeology |
| Accepted | `scanner-024`: run the accepted telemetry series physically exactly once | Accepted scanner-023 telemetry path |
| Accepted | `scanner-025`: implement the fixed selector-0x12 extension read offline | Accepted scanner-024 invariant standard telemetry |
| Accepted | `scanner-027`: run the selector-0x12 extension read once and command the blue light off | Accepted scanner-025/026 boundaries |
| Accepted | `scanner-028`: reconcile the physical short reply with the SDK and descriptor offline | Accepted scanner-027 evidence |
| Accepted | `scanner-029`: implement a fixed selector-0x12 `GET_LEN` reader offline | Accepted scanner-028 reconciliation |
| Accepted | `scanner-030`: run the fixed selector-0x12 `GET_LEN` once and command the blue light off | Accepted scanner-029 boundary |
| Accepted | `scanner-031`: implement a fixed selector-0x12 `GET_INFO` reader offline | Accepted scanner-030 raw length evidence |
| Accepted | `scanner-032`: run the fixed selector-0x12 `GET_INFO` once and command the blue light off | Accepted scanner-031 boundary |
| Suspended | selector-0x12 raw range reads | Owner-observed blue emitter remains on at idle |
| Accepted | `scanner-036`: trace an embedded shutdown path offline | Externally powered hub invalidates host-port power cuts |
| Inconclusive | `scanner-037`: test the one serial-matched reset-state command | No owner-confirmed optical result |
| Accepted | `scanner-038`: recover the embedded projector-power actuator | No direct GPIO/regulator mapping; RK1608/pre-ISP boundary identified |
| Accepted | `scanner-039`: recover the running RK1608/pre-ISP shutdown handler | Exact vendor service/driver control recovered from live process, binary, kernel and primary source |
| Negative | `scanner-040`: issue one exact RK1608 hardware-off command and retain passive state evidence | Count decremented but GPIO121 stayed high; execution accepted, shutdown not achieved |
| Negative | `scanner-041`: prepare, review, and run one conditional second RK1608 decrement | Second count decrement accepted but GPIO121 stayed high; original count was at least three |
| Accepted | `scanner-042`: audit RK1608 misc/V4L2 reference owners offline | Established count lower bounds and left exact supervisor cleanup unknown |
| Accepted | `scanner-043`: recover supervisor/watchdog/process metadata read-only | Same zxmain identity, watchdog/pre-ISP fds, log-only watchlog, stable cleanup; watchdog state still unknown |
| Accepted | `scanner-044`: recover watchdog-disable and exit control flow offline | No normal value-2 setter or proven disable-before-exit path exists |
| Accepted | `scanner-045`: prepare an exact-unit config/FDT, supervision, and fd-inheritance probe offline | Accepted scanner-044 negative teardown result |
| Partial | `scanner-046`: run the accepted exact-unit read-only probe once | Accepted one attempt; S99 selector/clear timed out before direct config/FDT reads |
| Accepted | `scanner-047`: revise the probe offline for direct-first recovery and bounded command settling | Accepted frozen helper after corrected orchestration tests |
| Zero attempt | `scanner-048`: run the accepted direct-first helper exactly once | Correctly stopped after autonomous re-enumeration changed devnum and rebound interfaces 0/1 |
| Accepted | `scanner-049`: adapt the helper offline for dynamic devnum and bound-driver ownership | Frozen helper preserves scanner-047 wire closure and leaves interfaces 0/1 detached |
| Negative | `scanner-050`: run the accepted dynamic-devnum direct-first helper exactly once | Helper refused executor-populated evidence root before scanner access; no retry |
| Partial | `scanner-051`: make one separately assigned corrected invocation with a truly empty helper-owned root | First read proved PID 569 is now watchlog.sh; cleanup passed and interfaces 0/1 remain detached |
| Current | `scanner-052`: adapt fixed-first collection and dynamically discover the current zxmain offline | Accepted scanner-051 PID drift and stable detached topology |
| 2 | Versioned scan-session manifest and atomic artifact store | Existing JSON/evidence models |
| 3 | Repeatable bounded depth session engine | Session contract |
| 4 | First RGB frame and repeatability proof | RGB branch resolution |
| 5 | Remaining factory calibration key, one closed whitelist at a time | Existing ownership lifecycle |
| 6 | Depth semantics and physical calibration protocol | Known target and captured frames |
| 7 | Metric depth conversion and calibration fixtures | Validated semantics |
| 8 | Single-frame point cloud and PLY export | Metric depth; dependency spike |
| 9 | RGB decoding, extrinsics, alignment, and timing evidence | RGB frames and calibration |
| 10 | Pairwise registration with quality/failure evidence | Point clouds |
| 11 | Pose graph and multi-view physical validation | Pairwise registration |
| 12 | Fusion, mesh generation, and standard exports | Accepted registration |
| 13 | Mapper artifact contract and boundary validation | Stable exports |
| 14 | Local service and job/session API | Stable capture/processing APIs |
| 15 | Web controls, artifact browser, and 3D viewer | Accepted service contract |
| 16 | Packaging, permissions, offline install, and recovery | Integrated application |
| 17 | Known-object and mapper-component end-to-end acceptance | All prior gates |

Calibration reads may be split into multiple tasks; registration, fusion, and UI
will almost certainly be split further. No task should combine a novel hardware
mutation with a broad product-code change.

## Decision gates owned by the user

| Decision | Needed by | Current state |
| --- | --- | --- |
| Authorize RGB-1's temporary root host writes | Current task | Authorized 2026-08-22 for scanner-013 exactly as bounded in `current-task.md` |
| Authorize an initial Git commit | Before relying on VCS rollback/history | Not authorized |
| Prioritize a depth-only interim checkpoint if RGB stays open | After RGB-1 classification | Optional; full v1 still requires RGB-D |
| Approve any persistent/unknown scanner mutation, firmware action, direct-stream ownership risk, or physical replug | Only if a future branch requires it | Pre-authorized 2026-08-22 when Codex judges it necessary; bound each action in the active task |
| Choose/approve a physical calibration target and acceptance tolerance | Before metric validation | Open |
| Approve runtime dependency and redistribution policy, including any vendor component | Before dependency lock/release | Vendor SDK redistribution unresolved |
| Define mapper's required artifact fields if repository inspection cannot settle them | Before Phase 8 acceptance | Open |
| Authorize commit, publish, deploy, or non-local exposure | When separately requested | Not authorized |

## Principal risks and responses

| Risk | Trigger/evidence | Response |
| --- | --- | --- |
| RGB requires undocumented device initialization | No payload after valid COMMIT/URB submission | Recover exact call ordering first; only propose reversible, observable writes with restore evidence. Preserve depth-only progress without weakening full v1. |
| Metric depth cannot be inferred from stored scalar | Known-target measurements contradict candidate interpretation | Preserve raw values, test alternate documented/measured conventions one at a time, and consider independent calibration rather than guessing. |
| Factory calibration record is misinterpreted | Off-center principal point or projections fail physical checks | Recover crop/coordinate semantics and validate raw record layout before using it. |
| Scanner or driver is not repeatable | Drops, hangs, or re-enumeration recur | Keep sessions bounded, capture kernel/device evidence, make recovery explicit, and do not hide instability behind retries. |
| CPython 3.14 lacks a suitable packaged 3D dependency | Candidate cannot install offline or fails fixtures | Prefer a smaller compatible dependency set, isolate interfaces for replacement, or implement only bounded primitives whose maintenance cost is justified. |
| Reconstruction looks plausible but is wrong | Residuals, dimensions, or coverage disagree | Gate every stage on quantitative evidence and retain rejected/uncertain regions. |
| Browser cannot handle full-resolution geometry | Memory/frame-time measurement exceeds bounds | Use server-derived previews/LOD/chunks while full processing and artifacts remain in Python. Native UI becomes eligible only after a measured web limitation. |
| Untracked worktree loses provenance | Overlapping edits or inability to identify a baseline | Establish the owner-authorized initial commit before broad implementation. |
| Vendor licensing blocks release | Required runtime path depends on unredistributable SDK | Keep the owned path independent; do not ship or silently require the vendor binary. |

## Evidence required at every accepted phase

- Exact source/worktree identity; until an initial commit exists, explicitly state
  the unborn/untracked limitation.
- Offline tests for successful, malformed, timeout, cancellation, cleanup, and
  resource-bound paths relevant to the phase.
- Hardware evidence only for claims actually exercised on the named scanner, with
  commands, attempt count, conditions, raw outputs/hashes, timing, and cleanup.
- A clear separation of fact, inference, hypothesis, and not-run checks.
- Reproducible package/build evidence after package contents or dependencies change.
- No claim that fixtures establish physical accuracy or hardware support.
- No new executor assignment until Codex reviews the complete prior handoff.

## Explicitly out of scope for v1 unless the owner changes it

- Licence bypass. Scanner firmware, reset, destructive/persistent device
  mutation, direct ownership, arbitrary vendor commands and replug are permitted
  only through explicit bounded tasks under the owner's 2026-08-22 authority.
- Browser WebUSB or giving browser input direct device/filesystem authority.
- LAN/public hosting, cloud services, external authentication, multi-user operation,
  or production deployment.
- Silent runtime dependence on Revo Scan or a bundled vendor SDK.
- Multiple simultaneous scanners unless the owner adds that product requirement.
- Scanner transport code inside the mapper repository.
