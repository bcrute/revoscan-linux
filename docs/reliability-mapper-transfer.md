# What this scanner research can contribute to the mapper

Recorded 2026-09-09 after the operator clarified that the scanner work also
serves the future mapper project. This adds a research-value interpretation;
it does not revise earlier results, declare the scanner reliable, authorize
mapper-repository changes, or resume the paused M27 batch.

The initial synthesis did not inspect the mapper's sensor, scale, motion regime
or architecture. The subsequent sibling-repository review below supplies that
context and narrows the proposed transfer. None is a measured cross-project
result. Evidence demonstrated on this MINI remains distinct from claims that
require another dataset or sensor.

## Sibling-repository review: different sensor and error budget

Read-only inspection on 2026-09-09 of `/home/ben/repositories/mapper`, HEAD
`ee9a7480d85b0262889a5922c54081cab5603a6f`. The inspected authority/code paths
have no reported worktree changes. No mapper files, runtime, workflow or hardware
were changed, and M27 remains paused.

The [current hardware baseline](/home/ben/repositories/mapper/docs/hardware-options.md:29)
plans a Livox Mid-70-class LiDAR, FLIR Blackfly S global-shutter color camera,
ZED-F9P-class RTK GNSS, raw IMU and timing MCU. These are planned sensor classes,
not purchased/working/field-validated equipment according to that document.
The generic spatial result is currently a point-layer map; temporary cone-course
reconstruction is its first application, not the only possible future use.
[Core boundary](/home/ben/repositories/mapper/docs/spatial-mapping-core.md:28).

The existing [pose estimator](/home/ben/repositories/mapper/src/course_mapper/observe/pose.py:1)
uses GNSS position, course-over-ground heading and integrated IMU yaw during
low-speed gaps. It is not the MINI's depth-only ICP tracking stack or a complete
tightly coupled GNSS/INS/LiDAR system. Its documentation explicitly retains
heading degradation during poor fixes and absence of a GNSS-denied estimator.
Those are current implementation limitations, not predictions that a LiDAR will
be faulty or that every future mapper architecture will have the same failure.

The [field plan](/home/ben/repositories/mapper/docs/field-testing.md:193) labels all
thresholds as provisional, not achieved physical accuracy. Representative
rectangle gates are 0.15 m side-length error, 0.20 m diagonal error and 0.25 m
landmark p95 after rigid alignment. The longer loop uses 0.30 m closure and
landmark-p95 gates. These are specific metrics for particular course tests,
not a universal claim of centimetre accuracy or a specification for a dense mesh.
For context, Livox specifies Mid-70 range precision of <=2 cm (1 sigma at 20 m)
under stated target/temperature conditions and detection ranges under 100 klx.
Those sensor figures do not establish this mapper's map accuracy.
[Manufacturer specifications](https://www.livoxtech.com/mid-70/specs).

Therefore, the relevant distinction is measurement method and system design,
not just blue versus another wavelength. The MINI's projector/exposure-depth
tradeoffs, fringe-mode firmware investigation and image-pixel normal radii have
little direct applicability to the planned LiDAR rig. Successful MINI settings
should not become mapper defaults. The scanner's existing geometric gate uses
1.5 nominal mm, not the operator-mentioned 0.02 mm advertising figure, and some
rejections concern gross misalignment rather than a tiny loss of detail.

Rough mapping relaxes small measurement-error requirements, but does not remove
wrong-frame placement, incorrect landmark association, reference drift, timing
or calibration errors. Independent GNSS/inertial evidence can help constrain
motion; whether it does so accurately depends on real timing and calibration.
Livox itself documents distortion when points acquired during motion are treated
as one instantaneous scan, and describes motion compensation using point times
and motion estimates. This is a relevant LiDAR-specific research direction,
not proof that MINI optical distortion and LiDAR scan distortion are identical.
[Livox motion-distortion explanation](https://www.livoxtech.com/showcase/211220).

Illustrative error arithmetic, not measurements of either rig:

- A 0.5-degree yaw error gives `20 * sin(0.5 degrees) = 0.17453 m` lateral
  displacement for a point initially straight ahead at 20 m.
- At 3 m/s, a 50 ms timestamp offset gives `3 * 0.05 = 0.15 m` translation error
  under constant straight-line motion, before any rotational contribution.

These are material against the mapper's proposed decimetre-scale gates even
with accurate individual range measurements. Missing cones, duplicate landmarks,
bad heading during GNSS degradation and uncompensated scan motion deserve more
mapper-specific attention than submillimetre normal tuning on this test object.

Revised research priority: retain all MINI findings, but prioritize reusable
timing/calibration, trajectory/association, uncertainty and replay evidence over
additional sensor-specific sweeps when the intended beneficiary is Mapper.
Finishing every MINI mechanism is not a prerequisite to Mapper development.
Any later dense LiDAR-registration work needs its own sensor-appropriate corpus
and error budget; these course-point results do not qualify it in advance.

Inspection identities (SHA-256): hardware baseline
`0e2900e68657d2fa72c9b87e9b2becfcd0163d11fb88f038cef6f5fb6840c9b3`;
field plan `e8a3a76383ebf5e65ef6aae79d6783be195f09572f51cf72b02c345bfe1016ed`;
pose estimator `6d0f0044db2b6dccf36ccb4343c398174c9c4a360a3f2849240e58067ec25c02`.

### Solid-state does not establish simultaneous point acquisition

The operator subsequently asked whether solid-state LiDAR acquires every point
at once. The construction label alone does not establish that timing behavior;
beam-steered systems can sample directions sequentially. Flash/snapshot LiDAR
is the relevant architecture for capturing a field together, though a particular
device's pulse accumulation and integration still need checking.
[TI's lidar architecture overview](https://www.ti.com/lit/wp/slyy150d/slyy150d.pdf).

For the Mid-70 class named in the current repo, Livox explicitly describes
non-repetitive scanning and increasing density by accumulating multiple frames.
It is not an instantaneous whole-field snapshot. Livox separately explains how
per-point times and motion estimates compensate motion within a point-cloud
accumulation interval. This is a known acquisition property, not evidence of a
fault or of a measured Mapper tracking failure.
[Mid-70 application description](https://www.livoxtech.com/showcase/10),
[motion compensation](https://www.livoxtech.com/showcase/211220).

This model-specific conclusion is conditional on the repo's sensor choice still
being current. A different selected device requires its own acquisition timing
check. No sensor purchase, mapper change or paused-test resumption follows.

## Two different kinds of success

Product success remains a reliable scan with defensible geometry, usable speed
and appropriate thermal behavior. Research success can instead be a reproducible
failure mechanism, a discriminating measurement, or a reusable testing component
that changes how we would build or evaluate the mapper. A rejected scanner
candidate may contribute to the latter without satisfying the former.

The recent broad sweeps still have diminishing returns as a route to fixing the
MINI. Their additional mapper value depends on extracting explanations and
testable boundaries, not simply retaining a longer list of rejected settings.

## Evidence-backed findings and proposed transfer

| Finding in this repository | Possible mapper implication | Boundary on the claim |
| --- | --- | --- |
| M21 accepts every input in four long scans while each fails geometric checks. M23's combined cost is effectively nonincreasing despite major trajectory failures. [Stage B](reliability-reverse-seed-stage-b.md), [mixed objective](reliability-mixed-metric.md) | Keep throughput, tracking acceptance, fit residual and map accuracy as separate measurements. Include checks that can expose a confidently wrong map. | These records refute those proxies as sufficient guarantees here; they do not establish a universal detector or calibrated uncertainty. |
| The same stationary face-A recording shows about 27.39 nominal mm apparent motion under production sampling versus 0.42 under hierarchical-grid sampling. Face B and moving cases behave differently. [Two-view review](reliability-view-b.md) | Audit how selection/preparation changes the measurements presented to the tracker over time. Retain stable input identity so measurement change can be distinguished from pose change. | Sampling contributes materially in this pipeline; neither these magnitudes nor a claim that stable sampling solves tracking transfers automatically. |
| A five-case M21 survivor fails the longer/handheld stage. Some saved adjacent alternatives repair local checks, but many late failures have no good saved alternative. [Longer-history evidence](reliability-reverse-seed-stage-b.md) | Test complete evolving trajectories, recovery and reference/model history, not only isolated pair fits or a favorable short run. | Saved alternatives are conditioned on the same old history; they are not evidence that a different live selection policy would succeed. |
| Optimized GPU/host processing reproduces the original recorded decisions and poses while improving speed; faster visible scans can still have wrong geometry. [GPU record](live-gpu-throughput.md) | Maintain a reference computation and an explicit numerical-equivalence test when accelerating processing. Benchmark acquisition, fitting, display and map updates separately. | Exact parity on this implementation/device does not qualify another GPU, precision, sensor or mapping algorithm. Equivalent wrong answers remain wrong. |
| The organized-normal adapter requires raw-frame, accepted-frame, model-pose and fit-direction identities. Matching coordinates alone do not identify a field whose inputs also include raw support and role. [Preflight](reliability-organized-model-preflight.md), [adapter](reliability-organized-model.md) | Make transform direction, ownership and cache dependencies explicit in reusable registration components. Preserve source indices through refusal and recovery. | M27 implementation/control checks support this contract; its changed-normal matrix is unfinished and no tracking benefit is qualified. Coordinate-keyed caches remain valid for functions that truly depend only on coordinates. |
| A local conditioning statistic does not consistently separate good from failed tracking; the same-view improvements are not explained by a simple condition cutoff. [Conditioning evidence](reliability-view-b.md), [learning index](reliability-learning-log.md) | Treat conditioning and residuals as diagnostics, not an automatic probability that the map is correct. Evaluate any proposed confidence rule against actual failure cases. | This does not disprove useful uncertainty models or all conditioning guards. No calibrated mapper confidence model was derived. |
| Shorter MINI timing increases raw FPS while losing previously stable returns; longer tests reveal effects missed by short trials. [Acquisition](scanner-fps-investigation.md), [validation](scanner-validation-results.md) | Measure useful scene information, coverage and latency together when comparing acquisition modes. Preserve timing and operating conditions with the frames. | The exposure/rate/temperature values and structured-light behavior are device-specific. They are not settings or thermal ratings for a different mapper sensor. |
| The new binding/inventory treats an approximate 1,800-point preparation target as a strict cap; 283 inventoried dense preparations exceed it. [Pause checkpoint](reliability-discussion-checkpoint.md) | Test contracts between pipeline stages, including unusual but legitimate inputs, and distinguish harness failures from algorithm or sensor failures. | The inventory is partial and the adapter limitation is not fixed. It does not explain tracking failures that predate this adapter. |

## What the negative results do and do not establish

The completed 139 whole configurations share a small set of recordings and a
substantial amount of pipeline code. They are not 139 independent validations
across scenes or hardware. Likewise, thousands of passing software tests are
not thousands of physical scans. Repeated endpoint comparisons may overlap.

The defensible negative statement is: a specific, recorded intervention did
not preserve the declared cases under the stated guards. It is not "robust
losses do not work," "all normal smoothing fails," "global mapping cannot help,"
or "the scanner hardware is bad." A local optimization failure may reflect
association, model/history, measurement or scene assumptions that this parameter
roster does not distinguish. These remain competing explanations.

Our repeated development comparisons are also not independent geometric ground
truth. Transferring an algorithmic conclusion to the mapper requires matching
the relevant assumptions and testing new conditions; a new sensor, independently
known motion/geometry, or a separately held-out recording can provide stronger
evidence than another setting on the same five cases. This is proposed future
validation, not a claim that those measurements already exist.

## Reusable assets already worth retaining

- Raw recordings, frame identities, decision/pose/reference histories and failed
  trajectories: a failure corpus for testing whether a proposed fix changes the
  observed behavior. Sensor-specific data needs an explicit adapter, not silent
  reinterpretation of coordinates or timestamps.
- Reference-versus-accelerated parity checks and fault-injection fixtures:
  examples of testing numerical behavior and truthful fallback/reporting.
- Fixed-endpoint rescoring, missing-input accounting, complete-trajectory tests
  and exact observation controls: evaluation patterns that can be adapted without
  inheriting the MINI's thresholds or acceptance policy.
- Source/runtime snapshots, immutable completed bundles, preserved failed
  attempts and result-to-input links: the reproducibility method, not a substitute
  for physical validity or an off-machine backup.

These are existing artifacts and patterns, not a newly extracted mapper library.
The [learning index](reliability-learning-log.md) remains the complete chronology
and experiment/evidence catalog; the [reproduction guide](reliability-reproduction.md)
describes retrieval, dependencies, local retention and safe fresh-output reruns.
This document is a synthesis and must not replace or discard the underlying record.

## Proposed research filter

For future work, record before execution:

1. The competing explanations and the exact observation that would separate them.
2. What either outcome would change in the scanner or mapper design.
3. Which assumption should transfer, and which sensor/scene details may not.
4. A bounded test and its stop condition; add more variants only to answer an
   unresolved distinction, test a declared interaction, or replicate a finding.
5. The actual result, failures/missing data, uncertainty and evidence links;
   retain attempted and corrected implementations separately.

Label findings as demonstrated locally, plausible transfer, or still untested.
Research that produces a useful counterexample or reusable evaluation component
can be worthwhile without fixing the scanner. Repeating equivalent failures with
no changed explanation or design consequence remains low-information work.

The official-software comparison still matters for the hardware-versus-host
question. It need not be treated as a prerequisite for every independent research
question. No further experiment or process resumption follows from this note;
the [pause checkpoint](reliability-discussion-checkpoint.md) remains operative.
