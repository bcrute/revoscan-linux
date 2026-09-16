# Handheld tracking and capture throughput

Historical September 6–7 results; see the [reliability learning index](reliability-learning-log.md)
for the current checkpoint. On September 9, all sixteen specifically named early
GPU/GUI artifacts and intermediate kernel reports were preserved unchanged under
`/home/ben/scans/diagnostics/20260909-historical-tracking-preservation/`.
Its `source-inventory.json` maps the original `/tmp` paths below to durable
copies. This preserves the files as found, not an unrecorded historical source
binding or an off-machine backup.

On 2026-09-06, the connected MINI delivered 160 full depth frames at approximately
4.5 fps with no reported drops when saving raw frames. The GUI capture path,
with tracking, colour capture and model generation enabled, delivered only
1.84 fps over a 50-second run. It saved 91 frames and reported 14 dropped frames.
These were measured through the real GUI capture job with headless widgets;
Qt rendering was not included in those timings.

At the lower rate, a moving scanner travels much farther between usable views.
The previous default also selected only one delivery in three. Once overlap is
lost, a later switch to finer spacing cannot recover the missing views.

The implementation now:

- Prepares raw frames with NumPy when available, materialising Python points
  only after depth clipping and thinning. The scalar fallback remains available.
  Point order, exclusive depth bounds, filtering and accounting are unchanged.
- Reuses a bounded cache of normal fields for repeated registration targets.
  Normal estimation and the pose solver retain their original arithmetic.
- Uses a single SciPy query worker for small tracking clouds, avoiding repeated
  thread creation inside ICP. Larger queries retain parallel execution.
- Returns each camera buffer immediately after copying it, before consumer
  callbacks. The immutable copy remains valid if the camera refills the mapping;
  a callback exception also leaves the buffer returned.
- Starts GUI captures at one frame in one, with automatic spacing off. The
  spacing controls remain available for a deliberate turntable capture.
- Applies model-refresh duty limiting above 50 ms, replacing a 700 ms threshold
  that exceeded three average camera intervals.
- Accelerates depth preview colouring, return statistics and plausibility
  diagnostics with NumPy, keeping scalar fallbacks. These run on the GUI thread
  and compete with tracking for Python execution time.

Preparation measured approximately 230 ms before and 25 ms after on a frame
from `/home/ben/scans/sweep/20260906-161033`.

With the updated implementation, the same headless capture job ran at **4.50 fps**, accepted
all **240/240 frames**, and reported **zero dropped or short frames**. The median
capture callback took 144 ms. The capture is
`/home/ben/scans/scan/20260906-163439`; the per-frame measurements are in
`/tmp/scanner-live-20260906-163439.json`.

The visible desktop application was tested separately, with the OpenGL model,
depth preview and colour preview drawing throughout the capture. With the core
changes but before preview acceleration it delivered **2.86 fps**, with 44
sequence-reported drops. After preview acceleration it delivered **3.80 fps**,
with **159/159 complete frames accepted** and no tracking loss. There were still
**13 sequence-reported drops and one short delivery**, so the visible application
has not reached the raw camera's full throughput. The final desktop capture is
`/home/ben/scans/scan/20260906-164722`; measurements are in
`/tmp/scanner-window-20260906-164722.json`. These are live turntable measurements,
not an assertion of handheld reliability or a hardware frame-rate specification.

## Accuracy checks and limits

Replaying identical clouds through the original and cached normal estimators
gave identical decisions and **zero pose-coordinate difference** on both the
first 90 frames of the fresh turntable capture and the 71 offered frames of
`/home/ben/scans/scan/20260828-105933`. The latter still accepts 67 frames before
three consecutive refusals: accelerating capture does not repair a historical
recording that already lost overlap. Residual, overlap, travel and agreement
gates have not been widened.

Regression tests compare accelerated and scalar preparation, all 65,536 preview
codes under multiple stretches and palettes, frame statistics, and diagnostics
on empty, uniform and varied frames with odd grid dimensions. Transport tests
also refill the camera mapping at re-queue time and raise from each callback to
check immutable payloads and buffer return on failure.

The turntable tests do not establish a maximum handheld speed. Hand motion can
change visibility and corrupt a structured-light measurement during acquisition;
those effects require a physical handheld test. Delivery times also must not be
assumed to be exposure times, so motion prediction was not changed to use them.

## Maximum-speed turntable follow-up

The operator increased the turntable to its maximum setting, estimated by the
operator as approximately twice the earlier speed. The visible desktop capture
`/home/ben/scans/scan/20260906-165225` delivered 360 buffers and saved **359 full
frames**, all accepted by the live tracker. It completed without tracking loss
at **3.31 fps**, with **63 sequence-reported drops and one short delivery**. Two
frames used coarse recovery (1.40 s and 0.68 s); median tracking time was 175 ms.
Later live-model builds took approximately 0.7 s, so keeping up with acquisition
remains a separate performance problem even when tracking stays connected.

Replaying all 359 saved frames also accepted every frame. To check geometry,
five image-matched revisit pairs were tested using the existing depth-projection
agreement calculation (1.5 mm tolerance, 75% required agreement):

| Frame pair | Raw image median difference, mm | Points agreeing within 1.5 mm |
| --- | ---: | ---: |
| 0, 187 | 0.035 | 80.3% |
| 30, 286 | 0.055 | 77.2% |
| 60, 314 | 0.050 | 77.5% |
| 89, 185 | 0.105 | 81.1% |
| 119, 278 | 0.165 | 99.2% |

All five passed, but the weakest pairs have little margin above the threshold;
this is not a metrology qualification. These are the actual matching pairs,
not pairs at a single median frame lag. On this multi-revolution capture, the
automatic image-period estimator mixed matches from different revolutions
(lags 96 to 256), chose 187, and its fixed-lag check incorrectly compared other
views at different phases. Its resulting 3% worst agreement is **not** the
agreement of the actual matching pairs listed above. No checker or tracking
threshold was changed for this test.

Detailed run measurements are in `/tmp/scanner-speed-20260906-165225.json`,
replayed poses in `/tmp/scanner-speed-coherence-20260906-165225.json`, and the
exact-pair checks in `/tmp/scanner-speed-exact-pairs-20260906-165225.json`.
The scanner stream was closed after the capture.

## Live meshing removed

At the operator's request, experimental occupancy meshing was removed from
live capture, not merely hidden or made less frequent. The capture job no longer
constructs an accumulator, folds clouds into a grid, marches triangles or
prepares surface arrays. Live snapshots contain points only, including when a
surface was selected before the capture. Offline reconstruction/export and
display of its signed-distance mesh are unchanged. The surface control explains
that it becomes available after Reconstruct + export.

The follow-up visible desktop capture `/home/ben/scans/scan/20260906-171316`
delivered **360 buffers at 4.49 fps**, with **zero sequence-reported drops** and
one short delivery. All **359 complete frames** were tracked, without tracking
loss or coarse recovery. This reaches the earlier measured raw-camera cadence;
it does not establish that 4.5 fps is a hardware limit. Physical turntable speed
was not independently controlled in this follow-up.

There were 90 point-model builds, with a median of **97 ms** and a maximum of
133 ms, versus the approximately 700 ms late mesh-bearing builds in the previous
run. Median tracking time was 163 ms. Details are in
`/tmp/scanner-no-live-mesh-20260906-171316.json`; the application log explicitly
identifies these builds as points-only.

Regression tests trap accumulator construction, marching and triangle-array
conversion, including a remembered surface preference, and verify that a new
capture clears a previous reconstructed mesh. Point-model duty limiting,
pause/recovery and offline reconstruction/export remain independently tested.

## GPU rendering is not GPU registration

On this host the 3D view uses OpenGL on the RX 9070 XT. Live preparation,
tracking and point-model construction still use CPU Python/NumPy/SciPy;
`select_backend("auto")` resolves to `scipy`. The reserved `gpu` compute backend
is not implemented. Its refusal now says that, instead of claiming that the
installed GPU generation lacks support.

AMD lists the RX 9070 XT in its
[ROCm Linux hardware support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html).
That does not validate this host's complete runtime/driver/kernel combination.
At the time live meshing was removed, no PyTorch, CuPy, PyOpenCL or JAX package
was installed in the scanner environment, and this process lacked read/write
access to `/dev/kfd`. No packages, drivers or device permissions were changed
during that investigation. The subsequent OpenGL compute experiment below
uses a different runtime and adds only an optional Python package.

A GPU prototype should keep reference clouds resident and benchmark the
registration loop as a whole, including transfers and synchronization, against
the current CPU path on these saved captures. Tiny per-query offloads can cost
more than the CPU search. Any speed gain must also preserve acceptance gates
and pass recorded-chain accuracy checks; GPU availability alone is not a
tracking speed or quality result.

## GPU processing experiment: actual compute, not just drawing

The opt-in `tools/gpu_tracking_probe.py` now exercises two float64 compute
shaders on the RX 9070 XT through its existing OpenGL driver. PyOpenGL 3.1.10
was added to this checkout's virtual environment; no system driver, ROCm
runtime, device permission or group membership was changed. The `gpu-probe`
extra records the optional dependencies. A ModernGL installation attempt
failed because its context package needed missing X11 development headers;
that path was not pursued and ModernGL was not installed.

The useful offload is **batched normal eigensolves**. CPU code retains the
neighbour selection, covariance arithmetic, normal classification, pose solve
and all acceptance gates. Nearest-neighbour offloading was tested separately
with resident target buffers; it preserved the first 30 frames' poses but was
slightly slower overall (3.16 s versus 3.05 s), so it was not used in the live
GPU experiment. A GPU being idle does not make every small operation faster.

The first normal kernel was faster but failed pose parity. Merely using
float64 was insufficient: native shader division differed from CPU arithmetic,
and multiply/add contraction in helper arguments also mattered. The corrected
kernel refines division with fused residuals and explicitly marks the relevant
intermediate expressions `precise`. The scope of that qualifier is described
in the [GLSL specification, section 4.9](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html).
The original cyclic Jacobi rotation order is retained, not replaced with a
different eigensolver. Neither convergence nor tracking gates were relaxed.

### Identical-recording comparisons

| Recording / run | CPU total tracking | GPU total tracking | CPU median/frame | GPU median/frame |
| --- | ---: | ---: | ---: | ---: |
| `20260906-171316`, 359 frames, first pair | 43.71 s | 33.60 s | 120.7 ms | 89.9 ms |
| Same recording, repeated CPU/GPU pair | 44.37 s | 34.65 s | 122.2 ms | 92.3 ms |
| `20260828-105933`, historical handheld | 11.17 s | 9.09 s | 119.0 ms | 88.8 ms |

Every comparison above had **zero pose-coordinate difference** and identical
acceptance, origin and failure-type decisions on the tested host. Both full
turntable replays accepted 359/359 frames. Both handheld replays offered 71
frames, accepted 67 and stopped at the same terminal tracking loss; faster
computation did not repair missing overlap in the old recording.

These are tracking-only wall times on pre-prepared identical clouds, including
GPU uploads, dispatch, synchronization, readback and Python conversions.
Preparation timings are recorded separately. Kernel compilation/warm-up occurs
before timing, and normal caches are cleared between runs. The shader hash is
included in new reports. Exact agreement on these inputs is an empirical
result, not a proof for arbitrary clouds, extreme values or other drivers.

Reports:

- `/tmp/scanner-gpu-precise-359-abab.json`
- `/tmp/scanner-gpu-precise-handheld.json`
- `/tmp/scanner-gpu-live-capture-parity.json`: the later 180-frame live GPU
  capture also replays with identical CPU/GPU poses and decisions, including
  both coarse recoveries (23.04 s CPU versus 17.24 s GPU tracking).
- `/tmp/scanner-gpu-first-probe.json` and the intermediate corrected/refined
  reports retain the failed experiments; they are not successful backend tests.

### Visible desktop capture

A separate process-scoped harness created the compute surface on the GUI
thread and its context on the capture worker, following the
[Qt offscreen-surface threading requirements](https://doc.qt.io/qt-6/qoffscreensurface.html).
The normal estimator was patched only for that capture process; concurrent
relocalization workers would retain the CPU path (none used it in this run).
Depth/RGB previews and the point cloud were drawn throughout. No surface mesh
was generated.

| Live run | Median tracking | Delivered rate | Complete frames tracked | Reported drops / short deliveries |
| --- | ---: | ---: | ---: | ---: |
| CPU `20260906-204709` | 161.9 ms | 4.49 fps | 179/179 | 0 / 1 |
| GPU `20260906-204754` | 133.0 ms | 4.45 fps | 180/180 | 1 / 0 |

The GPU run made 369 batched eigensolve calls on the RX 9070 XT. It had two
successful coarse recoveries (547 and 640 ms), and no tracking loss. Physical
turntable phase/speed was not independently controlled between these live
runs; the identical-recording comparison above is the stronger speed and
parity evidence. Live tracking time decreased, but delivered frame rate did
not increase and this is not a claim of drop-free capture or handheld accuracy.

Reports are `/tmp/scanner-live-cpu-20260906-204709.json` and
`/tmp/scanner-live-gpu-20260906-204754.json`. The harness is
`/tmp/scanner_gpu_live_test.py`; its report identifies the experimental offload,
whereas the application's ordinary backend selector still reports SciPy.

**Historical state at this experiment:** this remained an opt-in benchmark,
not a production backend or an automatic fallback. Promoting it requires
explicit backend/provenance reporting, per-worker context ownership and
initialization/failure handling, plus additional accuracy and driver tests.
It does not restore experimental live meshing.

**Later follow-up:** normal offload is now integrated into the capture worker,
with explicit CPU fallback and CPU/display optimizations. The last physical
GUI run reached 7.86 fps; 10 fps is not yet verified. See
[live GPU throughput](live-gpu-throughput.md) for results, intermediate exposure
tests and the USB reconnect needed to continue.

Reproduce the read-only recording comparison:

```bash
uv pip install --python .venv/bin/python '.[gpu-probe]'
.venv/bin/python tools/gpu_tracking_probe.py \
  --capture /home/ben/scans/scan/20260906-171316 --frames 359 \
  --modes cpu normals cpu normals --output /tmp/gpu-comparison-new.json
SCANNER3D_TEST_GPU=1 .venv/bin/pytest tests/test_gpu_tracking_probe.py
```

The output path must not already exist. The kernel tests are explicitly opt-in
and never open the scanner. They check strict gates, bounded resident buffers,
1,002 covariance eigensolves and complete normal fields including refusals.

## Camera frame-interval experiment

**Follow-up:** direct, firmware-verified timing controls subsequently raised
raw depth output to approximately 6, 9 and 13 fps. A five-minute 13 fps soak
delivered 3,891 unique complete frames without drops; CPU/NPU ended at
67.3/66.6°C. These are acquisition rates, not full-GUI tracking rates. See the
[in-depth scanner FPS investigation](scanner-fps-investigation.md) for the
recovered controls, A/B/A tests, quality limitations and restoration evidence.

The depth interface reports `V4L2_CAP_TIMEPERFRAME`, a current interval of
1/15 s, and discrete advertised intervals of 1/15, 1/10 and 1/5 s. A bounded
probe requested each supported interval after normal format selection and
measured 80 deliveries, then repeated 1/15 s:

| Requested / returned nominal rate | Measured delivery rate | Reported drops | Short deliveries |
| --- | ---: | ---: | ---: |
| 15 / 15 fps | 4.476 fps | 0 | 0 |
| 10 / 10 fps | 4.477 fps | 0 | 1 |
| 5 / 5 fps | 4.475 fps | 0 | 1 |
| 15 / 15 fps, repeated | 4.475 fps | 0 | 1 |

Thus changing this standard request did **not** change actual depth delivery
cadence. It is not evidence of a 4.5 fps hardware limit or a USB bandwidth
limit. The [V4L2 interval API](https://docs.kernel.org/userspace-api/media/v4l/vidioc-g-parm.html)
is not a measurement of actual sensor exposure timing.

`tools/camera_cadence_probe.py` restores the original interval in `finally`
and verified it was 1/15 s afterward. It never touches exposure or vendor
properties and saves no raw frames. Read-only standard exposure inventory
showed inconsistent metadata (the depth absolute-exposure maximum was zero,
while its value was 20,000); those values were not treated as reliable physical
exposure units and no exposure control was written. Scanner sessions were
closed after testing.
