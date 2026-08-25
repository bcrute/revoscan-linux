# revoscan-linux

A Linux-native toolchain for the [Revopoint MINI](https://www.revopoint3d.com/) structured-light
3D scanner (USB `2207:110c`). Revopoint ships no Linux consumer software, so this builds the
whole path — capture, registration, reconstruction and a desktop application — from the USB bus
up, with no vendor SDK at runtime.

## What works

```
scanner3d-gui                                    # desktop application
scanner3d sweep OUTPUT_DIR                       # capture a sweep
scanner3d reconstruct FRAMES_DIR --backend scipy # register and score it
```

The window shows three things at once: the colour camera for positioning, the depth camera for
what is actually being captured, and the accumulated model — orbitable, with under-scanned
surface in soft red. A live readout says whether your rotation speed is in the band that
registers well.

## The number this project is built around

Registration reports **held-out closure error**: revisited viewpoints are detected geometrically,
half are given to the optimiser, and the score comes from the half it never saw.

That is not decoration. Measured on a real two-lap capture of a model truck bed, a trajectory
that was **65 degrees wrong after one revolution** simultaneously reported per-pair alignment
rmse 0.65, 100% frame overlap, and healthy self-consistency. Every internal metric said the scan
was fine. Self-consistency cannot detect drift, because a drifted trajectory is perfectly
self-consistent — so this measures against evidence the optimiser was not allowed to fit.

Fixing what that exposed took drift on the same hardware from **65 degrees per revolution to
0.5**, via four independent changes: capture density, an ICP trim fraction that was suppressing
recovered rotation to 85% of truth, point-to-plane matching, and pose-graph optimisation against
detected loop closures.

If a sweep never revisits a viewpoint, reconstruction reports **no score at all** rather than a
number it cannot back.

## What is measured, and what is not

Everything quantitative here was measured on one unit. Findings that are not documented by the
vendor and were established here:

- The depth scale is millimetres — a 58.0 mm can measured 57.705 mm at ~122.7 mm standoff
  (−0.51%). Accuracy across the range is **not** established: −0.49% to −2.26%, no correction model.
- The device **re-enumerates on the USB bus every time a depth stream closes**, disappearing for
  about 0.39 s. Anything reopening immediately sees "no such device".
- The **RGB camera delivers nothing on its own** — a lone capture times out — but returns frames
  reliably while a depth stream runs. The vendor's depth-first ordering is a requirement.
- The full factory calibration set is readable over the vendor's own file-read protocol, including
  the depth→RGB extrinsics that make colour-to-geometry mapping possible.

## Honest gaps

- `keyframe.py` has no test suite. Everything else here does, mutation-tested. It is validated
  against real captures (2.4× less drift on an over-sampled sweep, no change on a well-sampled
  one) but that is not the same thing.
- Revisit detection can accept false positives on near-symmetric objects: two views half a
  revolution apart have similar depth histograms, and no local threshold separates them — the
  false ones score *better* on rmse than the true ones. The pipeline contains the damage by
  returning the unoptimised chain when optimisation makes held-out error worse.
- No mesh texturing yet, though the calibration needed for it is now in hand.
- The projector cannot be turned off. Several register writes and the device's own actuator ioctl
  are all *accepted* while the emitter stays lit; this project has never had optical readback.

## Configuration

The scanner serial is read from `SCANNER3D_SERIAL`, then `~/.config/scanner3d/config.toml`
(`serial` under `[scanner]`), then the single attached device. It refuses to guess when several
are attached, because several operations here detach kernel drivers.

## Requirements

- CPython **3.14** or newer
- Linux with `sysfs` mounted at `/sys` (only `uvcvideo`-exported metadata is read)
- No root, no udev rule, and no system packages are required for `doctor`

## Development setup

```bash
uv venv --python 3.14 .venv
uv pip install -e '.[dev]'
```

Or without `uv`:

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

This is the one verified path. It was checked by building a throwaway
environment from scratch and running every quality gate below in it.

**Do not use `uv sync` here.** The development tooling is declared as a
`[project.optional-dependencies]` *extra*, not a PEP 735
`[dependency-groups]` group, and this project has no dependency groups at all.
`uv sync --all-groups` therefore treats the dev tooling as unrequested and
**uninstalls** `pytest`, `ruff`, `mypy` and `coverage` from the environment
rather than installing them. If you prefer the lockfile-driven workflow, the
extra has to be named explicitly: `uv sync --extra dev`.

## Quality gates

Every gate runs offline and needs no hardware.

| Purpose | Command |
| --- | --- |
| Format | `.venv/bin/ruff format --check .` |
| Lint | `.venv/bin/ruff check .` |
| Types | `.venv/bin/mypy` |
| Tests | `.venv/bin/pytest` |

`ruff format .` rewrites files in place. The test suite is fixture-driven: it synthesizes fake
`sysfs`/`dev` trees and replays a recorded byte-for-byte configuration descriptor from the real
scanner, so it passes with no device attached and never touches hardware.

## CLI

```bash
scanner3d list                 # attached matching devices, one line each
scanner3d doctor               # human-readable diagnostic
scanner3d doctor --json        # machine-readable report on stdout
scanner3d doctor --json-out report.json
scanner3d trace capture.jsonl   # decode a recorded usbfs trace (offline)
scanner3d trace capture.jsonl --json
scanner3d firmware-version      # describe the one owned control read (opens nothing)
scanner3d firmware-version --execute --serial <serial>   # actually perform it
scanner3d firmware-version --detach-uvcvideo-from-interface-0 --execute --serial <serial>
                                # same read, but temporarily take interface 0 from uvcvideo
scanner3d inited-sequence       # describe the four-transfer /tmp/inited sequence (opens nothing)
scanner3d inited-sequence --execute-write-sequence --serial <serial>
                                # actually perform it — this WRITES to the scanner
scanner3d depth-intrinsics      # describe the bounded /data/camparam/Pl.bin read (opens nothing)
scanner3d depth-intrinsics --execute-calibration-read --serial <serial>
                                # actually perform it — this WRITES the key to the scanner
scanner3d depth-scale           # describe the bounded 0x918 depth-scale property read (opens nothing)
scanner3d depth-scale --execute-depth-scale-read --serial <serial>
                                # actually perform it — this WRITES the property id to the scanner
                                # (allow_abbrev=False: this flag must be typed in full)
scanner3d depth-frame           # describe the one bounded Y16 640x400 depth capture (opens nothing)
scanner3d depth-frame --execute-depth-frame-capture --serial <serial> --output-dir <dir>
                                # actually perform it — reads one frame over plain V4L2
scanner3d rgb-frame             # describe the one bounded MJPG 1280x800 RGB capture (opens nothing)
scanner3d rgb-frame --execute-rgb-frame-capture --serial <serial> --output-dir <dir>
                                # actually perform it — reads one frame over plain V4L2
scanner3d rgb-controls          # describe the fixed read-only RGB control inventory (opens nothing)
scanner3d rgb-controls --read-rgb-controls --serial <serial>
                                # query standard controls once; never changes a control or starts a stream
scanner3d rgb-exposure-extension # describe the fixed SDK RGB extension read (opens nothing)
scanner3d rgb-exposure-extension --execute-rgb-exposure-extension-read --serial <serial>
                                # opt in to exactly one fixed GET_CUR (scanner-027 ran it once)
scanner3d rgb-exposure-length # describe the fixed RGB extension GET_LEN (opens nothing)
scanner3d rgb-exposure-length --execute-rgb-exposure-length-read --serial <serial>
                                # scanner-030 ran one; every new physical run owes its own off finalizer
scanner3d rgb-exposure-info # describe the fixed RGB extension GET_INFO (opens nothing)
scanner3d rgb-exposure-info --execute-rgb-exposure-info-read --serial <serial>
                                # later physical task: one GET_INFO, then mandatory off finalizer
scanner3d paired-capture      # depth-first/RGB-second plan (opens nothing)
scanner3d paired-capture --json
scanner3d paired-capture --start-depth-first-rgb-second --serial <serial> --output-dir <dir>
                                # actually perform one fixed paired V4L2 attempt
scanner3d paired-rgb-series    # depth-first/RGB-series plan (opens nothing)
scanner3d paired-rgb-series --json
scanner3d paired-rgb-series --start-depth-first-rgb-series --serial <serial> --output-dir <dir>
                                # actually perform the fixed three-observation RGB diagnostic
scanner3d paired-rgb-series --read-rgb-controls-during-series
                                # request bounded read-only control telemetry (still dry by default)
```

The paired plan preserves the pinned 3DViewer ordering: resolve distinct depth
and RGB nodes by the exact serial, start depth, wait one second, start RGB, and
wait once for one depth and at most one RGB frame. Execution requires the exact
serial plus `--start-depth-first-rgb-second`, and a local `--output-dir`; it has
fixed 8-second deadlines, the existing depth/RGB memory ceilings, no retry, and
reverse cleanup. Verification is fully covered with injected/fake V4L2 syscalls;
accepted physical evidence is documented below separately from the synthetic tests. Scanner-014's raw
usbmon target records and their bounded decoder are preserved under
`tests/fixtures/` and do not claim that RGB hardware delivery works.

`paired-rgb-series` is an offline-verified diagnostic boundary. It preserves the accepted
depth-first start and one-second interval, then uses one absolute 8.0 s RGB deadline with exactly
three bounded dequeue opportunities. Every payload is written unchanged as indexed
`rgb-series/rgb-series-000.jpg` through `rgb-series-002.jpg` before any allowed requeue; the JSON
records V4L2 sequence/timestamp/flags/bytesused, hashes, strict JPEG inspection, EOI-prefix facts
and opaque trailer bytes. A complete JPEG prefix does not make a payload valid when bytes follow
EOI, and no trimmed derivative is written. The command is dry by default and requires the exact
`--start-depth-first-rgb-series` flag, the authorized serial and an explicit local output
directory. The live-capable implementation is verified only with injected/fake V4L2 boundaries.
Scanner-019 is the accepted physical three-observation run: all three payloads had four opaque
trailing bytes after their JPEG EOI, decoded nearly black, and showed no meaningful warm-up
improvement. Scanner-021 subsequently ran the exact read-only standard-control inventory once;
its measured result is recorded below.

The separately gated `--read-rgb-controls-during-series` mode is implemented and verified
offline with fake V4L2 calls. With the exact series opt-in it uses the already-open RGB
descriptor and reads gain, exposure-auto, and exposure-absolute in fixed order before the
first RGB poll and after each successfully dequeued observation. There are at most four points
and twelve attempts; ordinary errors and returned-ID mismatches remain typed records without
retries, and capture and telemetry statuses remain separate. The flag alone, a dry run, wrong
serial, and non-series capture never open anything or issue `G_CTRL`. Scanner-024 then performed
exactly one accepted physical telemetry invocation: all 4/4 points and 12/12 fixed reads
succeeded, with gain 16, exposure-auto index 3 and raw absolute exposure 131072 invariant.
No setting was changed; these raw values do not establish exposure units or the separate
selector-0x12 extension state.

`rgb-exposure-extension` is a separate, dry-by-default boundary for the MINI
SDK's selector-`0x12` wire request. Scanner-027 ran its immutable opt-in exactly
once: `a1 81`, `wValue=0x1200`, `wIndex=0x0400`, requested 60 bytes, and the
device returned 2/60 bytes (`0000`, SHA-256
`96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7`). The
typed result was `short_response` (exit 7), with no four-word decode. The same
run performed its mandatory immutable blue-light off write exactly once,
accepted 60/60, and passed release/reconnect plus stable postflight; that was
the last physical action. “Accepted” describes transfer length; it does not
claim optical darkness.

Scanner-028 reconciled this wire result offline with the hash-pinned SDK
(`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`) and the
descriptor fixture. The SDK's `PROPERTY_EXT_EXPOSURE_TIME_RGB=0x15` and
`...RANGE_RGB=0x16` are device-property IDs, not selector 0x12. Its CameraType
12 GET paths each issue one `a1 81`/`0x1200`/`0x0400`/60-byte/1000-ms transfer;
the wrapper zeroes the 60-byte buffer and treats any nonnegative libusb byte
count as wrapper success, discarding that count. The 0x15 path reads word 0;
the 0x16 path reads words 1–3 (offsets 4, 8, 12), with SDK scaling described
in `docs/architecture.md`. Entity 4's recorded `bmControls=ffa201` does not
advertise selector 0x12 (bit 17 is clear); this does not prove the device must
reject an unadvertised selector. Therefore `0000` is only the measured two-byte
reply: it is not established exposure, microseconds, enable state, success,
failure, or a valid zero. Scanner-029 now implements the offline-first
`GET_LEN` boundary in `rgb-exposure-length`. It is dry by default and its only
possible live request is `a1 85`, `wValue=0x1200`, `wIndex=0x0400`, `wLength=2`,
timeout 1000 ms, once through the established interface-0 lifecycle. It
preserves actual length, retained raw bytes, SHA-256 and typed transfer or
cleanup failures; only an exact two-byte reply is decoded little-endian as a
raw reported control length, never as support or exposure semantics. No
physical GET_LEN was run in scanner-029. Scanner-030 subsequently ran that
exact boundary once and completed its separate immutable blue-light off write
and clean restoration. Every new physical GET_LEN invocation must again issue
its own read followed by exactly one immutable off write and clean restoration;
neither result alone establishes support or semantics.

Scanner-030 then ran that exact GET_LEN boundary once on the authorized MINI.
It returned exactly 2/2 bytes, raw `0200` (little-endian raw value `2`), with
clean release/reconnect. The separate mandatory immutable blue-light-off write
was accepted 60/60 and restored the interface; three postflight samples stayed
stable. These are command/restoration facts only, not exposure or optical
readback evidence. Scanner-031 now adds the next offline-only boundary:
`rgb-exposure-info`, whose only possible live request is one `a1 86`,
`wValue=0x1200`, `wIndex=0x0400`, `wLength=1`, timeout 1000 ms. An exact
one-byte reply is shown as raw UVC GET_INFO capability bits using Linux-UAPI
syntax; that bit view does not prove selector advertisement, semantics, SET
acceptance, units or frame behavior. It remains dry by default, and a later
physical task must again end with one separate immutable off write and stable
restoration.

An earlier accidental opted-in host run is recorded only as an observed audit:
the already-created `/tmp/scanner3d-paired-capture` directory contained a
512000-byte depth artifact (SHA-256
`0503b236738b95167a4cf283e886746460078a19f17cbc670820d9baf3189ee8`) and a
16947-byte RGB JPEG (SHA-256
`cc5f702fd18746a04d3bc5dcd814764bf93907a34f032231160a4df5b7069463`), both timestamped
17:45:46. The RGB bytes had valid 1280x800 SOI/SOF/SOS/EOI structure but four
trailing bytes `95 81 7d 01`, so strict validation reported `trailing_data`.
Trimming those bytes was not performed and no derived file was written; the
hypothetical trimmed-byte hash was recorded as
`818e026092e3f342f3cc9b08c50efbc5cd7f83548cd2fde7e9446ac5c48f44ab`, not produced.
This does not establish useful image content, synchronization, or repeatability. Scanner-017
subsequently repeated the accepted paired lifecycle once; its evidence is recorded below.

The accepted scanner-017 run delivered a 22529-byte RGB payload (SHA-256
`650240568e5b97fcc8f61773035e03e2b48fb5bb940c2d73de8572f83609c07d`) with a baseline 1280x800
JPEG prefix through EOI at offset 22523 and opaque trailing bytes `54 52 d8 01`. Strict
classification remained `trailing_data`; the hypothetical 22525-byte prefix hash was
`77ad1601dc234b059e5e076380ba8cb8c2b4f64f59c4f3fdb98ec2572b549199`, and no trimmed file was
written. In-place viewing found a nearly black image with only faint features. The depth payload
was one complete 512000-byte `Y16 ` frame (SHA-256
`950c9640612dd135b104bcdff1ef7536da435ca4eb6e34f7043ee58a5cd10c6a`); no metric interpretation
is claimed.

Scanner-019 then delivered three indexed RGB payloads in one accepted physical
series run. Their JPEG prefixes all declared 1280x800, but each had four opaque
bytes after EOI, so all remained strict-invalid `trailing_data`; the images were
nearly black and no useful RGB or warm-up improvement is claimed. The
`rgb-controls` diagnostic is intentionally separate: it is inert by default and
requires the exact `--read-rgb-controls` opt-in plus the authorized serial. It
opens only the identity-resolved RGB capture node read-only and queries a fixed
whitelist of gain, auto/absolute/priority exposure, auto white balance and
white-balance temperature. It issues no control writes, menu enumeration, format
or stream operations, and does not guess the vendor RGB exposure extension.

Scanner-021 is the one accepted physical `rgb-controls` invocation. Preflight
resolved serial `EXAMPLESERIAL0001` at USB path `1-1.1`, devnum 46, RGB
interface 2 (`RGBCamEXAMPLESERIAL0001`), `uvcvideo`, V4L2 capture index 0 at
`/dev/video2`, mode `0660`, with read access and no holder. The command exited
0 with `complete`, `succeeded=true`, one close, stable post-close identity and
11 ioctls (the fixed maximum is 12); stdout and `outcome.json` were byte-
identical and JSON-equivalent. The six read-only observations were:

| Control (ID) | QUERYCTRL metadata | Flags | Current/status |
| --- | --- | --- | --- |
| gain (`0x00980913`) | `Gain`, integer, 1..16 step 1, default 7 | none | 16, supported |
| exposure_auto (`0x009a0901`) | `Auto Exposure`, menu, 0..3 step 1, default 0 | none | 3, supported |
| exposure_absolute (`0x009a0902`) | `Exposure Time, Absolute`, integer, 1..60000 step 1, default 496 | `INACTIVE` | 131072, inactive |
| exposure_auto_priority (`0x009a0903`) | `QUERYCTRL` returned `EINVAL` | — | unsupported; no value read |
| auto_white_balance (`0x0098090c`) | `White Balance, Automatic`, boolean, 0..1 step 1, default 1 | none | 1, supported |
| white_balance_temperature (`0x0098091a`) | `White Balance Temperature`, integer, 2800..7500 step 10, default 4600 | `INACTIVE` | 6500, inactive |

These are device-reported integers/menu or boolean values only; they do not
establish exposure quality, brightness, colour accuracy, safety or a setting
recommendation. Six read-only postflight samples spanning 6.257 seconds kept
both video-role bindings/nodes, permissions, devnum 46 and zero holders stable
(`depth` `/dev/video0`, RGB `/dev/video2`). The outcome SHA-256 was
`f6540078c317a5ce7cd8dc1be6f1fc23bf8f968917c2d402b96a92fbac8fc9c2`.

Scanner-024 is the accepted physical paired run: one read-only, unprivileged
invocation delivered one complete depth frame and three strict-invalid,
nearly-black RGB payloads with four opaque bytes after each JPEG EOI. Standard
gain 16, exposure-auto index 3 and absolute-exposure raw 131072 were invariant
at all four telemetry points; these raw values do not establish units or the
separate selector-`0x12` state. No setting, vendor/XU request, retry or
elevation occurred; a devnum change from 46 to 47 was observed during stable
postflight sampling without an attributed cause.

Both capture commands are dry/preflight by default and open no video node without their
`--execute-...` flag. Each is fixed at import time to exactly one profile: `depth-frame` to the
`depth` USB interface role and `Y16 ` 640x400, `rgb-frame` to the `rgb` role and `MJPG` 1280x800.
Neither accepts a format, size, fps, buffer-count, timeout, retry, node/device-path, JPEG-quality
or camera-control flag; the only caller choices are the exact serial and a local output directory.
They share one bounded V4L2/MMAP engine — one open, one `S_FMT`, one `REQBUFS`, one `STREAMON`,
one deadline-bounded poll, at most one `DQBUF`, no retry, and shielded cleanup — and report their
outcome as a status string plus a distinct exit code (for example `complete` 0, `poll_timeout` 4,
`format_mismatch` 17, `malformed_frame` 28). RGB-only capture keeps `uvcvideo` bound throughout.
An executed depth capture owes one fixed blue-light shutdown write after exhaustive V4L2 teardown;
that write temporarily takes interface 0 through the shared ownership lifecycle and then restores
the driver. A missing or failed shutdown is a failed capture, with recovery guidance; dry runs,
refusals before depth `STREAMON`, and RGB-only capture never send it.

### Depth blue-light finalization (scanner-026)

Every path that attempts depth `VIDIOC_STREAMON` records a mandatory final-state obligation. After
all owed V4L2 `STREAMOFF`/unmap/queue-release/close operations, the live depth command may issue
exactly one host-to-device `SET_CUR` through interface 0:

```
21 01 00 07 00 04 3c 00 e8 03 00 00
echo s 0x922 0 >/dev/rk_preisp       (NUL-padded to 60 bytes)
```

The payload SHA-256 is
`6030364daddcc4c9855d16da552c58161bbec5cf77ea23ace9de2f14839ffb0d`. The command is immutable,
has a one-transfer ceiling and has no optical readback: “accepted” means the transfer returned
the required 60-byte length; successful finalization additionally requires clean interface-0
driver restoration, not that an optical sensor verified darkness. If either condition fails, the
report explicitly says not to claim the blue light is off and recommends rerunning the bounded
finalizer or physically unplugging the scanner.

Required finalizer failures use write-accurate statuses: `short_write` and `overlong_write` mean
the fixed host write moved the wrong number of bytes; `ioctl_error`, `unexpected_exception`,
`interrupted`, and identity/driver-cleanup statuses retain typed phase evidence. A required result
is never successful without an exact 60-byte transfer record and clean restoration; an inert
`not_required` result is successful only when depth `STREAMON` was never attempted.

`doctor` distinguishes each failure class by both status string and exit code:

| Status | Exit | Meaning |
| --- | --- | --- |
| `ok` | 0 | Every check that ran passed |
| `device_absent` | 2 | No `2207:110c` device attached |
| `permission_denied` | 3 | Device found, but its nodes are not read-write for this user |
| `no_frames` | 4 | The reference probe ran but delivered no frame / no calibration |
| `unsupported_mode` | 5 | The device does not advertise a format this project needs |
| `unexpected_identity` | 6 | Something matched the USB id but is not the expected unit |
| `internal_error` | 7 | A diagnostic itself failed |

Checks that were deliberately skipped are reported as `not run` and never counted as passing.

Discovery reads only kernel-exported metadata: it opens no character device, needs no root, and
resolves the depth/RGB roles from each USB interface's **string descriptor**
(`DepthCam<serial>` / `RGBCam<serial>`) rather than from `/dev/videoN` numbering, which is assigned
in probe order and is not stable across replug or reboot.

## The reconstruction pipeline (entirely offline, no hardware)

Everything from here on is arithmetic over captured bytes. None of it opens a
device, and all of it runs on the nine real frames committed under
`tests/fixtures/frames/`.

| Module | What it does |
| --- | --- |
| `depth_metric` | one raw `Y16 ` frame -> 3D points, with the accounting for every excluded sample |
| `cloud_export` | joins that to the writer, carrying every caveat across |
| `pointcloud` | writes PLY, with the caveats inside the file |
| `geometry` | rigid transforms, rotation about an arbitrary axis, Horn's best-fit alignment |
| `registration` | aligns a turntable sweep and reports **loop-closure error** |
| `fusion` | merges registered clouds on a voxel grid |
| `mesh` | marching cubes over the fused occupancy grid, OBJ/PLY export |
| `validation` | fits a sphere or plane and compares it to a **caliper** reading |
| `simulate` | renders synthetic frames of known shapes, for testing against ground truth |

Two of those deserve a note.

**`simulate` exists because self-consistency is not correctness.** Registration
that converges, fusion that merges and a mesh that closes can all be wrong in
the same direction with nothing in the pipeline noticing. It renders real
512000-byte frames of shapes whose geometry is known exactly, so the pipeline
can be checked against an answer it did not compute. Measured: a rendered sphere
deprojects back onto its own surface to within uint16 quantisation
(0.002492 against a 0.002500 limit), so it is the exact inverse of the
production deprojection rather than a second opinion from the same arithmetic.

**`registration` reports loop-closure error**, which is why a turntable is worth
its cost: compose every frame-to-frame transform around a full revolution and
compare against the identity. That is a self-contained accuracy measurement
needing no external ground truth. `geometry.Rigid.rotation_angle` uses
`atan2` rather than `acos` specifically so that metric does not hit a numerical
floor — measured 1.7e-16 rad against `acos`'s 6.3e-8 rad on a 36-step
revolution.

**What none of it establishes:** no distance produced by this stack has been
metrically validated. The depth scale's physical unit is unknown, so a mesh has
a correct *shape* and an unknown *size*. Closing that is
[capture-checklist.md](docs/capture-checklist.md), and it needs calipers rather
than code. One known limitation is written up separately in
[shear-detection-limits.md](docs/shear-detection-limits.md).

## The one owned vendor control request (opt-in, touches hardware)

`scanner3d firmware-version` is the only owned/native command in this project that sends a vendor
USB request to the scanner directly, and it can send exactly one thing (`doctor --reference-probe` also causes
scanner traffic, but indirectly: it runs the vendor SDK, which issues its own requests):

`scanner3d rgb-controls` is a separate standard V4L2 read-only inventory. It can issue only the
fixed `VIDIOC_QUERYCTRL`/`VIDIOC_G_CTRL` metadata/value reads described above, and remains inert
unless its exact opt-in is supplied; it never sends vendor controls or changes a setting.

```
bmRequestType=0xa1  bRequest=0x81  wValue=0x0600  wIndex=0x0300  wLength=60  timeout=3000 ms
```

That setup packet is a UVC `GET_CUR` observed from the vendor SDK on this unit. It is read-only, it
changes no device state, and its correct answer is known independently — the SDK reports the same
firmware and algorithm version strings from `queryCameras` — so a wrong answer is unmistakable.

The safety boundary is the request itself, not a policy flag. The whitelist is checked *before* any
file descriptor is opened, there is no CLI flag that can vary any field of the packet, and
`scanner3d.control` knows exactly one usbfs command number (`USBDEVFS_CONTROL`). It performs no
write, no interface claim or kernel-driver detach, no device reset, no URB submission and no video
operation, and it closes its descriptor on every path. Tests assert each of those structurally.

Without `--execute` the command opens nothing: it prints the request it would send, the
serial-resolved target node with its `rdev` snapshot, the kernel driver currently bound to the
target interface, and the VideoControl entity the request's `wIndex` names.

| Status | Exit | Meaning |
| --- | --- | --- |
| `ok` | 0 | 60 bytes returned, NUL-padded, matching the known versions |
| `not_attempted` | 0 | Dry run; no node was opened |
| `device_absent` | 2 | No device with that serial, or its usbfs node is gone |
| `permission_denied` | 3 | The usbfs node is not open-able by this user |
| `timeout` | 4 | The transfer exceeded its bounded timeout |
| `ioctl_error` | 5 | The kernel refused the transfer (e.g. `EBUSY`) |
| `unexpected_content` | 6 | 60 bytes returned, but not the known payload |
| `short_response` | 7 | Fewer than 60 bytes returned |
| `device_replaced` | 8 | The device re-enumerated between resolution and the transfer |

**Measured result of the direct read on this unit: `ioctl_error` / `EBUSY`.** usbfs implicitly
claims the interface a class request names, and `uvcvideo` owns interface 0, so the kernel refuses
the transfer before it reaches the wire. See
[docs/architecture.md](docs/architecture.md#first-owned-control-request-measured).

### Temporarily taking interface 0 (opt-in, mutates kernel driver binding)

`--detach-uvcvideo-from-interface-0` clears exactly that obstacle, and nothing else. It performs one
fixed lifecycle on interface 0 and no other interface:

```
confirm uvcvideo -> USBDEVFS_DISCONNECT -> USBDEVFS_CLAIMINTERFACE
  -> the same unchanged read -> USBDEVFS_RELEASEINTERFACE -> USBDEVFS_CONNECT
```

The driver is confirmed twice — from sysfs before anything is opened, and again from
`USBDEVFS_GETDRIVER` on the open descriptor immediately before the detach — and the command refuses
outright unless the target is the expected `2207:110c` MINI at an explicitly given `--serial` with
interface 0 bound to exactly `uvcvideo`. No option varies the interface, the driver, the setup
packet, the timeout or the order, and no phase is ever retried.

The reconnect runs in a `finally` from the moment the detach is *attempted*, and the release runs
with it whenever the claim may be held: on success, on a short or unexpected payload, on a timeout,
on any ioctl failure, on an unexpected Python exception, and on SIGINT or SIGTERM — including a
signal that surfaces from the disconnect ioctl itself or between its return and the claim, where
whether the kernel completed the detach is unknowable. Only a disconnect that failed cleanly with an
errno undoes nothing, because the kernel then took nothing away. Cleanup happens at most once, so
none of this is a retry. A cleanup that does not fully succeed forces exit code 9 — it can never be
reported as success — and prints the exact rebind or replug needed to recover.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(cleanup failed)* | 9 | Overrides every row below, success included; recovery guidance is printed |
| `refused` | 10 | Preconditions not met; nothing was opened or detached |
| `detach_failed` | 11 | `USBDEVFS_DISCONNECT` failed; nothing was taken, nothing to undo |
| `claim_failed` | 12 | `USBDEVFS_CLAIMINTERFACE` failed; the driver was reconnected |
| `interrupted` | 13 | SIGINT/SIGTERM inside the ownership window; cleanup still ran |

**Current measured result on this unit: `ok`.** The device returned the full 60 bytes,
`v2.8.7.20220610|v7.5.7.0426`, in 1.3 ms — the first vendor bytes any code in this repository has
read, and they match what the vendor SDK independently reported for this unit. The detached window
lasted 127 ms, cleanup succeeded, and the device came back unchanged apart from `/dev/video0` and
`/dev/video1` being recreated by the `uvcvideo` re-probe.

That is **one read-only request**. Nothing about streaming or calibration is implied, and the effect
of a detach on an *active* capture has not been measured. See
[docs/architecture.md](docs/architecture.md#owned-code-reads-the-vendor-channel-measured).

## The `/tmp/inited` sequence (opt-in, WRITES to the scanner)

`scanner3d inited-sequence` is the only command in this project that sends bytes *to* the scanner.
It reproduces, exactly once and in order, the four 60-byte transfers the vendor SDK was observed
performing on this unit — rows 1–4 of the byte-by-byte capture in
[docs/architecture.md](docs/architecture.md#the-vendor-channel-byte-by-byte):

```
1. SET_CUR  bmRequestType=0x21 bRequest=0x01 wValue=0x0100 wIndex=0x0400 wLength=60
            payload: 01 "/tmp/inited" 00, zero-padded to 60
2. GET_CUR  bmRequestType=0xa1 bRequest=0x81 wValue=0x0200 wIndex=0x0400 wLength=60
3. GET_CUR  the same request again
4. SET_CUR  the same packet as (1), payload: 60 zero bytes   <- protocol cleanup
```

`wIndex=0x0400` is `(entity 4 << 8) | interface 0`, so every transfer addresses the extension unit
the device's own descriptors declare, on the interface the lifecycle takes from `uvcvideo`. Nothing
here reaches interface 4 or the HID function.

**Those four transfers are the only thing this command can send.** The path is a compile-time
constant, not an argument: there is no flag for a path, an opcode, a selector, an entity, an
interface, a length, a timeout or a repetition, and the other two paths in the same capture
(`/persist/sdk-license`, `/data/sdk-license`) do not appear in the module's executable source at
all. The whole sequence is validated before a file descriptor is opened, and each transfer again
before its ioctl argument is packed. Offline tests assert the exact packed setup bytes and the exact
outbound payload bytes.

The ownership lifecycle is the same one `--detach-uvcvideo-from-interface-0` uses, shared rather
than copied, with **one extra guarantee**: the clearing write in step 4 is protocol cleanup, not an
optional fifth idea. It is owed from the instant *before* the path write is attempted — so an
interrupted or ambiguous write still owes it — attempted exactly once, shielded from SIGINT and
SIGTERM, and run *before* the driver is given back, because it needs the claim. The driver handback
then runs from a `finally` around it, so a clearing write that fails, or raises outright, can never
cost you your driver. The two undos are reported separately and neither can be presented as success.

The live mode requires an explicit `--serial` and the deliberately separate `--execute-write-sequence`
flag — not the read command's `--execute` — so a habit formed on the read-only command cannot write
to the device by accident. Without it, nothing is opened and nothing is sent.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(driver cleanup failed)* | 9 | Outranks every row below; interface 0 may still be detached |
| *(clearing write failed)* | 14 | Outranks every row below except 9; the device may still hold the path selected |
| `ok` | 0 | All four transfers matched the SDK-observed sequence exactly |
| `not_attempted` | 0 | Dry run; nothing opened, nothing sent |
| `device_absent` | 2 | No device with that serial, or its usbfs node is gone |
| `permission_denied` | 3 | The usbfs node is not open-able by this user |
| `timeout` | 4 | A transfer exceeded its bounded 3 s timeout |
| `ioctl_error` | 5 | The kernel refused a transfer |
| `unexpected_content` | 6 | 60 bytes returned, but not what the SDK saw |
| `short_response` | 7 | A transfer moved fewer than 60 bytes |
| `device_replaced` | 8 | The device re-enumerated between resolution and the sequence |
| `refused` | 10 | Preconditions not met; nothing was opened, detached or written |
| `detach_failed` | 11 | `USBDEVFS_DISCONNECT` failed; nothing was taken, nothing written |
| `claim_failed` | 12 | `USBDEVFS_CLAIMINTERFACE` failed; nothing written, driver reconnected |
| `interrupted` | 13 | SIGINT/SIGTERM inside the window; both undos still ran |

**Current measured result on this unit: `ok`.** All four transfers matched the capture byte for
byte. The path write was accepted in full; the first read returned `0c 00 00 00` followed by
`v7.5.7.0426`, NUL-padded to 60 bytes; the second read returned 60 zeros; the clearing write was
accepted in full. The detached window lasted 40 ms, both cleanups succeeded, the kernel log gained
exactly one line (`uvcvideo` re-probing), and the device came back unchanged apart from
`/dev/video0` and `/dev/video1` being recreated.

**This is the first owned write, and the first multi-transfer vendor exchange reproduced without the
SDK.** It does not mean `/tmp/inited` was read — the reply is the algorithm version string, which is
not plausibly that path's contents — and it says nothing about any other path, selector, entity or
`SET_CUR`. The licence gate is unchanged and remains inside `lib3DCamera.so`. See
[docs/architecture.md](docs/architecture.md#owned-code-writes-to-the-vendor-channel-measured).

## Depth intrinsics (opt-in, WRITES a key to the scanner)

`scanner3d depth-intrinsics` performs, exactly once, the bounded three-transfer exchange the vendor
firmware's `libuvcReadFile` serves for the factory depth-intrinsics key `/data/camparam/Pl.bin`. It
is the same recovered protocol, the same entity and the same lifecycle as `inited-sequence`, with
one byte-string changed — the key — and a bounded chunk loop in the middle:

```
1. SET_CUR  bmRequestType=0x21 bRequest=0x01 wValue=0x0100 wIndex=0x0400 wLength=60
            payload: 01 "/data/camparam/Pl.bin" 00, zero-padded to 60
2. GET_CUR  bmRequestType=0xa1 bRequest=0x81 wValue=0x0200 wIndex=0x0400 wLength=60
            reply: uint32_le chunk length, then that many bytes; repeated until a zero length
3. SET_CUR  the same packet as (1), payload: 60 zero bytes   <- protocol cleanup
```

**This is not a file reader.** The key is a compile-time constant and the payload is derived from it
at import time. No function and no flag takes a path, an opcode, a selector, an entity, an
interface, a length, a timeout, a cap, a capacity or a repetition count. The sibling calibration
keys (`Prgb`, `LC_RT`, `Distort`), the licence paths and the write-file opcode do not appear in the
module's executable source at all, and the depth-scale channel (`wValue=0x0e00`) is not reachable
from it either.

Every bound is fixed at import and never widened at runtime: the output capacity is **256 bytes**, a
run performs at most **8** result reads — so at most **10** control transfers including the key
write and the clearing write — and a chunk declaring more than **56** bytes (more than fits behind
the 4-byte header) is rejected *before* anything is copied. A chunk that would pass the capacity is
not copied at all rather than truncated, and no transfer is ever retried.

The assembled bytes are the primary evidence: their length, SHA-256 and full hex are reported
whatever they turn out to be. Only a complete blob of exactly 40 bytes is additionally parsed as the
vendor `Intrinsics` record (`short width, height` then nine x86 little-endian floats laid out as
`fx, 0, cx, 0, fy, cy, 0, 0, 1`), and the parse never alters, normalizes or substitutes a value: a
structurally divergent record comes back with the divergence recorded, as typed evidence rather than
a crash — and never as accepted calibration. Structural checks and plausibility observations are
kept apart, and only a run whose clearing write *and* driver handback both succeeded offers the
record as calibration at all.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(driver cleanup failed)* | 9 | Outranks every row below; interface 0 may still be detached |
| *(clearing write failed)* | 14 | Outranks every row below except 9; the key may still be selected |
| `ok` / `complete` | 0 | A 40-byte, structurally exact record, both undos done |
| `not_attempted` | 0 | Dry run; nothing opened, nothing sent |
| `no_content` | 15 | The first reply carried a zero length word (cause not distinguishable) |
| `cap_exhausted` | 16 | 8 reads without a zero length word; the cap was not widened |
| `capacity_exhausted` | 17 | The next chunk would pass 256 bytes; it was not copied |
| `malformed_length` | 18 | A chunk declared more than 56 bytes; nothing was copied |
| `unexpected_content` | 6 | Terminated, but not a structurally exact 40-byte record |
| `short_response` | 7 | A transfer moved fewer than 60 bytes |
| `timeout` / `ioctl_error` | 4 / 5 | A transfer timed out at 1000 ms, or the kernel refused it |
| `refused` | 10 | Preconditions not met, or a serial other than the authorized one |
| `interrupted` | 13 | SIGINT/SIGTERM inside the window; both undos still ran |

The live mode requires an explicit `--serial`, its own `--execute-calibration-read` flag (not the
other commands' `--execute` or `--execute-write-sequence`), and refuses any serial but the single
unit the owner authorized this read for.

**Current measured result on this unit (2026-08-21, one run): `ok` / `complete`.** The key write was
accepted in full; the first result read returned `28 00 00 00` followed by a 40-byte chunk; the
second returned a zero length word; the clearing write was accepted in full. Two result reads, four
control transfers in total, a 5.4 ms read inside a 37.5 ms owned window, both cleanups clean, and
the device came back with interfaces 0–3 on `uvcvideo`, interface 4 on `usbhid` and only
`/dev/video0`/`/dev/video1` recreated. The 40-byte blob
(`fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf`) parses exactly as the vendor
layout:

```
frame  640x400                       (a mode this unit's own descriptors advertise)
fx     898.6829833984375   fy  898.6829833984375
cx      42.43747329711914  cy  192.8326873779297      (pixels of that 640x400 frame)
matrix [[898.68298, 0, 42.437473], [0, 898.68298, 192.83269], [0, 0, 1]]
```

All five structurally fixed entries were exact (`0, 0, 0, 0, 1`) and every float finite, which is
what makes the row-major pinhole reading of the layout more than a guess. One plausibility
observation is flagged and is recorded rather than explained: **cx sits 277.56 px left of the
320 px frame centre**, while cy is within 7.2 px of the vertical centre. *Inference, not
measurement:* a principal point that far off-centre is consistent with the record describing a
cropped or offset sensor window rather than the centred 640×400 frame it names — but nothing here
establishes that, and the values are reported exactly as they arrived. This read also says nothing
about metric accuracy, coordinate handedness, RGB alignment or depth scale; those need their own
evidence. See
[docs/architecture.md](docs/architecture.md#owned-code-reads-the-factory-depth-intrinsics-measured).

## Depth scale (opt-in, WRITES to the scanner)

`scanner3d depth-scale` performs, exactly once, the bounded two-transfer exchange the vendor
firmware's `libuvcReadValueExtension` serves for the depth-scale property `0x918`. It shares the
same entity and the same ownership lifecycle as `depth-intrinsics`, but it is a smaller primitive —
a closed two-transfer whitelist, not a property reader — and unlike the file-read protocols above it
has no protocol-clear transfer at all:

```
1. SET_CUR  bmRequestType=0x21 bRequest=0x01 wValue=0x0e00 wIndex=0x0400 wLength=4  timeout=1000 ms
            payload: 18 09 00 00 (0x918, little-endian)
2. GET_CUR  bmRequestType=0xa1 bRequest=0x81 wValue=0x0e00 wIndex=0x0400 wLength=60 timeout=1000 ms
            reply: the vendor's fixed 60-byte reply; byte 59 is the marker, bytes 0:4 the
            signed little-endian int32 denominator
```

**There is no third transfer, and none is invented.** The vendor helper's `memset(buf, 0, 60)`
between the two calls clears only its own host-side buffer; the function issues exactly these two
transfers and no clearing write of its own, so the file-read protocols' trailing 60-zero clearing
write is deliberately not borrowed here. `protocol_cleanup` reports "not applicable" and says why.
The driver handback — release, then reconnect — still runs on every exit, exactly as the shared
lifecycle enforces it elsewhere.

**This is not a property reader.** The property id `0x918` is a compile-time constant and the
4-byte payload is packed from it at import time. No function and no flag takes a property id, a
selector, an entity, an interface, a length, a timeout, a count or a retry, and no other `0x9xx` id
appears in the module's executable source. The whitelist is checked as a whole before anything is
opened, and each transfer again immediately before its ioctl argument is packed.

The 60-byte reply is preserved in full — length, SHA-256 and hex — as the primary evidence, and the
decode is reported beside it rather than in place of it. Byte 59 is the vendor's marker, accepted
only when it is `4`; bytes 0:4 are the signed little-endian `int32` denominator the vendor divides
`1.0f` by. Only an exact 60-byte reply carrying that marker and a positive denominator yields a
scale, and that scale is `float32(1.0f / denominator)`, reported with its exact bit pattern.

| Status | Exit | Meaning |
| --- | --- | --- |
| *(driver cleanup failed)* | 9 | Outranks every row below, success included |
| `ok` / `complete` | 0 | An exact 60-byte reply, marker 4, positive denominator, driver handback done |
| `not_attempted` | 0 | Dry run; nothing opened, nothing sent |
| `unexpected_marker` | 19 | Byte 59 was not 4; the reply is reported in full, no scale offered |
| `zero_denominator` | 20 | Bytes 0:4 were 0; the reciprocal is undefined |
| `negative_denominator` | 21 | Bytes 0:4 were negative; no scale offered |
| `non_finite` | 22 | Defensive only — unreachable through the gates above |
| `unexpected_content` | 6 | An exact 60-byte reply that failed one of the gates above |
| `short_response` | 7 | Fewer than 60 bytes returned on either transfer |
| `timeout` / `ioctl_error` | 4 / 5 | A transfer timed out at 1000 ms, or the kernel refused it |
| `refused` | 10 | Preconditions not met, or a serial other than the authorized one |
| `interrupted` | 13 | SIGINT/SIGTERM inside the window; the driver handback still ran |

The live mode requires an explicit `--serial` and its own `--execute-depth-scale-read` flag (not
any other command's execute flag), and refuses any serial but the single unit the owner authorized
this read for. The subcommand is parsed with `allow_abbrev=False`, so `--execute-depth-scale-read`
must be typed in full — argparse will not accept a truncated prefix of it, unlike most other flags
in this CLI.

**Current measured result on this unit (2026-08-21, one run): `ok` / `complete`.** The property
write was accepted in full (`18090000`); the result read returned the full 60 bytes in 17.715 ms,
SHA-256 `9873fa193adabb7daa87a77718d963d586d798838f4e6edc1a2ddac0bb337dd4`, hex
`c80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004`.
Byte 59 was `4` and bytes 0:4 decoded to the denominator `200`, giving
`vendor_depth_scale = float32(1.0f / 200) = 0.004999999888241291` (float32 bits `0x3ba3d70a`).
Exactly two control transfers, no retry, a 19.426 ms read inside a 27.880 ms owned window, and the
device came back with identical serial and devnum (no re-enumeration), identical descriptors, and
interfaces 0–3 back on `uvcvideo` and interface 4 on `usbhid`.

This value is bit-identical to the vendor SDK's compiled-in MINI fallback constant `0.005f` — an
**observation**, not a unit: **the scalar's physical unit is unknown from this evidence.** Two
transfers and one reciprocal prove what the device returned, not what it means, and nothing here
establishes millimetres, metres, accuracy, or a raw-depth conversion factor. That the SDK's authors
chose `0.005f` because it matches this device's reported value is, at most, an inference the
agreement is consistent with — never a fact this run establishes. See
[docs/architecture.md](docs/architecture.md#domain-4--calibration--depth-scale-fact-disassembled).

## Reference probe (opt-in, touches hardware)

The probe is a small standalone C++ program built against an **externally supplied** SDK. It is
never run unless you explicitly ask for it.

```bash
# Build (fails loudly if the SDK is absent; never downloads anything)
REVOPOINT_SDK_ROOT=/path/to/3DViewer/thirdparty/3DCamera reference/probe/build.sh

# Run through doctor, under a hard timeout, with SDK hashes verified
scanner3d doctor --reference-probe \
  --sdk-root /path/to/3DViewer/thirdparty/3DCamera \
  --probe-binary /tmp/3d-scanner-probe/probe \
  --probe-timeout 12 --json
```

The adapter refuses to run unless the SDK's SHA-256 hashes match
[`src/scanner3d/data/sdk-manifest.json`](src/scanner3d/data/sdk-manifest.json)
(override with `--allow-unverified-sdk`). The manifest ships inside the
`scanner3d` package itself so an installed CLI can verify the oracle without
the repository present. It caps the run at 15 s, kills the whole process group on timeout, and
leaves no stream open. See [`reference/probe/README.md`](reference/probe/README.md).

## usbfs tracer (opt-in, observation only)

[`reference/tracer/`](reference/tracer/README.md) is an `LD_PRELOAD` ioctl interposer that records
the `USBDEVFS_*` calls one process makes to **one exact** usbfs node, and `scanner3d trace` decodes
the result offline. It exists because the vendor SDK reaches this scanner entirely through
`ioctl()` on `/dev/bus/usb/BBB/DDD` using asynchronous URBs, which `strace` can only show as bare
pointers — from inside the process, the wire bytes are readable.

```bash
reference/tracer/build.sh                     # -> /tmp/3d-scanner-trace/libusbfs_trace.so

SCANNER3D_TRACE_TARGET=/dev/bus/usb/001/041 \
SCANNER3D_TRACE_OUT=/tmp/3d-scanner-trace/capture.jsonl \
LD_PRELOAD=/tmp/3d-scanner-trace/libusbfs_trace.so \
  timeout 15 /tmp/3d-scanner-probe/probe --serial EXAMPLESERIAL0001

scanner3d trace /tmp/3d-scanner-trace/capture.jsonl.<pid>
```

It observes and never alters: every hooked call is forwarded to the real libc symbol with its
arguments, return value and `errno` untouched, and the tracer issues no device operation of its
own. Capture is off unless both environment variables are set, and it fails closed — an absent
target, a target that is not a character device under `/dev/bus/usb/`, or an output path that
already exists all mean *nothing is recorded*, never *everything is recorded*. Payload bytes, line
length and total output size are all capped, and hitting a cap is written into the stream rather
than dropped silently. Output is one file per process (`<path>.<pid>`), because a traced program
may fork and exec.

`scanner3d trace` is a pure file read: no device, no SDK, no network. It validates the stream,
correlates URB submissions with their completions deterministically, and preserves unmatched,
failed and truncated events as evidence instead of discarding them.

What one bounded capture of the vendor SDK established about this scanner's control channel is in
[`docs/architecture.md`](docs/architecture.md#vendor-control-channel-measured-on-the-wire).

## Repository layout

```
src/scanner3d/
  errors.py            status vocabulary and exit codes
  usb_descriptors.py   tolerant parser for the raw USB config descriptor
  discovery.py         sysfs discovery, rooted for fixture replay
  control.py           the single whitelisted read-only USB control request
  ownership.py         the interface-0 ownership lifecycle, and that one read
  init_sequence.py     the four-transfer /tmp/inited sequence on entity 4
  intrinsics.py        the bounded /data/camparam/Pl.bin depth-intrinsics read
  depth_scale.py       the closed two-transfer 0x918 depth-scale property read
  v4l2_capture.py      the shared bounded V4L2/MMAP single-frame engine and the depth profile
  rgb_frame.py         the fixed MJPG 1280x800 RGB capture profile for that engine
  jpeg_inspect.py      dependency-free structural JPEG reader (markers only, no pixel decode)
  doctor.py            report assembly and status classification
  reference.py         opt-in external SDK probe adapter
  usbfs_trace.py       offline decoder for recorded usbfs traces
  cli.py               `scanner3d` entry point
reference/probe/       standalone C++ reference probe (source only)
reference/tracer/      LD_PRELOAD usbfs ioctl tracer (source only)
docs/                  architecture, roadmap, and measured-hardware notes
tests/                 fixture-driven test suite
```
