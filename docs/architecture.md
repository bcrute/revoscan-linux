# Architecture

## Summary

Today this project provides owned, dependency-light Linux **discovery** of the
Revopoint MINI: it identifies the device over sysfs, parses its raw USB
configuration descriptor to enumerate advertised video modes, checks node
permissions, and reports a diagnostic verdict (`scanner3d doctor` /
`scanner3d list`). Owned code has also captured one raw depth frame and, in one
accepted paired V4L2 run plus one unaccepted context run, delivered RGB payloads whose decodable 1280x800 JPEG
prefixes each ended with four opaque bytes after EOI. Both are strict-invalid
`trailing_data`; the observed images were nearly black/faint, so no useful RGB
claim is made. This remains measured evidence, not a repeatable capture path.
Two
bounded RGB attempts were made. The first refused itself at format negotiation,
because the driver reported a buffer size four times the device descriptor's
own — see
[Owned code attempts a raw RGB frame](#owned-code-attempts-a-raw-rgb-frame-measured).
The second, with the ceiling re-derived from that measurement, mapped four
8192000 B buffers and started the stream, and then no buffer became ready
within its 8.0 s deadline — see
[Owned code attempts a raw RGB frame with the measured ceiling](#owned-code-attempts-a-raw-rgb-frame-with-the-measured-ceiling-measured).
That silence has since been analysed offline, without touching the scanner:
the descriptor rules out bandwidth and alternate-setting negotiation entirely,
the depth function is a positive control on the identical kernel path, and the
kernel was configured (`trace=0`) such that it could not have reported what
happened after `STREAMON` — see
[Why the RGB stream is silent](#why-the-rgb-stream-is-silent-offline-reconstruction-scanner-012).

Nor, as it turns out, through any borrowed one. The intended source of frame
"ground truth" for this milestone was an external oracle — the closed-source
Revopoint SDK, invoked out-of-process through a standalone C++ probe
(`reference/probe/probe.cpp`) and gated by a pinned file-hash manifest
(`src/scanner3d/data/sdk-manifest.json`, shipped inside the package so an
installed CLI can verify the oracle without the repository present). That
oracle **enumerates this unit correctly
but cannot connect to it**: `connect()` fails with `ERROR_NOT_SUPPORT`
because the SDK's internal licence authorizer cannot find an `sdk-license`
blob on the camera. The first two owned standalone RGB attempts stopped before
delivery (one before streaming and one after 8.0 s with no buffer), see
[Owned code attempts a raw RGB frame](#owned-code-attempts-a-raw-rgb-frame-measured)
and
[Owned code attempts a raw RGB frame with the measured ceiling](#owned-code-attempts-a-raw-rgb-frame-with-the-measured-ceiling-measured)
— and no depth frame has ever come from the vendor SDK. Owned code has
obtained three things the SDK never reached: two calibration values, and
separately, one raw depth frame. It read the factory depth-intrinsics key
`/data/camparam/Pl.bin` off the unlicensed device as a 40-byte vendor
`Intrinsics` record, and separately read the depth-scale property `0x918`
off the same device as an exact 60-byte reply — see
[Owned code reads the factory depth intrinsics](#owned-code-reads-the-factory-depth-intrinsics-measured)
and [Domain 4](#domain-4--calibration--depth-scale-fact-disassembled). It also
captured one raw `Y16` 640×400 depth frame over plain V4L2, once, with neither
calibration value applied to it — see
[Owned code captures a raw depth frame](#owned-code-captures-a-raw-depth-frame-measured).
The SDK's own failure is unchanged; see
[Reference-oracle result](#reference-oracle-result-the-vendor-sdk-cannot-connect-to-this-unit)
for the full evidence.

What the blocked oracle *can* still do is be watched. `reference/tracer/`
records the SDK's usbfs traffic from inside its own process, and
`scanner3d trace` decodes it offline. The probe was run three times against
the hardware under the tracer; the first two produced no protocol data
because of tracer defects, fixed in between runs. The third — one bounded
capture of the failing `connect()` — established which interfaces, endpoints
and transfer types the SDK really uses, and recovered the actual wire bytes
of the vendor control channel up to the point authorization fails — see
[Vendor control channel](#vendor-control-channel-measured-on-the-wire). That is
observation only: that capture sent nothing itself, and the licence gate
remains a blocker rather than something worked around.

Owned code has since sent requests of its own, on a different footing. Across
four detach/claim runs it has issued **eleven** control transfers to this
unlicensed unit — a version read, the `/tmp/inited` write/read/clear exchange,
the `/data/camparam/Pl.bin` calibration read that returned the factory
depth intrinsics, and the two-transfer depth-scale property read. That is a
bounded, self-clearing foothold on the vendor control channel, not a
streaming transport: no ownership takeover was ever needed for it. Separately,
and without touching that channel at all, owned code has also pulled one raw
frame straight off `/dev/video0` through plain V4L2, once (`scanner-009`) —
the first depth frame this project has obtained by any path, from a device
that kept `uvcvideo` bound throughout.

So calibration is started but partial — one intrinsics key and the
depth-scale property, each read once — and owned transport is started but
partial too: one raw depth frame and two paired RGB payloads, each run once,
with no calibration applied and no useful-RGB or synchronization claim. The
sibling calibration keys, extrinsics and
distortion are untouched, and everything past that — point clouds,
registration, mesh export, and mapper artifacts — is planned and not started.

## Scanner-014 evidence and the next paired experiment

The scanner-014 usbmon capture is preserved offline in
`tests/fixtures/scanner-014-usbmon-target.txt`. It contains exactly the 26
records for bus 1/devnum 43 from the 810-line, 58,074-byte raw capture
(`15b43371f426c3efb617ce58d805b9f9c04519562fb641042dfd5259b9e2d664`). The
first wrapper accidentally matched `:43:` while usbmon rendered `:043:`;
the fixture records that defect and uses the bounded zero-padding-tolerant
selection rule. The offline decoder confirms successful 30 Hz Probe/Commit
negotiation (`sizeimage=8,192,000`, payload ceiling 524,288), five submitted
bulk URBs, and five cleanup completions (`-2`, zero bytes). There was no
successful RGB bulk completion, so this is not a frame or JPEG artifact and
does not establish physical RGB support.

The pinned 3DViewer source is fact at commit
`516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`: `CSCamera::startStream()` calls
`startDepthStream()` before `startRgbStream()` in
`src/cscamera/cscamera.cpp`, then applies default parameters only after both
streams report started. The local reference library is separately pinned by
SHA-256 `49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`.
The owned inference is therefore a safer next hardware experiment: resolve
both roles by serial, start depth, preserve the SDK's one-second interval,
start RGB, and wait once for one depth and at most one RGB frame. Firmware
initialization, RGB delivery, pairing quality, and synchronization remain
unknown until that experiment is explicitly run. The scanner-016 backend was
accepted offline; its accidental physical invocation is retained as
unaccepted context. Scanner-017 is the accepted reviewed physical
reproduction. Their narrow JPEG-prefix-plus-four-byte shape is documented
below without promoting strict-invalid bytes to valid RGB frames or claiming
transport repeatability.

`scanner3d paired-capture` prints this plan by default and opens nothing. The
concrete execution path requires the exact serial, the distinct opt-in
`--start-depth-first-rgb-second`, and a local `--output-dir`. It reuses the
single-frame V4L2/MMAP engine for both fixed profiles; preflight rejects
profile/role mismatches and node aliasing before opening, while the opened
node's identity is validated before the first ioctl. It starts depth before RGB, waits at most once
per role against absolute per-stream deadlines, and preserves each role's
existing artifact judgement. It has no retry, alternate mode, reset,
usbfs/HID/XU operation, or vendor SDK fallback. Cleanup is RGB reverse
teardown followed by depth reverse teardown on every exit; cleanup failure
overrides the primary result with exit 9. Offline verification is synthetic;
the accepted physical evidence below is reported separately and does not
establish useful RGB, synchronization, or repeatability.

`scanner3d paired-rgb-series` is the offline-verified follow-on diagnostic. It keeps the same
identity-resolved depth-first/RGB-second lifecycle, waits once for depth, then gives RGB one
absolute 8.0-second deadline and exactly three dequeue opportunities. Between observations it
requeues only the returned buffer; after the third observation it does not requeue. The shared
V4L2 engine accumulates every poll, dequeue and requeue record rather than overwriting the
single-frame state. Each payload is preserved before its requeue under an exclusive indexed name
(`rgb-series-000.jpg` through `rgb-series-002.jpg`) and reports its V4L2 metadata, SHA-256, strict
JPEG report, EOI-bounded prefix length/hash and opaque trailer length/hex. `trailing_data` remains
strict-invalid even when a complete JPEG prefix is present; no suffix is interpreted or trimmed.
The diagnostic is inert without `--start-depth-first-rgb-series`, the authorized serial and an
explicit output directory. The live-capable implementation is verified only with injected/fake
V4L2 boundaries. Scanner-019 is the accepted physical three-observation run: all three payloads
had four opaque trailing bytes after their JPEG EOI, decoded nearly black, and showed no meaningful
warm-up improvement. Scanner-021 subsequently ran the exact read-only standard-control inventory
once; its measured result is recorded below. Neither diagnostic establishes useful RGB,
synchronization, or scan accuracy.

### The fixed RGB control inventory (scanner-020 implementation and scanner-021 evidence)

Scanner-019's three physical RGB payloads all had decodable 1280x800 JPEG
prefixes but were strict-invalid trailing-data payloads and nearly black. To
measure the camera's reported standard state without changing it, scanner-020
adds `scanner3d rgb-controls`. The command is inert unless the exact
`--read-rgb-controls` opt-in and authorized serial are supplied. Scanner-021 is
the one accepted physical invocation described below.

The live-capable path reuses the existing serial/interface-string/V4L2-index RGB
resolution and post-open identity check, opens only the resolved capture node
read-only/nonblocking, then closes it exactly once. Its import-time whitelist is
`V4L2_CID_GAIN`, camera-class `V4L2_CID_EXPOSURE_AUTO`,
`V4L2_CID_EXPOSURE_ABSOLUTE`, `V4L2_CID_EXPOSURE_AUTO_PRIORITY`,
`V4L2_CID_AUTO_WHITE_BALANCE`, and
`V4L2_CID_WHITE_BALANCE_TEMPERATURE`. Each receives one `QUERYCTRL`; a supported
scalar or menu control gets at most one `G_CTRL` unless it is disabled or
write-only. Read-only, inactive and volatile flags remain readable evidence,
not permission to change anything. No control write, menu enumeration, format
negotiation, buffer/stream operation, depth access, vendor extension, usbfs,
HID or XU operation is in this diagnostic. Values remain device-reported
integers/booleans/menu indices; they do not establish exposure, brightness,
colour accuracy or a recommendation to change settings.

Scanner-021 preflight resolved serial `A22704522A6F00A64` at USB path `1-1.1`,
devnum 46, RGB interface 2 (`RGBCamA22704522A6F00A64`), `uvcvideo`, V4L2
capture index 0 at `/dev/video2`, mode `0660`, with read access and no holder.
The exact command ran once, exited 0 with `complete`/`succeeded=true`, closed
once, reported stable post-close identity, and issued 11 ioctls of the fixed
12-call ceiling. Its stdout and `outcome.json` were byte-identical and
JSON-equivalent. The complete device-reported result was:

| Control (ID) | QUERYCTRL metadata | Flags | Current/status |
| --- | --- | --- | --- |
| gain (`0x00980913`) | `Gain`, integer, 1..16 step 1, default 7 | none | 16, supported |
| exposure_auto (`0x009a0901`) | `Auto Exposure`, menu, 0..3 step 1, default 0 | none | 3, supported |
| exposure_absolute (`0x009a0902`) | `Exposure Time, Absolute`, integer, 1..60000 step 1, default 496 | `INACTIVE` | 131072, inactive |
| exposure_auto_priority (`0x009a0903`) | `QUERYCTRL` returned `EINVAL` | — | unsupported; no value read |
| auto_white_balance (`0x0098090c`) | `White Balance, Automatic`, boolean, 0..1 step 1, default 1 | none | 1, supported |
| white_balance_temperature (`0x0098091a`) | `White Balance Temperature`, integer, 2800..7500 step 10, default 4600 | `INACTIVE` | 6500, inactive |

The values remain device-reported integers/menu or boolean values; no exposure,
brightness, colour-accuracy, safety or setting-change inference is made. Six
read-only postflight samples spanning 6.257 seconds kept both video-role
bindings/nodes, permissions, devnum 46 and zero holders stable (`depth`
`/dev/video0`, RGB `/dev/video2`). The preserved `outcome.json` SHA-256 is
`f6540078c317a5ce7cd8dc1be6f1fc23bf8f968917c2d402b96a92fbac8fc9c2`.

The scanner-016 accidental physical invocation is preserved as unaccepted
context; the scanner-016 backend itself was accepted offline.
Read-only inspection of the already-created `/tmp/scanner3d-paired-capture`
directory found a 512000-byte depth artifact (SHA-256 prefix/suffix
`0503b236738b95167a4cf283e886746460078a19f17cbc670820d9baf3189ee8`) and a 16947-byte RGB JPEG (SHA-256
`cc5f702fd18746a04d3bc5dcd814764bf93907a34f032231160a4df5b7069463`), both timestamped 17:45:46. The RGB payload had valid
1280x800 SOI/SOF/SOS/EOI structure but ended with `95 81 7d 01`; strict
validation therefore returned `trailing_data`. The four bytes were not trimmed
and no derived file was written. The hypothetical trimmed-byte hash was
recorded as `818e026092e3f342f3cc9b08c50efbc5cd7f83548cd2fde7e9446ac5c48f44ab`, not produced. This proves only that that accidental run
opened/streamed/dequeued both roles; it does not establish acceptance, useful
image content, synchronization, or repeatability. Scanner-017 subsequently
repeated the paired lifecycle once. It delivered a 22529-byte RGB payload
(SHA-256
`650240568e5b97fcc8f61773035e03e2b48fb5bb940c2d73de8572f83609c07`) with a
baseline 1280x800 JPEG prefix through EOI at offset 22523 and opaque trailer
`54 52 d8 01`; strict validation remained `trailing_data`. The hypothetical
22525-byte prefix hash was
`77ad1601dc234b059e5e076380ba8cb8c2b4f64f59c4f3fdb98ec2572b549199`, and no
trimmed file was written. In-place viewing found a nearly black image with
only faint features. Its depth payload was one complete 512000-byte `Y16 `
frame (SHA-256
`950c9640612dd135b104bcdff1ef7536da435ca4eb6e34f7043ee58a5cd10c6a`).
The two runs' four-byte suffixes and payload sizes differ; neither establishes
that the suffix has meaning, useful RGB, synchronization, or repeatability.
Scene-bearing files are not copied into the repository.

## Layers

| Layer | Purpose | Status | Done means |
|---|---|---|---|
| L0 — vendor-SDK oracle (temporary) | External, closed-source `lib3DCamera.so` driven via `reference/probe`, used only to establish ground truth (does the unit stream at all, real calibration values, control-handshake shape). Hash-pinned against `src/scanner3d/data/sdk-manifest.json` (packaged inside `scanner3d` so an installed CLI can verify the oracle without the repository present), never vendored or auto-downloaded. | Exists, but **blocked on this unit** — enumerates correctly, `connect()` fails `ERROR_NOT_SUPPORT` at a licence check, so it has yielded no frames and no calibration | Deleted. This layer is explicitly scaffolding, not the long-term transport; it is done being useful when L2/L3 reproduce what it tells us and it can be removed. |
| L1 — owned Linux discovery | Sysfs-only enumeration of USB topology, interface roles, advertised UVC formats, and device-node permissions, rooted at injectable `sysfs_root`/`dev_root` so tests can replay fixtures. No `/dev` I/O beyond `os.access`/stat-style checks. | Exists | Reliably identifies the scanner, its roles, and its advertised modes on any Linux host with the device attached, with tests covering fixture replay and edge cases (multiple devices, missing nodes, permission denial). |
| L2 — owned Linux transport | Actually pulling depth and RGB frames without the vendor SDK: either V4L2 reads from `/dev/video0`/`/dev/video2` plus the vendor control channel (measured to be UVC class requests on endpoint 0 to entities 3 and 4 of interface 0, **not** the HID interrupt endpoint), or a from-scratch libusb/usbfs client. | **Started, partial** — owned code captured one raw `Y16` 640×400 depth frame straight off `/dev/video0` through plain V4L2, once, and scanner-017 is the one accepted reviewed physical paired reproduction. Executed depth capture now has a mandatory post-teardown blue-light finalizer: one immutable interface-0 `SET_CUR` (`0x21/0x01`, `wValue=0x0700`, `wIndex=0x0400`, 60 bytes) through the shared detach/claim/release/reconnect lifecycle. Scanner-026 implements and tests that obligation offline; no scanner-026 physical invocation or optical readback exists. The accepted scanner-017 run delivered one 22529-byte RGB payload with a decodable 1280x800 JPEG prefix plus four opaque trailing bytes, but strict status remained `trailing_data`; a prior scanner-016 physical invocation is unaccepted context from the offline-accepted backend. | A frame (raw depth + RGB) can be captured end-to-end by code in this repository with the vendor SDK absent, with explicit reported blue-light finalization evidence but no optical darkness claim. |
| L3 — calibration model | Turning raw Y16/YUY2 depth-sensor output into metric depth using device-stored intrinsics/depth-scale (not exposed via UVC descriptors), plus RGB-depth extrinsics. | **Started, partial** — owned code has retrieved two device-stored inputs on this unit: the factory depth intrinsics from `/data/camparam/Pl.bin` (scanner-007) and the `0x918` depth-scale scalar (scanner-008, decoded to `float32(1.0f/200)`), and separately captured one raw depth frame (scanner-009, L2). Still missing: applying either calibration input to that frame — its 16-bit samples carry no established unit, scale or zero/invalid convention yet — the depth scale's physical unit, the RGB-depth extrinsics, the metric conversion itself, and any validation of the result | A depth frame converts to millimetre-scale values that agree with the vendor SDK's calibrated output within a stated tolerance. |
| L4 — point-cloud generation | Deprojecting calibrated depth (+ RGB) frames into 3D point clouds. | Not started | A single-frame point cloud can be produced and visually/quantitatively sanity-checked against SDK output. |
| L5 — registration/fusion | Aligning and merging point clouds across frames/poses into a consistent scene. | Not started | Multiple captures of a known object register within a stated accuracy and fuse without visible seams on a test target. |
| L6 — mesh + export | Surface reconstruction from fused point clouds and export to standard mesh formats. | Not started | A fused scan exports to a mesh format openly loadable in common tooling (e.g. MeshLab), with no vendor tooling involved. |
| L7 — mapper artifacts | Producing whatever derived outputs the sibling mapping project consumes. | Not started | Defined by the sibling project's actual input contract, which has not yet been specified here. |

## Measured hardware facts

### USB topology

| Property | Value |
|---|---|
| USB identity | `2207:110c` |
| Manufacturer string | `Chishine3d` (Revopoint's OEM) |
| Product string | `ZXProduct` |
| USB serial | `A22704522A6F00A64` |
| bcdDevice | `0419` |
| bcdUSB | `2.00` |
| sysfs path | `/sys/bus/usb/devices/1-4.1` at time of measurement; later observed at `1-1` (direct) and `1-1.1` (behind the hub) after re-cabling (**not stable** — see note) |
| Bus / port path | bus 1, port 4.1 at time of measurement (**not stable**) |
| devnum | 24 at time of measurement (changes on every replug) |
| Negotiated link speed | 480 Mb/s (USB 2.0 high-speed), measured both behind a hub and attached directly to a SuperSpeed-capable root port — see **Link speed** below |
| Interfaces | 5, grouped under 2 Interface Association Descriptors (IADs) |

Bus, port path, sysfs path and `devnum` above are a snapshot, not identity. They
change on replug and on re-cabling; discovery therefore matches on USB
vendor/product id plus serial, never on a recorded path (see
`src/scanner3d/discovery.py`).

#### Link speed

Three observations, all on 2026-08-21, in the order they were taken:

1. **Scanner behind the Realtek hub, hub on a USB 2.0 upstream port.** The
   whole chain — Bus 001 root hub, Realtek hub, and scanner — enumerated at
   480 Mb/s on Bus 001, with the 10 Gb/s Bus 002 empty. On its own this said
   nothing about the scanner's ceiling: every upstream link was USB 2.0, so a
   SuperSpeed-capable device would have enumerated through its USB 2.0
   companion regardless. This observation is **superseded** as evidence about
   the scanner.
2. **Scanner behind the same hub, hub moved to a SuperSpeed upstream port.**
   The hub then presented both companions at once: `0bda:0401` on Bus 002 at
   5000 Mb/s (`/sys/bus/usb/devices/2-2`, `speed=5000`, `version=3.00`) and
   `0bda:5401` on Bus 001 at 480 Mb/s (`/sys/bus/usb/devices/1-1`,
   `speed=480`, `version=2.00`). The scanner attached below it enumerated as
   `usb 1-1.1: new high-speed USB device`, `2207:110c`, serial
   `A22704522A6F00A64` — 480 Mb/s on the hub's USB 2.0 side, while the hub's
   SuperSpeed side was simultaneously live. This ruled out the hub's
   *upstream* path but left the scanner-to-hub segment itself untested.
3. **Scanner attached directly to that same root port, hub removed.** The
   kernel log records `usb 1-1: new high-speed USB device number 39`,
   `idVendor=2207, idProduct=110c, bcdDevice= 4.19`,
   `SerialNumber: A22704522A6F00A64`, both `Found UVC 1.00 device ZXProduct
   (2207:110c)` lines, and `hid-generic 0003:2207:110C...: hiddev2,hidraw10 ...
   on usb-0000:00:14.0-1/input4`; sysfs read host path `1-1`, `speed=480`,
   `version=2.00`, and all four video nodes plus the HID interface were present.
   Nothing enumerated on Bus 002 during that window, even though the same
   physical receptacle (PCI `0000:00:14.0` port 1) carried the hub's
   SuperSpeed companion at 5000 Mb/s immediately before and after it. The owner
   identifies the cable in use as USB 3-rated.

**Established:** neither the hub nor the host port limits this link. Attached
directly to a port proven SuperSpeed-capable, over an owner-identified
USB 3-rated cable, the MINI enumerates as a USB 2.0 high-speed device at
480 Mb/s. 480 Mb/s is therefore the transport this scanner actually presents on
this host; `doctor` reports it as the normal case, not as a fault.

**This is not a bandwidth defect.** The scanner advertises all of the UVC modes
listed below over this same 480 Mb/s link, and the kernel log for this boot
contains no bandwidth-allocation failure at all (no `-ENOSPC`, no "not enough
bandwidth") on any of the three topologies. Revopoint's MINI quick start guide
asks for a USB 3.0-or-newer host port because a USB 2.0 port may not supply
sufficient *operating power*; it does not state that the MINI's data link
negotiates SuperSpeed ([official guide][mini-guide]). The measured 480 Mb/s is
consistent with that wording. (Separately, 24 `reset high-speed USB device`
entries and several `device not accepting address, error -71` entries appear
against the earlier path `1-4.1` during vendor-probe runs; those are device
resets and re-enumeration failures, not bandwidth allocation, and none appeared
during the direct-attach window.)

**Remaining uncertainty, low stakes:** only one cable has been tried, so a
device ceiling and a USB 2-only cable have not been separated from each other.
`bcdUSB 2.00` cannot distinguish them, because a SuperSpeed-capable device
reached over a USB 2-only cable also enumerates through its USB 2.0 companion
descriptors. Nothing measured so far depends on which it is; see "Checks not
run".

[mini-guide]: https://download.revopoint3d.com/support/download/mini/mini-quickstartguide-en-v3.0-2023.7.27.pdf

| Interface(s) | Function string | Class | Driver | Device nodes |
|---|---|---|---|---|
| 0 (VideoControl), 1 (VideoStreaming) | `DepthCamA22704522A6F00A64` | UVC | `uvcvideo` | `/dev/video0` (capture, v4l index 0), `/dev/video1` (UVC metadata, v4l index 1) |
| 2 (VideoControl), 3 (VideoStreaming) | `RGBCamA22704522A6F00A64` | UVC | `uvcvideo` | `/dev/video2` (capture), `/dev/video3` (metadata) |
| 4 | (none / HID) | HID (class 03) | `usbhid` | `/dev/hidraw10`, `hiddev0` — both `0600 root:root`, **not** accessible to the user on this host |

### Advertised UVC modes

Parsed from the raw configuration descriptor blob at
`/sys/bus/usb/devices/1-4.1/descriptors` (645 bytes) using
`scanner3d.usb_descriptors`, with no root privilege or `/dev` access
required.

| Stream | Format | Resolutions | Frame rates | Default fps |
|---|---|---|---|---|
| Depth (interface 1) | YUY2 (uncompressed, 16 bpp) | 640x400, 1280x800 | 5 / 10 / 15 | 15 |
| Depth (interface 1) | Y16 (uncompressed, 16 bpp) | 640x400, 1280x800, 640x800 | 5 / 10 / 15 | 15 |
| RGB (interface 3) | MJPG | 1280x800 | 10 / 15 / 20 / 30 | 30 |

### Permissions

`/dev/bus/usb/001/024` is `0660 root:plugdev` but carries a systemd-logind
`uaccess` POSIX ACL granting user `ben` (uid 1000, groups include `plugdev`
but not `video`) read/write. `/dev/video0`–`/dev/video3` are `0660 root:video`
and are likewise rw for `ben` via the same uaccess ACL. As a result, both
usbfs and V4L2 access work for an unprivileged, seat-local user with **no
udev rule installed**.

The vendor SDK ships `scripts/cs_uvc.rules` setting `MODE="0666"` for
`2207:110a`/`110b`/`110c`. This project deliberately did not install that
rule, because the logind uaccess ACL already suffices for a seat-local user.
That rule would still be needed for a headless or non-seat user (e.g. a
system service with no logind session).

The vendor SDK (`lib3DCamera.so` 3.2.23, from
`github.com/Revopoint/3DViewer` commit `516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`)
statically links libusb and opens the device through usbfs
(`/dev/bus/usb/BBB/DDD`), not through V4L2. It does not touch `/dev/video*`.

## Why UVC discovery alone did not produce frames

The scanner enumerates as a standard UVC device, and the kernel's `uvcvideo`
driver binds all four video interfaces, so `/dev/video0` and `/dev/video2`
exist and the modes tabulated above are real UVC descriptors read straight
off the device. But enumerating — and even opening — those nodes is not
expected to be sufficient to get usable depth data, for reasons that are
observed structure rather than speculation:

1. **The depth "camera" is a structured-light sensor.** Its Y16/YUY2 stream
   is raw sensor data, not metric depth. Converting it to millimetres
   requires a depth scale and intrinsics that live on the device and are not
   exposed through any UVC descriptor — UVC only describes pixel formats and
   frame geometry, not a calibration model.
2. **There is a vendor control channel that UVC video never touches, and it
   has now been read off the wire.** The SDK detaches the kernel driver from
   and claims **both** interface 0 (UVC) and interface 4 (HID) over usbfs,
   and its own trace log prints `enter openHid` / `UvcHid::openHid OK.`
   before it attempts anything else.

   What the earlier `strace` could not show — and what
   [the usbfs capture](#vendor-control-channel-measured-on-the-wire) now
   does — is *where the bytes actually go*. They do not go over the HID
   interrupt endpoint. Every vendor request in the captured session is a
   **UVC class control request on endpoint 0**, addressed to an extension
   unit on interface 0. Interface 4 was claimed and one interrupt-IN URB was
   armed on endpoint `0x85`, but it was cancelled with zero bytes
   transferred before authorization failed, so HID carried no data at all in
   this session. Streaming raw pixels without the endpoint-0 control traffic
   is still not expected to yield usable depth.

   Note that the SDK reaches HID *through usbfs*, not through
   `/dev/hidraw*`: `strings lib3DCamera.so` contains zero `hidraw`/`hidapi`
   references, and the `strace` shows no `/dev/hidraw*` or `/dev/video*`
   access at all. `/dev/hidraw10` being root-only on this host is therefore
   irrelevant to the SDK path.

   **Caveat on the negative result.** `connect()` aborts at authorization,
   so this says only that HID carried nothing *before that point*. It is not
   evidence that a successful session would leave HID idle.
3. **The vendor SDK bypasses V4L2 entirely.** It drives the device over
   usbfs with its own statically-linked libusb. That means the one known
   working reference path is not "open `/dev/videoN` and read"; it is a
   distinct transport that must detach from, or otherwise coexist with,
   `uvcvideo`.

Points 1–3 are structural reasons a *plain* UVC stream is not expected to
deliver usable depth. A direct attempt was also made and measured, not just
inferred:

4. **A direct plain-UVC stream open was attempted; it opened and negotiated
   but delivered no frames.** GStreamer opened `/dev/video0` as 1280x800
   GRAY16_LE at 15 fps and `/dev/video2` as 1280x800 MJPEG at 30 fps,
   concurrently. Both pipelines negotiated the format and opened the device
   successfully, but neither delivered a single buffer after more than 17
   seconds; both processes had to be explicitly terminated, since neither
   exited on its own. The kernel log showed no UVC error and no USB
   bandwidth error during the attempt. This result is consistent with —
   but does not by itself prove — the control-handshake hypothesis in point
   2: opening and format-negotiating a UVC stream is not the same as
   receiving usable frame content, and a clean, silent zero-buffer result
   (no kernel error, no bandwidth complaint) points at a missing vendor
   initialization step rather than a bandwidth or driver fault.

Given this, the current milestone treats the vendor SDK as an external
oracle for ground truth, while the sysfs/UVC discovery layer (L1) supplies
stable, dependency-free identification that will remain correct once L2
replaces the SDK as the owned transport.

**Important:** direct V4L2/UVC streaming from `/dev/videoN` has been
attempted once, with the zero-buffer result recorded in point 4 above. That
result is consistent with, but does not confirm, the point-2 hypothesis
that a vendor control-channel handshake is required — no HID/vendor-control
traffic was interleaved during this attempt, and no bus capture was taken to
see what the device was doing instead of delivering buffers.

## Reference-oracle result: the vendor SDK cannot connect to this unit

This is the central negative result of the foundation milestone, and it is
measured, not inferred.

The reference probe links against `lib3DCamera.so` V3.2.23
(`Jul  3 2023 18:33:24`) from Revopoint/3DViewer commit
`516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`.

**Enumeration works.** `cs::getSystemPtr()->queryCameras()` finds the unit and
returns correct metadata:

| Field | Value |
| --- | --- |
| `name` | `3DCamera-uvc` |
| `serial` / `uniqueId` | `A22704522A6F00A64` |
| `firmwareVersion` | `v2.8.7.20220610` |
| `algorithmVersion` | `v7.5.7.0426` |
| `getCameraTypeName(getCameraTypeBySN(...))` | `MINI` |

**Connection fails.** `cs::ICamera::connect(CameraInfo)` returns
`ERROR_NOT_SUPPORT` (code 8, `getCameraErrorString` → `"not support"`) after
roughly 0.9–1.6 s, every time. Because connect fails, nothing downstream was
reachable: no stream profiles, no intrinsics, no extrinsics, no depth scale,
and zero depth and zero RGB frames.

**Root cause (confirmed).** With `cs::enableLoging(true)`, the SDK prints its
own internal trace. The relevant lines, verbatim:

```
initDevice m_devh=0x55fbeab0e900
Driver successfully detached from kernel.interface number:0
claim interface ok,interface number:0
enter openHid
Driver successfully detached from kernel.interface number:4
claim interface ok,interface number:4
UvcHid::openHid OK.
connect to A22704522A6F00A64 camera
...
readFile failed C:0:/persist/sdk-license
uvc get user data failed:/persist/sdk-license
readFile failed C:0:/data/sdk-license
uvc get user data failed:/data/sdk-license
[SDK WARN ] not find license file
[SDK WARN ] Parse license file failed, error code : 1
[SDK TRACE] authorize failed
```

USB access, kernel-driver detach, interface claim, and the HID channel all
succeed. The SDK then tries to read an `sdk-license` blob from the camera's
own on-device storage (over the UVC user-data channel) at
`/persist/sdk-license` and `/data/sdk-license`, fails both, and reports
`authorize failed`. `ERROR_NOT_SUPPORT` is how that licensing failure is
surfaced to the caller. `strings` on the library corroborates this: it
contains `cs::LicenseAuthorizer`, `cs::InnerAuthorizer`,
`cs::AuthorizerManager`, and `cs::WhiteListAuthorizer` symbols alongside the
exact log strings observed.

**What was ruled out.** Each of these was tested and did not change the
outcome:

| Hypothesis | Test | Result |
| --- | --- | --- |
| Missing `/dev/hidraw10` access | `strings` + full `strace` | **Falsified.** The SDK never opens any `/dev/hidraw*`; it reaches HID over usbfs, which we already have. |
| usbfs permission problem | `strace` of `/dev/bus/usb/001/024` | Opened successfully. The one `USBDEVFS_CLAIMINTERFACE` `EBUSY` is immediately followed by a kernel-driver detach and a successful re-claim — normal libusb behaviour. |
| Wrong SDK backend | `cs::setSdkEnableLibuvc(true)` / `(false)` | `true`: unchanged. `false`: `queryCameras` finds 0 cameras, so discovery breaks entirely. Not the cause. |
| Wrong entry point | No-arg `connect()`, C API `systemConnectCamera`, `setCheckedCameraInited(false)` | All fail identically, so the gate is shared and internal, not wrapper-specific. |
| Camera-type / firmware gating | Searched 3DViewer source and `document/` | No `CameraType`→supported table, no firmware-minimum check, no bundled license file. Inconclusive, but nothing there explains `ERROR_NOT_SUPPORT`. |

**Consequence.** The vendor SDK does **not** work as a reference oracle for
this unit, so the milestone's optional "if the SDK supports this unit" branch
did not happen: no calibration values and no frames were obtained from it.
The oracle's usefulness is therefore limited to what enumeration and the
trace log revealed — which is still substantial, because the log documents
the real connection sequence (usbfs → detach `uvcvideo` → claim interface 0
and 4 → open HID → authorize) that an owned transport will have to
reproduce.

The smallest falsifiable next step is recorded in
[`roadmap.md`](roadmap.md#open-questions).

## Vendor control channel, measured on the wire

The `strace` behind the previous section proved *that* the SDK issued USB
transfers and on which interfaces, but printed every `struct usbdevfs_urb`
argument as a bare pointer, so the bytes themselves were unrecoverable from it.
`reference/tracer/` closes that gap: an `LD_PRELOAD` ioctl interposer that
records the SDK's `USBDEVFS_*` calls to one exact usbfs node from inside the
traced process, where the pointer can be dereferenced. It observes only — it
issues no device operation of its own, and it patches nothing.

**Capture.** The ordinary hash-pinned reference probe
(`--serial A22704522A6F00A64`) was run against the hardware three times under
the tracer, each bounded by `timeout 15`, writing only under `/tmp`, with no
elevated privilege. The first two runs captured no protocol data, both from
tracer defects rather than the device or the SDK: run 1 lost its output to an
`O_EXCL` collision between the `timeout` and probe images, both of which
re-run the constructor that opens the trace file, fixed by giving each
process its own lazily-created output file; run 2 missed the SDK's `open()`
of the usbfs node because it goes through `__open_2`, the `_FORTIFY_SOURCE`
spelling, which the interposer did not yet hook, fixed by hooking the
fortified spellings and adding an adoption fallback for a descriptor that no
intercepted open explains — that fallback matched on `st_rdev` alone when
these runs were made, and now also requires the descriptor's `/proc/self/fd`
path to equal the target. All three runs, including the two that captured
nothing, went through the SDK's ordinary teardown path — `USBDEVFS_RESET` then kernel-driver reattach — like
any other invocation of the probe.

The third run is the one evidence capture described in this section: 8.7 s
wall clock against `/dev/bus/usb/001/041`. Probe exit code `3`,
`status: "connect_failed"`, `ERROR_NOT_SUPPORT` — the same result as every
untraced run, so the tracer did not change the outcome. 144 events,
58 179 bytes, zero dropped events, zero truncated payloads, zero unparseable
lines. Every one of the 35 URBs correlated to its completion. The raw capture
and its decode live under `/tmp` and are deliberately not committed.

### What the SDK actually touches

| Fact | Measured value |
|---|---|
| Device access path | usbfs only. Six sequential `open()`s of the same node; no `/dev/video*`, no `/dev/hidraw*` |
| Transfer submission | Asynchronous URBs exclusively: 35 `USBDEVFS_SUBMITURB`, 70 `USBDEVFS_REAPURBNDELAY` (35 returning a URB, 35 `EAGAIN` polls), 1 `USBDEVFS_DISCARDURB` |
| Synchronous transfer ioctls | **None.** `USBDEVFS_CONTROL` and `USBDEVFS_BULK` never appear |
| Endpoints touched | Exactly two: control `0x00` and interrupt-IN `0x85` |
| Control endpoint `0x00` | 34 transfers, 3 785 B requested, 994 B actually transferred; 33 completed with status `0`, one with `-108` (`ESHUTDOWN`) during teardown |
| Interrupt-IN `0x85` | 1 transfer, 1 024 B requested, **0 B transferred**, status `-2` (cancelled by the `DISCARDURB`) |
| Bulk / isochronous | None. No stream was ever started |
| Interface 0 | Kernel driver `uvcvideo` detached (`USBDEVFS_IOCTL`/`USBDEVFS_DISCONNECT`), claimed twice (one `EBUSY` first, then success), released twice, reattached |
| Interface 4 | Kernel driver `usbhid` detached, claimed once, released once, reattached |
| Teardown | One `USBDEVFS_RESET`, then the node is closed |

### The control requests, by shape

Every vendor request is a control transfer on endpoint 0. Grouped by setup
packet, with counts:

| `bmRequestType` | `bRequest` | `wValue` | `wIndex` | `wLength` | Count | What it is |
|---|---|---|---|---|---|---|
| `0x80` | `0x06` | `0x0300` | `0x0000` | 4 | 12 | Standard `GET_DESCRIPTOR(String)`, LANGID list |
| `0x80` | `0x06` | `0x0301` | `0x0409` | 255 | 3 | Standard string 1 → `Chishine3d` |
| `0x80` | `0x06` | `0x0302` | `0x0409` | 255 | 4 | Standard string 2 → `ZXProduct` |
| `0x80` | `0x06` | `0x0303` | `0x0409` | 255 | 4 | Standard string 3 → `A22704522A6F00A64` |
| `0x21` | `0x01` | `0x0100` | `0x0400` | 60 | 6 | UVC class `SET_CUR`, 60-byte payload |
| `0xa1` | `0x81` | `0x0200` | `0x0400` | 60 | 4 | UVC class `GET_CUR`, 60-byte payload |
| `0xa1` | `0x81` | `0x0600` | `0x0300` | 60 | 1 | UVC class `GET_CUR`, 60-byte payload |

Totals: 634 B read from the device, 360 B written to it, across the whole
session.

The first four rows are ordinary USB string descriptors — nothing vendor
specific, and they return exactly the strings `scanner3d list` already reports
from sysfs. The last three rows are the vendor channel.

### The vendor channel, byte by byte

These are the payloads as captured. ASCII is shown for the printable runs;
everything else was zero padding to the fixed 60-byte length.

| Order | Direction | `wValue`/`wIndex` | Payload (leading bytes) |
|---|---|---|---|
| 1 | host → device | `0x0100`/`0x0400` | `01` `"/tmp/inited"` `00 …` |
| 2 | device → host | `0x0200`/`0x0400` | `0c 00 00 00` `"v7.5.7.0426"` `00 …` |
| 3 | device → host | `0x0200`/`0x0400` | all zero |
| 4 | host → device | `0x0100`/`0x0400` | all zero |
| 5 | device → host | `0x0600`/`0x0300` | `"v2.8.7.20220610|v7.5.7.0426"` `00 …` |
| 6 | host → device | `0x0100`/`0x0400` | `01` `"/persist/sdk-license"` `00 …` |
| 7 | device → host | `0x0200`/`0x0400` | all zero |
| 8 | host → device | `0x0100`/`0x0400` | all zero |
| 9 | host → device | `0x0100`/`0x0400` | `01` `"/data/sdk-license"` `00 …` |
| 10 | device → host | `0x0200`/`0x0400` | all zero |
| 11 | host → device | `0x0100`/`0x0400` | all zero |

Row 5 is worth calling out on its own: `v2.8.7.20220610|v7.5.7.0426` is exactly
the `firmwareVersion` and `algorithmVersion` pair the SDK reports from
`queryCameras`, concatenated with a `|`. That makes this one request a
**self-verifying read**: its correct answer is already known independently.

### Correlation with the SDK's own log

Both streams come from the same process (pid 1985765) in the same run, so this
correlation is direct rather than inferred across runs. The SDK's log lines and
the wire events interleave as follows:

| SDK log line | Corresponding wire event |
|---|---|
| `initDevice m_devh=0x…` | `open()` of the usbfs node, `USBDEVFS_GET_CAPABILITIES` → `0x1fd` |
| `Driver successfully detached from kernel.interface number:0` | `USBDEVFS_GETDRIVER`(0) → `uvcvideo`, then `USBDEVFS_IOCTL`/`USBDEVFS_DISCONNECT`(0) |
| `claim interface ok,interface number:0` | `USBDEVFS_CLAIMINTERFACE`(0) → 0, after one `EBUSY` |
| `enter openHid` … `Driver successfully detached from kernel.interface number:4` | `USBDEVFS_GETDRIVER`(4) → `usbhid`, `USBDEVFS_IOCTL`/`USBDEVFS_DISCONNECT`(4) |
| `claim interface ok,interface number:4` / `UvcHid::openHid OK.` | `USBDEVFS_CLAIMINTERFACE`(4) → 0, then one interrupt-IN URB armed on `0x85` |
| `connect to A22704522A6F00A64 camera` | — no distinct transfer; the next traffic is the licence read |
| `enter libuvcReadFile` / `libuvcReadFile open file ok` | `SET_CUR` carrying `01` + a NUL-terminated path (rows 1, 6, 9 above) |
| `readFile failed C:0:/persist/sdk-license` / `uvc get user data failed:/persist/sdk-license` | row 6 `SET_CUR "/persist/sdk-license"` → row 7 all-zero `GET_CUR` → row 8 all-zero `SET_CUR` |
| `readFile failed C:0:/data/sdk-license` / `uvc get user data failed:/data/sdk-license` | row 9 `SET_CUR "/data/sdk-license"` → row 10 all-zero `GET_CUR` → row 11 all-zero `SET_CUR` |
| `[SDK WARN ] not find license file` / `Parse license file failed, error code : 1` / `[SDK TRACE] authorize failed` | **no USB traffic at all.** The next events are `RELEASEINTERFACE`/reattach |
| `connect()` returns `ERROR_NOT_SUPPORT` (code 8) | `USBDEVFS_RESET`, then `close()` |

That last row but one is the important one. **The authorization decision is
made entirely inside `lib3DCamera.so`.** Between the final all-zero licence
response and `authorize failed` there is not one further transfer. The device
never refuses anything: every USB operation the SDK performed succeeded. The
gate is a host-side library check on a blob the library could not read, and it
stays a blocker for L0 — nothing in this repository works around it, and this
capture is not a step toward doing so.

### Inferences, labelled as such

Everything above is measured. The following is interpretation, and is recorded
separately because it is not yet proven:

- ~~**`wIndex` decoding.**~~ **Now measured, and the "two extension units"
  reading was wrong.** The `(entity ID << 8) | interface number` decoding holds:
  `0x0400` is entity 4 on interface 0 and `0x0300` is entity 3 on interface 0.
  But entity 3 is *not* an extension unit — the device's own descriptors
  declare it an output terminal. See
  [VideoControl entities](#videocontrol-entities-read-from-the-devices-own-descriptors).
- **The `0x01` leading byte** in rows 1, 6 and 9 is probably an opcode meaning
  "open this path", because each one is immediately preceded in the SDK log by
  `enter libuvcReadFile` and followed by `libuvcReadFile open file ok`. Only one
  opcode value was ever observed, so this is a single data point.
- **The `0c 00 00 00` prefix** in row 2 is probably a little-endian length: `12`
  is the length of `"v7.5.7.0426"` plus its terminator. One observation.
- **All-zero `GET_CUR` responses probably signal failure**, since both licence
  paths produced them and both were logged as `readFile failed`. But an empty
  file and a missing file would look identical this way, so "the blob is absent"
  is not distinguishable from "the blob is empty" on this evidence.
- **What row 2 has to do with row 1** is not established. Reading `/tmp/inited`
  returning the algorithm version string is not obviously sensible; the response
  may be a stale result buffer from an earlier operation rather than the content
  of that path. Not investigated.
  **Partly superseded by scanner-006:** static disassembly of
  `cs::libuvcReadFile` shows the reply format is a `uint32_le` length followed by
  up to 56 payload bytes, so `0c 00 00 00` + `"v7.5.7.0426"` is a well-formed
  12-byte chunk rather than a stale buffer, and the all-zero reply is the
  zero-length terminator. *Why* the device answers this path with the algorithm
  version is still open — see
  [Domain 4](#domain-4--calibration--depth-scale-fact-disassembled).

### What this capture cannot show

- Only the traced process's own calls are visible. Traffic from the kernel's
  `uvcvideo`/`usbhid` drivers, and from any other process, is out of reach of an
  ioctl interposer by construction.
- The session ends at authorization, so no streaming, projector, calibration-read
  or mode-select traffic exists to observe. What is known about the control
  channel beyond this capture comes from a *separate* evidence set that this
  capture had no part in: owned code's own eleven control transfers, ten
  carrying 60-byte payloads plus the 4-byte `0x918` depth-scale property write.
- One `USBDEVFS_RESET` at teardown is the SDK's own normal behaviour, and the
  kernel logged the consequence: `usb 1-1.1: reset high-speed USB device number
  41`, one transient `device not accepting address 41, error -71`, then a second
  reset and successful re-enumeration. The device was healthy afterwards — same
  serial, same devnum, all nodes present.

## VideoControl entities, read from the device's own descriptors

### Fixture provenance

`tests/fixtures/mini-config-descriptor.bin` is a byte-exact copy of
`/sys/bus/usb/devices/1-1.1/descriptors` for this unit: 645 bytes, SHA-256
`2fa267c0c41d4cc53390bb7d1562d4b54fb44e8be04e0d568de8633b5d626613`. That digest
was re-verified against the live sysfs attribute on 2026-08-21, before and after
the control request below, and is unchanged. (`sysfs` advertises a 65553-byte
`st_size` for that attribute; the actual readable content is 645 bytes, so tools
that trust `stat` will report the wrong length.) The blob is complete — the
parser flags no truncation — so nothing in the table below is missing evidence.

`scanner3d.usb_descriptors` now parses class-specific VideoControl descriptors
in addition to the streaming formats it already read. Each VideoControl
interface of this device (0 and 2, the depth and RGB functions) declares the
same five entities:

| Entity ID | Descriptor subtype | Type | `wTerminalType` | Source pins | GUID | `bNumControls` | `bmControls` |
|---|---|---|---|---|---|---|---|
| — | `0x01` | header | — | — | — | — | — |
| 1 | `0x02` | input terminal | `0x0201` (camera) | — | — | — | — |
| 2 | `0x05` | processing unit | — | 1 | — | — | `5f1f` |
| 3 | `0x03` | **output terminal** | `0x0101` (streaming) | 2 | — | — | — |
| 4 | `0x06` | **extension unit** | — | 2 | `C2211A6D-787E-3C4A-AC06-7E434B676F9A` | 9 | `ffa201` |

### What that answers, and what it breaks

The open question was whether the entity IDs the SDK addressed (3 and 4 on
interface 0) match the device's own extension-unit descriptors. They do not,
and only half the earlier reading survives:

- **Entity 4 is a real extension unit**, the only one this device declares per
  function, with nine controls and the GUID above. The SDK's licence/file-read
  traffic (`wIndex=0x0400`, `SET_CUR 0x0100` / `GET_CUR 0x0200`) is addressed to
  a genuine, descriptor-declared extension unit.
- **Entity 3 is an output terminal**, not an extension unit — the streaming
  output terminal fed by the processing unit. Yet the SDK's one firmware/algorithm
  version read (`wIndex=0x0300`, `GET_CUR` selector `0x06`) was addressed to it
  and, per the capture, succeeded and returned a sensible 60-byte payload.

So the device answers a vendor `GET_CUR` on an entity whose descriptor does not
advertise that control. That is the device's behaviour, not a decoding error:
both readings of `wIndex` (`0x0300` → entity 3/interface 0) and of the
descriptor (subtype `0x03`, `wTerminalType 0x0101`) are unambiguous, and the two
disagree about what entity 3 is. Recorded as a fact, not resolved.

The GUID is the UVC mixed-endian (`bytes_le`) rendering of the descriptor bytes
`6d1a21c2 7e78 4a3c ac06 7e434b676f9a`, which is how UVC stores format and
extension GUIDs.

## First owned control request (measured)

One request, once, from this repository's own code with the vendor SDK absent
from the process.

### Exact command

```
.venv/bin/python -m scanner3d.cli firmware-version --execute \
    --serial A22704522A6F00A64 --json --json-out /tmp/scanner-003/live-result.json
```

| | |
|---|---|
| Setup packet | `bmRequestType=0xa1` `bRequest=0x81` `wValue=0x0600` `wIndex=0x0300` `wLength=60` |
| Timeout | 3000 ms (the module's hard ceiling) |
| Target | serial `A22704522A6F00A64`, `1-1.1`, `/dev/bus/usb/001/041` |
| Node snapshot | rdev `189:40`, inode 2701, mode `0o660`, bus 1 device 41 |
| Interface 0 driver at the time | `uvcvideo` |
| Duration of the `ioctl` | 17.6 µs |
| Wall clock, whole process | 0.078 s |
| Exit code | 5 (`ioctl_error`) |
| Result | `EBUSY` (errno 16) — **no bytes were transferred** |
| Response | none; no `wLength` bytes to hash |

### Why it failed, and why that is informative

The kernel refused the transfer before it reached the wire. usbfs routes a
class request whose recipient is an interface through `checkintf()`
(`drivers/usb/core/devio.c`): if the calling file descriptor has not claimed
that interface, the kernel *implicitly claims it on the caller's behalf* and
logs a warning. Here that implicit claim could not succeed, because `uvcvideo`
already owns interface 0. The kernel logged exactly one line:

```
usb 1-1.1: usbfs: process 2324415 (python) did not claim interface 0 before use
```

That warning is the record of the kernel's own implicit-claim attempt, not of
anything this code asked for: `scanner3d.control` never issues
`USBDEVFS_CLAIMINTERFACE`, never detaches a driver, and never resets anything —
the module knows only one usbfs command number, and a test asserts that.

This is a clean, falsifiable negative, and it is worth being precise about how
far it reaches. What it establishes is the *first* obstacle for this particular
transport: a direct usbfs class request addressed to interface 0 does not get
past that interface while `uvcvideo` holds it. Owned code can resolve, open and
address the vendor control channel correctly, and this is where it currently
stops.

What the result does **not** establish:

- **Not that interface ownership is the only obstacle.** Nothing was learned
  about what happens after ownership, because no bytes reached the wire. The
  device may or may not answer owned code; whether some further gate (a licence,
  an init sequence, an ordering requirement) stands behind ownership is untested.
- **Not that detaching `uvcvideo` is the only viable transport.** Taking
  ownership over usbfs is the route the SDK was observed to use, which makes it
  the best-evidenced next experiment — not the only candidate. A driver-mediated
  route (for example UVC extension-unit controls surfaced by `uvcvideo` itself)
  has not been tried and is not ruled out by this result.
- **Not what detaching would do to the video nodes.** That effect is unmeasured;
  see the [roadmap](roadmap.md) for the conservative statement of that risk.

No vendor byte has yet been read by anything in this repository.

Per the task boundary, the request was issued once and not retried, and no
alternate tuple was tried.

**Superseded by the next section.** scanner-004 repeated this identical request
with interface 0 temporarily detached from `uvcvideo`, and the device answered.
The negative recorded above stands as written — it is what a direct request does
while the driver holds the interface — but "no vendor byte has yet been read"
is no longer true; see
[Owned code reads the vendor channel](#owned-code-reads-the-vendor-channel-measured).

### Effect on the device: none observed

| Check | Before | After |
|---|---|---|
| `devnum` | 41 | 41 |
| `serial` | `A22704522A6F00A64` | `A22704522A6F00A64` |
| Descriptor SHA-256 | `2fa267c0…6613` | `2fa267c0…6613` |
| Interfaces 0–3 driver | `uvcvideo` | `uvcvideo` |
| Interface 4 driver | `usbhid` | `usbhid` |
| Holders of `/dev/video0-3`, usbfs node, `/dev/hidraw10` | none | none |

The kernel log gained exactly one line across the run — the usbfs warning quoted
above. No detach, no reset, no re-enumeration, no `Found UVC 1.00 device` reprobe,
and no surviving process holding any scanner node. The device was not power-cycled
or replugged at any point in this task.

## Owned code reads the vendor channel (measured)

The same request, once more, after clearing the one obstacle the previous
section measured: interface 0 temporarily taken from `uvcvideo` and given back.
Authorized by the owner for this unit on 2026-08-21.

### Exact command

```
timeout --signal=TERM --kill-after=5s 20s \
  .venv/bin/python -m scanner3d.cli firmware-version \
    --detach-uvcvideo-from-interface-0 --execute \
    --serial A22704522A6F00A64 --json-out /tmp/scanner-004/live-result.json
```

Run once. No phase was retried, no alternate request was tried, and no second
live invocation was made.

### Per-phase result

| Phase | usbfs operation | Outcome | Duration |
|---|---|---|---|
| confirm driver (sysfs) | `driver` symlink of `1-1.1:1.0` | `uvcvideo` | — |
| confirm driver (kernel) | `USBDEVFS_GETDRIVER`(0) | `uvcvideo` | — |
| disconnect | `USBDEVFS_IOCTL`/`USBDEVFS_DISCONNECT`(0) | ok | 0.202 ms |
| claim | `USBDEVFS_CLAIMINTERFACE`(0) | ok | 0.010 ms |
| control read | `USBDEVFS_CONTROL`, the unchanged whitelisted packet | ok | 1.266 ms |
| release | `USBDEVFS_RELEASEINTERFACE`(0) | ok | 0.007 ms |
| reconnect | `USBDEVFS_IOCTL`/`USBDEVFS_CONNECT`(0) | ok | 122.915 ms |

| | |
|---|---|
| Setup packet | `bmRequestType=0xa1` `bRequest=0x81` `wValue=0x0600` `wIndex=0x0300` `wLength=60` |
| Timeout | 3000 ms (the module's hard ceiling), external deadline 20 s TERM / +5 s KILL |
| Target | serial `A22704522A6F00A64`, `1-1.1`, `/dev/bus/usb/001/041` |
| Node snapshot | rdev `189:40`, inode 2701, mode `0o660`, bus 1 device 41 |
| Total duration, detach through reconnect | 127.312 ms |
| Wall clock, whole process | 0.196 s |
| Exit code | 0 (`ok`, and cleanup succeeded) |

### The response

| | |
|---|---|
| Bytes returned | 60 of 60 |
| SHA-256 | `71b87fbcb738058473ffd828495dc8d788ce96ce892e94efc7cae7f0b0c8a334` |
| Text before the padding | `v2.8.7.20220610|v7.5.7.0426` |
| Padding | NUL to 60 bytes |
| Firmware version | `v2.8.7.20220610` — matches what the SDK reported |
| Algorithm version | `v7.5.7.0426` — matches what the SDK reported |

**These are the first vendor bytes any code in this repository has read from the
device.** The answer is the one the independent oracle predicts, so it is not a
plausible-looking artefact: the SDK's `queryCameras` reported the same two
strings for this unit before its authorization step failed.

### What this establishes, and what it does not

**Established.** Interface ownership was the obstacle the previous attempt hit,
and clearing it is sufficient for *this* request: with interface 0 detached from
`uvcvideo` and claimed by owned code, the device answered a UVC `GET_CUR` on
entity 3, selector `0x06`, in 1.3 ms with the correct 60-byte payload. No
licence, no HID init, no SDK, and no authorization step stood behind ownership
for this one read.

**Measured, not inferred:** every number above comes from the run's own JSON
artefact and from before/after sysfs and kernel-log snapshots.

What this does **not** establish:

- **Not that the whole vendor channel is open.** One read-only request, one
  entity, one selector. Nothing here says the device will answer any *other*
  request, and in particular nothing says a `SET_CUR` — which is what the SDK's
  authorization step actually needs — would be accepted. The SDK's own failure
  was at authorization, not at reading a version string.
- **Not that streaming, calibration, or depth data are reachable.** No frame,
  no intrinsics, no depth scale was requested or obtained.
- **Not that the HID channel is needed or not needed.** Interface 4 was never
  touched in this run, so this says nothing about whether `openHid` matters for
  requests other than this one.
- **Not a general claim about detach safety.** See the effect table below: the
  disruption measured here is for one unit, one 127 ms window, with nothing
  streaming.

### Effect on the device: reversible, and measured

| Check | Before | After |
|---|---|---|
| `devnum` | 41 | 41 |
| `serial` | `A22704522A6F00A64` | `A22704522A6F00A64` |
| Descriptor SHA-256 | `2fa267c0…6613` | `2fa267c0…6613` |
| Interfaces 0–3 driver | `uvcvideo` | `uvcvideo` |
| Interface 4 driver | `usbhid` | `usbhid` |
| `/dev/video0`, `/dev/video1` (depth) | present | present, **recreated** (new mtime) |
| `/dev/video2`, `/dev/video3` (RGB) | present | present, untouched (same mtime) |
| Holders of any scanner node | none | none |

The kernel log gained exactly one line across the run:

```
usb 1-1.1: Found UVC 1.00 device ZXProduct (2207:110c)
```

That is `uvcvideo` re-probing after the reconnect. Detaching interface 0 tore
down the *depth* function — `uvcvideo` binds the VideoControl interface and
claims its VideoStreaming sibling, so interface 1 went with it and `/dev/video0`
and `/dev/video1` were destroyed and recreated. The RGB pair on interfaces 2–3
was not disturbed. The node names and major/minor numbers came back identical
here; that is not guaranteed in general, since V4L2 assigns the lowest free
index and the freed indices happened to be reusable. There was no device reset,
no re-enumeration (`devnum` unchanged), and no replug at any point.

The reconnect cost 123 ms of the 127 ms window — almost all of it is `uvcvideo`
re-probing, not the detach or the transfer.

**Limitation of the disruption measurement.** Nothing was streaming from either
camera during the run, and no process held a video node. What a detach does to
an *active* capture was not measured and must not be inferred from this.

## Owned code writes to the vendor channel (measured)

The first write this repository has ever sent to the scanner, and the first
reproduction of a *multi-transfer* vendor exchange without the SDK: rows 1-4 of
[the byte-by-byte table](#the-vendor-channel-byte-by-byte) - the `/tmp/inited`
sequence on entity 4 - issued once, in order, inside the same interface-0
ownership lifecycle the previous section measured. Authorized by the owner for
this unit on 2026-08-21, narrowed to this one sequence.

### Exact command

```
timeout --signal=TERM --kill-after=10 45 \
  .venv/bin/python -m scanner3d.cli inited-sequence \
    --serial A22704522A6F00A64 --execute-write-sequence \
    --json-out /tmp/scanner005/live.json
```

Run once. No transfer was retried, no other path or selector was tried, and no
second live invocation was made.

### The four transfers, as sent

| # | Step | Setup packet | Payload sent |
|---|---|---|---|
| 1 | `path_write` | `21 01` `wValue=0x0100` `wIndex=0x0400` `wLength=60` | `01` `"/tmp/inited"` `00`, zero-padded to 60 |
| 2 | `result_read` | `a1 81` `wValue=0x0200` `wIndex=0x0400` `wLength=60` | — |
| 3 | `second_read` | the same packet again | — |
| 4 | `clear_write` | `21 01` `wValue=0x0100` `wIndex=0x0400` `wLength=60` | 60 zero bytes |

The packed `struct usbdevfs_ctrltransfer` prefixes were
`21 01 00 01 00 04 3c 00 b8 0b 00 00` and
`a1 81 00 02 00 04 3c 00 b8 0b 00 00` — the little-endian encodings of those
host values, with the 3000 ms ceiling. An offline test asserts those bytes.

### Per-phase result

| Phase | usbfs operation | Outcome | Duration |
|---|---|---|---|
| confirm driver (sysfs) | `driver` symlink of `1-1.1:1.0` | `uvcvideo` | — |
| confirm driver (kernel) | `USBDEVFS_GETDRIVER`(0) | `uvcvideo` | — |
| disconnect | `USBDEVFS_IOCTL`/`USBDEVFS_DISCONNECT`(0) | ok | 0.172 ms |
| claim | `USBDEVFS_CLAIMINTERFACE`(0) | ok | 0.007 ms |
| owned sequence | `USBDEVFS_CONTROL` x3 (transfers 1-3) | ok, all three matched | 5.289 ms |
| protocol cleanup | `USBDEVFS_CONTROL` (transfer 4) | ok, cleared | 2.250 ms |
| release | `USBDEVFS_RELEASEINTERFACE`(0) | ok | 0.016 ms |
| reconnect | `USBDEVFS_IOCTL`/`USBDEVFS_CONNECT`(0) | ok | 29.185 ms |

| | |
|---|---|
| Timeout | 3000 ms per transfer (the hard ceiling), external deadline 45 s TERM / +10 s KILL |
| Target | serial `A22704522A6F00A64`, `1-1.1`, `/dev/bus/usb/001/041` |
| Node snapshot | rdev `189:40`, inode 2701, mode `0o660`, bus 1 device 41 |
| Total duration, detach through reconnect | 39.743 ms |
| Exit code | 0 (`ok`, and both cleanups succeeded) |

### The responses

| Transfer | Bytes | SHA-256 | Content |
|---|---|---|---|
| 1 `path_write` | 60 of 60 accepted | `698deb02…4f77` (sent) | — |
| 2 `result_read` | 60 of 60 | `ad6c4887…3ae9` | `0c 00 00 00` then `"v7.5.7.0426"`, NUL, zero padding |
| 3 `second_read` | 60 of 60 | `5dcc1b58…ab31` | all zero |
| 4 `clear_write` | 60 of 60 accepted | `5dcc1b58…ab31` (sent) | — |

Leading 16 bytes of transfer 2: `0c00000076372e352e372e3034323600`. Leading 16
bytes of transfer 3: sixteen zero bytes. Transfers 3 and 4 share a digest
because 60 zero bytes are 60 zero bytes in either direction.

**All four transfers matched the SDK capture byte for byte.** The first read's
`0c 00 00 00` prefix decodes as little-endian 12, which is `len("v7.5.7.0426")`
plus its terminator, and the string is the algorithm version the SDK's
`queryCameras` independently reported for this unit. The second read returned 60
zeros, exactly as the trace recorded.

### What this establishes, and what it does not

**Established.**

- **The device accepts an owned vendor `SET_CUR`.** The write direction of the
  vendor channel is open to this repository without the SDK, without a licence
  blob, without HID initialization, and without any authorization step: the
  device took all 60 bytes of a `SET_CUR` to entity 4, selector `0x01`, in
  1.6 ms. Until this run, the only thing measured here was a read.
- **A multi-transfer vendor exchange is reproducible without the SDK.** Not one
  request in isolation but a four-step protocol interaction - select a path,
  read the result, read again, clear - reproduced in the same order with the
  same bytes and the same answers.
- **The device is stateful across transfers in the way the trace implied.** The
  first read after the path write returned a non-zero record and the second
  returned zeros. That difference came from the device, on this run, with
  nothing but the two identical reads between them.
- **The `0c 00 00 00` length-prefix reading now has a second, independent
  observation** — a different process, different code, no vendor library.

**Measured, not inferred:** every number above comes from the run's own JSON
artefact (`/tmp/scanner005/live.json`) and from before/after sysfs, node and
kernel-log snapshots taken by the same session.

What this does **not** establish:

- **Not that `/tmp/inited` was read.** The reply to a `/tmp/inited` path select
  is the algorithm version string, which is not plausibly the contents of that
  path. The [earlier inference](#inferences-labelled-as-such) that the response
  may be a stale result buffer from an earlier operation is neither confirmed
  nor refuted by reproducing it; it is now reproduced, not explained.
- **Not what the leading `01` byte means.** The same single opcode value was
  sent and the same answer came back. That is consistent with "open this path"
  and with several other readings. Still one opcode, never varied.
  **Superseded by scanner-006:** `cs::libuvcReadFile` `0x262861` writes
  `buf[0] = 0x01` then `strcpy(buf + 1, path)`, so `01` is the read-file opcode
  and this whole four-transfer sequence is the vendor's file-read protocol. That
  is static evidence about the host library, not a new observation of the
  device.
- **Not that other paths are readable.** `/persist/sdk-license` and
  `/data/sdk-license` were not sent, are not reachable from any code in this
  repository, and the licence gate remains exactly where
  [the reference-oracle section](#reference-oracle-result-the-vendor-sdk-cannot-connect-to-this-unit)
  left it: inside `lib3DCamera.so`, not on the wire.
- **Not that the write direction is generally safe.** One selector, one entity,
  two payloads, both of which the vendor's own software sent to this unit. No
  claim is made about any other `SET_CUR`.
- **Not that the second read's zeros mean failure.** The all-zero-means-failure
  reading came from the licence paths; it is not tested here.
- **Not that streaming, calibration, projector control or depth data are
  reachable.** None was requested or obtained.

### Effect on the device: reversible, and measured

| Check | Before | After |
|---|---|---|
| `devnum` | 41 | 41 |
| `serial` | `A22704522A6F00A64` | `A22704522A6F00A64` |
| Descriptor SHA-256 | `2fa267c0…6613` | `2fa267c0…6613` |
| Interfaces 0–3 driver | `uvcvideo` | `uvcvideo` |
| Interface 4 driver | `usbhid` | `usbhid` |
| usbfs node | `189:40`, inode 2701, mode `0o660` | identical |
| `/dev/video0`, `/dev/video1` (depth) | present | present, **recreated** (mtime 18:33 → 20:45) |
| `/dev/video2`, `/dev/video3` (RGB) | present | present, untouched (mtime 16:33) |
| Holders of the usbfs node | none | none |
| Surviving `scanner3d` processes | none | none |

The kernel log gained exactly one line across the run:

```
usb 1-1.1: Found UVC 1.00 device ZXProduct (2207:110c)
```

That is `uvcvideo` re-probing after the reconnect — the same single line the
read-only run produced. No reset, no re-enumeration, no error, no replug. The
reconnect cost 29 ms of the 40 ms window.

**The device-side state this run created was undone on the wire, not just left
alone.** The clearing `SET_CUR` is treated as protocol cleanup with the same
guarantees as the driver handback: owed from the instant before the path write
is attempted, attempted exactly once, shielded from SIGINT and SIGTERM, run
before the driver is given back, reported separately, and unable to be presented
as success when it fails (exit 14, against exit 9 for a failed handback). It
succeeded here.

**Limitation.** Whether the device would still hold the `/tmp/inited` selection
had the clearing write not been sent was not tested — that would have required
deliberately skipping cleanup, which is out of scope. "Cleared" here means the
clearing write completed, matching what the SDK does, not that a divergent
device state was observed and then observed to go away.

## Owned code reads the factory depth intrinsics (measured)

Ranked experiment 2 below was implemented and **run exactly once**, on
2026-08-21, against this unit. It is the first real calibration data this
project has obtained, and the first time the recovered `cs::libuvcReadFile`
protocol was driven with a path the vendor SDK was never observed sending here.

### Exact command, and the run count

```bash
timeout --signal=TERM --kill-after=10 30 \
  .venv/bin/python -m scanner3d.cli depth-intrinsics \
    --execute-calibration-read --serial A22704522A6F00A64 \
    --json-out /tmp/3d-scanner-intrinsics/live.json
```

Run **once**, exit `0`, well inside the external TERM-then-KILL deadline (the
whole process, interpreter start included, took under a second). An earlier
invocation of the same line died in `timeout` itself with exit `127` because the
interpreter path was relative to a directory the shell had changed out of; no
interpreter started, no node was opened and no byte reached the device, so the
one-run budget was not spent on it.

### The four transfers, as sent

`wIndex=0x0400` is `(entity 4 << 8) | interface 0` throughout; every timeout is
1000 ms; every transfer moved its full 60 bytes and none was retried.

| # | step | setup packet (12 B of the ioctl arg) | moved | length word | note |
|---|---|---|---|---|---|
| 1 | `path_write` | `2101 0001 0004 3c00 e8030000` | 60 B | — | `01` + `"/data/camparam/Pl.bin"` + NUL, zero-padded; sent SHA-256 `d033de5a…4378` |
| 2 | `chunk_read` | `a181 0002 0004 3c00 e8030000` | 60 B | **40** | reply SHA-256 `a7d06803…d077`; 40 bytes copied |
| 3 | `chunk_read` | `a181 0002 0004 3c00 e8030000` | 60 B | **0** | 60 zero bytes; the terminator, nothing copied |
| 4 | `clear_write` | `2101 0001 0004 3c00 e8030000` | 60 B | — | 60 zero bytes; protocol cleanup, sent SHA-256 `5dcc1b58…ab31` |

Timings: 1.144 ms, 2.891 ms, 1.209 ms, 1.385 ms. Two result reads of the fixed
maximum of eight; four control transfers of the fixed maximum of ten. The
capacity (256 B), the read cap (8) and the 56-byte chunk ceiling were all
untouched by this run — none of them was approached, and none can be widened at
runtime in any case.

Reply 2 in full:

```
28000000 80029001 b6ab6044 00000000 f9bf2942 00000000
b6ab6044 2bd54043 00000000 00000000 0000803f 00000000 …
```

`28 00 00 00` is `uint32_le` 40; the 40 bytes behind it are the record; the rest
of the 60-byte reply is zero padding outside the declared length and was not
copied.

### Per-phase result

| Phase | Result | Duration |
|---|---|---|
| `confirm_driver` (sysfs) | `uvcvideo` on interface 0 | — |
| `confirm_driver` (`USBDEVFS_GETDRIVER`) | `uvcvideo` | — |
| `disconnect` | detached `uvcvideo` from interface 0 | 0.137 ms |
| `claim` | `USBDEVFS_CLAIMINTERFACE(0)` succeeded | 0.006 ms |
| `owned_sequence` | complete 40-byte record | 5.427 ms |
| `protocol_cleanup` | the key selection cleared with 60 zero bytes | 1.385 ms |
| `release` | `USBDEVFS_RELEASEINTERFACE(0)` succeeded | 0.007 ms |
| `reconnect` | kernel asked to rebind | 25.519 ms |

Total owned window 37.488 ms, of which the read is 5.427 ms and the reconnect
25.519 ms. Both undos succeeded, so the run is reported as success; either
failing would have overridden it (exit 14 for the clear, exit 9 for the
handback).

### The blob, and the record it decodes to

The assembled bytes are the primary evidence and are preserved whatever the
parse says:

```
length  40 bytes
sha256  fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf
hex     80029001 b6ab6044 00000000 f9bf2942 00000000
        b6ab6044 2bd54043 00000000 00000000 0000803f
```

Decoded as the vendor `Types.hpp` `Intrinsics` layout (`short width, height`
then nine x86 little-endian IEEE-754 floats):

| field | value | |
|---|---|---|
| `width` × `height` | 640 × 400 | a mode this unit's own descriptors advertise |
| `fx` | 898.6829833984375 | pixels of that 640×400 frame |
| `fy` | 898.6829833984375 | pixels |
| `cx` | 42.43747329711914 | pixels |
| `cy` | 192.8326873779297 | pixels |
| fixed entries | `zero01 zero10 zero20 zero21 = 0`, `one22 = 1` | all exact |

Implied 3×3 pinhole matrix, built from the read floats alone:

```
[ 898.6829833984375        0.0     42.43747329711914 ]
[        0.0        898.6829833984375  192.8326873779297 ]
[        0.0               0.0             1.0          ]
```

Checks are recorded on two separate axes, and neither one alters a value:

- **Structural** — all six checks passed: one finiteness check plus the five
  fixed-entry checks. Every float is finite and the five entries the layout
  fixes really are `0, 0, 0, 0, 1`. That is what makes the row-major pinhole
  reading of these 40 bytes more than a guess: four independent zeros and a one
  landing exactly where the layout says they should is not what a misparse
  produces.
- **Plausibility** — four of the five observations passed: positive dimensions,
  positive and equal focal lengths, principal point inside the frame, and
  dimensions matching an advertised depth mode. The fifth is flagged: **`cx`
  sits 277.56 px left of the 320 px horizontal centre**, while `cy` is within
  7.17 px of the vertical centre.

`INFERENCE` (not measurement): a principal point that far off-centre
horizontally, with `fx == fy` and a near-centred `cy`, is consistent with the
factory record describing a **cropped or horizontally offset sensor window**
rather than the centred 640×400 frame it names. Nothing in this run establishes
that. The alternative readings — a different intended field order, a
device-specific convention, or a genuinely eccentric optical centre — are not
excluded by anything measured here, and the values are reported exactly as they
arrived rather than adjusted toward any of them.

### What this establishes, and what it does not

**Establishes.**

- The device **serves `/data/camparam/Pl.bin` to an unlicensed host.** The
  licence gate that stops the vendor SDK at connect does not gate *this* key.
  That answers the Domain 4 open question for `Pl.bin` only: the sibling keys
  (`Pr.bin`, `Prgb.bin`, `LC_RT.bin`, `Distort.bin` and the rest) were never
  requested and remain untested, and nothing here says they are served.
- The recovered `libuvcReadFile` protocol is correct in full, including the
  chunked `uint32_le`-prefixed reply, the zero-length terminator and the
  cleanup write, driven with a path this repo had never sent.
- The `Types.hpp` `Intrinsics` layout is correct for this firmware: 40 bytes,
  `short`/`short`, nine floats, with the fixed entries exactly where the layout
  puts them.
- The depth function has factory intrinsics on the device for a 640×400 frame,
  and they are readable in 5.4 ms without the SDK.

**Does not establish.**

- Nothing about **metric accuracy**. These are the numbers the device stores;
  whether they describe the optics well was not tested and cannot be from one
  read.
- Nothing about **coordinate handedness or origin convention** (top-left vs
  bottom-left, pixel centre vs corner). The struct carries no such statement.
- Nothing about **RGB alignment or extrinsics** — `Prgb`, `LC_RT` and `Distort`
  were not read and are out of scope here.
- Nothing about **depth scale**: the `wValue=0x0e00` read-by-id channel is
  experiment 3 and was not touched.
- Nothing about **streaming**. Intrinsics do not order a bring-up, and no V4L2
  or iso/bulk path was exercised.
- Whether the 640×400 the record names is the frame the depth stream actually
  delivers. It matches an advertised mode; that is a descriptor comparison, not
  a frame.

### Effect on the device: reversible, and measured

| Check | Before | After |
|---|---|---|
| `devnum` | 41 | 41 |
| `serial` | `A22704522A6F00A64` | `A22704522A6F00A64` |
| Descriptor SHA-256 | `2fa267c0…6613` | `2fa267c0…6613` |
| Interfaces 0–3 driver | `uvcvideo` | `uvcvideo` |
| Interface 4 driver | `usbhid` | `usbhid` |
| usbfs node | `189:40`, inode 2701, mode `0o660` | identical |
| `/dev/video0`, `/dev/video1` (depth) | present | present, **recreated** (mtime 20:45 → 22:44) |
| `/dev/video2`, `/dev/video3` (RGB) | present | present, untouched (mtime 16:33) |
| Holders of the usbfs node | none | none |
| Surviving `scanner3d` processes | none | none |

The kernel log gained exactly one line across the run:

```
usb 1-1.1: Found UVC 1.00 device ZXProduct (2207:110c)
```

That is `uvcvideo` re-probing after the reconnect — the same single line both
earlier owned runs produced. No reset, no re-enumeration, no error, no replug,
and no physical intervention was needed.

**Limitation**, unchanged from the write run: "cleared" means the clearing write
completed, matching what the vendor does. Whether the device would still hold
the `/data/camparam/Pl.bin` selection had that write not been sent was not
tested, because deliberately skipping cleanup is out of scope.

## Owned code captures a raw depth frame (measured)

Task `scanner-009` was implemented and **run exactly once**, on 2026-08-22 at
01:00:38 local time (finished 01:00:39), against this unit. It is the first
raw depth frame this project has obtained by any path. The vendor SDK never
reaches streaming — it fails at authorization, see
[Reference-oracle result](#reference-oracle-result-the-vendor-sdk-cannot-connect-to-this-unit)
— and owned code's earlier work on this device was confined to the vendor
control channel (the version read, `/tmp/inited`, and the two calibration
reads) and never touched a video node. The cumulative live-attempt count for
`scanner-009` is **one**, and this record must not be read as evidence that
the capture is repeatable on demand — it has been run once.

### Exact command

```bash
.venv/bin/scanner3d depth-frame --serial A22704522A6F00A64 \
  --execute-depth-frame-capture --output-dir /tmp/scanner-009-evidence/frame \
  --json-out /tmp/scanner-009-evidence/live.json
```

Unlike every earlier owned-control-channel run, this one took **no interface
away from `uvcvideo`**: it drove `/dev/video0` through ordinary V4L2 while the
kernel driver kept the depth function's interfaces bound throughout. No
XU/vendor-control transfer and no HID transfer is reported for this run —
only V4L2 `ioctl`s against the existing node.

### Device and node identification

The node was resolved, not assumed:

| fact | value |
|---|---|
| serial | `A22704522A6F00A64` |
| sysfs path | `/sys/bus/usb/devices/1-1.1` |
| bus / devnum at the time of the run | 1 / 41 |
| depth function interface | 0, driver `uvcvideo` |
| node resolution | USB interface string `DepthCamA22704522A6F00A64` plus V4L2 capture index 0 → `/dev/video0` (`81:0`) |
| driver | `uvcvideo` |
| card | `ZXProduct: DepthCamA22704522A6F` |
| kernel | `6.14.11` |
| `bus_info` | `usb-0000:00:14.0-1.1` |
| effective device caps | `0x04200001` (`V4L2_CAP_VIDEO_CAPTURE`\|`V4L2_CAP_STREAMING`) |

The node number was never assumed from a fixed `/dev/videoN` guess; it was
derived from the interface string plus a capture-index match.

### Format negotiation and buffer setup

The node's initial format was `YUYV` 640×400. The run negotiated:

| field | value |
|---|---|
| pixel format | `Y16 ` |
| resolution | 640×400 |
| field | `V4L2_FIELD_NONE` |
| bytesperline | 1280 (tight — no row padding) |
| sizeimage | 512000 |

4 MMAP buffers were requested and 4 were returned by `VIDIOC_REQBUFS`. Buffer
0 was dequeued after a single `poll()` — 1 call, 0 restarts — that waited
1.043 s of an 8.0 s deadline and returned `POLLIN`.

### The captured frame

| field | value |
|---|---|
| `bytesused` | 512000 |
| `sequence` | 0 |
| buffer timestamp | 272564.631765 s (kernel monotonic clock) |
| stream duration | 1.047 s |
| total run duration | 1.060 s |

### The buffer flags, and a corrected reading

Raw dequeued flags were `0x00012001`; the flags `QUERYBUF` reported for the
same buffer were `0x00012000`. **The round-0 record of this task named these
bits wrongly; the raw numbers themselves were always valid evidence.** Decoded
against this host's own `<linux/videodev2.h>`:

- `0x00010000` is `V4L2_BUF_FLAG_TSTAMP_SRC_SOE` — one of the two defined
  values inside `V4L2_BUF_FLAG_TSTAMP_SRC_MASK` (`0x00070000`), the other
  being `V4L2_BUF_FLAG_TSTAMP_SRC_EOF` (`0x00000000`). It is **not**
  `V4L2_BUF_FLAG_LAST`, which is `0x00100000` — a different bit — and which
  was **not** set on this frame.
- So `0x00012001` decodes as `V4L2_BUF_FLAG_MAPPED` (`0x1`) **|**
  `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC` (`0x2000`, the timestamp *type*, within
  `V4L2_BUF_FLAG_TIMESTAMP_MASK` = `0x0000e000`) **|**
  `V4L2_BUF_FLAG_TSTAMP_SRC_SOE` (`0x10000`, the timestamp *source*: start of
  exposure).
- `0x00012000` is the same timestamp type and source with `MAPPED` clear.

### The artifact

`/tmp/scanner-009-evidence/frame/depth-frame-raw.bin`, 512000 bytes, SHA-256
`0a1b667eeca5ed2ce1c64e6aab124580efc9474e5e3e13187b53944d0c296933`. Because the
rows were already tight (bytesperline 1280 == 640 × 2 bytes/sample), no
separate packed file was written; the packed hash is the same hash as the raw
one.

### Effect on the device: cleanup succeeded, and one causally unresolved event

Cleanup ran to completion inside the process itself: `VIDIOC_STREAMOFF`, four
`munmap`s, `VIDIOC_REQBUFS(count=0)`, then `close`, all reported successful.
None of it required taking the interface from `uvcvideo`, since it was never
taken.

**2.788 seconds after the frame was captured, and after the capturing process
had already exited, the device re-enumerated**: `devnum` moved from 41 to 42,
on the same serial and the same sysfs port (`/sys/bus/usb/devices/1-1.1`).

This is recorded as one observed, causally unresolved event, and no more than
that:

- It is **not** attributed here to `STREAMOFF`, to any missing vendor
  teardown, or to anything this run's code did. The evidence does not
  establish a cause.
- A single occurrence cannot distinguish a code-caused reset from an
  independent device- or host-side event — the process that ran the capture
  had already exited by the time it happened.
- Resolving it would need further, separately authorized observations — for
  example, repeat runs with and without this code's V4L2 sequence to see
  whether the re-enumeration tracks it, and a kernel-log/`usbmon` capture
  spanning the interval. None of that is scheduled or promised by this
  record.

### What this establishes, and what it does not

**Establishes.**

- Owned code, using only ordinary V4L2 `ioctl`s against the existing
  `/dev/video0` — no interface detach, no XU/vendor-control transfer, no HID
  transfer — can negotiate `Y16` 640×400 and dequeue one complete, correctly
  sized frame from this device's depth function.
- This is a positive answer, for this one run, to the discriminator posed for
  experiment **4a** in
  [Ranked next experiments](#ranked-next-experiments-each-mutation-risk-discriminating-evidence):
  "whether the depth function yields any frame without a prior `echo s …`
  register write." It does not generalize beyond this one run, and it does
  not by itself explain the earlier zero-buffer result in
  [Why UVC discovery alone did not produce frames](#why-uvc-discovery-alone-did-not-produce-frames):
  that attempt used GStreamer against 1280×800 `GRAY16_LE`, a different
  resolution and pixel format from the 640×400 `Y16` negotiated here, so the
  two results are not directly comparable and neither is treated as
  explaining the other.
- The 512000-byte frame, its exact buffer flags, and their corrected decoding
  are established, as recorded above.

**Does not establish.**

- **Repeatability.** This is one run. The cumulative live-attempt count for
  `scanner-009` remains one; this record must not be read as evidence the
  capture is repeatable on demand.
- **RGB capture.** Not attempted here.
- **Physical meaning of the samples.** The 512000 bytes are 640×400
  little-endian 16-bit values as the kernel presented them. What those values
  *mean* is not established by this run: no unit, no scale factor, no
  zero/invalid-sample convention, and no coordinate frame has been confirmed
  against the device. Nothing here licenses converting the samples to
  millimetres or to points, and the separately measured depth scale from
  scanner-008 (`float32(1.0f/200)`) has **not** been validated against this
  frame.
- **Corroboration against an L0 probe.** The vendor SDK cannot connect to this
  unit at all, so the M3 exit criterion's "corroborated against an L0 probe
  run on the same physical unit" clause remains unmet by construction, not by
  omission.
- **Why the earlier zero-buffer GStreamer attempt failed, or whether it would
  now succeed under the same resolution/format.** Different resolution,
  different pixel format, different tool; not tested here.
- **A cause for the 2.788 s post-exit re-enumeration.** See above.

## Owned code attempts a raw RGB frame (measured)

Task `scanner-010` was implemented and **run exactly once**, on 2026-08-22 at
02:21:13 local time (`07:21:13.352Z`), against this unit. It did **not**
produce an RGB frame: the run stopped at format negotiation, before any buffer
was requested or any stream was started, and it was not retried. The cumulative
live-attempt count for `scanner-010` is **one**.

The command reuses the same bounded V4L2/MMAP engine, the same nine `ioctl`s
and the same shielded cleanup as
[the depth capture](#owned-code-captures-a-raw-depth-frame-measured); only the
fixed profile differs. `uvcvideo` stayed bound throughout, no interface was
taken from the kernel, and no vendor/XU or HID transfer was issued.

### Exact command

```bash
.venv/bin/scanner3d rgb-frame --serial A22704522A6F00A64 \
  --execute-rgb-frame-capture --output-dir /tmp/scanner-010-evidence/artifact \
  --json --json-out /tmp/scanner-010-evidence/rgb-frame.json
```

Run under an external `timeout --signal=TERM --kill-after=5s 20s` deadline,
longer than the command's own 8.0 s internal poll budget. The external deadline
never fired: the process exited on its own after **0.120 s** wall
(0.0135 s measured inside the command), with exit code **17**
(`format_mismatch`).

### Node identification

The node was resolved from the USB interface string descriptor plus the V4L2
index, never from a `/dev/videoN` number and never from a caller-supplied path.

| Fact | Value |
| --- | --- |
| Serial | `A22704522A6F00A64` |
| sysfs path | `/sys/bus/usb/devices/1-1.1` (bus 1, devnum 42) |
| Interface | 2, string descriptor `RGBCamA22704522A6F00A64` |
| V4L2 index / kind | 0 / capture |
| Resolved node | `/dev/video2`, rdev `81:2`, inode 3119, mode `0o660` |
| Driver | `uvcvideo`, bound before, during and after |

`VIDIOC_QUERYCAP` answered driver `uvcvideo`, card
`ZXProduct: RGBCamA22704522A6F00`, bus `usb-0000:00:14.0-1.1`, version
`6.14.11` (`396811`), `capabilities=0x84a00001`, `device_caps=0x04200001` —
`V4L2_CAP_VIDEO_CAPTURE` and `V4L2_CAP_STREAMING` both present, nothing
required missing.

### The measured result: `sizeimage` is four times the descriptor's buffer size

This is the finding of the run.

`VIDIOC_G_FMT` on the freshly opened handle reported the only mode this
function advertises, and reported it with a buffer size far larger than the
device's own descriptor asks for:

| Field | Value from `G_FMT` | Value after the single `S_FMT` |
| --- | --- | --- |
| `pixelformat` | `MJPG` (`0x47504a4d`) | `MJPG` (`0x47504a4d`) |
| `width` x `height` | 1280x800 | 1280x800 |
| `field` | `V4L2_FIELD_NONE` | `V4L2_FIELD_NONE` |
| `bytesperline` | 0 (compressed) | 0 (compressed) |
| **`sizeimage`** | **8192000** | **8192000** |
| `colorspace` | 8 | 8 |

`VIDIOC_S_FMT` was issued exactly once, asking for `MJPG` 1280x800. The
`ioctl` itself **succeeded**; the driver answered with the format above. The
run then refused it, because:

- the recorded UVC frame descriptor for this mode gives
  `dwMaxVideoFrameBufferSize` = **2048000 B**, and the profile's fixed
  per-buffer map ceiling **at the time of this run** was derived from it as
  2 MiB (**2097152 B**);
- `uvcvideo` reported `sizeimage` = **8192000 B** — exactly **4x** the
  descriptor value, and also exactly `1280 x 800 x 8`.

Since 8192000 > 2097152, the shared engine declined to map the buffer and
stopped. The refusal is the designed behaviour, not a crash: a profile has one
mode and one ceiling, and a driver answering outside it is refused rather than
accommodated.

> **Superseded ceiling.** The 2097152 B figure above is historical. Its only
> justification was the descriptor, and this run is precisely the measurement
> that showed the descriptor does not predict what `uvcvideo` allocates. In
> scanner-011 the RGB profile's ceilings were re-derived from the measured
> value instead — 8192000 B per mapped buffer and per raw artifact, 32768000 B
> mapped in total across at most four buffers — so that the driver's own
> `sizeimage` is mappable. Nothing else about the profile changed, the
> descriptor's 2048000 B is still recorded verbatim as configuration, and the
> new ceiling is **not** a claim that a compressed 1280x800 JPEG occupies
> 8192000 bytes: only `bytesused` says how many bytes a frame carries, and only
> those bytes are ever written. The depth profile's ceilings are untouched.
>
> The raised ceiling was subsequently exercised against this unit exactly
> once. It proved sufficient — four 8192000 B buffers were allocated,
> mapped, queued and streamed — and the run then failed at a later
> boundary instead. See
> [Owned code attempts a raw RGB frame with the measured ceiling](#owned-code-attempts-a-raw-rgb-frame-with-the-measured-ceiling-measured).

**Nothing after that point ran.** No `VIDIOC_REQBUFS`, no `VIDIOC_QUERYBUF`,
no `mmap`, no `VIDIOC_QBUF`, no `VIDIOC_STREAMON`, no poll, no
`VIDIOC_DQBUF`. Zero bytes were dequeued, so no `rgb-frame.jpg` and no JPEG
structural evidence exist for this run — the output directory was never
created. Cleanup was the single `close()` of fd 4, which took 0.148 ms and
succeeded.

### Post-exit observation

`scanner-009` recorded an unexplained re-enumeration 2.788 s after its capture,
so this run was followed by a bounded **45.06 s** read-only observation window
sampling the same serial's `devnum` and V4L2 nodes. Observation is not a retry,
a reset or an attribution.

| Observed | Before | Through 45 s after |
| --- | --- | --- |
| `devnum` | 42 | 42, unchanged (no transition sampled) |
| V4L2 nodes | `video0`, `video1`, `video2`, `video3` | identical set, identical listing |
| Interface/driver bindings | `uvcvideo` on all four | byte-identical |
| Holders of any `/dev/video*` | none | none |
| New kernel messages | — | none |

**No re-enumeration occurred.** This does **not** explain the `scanner-009`
event: this run never reached `STREAMON`, so the two are not comparable, and
the cause of that earlier re-enumeration remains unresolved.

### What this run establishes

- The RGB function's capture node is reachable by role and index, opens, and
  answers `QUERYCAP` and `G_FMT` under plain V4L2 with `uvcvideo` bound.
- On this unit and this kernel (6.14.11), `uvcvideo` reports
  `sizeimage=8192000` for `MJPG` 1280x800 — **the device's own
  `dwMaxVideoFrameBufferSize` of 2048000 does not predict the driver's
  `sizeimage` for this mode.** That contradiction is measured, and it is the
  reason no frame was captured.
- The bounded lifecycle refuses and cleans up correctly at the format boundary:
  one open, one close, no leaked mapping, no holder, no device disturbance.

### What this run does not establish

- **Any RGB frame in this standalone attempt.** None was captured; later
  paired runs are documented in the higher-level evidence section above.
- **Whether a capture would succeed with a larger ceiling.** Not tested by
  *this* run. What the kernel would actually have filled — 2048000 B, 8192000 B,
  or whatever a compressed frame happens to be — was unknown here, because
  nothing was mapped or streamed. (The ceiling was subsequently raised to the
  measured value in scanner-011 and one further attempt was made; that attempt
  is recorded in
  [Owned code attempts a raw RGB frame with the measured ceiling](#owned-code-attempts-a-raw-rgb-frame-with-the-measured-ceiling-measured),
  and nothing in this section anticipates its result.)
- **Anything about JPEG content, colour, white balance, calibration or
  synchronisation with the depth camera.** Zero bytes were dequeued.
- **The descriptor's 30 fps default as a measured rate.** No `VIDIOC_S_PARM`
  was issued and no frame arrived; 30 fps remains a descriptor value only.
- **The cause of the `scanner-009` re-enumeration.** Still unresolved.

## Owned code attempts a raw RGB frame with the measured ceiling (measured)

Task `scanner-011` re-derived the RGB profile's map ceilings from the
`sizeimage` scanner-010 measured, and was then **run exactly once**, on
2026-08-22 at 03:08:36 local time (`08:08:36Z`), against this unit. It did
**not** produce an RGB frame either — but it failed at a different and
strictly later boundary than scanner-010 did. The format boundary is gone: the
stream started, and then nothing arrived. The cumulative live-attempt count
against the RGB node is **two** (one for scanner-010, one for scanner-011);
for `scanner-011` itself it is **one**.

`uvcvideo` stayed bound to all five interfaces throughout, no interface was
taken from the kernel, no vendor/XU, HID or usbfs transfer was issued, and the
depth node was not opened.

### Exact command

```bash
.venv/bin/scanner3d rgb-frame --serial A22704522A6F00A64 \
  --execute-rgb-frame-capture --output-dir /tmp/scanner-011-evidence/artifact \
  --json --json-out /tmp/scanner-011-evidence/result.json
```

Run under an external `timeout --signal=TERM --kill-after=20s 60s` deadline,
longer than the command's own 8.0 s internal poll budget. The external deadline
never fired: the process exited on its own after **8.148 s** wall
(8.027 s measured inside the command), with exit code **4**
(`poll_timeout`). The harness that ran it creates `/tmp/scanner-011-evidence/`
with a bare `mkdir` and writes a read-only attempt marker before launching, so
a second harness invocation would refuse with the harness's own exit 90 before
it could reach the device. That guard was never triggered, and this document
does not claim it was tested: the harness log records exactly one run
(`HARNESS DONE rc=4`) and the evidence root holds exactly one attempt.

Before the capture the same harness also ran `scanner3d rgb-frame --json`
*without* `--execute-rgb-frame-capture`, preserved as `before/dry-run.json`.
That dry run only resolved the node (`status: not_attempted`, one
`resolve_node` step, exit 0); it opened no video node, negotiated no format and
allocated no buffer. It is therefore not counted as a live attempt above.

### Node identification

Resolved exactly as before, from the USB interface string descriptor plus the
V4L2 index — never from a `/dev/videoN` number and never from a caller-supplied
path. The resolution was byte-for-byte the same as scanner-010's:
`/dev/video2`, rdev `81:2`, inode 3119, mode `0o660`, interface 2
(`RGBCamA22704522A6F00A64`), V4L2 index 0 capture, bus 1 devnum 42, driver
`uvcvideo`. `VIDIOC_QUERYCAP` again answered driver `uvcvideo`, card
`ZXProduct: RGBCamA22704522A6F00`, bus `usb-0000:00:14.0-1.1`, version
`6.14.11` (`396811`), `capabilities=0x84a00001`, `device_caps=0x04200001`.

### The raised ceiling is the right number: everything up to `STREAMON` succeeded

`VIDIOC_G_FMT` and the single `VIDIOC_S_FMT` reported `MJPG` 1280x800,
`field=V4L2_FIELD_NONE`, `bytesperline=0`, **`sizeimage=8192000`**,
`colorspace=8`, `priv=0xfeedcafe` — the same `sizeimage` scanner-010 measured,
reproduced on a second run. With the ceiling now equal to that value the run
did not refuse itself, and every step through the start of streaming
succeeded:

One caveat about the JSON this run emitted: the `initial_format` and
`negotiated_format` blocks carry `row_padded` and `tight_frame_bytes` fields
that are computed from the depth path's `Y16 ` arithmetic
(`width * height * 2`) by the `PixelFormat` record both profiles share. For a
compressed MJPEG bitstream neither field means anything — there is no row
stride (`bytesperline=0`) and no tight pixel packing — and the
`tight_frame_bytes=2048000` they report equals
`RGB_DESCRIPTOR_FRAME_BUFFER_BYTES` only by arithmetic coincidence, not
because it was derived from the descriptor. The human-readable renderer
already suppresses both for compressed profiles; the JSON does not. Read
`sizeimage` and, for a real frame, `bytesused` — never `tight_frame_bytes` —
as the RGB byte facts.

| Step | Result |
| --- | --- |
| `VIDIOC_REQBUFS` (count=4, MMAP) | 4 buffers allocated, `capabilities=0x00000017` |
| `VIDIOC_QUERYBUF` x4 | each `length=8192000`, `flags=0x00012000`, `bytesused=0` |
| `mmap` x4 | each **8192000 B** mapped read/write shared, 0.28–0.32 ms each |
| `VIDIOC_QBUF` x4 | all four queued |
| `VIDIOC_STREAMON` | succeeded in 1.66 ms; the 8.0 s deadline starts here |

The four buffers were reported at `mmap` offsets 0, 32768, 65536 and 98304.
Those are V4L2 buffer cookies, not byte positions in one contiguous region —
they are 32768 apart while each mapping is 8192000 B long — so what the
32768000 B total-mapped ceiling bounds is the sum of the four mapping lengths
(4 x 8192000 = 32768000 B of address space), which is exactly what was mapped
and exactly the ceiling. Nothing exceeded it and no fifth mapping was made.

### The measured result: the stream starts and produces no buffer

This is the finding of the run.

| Fact | Value |
| --- | --- |
| `poll` calls | 1 (no restarts, no retries) |
| Deadline given to that one poll | 7.99999 s of the 8.0 s budget |
| Waited | **8.007 s** |
| `revents` | **`0x0000`** — not `POLLIN`, not `POLLERR` |
| Buffers dequeued | **0** |
| `VIDIOC_DQBUF` | never issued |
| Stream duration (`STREAMON` to `STREAMOFF`) | 8.012 s |

The single poll returned zero revents. No buffer became ready in the 8.0 s
after `VIDIOC_STREAMON`, so `VIDIOC_DQBUF` was never reached, `bytesused` was
never read from a filled buffer, and zero bytes were dequeued. Per the profile
the run did **not** retry.

**No artifact exists for this run.** `/tmp/scanner-011-evidence/artifact/` is
empty: `preserve_rgb_frame` writes only `bytesused` bytes of a dequeued buffer,
there was no dequeued buffer, and mapping padding is never written. There is
therefore no `rgb-frame.jpg`, no SHA-256, no SOI/EOI or marker-segment
evidence, and no SOF dimension check for this run. No pixel was decoded,
because no bytes existed to decode.

### Cleanup

The shielded cleanup ran completely and every step succeeded: `VIDIOC_STREAMOFF`
(0.669 ms), four `munmap`s (0.257–0.299 ms each, releasing all 8192000 B x 4),
`VIDIOC_REQBUFS(count=0)` (0.928 ms) and `close()` of fd 4 (0.078 ms). No
mapping, buffer queue or file descriptor was leaked, and no recovery guidance
was emitted.

### Post-exit observation

The run was followed by a bounded **76 s** read-only observation window
(exit at 03:08:44, last sample at 03:10:00), sampling at +15 s, +30 s, +45 s
and +60 s. Every sample was a sysfs, procfs or udev-database read; no node,
usbfs endpoint or control was opened. Observation is not a retry, a reset or an
attribution.

| Observed | Before | Through 76 s after |
| --- | --- | --- |
| `devnum` | 42 | 42, unchanged (no transition sampled) |
| V4L2 nodes | `video0`, `video1`, `video2`, `video3` | identical set, identical listing |
| Interface/driver bindings | `uvcvideo` on interfaces 0–3, `usbhid` on 4 | byte-identical |
| Holders of any `/dev/video*` | none | none |
| `uvcvideo` module refcount | 0 | 0 |
| New kernel messages | — | **none** (`dmesg` line count 4066 before and after) |

**No re-enumeration occurred**, and this time the run did reach `STREAMON` —
so unlike scanner-010, this run *is* comparable to `scanner-009`'s streaming
depth capture on that axis, and it produced no such event. That still does not
explain the `scanner-009` re-enumeration; it only records that streaming the
RGB node for 8 s did not reproduce one.

### What this run establishes

- **The measured ceiling is correct and sufficient.** With the per-buffer
  ceiling set to the driver's own `sizeimage`, `uvcvideo` allocated, reported
  and mapped four 8192000 B MMAP buffers without complaint, and the run reached
  `VIDIOC_STREAMON`. The scanner-010 refusal was a ceiling that was too small,
  and nothing else.
- **`sizeimage=8192000` for `MJPG` 1280x800 reproduced** on a second,
  independent run against the same unit and kernel (6.14.11).
- **The RGB stream starts but delivers no frame under plain V4L2 alone.** On
  this unit, `VIDIOC_STREAMON` on the RGB capture node succeeds and then no
  buffer becomes ready within 8.0 s, with no error reported on the file
  descriptor (`revents=0x0000`, not `POLLERR`). Streaming is accepted and
  silently produces nothing.
- The bounded lifecycle refuses, stops and cleans up correctly at the *poll*
  boundary too: one open, one close, four mappings and four unmappings, no
  leaked buffer queue, no holder, no device disturbance.

### What this run does not establish

- **Any RGB frame in this standalone attempt.** None was captured. Later paired
  runs delivered strict-invalid RGB payloads, but the raw-RGB usefulness half
  of the M3 milestone remains unmet.
- **Why no buffer arrived.** Not determined. Nothing here distinguishes between
  a camera that needs a vendor initialisation or control write before it will
  emit MJPEG, a bandwidth/alt-setting negotiation that silently yields nothing,
  a sensor that needs longer than 8.0 s to produce its first frame, and any
  other cause. This run measured the outcome, not the mechanism, and it did not
  and will not probe for one.
- **That a longer deadline, a retry, another mode or another node would
  succeed.** The profile has one mode, one deadline, one dequeue and no
  retries, and exactly one attempt was made. Nothing was varied, so nothing
  about any variation is known.
- **Anything about JPEG content, structure, dimensions, colour, white balance,
  calibration or synchronisation with the depth camera.** Zero bytes were
  dequeued.
- **The descriptor's 30 fps default as a measured rate.** No `VIDIOC_S_PARM`
  was issued and no frame arrived; 30 fps remains a descriptor value only.
- **The cause of the `scanner-009` re-enumeration.** Still unresolved.

## Why the RGB stream is silent: offline reconstruction (scanner-012)

`scanner-012` spent no live attempt. It re-reads the frozen `scanner-009`,
`scanner-010` and `scanner-011` runs and the recorded configuration descriptor,
decodes the descriptor fields this repo had never extracted, deepens the pinned
SDK trace for MINI RGB start specifically, and asks the kernel module — not the
internet — what it does with this device. The point is to say which
explanations the existing evidence eliminates, which it leaves standing, and
what single experiment would separate the survivors.

**Headline: the descriptor removes bandwidth and alt-setting negotiation from
the hypothesis space entirely, and the depth function is a positive control on
the identical kernel path. What remains unmeasured is what happens on the wire
after `VIDIOC_STREAMON`, and the kernel was configured in a way that could not
have told us.**

### Evidence, and how to reproduce every number below

Three immutable inputs are now committed rather than living only in `/tmp`:

| fixture | SHA-256 | provenance |
| --- | --- | --- |
| `tests/fixtures/mini-config-descriptor.bin` | `2fa267c0…d626613` | byte-for-byte configuration descriptor of the physical MINI |
| `tests/fixtures/scanner-009-depth-capture.json` | `1fac16a8…8d56d0eef` | the one successful depth capture (`/tmp/scanner-009-evidence/live.json`) |
| `tests/fixtures/scanner-010-rgb-attempt.json` | `9b86f363…144f4f4d` | first RGB attempt, refused at the ceiling (`/tmp/scanner-010-evidence/rgb-frame.json`) |
| `tests/fixtures/scanner-011-rgb-attempt.json` | `cce52dcc…7ea98b33a` | the silent-stream run (`/tmp/scanner-011-evidence/result.json`) |

Each JSON fixture was verified byte-identical to its `/tmp` original with `cmp`
at the time of copying. Two offline helpers turn them into the tables below:

```bash
# the full UVC/V4L2 reconstruction
python reference/analysis/uvc_stream_state.py            # add --json for the machine form

# the uvcvideo module-parameter comparison
zstd -d /lib/modules/$(uname -r)/kernel/drivers/media/usb/uvc/uvcvideo.ko.zst -o /tmp/uvcvideo.ko
python reference/analysis/uvc_module_params.py --module /tmp/uvcvideo.ko
```

Both are covered by offline tests (`tests/test_uvc_stream_state.py`,
`tests/test_uvc_module_params.py`) that run in the default gate with no device,
no kernel module and no binutils.

### The RGB VideoStreaming interface, decoded

`docs/architecture.md` previously listed six values as **unrecorded** and named
them as the blocker on direct-transport work: endpoint address, endpoint
transfer type, `wMaxPacketSize`, alternate-setting number,
`bFormatIndex`/`bFrameIndex`, and the frame interval. All six are in the
recorded descriptor and are now decoded and pinned by tests. `FACT`.

| field | depth function | **RGB function** |
| --- | --- | --- |
| VideoControl interface | 0 | **2** |
| VideoStreaming interface | 1 | **3** |
| `bcdUVC` | `0x0100` (UVC 1.00) | **`0x0100` (UVC 1.00)** |
| VS Probe/Commit block | 26 bytes | **26 bytes** |
| `dwClockFrequency` | 48 000 000 Hz | 48 000 000 Hz |
| alternate settings | `[0]` — only one | **`[0]` — only one** |
| endpoint | `0x82` IN | **`0x84` IN** |
| endpoint transfer type | **bulk** | **bulk** |
| `wMaxPacketSize` | 512 | **512** |
| `bInterval` | 0 | 0 |
| `bTerminalLink` | 3 | 3 |
| `bmInfo` (dynamic format change) | `0x00` — no | `0x00` — no |
| `bNumFormats` | 2 (`YUY2`, `Y16 `) | **1 (`MJPG`)** |
| frames per format | 2 / 3 | **1** |
| `bFormatIndex` / `bFrameIndex` used | 2 / 1 | **1 / 1 (forced — nothing else exists)** |
| `dwMaxVideoFrameBufferSize` | 512 000 (`Y16 ` 640x400) | **2 048 000** |
| `dwDefaultFrameInterval` | 666 666 (15 Hz) | **333 333 (30 Hz)** |
| discrete intervals | 15 / 10 / 5 Hz | **30 / 20 / 15 / 10 Hz** |
| `dwMaxBitRate` | 40 960 000 bit/s | **163 840 000 bit/s** |

UVC 1.00 is confirmed independently of the descriptor: the kernel log carries
`Found UVC 1.00 device ZXProduct (2207:110c)` **twice** per enumeration, once
per video function. `FACT`.

Three consequences follow directly, and each kills a candidate explanation:

- **There is no alternate setting to select.** Every interface on this device
  declares exactly one alternate setting (0). The whole UVC isochronous
  bandwidth dance — pick an alt setting whose `wMaxPacketSize` covers the
  frame, fail with `-ENOSPC` if none does — does not apply. uvcvideo's
  `Selecting alternate setting %u (%u B/frame bandwidth)` and `No fast enough
  alt setting for requested bandwidth` messages are isochronous-only and are
  unreachable here. `FACT`.
- **Both streams are bulk, not isochronous.** Bulk transfers have no reserved
  bandwidth and no admission control, so `VIDIOC_STREAMON` cannot fail — or
  silently under-deliver — for bandwidth reasons. It returned success in
  1.66 ms. `FACT`.
- **The peak advertised RGB rate is 20.48 MB/s against a 53.248 MB/s ceiling.**
  USB 2.0 permits at most 13 bulk transactions of 512 bytes per 125 µs
  microframe on one high-speed endpoint; `dwMaxBitRate` needs **38.5%** of
  that. The depth mode that *worked* needs 9.6%. Bandwidth does not distinguish
  the two cases. `FACT` (arithmetic over descriptor values; not a measurement
  of achieved throughput).

An earlier hand-decode of this blob during `scanner-012` misread two interrupt
endpoints and produced a spurious "two interfaces share endpoint `0x83`"
finding. The decoder disproves it: interface 0 uses `0x81`, interface 2 uses
`0x83`, and `duplicate_endpoint_addresses()` returns empty. The test
`test_no_endpoint_address_is_declared_by_two_interfaces` exists to keep that
correction from having to be remembered.

### What `scanner-011` actually negotiated, and what it never did

Owned code issued exactly this, in order, all succeeding except the poll
(`tests/fixtures/scanner-011-rgb-attempt.json`, `steps`):

`open` → `VIDIOC_QUERYCAP` → `VIDIOC_G_FMT` → `VIDIOC_S_FMT` →
`VIDIOC_REQBUFS(4)` → 4 × (`VIDIOC_QUERYBUF` + `mmap`) → 4 × `VIDIOC_QBUF` →
`VIDIOC_STREAMON` → **`poll()` 8.007 s, `revents=0x0000`** → `VIDIOC_STREAMOFF`
→ 4 × `munmap` → `VIDIOC_REQBUFS(0)` → `close`.

Standard operations that owned code **never issued** — the honest list, since
"we did everything standard" would be false:

| never issued | what it would have done |
| --- | --- |
| `VIDIOC_S_PARM` / `VIDIOC_G_PARM` | request or read back the frame interval |
| `VIDIOC_ENUM_FMT` / `_FRAMESIZES` / `_FRAMEINTERVALS` | ask the driver what it believes is available |
| `VIDIOC_QUERYCTRL` / `G_CTRL` / `S_CTRL` | touch any UVC control (exposure, gain, white balance, power-line) |

The successful depth capture issued a **subset** of what the RGB attempt
issued, never a superset (`test_the_depth_run_negotiated_no_more_than_the_rgb_run`).
So whatever the RGB path lacks, **it is not a V4L2 call the working depth
capture made and the RGB attempt skipped.** `FACT`.

What `uvcvideo` did on the tool's behalf is not directly observed. `INFERENCE`,
and load-bearing: `VIDIOC_S_FMT` drives the driver's VS_PROBE `SET_CUR`/
`GET_CUR`, and `VIDIOC_STREAMON` drives VS_COMMIT `SET_CUR` and URB submission
on endpoint `0x84`. The *negotiated* `bmHint`, `dwFrameInterval`,
`dwMaxVideoFrameSize` and `dwMaxPayloadTransferSize` were never read back by
anything and are **unmeasured**.

### The one measured anomaly: `sizeimage` is exactly 4x the descriptor

| run | format | descriptor `dwMaxVideoFrameBufferSize` | driver `sizeimage` | ratio |
| --- | --- | --- | --- | --- |
| `scanner-009` depth | `Y16 ` 640x400, uncompressed | 512 000 | 512 000 | **1.0** |
| `scanner-010`/`011` RGB | `MJPG` 1280x800, compressed | 2 048 000 | 8 192 000 | **4.0** |

Every uncompressed frame in this descriptor declares exactly
`width × height × bpp / 8` bytes, and the depth capture's `sizeimage` matched
its descriptor value to the byte. The MJPG entry declares `1280 × 800 × 2` =
2 048 000 — the *uncompressed* 16-bpp size, a generically filled field that
carries no information about real JPEG sizes.

`INFERENCE`, and the most interesting one available offline: uvcvideo has no
descriptor-derived value it can trust for a compressed format, so `8192000` is
most plausibly the `dwMaxVideoFrameSize` the **device itself returned in its
VS_PROBE `GET_CUR` response**. If that is right, the device answered probe
negotiation on interface 3 with a live, non-descriptor value — which would mean
the control channel and mode negotiation are working, and the silence begins
strictly after COMMIT. This cannot be confirmed offline: the uvcvideo source is
not available on this host (see below), and no probe block has ever been read
back from this unit. It is the first thing the next experiment should measure.

### The control case: depth streams over the same kernel path

`scanner-009` captured a complete 512 000-byte `Y16 ` frame through the same
engine, the same `uvcvideo`, the same host and the same bulk-at-alt-0 topology,
with **no vendor initialisation in that run**, no `VIDIOC_S_PARM` and no UVC
control write. First buffer ready **1.043 s** after `STREAMON`.

That is the sharpest fact in this section. The difference between working and
silent is not the driver, the host, the transport type, the negotiation
sequence or the tool — it is **which USB function is asked to stream**.

Two caveats keep this from proving more than it does. The depth run used a
different resolution and format at a different frame rate, so the comparison is
between functions, not a controlled single-variable test. And `scanner-007`/
`scanner-008` had issued vendor XU traffic to the device earlier in its life;
whether any of that persisted across the intervening re-enumerations into the
`scanner-009` power cycle is **`UNKNOWN`**.

### The kernel said nothing — and why that is worth much less than it looks

Across the whole `scanner-011` attempt and 60 s of observation the kernel log
grew by **zero lines** (`kernel-delta.txt`: `before=4066 after=4066 delta=0`),
and a full ring-buffer read finds no `uvcvideo` message for this device beyond
routine `Found UVC 1.00 device` enumeration lines. `dmesg` was readable by the
invoking user (`dmesg_restrict=0`).

It is tempting to read that silence as "the driver hit no error path". Only
part of that survives scrutiny, and the module itself settles which part:

- The module carries **exactly one** dynamic-debug callsite. Its `__dyndbg`
  section is 0x38 bytes — one `_ddebug` descriptor — and it is
  `uvc_ctrl_restore_values`/`"restoring control %pUl/%u/%u"`. So uvcvideo's
  `uvc_dbg()` diagnostics are **not** `pr_debug` callsites gated by dyndbg;
  they are gated by the `trace` module parameter. `FACT`, reproducible with
  `readelf -SW`/`readelf -rW` on the decompressed module.
- `trace` was **0**. Every `uvc_dbg()` message — probe/commit contents, URB
  completion status, frame decoding, statistics — was therefore suppressed at
  the source. `FACT`.
- Consequently the absence of kernel output tells us only that **no
  `dev_err`/`dev_warn`/`dev_info` path fired**: no `Failed to set UVC probe
  control`, no `Failed to submit URB`, no `No valid video chain found`, no
  `Unable to initialize the status endpoint`. That is real but weak evidence.
  It says nothing whatsoever about whether URBs completed, with what status, or
  with how many bytes.

**The instrument was switched off.** This is the single most actionable finding
in `scanner-012`, and it is what makes the next experiment cheap.

### The module-parameter reading that had to be corrected

A first pass over `/sys/module/uvcvideo/parameters/` reported
`quirks = 4294967295` and concluded that every UVC quirk workaround had been
forced on for this host. **That is wrong**, and it would have poisoned the whole
analysis. `4294967295` is `(unsigned)-1`, the value the driver compiles in to
mean *no module parameter was supplied*.

The proof is offline and reproducible: the initialiser stored in the module's
own `.data` section is the same `ffffffff`. Every live parameter equals the
on-disk compiled default, byte for byte:

| parameter | compiled default | live | verdict |
| --- | --- | --- | --- |
| `quirks` | 4294967295 | 4294967295 | compiled default — **no quirk forced** |
| `nodrop` | 1 | 1 | compiled default |
| `timeout` | 5000 | 5000 | compiled default |
| `hwtimestamps` | 0 | 0 | compiled default |
| `trace` | 0 | 0 | compiled default |

`reference/analysis/uvc_module_params.py` performs exactly this comparison, and
`tests/test_uvc_module_params.py` pins the sentinel reading so the mistake
cannot recur silently. `FACT`.

One of these defaults matters directly. **`nodrop = 1`** — the module's own
description is `Don't drop incomplete frames`. Under that setting the driver is
configured to hand an incomplete frame up to userspace rather than discard it.
`INFERENCE` (the exact branch is in source this host does not have): "frames
arrived but the driver silently discarded them as erroneous" is therefore a
poor explanation for a queue that produced nothing at all.

### `2207:110c` is not a quirked device, and no UVC source or docs exist locally

- **uvcvideo source: `ABSENT`.** Only `linux-headers-*` packages are installed;
  they contain no `drivers/media/usb/uvc/*.c`. No dkms tree, no kernel git
  tree, nothing matching `uvc_driver.c`/`uvc_video.c` anywhere on the
  filesystem.
- **UVC/V4L2 documentation: `ABSENT`.** No `linux-doc`, no `v4l-utils` (no
  `v4l2-ctl`, no man page). The headers packages ship
  `Documentation/userspace-api/media/` directories containing only a `Makefile`
  — Ubuntu strips the `.rst` content.
- **Per-device quirk entry for `2207:110c`: `ABSENT`.** The 16-bit pattern
  `0x2207` does not occur **anywhere** in the compiled module — zero
  occurrences in the whole file. `0x110c` occurs four times, all inside
  `.BTF` debug-type metadata, none in `.rodata`/`.data`, and never adjacent to
  a vendor id. The `modinfo` alias list carries ~70 vendor-specific
  `usb:vXXXXpYYYY…` entries; this device is in none of them. It binds through
  the generic UVC class-code alias
  (`usb:v*p*d*dc*dsc*dp*ic0Eisc01ip00in*`) and receives **no driver_info
  quirks at all**. `FACT`.

Per the repository's own constraint this was established without browsing or
installing anything; the absences above are recorded as absences, not as
"probably fine".

### The SDK's MINI RGB start path, deepened

`scanner-006` established that streaming goes through a vendored **libuvc** over
`libusb` and never through V4L2. `scanner-012` follows the RGB case specifically
(pinned `lib3DCamera.so`, SHA-256 `49e0865c…cdae3c4`, re-verified). Ordering,
which is what static analysis could not previously supply:

```
cs::Camera::startStream                       0x1f50a8
  streamType == 0 -> OnRecvDepthFrameData 0x1fa5b4
  streamType != 0 -> OnRecvRgbFrameData   0x1fa580      # the ONLY depth/RGB difference here
  -> cs::UvcCamera::startStream               0x24d406
       -> cs::UvcStream::startStream          0x252fcc
            my_uvc_open_internal              0x2526ca  # claim the VideoControl interface
            my_uvc_get_stream_ctrl_format_size 0x252974 # match bFormatIndex/bFrameIndex/interval;
                                                        # claim the VideoStreaming interface
              uvc_query_stream_ctrl(probe=1, GET_MAX 0x83)   wValue=0x0100
              uvc_probe_stream_ctrl           0x260252
                uvc_query_stream_ctrl(probe=1, SET_CUR 0x01) wValue=0x0100
                uvc_query_stream_ctrl(probe=1, GET_CUR 0x81) wValue=0x0100
                _uvc_stream_params_negotiated 0x2601fb
            my_uvc_start_streaming            0x252d8a
              uvc_stream_open_ctrl            0x260b32
                uvc_stream_ctrl               0x260006
                  uvc_query_stream_ctrl(probe=0, SET_CUR 0x01) wValue=0x0200   # COMMIT
              uvc_stream_start                0x260c1e
                libusb_set_interface_alt_setting        0x260f6e  # AFTER commit; the only one
                2 x libusb_fill_{iso,bulk}_transfer, timeout 5000 ms
                libusb_submit_transfer
       ossSleep(1000 ms) on success           0x24d59f
```

Points that bear on the silence, each `FACT` unless marked:

- **No XU, ASCII, property or file command is issued anywhere in this chain**,
  for either stream. `libuvcExcuteCmd`, `libuvcSetExtension`,
  `libuvcReadValueExtension` and `libuvcReadFile` do not appear in
  `Camera::startStream`, `UvcCamera::startStream`, `UvcStream::startStream`, or
  anything they call down to `uvc_stream_start`. This is a *scoped* absence —
  the functions actually disassembled — not a whole-binary cross-reference
  sweep, and the 3DViewer application's own call sites above `startStream` were
  not traced. So it rules out an *automatic* pre-stream register poke inside
  `startStream`; it does not rule out one the application performs earlier.
- **`setStreamResolution` (`echo s 0x702 <w> <h>`) is a separate, explicitly
  invoked API**, not called by this chain.
- The probe acceptance gate `_uvc_stream_params_negotiated` compares only
  `bFormatIndex`, `bFrameIndex` and **`dwMaxPayloadTransferSize`**; a
  `dwFrameInterval` the device clamped is explicitly tolerated. Rejection
  returns `-51 UVC_ERROR_INVALID_MODE` with no retry.
- The SDK's post-start wait is **one `ossSleep(1000 ms)`** — there is no
  poll-for-first-frame loop at this layer. Per-transfer timeout is 5000 ms, and
  exactly **two** transfers are kept in flight, resubmitted unconditionally by
  `_uvc_stream_callback` `0x2608c7`.
- The libusb event pump (`OnHandleEventThread` `0x24f40e`, 100 ms granularity)
  is started at `connect()` time, long before any `startStream`.
- `bcdUVC > 0x010f ? 34 : 26` at `0x25fb74`/`0x25fb7e` selects the wire length.
  The SDK trace alone cannot say which applies; the descriptor can, and does:
  **26 bytes**. That is the kind of gap this task was meant to close.

`NOT RECOVERED`: the concrete interface/altsetting bytes passed to
`libusb_set_interface_alt_setting` (read at runtime from the matched
descriptor — the descriptor now says the only legal answer is interface 3,
alt 0), and anything about why a real MINI would accept COMMIT and then send
nothing, which is inherently on the far side of the USB boundary.

### Ranked evidence table

Ranked by how much the existing evidence constrains each explanation. "Status"
is about *this* body of evidence, not about the world.

| # | explanation | evidence for | evidence against | status |
| --- | --- | --- | --- | --- |
| 1 | **Device needs an out-of-band / vendor step before RGB payload flows** | depth streams on the identical path while RGB does not; SDK commands resolution out-of-band via `echo s 0x702`; `Types.hpp:490` names `start_stream[0x481]`/`stop_stream[0x483]` and the binary carries `0x484`/`0x487`; the RGB probe response appears to differ from the descriptor | the SDK's own `startStream` chain issues **no** such command for either stream (scoped absence); no such step was needed for depth | **leading, not established** |
| 2 | **Driver-side or device-side silence after COMMIT (URBs empty, stalled, or never completing)** | nothing was ever dequeued; `nodrop=1` means even incomplete frames would have been handed up, so no frame boundary was seen at all | none — this is simply **unobserved**; `trace=0` suppressed every message that would have shown it | **not eliminated; unmeasured** |
| 3 | **Something in the negotiated probe/commit block is wrong** (`dwMaxPayloadTransferSize`, `dwMaxVideoFrameSize`, interval) | `sizeimage` is 4x the descriptor for RGB and exact for depth; the SDK treats `dwMaxPayloadTransferSize` as a hard acceptance criterion | `STREAMON` succeeded; no `Failed to set UVC probe control` in the log (a `dev_err` path, so its absence *is* meaningful) | **open, directly measurable** |
| 4 | **Too-short poll** | none | 8.007 s ≈ 240 frame periods at the 30 Hz default; depth's first frame took 1.043 s; the SDK's own whole post-start wait is 1 s and its per-transfer timeout 5 s | **unlikely; not eliminated** (nothing beyond 8 s has ever been tried) |
| 5 | **Missing `S_PARM` / frame-interval negotiation** | `S_PARM` was never issued | uvcvideo commits an interval at `S_FMT` regardless, and the descriptor default (333 333 = 30 Hz) is one of only four legal values; the SDK explicitly tolerates a clamped interval; depth succeeded with no `S_PARM` either | **largely eliminated as a sole cause** |
| 6 | **Endpoint / bandwidth / alt-setting negotiation** | none | interface 3 has exactly one alt setting; the endpoint is **bulk**, which has no admission control; 20.48 MB/s of a 53.248 MB/s ceiling; `STREAMON` returned in 1.66 ms; the relevant kernel messages are isochronous-only and unreachable | **eliminated** |
| 7 | **Wrong node / role resolution** | none | node resolved from the USB interface string `RGBCamA22704522A6F00A64` on interface 2 plus V4L2 index 0; card `ZXProduct: RGBCam…`; `G_FMT` already reported `MJPG` 1280x800, a mode **only** the RGB function advertises | **eliminated** |
| 8 | **Device-specific uvcvideo quirk mis-applied** | none | `2207:110c` appears nowhere in the module; `quirks` is at its unset sentinel; every module parameter is at its compiled default | **eliminated** |

### What is still unknown after this task

- Whether any byte ever arrives on endpoint `0x84` after COMMIT. **Nothing in
  any run has ever observed the RGB endpoint.**
- The negotiated 26-byte VS Probe/Commit block for RGB — in particular whether
  `dwMaxVideoFrameSize` really is the source of `sizeimage=8192000`, and what
  `dwMaxPayloadTransferSize` was agreed.
- Whether the MINI requires an ISP/`rk_preisp` register write before the RGB
  sensor emits anything, and if so which. The host binary names candidates
  (`0x481`/`0x483`/`0x484`/`0x487`); their firmware-side semantics remain
  `UNKNOWN`, and none may be issued without a read-current-state and an exact
  restore (see experiment 5, still `NOT READY`).
- Whether the RGB stream behaves differently while the depth function is also
  streaming. Never tried; both functions have never been open at once.
- The `scanner-009` re-enumeration cause. Still unresolved.

## Static protocol archaeology: SDK call paths toward the wire

`scanner-006` is a reproducible, offline static-analysis pass over the pinned
vendor `lib3DCamera.so` and the official 3DViewer source. It traces the four
capture-relevant domains — **stream start/stop**, **stream/mode selection**,
**projector/illumination**, and **calibration/depth-scale** — from public SDK
entry points as far toward concrete USB operations as static evidence permits.
No scanner, device node, SDK execution or network capture was involved. Every
relationship below is backed by a symbol, offset, disassembly excerpt or pinned
source location; inference and absence-of-evidence are labelled `INFERENCE` /
`UNKNOWN`. Most control-transfer tuples are re-derivable with
[`reference/analysis/xu_trace.py`](../reference/analysis/README.md); the ones
that helper cannot reconstruct are cited to disassembly directly.

**How far it actually got, up front:** the XU control primitives
(`libuvcExcuteCmd`, `libuvcReadValueExtension`, `libuvcReadFile`) are recovered
to exact, complete on-wire shapes. Streaming is **not** — the call chain is
clear but the endpoint, alternate setting, format/frame indices and ordering are
not recovered. See [the per-domain ceiling](#ranked-next-experiments-each-mutation-risk-discriminating-evidence)
at the end of this section rather than reading "traced" as "ready to run".

### Evidence inventory and identity

- Read-only tools: GNU binutils 2.42 (`nm`, `objdump`, `readelf`, `c++filt`,
  `strings`), `file` 5.45, `git` 2.43.0, `curl` 8.5.0, CPython 3.14.7. No
  `radare2`/`ghidra`/`gdb` execution; no writes to the binary or its checkout.
- Vendor binary: `thirdparty/3DCamera/linux/x64/lib3DCamera.so`, 94,574,928 bytes,
  ELF64 x86-64 shared object, SHA-256
  `49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4` (matches
  [`sdk-manifest.json`](../src/scanner3d/data/sdk-manifest.json)), GNU build-id
  `8d81f1468c0173551e7fcdce55db802e90409f97`. **Not stripped** (38,016 defined
  `.symtab` symbols; 5,133 exported) and it carries DWARF for the statically
  linked third-party C. That symbol coverage is why static tracing reaches the
  `libusb_control_transfer` call sites; the vendor C++ has full symbols but no
  line info, so evidence there is symbol + disassembly, not source line.
- Source identity: `github.com/Revopoint/3DViewer` at commit
  `516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7` (shallow clone; working-tree
  content byte-identical to tag **v1.2.17**; SDK subtree identical to v1.2.15).
  SDK version 3.2.23. Headers: `thirdparty/3DCamera/include/hpp/{Types,Camera}.hpp`.
- The binary's RUNPATH/DWARF strings name the true OEM, **Chishine3D**
  (`/home/chishine3d/zhouzhe/sdk/branch-3.2.23/trunk/...`), with an internal
  layout including `src/usb/linuxUvc/linuxCamera.cpp`, `src/rpc/*`,
  `src/upgrade/upgrade.cpp`. `FACT`.
- Reachable official sources (read-only network, curl 8.5.0): the `Revopoint`
  GitHub org has **exactly one public repo, 3DViewer** (3 tags v1.2.15/16/17,
  2 releases). `NEGATIVE FINDING`: no firmware repo, no protocol-doc repo, no
  alternate/older open transport implementation, and the shipped `document/`
  manual carries no udev/USB/UVC/XU guidance. There is no older open
  implementation to mine; the binary is the primary evidence.

### The control-transfer primitive family (`FACT`, disassembled)

Every extension-unit (XU) operation funnels through four helpers over
`libusb_control_transfer`. They branch on `CameraType` (decoded from the serial
by `cs::getCameraTypeBySN` `0x20ed1f`): serials decoding `<= 9` use a pre-UVC
vendor request; **MINI-family serials all decode `> 9`** and use standard UVC
class transfers. The MINI branch is what the repo already observed on the wire
(see [The vendor channel, byte by byte](#the-vendor-channel-byte-by-byte)); the
disassembly now recovers it and its siblings exactly.

| helper (addr) | branch | bmRequestType | bRequest | wValue | wIndex | wLength |
|---|---|---|---|---|---|---|
| `cs::libuvcSetExtension` `0x262eae` | type ≤ 9 | `0x41` | `0xFF` | `sel<<8` | `0x0300` | 60 |
| | type > 9 (MINI) | `0x21` | `0x01` SET_CUR | `sel<<8` | `0x0400` | 60 |
| `cs::libuvcGetExtension` `0x262ff7` | type ≤ 9 | `0xC1` | `0xFF` | `sel<<8` | `0x0300` | 60 |
| | type > 9 (MINI) | `0xA1` | `0x81` GET_CUR | `sel<<8` | `0x0400` | 60 |
| `cs::libuvcReadValueExtension` `0x262d92` | no branch: always UVC class | `0x21`/`0x01` then `0xA1`/`0x81` | — | **fixed `0x0E00`** | `0x0400` | 4 then 60 |
| `cs::libuvcExcuteCmd` `0x262c25` | type ≤ 9 / > 9 | `0x41`/`0xFF` or `0x21`/`0x01` | — | **fixed `0x0700`** | `0x0300`/`0x0400` | 60 |
| `cs::libuvcReadFile` `0x262861` | type ≤ 9 / > 9 | `0x41`/`0xFF`+`0xC1`/`0xFF` or `0x21`/`0x01`+`0xA1`/`0x81` | — | **fixed `0x0100`** (write) / **`0x0200`** (read) | `0x0300`/`0x0400` | 60 |

- `wIndex = 0x0400 = (entity 4 << 8) | interface 0` — the extension unit the
  device's own descriptors declare, on interface 0; identical to the repo's
  measured `/tmp/inited` exchange. `FACT`.
- **Two of these four helpers take a caller-supplied selector; two do not.**
  This distinction matters and the previous version of this section blurred it:
  - `SetExtension` / `GetExtension` compute `wValue = (selector << 8) & 0xffff`
    from their `int` parameter (`mov -0x18(%rbp),%eax; shl $0x8,%eax;
    and $0xffff,%eax` at `0x262ecf`–`0x262ed5`, and the same at
    `0x263018`–`0x26301e`). Genuinely dynamic. `FACT`.
  - `libuvcReadValueExtension` uses a **fixed** `wValue = 0x0E00`, staged
    through a stack local (`movl $0xe00,-0x4(%rbp)` at `0x262daf`, read at
    `0x262dbf` and again at `0x262e3d`). Its `int` parameter is **not** a
    selector — it is a command/property id written into the payload (Domain 4).
    `FACT`.
  - `libuvcExcuteCmd` uses a **fixed** `wValue = 0x0700` (`movl $0x700,-0x54(%rbp)`
    at `0x262c4a`, read at `0x262ce3` and `0x262d1d`), and the *real* command
    travels as ASCII text in the 60-byte payload (Domain 3). `FACT`.
- `xu_trace.py` recovers all four cases and labels them: fixed values staged
  through a stack local print as e.g. `wValue=0xe00 [stack local -0x4]`, while
  parameter-derived ones print `dynamic`. A stack local is only reported as
  constant when the whole function body writes it exactly once, with an
  immediate — so `SetExtension`'s parameter-fed `-0x4(%rbp)` correctly stays
  `dynamic` even though it is the same slot offset. Both behaviours are pinned
  by regression tests over committed disassembly excerpts.
- `libuvcReadValueExtension` reads `CameraType` only to format a log message
  (`0x262dff`, `0x262e7d`); unlike its siblings it has no `cmp $0x9`
  branch and therefore no pre-UVC vendor-request form. `FACT`.
- A **fifth** helper, `cs::libuvcReadFile` `0x262861`, is the device-side
  file-read protocol: fixed `wValue=0x0100` for the request write and `0x0200`
  for the chunked reads. It is the primitive behind the `/tmp/inited` sequence
  this repo has already driven live, and behind all calibration reads — see
  [Domain 4](#domain-4--calibration--depth-scale-fact-disassembled). `FACT`.

### Domain 1 — stream start / stop (`FACT`: standard UVC video streaming)

`cs::Camera::startStream`/`stopStream` reach `cs::UvcStream`, a vendored
**libuvc**. Bring-up:

```
cs::UvcStream::my_uvc_start_streaming  0x252d8a
  -> uvc_stream_open_ctrl  0x260b32
  -> uvc_stream_start      0x260c1e
       uvc_query_stream_ctrl  0x25fb24          # VS PROBE/COMMIT negotiation
       libusb_set_interface_alt_setting          # select streaming altsetting
       libusb_alloc_transfer
       libusb_fill_iso_transfer  0x25f809  (or  libusb_fill_bulk_transfer 0x25f793)
         callback _uvc_stream_callback  0x2608c7
       libusb_submit_transfer
```

Frames arrive over **isochronous or bulk** transfers (both fill-paths present,
chosen by endpoint type) after an alt-setting switch. `FACT`.

**Which interface this happens on.** `uvc_query_stream_ctrl` `0x25fb24` takes
its `wIndex` from the `uvc_device_info` field at offset `0x24`
(`movzbl 0x24(%rax),%eax` at `0x25fd4e` → `%esi` → `%r8d` at `0x25fd91`) — the
**VideoStreaming** interface number, not a constant and not interface 0. On this
unit's descriptor that is **interface 1 for depth and interface 3 for RGB**
(see [USB topology](#usb-topology)); interfaces 0 and 2 are
the paired VideoControl interfaces, which is where `wIndex=0x0400` XU traffic
goes. `FACT`. The alt-setting switch and the iso/bulk submit likewise act on the
VideoStreaming interface, so **nothing in Domain 1 is reachable through the
interface-0 ownership lifecycle this repo already has** — that lifecycle owns
the VideoControl interface only. `FACT`.

`UNKNOWN`, and load-bearing for any future streaming experiment: this repo has
**never recorded** the streaming endpoint addresses, endpoint transfer types,
`wMaxPacketSize`, alternate-setting numbers, or `bFormatIndex`/`bFrameIndex`
values for either VS interface. The parsed descriptor gives formats and
resolutions only. Those bytes exist in
`tests/fixtures/mini-config-descriptor.bin` but have not been extracted or
checked, and no streaming endpoint has ever been observed in a capture.

### Domain 2 — stream / mode selection (`FACT`: two asymmetric mechanisms)

- **GET side is standard UVC.** `uvc_query_stream_ctrl` `0x25fb24` is the
  textbook UVC VS_PROBE/VS_COMMIT handshake: `bmRequestType = 0x21` (SET) /
  `0xA1` (GET); `bRequest` = SET_CUR/GET_CUR; `wValue = 0x0100`
  (VS_PROBE_CONTROL) or `0x0200` (VS_COMMIT_CONTROL); `wIndex` = the
  VideoStreaming interface; `wLength = 0x22` (34) or `0x1A` (26). It packs/unpacks
  a `uvc_stream_ctrl` whose **`wFormatIndex` (offset 2)** and **`wFrameIndex`
  (offset 3)** select pixel format + resolution. `cs::UvcCamera::getStreamInfos`
  `0x24d128` enumerates the standard USB config descriptor
  (`bInterfaceClass==0x0E`/`bInterfaceSubClass==1`) and parses UVC format/frame
  descriptors. `FACT`.
- **SET side is a vendor register poke.** `cs::Camera::setStreamResolution`
  `0x1fe36a` does not use the UVC control at all: it `snprintf`s
  `echo s 0x702 %d %d > /dev/rk_preisp` (width, height) and sends it via the
  `0xffff` → `libuvcExcuteCmd` path (Domain 3). `FACT`. So resolution is
  commanded out-of-band and the resulting formats are then read back over
  standard UVC descriptors.
- `cs::Camera::isStreamSupport` `0x1f98dc` is a pure local check (stream 0 always
  supported; stream 1 keys off the cached serial/model char) — no USB. `FACT`.
- Header leak (`Types.hpp:490`, `FACT`): a comment states fringe-pattern changes
  need `stop_stream[0x483]` and `start_stream[0x481]`; the binary carries
  `echo s 0x484 %d`/`echo s 0x487 %d` register commands. These are the same ISP
  command family — the SDK drives start/stop-adjacent state through
  `/dev/rk_preisp`, not only through UVC. `FACT` (strings) / `INFERENCE`
  (exact correspondence of 0x481/0x483 vs 0x484/0x487).

### Domain 3 — projector / illumination + HID (`FACT`, fully disassembled)

The illumination setters do **not** put their `PROPERTY_EXT_*` id on the wire as
a selector. Each `snprintf`s a 60-byte ASCII string of the form
`echo s 0x<id> <args> >/dev/rk_preisp` and calls
`setPropertyExtension(0xffff, buf)`, which (case `0xffff`, `0x24dd90`) reaches
`cs::libuvcExcuteCmd` → `libusb_control_transfer(0x21, 0x01, wValue=0x0700,
wIndex=0x0400, data=<ASCII>, wLength=60)`. **The real selector is ASCII text in
the payload**, not a `wValue`. `FACT`.

The payload text is *shell-like*: `echo s 0x922 1 >/dev/rk_preisp` names a
Rockchip pre-ISP device node using `echo`/redirect syntax. What this host binary
proves is only that this ASCII string is what goes on the wire; it says nothing
about how the camera consumes it. Whether firmware literally spawns a shell,
parses the string itself and writes the register, or hands it to something else,
is **`UNKNOWN`** — the interpreter is on the far side of the USB boundary. Treat
this as an **ASCII command / register passthrough** whose text happens to be
shell-shaped; do not infer remote command execution on the camera from it.

Recovered command vocabulary (`strings`), whose ids match the
`PROPERTY_TYPE_EXTENSION` enum values:

| id | command template | domain meaning (header) |
|---|---|---|
| `0x484` / `0x487` | `echo s 0x484 %d` / `echo s 0x487 %d` | stream start/stop-adjacent |
| `0x702` | `echo s 0x702 %d %d` | set resolution (w,h) |
| `0x705` | `echo s 0x705 %d` | contrast-min / background |
| `0x910`–`0x912` | `echo s 0x91x %d` | auto-exposure family (`0x912`=AE mode) |
| `0x913` | `echo s 0x913 %d %d %d %d` | depth ROI |
| `0x914` | `echo s 0x914 %d 0 0` | HDR mode |
| `0x915`/`0x916`/`0x917` | `echo s 0x91x …` | HDR scale/exposure/max-frametime-gain |
| `0x922` | `echo s 0x922 %d` | **laser on/off** |
| `0xa04` / `0xa05` | `echo s 0xa0x %d` | multiframe fusion / **fringe pattern** |
| `0xb00` | `echo s 0xb00 %d` | **LED on/off** |

A generic `echo s 0x%x %s>/dev/rk_preisp` template exists for the rest, and a
WiFi/HTTP mirror (`/cgi-bin/zx_cmd.cgi?system_cmd=echo s 0x912 1 >/dev/rk_preisp`)
for networked models — the `system_cmd` parameter name is the vendor's, and is
suggestive but is not proof of shell execution on either transport. `FACT` (the
strings exist) / `UNKNOWN` (what the far side does with them). `UNKNOWN` also:
the firmware-side semantics/units of each `%d` (e.g. laser brightness scale) are
not in this host-side binary.

**HID escape hatch** (`PROPERTY_EXT_WRITE_READ_HID = 0x1111`, present in the
3.2.23 header union though scrubbed from the visible enum in the next release):
`cs::UvcCamera::writeReadHid` `0x24f68c` frames a packet (magic `0x5A5A5A5A`,
opcode `0x103`) and drives `UvcHid`. `UvcHid::openHid` `0x25b608` walks the
config descriptor for `bInterfaceClass==3` — a **true USB HID interface,
separate from UVC interface 0**. `UvcHid::write` `0x25b85e` prefers an
interrupt-OUT endpoint and falls back to `libusb_control_transfer(0x21, 0x09
SET_REPORT, wValue=(0x02<<8)|reportID, wIndex=<HID iface>, timeout=1000)`;
`UvcHid::read` drains an async interrupt-IN queue. `FACT`. This is a second,
independent device channel that the repo has not touched. `hidUploadFile` /
`hidExecuteCommand` exist but were not decoded — `UNKNOWN`.

### Domain 4 — calibration / depth scale (`FACT`, disassembled)

- **Intrinsics/extrinsics/distortion come over the wire, through the same
  file-read protocol the repo has already driven live.** The first version of
  this section said these were local file reads with an unknown staging channel.
  That was wrong, and following the chain to the end also explains the
  `/tmp/inited` capture from scanner-005. The corrected chain:

  1. `cs::Camera::getIntrinsics` `0x1f8484` / `getExtrinsics` `0x1f86ca` /
     `getDistort` `0x1f894e` return a cached struct on the `Camera` object if
     populated. On a miss each `snprintf`s a **lookup key**, not a filesystem
     path: `snprintf(buf, 0x400, "%s/data/camparam/Pl.bin", "995b2f683da44aae")`
     (`0x1f858e`–`0x1f85a9`; format string at `0xeaee44`, the 16-character
     sentinel `995b2f683da44aae` at `0xeaee33`). Siblings: `Prgb.bin`
     (`0xeaee5c`), `LC_RT.bin` (`0xeaee9f`), `Distort.bin` (`0xeaef19`). `FACT`.
  2. They then virtual-call slot `+0xd0` on themselves →
     `cs::Camera::getUserData(char* buf, int& size)` `0x1f8eac`. It raises
     `size` to `0x400` if smaller, requires `memcmp(buf, "995b2f683da44aae", 16)
     == 0` (`0x1f8fb8`), and builds a `std::string` from **`buf + 16`** — i.e.
     **the sentinel is stripped and the key becomes `/data/camparam/Pl.bin`**.
     An optional `'|'`-delimited second component names a local cache file; the
     calibration keys contain no `'|'`, so that branch is not taken. `FACT`.
  3. It then calls slot `+0x58` on the transport object at `this+0xb8`, up to
     three times with `ossSleep(10)` between (indirect `call *%rbx` at
     `0x1f9251`, after `mov 0xb8(%rax),%rax; add $0x58,%rax` at
     `0x1f920a`–`0x1f9214`). For a USB camera that slot is
     `cs::UvcCamera::getUserData` `0x24e372` — `vtable for cs::UvcCamera` is at
     `0x14e32d0`, so its vptr is `0x14e32e0` and slot `+0x58` is the
     `.rela.dyn` `R_X86_64_RELATIVE` entry at `0x14e3338` → `0x24e372`. That
     function forwards to `cs::libuvcReadFile2` `0x262804` →
     `cs::libuvcReadFile` `0x262861` (`0x24e408`). `FACT`.
  4. **The only file I/O in `cs::Camera::getUserData` is a write, not a read,
     and it is not on the calibration path.** After the transport call returns,
     a branch guarded by `!second.empty()` (`0x1f9316`–`0x1f9320`) opens the
     `'|'`-named cache file in mode `"wb"` (`ossOpenFile`, `0x1f933b`, mode
     string at `0xeadece`), writes the **bytes just retrieved from the
     transport** (`ossWriteFile`, `0x1f936b`) and closes it (`0x1f937a`). Since
     the calibration keys carry no `'|'`, that branch is skipped for them, and
     in every case it is a cache write-out downstream of the fetch — never a
     source of the data. `FACT`. So the earlier "local file read with unknown
     staging" reading was wrong, but the correct statement is this specific
     one, not a blanket "no file I/O on this path".

- **`cs::libuvcReadFile` is the `/tmp/inited` sequence, generalized** — and its
  shape is therefore already proven to work on this unlicensed unit. Reading
  `0x262861`–`0x262bd4` (MINI branch, `CameraType > 9`):

  | # | direction | host arguments | packed setup prefix | payload / response |
  |---|---|---|---|---|
  | 1 | OUT SET_CUR | `0x21`, `0x01`, `wValue=0x0100`, `wIndex=0x0400`, `wLength=60`, `timeout=1000` | `21 01 00 01 00 04 3c 00 e8 03 00 00` | `01` + path + NUL, zero-padded to 60 |
  | 2… | IN GET_CUR (repeated) | `0xA1`, `0x81`, `wValue=0x0200`, `wIndex=0x0400`, `wLength=60`, `timeout=1000` | `a1 81 00 02 00 04 3c 00 e8 03 00 00` | `uint32_le` chunk length at `[0:4]`, then that many bytes at `[4:]` |
  | last | OUT SET_CUR | `0x21`, `0x01`, `wValue=0x0100`, `wIndex=0x0400`, `wLength=60`, `timeout=1000` | `21 01 00 01 00 04 3c 00 e8 03 00 00` | 60 zero bytes |

  The buffer is built as `buf[0] = 0x01` (`movb $0x1,-0x70(%rbp)` at `0x2628d2`)
  followed by `strcpy(buf + 1, path)` (`0x2628eb`). The read loop
  (`0x2629e8`–`0x262b2f`) re-issues the same GET_CUR, `memcpy`s
  `buf[4 : 4+len]` onto the output at the running offset, and **stops when the
  length word is zero**; it also stops when the caller's capacity is reached, and
  logs a warning if a chunk claims more than `0x38` = **56** bytes (the maximum
  that fits after the 4-byte header). The trailing 60-zero write is the vendor's
  own cleanup, issued unconditionally with its return value ignored
  (`0x262b38`–`0x262bd0`). `FACT`.

- **This resolves three questions scanner-005 left open**, without any new
  transfer:
  - The leading `01` byte is the **read-file opcode**; the rest of the payload is
    the NUL-terminated path. `FACT`.
  - The `0c 00 00 00` + `"v7.5.7.0426"` reply was not a stale buffer: it is a
    **length-prefixed 12-byte chunk** in exactly this format, and the all-zero
    second reply is the **zero-length end-of-data terminator**, not a failure
    code. `FACT` (the decode matches byte for byte). What remains `UNKNOWN` is
    only why the device answers a `/tmp/inited` read with the algorithm version
    string — a device-side question.
  - The fourth transfer the repo implemented as protocol cleanup **is** the
    vendor's own cleanup write, same selector and same all-zero 60-byte payload.
    `FACT`.
- **Depth scale is read over the XU channel, and `0x918` is a payload value,
  not a selector.** This was stated wrongly in the first version of this
  section; the corrected chain, disassembled end to end:

  1. `cs::Camera::getDepthScale(float&)` `0x1ffdee` returns the cache at
     `this+0x264` if non-zero (`0x1ffe14`–`0x1ffe30`). On a miss it zeroes a
     **60-byte** stack buffer at `-0x60(%rbp)` (seven `movq $0x0` plus one
     `movl $0x0`, `0x1ffe36`–`0x1ffe6e`), takes an `OssAutoCritical` lock, and
     virtual-calls slot `+0x60` of the device object at `this+0xb8` with
     `mov $0x918,%esi` / `lea -0x60(%rbp),%rdx` / `lea -0x70(%rbp),%rcx`
     (`0x1ffee4`–`0x1ffef1`). `FACT`.
  2. `cs::UvcCamera::readBuffer(int cmd, char* buf, int* len)` `0x24e4c4` is
     that slot's implementation. It passes `CameraType` from `this+0xa8` and the
     `libusb_device_handle*` from `this+0xe8`, then forwards **`cmd` as the third
     argument and `buf` as the fourth** to `cs::libuvcReadValueExtension`
     (`0x24e4f9`). `FACT`.
  3. `cs::libuvcReadValueExtension` `0x262d92` fixes `wValue = 0x0E00`
     (`movl $0xe00,-0x4(%rbp)` at `0x262daf`) and writes the caller's `cmd`
     into the **first four bytes of the caller's buffer**
     (`mov -0x18(%rbp),%edx; mov -0x28(%rbp),%rax; mov %edx,(%rax)` at
     `0x262db6`–`0x262dbd`). It then issues two transfers on the same fixed
     selector. `FACT`.

  | # | direction | host arguments | packed setup prefix (LE, as this repo prints usbfs) | payload |
  |---|---|---|---|---|
  | 1 | OUT (SET_CUR) | `bmRequestType=0x21`, `bRequest=0x01`, `wValue=0x0E00`, `wIndex=0x0400`, `wLength=4`, `timeout=1000` | `21 01 00 0e 00 04 04 00 e8 03 00 00` | `18 09 00 00` (= `0x918` little-endian) |
  | 2 | IN (GET_CUR) | `bmRequestType=0xA1`, `bRequest=0x81`, `wValue=0x0E00`, `wIndex=0x0400`, `wLength=60`, `timeout=1000` | `a1 81 00 0e 00 04 3c 00 e8 03 00 00` | — (60-byte response) |

  Between the two, `memset(buf, 0, 60)` at `0x262e38` clears the **host** buffer.
  That is a local buffer clear, not a protocol operation: the function issues
  **exactly these two transfers and no third**, and in particular sends no
  clearing/cleanup write of its own. `FACT` — do not model a clear from the
  `/tmp/inited` sequence onto this path.

  On return, `readBuffer` sets `*len = (int8_t)buf[59]` (`add $0x3b,%rax;
  movzbl (%rax),%eax; movsbl %al,%edx` at `0x24e501`–`0x24e513`). `getDepthScale`
  reads the same byte directly (`movzbl -0x25(%rbp)` = `-0x60 + 0x3b`, i.e.
  offset 59) and requires `== 4` (`0x1fff37`–`0x1fff3d`). On match it computes
  `depthScale = 1.0f / (float)(int32_le)buf[0:4]` (`cvtsi2ss` then
  `divss` against the constant `1.0f` at `0xeb15b4`) and caches it at
  `this+0x264`. On mismatch it falls back by camera type
  (`cs::getCameraTypeBySN`): type `0xC` → `0.005f` (`0xeb15dc`), otherwise
  `0.1f` (`0xeb15e0`). `FACT`.

  So `0x918` is a **command/property id carried in the four-byte payload**, and
  `0x0E00` is the fixed XU selector word for this whole read family. It is not
  the header's `PROPERTY_EXT_DEPTH_SCALE = 0x0`, and it is not a `wValue`.

- **Confirmed on hardware (measured, one run, 2026-08-21T23:44 local).** The
  two-transfer `libuvcReadValueExtension` pair above is no longer only a
  disassembly reading: `scanner3d depth-scale --execute-depth-scale-read` sent
  it, once, against the same unlicensed unit already exercised elsewhere in
  this document (`A22704522A6F00A64`, bus 1 device 41, `1-1.1`, `2207:110c`),
  under an external 30 s TERM / +10 s KILL deadline, and exited 0. `FACT`
  (measured):

  | # | direction | packed setup prefix | result |
  |---|---|---|---|
  | 1 | OUT (SET_CUR) | `21 01 00 0e 00 04 04 00 e8 03 00 00` | all 4 bytes accepted in 1.560 ms; sent hex `18090000` |
  | 2 | IN (GET_CUR) | `a1 81 00 0e 00 04 3c 00 e8 03 00 00` | 60 bytes returned in 17.715 ms |

  Exactly these two transfers ran, in this order, with no retry and no third
  transfer, confirming the disassembly's reading that this primitive issues no
  protocol-clear write of its own — the live run's `protocol_cleanup` reported
  "not applicable" rather than sending anything, matching the static finding
  that the `memset` at `0x262e38` clears only the host buffer. Read duration
  19.426 ms; total owned-window duration 27.880 ms. `FACT` (measured).

  The raw 60-byte reply — SHA-256
  `9873fa193adabb7daa87a77718d963d586d798838f4e6edc1a2ddac0bb337dd4`, hex
  `c80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004`
  — decodes exactly as the disassembly predicts: byte 59 (the marker) is `4`
  as both uint8 and int8, and bytes `0:4` read as a signed little-endian
  `int32` are `200` (`0xc8`). Both gates `getDepthScale` applies before it
  trusts the value pass. `FACT` (measured).

  `vendor_depth_scale = float32(1.0f / 200) = 0.004999999888241291` (float32
  bits `0x3ba3d70a`). `FACT` (measured) for this value and its bits.
  **The scalar's physical unit is unknown from this evidence** — two
  transfers and one reciprocal prove what the device returned, not what it
  means, and nothing here establishes millimetres, metres, accuracy, or a
  raw-depth conversion factor.

  **Observation:** this float32 bit pattern is identical to the SDK's
  compiled-in MINI fallback constant `0.005f` (`0x3ba3d70a`, the same
  `SDK_MINI_FALLBACK_SCALE` cited above) — ratio 1.0 to the bit. `FACT`
  (measured value compared against the already-cited static constant).
  `INFERENCE`, explicitly not established: that the SDK's authors chose
  `0.005f` as the MINI fallback *because* it matches what this class of
  device reports over this channel. The agreement is shown; it is not itself
  evidence of a unit, an accuracy, or a causal reason for the constant.

  Device state was unaffected beyond the expected re-probe: identical serial
  and devnum (41, no re-enumeration), identical USB descriptor SHA-256
  (`2fa267c0c41d4cc53390bb7d1562d4b54fb44e8be04e0d568de8633b5d626613`) before
  and after, interfaces 0–3 back on `uvcvideo` and interface 4 on `usbhid`
  both before and after, `/dev/video0..3`, `/dev/hidraw10` and
  `/dev/bus/usb/001/041` all present afterward, and no holders or surviving
  process. The kernel log gained only the expected `uvcvideo` re-probe line
  (`usb 1-1.1: Found UVC 1.00 device ZXProduct (2207:110c)`, plus unrelated
  bus-2 hub events). Driver handback (release, then reconnect) ran and
  reported restored. `FACT` (measured). This live run does not touch any
  other `0x9xx` id: only `0x918` was read, once.

- **`0x0E00` is a generic read-value-by-id channel, not a depth-scale-specific
  control.** Five `cs::Camera` methods reach the same device vtable slot
  (`this+0xb8` → `+0x60`, i.e. `readBuffer`), each with a different id in
  `%esi`, and every one of them therefore produces the same `wValue=0x0E00`
  four-byte-out / 60-byte-in pair with only the payload word differing. `FACT`:

  | id | caller | call site |
  |---|---|---|
  | `0x910` | `cs::Camera::getProperty(STREAM_TYPE, PROPERTY_TYPE, float&)` | `0x1f666e` |
  | `0x913` | `cs::Camera::getDepDataROI` | `0x2003b8` |
  | `0x916` | `cs::Camera::getHDRParamer` | `0x20021a` |
  | `0x918` | `cs::Camera::getDepthScale` | `0x1ffef1` |
  | *runtime* | `cs::Camera::getDepRangeHDRModeScaleSettingContrastMinAutoExposureMode(PROPERTY_TYPE_EXTENSION, …)` | `0x200107` (id from `-0x3c(%rbp)`, a parameter) |

  These ids are the same numeric family as the `echo s 0x9xx …` write commands
  in Domain 3 — `0x913` depth ROI, `0x916` HDR exposure — which reads as one id
  space with an XU read path and an ASCII write path. `INFERENCE` (the numbers
  match and the header groups them together; no code maps one to the other).
  `UNKNOWN`: the response layout for ids other than `0x918`, and whether byte 59
  always carries a payload length (it is returned as `*len` for every id, but
  only `getDepthScale`'s `== 4` check is evidenced here).
- Response struct shapes (`Types.hpp`): `Intrinsics` = `short width,height;
  float fx,_,cx,_,fy,cy,_,_,_` (3×3 pinhole, 40 B); `Extrinsics` =
  `float rotation[9]; float translation[3]` (48 B); `Distort` = `float k1..k5`.

### The licence gate, relative to the four domains

- `cs::LicenseAuthorizer::authorize(CameraInfo)` `0x2c583e` builds a
  `LicenseParser` and calls `LicenseParser::parse(sAllLicenseFilePaths,
  ICamera*, LicenseData&)` then `checkLicenseData`; `sAllLicenseFilePaths`
  (data symbol `0x158adf0`) resolves to the strings **`/data/sdk-license`** and
  **`/persist/sdk-license`**. Because `parse` takes an `ICamera*`, the licence
  blob is read **through the device**. `FACT`.
- A direct-call-site search finds no `call <addr>` to any `::authorize`. That is
  **not** dead code: each `authorize` address (`AuthorizerManager` `0x2c384c`,
  `LicenseAuthorizer` `0x2c583e`, `InnerAuthorizer` `0x2c509e`) appears as a
  pointer inside `.data.rel.ro` **vtable slots** with matching `.rela.dyn`
  relocations, i.e. `authorize` is **virtual** and dispatched via `call *reg`,
  which a direct-xref grep cannot see. `FACT` (relocation/vtable evidence).
  The repo's **measured** result — the vendor SDK's `connect()` returns
  `ERROR_NOT_SUPPORT`, its log naming the licence authorizer's failure to find
  an `sdk-license` blob — confirms the authorizer *does* run at connect time.
  `FACT` (prior milestones). The exact indirect call site on the connect path
  was not pinned by static xref (`UNKNOWN`).
- **The sampled domain operations contain no inline licence check.** The
  lower-level functions read for this pass (`libuvcExcuteCmd`,
  `libuvcSetExtension`/`libuvcGetExtension`, `libuvcReadValueExtension`,
  `uvc_query_stream_ctrl`, `UvcHid::write`, the `.bin` readers) call no
  authorizer and consult no licence state; they issue their transfer
  unconditionally. `FACT`, but note this is **absence of evidence in the
  functions sampled**, not a proof that no check exists anywhere on those paths.
- **What the device has actually been shown to accept is narrow.** Owned code
  has exercised exactly **eleven** control transfers against this unlicensed
  unit: one `GET_CUR` on entity 3 (`a1 81 wValue=0x0600 wIndex=0x0300 wLength=60`),
  the four-transfer `/tmp/inited` sequence on entity 4
  (`21 01 wValue=0x0100`, `a1 81 wValue=0x0200` twice, `21 01 wValue=0x0100`
  clearing write, all `wIndex=0x0400 wLength=60`), the four-transfer
  `/data/camparam/Pl.bin` read on entity 4 with the identical setup packets and
  a different 60-byte key payload, and the two-transfer `0x918` depth-scale
  property read on entity 4 with `wValue=0x0E00` (`21 01 wLength=4` carrying
  the property id, then `a1 81 wLength=60`). All eleven succeeded. `FACT`
  (measured). Separately, the vendor SDK's own pre-authorization traffic
  succeeded up to the point it gave up. `FACT` (measured).
- **The supported conclusion, and the part that is only a hypothesis.**
  `SUPPORTED`: authorization is a **host-library precondition inside
  `lib3DCamera.so` at connect time** — the SDK's own trace shows it fail after a
  licence read returns nothing, with no device refusal anywhere in the exchange,
  and the eleven owned transfers went through with no licence present at all,
  including a factory calibration key the SDK would only read after
  authorization. So the gate is not a per-transfer device lock *on the transfers
  that have been tried*.
  `HYPOTHESIS, NOT ESTABLISHED`: that the device therefore accepts UVC
  streaming, HID traffic, the other XU selector `0x0700`, any `0x0E00` property
  id other than `0x918`, or any Domain 1–4 operation from an unlicensed host.
  Nothing has tested those. The device could gate streaming or a specific
  selector independently of the library check, and static absence of a
  host-side check cannot rule that out. Experiments 2, 3 and 4a below each
  test one slice of this hypothesis; experiments 4b, 5 and 6 are marked
  `NOT READY` and test nothing until their prerequisites are met. Experiment 2
  widened the tested slice to a factory calibration key, and the depth-scale
  read has since widened it to one `0x0E00` property; "the device does not
  enforce it" is still a claim about eleven transfers on three selectors of two
  entities, not about the device.

### What static evidence cannot establish

- **Firmware-side semantics.** The ISP register ids (`0x922` laser, `0xb00` LED,
  `0x914` HDR, `0x484/0x487` stream, …) are interpreted inside the camera at
  `/dev/rk_preisp`; their argument units, valid ranges, ordering constraints and
  side-effects are not in this host binary. `UNKNOWN`.
- **The exact stream bring-up ordering on the MINI.** Whether depth streaming
  needs specific `echo s …` register writes *before* UVC COMMIT (and in what
  order) is not statically fixed; the SDK interleaves them at a layer above
  `uvc_stream_start`. `UNKNOWN` without a capture.
- **Calibration provenance on the device side.** The transport is now known
  (`libuvcReadFile` over the XU channel, keyed by `/data/camparam/*.bin`), and
  experiment 2 has since established by measurement that this device serves
  `/data/camparam/Pl.bin` to a host that never authorized. What produces those
  blobs inside the camera and whether they are factory-fixed are still
  `UNKNOWN`, as is whether any key other than `Pl.bin` is served — the siblings
  were never requested.
- **Device acceptance beyond eleven transfers.** Every claim in this section
  about what the device *will* answer rests on the eleven control transfers
  owned code has actually issued. Everything else — streaming, HID, selector
  `0x0700`, any `0x0E00` property id other than `0x918`, any path other than
  `/tmp/inited` and `/data/camparam/Pl.bin` — is untested. `UNKNOWN`.
- **Selector→property table for `getPropertyExtension`.** Scanner-028 completed
  the relevant RGB portion offline: property IDs `0x15` and `0x16` both use
  selector `0x12` through the direct UvcCamera override; the exact callsites,
  lengths, wrapper behavior and decode offsets are recorded below. Other
  extension properties remain outside that bounded result.
- No new on-wire bytes were produced: this task ran no scanner, SDK, probe or
  capture. The claims are what the code *would* do, not a fresh observation.

### Ranked next experiments (each: mutation, risk, discriminating evidence)

None of these requires a licence bypass. Ranked by evidence-per-risk.
Entries marked `NOT READY` are proposals only: each lists the exact
prerequisites it is missing, and none of them may be run until those are
met.

1. **Capture the SDK's own stream bring-up on a licensed unit (or via the
   existing tracer) — read-only.** Mutation: none beyond running the vendor SDK
   under the `LD_PRELOAD` usbfs tracer already in `reference/tracer/`. Risk:
   lowest (observation only; the SDK does the driving). Discriminates: the exact
   interleave of `echo s …>/dev/rk_preisp` register writes, UVC PROBE/COMMIT
   `(wFormatIndex,wFrameIndex)`, alt-setting and the first iso/bulk submit — the
   one thing static analysis cannot order. Blocked on this unit by the licence
   failure at connect; unblocked on any unit that authorizes, or by capturing a
   sibling model. This is the narrowest action that most reduces M2/M3
   uncertainty.
2. **Owned calibration read over `cs::libuvcReadFile` — `EXECUTED` on
   2026-08-21; see "Owned code reads the factory depth intrinsics (measured)"
   above for the run, the bytes and the record it returned. The design below is
   what was implemented, and is kept as the record of the bounds chosen *before*
   the run rather than after it.** Mutation:
   on **VideoControl interface 0**, already proven ownable, the exact sequence
   Domain 4 tabulates, with the path string `/data/camparam/Pl.bin` instead of
   `/tmp/inited`: SET_CUR `21 01 00 01 00 04 3c 00` carrying
   `01` + `"/data/camparam/Pl.bin"` + NUL zero-padded to 60; then repeated
   GET_CUR `a1 81 00 02 00 04 3c 00`; then the SET_CUR cleanup write of 60
   zeros. Every setup byte, the payload, the direction, the order, the
   termination condition and the cleanup are pinned by disassembly **and** by
   the repo's own live capture of the identical shape — only the path string
   differs from what has already run successfully on this unit.

   **Numeric bounds — chosen here, fixed before the run, never widened at
   runtime.** The vendor loop stops on a zero length word and on the caller's
   capacity, but does not bound its own iteration count, so an owned
   implementation must add both bounds itself:

   - allocate an output capacity of **256 bytes** — 5× the largest calibration
     struct in `Types.hpp` (`Extrinsics` 48 B; `Intrinsics` 40 B; `Distort`
     20 B) — and never write past it;
   - issue at most **8 GET_CUR transfers**, so the run is at most **10 control
     transfers** in total (1 SET_CUR + ≤8 GET_CUR + 1 SET_CUR cleanup), claims
     at most **448 bytes** of chunk payload (8 × 56) and copies at most **256**;
   - treat any chunk whose `uint32_le` length word is **`> 56`** (`0x38`, the
     maximum that fits after the 4-byte header — the value the vendor only
     warns about) as malformed: copy nothing and stop the loop immediately;
   - stop **before** any `memcpy` that would pass the 256-byte capacity, rather
     than truncating into it;
   - and, on **every** exit path, issue the single 60-zero SET_CUR cleanup
     write exactly once, never retried, at the same point the vendor issues it
     (`0x262b38`–`0x262bd0`). Unlike the vendor, which discards that status, an
     owned run must **act** on it, exactly as the accepted scanner-005
     lifecycle already does: a failed or ambiguous clear makes the whole run
     non-successful — nonzero exit, explicit recovery guidance — even when
     every read before it succeeded, and the driver handback
     (`USBDEVFS_RELEASEINTERFACE`, `USBDEVFS_CONNECT`) still runs afterwards
     regardless. A run whose protocol cleanup did not demonstrably succeed is
     never reported as a successful calibration read, however complete the
     bytes it collected look; those bytes are reported as unverified output of
     a failed run.

   **Outcome taxonomy — these are distinct results, not one undifferentiated
   "failure".**

   | outcome | condition | reading |
   |---|---|---|
   | `complete` | a zero length word arrived within the 8-GET cap | full blob; the terminator was actually seen |
   | `cap_exhausted` | 8 GET_CURs consumed, no zero terminator | **bounded incomplete** — the cap, not the device, ended the read |
   | `capacity_exhausted` | the next chunk would pass 256 bytes | **bounded incomplete** — the blob exceeds the chosen capacity |
   | `malformed_length` | a length word `> 56` | protocol mismatch, or the device is answering something else; nothing copied |
   | `transfer_error` | any SET_CUR/GET_CUR fails or stalls | the device refused this path or selector — also a clean answer |

   These five classify the **read phase** only. The cleanup result is a separate
   axis that **overrides** them: a `complete` read whose 60-zero clear failed or
   was ambiguous is a failed run reporting recovery guidance, not a successful
   calibration read.

   A `cap_exhausted` or `capacity_exhausted` run reports a bounded, explicitly
   incomplete result and stops. It must **never** silently raise its own cap or
   capacity and continue: widening either number is a new proposal, not a
   runtime decision. Risk: low, and lower than experiment 3 below, because
   nothing about the transfer shape is new; the single new variable is the path
   string. Discriminates: whether the device serves `/data/camparam/Pl.bin` to
   an unlicensed host. A success yields the **actual depth intrinsics** — the
   first real calibration data obtained from this unit, and a direct M3 unblock.
   *(Run once on 2026-08-21 under `scanner-007`: `complete`, a 40-byte
   `Intrinsics` record. The paragraphs below are the pre-run proposal, kept as
   written; the measured result is above.)*
   A zero-length first reply is itself a clean answer, but an **ambiguous** one:
   the same empty response is what an absent key, a refused key, and a
   genuinely empty file would all produce, and nothing in the recovered
   protocol distinguishes them — the vendor loop treats any zero length word as
   normal termination and carries no status code. Do not report it as "absent
   or refused"; report it as "no content returned, cause not distinguishable",
   and disambiguate it, if at all, by contrast against a second key.
   `CAVEAT`: the `Intrinsics` struct is 40 bytes and the vendor asks for a
   1024-byte capacity, so expect one or two chunks; do not assume a single
   response. Reading a path is a read, but it is still a path this repo has
   never sent — keep it to the calibration keys and never a licence path.
3. **Owned XU depth-scale read (`libuvcReadValueExtension` shape) —
   `EXECUTED` on 2026-08-21; see the "Confirmed on hardware" entry in Domain 4
   above for the run, the raw reply and the decoded scale. The design below is
   what was implemented, and is kept as the record of the bounds chosen
   *before* the run rather than after it.** This was ranked above any
   streaming attempt because it was the only remaining experiment whose
   on-wire shape was fully bounded by the evidence above. Mutation: on the
   **VideoControl interface 0** already proven ownable, a SET_CUR
   (`21 01 00 0e 00 04 04 00`, timeout 1000 ms) carrying the four payload bytes
   `18 09 00 00`, immediately followed by a GET_CUR
   (`a1 81 00 0e 00 04 3c 00`, timeout 1000 ms) reading 60 bytes — exactly two
   transfers, in that order, with **no cleanup write**, because the vendor
   function issues none (`0x262d92`–`0x262e6a`). Setup, payload, direction,
   order and cleanup are all pinned by disassembly, so this one is
   implementable as a new closed whitelist on the existing lifecycle. Risk: low
   — same interface, same entity `0x0400`, same `SET_CUR`/`GET_CUR` primitives
   the repo has already driven on this unit; the only novelty is selector
   `0x0E00` and a 4-byte rather than 60-byte write. Discriminates: whether the
   device answers selector `0x0E00` at all without a licence, and whether
   `buf[59] == 4` with a plausible `1/int32_le(buf[0:4])` — the first
   calibration value ever read by owned code, and a go/no-go for the whole
   read-by-id space (`0x910`/`0x913`/`0x916`/`0x918`).
   `CAVEAT`: a 4-byte write to an XU control the device declares as 60 bytes
   may be rejected with a stall; that is itself the discriminating evidence and
   is recoverable, but it means "SET_CUR succeeds" must not be assumed.
   *(Run once on 2026-08-21 under `scanner-008`: the SET_CUR was accepted in
   full, the GET_CUR returned an exact 60-byte reply with `buf[59] == 4` and
   `int32_le(buf[0:4]) == 200`, giving `float32(1.0f / 200) = 0.004999999888241291`
   (bits `0x3ba3d70a`) — bit-identical to the SDK's `0.005f` MINI fallback, an
   observation and not an established unit. The paragraph above is the pre-run
   proposal, kept as written; the measured result is in Domain 4.)*
4. **Streaming — 4a run once; 4b NOT ready to run, still blocked on descriptor
   facts this repo has not yet extracted.** The earlier version of this section
   said interface 0's lifecycle could be reused for UVC PROBE/COMMIT. That was
   wrong: PROBE/COMMIT `wIndex` names the **VideoStreaming** interface (1 for
   depth, 3 for RGB), and iso/bulk transfer needs ownership *and* an alternate
   setting on that same VS interface. Two separable paths:
   - **4a — kernel/V4L2, no ownership change.** Let `uvcvideo` keep both
     interfaces and drive the existing `/dev/video0` through ordinary V4L2
     (`VIDIOC_S_FMT`, `VIDIOC_REQBUFS`, `VIDIOC_STREAMON`, one buffer, then
     `VIDIOC_STREAMOFF`). Mutation: none at the USB-ownership layer; the driver
     performs PROBE/COMMIT and alt-setting itself. Risk: lowest of the two —
     no detach, no claim, fully reversible by `STREAMOFF` + close.
     Discriminates: whether the depth function yields **any** frame without a
     prior `echo s …` register write. A clean negative (`STREAMON` succeeds,
     zero buffers dequeued before timeout) localizes the missing step to the
     ISP register channel rather than to negotiation. Prerequisite: none beyond
     picking a format/resolution already parsed from the descriptor. **This is
     the correct first streaming attempt** and it needs no new transport code.
     *(Run once on 2026-08-22 under `scanner-009`: `Y16` 640×400 negotiated,
     `STREAMON` succeeded, and buffer 0 was dequeued after a single `poll()` —
     the **positive** outcome, not the clean negative described above, for this
     one run. The discriminator is answered for this run only; it does not by
     itself explain the earlier zero-buffer GStreamer result at a different
     resolution/format, and it says nothing about repeatability, RGB, or
     whether the ISP register channel is needed under other conditions. The
     paragraph above is the pre-run proposal, kept as written; the measured
     result is in
     [Owned code captures a raw depth frame](#owned-code-captures-a-raw-depth-frame-measured).)*
   - **4b — direct usbfs/libusb streaming.** 4a's one-run result does not make
     this unnecessary: it is one V4L2 frame through the kernel driver, not a
     multi-frame or driver-independent transport, and neither RGB nor
     repeatability was tested. Mutation: a *new* bounded ownership lifecycle
     for the VS interface, plus
     the coupling that detaching interface 0 also tears down interface 1 (the
     repo measured exactly that: `/dev/video0`/`video1` were destroyed and
     recreated, `/dev/video2`/`video3` untouched), so the depth **function** —
     both interfaces — must be owned and restored together. Risk: highest of
     the set: it takes the device away from the kernel driver, submits
     isochronous or bulk URBs, and must restore the alternate setting to 0 and
     reattach both interfaces on every failure path.
     `NOT READY`: endpoint address, endpoint transfer type, `wMaxPacketSize`,
     the alternate-setting number, `bFormatIndex`/`bFrameIndex` and the frame
     interval are **all unrecorded** for this device, and a bounded stop plus
     restoration has not been designed. Until those six values are extracted
     from `tests/fixtures/mini-config-descriptor.bin` — an offline, zero-risk
     parsing task — and a stop/restore path is written, this experiment cannot
     be called ready and must not be run.
5. **Owned single ISP register command (e.g. LED `echo s 0xb00 0`) — NOT
   READY.** Mutation: one `libuvcExcuteCmd`-shaped SET_CUR (`0x21/0x01`,
   `wValue=0x0700`, `wIndex=0x0400`, 60-byte ASCII payload) carrying one
   register id. Discriminates: whether the passthrough accepts owned `echo s`
   commands at all, decoupled from streaming. Risk: this **sets device state**.
   An earlier version of this entry called an LED write benign and
   self-reverting and borrowed the `/tmp/inited` cleanup as an analogy; both
   were wrong. `/tmp/inited` was a *read* whose trailing 60-zero write is the
   vendor's own cleanup of the **transfer channel** — it restores nothing about
   the device's LED. A state-setting command has no self-reverting property, so
   issuing one can leave the unit in a changed state this repo cannot currently
   undo.
   `NOT READY` until all of the following exist, none of which does today:
   - a **read of the current value** of the chosen id, so the pre-experiment
     state is known rather than assumed. The `0x0E00` read-by-id channel exists
     (Domain 4), but no evidence shows that `0xb00` is readable through it or
     what its response layout would be — `UNKNOWN`;
   - the **exact restore command** — the literal 60-byte ASCII payload that
     writes the captured value back — plus evidence that re-writing that value
     is accepted and actually restores the state;
   - the firmware-side **argument units and valid range** for the id, currently
     `UNKNOWN` (see "What static evidence cannot establish");
   - a restore path that runs on **every** exit, including transfer failure and
     process abort, and a stated, reported outcome if the restore itself fails.
   Without a proven read-current-state plus an exact restore, arbitrary
   state-setting writes must not run. The sole narrow exception is the
   scanner-026 final-state guard: after depth `STREAMON`, one fixed
   `echo s 0x922 0` off write is attempted only after V4L2 teardown and only
   through the shared interface-0 restoration lifecycle. It has no optical
   readback and never exposes a general laser or brightness control.
   Its typed write outcomes are `short_write` and `overlong_write` for
   non-exact host transfers; required success structurally needs `COMPLETE`
   plus an exact successful 60-byte record and clean restoration. Only the
   inert `NOT_REQUIRED`/not-attempted shape can succeed without an attempt.
6. **Probe the HID interface (`bInterfaceClass==3`) read-only — NOT READY.**
   The original rationale — that HID might be the staging path for the
   calibration `.bin` files — is **resolved and withdrawn**: Domain 4 establishes
   that the USB calibration getter fetches those blobs over the XU file-read path
   (`cs::UvcCamera::getUserData` `0x24e372` → `cs::libuvcReadFile2` `0x262804` →
   `cs::libuvcReadFile` `0x262861`), not over HID. What is left is only the
   separate, weaker question of what the HID channel carries at all.
   `NOT READY`: the frame layout, endpoints and response handling are not
   decoded. Before any live HID proposal this repo needs:
   - the **exact framed query** — the full byte layout of the
     `0x5A5A5A5A`/`0x103` frame including length, sequence/checksum fields and
     padding, taken from disassembly rather than from `strings`;
   - the **endpoint and fallback selection** — interrupt-OUT/IN endpoint
     addresses and `wMaxPacketSize` parsed from the descriptor fixture, plus the
     exact condition under which the SDK falls back to SET_REPORT/GET_REPORT;
   - **timeouts and a bounded read** — per-transfer timeout, a maximum number of
     IN transfers, a maximum accepted response size and the termination rule, so
     that a silent or chatty device cannot loop the host;
   - **response bounds and a decode** — what a valid reply looks like and what
     distinguishes it from no reply or from an error frame;
   - **interface-4 driver restoration** — interface 4 is bound to `usbhid`; the
     proposal must state the detach/claim/release/reattach sequence and its
     behaviour on every failure path, since this repo has never taken that
     interface.
   Risk once specified: moderate (a distinct, untouched interface). Until those
   five items are pinned, it must not run.

**Narrowest next action.** Two, in order:

- **Offline, zero-risk, and a hard precondition:** parse the six missing
  streaming-descriptor values out of the fixture already in this repository
  (`tests/fixtures/mini-config-descriptor.bin`, SHA-256 `2fa267c0…6613`) —
  endpoint addresses and types, `wMaxPacketSize`, alternate-setting numbers,
  `bFormatIndex`/`bFrameIndex` pairs and frame intervals per VS interface. Until
  these exist, experiment 4b cannot even be specified.
- **The narrowest action that puts new bytes on the wire:** experiment 2, the
  calibration read over `cs::libuvcReadFile`, run under the fixed bounds stated
  there — 256-byte capacity, at most 8 GET_CURs (≤10 transfers total), stop on a
  length word `> 56`, stop before overrunning capacity, one 60-zero cleanup write
  on every exit, and a bounded-incomplete report rather than any runtime widening
  of the cap. It reuses a transfer shape this repo has already executed
  successfully against this exact unlicensed unit, changing only the path string,
  and a success returns real depth intrinsics. That is the single highest
  evidence-per-risk step available and it is the one that most reduces M3
  uncertainty.

For M2 (streaming) specifically, experiment 4a — driving the existing
`/dev/video0` through ordinary V4L2 with no ownership change — was the
narrowest attempt, needed no new transport code, and has now been **run once**
under `scanner-009` with a positive result (one dequeued depth frame); see
[Owned code captures a raw depth frame](#owned-code-captures-a-raw-depth-frame-measured).

Note what is *not* on this list: experiments 4b (direct usbfs streaming), 5 (ISP
register write) and 6 (HID probe) are all `NOT READY`. 5 and 6 are additionally
**state-touching or interface-taking**, so neither can be the narrowest action
under any reading; each is gated on the explicit prerequisites listed with it.

**Honest ceiling, stated by domain rather than in aggregate.** The previous
summary claimed all four domains were traced to exact request shapes. That is
too broad. Precisely:

| domain | exact today | not established |
|---|---|---|
| 1 — stream start/stop | the libuvc call chain, the VS PROBE/COMMIT primitive, that `wIndex` is the VS interface number, that both iso and bulk fill-paths exist | the concrete PROBE/COMMIT payload, which endpoint/alt-setting/interval, and any ordering against register writes |
| 2 — mode selection | the GET side (descriptor enumeration; `wFormatIndex`/`wFrameIndex` are the selectors) and that the SET side is an `echo s 0x702 w h` register write, not a UVC control | the actual index values this device expects, and whether the UVC SET path is usable at all |
| 3 — projector/illumination | the exact `libuvcExcuteCmd` tuple (`0x21/0x01`, fixed `wValue=0x0700`, `wIndex=0x0400`, 60-byte ASCII payload) and the id vocabulary from `strings` | firmware-side semantics, argument units and ranges of every id; whether the device accepts these from a non-SDK host |
| 4 — calibration/depth scale | the full depth-scale pair (both setup packets, the `18 09 00 00` payload, the `buf[59]==4` / `1.0f/int32_le` decode); that the same `0x0E00` channel carries at least four other ids; the complete `libuvcReadFile` calibration protocol (opcode `01`, key `/data/camparam/Pl.bin`, chunked `0x0200` reads with a `uint32_le` length prefix, zero-length terminator, 60-zero cleanup); and — **measured, not static** — that this device serves `/data/camparam/Pl.bin` to an unlicensed host as a 40-byte `Intrinsics` record, and separately answers the `0x918` depth-scale property with an exact 60-byte reply (marker `4`, denominator `200`, `float32(1.0f/200) = 0.004999999888241291`, bits `0x3ba3d70a`) | response layouts for the other read-by-id ids (only `0x918` was read); whether the *other* calibration keys (`Prgb`, `LC_RT`, `Distort`) are served, which was not tested; the on-device provenance of the `.bin` contents; the metric accuracy, handedness and frame convention of the intrinsics that were read; and the physical unit of the depth-scale value, which this evidence does not establish |

What static analysis genuinely cannot reach is *ordering* and *firmware
preconditions*: they live above `uvc_stream_start` and inside the camera. Those
need a capture of a successful bring-up (experiment 1) or owned probing — and
only by the experiments that are actually ready, namely **2 (calibration read),
3 (XU depth scale) and 4a (bounded kernel/V4L2 stream attempt)** — of which
**2, 3 and 4a have now been run** and answered their questions (the device
serves `/data/camparam/Pl.bin` unlicensed and returned a 40-byte `Intrinsics`
record; the sibling keys remain untested — the device answers the `0x918`
depth-scale property unlicensed with an exact 60-byte reply, `float32(1.0f/200)
= 0.004999999888241291`, whose physical unit remains unestablished — and,
under 4a, plain V4L2 against `/dev/video0` with no ownership change delivered
one complete raw `Y16` 640×400 depth frame in one live run, the positive
outcome for the "does the depth function yield any frame without a prior
`echo s …` register write" discriminator, for that one run only).
Experiments 4b, 5 and 6 are `NOT READY` and cannot contribute until their listed
prerequisites are resolved. Either way, not more reading.

**Updated by `scanner-012`.** Experiment 4b's stated blocker — "endpoint
address, endpoint transfer type, `wMaxPacketSize`, the alternate-setting
number, `bFormatIndex`/`bFrameIndex` and the frame interval are **all
unrecorded**" — is **resolved**: all six were decoded from the committed
descriptor and are pinned by tests (see
[Why the RGB stream is silent](#why-the-rgb-stream-is-silent-offline-reconstruction-scanner-012)).
4b is therefore *specifiable* now, but it is still not the next thing to run:
it takes the device away from `uvcvideo` and submits URBs by hand, and a
stop/restore path is still unwritten. The single `READY` next experiment is
**`RGB-1`** — turn uvcvideo's own `trace` on and repeat the unchanged
scanner-011 command once — specified in full, with bounds, discriminating
outcomes and cleanup, in
[roadmap.md](roadmap.md#next-experiment--rgb-1-ready). Experiment 5 (owned ISP
register write) remains `NOT READY` for exactly the reasons listed above:
scanner-012 found nothing that supplies a read-current-state or an exact
restore for any `echo s …` id.

## RGB exposure and auto-exposure archaeology (scanner-022, offline)

This is a static result, not a new hardware observation. The source checkout
was verified at commit `516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`, and the
external SDK at SHA-256
`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`.
Evidence came from `rg`, `nm -C`, `readelf -rW`, and
`objdump -dr -Mintel/-s` over those exact local files. No SDK process,
scanner interface, `/dev/video*`, usbfs, sysfs or network access was used.

### Standard RGB auto-exposure control

- **FACT (source):** `src/cscamera/cscamera.cpp` maps
  `PARA_RGB_AUTO_EXPOSURE` to `STREAM_TYPE_RGB` plus
  `PROPERTY_ENABLE_AUTO_EXPOSURE`; its public getter requires
  `CAMERA_STARTED_STREAM` before dispatching `getProperty`.
- **FACT (binary):** `cs::UvcStream::getProperty` property 5 at `0x254062`
  calls `uvc_get_ae_mode` with request code `0x81`. The helper at `0x25529a`
  selects the parsed Camera Terminal (descriptor terminal type `0x0201`) and
  constructs `wIndex=(terminal_id<<8)|vc_interface`. The recorded RGB
  descriptor has terminal entity 1 and VC interface 2, so this is
  `wIndex=0x0102`. It sends one read-only UVC request:
  `a1 81 wValue=0x0200 wIndex=0x0102 wLength=1 timeout=0`; exactly one byte
  must be returned.
- **FACT (binary):** raw AE bytes `0x02` and `0x08` map to the float at
  `0xeb95d8` (`00 00 80 3f`, `1.0f`); every other byte maps to `0.0f`. The
  setter at `0x253894` uses `21 01`, selector `0x0200`, length 1, and sends
  `0x08` only for CameraType 10, otherwise `0x02`; CameraType 12 therefore
  selects `0x02`. The setter was not run.
- **FACT (kernel corroboration):** the local decompressed `uvcvideo.ko` is
  version 1.1.1, vermagic `6.14.0-37-generic SMP preempt mod_unload
  modversions`, SHA-256
  `3b5b8ccb696e19d484e182cda023b2b99e08c572b51014df4bbc3bdfd7af2788`.
  `nm` places `exposure_auto_mapping` at `.rodata+0x2610`, size 16;
  `objdump -s` shows little-endian u32 `[2,1,4,8]`. Thus V4L2 indices
  0,1,2,3 map to UVC bytes `0x02,0x01,0x04,0x08`. Scanner-021’s current
  index 3 proves the SDK getter returns enabled (`1.0f`) for this standard
  path; this is not an enum-name inference. The menu labels themselves were
  not queried.
- **FACT:** `VIDIOC_QUERYMENU` over 0..3 would add driver label evidence, but
  cannot change that enabled/disabled conclusion. It is optional read-only
  corroboration, not a prerequisite.

### Two distinct RGB exposure-time paths

The binary contains two paths that must not be merged.

1. **Dynamic `UvcCamera` extension path.** The UvcCamera vtable entry at
   `0x14e3358` relocates to `cs::UvcCamera::getPropertyExtension` at
   `0x24df50`; its 0x15/0x16 cases call `cs::libuvcGetExtension` directly.
   For CameraType 12, `0x262ff7` clears a 60-byte buffer and sends one
   read-only request: `a1 81 wValue=0x1200` (selector 0x12),
   `wIndex=0x0400` (entity 4/interface 0 in this repository’s notation),
   `wLength=60`, timeout 1000 ms. A nonnegative libusb result is treated as
   success.
   - For 0x15, `0x24e0e3` copies response bytes 0..3 as little-endian u32
     and multiplies by 1000 into `PropertyExtension.uiExposureTime`.
   - For 0x16, `0x24e19a` reads little-endian u32 words at offsets 4, 8 and
     12; words 4 and 8 are multiplied by 1000 into min/max, while step is
     replaced by fixed `1000.0f` at `0xeb8be4`. Word 12 is logged but not
     used in the returned range. There is no second transfer or clear write.
     `UvcCamera::getRGBExposureTimeRange` at `0x250472` repeats this same
     request and layout.
2. **Base `Camera::getExposureTimeAndRange` standard-UVC fallback.** The base
   implementation calls this helper at `0x1f83a1` for 0x15/0x16, but the
   UvcCamera virtual slot points to the direct override above. The fallback
   still explains the standard values:
   - `UvcStream::getProperty` exposure property 1 at `0x253e5c` calls
     `uvc_get_exposure_abs` with selector `0x0400`, request code `0x81`,
     four-byte response, then applies `log2f` at `0x253eaa`.
   - `UvcStream::getPropertyRange` at `0x25418c`/`0x2541e1` sends four-byte
     `GET_MIN` (`0x82`) and `GET_MAX` (`0x83`) requests with the same
     selector/entity construction, then applies `log2f` at `0x254268` and
     `0x25429d`; step is `1.0f` (`0xeb95d8`).
   - `getExposureTimeAndRange` at `0x200760` calls `getRGBExposureRange`
     (`0x2011b2`), floors max at `0x2007a5`, and for 0x15 uses
     `(current-min+1)*1000` when span is at most 10000. For 0x16 it emits
     min 1000, max `(max-min+1)*1000`, step 1000 for that small-span branch.
     The constants are `1.0f` at `0xeb15b4`, `1000.0f` at `0xeb15ec`, and
     `10000.0f` at `0xeb15e8`.

  **FACT (branch selection):** `getExposureTimeAndRange` recognizes camera
  types 0, 1, 2, 3, 4, 13 and 15 as the alternate branch. For other types it
  calls `compareVersion` against the embedded `v2.9.20` string at `0xeb0b24`
  and takes the standard branch when the current version compares lower. The
  observed MINI type 12 and firmware `v2.8.7` therefore select the standard
  branch whenever this base helper is the implementation reached. The vtable
  evidence above still matters: a dynamically dispatched UvcCamera extension
  call selects its direct selector-0x12 override rather than silently falling
  through to this helper.

### Relationship to scanner-021 and the next bounded task

- **FACT:** scanner-021 reported `V4L2_CID_EXPOSURE_ABSOLUTE` current 131072,
  inactive, outside its reported 1..60000 range. That fact remains
  uninterpreted.
- **FACT/INFERENCE:** the standard UvcStream path proves a log2 encoding and
  the base helper proves a later ×1000 conversion. If scanner-021’s standard
  raw values traverse that fallback, 131072 becomes `log2f=17`, raw range
  1..60000 becomes 0..15.87 then max 15 after floor, and the helper would
  return 18000 us with range 1000..16000 us. This does not prove that the
  firmware’s separate selector-0x12 response derives from that control.
  Because the dynamic UvcCamera override uses a different entity, selector
  and response layout, the direct extension-to-`V4L2_CID_EXPOSURE_ABSOLUTE`
  relationship is **UNKNOWN**; stale, physical-unit and firmware-side causes
  are not established.
- **FACT:** standard `VIDIOC_G_CTRL` is already proven read-only with
  `uvcvideo` bound by scanner-021. **UNKNOWN:** the selector-0x12 XU request
  cannot be reproduced through V4L2, and the earlier owned XU read returned
  `EBUSY` while `uvcvideo` owned VC interface 0. No in-place SDK-XU feasibility
  claim is made.

**Scanner-023 implementation (offline verified; no physical telemetry run):** the
already accepted three-observation `paired-rgb-series` path now has a separately
gated, read-only standard-control telemetry mode. After RGB `STREAMON`, it samples the
three fixed controls in this order — gain (`0x00980913`), exposure-auto
(`0x009a0901`), exposure-absolute (`0x009a0902`) — once before the first RGB
poll and once immediately after each successfully dequeued RGB observation,
before any requeue. This gives at most four sample points and exactly three
`VIDIOC_G_CTRL` attempts per reached point, maximum 12; each value or typed
error is stored with its sample point and associated frame sequence/timestamp
(the initial point has no frame association). A failed telemetry read is
recorded once without retry and never erases frames already captured; capture
status and telemetry status remain separate. The mode issues no `VIDIOC_QUERYCTRL`,
`VIDIOC_QUERYMENU`, `VIDIOC_S_CTRL`, XU request, format variation, added wait,
deadline reset or retry. It preserves the existing absolute RGB deadline and
signal-shielded `STREAMOFF`/unmap/free/close cleanup unchanged. Identity,
role, driver, node, holder and permission mismatches refuse before opening;
signals and cleanup failures retain the existing abort/cleanup precedence.
The implementation uses the existing RGB descriptor and performs no
`VIDIOC_QUERYCTRL`/`VIDIOC_QUERYMENU`, writes, extra waits, deadline resets or
retries. `VIDIOC_QUERYMENU` would provide
standard framework/driver labels or a valid-index mask, not material
device-specific semantics, so it is unnecessary before this read-only task.

**Scanner-024 accepted physical baseline:** one unprivileged paired invocation
delivered a complete 512000-byte depth frame and three preserved RGB payloads.
Each RGB payload had a decodable 1280x800 JPEG prefix followed by four opaque
bytes, so each remained strict `trailing_data` and nearly black. The four
standard-control telemetry points were invariant at gain 16, exposure-auto
index 3 and absolute-exposure raw 131072. These are raw standard-control facts,
not exposure units and not evidence about the separate SDK extension state. No
setting, vendor/XU request, retry or elevation occurred. A devnum change 46 to
47 and video-node inode changes were observed during six stable postflight
samples, but no cause is attributed.

**Scanner-025/027 fixed RGB extension boundary:** `scanner3d
rgb-exposure-extension` remains dry by default and cannot construct a live
backend, resolve a device, open a node or send an ioctl without its exact
opt-in. Its one immutable transfer is `bmRequestType=0xa1`, `bRequest=0x81`,
`wValue=0x1200`, `wIndex=0x0400` (entity 4/interface 0), `wLength=60`, timeout
1000 ms; no SET, second read, retry, V4L2 operation or caller-controlled setup
is expressible. Scanner-027 ran that GET exactly once and measured 2/60 bytes,
raw `0000`, SHA-256
`96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7`, typed
`short_response` (exit 7). The exact raw transfer and length were retained and
no four-word decode was made. Its same accepted run performed exactly one
mandatory 60/60 blue-light off write and passed release/reconnect and stable
postflight; that off write was the final physical action and proves restoration,
not optical darkness.

**Scanner-028 SDK reconciliation (offline):** The source checkout is the
hash-pinned Revopoint 3DViewer commit `516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`;
`sha256sum /tmp/revopoint-3dviewer.i2pZPb/thirdparty/3DCamera/linux/x64/lib3DCamera.so`
returned
`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4` (ELF
BuildID `8d81f1468c0173551e7fcdce55db802e90409f97`, unstripped with debug info).
The source header `thirdparty/3DCamera/include/hpp/Types.hpp` defines
`PROPERTY_EXT_EXPOSURE_TIME_RGB=0x15` and
`PROPERTY_EXT_EXPOSURE_TIME_RANGE_RGB=0x16`; these are SDK property IDs, not
the UVC XU control selector `0x12`. `src/cscamera/cscamera.cpp` maps the public
RGB parameter to 0x15 and dispatches range queries to 0x16.

The complete relevant binary call graph, recovered with `nm -C` and
`objdump -d -C -Mintel`, is:

- `cs::UvcCamera::getPropertyExtension` (`0x24df50`) calls
  `cs::libuvcGetExtension` (`0x262ff7`) at `0x24e14b` for property 0x15 and
  at `0x24e213` for property 0x16. The latter reads response dwords at offsets
  4, 8 and 12; the dwords at offsets 4 and 8 become min/max after `*1000`,
  while offset 12 is logged and the returned step is fixed 1000. The 0x15 branch reads dword 0 and
  stores it as the current value after `*1000`.
- `cs::UvcCamera::getRGBExposureTimeRange` (`0x250472`) calls the same helper
  at `0x25053e`, then reads only words 0 and 1 (offsets 0 and 4), multiplying
  both by 1000 and returning a fixed step. This is a third GET callsite, not a
  second transfer within either range operation.
- `cs::UvcCamera::setPropertyExtension` (`0x24dd90`) routes property 0x15 at
  `0x24df21` to `setRGBExposureTime` (`0x250348`), whose selector-0x12 SET
  call is `libuvcSetExtension` (`0x262eae`) at `0x25041a`. The setter divides
  the caller's microsecond-shaped integer to a word, zero-initializes 60 bytes,
  and places that word at offset 0. It is static evidence only; scanner-027
  did not run it.

For CameraType 12, `libuvcGetExtension` (`0x262ff7`) first `memset`s the
destination to 60 zero bytes (`0x263026`), then calls
`libusb_control_transfer(handle, 0xa1, 0x81, selector << 8, 0x0400, buffer,
60, timeout)` on the GET branch (`0x26307e`–`0x2630b3`). The caller supplies
timeout 1000 ms. The helper returns wrapper success 0 for any nonnegative
libusb return and failure 1 for a negative return; it discards the transferred
byte count. Thus the SDK performs one 60-byte GET, not a second read, but its
wrapper does not enforce exact length before the enclosing decoder reads the
zero-initialized buffer. The scanner-025/027 one-transfer wire model is
therefore faithful; its strict 2/60 result must remain a short raw observation,
not four fabricated u32 words. The corresponding setter uses
`libusb_control_transfer(..., 0x21, 0x01, selector << 8, 0x0400, buffer, 60,
timeout)` and similarly maps nonnegative return to wrapper success.

The recorded descriptor fixture
`tests/fixtures/mini-config-descriptor.bin` is 645 bytes with SHA-256
`2fa267c0c41d4cc53390bb7d1562d4b54fb44e8be04e0d568de8633b5d626613`.
Entity 4 is the declared extension unit (GUID
`C2211A6D-787E-3C4A-AC06-7E434B676F9A`) with bitmap `ffa201`. Interpreting the
bitmap's little-endian control bits, it advertises selectors 1–8, 10, 14, 16
and 17; selector 0x12 (bit 17, zero-based) is clear. It is therefore absent
from the recorded advertised bitmap, not proven reserved or accepted by the
firmware. UVC descriptors do not turn an unadvertised selector into a known
semantic control, and the short reply alone does not prove unsupported.

**Meaning boundary:** scanner-027 measured only two returned bytes `0000`.
The SDK's later zero-initialized-buffer reads explain how a nonnegative short
return could flow into a zero word, but do not establish that this device reply
means exposure, microseconds, enable state, success, failure or a valid zero.
No mapping is claimed.

**Scanner-029 fixed GET_LEN boundary (implemented offline):** the dedicated
`scanner3d.rgb_exposure_length` module and `rgb-exposure-length` CLI command
are dry by default. The exact opt-in is serial-gated and permits only one
device-to-host request: `a1 85`, `wValue=0x1200`, `wIndex=0x0400`, `wLength=2`,
timeout 1000 ms. It reuses the established interface-0 lifecycle, with no
protocol clear, retry, GET_CUR, V4L2 operation or setting path. The packed
setup is fixed to `a185001200040200e8030000`; the whitelist is checked before
the ioctl and the fake-syscall tests assert exactly one possible control
transfer.

The result retains the requested and actual lengths, at most the requested
raw bytes, SHA-256, phases, typed errno/exception/signal and cleanup/recovery
evidence. A short, zero-length or overlong result is not decoded; exactly two
bytes are decoded little-endian as the raw `reported_control_length`. Even an
exact zero is only a raw observation and does not establish support, exposure,
units, success or failure. Dry runs and wrong or missing serials construct no
live backend; a read-only device/preflight refusal may use the backend for
discovery but opens, detaches, claims and transfers nothing. No physical
GET_LEN was run in scanner-029. Scanner-030 subsequently issued that exact
boundary once and completed the mandatory immutable blue-light off write and
clean interface restoration. Every new physical GET_LEN invocation must again
issue its own read followed by that separate off write and clean restoration;
it must stop after that read plus finalizer, preserve typed short/stall/cleanup
evidence, and infer neither support nor meaning from either outcome.

**Scanner-030 accepted physical GET_LEN:** the exact serial-gated command ran
once and issued one `a185001200040200e8030000` request. The MINI returned 2/2
bytes, raw `0200`, SHA-256
`99be5efb88ca2013bd8e4eb035fd42d5245468fe9afa70d8ba9c1c419a48c4e8`, and the
interface-0 release/reconnect cleanup passed. The separate immutable
`0x922=0` off write then accepted 60/60 bytes with payload SHA-256
`6030364daddcc4c9855d16da552c58161bbec5cf77ea23ace9de2f14839ffb0d`; its
cleanup and three stable postflight samples passed. This proves raw command
and restoration evidence only; software has no optical readback.

**Scanner-031 fixed GET_INFO boundary (implemented offline):** the dedicated
`scanner3d.rgb_exposure_info` module and `rgb-exposure-info` command are dry by
default. The only expressible live request is `a186001200040100e8030000`:
`0xa1/0x86`, selector `0x12`, entity 4/interface 0, one byte, timeout 1000 ms.
The exact opt-in is serial-gated, abbreviation-disabled and permits one read
through the established interface-0 lifecycle, with no retry, protocol clear,
other UVC request, setting, V4L2, HID or illumination path. The result keeps
setup, requested/actual/retained length, raw byte, digest, phases, typed
errno/exception/signal and cleanup/recovery evidence; short/overlong replies
remain typed and are never decoded as capability bits.

Only an exact one-byte reply is decoded as `raw_info`. Its `supports_get`,
`supports_set`, `disabled`, `autoupdate`, `asynchronous` and reserved-high-bit
fields are a syntactic Linux-UAPI bit view. They do not prove selector
advertisement, exposure semantics, SET acceptance, units, current-value
validity or the cause of near-black RGB frames. Scanner-031 is offline-only;
a later physical GET_INFO task must issue one read and then one immutable
blue-light-off finalizer with clean restoration.

## Package structure (as of this milestone)

- `src/scanner3d/errors.py` — `DoctorStatus` `StrEnum` (`ok`, `no_frames`,
  `unsupported_mode`, `permission_denied`, `unexpected_identity`,
  `device_absent`, `internal_error`), the `EXIT_CODES` map, and the
  `ScannerError`/`DeviceAbsentError`/`ReferenceProbeError`/
  `ReferenceProbeTimeout` exception hierarchy.
- `src/scanner3d/usb_descriptors.py` — tolerant parser for the raw USB
  configuration-descriptor blob; extracts interfaces, interface
  associations, UVC format/frame descriptors, and class-specific VideoControl
  entities (terminals, units, extension-unit GUIDs and control bitmaps).
- `src/scanner3d/discovery.py` — sysfs discovery, rooted at injectable
  `sysfs_root`/`dev_root` so tests can replay recorded fixtures. Resolves a
  camera's logical role (`depth`/`rgb`) from the USB interface string
  descriptor (`DepthCam…`/`RGBCam…`), deliberately never from the
  `/dev/videoN` index, since that index is assigned in probe order and is
  unstable across replug/reboot.
- `src/scanner3d/doctor.py` — assembles the full diagnostic report and
  classifies overall status from the individual checks (presence, identity,
  topology, roles, modes, permissions, frames, calibration).
- `src/scanner3d/reference.py` — opt-in adapter that hashes and verifies the
  external SDK against `src/scanner3d/data/sdk-manifest.json` (packaged
  inside `scanner3d` so an installed CLI can verify the oracle without the
  repository present), then runs the
  reference probe binary under a hard timeout with process-group kill.
- `src/scanner3d/control.py` — the one owned, read-only USB control request.
  Deliberately not a general control interface: a closed whitelist of exactly
  one setup packet, checked before any file descriptor is opened; the target is
  resolved from `2207:110c` plus an exact serial and snapshotted (path, rdev,
  inode, mode) immediately before the open; open/`ioctl`/close/clock are all
  injectable, so every failure mode is tested without hardware. It knows one
  usbfs command number, `USBDEVFS_CONTROL`, and closes its descriptor on every
  path.
- `src/scanner3d/ownership.py` — the one bounded, reversible interface-0
  ownership lifecycle, and the one read that first used it. `InterfaceZeroLifecycle`
  owns the ownership and nothing else: confirm `uvcvideo` (from sysfs, then
  again from `USBDEVFS_GETDRIVER` on the open descriptor), `USBDEVFS_DISCONNECT`,
  `USBDEVFS_CLAIMINTERFACE`, one `OwnedOperation` supplied by a subclass, that
  operation's protocol undo, then `USBDEVFS_RELEASEINTERFACE` and
  `USBDEVFS_CONNECT` in a `finally`. Interface 0 is the only number its packers
  will encode and the operation is fixed at class level, never an argument; the
  lifecycle names no setup packet at all. The reconnect runs on every failure
  path from the *attempted* detach onwards - including a SIGINT/SIGTERM or
  unexpected exception surfacing from the disconnect ioctl, where kernel
  completion is ambiguous - the release runs with it whenever the claim may be
  held, the protocol undo runs before both whenever the operation was entered,
  cleanup runs at most once so nothing is retried, and either cleanup failing
  forces a nonzero exit (9 for the driver, 14 for the protocol undo) with
  explicit recovery guidance. `UsbfsOwnershipBackend` is that lifecycle carrying
  the unchanged whitelisted read, which writes nothing and so owes no undo. It
  knows six usbfs command numbers and no more; a test asserts the set and that
  `scanner3d.control` still knows only one.
- `src/scanner3d/init_sequence.py` — the four-transfer `/tmp/inited` sequence on
  entity 4, as a second closed whitelist on the same lifecycle. It can send two
  payloads and no others: `01` + `"/tmp/inited"` + NUL zero-padded to 60, and 60
  zeros. The path is a constant, not an argument; there is no way through it to
  vary the path, opcode, selector, entity, interface, length, timeout or
  repetition, and the other two paths in the same capture are absent from its
  executable source. The whole sequence is validated before a descriptor is
  opened and each transfer again before its ioctl argument is packed. The
  clearing write is protocol cleanup, not a fifth step: owed from the instant
  before the path write is attempted, so an ambiguous write still owes it. It
  knows one usbfs command number, `USBDEVFS_CONTROL`; the driver ioctls all
  belong to the lifecycle it reuses.
- `src/scanner3d/intrinsics.py` — the bounded `/data/camparam/Pl.bin`
  depth-intrinsics read, as a third closed whitelist on the same lifecycle: one
  key `SET_CUR`, at most eight length-prefixed `GET_CUR` result reads, one
  60-zero clearing `SET_CUR`. The key is an ASCII constant and the payload is
  derived from it at import; no function and no flag takes a path, opcode,
  selector, entity, interface, length, timeout, cap, capacity or repetition, and
  the sibling calibration keys, the licence paths, the write-file opcode and the
  `0x0e00` depth-scale selector are all absent from its executable source. Every
  bound is fixed at import and never widened at runtime: 256-byte capacity,
  8-read cap (10 control transfers total), and a declared chunk length over 56
  is rejected before anything is copied — a chunk that would pass the capacity
  is not copied at all rather than truncated, and nothing is ever retried. The
  assembled blob, its length and its SHA-256 are the primary evidence and are
  reported whatever they are; only a complete 40-byte blob is additionally
  parsed as the vendor `Intrinsics` record, structural and plausibility checks
  are kept on separate axes, and a divergent record is typed evidence rather
  than a crash. Ten read-phase outcomes are distinguished, and only a run whose
  clearing write and driver handback both succeeded offers the record as
  accepted calibration. It knows one usbfs command number, `USBDEVFS_CONTROL`.
- `src/scanner3d/depth_scale.py` — the closed two-transfer `0x918` depth-scale
  property read, as a fourth closed whitelist on the same lifecycle: one 4-byte
  property `SET_CUR`, one 60-byte result `GET_CUR`, and no third transfer — the
  vendor primitive has no clearing write of its own, so none is invented, and
  the lifecycle's driver handback runs regardless. The property id is a
  compile-time constant and the payload is packed from it at import; no
  function and no flag takes a property id, a selector, an entity, an
  interface, a length, a timeout, a count or a repetition, and no other `0x9xx`
  id appears in its executable source. Byte 59 of the reply is the marker
  (accepted only at `4`) and bytes 0:4 the signed little-endian `int32`
  denominator; only a reply passing both gates yields
  `float32(1.0f / denominator)`, reported with its exact bit pattern, and the
  scalar's physical unit is never claimed as known. The raw reply — length,
  SHA-256, hex — is the primary evidence and is reported whatever it is, and
  the SDK's static MINI fallback constant is compared beside it only as a
  clearly labelled observation. It knows one usbfs command number,
  `USBDEVFS_CONTROL`.
- `src/scanner3d/cli.py` — `scanner3d doctor`, `scanner3d list`,
  `scanner3d trace`, `scanner3d firmware-version` (offline by default;
  `--execute` opts in to the single read above, and
  `--detach-uvcvideo-from-interface-0` opts in to performing it through the
  ownership lifecycle instead), and `scanner3d inited-sequence` (offline by
  default; `--execute-write-sequence` — a deliberately separate and unmistakable
  flag, not the read command's `--execute` — opts in to the four-transfer
  sequence and requires an explicit `--serial`), and `scanner3d depth-intrinsics`
  (offline by default; `--execute-calibration-read` — again a separate,
  unmistakable flag — opts in to the bounded key read, requires an explicit
  `--serial`, and refuses any serial but the authorized unit), and
  `scanner3d depth-scale` (offline by default, parsed with `allow_abbrev=False`
  so the execute flag cannot be reached by prefix; `--execute-depth-scale-read`
  — again a separate, unmistakable flag — opts in to the bounded two-transfer
  property read, requires an explicit `--serial`, and refuses any serial but
  the authorized unit). The cleanup records are rendered and exited on
  separately.
- `src/scanner3d/usbfs_trace.py` — offline decoder for usbfs traces:
  validates the stream header, correlates asynchronous URB submissions with
  their completions deterministically, groups transfers structurally, and
  preserves unmatched, failed and truncated events as evidence rather than
  dropping them. Pure file reading; needs neither the device nor the SDK.
- `reference/probe/` — a standalone C++ probe (`probe.cpp`, `build.sh`)
  built against the externally supplied SDK. Source only; the SDK itself is
  not vendored and build artifacts go to `/tmp`.
- `reference/tracer/` — the `LD_PRELOAD` usbfs ioctl interposer
  (`usbfs_trace.c`, `build.sh`) that produced the capture above. Opt-in,
  targets one exact usbfs node, fails closed, bounds every payload and write,
  and adds no device operation of its own. Source only; it builds to `/tmp`
  and its captures stay there.
- `reference/analysis/` — an offline static-analysis helper (`xu_trace.py`,
  `README.md`) that reproduces the `libusb_control_transfer` argument tuples of
  the vendor XU primitives from a local `lib3DCamera.so` using only `nm`/`objdump`.
  It distinguishes immediates, fixed values staged through stack locals, and
  genuinely parameter-derived operands, reporting the last as `dynamic` rather
  than guessing; its limits are documented in its README.
  Not part of the `scanner3d` package and never imported by the CLI; its pure
  parser is covered by offline tests in `tests/test_reference_analysis.py`.
  Read-only, no device, no SDK, no network.

## Checks not run

- No RGB frame was obtained through the vendor SDK or either standalone RGB
  attempt; scanner-016's accidental physical bytes are unaccepted context,
  while the accepted reviewed scanner-017 paired run delivered one raw RGB
  payload, strict-invalid with four opaque bytes after EOI. No extrinsics or
  distortion was obtained, and no depth frame was ever obtained through the
  vendor SDK — it failed at authorization before those became reachable.
  Owned code has three exceptions, each requested and obtained exactly once:
  the depth intrinsics and the depth-scale property over the vendor control
  channel, and, separately and over plain V4L2 instead, one raw depth frame.
  scanner-007 read `/data/camparam/Pl.bin` from this unit once and got a
  complete 40-byte record — see
  [Owned code reads the factory depth intrinsics](#owned-code-reads-the-factory-depth-intrinsics-measured)
  — scanner-008 read the `0x918` depth-scale property once and got a
  complete, decoded 60-byte reply whose physical unit remains unestablished —
  see the "Confirmed on hardware" entry in
  [Domain 4](#domain-4--calibration--depth-scale-fact-disassembled) — and
  scanner-009 captured one raw `Y16` 640×400 depth frame once, with neither
  calibration value applied to it and no RGB counterpart — see
  [Owned code captures a raw depth frame](#owned-code-captures-a-raw-depth-frame-measured).
  Two bounded RGB attempts have been spent and both dequeued nothing.
  scanner-010 stopped at `S_FMT` because the driver's `sizeimage` exceeded the
  ceiling derived from the device's own descriptor — see
  [Owned code attempts a raw RGB frame](#owned-code-attempts-a-raw-rgb-frame-measured).
  scanner-011 then raised that ceiling to the measured `sizeimage` and spent one
  further attempt: the buffers were allocated, mapped, queued and streamed, and
  no buffer became ready within the 8.0 s deadline — see
  [Owned code attempts a raw RGB frame with the measured ceiling](#owned-code-attempts-a-raw-rgb-frame-with-the-measured-ceiling-measured).
  So a capture with the larger ceiling *has* now been tested, once, and did not
  succeed; **why** no buffer arrived was not investigated and is unknown.
  The sibling calibration keys and every other `0x0E00` property remain
  untested, and no repeat depth-frame capture has been attempted.
- The vendor control protocol rests on two separate evidence sets, and neither
  characterizes it fully. Keep them apart:
  - *SDK trace (observation of the vendor's own traffic).* The `LD_PRELOAD`
    capture reaches only as far as the authorization failure and stops there.
    What it observed on the wire is eleven vendor-channel class requests on
    entities 3 and 4 of interface 0, **all** of them 60-byte payloads — six
    `SET_CUR wValue=0x0100`, four `GET_CUR wValue=0x0200`, one
    `GET_CUR wValue=0x0600` (see
    [above](#vendor-control-channel-measured-on-the-wire)). No streaming,
    projector, mode-select or calibration-readout traffic appears in it,
    because the SDK never gets that far on this unit.
  - *Owned code (traffic this repository issued itself).* Independently of that
    capture, scanner-004, scanner-005, scanner-007 and scanner-008 issued
    eleven control transfers across four detach/claim runs — ten carrying
    60-byte payloads plus the 4-byte `0x918` property write — and those *do*
    include a calibration readout: the factory depth intrinsics from
    `/data/camparam/Pl.bin` and the `0x918` depth-scale property. The two
    counts matching at eleven is a coincidence, not the same transfers.
  Streaming, projector control and mode select remain unobserved on the wire by
  either evidence set, and unimplemented by owned code.
  Their call paths are recovered statically only —
  see [Static protocol archaeology](#static-protocol-archaeology-sdk-call-paths-toward-the-wire) —
  with **exact request shapes for the XU primitives** (`libuvcExcuteCmd`,
  `libuvcReadValueExtension`, `libuvcReadFile`) but **not** for streaming, where
  the endpoint, alternate setting, format/frame indices and ordering all remain
  unrecovered. A static call path is not an observation in any case.
- ~~The extension-unit entity IDs inferred from `wIndex` have not been
  cross-checked.~~ **Done**, and entity 3 turned out to be an output terminal —
  see [VideoControl entities](#videocontrol-entities-read-from-the-devices-own-descriptors).
- No kernel-level USB bus capture (e.g. `usbmon`/Wireshark) was taken. The
  capture that exists is a userspace ioctl interposition of one process, which
  cannot see kernel-driver traffic.
- ~~No vendor byte has been read from the scanner by owned code.~~ **Answered
  by scanner-004: 60 bytes, correct on the independent oracle**, once interface 0
  was temporarily taken from `uvcvideo` — see
  [Owned code reads the vendor channel](#owned-code-reads-the-vendor-channel-measured).
  That is one read-only request on one entity, and it says nothing about any
  other request.
- ~~No owned `SET_CUR` has ever been issued.~~ **Answered by scanner-005: the
  device accepts one.** The four-transfer `/tmp/inited` sequence on entity 4 —
  two writes and two reads — was reproduced exactly, without the SDK and without
  a licence, and both writes were accepted in full. See
  [Owned code writes to the vendor channel](#owned-code-writes-to-the-vendor-channel-measured).
  That is two payloads on one selector of one entity, both of which the vendor's
  own software sent to this unit; it says nothing about any other `SET_CUR`.
- ~~Whether the vendor file-read protocol can return the *contents* of any
  path.~~ **Answered by scanner-007 for one key:** `/data/camparam/Pl.bin`
  returned its 40-byte contents, a parseable factory `Intrinsics` record — see
  [Owned code reads the factory depth intrinsics](#owned-code-reads-the-factory-depth-intrinsics-measured).
  `/tmp/inited` still returns the algorithm version string rather than contents,
  the two licence paths were not sent, and no other `/data/camparam` key was
  requested.
- Owned code has still issued nothing outside those eleven vendor-control
  transfers: no third path, no `0x0E00` property id other than `0x918`, no
  other entity, no HID transfer, no device reset. Separately, one V4L2 stream
  start now exists outside that channel entirely — one frame, one run, no
  ownership change, no vendor-control traffic (scanner-009); see
  [Owned code captures a raw depth frame](#owned-code-captures-a-raw-depth-frame-measured).
- The disruption a detach causes to an *active* capture is unmeasured. All
  four measured detaches happened with nothing streaming and no process holding
  a video node.
- Whether the device would still hold a path selection had the clearing write
  not been sent is unmeasured: that would mean deliberately skipping cleanup.
  "Cleared" means the clearing write completed, as the SDK does it.
- No udev rule was installed; current access relies entirely on the logind
  uaccess ACL for a seat-local session.
- Multi-device behavior (more than one scanner attached at once) was not
  tested on hardware.
- Only one scanner cable has been tried. The 480 Mb/s link was reproduced both
  behind the Realtek hub and with the scanner attached directly to a root port
  proven SuperSpeed-capable (see [Link speed](#link-speed)), which excludes the
  hub and the host port, but does not separate a device ceiling from a
  USB 2-only cable. Smallest falsifiable next step, if it ever matters: attach
  the scanner directly with a second, known-good SuperSpeed cable and read
  `/sys/bus/usb/devices/<path>/speed`. If it still reads 480 the ceiling is the
  device; if it reads 5000 the cable was the limit. Nothing currently known
  about this unit depends on the answer.
