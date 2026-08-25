# Revopoint 3DCamera SDK reference probe

This is a standalone, single-file C++ program that exercises the read-only
query surface (and a short, bounded frame capture) of the vendor Revopoint
`3DCamera` SDK against a real, attached camera. It exists to answer factual
questions about SDK behavior (what does `getStreamInfos` actually return for
this device, what error code comes back from a given call, does streaming
actually work) independent of the Python package that lives elsewhere in
this repository.

The SDK itself is treated as an external, temporary oracle:

- It is **not vendored** into this repository. `build.sh` references it by
  path only (`REVOPOINT_SDK_ROOT` env var or `--sdk-root`), and refuses to
  run if the expected files are missing rather than downloading anything.
- Build artifacts and JSON output are written under `/tmp`, never into the
  repo.

## Files

- `probe.cpp` - the probe itself.
- `build.sh` - locates the SDK by path, prints sha256 hashes of the header
  and shared library it is about to link against, and builds to
  `/tmp/3d-scanner-probe/probe` by default.

## SDK inputs used for this probe run (sha256)

SDK root: `/tmp/revopoint-3dviewer.i2pZPb/thirdparty/3DCamera` (from
reference repo commit `516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`), SDK
version 3.2.23.

```
160c609a1ac1f4f3513ec79f7ef6ae890bad058ef620b93833dfa3888fed2cdb  include/3DCamera.hpp
49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4  linux/x64/lib3DCamera.so
```

Note: that `/tmp/revopoint-3dviewer.i2pZPb` path is itself an ephemeral
extraction directory outside this repo's lifecycle; if it's gone, re-point
`REVOPOINT_SDK_ROOT` at wherever the vendor SDK tree lives and re-run
`build.sh` -- it will print fresh hashes for whatever it finds.

## Build

```bash
REVOPOINT_SDK_ROOT=/tmp/revopoint-3dviewer.i2pZPb/thirdparty/3DCamera \
    /home/ben/repositories/3d-scanner/reference/probe/build.sh
```

or equivalently:

```bash
/home/ben/repositories/3d-scanner/reference/probe/build.sh \
    --sdk-root /tmp/revopoint-3dviewer.i2pZPb/thirdparty/3DCamera \
    --out /tmp/3d-scanner-probe/probe
```

This runs, under the hood:

```bash
g++ -std=c++11 -O2 -Wall -Wextra \
    -DSDK_LIB_PATH_STR="<libdir>/lib3DCamera.so" \
    -I<sdk-root>/include probe.cpp -o /tmp/3d-scanner-probe/probe \
    -L<sdk-root>/linux/x64 -l3DCamera \
    -Wl,-rpath,<sdk-root>/linux/x64 -lpthread
```

## Run

```bash
timeout 15 /tmp/3d-scanner-probe/probe \
    --serial EXAMPLESERIAL0001 \
    --json-out /tmp/3d-scanner-probe/probe-run.json
```

CLI flags:

- `--serial <SN>` (required) - the USB serial of the camera to connect to.
- `--json-out <path>` (optional) - also write the JSON result to this file.
- `--timeout-ms <int>` (default `12000`) - hard wall-clock budget. A
  watchdog thread force-exits the process with a partial JSON dump
  (`status: "timeout"`, exit code 4) if the main thread hasn't finished by
  then.
- `--frames <n>` (default `5`) - number of paired depth+RGB frames to
  attempt to capture via `getPairedFrame`.

The program always runs under an external `timeout 15` in addition to its
own internal watchdog; no single invocation should exceed 15 seconds wall
clock.

Exit codes: `0` success (both a depth and an RGB frame were obtained), `2`
device not found (serial not in `queryCameras` results), `3` connect
failed, `4` no frames obtained / internal watchdog timeout, `5` other SDK
error (e.g. `getSystemPtr`/`getCameraPtr` returned null).

## Output

A single JSON object is written to stdout (and to `--json-out` if given)
with SDK version/lib info, the list of enumerated cameras, whether the
requested serial matched, stream support/profiles, intrinsics/extrinsics/
distortion, depth scale, frame capture stats, a list of every non-SUCCESS
`ERROR_CODE` encountered (stage, numeric code, symbolic name, message from
`getCameraErrorString`), elapsed time, and an overall `status` string.

## Diagnostic flags

These exist only to investigate SDK behaviour; none of them changes the
default run, which is kept byte-identical so recorded evidence stays
reproducible.

- `--sdk-log-dir <dir>` - call `cs::setLogSavePath(dir)` and
  `cs::enableLoging(true)` before doing anything else. The SDK ignores the
  path on Linux but does print its full internal trace to stdout/stderr,
  which is where the licence-failure evidence below came from.
- `--libuvc true|false` - call `cs::setSdkEnableLibuvc()` before
  `queryCameras`.
- `--checked-camera-inited true|false` - call
  `cs::setCheckedCameraInited()`.
- `--init-driver-code <code>` - call `initDriver()` with an activation code.
- `--connect-mode <mode>` - use the no-argument `connect()` overload instead
  of `connect(CameraInfo)`.
- `--c-api-connect` - go through the plain-C `createSystem` /
  `systemCreateCameraInfoList` / `systemConnectCamera` path instead of the
  C++ `cs::` API.

## Recorded result on the unit tested (serial `EXAMPLESERIAL0001`)

**Enumeration succeeds; connection does not.** `queryCameras` returns the
camera with `name=3DCamera-uvc`, `firmwareVersion=v2.8.7.20220610`,
`algorithmVersion=v7.5.7.0426`, and `getCameraTypeName(getCameraTypeBySN())`
resolves to `MINI`. `connect(CameraInfo)` then fails with
`ERROR_NOT_SUPPORT` (code 8) in ~1.0-1.6 s, so no stream profiles,
intrinsics, extrinsics, depth scale, or frames were reachable. Probe exit
code `3`, `status: "connect_failed"`.

With `--sdk-log-dir` the SDK's own trace shows why: it detaches `uvcvideo`
and claims interfaces 0 and 4, opens the HID channel successfully, then
fails to read an `sdk-license` blob from the camera's on-device storage and
logs `authorize failed`. This is a vendor licensing gate, not a permission,
transport, or firmware-compatibility problem. `--libuvc`, `--connect-mode`,
`--checked-camera-inited`, and `--c-api-connect` were all tried and none
changes the outcome.

Full evidence, including the falsified `/dev/hidraw` hypothesis, is in
[`../../docs/architecture.md`](../../docs/architecture.md#reference-oracle-result-the-vendor-sdk-cannot-connect-to-this-unit).

## Safety properties

- Every hardware run is bounded twice: an internal watchdog thread and an
  external `timeout 15`.
- `stopStream` and `disconnect` run on all normal exit paths via an RAII
  guard, so no stream is left open.
- The SDK writes a `tracksData/` directory into its working directory at
  runtime; it is gitignored. Run the probe from `/tmp` if you would rather
  it not appear in the repo at all.
