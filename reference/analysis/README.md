# Static / offline analysis helpers

This directory holds read-only discovery aids. **None of them is part of the
`scanner3d` package**: none is imported by the CLI, none adds a runtime
dependency, and none opens a device node, sends a USB transfer or touches the
network. Each has offline tests that run in the default gate.

| helper | question it answers | tests |
| --- | --- | --- |
| [`xu_trace.py`](#static-protocol-analysis-helper-xu_tracepy) | what `libusb_control_transfer` tuples does the vendor SDK use for its XU channel? | [`tests/test_reference_analysis.py`](../../tests/test_reference_analysis.py) |
| [`uvc_stream_state.py`](#uvc-stream-state-reconstruction-uvc_stream_statepy) | what UVC/V4L2 state did the silent scanner-011 RGB attempt actually reach? | [`tests/test_uvc_stream_state.py`](../../tests/test_uvc_stream_state.py) |
| [`uvc_module_params.py`](#uvcvideo-module-parameters-uvc_module_paramspy) | are the live `uvcvideo` module parameters overridden, or at their compiled defaults? | [`tests/test_uvc_module_params.py`](../../tests/test_uvc_module_params.py) |

## Static protocol-analysis helper (`xu_trace.py`)

`xu_trace.py` reproduces part of the scanner-006 evidence: the
`libusb_control_transfer` argument tuples the vendor `lib3DCamera.so` uses for
its extension-unit (XU) control channel and UVC stream negotiation. It is a
discovery/verification aid, **not** part of the `scanner3d` package — it is
never imported by the CLI and adds no runtime dependency.

It is read-only and offline by construction:

- it accepts a **local path** to an externally supplied `lib3DCamera.so` (the
  SDK is deliberately not vendored in this repository);
- it shells out only to `nm` and `objdump` (ordinary binutils) against that
  file, and opens no device node, sends no USB or SDK call, and does no network;
- its pure core, `extract_control_transfers()`, parses `objdump -d` text and is
  covered by offline tests in [`tests/test_reference_analysis.py`](../../tests/test_reference_analysis.py)
  using committed disassembly excerpts (no binary required to run the tests).

## What it does and does not recover

The parser recovers the argument tuple of each call site. It distinguishes
three cases and never collapses them:

| Case | Example | Reported as |
| --- | --- | --- |
| Immediate straight into the argument register | `mov $0x21,%esi` | the constant |
| Fixed value staged through an `%rbp` stack local | `movl $0xe00,-0x4(%rbp)` … `mov -0x4(%rbp),%eax` | the constant, annotated `[stack local -0x4]` |
| Value derived from a parameter or arithmetic | `(selector << 8) & 0xffff` | `dynamic` |

A stack local counts as constant only if the whole disassembled function body
writes it exactly once, with an immediate. Any register-sourced write, or a
second write, makes it unknown — so a parameter-derived slot can never be
reported as a constant. Register state is cleared at every call and at every
jump target; unmodelled instructions (`shl`, `and`, `lea`, arithmetic, non-`rbp`
memory loads) set their destination to unknown rather than leaving a stale
immediate behind.

The parser also tracks the provenance of the `data` pointer and stores made
into it, which is how the four-byte depth-scale payload is recovered rather than
asserted: `payload[0:4] = arg3`.

**Limits, stated so the output is not over-read.** This is a linear,
intra-function scan, not a full dataflow analysis. It does not model writes
through arbitrary pointers into a tracked slot, `lea`-formed buffer addresses,
`%rsp`-relative frames, or values produced by arithmetic. Those surface as
`dynamic`, which means **"not recovered by this tool"** — not "proven to vary at
run time". Two cases in the output below make that concrete:

- `uvc_query_stream_ctrl` reports everything except `timeout` as `dynamic`. The
  PROBE/COMMIT `wValue` is in fact `0x0100`/`0x0200` chosen by a branch
  (`0x25fd5b`/`0x25fd63`) and the `wIndex` is the VideoStreaming interface
  number loaded from `uvc_device_info+0x24` (`0x25fd4e`).
- `libuvcReadFile` reports only its first `wValue` (`0x0100`) as recovered. The
  later `0x0200` reads and the `0x0100` cleanup write go through the *same*
  stack slot `-0x78(%rbp)`, written three times, so the conservative
  written-exactly-once rule refuses it. The values are plain immediates at
  `0x2628f0`, `0x2629da` and `0x262b4e`.

Both are cited to disassembly directly in
[`docs/architecture.md`](../../docs/architecture.md), not through this tool.

## Offline invocation

Supply the SDK path (same layout the reference probe expects). The pinned hash
is in [`src/scanner3d/data/sdk-manifest.json`](../../src/scanner3d/data/sdk-manifest.json):

```bash
python reference/analysis/xu_trace.py \
  --so /path/to/3DViewer/thirdparty/3DCamera/linux/x64/lib3DCamera.so \
  --expect-sha256 49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4
```

Expected output for the pinned build (`8d81f146…`):

```
## libuvcSetExtension
  @262f38: bmRequestType=0x41 bRequest=0xff wValue=dynamic wIndex=0x300 wLength=0x3c timeout=dynamic
  @262f94: bmRequestType=0x21 bRequest=0x1 wValue=dynamic wIndex=0x400 wLength=0x3c timeout=dynamic

## libuvcGetExtension
  @263070: bmRequestType=0xc1 bRequest=0xff wValue=dynamic wIndex=0x300 wLength=0x3c timeout=dynamic
  @2630aa: bmRequestType=0xa1 bRequest=0x81 wValue=dynamic wIndex=0x400 wLength=0x3c timeout=dynamic

## libuvcReadValueExtension
  @262dec: bmRequestType=0x21 bRequest=0x1 wValue=0xe00 [stack local -0x4] wIndex=0x400 wLength=0x4 timeout=0x3e8
      payload[0:4] = arg3 (@262dbd)
  @262e6a: bmRequestType=0xa1 bRequest=0x81 wValue=0xe00 [stack local -0x4] wIndex=0x400 wLength=0x3c timeout=0x3e8

## libuvcExcuteCmd
  @262d0c: bmRequestType=0x41 bRequest=0xff wValue=0x700 [stack local -0x54] wIndex=0x300 wLength=0x3c timeout=dynamic
  @262d46: bmRequestType=0x21 bRequest=0x1 wValue=0x700 [stack local -0x54] wIndex=0x400 wLength=0x3c timeout=dynamic

## libuvcReadFile
  @26293c: bmRequestType=0x41 bRequest=0xff wValue=dynamic wIndex=0x300 wLength=0x3c timeout=0x3e8
  @26297a: bmRequestType=0x21 bRequest=0x1 wValue=0x100 [immediate] wIndex=0x400 wLength=0x3c timeout=0x3e8
  @262a40: bmRequestType=0xc1 bRequest=0xff wValue=dynamic wIndex=0x300 wLength=0x3c timeout=0x3e8
  @262a81: bmRequestType=0xa1 bRequest=0x81 wValue=dynamic wIndex=0x400 wLength=0x3c timeout=0x3e8
  @262b90: bmRequestType=0x41 bRequest=0xff wValue=dynamic wIndex=0x300 wLength=0x3c timeout=0x3e8
  @262bcb: bmRequestType=0x21 bRequest=0x1 wValue=dynamic wIndex=0x400 wLength=0x3c timeout=0x3e8

## uvc_query_stream_ctrl
  @25fd9d: bmRequestType=dynamic bRequest=dynamic wValue=dynamic wIndex=dynamic wLength=dynamic timeout=0x0
```

The two branches per extension helper are the pre-MINI vendor request
(`CameraType <= 9`) and the MINI-family UVC class request (`CameraType > 9`).
`libuvcReadValueExtension` has no such branch: it always issues the UVC class
form. See [`docs/architecture.md`](../../docs/architecture.md) for the full
call-path analysis this helper supports.


## UVC stream-state reconstruction (`uvc_stream_state.py`)

`scanner-011` negotiated `MJPG` 1280x800 on the RGB node, mapped four buffers,
started the stream, and then no buffer became ready in 8.007 s. This helper
reconstructs — from committed evidence only — exactly what standard state that
run reached, and what it never negotiated.

Inputs are all committed fixtures:

- `tests/fixtures/mini-config-descriptor.bin` — the byte-for-byte
  configuration descriptor of the physical MINI;
- `tests/fixtures/scanner-009-depth-capture.json` — the one **successful**
  depth capture, which is the positive control;
- `tests/fixtures/scanner-010-rgb-attempt.json` and
  `scanner-011-rgb-attempt.json` — the two RGB attempts.

```bash
python reference/analysis/uvc_stream_state.py           # human-readable report
python reference/analysis/uvc_stream_state.py --json     # machine-readable
```

It decodes what `scanner3d.usb_descriptors` deliberately does not: alternate
settings, endpoints and their transfer types, `bcdUVC`, the VS input header,
and per-format/per-frame indices, buffer sizes, bit rates and frame intervals.
The product parser answers "which modes does this unit advertise" and its
contract is untouched; these are the fields the *streaming* question turns on,
and [`docs/architecture.md`](../../docs/architecture.md) previously listed all
six of them as unrecorded prerequisites for direct-transport work.

What it establishes for this unit (all pinned by tests):

- both VideoStreaming interfaces have **exactly one alternate setting** and a
  **bulk** endpoint — so there is no isochronous bandwidth negotiation to fail;
- RGB is interface 3, endpoint `0x84`, `wMaxPacketSize` 512, `bFormatIndex` 1,
  `bFrameIndex` 1, default interval 333333 (30 Hz), and `MJPG` 1280x800 is the
  **only** mode it advertises;
- `bcdUVC` is `0x0100`, fixing the VS Probe/Commit block at **26** bytes;
- the driver's `sizeimage` matched the descriptor exactly for the uncompressed
  depth capture and is **4x** the descriptor for compressed RGB;
- which V4L2 operations owned code issued, and which it never did.

**What it does not do:** it reports descriptor and recorded-run values. It never
claims a runtime value that was not measured — in particular the negotiated
probe/commit block has never been read back from this unit, and is reported as
unmeasured rather than guessed.

## `uvcvideo` module parameters (`uvc_module_params.py`)

Reading `/sys/module/uvcvideo/parameters/quirks` gives `4294967295`, which looks
exactly like "every UVC quirk has been forced on for this host". It is not: that
value is `(unsigned)-1`, the sentinel meaning *no module parameter was
supplied*. During `scanner-012` that reading was made and had to be corrected,
so the check now exists as tested code rather than a one-off shell result.

The helper compares each live parameter against the initialiser compiled into
the module's own `.data`/`.bss`, and reports a parameter as overridden **only**
when the two provably differ.

```bash
zstd -d /lib/modules/$(uname -r)/kernel/drivers/media/usb/uvc/uvcvideo.ko.zst -o /tmp/uvcvideo.ko
python reference/analysis/uvc_module_params.py --module /tmp/uvcvideo.ko
```

It contains a small pure-Python ELF64 section/symbol reader, so it shells out to
nothing; decompressing the `.ko.zst` is left to the caller. `--no-live` prints
the compiled defaults alone, and `--sysfs` points the live read somewhere else
(which is how the tests drive it). Its tests build a minimal ELF64 object
in-process, so they need no kernel module, no binutils and no root.

## Offline tests for everything here

```bash
.venv/bin/pytest tests/test_reference_analysis.py tests/test_uvc_stream_state.py \
    tests/test_uvc_module_params.py
```
