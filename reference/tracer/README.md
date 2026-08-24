# usbfs transaction tracer

An opt-in `LD_PRELOAD` interposer that records — and never alters — the
`USBDEVFS_*` ioctls one process issues to **one exact** usbfs node, plus the
offline decoder that reads the result back (`scanner3d trace`).

This exists because of a specific, measured gap. The vendor Revopoint SDK talks
to this scanner entirely through `ioctl()` on `/dev/bus/usb/BBB/DDD` — its
statically linked libusb never opens `/dev/video*` or `/dev/hidraw*` — and it
uses the asynchronous URB model (`USBDEVFS_SUBMITURB` + `USBDEVFS_REAPURB*`).
A plain `strace` therefore proves *that* transfers happened and on which
interface, but prints the `struct usbdevfs_urb` argument as a bare pointer, so
the wire bytes are unrecoverable from it. Running inside the traced process, we
can dereference that pointer.

## What this is not

- It is **not** a license bypass. It records what the SDK does and stops there.
  The SDK's `authorize failed` path is evidence to be described, not a gate to
  be worked around, and nothing here patches, hooks, or short-circuits it.
- It is **not** a replay tool. Nothing in this directory sends anything to the
  scanner. The tracer adds no device operation of its own — no ioctl, no read,
  no write against the traced descriptor.
- It is **not** a bus capture. It sees only what the traced process itself
  asks for. Traffic issued by other processes, and by the kernel's own
  `uvcvideo`/`usbhid` drivers, is out of reach of this mechanism entirely.

## Files

- `usbfs_trace.c` — the interposer.
- `build.sh` — builds it to `/tmp/3d-scanner-trace/libusbfs_trace.so` by
  default and prints the sha256 of what it built.
- `../../src/scanner3d/usbfs_trace.py` — the offline decoder.
- `../../tests/test_usbfs_trace.py` — synthetic fixtures and tracer tests.

## Safety properties

These are the properties the implementation is built to hold, in priority
order. Each is enforced in code, not by convention:

1. **Observe, never alter.** Every hooked call is forwarded to the real libc
   symbol with its arguments untouched, and the caller's return value is
   always the real call's. `errno` transparency is explicit, not incidental:
   each hook saves the caller's `errno` on entry and restores it immediately
   before forwarding to the real symbol, then restores the real call's own
   `errno` immediately before returning — so the tracer's own bookkeeping in
   between (`fstat`, `readlink`, `clock_gettime`, `write`, mutex operations)
   never leaks into what the caller sees. A hooked call that succeeds leaves
   the same `errno` behind as it would untraced. The one place this is not
   silent is diagnostics: a fail-closed condition still writes to the traced
   program's stderr (see Known capture limitations).
2. **Opt in, and fail closed.** Capture happens only when both
   `SCANNER3D_TRACE_TARGET` and `SCANNER3D_TRACE_OUT` are set. If the target
   does not resolve, is not under `/dev/bus/usb/`, or is not a character
   device, the tracer disables itself and records *nothing*. "Disabled" always
   means "captures nothing", never "captures everything".
3. **One node snapshot only.** A descriptor is traced only if it was opened on
   the exact canonical target path **and** its `st_rdev` matches the target's,
   and both halves are required for every newly observed descriptor — so the
   capture never broadens to another usbfs node that is open at the same time.
   Descriptors for other USB devices, for regular files, and for the tracer's
   own output are never recorded.

   What this does **not** do is authenticate physical identity across a
   hotplug. Linux may hand the same bus/device number, the same canonical
   `/dev/bus/usb/BBB/DDD` path, and therefore the same `st_rdev` to whatever
   enumerates next, so path plus rdev cannot prove the scanner stayed where it
   was. The tracer scopes one *snapshot* of one node; the operator must resolve
   that node from the device serial immediately before the bounded run, and
   must treat any disconnect or re-enumeration as invalidating the target and
   requiring a fresh resolution. Closing that gap inside the tracer would mean
   issuing USB I/O of its own to re-read the serial, which property 1 forbids.
4. **Never clobber, and one file per process.** The output file is created
   `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW`. An existing path, or a symlink at that
   path, disables capture instead of overwriting anything. The name carries the
   pid — `<SCANNER3D_TRACE_OUT>.<pid>` — and the file is created lazily, on the
   first event a process actually records. Both are forced by the same fact:
   the traced program is free to fork and exec (even a bare `timeout 15 probe`
   wrapper means two images run with this library loaded), every image re-runs
   the constructor, and a fork child that is about to exec must not claim the
   filename the exec'd image will need. A process that never touches the target
   device therefore leaves no file at all, which is the truthful result rather
   than an empty one.
5. **Everything is bounded, and every write is completed or accounted for.**
   Per-event payload bytes (`max_payload_bytes`) are clamped into
   `[0, 4096]`, default 256; total output bytes (`max_output_bytes`) are
   clamped into `[8192, 268435456]` (8 KiB–256 MiB), default 32 MiB — the
   floor exists so every file the tracer creates is large enough to hold a
   header and the two reserved records below, never a truncated fragment. A
   clamp is never silent: the header carries both what was requested
   (`requested_max_payload_bytes`, `requested_max_output_bytes`) and what is
   in force (`max_payload_bytes`, `max_output_bytes`), plus `limits_clamped`.
   The last `limit_reserve_bytes` (1024) of the output cap are held back for
   the two records that explain a truncated capture — the `limit` record
   announcing the stop and the closing `summary` record stating how many
   events were lost to it. Only those two may spend the reserve, which is what
   guarantees it is still free when they are written; ordinary events therefore
   stop short of the cap, not at it, and a capture that hit its cap still
   accounts for its own losses. Every write to the output file loops over
   short successful writes and `EINTR`, so a record the file accepts is
   written in full rather than as a partial prefix. An unrecoverable write
   failure is bounded rather than silent: the event is counted in
   `events_dropped` and a newline is written to resynchronise, so the damage
   is confined to one explicitly damaged line, which the decoder reports as a
   problem rather than letting it corrupt the rest of the stream.
6. **Identification does not depend on catching the open.** Hooking the open
   family is not sufficient on its own, and that is measured rather than
   precautionary: the vendor SDK imports `__open_2`, the `_FORTIFY_SOURCE`
   rewrite of `open()`, so an earlier build of this tracer that hooked only
   `open`/`openat` watched the SDK open the device and recorded *nothing*. The
   fortified spellings are now hooked too, but the real fix is the fallback: an
   untracked descriptor that receives a usbfs ioctl is adopted only if **both**
   halves of an identity test pass — its `st_rdev` (via `fstat()`) equals the
   target's, **and** its `/proc/self/fd/N` link (via `readlink()`) resolves to
   the exact canonical target path. Requiring both means adoption is never a
   weaker door into the capture than an intercepted open would have been, and
   it holds however the descriptor came to exist — a raw syscall, an inherited
   fd, or a spelling of `open()` that does not exist yet. It also fails closed:
   if `/proc` is not mounted, or the link cannot be read, nothing is adopted,
   and the capture falls back to whatever the open hooks saw. Such a descriptor
   is recorded with `"op":"adopt"` rather than `"op":"open"`, so the
   distinction stays visible in the data. `fstat()` and `readlink()` read inode
   metadata and path text; neither issues a USB operation.
7. **Reading user memory cannot crash the traced process.** Payloads are read
   with `process_vm_readv()` against our own pid, so a stale or unmapped
   buffer pointer yields `EFAULT` and a `payload_error` field instead of
   `SIGSEGV` in the middle of somebody else's program.

## Build

```bash
/home/ben/repositories/3d-scanner/reference/tracer/build.sh
```

Compiles clean under `gcc -std=gnu11 -O2 -g -Wall -Wextra -Wpedantic`. No SDK,
no vendor header, and no elevated privilege is needed to build or to run it.

## Run

```bash
SCANNER3D_TRACE_TARGET=/dev/bus/usb/001/041 \
SCANNER3D_TRACE_OUT=/tmp/3d-scanner-trace/capture.jsonl \
LD_PRELOAD=/tmp/3d-scanner-trace/libusbfs_trace.so \
    timeout 15 /tmp/3d-scanner-probe/probe --serial A22704522A6F00A64
```

Environment:

| Variable | Meaning | Default |
| --- | --- | --- |
| `SCANNER3D_TRACE_TARGET` | Exact usbfs node to record. Required. | unset → inert |
| `SCANNER3D_TRACE_OUT` | Output file. Must not already exist. Required. | unset → inert |
| `SCANNER3D_TRACE_MAX_PAYLOAD` | Payload bytes kept per event, clamped into `[0, 4096]`. | 256 |
| `SCANNER3D_TRACE_MAX_BYTES` | Total output bytes, clamped into `[8192, 268435456]` (8 KiB–256 MiB). | 33554432 |

An out-of-range value is clamped rather than rejected, and the value actually
requested is recorded in the header (`requested_max_payload_bytes`,
`requested_max_output_bytes`) next to the effective one, so a clamp is always
visible in the evidence. A value that is not a decimal number at all is not a
request: it is discarded and the default applies, and the header then records
the default as what was requested — the header cannot tell you that a variable
held garbage, only what limit was asked for and what is in force.

With neither of the two required variables set, the library loads and does
nothing: every hook is a straight passthrough. That is the intended state for
any process that happens to inherit it.

## Decode

```bash
ls /tmp/3d-scanner-trace/capture.jsonl.*        # one file per traced process
scanner3d trace /tmp/3d-scanner-trace/capture.jsonl.<pid>
scanner3d trace /tmp/3d-scanner-trace/capture.jsonl.<pid> --json
```

`scanner3d trace` is a pure file read: it never opens a device, never runs the
vendor SDK, and never sends anything to the scanner. It validates the header,
correlates URB submissions with their completions, groups transfers
structurally, and reports every failure, truncation, and unusable line rather
than dropping them.

## Trace format (version 1)

Line-delimited JSON. Line 1 is the header; every later line is one record.

```json
{"op":"header","format":"scanner3d-usbfs-trace","version":1,"target":"/dev/bus/usb/001/041",
 "target_rdev_major":189,"target_rdev_minor":40,"pid":12345,"ppid":12300,
 "path":"/tmp/3d-scanner-trace/capture.jsonl.12345","max_payload_bytes":256,
 "max_output_bytes":33554432,"requested_max_payload_bytes":256,
 "requested_max_output_bytes":33554432,"limits_clamped":false,"limit_reserve_bytes":1024,
 "clock":"CLOCK_MONOTONIC","started_realtime_ns":...,"limitations":["..."]}
```

Fields, in the order the header writes them: `format`, `version`, `target`
(canonical target path), `target_rdev_major`/`target_rdev_minor`, `pid`,
`ppid`, `path` (the file this process actually writes — the `<SCANNER3D_TRACE_OUT>.<pid>`
name), `max_payload_bytes`/`max_output_bytes` (effective, after clamping),
`requested_max_payload_bytes`/`requested_max_output_bytes` (what the
environment asked for), `limits_clamped` (true if either differs from what was
requested), `limit_reserve_bytes`, `clock`, `started_realtime_ns`,
`limitations`.

Every event carries `seq`, `t_ns`, `pid`, `tid`, `op`, with one exception: the
`limit` record for a byte-cap stop (`"reason":"max_output_bytes"`) is written
directly rather than through the normal per-event path and carries only `op`,
`reason`, `note` — see the `limit` row below. That record and the closing
`summary` are the only two allowed to spend `limit_reserve_bytes`.

| `op` | Meaning |
| --- | --- |
| `open` | A descriptor for the target became traced via an intercepted `open`-family call. Carries `fd`, `path`. |
| `dup` | A descriptor inherited tracking via `dup`/`dup2`/`dup3`. Carries `fd`, `from_fd`, `path`. |
| `adopt` | Same as `open`, but the descriptor was identified on its first usbfs ioctl rather than by an intercepted open: its `st_rdev` matched the target's **and** its `/proc/self/fd` link resolved to the target path. Carries `fd`, `path`, `note`. |
| `close` | A traced descriptor was closed, or replaced by `dup2`/`dup3` onto an already-tracked fd. Carries `fd`, and `note` when replaced rather than actually closed. |
| `ioctl` | One recorded call. Carries `fd`, `request`, `name`, `ret`, `errno`, `errno_name` (on failure), `duration_ns`. |
| `limit` | A cap was hit; later calls are **absent, not idle**. Two shapes: `"reason":"max_output_bytes"` (only `op`, `reason`, `note` — no `seq`/`t_ns`/`pid`/`tid`, since this record bypasses the normal per-event path so it is guaranteed to fit in the reserve) and `"reason":"fd_out_of_table"` (the usual common fields plus `fd`, `reason`, `note`). |
| `summary` | Written at **this library's** teardown: `events_written`, `bytes_written`, `events_dropped`. Not necessarily the last record — see below. |

`ioctl` records add fields by request:

- `USBDEVFS_SUBMITURB` (`"phase":"submit"`) — `urb` (pointer, the correlation
  key), `urb_type`/`urb_type_name`, `endpoint`/`endpoint_number`/`direction`,
  `status`, `flags`, `buffer_length`, `actual_length`, `start_frame`,
  `number_of_packets`, `error_count`, `signr`, `usercontext`. For control URBs
  the 8-byte setup packet is decoded into `bmRequestType`, `bRequest`,
  `wValue`, `wIndex`, `wLength`, `setup_direction`. OUT payload bytes are in
  `data` (lowercase hex) alongside `payload_available`, `payload_captured`,
  `payload_truncated`.
- `USBDEVFS_REAPURB` / `USBDEVFS_REAPURBNDELAY` (`"phase":"reap"`) — the same
  URB fields re-read at completion; IN payload bytes in `data`.
- `USBDEVFS_DISCARDURB` (`"phase":"discard"`) — `urb`.
- `USBDEVFS_CLAIMINTERFACE` / `USBDEVFS_RELEASEINTERFACE` — `interface`.
- `USBDEVFS_GETDRIVER` — `interface`, `driver` (the bound kernel driver name).
- `USBDEVFS_IOCTL` — `interface`, `inner_request`, `inner_name` (this is how
  libusb detaches and reattaches a kernel driver).
- `USBDEVFS_CONTROL` / `USBDEVFS_BULK` — the synchronous transfer ioctls,
  decoded with setup/endpoint and bounded `data`.

## Known capture limitations

Most of these are recorded in the header's `limitations` array, so they travel
with the data and are repeated in every decoded report; the two marked
"not in the header" below are structural facts about this tracer rather than
per-capture evidence, so they are not restated there:

- `seq` is **emission** order (assigned after the call returns); `t_ns` is
  call-entry time on `CLOCK_MONOTONIC`. A completion handled on one thread can
  therefore be written before the submission record of another. The decoder
  handles that case explicitly and marks such pairs `deferred`.
- Payload bytes are capped per event by `max_payload_bytes`;
  `payload_available` always records the untruncated length.
- Isochronous per-packet descriptors are not decoded.
- Only ioctls on file descriptors whose `/proc/self/fd` link equals the target
  path and whose `st_rdev` matches are recorded.
- The last `limit_reserve_bytes` of `max_output_bytes` are held back for the
  `limit` and `summary` records, so events stop short of the cap rather than
  exactly at it and a capped capture still states its own losses.
- URBs still in flight when the process exits have no reap record; the decoder
  reports them as `submit_only` rather than inventing a completion.
- A relative `openat()` against a directory descriptor is not reconstructed, so
  a target reached that way would not be traced (not in the header). libusb
  does not do this.
- The closing `summary` record is written when *this* library is torn down,
  which is not necessarily last. A library unloaded after this one — the vendor
  SDK's own static destructors, for instance — can still record events, so
  records may follow the summary and its `events_written` can undercount. The
  decoder's own counts are authoritative; this was observed in the real
  capture, where 33 further events followed the summary record.
- Diagnostics from a fail-closed condition go to the traced program's stderr
  (not in the header). That is deliberate (silence would be worse), but it
  does mean stderr is not byte-identical to an untraced run when capture is
  refused. stdout is never touched.
- Traffic from other processes, and from the kernel's own `uvcvideo`/`usbhid`
  drivers, is out of reach of this mechanism.
