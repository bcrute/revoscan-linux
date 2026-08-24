# scanner-015 — accepted

- Accepted review round: `2`
- Ref: `unborn main`

## Result

Scanner-014's 26 target usbmon records are now a compact provenance-complete
fixture with a deterministic decoder. The decoder tolerates padded/unpadded
device numbers, pairs URBs one-to-one, decodes 26-byte UVC Probe/Commit fields,
reports unmatched records, and keeps completion status, actual byte count and
captured text prefix as separate facts. It reproduces seven successful control
pairs, five 16384-byte bulk submissions, five cleanup cancellations with zero
actual bytes, and no successful RGB transfer or frame.

A backend-agnostic `paired-capture` lifecycle and dry-run CLI now encode the
pinned 3DViewer application's depth-first/RGB-second ordering, the one-second
SDK interval, exact serial/opt-in gates, fixed depth/RGB profiles, frame and
memory caps, per-role deadlines, no retry, and reverse teardown. Map/queue and
STREAMON obligations are pessimistic across exception/signal boundaries. The
CLI intentionally has no live backend yet and makes no physical-support claim.

## Verification

- 1050 tests pass.
- Ruff format and lint pass.
- Strict mypy passes for 43 source files.
- Wheel SHA-256:
  `bf01813dbfa31b532d2ae2abeecb594fa3fe2b8fd1d2d5642967771d67bcbed9`
- Sdist SHA-256:
  `60dd46602ff550fb5bce389b0d1edf4eed57ecbeea6a348cc97cb4f59e0cf676`
- Wheel/sdist contents independently inspected.

No device node, scanner, usbfs, vendor SDK, privilege, module, trace, network or
external-system action occurred.

## Next branch

Implement a concrete dual-stream V4L2 backend using the reviewed lifecycle and
existing single-frame safety engine, entirely against injected/fake syscalls.
Only after that implementation receives Codex review should a separately
bounded physical paired-stream attempt be assigned.
