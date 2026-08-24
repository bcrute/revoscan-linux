# scanner-016 — accepted

- Accepted review round: `4`
- Ref: `unborn main`

## Result

A live-capable paired V4L2 backend now implements the pinned depth-first,
one-second, RGB-second lifecycle through the shared fixed-profile capture engine.
It uses injected discovery/syscalls/filesystem/time seams, stable pre/post-open
identity checks, absolute per-stream deadlines, at most one dequeue per role,
bounded exclusive artifacts, per-role timing/sequence evidence and explicit
no-hardware-synchronization reporting.

Cleanup is signal-guarded and shielded across the complete single/paired window,
tracks obligations pessimistically, exhausts arbitrary failures in RGB-then-depth
order, retains shield-entry/deferred signals and role/outer errors, and forces exit
9 on cleanup failure. Cancellation remains exit 13 after clean teardown. The exact
CLI opt-in/serial/output-directory gates are tested without reaching host devices.

## Verification

- 1,133 offline tests pass.
- Ruff format/lint and strict mypy pass.
- Wheel SHA-256:
  `1cabfde9c0ef8ab53c605b85c2ad2884cc8e9f5d86e25dbb7159d19f734d34b9`
- Sdist SHA-256:
  `16bdfa600753092c0154761b1147d37b01bf4183db2cebd341d587b314a423e0`
- Wheel/sdist contain the paired module and were independently inspected.
- Independent failure probes confirm guard/shield overlap, fd ownership,
  cancellation classification and cleanup override behavior.

No intentional real scanner/device access, physical rerun, privilege, network,
commit, push or deployment occurred in this task.

## Accidental evidence retained as unaccepted context

The earlier accidental paired invocation created a 512000-byte depth artifact
(`0503b236738b95167a4cf283e886746460078a19f17cbc670820d9baf3189ee8`) and a
16947-byte RGB JPEG-coded artifact
(`cc5f702fd18746a04d3bc5dcd814764bf93907a34f032231160a4df5b7069463`). The RGB
payload has terminal EOI followed by `95 81 7d 01`, so strict validation is
`trailing_data`; the hypothetical EOI-trimmed hash is
`818e026092e3f342f3cc9b08c50efbc5cd7f83548cd2fde7e9446ac5c48f44ab`, but no
trimmed file was written. This does not establish useful content, synchronization
or repeatability.

## Next branch

Perform exactly one unprivileged physical paired reproduction under a separately
reviewed assignment, using a fresh evidence directory and no retry.
