# scanner-010 — accepted

- Accepted review round: `1`
- Ref: `unborn main`

## Result

Implemented a shared immutable-profile V4L2/MMAP engine serving the accepted
depth path and a new stable-role RGB path, plus `scanner3d rgb-frame` and a
bounded dependency-free structural JPEG inspector. Exactly one live MJPG
1280x800 attempt ran.

The driver retained the requested mode but returned `sizeimage=8192000`, above
the descriptor-derived 2097152-byte ceiling. The command returned bounded
`format_mismatch` before buffer allocation or streaming, closed cleanly, and
did not retry. No RGB frame was captured. A 45.06-second observation showed
stable devnum 42, nodes and bindings, no holder and no kernel event.

## Verification

- 944 tests, format, lint and strict mypy pass.
- Evidence: `/tmp/scanner-010-evidence/`.
- Final wheel SHA-256:
  `bd464f242f89d6f3da1618652c6a054ff28a50abdacda7f97e1bc2043a0420cf`.
- Final sdist SHA-256:
  `6a1e5386e92756bd1b1af2535113e52d604317856dacadfc20f281dc2799b2c0`.

## Limits

Zero RGB bytes were dequeued, so JPEG content, color, calibration and RGB/depth
synchronization remain unmeasured. The scanner-009 re-enumeration cause remains
unresolved; scanner-010 never streamed and had no comparable event.
