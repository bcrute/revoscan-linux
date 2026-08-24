# scanner-011 — accepted

- Accepted review round: `0`
- Ref: `unborn main`

## Result

Raised only the fixed RGB mapping/artifact ceilings to the scanner-010 driver
measurement (`8192000` bytes each; `32768000` total) and spent exactly one new
live attempt. The path reached four mapped/queued buffers and STREAMON, but one
8.007-second poll returned no ready buffer. Result: `poll_timeout`, exit 4,
zero frame bytes and no artifact.

Cleanup stopped the stream, unmapped all buffers, released the queue and closed
the fd. A 60-second observation found stable devnum 42, nodes, bindings/modules,
no kernel delta and no video holder. Evidence is frozen under
`/tmp/scanner-011-evidence/`.

## Verification

- 951 tests; format, lint and strict mypy pass.
- Final wheel SHA-256:
  `2aaddb5f4ed75b2f4678515c002c83631659d9db61e4b537901187054e7e8e4d`.
- Final sdist SHA-256:
  `ea9bbfa57b79b8ed5047e42bf579302947727ce03bb8cbc735749d8c5c34cda1`.

## Limits

No RGB/JPEG evidence exists. The cause of the silent stream is unknown; a
longer timeout, missing standard control, vendor/XU initialization, bandwidth
or another cause is not established by this run.
