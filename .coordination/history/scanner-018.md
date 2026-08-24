# scanner-018 — accepted

- Accepted review round: `3`
- Scope: offline implementation and verification only
- Physical scanner access: none

Implemented the fixed `paired-rgb-series` diagnostic on the shared paired V4L2
backend. It keeps depth-first start, the exact one-second RGB start interval, one
depth dequeue, one absolute eight-second RGB deadline and exactly three bounded
RGB observation opportunities. Every copied RGB payload is represented with
indexed exclusive preservation, V4L2 metadata, SHA-256, strict JPEG status,
EOI-prefix facts and bounded opaque-trailer facts; no trimmed derivative exists.

Signals, partial progress, preservation failure and ordinary transport failures
retain cumulative typed evidence and exhaustive RGB-then-depth teardown. Byte
delivery, artifact success, strict JPEG validity and command success remain
separate facts. The legacy one-frame command remains compatible, and no series
path creates `rgb-frame.jpg`.

Independent acceptance evidence: 1,158 tests passed; Ruff format/lint and strict
mypy passed; the fresh wheel and sdist had 23/68 members and SHA-256 values
`94eca53d9abb9dbe2b70c52fa04b6fddd096fd1457e3291313d12a9138617994` and
`b13b0cce8efd337164dab701ade33056b3d7d4e2ece89eaacb827993d0fc6a3f`.

This milestone makes no physical multi-frame, useful RGB, suffix-meaning,
auto-exposure, synchronization or accuracy claim.
