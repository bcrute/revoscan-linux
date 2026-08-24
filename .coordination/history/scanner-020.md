# scanner-020 — accepted

- Accepted review round: `1`
- Physical diagnostic invocations: `0`
- Independent build: `/tmp/scanner020-root-review.QcX5EL`

Implemented and offline-verified a dry-by-default, fixed-whitelist, read-only
standard V4L2 RGB-control inventory. The only possible device requests are one
`VIDIOC_QUERYCTRL` for each of six fixed gain/exposure/white-balance IDs and at
most one `VIDIOC_G_CTRL` for each readable understood control, capped at 12 total.
No control write, menu query, format/buffer/stream operation, depth access or
vendor transport exists in the module.

Lifecycle and CLI evidence cover exact opt-in/serial/parser gates, stable RGB
resolution, open/query/get signals and errors, returned-ID validation, typed
flags and partial observations, continuation without retry, exactly-once cleanup,
cleanup precedence and complete post-close RGB/uvcvideo identity. Acceptance
passed 1,198 offline tests, Ruff, strict mypy and a fresh wheel/sdist build. The
new diagnostic has not yet been run on the physical scanner.
