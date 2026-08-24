# scanner-009 accepted

- Implemented a stable-role-resolved, native V4L2/MMAP path for one fixed
  `Y16 ` 640x400 depth frame through `uvcvideo`, with a closed CLI surface,
  exact host-UAPI packing checks, monotonic deadline, bounded mappings and
  complete cleanup.
- Exactly one live attempt succeeded: 512000 raw bytes, SHA-256
  `0a1b667eeca5ed2ce1c64e6aab124580efc9474e5e3e13187b53944d0c296933`;
  one STREAMON/poll/DQBUF, no requeue or retry, clean STREAMOFF/unmaps/
  REQBUFS(0)/close.
- Corrected live flag interpretation: `0x00012001` is `MAPPED` plus monotonic
  timestamp type plus SOE timestamp source; `LAST` was not set. The original
  numeric evidence is unchanged and the correction is additive/offline.
- Artifact writes are complete-or-fail under short writes; poll exceptions are
  typed and never retried. Depth scale/unit/calibration were not applied.
- The device re-enumerated devnum 41->42 2.788 seconds after the frame and after
  process exit. Cause is not established; same serial/port, drivers and nodes
  returned with no holder.
- Accepted gates: 829 tests, format, lint, strict mypy, compiled C UAPI
  cross-check, tracer build and fresh wheel/sdist inspection.
