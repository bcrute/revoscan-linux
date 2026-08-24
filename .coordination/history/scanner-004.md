# scanner-004 accepted summary

- Review round accepted: 1
- Added a fixed interface-0 ownership lifecycle: confirm `uvcvideo`, disconnect,
  claim, one whitelisted read, release, reconnect.
- The one authorized live run succeeded and returned the exact known 60-byte
  firmware/algorithm payload in 1.266 ms. Cleanup succeeded; no repeat occurred.
- Measured detach effect: depth nodes recreated; RGB nodes untouched; no USB
  reset or device re-enumeration.
- Review found asynchronous cleanup gaps; the accepted implementation treats
  disconnect/claim completion conservatively and shields cleanup from
  SIGINT/SIGTERM without retrying phases.
- Accepted gates: 307 tests, format, lint, strict mypy, tracer build, fresh
  wheel/sdist build and content inspection.
- Next observed useful sequence contains transient device-directed `SET_CUR`
  writes and requires separate owner authorization.
