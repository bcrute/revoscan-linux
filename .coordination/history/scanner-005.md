# scanner-005 accepted summary

- Review round accepted: 0
- Added one closed-whitelist operation for the SDK-observed four-transfer
  entity-4 `/tmp/inited` exchange, reusing the accepted interface-0 lifecycle.
- The one authorized live run matched all setup, payload, and response bytes:
  algorithm `v7.5.7.0426`, then 60 zeros; both writes accepted 60 bytes.
- Protocol clear, release, and reconnect succeeded in a 39.743 ms detached
  window. Depth nodes were recreated; all drivers and nodes are restored.
- Cleanup is conservative across asynchronous interruption, independently
  reports protocol and driver restoration, and never retries a transfer.
- Accepted gates: 397 tests, format, lint, strict mypy, warning-free tracer
  build, and fresh wheel/sdist inspection.
- Next step is offline archaeology of the unobserved post-authorization stream,
  mode, projector, and calibration protocol; no authorization bypass is allowed.
