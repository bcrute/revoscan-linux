# scanner-007 accepted

- Implemented a closed-whitelist `/data/camparam/Pl.bin` read on the accepted
  interface-0 lifecycle: one fixed key SET, at most eight 60-byte chunk GETs,
  and one mandatory zero clear, with 256-byte capacity and no retries.
- The one live run completed in 37.488 ms with clean protocol and driver undo.
  It returned one 40-byte chunk then a zero terminator; blob SHA-256
  `fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf`.
- Parsed factory depth intrinsics for 640x400:
  `fx=fy=898.6829833984375`, `cx=42.43747329711914`,
  `cy=192.8326873779297`, fixed matrix entries `0,0,0,0,1`.
- The off-centre `cx` is preserved and flagged as a plausibility observation;
  no metric accuracy, coordinate convention, RGB alignment, or depth-scale
  claim is inferred.
- Final device state: interfaces 0-3 on `uvcvideo`, interface 4 on `usbhid`,
  nodes restored, no holder/process. No replug was needed.
- Accepted gates: 522 tests, format, lint, strict mypy, tracer build, fresh
  wheel/sdist and package-content inspection.
