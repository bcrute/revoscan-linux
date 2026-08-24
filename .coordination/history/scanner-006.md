# scanner-006 accepted summary

- Review round accepted: 3
- Statically traced the pinned, unstripped vendor ELF across streaming, mode,
  illumination/HID, calibration, and host-library authorization paths.
- Recovered exact XU command (`0x0700`), read-by-id (`0x0e00`), and file-read
  primitives; the latter explains scanner-005's length chunk, terminator, and
  cleanup and identifies fixed factory calibration keys.
- Corrected depth-scale semantics: command id `0x918` is payload bytes
  `18 09 00 00`; fixed `wValue=0x0e00`; one 4-byte SET and one 60-byte GET.
- Standard streaming acts on VS interface 1/3, not the accepted VC-interface-0
  lifecycle. Direct streaming remains NOT READY pending descriptor and recovery
  work; V4L2 remains the lowest-risk streaming route.
- Bounded future calibration read: 256-byte output, at most eight GETs, chunk
  length ≤56, one mandatory zero clear, cleanup failure overrides success.
- Added an offline static-analysis helper and 20 focused tests. Accepted gates:
  417 tests, format, lint, strict mypy, and fresh package inspection.
- No scanner or SDK execution occurred in this milestone.
