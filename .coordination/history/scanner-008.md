# scanner-008 accepted

- Implemented a closed two-transfer read of property `0x918`: fixed SET_CUR
  `21 01 wValue=0x0e00 wIndex=0x0400 wLength=4` carrying `18090000`, then one
  60-byte GET_CUR on the same selector. No retry and no invented protocol clear.
- The one live run returned marker 4 and signed denominator 200. Raw reply
  SHA-256 is
  `9873fa193adabb7daa87a77718d963d586d798838f4e6edc1a2ddac0bb337dd4`.
- Exact vendor binary32 decode:
  `float32(1.0f/200) = 0.004999999888241291`, bits `0x3ba3d70a`. This matches
  the SDK's MINI fallback bit-for-bit as an observation only; the physical unit
  and metric accuracy remain unknown.
- Release/reconnect succeeded; interfaces 0-3 returned to `uvcvideo`, interface
  4 remained on `usbhid`, nodes were restored, and no holder/process survived.
- Accepted gates: 630 tests, format, lint, strict mypy, tracer build, fresh
  post-edit wheel/sdist and package-content inspection.
