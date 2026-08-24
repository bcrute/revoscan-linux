# scanner-031 — accepted

- Accepted review round: `1`
- Physical scanner invocations: `0`
- Accepted artifacts: `/tmp/scanner-031-round1-final.g5Yn5q`

A dedicated dry-by-default selector-0x12 UVC GET_INFO boundary now permits only
setup `a186001200040100e8030000`: one byte, entity 4/interface 0, timeout
1000 ms. It is exact-serial gated, has one attempt and no retry, alternate
request, protocol clear, setting, V4L2, HID, vendor or illumination path.

An exact byte is retained raw and rendered through Linux-UAPI GET/SET/DISABLED/
AUTOUPDATE/ASYNCHRONOUS bit syntax with reserved bits. No bit is treated as
proof of selector advertisement, exposure semantics, current validity, actual
SET acceptance, units or frame behavior. Acceptance passed all 1,349 offline
tests and installed artifact checks. No scanner access occurred.
