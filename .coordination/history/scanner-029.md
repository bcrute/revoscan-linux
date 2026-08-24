# scanner-029 — accepted

- Accepted review round: `1`
- Physical scanner invocations: `0`
- Accepted artifacts: `/tmp/scanner-029-round1-final2.LWUBbz`

One fixed, dry-by-default UVC `GET_LEN` boundary now exists for entity 4,
interface 0, selector `0x12`. Its exact live setup is
`a185001200040200e8030000`; the serial-gated CLI can perform at most one
two-byte read with no retry, SET, alternate request, protocol clear, V4L2,
HID, illumination or arbitrary-command path.

Only an exact two-byte reply is decoded little-endian as the raw reported
control length. The implementation retains exact wire, raw, digest, timing,
identity, typed failure and cleanup evidence and assigns no support, exposure,
unit, validity or status semantics to the number. Dry/wrong/missing-serial
paths are inert.

Acceptance passed all 1,314 offline tests, Ruff format/check, strict mypy,
isolated installed-wheel help/dry checks and exact artifact comparisons. Wheel
SHA-256 is `44b090f7543814d6c7a97ed6919c34b19c18a45edeae9fac68cef5ad697742b8`;
sdist SHA-256 is `bdf9fde07be91692803dbeb1cb07889a36d7d6d3805b7d0444597aea63116044`.
No physical scanner access occurred.
