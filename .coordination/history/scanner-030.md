# scanner-030 — accepted

- Accepted review round: `0`
- Physical GET_LEN attempts: `1`
- Physical blue-light-off attempts: `1`
- Evidence root: `/tmp/scanner-030-evidence.2Wd7Cq`

The fixed selector-0x12 GET_LEN ran exactly once with setup
`a185001200040200e8030000`. It returned exactly two bytes `0200` (SHA-256
`99be5efb88ca2013bd8e4eb035fd42d5245468fe9afa70d8ba9c1c419a48c4e8`),
so the raw little-endian reported length is 2. Release/reconnect passed. The
number has no assigned exposure, unit, validity, support, status or near-black
frame meaning.

The mandatory immutable `0x922=0` finalizer then ran exactly once as the last
device control transfer. Its exact 60-byte payload was accepted 60/60 and
release/reconnect passed. Three postflight samples over 2.541 seconds kept
devnum 47, all UVC/HID bindings and four video nodes stable with no holders.
This establishes command acceptance/restoration, not optical darkness.
