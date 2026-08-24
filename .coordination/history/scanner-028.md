# scanner-028 — accepted

- Accepted review round: `1`
- Physical scanner invocations: `0`
- Focused tests: `45 passed`

Offline reconciliation proved the pinned SDK asks for one 60-byte entity-4
selector-0x12 GET and then loses libusb's actual transferred-byte count by
mapping every nonnegative value to wrapper success. Because its destination is
zero-filled, scanner-027's measured 2/60-byte `0000` reply could flow into SDK
zero decodes, but it is not a valid four-word response and carries no proven
exposure, unit, enable, success, failure or valid-zero meaning.

The relevant selector-0x12 call graph and both range layouts are recorded with
addresses. The descriptor fixture shows selector 18's bitmap bit clear, so it
is unadvertised rather than proven reserved or unsupported. The next bounded
direction is a fixed offline-first `GET_LEN` reader, followed in any later
physical task by exactly one mandatory immutable off finalizer.

No runtime source/test/reference file changed: root byte-compared all 68
non-cache files with the accepted scanner-026 sdist. No scanner/device runtime
was accessed; the last physical action remains the accepted scanner-027 off
write and stable restoration.
