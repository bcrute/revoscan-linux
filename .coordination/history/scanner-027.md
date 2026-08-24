# scanner-027 — accepted

- Accepted review round: `0`
- Physical selector reads: `1`
- Physical blue-light off writes: `1`
- Evidence root: `/tmp/scanner-027-evidence.sp2Y0b`

The attached MINI's fixed entity-4/interface-0 selector-0x12 GET_CUR was invoked
exactly once with no retry. Setup `a181001200043c00e8030000` returned only 2
of the requested 60 bytes: raw `0000`, SHA-256
`96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7`.
The command typed this as `short_response`/exit 7, retained the raw evidence,
decoded no u32 words and made no exposure-unit or near-black-frame claim.
Release/reconnect passed.

The mandatory final-state `0x922=0` off command then ran exactly once. It was
accepted 60/60 bytes in 1.572 ms; cleanup passed. Three postflight samples over
2.205 seconds retained devnum 47, all UVC/HID bindings and video nodes, and no
holders. All evidence hashes verify. There is no optical readback.
