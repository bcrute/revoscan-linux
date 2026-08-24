# scanner-012 — accepted

- Accepted review round: `0`
- Ref: `unborn main`

## Result

Offline reconstruction decoded the complete RGB UVC streaming path and compared
the successful depth run with both RGB attempts. RGB uses VS interface 3,
endpoint `0x84` IN bulk, only alt 0, format/frame 1/1 and a 26-byte UVC 1.00
Probe/Commit block. Bandwidth/alt-setting, wrong-node and device-quirk theories
are eliminated; missing S_PARM is largely eliminated.

The host uvcvideo module had `trace=0`, suppressing probe/commit, URB and frame
decode diagnostics. `quirks=4294967295` was corrected to its compiled unset
sentinel. The pinned SDK starts RGB with standard Probe/Commit/bulk transfers
and no automatic XU/vendor command in the scoped start chain.

## Verification

- 1010 tests; format, lint and strict mypy pass.
- Wheel SHA-256:
  `2aaddb5f4ed75b2f4678515c002c83631659d9db61e4b537901187054e7e8e4d`.
- Sdist SHA-256:
  `2b71e8dfc02ef7599edbd008366291eaeeccca58c37c70a3c1c843dc8337b4db`.

## Next experiment and blocker

RGB-1 would temporarily set the root-owned host module parameter
`/sys/module/uvcvideo/parameters/trace` from 0 to `0xFFFFFFFF`, run the unchanged
bounded RGB attempt exactly once with capped dmesg evidence, and restore/read
back 0 on every path. It does not mutate the scanner, but elevation needs
explicit owner authorization.
