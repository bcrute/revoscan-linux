# scanner-057 accepted review

The isolated b01-disable helper was accepted at SHA-256
`75fd909049ca3bb9ee3c785bc81e043148be5915e7446d45acd1af70acb38444`;
its focused test hash is
`46178589e5b11346de9f87181f0621c5524ded2fa63e60d872789aef006564db`.
SDK disassembly independently proves IR id 0 `DISABLE_LED` emits exact command
`echo s 0xb01 0 > /dev/rk_preisp`, while `ENABLE_LED` emits b01=1. The helper
contains only the disable command. No live action occurred in scanner-057.
