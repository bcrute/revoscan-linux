# Scanner-039 accepted — RK1608 shutdown ownership recovered

PID 569 `/tmp/zxmain` is the live vendor controller and durable holder of
`/dev/rk_preisp`. Its exact 166,412-byte ARM ELF (SHA-256
`decdc3ad4e34449cefee8543c790866cfc87dd9279a873c3abab124b9754b0f1`)
contains active calls to `system("echo off >/dev/rk_preisp")`, including an
RK1608 off/on cycle in the UVC stream-control path.

The matching upstream Rockchip driver maps textual `off` to a reference-counted
RK1608 hardware power-down. On transition one-to-zero it disables IRQ and MCLK
and drives RK1608 wakeup/reset low. The vendor init stop path has no cleanup,
and BusyBox `poweroff` still has no proven board hook. The exact `preisp.rkl`
firmware was loaded successfully at boot but is absent from all searched live
filesystem paths.

Do not query `/dev/rk_preisp` after shutdown: opening it at count zero invokes
the full power-on path. The next one-shot intervention uses only the exact
vendor `off` command, then passive kernel-log and GPIO evidence. Optical
darkness remains pending the owner's return.
