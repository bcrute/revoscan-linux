# Scanner-044 accepted — offline watchdog and exit control flow

The final immutable analysis root is
`/tmp/scanner-044-analysis.tFwwmN`; input SHA-256
`decdc3ad4e34449cefee8543c790866cfc87dd9279a873c3abab124b9754b0f1`,
manifest SHA-256
`3baec0f01415f9f2485edef45fbfe7b0e162f7cf32e98a59526ef0e5bdd4db04`.
Root and independent Codex reproductions verified every selected artifact and
all 66 extracted disassembler files.

The binary has only two stores to watchdog flag `0x142b00`, and both write 1;
there is no value-2 setter. The state-2 branch passes raw address 1 to
`WDIOC_SETOPTIONS` despite the generic v4.19 `int *` ABI, ignores the result,
then reaches the shared close. No mapped normal, signal, abort, pthread-exit,
main-return, init/fini, or atexit path proves successful disable-before-exit.

Decoded direct import coverage contains 2,327 ordinary calls plus 63 tail
transfers, including 53 close and 34 system transfers. Of all 53 close
arguments, only `0x2d7d4` is proven to consume the watchdog open result; the
other 52 remain explicitly unresolved. Exact unit `nowayout`, device-tree
reset binding, magic-close outcome, and live supervision remain unknown, so no
process signal or teardown is authorized by this task. The emitter remains
presumed on.
