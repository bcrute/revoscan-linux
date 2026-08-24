# Scanner-051 accepted-partial — remote PID identity drift

One exact invocation of frozen helper
`8b630434a44a1059655618d3a50b47351094c45715e3ae4390fb5ff4c8e0b57c`
ran from a genuinely empty root at
`/tmp/scanner-049-evidence.FbApS9SZfcAb`. Manifest SHA-256 is
`8ccc62306db61b437574a38bd88d66668e05f03b11d992290252502db3d44de3`.

GETDRIVER, detach and claim each succeeded once. The first required
`/proc/569/stat` read completed in six chunks plus terminator and exact clear,
then reported `569 (watchlog.sh)` with starttime 1004 rather than the pinned
`zxmain`/998 identity. The helper stopped before all 54 direct paths and twelve
commands, released/closed successfully, and returned `final_ok=false`.

Root and independent Codex validated the complete twenty-artifact chain and
237-byte blob SHA-256
`d8503875cf54c2b637febbb1bae62afc112384f0e5542667c259a556fa4dacd6`.
The same devnum 50 remained stable with interfaces 0/1 unbound and 2/3/4 still
owned through the passive window and later root observation. No retry,
reconnect, shutdown, power or optical test occurred. The process-table change
does not by itself establish a reset cause.
