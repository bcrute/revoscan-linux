# Scanner-040 accepted — exact RK1608 decrement did not power hardware off

The exact vendor `echo off >/dev/rk_preisp` command was issued once through the
established entity-4 bridge after complete identity, holder, descriptor, GPIO,
and binding preflight. It returned 60/60 bytes and the kernel added exactly one
`do cmd power off, count--` line.

GPIO121 reset remained high and its complete before/after snapshots were
byte-identical. The shared RK1608 reference count therefore decremented without
entering the physical one-to-zero shutdown path. Cleanup passed, devnum 49 was
stable, interface 0 was left unbound, and no later scanner transfer occurred.

Evidence root: `/tmp/scanner-040-evidence.AHnFJl`; manifest SHA-256
`4eb55f85c2a437cc3bb6dac9baf7f83442f49d1a7a7a6ab073cacf36599b7a8f`.
The emitter remains presumed on. A separately reviewed second decrement is the
only next live action; no automatic retry, process signal, or third decrement
is authorized.
