# scanner-056 live result

The accepted scanner-055 helper was invoked exactly once at
`/tmp/scanner-055-evidence.ALVrdn`. Manifest SHA-256 is
`4a197ad6b99e77f57212393535b000c9fe345311dd5cbba697bfd01c2877f084`.
The pinned device stayed bus 1/devnum 50, rdev `189:49`. The sole exact
`echo s 0xb00 0 > /dev/rk_preisp` SET_CUR was physically submitted and accepted
60/60 bytes. Claim/release/close, final detached topology, identity and evidence
chain all passed. The owner subsequently observed that the front blue emitter
was still on. Therefore b00 command acceptance did not shut down the visible
emitter and must not be repeated as that actuator.
