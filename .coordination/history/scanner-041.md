# Scanner-041 accepted — second RK1608 decrement also stayed powered

One separately reviewed second `echo off >/dev/rk_preisp` command ran after
strict immutable-evidence, identity, process, holder, log, GPIO, and binding
preflight. The 60-byte transfer completed exactly once and the kernel added one
new `do cmd power off, count--` event.

GPIO121 reset remained high; its complete before/after snapshots were
byte-identical. Release and fixed host-sysfs postflight passed, devnum 49 stayed
stable, interface 0 was left unbound, and no later scanner transfer occurred.
Evidence root `/tmp/scanner-041-evidence.F3ZTLb`; manifest SHA-256
`56fb90433b85aebf253295c9d48d08534b40035f7b17891166fe50b9718e9f49`.

Two accepted decrements without physical shutdown imply an original shared
count of at least three under unchanged/no-race semantics. The emitter remains
presumed on. Do not issue a third blind decrement or signal the watchdog-owning
vendor process; audit the actual misc/V4L2 reference owners first.
