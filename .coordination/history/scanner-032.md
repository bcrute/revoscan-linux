# scanner-032 — accepted

- Accepted review round: `1`
- Physical GET_INFO attempts: `1`
- Physical blue-light-off attempts: `1`
- Evidence root: `/tmp/scanner-032-evidence.vu83Lm`

One fixed selector-0x12 GET_INFO returned exact raw `03` (SHA-256
`084fed08b978af4d7d196a7446a86b58009e636b611db16211b65a9aadff29c5`).
Linux-UAPI syntax marks GET and SET, with other defined/reserved bits clear;
this does not establish actual SET acceptance or exposure semantics. Cleanup
passed.

The mandatory off finalizer then accepted the exact payload 60/60 as the last
device transfer. Cleanup and three postflight snapshots passed with devnum 47,
all nodes/bindings and no holders. There is no optical readback.

Evidence limitation: attempt markers were mode `0664` during review, not the
assigned read-only mode. They were corrected post hoc to `0444`; both original
and replacement manifests verify, and the correction record explicitly says
historical immutability cannot be proved. No action was rerun.
