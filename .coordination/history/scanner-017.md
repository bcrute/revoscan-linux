# scanner-017 — accepted

- Accepted review round: `0`
- Ref: `unborn main`
- Evidence root: `/tmp/scanner-017-evidence.GNX5bU`

## Result

Exactly one bounded unprivileged physical paired invocation ran against MINI serial
`A22704522A6F00A64`. It completed without external timeout or retry and returned
application exit 5 because depth completed while RGB was strict-invalid.

Depth produced one complete 512000-byte `Y16 ` 640x400 observation with SHA-256
`950c9640612dd135b104bcdff1ef7536da435ca4eb6e34f7043ee58a5cd10c6a`.
No metric interpretation is claimed.

RGB produced an unchanged 22529-byte observation with SHA-256
`650240568e5b97fcc8f61773035e03e2b48fb5bb940c2d73de8572f83609c07d`.
It contains a baseline 1280x800 JPEG through EOI at offset 22523 followed by four
bytes `54 52 d8 01`; strict classification is `trailing_data`. The hypothetical
EOI-bounded prefix is 22525 bytes with SHA-256
`77ad1601dc234b059e5e076380ba8cb8c2b4f64f59c4f3fdb98ec2572b549199`, but no
derived file was written. Independent in-place viewing showed a nearly black image
with only faint features, not useful RGB content.

Both sequences were 0 and observed timestamp skew was 1003506000 ns. This does not
establish synchronization. Cleanup completed exhaustively in RGB-then-depth order.
Six postflight samples retained stable serial/path/nodes/bindings and no holders.

Preflight devnum 44 changed to 45 before postflight while stable identity and path
returned. The re-enumeration cause is unresolved.

The checksum manifest independently passed in full; its SHA-256 is
`870a370d5b7da5c003f2eeff2aeda960b93db49b58de2c7c205d4b4f61c5eea8`.

## Next branch

Create and offline-test one tightly bounded multi-frame paired RGB diagnostic.
Preserve every raw dequeue and classify a complete JPEG prefix plus opaque suffix
without accepting or silently trimming strict-invalid bytes. No further live run is
authorized by this history record.
