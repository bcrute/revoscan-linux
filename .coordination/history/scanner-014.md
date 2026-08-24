# scanner-014 — accepted

- Accepted review round: `1`
- Ref: `unborn main`

## Result

Exactly one unchanged MJPG 1280x800 RGB capture ran under a bounded bus-specific
usbmon observer. Probe and Commit both completed successfully with 26-byte UVC
1.00 blocks. The negotiated GET_CUR/Commit state selected format/frame `1/1`,
interval `333333`, `dwMaxVideoFrameSize=8192000`, and
`dwMaxPayloadTransferSize=524288`.

The driver submitted five endpoint `0x84` bulk-IN URBs of 16384 bytes each. No
URB completed successfully during the 8.008-second poll. All five completed only
during cleanup with status `-2` and actual length `0`; consequently this run
captured no UVC header, JPEG byte, V4L2 buffer, or RGB artifact. Probe/Commit
inconsistency and host-side failure to submit URBs are eliminated for this run.
Missing device-side RGB production or a required higher-level start ordering is
the leading branch.

Capture cleanup succeeded, the usbmon follower was collected, the temporary
usbmon module was unloaded, trace remained `0`, and serial/devnum/nodes/bindings/
holders stayed stable through 50 seconds.

## Evidence and limitation

- Evidence root: `/tmp/scanner-014-evidence/`
- Raw usbmon: 810 lines / 58074 bytes, SHA-256
  `15b43371f426c3efb617ce58d805b9f9c04519562fb641042dfd5259b9e2d664`
- Preserved wrapper: `/tmp/scanner-014-root-wrapper`, independently reviewed
  SHA-256
  `689f6d234035ff863c7edff3ab4969a16bcb3edd61eb9dbd49a34867fabe4441`
- Every artifact listed by the root-owned evidence manifest verifies.

The wrapper filtered on unpadded devnum `43`, while usbmon rendered `043`, so
the root-owned filtered and decoded derivative files are empty. The immutable
raw stream contains exactly 26 target lines, and both the coder report and Codex
review reproduce every accepted field/status directly from those lines. The
wrapper hash is not chained into the root-owned manifest. These are medium
evidence-packaging defects, not reasons to rerun a physical attempt.

## Next branch

Freeze the target usbmon observation as an offline regression fixture, add a
padding-tolerant decoder, and recover/implement the vendor application's actual
higher-level start ordering. The published 3DViewer source starts depth before
RGB; this safer standard-UVC hypothesis precedes an undocumented XU/register
write.
