# scanner-013 — accepted

- Accepted review round: `4`
- Ref: `unborn main`

## Result

Exactly one RGB-1 attempt ran with all uvcvideo trace bits enabled. It again
timed out after one 8.007-second poll with no dequeue or RGB artifact, but the
trace proved successful URB completion with valid header-only UVC payloads and
an FID transition. The dumped prior frame contained one total packet, zero
non-empty packets and no PTS/SCR; no JPEG bytes or V4L2 frame completed.

Trace restored `0`, all capture resources released, 50 seconds of observation
kept serial/devnum/nodes/bindings/holders stable, and all evidence hashes pass
under `/tmp/scanner-013-evidence/`.

## Next branch

The kernel has no successful Probe/Commit byte or generic successful-completion
trace callsite. Use bounded usbmon observation of one unchanged attempt before
issuing a device pre-roll command or proposing a decoder/quirk change.
