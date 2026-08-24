# scanner-024 — accepted

- Accepted review round: `0`
- Capture CLI invocations: exactly `1`; no retry
- Preflight device-node opens: exactly `2`, read-only, no ioctl
- Evidence: `/tmp/scanner-024-evidence.VMc3vD`

One unprivileged paired run delivered a complete 512000-byte depth artifact and
three preserved RGB payloads. Each RGB payload had a decodable 1280x800 JPEG
prefix plus four opaque trailing bytes and remained nearly black. The command
therefore truthfully exited 5 with strict `trailing_data` evidence.

The accepted scanner-023 telemetry path completed all four points and all twelve
fixed G_CTRL reads. Gain 16, auto-exposure index 3 and absolute-exposure raw
131072 were invariant before the first RGB poll and after each delivered frame;
all requested/returned IDs matched and no telemetry error occurred. These raw
values do not establish exposure units or the separate selector-0x12 state.

Cleanup was exhaustive and six postflight samples were stable. Devnum changed
46 to 47 without an attributed cause. The evidence manifest, JSON equivalence,
artifact hashes, trailer bytes and decoded images were independently reviewed.
No setting, extension-unit/vendor request, elevation or retry occurred.
