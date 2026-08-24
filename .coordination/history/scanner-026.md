# scanner-026 — accepted

- Accepted review round: `2`
- Physical scanner invocations: `0`
- Independent build: `/tmp/scanner-026-round2-final2.P2SZL2`

Mandatory automatic blue-light finalization now follows every existing depth
STREAMON attempt. The obligation is raised before STREAMON, all V4L2 roles are
torn down through close first, and exactly one immutable 60-byte `0x922=0`
SET is then attempted through the shared interface-0 ownership lifecycle.
Finalizer construction through restoration is termination-shielded, including
a fresh-shield entry-signal fallback, and deferred signals remain evidence.

Exact command acceptance and complete restoration are separate invariants.
Malformed or contradictory shutdown records cannot produce success; a required
failed write or cleanup blocks terminal success and produces nonzero recovery
output without erasing capture/artifact/telemetry truth. RGB-only, dry,
refused and pre-STREAMON paths construct no shutdown backend. No on,
brightness, LED, GET, arbitrary request or retry surface exists.

Acceptance passed 408 focused and all 1,285 offline tests, Ruff format/check,
strict mypy and an independently inspected fresh wheel/sdist. No physical
scanner interface was accessed, and no optical-darkness claim is made.
