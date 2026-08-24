# scanner-025 — accepted

- Accepted review round: `1`
- Physical scanner invocations: `0`
- Independent build: `/tmp/scanner025-root-review.tN73S8`

A dry-by-default `rgb-exposure-extension` command now models only the pinned
SDK's fixed entity-4 selector-0x12 GET_CUR: one 60-byte read, 1000 ms timeout,
no SET, retry, clear or arbitrary wire surface. The future live flag reuses the
accepted interface-0 detach/claim/release/reconnect lifecycle; default execution
constructs no backend.

Exact replies retain setup, requested/actual lengths, raw bytes and SHA-256.
Four little-endian u32 words are reported first. The separate SDK-observed
mapping (`word0/1/2*1000`, fixed step 1000) remains labelled unit-unknown,
word3 is logged-only and contradictory ranges remain typed warnings.

Acceptance passed 22 focused tests, all 1,231 offline tests, Ruff, strict mypy
and an independently reproduced wheel/sdist build with archive inspection.
No physical device interface was accessed, so selector-0x12 behavior and values
on the attached MINI remain unclaimed pending a later bounded task.
