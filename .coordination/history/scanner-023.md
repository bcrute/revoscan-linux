# scanner-023 — accepted

- Accepted review round: `1`
- Physical scanner invocations: `0`
- Independent build: `/tmp/scanner023-root-review.DxS4MY`

An offline-tested, separately gated in-stream RGB telemetry mode now reads only
gain `0x00980913`, exposure-auto `0x009a0901` and exposure-absolute
`0x009a0902` through `VIDIOC_G_CTRL` on the existing RGB fd. It samples once
after STREAMON before the first poll and after each copied/preserved dequeue
before requeue, for at most four points and exactly twelve attempts on a complete
three-observation run. It adds no retry, sleep, deadline extension, open, stream,
setting, query-menu, extension-unit or vendor action.

Telemetry values, returned-ID mismatch, ordinary ioctl error, interruption and
unexpected exception are typed independently from capture status. Partial points
and frame associations survive failures and cleanup. The opt-in, serial, mode and
output gates remain explicit, while telemetry-off paths issue zero G_CTRL calls.

Acceptance passed 36 focused tests, all 1,209 offline tests, Ruff, strict mypy
and an independently reproduced wheel/sdist build. No physical device interface
was accessed; physical behavior remains unclaimed until a separate bounded run.
