# scanner-022 — accepted

- Accepted review round: `1`
- Physical scanner invocations: `0`
- Independent build: `/tmp/scanner022-root-review.UQGNbe`

Offline static archaeology separated the two RGB exposure paths. Standard UVC
auto exposure reads Camera Terminal selector 2; the installed uvcvideo mapping
`[2, 1, 4, 8]` proves scanner-021 V4L2 index 3 is raw mode 8 and therefore maps
to SDK enabled. The standard exposure getter/range use Camera Terminal selector
4 and `log2f`. The physical UvcCamera extension override instead performs one
60-byte entity-4/interface-0 selector-0x12 read for IDs 0x15/0x16; its relationship
to standard V4L2 exposure remains unknown.

CameraType 12 and firmware v2.8.7 select the standard branch only when the base
fallback is reached. Under that conditional path, scanner-021 raw current 131072
would become 17 / 18000 us and raw range 1..60000 would become 1000..16000 us.
This does not normalize or explain the device-reported out-of-range value.

The accepted next boundary is an offline implementation/test task that adds a
separately gated read-only telemetry mode to the three-frame RGB series. A later
physical task may permit at most 12 fixed G_CTRL attempts at four sample points;
no setting, QUERYMENU, XU, retry, new wait or deadline reset is justified.

Acceptance passed 1,198 tests, Ruff, strict mypy and an independently inspected
fresh wheel/sdist build. Product runtime was not changed and no physical device
interface was accessed.
