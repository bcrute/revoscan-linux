# scanner-003 accepted summary

- Review round accepted: 2
- Added tolerant VideoControl entity parsing over the real 645-byte MINI
  descriptor fixture.
- Added an opt-in `scanner3d firmware-version` command whose hardware surface is
  one fixed read-only UVC request and whose default mode is dry-run.
- Confirmed entity 3 is an output terminal; entity 4 is the declared extension
  unit.
- Issued the request once against serial `A22704522A6F00A64`; it returned
  `EBUSY` before reaching the wire because `uvcvideo` owns interface 0.
- Accepted gates: 214 tests, format, lint, strict mypy, warning-free tracer
  build, and distribution inspection.
- Next step requires owner authority to transiently change interface-0 driver
  binding, with explicit reconnect and replug fallback.
