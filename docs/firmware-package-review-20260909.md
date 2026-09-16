# Original MINI: static official-package investigation

Started September 9, 2026, 07:58 UTC. Evidence:
`/home/ben/scans/diagnostics/20260909-official-package-review/`.
This extends the [earlier public-page review](firmware-public-review-20260909.md).
No vendor executable/library was run, loaded or installed. No scanner, firmware,
calibration or settings were changed, and no referenced OTA service was queried.

## Verified package and bounded extraction

The [manufacturer download page](https://www.revopoint3d.com/pages/support-download)
links [Revo Scan 5.5.5.2012 for Windows](https://dl.revopoint3d.com/softs/revoscan5/RevoScan5-5.5.5.2012-windows-x86_64.exe),
listed as compatible with the original MINI. Complete download: 682,230,392
bytes; SHA-256 `b7c632591926d6ac80693ba5255c2855a868497ffe1e31993fe61f961d9244e4`.
Site release date, server Last-Modified and outer-PE build timestamp differ;
none is this scanner's firmware date. HTTPS provenance and matching file hashes
are not a completed Authenticode trust-chain or exact-unit firmware validation.

Existing 7-Zip sees only the PE wrapper. Its 64-byte loader offset resource has
an exact revision-2 CRC; metadata identifies Inno Setup 6.5.2. The exact
[upstream tag source](https://github.com/jrsoftware/issrc/tree/828de5a5a0d2f3b9ea67957060314f0f8cca114a)
is pinned and Git-blob/size/hash checked, with its license preserved. Main's
7.1.0 metadata sizes differ, so that newer format was not substituted.

All 93,073 metadata bytes are consumed: two decoded blocks of 425,606 and
126,380 bytes, every block/chunk CRC verified. The complete file table has
1,422 entries and 1,420 distinct locations. An empty-name uninstall executable
precedes the first application file; the initially incomplete walk and exact
failure location are retained. No filename matches firmware/recovery/upgrade
or `.img/.fw/.rkp/.bin`. This does not exclude a hidden or later-downloaded
payload. The lone ZIP is named `res/Project_Example.zip`; it was not extracted.

The solid data is LZMA1, 8-MiB dictionary, not LZMA2. Only a 406,660,528-byte
expanded prefix was processed, and five files retained. The entire solid stream
was **not** validated through EOF. All five saved files match their original
metadata SHA-256 after restoring the documented per-file 64-KiB instruction
transform; only our own data decoder and existing static tools were used.

| Selected file | Bytes | SHA-256 |
| --- | ---: | --- |
| `3DCamera.dll` | 63,965,856 | `d24637d2776f37072a5b38b46b5e57fbc106f9409b04dfaaec78eb6d095b5728` |
| `RevoScan5.exe` | 26,604,704 | `c07f381c3adfd24675931b4c67c21a47881e8b586baf0cdf60fcec098079a207` |
| `TrackerScanner.dll` | 52,656 | `ef6508c54dcf55dec8a2f4a7377f7a65b81bf194229f920ad77e0fb16a21c39b` |
| `config/calibration/0A6/algo.json` | 55 | `a812ed005e0f0c391dde6af5b581f6dec9a25f79ace359881d68ffc7b078870a` |
| `config/calibration/0A6/calibration.json` | 170 | `964032784e26bf1a9ed88bd5aaae0189a3141e86d5ccf0a93a5eb0cd27f23537` |

Metadata decoder: 25 synthetic tests. Selected-stream/byte-restoration decoder:
54 final synthetic tests; the first run's five progress-check failures and
49 passes are retained with pre-correction source. These are separate from
the repository's 1,544-test selected reliability suite and prove no scan quality.

The calibration JSON describes a 9-by-5 pattern, cell size 15, special-circle
positions and empty real-point data. `algo.json` has detection method 1 and
median-blur setting 3. These are calibration-target/detection settings, **not
the unit's factory calibration backup or its depth/tracking filter settings**.

## Firmware references and a real SDK-version distinction

The app contains a serial-aware OTA-version endpoint, package/MD5 checks,
USB-mode checks and recovery-mode messages. The camera library contains normal,
network and alternative-system update references. None proves that a package
exists for this serial or that recovery works on this hardware. No endpoint
was contacted, update command issued or updater installed.

Three camera-library timing strings have verified static x64 instruction
references. Bounded enclosing context identifies `cs::Camera::setFringePattern`,
pattern enum range 0..6, command template `0xa05`, and actual float constants
6600/6200 selected only on the pattern-6 path. Runtime unwind-table entries can
be fragments, not complete functions; the additional enclosing window and
two gate-callee listings preserve that distinction. No machine code is emulated.

The earlier Linux SDK is separately preserved and rechecked at 12:07 UTC:
94,574,928 bytes, SHA-256
`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`.
Its complete symbol-bounded `setFringePattern` listing is 0x1fd5fa..0x1fdafc.
The two binaries genuinely differ:

| Host-library predicate | Older pinned Linux SDK | Windows library in 5.5.5.2012 |
| --- | --- | --- |
| Model before POP 2 (ID below 10) | Reject | Reject |
| Firmware below 2.7.35 | Reject every otherwise eligible model | Check only POP 2 (ID 10) |
| Patterns 5/6 below 2.9.43 | Reject every otherwise eligible model | Check only POP 2 (ID 10) |
| Pattern 6 after successful command | Choose 6600/6200 around 2.9.45 | Same two constants/version branch |

The Windows serial parser reads the 17-character unit serial's `A6` suffix
and non-NUL final character as model ID 12; the independently pinned public
header names ID 12 `CAMERA_MINI_NORMAL`. Thus the Windows checks above do not
themselves rule out this MINI's 2.8.7 firmware. This is **static inference about
host control flow**, not a firmware implementation or depth-quality result.

The earlier [Linux-specific conclusion](scanner-validation-results.md#p6-firmware-evidence-and-flash-gate)
was correct for that binary. It must not be generalized into a universal ban
on the MINI or a claim that flashing is inherently necessary. Conversely,
omitting a host check does not make an unsupported device command safe or useful.
The public header describes pattern 6 as a high-frame-rate 18-frame mode and
requires stream stop/start around pattern changes. Existing normal-operation
logs mention pattern 3. Neither is fresh readback of a successful mode-6 change.

## What remains open

Determine the exact unit's pattern-command implementation, an observable
starting mode and a verified restoration path before any mode-changing test.
Keep timing/exposure/gain fixed initially if such a test is justified; changing
pattern and exposure together would confound depth retention. Retain complete
raw deliveries, fixed whole-scene masks, numerical bias/retention checks and
thermal telemetry. No throughput result from this static review is claimed.

Firmware flashing still requires an authenticated exact-compatible image,
complete device-specific calibration preservation and demonstrated recovery.
No Stage A/Stage B tracking candidate, live-FPS target or reliability gate is
qualified by this package work. All failed extraction/format assumptions remain
in the evidence root and its retrieval notes, rather than being erased.

Archive sealed at 12:20:52 UTC: 121 files / 872,679,386 bytes, manifest SHA-256
`93dd2082a9ff1d15e447d57b6eb0b6c6ce90f2d4dd36231cc74171a7bd39cb71`.
The outside-root verification report belongs to the separate fringe-capability
root. Further live reads or static follow-ups must not mutate this archive.
