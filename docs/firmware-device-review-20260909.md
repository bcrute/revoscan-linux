# Device firmware and calibration review, 2026-09-09

Sealed evidence root:
`/home/ben/scans/diagnostics/20260909-fringe-capability-review/`.
This follows the [official-package review](firmware-package-review-20260909.md):
the newer Windows SDK's MINI model gate is less restrictive than the preserved
Linux SDK's gate. That does not establish device support for another pattern.

## Read-only acquisition

The 12:12 UTC health inventory freezes and verifies 173 executing repository
sources. Three fixed, nonrecursive directory catalogs follow at 12:15, 12:19
and 12:22 UTC (six, three and four paths). Their pure command tests pass
21, 18 and 19 checks respectively. Observed paths lead to
`/userdata/firmware/preisp.rkl` and `/oem/camparam/`; unrelated radio/network
configuration files are not read. Symlink destinations and failures are retained.

A sixteen-path file catalog has 23 passing tests before acquisition. The
12:25:21–12:25:41 read obtains every expected file and a second firmware copy:
1,522,600 retained bytes. The two 757,024-byte firmware reads independently
have SHA-256 `f1d3aa9756b69ec6435abfc6a89a30e352c9f3df5a3acb385c61c238dca72dc6`.
Original paths, lengths, hashes, commands, selectors and temperatures are in
`file-read-plan.json` and `file-read-report.json`. Baseline timing remains
20000/20000/0; the health inventory reads gain 24. All source comparisons and
selector cleanup pass. File-read CPU/NPU temperatures range up to 58.9/58.2 C.

No stream, mode, timing, gain, calibration or persistent firmware change is
made. The existing command/getter mechanism creates unique temporary RAM log
files on the scanner: this is not a claim of literally zero RAM writes.
The saved files are not a complete device backup or a verified recovery image,
and the local evidence tree is not an off-machine backup.

## Container interpretation, not firmware modification

The file tag is `rkceva4k2022042615193025`, with a 640-byte header and fourteen
20-byte records. The primary Rockchip loader source at commit
`f39b590ea11da2d897644e42119ba5099b2eefc8` supplies the compatible record layout.
Its named section array has ten slots, whereas this blob has fourteen: it is
not asserted to be the exact running driver version.
[Record definition](https://github.com/rockchip-linux/kernel/blob/f39b590ea11da2d897644e42119ba5099b2eefc8/drivers/media/spi/rk1608_core.h),
[loader](https://github.com/rockchip-linux/kernel/blob/f39b590ea11da2d897644e42119ba5099b2eefc8/drivers/media/spi/rk1608_core.c).

The initial parser passes 23 synthetic checks but rejects a real record copying
the whole container header to DSP memory. That source, its tests and failed
inspection plan remain intact. The additive v2 parser permits only the exact
observed header-copy form; 27 tests pass and all fourteen records parse at
12:36:26 UTC. Ten contain payloads, four contain zero-length control records.
Extracted data, file offsets, declared destinations and hashes are preserved.
At that first inspection, CRC fields are not yet independently validated.
The separate 12:51:21 CRC review then checks all nine flagged payloads against
sixteen predeclared polynomial/direction/seed/final-xor combinations. Exactly
one combination matches every section: polynomial 0x1021, MSB-first, initial
65535, final xor zero. All 2,600 recurrence/standard-library synthetic checks
pass. The 12:53:31 independent closeout also verifies all nine full payloads
using standard-library `crc_hqx`, and proves that the ten data intervals cover
the entire saved file exactly once. The unflagged header copy is not a failed
CRC omitted from the comparison. No binary or downloaded source is executed,
installed, compiled, loaded or repacked; CRCs are not authentication.

The blob contains strings for frame-time limits, algorithm type and fringe-
pattern changes. Their mapped offsets are evidence of references, not decoded
branches or proof that patterns 5/6 work. The actual running image has not yet
been cryptographically bound to the saved file. No CEVA instruction decoder,
safe reversible mode trial or new quality-qualified speed is established.
CEVA's own RK1608 description identifies multiple CEVA-XM4 cores; that supports
the processor-family interpretation, not an ARM instruction listing or an
exact firmware variant.
[CEVA announcement](https://www.ceva-ip.com/press/ceva-and-lg-electronics-partner-for-smart-3d-camera-solution/).
The checked older Linux SDK symbol listing contains a fringe setter but no
corresponding named getter. This bounded search does not prove no getter exists.
The saved normal-boot log does not contain a matching container/version table;
runtime-image identity remains unproven. No reboot is initiated to obtain it.

## Calibration consistency, not physical accuracy

The fresh binary depth intrinsics exactly preserve the previously recorded
SHA-256 `fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf`.
The newly saved complete YAML names a different unit; the existing
`rgb_project.py` documentation already warns of that mismatch. Do not claim
this is a newly discovered calibration fault, apply the YAML as calibration,
or replace rectified intrinsics with raw-camera intrinsics.

The subsequent strict whole-record parser passes 21 synthetic checks. The
source-bound comparison completes without device access: all four current
binary fixtures (Pl, Prgb, LC_RT, Distort) exactly match the fresh device files.
Every Pl matrix entry equals the left 3x3 part of YAML pL rounded to float32;
all twelve LR-P entries equal pL, and all sixteen LR-Q entries equal Q.
LR-P does not equal pR: their principal-point and translation entries differ.
Both mapparam files contain exact float32 copies of their respective YAML
distortion vectors at byte 64; this is not a complete map-file schema.

The OEM and persist scale files both contain little-endian integer 200, crop
files contain integer 50, and the resonant-frequency file contains integer
3121. Recording alternative raw float interpretations does not make these
files float-valued parameters or establish the resonant value's units.
All raw values, mismatches and current fixture hashes are retained in
`calibration-comparison.json`. The serial text remains inconsistent, but
the matrices are not simply unrelated numbers. No optical truth or calibration
correction is established. The older YAML fixture is truncated/sanitized and
is not used as an independent complete calibration.

## Closeout

`device-review-audit.json` independently verifies all 173 frozen executing
sources, all thirteen directory results, all seventeen file reads, seven
retained JUnit suites, primary source Git-blob identities and licenses,
container coverage, strings and CRCs. Its tests include superseded parser
variants; their counts are not additional independent hardware trials.
Failed parser assumptions and bounded unsuccessful lookups are retained in
`RETRIEVAL-AND-FAILURE-NOTES.md`. Host-only sensors at 12:51:39 report GPU
edge/junction/memory 57/60/73 C and CPU package 63 C with no CPU critical alarm;
these do not substitute for the earlier actual scanner temperature readings.

The remaining live-mode gates are a reliable current-mode observation, decoded
or otherwise verified command semantics and a demonstrated restoration route.
Flashing additionally needs a compatible authenticated image, complete
device-specific preservation and recovery evidence. No firmware claim is
promoted into the production tracker, and M1–M26 trajectory totals are unchanged.

Sealed at 12:56:59 UTC: 334 files / 7,563,546 bytes, manifest SHA-256
`b233bd2d1329f39ba8a887fdf6f1bbfaaffb5777ba0a7f0d3698e96d672eab66`.
The separate organized-plane review verifies exact inventory and every hash
at 12:58:42. No further output belongs in this sealed root.
