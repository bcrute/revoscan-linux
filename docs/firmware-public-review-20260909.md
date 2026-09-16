# Original MINI: public firmware-path recheck

Follow-up: the [static official-package review](firmware-package-review-20260909.md)
inspects the manufacturer-linked installer and reconciles differing SDK version
gates. This earlier page-only review remains a bounded historical result.

Read-only review, 2026-09-09 03:33 UTC. This supplements the exact-unit
[firmware gate](scanner-validation-results.md#p6-firmware-evidence-and-flash-gate),
not a device readback, installation, support request or flash.

The manufacturer's download page lists Revo Scan 5.5.5 for Windows/macOS,
dated 2025-12-20, with original MINI compatibility. Desktop Revo Scan 6's
current lists include MINI 2 but omit original MINI; mobile Revo Scan 6 lists
include both. No Linux edition is listed in those platform sections. The
Windows 6 entry says 6.3.2, dated 2026-09-08; some regional/search copies still
show 6.3.1. These are software listings, not exact-unit firmware availability.
[Manufacturer downloads](https://www.revopoint3d.com/pages/support-download).

In the original MINI firmware question, manufacturer representative
Revopoint-Jane's 2024-06-17 response requests the scanner model and firmware
information. That thread does not supply a public recovery image or establish
the latest version for our serial.
[Manufacturer forum response](https://forum.revopoint3d.com/t/mini-firmware-update-how/21792/3).

The earlier update-failure thread's official 2022-11-16 response requests
diagnostic logs and support escalation. Community reset/power anecdotes in
that thread are not an authenticated recovery procedure. Its linked log tool
was not downloaded or executed.
[Update-failure thread](https://forum.revopoint3d.com/t/mini-firmware-update-issue/18408).

The original MINI quick-start's indexed text describes in-app update
notification and stable power. It also specifies a 0–40°C operating environment;
this is not a CPU/NPU junction-temperature rating and does not establish a
90°C continuous internal limit. The direct PDF open failed in this review;
these statements were available in the search index, not a fresh local PDF.
[Original MINI quick-start](https://download.revopoint3d.com/support/download/mini/mini-quickstartguide-en-v3.0-2023.7.27.pdf).

The current manual's indexed firmware screenshot is explicitly **MINI 2**.
Its displayed v4.x image name is not evidence of compatibility with our
original MINI v2.8.7. The online Revopedia route rendered only a feedback shell.
[Revo Scan manual](https://download.revopoint3d.com/support/download/soft/revoscan-pc-usermanual-en.pdf).

Conclusion: these inspected public sources do not close the exact-model image,
calibration-preservation and supported-recovery requirements. This is a bounded
negative search result, not proof that Revopoint has no suitable image. A future
vendor-assisted update check must identify the exact model/version/serial and
request the recoverable image plus calibration handling. No contact was sent,
package installed, scanner setting changed or firmware executed here.
