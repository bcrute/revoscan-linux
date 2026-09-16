# M26: bounded image median and reverse initialization

Declared September 9, 2026, around 07:01 UTC, before implementation or numerical
testing. Evidence:
`/home/ben/scans/diagnostics/20260909-reliability-depth-median/`.
At 07:34, 95 focused checks and the final 1,465-test selected suite pass, with
clean lint and cold typing of 130 files. Four numerical fixtures exercise the
actual replay/tracker through explicitly substituted CPU-reference GPU and
preparation/coherence fixtures; they are not hardware or accuracy validation.
The first selected run's two private-import typing errors and correction are
retained. The 07:23:30 freeze contains 173 executing files and the separate
verification snapshot contains 256 files. Twelve recorded GPU observation
controls start at 07:23:50 after M25 is fully rejected; all twelve reproduce
exactly. Forty changed trajectories start at 07:36:41, with two GPU children.
All forty complete; the 08:32:08 UTC closeout confirms no whole survivor.

The [vendor-source review](reliability-calibration-semantics.md) identifies
optional preprocessing before point generation. M26 is a separately specified
rule, not reproduction of that filtering or proof it is used in Revo Scan.
Prior covariance/solver weighting and stable-pixel sampling do not test this
image-domain representation.

One nonrecursive 3x3 median is allowed only where all nine original samples
are inside the existing 60..300 band and their scaled range is at most one
original 1.5-unit tracking voxel. Border, missing and band membership remain
unchanged. The original raw bytes are never rewritten, and all geometry scores
still use them. Every derived preparation records its source/result hashes,
changed/eligible pixel counts and signed/absolute code-change distributions.
This is not hole filling, temporal smoothing, an uncertainty estimate, or
evidence that a displaced depth sample is physically more accurate.

The original model-first preference, gates and refusal/stop rules stay fixed.
Cross two source capacities, two forward/reverse initialization choices and
two finite-update rules: eight whole configurations, forty trajectories and
840 required fixed raw pairs. Twelve observe-raw controls must reproduce their
matching M21 reports, raw checks, solver and seed traces exactly first. No
per-case selection, new crop, missing-pair imputation or regression waiver.

Only opt-in diagnostic preparation scopes and adapters are added. Original
M1–M25 sources, production, firmware, calibration, dependencies, driver, scanner
settings and experimental meshing remain untouched. A whole Stage A survivor
must pass the separate six-case longer/handheld stage before further work.

## Completed result

All eight configurations accept all 2,524 original inputs: 20,192 repeated
frame occurrences, all 840 required raw comparisons checkable. Two configurations
pass four cases, but none passes five. No Stage B replay is earned or run.
The original 90% absolute and one-percentage-point regression gates both apply;
a minimum above 90% does not by itself pass a case.

| Capacity | Seed | Update | A / B / A1 / A2 / C minimum agreement (%) | Cases passing |
| --- | --- | --- | --- | --- | --- |
| 900 | Forward | Original | 97.4343 / 99.7801 / 96.8856 / 8.5659 / 15.6228 | A, B |
| 900 | Forward | Centered | 97.3701 / 99.7801 / 98.4026 / 98.0011 / 23.8944 | A, B, A1, A2 |
| 900 | Reverse-forward | Original | 97.5016 / 99.7801 / 98.2603 / 6.5397 / 98.9631 | A, B, A1, C |
| 900 | Reverse-forward | Centered | 97.5016 / 99.7801 / 97.2565 / 5.1282 / 99.1875 | A, B, C |
| 1800 | Forward | Original | 96.1152 / 93.6000 / 98.7492 / 97.8869 / 32.3179 | A, A1, A2 |
| 1800 | Forward | Centered | 95.6493 / 93.3752 / 98.3233 / 99.2159 / 30.1418 | A, A1, A2 |
| 1800 | Reverse-forward | Original | 96.2840 / 93.7215 / 99.1304 / 22.0577 / 99.1851 | A, A1, C |
| 1800 | Reverse-forward | Centered | 95.5823 / 93.7179 / 97.8710 / 98.3954 / 47.0091 | A, A2 |

Every frame changes some depth codes; all nonzero and band memberships remain
exact. Across repeated preparations, 1,009,866,984 pixel occurrences are eligible
and 543,355,576 change (273,073,384 upward / 270,282,192 downward). All 2,524
distinct original raw hashes produce identical derived hashes and complete
statistics across variants. These are repeated computations, not independent
data or evidence that changed depth is more accurate.

Saved traces contain 60,774 completed fit requests, four refused fit requests,
91,400 completed stages and 255,340 iterations. Eighty-five reverse seeds
finish unconverged; original recovery still accepts every input. A refused fit
is not necessarily a refused frame. No convergence or refusal gate changed.
All 173 frozen executing sources still match. All 420 seed/preparation and 420
capacity/preparation descriptive contrasts are computable; they reuse the same
840 raw comparisons across the two contrast families.

Authoritative artifacts: `m26/audit.json`, `m26/interactions.json`,
`execution-summary.json` and `through-m26-audit.json` in the evidence root.
Completed M1–M26 totals: 695 changed trajectories, 139 configurations, 14,595
required/checkable pairs and 134 exact observation pilots. The only Stage A
survivor remains M21's already Stage-B-rejected configuration. No production
tracking, live throughput or physical quality is qualified.

The evidence root is sealed at 08:36:41 UTC: 776 files / 1,916,206,029 bytes,
manifest SHA-256
`39ce482bab20f4aa8e8e149aa917b993c104cafe842ae339e53005daa458165f`.
Outside-root verification is retained in the official-package-review root.
Later work must not append or rewrite this archive.
