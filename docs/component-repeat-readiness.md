# Controlled component repeat — readiness, 2026-09-09

## Outcome

Subsequent execution: the operator adjusted the setup and the
[controlled repeat completed](component-repeat-results.md). The setup-only
outcome below is preserved as the earlier checkpoint, not a current request
to repeat the adjustment.

The scanner is available at the unchanged 20 ms baseline. A fresh rotating
repeat has **not** started: the short view check was stationary, and the colour
view clips the left-hand portion of the printed target. Recenter the entire
target, start the turntable slowly, and leave the camera fixed before acquiring
the repeat. No new dimensional result or reliability qualification is claimed.

The visible ramp, pocket plateau and steps are consistent with the measured
reference. Visual consistency is not proof of physical print identity; do not
treat the earlier description of a different white model as established fact.
The existing caliper record remains the comparison reference, not a request to
repeat the owner's measurement preparation.

Evidence is in the new, currently unsealed directory:
`/home/ben/scans/diagnostics/20260909-component-repeat-AaLYjA/`.
`readiness-result.json` completed at 18:03:45 UTC. The previous
[saved-capture baseline](component-baseline-results.md) and its sealed evidence
remain unchanged.

## Checks completed

- Exact scanner serial: `A22704522A6F00A64`.
- Before/after settings: frame time 20,000 us, exposure 20,000 us, manual
  exposure mode 0, gain 24. No settings were written.
- Depth view: 32 complete distinct frames, no sequence drops or short frames,
  4.413 observed fps. The familiar alternating arrival gaps remain visible.
- First/last selected depth views: 40,516 mutually valid pixels; absolute
  depth-difference median/p90/p99 = 0.04/0.17/1.44 mm. Of those pixels, 98.43%
  differ by no more than 1 mm. This supports a stationary-scene assessment,
  not a tracking or shape-accuracy conclusion.
- Colour check: 24 complete depth frames with no drops/short frames and nine
  RGB images. Inspect `colour-check/colour_0007.jpg` and `view-31.png` together.
  Colour and depth fields of view differ: RGB clipping alone does not prove
  the depth sensor clipped the same edge. The depth view is strongly left-biased
  and shows limited left-side geometry, so recentering is still warranted.
- Largest depth component in the last view: bounding box (23, 93)-(271, 333)
  in the 640-by-400 image. Zero component pixels touch the image border; this
  does not establish that all surfaces of the object returned depth.
- Internal CPU/NPU readings peaked at 58.9/58.2 degrees C during the depth
  check and read 58.5/57.7 degrees C after the colour check. These are short
  readiness observations, not a thermal soak. The 80-degree stop policy remains
  a test policy, not a manufacturer temperature rating.
- No reconstruction, scale adjustment, firmware/calibration change or
  experimental mesh generation occurred. M27 remains paused.

## Capture timing and continuation

The view probe buffers raw frames and writes them after stream closure. Its
authoritative delivery times are in `view-check/raw/capture-progress.json`;
file modification times must not be substituted for those times in fixed-axis
reconstruction. Normal CLI sweep writes on arrival and preserves its frame
times in `scan.scanner3d`. A full repeat must use actual capture timing and a
bounded acquisition with internal-temperature monitoring.

The initial `python -m scanner3d` invocation failed before hardware access
because the package has no `__main__`. The installed `.venv/bin/scanner3d`
entry point completed the colour check successfully.

After the physical adjustment, check framing and motion again, then acquire
the controlled repeat into a fresh directory. Retain the same print correction,
60-300 mm band, fixed-axis reconstruction and placement seed 5 for the baseline
comparison. The reduced-SVD evaluator is diagnostic-only. Check explicit
opposing-face dimensions and coverage before interpreting any repeated
approximately 1% shortfall; do not fit it away with a scale correction.
