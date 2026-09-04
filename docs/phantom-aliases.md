# The near-field phantoms are fringe-order aliases, measured

Status: **measured on two captures a day apart, replicated to three digits.**
This closes the question the memory of 2026-08-31 opened as a "near-field
obstruction" and the 2026-09-03 session reopened as "decoding phantoms".

## The test

`fringe-projection-depth-failure-modes.pdf` (research report, repo root, dated
2026-09-04) gives the signature of a wrong fringe order: adjacent aliases are
separated by a constant increment in **reciprocal depth**,

```
1/z_bad - 1/z_true = k * D        (k integer, D = P / F one fringe order)
```

and not by a constant ratio of depths. It recommends a distance sweep of a
matte board to establish it. The sweep was unnecessary: the scene already
contains a surface (the white cylinder behind the turntable, about 260 mm out)
that the sensor reads correctly in some frames and wrongly in others, on the
SAME pixels, so every such pixel is one sample of the law.

Over the top band of the image (rows 0-100, columns 40-300), taking each
pixel's median phantom reading (< 130 mm) and median far reading (245-330 mm):

| capture | pixels | far depth | phantoms | reciprocal shift clusters | ratio of clusters |
| --- | --- | --- | --- | --- | --- |
| `20260903-142612` | 2515 | 259.7 mm | 107.2 and 73.3 mm | 0.00540 and 0.01043 mm⁻¹ | 1.93 |
| `20260902-224804` | 2422 | 259.8 mm | 106.8 and 73.3 mm | 0.00542 and 0.01071 mm⁻¹ | 1.98 |

Two clusters, at one and two increments of **D = 0.0054 mm⁻¹**, in both
captures. The alias model fits the per-pixel data at 7.2-7.7 mm RMS against
12.5-13.2 mm for a constant-ratio model. Order 1 and order 2 are interleaved
across the same columns and rows, which no fixed reflection could produce.

## What it means

- **Nothing sits at 6-10 cm.** The points are the surface at 26 cm decoded one
  or two fringe orders too near. A hand at 8 cm returns nothing itself
  (inside the near limit) and blocks the rays only where they actually run,
  which is above the target toward that surface.
- **The alias window is narrow.** With D = 0.0054 mm⁻¹, a surface at 250 mm
  lands at 106 or 68 mm; at 300 mm, at 114 or 71 mm; at 480 mm, at 134 or
  78 mm. Any surface the projector reaches behind the subject can produce
  points inside the working range. The subject's own far faces are at risk
  too once they pass roughly 220 mm, so the target belongs nearer than the
  ~215 mm it stands at now, and the space behind it must be empty or black
  out to well beyond 50 cm.
- **The registration band was the right defence** but a downstream one: it
  drops the aliases, it does not prevent them.

## What is still open, mapped to the report's campaign

- Phase A (the law): done above. A matte-board sweep would only refine D.
- Phase B (light transport): the projector cannot be switched off on this
  unit (`docs/dsp-protocol.md`), so the projector-off frame is unavailable;
  a black absorber behind the subject is the substitute. **Done on
  2026-09-04:** with matte grey-black cloth about 30 cm behind the subject,
  capture `20260904-020845` carries 0.00% of returns under 84 mm (0.8-1.0% on
  the two captures above), no returns in the 84-130 mm band, and a single
  depth mode from 160 to 260 mm. The phantoms vanished rather than
  strengthened, which is the alias outcome and not the ghost outcome.
- Phase C (firmware controls): the coprocessor exposes `id_msg_algo_set_z_range`,
  `id_msg_algo_set_background_threshold` and `id_msg_algo_set_gradient_threshold`
  by name. The report's conventional readings apply; a bounded one-factor
  sweep with readback is the experiment, and the 2026-08 sweep that zeroed
  exposure and needed a power cycle is the cautionary case.
- Phase D (calibration): not before the aliases are removed at the source.
