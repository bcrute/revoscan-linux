"""Gate 2, with accuracy and completeness separated.

A CAD-side nearest-neighbour distance answers two questions at once and must
not: it is large both when a surface was measured badly and when it was never
measured. So report them apart -

  completeness : the fraction of a surface's CAD points that have ANY scan
                 point within the observation radius
  accuracy     : the deviation over only those points, which is the only
                 subset where "how far off" means anything

- exactly the split the coverage research names (docs/coverage-and-subject.md
  section 3, and mapper's observed_geometry / survey_extent).
"""
import sys, json; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np

T = Path("/home/ben/repositories/3d-scanner/reference/targets/reconstruction_v1")
SCR = Path("/tmp/claude-1000/-home-ben-repositories-3d-scanner/f0053f76-900a-4117-882a-7aa19f4bc270/scratchpad")
man = json.loads((T / "reconstruction-target-v1.json").read_text())
gates = json.loads((T / "gates/tolerance-v1.json").read_text())
tol = [g for g in gates["gates"] if g["id"] == "planar_rms"][0]["tolerance_mm"]
name_of = {v["id"]: k for k, v in man["surface_ids"].items()}

raw = (T / "reconstruction-target-v1-reference.ply").read_bytes()
he = raw.index(b"end_header\n") + len(b"end_header\n")
n = int([l for l in raw[:he].split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
rec = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("nx","<f4"),("ny","<f4"),("nz","<f4"),("sid","<u2")])
ref = np.frombuffer(raw, dtype=rec, count=n, offset=he)
sid = np.array(ref["sid"])
d = np.load(SCR / "cad_err.npy")

SEEN = 3.0   # a CAD point with no scan point inside this was not observed
print(f"observation radius {SEEN} mm; accuracy gate {tol} mm\n")
print(f"{'surface':<20} {'CAD':>5} {'seen':>6} {'ACCURACY (of seen)':>20}   verdict")
print(f"{'':<20} {'':>5} {'':>6} {'RMS':>8} {'p95':>10}")
rows = []
for s in sorted(set(int(x) for x in sid)):
    m = sid == s
    if m.sum() < 30: continue
    ds = d[m]
    seen = ds < SEEN
    frac = float(seen.mean())
    if seen.sum() < 15:
        rows.append((name_of.get(s,str(s)), int(m.sum()), frac, None, None)); continue
    rms = float(np.sqrt((ds[seen]**2).mean())); p95 = float(np.percentile(ds[seen], 95))
    rows.append((name_of.get(s,str(s)), int(m.sum()), frac, rms, p95))
for nm, c, frac, rms, p95 in sorted(rows, key=lambda r: r[2]):
    if rms is None:
        print(f"{nm:<20} {c:>5} {frac:>5.0%} {'':>8} {'':>10}   NOT OBSERVED")
    else:
        v = "PASS" if rms <= tol else "FAIL"
        low = "  (thin coverage)" if frac < 0.5 else ""
        print(f"{nm:<20} {c:>5} {frac:>5.0%} {rms:>8.2f} {p95:>10.2f}   {v}{low}")

ok = [r for r in rows if r[3] is not None]
allseen = d < SEEN
print(f"\nwhole target: {allseen.mean():.0%} of CAD points observed; "
      f"accuracy over those RMS {np.sqrt((d[allseen]**2).mean()):.2f} mm, "
      f"p95 {np.percentile(d[allseen],95):.2f} mm")
print(f"surfaces passing the {tol} mm accuracy gate: "
      f"{sum(1 for r in ok if r[3] <= tol)}/{len(ok)}")
