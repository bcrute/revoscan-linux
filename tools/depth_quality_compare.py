"""Compare stored rotating-object depth sets using nearest-view matching.

This is an exploratory coverage comparison, NOT a stationary exposure test or
metrology. Motion, occlusion and temporal noise remain confounders. Each match
and its residual are recorded so a bad match cannot masquerade as depth loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, label

from scanner3d.cli import DEPTH_SCALE_METRES


def load(directory: Path) -> tuple[list[Path], np.ndarray]:
    paths = sorted(directory.glob("frame_*.raw"))
    if not 2 <= len(paths) <= 256:
        raise ValueError("Expected 2--256 complete stored frames")
    frames = np.stack([np.frombuffer(p.read_bytes(), dtype="<u2").reshape(400, 640) for p in paths])
    return paths, frames.astype(np.float64) * DEPTH_SCALE_METRES


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_paths, references = load(reference)
    paths, frames = load(candidate)
    reference_masks = (references > 60) & (references < 300)
    candidate_masks = (frames > 60) & (frames < 300)
    # Strided descriptors choose the nearest recorded viewpoint, not a rigid
    # transform that might hide a change in the measured shape.
    small = references[:, ::8, ::8]
    small_masks = reference_masks[:, ::8, ::8]
    rows: list[dict[str, Any]] = []
    for index, (frame, mask) in enumerate(zip(frames, candidate_masks, strict=True)):
        mutual = small_masks & mask[::8, ::8]
        union = small_masks | mask[::8, ::8]
        residuals = np.minimum(np.abs(small - frame[::8, ::8]), 10)
        score = (
            np.sum(residuals * mutual, axis=(1, 2)) + 10 * np.sum(union & ~mutual, axis=(1, 2))
        ) / np.maximum(1, np.sum(union, axis=(1, 2)))
        nearest = int(np.argmin(score))
        regions, count = label(reference_masks[nearest])
        if not count:
            continue
        sizes = np.bincount(regions.ravel())
        sizes[0] = 0
        object_mask = regions == np.argmax(sizes)
        core = binary_erosion(object_mask, iterations=3)
        common = core & mask
        differences = np.abs(frame[common] - references[nearest][common])
        rows.append(
            {
                "candidate": paths[index].name,
                "reference": reference_paths[nearest].name,
                "matching_score_mm_plus_missing_penalty": float(score[nearest]),
                "reference_object_pixels": int(np.count_nonzero(object_mask)),
                "reference_core_pixels": int(np.count_nonzero(core)),
                "core_return_retention": int(np.count_nonzero(common))
                / max(1, int(np.count_nonzero(core))),
                "mutual_core_depth_difference_mm_p50_p90": np.percentile(
                    differences, [50, 90]
                ).tolist()
                if len(differences)
                else None,
            }
        )
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "scope": (
            "Nearest-view rotating-object comparison; motion/occlusion remain confounders. "
            "Not metrology."
        ),
        "frames": rows,
        "median_core_return_retention": float(
            np.median([r["core_return_retention"] for r in rows])
        ),
        "median_matching_score": float(
            np.median([r["matching_score_mm_plus_missing_penalty"] for r in rows])
        ),
        "median_mutual_depth_difference_mm": float(
            np.median(
                [
                    r["mutual_core_depth_difference_mm_p50_p90"][0]
                    for r in rows
                    if r["mutual_core_depth_difference_mm_p50_p90"] is not None
                ]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite report")
    reports = []
    for candidate in args.candidate:
        report = compare(args.reference, candidate)
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != "frames"}), flush=True)
    with args.output.open("x") as stream:
        json.dump(reports, stream, indent=2)


if __name__ == "__main__":
    main()
