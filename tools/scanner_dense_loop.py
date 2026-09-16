"""G2 raw-only closure proposals with frozen validation endpoint exclusion."""

from __future__ import annotations

from typing import Any

import numpy as np
from scanner_moving_abba import descriptor, nearest_view


def proposals(raws: list[bytes], partition: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not 100 <= len(raws) <= 3000 or any(len(raw) != 512000 for raw in raws):
        raise ValueError("100..3000 complete raw frames required")
    descriptors = [descriptor(np.frombuffer(raw, dtype="<u2").reshape(400, 640)) for raw in raws]
    depths = np.stack([d for d, _ in descriptors])
    masks = np.stack([m for _, m in descriptors])
    return select(depths, masks, partition)


def select(
    depths: np.ndarray, masks: np.ndarray, partition: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if depths.shape != masks.shape or depths.ndim != 3 or not np.isfinite(depths).all():
        raise ValueError("Finite matching depth/mask descriptor arrays required")
    protected = [
        r["sample"][key]
        for r in partition
        if r["role"] == "validation" and r["sample"]["credible"]
        for key in ("frame", "best_frame")
    ]
    available = [i for i in range(len(depths)) if all(abs(i - p) > 4 for p in protected)]
    used = set()
    rows = []
    for query in range(0, len(depths), 32):
        reasons: list[str] = []
        row: dict[str, Any] = dict(query=query, exclusion_reasons=reasons)
        rows.append(row)
        if query not in available:
            reasons.append("protected_validation_endpoint")
            continue
        candidates = [i for i in available if abs(i - query) >= 100]
        if not candidates:
            reasons.append("no_separated_unprotected_candidate")
            continue
        match = nearest_view(
            (depths[query], masks[query]), depths[candidates], masks[candidates], candidates
        )
        row["raw_match"] = match
        pair = tuple(sorted((query, match["source_index"])))
        row["accepted_indices"] = list(pair)
        if not match["comparable"]:
            reasons.append("raw_descriptor_not_comparable")
        elif pair in used:
            reasons.append("duplicate_unordered_pair")
        else:
            used.add(pair)
    return rows
