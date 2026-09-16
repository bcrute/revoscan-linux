"""Fixed G1 offline decision rules, independent of graph fitting and filenames."""

from __future__ import annotations

import math
from typing import Any


def measurable(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def activation(checks: list[dict[str, Any]]) -> str:
    """Held-out or excluded pairs cannot activate correction or a second try."""
    fit = [r for r in checks if r["role"] == "fit" and not r["exclusion_reasons"]]
    if len(fit) < 5 or any(not measurable(r["before"]["agreement"]) for r in fit):
        return "insufficient"
    return "correct" if any(r["before"]["agreement"] < 0.9 for r in fit) else "keep"


def rejection_reasons(checks: list[dict[str, Any]], local: list[dict[str, Any]]) -> list[str]:
    """No successful optimiser status can override these raw geometry checks."""
    reasons = []
    held = [r for r in checks if r["role"] == "validation"]
    if len(held) < 5:
        reasons.append("fewer_than_five_held_out_pairs")
    if any(not measurable(r[when]["agreement"]) for r in checks for when in ("before", "after")):
        reasons.append("uncheckable_credible_revisit")
    else:
        if any(r["after"]["agreement"] < 0.9 for r in checks):
            reasons.append("sampled_revisit_below_90_percent")
        if any(
            r["before"]["agreement"] >= 0.9
            and r["after"]["agreement"] < r["before"]["agreement"] - 0.01
            for r in checks
        ):
            reasons.append("previously_good_pair_degraded_over_one_point")
    if not local or any(
        not measurable(r[when]["agreement"]) for r in local for when in ("before", "after")
    ):
        reasons.append("missing_or_uncheckable_local_checks")
    elif any(r["after"]["agreement"] < r["before"]["agreement"] - 0.01 for r in local):
        reasons.append("local_pair_degraded_over_one_point")
    return reasons
