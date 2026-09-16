"""Dense raw proposals cannot silently reuse held-out endpoints or empty depth."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def dense(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_dense_loop")


def test_choice_is_deterministic_separated_and_protected_at_both_ends(dense):
    depth = np.full((256, 10, 10), 200.0)
    masks = np.ones_like(depth, dtype=bool)
    partition = [dict(role="validation", sample=dict(frame=0, best_frame=128, credible=True))]
    rows = dense.select(depth, masks, partition)
    assert rows == dense.select(depth, masks, partition)
    assert rows[0]["exclusion_reasons"] == ["protected_validation_endpoint"]
    for row in rows:
        if "accepted_indices" not in row:
            continue
        a, b = row["accepted_indices"]
        assert b - a >= 100
        assert all(abs(i - p) > 4 for i in (a, b) for p in (0, 128))
    assert rows[1]["raw_match"]["source_index"] == 133


def test_unordered_duplicate_not_counted_twice(dense):
    depth = np.full((256, 10, 10), 250.0)
    depth[0] = depth[128] = 200.0
    rows = dense.select(depth, np.ones_like(depth, dtype=bool), [])
    assert rows[0]["accepted_indices"] == [0, 128]
    assert rows[0]["exclusion_reasons"] == []
    assert rows[4]["accepted_indices"] == [0, 128]
    assert rows[4]["exclusion_reasons"] == ["duplicate_unordered_pair"]


def test_no_returns_are_not_repeated_views(dense):
    depth = np.zeros((256, 10, 10))
    rows = dense.select(depth, np.zeros_like(depth, dtype=bool), [])
    assert all(r["exclusion_reasons"] == ["raw_descriptor_not_comparable"] for r in rows)


def test_invalid_or_short_inputs_refused(dense):
    with pytest.raises(ValueError, match="complete"):
        dense.proposals([b"short"] * 100, [])
    with pytest.raises(ValueError, match="Finite"):
        dense.select(np.full((2, 3, 3), float("nan")), np.ones((2, 3, 3), dtype=bool), [])
