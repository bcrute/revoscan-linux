"""Phase comparison retains missing orientations and never aligns away differences."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def moving(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_moving_abba")


def test_anchors_use_fixed_time_bin_centres_not_depth_or_scores(moving):
    rows = [dict(monotonic_s=i * 0.2, in_comparison_window=12 <= i < 312) for i in range(640)]
    chosen = moving.anchor_indices(rows)
    assert len(chosen) == len(set(chosen)) == 60
    for second, index in enumerate(chosen):
        assert abs(rows[index]["monotonic_s"] - (rows[12]["monotonic_s"] + second + 0.5)) <= 0.11
    with pytest.raises(ValueError, match="Missing"):
        moving.anchor_indices([])


def test_best_view_uses_shape_and_missing_penalty_not_candidate_order(moving):
    depth = np.arange(144).reshape(12, 12) * 0.02 + 100
    mask = np.ones_like(depth, dtype=bool)
    depths = np.stack([depth + 8, depth + 0.2, depth + 2])
    masks = np.ones_like(depths, dtype=bool)
    result = moving.nearest_view((depth, mask), depths, masks, [13, 24, 52])
    assert result["source_index"] == 24 and result["comparable"]
    assert result["median_mutual_depth_difference"] == pytest.approx(0.2)
    assert result["descriptor_mask_iou"] == 1


def test_sparse_perfect_residual_cannot_beat_complete_close_view(moving):
    depth = np.full((20, 20), 100.0)
    mask = np.ones_like(depth, dtype=bool)
    depths = np.stack([depth, depth + 0.2])
    masks = np.ones_like(depths, dtype=bool)
    masks[0, :, 10:] = False
    result = moving.nearest_view((depth, mask), depths, masks, [5, 6])
    assert result["source_index"] == 6


@pytest.mark.parametrize("reference_empty", [False, True])
def test_no_mutual_geometry_is_never_a_perfect_match(moving, reference_empty):
    depth = np.full((12, 12), 100.0)
    mask = np.full_like(depth, not reference_empty, dtype=bool)
    result = moving.nearest_view(
        (depth, mask), np.zeros((2, 12, 12)), np.zeros((2, 12, 12), dtype=bool), [12, 13]
    )
    assert result["comparable"] is False
    assert result["score"] is None
    assert result["descriptor_mask_iou"] == (None if reference_empty else 0)
    assert result["median_mutual_depth_difference"] is None
    assert len(result["reasons"]) == 4


def test_best_available_view_can_still_be_unmatchable(moving):
    depth = np.full((12, 12), 100.0)
    mask = np.ones_like(depth, dtype=bool)
    result = moving.nearest_view(
        (depth, mask), np.stack([depth + 2, depth + 3]), np.stack([mask, mask]), [17, 99]
    )
    assert result["source_index"] == 17
    assert result["comparable"] is False
    assert result["reasons"] == [
        "missing_aware_score_above_1.5",
        "median_mutual_depth_difference_above_1.5",
    ]


def test_matching_requires_one_to_one_index_evidence_and_shapes(moving):
    depth = np.full((12, 12), 100.0)
    mask = np.ones_like(depth, dtype=bool)
    with pytest.raises(ValueError, match="one-to-one"):
        moving.nearest_view((depth, mask), depth[None], mask[None], [])
    with pytest.raises(ValueError, match="shapes"):
        moving.nearest_view((depth, mask), depth[None, :, :-1], mask[None], [12])


def test_region_loss_and_depth_shift_are_not_warped_away(moving):
    initial = np.zeros((20, 20), dtype=np.uint16)
    initial[3:17, 3:17] = 20000
    changed = initial.copy()
    changed[3:17, 3:17] += 40
    changed[3:17, 3] = 0
    result = moving.paired_regions(initial, changed)
    assert result["component"]["return_retention"] == pytest.approx(13 / 14)
    assert result["core"]["return_retention"] == 1
    assert result["edge"]["return_retention"] < 1
    assert result["core"]["absolute_depth_difference_p50_p90"] == pytest.approx([0.2, 0.2])


def test_empty_reference_regions_remain_explicit(moving):
    empty = np.zeros((20, 20), dtype=np.uint16)
    result = moving.paired_regions(empty, empty)
    assert set(result) == {"all_reference_returns", "component", "core", "edge"}
    assert all(row["return_retention"] is None for row in result.values())
    assert all(row["reference_pixels"] == 0 for row in result.values())


def test_common_bin_summary_reports_omissions_and_same_anchors(moving):
    def records(comparable, retained):
        return [
            dict(
                anchor_source_index=i * 5,
                match=dict(comparable=i in comparable),
                regions={
                    name: dict(return_retention=retained)
                    for name in ("all_reference_returns", "component", "core", "edge")
                },
            )
            for i in range(60)
        ]

    comparisons = {
        "a2-20000": records(set(range(50)), 0.95),
        "b1-14000": records(set(range(10, 60)), 0.8),
        "b2-14000": records(set(), 0.9),
    }
    result = moving.common_bin_summary(comparisons)
    one = result["b1-14000"]
    assert one["comparable_to_control_and_candidate_bins"] == list(range(10, 50))
    assert one["omitted_from_matched_summary_bins"] == list(range(10)) + list(range(50, 60))
    assert one["regions"]["core"]["candidate_minus_control_p10_p50_p90"] == pytest.approx(
        [-0.15] * 3
    )
    assert result["b2-14000"]["regions"]["core"]["candidate_retention_p10_p50_p90"] is None
    assert len(comparisons["b1-14000"]) == 60
    comparisons["b1-14000"][0]["anchor_source_index"] = 999
    with pytest.raises(ValueError, match="identical anchor"):
        moving.common_bin_summary(comparisons)


def test_second_raw_read_rechecks_hash(moving, tmp_path):
    path = tmp_path / "frame_0012.raw"
    path.write_bytes(bytes(512000))
    with pytest.raises(ValueError, match="changed"):
        moving.verified_raw(dict(source_path=str(path), sha256="0" * 64))
