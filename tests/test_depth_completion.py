"""The spatial completion primitive, proved conservative rather than merely dense.

``tests/test_depth_completeness.py`` scores this module inside the accepted
benchmark, against rendered scenes and held-out truth. This file owns the
properties underneath that score, on patches small enough to reason about by
hand: that a returned sample is never edited, that a hole with no evidence
behind it stays a hole, that support is always original measurement, that the
confidence moves in the direction it claims to, and that the work is bounded by
something derivable before the run.

The refusals are the load-bearing part. A completion that filled a silhouette,
extrapolated past the edge of its evidence, averaged across a depth step or
cited its own output as support would produce a denser and entirely wrong
frame, so each of those is a test rather than a comment.

Nothing here touches a scanner, a capture under ``/home/ben/scans``, a display
server, a GPU, the network or the clock.
"""

from __future__ import annotations

import dataclasses
import math
import struct
from collections.abc import Sequence
from typing import Final

import pytest

from scanner3d.compute import ComputeBackendUnavailable, ComputeBackendUnknown
from scanner3d.depth_completion import (
    DEFAULT_POLICY,
    MAX_CODE,
    MAX_COMPONENT_SAMPLES,
    MAX_COMPONENT_SPAN,
    MAX_PATCH_DIMENSION,
    MAX_PATCH_SAMPLES,
    MAX_RADIUS,
    MAX_SPREAD,
    MAX_SUPPORT,
    MAX_WORK_UNITS,
    REJECTION_REASONS,
    STATE_MEASURED,
    STATE_MISSING,
    STATE_SPATIALLY_RECONSTRUCTED,
    CompletedPatch,
    CompletionPolicy,
    CompletionWork,
    DepthPatch,
    PatchUnusable,
    PolicyUnusable,
    ResultUnusable,
    SupportFit,
    SupportSample,
    WorkTooLarge,
    complete_patch,
    fit_support,
    hole_components,
    select_backend,
    support_confidence,
)

WIDTH = 21
HEIGHT = 21


def _plane_codes(width=WIDTH, height=HEIGHT, *, base=10000, per_column=3, per_row=2):
    """A tilted plane in raw codes, exact by construction and easy to check by eye."""
    return [
        base + per_column * column + per_row * row
        for row in range(height)
        for column in range(width)
    ]


def _drop(codes, width, positions):
    """The same codes with the named ``(column, row)`` samples set to no-return."""
    out = list(codes)
    for column, row in positions:
        out[row * width + column] = 0
    return out


def _patch(codes, width=WIDTH, height=HEIGHT) -> DepthPatch:
    return DepthPatch.of(width, height, codes)


def _reasons(result: CompletedPatch) -> dict[str, int]:
    return {reason: count for reason, count in result.work.rejections.items() if count}


# --- the patch ------------------------------------------------------------------------


def test_a_patch_that_is_not_its_own_rectangle_is_refused():
    with pytest.raises(PatchUnusable, match="denominator is the rectangle"):
        DepthPatch.of(4, 4, [1] * 15)


@pytest.mark.parametrize("size", [(0, 4), (4, 0), (-1, 4)])
def test_a_patch_that_spans_nothing_is_refused(size):
    with pytest.raises(PatchUnusable, match=r"spans nothing|not a size"):
        DepthPatch.of(size[0], size[1], [])


@pytest.mark.parametrize("code", [-1, MAX_CODE + 1, 1.5, True, "8"])
def test_a_patch_carrying_something_that_is_not_a_uint16_code_is_refused(code):
    with pytest.raises(PatchUnusable, match="not a uint16 depth code"):
        DepthPatch.of(2, 1, [7, code])


def test_a_patch_read_out_of_a_frame_is_the_rectangle_it_names():
    frame = bytearray(8 * 4 * 2)
    for row in range(4):
        for column in range(8):
            struct.pack_into("<H", frame, (row * 8 + column) * 2, 100 * row + column)
    patch = DepthPatch.from_frame(
        bytes(frame), frame_width=8, frame_height=4, u0=2, v0=1, width=3, height=2
    )
    assert patch.codes == (102, 103, 104, 202, 203, 204)


def test_a_frame_whose_byte_count_is_not_what_it_was_told_is_refused():
    with pytest.raises(PatchUnusable, match="byte"):
        DepthPatch.from_frame(b"\x00" * 10, frame_width=8, frame_height=4)


def test_a_patch_that_leaves_its_frame_is_refused():
    with pytest.raises(PatchUnusable, match="leaves the"):
        DepthPatch.from_frame(
            b"\x00" * 64, frame_width=8, frame_height=4, u0=6, v0=0, width=4, height=2
        )


def test_a_zero_code_is_a_hole_and_never_a_measurement():
    patch = _patch(_drop(_plane_codes(), WIDTH, [(5, 5)]))
    assert 5 * WIDTH + 5 in patch.hole_indices()
    assert not patch.is_measured(5 * WIDTH + 5)


# --- the policy -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("radius", 0),
        ("radius", -1),
        ("radius", True),
        ("radius", 2.5),
        ("min_support", 0),
        ("max_component_samples", 0),
        ("max_component_span", 0),
        ("min_spread", 0.0),
        ("min_spread", float("inf")),
        ("min_spread", float("nan")),
        ("cluster_gap_codes", -1.0),
        ("max_residual_codes", 0.0),
        ("max_residual_codes", float("inf")),
        ("max_overshoot_codes", float("nan")),
        ("max_support_span_codes", float("inf")),
        ("min_confidence", 0.0),
        ("min_confidence", 1.5),
        ("min_confidence", True),
        ("min_confidence", float("nan")),
        ("passes", 2),
        ("passes", 0),
        ("min_code", 0),
        ("max_code", MAX_CODE + 1),
    ],
)
def test_a_policy_bound_that_is_not_a_bound_is_refused(field, value):
    with pytest.raises(PolicyUnusable):
        CompletionPolicy(**{field: value})


def test_a_policy_may_not_want_more_support_than_it_allows():
    with pytest.raises(PolicyUnusable, match="allows at most"):
        CompletionPolicy(min_support=12, max_support=8)


def test_a_policy_may_not_want_more_support_than_its_radius_can_hold():
    with pytest.raises(PolicyUnusable, match="holds at most"):
        CompletionPolicy(radius=1, min_support=9, max_support=9)


def test_a_policy_whose_coarse_bound_could_never_fire_is_refused():
    with pytest.raises(PolicyUnusable, match="could never fire"):
        CompletionPolicy(cluster_gap_codes=500.0, max_support_span_codes=100.0)


def test_a_policy_predicting_into_no_range_at_all_is_refused():
    with pytest.raises(PolicyUnusable, match="not a range"):
        CompletionPolicy(min_code=900, max_code=900)


# --- the backend ----------------------------------------------------------------------


def test_auto_and_python_both_give_the_reference():
    assert select_backend() == "python"
    assert select_backend("python") == "python"


@pytest.mark.parametrize("name", ["numpy", "scipy", "gpu"])
def test_a_recognised_backend_this_module_has_not_written_is_refused(name):
    with pytest.raises(ComputeBackendUnavailable, match="only implementation"):
        select_backend(name)


def test_a_backend_nobody_named_is_refused():
    with pytest.raises(ComputeBackendUnknown):
        select_backend("cuda")


def test_a_completion_never_silently_downgrades_to_a_backend_it_has():
    with pytest.raises(ComputeBackendUnavailable):
        complete_patch(_patch(_plane_codes()), backend="numpy")


# --- raw evidence is never edited ------------------------------------------------------


def test_every_returned_sample_comes_back_exactly_and_is_called_a_measurement():
    codes = _drop(_plane_codes(), WIDTH, [(5, 5), (10, 10)])
    patch = _patch(codes)
    result = complete_patch(patch)
    for index, code in enumerate(codes):
        if code == 0:
            continue
        assert result.codes[index] == code
        assert result.state_at(index) == "measured"
        assert result.confidence[index] == 1.0
        assert result.support_at(index) == ()


def test_a_refused_hole_stays_code_zero_confidence_zero_and_missing():
    # One sample in an otherwise empty patch: there is nothing around it, and
    # a completion with nothing to reason from must leave it exactly as it was.
    codes = [0] * (WIDTH * HEIGHT)
    codes[10 * WIDTH + 10] = 12345
    result = complete_patch(_patch(codes))
    assert result.work.accepted == 0
    for index in range(result.sample_count):
        if index == 10 * WIDTH + 10:
            continue
        assert result.codes[index] == 0
        assert result.confidence[index] == 0.0
        assert result.state_at(index) == "missing"


def test_the_input_patch_is_not_mutated_by_completing_it():
    patch = _patch(_drop(_plane_codes(), WIDTH, [(5, 5)]))
    before = patch.codes
    complete_patch(patch)
    assert patch.codes == before


# --- what is recovered, and what is refused --------------------------------------------


def test_an_isolated_hole_on_a_tilted_plane_is_recovered_exactly():
    codes = _plane_codes()
    truth = codes[7 * WIDTH + 7]
    result = complete_patch(_patch(_drop(codes, WIDTH, [(7, 7)])))
    assert result.work.accepted == 1
    assert result.codes[7 * WIDTH + 7] == truth
    assert result.state_at(7 * WIDTH + 7) == "spatially_reconstructed"
    assert 0.0 < result.confidence[7 * WIDTH + 7] < 1.0


#: A policy whose component bounds cannot fire, so a control aimed at a
#: PER-SAMPLE refusal gets that refusal rather than the whole-component one.
#: Everything else is the shipped default.
NO_COMPONENT_BOUND = CompletionPolicy(
    max_component_samples=WIDTH * HEIGHT, max_component_span=max(WIDTH, HEIGHT)
)


def test_a_hole_with_nothing_around_it_is_refused_for_want_of_support():
    codes = [0] * (WIDTH * HEIGHT)
    for column in range(WIDTH):
        codes[0 * WIDTH + column] = 5000
    result = complete_patch(_patch(codes), policy=NO_COMPONENT_BOUND)
    assert result.work.accepted == 0
    assert _reasons(result)["insufficient_support"] > 0


def test_a_hole_with_support_on_one_side_only_is_refused():
    """The silhouette shape. Everything above row 10 measured, nothing below it.

    A fit through that support determines a plane, and every value it would put
    below the boundary is a guess about a surface nothing measured. Refused as
    extrapolation past the edge of the evidence rather than published at a low
    confidence.
    """
    codes = _plane_codes()
    for row in range(10, HEIGHT):
        for column in range(WIDTH):
            codes[row * WIDTH + column] = 0
    result = complete_patch(_patch(codes), policy=NO_COMPONENT_BOUND)
    assert result.work.accepted == 0
    assert _reasons(result)["one_sided_support"] > 0


def test_a_hole_between_two_incompatible_surfaces_is_refused_rather_than_averaged():
    """A depth step: near surface left of column 10, far surface right of it.

    The one hole sits on the boundary with support from both, and the gap
    between the two clusters is what refuses it. Averaging them would put the
    sample in the empty band between two real surfaces, which is a flying pixel
    however plausible the arithmetic looked.
    """
    codes = [(8000 if column < 10 else 20000) for row in range(HEIGHT) for column in range(WIDTH)]
    codes[10 * WIDTH + 10] = 0
    result = complete_patch(_patch(codes))
    assert result.work.accepted == 0
    assert _reasons(result)["incompatible_clusters"] == 1


def test_a_large_connected_hole_is_refused_whole_and_by_name():
    codes = _plane_codes()
    for row in range(4, 16):
        for column in range(4, 16):
            codes[row * WIDTH + column] = 0
    result = complete_patch(_patch(codes))
    assert result.work.accepted == 0
    assert _reasons(result) == {"component_too_large": 144}


def test_a_long_thin_hole_is_refused_even_when_it_is_small():
    """A collapsed column. Few samples, and it spans the whole patch.

    The narrow-slot ambiguity in miniature: measured surface either side at the
    same depth, and nothing inside one frame that says whether the gap is
    dropout on one surface or a real gap with something else behind it.
    """
    policy = CompletionPolicy(max_component_samples=WIDTH * HEIGHT)
    codes = _drop(_plane_codes(), WIDTH, [(10, row) for row in range(HEIGHT)])
    result = complete_patch(_patch(codes), policy=policy)
    assert result.work.accepted == 0
    assert _reasons(result)["component_too_long"] == HEIGHT


def test_support_is_always_a_sample_the_input_returned():
    """No reconstruction is evidence for another one, at any distance.

    Two holes side by side: whichever is completed first, the other one's
    support may not contain it. Checked against the INPUT rather than against
    the output, which is the only version of this claim that a fill creeping
    across a frame one sample at a time could not satisfy.
    """
    codes = _drop(_plane_codes(), WIDTH, [(7, 7), (8, 7), (7, 8)])
    patch = _patch(codes)
    result = complete_patch(patch)
    assert result.work.accepted == 3
    for index in result.reconstructed_indices():
        for source in result.support_at(index):
            assert patch.is_measured(source)
            assert result.state_at(source) == "measured"


def test_a_second_pass_over_the_result_is_refused_by_the_policy():
    with pytest.raises(PolicyUnusable, match="no reconstruction is evidence"):
        CompletionPolicy(passes=2)


# --- reproducing an accepted sample from its own evidence -------------------------------


def _independent_plane(offsets, values):
    """A least-squares plane, solved by a different route than the module's.

    Centred normal equations with an explicit two-by-two inverse, rather than
    the module's three-by-three elimination about the sample. Same answer if
    both are right, and no shared code path, which is what makes this a
    reproduction rather than a restatement.
    """
    count = len(values)
    mean_column = sum(column for column, _row in offsets) / count
    mean_row = sum(row for _column, row in offsets) / count
    mean_value = sum(values) / count
    s_cc = sum((column - mean_column) ** 2 for column, _row in offsets)
    s_rr = sum((row - mean_row) ** 2 for _column, row in offsets)
    s_cr = sum((column - mean_column) * (row - mean_row) for column, row in offsets)
    s_cv = sum(
        (column - mean_column) * (value - mean_value)
        for (column, _row), value in zip(offsets, values, strict=True)
    )
    s_rv = sum(
        (row - mean_row) * (value - mean_value)
        for (_column, row), value in zip(offsets, values, strict=True)
    )
    determinant = s_cc * s_rr - s_cr * s_cr
    slope_column = (s_rr * s_cv - s_cr * s_rv) / determinant
    slope_row = (s_cc * s_rv - s_cr * s_cv) / determinant
    intercept = mean_value - slope_column * mean_column - slope_row * mean_row
    return intercept


def test_every_accepted_sample_is_reproducible_from_only_its_named_support():
    """The claim the whole benchmark rests on, checked sample by sample.

    Nothing but the support indices the result published and the policy it
    published beside them. If a completion could produce a value its own stated
    evidence does not give, "supported" would be a label rather than a fact.
    """
    codes = _plane_codes()
    holes = [(6, 6), (7, 12), (13, 7), (14, 14)]
    patch = _patch(_drop(codes, WIDTH, holes))
    result = complete_patch(patch)
    assert result.work.accepted == len(holes)
    for index in result.reconstructed_indices():
        column, row = result.position_of(index)
        sources = result.support_at(index)
        offsets = []
        values = []
        for source in sources:
            source_column, source_row = result.position_of(source)
            offsets.append((source_column - column, source_row - row))
            values.append(float(patch.codes[source]))
        assert result.codes[index] == round(_independent_plane(offsets, values))


def test_omitting_any_support_fact_changes_the_value_it_reproduces():
    """Fail closed: the support list is the evidence, not a decoration beside it."""
    patch = _patch(_drop(_plane_codes(), WIDTH, [(6, 6)]))
    result = complete_patch(patch)
    index = result.reconstructed_indices()[0]
    column, row = result.position_of(index)
    sources = list(result.support_at(index))
    trimmed = sources[:-1]
    offsets = []
    values = []
    for source in trimmed:
        source_column, source_row = result.position_of(source)
        offsets.append((source_column - column, source_row - row))
        values.append(float(patch.codes[source]) + 400.0)
    assert round(_independent_plane(offsets, values)) != result.codes[index]


# --- confidence -------------------------------------------------------------------------


def _fit_of(support):
    fit = fit_support(support)
    assert fit is not None
    return fit


def _ring(code_at):
    """Eight supporting samples around a hole, with a code chosen per offset."""
    return [
        SupportSample(
            index=index,
            column_offset=column,
            row_offset=row,
            code=code_at(column, row),
        )
        for index, (column, row) in enumerate(
            ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
        )
    ]


#: A fit in the middle of every bound the default policy states, so a control
#: can vary exactly one of its terms and read the effect of that term alone.
#: Varying a real neighbourhood instead would move several at once - moving the
#: support outwards spreads it as well as distancing it - and a test that
#: cannot say which term moved is not a test of a formula.
BASELINE_FIT = SupportFit(
    predicted=10000.0,
    worst_residual=0.5,
    mean_distance=1.5,
    code_span=20.0,
    weaker_spread=2.0,
    support_count=12,
)


def _confidence(**overrides):
    fields = {
        "predicted": BASELINE_FIT.predicted,
        "worst_residual": BASELINE_FIT.worst_residual,
        "mean_distance": BASELINE_FIT.mean_distance,
        "code_span": BASELINE_FIT.code_span,
        "weaker_spread": BASELINE_FIT.weaker_spread,
        "support_count": BASELINE_FIT.support_count,
    }
    return support_confidence(SupportFit(**(fields | overrides)), DEFAULT_POLICY)


def test_confidence_falls_when_the_fit_gets_worse():
    assert _confidence(worst_residual=1.5) < _confidence()


def test_confidence_falls_when_the_support_moves_further_away():
    assert _confidence(mean_distance=3.0) < _confidence()


def test_confidence_falls_when_the_support_disagrees_about_the_depth():
    assert _confidence(code_span=200.0) < _confidence()


def test_confidence_rises_with_more_supporting_samples():
    assert _confidence(support_count=16) > _confidence()


def test_confidence_rises_when_the_support_surrounds_the_sample_more_evenly():
    assert _confidence(weaker_spread=6.0) > _confidence()


def test_confidence_is_never_outside_the_fraction_it_is_declared_to_be():
    """Every term saturates rather than escaping, in both directions."""
    assert _confidence(support_count=1000, weaker_spread=1e6, mean_distance=0.0) <= 1.0
    assert _confidence(worst_residual=1e6) == 0.0
    assert _confidence(mean_distance=float("inf")) == 0.0


def test_a_real_neighbourhood_is_scored_by_the_same_formula():
    """The formula and the fit agree: no second confidence rule anywhere."""
    ring = _ring(lambda column, row: 10000 + column + row)
    fit = _fit_of(ring)
    codes = _drop(_plane_codes(base=10000, per_column=1, per_row=1), WIDTH, [(7, 7)])
    result = complete_patch(_patch(codes))
    index = 7 * WIDTH + 7
    assert result.confidence[index] > 0.0
    assert support_confidence(fit, DEFAULT_POLICY) > 0.0


def test_confidence_is_a_finite_fraction_of_one_everywhere_it_is_published():
    codes = _drop(_plane_codes(), WIDTH, [(6, 6), (7, 12), (13, 7)])
    result = complete_patch(_patch(codes))
    for value in result.confidence:
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    for index in result.reconstructed_indices():
        assert result.confidence[index] < 1.0, "a reconstruction is not a measurement"


def test_raising_the_confidence_threshold_only_ever_refuses_more():
    """Thresholded rejection is stable: the accepted set shrinks and never moves."""
    codes = _drop(
        _plane_codes(),
        WIDTH,
        [(6, 6), (7, 12), (13, 7), (14, 14), (5, 15), (15, 5)],
    )
    patch = _patch(codes)
    accepted = []
    for threshold in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0):
        result = complete_patch(patch, policy=CompletionPolicy(min_confidence=threshold))
        accepted.append(set(result.reconstructed_indices()))
    for tighter, looser in zip(accepted[1:], accepted[:-1], strict=True):
        assert tighter <= looser


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("radius", 3),
        ("min_support", 12),
        ("max_residual_codes", 0.5),
        ("cluster_gap_codes", 1.0),
        ("max_component_samples", 1),
        ("max_component_span", 1),
        ("max_overshoot_codes", 0.5),
        ("min_spread", 4.0),
    ],
)
def test_tightening_any_bound_only_ever_refuses_more(field, value):
    """Every knob in the policy is a conservatism knob, in the same direction."""
    codes = _drop(
        _plane_codes(),
        WIDTH,
        [(6, 6), (7, 12), (13, 7), (14, 14), (5, 15), (15, 5), (9, 9), (10, 9)],
    )
    patch = _patch(codes)
    loose = set(complete_patch(patch).reconstructed_indices())
    tight = set(
        complete_patch(patch, policy=CompletionPolicy(**{field: value})).reconstructed_indices()
    )
    assert tight <= loose


# --- work is bounded, and reported ------------------------------------------------------


def test_work_is_bounded_by_a_budget_derivable_before_the_run():
    codes = _drop(_plane_codes(), WIDTH, [(6, 6), (7, 12), (13, 7)])
    patch = _patch(codes)
    result = complete_patch(patch)
    expected = len(patch.hole_indices()) * DEFAULT_POLICY.neighbourhood_samples
    assert result.work.work_budget == expected
    assert result.work.work_units <= expected
    assert result.work.within_budget


def test_every_hole_is_accounted_for_by_acceptance_or_by_a_named_reason():
    codes = _plane_codes()
    for row in range(4, 16):
        for column in range(4, 16):
            codes[row * WIDTH + column] = 0
    codes[1 * WIDTH + 1] = 0
    result = complete_patch(_patch(codes))
    work = result.work
    assert work.holes == 145
    assert work.attempted == work.holes
    assert work.accepted + work.rejected == work.attempted
    assert sum(work.rejections.values()) == work.rejected
    assert sorted(work.rejections) == sorted(REJECTION_REASONS)


def test_the_same_patch_always_gives_the_same_result():
    codes = _drop(_plane_codes(), WIDTH, [(6, 6), (7, 12), (13, 7), (9, 9)])
    first = complete_patch(_patch(codes))
    second = complete_patch(_patch(codes))
    assert first.codes == second.codes
    assert first.state_codes == second.state_codes
    assert first.confidence == second.confidence
    assert first.support == second.support
    assert first.work.as_dict() == second.work.as_dict()


def test_hole_components_are_four_connected_and_deterministic():
    codes = _drop(_plane_codes(), WIDTH, [(5, 5), (6, 6)])
    components = hole_components(_patch(codes))
    assert [component.indices for component in components] == [
        (5 * WIDTH + 5,),
        (6 * WIDTH + 6,),
    ]


# --- a result that does not hold up is refused -------------------------------------------


#: A policy that asks for one supporting sample, so a control aimed at a
#: RESULT rule is not refused first by the work record's support arithmetic.
ONE_SUPPORT = CompletionPolicy(min_support=1)


def _work_of(**overrides: object) -> CompletionWork:
    fields: dict[str, object] = {
        "backend": "python",
        "policy": DEFAULT_POLICY,
        "holes": 1,
        "attempted": 1,
        "accepted": 0,
        "rejected": 1,
        "rejections": dict.fromkeys(REJECTION_REASONS, 0) | {"insufficient_support": 1},
        "support_samples": 0,
        "work_units": 1,
        "work_budget": DEFAULT_POLICY.neighbourhood_samples,
    }
    return CompletionWork(**(fields | overrides))  # type: ignore[arg-type]


def test_a_work_record_that_does_not_add_up_is_refused():
    with pytest.raises(ResultUnusable, match="accounts for"):
        _work_of(accepted=1)


def test_a_work_record_that_looked_at_fewer_holes_than_it_had_is_refused():
    with pytest.raises(ResultUnusable, match="every hole is examined once"):
        _work_of(holes=2, attempted=1, accepted=0, rejected=1)


def test_a_work_record_whose_reasons_do_not_cover_its_refusals_is_refused():
    with pytest.raises(ResultUnusable, match="gives reasons for"):
        _work_of(rejections=dict.fromkeys(REJECTION_REASONS, 0))


def test_a_work_record_with_a_reason_nobody_declared_is_refused():
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"felt_wrong": 1}
    with pytest.raises(ResultUnusable, match="declared vocabulary"):
        _work_of(rejections=reasons)


def test_a_work_record_that_overspent_its_own_budget_is_refused():
    with pytest.raises(ResultUnusable, match="not advisory"):
        _work_of(work_units=10_000, work_budget=10)


def test_a_work_record_naming_support_its_policy_could_not_have_used_is_refused():
    with pytest.raises(ResultUnusable, match="its own policy allows"):
        _work_of(accepted=0, rejected=1, support_samples=5)


def _source_with_holes(codes: Sequence[int], holes: Sequence[int]) -> DepthPatch:
    """The raw 4x4 frame a hand-built result claims to complete.

    Every control below builds a result by hand; each one needs the immutable
    frame it is supposed to be derived from, with a no-return sample exactly
    where the result claims to have reconstructed one.
    """
    raw = list(codes)
    for index in holes:
        raw[index] = 0
    return DepthPatch.of(4, 4, raw)


def _result_of(**overrides: object) -> CompletedPatch:
    codes = _drop(_plane_codes(4, 4, base=9000, per_column=0, per_row=0), 4, [(1, 1)])
    fields: dict[str, object] = {
        "width": 4,
        "height": 4,
        "source": DepthPatch.of(4, 4, codes),
        "codes": tuple(codes),
        "state_codes": bytes(STATE_MISSING if code == 0 else STATE_MEASURED for code in codes),
        "confidence": tuple(0.0 if code == 0 else 1.0 for code in codes),
        "support": {},
        "work": _work_of(),
    }
    return CompletedPatch(**(fields | overrides))  # type: ignore[arg-type]


def test_a_result_that_is_not_the_whole_rectangle_is_refused():
    with pytest.raises(ResultUnusable, match="cannot be counted against it"):
        _result_of(codes=tuple(range(1, 16)))


def test_a_result_carrying_a_confidence_outside_the_fraction_is_refused():
    confidence = [1.0] * 16
    confidence[0] = 1.5
    with pytest.raises(ResultUnusable, match="finite fraction of 1"):
        _result_of(confidence=tuple(confidence))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_a_result_carrying_a_confidence_that_is_not_a_number_is_refused(value):
    confidence = [1.0] * 16
    confidence[0] = value
    with pytest.raises(ResultUnusable):
        _result_of(confidence=tuple(confidence))


def test_a_result_citing_a_reconstruction_as_support_is_refused():
    """The recursive-support failure, stated at the type rather than left to a caller.

    Checked against the SOURCE frame, not against the result's own labels: index
    5 is a hole in the frame that was handed over, so naming it as support is
    naming a sample the sensor never returned, whatever the result calls it.
    """
    codes = [9000] * 16
    states = bytearray(STATE_MEASURED for _ in codes)
    states[5] = STATE_SPATIALLY_RECONSTRUCTED
    states[6] = STATE_SPATIALLY_RECONSTRUCTED
    with pytest.raises(ResultUnusable, match="Only an original measurement is evidence"):
        CompletedPatch(
            width=4,
            height=4,
            source=_source_with_holes(codes, (5, 6)),
            codes=tuple(codes),
            state_codes=bytes(states),
            confidence=tuple(0.5 if index in (5, 6) else 1.0 for index in range(16)),
            support={5: (0,), 6: (5,)},
            work=_work_of(
                policy=ONE_SUPPORT,
                holes=2,
                attempted=2,
                accepted=2,
                rejected=0,
                rejections=dict.fromkeys(REJECTION_REASONS, 0),
                support_samples=2,
                work_units=2,
                work_budget=2 * ONE_SUPPORT.neighbourhood_samples,
            ),
        )


def test_a_reconstruction_with_no_named_support_is_refused():
    codes = [9000] * 16
    states = bytearray(STATE_MEASURED for _ in codes)
    states[5] = STATE_SPATIALLY_RECONSTRUCTED
    states[6] = STATE_SPATIALLY_RECONSTRUCTED
    with pytest.raises(ResultUnusable, match="invention"):
        CompletedPatch(
            width=4,
            height=4,
            source=_source_with_holes(codes, (5, 6)),
            codes=tuple(codes),
            state_codes=bytes(states),
            confidence=tuple(0.5 if index in (5, 6) else 1.0 for index in range(16)),
            support={5: (0, 1)},
            work=_work_of(
                policy=ONE_SUPPORT,
                holes=2,
                attempted=2,
                accepted=2,
                rejected=0,
                rejections=dict.fromkeys(REJECTION_REASONS, 0),
                support_samples=2,
                work_units=2,
                work_budget=2 * ONE_SUPPORT.neighbourhood_samples,
            ),
        )


def test_a_result_whose_reconstruction_count_is_not_its_work_record_is_refused():
    codes = [9000] * 16
    states = bytearray(STATE_MEASURED for _ in codes)
    states[5] = STATE_SPATIALLY_RECONSTRUCTED
    with pytest.raises(ResultUnusable, match="beside work that accepted"):
        CompletedPatch(
            width=4,
            height=4,
            source=_source_with_holes(codes, (5,)),
            codes=tuple(codes),
            state_codes=bytes(states),
            confidence=tuple(0.5 if index == 5 else 1.0 for index in range(16)),
            support={5: (0,)},
            work=_work_of(
                policy=ONE_SUPPORT,
                holes=1,
                attempted=1,
                accepted=0,
                rejected=1,
                support_samples=0,
                work_units=1,
            ),
        )


def test_a_fit_that_no_plane_determines_is_none_rather_than_an_answer():
    collinear = [
        SupportSample(index=index, column_offset=offset, row_offset=0, code=10000 + offset)
        for index, offset in enumerate((-3, -2, -1, 1, 2, 3))
    ]
    fit = fit_support(collinear)
    assert fit is None or fit.weaker_spread == 0.0
    assert fit_support(()) is None


def test_the_summary_of_a_result_counts_every_sample_once():
    codes = _drop(_plane_codes(), WIDTH, [(6, 6), (7, 12)])
    summary = complete_patch(_patch(codes)).as_dict()
    counted = sum(
        int(str(summary[key])) for key in ("measured", "spatially_reconstructed", "missing")
    )
    assert counted == summary["sample_count"]


# --- the result is bound to the frame it claims to preserve ------------------------------
#
# Every control below is a direct construction: it does not go through
# complete_patch, because the question these answer is what the public type
# accepts from anyone, not what this module's own algorithm happens to build.


#: The 4x4 frame every control in this section perturbs: constant depth with
#: exactly one no-return sample in it, at index 5.
_ONE_HOLE_SOURCE: Final = DepthPatch.of(4, 4, [9000] * 5 + [0] + [9000] * 10)

#: The support this module itself selects for that hole under
#: :data:`ONE_SUPPORT`: every other sample of the patch, because all fifteen of
#: them are inside the radius and the policy allows sixteen.
_ONE_HOLE_SUPPORT: Final = tuple(index for index in range(16) if index != 5)


def _reconstruction(**overrides: object) -> CompletedPatch:
    """A real 4x4 result that reconstructs one hole, with one thing changed.

    Derived rather than hand-built, and that is the round-three rule rather
    than a convenience: a published result is now required to *be* the
    deterministic outcome of the source it names under the policy it
    publishes, so a plausible-looking result assembled by hand is refused
    however consistent its own labels are. The valid starting point is
    therefore one this module produced, and every control below perturbs
    exactly one field of it, so what each of them tests is the rule it names
    and not some other rule it tripped on the way.
    """
    base = complete_patch(_ONE_HOLE_SOURCE, policy=ONE_SUPPORT)
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def test_the_valid_derived_reconstruction_is_accepted():
    """So that every refusal below is the rule it names and not the fixture."""
    result = _reconstruction()
    assert result.state_at(5) == "spatially_reconstructed"
    assert result.support_at(5) == _ONE_HOLE_SUPPORT
    assert result.codes[5] == 9000
    assert 0.0 < result.confidence[5] < 1.0


def test_a_result_that_changes_a_measured_code_and_keeps_the_label_is_refused():
    """The reviewer's first reproduction, at the constructor.

    Complete input, one result code replaced, the state left at ``measured``.
    Nothing about the result's own labels contradicts that; only the source
    frame does, which is exactly why the source is a field.
    """
    result = complete_patch(DepthPatch.of(3, 3, [100] * 9))
    with pytest.raises(ResultUnusable, match="byte-for-byte"):
        dataclasses.replace(result, codes=(999, *result.codes[1:]))


def test_a_result_that_gives_a_missing_sample_a_depth_and_a_confidence_is_refused():
    """The reviewer's second reproduction: a refused hole handed a value anyway."""
    result = complete_patch(DepthPatch.of(3, 3, [0] * 9))
    assert result.state_at(4) == "missing"
    codes = list(result.codes)
    codes[4] = 777
    confidence = list(result.confidence)
    confidence[4] = 0.9
    with pytest.raises(ResultUnusable, match="exactly code zero, confidence zero"):
        dataclasses.replace(result, codes=tuple(codes), confidence=tuple(confidence))


def test_a_result_that_relabels_a_no_return_sample_as_a_measurement_is_refused():
    states = bytearray(_reconstruction().state_codes)
    states[5] = STATE_MEASURED
    with pytest.raises(ResultUnusable, match="never be relabelled"):
        _reconstruction(state_codes=bytes(states), support={}, confidence=(1.0,) * 16)


def test_a_result_that_publishes_a_measurement_at_less_than_full_confidence_is_refused():
    confidence = [0.5 if index == 5 else 1.0 for index in range(16)]
    confidence[0] = 0.9
    with pytest.raises(ResultUnusable, match=r"carried at 1\.0 and nothing else"):
        _reconstruction(confidence=tuple(confidence))


def test_a_result_that_names_support_for_a_measurement_is_refused():
    support = {0: (1,), 5: _ONE_HOLE_SUPPORT}
    with pytest.raises(ResultUnusable, match="rests on the sensor"):
        _reconstruction(
            support=support,
            work=_accepted_one(support_samples=len(_ONE_HOLE_SUPPORT) + 1),
        )


def test_a_reconstruction_published_at_camera_confidence_is_refused():
    """One is the sensor's, and no reconstruction reaches it however good its fit."""
    confidence = [1.0] * 16
    with pytest.raises(ResultUnusable, match="reserved for what the sensor returned"):
        _reconstruction(confidence=tuple(confidence))


def test_a_reconstruction_published_at_no_confidence_at_all_is_refused():
    confidence = [0.0 if index == 5 else 1.0 for index in range(16)]
    with pytest.raises(ResultUnusable, match="worth more than nothing"):
        _reconstruction(confidence=tuple(confidence))


def test_a_reconstruction_that_carries_code_zero_is_refused():
    codes = list(_reconstruction().codes)
    codes[5] = 0
    with pytest.raises(ResultUnusable, match="no measurement rather than a depth"):
        _reconstruction(codes=tuple(codes))


def test_a_result_that_names_the_same_support_sample_twice_is_refused():
    with pytest.raises(ResultUnusable, match="same sample twice"):
        _reconstruction(support={5: (0, 0)}, work=_accepted_one(support_samples=2))


def test_support_is_believed_from_the_source_and_never_from_the_results_own_label():
    """A forged ``measured`` label cannot vouch for its own evidence.

    Index 6 is a hole in the frame that was handed over. Labelling it measured
    and citing it as support for another reconstruction is the attack; the
    source is what refuses it.
    """
    raw = [9000] * 16
    raw[5] = 0
    raw[6] = 0
    source = DepthPatch.of(4, 4, raw)
    codes = [9000] * 16
    states = bytearray(STATE_MEASURED for _ in codes)
    states[5] = STATE_SPATIALLY_RECONSTRUCTED
    with pytest.raises(ResultUnusable, match="Only an original measurement is evidence"):
        CompletedPatch(
            width=4,
            height=4,
            source=source,
            codes=tuple(codes),
            state_codes=bytes(states),
            confidence=tuple(0.5 if index == 5 else 1.0 for index in range(16)),
            support={5: (6,)},
            work=_work_of(
                policy=ONE_SUPPORT,
                holes=2,
                attempted=2,
                accepted=1,
                rejected=1,
                rejections=dict.fromkeys(REJECTION_REASONS, 0) | {"insufficient_support": 1},
                support_samples=1,
                work_units=2,
                work_budget=ONE_SUPPORT.worst_case_work(2),
            ),
        )


def test_a_result_whose_dimensions_are_not_its_sources_is_refused():
    with pytest.raises(ResultUnusable, match="does not get to move it"):
        _reconstruction(width=8, height=2)


def test_a_result_with_no_source_patch_is_refused():
    with pytest.raises(ResultUnusable, match="not a DepthPatch"):
        _reconstruction(source={"width": 4, "height": 4})


def test_a_result_stating_a_schema_version_this_module_does_not_write_is_refused():
    with pytest.raises(ResultUnusable, match="schema version"):
        _reconstruction(schema_version=2)


# --- the work record is bound to the source, the result and the policy -------------------


def _accepted_one(**overrides: object) -> CompletionWork:
    """The real work record of :func:`_reconstruction`, with counts changed.

    Derived from the run rather than restated, so a control that changes one
    count is changing it away from what actually happened.
    """
    return dataclasses.replace(_reconstruction().work, **overrides)  # type: ignore[arg-type]


def test_work_that_states_a_hole_count_the_source_does_not_have_is_refused():
    """How many holes there were is the frame's fact, not the run's."""
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"insufficient_support": 1}
    changed = _accepted_one(
        holes=2,
        attempted=2,
        rejected=1,
        rejections=reasons,
        work_budget=ONE_SUPPORT.worst_case_work(2),
    )
    with pytest.raises(ResultUnusable, match="the input's fact"):
        _reconstruction(work=changed)


def test_work_that_states_a_budget_it_did_not_derive_is_refused():
    with pytest.raises(ResultUnusable, match="not a number the run chose"):
        _reconstruction(work=_accepted_one(work_budget=10_000))


def test_work_that_claims_a_conservatism_the_result_does_not_show_is_refused():
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"residual_too_large": 1}
    changed = _accepted_one(accepted=0, rejected=1, support_samples=0, rejections=reasons)
    with pytest.raises(ResultUnusable, match="beside work that accepted"):
        _reconstruction(work=changed)


def test_work_naming_a_backend_that_cannot_have_produced_it_is_refused():
    with pytest.raises(ResultUnusable, match="only one that can have produced a run"):
        _work_of(backend="numpy")


def test_work_whose_policy_is_not_the_validated_type_is_refused():
    with pytest.raises(ResultUnusable, match="not a CompletionPolicy"):
        _work_of(policy=DEFAULT_POLICY.as_dict())


def test_work_stating_a_schema_version_this_module_does_not_write_is_refused():
    with pytest.raises(ResultUnusable, match="schema version"):
        _work_of(schema_version=0)


@pytest.mark.parametrize(
    "name", ["holes", "attempted", "accepted", "rejected", "support_samples", "work_units"]
)
def test_a_work_count_that_is_a_boolean_is_refused(name):
    with pytest.raises(ResultUnusable, match="not a count"):
        _work_of(**{name: True})


def test_a_negative_work_count_is_refused():
    with pytest.raises(ResultUnusable, match="not negative"):
        _work_of(work_units=-1, work_budget=DEFAULT_POLICY.neighbourhood_samples)


def test_a_boolean_rejection_count_is_refused():
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"insufficient_support": True}
    with pytest.raises(ResultUnusable, match="not a count"):
        _work_of(rejections=reasons)


# --- a published result is the derivation its own source and policy give ------------------
#
# The rules above say a result has a *shape* its source permits. These say it
# has the one *value* its source and its published policy determine. Every
# control here is a direct ``dataclasses.replace`` on a result this module
# produced, which is the exact form the round-two review used to forge one: the
# source, the labels and the support indices all stay real and only the
# published number changes.


def _centre_hole() -> CompletedPatch:
    """A 9x9 constant patch missing only its centre, completed for real."""
    codes = [100] * 81
    codes[40] = 0
    return complete_patch(DepthPatch.of(9, 9, codes))


#: A policy whose disc is small enough that the support of a hole is every
#: measurement near it rather than the ``max_support`` nearest, so a control can
#: add a support sample without exceeding what the policy allows one
#: reconstruction to rest on.
_SMALL_DISC: Final = CompletionPolicy(radius=2, min_support=8, max_support=16)


def _small_disc_hole() -> CompletedPatch:
    """The same constant 9x9, completed at :data:`_SMALL_DISC`."""
    codes = [100] * 81
    codes[40] = 0
    return complete_patch(DepthPatch.of(9, 9, codes), policy=_SMALL_DISC)


def _corner_and_centre() -> CompletedPatch:
    """A constant 9x9 missing its centre and its corner.

    The centre is recovered and the corner is refused for one-sided support, so
    one real result carries both an acceptance and a named refusal to forge.
    """
    codes = [100] * 81
    codes[0] = 0
    codes[40] = 0
    return complete_patch(DepthPatch.of(9, 9, codes))


def _all_holes() -> CompletedPatch:
    """A 9x9 patch that returned nothing at all, completed for real."""
    return complete_patch(DepthPatch.of(9, 9, [0] * 81))


def test_the_centre_of_a_constant_patch_is_recovered_and_nothing_else_is():
    """The fixture the controls below perturb, stated so each refusal is its own."""
    result = _centre_hole()
    assert result.reconstructed_indices() == (40,)
    assert result.codes[40] == 100
    assert result.work.work_units == DEFAULT_POLICY.neighbourhood_samples
    assert dataclasses.replace(result).codes == result.codes


def test_a_reconstructed_code_the_evidence_does_not_give_is_refused():
    """The review's first forgery: the same source and support, a different depth."""
    result = _centre_hole()
    codes = list(result.codes)
    codes[40] = 65000
    with pytest.raises(ResultUnusable, match="a number with a provenance label attached"):
        dataclasses.replace(result, codes=tuple(codes))


def test_a_confidence_the_formula_does_not_give_is_refused():
    """The review's second forgery: a real reconstruction published as near-certain."""
    result = _centre_hole()
    confidence = list(result.confidence)
    confidence[40] = 0.99
    with pytest.raises(ResultUnusable, match="not a number a result may state"):
        dataclasses.replace(result, confidence=tuple(confidence))


def test_a_policy_the_published_reconstruction_would_not_have_passed_is_refused():
    """The review's third forgery: a policy the result's own value falls below.

    ``min_confidence=0.9`` is above the confidence this reconstruction carries,
    so under the policy the result publishes the hole is refused rather than
    recovered - and a result that publishes a policy it did not obey is stating
    bounds nothing held it to.
    """
    result = _centre_hole()
    assert result.confidence[40] < 0.9
    strict = dataclasses.replace(result.work, policy=CompletionPolicy(min_confidence=0.9))
    with pytest.raises(ResultUnusable, match="not a decision a result reports"):
        dataclasses.replace(result, work=strict)


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("one support sample omitted", lambda chosen: chosen[1:]),
        ("one support sample added", lambda chosen: (*chosen, 0)),
        ("one support sample exchanged", lambda chosen: (0, *chosen[1:])),
    ],
)
def test_support_that_is_not_the_selection_the_policy_makes_is_refused(name, change):
    """Changed, omitted and extra support are all refused, not only invalid support.

    Every index in each of these is still an original measurement of the same
    frame, so nothing about the support's *validity* is what refuses them: the
    policy chooses the support, and a result that got to choose it could name
    evidence for a value that evidence does not give.
    """
    result = _small_disc_hole()
    chosen = result.support_at(40)
    assert len(chosen) < _SMALL_DISC.max_support
    forged = change(chosen)
    work = dataclasses.replace(result.work, support_samples=len(set(forged)))
    with pytest.raises(ResultUnusable, match="selects"):
        dataclasses.replace(result, support={40: forged}, work=work)


@pytest.mark.parametrize("spent", [0, 1, 80])
def test_a_work_spend_the_run_did_not_make_is_refused(spent):
    """The review's fourth forgery, and both directions of it.

    ``work_units=0`` is the exact attack: it stays under the budget, so the
    ceiling test cannot see it, and without recomputation a run that claims to
    have looked at nothing is indistinguishable from one that did.
    """
    result = _centre_hole()
    assert result.work.work_units == 81
    with pytest.raises(ResultUnusable, match="not a total that merely fits under the budget"):
        dataclasses.replace(result, work=dataclasses.replace(result.work, work_units=spent))


def test_a_rejection_taxonomy_that_only_adds_up_to_the_right_total_is_refused():
    """The review's fifth forgery: 81 real refusals moved to a reason that never fired."""
    result = _all_holes()
    assert result.work.rejections["component_too_large"] == 81
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"insufficient_support": 81}
    forged = dataclasses.replace(result.work, rejections=reasons)
    assert sum(forged.rejections.values()) == result.work.rejected
    with pytest.raises(ResultUnusable, match="the reason that actually fired"):
        dataclasses.replace(result, work=forged)


def test_a_hole_this_policy_refuses_cannot_be_published_as_a_recovery():
    """Presence: an accepted sample this source and policy do not accept.

    The corner of a constant patch has support on one side only, so it stays
    missing. Everything the forgery states about it is otherwise legal - a
    nonzero code, a confidence inside the fraction, eight support indices that
    really are measurements - and only recomputation refuses it.
    """
    result = _corner_and_centre()
    assert result.state_at(0) == "missing"
    assert result.work.rejections["one_sided_support"] == 1
    codes = list(result.codes)
    codes[0] = 100
    states = bytearray(result.state_codes)
    states[0] = STATE_SPATIALLY_RECONSTRUCTED
    confidence = list(result.confidence)
    confidence[0] = 0.5
    support = dict(result.support) | {0: (1, 2, 9, 10, 11, 18, 19, 20)}
    work = dataclasses.replace(
        result.work,
        accepted=2,
        rejected=0,
        rejections=dict.fromkeys(REJECTION_REASONS, 0),
        support_samples=result.work.support_samples + 8,
    )
    with pytest.raises(ResultUnusable, match="not a decision a result reports"):
        dataclasses.replace(
            result,
            codes=tuple(codes),
            state_codes=bytes(states),
            confidence=tuple(confidence),
            support=support,
            work=work,
        )


def test_a_hole_this_policy_recovers_cannot_be_published_as_refused():
    """Absence: hiding a real recovery is refused by the same recomputation.

    The opposite direction of the control above, and it matters for the same
    reason: a coverage number that can be lowered at will is as unreadable as
    one that can be raised.
    """
    result = _centre_hole()
    codes = list(result.codes)
    codes[40] = 0
    states = bytearray(result.state_codes)
    states[40] = STATE_MISSING
    confidence = list(result.confidence)
    confidence[40] = 0.0
    work = dataclasses.replace(
        result.work,
        accepted=0,
        rejected=1,
        rejections=dict.fromkeys(REJECTION_REASONS, 0) | {"insufficient_support": 1},
        support_samples=0,
    )
    with pytest.raises(ResultUnusable, match="not a decision a result reports"):
        dataclasses.replace(
            result,
            codes=tuple(codes),
            state_codes=bytes(states),
            confidence=tuple(confidence),
            support={},
            work=work,
        )


def test_a_policy_change_that_changes_nothing_it_derives_is_not_detectable():
    """The stated limit of this binding, written down rather than left implied.

    Recomputation binds a result to the *outcome* of its published policy, so a
    policy edit that no sample of this frame is sensitive to leaves an
    identical derivation and is accepted. Every published code, confidence and
    support selection still reproduces exactly from that policy - which is what
    a reader of the result needs - but "the policy field is the one the run was
    invoked with" is not something a single result can establish, and this
    module does not claim it.
    """
    result = _centre_hole()
    loosened = dataclasses.replace(result.work.policy, max_component_samples=47)
    assert loosened != result.work.policy
    rebound = dataclasses.replace(result, work=dataclasses.replace(result.work, policy=loosened))
    assert rebound.codes == result.codes
    assert rebound.confidence == result.confidence


# --- a number that is not the type it is compared against ---------------------------------
#
# ``True == 1`` and ``1.0 == 1``. Every equality and range test below used to
# run before the type test, so a boolean width and a float schema version were
# accepted for being numerically equal to the right answer.


@pytest.mark.parametrize("value", [True, False, 1.0, 9.0])
def test_a_result_dimension_that_is_not_an_integer_is_refused(value):
    with pytest.raises(ResultUnusable, match="which is not an integer"):
        dataclasses.replace(_centre_hole(), width=value)


@pytest.mark.parametrize("value", [True, 9.0])
def test_a_result_height_that_is_not_an_integer_is_refused(value):
    with pytest.raises(ResultUnusable, match="which is not an integer"):
        dataclasses.replace(_centre_hole(), height=value)


@pytest.mark.parametrize("value", [-1, 0, 10**9])
def test_a_result_dimension_that_is_not_its_sources_is_refused(value):
    with pytest.raises(ResultUnusable, match="does not get to move it"):
        dataclasses.replace(_centre_hole(), width=value)


@pytest.mark.parametrize("value", [-1, 0, 10**9])
def test_a_result_height_that_is_not_its_sources_is_refused(value):
    with pytest.raises(ResultUnusable, match="does not get to move it"):
        dataclasses.replace(_centre_hole(), height=value)


@pytest.mark.parametrize("value", [True, 1.0])
def test_a_result_schema_version_that_is_not_an_integer_is_refused(value):
    with pytest.raises(ResultUnusable, match="which is not an integer"):
        dataclasses.replace(_centre_hole(), schema_version=value)


@pytest.mark.parametrize("value", [True, 1.0])
def test_a_work_schema_version_that_is_not_an_integer_is_refused(value):
    with pytest.raises(ResultUnusable, match="which is not an integer"):
        dataclasses.replace(_centre_hole().work, schema_version=value)


@pytest.mark.parametrize("value", [0, 2, -1, 10**9])
def test_a_schema_version_this_module_does_not_write_is_refused(value):
    with pytest.raises(ResultUnusable, match="schema version"):
        dataclasses.replace(_centre_hole().work, schema_version=value)


# --- the immutable types are immutable all the way down ----------------------------------


def test_a_patch_does_not_change_when_the_list_it_was_built_from_does():
    codes = [9000] * 9
    patch = DepthPatch(width=3, height=3, codes=codes)  # type: ignore[arg-type]
    codes[0] = 1
    assert patch.codes == (9000,) * 9


def test_a_result_does_not_change_when_the_containers_it_was_built_from_do():
    """Every mutable input copied at construction, or "immutable" is a word."""
    derived = _reconstruction()
    codes = list(derived.codes)
    states = bytearray(derived.state_codes)
    confidence = list(derived.confidence)
    support = {5: list(_ONE_HOLE_SUPPORT)}
    result = _reconstruction(
        codes=codes, state_codes=states, confidence=confidence, support=support
    )
    codes[0] = 1
    states[0] = STATE_MISSING
    confidence[0] = 0.0
    support[5].append(2)
    support[7] = [3]
    assert result.codes[0] == 9000
    assert result.state_at(0) == "measured"
    assert result.confidence[0] == 1.0
    assert result.support_at(5) == _ONE_HOLE_SUPPORT
    assert result.support_at(7) == ()


def test_a_results_support_mapping_cannot_be_written_to():
    result = _reconstruction()
    with pytest.raises(TypeError):
        result.support[6] = (0,)  # type: ignore[index]


def test_a_work_records_rejections_cannot_be_written_to_or_changed_underneath_it():
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"insufficient_support": 1}
    work = _work_of(rejections=reasons)
    reasons["insufficient_support"] = 99
    assert work.rejections["insufficient_support"] == 1
    with pytest.raises(TypeError):
        work.rejections["insufficient_support"] = 99  # type: ignore[index]


# --- the ceilings, which are what makes the work bounded ---------------------------------


def test_the_reviewers_unbounded_policy_is_refused():
    """``radius=10**9`` is finite, and publishes a neighbourhood of 4e18 samples."""
    with pytest.raises(PolicyUnusable, match="A finite integer is not a bound on work"):
        CompletionPolicy(radius=10**9, min_support=8, max_support=16)


@pytest.mark.parametrize(
    ("name", "ceiling"),
    [
        ("radius", MAX_RADIUS),
        ("min_support", MAX_SUPPORT),
        ("max_support", MAX_SUPPORT),
        ("max_component_samples", MAX_COMPONENT_SAMPLES),
        ("max_component_span", MAX_COMPONENT_SPAN),
    ],
)
def test_every_integer_bound_has_a_ceiling_and_it_is_reachable(name, ceiling):
    at_maximum = {"radius": MAX_RADIUS, "min_support": MAX_SUPPORT, "max_support": MAX_SUPPORT}
    assert CompletionPolicy(**(at_maximum | {name: ceiling}))
    with pytest.raises(PolicyUnusable, match="past the ceiling"):
        CompletionPolicy(**(at_maximum | {name: ceiling + 1}))
    with pytest.raises(PolicyUnusable, match="past the ceiling"):
        CompletionPolicy(**(at_maximum | {name: 10**18}))


@pytest.mark.parametrize(
    ("name", "ceiling"),
    [
        ("min_spread", MAX_SPREAD),
        ("cluster_gap_codes", float(MAX_CODE)),
        ("max_support_span_codes", float(MAX_CODE)),
        ("max_residual_codes", float(MAX_CODE)),
        ("max_overshoot_codes", float(MAX_CODE)),
    ],
)
def test_every_numeric_bound_has_a_ceiling_and_it_is_reachable(name, ceiling):
    widest = {"cluster_gap_codes": float(MAX_CODE), "max_support_span_codes": float(MAX_CODE)}
    assert CompletionPolicy(**(widest | {name: ceiling}))
    with pytest.raises(PolicyUnusable, match="past the ceiling"):
        CompletionPolicy(**(widest | {name: ceiling + 1.0}))
    with pytest.raises(PolicyUnusable, match="past the ceiling"):
        CompletionPolicy(**(widest | {name: 1e18}))


@pytest.mark.parametrize(
    "name",
    [
        "radius",
        "min_support",
        "max_support",
        "max_component_samples",
        "max_component_span",
        "min_spread",
        "cluster_gap_codes",
        "max_support_span_codes",
        "max_residual_codes",
        "max_overshoot_codes",
        "min_code",
        "max_code",
        "min_confidence",
        "passes",
    ],
)
def test_no_bound_accepts_a_boolean(name):
    with pytest.raises(PolicyUnusable):
        CompletionPolicy(**{name: True})


def test_a_patch_larger_than_the_ceiling_is_refused_before_its_codes_are_counted():
    with pytest.raises(PatchUnusable, match="past the ceiling"):
        DepthPatch(width=2048, height=2048, codes=())
    with pytest.raises(PatchUnusable, match="ceiling"):
        DepthPatch(width=MAX_PATCH_DIMENSION + 1, height=1, codes=())
    with pytest.raises(PatchUnusable, match="corrupted header"):
        DepthPatch.from_frame(b"", frame_width=MAX_PATCH_DIMENSION + 1, frame_height=1)


def test_the_whole_depth_frame_at_the_default_policy_is_inside_every_ceiling():
    """The real boundary: a 640x400 frame in which every sample is a hole."""
    samples = 640 * 400
    assert samples <= MAX_PATCH_SAMPLES
    assert DEFAULT_POLICY.worst_case_work(samples) == samples * 81 == 20_736_000
    assert DEFAULT_POLICY.worst_case_work(samples) <= MAX_WORK_UNITS


def test_a_run_whose_worst_case_is_past_the_ceiling_is_refused_before_it_scans():
    """Refused by arithmetic on the input, not by noticing how long it took."""
    policy = CompletionPolicy(radius=MAX_RADIUS)
    holes = MAX_WORK_UNITS // policy.neighbourhood_samples + 1
    assert policy.worst_case_work(holes) > MAX_WORK_UNITS
    patch = DepthPatch.of(holes // 2, 2, [0] * (2 * (holes // 2)))
    assert len(patch.hole_indices()) * policy.neighbourhood_samples > MAX_WORK_UNITS
    with pytest.raises(WorkTooLarge, match="past the ceiling"):
        complete_patch(patch, policy=policy)


def test_worst_case_work_is_exact_integer_arithmetic_and_refuses_a_boolean():
    assert DEFAULT_POLICY.worst_case_work(0) == 0
    assert DEFAULT_POLICY.worst_case_work(7) == 7 * DEFAULT_POLICY.neighbourhood_samples
    with pytest.raises(PolicyUnusable, match="not a number of holes"):
        DEFAULT_POLICY.worst_case_work(True)
    with pytest.raises(PolicyUnusable, match="not a number of holes"):
        DEFAULT_POLICY.worst_case_work(-1)
