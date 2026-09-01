"""What a frame has to prove before :mod:`scanner3d.tsdf` will weigh it.

``src/scanner3d/tsdf_input.py`` is the boundary between "a completion said so"
and "the field counted it". Everything here is about that boundary being
checkable rather than declared, so the tests below fall into three groups:

* the weighting itself - measured is exactly full weight, an accepted temporal
  reconstruction is strictly between zero and that and strictly increasing in
  its own confidence, and an unconfirmed spatial estimate is exactly zero;
* the adapters - what each accepted product result actually turns into, and
  in particular that a spatial-only completion turns into the raw frame it
  came from;
* the corruption controls - a hand-built frame that changes exactly one fact
  about an otherwise valid one, each of which must be refused for its own
  reason rather than merely produce a worse field.

Nothing here touches a scanner, a capture, a display server, a GPU, the network
or the clock, and no number produced here is a claim about the device.
"""

from __future__ import annotations

import itertools
import math
import struct
from typing import Any, Final

import pytest

from scanner3d import depth_temporal, tsdf_input
from scanner3d.depth_completion import (
    STATE_MEASURED,
    STATE_MISSING,
    STATE_SPATIALLY_RECONSTRUCTED,
    CompletedPatch,
    CompletionPolicy,
    DepthPatch,
)
from scanner3d.depth_completion import complete_patch as complete_spatially
from scanner3d.depth_temporal import (
    MAX_RECONSTRUCTION_CONFIDENCE,
    STATE_TEMPORALLY_RECONSTRUCTED,
    CameraModel,
    PoseRecord,
    TemporalFrame,
    TemporalPatch,
    complete_temporally,
)
from scanner3d.geometry import IDENTITY_MATRIX, Rigid
from scanner3d.tsdf_input import (
    ADAPTERS,
    EXCLUDED_WEIGHT,
    FRAME_SAMPLES,
    INDEX_ENTRY_BYTES,
    MAX_FRAME_BYTES,
    MAX_RECONSTRUCTIONS,
    MEASURED_WEIGHT,
    PROVENANCE_MEASURED,
    PROVENANCE_MISSING,
    PROVENANCE_NAMES,
    PROVENANCE_SPATIALLY_EXCLUDED,
    PROVENANCE_TEMPORALLY_RECONSTRUCTED,
    RECONSTRUCTION_ENTRY_BYTES,
    TEMPORAL_WEIGHT_CEILING,
    WEIGHT_ENTRY_BYTES,
    FrameUnusable,
    ProvenanceUnusable,
    SourceUnusable,
    SparseWeights,
    SummaryUnusable,
    WeightedDepthFrame,
    WeightedFrameSummary,
    WeightedInputNotAllowed,
    summarise,
    temporal_fusion_weight,
    weighted_completed_frame,
    weighted_raw_frame,
    weighted_temporal_frame,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_WIDTH

# --- the one small scene every case here is built on -------------------------------------

#: A 4x4 patch of constant code 9000 with one sample missing, placed at this
#: origin in a real 640x400 frame. The same fixture the accepted temporal suite
#: uses, so what the product does with it is already established there and this
#: file only has to check what the weighting does with what the product said.
PATCH_WIDTH: Final = 4
PATCH_CODE: Final = 9000
HOLE: Final = 5  # (column 1, row 1) of a 4x4 patch
ORIGIN: Final = (300, 190)
FRAME_INDEX: Final = (ORIGIN[1] + 1) * FRAME_WIDTH + (ORIGIN[0] + 1)

CAMERA: Final = CameraModel(fx=50.0, fy=50.0, cx=1.5, cy=1.5, depth_scale=1.0)


def _pose(dx: float = 0.0) -> PoseRecord:
    return PoseRecord(
        transform=Rigid.of(IDENTITY_MATRIX, (dx, 0.0, 0.0)),
        accepted=True,
        status="test pose, exact by construction",
        quality=1.0,
    )


def _codes(hole: bool = True) -> list[int]:
    codes = [PATCH_CODE] * 16
    if hole:
        codes[HOLE] = 0
    return codes


def _patch(codes: list[int]) -> DepthPatch:
    return DepthPatch.of(PATCH_WIDTH, PATCH_WIDTH, codes)


def _raw(codes: list[int], origin: tuple[int, int] = ORIGIN) -> bytes:
    """A real 512000-byte frame carrying ``codes`` at ``origin`` and nothing else."""
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    u0, v0 = origin
    for index, code in enumerate(codes):
        row, column = divmod(index, PATCH_WIDTH)
        at = ((v0 + row) * FRAME_WIDTH + (u0 + column)) * 2
        struct.pack_into("<H", buffer, at, code)
    return bytes(buffer)


def _spatial() -> CompletedPatch:
    """The accepted single-frame completion of that patch: it recovers the hole."""
    return complete_spatially(_patch(_codes()), policy=CompletionPolicy(min_support=1))


def _frame(index: int, codes: list[int], pose: PoseRecord) -> TemporalFrame:
    return TemporalFrame(index=index, patch=_patch(codes), u0=ORIGIN[0], v0=ORIGIN[1], pose=pose)


def _sources(code: int) -> list[TemporalFrame]:
    """Two other posed frames that each measured the target's hole."""
    frames = []
    for position, (index, pose) in enumerate(((1, _pose()), (2, _pose(dx=0.1)))):
        codes = [0] * 16
        codes[HOLE] = code
        frames.append(_frame(index, codes, pose))
        assert position < 2
    return frames


def _temporal(code: int = PATCH_CODE, *, spatial: CompletedPatch | None = None) -> TemporalPatch:
    """The accepted pose-compensated completion of the same patch."""
    return complete_temporally(
        _frame(0, _codes(), _pose()),
        _sources(code),
        camera=CAMERA,
        spatial=spatial,
    )


def _unobserved_temporal() -> TemporalPatch:
    """A temporal run that reaches the hole with nothing, beside a spatial estimate.

    The accepted result then leaves the sample ``spatially_reconstructed`` and
    ``not_observed``, which is the case the exclusion rule exists for.
    """
    empty = [0] * 16
    return complete_temporally(
        _frame(0, _codes(), _pose()),
        [_frame(1, empty, _pose()), _frame(2, empty, _pose(dx=0.1))],
        camera=CAMERA,
        spatial=_spatial(),
    )


# =========================================================================================
# 1. The weighting itself
# =========================================================================================


def test_a_measurement_weighs_exactly_one_and_an_excluded_sample_exactly_nothing():
    assert MEASURED_WEIGHT == 1.0
    assert EXCLUDED_WEIGHT == 0.0


def test_a_temporal_reconstruction_weighs_strictly_between_nothing_and_a_measurement():
    for confidence in (0.01, 0.25, 0.5, 0.9, MAX_RECONSTRUCTION_CONFIDENCE):
        weight = temporal_fusion_weight(confidence)
        assert 0.0 < weight < MEASURED_WEIGHT
    # The widest margin is at the top, not the narrowest: even a perfect
    # accepted reconstruction is worth strictly less than half a measurement.
    assert temporal_fusion_weight(MAX_RECONSTRUCTION_CONFIDENCE) < TEMPORAL_WEIGHT_CEILING


def test_lowering_an_accepted_confidence_strictly_lowers_what_it_is_worth():
    values = [0.05, 0.2, 0.4, 0.6, 0.8, MAX_RECONSTRUCTION_CONFIDENCE]
    weights = [temporal_fusion_weight(value) for value in values]
    assert weights == sorted(weights)
    assert all(low < high for low, high in itertools.pairwise(weights))


def test_the_weight_of_one_confidence_is_the_same_float_every_time():
    first = temporal_fusion_weight(0.37)
    assert first == temporal_fusion_weight(0.37)
    assert first == TEMPORAL_WEIGHT_CEILING * 0.37


@pytest.mark.parametrize(
    "confidence",
    [0.0, -0.1, 1.0, 1.5, float("nan"), float("inf"), True, False, None, "0.5"],
)
def test_a_confidence_the_accepted_module_could_not_have_published_earns_no_weight(confidence):
    with pytest.raises(ProvenanceUnusable):
        temporal_fusion_weight(confidence)


def test_the_provenance_vocabulary_is_closed_and_indexed_by_its_own_codes():
    assert len(PROVENANCE_NAMES) == 4
    for code, name in (
        (PROVENANCE_MEASURED, "measured"),
        (PROVENANCE_TEMPORALLY_RECONSTRUCTED, "temporally_reconstructed"),
        (PROVENANCE_SPATIALLY_EXCLUDED, "spatially_excluded"),
        (PROVENANCE_MISSING, "missing"),
    ):
        assert PROVENANCE_NAMES[code] == name


# =========================================================================================
# 2. The adapters
# =========================================================================================


def test_a_raw_frame_becomes_full_weight_returns_and_weightless_holes():
    raw = _raw(_codes())
    frame = weighted_raw_frame(raw)

    assert frame.adapter == "raw"
    assert frame.codes == raw
    assert frame.summary.measured == 15
    assert frame.summary.missing == FRAME_SAMPLES - 15
    assert frame.summary.temporally_reconstructed == 0
    assert frame.summary.spatially_excluded == 0
    assert frame.summary.total_weight == 15.0
    assert frame.weight_at(FRAME_INDEX) == EXCLUDED_WEIGHT
    assert frame.provenance_at(FRAME_INDEX) == "missing"
    corner = ORIGIN[1] * FRAME_WIDTH + ORIGIN[0]
    assert frame.weight_at(corner) == MEASURED_WEIGHT
    assert frame.code_at(corner) == PATCH_CODE


def test_an_accepted_spatial_completion_offers_the_raw_frame_and_nothing_more():
    """The whole point of running the spatial adapter: it adds no geometry.

    The estimate is real, accepted and counted - and the codes offered to
    fusion are byte-identical to the raw frame's, because a hypothesis one
    frame cannot tell from a sensor dropout is not fused at a discount.
    """
    raw = _raw(_codes())
    spatial = _spatial()
    assert spatial.state_codes[HOLE] == STATE_SPATIALLY_RECONSTRUCTED

    frame = weighted_completed_frame(raw, spatial, origin=ORIGIN)

    assert frame.adapter == "spatial_completion"
    assert frame.codes == raw
    assert frame.summary.spatially_excluded == 1
    assert frame.summary.measured == 15
    assert frame.summary.total_weight == 15.0
    assert frame.provenance_at(FRAME_INDEX) == "spatially_excluded"
    assert frame.weight_at(FRAME_INDEX) == EXCLUDED_WEIGHT
    assert frame.code_at(FRAME_INDEX) == 0


def test_an_accepted_temporal_reconstruction_is_offered_at_its_own_confidence():
    raw = _raw(_codes())
    temporal = _temporal()
    assert temporal.state_codes[HOLE] == STATE_TEMPORALLY_RECONSTRUCTED

    frame = weighted_temporal_frame(raw, temporal, origin=ORIGIN)

    assert frame.adapter == "temporal_completion"
    assert frame.provenance_at(FRAME_INDEX) == "temporally_reconstructed"
    assert frame.code_at(FRAME_INDEX) == temporal.codes[HOLE]
    assert frame.weight_at(FRAME_INDEX) == temporal_fusion_weight(temporal.confidence[HOLE])
    assert 0.0 < frame.weight_at(FRAME_INDEX) < MEASURED_WEIGHT
    assert frame.summary.temporally_reconstructed == 1
    assert frame.summary.total_weight == math.fsum((15.0, frame.weight_at(FRAME_INDEX)))
    # Every measured sample is still the raw byte it was.
    assert frame.codes[: 2 * FRAME_INDEX] == raw[: 2 * FRAME_INDEX]
    assert frame.codes[2 * FRAME_INDEX + 2 :] == raw[2 * FRAME_INDEX + 2 :]


def test_a_spatial_estimate_no_independent_view_observed_is_still_excluded():
    """The confirmation rule, at the fusion boundary rather than in the label."""
    raw = _raw(_codes())
    temporal = _unobserved_temporal()
    assert temporal.confirmation_at(HOLE) == "not_observed"
    assert temporal.state_codes[HOLE] == STATE_SPATIALLY_RECONSTRUCTED

    frame = weighted_temporal_frame(raw, temporal, origin=ORIGIN)

    assert frame.provenance_at(FRAME_INDEX) == "spatially_excluded"
    assert frame.weight_at(FRAME_INDEX) == EXCLUDED_WEIGHT
    assert frame.codes == raw


def test_a_contradicted_estimate_is_weighted_as_the_independent_measurement():
    """Contradiction republishes the independent value, so that is what is weighed."""
    raw = _raw(_codes())
    spatial = _spatial()
    temporal = _temporal(code=200, spatial=spatial)
    assert temporal.confirmation_at(HOLE) == "contradicted"

    frame = weighted_temporal_frame(raw, temporal, origin=ORIGIN)

    assert frame.code_at(FRAME_INDEX) == 200
    assert frame.code_at(FRAME_INDEX) != spatial.codes[HOLE]
    assert frame.weight_at(FRAME_INDEX) == temporal_fusion_weight(temporal.confidence[HOLE])


def test_a_frame_with_nothing_missing_is_all_measurement_at_full_weight():
    raw = _raw(_codes(hole=False))
    frame = weighted_raw_frame(raw)
    assert frame.summary.measured == 16
    assert frame.summary.total_weight == 16.0
    assert frame.summary.weighted_samples == 16
    assert frame.summary.sample_count == FRAME_SAMPLES


def test_every_adapter_names_itself_from_the_closed_vocabulary():
    raw = _raw(_codes())
    built = (
        weighted_raw_frame(raw),
        weighted_completed_frame(raw, _spatial(), origin=ORIGIN),
        weighted_temporal_frame(raw, _temporal(), origin=ORIGIN),
    )
    assert sorted(frame.adapter for frame in built) == sorted(ADAPTERS)


def test_an_adapter_offered_the_other_modules_result_type_is_refused():
    raw = _raw(_codes())
    with pytest.raises(SourceUnusable, match="CompletedPatch"):
        weighted_completed_frame(raw, _temporal(), origin=ORIGIN)  # type: ignore[arg-type]
    with pytest.raises(SourceUnusable, match="TemporalPatch"):
        weighted_temporal_frame(raw, _spatial(), origin=ORIGIN)  # type: ignore[arg-type]


def test_a_result_derived_from_other_bytes_cannot_be_attached_to_these():
    """The raw/source binding: a result about another frame says nothing here."""
    other = list(_codes())
    other[0] = 12345
    with pytest.raises(SourceUnusable, match="derived from code"):
        weighted_completed_frame(_raw(other), _spatial(), origin=ORIGIN)


def test_a_rectangle_placed_outside_the_frame_is_refused():
    with pytest.raises(SourceUnusable, match="leaves the"):
        weighted_completed_frame(_raw(_codes()), _spatial(), origin=(639, 399))


@pytest.mark.parametrize("length", [0, EXPECTED_FRAME_BYTES - 2, EXPECTED_FRAME_BYTES + 2])
def test_a_frame_that_is_not_one_whole_depth_frame_is_refused(length):
    with pytest.raises(FrameUnusable):
        weighted_raw_frame(bytes(length))


def test_the_adapter_copies_a_mutable_buffer_rather_than_aliasing_it():
    codes = _codes()
    buffer = bytearray(_raw(codes))
    frame = weighted_raw_frame(buffer)
    before = frame.codes
    struct.pack_into("<H", buffer, 0, 4242)
    assert frame.raw is not buffer
    assert frame.codes == before
    assert frame.code_at(0) == 0


def test_the_weight_mapping_a_caller_still_holds_cannot_change_the_frame():
    raw = _raw(_codes())
    built = weighted_temporal_frame(raw, _temporal(), origin=ORIGIN)
    weights = dict(built.weights)
    weights[FRAME_INDEX] = 0.99
    assert built.weight_at(FRAME_INDEX) != 0.99
    with pytest.raises(TypeError):
        built.weights[FRAME_INDEX] = 0.99  # type: ignore[index]


# =========================================================================================
# 3. The corruption controls
# =========================================================================================


def _parts(frame: WeightedDepthFrame) -> dict[str, Any]:
    """One valid weighted frame, opened up so a control can change one thing."""
    return {
        "raw": frame.raw,
        "codes": frame.codes,
        "provenance": frame.provenance,
        "weights": dict(frame.weights),
        "summary": frame.summary,
        "adapter": frame.adapter,
        "completion": frame.completion,
        "origin": frame.origin,
    }


def _temporal_frame() -> WeightedDepthFrame:
    return weighted_temporal_frame(_raw(_codes()), _temporal(), origin=ORIGIN)


def _spatial_frame() -> WeightedDepthFrame:
    return weighted_completed_frame(_raw(_codes()), _spatial(), origin=ORIGIN)


def test_promoting_a_reconstruction_to_a_measurement_is_refused():
    parts = _parts(_temporal_frame())
    provenance = bytearray(parts["provenance"])
    provenance[FRAME_INDEX] = PROVENANCE_MEASURED
    parts["provenance"] = bytes(provenance)
    parts["weights"] = {}
    parts["summary"] = summarise(bytes(provenance), {})
    with pytest.raises(ProvenanceUnusable, match="provenance"):
        WeightedDepthFrame(**parts)


def test_demoting_a_measurement_or_changing_its_code_is_refused():
    parts = _parts(_temporal_frame())
    corner = ORIGIN[1] * FRAME_WIDTH + ORIGIN[0]
    codes = bytearray(parts["codes"])
    struct.pack_into("<H", codes, 2 * corner, PATCH_CODE + 1)
    parts["codes"] = bytes(codes)
    with pytest.raises(ProvenanceUnusable, match="republished byte for byte"):
        WeightedDepthFrame(**parts)


def test_giving_an_unconfirmed_spatial_estimate_any_weight_at_all_is_refused():
    parts = _parts(_spatial_frame())
    parts["weights"] = {FRAME_INDEX: 0.25}
    parts["summary"] = summarise(parts["provenance"], parts["weights"])
    with pytest.raises(ProvenanceUnusable, match="weights"):
        WeightedDepthFrame(**parts)


def test_giving_a_missing_sample_any_weight_at_all_is_refused():
    parts = _parts(weighted_raw_frame(_raw(_codes())))
    parts["weights"] = {FRAME_INDEX: 0.25}
    parts["summary"] = summarise(parts["provenance"], parts["weights"])
    with pytest.raises(ProvenanceUnusable, match="weights"):
        WeightedDepthFrame(**parts)


def test_zeroing_an_accepted_temporal_weight_is_refused_rather_than_silently_dropped():
    """A weight of zero means absence, so it may not stand where evidence exists.

    Dropping a reconstruction's weight looks conservative and is not: the
    published counts still say a reconstruction was offered, and a reader
    comparing contribution against those counts would be reading a frame that
    is not the one it describes.
    """
    parts = _parts(_temporal_frame())
    parts["weights"] = {}
    with pytest.raises(ProvenanceUnusable, match="weights"):
        WeightedDepthFrame(**parts)


def test_substituting_a_reconstructed_code_the_accepted_result_did_not_give_is_refused():
    parts = _parts(_temporal_frame())
    codes = bytearray(parts["codes"])
    struct.pack_into("<H", codes, 2 * FRAME_INDEX, 4321)
    parts["codes"] = bytes(codes)
    with pytest.raises(ProvenanceUnusable, match="accepted result's code"):
        WeightedDepthFrame(**parts)


def test_substituting_a_confidence_the_accepted_result_did_not_earn_is_refused():
    frame = _temporal_frame()
    parts = _parts(frame)
    parts["weights"] = {FRAME_INDEX: temporal_fusion_weight(MAX_RECONSTRUCTION_CONFIDENCE)}
    parts["summary"] = summarise(parts["provenance"], parts["weights"])
    with pytest.raises(ProvenanceUnusable, match="times the accepted confidence"):
        WeightedDepthFrame(**parts)


def test_a_frame_whose_accepted_result_is_removed_cannot_keep_its_reconstruction():
    parts = _parts(_temporal_frame())
    parts["completion"] = None
    with pytest.raises(SourceUnusable, match="adapter"):
        WeightedDepthFrame(**parts)


def test_a_frame_that_names_the_wrong_adapter_for_its_result_is_refused():
    parts = _parts(_temporal_frame())
    parts["adapter"] = "spatial_completion"
    with pytest.raises(SourceUnusable, match="adapter"):
        WeightedDepthFrame(**parts)


def test_a_frame_that_names_an_adapter_nobody_implemented_is_refused():
    parts = _parts(_temporal_frame())
    parts["adapter"] = "gpu_completion"
    with pytest.raises(FrameUnusable, match="names adapter"):
        WeightedDepthFrame(**parts)


def test_a_frame_that_names_an_unwritten_schema_version_is_refused():
    parts = _parts(_temporal_frame())
    with pytest.raises(FrameUnusable, match="schema version"):
        WeightedDepthFrame(**parts, schema_version=2)


def test_a_forged_summary_is_refused_against_the_provenance_it_describes():
    frame = _temporal_frame()
    parts = _parts(frame)
    parts["summary"] = WeightedFrameSummary(
        measured=frame.summary.measured + 1,
        temporally_reconstructed=frame.summary.temporally_reconstructed,
        spatially_excluded=frame.summary.spatially_excluded,
        missing=frame.summary.missing - 1,
        total_weight=frame.summary.total_weight + 1.0,
    )
    with pytest.raises(SummaryUnusable, match="the frame counted"):
        WeightedDepthFrame(**parts)


def test_a_summary_that_is_not_the_published_type_is_refused():
    parts = _parts(_temporal_frame())
    parts["summary"] = {"measured": 15}
    with pytest.raises(SummaryUnusable, match="not a WeightedFrameSummary"):
        WeightedDepthFrame(**parts)


def test_an_altered_raw_frame_no_longer_matches_the_result_it_carries():
    parts = _parts(_temporal_frame())
    raw = bytearray(parts["raw"])
    corner = ORIGIN[1] * FRAME_WIDTH + ORIGIN[0]
    struct.pack_into("<H", raw, 2 * corner, PATCH_CODE + 7)
    parts["raw"] = bytes(raw)
    with pytest.raises(SourceUnusable, match="derived from code"):
        WeightedDepthFrame(**parts)


@pytest.mark.parametrize("weight", [0.0, -0.5, 1.0, 1.5, float("nan"), float("inf"), True])
def test_a_weight_that_is_not_strictly_between_absence_and_a_measurement_is_refused(weight):
    parts = _parts(_temporal_frame())
    parts["weights"] = {FRAME_INDEX: weight}
    with pytest.raises(WeightedInputNotAllowed):
        WeightedDepthFrame(**parts)


@pytest.mark.parametrize("index", [-1, FRAME_SAMPLES, True, "5"])
def test_a_weight_for_something_that_is_not_a_sample_of_this_frame_is_refused(index):
    parts = _parts(_temporal_frame())
    parts["weights"] = {index: 0.25}
    with pytest.raises(FrameUnusable):
        WeightedDepthFrame(**parts)


def test_a_provenance_byte_nobody_defined_is_refused():
    parts = _parts(_temporal_frame())
    provenance = bytearray(parts["provenance"])
    provenance[0] = 9
    parts["provenance"] = bytes(provenance)
    with pytest.raises(WeightedInputNotAllowed):
        WeightedDepthFrame(**parts)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw", bytes(10)),
        ("codes", bytes(10)),
        ("provenance", bytes(10)),
        ("raw", "not-bytes"),
        ("weights", [1.0]),
        ("origin", (0, 0, 0)),
        ("origin", (True, 0)),
    ],
)
def test_a_malformed_field_is_refused_before_the_frame_exists(field, value):
    parts = _parts(_temporal_frame())
    parts[field] = value
    with pytest.raises(WeightedInputNotAllowed):
        WeightedDepthFrame(**parts)


def test_a_frame_with_no_completion_result_may_not_place_a_rectangle():
    parts = _parts(weighted_raw_frame(_raw(_codes())))
    parts["origin"] = ORIGIN
    with pytest.raises(SourceUnusable, match="no rectangle to place"):
        WeightedDepthFrame(**parts)


def test_a_completion_result_that_is_not_an_accepted_type_is_refused():
    parts = _parts(_temporal_frame())
    parts["completion"] = object()
    with pytest.raises(SourceUnusable):
        WeightedDepthFrame(**parts)


# =========================================================================================
# 4. The published account
# =========================================================================================


def test_the_summary_partitions_the_whole_frame_and_says_so():
    frame = weighted_temporal_frame(_raw(_codes()), _temporal(), origin=ORIGIN)
    record: dict[str, Any] = frame.summary.as_dict()
    assert record["sample_count"] == FRAME_SAMPLES
    assert (
        record["measured"]
        + record["temporally_reconstructed"]
        + record["spatially_excluded"]
        + record["missing"]
        == FRAME_SAMPLES
    )
    assert record["weighted_samples"] == 16
    assert "correctly rounded" in str(record["total_weight_is"])


def test_the_published_account_states_the_weighting_and_claims_no_accuracy():
    frame = weighted_temporal_frame(_raw(_codes()), _temporal(), origin=ORIGIN)
    record: dict[str, Any] = frame.as_dict()
    assert record["adapter"] == "temporal_completion"
    weighting = record["weighting"]
    assert isinstance(weighting, dict)
    assert weighting["measured_weight"] == MEASURED_WEIGHT
    assert weighting["excluded_weight"] == EXCLUDED_WEIGHT
    assert "strictly below measured_weight" in str(weighting["temporal_weight_is"])


def test_the_states_this_module_reads_are_the_accepted_modules_own_constants():
    """No second copy of the provenance vocabulary. These are imports, not literals."""
    states = {
        STATE_MISSING,
        STATE_MEASURED,
        STATE_SPATIALLY_RECONSTRUCTED,
        STATE_TEMPORALLY_RECONSTRUCTED,
    }
    assert len(states) == 4
    assert states == set(range(len(depth_temporal.STATE_NAMES)))


def test_the_published_storage_is_every_byte_the_frame_owns_and_not_only_the_fixed_part():
    """``owned_bytes`` is read off the buffers, including the reconstructions.

    The round-two report advertised the fixed part alone while the weights sat
    in a Python dict beside it, so the published number was not the frame's
    cost. It is now the sum of the buffers themselves.
    """
    frame = weighted_temporal_frame(_raw(_codes()), _temporal(), origin=ORIGIN)
    reconstructions = len(frame.weights)
    assert reconstructions > 0
    storage = frame.as_dict()["storage"]
    assert isinstance(storage, dict)
    assert tsdf_input.FIXED_FRAME_BYTES == 2 * EXPECTED_FRAME_BYTES + FRAME_SAMPLES
    assert RECONSTRUCTION_ENTRY_BYTES == INDEX_ENTRY_BYTES + WEIGHT_ENTRY_BYTES
    expected = (
        len(frame.raw)
        + len(frame.codes)
        + len(frame.provenance)
        + reconstructions * RECONSTRUCTION_ENTRY_BYTES
    )
    assert frame.owned_bytes == expected
    assert storage["owned_bytes"] == expected
    assert storage["fixed_bytes"] == tsdf_input.FIXED_FRAME_BYTES
    assert storage["reconstruction_bytes"] == reconstructions * RECONSTRUCTION_ENTRY_BYTES
    assert storage["max_owned_bytes"] == MAX_FRAME_BYTES
    # A frame carrying reconstructions costs strictly more than the fixed part,
    # which is exactly the fact the fixed-only number hid.
    assert frame.owned_bytes > tsdf_input.FIXED_FRAME_BYTES


def test_the_published_bound_holds_at_the_worst_frame_this_contract_allows():
    """Demonstrated at the maximum population, not argued for at a typical one.

    The bound is a byte count of two ``bytes`` objects, so it is the same on
    every interpreter. :func:`sys.getsizeof` is deliberately not consulted: it
    measures one implementation's container overhead and is not a portable
    contract.
    """
    weights = SparseWeights({index: 0.5 for index in range(MAX_RECONSTRUCTIONS)})
    assert len(weights) == MAX_RECONSTRUCTIONS == FRAME_SAMPLES
    assert weights.owned_bytes == MAX_RECONSTRUCTIONS * RECONSTRUCTION_ENTRY_BYTES
    assert tsdf_input.FIXED_FRAME_BYTES + weights.owned_bytes == MAX_FRAME_BYTES
    # And the worst frame is the fixed part plus that, which is the published
    # ceiling every buildable frame is at most.
    assert weighted_raw_frame(_raw(_codes())).owned_bytes <= MAX_FRAME_BYTES
    assert (
        weighted_temporal_frame(_raw(_codes()), _temporal(), origin=ORIGIN).owned_bytes
        <= MAX_FRAME_BYTES
    )


def test_the_compact_weight_encoding_reads_back_the_exact_accepted_floats():
    """Compactness is not paid for in precision: ``float64`` in, the same out."""
    exact = {
        3: temporal_fusion_weight(0.9),
        1: temporal_fusion_weight(0.123456789012345),
        FRAME_SAMPLES - 1: temporal_fusion_weight(depth_temporal.MAX_RECONSTRUCTION_CONFIDENCE),
    }
    weights = SparseWeights(exact)
    assert dict(weights) == exact
    for index, weight in exact.items():
        assert weights[index] == weight
        assert index in weights
    # Ascending sample order, whatever order the caller offered them in.
    assert list(weights) == sorted(exact)
    assert 0 not in weights
    with pytest.raises(KeyError):
        weights[0]


def test_the_compact_weight_encoding_cannot_be_written_to_after_it_is_checked():
    weights = SparseWeights({7: 0.25})
    with pytest.raises(TypeError):
        weights[7] = 0.5  # type: ignore[index]
    with pytest.raises(AttributeError):
        weights._weights = b""
    with pytest.raises(AttributeError):
        del weights._indices
    assert weights[7] == 0.25
