"""What the value half of frame health will and will not say.

The presence half (:mod:`scanner3d.return_health`) has its own suite. This one
covers the claim that module could not make: that the DISTANCES in a frame
describe a scene.

Everything here is synthetic and exact. The frames are built to contain a
stated shape - one surface, two surfaces with a void between them, a patch that
holds still - because the subject is the decision, not the sensor. What the
detector does on the owner's real captures is recorded in
docs/depth-plausibility.md and is evidence of a different kind: it cannot be a
test, because those files are not in this repository and a test that needed
them would fail everywhere else.
"""

from __future__ import annotations

import json
import math
import sys

import pytest

import scanner3d.depth_plausibility as plausibility_module
from scanner3d.depth_metric import NO_RETURN_SAMPLE
from scanner3d.depth_plausibility import (
    DETACHED_CLUSTER,
    ENVELOPE_NEAR,
    LIMITATIONS,
    NO_RETURNS,
    OUTSIDE_ENVELOPE,
    PLAUSIBLE,
    STATIC_COLUMNS,
    STATIC_HISTORY_FRAMES,
    STATIC_REGION,
    STATIC_ROWS,
    DepthCluster,
    PlausibilityEnvelopeUnusable,
    PlausibilityFrameShapeMismatch,
    PlausibilityScaleUnusable,
    StaticRegionTracker,
    depth_profile,
    locate_cluster,
    plausibility,
)

#: The scale this unit returns: raw code / 200.
SCALE = 1.0 / 200.0

WIDTH, HEIGHT = 64, 40


@pytest.mark.parametrize("shape", [(1, 1), (5, 3), (17, 13), (64, 40), (640, 400)])
@pytest.mark.parametrize("fill", [0, 32000, None])
def test_bulk_diagnostics_match_scalar_with_odd_grids_and_empty_cells(monkeypatch, shape, fill):
    pytest.importorskip("numpy")
    width, height = shape
    samples = tuple(
        (index * 997 % 65536 if index % 3 else 0) if fill is None else fill
        for index in range(width * height)
    )
    expected_grid = plausibility_module._cell_medians(samples, width, height, SCALE)
    expected_profile = depth_profile(samples, depth_scale=SCALE, bins=47)
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert plausibility_module._cell_medians(samples, width, height, SCALE) == expected_grid
    assert depth_profile(samples, depth_scale=SCALE, bins=47) == expected_profile


def _code(distance: float) -> int:
    """The raw code that decodes to ``distance``."""
    return round(distance / SCALE)


def _frame(*bands: tuple[float, int], blanks: int = 0) -> list[int]:
    """A frame holding ``count`` returns at each stated distance, then no-returns."""
    samples: list[int] = []
    for distance, count in bands:
        samples.extend([_code(distance)] * count)
    samples.extend([NO_RETURN_SAMPLE] * blanks)
    return samples


# --- one surface is one cluster ---------------------------------------------


def test_a_single_surface_is_plausible_and_says_where_it_is() -> None:
    """The commonest frame there is, and the one that must never warn."""
    frame = _frame((170.0, 400), (175.0, 400), (180.0, 200), blanks=1000)

    profile = depth_profile(frame, depth_scale=SCALE)
    verdict = plausibility(profile)

    assert len(profile.clusters) == 1
    assert profile.returned == 1000
    assert verdict.verdict == PLAUSIBLE
    assert not verdict.warning
    assert "170" in verdict.detail


def test_a_frame_with_no_returns_at_all_says_so() -> None:
    """Distinct from a frame whose returns are wrong: there is nothing to judge."""
    verdict = plausibility(depth_profile([NO_RETURN_SAMPLE] * 500, depth_scale=SCALE))

    assert verdict.verdict == NO_RETURNS
    assert verdict.warning


def test_a_frame_at_exactly_one_distance_is_one_cluster_and_not_a_crash() -> None:
    """A real state - a flat wall square on - and the one the binning divides by zero on."""
    profile = depth_profile(_frame((150.0, 300)), depth_scale=SCALE)

    assert len(profile.clusters) == 1
    assert profile.clusters[0].extent == 0.0
    assert profile.widest_gap == 0.0
    assert plausibility(profile).verdict == PLAUSIBLE


def test_an_object_in_front_of_its_background_is_still_one_scene() -> None:
    """The gate has to be wide enough not to fire on ordinary depth.

    An object standing clear of the surface behind it is the normal case for
    every capture this appliance takes. Splitting that into "two things" would
    make the warning meaningless on the first real frame.
    """
    frame = _frame((165.0, 500), (168.0, 400), (172.0, 300), (176.0, 200))

    assert plausibility(depth_profile(frame, depth_scale=SCALE)).verdict == PLAUSIBLE


# --- two things in the room -------------------------------------------------


def test_a_detached_near_cluster_is_reported_with_where_and_how_far() -> None:
    """The failure this module was written for, in its smallest form.

    The shape is the one measured on the owner's captures: a small group of
    returns much nearer than the object, with a void between them that no
    single surface would produce.
    """
    # Above the envelope floor on purpose: a cluster mostly BELOW it is the
    # "too close" case, which has its own test and its own sentence.
    frame = _frame((95.0, 60), (98.0, 40), (170.0, 500), (180.0, 400))

    profile = depth_profile(frame, depth_scale=SCALE)
    verdict = plausibility(profile)

    assert len(profile.clusters) == 2
    assert profile.widest_gap > 50.0
    assert verdict.verdict == DETACHED_CLUSTER
    assert verdict.warning
    assert "95" in verdict.detail
    # The intent, not the wording: it has to send the operator to the space
    # between the scanner and the object.
    assert "between the scanner and the target" in verdict.advice


def test_an_object_with_a_distant_background_behind_it_is_not_an_intruder() -> None:
    """The false positive the first version of this module had.

    An object standing well clear of a wall splits into two groups with a wide
    void between them, and it is a completely ordinary scene. The subject is
    whatever most of the returns came from, so the near group being the BULK of
    the frame is what tells the two cases apart.
    """
    profile = depth_profile(_frame((90.0, 900), (280.0, 100)), depth_scale=SCALE)

    assert len(profile.clusters) == 2
    assert profile.detached is None
    assert plausibility(profile).verdict == PLAUSIBLE


def test_two_comparable_groups_are_two_answers_and_say_nothing() -> None:
    """Which one is the subject is not decidable from the split alone."""
    profile = depth_profile(_frame((90.0, 500), (280.0, 500)), depth_scale=SCALE)

    assert len(profile.clusters) == 2
    assert profile.detached is None


def test_a_minority_near_group_behind_a_void_is_the_intruder() -> None:
    """The measured shape: a small near group, the object behind it, a wide gap."""
    profile = depth_profile(_frame((60.0, 60), (250.0, 940)), depth_scale=SCALE)

    assert profile.detached is not None
    assert profile.detached.near == pytest.approx(60.0, abs=0.5)
    assert plausibility(profile).verdict == DETACHED_CLUSTER


def test_a_handful_of_stray_samples_is_not_a_cluster() -> None:
    """Every frame has a few; firing on them would make this noise."""
    frame = _frame((60.0, 2), (170.0, 998))

    profile = depth_profile(frame, depth_scale=SCALE)

    assert profile.detached is None
    assert plausibility(profile).verdict == PLAUSIBLE


# --- outside what this unit was characterised at ----------------------------


def test_returns_outside_the_envelope_are_counted_both_ways_and_never_removed() -> None:
    """Counted rather than clipped, which is the whole difference from preparation.

    :data:`~scanner3d.preparation.DEPTH_CLIP_NEAR` throws these away before a
    cloud is built. This module's job is to say they were there.
    """
    frame = _frame((70.0, 100), (170.0, 800), (295.0, 100))

    profile = depth_profile(frame, depth_scale=SCALE)

    assert profile.below_envelope == 100
    assert profile.above_envelope == 100
    assert profile.returned == 1000
    assert profile.out_of_envelope_fraction == pytest.approx(0.2)


def test_a_frame_inside_the_envelope_reports_nothing_outside_it() -> None:
    profile = depth_profile(
        _frame((ENVELOPE_NEAR + 1, 500), (ENVELOPE_NEAR + 6, 500)), depth_scale=SCALE
    )

    assert profile.out_of_envelope == 0
    assert plausibility(profile).verdict == PLAUSIBLE


def test_enough_returns_outside_the_envelope_earns_its_own_sentence() -> None:
    """Reached when the returns are odd but not split into two groups.

    This is the shape the owner's 2026-08-31 capture had: out-of-range returns
    smeared close enough to the object that no clean void separates them.
    """
    profile = depth_profile(_frame((80.0, 200), (86.0, 800)), depth_scale=SCALE)
    verdict = plausibility(profile)

    assert profile.detached is None
    assert verdict.verdict == OUTSIDE_ENVELOPE
    assert verdict.warning
    assert "nearer" in verdict.detail


def test_a_near_group_that_is_mostly_unmeasurable_says_move_back() -> None:
    """The wrong sentence this module said once, asserted so it cannot come back.

    On 2026-09-01 it detected a near cluster correctly and told the owner
    "something is in the scanner's view much closer than the object". There was
    nothing: the near returns were the target's own ramp, and the scanner was
    simply too close. What tells the two cases apart is how much of the cluster
    was measurable at all.
    """
    # A near group entirely below the envelope, and the subject behind it.
    frame = _frame((40.0, 300), (60.0, 200), (190.0, 500))

    verdict = plausibility(depth_profile(frame, depth_scale=SCALE))

    assert verdict.verdict == DETACHED_CLUSTER
    assert "Move the scanner back" in verdict.advice
    assert "between the scanner and the target" not in verdict.advice


def test_a_near_group_that_was_measurable_still_looks_for_an_intruder() -> None:
    """A foreign object sits in front of a subject that is itself in range."""
    frame = _frame((95.0, 60), (250.0, 940))

    verdict = plausibility(depth_profile(frame, depth_scale=SCALE))

    assert verdict.verdict == DETACHED_CLUSTER
    assert "between the scanner and the target" in verdict.advice
    assert "Move the scanner back" not in verdict.advice


# --- where in the frame it is -----------------------------------------------


def _placed_frame(near_cells: set[tuple[int, int]]) -> list[int]:
    """A frame at 170, with a measurable near group in the named coarse thirds."""
    samples = []
    for row in range(HEIGHT):
        for column in range(WIDTH):
            third = (row * 3 // HEIGHT, column * 3 // WIDTH)
            samples.append(_code(95.0) if third in near_cells else _code(170.0))
    return samples


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ({(0, 0)}, "upper left"),
        ({(0, 2)}, "upper right"),
        ({(1, 1)}, "the middle"),
        ({(2, 0)}, "lower left"),
        ({(2, 2)}, "lower right"),
    ],
)
def test_a_cluster_is_named_by_the_part_of_the_view_it_sits_in(
    cells: set[tuple[int, int]], expected: str
) -> None:
    """The half that turns a thirty-second search into a three-second one."""
    frame = _placed_frame(cells)
    profile = depth_profile(frame, depth_scale=SCALE)
    assert profile.detached is not None

    located = locate_cluster(frame, WIDTH, HEIGHT, profile.detached, depth_scale=SCALE)

    assert located is not None
    assert located.where == expected
    assert plausibility(profile, None, located).detail.endswith(
        "clear of everything else in the frame"
    )
    assert expected in plausibility(profile, None, located).advice


def test_the_words_come_from_the_centroid_not_the_bounding_box() -> None:
    """A box spanning the frame with its mass in one corner is a thing in that corner.

    An operator told "the whole view" would look nowhere, which is the same as
    not being told.
    """
    frame = _placed_frame({(0, 0), (0, 1)})
    profile = depth_profile(frame, depth_scale=SCALE)
    assert profile.detached is not None

    located = locate_cluster(frame, WIDTH, HEIGHT, profile.detached, depth_scale=SCALE)

    assert located is not None
    assert located.left == 0
    assert located.right > WIDTH // 2, "the box spans more than half the frame"
    assert located.where == "upper left"


def test_locating_a_cluster_no_sample_belongs_to_returns_nothing() -> None:
    """Only reachable when the cluster came from a different frame. Stated, not assumed."""
    frame = _frame((170.0, WIDTH * HEIGHT))
    elsewhere = DepthCluster(near=60.0, far=62.0, samples=10, fraction=0.1)

    assert locate_cluster(frame, WIDTH, HEIGHT, elsewhere, depth_scale=SCALE) is None


def test_locating_refuses_a_frame_of_another_shape() -> None:
    profile = depth_profile(_frame((95.0, 60), (170.0, 940)), depth_scale=SCALE)
    assert profile.detached is not None

    with pytest.raises(PlausibilityFrameShapeMismatch):
        locate_cluster([0] * 99, WIDTH, HEIGHT, profile.detached, depth_scale=SCALE)


def test_a_verdict_with_no_location_still_reads_as_a_sentence() -> None:
    """Localising is optional, and the message must not depend on it."""
    profile = depth_profile(_frame((95.0, 60), (170.0, 940)), depth_scale=SCALE)

    verdict = plausibility(profile)

    assert verdict.verdict == DETACHED_CLUSTER
    assert "edges of the view" in verdict.advice


# --- a region that will not move --------------------------------------------


def _grid_frame(depths: list[float]) -> list[int]:
    """One frame whose every coarse cell holds a stated distance."""
    samples = []
    for row in range(HEIGHT):
        for column in range(WIDTH):
            cell = (row * STATIC_ROWS // HEIGHT) * STATIC_COLUMNS + (
                column * STATIC_COLUMNS // WIDTH
            )
            samples.append(_code(depths[cell]))
    return samples


def test_the_static_test_says_nothing_until_it_has_seen_enough_frames() -> None:
    """A verdict from two frames would fire during every capture's first second."""
    tracker = StaticRegionTracker()
    cells = STATIC_COLUMNS * STATIC_ROWS

    for step in range(STATIC_HISTORY_FRAMES - 1):
        region = tracker.observe(
            _grid_frame([150.0 + step * 5.0] * cells), WIDTH, HEIGHT, depth_scale=SCALE
        )
        assert not region.found
        assert region.compared == 0


def test_a_completely_still_scene_reports_no_static_region() -> None:
    """Framing a shot, or a paused turntable: everything is still, and that is fine.

    Reporting an intruder here would fire on the commonest harmless state there
    is, which is why the test requires most of the view to be MOVING first.
    """
    tracker = StaticRegionTracker()
    cells = STATIC_COLUMNS * STATIC_ROWS

    for _ in range(STATIC_HISTORY_FRAMES + 2):
        region = tracker.observe(_grid_frame([150.0] * cells), WIDTH, HEIGHT, depth_scale=SCALE)

    assert not region.found


def test_a_patch_that_holds_still_while_the_view_moves_is_found_and_located() -> None:
    """Fixed relative to the camera on a scene that is supposed to be turning."""
    tracker = StaticRegionTracker()
    cells = STATIC_COLUMNS * STATIC_ROWS
    stuck = {0, 1, 2}

    for step in range(STATIC_HISTORY_FRAMES + 2):
        depths = [65.0 if cell in stuck else 150.0 + step * 4.0 for cell in range(cells)]
        region = tracker.observe(_grid_frame(depths), WIDTH, HEIGHT, depth_scale=SCALE)

    assert region.found
    assert set(region.cells) == {(0, 0), (1, 0), (2, 0)}
    assert region.distance == pytest.approx(65.0, abs=0.5)
    assert region.moving_fraction > 0.9

    verdict = plausibility(depth_profile(_grid_frame([150.0] * cells), depth_scale=SCALE), region)
    assert verdict.verdict == STATIC_REGION
    assert verdict.warning
    assert "columns 0-2" in verdict.detail


def test_a_reset_forgets_everything() -> None:
    """A capture boundary is not evidence about the next capture."""
    tracker = StaticRegionTracker()
    cells = STATIC_COLUMNS * STATIC_ROWS
    for _ in range(STATIC_HISTORY_FRAMES):
        tracker.observe(_grid_frame([150.0] * cells), WIDTH, HEIGHT, depth_scale=SCALE)
    assert tracker.frames == STATIC_HISTORY_FRAMES

    tracker.reset()

    assert tracker.frames == 0


# --- refusals, at the boundary ----------------------------------------------


def test_a_frame_of_another_shape_is_refused_rather_than_reshaped() -> None:
    tracker = StaticRegionTracker()

    with pytest.raises(PlausibilityFrameShapeMismatch):
        tracker.observe([0] * 99, WIDTH, HEIGHT, depth_scale=SCALE)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_a_scale_that_cannot_make_a_distance_is_refused(scale: float) -> None:
    with pytest.raises(PlausibilityScaleUnusable):
        depth_profile(_frame((150.0, 10)), depth_scale=scale)


@pytest.mark.parametrize(
    "envelope", [(200.0, 100.0), (-5.0, 100.0), (100.0, 100.0), (math.nan, 100.0)]
)
def test_an_envelope_that_is_not_an_interval_is_refused(envelope: tuple[float, float]) -> None:
    with pytest.raises(PlausibilityEnvelopeUnusable):
        depth_profile(_frame((150.0, 10)), depth_scale=SCALE, envelope=envelope)


def test_a_histogram_needs_more_than_one_bin() -> None:
    with pytest.raises(PlausibilityEnvelopeUnusable):
        depth_profile(_frame((150.0, 10)), depth_scale=SCALE, bins=1)


# --- it is a value, and it decides nothing ----------------------------------


def test_the_same_frame_gives_the_same_answer_every_time() -> None:
    """Nothing here reads a clock or a random number."""
    frame = _frame((62.0, 80), (170.0, 900))

    first = depth_profile(frame, depth_scale=SCALE).as_dict()
    second = depth_profile(frame, depth_scale=SCALE).as_dict()

    assert first == second


def test_every_record_survives_a_strict_json_round_trip() -> None:
    """A record written with ``allow_nan=False`` refuses anything JSON has no word for."""
    profile = depth_profile(_frame((62.0, 80), (170.0, 900)), depth_scale=SCALE)
    verdict = plausibility(profile)

    for payload in (profile.as_dict(), verdict.as_dict()):
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_the_advice_names_what_to_look_at_and_never_what_it_is() -> None:
    """The bytes do not carry a cause, so no sentence here may claim one.

    Asserted as a word list rather than argued, because "it never names a
    cause" is the kind of promise that decays one helpful edit at a time.
    """
    frames = [
        _frame((62.0, 80), (170.0, 900)),
        _frame((70.0, 300), (170.0, 700)),
        [NO_RETURN_SAMPLE] * 100,
    ]
    forbidden = ("clamp", "cable", "hand", "finger", "projector", "emitter", "lens", "dirt")

    for frame in frames:
        advice = plausibility(depth_profile(frame, depth_scale=SCALE)).advice.lower()
        for word in forbidden:
            assert word not in advice, f"advice names a cause it cannot know: {word}"


def test_the_limitations_are_carried_and_name_the_weakest_number() -> None:
    """The envelope's provenance is thin, and the module says so itself."""
    assert LIMITATIONS
    joined = " ".join(LIMITATIONS).lower()
    assert "advisory" in joined
    assert "nine early" in joined or "nine" in joined
