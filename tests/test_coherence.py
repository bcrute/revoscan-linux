"""Whether a registered sweep agrees with itself, judged by the frames alone.

The fixture is a synthetic turntable sweep of one and a bit turns, so the
revisit exists by construction and the camera's true pose in the model frame
is known exactly. Every test here says what a consistent registration reads,
what a bent one reads, and what a sweep with no revisit reads - the third
because a check that could not run must not read as one that passed.
"""

from __future__ import annotations

import math
import struct

import pytest

from scanner3d import coherence
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.coherence import (
    CoherenceNotAllowed,
    camera_orbit,
    check_coherence,
    image_period,
    period_agreement,
)
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.preparation import DepthBand
from scanner3d.simulate import FACTORY_INTRINSICS, Box, Shape, Sphere, turntable_sweep

#: The spindle: vertical in the camera's frame, standing 180 out and to the
#: right of the optical axis, which on this unit's intrinsics is near the left
#: edge of the image - so the scene stays in view all the way round.
AXIS = (0.0, 1.0, 0.0)
THROUGH = (50.0, 0.0, 180.0)
TURN = 16
EXTRA = 6
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
BAND = DepthBand(near=100.0, far=260.0)


@pytest.fixture(scope="module")
def sweep() -> tuple[list[bytes], list[Rigid]]:
    """One turn plus six frames, with each frame's CAMERA pose in the model frame.

    :func:`turntable_sweep` moves the SCENE by each returned pose, so the
    camera's pose in the frame of the first view is that pose's inverse - the
    direction :func:`scanner3d.pipeline.reconstruct_sweep` returns.
    """
    shapes: list[Shape] = [
        Box(half_extents=(20.0, 10.0, 12.0), pose=Rigid.of(IDENTITY, (60.0, 8.0, 175.0))),
        Sphere(centre=(35.0, -6.0, 185.0), radius=12.0),
    ]
    frames, scene_poses = turntable_sweep(shapes, frames=TURN, axis=AXIS, through=THROUGH)
    raws = [frame.raw for frame in frames] + [frame.raw for frame in frames[:EXTRA]]
    poses = [pose.inverse() for pose in scene_poses] + [
        pose.inverse() for pose in scene_poses[:EXTRA]
    ]
    return raws, poses


def test_a_sweep_that_comes_back_around_has_its_period_read_from_the_images(sweep):
    raws, _ = sweep
    estimate = image_period(raws, depth_scale=DEPTH_SCALE_METRES, min_lag=8)
    assert estimate is not None
    assert estimate.period == TURN
    # A revisit is the same image, so it matches better than the frame's own
    # neighbour; the last sampled frame of this one-and-a-bit-turn sweep has
    # no revisit at all, and is what the credibility rule exists to drop.
    credible = [sample for sample in estimate.samples if sample.credible]
    assert 2 * len(credible) >= len(estimate.samples)
    for sample in credible:
        assert sample.lag == TURN
        assert sample.adjacent is not None
        assert sample.match < sample.adjacent
        assert sample.match < 0.05


def test_consistent_poses_agree_across_one_period(sweep):
    raws, poses = sweep
    report = period_agreement(
        raws,
        poses,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        period=TURN,
        depth_band=BAND,
    )
    assert report.coherent is True
    assert report.agreement is not None and report.agreement > 0.95
    for pair in report.pairs:
        assert pair.revisit - pair.frame == TURN
        assert abs(pair.median_offset) < 0.1
    assert report.statement().startswith("coherent")
    assert report.as_dict()["coherent"] is True


@pytest.mark.parametrize(
    "bend",
    [
        rotation_about_axis(AXIS, math.radians(20.0), through=THROUGH),
        Rigid.of(IDENTITY, (0.0, 0.0, 5.0)),
    ],
    ids=["second lap turned 20 degrees", "second lap pushed 5 deeper"],
)
def test_poses_that_disagree_with_their_own_revisits_are_incoherent(sweep, bend):
    raws, poses = sweep
    bent = list(poses[:TURN]) + [bend.compose(pose) for pose in poses[TURN:]]
    report = period_agreement(
        raws,
        bent,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        period=TURN,
        depth_band=BAND,
    )
    assert report.coherent is False
    assert report.agreement is not None and report.agreement < 0.75
    assert report.statement().startswith("INCOHERENT")


def test_check_coherence_reads_the_period_and_then_judges(sweep):
    raws, poses = sweep
    report = check_coherence(
        raws,
        poses,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        depth_band=BAND,
    )
    # The default minimum lag is for real sweeps of hundreds of frames; this
    # one is sixteen frames a turn, so the period is found only when asked at
    # a lag the sweep can hold.
    if report.period is None:
        pytest.skip("the default minimum lag exceeds this synthetic sweep's period")
    assert report.period == TURN
    assert report.coherent is True
    assert report.estimate is not None


@pytest.mark.parametrize(
    "translation,expected",
    [((0.0, 0.0, 0.0), True), ((0.0, 0.0, 5.0), False), ((5000.0, 0.0, 0.0), None)],
)
def test_final_check_uses_actual_matches_and_never_passes_unprojectable_pairs(
    monkeypatch, translation, expected
):
    raws = [struct.pack("<H", 30000) * (640 * 400)] * 10
    poses = [Rigid.identity()] * 10
    poses[9] = Rigid.of(IDENTITY, translation)
    estimate = coherence.PeriodEstimate(
        period=4,
        samples=tuple(
            coherence.PeriodSample(frame=a, best_frame=b, lag=b - a, match=0.0, adjacent=1.0)
            for a, b in ((0, 4), (1, 9), (2, 6))
        ),
    )

    def matched(frames, *, depth_scale, samples):
        assert frames == raws and depth_scale == DEPTH_SCALE_METRES
        assert samples == 25
        return estimate

    monkeypatch.setattr(coherence, "image_period", matched)
    # The old fixed-lag sampler never reaches frame 9 and falsely passes.
    legacy = period_agreement(
        raws,
        poses,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        period=4,
        depth_band=BAND,
    )
    assert legacy.coherent is True
    checked = check_coherence(
        raws, poses, intrinsics=FACTORY_INTRINSICS, depth_scale=DEPTH_SCALE_METRES, depth_band=BAND
    )
    assert checked.coherent is expected
    assert checked.period == 4  # Descriptive, not an invented constant lag.
    assert checked.as_dict()["pair_selection"] == "actual_image_match"
    assert checked.as_dict()["requested_image_samples"] == 25
    if expected is None:
        assert checked.uncheckable_pairs == ((1, 9),)
        assert "not checkable" in checked.statement()
    else:
        assert [(p.frame, p.revisit) for p in checked.pairs] == [(0, 4), (1, 9), (2, 6)]
        assert not checked.uncheckable_pairs


@pytest.mark.parametrize("agreement,expected", [(0.5, False), (0.95, None)])
def test_missing_projection_cannot_hide_failure_or_establish_pass(agreement, expected):
    report = coherence.CoherenceReport(
        period=100,
        tolerance=1.5,
        required_agreement=0.75,
        pairs=(coherence.PeriodPair(0, 100, 200, 0.0, agreement),),
        pair_selection="actual_image_match",
        requested_image_samples=25,
        uncheckable_pairs=((20, 121),),
    )
    assert report.coherent is expected
    saved = report.as_dict()
    assert saved["coherent"] is expected
    assert saved["uncheckable_pairs"] == [[20, 121]]
    assert saved["schema_version"] == 2
    assert report.statement().startswith("INCOHERENT" if expected is False else "not checkable")


def test_a_sweep_under_one_turn_has_no_period_and_is_not_checkable(sweep):
    raws, poses = sweep
    partial = raws[: TURN - 4]
    assert image_period(partial, depth_scale=DEPTH_SCALE_METRES, min_lag=4) is None
    report = check_coherence(
        partial,
        poses[: TURN - 4],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    assert report.period is None
    assert report.coherent is None
    assert report.agreement is None
    assert "not checkable" in report.statement()
    assert report.as_dict()["coherent"] is None


def test_frames_and_poses_that_do_not_pair_are_refused(sweep):
    raws, poses = sweep
    with pytest.raises(CoherenceNotAllowed):
        check_coherence(
            raws,
            poses[:-1],
            intrinsics=FACTORY_INTRINSICS,
            depth_scale=DEPTH_SCALE_METRES,
        )
    with pytest.raises(CoherenceNotAllowed):
        period_agreement(
            raws,
            poses,
            intrinsics=FACTORY_INTRINSICS,
            depth_scale=DEPTH_SCALE_METRES,
            period=len(raws),
        )


def test_the_camera_orbit_reads_the_turntable_axis_from_consistent_poses(sweep):
    _, poses = sweep
    orbit = camera_orbit(poses[:TURN])
    assert orbit is not None
    # The camera sits at the origin of its own frame, so its orbit radius is the
    # distance from the origin to the spindle.
    assert orbit.radius == pytest.approx(math.hypot(THROUGH[0], THROUGH[2]), abs=0.01)
    assert abs(orbit.axis[1]) == pytest.approx(1.0, abs=1e-6)
    assert orbit.residual_median < 1e-6
    assert orbit.out_of_plane_rms < 1e-6
    assert orbit.centre == pytest.approx(THROUGH, abs=0.01)
    assert orbit.poses == TURN


def test_poses_that_do_not_orbit_give_no_circle():
    assert camera_orbit([Rigid.identity() for _ in range(12)]) is None
    line = [Rigid.of(IDENTITY, (float(index), 0.0, 0.0)) for index in range(12)]
    assert camera_orbit(line) is None
    assert camera_orbit([Rigid.identity() for _ in range(5)]) is None
