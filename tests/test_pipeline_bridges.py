"""Tests for the relocalization-bridge wiring in :mod:`scanner3d.pipeline` and
:mod:`scanner3d.cli`'s ``reconstruct`` command.

What is at stake
-----------------
A capture that was repositioned mid-sweep has one pair of consecutive frames
whose relationship did NOT come from ordinary adjacent alignment - it came
from a global relocalization, decided live, and recorded to disk by
:mod:`scanner3d.capture_sequence`. Fitting that pair with ICP anyway is the
one thing this machinery exists to make impossible: the two frames are not a
small step apart, and ICP handed them returns a confident, small residual for
a transform that is nonsense.

So the two claims this file has to prove, end to end, are:

1. A sweep with NO bridges is fit exactly as it always was - same pairs, same
   order, same numbers - so every capture taken before pausing existed
   reconstructs identically. This is checked by instrumenting
   :func:`scanner3d.pipeline._fit_pair` directly and recording every pair it
   actually received, the same "capture the real data flow" approach
   ``tests/test_pipeline.py``'s own held-out-partition test uses, for the
   same reason: a count-only check cannot distinguish "the right pairs were
   fitted" from "some pairs were fitted and the total happens to match".
2. A sweep WITH bridges skips exactly the bridged pairs, still fits every
   other one, composes the recorded transform in exactly the place a fitted
   one would have gone, and reports the bridge in every place a reader would
   look for it - ``SweepReconstruction.bridges``, ``as_dict()``,
   ``limitations``, and the CLI's own JSON payload.

Runtime
-------
Every pipeline-level test that does not need real geometry (guards,
ordering, skipping, naming, refusals, cancellation) uses tiny, degenerate,
finite point clouds and monkeypatches ``pipeline._fit_pair`` out entirely -
each of those runs in well under a second. Two pipeline-level tests need
real ICP and real revisit detection to prove those still run on a bridged
sweep; they reuse a short cut of ``tests/test_pipeline.py``'s own
non-closing arc recipe (duplicated rather than imported, per this
codebase's stated policy - see that module's own fixtures), which is the
cheapest scene in this repository already proven to exercise real ICP
without finding a spurious revisit. The CLI-level tests are the expensive
part: the CLI reads raw bytes off disk and deprojects them itself, so a
synthetic cloud cannot stand in for a frame there, and the CLI's own
default revisit-detection gap (see ``_MIN_CLI_FRAMES`` below) sets a hard
floor of 9 frames under any capture the CLI is asked to reconstruct, bridge
or no bridge. Four capture fixtures are built once each, at module scope,
and reused across every test that only needs to read them. Measured total
for this file: under 60 seconds.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import scanner3d.pipeline as pipeline_module
import scanner3d.tracking as tracking_module
from scanner3d.capture_sequence import BridgeBoundary, publish_segment
from scanner3d.cli import main
from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, rotation_about_axis
from scanner3d.icp_plane import IcpPlaneCorrespondencesInsufficient
from scanner3d.pipeline import PipelineNotAllowed, reconstruct_sweep
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.relocalize import LocalFit
from scanner3d.simulate import Box, points_of, render

# --- section 1: cheap, degenerate clouds for the monkeypatched tests ----------------

#: A trivial, non-degenerate square - shaped like a cloud, nothing more.
#: ``_fit_pair`` is monkeypatched out in every test that uses this, so
#: nothing here is ever actually aligned; the only real machinery that
#: touches these numbers is ``detect_revisits``' own Stage 1/2 search, and a
#: degenerate candidate simply failing to verify is documented, ordinary
#: behaviour for that module (see its own module docstring on Stage 2) -
#: not an error - so tiny clouds like these are safe to hand it.
_DUMMY_CLOUD: tuple[Point, ...] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)


def _dummy_cloud(index: int) -> tuple[Point, ...]:
    """A distinct cloud per frame index (offset in z), so a recorded pair
    can be checked by VALUE against ``clouds[i]``/``clouds[i - 1]`` rather
    than merely by count.
    """
    return tuple((x, y, z + index) for x, y, z in _DUMMY_CLOUD)


def _dummy_sweep(n: int) -> list[tuple[Point, ...]]:
    return [_dummy_cloud(i) for i in range(n)]


def _settled_fit(**overrides: object) -> LocalFit:
    """An adjacent fit that clears every gate the tracker judges one by.

    A transform alone is not a fit any more. The tracker judges a candidate on
    its residual, convergence, matched and used fractions before it will place
    anything by it, so a stand-in carrying only a transform would report a NaN
    residual and no correspondences to that judgement - which is a refusal, and
    would make every stubbed test below assert the refusal path by accident.

    ``overrides`` is how a test asks for that path ON PURPOSE.
    """
    fit: dict[str, object] = {
        "transform": Rigid.identity(),
        "rmse": 0.2,
        "converged": True,
        "iterations": 9,
        "matched_fraction": 0.9,
        "used_fraction": 0.8,
        "resolved_backend": "python",
    }
    fit.update(overrides)
    return cast(LocalFit, SimpleNamespace(**fit))


def _recording_fit_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[Point, ...], tuple[Point, ...]]]:
    """Monkeypatch ``pipeline._fit_pair`` with a spy and return its call log.

    Records exactly the ``(source, target)`` pair each call received, in
    call order, which is what lets the tests below assert on the actual
    data flow rather than inferring it from the output.
    """
    calls: list[tuple[tuple[Point, ...], tuple[Point, ...]]] = []

    def fake(job: tuple[object, tuple[Point, ...], tuple[Point, ...]]) -> LocalFit:
        _settings, source, target = job
        calls.append((source, target))
        return _settled_fit()

    def no_other_reference(*_args: object, **_kwargs: object) -> object:
        raise IcpPlaneCorrespondencesInsufficient(
            "the keyframe and model references are switched off for this test"
        )

    monkeypatch.setattr(pipeline_module, "_fit_pair", fake)
    # The tracker's OTHER references are switched off, and that is the whole
    # point of these tests rather than a shortcut in them. What is being
    # asserted here is which adjacent pairs reach ``_fit_pair`` and in what
    # order - a data-flow claim - over clouds chosen to be cheap rather than to
    # be geometry. A keyframe or model candidate fitted for real over four
    # coplanar points would be answering a question about that geometry, and it
    # would disagree with the stub above by construction, because the stub says
    # the frames did not move and the clouds say they moved in z.
    monkeypatch.setattr(tracking_module, "icp_point_to_plane", no_other_reference)
    return calls


# --- section 2: the equivalence guarantee with no bridges ----------------------------


def test_no_bridges_fits_every_adjacent_pair_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The byte/behaviour-equivalence guarantee itself: with no bridges this
    must fit exactly the pairs, in exactly the order, it has always fit.
    """
    calls = _recording_fit_pair(monkeypatch)
    clouds = _dummy_sweep(6)

    result = reconstruct_sweep(clouds, workers=1, min_index_gap=1, robust=False)

    assert result.bridges == ()
    assert calls == [(clouds[i], clouds[i - 1]) for i in range(1, len(clouds))]


# --- section 3: a single bridge replaces exactly its pair -----------------------------


def test_a_bridge_replaces_its_pair_and_the_recorded_transform_is_composed_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the bridged position, ``_fit_pair`` must never be called for that
    pair; every OTHER pair must still be fitted; and the resulting pose must
    be exactly the predecessor pose composed with the RECORDED transform -
    not a fitted one, which the recording spy would have made ``identity``
    for every pair including this one, making any drift from ``identity``
    here proof the recorded transform (not a fit) produced the pose.
    """
    calls = _recording_fit_pair(monkeypatch)
    clouds = _dummy_sweep(6)
    transform = rotation_about_axis((0.2, 0.4, 0.9), math.radians(37.0), through=(1.0, 2.0, 3.0))
    bridge = BridgeBoundary(position=3, transform=transform, source="segment-0001", limitations=())

    result = reconstruct_sweep(clouds, workers=1, min_index_gap=1, robust=False, bridges=[bridge])

    assert (clouds[3], clouds[2]) not in calls
    assert calls == [(clouds[i], clouds[i - 1]) for i in range(1, len(clouds)) if i != 3]
    expected = result.poses[2].compose(transform)
    assert result.poses[3].rotation == expected.rotation
    assert result.poses[3].translation == expected.translation
    # And it did not merely happen to equal the predecessor's own pose -
    # the fake fitter would have produced THAT for a fitted pair.
    assert result.poses[3].rotation != result.poses[2].rotation


def test_several_bridges_in_one_sweep_all_take_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """More than one reposition in a sweep: every bridged pair is skipped,
    every other pair still fitted, and each bridge's own transform is
    composed at its own position, not just the first bridge found.
    """
    calls = _recording_fit_pair(monkeypatch)
    clouds = _dummy_sweep(9)
    t1 = rotation_about_axis((1.0, 0.0, 0.0), math.radians(12.0))
    t2 = rotation_about_axis((0.0, 1.0, 0.0), math.radians(51.0), through=(4.0, -1.0, 2.0))
    bridges = [
        BridgeBoundary(position=6, transform=t2, source="segment-0002", limitations=()),
        BridgeBoundary(position=2, transform=t1, source="segment-0001", limitations=()),
    ]

    result = reconstruct_sweep(clouds, workers=1, min_index_gap=1, robust=False, bridges=bridges)

    skipped = {2, 6}
    assert calls == [(clouds[i], clouds[i - 1]) for i in range(1, len(clouds)) if i not in skipped]
    expected_2 = result.poses[1].compose(t1)
    assert result.poses[2].rotation == expected_2.rotation
    assert result.poses[2].translation == expected_2.translation
    expected_6 = result.poses[5].compose(t2)
    assert result.poses[6].rotation == expected_6.rotation
    assert result.poses[6].translation == expected_6.translation
    assert result.bridges == ("segment-0001", "segment-0002")


# --- section 4: what a reader is told - bridges, as_dict, limitations ----------------


def test_no_bridges_means_bridges_field_empty_and_as_dict_omits_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary capture must produce byte-for-byte the record it always
    did: no ``bridges`` entries and no ``relocalization_bridges`` key at
    all, not even an empty one.
    """
    _recording_fit_pair(monkeypatch)
    clouds = _dummy_sweep(5)

    result = reconstruct_sweep(clouds, workers=1, min_index_gap=1, robust=False)

    assert result.bridges == ()
    assert "relocalization_bridges" not in result.as_dict()


def test_bridges_are_named_in_position_order_regardless_of_argument_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SweepReconstruction.bridges`` names segments by POSITION, not by
    the order ``bridges=`` was given in - proven by passing them reversed -
    and ``as_dict()`` must carry the key, in that same order, once any
    bridge exists.
    """
    _recording_fit_pair(monkeypatch)
    clouds = _dummy_sweep(9)
    bridges = [
        BridgeBoundary(
            position=6, transform=Rigid.identity(), source="segment-0002", limitations=()
        ),
        BridgeBoundary(
            position=2, transform=Rigid.identity(), source="segment-0001", limitations=()
        ),
    ]

    result = reconstruct_sweep(clouds, workers=1, min_index_gap=1, robust=False, bridges=bridges)

    assert result.bridges == ("segment-0001", "segment-0002")
    record = json.loads(json.dumps(result.as_dict()))
    assert record["relocalization_bridges"] == ["segment-0001", "segment-0002"]


def test_limitations_name_the_reposition_first_and_carry_each_bridges_own_limitations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reposition sentence leads every other limitation, because it
    changes what everything below it means, and each bridge's own recorded
    limitations (from its segment's bridge record) must survive into the
    sweep's limitations verbatim, not be summarised away.
    """
    _recording_fit_pair(monkeypatch)
    clouds = _dummy_sweep(6)
    bridge = BridgeBoundary(
        position=3,
        transform=Rigid.identity(),
        source="segment-0001",
        limitations=("this segment's own relocalization note",),
    )

    result = reconstruct_sweep(clouds, workers=1, min_index_gap=1, robust=False, bridges=[bridge])

    assert "REPOSITIONED" in result.limitations[0]
    assert "segment-0001" in result.limitations[0]
    assert result.limitations[1] == "this segment's own relocalization note"
    assert any("revisits were found" in item for item in result.limitations[2:])


# --- section 5: refusals - a bridge REPLACES a measurement, so a bad one refuses ------


def test_bridge_at_position_zero_refuses() -> None:
    """Position 0 has no predecessor pair to replace."""
    clouds = _dummy_sweep(5)
    bridge = BridgeBoundary(
        position=0, transform=Rigid.identity(), source="segment-0001", limitations=()
    )
    with pytest.raises(PipelineNotAllowed, match="position 0"):
        reconstruct_sweep(clouds, bridges=[bridge])


def test_bridge_at_or_past_the_end_of_the_sweep_refuses() -> None:
    clouds = _dummy_sweep(5)
    bridge = BridgeBoundary(
        position=5, transform=Rigid.identity(), source="segment-0001", limitations=()
    )
    with pytest.raises(PipelineNotAllowed, match="position 5"):
        reconstruct_sweep(clouds, bridges=[bridge])


def test_a_negative_bridge_position_refuses() -> None:
    clouds = _dummy_sweep(5)
    bridge = BridgeBoundary(
        position=-1, transform=Rigid.identity(), source="segment-0001", limitations=()
    )
    with pytest.raises(PipelineNotAllowed, match="position -1"):
        reconstruct_sweep(clouds, bridges=[bridge])


def test_two_bridges_at_the_same_position_refuses() -> None:
    """A pair has one relationship; two bridges claiming the same position
    is self-contradictory evidence, not a preference to pick between.
    """
    clouds = _dummy_sweep(5)
    bridges = [
        BridgeBoundary(
            position=2, transform=Rigid.identity(), source="segment-0001", limitations=()
        ),
        BridgeBoundary(
            position=2, transform=Rigid.identity(), source="segment-0002", limitations=()
        ),
    ]
    with pytest.raises(PipelineNotAllowed, match="both bridge position 2"):
        reconstruct_sweep(clouds, bridges=bridges)


# --- section 6: cancellation and progress still work with bridges --------------------


def test_on_progress_exception_propagates_and_progress_mentions_bridge_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation is implemented as ``on_progress`` raising, and that must
    still work with bridges registered. Raises on the message that reports
    the bridge count, which proves two things at once: the exception really
    does propagate out of ``reconstruct_sweep``, and that particular message
    really is emitted (a caller reading progress text would have seen how
    many bridges this sweep has before it was cancelled).
    """
    _recording_fit_pair(monkeypatch)
    clouds = _dummy_sweep(6)
    bridge = BridgeBoundary(
        position=3, transform=Rigid.identity(), source="segment-0001", limitations=()
    )
    messages: list[str] = []

    def on_progress(message: str) -> None:
        messages.append(message)
        if "relocalization bridge" in message:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        reconstruct_sweep(
            clouds,
            workers=1,
            min_index_gap=1,
            robust=False,
            bridges=[bridge],
            on_progress=on_progress,
        )

    assert any("1 recorded relocalization bridge(s)" in message for message in messages)


# --- section 7: real geometry - revisit scoring and workers still work bridged -------

#: The same oblique axis, box and voxel fraction ``tests/test_pipeline.py``'s
#: own ``_arc_clouds`` fixture uses - duplicated rather than imported, per
#: this codebase's established policy of duplicating a small fixture rather
#: than coupling two test files together (see that module's own comment on
#: the point). Chosen here for the same reason it was chosen there: it is
#: the cheapest scene in this repository already measured to exercise real
#: ICP and real revisit detection without finding a spurious revisit.
_AXIS: Point = (0.55, 0.6, 0.58)
_THROUGH: Point = (0.0, 0.0, 117.0)
_BOX = Box(half_extents=(8.0, 12.0, 6.0), pose=Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 117.0)))
_VOXEL_FRACTION = 1.0 / 28.0
_MIN_INDEX_GAP = 3
_MAX_CANDIDATES = 8
_ITERATIONS = 60


def _real_frame_cloud(angle_deg: float, *, seed: int) -> tuple[Point, ...]:
    pose = rotation_about_axis(_AXIS, math.radians(angle_deg), through=_THROUGH)
    frame = render([_BOX], pose=pose, jitter=0.65, seed=seed)
    points = tuple(points_of(frame))
    extent = cloud_extent(points)
    return voxel_downsample(points, extent * _VOXEL_FRACTION)


@functools.cache
def _bridged_arc_clouds() -> tuple[tuple[Point, ...], ...]:
    """4 real frames spanning a 45-degree arc that never returns - a
    shorter cut of the same non-closing scene as ``test_pipeline.py``'s own
    ``_arc_clouds``, so the "no revisits" branch this exercises is
    known-good geometry rather than a fixture invented for this file and
    never checked against anything else. Cut to the fewest frames that
    still leave a real predecessor and a real bridge position (need at
    least 3 for a sweep, one of which is the bridge) - the fewer frames,
    the fewer revisit CANDIDATES real ``detect_revisits`` spends full ICP
    verifying, which is what this pair of tests spends most of its time on.
    Cached so the render cost is paid once for both tests that use it
    below.
    """
    return tuple(_real_frame_cloud(15.0 * k, seed=300 + k) for k in range(4))


def _true_step(from_index: int, to_index: int) -> Rigid:
    """The transform carrying frame ``to_index``'s coordinates into
    ``from_index``'s, from the poses the scene was RENDERED with.

    A bridge stands in for a pair that was never fitted, so a test that supplies
    one is asserting what the capture recorded, not what ICP would say. Supplying
    the identity for a pair that is genuinely 15 degrees apart does not test that
    - it builds a model whose accepted poses contradict its own clouds, and the
    tracker refuses such a frame on purpose, because two references disagreeing
    is exactly what a wrong bridge looks like. So the bridge here is the step the
    renderer actually applied, which is known exactly and is not a measurement.
    """
    source = rotation_about_axis(_AXIS, math.radians(15.0 * from_index), through=_THROUGH)
    target = rotation_about_axis(_AXIS, math.radians(15.0 * to_index), through=_THROUGH)
    return source.compose(target.inverse())


def test_revisit_detection_and_no_revisit_limitation_still_run_on_a_bridged_sweep() -> None:
    """A bridge must not shortcut past revisit scoring for the rest of the
    sweep: the other pairs are still fitted with real ICP, real revisit
    detection still runs against the real clouds, and - since this scene
    never closes - the sweep still reaches the same documented refusal to
    invent confidence that an unbridged sweep with no revisits reaches.
    """
    clouds = _bridged_arc_clouds()
    bridge = BridgeBoundary(
        position=2, transform=_true_step(1, 2), source="segment-0001", limitations=()
    )

    result = reconstruct_sweep(
        clouds,
        min_index_gap=_MIN_INDEX_GAP,
        max_candidates=_MAX_CANDIDATES,
        iterations=_ITERATIONS,
        robust=False,
        bridges=[bridge],
        workers=1,
    )

    assert result.revisits_found == 0
    assert result.optimised is False
    assert result.bridges == ("segment-0001",)
    assert any("revisits were found" in item for item in result.limitations)


def test_workers_two_produces_identical_poses_to_workers_one_on_a_bridged_sweep() -> None:
    """Parallelising the FITTED pairs must not change which pairs are
    fitted or how a bridged position is filled in - proven by exact pose
    agreement between ``workers=1`` and ``workers=2`` on the same bridged
    sweep. If spawning subprocesses is unavailable in this sandbox,
    ``reconstruct_sweep`` reports the fallback rather than raising (see its
    own module docstring), and the poses it returns must still match the
    ``workers=1`` run exactly - so this assertion holds either way.
    """
    clouds = _bridged_arc_clouds()
    bridge = BridgeBoundary(
        position=2, transform=_true_step(1, 2), source="segment-0001", limitations=()
    )
    kwargs: dict[str, object] = {
        "min_index_gap": _MIN_INDEX_GAP,
        "max_candidates": _MAX_CANDIDATES,
        "iterations": _ITERATIONS,
        "robust": False,
        "bridges": [bridge],
    }

    serial = reconstruct_sweep(clouds, workers=1, **kwargs)  # type: ignore[arg-type]
    parallel = reconstruct_sweep(clouds, workers=2, **kwargs)  # type: ignore[arg-type]

    assert serial.bridges == parallel.bridges == ("segment-0001",)
    assert len(serial.poses) == len(parallel.poses)
    for serial_pose, parallel_pose in zip(serial.poses, parallel.poses, strict=True):
        assert serial_pose.rotation == parallel_pose.rotation
        assert serial_pose.translation == parallel_pose.translation


# --- section 8: the CLI's own reconstruct command, end to end ------------------------

#: ``reconstruct_sweep`` (called by the CLI with no override) computes its
#: revisit-detection gap as ``max(8, frames // 8)``, and ``detect_revisits``
#: itself refuses outright unless that gap is STRICTLY SMALLER than the
#: frame count - see ``scanner3d.revisit._ensure_parameters``. The CLI
#: exposes no flag to change the gap, so any capture the CLI is asked to
#: reconstruct needs at least 9 frames left after striding, or the command
#: refuses before it ever gets to reporting a bridge. Every fixture below is
#: sized to clear that floor - it is a floor this file measured empirically,
#: not a realism choice.
_MIN_CLI_FRAMES = 9

#: The same axis/box recipe as section 7's real-geometry fixtures, reused
#: here as raw bytes: the CLI reads frames off disk and deprojects them
#: itself, so a synthetic cloud cannot stand in for a frame at this layer.
_CLI_AXIS: Point = (0.55, 0.6, 0.58)
_CLI_THROUGH: Point = (0.0, 0.0, 117.0)
_CLI_BOX = Box(half_extents=(8.0, 12.0, 6.0), pose=Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 117.0)))


def _cli_raw_frame(angle_deg: float, *, seed: int) -> bytes:
    pose = rotation_about_axis(_CLI_AXIS, math.radians(angle_deg), through=_CLI_THROUGH)
    return render([_CLI_BOX], pose=pose, jitter=0.65, seed=seed).raw


#: A structurally valid accepted-evidence object for a synthetic bridge: an
#: object, ``accepted: true``, a ``thresholds`` object and a non-empty
#: ``candidates`` list, which is what capture_sequence.read_capture_sequence
#: now requires of every published boundary's evidence (see that module's
#: ``_evidence_from``). The note is kept so a reader of a written record can
#: still tell this bridge was never a real relocalization.
def _synthetic_evidence(batch_count: int) -> dict[str, object]:
    """Accepted-evidence shaped as :func:`publish_segment` requires, for a
    batch of ``batch_count`` frames: the global decision plus a local chain of
    one anchor and an accepted, measured link into every frame after it. The
    numbers are synthetic - what these tests are about is the pipeline's use
    of the bridge, not the relocalization that produced it - but the SHAPE is
    the real one, because the writer and the reader both check it.
    """
    return {
        "accepted": True,
        "thresholds": {},
        "candidates": [{"note": "synthetic bridge built for tests/test_pipeline_bridges.py"}],
        "note": "synthetic bridge built for tests/test_pipeline_bridges.py",
        "local_chain": {
            "gates": {"note": "synthetic"},
            "anchor_frames": 1,
            "edges": [
                {
                    "frame": frame,
                    "rmse": 0.02,
                    "converged": True,
                    "iterations": 9,
                    "matched_fraction": 0.9,
                    "used_fraction": 0.8,
                    "resolved_backend": "python",
                    "accepted": True,
                    "refusals": [],
                }
                for frame in range(1, max(batch_count, 1))
            ],
        },
    }


@pytest.fixture(scope="module")
def legacy_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """9 root-only frames: a capture exactly as every capture looked before
    pausing existed, sized to clear ``_MIN_CLI_FRAMES``.
    """
    directory = tmp_path_factory.mktemp("legacy-capture")
    for index in range(_MIN_CLI_FRAMES):
        (directory / f"frame_{index:04d}.raw").write_bytes(_cli_raw_frame(3.0 * index, seed=index))
    return directory


@pytest.fixture(scope="module")
def small_segmented_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """7 root frames plus a 3-frame published segment: 10 frames total, the
    smallest capture that both clears ``_MIN_CLI_FRAMES`` and carries one
    bridge (a published batch must hold at least ``MIN_TENTATIVE_FRAMES``
    frames), for the basic "reconstructs and reports it" case.
    """
    directory = tmp_path_factory.mktemp("small-segmented-capture")
    root = 7
    for index in range(root):
        (directory / f"frame_{index:04d}.raw").write_bytes(_cli_raw_frame(3.0 * index, seed=index))
    publish_segment(
        directory,
        frames=[
            (root + offset, _cli_raw_frame(3.0 * (root + offset), seed=root + offset))
            for offset in range(3)
        ],
        predecessor_index=root - 1,
        transform=Rigid.identity(),
        evidence=_synthetic_evidence(3),
        limitations=("synthetic; no real relocalization was performed",),
    )
    return directory


def _segmented_capture(
    tmp_path_factory: pytest.TempPathFactory, name: str, root: int, segment: int
) -> Path:
    directory = tmp_path_factory.mktemp(name)
    for index in range(root):
        (directory / f"frame_{index:04d}.raw").write_bytes(_cli_raw_frame(3.0 * index, seed=index))
    publish_segment(
        directory,
        frames=[
            (root + offset, _cli_raw_frame(3.0 * (root + offset), seed=root + offset))
            for offset in range(segment)
        ],
        predecessor_index=root - 1,
        transform=Rigid.identity(),
        evidence=_synthetic_evidence(segment),
        limitations=("synthetic; no real relocalization was performed",),
    )
    return directory


@pytest.fixture(scope="module")
def stride2_segmented_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4 root frames plus an 11-frame published segment.

    ``CaptureSequence.strided`` keeps each run's own first frame, its own
    tail when a bridge depends on it, and one in ``stride`` of the rest (see
    its own docstring) - so striding this by 2 leaves exactly
    ``_MIN_CLI_FRAMES`` frames, the smallest (root, segment) pair this file
    found, by exhaustive search over small run lengths, that lands exactly
    there at stride 2. Landing exactly at the floor rather than well above
    it matters for runtime: the number of revisit CANDIDATES real
    ``detect_revisits`` verifies with full ICP grows with how far the frame
    count sits above its own gap, so a capture strided down to a few frames
    over the floor was measured taking several times longer here than one
    landing on it.
    """
    return _segmented_capture(tmp_path_factory, "stride2-segmented-capture", root=4, segment=11)


@pytest.fixture(scope="module")
def stride3_segmented_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """5 root frames plus a 16-frame published segment - the smallest
    (root, segment) pair this file found that lands exactly on
    ``_MIN_CLI_FRAMES`` when strided by 3, for the same runtime reason as
    :func:`stride2_segmented_capture`.
    """
    return _segmented_capture(tmp_path_factory, "stride3-segmented-capture", root=5, segment=16)


def _reconstruct_json(
    argv_tail: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, object]]:
    code = main(["reconstruct", *argv_tail, "--backend", "python", "--workers", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    return code, payload


def test_a_legacy_root_only_capture_reconstructs_with_no_bridge_key(
    legacy_capture: Path, capsys: pytest.CaptureFixture[str], readiness_stood_down: None
) -> None:
    """A capture with no segment directory must reconstruct exactly as it
    always did: no ``relocalization_bridges`` key anywhere in the JSON
    payload, not even an empty one.
    """
    code, payload = _reconstruct_json([str(legacy_capture)], capsys)
    command = payload["command"]
    assert isinstance(command, dict)
    assert code == 0
    assert "relocalization_bridges" not in command
    assert payload["frames"] == _MIN_CLI_FRAMES


def test_a_segmented_capture_reconstructs_and_reports_its_bridge(
    small_segmented_capture: Path, capsys: pytest.CaptureFixture[str], readiness_stood_down: None
) -> None:
    """A capture with one published segment must reconstruct successfully
    and name the bridge, its position and its transform in the JSON
    payload's ``command.relocalization_bridges``.
    """
    code, payload = _reconstruct_json([str(small_segmented_capture)], capsys)
    assert code == 0
    command = payload["command"]
    assert isinstance(command, dict)
    (bridge,) = command["relocalization_bridges"]
    assert bridge["segment"] == "segment-0001"
    assert bridge["position"] == 7
    assert bridge["transform"] == Rigid.identity().as_dict()
    assert bridge["limitations"] == ["synthetic; no real relocalization was performed"]


def test_stride_2_on_a_segmented_capture_still_reconstructs_and_reports_the_bridge(
    stride2_segmented_capture: Path, capsys: pytest.CaptureFixture[str], readiness_stood_down: None
) -> None:
    """The boundary must never be strided away: ``CaptureSequence.strided``
    keeps both frames of a bridged pair within each run (see its own
    docstring), and this proves that guarantee end to end through the real
    CLI command at ``--stride 2`` rather than only at the
    ``CaptureSequence`` level.
    """
    code, payload = _reconstruct_json([str(stride2_segmented_capture), "--stride", "2"], capsys)
    assert code == 0
    command = payload["command"]
    assert isinstance(command, dict)
    assert command["stride"] == 2
    (bridge,) = command["relocalization_bridges"]
    assert bridge["segment"] == "segment-0001"


def test_stride_3_on_a_segmented_capture_still_reconstructs_and_reports_the_bridge(
    stride3_segmented_capture: Path, capsys: pytest.CaptureFixture[str], readiness_stood_down: None
) -> None:
    """Same guarantee as the ``--stride 2`` case above, checked again at
    ``--stride 3`` on a capture with a different (root, segment) split -
    proving the boundary survives striding generally, not for one
    coincidental split of run lengths.
    """
    code, payload = _reconstruct_json([str(stride3_segmented_capture), "--stride", "3"], capsys)
    assert code == 0
    command = payload["command"]
    assert isinstance(command, dict)
    assert command["stride"] == 3
    (bridge,) = command["relocalization_bridges"]
    assert bridge["segment"] == "segment-0001"


def test_a_malformed_segment_fails_the_whole_command_rather_than_reconstructing_the_root_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A capture whose segment cannot be trusted must refuse outright and
    name the capture in its error, not silently reconstruct the root frames
    as if the segment did not exist - that would report a healthy-looking
    number for a sweep that is actually missing however much the segment
    held.
    """
    directory = tmp_path / "capture"
    directory.mkdir()
    for index in range(3):
        (directory / f"frame_{index:04d}.raw").write_bytes(b"not a real depth frame")
    publish_segment(
        directory,
        frames=[(3 + offset, b"not a real depth frame either") for offset in range(3)],
        predecessor_index=2,
        transform=Rigid.identity(),
        evidence=_synthetic_evidence(3),
        limitations=(),
    )
    (directory / "segment-0001" / "bridge.json").write_text("{not valid json", encoding="utf-8")

    code = main(["reconstruct", str(directory), "--backend", "python"])

    assert code != 0
    assert str(directory) in capsys.readouterr().err
