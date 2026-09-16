"""Input sequence is not accepted-frame identity; rejected offers and eviction."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from tests.test_scanner_organized_model_normals import PIXEL, deproject


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_organized_frame_binding")


def frame(change=0):
    codes = np.full(256000, 40000, "<u2")
    codes[PIXEL + 1] += change
    raw = codes.tobytes()
    return raw, deproject(raw, (PIXEL,))


def accepted(module):
    ledger, tracker = module.FrameBinding(900), object()
    raw, points = frame()
    ledger.prepare(raw, points)
    ledger.begin(tracker, 0, points)
    ledger.finish(1, {0: points})
    return ledger, tracker, raw, points


@pytest.mark.parametrize("capacity", [900, 1800])
def test_refusal_consumes_offer_but_not_accepted_index(module, capacity):
    ledger, tracker = module.FrameBinding(capacity), object()
    first_raw, points = frame()
    second_raw, same_points = frame(50)
    assert points == same_points and first_raw != second_raw
    first = ledger.prepare(first_raw, points)
    ledger.begin(tracker, 0, points)
    ledger.finish(0, {})
    second = ledger.prepare(second_raw, points)
    ledger.begin(tracker, 0, points)
    ledger.finish(1, {0: points})
    assert (first.ordinal, second.ordinal) == (0, 1)
    assert ledger.accepted_count == 1 and ledger.prepared_count == 2
    assert ledger.retained[0].offer is second
    native = ledger.native(0, 5, points)
    assert native.raw == second_raw and ledger.native(0, 5, points) is native
    assert ledger.native(0, 10, points).radius == 10
    ledger.complete()


def test_equal_points_distinct_accepted_frames_and_eviction(module):
    ledger, tracker, _, points = accepted(module)
    first = ledger.native(0, 5, points)
    raw2, _ = frame(50)
    ledger.prepare(raw2, points)
    ledger.begin(tracker, 1, points)
    ledger.finish(2, {0: points, 1: points})
    second = ledger.native(1, 5, points)
    assert first.points == second.points and first.raw_sha256 != second.raw_sha256
    raw3, _ = frame(100)
    ledger.prepare(raw3, points)
    ledger.begin(tracker, 2, points)
    ledger.finish(3, {0: points, 2: points})
    assert set(ledger.retained) == {0, 2} and set(ledger.fields) == {(0, 5)}
    with pytest.raises(ValueError, match="No retained raw ownership"):
        ledger.native(1, 5, points)
    assert ledger.native(0, 5, points) is first
    ledger.complete()


@pytest.mark.parametrize(
    "action",
    [
        "missing",
        "stale",
        "reentrant",
        "second-tracker",
        "index-drift",
        "wrong-points",
        "unfinished",
        "unconsumed",
    ],
)
def test_pending_and_tracker_identity_guards(module, action):
    ledger, tracker, raw, points = accepted(module)
    with pytest.raises(ValueError):
        if action == "missing":
            ledger.begin(tracker, 1, points)
        else:
            ledger.prepare(raw, points)
            if action == "stale":
                ledger.prepare(raw, points)
            elif action == "second-tracker":
                ledger.begin(object(), 1, points)
            elif action == "index-drift":
                ledger.begin(tracker, 2, points)
            elif action == "wrong-points":
                ledger.begin(tracker, 1, ((0.0, 0.0, 200.0),))
            elif action == "unconsumed":
                ledger.complete()
            else:
                ledger.begin(tracker, 1, points)
                if action == "reentrant":
                    ledger.prepare(raw, points)
                else:
                    ledger.complete()


@pytest.mark.parametrize(
    "count,mapping",
    [
        (3, {0: "same", 1: "same", 2: "same"}),
        (2, {0: "same"}),
        (2, {0: "changed", 1: "same"}),
        (2, {0: "same", 1: "same", 8: "same"}),
    ],
)
def test_finish_failure_is_atomic(module, count, mapping):
    ledger, tracker, raw, points = accepted(module)
    original = ledger.native(0, 5, points)
    ledger.prepare(raw, points)
    ledger.begin(tracker, 1, points)
    actual = {i: points if p == "same" else ((0.0, 0.0, 200.0),) for i, p in mapping.items()}
    with pytest.raises(ValueError):
        ledger.finish(count, actual)
    assert ledger.active is not None and ledger.accepted_count == 1
    assert set(ledger.retained) == {0} and ledger.fields[(0, 5)] is original
    ledger.finish(1, {0: points})
    ledger.complete()


@pytest.mark.parametrize("capacity", [0, 899, 901, 1801, True, 900.0])
def test_undeclared_capacity_refused(module, capacity):
    with pytest.raises(ValueError):
        module.FrameBinding(capacity)


def test_cannot_resurrect_evicted_equal_coordinate_frame(module):
    ledger, tracker, raw, points = accepted(module)
    ledger.prepare(raw, points)
    ledger.begin(tracker, 1, points)
    ledger.finish(2, {1: points})
    ledger.prepare(raw, points)
    ledger.begin(tracker, 2, points)
    with pytest.raises(ValueError, match="resurrected"):
        ledger.finish(3, {0: points, 1: points, 2: points})


def test_mutable_inner_prepared_points_refused(module):
    raw, points = frame()
    with pytest.raises(ValueError, match=r"[Ii]mmutable"):
        module.FrameBinding(900).prepare(raw, (list(points[0]),))
