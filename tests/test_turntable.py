"""scanner3d.turntable: one axis and one rate recover a synthetic sweep.

The object is a box with a dome on it, turned about a known axis by a known
rate, with noise on every point. The registration must find the axis to a
few hundredths of a degree, the centre to a fraction of a millimetre and the
rate to the fractional period, and it must do so whether the frames are
evenly spaced or timed unevenly, as the desktop application's held spacing
leaves them.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from scanner3d.geometry import Point, Rigid, dot, normalize, subtract
from scanner3d.pipeline import SweepReconstruction
from scanner3d.turntable import (
    TurntableNotDetected,
    TurntableRegistration,
    TurntableUnusable,
    axis_angle,
    frame_angles,
    register_turntable,
    rotation_about,
)

AXIS: Point = normalize((0.02, 0.91, 0.41))
CENTRE: Point = (50.0, -90.0, 190.0)
RATE = 4.0  # degrees per frame interval: 90 frames per turn


def _object(rng: random.Random) -> list[Point]:
    """A 40 mm box standing on the turntable with a 15 mm dome on its lid, in mm."""
    points: list[Point] = []
    e1 = normalize((1.0, 0.0, -AXIS[0] / AXIS[2])) if abs(AXIS[2]) > 0.1 else (1.0, 0.0, 0.0)
    e2 = normalize(
        (
            AXIS[1] * e1[2] - AXIS[2] * e1[1],
            AXIS[2] * e1[0] - AXIS[0] * e1[2],
            AXIS[0] * e1[1] - AXIS[1] * e1[0],
        )
    )

    def place(u: float, v: float, h: float) -> Point:
        # u, v across the table, h up the axis; the box stands 30 mm off-centre
        return (
            CENTRE[0] + (u + 30.0) * e1[0] + v * e2[0] + h * AXIS[0],
            CENTRE[1] + (u + 30.0) * e1[1] + v * e2[1] + h * AXIS[1],
            CENTRE[2] + (u + 30.0) * e1[2] + v * e2[2] + h * AXIS[2],
        )

    for _ in range(700):
        face = rng.randrange(5)
        a, b = rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0)
        if face == 0:
            points.append(place(a, b, 40.0))
        elif face == 1:
            points.append(place(20.0, a, b + 20.0))
        elif face == 2:
            points.append(place(-20.0, a, b + 20.0))
        elif face == 3:
            points.append(place(a, 20.0, b + 20.0))
        else:
            points.append(place(a, -20.0, b + 20.0))
    for _ in range(300):
        theta = rng.uniform(0.0, math.pi / 2)
        phi = rng.uniform(0.0, 2 * math.pi)
        r = 15.0
        points.append(
            place(
                r * math.sin(theta) * math.cos(phi),
                r * math.sin(theta) * math.sin(phi),
                40.0 + r * math.cos(theta),
            )
        )
    return points


def _sweep(taus: Sequence[float], *, noise: float = 0.05, seed: int = 1) -> list[list[Point]]:
    """Frames of the object turned by RATE * tau about AXIS through CENTRE, plus noise."""
    rng = random.Random(seed)
    body = _object(rng)
    frames = []
    for tau in taus:
        rotation = rotation_about(AXIS, RATE * tau)
        turned = []
        for p in body:
            d = subtract(p, CENTRE)
            q = (
                CENTRE[0] + rotation[0][0] * d[0] + rotation[0][1] * d[1] + rotation[0][2] * d[2],
                CENTRE[1] + rotation[1][0] * d[0] + rotation[1][1] * d[1] + rotation[1][2] * d[2],
                CENTRE[2] + rotation[2][0] * d[0] + rotation[2][1] * d[1] + rotation[2][2] * d[2],
            )
            turned.append(
                (
                    q[0] + rng.gauss(0.0, noise),
                    q[1] + rng.gauss(0.0, noise),
                    q[2] + rng.gauss(0.0, noise),
                )
            )
        frames.append(turned)
    return frames


def _axis_error(result: TurntableRegistration) -> float:
    return math.degrees(math.acos(min(1.0, abs(dot(result.axis, AXIS)))))


def _centre_error(result: TurntableRegistration) -> float:
    d = subtract(result.centre, CENTRE)
    along = dot(d, AXIS)
    return math.sqrt(max(0.0, dot(d, d) - along * along))


def _landing_error(result: TurntableRegistration, frames: Sequence[Sequence[Point]]) -> float:
    """Median distance between frame 0's points and frame k's, brought back by the poses."""
    errors = []
    origin = list(result.poses[0].apply_all(frames[0]))
    for k in range(10, len(frames), 20):
        back = list(result.poses[k].apply_all(frames[k]))
        for a, b in zip(origin[::40], back[::40], strict=True):
            errors.append(math.dist(a, b))
    errors.sort()
    return errors[len(errors) // 2]


# -- the model ----------------------------------------------------------------


def test_rotation_about_and_axis_angle_round_trip() -> None:
    axis = normalize((0.3, -0.5, 0.8))
    found, angle = axis_angle(rotation_about(axis, 37.5))
    assert angle == pytest.approx(37.5, abs=1e-9)
    assert dot(found, axis) == pytest.approx(1.0, abs=1e-9)


def test_frame_angles_are_the_index_for_even_spacing_and_the_time_otherwise() -> None:
    assert frame_angles(None, 4) == (0.0, 1.0, 2.0, 3.0)
    assert frame_angles([0.0, 0.443, 0.886, 1.329], 4) == (0.0, 1.0, 2.0, 3.0)
    uneven = frame_angles([0.0, 0.5, 1.0, 2.0, 2.5, 3.5], 6)
    assert uneven[0] == 0.0
    assert uneven[3] == pytest.approx(4.0)  # 2.0 s is four median intervals of 0.5 s
    assert uneven != (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


# -- registration -------------------------------------------------------------


def test_an_even_sweep_is_registered_to_the_axis_the_centre_and_the_fractional_period() -> None:
    frames = _sweep([float(k) for k in range(100)])
    result = register_turntable(frames, period=90)
    assert _axis_error(result) < 0.05
    assert _centre_error(result) < 0.3
    assert result.frames_per_turn == pytest.approx(90.0, abs=0.05)
    assert result.times_used is False
    assert result.pairs_used >= 20
    assert _landing_error(result, frames) < 0.15
    assert len(result.poses) == 100
    assert result.poses[0].rotation_angle == pytest.approx(0.0, abs=1e-9)


def test_an_unevenly_timed_sweep_registers_from_its_times() -> None:
    rng = random.Random(7)
    times = [0.0]
    for _ in range(99):
        times.append(times[-1] + rng.choice([0.5, 0.75, 1.0, 1.25, 1.5]))
    spacing = 0.75  # the median interval: the model's unit of tau
    taus = [t / spacing for t in times]
    frames = _sweep(taus)
    result = register_turntable(frames, period=90, times=times)
    assert result.times_used is True
    assert _axis_error(result) < 0.05
    assert _centre_error(result) < 0.3
    assert _landing_error(result, frames) < 0.15


def test_the_other_direction_of_turning_registers_too() -> None:
    frames = _sweep([-float(k) for k in range(100)])
    result = register_turntable(frames, period=90)
    assert _axis_error(result) < 0.05
    assert result.rate_degrees < 0.0 or dot(result.axis, AXIS) < 0.0
    assert _landing_error(result, frames) < 0.15


def test_the_registration_can_be_exported_like_a_sweep() -> None:
    frames = _sweep([float(k) for k in range(100)])
    result = register_turntable(frames, period=90)
    record = result.as_reconstruction()
    assert isinstance(record, SweepReconstruction)
    assert record.frames == 100
    assert len(record.poses) == 100
    assert any("turntable model" in line for line in record.limitations)
    assert "frames per turn" in result.statement()
    as_dict = result.as_dict()
    assert as_dict["registration"] == "turntable"
    assert isinstance(as_dict["agreement"], list)


# -- refusals -----------------------------------------------------------------


def test_frames_that_do_not_repeat_are_not_a_turntable() -> None:
    frames = _sweep([float(k) for k in range(30)])
    with pytest.raises(TurntableNotDetected):
        register_turntable(frames)  # no raws, no period: nothing to detect a turn from
    with pytest.raises(TurntableNotDetected):
        register_turntable(frames, raws=[], depth_scale=0.005)


def test_too_few_frames_are_refused() -> None:
    frames = _sweep([float(k) for k in range(8)])
    with pytest.raises(TurntableUnusable):
        register_turntable(frames, period=90)


def test_progress_lines_name_the_stages() -> None:
    frames = _sweep([float(k) for k in range(100)])
    lines: list[str] = []
    register_turntable(frames, period=90, on_progress=lines.append)
    assert lines[0].startswith("turntable: image period 90 frames")
    assert any("seeded from a chain" in line for line in lines)
    assert any(line.strip().startswith("round 1:") for line in lines)
    assert any("frames 1 apart" in line for line in lines)


def _identity_like(pose: Rigid) -> bool:
    return pose.rotation_angle < 1e-9 and pose.translation_magnitude < 1e-9


def test_frame_zero_is_the_reference_frame() -> None:
    frames = _sweep([float(k) for k in range(100)])
    result = register_turntable(frames, period=90)
    assert _identity_like(result.poses[0])


def test_a_fit_that_runs_out_of_rounds_says_it_did_not_converge(monkeypatch) -> None:
    import scanner3d.turntable as turntable_module

    frames = _sweep([float(k) for k in range(100)])
    settled = register_turntable(frames, period=90)
    assert settled.converged is True
    assert settled.rounds >= 1
    assert settled.as_reconstruction().converged is True
    assert "settled in" in settled.statement()
    assert settled.as_dict()["converged"] is True

    # With no round allowed to count as settled, the budget runs out and the
    # model must say so instead of being exported as a converged fit.
    monkeypatch.setattr(turntable_module, "CONVERGED_MM", -1.0)
    monkeypatch.setattr(turntable_module, "MAX_ROUNDS", 2)
    lines: list[str] = []
    capped = register_turntable(frames, period=90, on_progress=lines.append)
    assert capped.converged is False
    assert capped.rounds == 2
    assert capped.as_reconstruction().converged is False
    assert "NOT converged" in capped.statement()
    assert capped.as_dict()["converged"] is False
    assert any("not converged" in line for line in lines)
