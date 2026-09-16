"""One unit's measured metric correction, and the deprojection that undoes it.

Why this exists
---------------
Every point this pipeline makes comes from ``x = (u - cx) * z / fx``,
``y = (v - cy) * z / fy`` and ``z = code * scale``, with the factory record
(:data:`scanner3d.simulate.FACTORY_INTRINSICS`) and the SDK's depth scale.
Measured against a caliper-checked sphere (``docs/cue-ball-scale-20260911.md``),
this unit's output is not quite metric: horizontally and in depth it reads
about 0.15% short, vertically about 0.55% short. Those three factors are what
the record below carries, as *measured over true* along the camera axes, and
:meth:`MetricCorrection.deprojection` turns them into the intrinsics and depth
scale that undo them: ``fx * lx / lz``, ``fy * ly / lz`` and ``scale / lz``.

What this is not: a change to the factory record, a firmware or calibration
write, or a claim about any other unit. It is bound to a serial and applied at
deprojection time by the entry points that choose to, which say so in the
provenance they export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from scanner3d.depth_scale import SDK_MINI_FALLBACK_SCALE
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.simulate import FACTORY_INTRINSICS, intrinsics_of

SCHEMA_VERSION: Final = 1

#: A factor outside this band is not a calibration of this scanner, it is a bug.
FACTOR_BAND: Final = (0.9, 1.1)

#: The depth scale the factory deprojection uses: float32 0.005 mm per code.
#: ``scanner3d.cli.DEPTH_SCALE_METRES`` carries the same value under a name
#: that predates the unit being established.
FACTORY_DEPTH_SCALE: Final = SDK_MINI_FALLBACK_SCALE


class MetricCorrectionUnusable(ValueError):
    """A correction record whose factors cannot be a scanner's."""


@dataclass(frozen=True, slots=True)
class MetricCorrection:
    """Measured-over-true factors along the camera axes for one unit."""

    serial: str
    x_over_true: float
    y_over_true: float
    z_over_true: float
    measured_on: str
    method: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        low, high = FACTOR_BAND
        for name, value in (
            ("x_over_true", self.x_over_true),
            ("y_over_true", self.y_over_true),
            ("z_over_true", self.z_over_true),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise MetricCorrectionUnusable(
                    f"{name}={value!r} is not a plausible scanner factor (needs {low}..{high})"
                )

    @property
    def fx_factor(self) -> float:
        """What ``fx`` is multiplied by: x reads ``lx`` times truth and z ``lz`` times."""
        return self.x_over_true / self.z_over_true

    @property
    def fy_factor(self) -> float:
        return self.y_over_true / self.z_over_true

    @property
    def depth_factor(self) -> float:
        """What the depth scale is multiplied by, so that z reads true."""
        return 1.0 / self.z_over_true

    def deprojection(
        self, intrinsics: IntrinsicsRecord, depth_scale: float
    ) -> tuple[IntrinsicsRecord, float]:
        """The record and scale that undo the measured factors.

        ``x_corrected = (u - cx) * z_corrected / (fx * lx / lz)`` equals
        ``x_measured / lx``, and ``z_corrected = code * scale / lz`` equals
        ``z_measured / lz``: each axis divided by its own factor. The corrected
        record is built through the same parser the factory record came
        through, so it is exactly as constrained as one read off the device.
        """
        values = intrinsics.values
        corrected = intrinsics_of(
            fx=values["fx"] * self.fx_factor,
            fy=values["fy"] * self.fy_factor,
            cx=values["cx"],
            cy=values["cy"],
            width=intrinsics.width,
            height=intrinsics.height,
        )
        return corrected, depth_scale * self.depth_factor

    def statement(self) -> str:
        return (
            f"metric correction for unit {self.serial}, measured {self.measured_on} "
            f"({self.method}): measured over true x {self.x_over_true:.4f}, "
            f"y {self.y_over_true:.4f}, z {self.z_over_true:.4f}; applied as "
            f"fx x {self.fx_factor:.4f}, fy x {self.fy_factor:.4f}, depth scale x "
            f"{self.depth_factor:.4f}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "serial": self.serial,
            "measured_over_true": {
                "x": self.x_over_true,
                "y": self.y_over_true,
                "z": self.z_over_true,
            },
            "applied": {
                "fx_factor": self.fx_factor,
                "fy_factor": self.fy_factor,
                "depth_scale_factor": self.depth_factor,
            },
            "measured_on": self.measured_on,
            "method": self.method,
            "evidence": list(self.evidence),
        }


#: This unit's correction. Sphere fits of a 57.265 mm cue ball at two poses,
#: its orbit on the turntable, and a rendered-sphere model that reproduces the
#: fitted diameter and the horizontal-over-vertical ratio; see the document.
CUE_BALL_20260911: Final = MetricCorrection(
    serial="A22704522A6F00A64",
    x_over_true=0.9985,
    y_over_true=0.9945,
    z_over_true=0.9985,
    measured_on="2026-09-11",
    method=(
        "single-frame sphere fits of a caliper-measured 57.265 mm cue ball, the ball's "
        "orbit on the turntable, and a rendered-sphere model matching both; "
        "docs/cue-ball-scale-20260911.md"
    ),
    evidence=(
        "/home/ben/scans/diagnostics/20260911-cueball-bracket-2930410420/",
        "/home/ben/scans/diagnostics/20260911-cueball-rotated-138616680/",
        "/home/ben/scans/diagnostics/20260911-ballpair-longturn-69468320/",
    ),
)


def calibrated_deprojection(
    correction: MetricCorrection | None = CUE_BALL_20260911,
) -> tuple[IntrinsicsRecord, float, MetricCorrection | None]:
    """The intrinsics and depth scale to deproject this unit's frames with.

    With ``correction`` ``None`` the factory record and scale come back
    untouched, so a caller can state explicitly that it wants uncorrected
    geometry (a control run, a test of the record itself).
    """
    if correction is None:
        return FACTORY_INTRINSICS, FACTORY_DEPTH_SCALE, None
    intrinsics, scale = correction.deprojection(FACTORY_INTRINSICS, FACTORY_DEPTH_SCALE)
    return intrinsics, scale, correction


def calibration_source_statement(
    correction: MetricCorrection | None = CUE_BALL_20260911,
) -> str:
    """The provenance sentence an export carries for the geometry it holds."""
    base = (
        "the factory intrinsics record this repository carries as "
        "scanner3d.simulate.FACTORY_INTRINSICS"
    )
    if correction is None:
        return (
            f"{base}, applied unmodified to every frame of this capture; no calibration was "
            "performed for this scan, and the record's own principal-point observation stands"
        )
    return (
        f"{base}, with the {correction.statement()}, applied to every frame of this capture; "
        "the record's own principal-point observation stands"
    )
