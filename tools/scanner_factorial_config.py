"""Closed F1 factors: existing implementations combined only in offline replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

SAMPLING = ("stride", "rank", "grid")
MODEL_NORMALS = ("pooled", "native")
CONVERGENCE = ("residual", "settling")


@dataclass(frozen=True)
class Factors:
    sampling: str = "stride"
    model_normals: str = "pooled"
    convergence: str = "residual"

    def __post_init__(self) -> None:
        for value, choices in (
            (self.sampling, SAMPLING),
            (self.model_normals, MODEL_NORMALS),
            (self.convergence, CONVERGENCE),
        ):
            if value not in choices:
                raise ValueError(f"Unknown factorial value {value!r}; expected one of {choices}")

    @property
    def identifier(self) -> str:
        return f"{self.sampling}-{self.model_normals}-{self.convergence}"

    @property
    def values(self) -> tuple[str, str, str]:
        return self.sampling, self.model_normals, self.convergence

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def sampling_rule(self) -> str:
        return {
            "stride": "production_compacted_stride",
            "rank": "fixed_pixel_rank_same_count",
            "grid": "fixed_pixel_grid_same_count",
        }[self.sampling]

    @property
    def model_normal_rule(self) -> str:
        return (
            "native_frame_pca_nearest_voxel_contributor"
            if self.model_normals == "native"
            else "pooled_target_pca"
        )

    @property
    def convergence_rule(self) -> str:
        return (
            "residual_and_surface_motion" if self.convergence == "settling" else "residual_change"
        )


CONFIGURATIONS = tuple(Factors(*values) for values in product(SAMPLING, MODEL_NORMALS, CONVERGENCE))
