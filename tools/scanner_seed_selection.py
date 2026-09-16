"""M25: change preference only inside the original agreed candidate cluster."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

from scanner3d import tracking

PROFILES = ("observe-model-first", "adjacent-first")
ORIGINAL_PREFERENCE = tracking.PREFERENCE_ORDER
EXPECTED_ORIGINAL = (tracking.MODEL_ORIGIN, tracking.ADJACENT_ORIGIN, tracking.KEYFRAME_ORIGIN)
_ACTIVE: ContextVar[bool] = ContextVar("m25_seed_selection", default=False)


class SeedSelectionExperiment:
    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("M25 requires an explicit closed selection profile")
        self.profile = profile
        self.preference = (
            ORIGINAL_PREFERENCE
            if profile == "observe-model-first"
            else (tracking.ADJACENT_ORIGIN, tracking.MODEL_ORIGIN, tracking.KEYFRAME_ORIGIN)
        )

    @contextmanager
    def active(self) -> Iterator[None]:
        if (
            _ACTIVE.get()
            or ORIGINAL_PREFERENCE != EXPECTED_ORIGINAL
            or tracking.PREFERENCE_ORDER is not ORIGINAL_PREFERENCE
        ):
            raise RuntimeError("M25 requires the original preference and a nonnested scope")
        token = _ACTIVE.set(True)
        try:
            with patch.object(tracking, "PREFERENCE_ORDER", self.preference):
                yield
                if tracking.PREFERENCE_ORDER is not self.preference:
                    raise RuntimeError("M25 preference changed during its declared scope")
        finally:
            _ACTIVE.reset(token)
