"""The live UI says one current thing, while retaining the detailed evidence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QToolButton, QWidget

from scanner3d import gui
from scanner3d.gui import ScannerWindow, ScanStatus
from scanner3d.viewer import render_frame
from tests.gui_harness import CaptureWindow
from tests.qt_offscreen import qt_app

__all__ = ["qt_app"]


def status(
    source: str,
    title: str,
    *,
    priority: int,
    severity: str = "ok",
) -> ScanStatus:
    return ScanStatus(
        source=source,
        title=title,
        summary=f"{title} summary",
        action=f"{title} action",
        severity=severity,
        priority=priority,
        detail=f"full technical evidence for {title}",
    )


def test_a_depth_warning_is_not_erased_by_an_ordinary_tracking_update() -> None:
    """Independent signals arbitrate by urgency instead of last writer wins."""
    window = CaptureWindow(frames=1)
    window._replace_scan_statuses(status("tracking", "TRACKING OK", priority=20))
    window._show_scan_status(
        status("depth", "DEPTH SIGNAL COLLAPSING", priority=80, severity="warning")
    )

    window._show_scan_status(status("tracking", "NEXT FRAME ALIGNED", priority=20))

    assert window._scan_status_title.text == "DEPTH SIGNAL COLLAPSING"
    assert window._scan_status_summary.text == "DEPTH SIGNAL COLLAPSING summary"
    assert window._scan_status_action.text == "Next: DEPTH SIGNAL COLLAPSING action"

    window._clear_scan_status("depth")
    assert window._scan_status_title.text == "NEXT FRAME ALIGNED"


def test_the_depth_signal_status_is_short_and_the_evidence_is_a_tooltip() -> None:
    window = CaptureWindow(frames=1)
    raw = b"\x00\x00" * (gui.FRAME_WIDTH * gui.FRAME_HEIGHT)
    stats = render_frame(raw).stats

    ScannerWindow._show_return_health(cast(ScannerWindow, window), stats)

    assert window._scan_status_title.text == "NO DEPTH SIGNAL"
    assert window._scan_status_summary.text == "This view returned no usable depth."
    assert window._scan_status_action.text.startswith("Next: Stop moving.")
    assert "256000 samples" in window._scan_status_title.tooltip
    assert "no returns anywhere" not in window._scan_status_summary.text


def test_technical_widgets_are_collapsed_together_and_remain_available(qt_app) -> None:
    del qt_app
    parent = QWidget()
    label = QLabel("raw evidence", parent)
    report = QPlainTextEdit(parent)
    advanced = QAction("advanced capture setting", parent)
    toggle = QToolButton(parent)
    shell = SimpleNamespace(
        _technical_details_visible=False,
        _technical_widgets=[],
        _technical_toggle=toggle,
    )

    ScannerWindow._register_technical_widget(cast(ScannerWindow, shell), label)
    ScannerWindow._register_technical_widget(cast(ScannerWindow, shell), report)
    ScannerWindow._register_technical_widget(cast(ScannerWindow, shell), advanced)

    assert label.isHidden()
    assert report.isHidden()
    assert not advanced.isVisible()

    ScannerWindow._toggle_technical_details(cast(ScannerWindow, shell), True)

    assert not label.isHidden()
    assert not report.isHidden()
    assert advanced.isVisible()
    assert toggle.text() == "Hide technical details"
