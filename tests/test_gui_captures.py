"""Where the desktop application puts captures, tested without a screen.

None of this constructs a widget. Every method under test is called unbound,
against a small stand-in holding only the handful of attributes it touches, so
the tests exercise the window's real logic on a machine with no display server
and no ``QApplication``. That is deliberate rather than merely convenient: the
question here is which directory the GUI reports and captures into, and a Qt
event loop contributes nothing to answering it while making the tests unable
to run in CI at all.

The stand-in is also the assertion for the refusal test - it deliberately does
NOT carry the attributes ``_start_capture`` reaches for after it has a target,
so a capture that started anyway raises ``AttributeError`` instead of passing.

The methods are reached through the three ``_choose``/``_target``/``_start``
helpers below rather than bound onto the stand-in, so that the cast asserting
"this is window-shaped enough" appears once, in the open, instead of being
spread implicitly across every call.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog

from scanner3d import gui
from scanner3d.config import (
    CAPTURES_DIR_ENV_VAR,
    GUI_CAPTURE_COMMAND,
    CaptureDirUnavailable,
    set_captures_dir,
)
from scanner3d.gui import ScannerWindow
from scanner3d.preparation import DepthBand


class FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, text: str, _timeout: int = 0) -> None:  # Qt's name
        self.messages.append(text)


class FakeAction:
    def __init__(self) -> None:
        self.tooltip = ""

    def setToolTip(self, text: str) -> None:  # Qt's name
        self.tooltip = text


class FakeSpin:
    def __init__(self, value: int) -> None:
        self._value = value

    def value(self) -> int:  # Qt's name
        return self._value


class Window:
    """The parts of :class:`ScannerWindow` the captures methods actually touch."""

    def __init__(self) -> None:
        self.logged: list[str] = []
        self._captures_action = FakeAction()
        self._status = FakeStatusBar()
        # The depth controls, at the sensor's own interval: ``_start_capture``
        # reads them before it allocates a folder, and a band it cannot apply
        # is refused there - which is a different subject from this module's.
        self._depth_near = FakeSpin(60)
        self._depth_far = FakeSpin(300)

    def _log(self, message: str) -> None:
        self.logged.append(message)

    def statusBar(self) -> FakeStatusBar:  # Qt's name
        return self._status

    # The four the window calls on itself, delegated to the real ones. Only
    # these: everything ``_start_capture`` touches once it HAS a target is
    # absent on purpose, so a capture that started anyway fails loudly.
    def _refresh_captures_action(self) -> None:
        ScannerWindow._refresh_captures_action(cast(ScannerWindow, self))

    def _depth_band(self) -> DepthBand:
        return ScannerWindow._depth_band(cast(ScannerWindow, self))

    def _effective_captures_dir(self) -> tuple[Path | None, str | None]:
        return ScannerWindow._effective_captures_dir(cast(ScannerWindow, self))

    def _capture_name(self) -> str | None:
        return ScannerWindow._capture_name(cast(ScannerWindow, self))

    def _capture_target(self) -> Path | None:
        return ScannerWindow._capture_target(cast(ScannerWindow, self))


def _choose(window: Window) -> None:
    ScannerWindow._choose_captures_dir(cast(ScannerWindow, window))


def _target(window: Window) -> Path | None:
    return window._capture_target()


def _start(window: Window) -> None:
    ScannerWindow._start_capture(cast(ScannerWindow, window))


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    """Capture every ``QMessageBox.warning`` instead of showing one."""
    shown: list[str] = []

    class FakeMessageBox:
        @staticmethod
        def warning(_parent: object, _title: str, text: str) -> None:
            shown.append(text)

    monkeypatch.setattr(gui, "QMessageBox", FakeMessageBox)
    return shown


def _answer_dialog_with(monkeypatch, directory: str) -> list[str]:
    """Answer the folder chooser with ``directory``; "" is a cancelled chooser.

    Returns the list every call appends to, so a test can assert the chooser was
    never opened at all - which is the only way to prove a cancelled or refused
    scan name returned before anything else was asked.
    """
    opened: list[str] = []

    def chosen(*_a, **_k) -> str:
        opened.append(directory)
        return directory

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(chosen))
    return opened


def _answer_name_with(monkeypatch, text: str | None) -> list[str]:
    """Answer the scan-name prompt with ``text``. None presses Cancel.

    Cancel returns ``(text, False)`` from Qt, and the text it carries is
    deliberately non-empty here: a window that reads the string and ignores the
    flag would capture into a folder called "ignored" instead of stopping, and
    this is what makes that show up as a failure rather than as a pass.
    """
    asked: list[str] = []

    def get_text(_parent: object, title: str, label: str, *_a, **_k) -> tuple[str, bool]:
        asked.append(f"{title}\n{label}")
        return ("ignored" if text is None else text, text is not None)

    class FakeInputDialog:
        getText = staticmethod(get_text)  # Qt's name

    monkeypatch.setattr(gui, "QInputDialog", FakeInputDialog)
    return asked


def _configured_root(monkeypatch, root: Path) -> Path:
    """Record ``root`` as the captures root, the way the toolbar action does."""
    set_captures_dir(root)
    return root


def test_choosing_a_folder_under_an_override_reports_the_effective_root(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """The reproduced surprise: it used to say captures go where you clicked.

    The environment variable outranks the file the dialog just wrote, so the
    next capture goes somewhere else entirely. Reporting the chosen folder
    would send someone looking for their frames in a directory that will stay
    empty.
    """
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv(CAPTURES_DIR_ENV_VAR, str(elsewhere))
    chosen = tmp_path / "picked"
    _answer_dialog_with(monkeypatch, str(chosen))

    window = Window()
    _choose(window)

    assert warnings == []
    message = "\n".join(window.logged)
    assert str(elsewhere) in message
    assert CAPTURES_DIR_ENV_VAR in message
    assert "Captures will go under" not in message
    # The file was still written, and the toolbar names the source that wins.
    assert str(elsewhere) in window._captures_action.tooltip
    assert CAPTURES_DIR_ENV_VAR in window._captures_action.tooltip
    assert window._status.messages == [f"captures folder: {elsewhere}"]


def test_choosing_a_folder_with_no_override_reports_the_folder_itself(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """The companion case: with nothing outranking it, the file is the answer."""
    chosen = tmp_path / "picked"
    _answer_dialog_with(monkeypatch, str(chosen))

    window = Window()
    _choose(window)

    assert warnings == []
    assert f"Captures will go under {chosen}" in "\n".join(window.logged)


def test_two_captures_in_a_row_get_two_different_directories(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """A shared folder would mix two scans' frames, and neither survives it."""
    chosen = tmp_path / "picked"
    _answer_dialog_with(monkeypatch, str(chosen))
    _choose(Window())
    _answer_name_with(monkeypatch, "")

    first = _target(Window())
    second = _target(Window())

    assert first is not None and second is not None
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.parent == second.parent == chosen / GUI_CAPTURE_COMMAND
    assert warnings == []


def test_a_capture_that_cannot_be_allocated_never_starts(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """Refused before the capture, not demoted to a dialog and captured anyway.

    ``Window`` carries none of the attributes ``_start_capture`` uses once it
    has a target, so this passes only if it returned at the refusal - well
    before the preview is released or a worker is started.
    """
    refusal = "every name under that folder is taken, so nothing was captured"

    def unavailable(_command: str, **_kwargs: object) -> Path:
        raise CaptureDirUnavailable(refusal)

    monkeypatch.setattr(gui, "allocate_capture_dir", unavailable)
    _configured_root(monkeypatch, tmp_path / "scans")
    _answer_name_with(monkeypatch, "chassis")
    # If the refusal were ignored, this dialog would silently supply a target.
    _answer_dialog_with(monkeypatch, str(tmp_path / "somewhere-else"))

    window = Window()
    assert _target(window) is None
    assert warnings == [refusal]

    warnings.clear()
    _start(window)
    assert warnings == [refusal]
    assert window.logged == []


# --- one fresh, optionally named folder per scan ------------------------------


def test_a_typed_name_becomes_the_folder_the_frames_go_into(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """The point of asking: the folder is called what the person called the scan."""
    root = _configured_root(monkeypatch, tmp_path / "scans")
    asked = _answer_name_with(monkeypatch, "chassis left side")
    opened = _answer_dialog_with(monkeypatch, "")

    target = _target(Window())

    assert target == root / GUI_CAPTURE_COMMAND / "chassis left side"
    assert target is not None and target.is_dir()
    assert warnings == []
    # Asked once, and the recorded root meant the folder chooser stayed shut.
    assert len(asked) == 1
    assert opened == []


def test_surrounding_whitespace_is_trimmed_from_the_typed_name(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """A stray keystroke or a paste, not a folder somebody must quote forever."""
    root = _configured_root(monkeypatch, tmp_path / "scans")
    _answer_name_with(monkeypatch, "  turntable pass  ")

    target = _target(Window())

    assert target == root / GUI_CAPTURE_COMMAND / "turntable pass"
    assert warnings == []


def test_a_blank_name_still_gets_a_dated_folder_of_its_own(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """Leaving the box empty is an answer - the one this window used to assume."""
    root = _configured_root(monkeypatch, tmp_path / "scans")
    _answer_name_with(monkeypatch, "   ")

    target = _target(Window())

    assert target is not None and target.is_dir()
    assert target.parent == root / GUI_CAPTURE_COMMAND
    assert target.name != ""
    # The readable local stamp, not a name and not the root.
    assert len(target.name) == len("20260825-174501")
    assert target.name.replace("-", "").isdigit()
    assert warnings == []


def test_the_same_name_twice_gets_two_folders_rather_than_one_ruined_one(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """Frames are written by fixed name, so a reused folder destroys both scans."""
    root = _configured_root(monkeypatch, tmp_path / "scans")
    _answer_name_with(monkeypatch, "chassis")

    first = _target(Window())
    assert first is not None
    (first / "frame_0000.raw").write_bytes(b"the first scan")
    second = _target(Window())

    assert first == root / GUI_CAPTURE_COMMAND / "chassis"
    assert second == root / GUI_CAPTURE_COMMAND / "chassis-2"
    assert second is not None and list(second.iterdir()) == []
    assert (first / "frame_0000.raw").read_bytes() == b"the first scan"
    assert warnings == []


def test_with_no_recorded_folder_the_chosen_one_is_the_root_not_the_frames_folder(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """The reproduced defect: frames used to be written straight into the pick.

    A folder somebody points at is a place to keep scans, not a scan. Writing
    frame_0000.raw into it means the next scan pointed at the same folder
    overwrites this one frame by frame, and it means a folder that already holds
    something quietly gains half a scan's worth of raw files.
    """
    chosen = tmp_path / "picked"
    chosen.mkdir()
    (chosen / "notes.txt").write_text("not mine to write into\n", encoding="utf-8")
    _answer_name_with(monkeypatch, "chassis")
    opened = _answer_dialog_with(monkeypatch, str(chosen))

    target = _target(Window())

    assert target == chosen / GUI_CAPTURE_COMMAND / "chassis"
    assert target is not None and target.is_dir()
    assert target != chosen
    assert sorted(path.name for path in chosen.iterdir()) == sorted(
        [GUI_CAPTURE_COMMAND, "notes.txt"]
    )
    assert opened == [str(chosen)]
    assert warnings == []


def test_a_dated_scan_into_a_chosen_folder_also_lands_one_level_down(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """The blank-name half of the same path, which is the common one."""
    chosen = tmp_path / "picked"
    _answer_name_with(monkeypatch, "")
    _answer_dialog_with(monkeypatch, str(chosen))

    target = _target(Window())

    assert target is not None
    assert target.parent == chosen / GUI_CAPTURE_COMMAND
    assert target.is_dir()
    assert warnings == []


def test_cancelling_the_name_prompt_cancels_the_scan_and_creates_nothing(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """Cancel must cost nothing: no folder, no dialog, no preview interrupted.

    ``Window`` carries none of the attributes ``_start_capture`` touches once it
    has a target, so the ``_start`` half passes only if it returned at the
    cancelled prompt - before the preview is released or a worker is started.
    """
    root = _configured_root(monkeypatch, tmp_path / "scans")
    _answer_name_with(monkeypatch, None)
    opened = _answer_dialog_with(monkeypatch, str(tmp_path / "somewhere-else"))

    window = Window()
    assert _target(window) is None
    _start(window)

    assert warnings == []
    assert window.logged == []
    assert opened == []
    assert not root.exists()


def test_cancelling_the_folder_chooser_cancels_the_scan_and_creates_nothing(
    tmp_path: Path, monkeypatch, warnings
) -> None:
    """The other cancel, on the path where the root has to be asked for."""
    _answer_name_with(monkeypatch, "chassis")
    _answer_dialog_with(monkeypatch, "")

    window = Window()
    assert _target(window) is None
    _start(window)

    assert warnings == []
    assert window.logged == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name",
    [
        ".",
        "..",
        "  ..  ",
        "scans/chassis",
        "/absolute",
        "../escape",
        "with\x00nul",
        # A backslash names a place too, on this host as much as on a Windows
        # one - the captures tree is copied off this machine.
        "parent\\child",
        "C:\\captures\\scan",
        # Unicode Cc controls, including the C1 ones an ASCII comparison misses
        # and the ones str.strip() would erase into a silent dated folder.
        "with\x85next-line",
        "with\x9fcontrol",
        "line\nbreak",
        "\t",
        "\x85",
    ],
)
def test_a_name_that_is_not_one_folder_name_is_refused_before_anything_happens(
    tmp_path: Path, monkeypatch, warnings, name: str
) -> None:
    """Refused with a message, before allocation and before the capture starts.

    Every one of these names, honoured, would put frames somewhere other than
    inside the captures folder's scan namespace - which is the one promise this
    whole flow makes.
    """
    root = _configured_root(monkeypatch, tmp_path / "scans")
    _answer_name_with(monkeypatch, name)
    opened = _answer_dialog_with(monkeypatch, str(tmp_path / "somewhere-else"))

    window = Window()
    assert _target(window) is None
    assert len(warnings) == 1
    assert "nothing was captured" in warnings[0].lower()

    warnings.clear()
    _start(window)
    assert len(warnings) == 1
    assert window.logged == []
    assert opened == []
    assert not root.exists()
