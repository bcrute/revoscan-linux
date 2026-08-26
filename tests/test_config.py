"""Where captures go, who says so, and what a write is allowed to disturb.

Every test here drives the real file on a temporary XDG root (see the autouse
``isolated_config`` fixture in ``conftest``), because the thing worth proving
about a setting that persists is what actually lands on disk.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from scanner3d.config import (
    CAPTURES_DIR_ENV_VAR,
    EXAMPLE_SERIAL,
    SERIAL_ENV_VAR,
    ConfigFileUnreadable,
    ConfigWriteRefused,
    captures_dir_for,
    captures_dir_source,
    clear_captures_dir,
    config_path,
    configured_captures_dir,
    configured_serial,
    dated_capture_name,
    serial_source,
    set_captures_dir,
    stated_serial,
)


def _write_config(text: str) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- reading -----------------------------------------------------------------


def test_nothing_configured_reads_as_not_stated_rather_than_an_error():
    """An absent file is an ordinary condition: no captures directory is set."""
    assert configured_captures_dir() is None
    assert captures_dir_source() is None


def test_captures_directory_is_read_from_the_config_file():
    path = _write_config('[captures]\ndirectory = "/data/scans"\n')
    assert configured_captures_dir() == Path("/data/scans")
    assert captures_dir_source() == str(path)


def test_environment_wins_over_the_file_and_says_so(monkeypatch):
    """One shell pointed elsewhere must not need the file edited and edited back."""
    _write_config('[captures]\ndirectory = "/data/scans"\n')
    monkeypatch.setenv(CAPTURES_DIR_ENV_VAR, "/mnt/scratch")
    assert configured_captures_dir() == Path("/mnt/scratch")
    assert captures_dir_source() == CAPTURES_DIR_ENV_VAR


def test_a_recorded_relative_path_is_read_back_absolute(monkeypatch, tmp_path):
    """A default that means a different directory per terminal is not a default."""
    monkeypatch.chdir(tmp_path)
    _write_config('[captures]\ndirectory = "scans"\n')
    assert configured_captures_dir() == tmp_path / "scans"


def test_a_tilde_in_the_file_is_expanded():
    _write_config('[captures]\ndirectory = "~/scans"\n')
    assert configured_captures_dir() == Path.home() / "scans"


def test_an_empty_value_is_the_same_as_no_value():
    _write_config('[captures]\ndirectory = "   "\n')
    assert configured_captures_dir() is None
    assert captures_dir_source() is None


def test_a_malformed_config_is_refused_rather_than_ignored():
    """Silently ignoring it would hide the very setting the file exists to make."""
    _write_config("[captures\ndirectory = nonsense\n")
    with pytest.raises(ConfigFileUnreadable):
        configured_captures_dir()
    with pytest.raises(ConfigFileUnreadable):
        configured_serial()


def test_a_malformed_config_does_not_break_importing_the_package():
    """The module-level AUTHORIZED_SERIAL constants are read while importing.

    A file too broken to parse must not take down every command, including the
    one that exists to fix it - so this reader falls back where the readers
    that aim an operation at hardware refuse.
    """
    _write_config("[captures\nnonsense\n")
    assert stated_serial() == EXAMPLE_SERIAL


def test_the_serial_still_reports_its_own_source(monkeypatch):
    path = _write_config('[scanner]\nserial = "ABC123"\n')
    assert configured_serial() == "ABC123"
    assert serial_source() == str(path)
    monkeypatch.setenv(SERIAL_ENV_VAR, "OTHER")
    assert serial_source() == SERIAL_ENV_VAR


# --- what commands do with it ------------------------------------------------


def test_each_command_gets_its_own_subdirectory():
    """Two commands writing fixed filenames into one directory would collide."""
    _write_config('[captures]\ndirectory = "/data/scans"\n')
    assert captures_dir_for("depth-frame") == Path("/data/scans/depth-frame")
    assert captures_dir_for("rgb-frame") == Path("/data/scans/rgb-frame")


def test_the_fallback_is_used_only_when_nothing_is_configured():
    fallback = Path("/tmp/scanner3d-depth-frame")
    assert captures_dir_for("depth-frame", fallback) == fallback
    _write_config('[captures]\ndirectory = "/data/scans"\n')
    assert captures_dir_for("depth-frame", fallback) == Path("/data/scans/depth-frame")


def test_a_dated_name_is_unique_per_second_and_reads_as_a_date():
    stamped = dated_capture_name("sweep", now=datetime(2026, 8, 25, 17, 45, 1))
    assert stamped == "sweep-20260825-174501"


# --- writing -----------------------------------------------------------------


def test_setting_a_directory_creates_the_file_and_is_read_back(tmp_path):
    target = set_captures_dir(tmp_path / "scans")
    assert target == tmp_path / "scans"
    assert configured_captures_dir() == tmp_path / "scans"
    assert 'directory = "' in config_path().read_text(encoding="utf-8")


def test_setting_a_directory_neither_creates_nor_requires_it(tmp_path):
    """Whether the path exists is the writing command's business, not this one's."""
    absent = tmp_path / "not-there"
    assert set_captures_dir(absent) == absent
    assert not absent.exists()


def test_a_relative_path_is_recorded_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert set_captures_dir(Path("scans")) == tmp_path / "scans"
    assert str(tmp_path) in config_path().read_text(encoding="utf-8")


def test_writing_preserves_comments_and_every_other_setting(tmp_path):
    """Losing someone's notes to record a default is not a trade this makes."""
    _write_config(
        "# the scanner on the bench\n"
        "[scanner]\n"
        'serial = "ABC123"  # engraved on the base\n'
        "\n"
        "[captures]\n"
        "# where scans land\n"
        'directory = "/old/place"\n'
    )
    set_captures_dir(tmp_path / "new-place")
    text = config_path().read_text(encoding="utf-8")
    assert "# the scanner on the bench" in text
    assert "# engraved on the base" in text
    assert "# where scans land" in text
    assert "/old/place" not in text
    assert configured_serial() == "ABC123"
    assert configured_captures_dir() == tmp_path / "new-place"


def test_a_new_table_is_appended_without_disturbing_an_existing_one(tmp_path):
    _write_config('[scanner]\nserial = "ABC123"\n')
    set_captures_dir(tmp_path / "scans")
    assert configured_serial() == "ABC123"
    assert configured_captures_dir() == tmp_path / "scans"


def test_clearing_removes_the_setting_and_reports_whether_there_was_one(tmp_path):
    set_captures_dir(tmp_path / "scans")
    assert clear_captures_dir() is True
    assert configured_captures_dir() is None
    assert clear_captures_dir() is False


def test_clearing_leaves_the_rest_of_the_file_alone(tmp_path):
    _write_config('[scanner]\nserial = "ABC123"\n\n[captures]\ndirectory = "/old"\n')
    assert clear_captures_dir() is True
    assert configured_serial() == "ABC123"


def test_a_key_of_the_same_name_in_another_table_is_left_alone(tmp_path):
    """The table boundary is what decides which 'directory' this owns."""
    _write_config('[captures]\ndirectory = "/old"\n\n[export]\ndirectory = "/exports"\n')
    set_captures_dir(tmp_path / "scans")
    text = config_path().read_text(encoding="utf-8")
    assert '/exports"' in text
    assert configured_captures_dir() == tmp_path / "scans"


def test_a_setting_below_an_array_of_tables_is_not_swallowed_into_it(tmp_path):
    _write_config('[captures]\ndirectory = "/old"\n\n[[profiles]]\ndirectory = "/elsewhere"\n')
    set_captures_dir(tmp_path / "scans")
    assert configured_captures_dir() == tmp_path / "scans"
    assert "/elsewhere" in config_path().read_text(encoding="utf-8")


def test_a_malformed_file_is_never_rewritten(tmp_path):
    path = _write_config("[captures\nnonsense\n")
    with pytest.raises(ConfigFileUnreadable):
        set_captures_dir(tmp_path / "scans")
    assert path.read_text(encoding="utf-8") == "[captures\nnonsense\n"


def test_a_form_this_cannot_edit_in_place_is_refused_not_mangled(tmp_path):
    """A dotted key states the same setting; appending a table would break the file."""
    original = 'captures.directory = "/old/place"\n'
    path = _write_config(original)
    with pytest.raises(ConfigWriteRefused):
        set_captures_dir(tmp_path / "scans")
    assert path.read_text(encoding="utf-8") == original
    assert configured_captures_dir() == Path("/old/place")


def test_a_path_needing_escaping_survives_the_round_trip(tmp_path):
    awkward = tmp_path / 'quote"and\\backslash'
    assert set_captures_dir(awkward) == awkward
    assert configured_captures_dir() == awkward
