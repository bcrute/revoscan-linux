"""Where captures go, who says so, and what a write is allowed to disturb.

Every test here drives the real file on a temporary XDG root (see the autouse
``isolated_config`` fixture in ``conftest``), because the thing worth proving
about a setting that persists is what actually lands on disk.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from scanner3d import config as config_module
from scanner3d.config import (
    CAPTURES_DIR_ENV_VAR,
    EXAMPLE_SERIAL,
    SERIAL_ENV_VAR,
    CaptureDirUnavailable,
    CaptureNameInvalid,
    ConfigFileUnreadable,
    ConfigWriteRefused,
    allocate_capture_dir,
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
    validate_capture_name,
)
from scanner3d.rgb_frame import RGB_ARTIFACT_NAME, preserve_rgb_frame
from scanner3d.v4l2_capture import (
    MIN_BYTES_PER_LINE,
    RAW_ARTIFACT_NAME,
    PixelFormat,
    preserve_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_a_relative_path_in_the_file_is_resolved_against_the_file(monkeypatch, tmp_path):
    """A default that means a different directory per terminal is not a default.

    The config file has one location, so a relative path written into it has
    one meaning. The working directory of whichever shell happened to run the
    command is not it.
    """
    path = _write_config('[captures]\ndirectory = "scans"\n')
    monkeypatch.chdir(tmp_path)
    assert configured_captures_dir() == path.parent / "scans"
    other = tmp_path / "somewhere-else"
    other.mkdir()
    monkeypatch.chdir(other)
    assert configured_captures_dir() == path.parent / "scans"


def test_a_relative_environment_override_is_resolved_against_the_shell(monkeypatch, tmp_path):
    """The opposite rule, deliberately: a variable is one process's override.

    Unlike the file, it was typed at a prompt moments ago and dies with the
    shell, so the directory that shell is standing in is the only base it could
    sensibly mean.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CAPTURES_DIR_ENV_VAR, "scratch")
    assert configured_captures_dir() == tmp_path / "scratch"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert configured_captures_dir() == elsewhere / "scratch"


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


def test_a_dated_name_reads_as_a_date_and_claims_nothing_more():
    """It is a readable name. Uniqueness is allocate_capture_dir's job, not its."""
    now = datetime(2026, 8, 25, 17, 45, 1)
    assert dated_capture_name("sweep", now=now) == "sweep-20260825-174501"
    assert dated_capture_name("sweep", now=now) == dated_capture_name("sweep", now=now)


# --- writing -----------------------------------------------------------------


def test_setting_a_directory_creates_the_file_and_is_read_back(tmp_path):
    recorded = set_captures_dir(tmp_path / "scans")
    assert recorded.path == tmp_path / "scans"
    assert recorded.changed is True
    assert configured_captures_dir() == tmp_path / "scans"
    assert 'directory = "' in config_path().read_text(encoding="utf-8")


def test_setting_a_directory_neither_creates_nor_requires_it(tmp_path):
    """Whether the path exists is the writing command's business, not this one's."""
    absent = tmp_path / "not-there"
    assert set_captures_dir(absent).path == absent
    assert not absent.exists()


def test_a_relative_path_is_recorded_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert set_captures_dir(Path("scans")).path == tmp_path / "scans"
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
    assert set_captures_dir(awkward).path == awkward
    assert configured_captures_dir() == awkward


# --- table boundaries the editor must not walk through ------------------------


def test_a_commented_table_header_still_ends_the_table_above_it(tmp_path):
    """The reproduced bug: [export] # comment was not recognised as a header.

    Starting from [captures], the scan ran straight through the header and
    deleted the next 'directory' assignment it found - one belonging to a table
    this has no business touching. The value the caller asked about was indeed
    absent afterwards, so a check that only looked at the target value said the
    edit had worked.
    """
    path = _write_config(
        "[captures]\n"
        'directory = "/old"\n'
        "\n"
        "[export]  # where finished models are published\n"
        'directory = "/exports"\n'
    )
    assert clear_captures_dir() is True
    text = path.read_text(encoding="utf-8")
    assert 'directory = "/exports"' in text
    assert "[export]  # where finished models are published" in text
    assert configured_captures_dir() is None


def test_an_empty_captures_table_never_reaches_a_later_commented_table(tmp_path):
    """The exact original reproducer, with nothing between the two headers.

    Every other boundary test puts a captures ``directory`` before the header,
    so the scan stops at that assignment and never reaches the boundary at all.
    Only an EMPTY ``[captures]`` makes the scan run to the end of the table,
    which is the walk that used to step through ``[export]  # comment`` and
    delete a setting belonging to somebody else's table.

    Nothing here should be written: there is no captures value to remove, so
    the clear is a no-op and the file must come back byte for byte.
    """
    original = (
        '[captures]\n\n[export]  # where finished models are published\ndirectory = "/exports"\n'
    )
    path = _write_config(original)
    before = path.read_bytes()

    assert clear_captures_dir() is False
    assert path.read_bytes() == before
    assert config_module._load_config() == {
        "captures": {},
        "export": {"directory": "/exports"},
    }


def test_a_commented_array_of_tables_header_is_a_boundary_too(tmp_path):
    path = _write_config(
        "[captures]\n"
        'directory = "/old"\n'
        "\n"
        "[[profiles]] # one per turntable\n"
        'directory = "/elsewhere"\n'
    )
    set_captures_dir(tmp_path / "scans")
    assert '/elsewhere"' in path.read_text(encoding="utf-8")
    assert configured_captures_dir() == tmp_path / "scans"


def test_a_quoted_or_spaced_table_header_is_a_boundary_too(tmp_path):
    path = _write_config(
        "[captures]\n"
        'directory = "/old"\n'
        "\n"
        '[ "odd name" ]\t# quoted, padded and commented, and all of it legal\n'
        'directory = "/quoted"\n'
    )
    assert clear_captures_dir() is True
    assert '/quoted"' in path.read_text(encoding="utf-8")


def test_an_unbalanced_bracket_line_is_not_mistaken_for_a_header(tmp_path):
    """[[x] is not a table header, and treating it as one would end a table early."""
    assert config_module._table_header("[[x]\n") is None
    assert config_module._table_header("[x]]\n") is None
    assert config_module._table_header("[x] # fine\n") == (("x",), False)
    assert config_module._table_header("[[x]] # fine\n") == (("x",), True)
    assert config_module._table_header("[a.b]\n") == (("a", "b"), False)
    assert config_module._table_header('["a.b"]\n') == (("a.b",), False)


def test_an_escaped_quoted_table_header_is_the_table_it_parses_as(tmp_path):
    """``["cap\u0074ures"]`` IS ``[captures]``, and clearing it must say so.

    The hand-written unquoter only undid ``\\"`` and ``\\\\``, so the header
    read back as the literal text ``cap\u0074ures``. ``configured_captures_dir``
    goes through tomllib and saw the setting; the editor did not, found no
    table to edit, and reported ``False`` while leaving the value in the file -
    a clear that silently did nothing.
    """
    path = _write_config('["cap\\u0074ures"]\ndirectory = "/old"\n')
    assert configured_captures_dir() == Path("/old")

    assert clear_captures_dir() is True
    assert configured_captures_dir() is None
    assert path.read_text(encoding="utf-8") == '["cap\\u0074ures"]\n'


def test_an_escaped_quoted_table_header_is_edited_in_place_on_a_set(tmp_path):
    """A set must reuse that table, not append a second declaration of it.

    Appending ``[captures]`` below a table already declared as
    ``["cap\u0074ures"]`` states the same table twice, which is not valid TOML,
    so the write was refused outright rather than done.
    """
    path = _write_config('# kept\n["cap\\u0074ures"]\ndirectory = "/old"\n')
    recorded = set_captures_dir(tmp_path / "scans")

    assert recorded.changed is True
    assert configured_captures_dir() == tmp_path / "scans"
    text = path.read_text(encoding="utf-8")
    assert text == f'# kept\n["cap\\u0074ures"]\ndirectory = "{tmp_path / "scans"}"\n'
    assert "[captures]" not in text  # no second declaration of the same table
    assert text.count("[") == 1


def test_an_escaped_quoted_assignment_key_is_the_key_it_parses_as(tmp_path):
    """The same defect on the assignment side: ``"direc\u0074ory"`` is ``directory``."""
    path = _write_config('[captures]\n"direc\\u0074ory" = "/old"  # trailing note\n')
    assert configured_captures_dir() == Path("/old")

    assert clear_captures_dir() is True
    assert configured_captures_dir() is None
    assert path.read_text(encoding="utf-8") == "[captures]\n"


def test_setting_over_an_escaped_quoted_assignment_key_replaces_that_one_line(tmp_path):
    """One line changes; every other byte, comments included, is left as it was."""
    path = _write_config(
        "# a note someone wrote\n"
        "[captures]\n"
        '"direc\\u0074ory" = "/old"\n'
        "\n"
        "[export]\n"
        'directory = "/exports"\n'
    )
    set_captures_dir(tmp_path / "scans")

    assert path.read_text(encoding="utf-8") == (
        "# a note someone wrote\n"
        "[captures]\n"
        f'directory = "{tmp_path / "scans"}"\n'
        "\n"
        "[export]\n"
        'directory = "/exports"\n'
    )
    assert config_module._load_config()["export"] == {"directory": "/exports"}


def test_a_long_form_unicode_escape_in_a_key_is_decoded_too(tmp_path):
    r"""``\U0000....`` is as valid as ``\u....`` and must decode the same way."""
    path = _write_config('["\\U00000063aptures"]\n"\\U00000064irectory" = "/old"\n')
    assert configured_captures_dir() == Path("/old")

    assert clear_captures_dir() is True
    assert path.read_text(encoding="utf-8") == '["\\U00000063aptures"]\n'


def test_every_basic_string_escape_a_key_may_carry_decodes(tmp_path):
    """The decoder, directly - one case per escape TOML defines for basic strings."""
    unquote = config_module._unquote_key_part
    assert unquote(r'"\b\t\n\f\r\"\\"') == '\b\t\n\f\r"\\'
    assert unquote(r'"cap\u0074ures"') == "captures"
    assert unquote(r'"\U0001F600"') == "\U0001f600"
    assert unquote("bare-key") == "bare-key"


def test_a_literal_quoted_key_still_keeps_its_backslashes_verbatim(tmp_path):
    r"""``'cap\u0074ures'`` is a DIFFERENT table, exactly as tomllib reads it.

    Literal strings define no escapes, so decoding one here would make the
    editor rewrite a table the parser never called ``captures``.
    """
    assert config_module._unquote_key_part(r"'cap\u0074ures'") == r"cap\u0074ures"

    original = "['cap\\u0074ures']\ndirectory = \"/theirs\"\n"
    path = _write_config(original)
    assert configured_captures_dir() is None
    assert clear_captures_dir() is False
    assert path.read_text(encoding="utf-8") == original


def test_an_escaped_quoted_key_in_another_table_is_still_left_alone(tmp_path):
    """Decoding must find the right key, not merely more keys."""
    path = _write_config(
        '[captures]\ndirectory = "/old"\n\n[export]\n"direc\\u0074ory" = "/exports"\n'
    )
    assert clear_captures_dir() is True
    assert path.read_text(encoding="utf-8") == (
        '[captures]\n\n[export]\n"direc\\u0074ory" = "/exports"\n'
    )
    assert config_module._load_config()["export"] == {"directory": "/exports"}


# --- either line ending TOML defines, and the one the file already uses -------


def _write_config_bytes(raw: bytes) -> Path:
    """Put exact bytes on disk: line endings are the thing under test here."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def test_a_table_header_is_recognized_with_either_terminator_or_none():
    r"""LF, CRLF and no terminator at all are one header, as tomllib reads them.

    The pattern used to be anchored at ``$``, which accepts ``[captures]\n``
    and rejects ``[captures]\r\n``. Nothing about which bytes end a line says
    anything about which table a line declares.
    """
    header = config_module._table_header
    assert header("[captures]") == (("captures",), False)
    assert header("[captures]\n") == (("captures",), False)
    assert header("[captures]\r\n") == (("captures",), False)
    assert header("  [ captures ]  # note\r\n") == (("captures",), False)
    assert header('["cap\\u0074ures"]\r\n') == (("captures",), False)
    assert header("[[frames]] # note\r\n") == (("frames",), True)
    assert header('directory = "/x"\r\n') is None


def test_lines_are_split_only_where_toml_ends_a_line():
    r"""``str.splitlines`` also breaks on \v, \f and \u2028; TOML does not.

    A form feed inside a multi-line string would otherwise split one logical
    line in two, and a table boundary computed from those lines could land in
    the wrong place.
    """
    split = config_module._split_lines
    assert split("a\r\nb\n") == ["a\r\n", "b\n"]
    assert split("a\fb\u2028c\n") == ["a\fb\u2028c\n"]
    assert split("no trailing newline") == ["no trailing newline"]
    assert split("") == []


def test_a_crlf_config_clears_its_ordinary_captures_header(tmp_path):
    r"""The reviewed defect: ``[captures]\r\n`` parsed, but did not clear.

    ``configured_captures_dir()`` went through tomllib and returned ``/old``
    while ``clear_captures_dir()`` returned False and left the setting in the
    file - a clear that reported nothing to do and did nothing.
    """
    path = _write_config_bytes(b'[captures]\r\ndirectory = "/old"\r\n')
    assert configured_captures_dir() == Path("/old")

    assert clear_captures_dir() is True
    assert configured_captures_dir() is None
    assert path.read_bytes() == b"[captures]\r\n"


def test_a_crlf_config_sets_over_its_ordinary_captures_header(tmp_path):
    """A set replaces the one line and leaves the file CRLF throughout."""
    path = _write_config_bytes(b'# kept\r\n[captures]\r\ndirectory = "/old"\r\n')
    target = tmp_path / "scans"

    assert set_captures_dir(target).changed is True
    assert configured_captures_dir() == target
    assert path.read_bytes() == f'# kept\r\n[captures]\r\ndirectory = "{target}"\r\n'.encode()


def test_a_crlf_escaped_quoted_header_and_key_clear_together(tmp_path):
    r"""Both round-2 escape cases again, this time CRLF-terminated."""
    path = _write_config_bytes(b'["cap\\u0074ures"]\r\n"direc\\u0074ory" = "/old"  # note\r\n')
    assert configured_captures_dir() == Path("/old")

    assert clear_captures_dir() is True
    assert configured_captures_dir() is None
    assert path.read_bytes() == b'["cap\\u0074ures"]\r\n'


def test_a_crlf_escaped_quoted_header_is_set_in_place(tmp_path):
    """The escaped header is reused, not declared a second time below itself."""
    path = _write_config_bytes(b'["cap\\u0074ures"]\r\ndirectory = "/old"\r\n')
    target = tmp_path / "scans"

    assert set_captures_dir(target).changed is True
    assert configured_captures_dir() == target
    raw = path.read_bytes()
    assert raw == f'["cap\\u0074ures"]\r\ndirectory = "{target}"\r\n'.encode()
    assert b"[captures]" not in raw  # no second declaration of the same table


def test_a_crlf_clear_stops_at_a_later_commented_table(tmp_path):
    """The cross-table scan again, with CRLF headers on both tables.

    A boundary missed because of the carriage return would put the scan inside
    ``[export]`` and delete a ``directory`` belonging to somebody else.
    """
    path = _write_config_bytes(
        b'[captures]\r\ndirectory = "/old"\r\n[export] # inline comment\r\n'
        b'directory = "/exports"\r\n'
    )
    assert clear_captures_dir() is True

    assert path.read_bytes() == (
        b'[captures]\r\n[export] # inline comment\r\ndirectory = "/exports"\r\n'
    )
    assert config_module._load_config()["export"] == {"directory": "/exports"}


def test_an_empty_crlf_captures_table_before_a_commented_table_is_left_alone(tmp_path):
    """Nothing to clear, and the unrelated ``directory`` stays byte-identical."""
    original = b'[captures]\r\n[export] # inline comment\r\ndirectory = "/exports"\r\n'
    path = _write_config_bytes(original)

    assert clear_captures_dir() is False
    assert path.read_bytes() == original


def test_inserting_into_a_crlf_config_does_not_make_it_mixed(tmp_path):
    """An inserted assignment takes the terminator of the header it goes under."""
    path = _write_config_bytes(b"[captures]\r\n# a note\r\n")
    target = tmp_path / "scans"

    assert set_captures_dir(target).changed is True
    raw = path.read_bytes()
    assert raw == f'[captures]\r\ndirectory = "{target}"\r\n# a note\r\n'.encode()
    assert raw.count(b"\n") == raw.count(b"\r\n")  # no lone LF anywhere


def test_appending_a_captures_table_to_a_crlf_config_keeps_crlf(tmp_path):
    """A file with no captures table at all still keeps its own line ending."""
    path = _write_config_bytes(b'[export]\r\ndirectory = "/exports"\r\n')
    target = tmp_path / "scans"

    assert set_captures_dir(target).changed is True
    raw = path.read_bytes()
    assert (
        raw
        == (
            f'[export]\r\ndirectory = "/exports"\r\n\r\n[captures]\r\ndirectory = "{target}"\r\n'
        ).encode()
    )
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_an_lf_config_is_still_written_back_as_lf(tmp_path):
    """The retained side of the same rule: LF files do not gain carriage returns."""
    path = _write_config_bytes(b"[captures]\r\n")
    path.write_bytes(b'# kept\n[captures]\ndirectory = "/old"\n')
    target = tmp_path / "scans"

    assert set_captures_dir(target).changed is True
    raw = path.read_bytes()
    assert raw == f'# kept\n[captures]\ndirectory = "{target}"\n'.encode()
    assert b"\r" not in raw


def test_the_edited_line_keeps_its_own_ending_in_a_mixed_file(tmp_path):
    """Preservation is per line, so an already-mixed file is not re-flattened."""
    path = _write_config_bytes(b'[captures]\ndirectory = "/old"\r\n[export]\n')
    target = tmp_path / "scans"

    assert set_captures_dir(target).changed is True
    assert path.read_bytes() == f'[captures]\ndirectory = "{target}"\r\n[export]\n'.encode()


def test_an_edit_that_would_disturb_anything_else_is_refused_and_writes_nothing(
    monkeypatch, tmp_path
):
    """The backstop for every boundary case nobody thought of.

    The requested value being right is not evidence that nothing else moved, so
    the whole parse is compared key by key. Here an editor that also drops an
    unrelated table is injected; the check must catch it on the difference
    alone, without knowing what went wrong.
    """
    original = '[captures]\ndirectory = "/old"\n\n[export]\ndirectory = "/exports"\n'
    path = _write_config(original)

    def clumsy(text: str, table: str, key: str, value: str | None) -> str:
        return f'[captures]\ndirectory = "{value}"\n'

    monkeypatch.setattr(config_module, "_edit_toml", clumsy)
    with pytest.raises(ConfigWriteRefused) as excinfo:
        set_captures_dir(tmp_path / "scans")
    assert "left alone" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == original


# --- what a rewrite is allowed to do to the file itself -----------------------


def _mode_of(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_an_existing_private_config_keeps_its_permissions(tmp_path):
    """A 0600 config became 0664 before, because the scratch file took the umask."""
    path = _write_config('[scanner]\nserial = "ABC123"\n')
    path.chmod(0o600)
    set_captures_dir(tmp_path / "scans")
    assert _mode_of(path) == 0o600
    assert configured_captures_dir() == tmp_path / "scans"


def test_a_config_file_created_here_is_private(tmp_path):
    """The next thing written into it is a path inside somebody's home directory."""
    set_captures_dir(tmp_path / "scans")
    assert _mode_of(config_path()) == 0o600


def test_a_successful_write_leaves_no_temporary_file_behind(tmp_path):
    set_captures_dir(tmp_path / "scans")
    assert [p.name for p in config_path().parent.iterdir()] == [config_path().name]


def test_a_symlinked_config_path_is_refused_and_leaves_link_and_target_alone(tmp_path):
    """Replacing the link would strand the file it points at, silently.

    Especially bad here, where the whole promise is that settings this module
    does not understand survive a write.
    """
    real = tmp_path / "real-config.toml"
    real.write_text('[scanner]\nserial = "ABC123"\n', encoding="utf-8")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(real)

    with pytest.raises(ConfigWriteRefused) as excinfo:
        set_captures_dir(tmp_path / "scans")
    assert "symbolic link" in str(excinfo.value)
    assert path.is_symlink()
    assert path.readlink() == real
    assert real.read_text(encoding="utf-8") == '[scanner]\nserial = "ABC123"\n'


# --- injected write failures are sentences, not tracebacks --------------------


def _write_failure_leaves_the_original(tmp_path, monkeypatch, target, name, boom):
    original = '[captures]\ndirectory = "/old"\n'
    path = _write_config(original)
    monkeypatch.setattr(target, name, boom)
    with pytest.raises(ConfigWriteRefused) as excinfo:
        set_captures_dir(tmp_path / "scans")
    assert path.read_text(encoding="utf-8") == original
    assert [p.name for p in path.parent.iterdir()] == [path.name]
    return str(excinfo.value)


def test_a_failed_sync_is_reported_as_an_unchanged_file(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    message = _write_failure_leaves_the_original(tmp_path, monkeypatch, os, "fsync", boom)
    assert "could not be written" in message
    assert "is unchanged" in message


def test_a_failed_replace_is_reported_as_an_unchanged_file(tmp_path, monkeypatch):
    def boom(self, _target):
        raise OSError(13, "Permission denied")

    message = _write_failure_leaves_the_original(tmp_path, monkeypatch, Path, "replace", boom)
    assert "could not be put in place" in message
    assert "is unchanged" in message


def test_a_failed_chmod_is_reported_as_an_unchanged_file(tmp_path, monkeypatch):
    def boom(self, _mode):
        raise OSError(1, "Operation not permitted")

    message = _write_failure_leaves_the_original(tmp_path, monkeypatch, Path, "chmod", boom)
    assert "is unchanged" in message


def test_an_uncreatable_temporary_file_is_reported_as_an_unchanged_file(tmp_path, monkeypatch):
    def boom(**_kwargs):
        raise OSError(30, "Read-only file system")

    message = _write_failure_leaves_the_original(tmp_path, monkeypatch, tempfile, "mkstemp", boom)
    assert "temporary file" in message
    assert "is unchanged" in message


# --- one fresh directory per capture, never a shared one ----------------------

FIXED = datetime(2026, 8, 25, 17, 45, 1)


def test_nothing_is_allocated_when_no_captures_root_is_configured():
    """None is not an error: the caller keeps whatever default it always had."""
    assert allocate_capture_dir("depth-frame") is None


def test_an_allocated_directory_is_created_under_the_command_that_asked(tmp_path):
    set_captures_dir(tmp_path / "scans")
    allocated = allocate_capture_dir("sweep", now=FIXED)
    assert allocated is not None
    assert allocated == tmp_path / "scans" / "sweep" / "20260825-174501"
    assert allocated.is_dir()


def test_two_allocations_in_the_same_second_get_different_directories(tmp_path):
    """Wall-clock seconds do not make a name unique, and this never pretends they do."""
    set_captures_dir(tmp_path / "scans")
    # A directory already carrying the name this second would produce - a
    # restart, an earlier run, anything.
    taken = tmp_path / "scans" / "sweep" / "20260825-174501"
    taken.mkdir(parents=True)
    (taken / "frame_0000.raw").write_bytes(b"older evidence")

    first = allocate_capture_dir("sweep", now=FIXED)
    second = allocate_capture_dir("sweep", now=FIXED)
    assert first is not None and second is not None

    assert len({first, second, taken}) == 3
    assert first.is_dir() and second.is_dir()
    assert list(first.iterdir()) == [] and list(second.iterdir()) == []
    assert (taken / "frame_0000.raw").read_bytes() == b"older evidence"


def test_a_clock_dragged_backwards_still_gets_a_fresh_directory(tmp_path):
    """A repeated local hour - a DST fall-back, an NTP correction - repeats names."""
    set_captures_dir(tmp_path / "scans")
    later = allocate_capture_dir("scan", now=datetime(2026, 10, 25, 2, 30, 0))
    earlier = allocate_capture_dir("scan", now=datetime(2026, 10, 25, 1, 30, 0))
    repeated = allocate_capture_dir("scan", now=datetime(2026, 10, 25, 2, 30, 0))
    assert later is not None and earlier is not None and repeated is not None
    assert len({later, earlier, repeated}) == 3
    assert all(p.is_dir() for p in (later, earlier, repeated))


def test_allocation_refuses_rather_than_reuse_when_every_name_is_taken(tmp_path, monkeypatch):
    set_captures_dir(tmp_path / "scans")
    monkeypatch.setattr(config_module, "CAPTURE_DIR_ATTEMPTS", 2)
    parent = tmp_path / "scans" / "sweep"
    parent.mkdir(parents=True)
    (parent / "20260825-174501").mkdir()
    (parent / "20260825-174501-2").mkdir()
    with pytest.raises(CaptureDirUnavailable) as excinfo:
        allocate_capture_dir("sweep", now=FIXED)
    assert "nothing was captured" in str(excinfo.value)


def test_a_captures_root_that_cannot_be_created_is_a_refusal_not_a_traceback(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("I am a file\n", encoding="utf-8")
    set_captures_dir(blocker)
    with pytest.raises(CaptureDirUnavailable) as excinfo:
        allocate_capture_dir("sweep", now=FIXED)
    assert "Nothing was captured" in str(excinfo.value)


# --- an optionally named leaf, under the same collision policy ----------------


def test_a_caller_that_supplies_no_name_still_gets_the_dated_directory(tmp_path):
    """The retained side: every CLI command passes no name and must not notice."""
    set_captures_dir(tmp_path / "scans")
    assert allocate_capture_dir("sweep", now=FIXED) == (
        tmp_path / "scans" / "sweep" / "20260825-174501"
    )


def test_a_supplied_name_becomes_the_leaf_under_the_same_namespace(tmp_path):
    set_captures_dir(tmp_path / "scans")
    allocated = allocate_capture_dir("scan", name="chassis left side", now=FIXED)
    assert allocated == tmp_path / "scans" / "scan" / "chassis left side"
    assert allocated is not None and allocated.is_dir()


def test_a_blank_or_whitespace_name_falls_back_to_the_timestamp(tmp_path):
    """ "You name it" is an answer, and it is the answer the old behaviour gave."""
    set_captures_dir(tmp_path / "scans")
    blank = allocate_capture_dir("scan", name="", now=FIXED)
    # Spaces only: a tab is a control character, and a control character is a
    # refusal rather than a roundabout way of saying "you name it".
    spaces = allocate_capture_dir("scan", name="     ", now=FIXED)
    assert blank == tmp_path / "scans" / "scan" / "20260825-174501"
    assert spaces == tmp_path / "scans" / "scan" / "20260825-174501-2"


def test_surrounding_whitespace_is_trimmed_off_a_supplied_name(tmp_path):
    """A folder whose name starts with a space is one somebody must quote forever."""
    set_captures_dir(tmp_path / "scans")
    allocated = allocate_capture_dir("scan", name="  turntable pass  ", now=FIXED)
    assert allocated == tmp_path / "scans" / "scan" / "turntable pass"


def test_a_repeated_name_is_suffixed_rather_than_written_into(tmp_path):
    """The whole point: the same name twice is two scans, not one ruined one."""
    set_captures_dir(tmp_path / "scans")
    first = allocate_capture_dir("scan", name="chassis", now=FIXED)
    assert first is not None
    (first / "frame_0000.raw").write_bytes(b"older evidence")
    second = allocate_capture_dir("scan", name="chassis", now=FIXED)
    third = allocate_capture_dir("scan", name="chassis", now=FIXED)

    assert second == tmp_path / "scans" / "scan" / "chassis-2"
    assert third == tmp_path / "scans" / "scan" / "chassis-3"
    assert second is not None and third is not None
    assert list(second.iterdir()) == [] and list(third.iterdir()) == []
    assert (first / "frame_0000.raw").read_bytes() == b"older evidence"


def test_a_named_allocation_refuses_rather_than_reuse_when_every_name_is_taken(
    tmp_path, monkeypatch
):
    set_captures_dir(tmp_path / "scans")
    monkeypatch.setattr(config_module, "CAPTURE_DIR_ATTEMPTS", 2)
    parent = tmp_path / "scans" / "scan"
    parent.mkdir(parents=True)
    (parent / "chassis").mkdir()
    (parent / "chassis-2").mkdir()
    with pytest.raises(CaptureDirUnavailable) as excinfo:
        allocate_capture_dir("scan", name="chassis", now=FIXED)
    message = str(excinfo.value)
    assert "starting chassis" in message
    assert "nothing was captured" in message


@pytest.mark.parametrize(
    "name",
    [
        ".",
        "..",
        "  ..  ",
        "scans/chassis",
        "/absolute",
        "../escape",
        "trailing/",
        "with\x00nul",
        "with\nnewline",
        "with\x7fdelete",
        # Path-shaped on a host this tree can be copied to, whatever os.sep
        # says here.
        "parent\\child",
        "C:\\captures\\scan",
        "trailing\\",
        # C1 controls: below neither " " nor "\x7f", and category Cc all the
        # same.
        "with\x85next-line",
        "with\x9fcontrol",
        # str.strip() would erase these entirely, turning a refusal into a
        # silent request for a dated folder.
        "\x85",
        "\n",
        "\t",
        " \x1f ",
    ],
)
def test_a_name_that_is_not_one_folder_name_is_refused_and_creates_nothing(tmp_path, name):
    """Refused BEFORE anything exists - including the command's own namespace."""
    set_captures_dir(tmp_path / "scans")
    with pytest.raises(CaptureNameInvalid) as excinfo:
        allocate_capture_dir("scan", name=name, now=FIXED)
    assert "nothing was captured" in str(excinfo.value).lower()
    assert not (tmp_path / "scans").exists()


def test_a_refused_name_is_refused_before_the_captures_root_is_even_read(tmp_path, monkeypatch):
    """Validation is first, so an invalid name cannot depend on configuration."""

    def unreachable() -> Path:
        raise AssertionError("the configuration was read for a name already known bad")

    monkeypatch.setattr(config_module, "configured_captures_dir", unreachable)
    with pytest.raises(CaptureNameInvalid):
        allocate_capture_dir("scan", name="../elsewhere", now=FIXED)


def test_an_invalid_name_is_also_a_capture_dir_unavailable():
    """A caller already guarding allocation refuses safely without knowing this."""
    assert issubclass(CaptureNameInvalid, CaptureDirUnavailable)


def test_validating_a_name_returns_the_trimmed_component_or_empty():
    assert validate_capture_name("chassis left side") == "chassis left side"
    assert validate_capture_name("  turntable pass  ") == "turntable pass"
    assert validate_capture_name("") == ""
    assert validate_capture_name("   ") == ""
    # Ordinary names a cautious validator would be tempted to mangle.
    for kept in ("...", "-2", "scan.v2", "Chassis (left)", "\u00e9tude"):
        assert validate_capture_name(kept) == kept


def test_a_backslash_names_a_place_on_this_host_too():
    """os.sep describes this machine; a captures tree outlives this machine."""
    for path_shaped in ("parent\\child", "C:\\captures\\scan", "..\\escape"):
        with pytest.raises(CaptureNameInvalid) as excinfo:
            validate_capture_name(path_shaped)
        assert "path separator" in str(excinfo.value)


def test_every_unicode_control_is_refused_before_the_trim_can_hide_it():
    """str.strip() removes \\x85 and \\x1c-\\x1f, so trimming first loses the refusal."""
    for controlled in ("with\x85next-line", "with\x9fcontrol", "\x85", " \x1f ", "tab\there"):
        with pytest.raises(CaptureNameInvalid) as excinfo:
            validate_capture_name(controlled)
        assert "control characters" in str(excinfo.value)
    # And the thing being protected: ordinary surrounding spaces still trim.
    assert validate_capture_name("  turntable pass  ") == "turntable pass"


def test_an_explicit_root_bypasses_the_configuration_but_not_the_namespace(tmp_path):
    """What the GUI's folder picker does: the chosen folder is the ROOT, not the leaf."""
    picked = tmp_path / "picked"
    allocated = allocate_capture_dir("scan", root=picked, name="chassis", now=FIXED)
    assert allocated == picked / "scan" / "chassis"
    assert allocated is not None and allocated.is_dir()


# --- the real preservation code, run twice ------------------------------------


def _depth_format() -> PixelFormat:
    """A kernel answer with no row padding, so preserve_frame writes one file."""
    return PixelFormat(
        width=640,
        height=400,
        pixelformat=0,
        fourcc="Y16 ",
        field=1,
        field_name="none",
        bytesperline=MIN_BYTES_PER_LINE,
        sizeimage=MIN_BYTES_PER_LINE * 400,
        colorspace=0,
        priv=0,
        flags=0,
        ycbcr_enc=0,
        quantization=0,
        xfer_func=0,
    )


def test_two_rgb_captures_in_one_second_both_preserve_their_frame(tmp_path):
    """The reproduced failure: the second write said 'rgb-frame.jpg: File exists'.

    This drives the real preservation function, not a seam that records a path:
    the bug was that the bytes could not be written, and only writing them
    proves it is gone.
    """
    set_captures_dir(tmp_path / "scans")
    first_dir = allocate_capture_dir("rgb-frame", now=FIXED)
    second_dir = allocate_capture_dir("rgb-frame", now=FIXED)
    assert first_dir is not None and second_dir is not None

    first_record = preserve_rgb_frame(b"first frame", first_dir)
    second_record = preserve_rgb_frame(b"second frame", second_dir)

    assert first_record.ok is True, first_record.note
    assert second_record.ok is True, second_record.note
    written = sorted(
        (path.parent.name, path.read_bytes())
        for path in (tmp_path / "scans" / "rgb-frame").rglob(RGB_ARTIFACT_NAME)
    )
    assert [payload for _name, payload in written] == [b"first frame", b"second frame"]


def test_two_depth_captures_in_one_second_both_preserve_their_frame(tmp_path):
    set_captures_dir(tmp_path / "scans")
    pixel_format = _depth_format()
    first_dir = allocate_capture_dir("depth-frame", now=FIXED)
    second_dir = allocate_capture_dir("depth-frame", now=FIXED)
    assert first_dir is not None and second_dir is not None

    first_record = preserve_frame(b"first frame", pixel_format, first_dir)
    second_record = preserve_frame(b"second frame", pixel_format, second_dir)

    assert first_record.ok is True, first_record.note
    assert second_record.ok is True, second_record.note
    payloads = sorted(
        path.read_bytes() for path in (tmp_path / "scans" / "depth-frame").rglob(RAW_ARTIFACT_NAME)
    )
    assert payloads == [b"first frame", b"second frame"]


# --- isolation that has to happen before collection ---------------------------


def test_a_hosts_own_configuration_cannot_reach_an_import_time_constant(tmp_path):
    """Run a fresh pytest with a host config set, and prove it does not leak in.

    ``AUTHORIZED_SERIAL`` is evaluated while its module is imported, which
    happens during collection - so a function-scoped fixture, however early, is
    already too late. Only a separate process with a genuinely populated
    ``XDG_CONFIG_HOME`` can show whether the isolation runs early enough, so
    that is what this does.
    """
    home = tmp_path / "host-config"
    (home / "scanner3d").mkdir(parents=True)
    (home / "scanner3d" / "config.toml").write_text(
        '[scanner]\nserial = "BENCH-SERIAL"\n\n[captures]\ndirectory = "/tmp/bench-scans"\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(home)
    env["SCANNER3D_SERIAL"] = "BENCH-SERIAL"
    env["SCANNER3D_CAPTURES_DIR"] = str(tmp_path / "bench-captures")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "tests/test_depth_scale.py::test_the_authorized_serial_is_the_one_the_owner_named",
            "tests/test_config.py::test_nothing_configured_reads_as_not_stated_rather_than_an_error",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


# --- the registration depth band ------------------------------------------------


def test_no_stated_depth_band_reads_as_not_stated():
    assert config_module.configured_depth_band() is None
    assert config_module.depth_band_source() is None


def test_the_registration_depth_band_is_read_from_the_config_file():
    path = _write_config("[registration]\ndepth_band = [130, 275.5]\n")
    assert config_module.configured_depth_band() == (130.0, 275.5)
    assert config_module.depth_band_source() == str(path)


@pytest.mark.parametrize(
    "value",
    [
        '"130-275"',
        "[130]",
        "[130, 275, 300]",
        "[275, 130]",
        "[130, 130]",
        "[0, 275]",
        "[-5, 275]",
        "[true, 275]",
        '["130", "275"]',
        "[nan, 275]",
        "[130, inf]",
    ],
)
def test_a_depth_band_the_clip_cannot_honour_is_refused_rather_than_ignored(value):
    """Silently ignoring it would hide the very setting the file exists to make."""
    _write_config(f"[registration]\ndepth_band = {value}\n")
    with pytest.raises(ConfigFileUnreadable) as caught:
        config_module.configured_depth_band()
    assert "depth_band" in str(caught.value)


def test_recording_the_depth_band_writes_a_list_the_readers_accept():
    _write_config('[captures]\ndirectory = "/scans"  # kept\n')
    assert config_module.set_depth_band(130, 275) is True
    text = config_path().read_text(encoding="utf-8")
    assert "[registration]\ndepth_band = [130, 275]\n" in text
    assert 'directory = "/scans"  # kept' in text
    assert config_module.configured_depth_band() == (130.0, 275.0)
    # Recording the same band again changes nothing, and says so.
    assert config_module.set_depth_band(130, 275) is False


def test_a_fractional_band_is_written_as_floats_and_read_back_exactly():
    assert config_module.set_depth_band(127.5, 262.25) is True
    assert config_module.configured_depth_band() == (127.5, 262.25)


def test_clearing_the_depth_band_removes_only_that_key():
    _write_config('[registration]\ndepth_band = [130, 275]\nother = "kept"\n')
    assert config_module.clear_depth_band() is True
    assert config_module.configured_depth_band() is None
    assert 'other = "kept"' in config_path().read_text(encoding="utf-8")
    assert config_module.clear_depth_band() is False


@pytest.mark.parametrize("near, far", [(275, 130), (0, 275), (130, 130), (-1, 100)])
def test_a_band_the_clip_could_not_apply_is_refused_before_the_file_is_touched(near, far):
    _write_config('[captures]\ndirectory = "/scans"\n')
    before = config_path().read_text(encoding="utf-8")
    with pytest.raises(ConfigWriteRefused):
        config_module.set_depth_band(near, far)
    assert config_path().read_text(encoding="utf-8") == before
