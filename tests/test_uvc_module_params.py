"""Offline tests for the scanner-012 ``reference/analysis/uvc_module_params`` helper.

The point of the helper is a single correction: ``quirks = 4294967295`` in sysfs
is the driver's "no forced quirks" sentinel, not "every quirk forced on". These
tests pin that reading, and pin the ELF reader that establishes it, against a
minimal ELF64 object synthesized in-process -- so they need no kernel module, no
binutils, no root and no device.

One test additionally runs against the host's real ``uvcvideo.ko`` when a
decompressed copy happens to be present, and skips otherwise. It is a bonus
cross-check, never a gate.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parent.parent / "reference" / "analysis" / "uvc_module_params.py"
_spec = importlib.util.spec_from_file_location("uvc_module_params", _HELPER)
assert _spec is not None
uvc_module_params = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules.setdefault("uvc_module_params", uvc_module_params)
_spec.loader.exec_module(uvc_module_params)


# --------------------------------------------------------------------------
# A minimal ELF64 object builder: header + .data + .bss + .symtab + .strtab
# + .shstrtab. Enough for the reader under test, and nothing more.
# --------------------------------------------------------------------------

_SHT_PROGBITS = 1
_SHT_SYMTAB = 2
_SHT_STRTAB = 3
_SHT_NOBITS = 8


def _strtab(names):
    blob = bytearray(b"\x00")
    offsets = {}
    for name in names:
        offsets[name] = len(blob)
        blob += name.encode("ascii") + b"\x00"
    return bytes(blob), offsets


def build_elf64(data_values, bss_symbols=()):
    """Build an ELF64 object placing ``data_values`` (name -> u32) in ``.data``.

    ``bss_symbols`` are declared in a NOBITS section, i.e. with no bytes in the
    file, which is how a zero-initialised module parameter is stored.
    """
    data_names = list(data_values)
    data_blob = b"".join(struct.pack("<I", data_values[n]) for n in data_names)
    data_offsets = {name: index * 4 for index, name in enumerate(data_names)}

    section_names = [".data", ".bss", ".symtab", ".strtab", ".shstrtab"]
    shstrtab_blob, shstr_offsets = _strtab(section_names)
    symbol_names = data_names + list(bss_symbols)
    strtab_blob, str_offsets = _strtab(symbol_names)

    # Section indices: 0 null, 1 .data, 2 .bss, 3 .symtab, 4 .strtab, 5 .shstrtab
    symbols = [b"\x00" * 24]
    for name in data_names:
        symbols.append(struct.pack("<IBBHQQ", str_offsets[name], 0, 0, 1, data_offsets[name], 4))
    for name in bss_symbols:
        symbols.append(struct.pack("<IBBHQQ", str_offsets[name], 0, 0, 2, 0, 4))
    symtab_blob = b"".join(symbols)

    header_size = 64
    shentsize = 64
    shnum = 6
    offset = header_size
    layout = {}
    for name, blob in (
        (".data", data_blob),
        (".symtab", symtab_blob),
        (".strtab", strtab_blob),
        (".shstrtab", shstrtab_blob),
    ):
        layout[name] = (offset, len(blob))
        offset += len(blob)
    shoff = offset

    body = data_blob + symtab_blob + strtab_blob + shstrtab_blob

    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # ELF64
    header[5] = 1  # little-endian
    header[6] = 1  # version
    struct.pack_into("<HH", header, 0x10, 1, 0x3E)  # ET_REL, x86-64
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HHH", header, 0x3A, shentsize, shnum, 5)

    def section(name_offset, sh_type, sh_offset, sh_size, sh_link=0, sh_entsize=0):
        entry = bytearray(64)
        struct.pack_into("<II", entry, 0, name_offset, sh_type)
        struct.pack_into("<QQ", entry, 0x18, sh_offset, sh_size)
        struct.pack_into("<I", entry, 0x28, sh_link)
        struct.pack_into("<Q", entry, 0x38, sh_entsize)
        return bytes(entry)

    sections = [
        bytes(64),
        section(shstr_offsets[".data"], _SHT_PROGBITS, *layout[".data"]),
        section(shstr_offsets[".bss"], _SHT_NOBITS, 0, 4 * len(bss_symbols)),
        section(
            shstr_offsets[".symtab"], _SHT_SYMTAB, *layout[".symtab"], sh_link=4, sh_entsize=24
        ),
        section(shstr_offsets[".strtab"], _SHT_STRTAB, *layout[".strtab"]),
        section(shstr_offsets[".shstrtab"], _SHT_STRTAB, *layout[".shstrtab"]),
    ]
    return bytes(header) + body + b"".join(sections)


UVCVIDEO_6_14_DEFAULTS = {
    "uvc_quirks_param": 0xFFFFFFFF,
    "uvc_no_drop_param": 1,
    "uvc_timeout_param": 5000,
}


@pytest.fixture(scope="module")
def elf():
    blob = build_elf64(UVCVIDEO_6_14_DEFAULTS, bss_symbols=("uvc_hw_timestamps_param",))
    return uvc_module_params.parse_elf64(blob)


# --------------------------------------------------------------------------
# The ELF reader
# --------------------------------------------------------------------------


def test_reads_initialised_values_out_of_data(elf):
    assert elf.read_u32("uvc_quirks_param") == 0xFFFFFFFF
    assert elf.read_u32("uvc_no_drop_param") == 1
    assert elf.read_u32("uvc_timeout_param") == 5000


def test_a_nobits_symbol_reads_as_zero_not_as_garbage(elf):
    """A ``.bss`` symbol has no bytes in the file; it is zeroed at load."""
    assert elf.read_u32("uvc_hw_timestamps_param") == 0


def test_absent_symbol_reads_as_none(elf):
    assert elf.read_u32("uvc_no_such_param") is None
    assert elf.symbol("uvc_no_such_param") is None


def test_sections_are_named(elf):
    assert [s.name for s in elf.sections][1:3] == [".data", ".bss"]


def test_non_elf_input_is_rejected():
    with pytest.raises(uvc_module_params.ElfError):
        uvc_module_params.parse_elf64(b"not an elf at all, not even close")


def test_elf32_is_rejected():
    blob = bytearray(build_elf64(UVCVIDEO_6_14_DEFAULTS))
    blob[4] = 1  # ELFCLASS32
    with pytest.raises(uvc_module_params.ElfError):
        uvc_module_params.parse_elf64(bytes(blob))


def test_big_endian_is_rejected():
    blob = bytearray(build_elf64(UVCVIDEO_6_14_DEFAULTS))
    blob[5] = 2  # ELFDATA2MSB
    with pytest.raises(uvc_module_params.ElfError):
        uvc_module_params.parse_elf64(bytes(blob))


# --------------------------------------------------------------------------
# The correction this helper exists for
# --------------------------------------------------------------------------


def test_all_ones_quirks_is_the_unset_sentinel_not_a_forced_mask(elf):
    """4294967295 == (unsigned)-1 == "no module parameter was supplied"."""
    live = {"quirks": "4294967295", "nodrop": "1", "timeout": "5000", "trace": "0"}
    comparisons = uvc_module_params.compare(elf, live)
    assert uvc_module_params.quirks_are_forced(comparisons) is False
    quirks = next(c for c in comparisons if c.sysfs_name == "quirks")
    assert quirks.compiled == uvc_module_params.QUIRKS_UNSET_SENTINEL
    assert quirks.overridden is False


def test_a_real_forced_quirk_mask_is_reported_as_forced(elf):
    live = {"quirks": "256"}
    comparisons = uvc_module_params.compare(elf, live)
    assert uvc_module_params.quirks_are_forced(comparisons) is True
    quirks = next(c for c in comparisons if c.sysfs_name == "quirks")
    assert quirks.overridden is True


def test_zero_quirks_is_also_a_forced_value(elf):
    """Explicitly forcing 0 is a deliberate override, not the default."""
    comparisons = uvc_module_params.compare(elf, {"quirks": "0"})
    assert uvc_module_params.quirks_are_forced(comparisons) is True


def test_unreadable_live_value_is_not_comparable(elf):
    comparisons = uvc_module_params.compare(elf, {"quirks": None})
    quirks = next(c for c in comparisons if c.sysfs_name == "quirks")
    assert quirks.overridden is None
    assert uvc_module_params.quirks_are_forced(comparisons) is None


def test_non_integer_live_value_is_not_comparable(elf):
    """`clock` reads as a string; a raw integer comparison must refuse it."""
    comparisons = uvc_module_params.compare(elf, {"trace": "CLOCK_MONOTONIC"})
    trace = next(c for c in comparisons if c.sysfs_name == "trace")
    assert trace.live_int is None
    assert trace.overridden is None


def test_compare_without_live_values_reports_defaults_only(elf):
    comparisons = uvc_module_params.compare(elf, None)
    assert all(c.live is None for c in comparisons)
    assert all(c.overridden is None for c in comparisons)
    assert {c.symbol for c in comparisons} == set(uvc_module_params.UVC_INT_PARAMETERS)


def test_nodrop_default_is_one_so_incomplete_frames_are_handed_up(elf):
    """Pinned because the scanner-011 reasoning leans on it.

    With ``nodrop = 1`` the driver is configured not to discard incomplete
    frames, so "frames arrived but were silently dropped" is not available as
    an explanation for a queue that produced nothing.
    """
    assert elf.read_u32("uvc_no_drop_param") == 1


# --------------------------------------------------------------------------
# sysfs reading and rendering
# --------------------------------------------------------------------------


def test_read_live_parameters_from_a_fake_sysfs_tree(tmp_path):
    for name, value in (("quirks", "4294967295"), ("nodrop", "1"), ("timeout", "5000")):
        (tmp_path / name).write_text(value + "\n", encoding="utf-8")
    live = uvc_module_params.read_live_parameters(tmp_path)
    assert live["quirks"] == "4294967295"
    assert live["nodrop"] == "1"
    assert live["trace"] is None  # absent file, not an exception


def test_render_names_the_sentinel_explicitly(elf):
    text = uvc_module_params.render(uvc_module_params.compare(elf, {"quirks": "4294967295"}))
    assert "no quirk was forced" in text
    assert "compiled default" in text


def test_render_flags_a_real_override(elf):
    text = uvc_module_params.render(uvc_module_params.compare(elf, {"quirks": "512"}))
    assert "OVERRIDDEN" in text
    assert "a mask WAS forced" in text


def test_cli_json_against_a_synthesized_module(tmp_path, capsys):
    module = tmp_path / "fake.ko"
    module.write_bytes(build_elf64(UVCVIDEO_6_14_DEFAULTS))
    sysfs = tmp_path / "parameters"
    sysfs.mkdir()
    (sysfs / "quirks").write_text("4294967295\n", encoding="utf-8")
    exit_code = uvc_module_params.main(["--module", str(module), "--sysfs", str(sysfs), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quirks_forced"] is False


def test_cli_no_live_skips_sysfs_entirely(tmp_path, capsys):
    module = tmp_path / "fake.ko"
    module.write_bytes(build_elf64(UVCVIDEO_6_14_DEFAULTS))
    exit_code = uvc_module_params.main(
        ["--module", str(module), "--sysfs", str(tmp_path / "missing"), "--no-live", "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quirks_forced"] is None
    assert all(entry["live"] is None for entry in payload["parameters"])


# --------------------------------------------------------------------------
# Optional cross-check against the host's real module
# --------------------------------------------------------------------------

_REAL_MODULE = Path("/tmp/uvc-investigate/uvcvideo.ko")


@pytest.mark.skipif(
    not _REAL_MODULE.exists(),
    reason="no decompressed uvcvideo.ko present; this cross-check is a bonus, not a gate",
)
def test_real_uvcvideo_module_carries_the_documented_defaults():
    elf = uvc_module_params.parse_elf64(_REAL_MODULE.read_bytes())
    assert elf.read_u32("uvc_quirks_param") == uvc_module_params.QUIRKS_UNSET_SENTINEL
    assert elf.read_u32("uvc_no_drop_param") == 1
    assert elf.read_u32("uvc_timeout_param") == 5000
