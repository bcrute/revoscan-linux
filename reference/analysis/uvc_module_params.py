#!/usr/bin/env python3
"""Compare the live ``uvcvideo`` module parameters against the module's own defaults.

A parameter read out of ``/sys/module/uvcvideo/parameters/`` looks like a
deliberate setting even when it is nothing of the kind. ``quirks`` is the trap:
it reads ``4294967295``, which invites the conclusion that every UVC quirk
workaround has been forced on for this host. It has not. ``4294967295`` is
``(unsigned)-1``, the sentinel the driver compiles in to mean *no forced
quirks*, and reading the compiled ``.ko`` proves it: the initialiser stored in
the module's own ``.data`` section is the same ``ffffffff``.

That distinction matters for the scanner-011 RGB analysis, so this helper makes
it reproducible instead of a one-off shell result. It answers one question:

    for each uvcvideo module parameter, does the live value differ from the
    value the on-disk module was compiled with?

Only a difference is evidence of a deliberate override.

Offline and read-only by construction. It reads an ELF object file and, when
asked, files under ``/sys/module/<name>/parameters/`` -- which are *module*
attributes, not device nodes. It opens no device, issues no USB transfer, runs
no subprocess and does no network I/O. Decompressing a ``.ko.zst`` is left to
the caller (``zstd -d``), so this helper never shells out.

Invocation:

    zstd -d /lib/modules/$(uname -r)/kernel/drivers/media/usb/uvc/uvcvideo.ko.zst \
        -o /tmp/uvcvideo.ko
    python reference/analysis/uvc_module_params.py --module /tmp/uvcvideo.ko

Add ``--no-live`` to print the compiled defaults alone.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Parameters this helper knows how to read out of ``.data``/``.bss`` by symbol
# name, mapped to the sysfs attribute that exposes them. Every one is a plain
# 32-bit integer; ``clock`` is deliberately absent because its sysfs form is a
# string ("CLOCK_MONOTONIC") produced by a custom getter, so a raw integer
# comparison would be meaningless.
UVC_INT_PARAMETERS: dict[str, str] = {
    "uvc_quirks_param": "quirks",
    "uvc_no_drop_param": "nodrop",
    "uvc_timeout_param": "timeout",
    "uvc_hw_timestamps_param": "hwtimestamps",
    "uvc_dbg_param": "trace",
}

# ``quirks`` is an ``unsigned int`` initialised to ``-1``. The driver treats
# that value as "the module parameter was not supplied"; any other value is a
# forced override.
QUIRKS_UNSET_SENTINEL = 0xFFFFFFFF

_SHT_NOBITS = 8
_SHT_SYMTAB = 2


class ElfError(ValueError):
    """The file is not an ELF64 little-endian object this reader understands."""


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    sh_type: int
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    section_index: int
    value: int
    size: int


@dataclass(frozen=True, slots=True)
class Elf64:
    """The minimum of an ELF64 object needed to read an initialised variable."""

    sections: tuple[Section, ...]
    symbols: tuple[Symbol, ...]
    data: bytes

    def symbol(self, name: str) -> Symbol | None:
        for sym in self.symbols:
            if sym.name == name:
                return sym
        return None

    def read_u32(self, name: str) -> int | None:
        """Read the compiled initialiser of a 4-byte integer symbol.

        Returns ``0`` for a symbol in a ``NOBITS`` section (``.bss``), whose
        contents are not stored in the file but are zeroed at load. Returns
        ``None`` when the symbol is absent, the wrong size, or its section is
        out of range.
        """
        sym = self.symbol(name)
        if sym is None or sym.size != 4:
            return None
        if not 0 <= sym.section_index < len(self.sections):
            return None
        section = self.sections[sym.section_index]
        if section.sh_type == _SHT_NOBITS:
            return 0
        start = section.offset + sym.value
        if start + 4 > len(self.data):
            return None
        return int(struct.unpack_from("<I", self.data, start)[0])


def _cstr(blob: bytes, offset: int) -> str:
    if offset >= len(blob):
        return ""
    end = blob.find(b"\x00", offset)
    if end < 0:
        end = len(blob)
    return blob[offset:end].decode("utf-8", errors="replace")


def parse_elf64(data: bytes) -> Elf64:
    """Parse the section and symbol tables of a little-endian ELF64 object."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ElfError("not an ELF file")
    if data[4] != 2:
        raise ElfError("not ELF64")
    if data[5] != 1:
        raise ElfError("not little-endian")

    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    if e_shoff == 0 or e_shnum == 0:
        raise ElfError("no section headers")
    if e_shoff + e_shnum * e_shentsize > len(data):
        raise ElfError("section headers out of range")

    raw: list[tuple[int, int, int, int, int, int]] = []
    for index in range(e_shnum):
        base = e_shoff + index * e_shentsize
        sh_name, sh_type = struct.unpack_from("<II", data, base)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, base + 0x18)
        (sh_link,) = struct.unpack_from("<I", data, base + 0x28)
        (sh_entsize,) = struct.unpack_from("<Q", data, base + 0x38)
        raw.append((sh_name, sh_type, sh_offset, sh_size, sh_link, sh_entsize))

    if e_shstrndx >= e_shnum:
        raise ElfError("section-name string table index out of range")
    shstr_offset, shstr_size = raw[e_shstrndx][2], raw[e_shstrndx][3]
    shstrtab = data[shstr_offset : shstr_offset + shstr_size]

    sections = tuple(
        Section(name=_cstr(shstrtab, entry[0]), sh_type=entry[1], offset=entry[2], size=entry[3])
        for entry in raw
    )

    symbols: list[Symbol] = []
    for entry in raw:
        if entry[1] != _SHT_SYMTAB or entry[5] == 0:
            continue
        strtab_index = entry[4]
        if strtab_index >= len(raw):
            continue
        str_offset, str_size = raw[strtab_index][2], raw[strtab_index][3]
        strtab = data[str_offset : str_offset + str_size]
        count = entry[3] // entry[5]
        for position in range(count):
            base = entry[2] + position * entry[5]
            if base + 24 > len(data):
                break
            (st_name,) = struct.unpack_from("<I", data, base)
            (st_shndx,) = struct.unpack_from("<H", data, base + 6)
            st_value, st_size = struct.unpack_from("<QQ", data, base + 8)
            name = _cstr(strtab, st_name)
            if name:
                symbols.append(
                    Symbol(name=name, section_index=st_shndx, value=st_value, size=st_size)
                )

    return Elf64(sections=sections, symbols=tuple(symbols), data=data)


@dataclass(frozen=True, slots=True)
class ParameterComparison:
    """One module parameter's compiled default beside its live value."""

    symbol: str
    sysfs_name: str
    compiled: int | None
    live: str | None

    @property
    def live_int(self) -> int | None:
        if self.live is None:
            return None
        try:
            return int(self.live.strip())
        except ValueError:
            return None

    @property
    def overridden(self) -> bool | None:
        """True only when the live value provably differs from the default.

        ``None`` means the comparison could not be made -- an absent symbol, an
        unreadable sysfs attribute, or a non-integer live value -- and must not
        be reported as either "default" or "overridden".
        """
        if self.compiled is None or self.live_int is None:
            return None
        return self.compiled != self.live_int

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sysfs_name": self.sysfs_name,
            "compiled": self.compiled,
            "live": self.live,
            "live_int": self.live_int,
            "overridden": self.overridden,
        }


def compiled_defaults(elf: Elf64) -> dict[str, int | None]:
    return {symbol: elf.read_u32(symbol) for symbol in UVC_INT_PARAMETERS}


def read_live_parameters(root: Path) -> dict[str, str | None]:
    """Read each parameter file under a ``/sys/module/<name>/parameters`` path."""
    live: dict[str, str | None] = {}
    for sysfs_name in UVC_INT_PARAMETERS.values():
        path = root / sysfs_name
        try:
            live[sysfs_name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            live[sysfs_name] = None
    return live


def compare(
    elf: Elf64,
    live: dict[str, str | None] | None = None,
) -> list[ParameterComparison]:
    defaults = compiled_defaults(elf)
    return [
        ParameterComparison(
            symbol=symbol,
            sysfs_name=sysfs_name,
            compiled=defaults[symbol],
            live=None if live is None else live.get(sysfs_name),
        )
        for symbol, sysfs_name in UVC_INT_PARAMETERS.items()
    ]


def quirks_are_forced(comparisons: Sequence[ParameterComparison]) -> bool | None:
    """Whether a quirk mask was actually forced by module parameter.

    The answer is ``False`` for the sentinel ``0xffffffff`` however alarming it
    looks, and ``None`` when the live value could not be read.
    """
    for comparison in comparisons:
        if comparison.sysfs_name != "quirks":
            continue
        value = comparison.live_int
        if value is None:
            return None
        return value != QUIRKS_UNSET_SENTINEL
    return None


def render(comparisons: Sequence[ParameterComparison]) -> str:
    lines = [f"{'parameter':16s}{'compiled':>12s}{'live':>20s}  verdict"]
    for comparison in comparisons:
        compiled = "absent" if comparison.compiled is None else str(comparison.compiled)
        live = "unreadable" if comparison.live is None else comparison.live
        if comparison.overridden is None:
            verdict = "not comparable"
        elif comparison.overridden:
            verdict = "OVERRIDDEN"
        else:
            verdict = "compiled default"
        lines.append(f"{comparison.sysfs_name:16s}{compiled:>12s}{live:>20s}  {verdict}")
    forced = quirks_are_forced(comparisons)
    lines.append("")
    if forced is None:
        lines.append("quirks: live value unreadable; forced-quirks question unanswered")
    elif forced:
        lines.append("quirks: a mask WAS forced by module parameter")
    else:
        lines.append(
            f"quirks: 0x{QUIRKS_UNSET_SENTINEL:08x} is the driver's "
            "'not supplied' sentinel -- no quirk was forced"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--module",
        type=Path,
        required=True,
        help="decompressed uvcvideo.ko (zstd -d it first; this tool never shells out)",
    )
    parser.add_argument(
        "--sysfs",
        type=Path,
        default=Path("/sys/module/uvcvideo/parameters"),
        help="module-parameter directory to read live values from",
    )
    parser.add_argument("--no-live", action="store_true", help="print compiled defaults only")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    elf = parse_elf64(args.module.read_bytes())
    live = None if args.no_live else read_live_parameters(args.sysfs)
    comparisons = compare(elf, live)

    if args.json:
        json.dump(
            {
                "parameters": [c.as_dict() for c in comparisons],
                "quirks_forced": quirks_are_forced(comparisons),
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render(comparisons) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
