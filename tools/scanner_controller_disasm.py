"""Offline ARM inspection of the already captured, hash-pinned MINI controller.

Requires Capstone/pyelftools in the separate analysis environment. Never loads
or executes the captured program and has no device access or firmware writer.
Literal references are syntactic evidence, not proof a runtime branch executed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import struct
from pathlib import Path
from typing import Any

from capstone import CS_ARCH_ARM, CS_MODE_ARM, Cs
from capstone.arm import ARM_OP_IMM, ARM_OP_MEM, ARM_REG_PC
from elftools.elf.elffile import ELFFile

CONTROLLER_SHA256 = "decdc3ad4e34449cefee8543c790866cfc87dd9279a873c3abab124b9754b0f1"


def inspect(path: Path, pattern: str | None, start: int | None, end: int | None) -> None:
    if path.stat().st_size != 166412:
        raise ValueError("Only the known 166412-byte controller artifact")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != CONTROLLER_SHA256:
        raise ValueError("Controller SHA256 differs from the captured MINI binary")
    elf = ELFFile(io.BytesIO(blob))
    code = elf.get_section_by_name(".text")
    rodata = elf.get_section_by_name(".rodata")
    code_base, data_base = code["sh_addr"], rodata["sh_addr"]
    code_bytes, data_bytes = code.data(), rodata.data()
    engine = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    engine.detail = True
    engine.skipdata = True
    imports = {}
    symbols = elf.get_section_by_name(".dynsym")
    plt = elf.get_section_by_name(".plt")["sh_addr"]
    for index, relocation in enumerate(elf.get_section_by_name(".rel.plt").iter_relocations()):
        imports[plt + 20 + 12 * index] = symbols.get_symbol(relocation["r_info_sym"]).name

    def literal(instruction: Any) -> tuple[int, int] | None:
        if instruction.id == 0 or not instruction.mnemonic.startswith("ldr"):
            return None
        for operand in instruction.operands:
            if operand.type == ARM_OP_MEM and operand.mem.base == ARM_REG_PC:
                address = instruction.address + 8 + operand.mem.disp
                offset = address - code_base
                if 0 <= offset <= len(code_bytes) - 4:
                    value = struct.unpack_from("<I", code_bytes, offset)[0]
                    return address, value
        return None

    def string_at(address: int) -> str | None:
        offset = address - data_base
        if not 0 <= offset < len(data_bytes):
            return None
        return data_bytes[offset:].split(b"\0", 1)[0].decode("ascii", errors="replace")

    if pattern is not None:
        matcher = re.compile(pattern, re.IGNORECASE)
        for instruction in engine.disasm(code_bytes, code_base):
            loaded = literal(instruction)
            if loaded is None:
                continue
            string = string_at(loaded[1])
            if string is not None and matcher.search(string):
                print(
                    f"{instruction.address:#010x} literal={loaded[0]:#x} "
                    f"value={loaded[1]:#x} {string!r}"
                )
        return
    if start is None or end is None or not code_base <= start < end <= code_base + len(code_bytes):
        raise ValueError("Explicit disassembly range within .text required")
    if end - start > 16384 or start % 4 or end % 4:
        raise ValueError("Aligned range of at most 16 KiB required")
    for instruction in engine.disasm(code_bytes[start - code_base : end - code_base], start):
        annotation = ""
        loaded = literal(instruction)
        if loaded:
            annotation = f" ; literal {loaded[0]:#x} = {loaded[1]:#x}"
            string = string_at(loaded[1])
            if string is not None:
                annotation += f" {string!r}"
        if instruction.id and instruction.mnemonic in ("bl", "b"):
            target = instruction.operands[0]
            if target.type == ARM_OP_IMM and target.imm in imports:
                annotation += " ; " + imports[target.imm] + "@plt"
        print(
            f"{instruction.address:08x}  {instruction.bytes.hex():8s}  "
            f"{instruction.mnemonic:8s} {instruction.op_str}{annotation}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--find")
    parser.add_argument("--start", type=lambda value: int(value, 0))
    parser.add_argument("--end", type=lambda value: int(value, 0))
    args = parser.parse_args()
    inspect(args.binary, args.find, args.start, args.end)
