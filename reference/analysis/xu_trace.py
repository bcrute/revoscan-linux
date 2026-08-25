#!/usr/bin/env python3
"""Offline recovery of ``libusb_control_transfer`` argument tuples from a vendor
shared object, for reproducing the scanner-006 protocol-archaeology evidence.

This helper is deliberately *not* part of the ``scanner3d`` package: it adds no
runtime dependency and is never imported by the CLI. It accepts a local path to
an externally supplied ``lib3DCamera.so`` (the SDK is not vendored here) and
shells out to the ordinary read-only binutils ``nm``/``objdump`` only. It opens
no device node, issues no USB or SDK call, and performs no network access.

The pure core, :func:`extract_control_transfers`, parses AT&T-syntax x86-64
``objdump -d`` text and reconstructs the arguments passed to each
``libusb_control_transfer`` call. The System V AMD64 calling convention places
the arguments as::

    libusb_control_transfer(handle,         # rdi
                            bmRequestType,  # esi
                            bRequest,       # edx
                            wValue,         # ecx
                            wIndex,         # r8d
                            data,           # r9
                            wLength,        # stack arg 7
                            timeout)        # stack arg 8

What the parser recovers, and what it refuses to claim
------------------------------------------------------

The vendor code does not always load a constant straight into the argument
register. ``cs::libuvcReadValueExtension`` and ``cs::libuvcExcuteCmd`` stage a
*fixed* ``wValue`` through an ``%rbp``-relative stack local
(``movl $0xe00,-0x4(%rbp)`` / ``movl $0x700,-0x54(%rbp)``), while
``cs::libuvcSetExtension`` and ``cs::libuvcGetExtension`` compute a *genuinely
dynamic* ``wValue`` from a parameter (``(selector << 8) & 0xffff``). Reporting
both as "dynamic" understates the evidence; reporting both as constant would be
a fabrication. The parser therefore distinguishes them:

* **Stack slots** are resolved by a whole-function pre-pass. A slot is treated
  as a constant only if the entire disassembled body writes it exactly once and
  that write is an immediate. Any register-sourced write, or a second write,
  makes the slot unknown, so a parameter-derived slot can never be reported as
  a constant regardless of instruction order.
* **Registers** are tracked by a linear scan that is cleared at every call (the
  callee clobbers caller-saved registers) and at every jump target inside the
  function (incoming register state is ambiguous there). Stack-slot values
  survive both, which is what makes the fixed ``wValue`` recoverable.
* Any instruction that writes a register in a form the parser does not model
  (``shl``, ``and``, ``lea``, arithmetic, loads from memory other than an
  ``%rbp`` slot) sets that register to unknown rather than leaving a stale
  immediate behind.

Unknown fields are reported as ``None`` (printed ``dynamic``) so a computed
operand is never silently presented as a constant.

Additionally the parser tracks the provenance of the ``data`` pointer and any
store made into it before the call, which is how the four-byte depth-scale
payload is recovered: ``readBuffer``'s ``int`` argument is stored to the first
four bytes of the caller's buffer (``mov %edx,(%rax)`` at ``0x262dbd``) and the
SET_CUR that follows has ``wLength = 4``. That relationship is reported as a
:class:`PayloadStore`, not asserted from outside.

Caveats, stated so the output is not over-read: this is a linear, intra-function
scan, not a full dataflow analysis. It does not model writes through arbitrary
pointers into a tracked stack slot, ``%rsp``-relative frames, or values produced
by arithmetic. Those cases surface as ``None`` (unknown), never as a wrong
constant, but "unknown" here means "not recovered by this tool", not "proven
dynamic".
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Register canonicalisation
# --------------------------------------------------------------------------
# All width aliases of one architectural register share a slot, so a value
# written as %edx is visible to a later read of %rdx or %dx.
_REG_ALIASES: dict[str, tuple[str, int]] = {}
for _base in ("a", "b", "c", "d"):
    _REG_ALIASES[f"r{_base}x"] = (_base, 8)
    _REG_ALIASES[f"e{_base}x"] = (_base, 4)
    _REG_ALIASES[f"{_base}x"] = (_base, 2)
    _REG_ALIASES[f"{_base}l"] = (_base, 1)
for _base in ("si", "di", "bp", "sp"):
    _REG_ALIASES[f"r{_base}"] = (_base, 8)
    _REG_ALIASES[f"e{_base}"] = (_base, 4)
    _REG_ALIASES[_base] = (_base, 2)
    _REG_ALIASES[f"{_base}l"] = (_base, 1)
for _n in range(8, 16):
    _REG_ALIASES[f"r{_n}"] = (f"r{_n}", 8)
    _REG_ALIASES[f"r{_n}d"] = (f"r{_n}", 4)
    _REG_ALIASES[f"r{_n}w"] = (f"r{_n}", 2)
    _REG_ALIASES[f"r{_n}b"] = (f"r{_n}", 1)

# Integer argument registers in System V AMD64 order, for labelling the
# prologue spills that name a function's parameters.
_ARG_ORDER: tuple[str, ...] = ("di", "si", "d", "c", "r8", "r9")


def canonical_register(token: str) -> tuple[str, int] | None:
    """Map an objdump register token (with or without ``%``) to ``(name, width)``."""
    return _REG_ALIASES.get(token.lstrip("%"))


# --------------------------------------------------------------------------
# Line/operand parsing
# --------------------------------------------------------------------------
_LINE = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )*\s*([a-z][a-z0-9.]*)\s*(.*?)\s*$")
_JUMP_TARGET = re.compile(r"^j[a-z]+$")
_IMM = re.compile(r"^\$0x([0-9a-fA-F]+)$")
_REG = re.compile(r"^%([a-z0-9]+)$")
_RBP_SLOT = re.compile(r"^(-?0x[0-9a-fA-F]+)\(%rbp\)$")
_DEREF = re.compile(r"^(?:(-?0x[0-9a-fA-F]+))?\(%([a-z0-9]+)\)$")
_CALL_TARGET = re.compile(r"<([^>]+)>")


def _split_operands(text: str) -> list[str]:
    """Split an AT&T operand list on top-level commas (SIB forms contain commas)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    # objdump appends a "# eba0c0 <sym>" comment to rip-relative operands.
    return [p.split("#", 1)[0].strip() for p in parts if p.split("#", 1)[0].strip()]


# --------------------------------------------------------------------------
# Recovered values
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RegValue:
    """What a register is known to hold at a program point.

    ``const`` is a recovered immediate (``None`` when not recovered). ``origin``
    names the ``%rbp``-relative stack slot the value was loaded from, which is
    how two registers are recognised as pointing at the same buffer.
    """

    const: int | None = None
    origin: str | None = None


@dataclass(frozen=True)
class PayloadStore:
    """A store into the buffer later passed as the ``data`` argument."""

    address: str
    offset: int
    width: int
    const: int | None
    source: str | None  # e.g. "arg3" when the stored value is a parameter

    def describe(self) -> str:
        what = f"{self.const:#x}" if self.const is not None else (self.source or "runtime value")
        return f"payload[{self.offset}:{self.offset + self.width}] = {what} (@{self.address})"


@dataclass
class ControlTransfer:
    """One recovered ``libusb_control_transfer`` call site.

    Integer fields are the recovered constant, or ``None`` when the operand was
    not recovered as a constant by the rules documented in the module docstring.
    """

    address: str
    bmRequestType: int | None = None
    bRequest: int | None = None
    wValue: int | None = None
    wIndex: int | None = None
    wLength: int | None = None
    timeout: int | None = None
    wValue_source: str | None = None  # "immediate" | "stack local <slot>" | None
    data_origin: str | None = None
    payload: list[PayloadStore] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "address": self.address,
            "bmRequestType": self.bmRequestType,
            "bRequest": self.bRequest,
            "wValue": self.wValue,
            "wIndex": self.wIndex,
            "wLength": self.wLength,
            "timeout": self.timeout,
            "wValue_source": self.wValue_source,
            "payload": [p.describe() for p in self.payload],
        }


# --------------------------------------------------------------------------
# Whole-function pre-passes
# --------------------------------------------------------------------------
_POISON = object()


def _instructions(disasm: str) -> list[tuple[str, str, list[str]]]:
    out: list[tuple[str, str, list[str]]] = []
    for raw in disasm.splitlines():
        match = _LINE.match(raw)
        if match:
            out.append((match.group(1), match.group(2), _split_operands(match.group(3))))
    return out


def _constant_slots(insns: list[tuple[str, str, list[str]]]) -> dict[str, int]:
    """Stack slots written exactly once, by an immediate, in the whole function.

    Order-independent and therefore branch-safe: a slot fed from a parameter is
    excluded no matter where in the body it is read.
    """
    seen: dict[str, object] = {}
    for _addr, mnem, ops in insns:
        if not mnem.startswith("mov") or len(ops) != 2:
            continue
        slot = _RBP_SLOT.match(ops[1])
        if slot is None:
            continue
        key = slot.group(1)
        imm = _IMM.match(ops[0])
        value: object = int(imm.group(1), 16) if imm else _POISON
        seen[key] = _POISON if key in seen else value
    return {k: v for k, v in seen.items() if isinstance(v, int)}


def _parameter_slots(insns: list[tuple[str, str, list[str]]]) -> dict[str, str]:
    """Label the prologue spills ``mov %edi,-0x14(%rbp)`` as ``arg1`` … ``arg6``."""
    labels: dict[str, str] = {}
    for _addr, mnem, ops in insns:
        if not mnem.startswith("mov") or len(ops) != 2:
            continue
        reg = _REG.match(ops[0])
        slot = _RBP_SLOT.match(ops[1])
        if reg is None or slot is None:
            continue
        canon = canonical_register(reg.group(1))
        if canon is None or canon[0] not in _ARG_ORDER:
            continue
        labels.setdefault(slot.group(1), f"arg{_ARG_ORDER.index(canon[0]) + 1}")
    return labels


def _jump_targets(insns: list[tuple[str, str, list[str]]]) -> set[str]:
    targets: set[str] = set()
    for _addr, mnem, ops in insns:
        if _JUMP_TARGET.match(mnem) and ops:
            token = ops[0].split()[0]
            if re.fullmatch(r"[0-9a-f]+", token):
                targets.add(token)
    return targets


# --------------------------------------------------------------------------
# Linear scan
# --------------------------------------------------------------------------
@dataclass
class _State:
    regs: dict[str, RegValue] = field(default_factory=dict)
    pushes: list[int | None] = field(default_factory=list)
    stores: list[tuple[str, int, PayloadStore]] = field(default_factory=list)

    def clear(self) -> None:
        """Forget register/stack-argument state (call boundary or jump target)."""
        self.regs.clear()
        self.pushes.clear()
        self.stores.clear()


def extract_control_transfers(disasm: str) -> list[ControlTransfer]:
    """Recover the arguments of every ``libusb_control_transfer`` call.

    ``disasm`` is the text of ``objdump -d`` (AT&T syntax) for a single
    function. See the module docstring for exactly what is and is not claimed.
    """
    insns = _instructions(disasm)
    slots = _constant_slots(insns)
    params = _parameter_slots(insns)
    targets = _jump_targets(insns)

    state = _State()
    found: list[ControlTransfer] = []

    for addr, mnem, ops in insns:
        if addr in targets:
            state.clear()

        if mnem == "call":
            target = _CALL_TARGET.search(" ".join(ops))
            if target and "libusb_control_transfer" in target.group(1):
                found.append(_snapshot(addr, state))
            state.clear()
            continue

        if _JUMP_TARGET.match(mnem) or mnem in {"cmp", "test", "ret", "leave", "nop"}:
            continue

        if mnem == "push":
            if ops:
                imm = _IMM.match(ops[0])
                if imm:
                    state.pushes.append(int(imm.group(1), 16))
                    continue
                reg = _REG.match(ops[0])
                canon = canonical_register(reg.group(1)) if reg else None
                value = state.regs.get(canon[0]) if canon else None
                state.pushes.append(value.const if value else None)
            continue

        if len(ops) != 2:
            continue
        src, dst = ops

        # Store into a dereferenced pointer: candidate payload write.
        deref = _DEREF.match(dst)
        if deref is not None:
            _record_store(state, addr, mnem, src, deref, slots, params)
            continue

        dst_reg = _REG.match(dst)
        if dst_reg is None:
            continue
        canon = canonical_register(dst_reg.group(1))
        if canon is None:
            continue
        state.regs[canon[0]] = _evaluate(mnem, src, state, slots, params)

    return found


def _evaluate(
    mnem: str,
    src: str,
    state: _State,
    slots: dict[str, int],
    params: dict[str, str],
) -> RegValue:
    """Value a modelled ``mov``-family instruction puts in its destination."""
    if not mnem.startswith("mov"):
        # shl/and/add/lea/... — not modelled, so the destination becomes unknown.
        return RegValue()

    imm = _IMM.match(src)
    if imm is not None:
        return RegValue(const=int(imm.group(1), 16))

    slot = _RBP_SLOT.match(src)
    if slot is not None:
        key = slot.group(1)
        return RegValue(const=slots.get(key), origin=key)

    reg = _REG.match(src)
    if reg is not None:
        canon = canonical_register(reg.group(1))
        if canon is None:
            return RegValue()
        value = state.regs.get(canon[0], RegValue())
        const = value.const
        if const is not None and mnem.startswith("movz"):
            const &= (1 << (canon[1] * 8)) - 1
        return RegValue(const=const, origin=value.origin)

    # Any other memory form (rip-relative, SIB, non-rbp base) is unknown.
    return RegValue()


_WIDTH_BY_SUFFIX = {"movb": 1, "movw": 2, "movl": 4, "movq": 8}


def _record_store(
    state: _State,
    addr: str,
    mnem: str,
    src: str,
    deref: re.Match[str],
    slots: dict[str, int],
    params: dict[str, str],
) -> None:
    base = canonical_register(deref.group(2))
    if base is None or not mnem.startswith("mov"):
        return
    pointer = state.regs.get(base[0], RegValue())
    if pointer.origin is None:
        return
    offset = int(deref.group(1), 16) if deref.group(1) else 0

    imm = _IMM.match(src)
    if imm is not None:
        width = _WIDTH_BY_SUFFIX.get(mnem, 4)
        store = PayloadStore(addr, offset, width, int(imm.group(1), 16), None)
        state.stores.append((pointer.origin, offset, store))
        return

    reg = _REG.match(src)
    if reg is None:
        return
    canon = canonical_register(reg.group(1))
    if canon is None:
        return
    value = state.regs.get(canon[0], RegValue())
    source = params.get(value.origin) if value.origin else None
    store = PayloadStore(addr, offset, canon[1], value.const, source)
    state.stores.append((pointer.origin, offset, store))


def _snapshot(addr: str, state: _State) -> ControlTransfer:
    ct = ControlTransfer(address=addr)
    ct.bmRequestType = _const(state, "si")
    ct.bRequest = _const(state, "d")
    ct.wValue = _const(state, "c")
    ct.wIndex = _const(state, "r8")
    wvalue = state.regs.get("c", RegValue())
    if wvalue.const is not None:
        ct.wValue_source = f"stack local {wvalue.origin}" if wvalue.origin else "immediate"
    data = state.regs.get("r9", RegValue())
    ct.data_origin = data.origin
    if data.origin is not None:
        ct.payload = [store for origin, _off, store in state.stores if origin == data.origin]
    # Stack args 7 (wLength) and 8 (timeout) are pushed right-to-left, so the
    # push nearest the call is wLength and the one before it is timeout.
    if state.pushes:
        ct.wLength = state.pushes[-1]
    if len(state.pushes) >= 2:
        ct.timeout = state.pushes[-2]
    return ct


def _const(state: _State, reg: str) -> int | None:
    return state.regs.get(reg, RegValue()).const


# --------------------------------------------------------------------------
# Read-only binary access
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        raise FileNotFoundError(f"required read-only tool not found on PATH: {name}")
    return tool


def _symbol_span(so: Path, needle: str) -> tuple[int, int, str] | None:
    """Return (addr, size, name) of the first defined function whose mangled
    symbol contains ``needle``, via ``nm -S`` (read-only)."""
    out = subprocess.run(
        [_require_tool("nm"), "-S", "--defined-only", str(so)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        addr, size, _type, name = parts
        if needle in name:
            return int(addr, 16), int(size, 16), name
    return None


def disassemble(so: Path, needle: str) -> str:
    """Read-only disassembly of the function matching ``needle`` via objdump."""
    span = _symbol_span(so, needle)
    if span is None:
        raise LookupError(f"no defined function symbol contains {needle!r}")
    addr, size, _name = span
    return subprocess.run(
        [
            _require_tool("objdump"),
            "-d",
            f"--start-address={addr:#x}",
            f"--stop-address={addr + size:#x}",
            "--no-show-raw-insn",
            str(so),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


# The primitive helpers whose control-transfer shapes are the scanner-006
# evidence. Keyed by a mangled-symbol substring stable across the pinned build.
PRIMITIVES: dict[str, str] = {
    "libuvcSetExtension": "18libuvcSetExtension",
    "libuvcGetExtension": "18libuvcGetExtension",
    "libuvcReadValueExtension": "24libuvcReadValueExtension",
    "libuvcExcuteCmd": "15libuvcExcuteCmd",
    "libuvcReadFile": "14libuvcReadFileE",
    "uvc_query_stream_ctrl": "uvc_query_stream_ctrl",
}


def _fmt(value: int | None) -> str:
    return "dynamic" if value is None else f"{value:#x}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so", required=True, type=Path, help="path to lib3DCamera.so")
    parser.add_argument(
        "--expect-sha256",
        default=None,
        help="fail unless the file's SHA-256 matches this pinned hash",
    )
    args = parser.parse_args(argv)

    so: Path = args.so
    if not so.is_file():
        print(f"error: not a file: {so}", file=sys.stderr)
        return 2

    digest = sha256_file(so)
    print(f"file: {so}")
    print(f"sha256: {digest}")
    if args.expect_sha256 and digest != args.expect_sha256:
        print(
            f"error: sha256 mismatch (expected {args.expect_sha256})",
            file=sys.stderr,
        )
        return 3

    for label, needle in PRIMITIVES.items():
        print(f"\n## {label}")
        try:
            text = disassemble(so, needle)
        except LookupError as exc:
            print(f"  (not found: {exc})")
            continue
        transfers = extract_control_transfers(text)
        if not transfers:
            print("  (no libusb_control_transfer call sites)")
        for ct in transfers:
            source = f" [{ct.wValue_source}]" if ct.wValue_source else ""
            print(
                f"  @{ct.address}: bmRequestType={_fmt(ct.bmRequestType)} "
                f"bRequest={_fmt(ct.bRequest)} wValue={_fmt(ct.wValue)}{source} "
                f"wIndex={_fmt(ct.wIndex)} wLength={_fmt(ct.wLength)} "
                f"timeout={_fmt(ct.timeout)}"
            )
            for store in ct.payload:
                print(f"      {store.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
