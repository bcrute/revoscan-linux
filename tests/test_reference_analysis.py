"""Offline tests for the scanner-006 ``reference/analysis/xu_trace`` helper.

These exercise the pure disassembly parser against small committed excerpts of
real ``objdump -d`` output (source-derived evidence, not the vendor binary).
They run with no SDK, no binary, no device and no network, so they are part of
the default gate.

The excerpts below are verbatim from the pinned build (`lib3DCamera.so`,
SHA-256 ``49e0865c…cdae3c4``, build-id ``8d81f146…``); each is reproducible with

    objdump -d --start-address=<addr> --stop-address=<addr+size> \
        --no-show-raw-insn lib3DCamera.so
"""

import importlib.util
import struct
import sys
from pathlib import Path

_HELPER = Path(__file__).resolve().parent.parent / "reference" / "analysis" / "xu_trace.py"
_spec = importlib.util.spec_from_file_location("xu_trace", _HELPER)
assert _spec is not None
xu_trace = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules.setdefault("xu_trace", xu_trace)
_spec.loader.exec_module(xu_trace)


# Real objdump -d excerpt of cs::libuvcSetExtension @0x262eae, both branches:
# the vendor-request (0x41/0xFF, wIndex 0x0300) and UVC class SET_CUR
# (0x21/0x01, wIndex 0x0400) forms. Here wValue is genuinely dynamic: it is
# computed as (selector << 8) & 0xffff from parameter 3 and staged through
# -0x4(%rbp), so the slot is register-written and must stay unknown.
SET_EXTENSION_EXCERPT = """\
  262ebd:\tmov    %edx,-0x18(%rbp)
  262ecf:\tmov    -0x18(%rbp),%eax
  262ed2:\tshl    $0x8,%eax
  262ed5:\tand    $0xffff,%eax
  262eda:\tmov    %eax,-0x4(%rbp)
  262f0c:\tmov    -0x2c(%rbp),%ecx
  262f0f:\tmov    -0x4(%rbp),%eax
  262f12:\tmovzwl %ax,%edx
  262f15:\tmov    -0x28(%rbp),%rsi
  262f19:\tmov    -0x20(%rbp),%rax
  262f1d:\tpush   %rcx
  262f1e:\tpush   $0x3c
  262f20:\tmov    %rsi,%r9
  262f23:\tmov    $0x300,%r8d
  262f29:\tmov    %edx,%ecx
  262f2b:\tmov    $0xff,%edx
  262f30:\tmov    $0x41,%esi
  262f35:\tmov    %rax,%rdi
  262f38:\tcall   1480b0 <libusb_control_transfer@plt>
  262f68:\tmov    -0x2c(%rbp),%ecx
  262f6b:\tmov    -0x4(%rbp),%eax
  262f6e:\tmovzwl %ax,%edx
  262f71:\tmov    -0x28(%rbp),%rsi
  262f75:\tmov    -0x20(%rbp),%rax
  262f79:\tpush   %rcx
  262f7a:\tpush   $0x3c
  262f7c:\tmov    %rsi,%r9
  262f7f:\tmov    $0x400,%r8d
  262f85:\tmov    %edx,%ecx
  262f87:\tmov    $0x1,%edx
  262f8c:\tmov    $0x21,%esi
  262f91:\tmov    %rax,%rdi
  262f94:\tcall   1480b0 <libusb_control_transfer@plt>
"""

# Real objdump -d excerpt of cs::libuvcReadValueExtension @0x262d92 (size
# 0x11c), the whole function. Unlike SetExtension, wValue here is a FIXED
# 0x0e00 staged through the stack local -0x4(%rbp), and the SET_CUR payload is
# the caller's int argument (arg3) written to the first four bytes of the
# caller-supplied buffer, with wLength = 4. The 60-byte GET_CUR that follows
# reuses the same fixed selector.
READ_VALUE_EXTENSION_EXCERPT = """\
  262d9a:\tmov    %edi,-0x14(%rbp)
  262d9d:\tmov    %rsi,-0x20(%rbp)
  262da1:\tmov    %edx,-0x18(%rbp)
  262da4:\tmov    %rcx,-0x28(%rbp)
  262da8:\tmovl   $0x0,-0x8(%rbp)
  262daf:\tmovl   $0xe00,-0x4(%rbp)
  262db6:\tmov    -0x18(%rbp),%edx
  262db9:\tmov    -0x28(%rbp),%rax
  262dbd:\tmov    %edx,(%rax)
  262dbf:\tmov    -0x4(%rbp),%eax
  262dc2:\tmovzwl %ax,%edx
  262dc5:\tmov    -0x28(%rbp),%rcx
  262dc9:\tmov    -0x20(%rbp),%rax
  262dcd:\tpush   $0x3e8
  262dd2:\tpush   $0x4
  262dd4:\tmov    %rcx,%r9
  262dd7:\tmov    $0x400,%r8d
  262ddd:\tmov    %edx,%ecx
  262ddf:\tmov    $0x1,%edx
  262de4:\tmov    $0x21,%esi
  262de9:\tmov    %rax,%rdi
  262dec:\tcall   1480b0 <libusb_control_transfer@plt>
  262df1:\tadd    $0x10,%rsp
  262df5:\tmov    %eax,-0x8(%rbp)
  262df8:\tmov    -0x8(%rbp),%eax
  262dfb:\ttest   %eax,%eax
  262dfd:\tjns    262e27 <_ZN2cs24libuvcReadValueExtensionE+0x95>
  262e1b:\tmovl   $0x1,-0x8(%rbp)
  262e22:\tjmp    262ea9 <_ZN2cs24libuvcReadValueExtensionE+0x117>
  262e27:\tmov    -0x28(%rbp),%rax
  262e2b:\tmov    $0x3c,%edx
  262e30:\tmov    $0x0,%esi
  262e35:\tmov    %rax,%rdi
  262e38:\tcall   148a70 <memset@plt>
  262e3d:\tmov    -0x4(%rbp),%eax
  262e40:\tmovzwl %ax,%edx
  262e43:\tmov    -0x28(%rbp),%rcx
  262e47:\tmov    -0x20(%rbp),%rax
  262e4b:\tpush   $0x3e8
  262e50:\tpush   $0x3c
  262e52:\tmov    %rcx,%r9
  262e55:\tmov    $0x400,%r8d
  262e5b:\tmov    %edx,%ecx
  262e5d:\tmov    $0x81,%edx
  262e62:\tmov    $0xa1,%esi
  262e67:\tmov    %rax,%rdi
  262e6a:\tcall   1480b0 <libusb_control_transfer@plt>
  262ea9:\tmov    -0x8(%rbp),%eax
"""

# Real objdump -d excerpt of cs::libuvcExcuteCmd @0x262c25: the fixed wValue
# 0x0700 is staged through -0x54(%rbp) at the top of the function and read in
# both CameraType branches, the second of which is a jump target.
EXCUTE_CMD_EXCERPT = """\
  262c38:\tmov    %ecx,-0x68(%rbp)
  262c4a:\tmovl   $0x700,-0x54(%rbp)
  262cd1:\tmovl   $0x0,-0x58(%rbp)
  262cd8:\tmov    -0x64(%rbp),%eax
  262cdb:\tcmp    $0x9,%eax
  262cde:\tjg     262d1a <_ZN2cs15libuvcExcuteCmdE+0xf5>
  262ce0:\tmov    -0x68(%rbp),%ecx
  262ce3:\tmov    -0x54(%rbp),%eax
  262ce6:\tmovzwl %ax,%edx
  262ce9:\tlea    -0x50(%rbp),%rsi
  262ced:\tmov    -0x70(%rbp),%rax
  262cf1:\tpush   %rcx
  262cf2:\tpush   $0x3c
  262cf4:\tmov    %rsi,%r9
  262cf7:\tmov    $0x300,%r8d
  262cfd:\tmov    %edx,%ecx
  262cff:\tmov    $0xff,%edx
  262d04:\tmov    $0x41,%esi
  262d09:\tmov    %rax,%rdi
  262d0c:\tcall   1480b0 <libusb_control_transfer@plt>
  262d11:\tadd    $0x10,%rsp
  262d15:\tmov    %eax,-0x58(%rbp)
  262d18:\tjmp    262d52 <_ZN2cs15libuvcExcuteCmdE+0x12d>
  262d1a:\tmov    -0x68(%rbp),%ecx
  262d1d:\tmov    -0x54(%rbp),%eax
  262d20:\tmovzwl %ax,%edx
  262d23:\tlea    -0x50(%rbp),%rsi
  262d27:\tmov    -0x70(%rbp),%rax
  262d2b:\tpush   %rcx
  262d2c:\tpush   $0x3c
  262d2e:\tmov    %rsi,%r9
  262d31:\tmov    $0x400,%r8d
  262d37:\tmov    %edx,%ecx
  262d39:\tmov    $0x1,%edx
  262d3e:\tmov    $0x21,%esi
  262d43:\tmov    %rax,%rdi
  262d46:\tcall   1480b0 <libusb_control_transfer@plt>
  262d52:\tcmpl   $0x3c,-0x58(%rbp)
"""


# --------------------------------------------------------------------------
# libuvcSetExtension: genuinely dynamic selector
# --------------------------------------------------------------------------
def test_extracts_both_control_transfer_branches():
    transfers = xu_trace.extract_control_transfers(SET_EXTENSION_EXCERPT)
    assert [t.address for t in transfers] == ["262f38", "262f94"]


def test_vendor_request_branch_shape():
    vendor = xu_trace.extract_control_transfers(SET_EXTENSION_EXCERPT)[0]
    assert vendor.bmRequestType == 0x41
    assert vendor.bRequest == 0xFF
    assert vendor.wIndex == 0x0300
    assert vendor.wLength == 0x3C


def test_uvc_set_cur_branch_shape():
    uvc = xu_trace.extract_control_transfers(SET_EXTENSION_EXCERPT)[1]
    assert uvc.bmRequestType == 0x21  # UVC class, interface, host-to-device
    assert uvc.bRequest == 0x01  # SET_CUR
    assert uvc.wIndex == 0x0400  # (entity 4 << 8) | interface 0
    assert uvc.wLength == 0x3C  # 60-byte payload


def test_parameter_derived_wvalue_is_reported_dynamic_not_guessed():
    # wValue = (selector << 8) & 0xffff is built from parameter 3 and staged
    # through -0x4(%rbp). The slot is register-written, so it must never be
    # reported as a constant even though an immediate never touches %ecx.
    for ct in xu_trace.extract_control_transfers(SET_EXTENSION_EXCERPT):
        assert ct.wValue is None
        assert ct.wValue_source is None


# --------------------------------------------------------------------------
# libuvcReadValueExtension: fixed selector + four-byte depth-scale payload
# --------------------------------------------------------------------------
def test_read_value_extension_set_cur_is_the_fixed_selector_and_four_byte_write():
    set_cur = xu_trace.extract_control_transfers(READ_VALUE_EXTENSION_EXCERPT)[0]
    assert set_cur.address == "262dec"
    assert set_cur.bmRequestType == 0x21  # SET_CUR, class, interface
    assert set_cur.bRequest == 0x01
    assert set_cur.wValue == 0x0E00  # fixed, NOT the caller's command id
    assert set_cur.wValue_source == "stack local -0x4"
    assert set_cur.wIndex == 0x0400
    assert set_cur.wLength == 4  # four-byte payload, not 60
    assert set_cur.timeout == 1000


def test_read_value_extension_get_cur_reuses_the_same_fixed_selector():
    get_cur = xu_trace.extract_control_transfers(READ_VALUE_EXTENSION_EXCERPT)[1]
    assert get_cur.address == "262e6a"
    assert get_cur.bmRequestType == 0xA1  # GET_CUR, class, interface, IN
    assert get_cur.bRequest == 0x81
    assert get_cur.wValue == 0x0E00  # same selector as the SET_CUR above
    assert get_cur.wValue_source == "stack local -0x4"
    assert get_cur.wIndex == 0x0400
    assert get_cur.wLength == 0x3C  # 60-byte response
    assert get_cur.timeout == 1000


def test_four_byte_payload_is_the_callers_command_id_argument():
    # The command/property id is NOT the wValue selector: it is written into
    # the first four bytes of the caller's buffer at 0x262dbd and sent as the
    # four-byte SET_CUR payload.
    set_cur = xu_trace.extract_control_transfers(READ_VALUE_EXTENSION_EXCERPT)[0]
    (store,) = set_cur.payload
    assert store.address == "262dbd"
    assert store.offset == 0
    assert store.width == 4
    assert store.source == "arg3"  # cs::libuvcReadValueExtension(type, handle, int, char*)
    assert store.const is None  # supplied by the caller, not an immediate
    assert set_cur.wLength == store.width


def test_depth_scale_payload_bytes_for_command_id_0x918():
    # getDepthScale passes 0x918 to readBuffer, which forwards it as arg3, so
    # the four payload bytes on the wire are the little-endian encoding.
    assert struct.pack("<I", 0x918) == b"\x18\x09\x00\x00"
    set_cur = xu_trace.extract_control_transfers(READ_VALUE_EXTENSION_EXCERPT)[0]
    assert set_cur.wLength == len(struct.pack("<I", 0x918))


def test_get_cur_carries_no_payload_store():
    # memset() clears the host buffer between the two transfers; the function
    # performs no additional protocol write, so the GET has no payload store.
    get_cur = xu_trace.extract_control_transfers(READ_VALUE_EXTENSION_EXCERPT)[1]
    assert get_cur.payload == []


# --------------------------------------------------------------------------
# libuvcExcuteCmd: fixed selector read on both sides of a branch
# --------------------------------------------------------------------------
def test_excute_cmd_recovers_fixed_selector_in_both_branches():
    transfers = xu_trace.extract_control_transfers(EXCUTE_CMD_EXCERPT)
    assert [t.address for t in transfers] == ["262d0c", "262d46"]
    for ct in transfers:
        assert ct.wValue == 0x0700
        assert ct.wValue_source == "stack local -0x54"
        assert ct.wLength == 0x3C


def test_excute_cmd_branch_request_types():
    vendor, uvc = xu_trace.extract_control_transfers(EXCUTE_CMD_EXCERPT)
    assert (vendor.bmRequestType, vendor.bRequest, vendor.wIndex) == (0x41, 0xFF, 0x0300)
    assert (uvc.bmRequestType, uvc.bRequest, uvc.wIndex) == (0x21, 0x01, 0x0400)


def test_excute_cmd_timeout_is_a_caller_argument_not_a_constant():
    for ct in xu_trace.extract_control_transfers(EXCUTE_CMD_EXCERPT):
        assert ct.timeout is None


# --------------------------------------------------------------------------
# Conservatism of the parser
# --------------------------------------------------------------------------
def test_no_call_sites_yields_empty():
    assert xu_trace.extract_control_transfers("  0:\tret\n") == []


def test_register_sourced_argument_not_treated_as_constant():
    # A bmRequestType loaded from another register must not carry a stale
    # immediate forward from an earlier unrelated mov.
    text = (
        "  10:\tmov    $0x21,%esi\n"
        "  14:\tmov    %eax,%esi\n"
        "  18:\tcall   1480b0 <libusb_control_transfer@plt>\n"
    )
    (ct,) = xu_trace.extract_control_transfers(text)
    assert ct.bmRequestType is None


def test_slot_written_twice_is_not_a_constant():
    text = (
        "  10:\tmovl   $0xe00,-0x4(%rbp)\n"
        "  14:\tmovl   $0x700,-0x4(%rbp)\n"
        "  18:\tmov    -0x4(%rbp),%ecx\n"
        "  1c:\tcall   1480b0 <libusb_control_transfer@plt>\n"
    )
    (ct,) = xu_trace.extract_control_transfers(text)
    assert ct.wValue is None


def test_unmodelled_arithmetic_clears_a_tracked_register():
    text = (
        "  10:\tmov    $0x21,%ecx\n"
        "  14:\tshl    $0x8,%ecx\n"
        "  18:\tcall   1480b0 <libusb_control_transfer@plt>\n"
    )
    (ct,) = xu_trace.extract_control_transfers(text)
    assert ct.wValue is None


def test_registers_are_cleared_at_a_jump_target():
    # Incoming register state at a branch target is ambiguous, so an immediate
    # from the fall-through path must not leak across it. The stack slot does
    # survive, which is what makes the fixed selectors recoverable.
    text = (
        "  10:\tmovl   $0xe00,-0x4(%rbp)\n"
        "  14:\tmov    $0x21,%esi\n"
        "  18:\tjmp    1c <f>\n"
        "  1c:\tmov    -0x4(%rbp),%ecx\n"
        "  20:\tcall   1480b0 <libusb_control_transfer@plt>\n"
    )
    (ct,) = xu_trace.extract_control_transfers(text)
    assert ct.bmRequestType is None  # cleared at the 0x1c jump target
    assert ct.wValue == 0x0E00  # stack slot survives


def test_call_clears_registers_and_stack_arguments():
    text = (
        "  10:\tmov    $0x21,%esi\n"
        "  14:\tpush   $0x3c\n"
        "  18:\tcall   148a70 <memset@plt>\n"
        "  1c:\tcall   1480b0 <libusb_control_transfer@plt>\n"
    )
    (ct,) = xu_trace.extract_control_transfers(text)
    assert ct.bmRequestType is None
    assert ct.wLength is None


# Real objdump -d excerpt of cs::libuvcReadFile @0x262861 (MINI branch only):
# the request write (wValue 0x0100), one chunk read (0x0200) and the cleanup
# write (0x0100) all stage wValue through the SAME slot -0x78(%rbp), written
# three times. The conservative rule must refuse all three rather than pick
# whichever immediate happened to appear last.
READ_FILE_EXCERPT = """\
  2628d2:\tmovb   $0x1,-0x70(%rbp)
  2628f0:\tmovl   $0x100,-0x78(%rbp)
  26294d:\tlea    -0x70(%rbp),%rdx
  262951:\tmov    -0xa0(%rbp),%rax
  262958:\tpush   $0x3e8
  26295d:\tpush   $0x3c
  26295f:\tmov    %rdx,%r9
  262962:\tmov    $0x400,%r8d
  262968:\tmov    $0x100,%ecx
  26296d:\tmov    $0x1,%edx
  262972:\tmov    $0x21,%esi
  262977:\tmov    %rax,%rdi
  26297a:\tcall   1480b0 <libusb_control_transfer@plt>
  2629da:\tmovl   $0x200,-0x78(%rbp)
  262a51:\tmov    -0x78(%rbp),%eax
  262a54:\tmovzwl %ax,%edx
  262a57:\tlea    -0x70(%rbp),%rcx
  262a5b:\tmov    -0xa0(%rbp),%rax
  262a62:\tpush   $0x3e8
  262a67:\tpush   $0x3c
  262a69:\tmov    %rcx,%r9
  262a6c:\tmov    $0x400,%r8d
  262a72:\tmov    %edx,%ecx
  262a74:\tmov    $0x81,%edx
  262a79:\tmov    $0xa1,%esi
  262a7e:\tmov    %rax,%rdi
  262a81:\tcall   1480b0 <libusb_control_transfer@plt>
  262b4e:\tmovl   $0x100,-0x78(%rbp)
"""


def test_read_file_request_write_recovers_its_direct_immediate():
    # The first transfer loads wValue straight into %ecx, so it is recovered
    # even though the slot feeding the later transfers is not.
    request = xu_trace.extract_control_transfers(READ_FILE_EXCERPT)[0]
    assert request.address == "26297a"
    assert (request.bmRequestType, request.bRequest) == (0x21, 0x01)
    assert request.wValue == 0x0100
    assert request.wValue_source == "immediate"
    assert (request.wIndex, request.wLength, request.timeout) == (0x0400, 0x3C, 1000)


def test_read_file_chunk_read_refuses_the_thrice_written_slot():
    # -0x78(%rbp) holds 0x0100, then 0x0200, then 0x0100 again. Reporting any
    # of them as "the" constant would be a guess, so the field stays unknown
    # and docs/architecture.md cites the three stores directly instead.
    chunk = xu_trace.extract_control_transfers(READ_FILE_EXCERPT)[1]
    assert chunk.address == "262a81"
    assert (chunk.bmRequestType, chunk.bRequest) == (0xA1, 0x81)
    assert chunk.wValue is None
    assert chunk.wValue_source is None
    assert (chunk.wIndex, chunk.wLength, chunk.timeout) == (0x0400, 0x3C, 1000)
