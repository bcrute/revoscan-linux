"""Closed volatile gain experiment for MINI controller SHA256 decdc3ad...0f1.

Recovered ARM 0x21c18 constructs {3, 0x903, value}; 0x21c70 queries 0x9a3.
The depth gain handler at 0x19844 validates user gain 1..16, shifts left four
bits, clips to 255, and calls 0x21c18. Our range is deliberately only 16, 24
(fresh measured baseline), 32. This is sensor gain, NOT projector current.
"""

from scanner_timing_probe import xu

GAINS = (16, 24, 32)


def set_gain(fd: int, value: int) -> None:
    if value not in GAINS:
        raise ValueError("Only the three bounded gain candidates are permitted")
    payload = f"echo s 0x903 {value} >/dev/rk_preisp".encode().ljust(60, b"\x00")
    xu(fd, 7, 1, payload)
