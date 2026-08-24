# Scanner-050 accepted-negative — non-empty-root refusal

After a passing exact-identity preflight at devnum 50, the executor created
stdout/stderr capture files inside `/tmp/scanner-049-evidence.h1cHnBb8` before
the one permitted helper invocation. `/tmp/scanner049_readonly.py` correctly
refused the non-empty directory and exited 1 before `resolve_target`, usbfs
open, any ownership ioctl/control transfer, or any remote command.

Root and independent Codex review verified the six preserved artifact hashes,
the early control-flow boundary, exactly one invocation, no retry/second root,
and the scanner's later unchanged fully bound topology. The reported 48.18-
second passive samples were not independently hash-linked, so they remain an
executor observation. This is an orchestration failure with zero scanner
action, not a live scanner result. Power and optical state were not tested.
