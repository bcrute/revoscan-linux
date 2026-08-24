# Scanner-049 accepted — dynamic-devnum detached collector

Root and independent Codex review accepted `/tmp/scanner049_readonly.py` at
SHA-256
`8b630434a44a1059655618d3a50b47351094c45715e3ae4390fb5ff4c8e0b57c`
and its pure-offline suite at SHA-256
`01db1b8bfe6908ccc614bb88fdca15887d9463d6035d48f89b52407d7a6dda5b`.

The helper preserves scanner-047's direct-first 54-path/12-command evidence
protocol while accepting and pinning the exact positive devnum resolved for the
fixed scanner identity. It requires the rebound topology, verifies interface 0
is owned by `uvcvideo`, detaches it once, requires interfaces 0/1 unbound, claims
once, then performs release and close without reconnect. Identity, signal,
clear-debt, release/close and final-state failure paths block success.

The exhaustive offline closure permits only CONTROL, GETDRIVER, DISCONNECT,
CLAIM and RELEASE behavior and excludes reset, reconnect, power, firmware,
V4L2/HID and process mutation. No live scanner access occurred in scanner-049,
and no shutdown or optical-darkness claim exists.
