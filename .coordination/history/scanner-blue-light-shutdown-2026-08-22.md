# Blue-light final-state shutdown — accepted operational intervention

- Physical command invocations: exactly `1`; no retry
- Device: `2207:110c`, serial `A22704522A6F00A64`
- Observed path: sysfs `1-1.1`, bus 1, devnum 47

After the owner made blue-light shutdown an explicit completion condition,
Codex performed one unprivileged SDK-derived laser-off command using the shared
interface-0 ownership lifecycle. The immutable setup was
`21 01 00 07 00 04 3c 00 e8 03 00 00`; its 60-byte payload was the ASCII text
`echo s 0x922 0 >/dev/rk_preisp` followed by NUL padding, SHA-256
`6030364daddcc4c9855d16da552c58161bbec5cf77ea23ace9de2f14839ffb0d`.

The sole SET returned 60/60 bytes in about 1.7 ms. Detach, claim, release and
reconnect all passed. Six postflight samples over five seconds retained sysfs
path `1-1.1` and devnum 47. A corrected canonical-path check confirmed
`/dev/video0` through `/dev/video3`, interfaces 0-3 bound to `uvcvideo`,
interface 4 bound to `usbhid`, and no node holders.

This establishes exact command acceptance and clean host-driver restoration.
The stack has no optical readback, so it does not independently prove the
visible light state. Future depth-streaming tasks must automate and evidence
this off command after V4L2 teardown before they may be considered complete.
