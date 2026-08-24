# Scanner-036 accepted — offline embedded-shutdown archaeology

The pinned `lib3DCamera.so` (SHA-256
`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`)
contains no supported scanner shutdown path. `Camera::restart()` sends literal
`reboot`; stop/disconnect does not issue illumination or power control; no
device-command `poweroff` or `halt` exists.

`Camera::resetTrackerScannerState()` has no direct caller. For a 17-character
serial with byte 14 `A`, matching `A22704522A6F00A64`, it sends exact command
`echo 0 > /sys/class/gpio/gpio117/value`. Its electrical effect is unknown and
it is not shutdown evidence. The owner's broad device-reset/bricking authority
permits one separately bounded optical experiment, scanner-037.
