# scanner-059 internal-ioctl feasibility

Offline evidence supports a syscall-only ARM EABI helper for the recovered
`/dev/v4l-subdev3` ioctl, but no local ARM toolchain is installed. The helper
can later be emitted as a minimal pinned ELF and checked structurally. Delivery
is not yet proven: remote byte readback is established, while upload/append,
chmod/execute/status and cleanup are not. Scanner-060 will implement only a
four-command `/tmp` primitive inventory plus direct `/proc/mounts` read.
