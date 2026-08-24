# Scanner-038 accepted — embedded projector-control archaeology

The live scanner is an RV1126 RMSL DDR3L board running Buildroot/BusyBox. The
SDK-derived GPIO117 path does not exist. Exact live FDT bytes (98,304 bytes,
SHA-256 `076ae86c2f334e9fbd73942b754590a165a94075f142c33834c27c9d69fbfdc5`)
map GPIO55/60 to image-sensor PWDN, GPIO61 to sensor reset, and GPIO121/125 to
the RK1608 pre-ISP reset/IRQ. No projector/illumination GPIO or regulator node
is present.

The pinned SDK provides reboot and firmware-upgrade paths but no supported
shutdown. No firmware image exists locally. A projector-off function cannot
yet be implemented honestly; the next boundary is the live vendor init/process
stack and the RK1608/`rk_preisp` handler, followed by exact firmware recovery
before any persistent patch.
