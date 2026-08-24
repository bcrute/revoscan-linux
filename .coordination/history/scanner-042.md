# Scanner-042 accepted — shared RK1608 ownership audit

Official Rockchip source and immutable scanner-039 through scanner-041 evidence
were audited without accessing the scanner. The RK1608 misc device and forwarded
V4L2 `.s_power` path share one count; ordinary video/subdevice opens do not by
themselves prove a reference. Each completed textual-off shell command reduced
the pre-existing count by one.

The two accepted decrements with GPIO121 still high prove only that the count
before scanner-040 was at least three and the current count is at least one
under unchanged/no-race semantics. PID 569's fd114 is a likely remaining misc
reference, but the exact count and other owners remain unknown. Its watchdog fd,
caught signals, and cleanup strings do not prove that termination is safe.

No third blind decrement or process signal is justified. The next task is one
bounded read-only recovery of supervisor, watchdog, process, and fd metadata;
the emitter remains presumed on pending the owner's visual check.
