# Scanner-047 accepted — direct-first read-only collector

The final helper `/tmp/scanner047_readonly.py` has SHA-256
`f1d7aaa7255a5c187b40957a384938c35995a527002a124d13dcbb09968f9874`;
its pure-offline suite `/tmp/scanner047_offline_test.py` has SHA-256
`6646e49bd997d4fb3c58017452b9d9fd2b1d4d8e998ae9fdf84df6a724c395a4`.
Root and independent Codex review accepted round 1.

The helper validates the accepted scanner-045 lineage and scanner-046 failure
manifest, performs a required remote PID-stat gate, attempts all 54 direct
config/FDT/watchdog/supervision paths before any command, removes the three
init-script `cat` commands, and runs 12 remaining read-only commands through a
recorded 250 ms settle. The strict protocol/evidence/budget/final-clear and
release/close invariants are preserved. This task was offline only and makes no
shutdown or optical-darkness claim.
