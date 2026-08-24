# Scanner-045 accepted — exact-unit read-only helper

The final standalone helper is `/tmp/scanner045_readonly.py`, SHA-256
`301cfa2a5b39549aa299918b24bc463a5d624f152acd5a8ca0eb50b3245cd114`.
Its pure offline test is `/tmp/scanner045_offline_test.py`, SHA-256
`5da406e90c22f5b49cd632d7b7239cb789a7dd30b46441b00e735f1607ad856c`.
Root and independent Codex review accepted round 3 after reproducing every
prior malformed/status/reservation case.

The helper pins 15 commands, 51 selector paths, exact scanner/PID identity,
four startup-script hashes, bounded config gzip and strict FDT parsing, command-
child fd metadata, aggregate budgets, one clear per selector, and an exact last
`clear:final_proc_stat` transfer before release/close. It was not invoked by
this task and makes no shutdown or optical-darkness claim. One separately
assigned single-use live read-only run is next.
