# Scanner-043 accepted — bounded supervisor/watchdog metadata recovery

One separately reviewed read-only helper ran exactly once against USB
`2207:110c`, serial `A22704522A6F00A64`, at devnum 49. Evidence root
`/tmp/scanner-043-evidence.BWTqs7Haw8k4`; manifest SHA-256
`3a68f22f7286bf19cf137da5adba92348e6b294830bf866bf85e67933f02242e`.

Root and an independent Codex auditor recomputed the complete 558-event hash
chain, all 328 response blobs, all 110 decoded blobs, and the exact final
`clear:final_proc_stat` transfer. Interface-0 claim/release and usbfs close
succeeded; pinned identity and all interface bindings remained stable; no
scanner control followed the final clear.

PID 569 remained the same 32-thread `/tmp/zxmain` process and still owned fd 5
`/dev/watchdog` and fd 114 `/dev/rk_preisp`. The recovered
`/userdata/watchlog.sh` is only a periodic log trimmer, not a process supervisor.
The watchdog class exposes no readable timeout/nowayout state on this firmware,
so process termination remains unsafe pending offline recovery of zxmain's exact
watchdog-disable control flow. No shutdown or optical-darkness claim is made;
the blue emitter remains presumed on.
