# Scanner-046 accepted — one contained partial live read-only run

The exact accepted scanner-045 helper was invoked once, without retry, in
`/tmp/scanner-045-evidence.aG7OZ6Aq`. Its manifest SHA-256 is
`4044e9a65da32bbbb84d8688bfc39497b47d4b9ee09f349b72f8f09d4a11b312`.
Root and independent Codex audit reproduced all 159 hash-linked transfer events,
121 response blobs, 12 decoded artifacts, and marker/manifest bindings.

`command:s99_cat` succeeded, but its immediate staging selector timed out and
the exactly once owed clear attempt also timed out. The helper returned 1 and
did not reach `clear:final_proc_stat`. Interface release and usbfs close passed,
no later scanner control occurred, and device identity/interfaces stayed stable.
Partial process/fd/init evidence is accepted; config, FDT, remaining supervision,
fdinfo, watchdog semantics, and teardown safety remain unknown. No shutdown or
optical test occurred, and the front emitter remains presumed on.
