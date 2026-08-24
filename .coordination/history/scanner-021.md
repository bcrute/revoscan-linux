# scanner-021 — accepted

- Accepted review round: `0`
- Live diagnostic invocations: exactly `1`; no retry
- Evidence: `/tmp/scanner-021-evidence.fANtK8`

The fixed read-only RGB-control inventory completed on serial
`A22704522A6F00A64` at devnum 46 with 11 ioctls, one successful close and stable
post-close identity. It found gain current/max 16/16; auto-exposure menu current
index 3; inactive absolute exposure reporting 131072 outside its 1..60000 range;
unsupported auto-exposure priority; enabled auto white balance; and inactive
white-balance temperature current 6500. These remain raw device-reported values,
not setting recommendations or proof of image quality.

Stdout and `outcome.json` are byte-identical at 10417 bytes with SHA-256
`f6540078c317a5ce7cd8dc1be6f1fc23bf8f968917c2d402b96a92fbac8fc9c2`.
Six postflight samples over 6.257 seconds kept devnum, bindings, nodes,
permissions and zero holders stable. No setting, menu, stream, depth or vendor
operation and no privilege escalation occurred.
