# scanner-019 — accepted

- Accepted review round: `0`
- Live diagnostic invocations: exactly `1`; no retry
- Evidence: `/tmp/scanner-019-evidence.CjqNHU`

The fixed three-observation physical run delivered one complete 512000-byte depth
artifact and three RGB payloads of 22271, 22323 and 22321 bytes. Every RGB payload
contains a decodable baseline 1280x800 JPEG prefix followed by exactly four opaque
bytes; strict status therefore remains `trailing_data` and the command correctly
returned exit 5. Raw SHA-256 values are `50a5a18c...a40a331`,
`34279a4b...561cb`, and `4153d15f...8206b`; trailers are `256e1d02`,
`c46e1d02`, and `e06f1d02`. No trimming or suffix interpretation occurred.

The series used three polls/dequeues, two requeues, retained 66915 bytes and
reported sequences `0,4,6`, gaps `[3,1]`, and intervals 169476000/283789000 ns.
Cleanup was exhaustive and clean. Devnum changed 45-to-46 for an unresolved
reason, then six samples over 5.35 seconds were stable with the same serial/path,
nodes, bindings, permissions and no holders.

Independent GStreamer decode succeeded for all unchanged raw payloads. All three
were almost black with essentially unchanged statistics: mean channels remained
below 0.85/255 and over 99.16% of pixels had all channels below 16. No useful RGB,
suffix meaning, exposure behavior, synchronization or scan accuracy is claimed.
