# Review: scanner-061 independent authorization review

- Verdict: `changes_requested`
- Review round: `0`
- Examined ref: `unborn main`; scanner-060 helper
- Helper SHA-256 examined:
  `561343da789c909800092590430774edc873f44417bc09bdbaa1c169d16194d2`
- Focused-test SHA-256 examined:
  `dc961690ce6a4134fcaeb0fa80b0ea1e4ea18927845e11d88754220d760c2d2f`

## Blocking findings

1. A SIGINT/SIGTERM arriving during shielded cleanup or release is appended to
   `deferred_signals`, but the helper neither records that list in the manifest
   nor requires it to be empty for `final_ok`. A signalled run could therefore
   be reported successful.
2. After release/close the helper records final interface topology but not a
   fresh exact target identity snapshot. Scanner-061 requires final serial,
   USB ID/path, bus/devnum and node identity evidence, not just interface
   bindings.

The helper/test and all five lineage hashes matched, and the independent review
found the fixed command/path/wire, claimed topology, cleanup ordering and
absence-unknown semantics otherwise sound. No live access or helper-main
invocation occurred in this review.

## Next action

Reopen scanner-060 for only these two offline corrections and focused tests.
Scanner-061 remains unauthorized until the corrected immutable helper receives
root review and a fresh independent authorization review.
