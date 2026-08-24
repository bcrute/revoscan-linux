# Current task

- Task ID: `scanner-060`
- State: `changes_requested`
- Review round: `1`
- Starting ref: `unborn main; scanner-061 independent review`

## Objective

Correct exactly two offline evidence/success defects in the scanner-060 helper
and focused suite. Do not broaden its live plan or capabilities.

## Candidate under correction

- `/tmp/scanner060_inventory.py` SHA-256
  `561343da789c909800092590430774edc873f44417bc09bdbaa1c169d16194d2`
- `/tmp/scanner060_offline_test.py` SHA-256
  `dc961690ce6a4134fcaeb0fa80b0ea1e4ea18927845e11d88754220d760c2d2f`

Rehash both plus all five immutable helper-gated lineage inputs before editing.

## Required corrections

1. Preserve the complete deferred SIGINT/SIGTERM list in final manifest and
   completion evidence. `final_ok` must be false whenever any deferred signal
   occurred, including during cleanup, release or close. Add faithful focused
   tests that exercise signals in those shielded finalization intervals and
   prove the result cannot report success.
2. After release/close, perform and preserve a fresh read-only exact target
   identity snapshot. Prove the same serial `A22704522A6F00A64`, USB ID
   `2207:110c`, path `1-1.1`, bus/devnum, node rdev/major/minor/inode as the
   originally pinned unit, alongside detached interface topology. Identity
   absence, re-enumeration or drift must make `final_ok` false. Add focused
   exact-match, drift and absence tests.

## Preserve unchanged

All exact command/path allowlists, wire setup/framing, one-open/claim/release/
close lifecycle, no-reconnect behavior, ceilings, cleanup/selector debt,
absence-unknown semantics, hash gate, marker ordering, and no optical claim.
Do not introduce another target command, path, ioctl class or product import.

## Proportional proof and handoff

In-memory compile and the one focused offline suite only. Report new exact
helper/test hashes, the two corrected behaviors, and tests run. Update only the
executor-owned status/latest report plus the two `/tmp` candidates. Do not edit
product code or Codex-owned planner/review files.

## Stop conditions

No helper main, evidence root, live USB/sysfs/devnode, target command/file/
process, network/download, sudo, package sync/install, firmware, reset/power,
product edit, Claude, commit/push/deploy or purchase. Scanner-061 remains
unauthorized pending root and independent review of the corrected candidates.
