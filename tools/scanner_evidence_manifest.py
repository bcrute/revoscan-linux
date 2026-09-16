"""Hash bounded diagnostic evidence without changing inputs or opening hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def manifest(roots: list[Path], output: Path) -> dict[str, Any]:
    if not 1 <= len(roots) <= 4 or output.exists():
        raise ValueError("One to four roots and a new output file required")
    paths: set[Path] = set()
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Evidence roots must be real directories")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Do not follow evidence symlinks: {path}")
            if path.is_file():
                paths.add(path.resolve())
            if len(paths) > 10000:
                raise ValueError("Evidence exceeds 10,000-file bound")
    if sum(path.stat().st_size for path in paths) > 15_000_000_000:
        raise ValueError("Evidence exceeds 15 GB bound")
    records = []
    total_bytes = 0
    for path in sorted(paths):
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise RuntimeError(f"Evidence changed while hashing: {path}")
        records.append(dict(path=str(path), bytes=after.st_size, sha256=digest))
        total_bytes += after.st_size
    report: dict[str, Any] = dict(
        created_utc=datetime.now(UTC).isoformat(),
        roots=[str(root.resolve()) for root in roots],
        scope="File integrity only, not measurement correctness; excludes this manifest",
        files=records,
        file_count=len(records),
        total_bytes=total_bytes,
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = manifest(args.root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))
