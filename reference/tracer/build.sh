#!/usr/bin/env bash
# build.sh - build the repository-owned usbfs ioctl tracer.
#
# This produces a small LD_PRELOAD shared object that records USBDEVFS ioctls
# issued to one exact usbfs node. It links against nothing outside libc and
# needs no SDK, no headers from the vendor, and no elevated privilege.
#
# Usage:
#   ./build.sh [--out /tmp/3d-scanner-trace/libusbfs_trace.so]
#
# Build output goes under /tmp by default, never into the repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_PATH="/tmp/3d-scanner-trace/libusbfs_trace.so"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--out <output-so-path>]"
            exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1 ;;
    esac
done

mkdir -p "$(dirname "$OUT_PATH")"

echo "== Building usbfs tracer -> $OUT_PATH"
set -x
gcc -std=gnu11 -O2 -g -Wall -Wextra -Wpedantic -fPIC -shared \
    "$SCRIPT_DIR/usbfs_trace.c" \
    -o "$OUT_PATH" \
    -ldl -lpthread
set +x

echo "== Build OK: $OUT_PATH"
sha256sum "$OUT_PATH"
