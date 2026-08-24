#!/usr/bin/env bash
# build.sh - build the standalone Revopoint 3DCamera SDK reference probe.
#
# The SDK (headers + lib3DCamera.so) is treated as an external, temporary
# oracle: it is never vendored into this repository. This script only
# references it by path, taken from --sdk-root or the REVOPOINT_SDK_ROOT
# env var, and fails loudly if it cannot find the expected files there.
#
# Usage:
#   REVOPOINT_SDK_ROOT=/path/to/thirdparty/3DCamera ./build.sh
#   ./build.sh --sdk-root /path/to/thirdparty/3DCamera [--out /tmp/.../probe]
#
# Build output goes under /tmp by default (never under the repo).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="${REVOPOINT_SDK_ROOT:-}"
OUT_PATH="/tmp/3d-scanner-probe/probe"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sdk-root)
            SDK_ROOT="$2"; shift 2 ;;
        --out)
            OUT_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--sdk-root <path-to-thirdparty/3DCamera>] [--out <output-binary-path>]"
            exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1 ;;
    esac
done

if [[ -z "$SDK_ROOT" ]]; then
    echo "ERROR: SDK root not set. Pass --sdk-root <path> or set REVOPOINT_SDK_ROOT." >&2
    echo "       Expected it to point at a 'thirdparty/3DCamera' directory containing" >&2
    echo "       include/3DCamera.hpp and linux/x64/lib3DCamera.so." >&2
    exit 1
fi

HEADER="$SDK_ROOT/include/3DCamera.hpp"
INCLUDE_DIR="$SDK_ROOT/include"
LIB_DIR="$SDK_ROOT/linux/x64"
LIB_FILE="$LIB_DIR/lib3DCamera.so"

MISSING=0
for f in "$HEADER" "$LIB_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: expected SDK file not found: $f" >&2
        MISSING=1
    fi
done
if [[ "$MISSING" -ne 0 ]]; then
    echo "ERROR: SDK root '$SDK_ROOT' does not look like a valid thirdparty/3DCamera checkout." >&2
    echo "       This script will NOT download the SDK. Fix REVOPOINT_SDK_ROOT/--sdk-root and retry." >&2
    exit 1
fi

echo "== SDK root: $SDK_ROOT"
echo "== Header:   $HEADER"
echo "== Library:  $LIB_FILE"
echo "== sha256sum of SDK inputs (external oracle, not vendored):"
sha256sum "$HEADER" "$LIB_FILE"

OUT_DIR="$(dirname "$OUT_PATH")"
mkdir -p "$OUT_DIR"

echo "== Building probe -> $OUT_PATH"
set -x
g++ -std=c++11 -O2 -Wall -Wextra \
    -D"SDK_LIB_PATH_STR=\"$LIB_FILE\"" \
    -I"$INCLUDE_DIR" \
    "$SCRIPT_DIR/probe.cpp" \
    -o "$OUT_PATH" \
    -L"$LIB_DIR" -l3DCamera \
    -Wl,-rpath,"$LIB_DIR" \
    -lpthread
set +x

echo "== Build OK: $OUT_PATH"
