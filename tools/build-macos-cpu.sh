#!/usr/bin/env bash

set -euo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Error: build-macos-cpu.sh must run on macOS" >&2
    exit 1
fi

JOBS="${JOBS:-8}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/diarization-ggml/build-macos-cpu"

echo "[build-macos-cpu] configure: $BUILD_DIR"
cmake -S "$ROOT_DIR/diarization-ggml" -B "$BUILD_DIR" -DGGML_METAL=OFF

echo "[build-macos-cpu] build -j$JOBS"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "[build-macos-cpu] done"
