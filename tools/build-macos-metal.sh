#!/usr/bin/env bash

set -euo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Error: build-macos-metal.sh must run on macOS" >&2
    exit 1
fi

JOBS="${JOBS:-8}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/diarization-ggml/build-metal"

echo "[build-macos-metal] configure: $BUILD_DIR"
cmake -S "$ROOT_DIR/diarization-ggml" -B "$BUILD_DIR" \
  -DGGML_METAL=ON \
  -DEMBEDDING_COREML=ON \
  -DSEGMENTATION_COREML=ON

echo "[build-macos-metal] build -j$JOBS"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "[build-macos-metal] done"
