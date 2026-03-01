#!/usr/bin/env bash

set -euo pipefail

JOBS="${JOBS:-8}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/diarization-ggml/build-x86-cuda"

echo "[build-linux-cuda] configure: $BUILD_DIR"
cmake -S "$ROOT_DIR/diarization-ggml" -B "$BUILD_DIR"

echo "[build-linux-cuda] build -j$JOBS"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "[build-linux-cuda] done"
