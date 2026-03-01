#!/usr/bin/env bash

set -euo pipefail

JOBS="${JOBS:-8}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/diarization-ggml/build-linux-cpu"

echo "[build-linux-cpu] configure: $BUILD_DIR"
cmake -S "$ROOT_DIR/diarization-ggml" -B "$BUILD_DIR"

echo "[build-linux-cpu] build -j$JOBS"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "[build-linux-cpu] done"
