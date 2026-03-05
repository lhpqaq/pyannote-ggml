#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <audio.wav> [backend: auto|cpu|metal|cuda] [language]" >&2
    echo "Example: $0 mulpeo.wav cuda zh" >&2
    exit 1
fi

# Segmentation CUDA LSTM defaults (same as run-diarization.sh)
export DIARIZATION_SEG_LSTM_COOP=1
: "${DIARIZATION_SEG_LSTM_COOP_WARP:=1}"
: "${DIARIZATION_SEG_LSTM_COOP_WARPS:=4}"
: "${DIARIZATION_SEG_LSTM_COOP_WARP_NOSH:=1}"
: "${DIARIZATION_SEG_LSTM_COOP_BIDIR:=1}"
export DIARIZATION_SEG_LSTM_COOP_WARP
export DIARIZATION_SEG_LSTM_COOP_WARPS
export DIARIZATION_SEG_LSTM_COOP_WARP_NOSH
export DIARIZATION_SEG_LSTM_COOP_BIDIR

AUDIO_PATH="$1"
BACKEND="${2:-cuda}"
LANGUAGE="${3:-zh}"

case "$BACKEND" in
    auto|cpu|metal|cuda) ;;
    *)
        echo "Error: backend must be one of auto|cpu|metal|cuda" >&2
        exit 1
        ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$BACKEND" = "metal" ]; then
    if [ "$(uname -s)" != "Darwin" ]; then
        echo "Error: backend 'metal' requires macOS" >&2
        exit 1
    fi
    BUILD_DIR="$ROOT_DIR/diarization-ggml/build-metal"
else
    BUILD_DIR="$ROOT_DIR/diarization-ggml/build-x86-cuda"
fi

BIN="$BUILD_DIR/bin/transcribe"
SEG_MODEL="$ROOT_DIR/models/segmentation-ggml/segmentation.gguf"
EMB_MODEL="$ROOT_DIR/models/embedding-ggml/embedding.gguf"
PLDA_MODEL="$ROOT_DIR/diarization-ggml/plda.gguf"

# whisper.cpp model path (can be overridden)
: "${WHISPER_MODEL:=$ROOT_DIR/../whisper.cpp/models/ggml-small.bin}"

GPU_DEVICE="${GPU_DEVICE:-0}"

if [ ! -x "$BIN" ]; then
    echo "Error: binary not found: $BIN" >&2
    if [ "$BACKEND" = "metal" ]; then
        echo "Build first:" >&2
        echo "  cmake -S diarization-ggml -B diarization-ggml/build-metal -DGGML_METAL=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON" >&2
        echo "  cmake --build diarization-ggml/build-metal -j8" >&2
    else
        echo "Build first: cmake --build diarization-ggml/build-x86-cuda -j8" >&2
    fi
    exit 1
fi

if [ ! -f "$AUDIO_PATH" ]; then
    echo "Error: audio file not found: $AUDIO_PATH" >&2
    exit 1
fi

if [ ! -f "$SEG_MODEL" ] || [ ! -f "$EMB_MODEL" ] || [ ! -f "$PLDA_MODEL" ]; then
    echo "Error: model file missing (segmentation/embedding/plda)" >&2
    exit 1
fi

if [ ! -f "$WHISPER_MODEL" ]; then
    echo "Error: whisper model not found: $WHISPER_MODEL" >&2
    echo "Set WHISPER_MODEL to override." >&2
    exit 1
fi

CMD=(
    "$BIN"
    "$AUDIO_PATH"
    --seg-model "$SEG_MODEL"
    --emb-model "$EMB_MODEL"
    --whisper-model "$WHISPER_MODEL"
    --plda "$PLDA_MODEL"
    --backend "$BACKEND"
)

if [ "$BACKEND" = "cuda" ]; then
    CMD+=( --gpu-device "$GPU_DEVICE" )
fi

if [ -n "$LANGUAGE" ]; then
    CMD+=( --language "$LANGUAGE" )
fi

echo "Running transcribe"
echo "  audio:    $AUDIO_PATH"
echo "  backend:  $BACKEND"
echo "  gpu:      $GPU_DEVICE"
echo "  language: $LANGUAGE"
echo "  whisper:  $WHISPER_MODEL"

"${CMD[@]}"
