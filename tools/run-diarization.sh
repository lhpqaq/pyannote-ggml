#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <audio.wav> <backend: auto|cpu|metal|cuda> [output.rttm]" >&2
    exit 1
fi

AUDIO_PATH="$1"
BACKEND="$2"
OUTPUT_PATH="${3:-}"
# export DIARIZATION_DEBUG_BACKEND_ASSIGN_RATIO=1 
# export DIARIZATION_SEG_FORCE_ALL_GPU=1
# export  DIARIZATION_SEG_LSTM_NOCUSTOM_DEBUG=1
export DIARIZATION_SEG_OP_GAP=1
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

BIN="$BUILD_DIR/bin/diarization-ggml"
SEG_MODEL="$ROOT_DIR/models/segmentation-ggml/segmentation.gguf"
EMB_MODEL="$ROOT_DIR/models/embedding-ggml/embedding.gguf"
PLDA_MODEL="$ROOT_DIR/diarization-ggml/plda.gguf"

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

CMD=(
    "$BIN"
    "$SEG_MODEL"
    "$EMB_MODEL"
    "$AUDIO_PATH"
    --plda "$PLDA_MODEL"
    --backend "$BACKEND"
)

if [ -n "$OUTPUT_PATH" ]; then
    CMD+=( -o "$OUTPUT_PATH" )
fi

echo "Running diarization"
echo "  audio:   $AUDIO_PATH"
echo "  backend: $BACKEND"
if [ -n "$OUTPUT_PATH" ]; then
    echo "  output:  $OUTPUT_PATH"
else
    echo "  output:  stdout"
fi

if [ -n "${SEG_GPU_PARTITION_MODE:-}" ]; then
    case "$SEG_GPU_PARTITION_MODE" in
        classifier|linear|all)
            export DIARIZATION_SEG_GPU_PARTITION_MODE="$SEG_GPU_PARTITION_MODE"
            echo "  seg-gpu-partition: $SEG_GPU_PARTITION_MODE"
            ;;
        *)
            echo "Error: SEG_GPU_PARTITION_MODE must be classifier|linear|all" >&2
            exit 1
            ;;
    esac
fi

"${CMD[@]}"
