#!/bin/bash
# Benchmark CoreML compute unit configurations with proper statistical rigor.
#
# Protocol:
#   1. For each config: 1 warmup run (discarded), then N timed runs.
#   2. All runs save RTTM → compare for consistency across runs & configs.
#   3. Reports min / median / max for segmentation, embedding, and total times.
#
# Usage: bash tools/bench_coreml_compute_units.sh [audio_file] [N]

set -euo pipefail

AUDIO="${1:-samples/sample.wav}"
N="${2:-5}"
BIN="./diarization-ggml/build/bin/diarization-ggml"
SEG_GGUF="models/segmentation-ggml/segmentation.gguf"
EMB_GGUF="models/embedding-ggml/embedding.gguf"
PLDA="diarization-ggml/plda.gguf"
EMB_COREML="models/embedding-ggml/embedding.mlpackage"
SEG_COREML="models/segmentation-ggml/segmentation.mlpackage"

CONFIGS=("all" "cpu_ane" "cpu_gpu")
BENCH_DIR="/tmp/bench_coreml_$$"
mkdir -p "$BENCH_DIR"

audio_name=$(basename "$AUDIO" .wav)

echo "============================================"
echo "CoreML Compute Unit Benchmark"
echo "Audio: $AUDIO"
echo "Runs per config: $N (+ 1 warmup)"
echo "Temp dir: $BENCH_DIR"
echo "============================================"
echo ""

# Extract a timing value from the === Timing Summary === section.
# The summary lines look like:  "  Segmentation:       437 ms  (50.5%)"
extract_timing() {
    local label="$1"
    local text="$2"
    echo "$text" | awk -v lbl="$label" '
        /=== Timing Summary ===/ { in_summary=1; next }
        in_summary && $0 ~ lbl":" {
            for (i=1; i<=NF; i++) {
                if ($i == "ms" && i > 1) { print $(i-1); exit }
            }
        }'
}

ref_rttm=""

for cfg in "${CONFIGS[@]}"; do
    export COREML_COMPUTE_UNITS="$cfg"
    echo "--- Config: COREML_COMPUTE_UNITS=$cfg ---"

    rttm_out="$BENCH_DIR/${audio_name}_${cfg}.rttm"

    # Warmup run (discard timing, keep RTTM for comparison)
    $BIN $SEG_GGUF $EMB_GGUF "$AUDIO" \
        --plda $PLDA \
        --coreml $EMB_COREML \
        --seg-coreml $SEG_COREML \
        -o "$rttm_out" 2>&1 >/dev/null
    echo "  warmup done"

    seg_times=()
    emb_times=()
    tot_times=()
    rttm_info=""

    for i in $(seq 1 $N); do
        rttm_run="$BENCH_DIR/${audio_name}_${cfg}_run${i}.rttm"
        output=$($BIN $SEG_GGUF $EMB_GGUF "$AUDIO" \
            --plda $PLDA \
            --coreml $EMB_COREML \
            --seg-coreml $SEG_COREML \
            -o "$rttm_run" 2>&1)

        seg=$(extract_timing "Segmentation" "$output")
        emb=$(extract_timing "Embeddings" "$output")
        tot=$(extract_timing "Total" "$output")
        info=$(echo "$output" | grep "Diarization complete:" || true)

        seg_times+=("${seg:-0}")
        emb_times+=("${emb:-0}")
        tot_times+=("${tot:-0}")
        rttm_info="$info"

        printf "  run %d: seg=%4sms  emb=%4sms  total=%4sms\n" "$i" "$seg" "$emb" "$tot"
    done

    # Statistics: sort, get min/median/max
    sorted_seg=($(printf '%s\n' "${seg_times[@]}" | sort -n))
    sorted_emb=($(printf '%s\n' "${emb_times[@]}" | sort -n))
    sorted_tot=($(printf '%s\n' "${tot_times[@]}" | sort -n))
    mid=$(( (N - 1) / 2 ))
    last=$(( N - 1 ))

    printf "  ── min:    seg=%4sms  emb=%4sms  total=%4sms\n" "${sorted_seg[0]}" "${sorted_emb[0]}" "${sorted_tot[0]}"
    printf "  ── median: seg=%4sms  emb=%4sms  total=%4sms\n" "${sorted_seg[$mid]}" "${sorted_emb[$mid]}" "${sorted_tot[$mid]}"
    printf "  ── max:    seg=%4sms  emb=%4sms  total=%4sms\n" "${sorted_seg[$last]}" "${sorted_emb[$last]}" "${sorted_tot[$last]}"
    echo "  ── result: $rttm_info"

    # RTTM consistency check across runs within this config
    run_mismatch=0
    for i in $(seq 2 $N); do
        if ! diff -q "$BENCH_DIR/${audio_name}_${cfg}_run1.rttm" "$BENCH_DIR/${audio_name}_${cfg}_run${i}.rttm" >/dev/null 2>&1; then
            echo "  ⚠ RTTM mismatch: run1 vs run$i"
            run_mismatch=1
        fi
    done
    if [ "$run_mismatch" -eq 0 ]; then
        echo "  ✓ All $N runs produced identical RTTM"
    fi

    # Cross-config RTTM consistency
    if [ -z "$ref_rttm" ]; then
        ref_rttm="$BENCH_DIR/${audio_name}_${cfg}_run1.rttm"
    else
        if ! diff -q "$ref_rttm" "$BENCH_DIR/${audio_name}_${cfg}_run1.rttm" >/dev/null 2>&1; then
            echo "  ⚠ RTTM differs from first config ($(basename $ref_rttm)):"
            diff "$ref_rttm" "$BENCH_DIR/${audio_name}_${cfg}_run1.rttm" | head -10
        else
            echo "  ✓ RTTM matches reference config"
        fi
    fi

    echo ""
done

echo "============================================"
echo "All benchmark results saved in: $BENCH_DIR"
echo "============================================"
