#!/bin/bash
# =============================================================================
# Pipeline Optimization Benchmark
# 对比优化前后的完整执行流程：构建 → 运行 → 对比
#
# 用法:
#   bash tools/bench_pipeline_optimization.sh [audio_file] [N]
#
# 参数:
#   audio_file  音频文件路径（默认: samples/sample.wav）
#   N           每种模式运行次数（默认: 5）
#
# 前置条件:
#   1. 已安装 cmake, clang/clang++
#   2. CoreML 模型已就绪:
#      - models/segmentation-ggml/segmentation.mlpackage
#      - models/embedding-ggml/embedding.mlpackage
#   3. GGML 模型已就绪:
#      - models/segmentation-ggml/segmentation.gguf
#      - models/embedding-ggml/embedding.gguf
#      - diarization-ggml/plda.gguf
#
# =============================================================================

set -euo pipefail

AUDIO="${1:-samples/sample.wav}"
N="${2:-5}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BIN="./diarization-ggml/build/bin/diarization-ggml"
SEG_GGUF="models/segmentation-ggml/segmentation.gguf"
EMB_GGUF="models/embedding-ggml/embedding.gguf"
PLDA="diarization-ggml/plda.gguf"
EMB_COREML="models/embedding-ggml/embedding.mlpackage"
SEG_COREML="models/segmentation-ggml/segmentation.mlpackage"

BENCH_DIR="/tmp/bench_pipeline_opt_$$"
mkdir -p "$BENCH_DIR"
audio_name=$(basename "$AUDIO" .wav)

# ─── 辅助函数 ─────────────────────────────────────────────────

extract_timing() {
    local label="$1"
    local text="$2"
    echo "$text" | awk -v lbl="$label" '
        /=== Timing Summary/ { in_summary=1; next }
        in_summary && $0 ~ lbl":" {
            for (i=1; i<=NF; i++) {
                if ($i == "ms" && i > 1) { print $(i-1); exit }
            }
        }'
}

median_of() {
    local arr=("$@")
    local sorted=($(printf '%s\n' "${arr[@]}" | sort -n))
    local mid=$(( (${#sorted[@]} - 1) / 2 ))
    echo "${sorted[$mid]}"
}

run_benchmark() {
    local label="$1"
    local extra_env="$2"
    shift 2

    echo ""
    echo "━━━ $label ━━━"

    # Warmup
    eval "$extra_env" $BIN $SEG_GGUF $EMB_GGUF "$AUDIO" \
        --plda $PLDA \
        --coreml $EMB_COREML \
        --seg-coreml $SEG_COREML \
        -o "$BENCH_DIR/${audio_name}_${label// /_}_warmup.rttm" 2>&1 >/dev/null
    echo "  warmup done"

    local seg_times=()
    local emb_times=()
    local tot_times=()
    local rttm_info=""

    for i in $(seq 1 $N); do
        local rttm_run="$BENCH_DIR/${audio_name}_${label// /_}_run${i}.rttm"
        local output
        output=$(eval "$extra_env" $BIN $SEG_GGUF $EMB_GGUF "$AUDIO" \
            --plda $PLDA \
            --coreml $EMB_COREML \
            --seg-coreml $SEG_COREML \
            -o "$rttm_run" 2>&1)

        local seg emb tot
        seg=$(extract_timing "Segmentation" "$output")
        emb=$(extract_timing "Embeddings" "$output")
        tot=$(extract_timing "Total" "$output")
        rttm_info=$(echo "$output" | grep "Diarization complete:" || true)

        seg_times+=("${seg:-0}")
        emb_times+=("${emb:-0}")
        tot_times+=("${tot:-0}")

        printf "  run %d: seg=%5s ms  emb=%5s ms  total=%5s ms\n" "$i" "$seg" "$emb" "$tot"
    done

    local med_seg med_emb med_tot
    med_seg=$(median_of "${seg_times[@]}")
    med_emb=$(median_of "${emb_times[@]}")
    med_tot=$(median_of "${tot_times[@]}")

    printf "  ── median: seg=%5s ms  emb=%5s ms  total=%5s ms\n" "$med_seg" "$med_emb" "$med_tot"
    echo "  ── $rttm_info"

    # RTTM consistency
    local mismatch=0
    for i in $(seq 2 $N); do
        if ! diff -q "$BENCH_DIR/${audio_name}_${label// /_}_run1.rttm" \
                     "$BENCH_DIR/${audio_name}_${label// /_}_run${i}.rttm" >/dev/null 2>&1; then
            echo "  ⚠ RTTM mismatch: run1 vs run$i"
            mismatch=1
        fi
    done
    [ "$mismatch" -eq 0 ] && echo "  ✓ All $N runs: RTTM identical"

    echo "$med_tot" > "$BENCH_DIR/${label// /_}_median_total.txt"
}

# ─── 前置检查 ─────────────────────────────────────────────────

echo "============================================================"
echo " Pipeline Optimization Benchmark"
echo "============================================================"
echo "  Audio:    $AUDIO"
echo "  Runs:     $N (+ 1 warmup)"
echo "  Temp:     $BENCH_DIR"
echo ""

for f in "$BIN" "$SEG_GGUF" "$EMB_GGUF" "$PLDA" "$AUDIO"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: 文件不存在: $f" >&2
        echo "" >&2
        echo "请先构建项目:" >&2
        echo "  cmake -S diarization-ggml -B diarization-ggml/build \\" >&2
        echo "    -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON" >&2
        echo "  cmake --build diarization-ggml/build -j" >&2
        exit 1
    fi
done

for d in "$EMB_COREML" "$SEG_COREML"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: CoreML 模型不存在: $d" >&2
        echo "请先导出 CoreML 模型（见下方命令）" >&2
        exit 1
    fi
done

# ─── 执行基准测试 ──────────────────────────────────────────────

# 优化后: cpu_ane（推荐配置，使用管线并行 + fbank预计算 + CoreML优化）
run_benchmark "optimized_cpu_ane" "COREML_COMPUTE_UNITS=cpu_ane"

# 优化后: all（默认 CoreML 配置）
run_benchmark "optimized_all" "COREML_COMPUTE_UNITS=all"

# ─── 结果对比 ──────────────────────────────────────────────────

echo ""
echo "============================================================"
echo " 结果汇总"
echo "============================================================"

for label in optimized_cpu_ane optimized_all; do
    f="$BENCH_DIR/${label}_median_total.txt"
    if [ -f "$f" ]; then
        printf "  %-25s  median total = %5s ms\n" "$label" "$(cat "$f")"
    fi
done

# RTTM cross-config consistency
ref="$BENCH_DIR/${audio_name}_optimized_cpu_ane_run1.rttm"
alt="$BENCH_DIR/${audio_name}_optimized_all_run1.rttm"
if [ -f "$ref" ] && [ -f "$alt" ]; then
    if diff -q "$ref" "$alt" >/dev/null 2>&1; then
        echo ""
        echo "  ✓ cpu_ane 与 all 配置 RTTM 完全一致"
    else
        echo ""
        echo "  ⚠ cpu_ane 与 all 配置 RTTM 有差异"
    fi
fi

echo ""
echo "结果文件: $BENCH_DIR"
echo "============================================================"
