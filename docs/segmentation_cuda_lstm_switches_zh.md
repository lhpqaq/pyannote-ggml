# 分割模型 BiLSTM CUDA 开关使用方案（中文）

本文档说明本仓库中用于加速 *pyannote segmentation* BiLSTM CUDA 路径的环境变量开关，给出推荐组合与调参策略。

实现位置（供定位代码）：

- `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`

## 适用范围与前提

- 这些开关作用于 segmentation BiLSTM 的 `GGML_OP_CUSTOM` CUDA fast-path。
- cooperative kernel 仅在以下条件同时满足时生效：
  - 设备支持 cooperative launch
  - `DIARIZATION_SEG_LSTM_COOP=1`
  - `B==1`（该自定义实现只在 batch=1 的推理路径上启用 coop；否则走其它路径/回退）

## 开关一览

### 1) Cooperative 总开关

- `DIARIZATION_SEG_LSTM_COOP=1`
  - 启用 cooperative recurrence（内核中每个 timestep 使用 `grid.sync()` 强制跨 block 同步）。
  - 关闭/置 0 将回退到非 cooperative 路径（通常更慢）。

### 2) Warp-per-hidden 映射开关

- `DIARIZATION_SEG_LSTM_COOP_WARP=1`
  - 启用“1 个 warp 负责 1 个 hidden 单元”的映射方式。
  - 该映射针对 `w_hh` 的列主存储（ld=H）做了访问模式优化，使得 warp 内读取更易合并（coalesced）。

### 3) 每个 block 的 warp 数（关键调参）

- `DIARIZATION_SEG_LSTM_COOP_WARPS={2|4|8|16}`
  - 仅对 `DIARIZATION_SEG_LSTM_COOP_WARP=1` 生效。
  - 含义：每个 block 内 warp 的数量（也等价于一个 block 同时计算多少个 hidden 单元）。
  - T4（sm_75）经验最优：`4`。
  - 原理：
    - warp 数太小：并行度不足
    - warp 数太大：block 太重，cooperative resident/调度受限，且每步 barrier 参与 block 可能过少/过多导致吞吐变化

### 4) hp 直接从全局内存读取（去掉每步 shared staging）

- `DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=1`
  - 仅对 warp 路径生效。
  - 含义：每个 timestep 不再把 `h_{t-1}` 缓存到 shared（也不需要 `__syncthreads()`），而是 warp lane 直接从全局内存读取 `hp[k]`。
  - T4 上通常更快，且数值输出与 shared staging 版本一致（只是读取位置不同，计算顺序不变）。

### 5) Fused 双向（一次 cooperative kernel 同时算 forward+reverse）

- `DIARIZATION_SEG_LSTM_COOP_BIDIR=1`
  - 仅对 warp cooperative 路径生效。
  - 含义：把 forward/reverse 两个方向合并到一次 cooperative launch。
  - 目的：把每步 `grid.sync()` 从 “每方向一次（两次）” 降为 “每 timestep 一次”。
  - 当前实现要求 `B==1`（cooperative 模式本身也只在 B==1 生效）。

## Embedding 相关（补充）

Embedding 阶段的 CUDA 优化主要发生在 ggml-cuda 的 conv2d/im2col 路径，不依赖额外运行时开关。
本文档不逐项列出内核选择逻辑，详细算法与实验协议见：

- `docs/paper_cuda_embedding_optimization_notes_zh.md`

## 推荐使用方案（从保守到激进）

### 方案 A：当前最快（T4 推荐）

适用：Tesla T4 优先，追求最快 segmentation 推理，允许使用 fused 双向。

```bash
export DIARIZATION_SEG_LSTM_COOP=1
export DIARIZATION_SEG_LSTM_COOP_WARP=1
export DIARIZATION_SEG_LSTM_COOP_WARPS=4
export DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=1
export DIARIZATION_SEG_LSTM_COOP_BIDIR=1
```

### 方案 B：warp + nosh，但不做 fused（方便定位/回归）

```bash
export DIARIZATION_SEG_LSTM_COOP=1
export DIARIZATION_SEG_LSTM_COOP_WARP=1
export DIARIZATION_SEG_LSTM_COOP_WARPS=4
export DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=1
export DIARIZATION_SEG_LSTM_COOP_BIDIR=0
```

### 方案 C：warp 映射但保留 shared staging（更传统、便于对照）

```bash
export DIARIZATION_SEG_LSTM_COOP=1
export DIARIZATION_SEG_LSTM_COOP_WARP=1
export DIARIZATION_SEG_LSTM_COOP_WARPS=4
export DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=0
export DIARIZATION_SEG_LSTM_COOP_BIDIR=0
```

### 方案 D：回退到旧的 cooperative（1 thread / hidden）

```bash
export DIARIZATION_SEG_LSTM_COOP=1
export DIARIZATION_SEG_LSTM_COOP_WARP=0
```

## 调参建议（多 GPU 兼容）

1. 先固定为方案 A，在目标 GPU 上跑 `--bypass-embeddings` 的 segmentation-only 基准。
2. 扫描 `DIARIZATION_SEG_LSTM_COOP_WARPS`：2/4/8/16。
3. 在最优 `WARPS` 下，对比：
   - `DIARIZATION_SEG_LSTM_COOP_BIDIR=0/1`
   - `DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=0/1`

经验：

- T4 上 `WARPS=4` 一般最优。
- `WARP_NOSH=1` 往往更快。
- `COOP_BIDIR=1` 在 barrier 成本占比较高时收益很大。

## Profiling 与正确性回归

### Profiling（推荐 nsys）

Nsight Compute（`ncu`）可能因为权限问题无法使用（`ERR_NVGPUCTRPERM`），建议用 Nsight Systems（`nsys`）先做耗时归因。

用于隔离 segmentation 的命令（不跑 embedding）：

```bash
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
  -o /tmp/out.rttm
```

### Dump 回归（推荐）

通过 dump 中间张量对齐不同开关组合：

```bash
export DIARIZATION_SEG_DEBUG_DUMP_DIR=/tmp/seg-dump
export DIARIZATION_SEG_DEBUG_DUMP_MAX=1

python3 tools/compare-seg-dumps.py /tmp/seg-a /tmp/seg-b
```

## 与脚本默认值的关系

`tools/run-diarization.sh` 会为 CUDA 路径设置一组“偏快”的默认值（但允许外部 env 覆盖）。
若需要强制特定方案，请在运行脚本前显式 `export` 对应变量。
