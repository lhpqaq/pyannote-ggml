# 学术论文写作材料：面向 Turing(T4) 的 BiLSTM CUDA 优化（Pyannote Segmentation）

本文档为学术论文/技术报告准备，系统性描述本仓库中对 segmentation BiLSTM CUDA 推理路径的优化动机、算子形态、关键设计与实验验证方式。

目标读者：熟悉 GPU 编程/深度学习推理，但不一定了解 ggml 内部张量布局。

代码位置（实现参考）：

- `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`

## 1. 问题背景与算子形态

### 1.1 模型结构与推理约束

Pyannote segmentation 模型包含 4-layer BiLSTM（双向 LSTM）以及后续的线性层/分类器。

在本工程的推理设置中，关键形状通常为：

- hidden size：`H = 128`
- 序列长度：`T = 589`
- batch：`B = 1`（分块推理）

该设置导致 LSTM recurrence 的计算呈现“强时序依赖”：每个 timestep 必须在前一 timestep 完成后才能继续。
因此性能优化重点从“大矩阵乘”转向“减少每步开销与同步成本”。

### 1.2 原始 CUDA 实现的瓶颈

项目使用 ggml 的张量布局存储 recurrent 权重 `w_hh`，其视作 `ne0=H, ne1=4H` 的矩阵。
在原始 cooperative recurrence 内核中：

- 每个 hidden 单元 `h` 由一个 thread 计算（1 thread / hidden）
- thread 在 `k=0..H-1` 上累加四个 gate（i/f/g/o）的 dot-product
- 每个 timestep 使用 cooperative 的 `grid.sync()` 做跨 block 同步

瓶颈来源：

1. `w_hh` 按列主（ld=H）访问时，warp lane 对同一列读取呈 stride=H 的模式，导致 **global load 难以合并（coalescing 较差）**。
2. 每个 timestep 的 `grid.sync()` 是硬串行点；如果每步计算吞吐不高，barrier 成本占比迅速上升。
3. gate 计算涉及大量半精度权重的 `half->float` 转换，放在热循环中会产生显著指令开销。

## 2. 优化总体思路

针对上述形态，本优化遵循三条主线：

1. **改变线程映射**：使 `w_hh` 读取在 warp 内连续，最大化内存合并访问效率。
2. **减少每步指令数**：对固定形状 `H=128` 做 loop unroll 与 half2 向量化，减少 load/convert 指令。
3. **减少全局同步次数**：将 forward/reverse 两个方向融合为单 kernel，使每 timestep 的 `grid.sync()` 次数减半。

所有优化保持“推理结果可回归验证”，并通过 dump 中间张量对齐输出。

## 3. 具体优化方法

### 3.1 Warp-per-hidden 映射（coalesced weight loads）

**核心改变**：从 1 thread / hidden 改为 1 warp / hidden。

对每个 hidden 单元 `h`：

- 一个 warp 的 32 个 lane 分摊 `k` 维度的部分和：lane 处理 `k = lane + 32*i`。
- 对于列主存储的 `w_hh[col + k]`，当 col 固定时，warp 内 lane 访问的地址在 k 上连续，从而提升合并访问。
- 之后使用 `__shfl_down_sync` 进行 warp 归约得到最终 dot。

该方法的特点：

- 优化点直接作用于 memory coalescing，通常是此算子在 T4 上的主要收益来源。
- FP32 归约顺序与 baseline 不同，可能产生微小数值差异（舍入误差）；需要用 dump 对齐验证差异边界。

### 3.2 H=128 特化与 unroll

当 `H==128` 时：

- 将 `k` 循环展开为固定迭代（例如 4 次），减少分支/循环控制开销。
- 使用 `fmaf()` 提示编译器生成 fused-multiply-add 指令序列。

该优化通常提供小幅加速（single-digit %），但实现简单、风险低。

### 3.3 half2 向量化读取 recurrent 权重（H=128）

对 `w_hh`（FP16）在 H=128 情况下使用 `half2`：

- 每个 lane 一次处理两个 k（`k0=2*lane+64*i`, `k1=k0+1`）。
- 通过 `__half22float2` 将两个半精度权重一次转换为两个 float。
- 对四个 gate 分别累加两次 FMA。

该方法减少了：

- weight load 指令数量
- half->float 转换指令数量

### 3.4 去掉每步 shared hp staging（NO-SH）

传统做法会在每个 timestep 将 `hp` 写入 shared，并 `__syncthreads()`。
在本算子形态上，这个 staging 的成本可能超过节省的全局读开销。

NO-SH 变体直接从全局内存读取 `hp[k]`：

- 移除每步 shared 写入
- 移除每步 `__syncthreads()`
- 保留 `grid.sync()`（保证时序依赖）

实验观察：T4 上 NO-SH 通常更快，且与 shared 版本输出一致（读取位置不同但计算顺序不变）。

### 3.5 fused bidirectional（减少 grid.sync 次数）

基线实现中 forward 与 reverse 方向分别由两个 cooperative kernel 完成。
每个方向都包含 per-timestep 的 `grid.sync()`，因此 barrier 次数翻倍。

fused bidirectional 将双向融合为单 kernel：

- 一个 block 内含 `2 * WARPS_PER_BLOCK` 个 warp
- 前半部分 warp 计算 forward，后半部分 warp 计算 reverse
- 每个 timestep 只执行一次 `grid.sync()`，同时推进两个方向

工程注意点：

- fused 模式需要在启动 fused kernel 前确保 reverse 的 `ih_rev` GEMM 已经计算完成
- 本实现仅在 `B==1` 的 cooperative 模式下启用

## 4. 开关设计与实验协议

### 4.1 环境变量开关

开关的中文使用方案见：

- `docs/segmentation_cuda_lstm_switches_zh.md`

用于论文实验的推荐组合（T4）：

```bash
export DIARIZATION_SEG_LSTM_COOP=1
export DIARIZATION_SEG_LSTM_COOP_WARP=1
export DIARIZATION_SEG_LSTM_COOP_WARPS=4
export DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=1
export DIARIZATION_SEG_LSTM_COOP_BIDIR=1
```

### 4.2 性能测量方法

建议使用 segmentation-only 模式隔离 embedding 干扰：

```bash
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
  -o /tmp/out.rttm
```

使用 `nsys` 获取 kernel 级时间占比与实例数：

- `nsys profile ...`
- `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum ...`

### 4.3 正确性回归（建议写入论文方法部分）

性能优化可能改变归约顺序导致细微数值漂移，因此采用“中间张量 dump + 对齐指标”进行回归。

方法：

1. 设置：

```bash
export DIARIZATION_SEG_DEBUG_DUMP_DIR=/tmp/seg-dump
export DIARIZATION_SEG_DEBUG_DUMP_MAX=1
```

2. 运行两种配置（例如 baseline vs optimized）。
3. 使用：

```bash
python3 tools/compare-seg-dumps.py /tmp/seg-a /tmp/seg-b
```

建议报告指标：

- `max_abs` / `mean_abs` / `rmse`
- `argmax_diff_frames`

对于 NO-SH 与 fused bidir，在正确实现的情况下可达到“完全一致”（0 diff）。

## 5. 讨论与局限

- cooperative `grid.sync()` 仍是 per-step 的硬同步点；当计算被极致优化后，barrier 成本可能再次成为主要瓶颈。
- fused bidirectional 有助于减少 barrier 次数，但会增加单 kernel 的线程数与资源占用，对不同 GPU 的最优参数可能不同。
- Nsight Compute 可能受限于 performance counter 权限，论文中应说明 profiling 工具与环境限制。

## 6. 可复现实验表格建议（论文写作提示）

建议在论文中给出一张表，列为：

- baseline coop (thread/hidden)
- warp-per-hidden
- + unroll
- + half2
- + NO-SH
- + fused bidir

行内容：

- segmentation-only 总耗时
- LSTM kernel 总耗时（nsys kernel sum）
- speedup
- 数值对齐指标（max_abs/argmax_diff_frames）
