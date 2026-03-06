# CUDA Embedding 正确性诊断与修复

本文档记录对 CUDA 路径 embedding 正确性问题的分析和添加的诊断工具。

## 1. 问题背景

### 1.1 历史问题

`CUDA_CONTEXT.md` 记录了以下问题：

- 全流程 CUDA embedding 路径产生异常：`Filter: 0 embeddings`
- embedding 向量全零或 NaN，导致聚类回退到单说话人
- RTTM 与 CPU 基线差异显著
- 但独立 `embedding-ggml --test-inference` 的 CPU/CUDA 数值接近，可通过

### 1.2 当前状态

检查发现：
- 早期的 "CPU 权重保护策略"（当 backend=cuda 时强制 seg/emb 权重 CPU）已被移除
- 当前代码直接将 `config.ggml_backend`（如 "cuda"）传给 `model_load` 的 `weight_backend` 参数
- 权重直接加载到 GPU 内存

### 1.3 已有的修复

embedding 模型中 TSTP 方差路径已有 clamp 保护：

```cpp
// CUDA path can produce tiny negative variance from FP roundoff; clamp before sqrt
struct ggml_tensor* var_unbiased_nonneg = ggml_clamp(ctx, var_unbiased, 0.0f, INFINITY);
```

此修复防止了 FP 舍入误差导致的负方差 → NaN 传播问题。

## 2. 添加的诊断工具

### 2.1 环境变量 `DIARIZATION_EMB_DEBUG`

替代旧的 `DIARIZATION_EMB_DEBUG_NAN`（功能超集），提供更全面的 embedding 健康检查。

#### 逐 embedding 检查（在 `extract_embeddings` 和 `pipeline_parallel_seg_emb_ggml` 中）

对每个生成的 embedding 向量检查：

- NaN 个数
- Inf 个数
- L2 范数

异常触发条件：`nan > 0 || inf > 0 || L2 < 1e-6 || L2 > 1e6`

输出格式：
```
[emb] chunk=5 spk=1 nan=0 inf=0 L2=0.0000
[emb-par] chunk=12 spk=0 nan=256 inf=0 L2=0.0000
```

#### 过滤前汇总（在 `diarize_from_samples` 的 Filter 步骤前）

统计所有 embedding 的健康分布：

```
[emb-summary] total=663 nan=220 zero=3 normal=440
```

| 指标 | 含义 |
|------|------|
| total | 总 embedding 数（num_chunks × 3 speakers） |
| nan | 含 NaN 的 embedding 数（通常是静默 speaker） |
| zero | L2 < 1e-6 的 embedding 数（异常：应关注） |
| normal | 正常 embedding 数 |

### 2.2 使用方法

```bash
export DIARIZATION_EMB_DEBUG=1

./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --plda diarization-ggml/plda.gguf \
  -o /tmp/out.rttm
```

### 2.3 诊断流程

1. **正常情况**：`nan` 数等于静默 speaker 数，`zero=0`，`normal` 数等于活跃 speaker-chunk 数
2. **CUDA 问题**：`zero > 0` 或 `nan` 远超预期 → 说明 CUDA embedding 推理产生异常值
3. **对比 CPU**：用 `--backend cpu` 运行同一音频，对比 summary 数据

## 3. 常见诊断场景

### 场景 A：`Filter: 0 embeddings`

```bash
export DIARIZATION_EMB_DEBUG=1
# 运行后查看 [emb-summary]
# 如果 normal=0 且 nan=total → embedding 推理全部失败
# 如果 normal>0 但 zero=normal → embedding 推理返回零向量
```

### 场景 B：RTTM 与 CPU 差异大

```bash
# CPU 基线
./diarization-ggml ... --backend cpu -o /tmp/cpu.rttm

# CUDA
export DIARIZATION_EMB_DEBUG=1
./diarization-ggml ... --backend cuda -o /tmp/cuda.rttm

# 对比 RTTM
diff /tmp/cpu.rttm /tmp/cuda.rttm

# 对比 embedding summary 中的 normal 数是否一致
```

### 场景 C：逐 embedding 异常定位

设置 `DIARIZATION_EMB_DEBUG=1` 后，异常 embedding 会打印 chunk 和 speaker 索引，可用于精确定位问题出现的位置。

## 4. 修改文件

| 文件 | 改动 |
|------|------|
| `diarization-ggml/src/diarization.cpp` | `extract_embeddings`: 扩展 NaN 检查为完整健康检查；`pipeline_parallel_seg_emb_ggml`: 添加健康检查；`diarize_from_samples`: 过滤前添加 summary |

## 5. 后续优化方向

| 方向 | 状态 |
|------|------|
| 确认当前全 CUDA 路径在 T4 上的正确性 | 需在有 GPU 的环境验证 |
| 如果仍有问题，添加 per-tensor dump 对比 CPU/CUDA | 工具已就绪（`DIARIZATION_SEG_DEBUG_DUMP_DIR`） |
| weight placement 自动回退策略 | 可选：检测到异常时自动切换 CPU 权重 |
