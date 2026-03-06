# CUDA/GGML 路径 Seg/Emb 管线并行优化

本文档记录将 CoreML 路径中的 Segmentation–Embedding 管线并行优化移植到 GGML（CPU/CUDA/Metal）路径的改动。

## 1. 动机

### 1.1 基线问题

在 GGML 路径中，diarization 管线完全串行执行：

```
基线执行流（串行）:
for c = 0 to N-1:
    seg_logits[c] = segmentation_infer(audio_chunk[c])    // GGML (CPU/CUDA)
powerset_to_multilabel(seg_logits) → binarized            // CPU
for c = 0 to N-1:
    for s = 0 to 2:
        fbank[c] = compute_fbank(audio_chunk[c])          // CPU
        embedding[c][s] = embedding_infer(masked_fbank)   // GGML (CPU/CUDA)
```

总耗时 = t_seg + t_emb。Segmentation 和 Embedding 使用不同的模型、不同的 ggml 后端实例和不同的 scheduler，理论上可并行执行。

### 1.2 CoreML 路径已有方案

CoreML 路径的 `pipeline_parallel_seg_emb` 函数已使用 Producer-Consumer 模式实现双线程并行，在 Apple Silicon 上实现了 ~1.5x 加速（5644ms → ~3700ms）。

### 1.3 GGML 路径缺失

GGML 路径（包括 CUDA 后端）没有并行执行路径，所有 seg 推理完成后才开始 emb 推理。

## 2. 实现方案

### 2.1 新函数 `pipeline_parallel_seg_emb_ggml`

在 `diarization.cpp` 中新增 GGML 版本的并行管线函数，采用与 CoreML 版本相同的 Producer-Consumer 模式：

```
时间轴 →

Seg 线程:  [seg(0)][seg(1)][seg(2)][seg(3)]...
              │       │       │       │
              ▼       ▼       ▼       ▼
Emb 线程:        [emb(0)][emb(1)][emb(2)]...
```

- **Producer 线程**（segmentation）：顺序对每个 chunk 执行 GGML segmentation 推理 + 输出转置（`[class,frame]` → `[frame,class]`）+ per-chunk powerset→binary
- **Consumer 线程**（main，embedding）：全局 fbank 预计算（Opt-1），等待 chunk c 的 segmentation 完成后，切片 fbank + CMN + mask + GGML embedding 推理

### 2.2 同步机制

```cpp
std::atomic<int> chunks_segmented{0};
std::atomic<bool> seg_error{false};
std::mutex mtx;
std::condition_variable cv;
```

Producer 每完成一个 chunk 就递增 `chunks_segmented` 并通知 consumer。Consumer 通过 `cv.wait` 阻塞等待对应 chunk 就绪。

### 2.3 激活条件

在 `diarize_from_samples` 中，GGML 并行路径的激活条件为：

```cpp
if (!parallel_done && !use_seg_coreml && !use_emb_coreml &&
    seg_model.ctx && emb_model.ctx && !config.bypass_embeddings)
```

即：CoreML 并行路径未激活、两个 GGML 模型都已加载、且非 bypass 模式。

### 2.4 GGML 特殊处理

与 CoreML 版本的差异：

1. **输出转置**：GGML segmentation 输出为 `[class, frame]` 布局，需在 producer 端转置为 `[frame, class]`
2. **Embedding 输入布局**：GGML 期望列主 `[80, T]` 布局，consumer 端构建 masked+transposed fbank
3. **错误传播**：新增 `seg_error` 原子标志，segmentation 失败时安全终止 consumer

## 3. CUDA 并发分析

### 3.1 线程安全性

两个 GGML 模型使用独立的 `ggml_backend_sched` 实例。在 CUDA 后端下：

- 每个 scheduler 通过 `ggml_backend_cuda_init()` 获取独立的后端实例
- 每个后端实例有独立的 CUDA stream
- kernel 提交到不同 stream 上，GPU 可以尝试并发执行

### 3.2 预期收益

| 场景 | 串行 | 并行 | 理论加速 |
|------|------|------|---------|
| CPU 后端 | t_seg + t_emb | max(t_seg, t_emb) | ~1.5-1.85x |
| CUDA 后端 | t_seg + t_emb | max(t_seg, t_emb) + GPU 争抢开销 | ~1.3-1.7x |

CUDA 后端的实际收益取决于 GPU 资源竞争程度。即使 GPU kernel 不能完全并行，CPU 侧的工作（fbank 计算、masking、graph 构建）也能与另一个模型的 GPU 推理重叠。

### 3.3 与 Opt-1 (fbank 预计算) 的协同

GGML 并行路径内部直接集成了全局 fbank 预计算（consumer 线程入口处一次性计算），两项优化叠加生效。

## 4. 正确性保证

- Producer 端的 segmentation 推理 + 转置 + powerset 操作与串行路径完全一致
- Consumer 端的 fbank + mask + embedding 推理操作与 `extract_embeddings` 函数完全一致
- 两个线程不共享可变状态（通过独立的 model/state 实例保证）
- 模型输出应与串行路径 bit-exact 一致（同一后端、同一权重、相同输入）

## 5. 修改文件

| 文件 | 改动 |
|------|------|
| `diarization-ggml/src/diarization.cpp` | 新增 `pipeline_parallel_seg_emb_ggml` 函数；在 `diarize_from_samples` 中添加 GGML 并行路径 |

## 6. 实测结果与状态

### 6.1 Tesla T4 单 GPU 实测（sample.wav 30s, 21 chunks）

| 模式 | Seg | Emb | Total | 正确性 |
|------|----:|----:|------:|:------:|
| 基线（串行） | 185 ms | 1105 ms | 1486 ms | 14 段 2 人 |
| 并行路径 | 402 ms | 1167 ms | 1487 ms | 14 段 2 人 |

**结论**：单 GPU CUDA 下，并行路径因 GPU 资源竞争反而无收益（seg 从 185→402ms，emb 从 1105→1167ms）。Wall-clock 大致持平。

### 6.2 当前策略

并行路径改为 **opt-in** 模式（设置 `DIARIZATION_PARALLEL=1` 启用），默认走更快的串行路径。适用场景：
- CPU-only 后端（seg/emb 可利用不同核心真正并行）
- 未来多 GPU 配置

### 6.3 验证方法

```bash
# 启用并行路径
export DIARIZATION_PARALLEL=1

./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --plda diarization-ggml/plda.gguf \
  -o /tmp/out.rttm
```

验证：

1. 日志中应出现 `parallel/ggml` 字样
2. RTTM 输出应与串行路径完全一致
