# 学术论文写作材料：Embedding (ResNet34+TSTP) CUDA 优化（面向 T4）

本文档总结本仓库中 *embedding* 阶段（WeSpeaker ResNet34 + TSTP pooling）在 CUDA 后端上的优化工作，目标是以 end-to-end pipeline 的 *Embeddings* 计时为准实现可观加速，同时保持 diarization 输出一致性。

实现相关代码主要位于：

- `whisper.cpp/ggml/src/ggml-cuda/im2col.cu`
- `models/embedding-ggml/src/model.cpp`
- `diarization-ggml/src/diarization.cpp`
- `diarization-ggml/src/streaming.cpp`

## 1. 背景与瓶颈定位

### 1.1 Pipeline 约束

在 segmentation LSTM 优化后，离线 diarization pipeline 在 T4 上常转为 embedding 阶段主导：

- `Segmentation` 可压到数百毫秒级
- `Embeddings` 仍常为 1s+（sample.wav）或 10s+（mulpeo.wav）

因此优化目标以 `diarization-ggml` 输出的阶段计时为准：

- `Embeddings: ... done [X ms]`

### 1.2 nsys 观测：im2col 为主瓶颈

使用 `nsys profile` + `nsys stats --report cuda_gpu_kern_sum`，在 embedding-only 与 pipeline 两种场景均观察到：

- embedding ResNet 的 conv2d 路径大量时间花在 `im2col_kernel<__half>` 及其变体

在 ggml 的 conv2d 实现中（`whisper.cpp/ggml/src/ggml.c:ggml_conv_2d`），conv2d 被表达为：

1. `im2col(a, b)` 将输入展开为矩阵
2. `mul_mat` 做 GEMM

因此 im2col 的 kernel 形态直接决定 conv2d 的总体效率。

## 2. 优化目标与设计原则

### 2.1 优化目标

1. 以 pipeline `Embeddings` 阶段耗时为主指标
2. 保持 diarization 输出一致（尤其是 `mulpeo.wav` 的金标准：`Diarization complete: 99 segments, 4 speakers`）

### 2.2 设计原则

- 优先选择数值等价的算子级优化（例如仅改变并行映射/访存方式，不改变数学表达）
- 避免改变 embedding 计算次数的启发式剪枝（剪枝往往会影响最终聚类与 RTTM）

## 3. 已实现优化

### 3.1 CPU 侧：融合 mask + transpose，消除 per-speaker 全量拷贝

离线与 streaming 路径都存在如下低效模式：

- 每个 speaker 复制一份完整的 fbank（`masked_fbank = fbank`）
- 按帧将非活动区域置零
- `embedding::model_infer()` 内部再执行一次 row-major → ggml-layout 的转置

优化后：

- GGML embedding 路径直接构建 ggml-layout 的 `fbank_transposed`（[80][T] 视图）并调用 `model_infer_transposed()`
- CoreML 路径仍使用 row-major（仅在 CoreML 模式下才分配对应 buffer）

该优化减少 CPU 内存带宽与重复分配开销，对长音频更加稳定。

### 3.2 CUDA 侧：im2col 参数特化 fast-path

在 `im2col_cuda()` 内增加对 ResNet 常见参数的匹配，选择更轻量的 kernel：

- 3x3, pad=1, stride=1, dil=1
- 3x3, pad=1, stride=2, dil=1
- 1x1, pad=0, stride=1, dil=1
- 1x1, pad=0, stride=2, dil=1

这些特化主要减少通用路径中的额外算术与分支。

### 3.3 CUDA 侧：out-parallel im2col（关键提速点）

#### 3.3.1 动机

通用 im2col 的网格结构中，block 数量与 `OW * (N*OH) * ceil(IC*KH*KW / 256)` 成正比。
当 `IC*KH*KW` 较大（ResNet 中后层常见），block 数会膨胀，导致更高的 launch/调度开销，并降低每个 block 的有效工作密度。

#### 3.3.2 方法

对 `3x3 pad=1 dil=1` 场景，引入 out-parallel kernel：

- 1 个 block 对应 1 个输出位置（out_idx = n*OH*OW + oh*OW + ow）
- block 内 threads 在 `i in [0, IC*KH*KW)` 上循环完成该输出位置的展开写入

当 `IC*KH*KW > 256` 时，选择 out-parallel kernel；否则保留原 kernel（避免小尺寸场景下并行度不足）。

对应内核：

- `im2col_kernel_p1s1d1_outpar<T>`
- `im2col_kernel_p1s2d1_outpar<T>`

#### 3.3.3 结果

以 pipeline 的 `Embeddings` 阶段计时为准，在 `samples/sample.wav` 上可稳定降低 `Embeddings` 耗时约 10% 量级；
在 `mulpeo.wav` 上同样可观察到 Embeddings 总耗时下降，并保持 diarization 输出金标准一致。

## 4. 正确性验证协议（论文方法部分建议）

### 4.1 Pipeline-level 金标准

在有把握时使用 `mulpeo.wav` 验证：

- 期望输出：`Diarization complete: 99 segments, 4 speakers`

该用例较慢，不建议每次迭代都运行。

### 4.2 快速回归

每次改动先用 `samples/sample.wav`：

- 记录 `Embeddings: ... done [X ms]`
- 确认 `Diarization complete` 输出结构合理（segments/speakers 不异常）

建议重复运行 3 次取中位数，以减少抖动造成的误判。

### 4.3 反例（不建议写入主line的优化）

尝试“按 min_active_ratio 提前跳过 embedding 推理”虽然能显著降低 Embeddings 时间，但会改变最终聚类与 RTTM（不满足输出一致性要求）。

论文中可作为讨论部分：剪枝类优化可能牺牲最终准确性。

## 5. Profiling 方法（论文附录建议）

### 5.1 nsys

推荐命令：

```bash
nsys profile -o /tmp/nsys/full --force-overwrite=true --trace=cuda,osrt \
  ./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
    models/segmentation-ggml/segmentation.gguf \
    models/embedding-ggml/embedding.gguf \
    samples/sample.wav \
    --backend cuda --plda diarization-ggml/plda.gguf \
    -o /tmp/out.rttm

nsys stats --report cuda_gpu_kern_sum /tmp/nsys/full.nsys-rep
```

用 `cuda_gpu_kern_sum` 确认 im2col kernel 是否仍为主耗时，并观察新 kernel（例如 out-parallel）是否被命中。

## 6. 局限与未来工作

- 本文优化仍基于“显式 im2col + GEMM”的 conv2d 表达，理论上更高效的路径是绕开显式 im2col（implicit GEMM / cuDNN / 更深度的 cutlass conv 实现）。
- 后续可考虑在特定固定 shape 下引入更激进的 conv 实现，并通过开关控制用于回归与部署。
