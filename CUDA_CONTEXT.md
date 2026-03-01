# pyannote-ggml CUDA 工作上下文（截至当前）

## 目标

- 在 x86 环境启用并验证 `pyannote-ggml` 的 CUDA 支持（从 ARM 迁移后）。
- **优先保证推理正确性**（用户明确要求结果正确优先于速度）。
- 参考 CoreML 集成方式，评估可否迁移/借鉴到 CUDA。

## 关键要求

- 使用最多 `-j8` 线程进行构建。
- 持续推进 CUDA embedding 路径排查，不回退为“放弃 CUDA”。
- 对比 CPU 与 CUDA 输出，验证结果一致性/可接受性。

## 已确认环境与基础结论

- NVIDIA GPU 可用（`Tesla T4`），`nvidia-smi` 正常。
- 需要新建 x86 CUDA 构建目录，避免 ARM 缓存污染：`diarization-ggml/build-x86-cuda`。

## 主要问题与处理过程

### 1) 构建阻塞：BLAS 头文件缺失

- 初始构建报错缺少 `cblas.h`。
- 安装 `libopenblas-dev` 后恢复。
- 已补充 Linux BLAS 链接与 Apple/非 Apple 条件分支。

### 2) CUDA 初始化正常，但执行落到 CPU

- `--backend cuda` 可正常初始化 CUDA。
- 早期观察到调度仍显示 `gpu=0`（虽显示 op 支持率 100%）。

### 3) 强制 GPU 节点分配实验导致崩溃

- 激进强制分配 GPU 节点会在 segmentation 路径触发 CUDA illegal memory access。
- 对应激进改动已回退。

### 4) 结果正确性关键问题（当前核心）

- 当 diarization 全流程中 embedding 权重/计算走 CUDA 时，embedding 速度提升明显，但结果质量异常：
  - `Filter: 0 embeddings`
  - 回退到单簇/单说话人
  - RTTM 与 CPU 基线差异显著
- 但独立 `embedding-ggml --test-inference` 的 CPU/CUDA 数值接近并可通过，说明可能是**流水线交互问题**（mask/filter/scheduler/weight placement 等），而非“CUDA embedding 完全不可用”。

### 5) CoreML 路径的启发

- 仓库内 CoreML 属于模型桥接/旁路执行，并非 ggml 通用逐算子调度。
- 对 CUDA 方案设计有参考意义，但不能 1:1 套用。

## 已完成改动

1. x86 CUDA 构建打通（`-j8`）并可运行。
2. 增加跨平台 BLAS 兼容修复（Apple 与 Linux 分支）。
3. 验证 CUDA backend 初始化与执行路径日志。
4. 实验并回退不安全的强制 GPU 节点分配方案。
5. 为 segmentation/embedding 增加 backend-aware `model_load` 重载（支持 `weight_backend` / `gpu_device`）。
6. 在 embedding TSTP 方差路径加入 clamp 后再 `sqrt`（防止负方差导致数值异常）。
7. 为 `embedding-ggml` 增加调试参数：`--backend`、`--gpu-device`。
8. 在 diarization/streaming/model_cache 中加入当前保护策略：当主 backend 为 CUDA 时，seg/emb 权重暂强制 CPU（保证结果稳定）。

## 当前状态

- **已完成**：CUDA x86 基础构建、BLAS 兼容、关键实验与回退、安全护栏、调试入口。
- **进行中**：定位“全流程 CUDA embedding 导致过滤后 0 向量/聚类异常”的根因。
- **暂行策略**：以正确性优先，默认保守路径（CUDA 主后端 + seg/emb CPU 权重）。

## 下一步计划

1. 在 `extract_embeddings` 与预过滤阶段加细粒度诊断（NaN/Inf、L2 范围、active ratio）。
2. 在 **diarization 流水线内部** 做 CPU vs CUDA embedding 向量逐段对比（不仅是独立 embedding 工具）。
3. 拆分验证问题来源：
   - weight placement
   - scheduler/backend split
   - 输入 mask 差异
   - 过滤与有效性判定阈值
4. 在保持正确性默认值前提下，提供显式实验开关用于 CUDA embedding 试验。

## 关键改动文件

### 构建与链接

- `diarization-ggml/CMakeLists.txt`
- `models/segmentation-ggml/CMakeLists.txt`

### BLAS/Accelerate 兼容

- `diarization-ggml/src/vbx.cpp`
- `diarization-ggml/src/plda.cpp`

### Embedding 模型（backend + 数值修复 + CLI）

- `models/embedding-ggml/src/model.h`
- `models/embedding-ggml/src/model.cpp`
- `models/embedding-ggml/src/main.cpp`

### Segmentation 模型（backend-aware load）

- `models/segmentation-ggml/src/model.h`
- `models/segmentation-ggml/src/model.cpp`

### Diarization 调用链（当前保护策略）

- `diarization-ggml/src/diarization.cpp`
- `diarization-ggml/src/model_cache.cpp`
- `diarization-ggml/src/streaming.cpp`

## 构建与产物路径

- 构建目录：`diarization-ggml/build-x86-cuda`
- 二进制：`diarization-ggml/build-x86-cuda/bin/*`
- 对比输出（临时）：`/tmp/test_cpu*.rttm`、`/tmp/test_cuda*.rttm`、`/tmp/cmp_cpu.rttm`、`/tmp/cmp_cuda*.rttm`
