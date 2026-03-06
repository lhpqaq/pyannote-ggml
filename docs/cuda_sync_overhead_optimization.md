# CUDA 同步开销分析与 Segmentation 图缓存（未完成）

本文档记录对 ggml scheduler CUDA 同步开销的分析和图缓存优化的尝试。

## 1. 瓶颈分析

### 1.1 nsys 观测结论

在 T4 上使用 `nsys stats --report cuda_api_sum` 分析，`cudaStreamSynchronize` 是 CPU API 时间的最大消耗。每次 `model_infer` 调用触发的同步点包括：

| 同步点 | 来源 | 调用次数/chunk |
|--------|------|---------------|
| `ggml_backend_tensor_set` | 输入数据 H2D 拷贝后同步 | 1-2 |
| `ggml_backend_sched_graph_compute` → `ggml_backend_sched_synchronize` | 计算完成后同步所有后端 | 1 |
| `ggml_backend_tensor_get` | 输出数据 D2H 拷贝前同步 | 1 |

Segmentation 模型每个 chunk 至少触发 3 次 `cudaStreamSynchronize`。

### 1.2 额外开销：graph 重建

Segmentation 的 `model_infer` 每次调用都重新执行：

1. `build_graph()` — 分配新 ggml 上下文，构建完整计算图
2. `prefer_gpu_for_supported_nodes()` — 遍历所有节点设置 GPU 偏好
3. `ggml_backend_sched_alloc_graph()` — 分配/验证所有张量缓冲区

其中 1-2 在输入 shape 固定（160000 samples）的情况下结果完全不变，存在冗余。

## 2. 图缓存优化尝试（已回退）

### 2.1 方案

在 `segmentation_state` 中新增持久化图缓存字段，首次构建图后缓存所有指针，后续直接复用。

### 2.2 遇到的问题

实测发现图缓存导致 **segmentation 输出全零**（"no speakers detected"），根本原因：

**`ggml_backend_sched_reset()` 会清除 tensor→backend 映射表**。原始代码每次 `model_infer` 都重新调用 `build_graph()` + `prefer_gpu_for_supported_nodes()`，在 scheduler reset 后重新建立映射。缓存方案只在首次建立映射，后续调用丢失了 GPU 分配信息。

即使在每次 `model_infer` 中重新调用 `prefer_gpu_for_supported_nodes()`，仍然无法解决问题。推测 `ggml_backend_sched` 在处理持久化 graph context 时存在其他兼容性问题（例如 tensor buffer 指针在 `reset` 后变为悬空指针）。

### 2.3 当前状态

**已完全回退**。`model.h` 和 `model.cpp` 恢复到优化前状态。此优化需要更深入理解 `ggml_backend_sched` 的内部生命周期管理才能安全实现。

## 3. 未优化的同步点（保持现状）

以下同步点属于 ggml 后端架构的必要组成，不应在模型层移除：

1. **`ggml_backend_tensor_set` 后的 sync** — 确保 H2D 拷贝完成后才能开始计算
2. **`ggml_backend_sched_synchronize` 在计算后** — 确保所有 GPU 工作完成后才能读输出
3. **`ggml_backend_tensor_get` 前的 sync** — 确保 D2H 拷贝读到正确数据

## 4. 进一步优化方向

### 4.1 正确实现图缓存

需要研究 `ggml_gallocr` 和 `ggml_backend_sched` 的精确语义：
- `reset` 后哪些状态被清除
- 如何安全地在 `reset` → `alloc_graph` 周期中复用同一个图
- 是否需要手动管理 tensor buffer assignment

### 4.2 CUDA Events 替代 Stream Sync（ggml 层面）

ggml 后端可以使用 `cudaEventRecord` + `cudaStreamWaitEvent` 替代 `cudaStreamSynchronize`。

### 4.3 CUDA Graphs（未来）

对于固定 shape 的推理，CUDA Graph capture 可以消除每次 launch 的 kernel dispatch 开销。
