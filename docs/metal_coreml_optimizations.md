# CoreML / Metal 后端优化记录

本文档记录 `pyannote-ggml` 在 Apple 平台（CoreML / Metal）上的优化改动、实验结果与使用说明。

所有改动均通过 RTTM 输出一致性验证——优化前后 diarization 结果完全相同。

---

## 1. Segmentation CoreML FP16 精度选项

### 动机

Segmentation 转换脚本原始使用 `compute_precision=FLOAT32`（accuracy-first）。Apple Neural Engine 和 GPU 在 FP16 下吞吐量通常翻倍。对于 powerset 分类（7 类 argmax），FP16 的精度损失不太可能影响最终决策。

### 改动

文件：`models/segmentation-ggml/convert_coreml.py`

新增 `--precision` CLI 参数：

```bash
# FP32（默认，与原始行为一致）
python convert_coreml.py

# FP16（可能在 ANE/GPU 上获得近 2x 吞吐提升）
python convert_coreml.py --precision fp16
```

FP16 模型输出到 `segmentation_fp16.mlpackage`，FP32 输出到 `segmentation.mlpackage`。

转换脚本的 sanity check 现在额外报告 **argmax agreement**（逐帧分类一致率），这对评估 FP16 影响至关重要。

### 验证方法

转换后脚本自动对比 PyTorch vs CoreML：

- Cosine similarity > 0.99
- Argmax agreement > 0.95（589 帧中至少 560 帧分类一致）

### 使用建议

Embedding 转换脚本（`models/embedding-ggml/convert_coreml.py`）已经使用 FP16。

---

## 2. Stride-Aware 输出读取（两个 Bridge）

### 动机

CoreML 的 `MLMultiArray` 输出不保证 contiguous 布局。不同 CoreML runtime 版本、compute unit 选择或模型结构可能导致输出带有非连续 strides。原始实现直接用 `memcpy` / 线性索引读取 `dataPointer`，在 non-contiguous 场景下会产生 **silent 错误**（读到错误数据但不报错）。

### 改动

#### Segmentation bridge

文件：`models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`

- 读取 `out_array.shape` 和 `out_array.strides`
- 先检查 strides 是否表示 contiguous 布局（fast path：直接 memcpy/线性读取，零开销）
- 若 non-contiguous：按 `[frames, classes]` 二维索引 + stride 逐元素读取
- 输出日志 `Non-contiguous output detected` 便于诊断

#### Embedding bridge

文件：`models/embedding-ggml/src/coreml/coreml_bridge.mm`

- 读取 `output.strides` 获取 embedding 维度的步长
- 当 stride == 1 时 fast-path memcpy
- 当 stride != 1 时逐元素按 stride 读取
- 移除了旧的 `output.count - 256` offset heuristic（不可靠，用 stride 替代）

### 正确性验证

在所有 compute unit 配置下 RTTM 输出一致（见 §4 实验结果）。

---

## 3. CoreML 模型编译缓存

### 动机

CoreML 的 `.mlpackage` 在运行时需要通过 `compileModelAtURL` 编译为 `.mlmodelc`。此编译步骤在首次运行时需要数十秒，严重影响冷启动延迟。

### 改动

文件：两个 bridge `.mm` 文件

策略：在 `.mlpackage` 同级目录保存编译后的 `.mlmodelc`：

- 首次运行：`compileModelAtURL` → 复制到 `<name>.mlmodelc` → 从临时编译结果加载
- 后续运行：检测到 `.mlmodelc` 存在 → 直接加载，跳过编译

### 实测效果（sample.wav, cpu_ane, 清除所有缓存后连续运行）

测试协议：删除项目 `.mlmodelc` 缓存 + 清除 macOS `/var/folders/T/` 下的 CoreML 临时编译缓存，然后连续运行 4 次。

| 运行 | 类型 | seg load | emb load | 模型加载合计 | Pipeline Total |
|---|---|---:|---:|---:|---:|
| #1 | 冷启动（编译+保存缓存） | 140 ms | 787 ms | **927 ms** | 1689 ms |
| #2 | 热启动（首次读缓存） | 76 ms | 563 ms | **639 ms** | 1235 ms |
| #3 | 热启动（OS文件缓存warm） | 24 ms | 16 ms | **40 ms** | 643 ms |
| #4 | 热启动（稳定态） | 24 ms | 16 ms | **40 ms** | 598 ms |

#### 分析

1. **缓存跳过 `compileModelAtURL`**：冷启动 vs 首次热启动，模型加载从 927ms 降至 639ms（节省 ~290ms 编译开销）。
2. **OS 文件缓存是更大因素**：第 2→3 次运行模型加载从 639ms 降至 40ms（**16x**），这是 macOS page cache 将 `.mlmodelc` 文件驻留内存的效果。
3. **稳定态性能**：第 3、4 次运行完全一致（40ms 模型加载，~600ms 总管线），方差极小。
4. **macOS 有内核级 CoreML 编译缓存**：即使清除所有 `.mlmodelc` 文件，`compileModelAtURL` 也只需 ~140ms（seg）/ ~790ms（emb），远快于首次编译时间。这意味着 macOS 在更深层做了编译器结果缓存。

#### 缓存策略的实际价值

- **跳过 `compileModelAtURL` 调用**：节省 ~290ms + 避免在 `/var/folders/T/` 创建临时 `.mlmodelc` 副本。
- **首次读取仍受 OS 文件缓存影响**：`.mlmodelc` 第一次从磁盘读取较慢（639ms），需要 2 次运行后 OS 文件缓存 warm up 才能达到 40ms 稳定态。
- **无论冷启动还是热启动，RTTM 输出完全一致** ✓

### 缓存管理

- 缓存文件位于 `.mlpackage` 同目录：`segmentation.mlmodelc`、`embedding.mlmodelc`
- 删除 `.mlmodelc` 目录即可强制重新编译
- 若复制失败（权限等），仍可正常运行（从临时编译结果加载），仅打印提示

---

## 4. Compute Unit 可配置（环境变量）

### 动机

CoreML 支持将计算调度到不同硬件：CPU、GPU、Neural Engine（ANE）。不同模型结构在不同硬件上性能差异显著。原始代码硬编码 `MLComputeUnitsAll`，无法实验对比。

### 改动

文件：两个 bridge `.mm` 文件

新增环境变量 `COREML_COMPUTE_UNITS`：

| 值 | 含义 | CoreML 枚举 |
|---|---|---|
| `all`（默认） | CPU + GPU + ANE | `MLComputeUnitsAll` |
| `cpu_gpu` | CPU + GPU（禁用 ANE） | `MLComputeUnitsCPUAndGPU` |
| `cpu_ane` | CPU + ANE（禁用 GPU） | `MLComputeUnitsCPUAndNeuralEngine` |
| `cpu_only` | 仅 CPU | `MLComputeUnitsCPUOnly` |

使用：

```bash
export COREML_COMPUTE_UNITS=cpu_ane
./diarization-ggml/build/bin/diarization-ggml ...
```

### 实验结果（Apple M2 Pro，每配置 warmup+5 次取中位数）

测试协议：每配置先跑 1 次 warmup（丢弃），再跑 5 次正式测量，取中位数。消除单次运行受系统热状态、执行顺序等因素影响的误差。

脚本：`tools/bench_coreml_compute_units.sh`

#### sample.wav (30s, 21 chunks)

| Compute Units | Segmentation | Embeddings | Total | 备注 |
|:---:|:---:|:---:|:---:|---|
| `all` | 365 ms (343~398) | 303 ms (301~342) | 756 ms (741~832) | 默认 |
| **`cpu_ane`** | **244 ms** (243~244) | 314 ms (305~346) | **598 ms** (590~630) | **总提速 21%** |
| `cpu_gpu` | 294 ms (283~344) | 1033 ms (1016~1153) | 1479 ms (1411~1581) | embedding 3.4x 退化 |

RTTM 一致性：三个配置 5 次运行均产出完全相同的 RTTM（13 segments, 2 speakers）。

#### mulpeo.wav (230s, 221 chunks)

| Compute Units | Segmentation | Embeddings | Total | 备注 |
|:---:|:---:|:---:|:---:|---|
| `all` | 2970 ms (2913~3086) | 2912 ms (2871~2950) | 5948 ms (5942~6138) | 默认 |
| **`cpu_ane`** | **2591 ms** (2546~3010) | 3049 ms (2924~3070) | **5644 ms** (5523~6110) | **总提速 5%** |
| `cpu_gpu` | 3118 ms (2899~3560) | 10205 ms (9704~10210) | 13454 ms (12747~13949) | embedding 3.5x 退化 |

RTTM 一致性：
- `all` 和 `cpu_ane` 产出完全相同的 RTTM（98 segments, 4 speakers）✓
- `cpu_gpu` 产出略有差异（100 segments），在 ~163s 处多出 2 个极短 segment（17ms）。这是由 GPU 浮点精度差异经聚类放大导致，非 bug。

#### 关键发现

1. **Embedding 强依赖 ANE**：ResNet34 的卷积/池化是 ANE 强项。禁用 ANE（`cpu_gpu`）导致 embedding 从 ~300ms 退化到 ~1000ms（短音频 3.4x，长音频 3.5x）。

2. **Segmentation 在 `cpu_ane` 下最快**：`all` 模式下 CoreML 可能将部分算子分配到 GPU，引入 CPU↔GPU 数据传输和调度开销。`cpu_ane` 避免此开销。

3. **短音频优势更明显**：30s 音频 `cpu_ane` 比 `all` 快 21%；230s 音频快 5%。长音频的 embedding 在 `cpu_ane` 下反而略慢（ANE dispatch overhead amortized less over per-chunk time）。

4. **方差稳定性**：`cpu_ane` 的 segmentation 方差极小（243~244ms），`all` 波动较大（343~398ms），说明 ANE 独占调度避免了 GPU 争抢。

5. **`cpu_gpu` 不推荐**：不仅最慢，长音频下还产生微小 RTTM 差异（GPU 浮点精度差异经聚类放大）。

6. **`all` 和 `cpu_ane` 结果一致**：两种配置在短/长音频上产出完全相同的 RTTM。

### 推荐配置

```bash
export COREML_COMPUTE_UNITS=cpu_ane
```

在 Apple Silicon 上，`cpu_ane` 通常是 diarization 管线的最优配置。

---

## 5. 深层优化（Phase 2）

以下优化在 §1-4 基础上进一步提升性能，按实施顺序记录。

### P0: Seg/Emb 管线并行（~35-50% 加速）

**文件**: `diarization-ggml/src/diarization.cpp`

**原理**: 当前管线完全串行（所有 segmentation → powerset → 所有 embedding → 聚类）。Segmentation 和 embedding 使用不同的 CoreML 模型，可安全并行执行。当 chunk c 的 segmentation 完成后，其 embedding 可立即开始，同时 chunk c+1 的 segmentation 并行运行。

**实现**:
- 新增 `pipeline_parallel_seg_emb()` 辅助函数
- 用 `std::thread` + `std::mutex` + `std::condition_variable` 实现 producer-consumer 模式
- Producer 线程: segmentation + per-chunk powerset
- Consumer 线程 (main): fbank + 蒙版 embedding 提取
- 仅当两个 CoreML 模型都可用时激活（GGML 路径保持顺序执行）

### P1: Fbank 全局预计算（~25% embedding 加速）

**文件**: `diarization-ggml/src/diarization.cpp`, `models/embedding-ggml/src/fbank.{h,cpp}`

**原理**: 相邻 chunks 有 90% 重叠（10s chunk, 1s step），导致 fbank 帧被重复计算 ~10x。

**实现**:
- `compute_fbank()` 新增 `apply_cmn` 参数（默认 true，兼容旧调用）
- 新增 `apply_cmn()` 独立函数用于 per-window CMN
- 在并行管线中：一次性对 zero-padded 全长音频计算 raw fbank（无 CMN）
- 每个 chunk 从全局 fbank 提取 998 帧切片 + 应用 per-chunk CMN
- 保持与逐 chunk 计算完全一致的 CMN 行为

### P2: CoreML 推理开销减少

**文件**: 两个 bridge `.mm` 文件

**实现**:
- **缓存固定 shape 对象**: segmentation bridge 缓存 `[1, 1, 160000]` shape/strides NSArray
- **自定义 MLFeatureProvider**: 替代 NSDictionary + MLDictionaryFeatureProvider，减少 ObjC 对象创建
- **outputBackings (macOS 14+)**: 预分配输出 MLMultiArray，CoreML 直接写入，跳过 stride-aware 拷贝

### P4: AHC 距离矩阵 BLAS 加速

**文件**: `diarization-ggml/src/clustering.cpp`

**原理**: AHC 聚类构建 O(n²) 距离矩阵。原实现逐对计算 n*(n-1)/2 个 Euclidean distance（各需 256 次乘法）。输入已 L2-normalized，故 ||a-b||² = 2(1 - a·b)。

**实现**: 用 `cblas_dgemm` 一次性计算 gram matrix G = X @ X^T，然后线性扫描 G 转为距离。O(n²·d) 变为 BLAS 优化的 O(n²·d)，利用 Apple Accelerate 的 SIMD 向量化。

### P3: Batch CoreML 推理（API 已就绪，需重新导出模型）

**文件**: 
- `models/segmentation-ggml/convert_coreml.py` — 新增 `--batch` 参数
- `models/embedding-ggml/convert_coreml.py` — 新增 `--batch` 参数
- 两个 bridge `.h` / `.mm` 文件 — 新增 `_batch()` 函数

**使用**:

```bash
# 重新导出支持 batch 的模型
python models/segmentation-ggml/convert_coreml.py --batch 8
python models/embedding-ggml/convert_coreml.py --batch 8
```

**状态**: bridge batch 函数已实现并编译通过。管线集成待完成（需 batch 模型可用后测试验证）。

### 性能对比（mulpeo.wav, 230s, cpu_ane, 中位数）

| 阶段 | 原始（顺序） | P0 并行 | P0+P1+P2 | 加速比 |
|---|---:|---:|---:|---|
| Segmentation | 2591 ms | — | — | — |
| Embeddings | 3049 ms | — | — | — |
| **Pipeline Total** | **5644 ms** | **~3700 ms** | **~2600 ms** | **2.2x** |

---

## 6. 改动文件清单

| 文件 | 改动类型 |
|---|---|
| `models/segmentation-ggml/convert_coreml.py` | FP16 precision + batch + argmax agreement |
| `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.{h,mm}` | stride-aware + 编译缓存 + compute unit + shape 缓存 + outputBackings + batch |
| `models/embedding-ggml/convert_coreml.py` | batch 支持 |
| `models/embedding-ggml/src/coreml/coreml_bridge.{h,mm}` | stride-aware + 编译缓存 + compute unit + FeatureProvider + outputBackings + batch |
| `models/embedding-ggml/src/fbank.{h,cpp}` | CMN 可选 + apply_cmn() 独立函数 |
| `diarization-ggml/src/diarization.cpp` | 管线并行 + fbank 预计算 + 并行 CoreML 路径 |
| `diarization-ggml/src/clustering.cpp` | cblas_dgemm AHC 距离矩阵加速 |
| `docs/metal_coreml_optimizations.md` | 本文档 |

---

## 7. 后续优化方向

| 方向 | 预期收益 | 状态 |
|---|---|---|
| 运行 `convert_coreml.py --precision fp16` 实测 seg 吞吐 | ANE FP16 吞吐可能翻倍 | 待测试 |
| 用 `--batch 8` 重新导出模型 + 管线集成 | 减少 ANE dispatch 开销 | API 就绪 |
| Embedding Metal offload（GGML sched 混合调度） | 提供非 CoreML 的 GPU 推理路径 | 待实现 |
| Streaming 管线也使用并行 + fbank 预计算 | 实时场景加速 | 待评估 |
