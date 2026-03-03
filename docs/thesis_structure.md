# 毕业论文整体结构建议（面向学术硕士）

本文档给出一份“学术化、可论证、有创新点”的毕业论文整体结构，覆盖：

- 第三章：Whisper 混合精度量化与代价建模（对应 `/home/aim/project/reverse/mse.py` 与 `/home/aim/project/reverse/score.py` 的方法链条）
- 第四章：将 PyTorch 版 Pyannote diarization 模型**重构为 GGML 实现**，并进一步在 CUDA/CoreML 后端做算子级实现与优化（本仓库 `diarization-ggml/`、`models/`、`whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`）
- 第五章（可选但强烈建议）：将“语音转文字 + 说话人识别”整合为可运行的端侧原型（`./diarization-ggml/build-x86-cuda/bin/transcribe` 路线），但写法上强调**系统化研究问题**与**可验证机制**，避免变成“实现了一个系统”的工程报告。

目标读者：计算机学术硕士论文评审。写作基调：以“问题定义 -> 方法/机制 -> 实现细节 -> 可验证证据链 -> 讨论局限”为主，避免以功能堆砌为主。

---

## 0. 题目候选（推荐 2 个主选 + 1 个备选）

### 主选 1（强调非对称瓶颈 + 两类创新）

面向端侧异构语音管线的非对称瓶颈优化：Whisper 混合精度量化与基于 GGML 的说话人日志算子重构

### 主选 2（强调“从 PyTorch 到 GGML”的学术贡献）

面向端侧部署的说话人日志推理重构：从 PyTorch 计算图到 GGML 算子级实现与异构后端映射

### 备选（更系统论文味，但需要你后续评估章节支撑）

端侧长音频理解的软硬协同优化：量化代价建模与异构算子映射

---

## 1. 一句话总叙事（放在摘要/绪论开头）

端侧长音频理解包含生成式转写与判别式说话人日志两条子管线，其瓶颈呈现显著非对称：Whisper 更易受访存墙约束，而 diarization 更受热点算子并行度与后端映射能力约束。本文提出一套软硬协同优化思路：以代价感知混合精度量化压缩 Whisper 的数据搬运量，并将 PyTorch 版 diarization 模型重构为 GGML 计算图及算子实现，进一步在 CUDA/CoreML 上对关键算子进行专用实现以释放异构加速器吞吐。

---

## 2. 建议明确写出的“学术贡献点”（答辩/评审最关心）

建议在绪论末尾以 4–6 条列出，且每条都能落到“机制 + 证据”。

1) **Whisper 侧：代价感知混合精度量化策略**
   - 机制：基于 GGML 块级量化数学模拟 + logits-MSE 敏感度画像 + 鲁棒代价映射 + 多目标收益模型，生成分层混合位宽配置。
   - 证据链：`/home/aim/project/reverse/mse.py`（逐层 MSE 测量与 CSV）+ `/home/aim/project/reverse/score.py`（分位数截断代价、收益模型、位宽选择）。

2) **Diarization 侧：从 PyTorch 到 GGML 的模型重构（工作量必须学术化表达）**
   - 机制：把 PyanNet segmentation 与 WeSpeaker embedding 的 PyTorch 运算图映射为 GGML 张量形状与算子序列（conv/norm/lstm/linear/softmax 等），形成可调度、可离线/流式推理的图。
   - 学术表述建议：将其表述为“计算图翻译/图语义对齐（Computational Graph Translation with Semantic Equivalence）”而不是“重写了代码”。
   - 证据链：`models/segmentation-ggml/src/*.cpp`、`models/embedding-ggml/src/*.cpp`、`diarization-ggml/src/*.cpp`。

3) **算子级创新：Segmentation BiLSTM 的 CUDA cooperative 递推算子实现**
   - 机制：在 `GGML_OP_CUSTOM` 上构建 contract-based dispatch，并采用“cuBLAS 输入投影 + cooperative-groups grid.sync 递推”的并行化路径（`export DIARIZATION_SEG_LSTM_COOP=1`）。
   - 证据链：`whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` + 说明文档 `docs/segmentation_cuda_lstm_coop_route.md`。

4) **CoreML 后端工程机制：零拷贝绑定与 stride-aware 输出契约**
   - 机制：在 Objective-C bridge 中用 `MLMultiArray` 的显式 shape/stride 建立输入输出内存契约，避免隐式 contiguous 假设导致的 silent 错误。
   - 证据链：segmentation bridge（stride-aware）`models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`；embedding bridge 当前为 contiguous 假设，可作为局限与改进方向。

5) （可选）**端到端异构管线原型：转写 + 说话人日志的可运行集成与资源协同**
   - 机制：将两条子管线在 host 侧做数据接力、缓存与调度，给出可复现实验入口（`transcribe`）。
   - 学术表述建议：将其写成“端侧长音频异构推理的系统化实验平台（prototype for evaluation and co-design）”，而不是“产品系统”。

---

## 3. 推荐论文目录（含关键过渡）

### 第 1 章 绪论

- 1.1 端侧长音频理解的挑战：延迟、峰值内存、异构硬件
- 1.2 研究对象：Whisper 转写 + Pyannote diarization（segmentation + embedding + clustering）
- 1.3 研究问题：为何需要非对称优化（访存 vs 算子/后端映射）
- 1.4 本文贡献（对应第 2 节的 4–6 条）
- 1.5 章节安排

写作要点：绪论中要强调“创新点不是系统功能，而是可复用的优化机制与实现证据”。

### 第 2 章 背景与相关工作

- 2.1 Whisper：Encoder–Decoder、自回归推理与端侧代价来源（访存/缓存/算子密度）
- 2.2 GGML 量化与块级存储格式：Block-wise scaling、meta-data、访存行为
- 2.3 Diarization：segmentation/embedding 的建模与推理形态（滑窗、长序列）
- 2.4 异构推理后端：CUDA、CoreML/ANE、GGML scheduler
- 2.5 相关工作：混合精度量化（HAWQ 等）、端侧推理系统、算子级优化与自定义 kernel

写作要点：这章的任务是把第 3/4/5 章“看似不同的技术”放到统一的系统优化语境中。

### 第 3 章 面向端侧访存瓶颈的 Whisper 混合精度量化策略与代价建模

- 3.1 块级量化理论与 GGML 张量映射（Q8_0/Q5_0/Q4_0/Q2_K）
- 3.2 量化误差敏感度评估：逐层替换与 logits-MSE（校准集与流程）
- 3.3 多目标收益模型：延迟收益 + 内存收益（端侧画像/查表/归一化）
- 3.4 鲁棒代价建模：`log10(MSE)` 分位数截断与区间映射
- 3.5 位宽分配算法：候选集合上的评分/贪心选择与配置生成
- 3.6 小结：从“数据体积/带宽”角度缓解访存墙，为 diarization 释放资源预算

关键过渡（3 -> 4）建议放在 3.6 末尾：

> 第三章通过压缩权重表示降低端侧数据搬运量，使转写子管线在固定资源预算下可运行。然而在同一设备上引入 diarization 后，整体延迟的主导因素不再仅由访存决定，而更多转移到 diarization 模型中的热点算子实现与后端映射效率。因此第四章进一步从“计算图重构与算子级实现”的角度，对 PyTorch 版 diarization 推理进行 GGML 化重构，并针对关键算子设计异构后端实现。

### 第 4 章 面向算子瓶颈的说话人日志推理重构：从 PyTorch 到 GGML 及异构后端实现

这一章要突出两层贡献：

- (A) **PyTorch -> GGML 的计算图翻译与语义对齐**（这是关键工作量）
- (B) 在 GGML 表达基础上，进一步做 **CUDA/CoreML 的算子级实现/优化**

建议章节展开如下：

- 4.1 diarization 端到端流程与张量约定（seg 输出布局、emb 输入布局、滑窗策略）
- 4.2 计算图翻译：segmentation 模型的 GGML 实现
  - 4.2.1 SincNet: conv/pool/norm/leaky_relu 的 GGML 映射与形状定义
  - 4.2.2 BiLSTM: 以 `GGML_OP_CUSTOM` 表达双向多层 LSTM（图规模与中间张量控制）
  - 4.2.3 Linear + classifier + log_softmax 的 GGML 实现与数值对齐策略
- 4.3 计算图翻译：embedding 模型（ResNet34 + TSTP）的 GGML 实现
  - 4.3.1 BN 推理形态融合（scale+shift）与图节点减少
  - 4.3.2 TSTP 统计池化的 GGML 表达：reduction 与数值稳定性
- 4.4 CUDA 后端：segmentation BiLSTM cooperative 递推算子（主路线 `DIARIZATION_SEG_LSTM_COOP=1`）
  - 4.4.1 contract-based dispatch：识别特定 `GGML_OP_CUSTOM` 并接管执行
  - 4.4.2 GEMM+recurrence 分解：cuBLAS 输入投影 + 自定义递推 kernel
  - 4.4.3 cooperative-groups `grid.sync()` 并行化设计与状态 ping-pong
- 4.5 CoreML 后端：静态图转换与运行时桥接
  - 4.5.1 segmentation CoreML：冻结 sinc filters、log_softmax 入图、stride-aware 输出读取
  - 4.5.2 embedding CoreML：变长输入、zero-copy 输入绑定、输出读取约束
- 4.6 小结：GGML 化重构带来的端侧内存/可移植性优势；算子级后端实现带来的吞吐释放

写作要点（确保“学术性而非工程流水账”）：

- 对“PyTorch -> GGML”要写成方法：定义张量布局、算子等价、误差来源与对齐策略（不要写成“把代码翻译了”）。
- 对 CUDA/CoreML 要写成机制：合同、调度、布局契约、同步语义、数据类型选择。

### 第 5 章（可选但建议）端侧异构语音理解原型：转写与说话人日志的联合管线与资源协同

本章的写法关键：必须把 `transcribe` 视为“验证平台/系统化机制”，而不是“产品系统”。

- 5.1 统一任务定义：带说话人标签的转写输出（transcription + diarization alignment）
- 5.2 联合管线的数据流与接口契约（音频缓冲、滑窗、特征复用的可能性）
- 5.3 资源协同机制：内存预算、缓冲复用、异步执行与同步点分析
- 5.4 原型实现与可复现实验入口：`./diarization-ggml/build-x86-cuda/bin/transcribe`
- 5.5 讨论：端到端延迟分解、瓶颈迁移与对第 3/4 章方法的系统级印证

写作要点：把第 5 章写成“系统级讨论章 + 证据章”，强调你提出的机制在完整管线中如何发挥作用，以及哪些开销仍存在（转置、输出拷贝、CPU fallback 等）。

### 第 6 章 结论与展望

- 6.1 工作总结：非对称瓶颈框架下的两类优化与落地证据
- 6.2 局限性：指标选择（logits-MSE 与 WER/DER 的关系）、后端覆盖率、输出布局拷贝
- 6.3 未来工作：embedding CoreML stride-aware、更多算子融合、减少转置/cont、端到端流水线优化

---

## 4. 第四章“GGML 化重构”如何写出学术创新（建议的表达模板）

你需要把重构写成“可复用的方法/框架”，而不是“实现细节”。下面给一个可直接套用的写作框架：

1) **问题定义**：高层框架（PyTorch）在端侧部署存在运行时/内存/后端可控性问题；需要一个可控的计算图表示与后端分派机制。
2) **图翻译方法**：给出算子集合与张量布局约定；说明每个 PyTorch 子模块如何映射到 GGML 图（conv/norm/lstm/linear/pooling）。
3) **语义对齐与数值对齐**：说明如何保证输出语义一致（例如 segmentation 输出需要 frame-major；CoreML bridge stride-aware）；说明误差来源（量化/precision/cast）。
4) **系统收益**：强调“可控内存占用、可移植后端、关键算子可特化”。
5) **算子级创新点**：把 cooperative BiLSTM 写成关键创新算子，提供合同、调度、同步语义与性能动机。

---

## 5. 你可以直接引用的代码证据指针（建议写进论文脚注/附录）

- 第 3 章：
  - `/home/aim/project/reverse/mse.py`（逐层敏感度与 GGML 量化模拟）
  - `/home/aim/project/reverse/score.py`（代价/收益模型与位宽选择）

- 第 4 章：
  - `models/segmentation-ggml/src/sincnet.cpp`
  - `models/segmentation-ggml/src/lstm.cpp`
  - `models/embedding-ggml/src/model.cpp`
  - `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`（`DIARIZATION_SEG_LSTM_COOP=1` cooperative LSTM）
  - `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`（stride-aware output）
  - `models/embedding-ggml/src/coreml/coreml_bridge.mm`（zero-copy input；输出读取约束）

- 第 5 章：
  - `diarization-ggml/src/diarization.cpp`
  - `diarization-ggml/src/streaming.cpp`
  - `diarization-ggml/build-x86-cuda/bin/transcribe`（原型入口，写作中作为实验平台）
