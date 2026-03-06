# Markdown 文档索引（排除 whisper.cpp）

本索引基于仓库 `pyannote-ggml` 生成。

- 范围：所有 `*.md` 文件，但排除 `whisper.cpp/` 子树
- 备注：
  - 以 `session-` 开头的文件通常是历史对话/快照/过程记录。
  - `.sisyphus/` 主要是计划与研发笔记（常有用，但不一定代表当前最终实现）。

## A. 项目入口与通用约定

- `README.md`
  - 项目总览：原生 C++ speaker diarization 管线（pyannote 移植）+ GGML/CoreML；并包含可选的 Whisper 转写集成。
  - 给出构建/模型转换/运行命令；描述离线与流式 API；列出关键常量（10s chunk、1s hop、589 帧、3 个 local speakers、256 维 embedding）。

- `AGENTS.md`
  - 本仓库的 coding agent 指南、repo 地图、构建/测试命令，以及风格约束。

- `DEPLOYMENT_ZH.md`
  - 中文部署指南：macOS/Apple Silicon 环境、Python/coremltools 版本注意、HF token 要求、模型转换与运行示例。

## B. 后端 / 平台运行手册

- `METAL_RUNBOOK.md`
  - Metal 后端构建/运行的最小手册，以及行为说明（未编译 Metal 时 `--backend metal` 会直接报错）。

- `CUDA_CONTEXT.md`
  - CUDA x86 迁移与排障上下文：正确性优先、已知失败/护栏策略、构建目录、下一步诊断计划。

## C. Whisper + Diarization 集成（设计文档）

- `INTEGRATION_PLAN.md`
  - 详细流式集成设计：静音过滤（filtered timeline）、音频缓冲区时间戳、基于 pyannote VAD 帧的分段结束检测、Whisper 分块规则、WhisperX 风格对齐。

- `INTEGRATION_PLAN_ZH.md`
  - `INTEGRATION_PLAN.md` 的中文版本。

- `diarization_whisper_integration.md`
  - 更早的英文设计草案（同主题，但更粗略）。

- `DIARIZATION_WHISPER_INTEGRATION_ZH.md`
  - 更早的中文设计草案。

## D. Streaming diarization 设计与工具

- `diarization-ggml/STREAMING_DESIGN.md`
  - streaming diarization 的设计：周期性 recluster；解释为什么需要保留完整历史（embeddings + binarized）才能做到 offline-identical 的重聚类。

- `tools/streaming-viewer/README.md`
  - Flask 可视化工具：读取 `streaming_test --json-stream` 的 JSON lines（push/recluster/finalize）。

## E. 模型子项目（Segmentation / Embedding）

- `models/segmentation-ggml/README.md`
  - segmentation GGML 实现说明：当前状态、与 PyTorch 的精度对齐结果、build/convert/run/test 入口。

- `models/segmentation-ggml/docs/architecture.md`
  - 超长结构说明：逐层形状/参数，PyanNet（SincNet + 4-layer BiLSTM + MLP + 7 类 powerset）。

- `models/embedding-ggml/kaldi-native-fbank/README.md`
  - 上游依赖说明：Kaldi 兼容在线 fbank 特征提取。

- `models/embedding-ggml/kaldi-native-fbank/cmake/Modules/README.md`
  - 上游注记：FetchContent 模块来源。

- `models/embedding-ggml/kaldi-native-fbank/toolchains/README.md`
  - 上游注记：ARM 交叉编译工具链下载/解包说明。

## F. `docs/` 下的工程 / 论文素材

- `docs/coreml_and_ggml_backend_notes.md`
  - 读码式实现笔记：CoreML 转换 + runtime bridge、GGML CPU/CUDA 图构建、布局/转置点、scheduler 行为。

- `docs/coreml_segmentation_implementation.md`
  - CoreML segmentation 的论文式章节草案：I/O contract、转换 wrapper 决策、zero-copy + stride-aware 输出、验证建议。

- `docs/operator_level_optimizations.md`
  - segmentation 的算子级优化笔记（LSTM recurrence 是主要瓶颈 + 可推进方向）。

- `docs/operator_level_optimizations_fullflow.md`
  - seg+emb+CUDA+CoreML 的全流程算子与布局笔记；包含 segmentation BiLSTM 的 contract-based CUDA custom-op dispatch。

- `docs/segmentation_cuda_lstm_coop_route.md`
  - CUDA cooperative BiLSTM 的实现契约文档（env 开关、shape/type/name 匹配、GEMM+recurrence 分解、kernel launch、grid.sync 语义）。

- `docs/segmentation_cuda_lstm_parallel_optimization.md`
  - 面向论文的完整叙事：profiling 证据 -> kernel 结构重写 -> 验证方法 -> 代表性结果。

- `docs/context_segmentation_cuda_bottleneck_summary.md`
  - 压缩上下文：稳定结论、弃用路线、当前主方向、最小复现/profiling 命令。

- `docs/thesis_structure.md`
  - 中文论文结构建议：如何把“工程重写”写成学术贡献（图翻译、语义对齐、后端契约、布局/stride 契约等）。

- `docs/thesis_ch4_ch5_materials_to_extract.md`
  - 写论文第 4/5 章建议从代码提取的材料清单，附大量文件/函数指针。

- `docs/cuda_fbank_precompute_optimization.md`
  - CUDA/GGML 路径 fbank 全局预计算优化：消除 ~10x 冗余 FFT/mel 计算（从 CoreML 路径移植）。

- `docs/cuda_pipeline_parallel_optimization.md`
  - CUDA/GGML 路径 Seg/Emb 管线并行优化：producer-consumer 双线程，CUDA 并发分析。

- `docs/cuda_sync_overhead_optimization.md`
  - CUDA 同步开销分析：segmentation 计算图缓存尝试（已回退），ggml scheduler 同步点分析。

- `docs/cuda_embedding_correctness_fix.md`
  - CUDA embedding 正确性诊断工具：`DIARIZATION_EMB_DEBUG` 环境变量，L2/NaN/Inf 检查，过滤前汇总。

- `docs/cuda_benchmark_results.md`
  - CUDA 优化基准测试结果：Tesla T4 上基线 vs 优化版对比（Embedding -10.4%, Total -7.3%），各优化项状态总结。

## G. 学术正文草稿

- `CHAPTER4_ACADEMIC.md`
  - 第四章正文式草稿：PyTorch->GGML 静态图、算子映射、内存池/生命周期、PLDA+AHC+VBx 复杂度叙事等。

## H. 归档实验

- `notes-cudnn-experiment.md`
  - 归档实验：尝试用 cuDNN 替换 CUDA LSTM custom path；因为端到端对数值差异敏感，最后以 patch 形式保留。

## I. Session / 历史记录（通常是历史信息）

- `session-2026-03-02-archive.md`
  - CUDA 工作状态快照与 downstream whisper.cpp patch 线索。

- `session-changshigpu.md`
  - 关于强制 segmentation 上 CUDA 的尝试与回退等对话记录。

- `session-ses_3572_huijia.md`
  - 对话记录：包含 seg dump 对比工具等实验过程。

- `session-ses_35b4.md`
  - 更早的对话记录，含环境噪音（路径/LSP）。

- `session-论文.md`
  - session 壳/残留，信息密度低。

## J. Node 绑定

- `bindings/node/packages/pyannote-cpp-node/README.md`
  - Node.js 原生扩展 API：共享模型缓存、offline/one-shot/streaming session 模式、事件语义、TypeScript 类型。

## K. 第三方 / 构建产物 README（低优先级）

- `diarization-ggml/build/_deps/kissfft-src/README.md`
- `diarization-ggml/build-linux/_deps/kissfft-src/README.md`
- `diarization-ggml/build-x86-cuda/_deps/kissfft-src/README.md`
  - KISS FFT 上游 README（由构建依赖带入）。

---

# 压缩上下文（内容丰富版）

本节是为后续任务准备的“高信号上下文胶囊”，用于快速对齐目标与关键约束。

## 系统目标

- 将 `pyannote/speaker-diarization-community-1` 的 diarization pipeline（segmentation + embedding + clustering）移植到原生 C++，用 GGML 表达计算图并支持多后端（CPU/CUDA/Metal/CoreML）。
- 可选：将 Whisper 转写与 streaming diarization 集成，输出带说话人标签的转写（word/segment 时间戳）。

## 关键常量 / 数据契约

- 音频：16 kHz mono float samples。
- Segmentation：
  - 滑窗：10s chunk（`160000` samples），1s hop（`16000` samples）。
  - 输出：每个 10s chunk 输出 `589` 帧。
  - Powerset：7 类 log-probabilities -> 确定性映射到 3 个 local speaker 活跃轨迹。
- Embedding：
  - 特征：fbank（80 bins）+ ResNet34 + TSTP pooling。
  - 输出：256 维 embedding；无效/静音轨迹可用 NaN 标记。
- 聚类：
  - 先按“clean（非重叠）活跃比例”过滤 embedding。
  - PLDA 变换（256 -> 128）+ AHC 初始化 + VBx 精炼。
  - 每个 chunk 内用 Hungarian 做 constrained assignment：将 3 个 local speakers 映射到 K 个全局 cluster。

## Streaming diarization 的状态不变式

- 为了支持周期性 recluster 并能“回溯修正过去标签”，streaming 必须保留完整历史：
  - embeddings（永久增长）
  - binarized（永久增长，保存每 chunk 每帧的 local 活跃情况）
- 只有 segmentation 推理需要最近 10s 音频滑窗；但 recluster 需要全历史数据。

## 最容易出错 / 最重要的工程契约

- 布局/stride 正确性是第一优先级：
  - powerset 解码期望“每帧连续的 7 类”访问（frame-major）。
  - 历史上出现过将布局错配误诊为“精度问题”的情况；真实原因是 `[class, frame]` vs `[frame, class]` 的 layout mismatch。
- CoreML bridge 的鲁棒性：
  - segmentation bridge 已做 stride-aware 输出读取，避免 CoreML 返回非 contiguous 造成 silent 错误。
  - embedding bridge 目前更依赖 contiguous 假设（潜在鲁棒性缺口/未来工作点）。
- 输入转置拷贝是热点：
  - embedding 的 fbank 往往需要 transpose 拷贝以适配 GGML 布局，属于性能与内存带宽开销点。

## CUDA 现实与优化叙事

- CUDA 上 segmentation 的瓶颈几乎完全在 BiLSTM recurrence custom op。
- 仓库记录了一条 CUDA fast-path：
  - contract-based dispatch（仅匹配特定 `GGML_OP_CUSTOM` 形状/类型/命名）
  - cuBLAS 做输入投影（`W_ih * X`）
  - cooperative-groups recurrence kernel：每步 `grid.sync()`，并行 hidden 单元但保持时间步依赖
  - 通过 `DIARIZATION_SEG_LSTM_COOP=1` 启用
- 正确性验证优先用 logits/log-probabilities 级别（max/mean abs diff + argmax agreement），因为 RTTM/DER 对阈值和离散决策非常敏感。
- CUDA 端到端正确性可能受权重放置/scheduler 拆分影响，`CUDA_CONTEXT.md` 记录了保守护栏与下一步诊断方向。

## Whisper + diarization 集成（流式）

- 采用 filtered timeline：先用 VAD 静音过滤压缩长静音到约 2s（并保持自然过渡）。
- 分段结束检测：使用 pyannote streaming VAD 帧 + 累积帧计数，保证时间戳精确（自然处理 59/60 帧交替）。
- Whisper 分块规则：
  - segment-end 且缓冲区 >= 20s 才送 Whisper
  - silence flush 与 finalize 强制送出缓冲区内容
- 对齐：WhisperX 风格最大交叠分配 speaker，gap 用最近 segment fallback。

## Node 绑定

- `pyannote-cpp-node` 封装 integrated pipeline：
  - 共享 model cache（一次加载，多次复用）
  - offline / one-shot / streaming session API
  - session 以事件形式发出累计 `segments` 更新
