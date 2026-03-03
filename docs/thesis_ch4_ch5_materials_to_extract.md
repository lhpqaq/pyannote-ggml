# 第四章/第五章写作需要从代码提取的材料清单（基于读码整理）

本文件列出撰写毕业论文第 4 章（PyTorch -> GGML 重构 + CUDA/CoreML 算子实现）与第 5 章（可选：转写+说话人识别联合管线）时，建议从代码中抽取/整理的“可引用材料”。

目标：让论文内容具备学术性与可验证性。

每条材料都给出：需要提取的内容类型、为什么需要、以及代码证据位置（文件路径/函数）。

---

## A. 第四章：PyTorch -> GGML 重构 + 异构后端算子实现

### A1. 模型与算子语义对齐材料（必须有）

你需要从代码中提取“GGML 版本与 PyTorch 版本语义一致”的证据链条，避免被质疑为纯工程重写。

- **算子映射表（PyTorch module -> GGML op 序列）**
  - Segmentation（PyanNet）：SincNet / BiLSTM / Linear / Classifier
  - Embedding（WeSpeaker ResNet34）：Conv2d+BN+ReLU+Residual / TSTP / Linear
  - 证据位置：
    - `models/segmentation-ggml/src/model.cpp`：`model_forward()` 调用链
    - `models/segmentation-ggml/src/sincnet.cpp`：`sincnet_forward()`
    - `models/segmentation-ggml/src/lstm.cpp`：`lstm_forward()` 与 `ggml_custom_4d(...)`
    - `models/embedding-ggml/src/model.cpp`：`model_forward()`、`basic_block()`、`batch_norm_2d()`、TSTP 实现

- **张量形状/布局约定（最容易写错，论文必须明确）**
  - Segmentation：输入 waveform 固定为 `[160000, 1, 1]`（10s@16kHz）；输出 logits 目标语义为 frame-major `[589, 7]`
  - Embedding：输入 fbank 为 `(T, 80)`；GGML 图输入为 `[T, 80, 1, 1]`（ne[0]=T 连续）
  - Diarization 层面还涉及 powerset mapping：7 类 powerset -> 3 说话人多标签
  - 证据位置：
    - `models/segmentation-ggml/src/model.cpp`：`build_graph()` 创建 `waveform` 输入张量
    - `diarization-ggml/src/streaming.cpp`：`CHUNK_SAMPLES=160000`、`FRAMES_PER_CHUNK=589`、`NUM_POWERSET_CLASSES=7`
    - `diarization-ggml/src/powerset.cpp`：`powerset_to_multilabel()` 7->3 映射矩阵

- **连续性/转置/拷贝（contiguity constraints）清单**
  - 目的：解释为什么 GGML 图里大量出现 `permute` / `transpose` / `cont`，以及这些操作会带来哪些额外代价。
  - 证据位置：
    - Seg classifier：`models/segmentation-ggml/src/model.cpp`：`classifier_forward()` 中 `ggml_permute` + `ggml_cont` + `ggml_transpose` + `ggml_cont`
    - Embedding TSTP：`models/embedding-ggml/src/model.cpp`：`ggml_permute` + `ggml_cont` 构建 reduction 计算

写作建议：在 4.2/4.3 的开头做一张“形状表 + 布局表”，并给出“GGML 的 ne[i] 与 PyTorch 的维度对应关系”。

---

### A2. GGML 图构建与调度机制材料（把工程写成系统机制）

这部分用于支撑“为什么重构到 GGML 会带来端侧收益”，核心是：显式图、显式后端、显式 buffer 管理。

- **graph building 的纯函数模式与 no_alloc 元数据上下文**
  - 证据位置：
    - `models/segmentation-ggml/src/model.cpp`：`build_graph()` 使用 `ggml_init_params{..., .no_alloc=true}` + `ggml_new_graph_custom`
    - `models/embedding-ggml/src/model.cpp`：`build_graph()` 同模式（按 num_frames 构图）

- **backend scheduler 的分配/执行/重置流程**
  - 你需要从代码提取一段“统一的调度流程”，用于论文中的执行模型描述。
  - 证据位置：
    - `models/segmentation-ggml/src/model.cpp`：`model_infer()`
      - `prefer_gpu_for_supported_nodes()`
      - `ggml_backend_sched_alloc_graph()`
      - `ggml_backend_tensor_set()`
      - `ggml_backend_sched_graph_compute()`
      - `ggml_backend_tensor_get()`
      - `ggml_backend_sched_reset()`
    - `models/embedding-ggml/src/model.cpp`：`model_infer()` 同模式

- **权重后端分配与模型缓存（减少重复初始化）**
  - 证据位置：
    - `diarization-ggml/src/model_cache.cpp`：`model_cache_load()`
      - 可同时加载 CoreML segmentation/embedding、GGML segmentation/embedding、PLDA、Whisper、VAD
      - 体现“后端可插拔”和“缓存复用”的系统设计
    - `diarization-ggml/src/pipeline.cpp`：`pipeline_init_with_cache()` 借用 cache 里的模型/上下文

---

### A3. CUDA 算子级实现材料（第四章硬核创新点）

重点是 segmentation BiLSTM 的 cooperative 递推路线（你的主路线 `export DIARIZATION_SEG_LSTM_COOP=1`）。

- **custom op 合同（contract）与分派条件**
  - 你需要从代码中抽取：op 类型、src 数量、权重类型、bias 类型、name 前缀匹配、shape 检查。
  - 证据位置：
    - `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`：`ggml_cuda_is_pyannote_seg_lstm_custom()`

- **算子分解：cuBLAS 输入投影 + 自定义 recurrence**
  - 需要抽取：IH buffer 的形状/布局、G=4H、以及为何要把 X 从 FP32 cast 到 FP16。
  - 证据位置：
    - `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`：`ggml_cuda_pyannote_seg_lstm_custom()` 中 `k_f32_to_f16` + `cublasGemmEx`/`cublasGemmStridedBatchedEx`

- **cooperative-groups kernel 设计要点**
  - 需要抽取：grid/block 配置、hidden 维度分块、`grid.sync()` 每步同步、state ping-pong buffer。
  - 证据位置：
    - `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`：`k_pyannote_seg_lstm_dir_coop<...>` 与 `cudaLaunchCooperativeKernel` 调用处
  - 环境开关：`DIARIZATION_SEG_LSTM_COOP`

- **主路线限定条件（论文要写清楚）**
  - cooperative kernel 仅在 B==1 且设备支持 cooperative launch 时启用。
  - 证据位置：
    - `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`：`supports_cooperative_launch` + env var 判断 + `B==1`

写作建议：第四章建议单列一个“实现契约表”：列出该 custom op 的输入/输出 shape、dtype、layout、以及 env 开关。

---

### A4. CoreML 后端材料（第四章另一条异构路线）

你需要从 CoreML 转换脚本与 runtime bridge 中提取“静态化、zero-copy、layout contract”三类材料。

- **转换（conversion）阶段的静态化策略**
  - Segmentation：冻结 sinc filters、permute 替代 einops、log_softmax 入图
  - Embedding：permute、std(unbiased=False)、变长 RangeDim
  - 证据位置：
    - `models/segmentation-ggml/convert_coreml.py`
    - `models/embedding-ggml/convert_coreml.py`

- **runtime bridge 的 zero-copy 输入绑定（shape/stride 明确）**
  - 证据位置：
    - `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`：`initWithDataPointer` + 输入 strides
    - `models/embedding-ggml/src/coreml/coreml_bridge.mm`：同模式

- **segmentation 输出 stride-aware 读取（关键工程正确性材料）**
  - 证据位置：
    - `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`：读取 `out_array.shape` / `out_array.strides` 并按 stride 拷贝

- **embedding 输出读取的布局假设（作为局限/未来工作材料）**
  - 证据位置：
    - `models/embedding-ggml/src/coreml/coreml_bridge.mm`：未读取 strides，使用 `output.count`/offset 逻辑

---

## B. 第五章（可选）：转写 + diarization 的联合管线（把系统写成“研究平台/机制验证”）

第五章写作要点：重点不是“我实现了一个系统”，而是“我构建了一个可重复验证的端侧异构管线原型，用来印证第三/四章提出的机制”。因此需要从代码提取的是“数据流/接口契约/资源协同点/瓶颈位置”。

### B1. 联合管线阶段划分与数据流

你需要从代码中抽取清晰的 stage 定义与串联关系：

- Stage 1：Whisper 全音频转写（offline）或分段转写（streaming）
- Stage 2：Diarization（seg+emb+clustering）
- Stage 3：Alignment（把转写段与说话人段对齐）

证据位置：

- `diarization-ggml/src/offline_pipeline.cpp`：`offline_transcribe()`
  - Step 1 init Whisper -> Step 2 whisper_full -> Step 3 segments -> Step 4 diarize -> Step 5 align
- `diarization-ggml/src/pipeline.cpp`：流式 pipeline（VAD/silence filter、segment detector、transcriber submit、streaming diarization、recluster、align）
- `diarization-ggml/src/main_transcribe.cpp`：CLI/入口，解析参数并调用 offline/streaming

### B2. 资源/缓存协同与“避免重复初始化”的证据

你需要从代码中提取“缓存复用 + 模型一次加载多次使用”的机制材料：

- `diarization-ggml/src/model_cache.cpp`：`model_cache_load()` / `model_cache_free()`
  - 同时缓存：Whisper ctx、VAD ctx、CoreML ctx、GGML models/states、PLDA
- `diarization-ggml/src/pipeline.cpp`：`pipeline_init_with_cache()`
  - 复用 cache 中 whisper_ctx、seg/emb ggml 或 coreml ctx

### B3. 滑窗/分段策略的常量与时间坐标定义（论文需要精确写出来）

这些常量决定了系统吞吐与边界行为，建议直接引用定义。

- `diarization-ggml/src/streaming.cpp`：
  - `CHUNK_SAMPLES=160000`（10s）
  - `STEP_SAMPLES=16000`（1s step）
  - `FRAMES_PER_CHUNK=589`（seg 输出帧数）
  - `FRAME_STEP_SEC=0.016875`（帧步长）

### B4. 端到端输出对齐（alignment）机制

需要提取：对齐的输入是什么、输出是什么、何时触发。

- `diarization-ggml/src/offline_pipeline.cpp`：`align_segments(transcribe_segments, diar_result)`
- `diarization-ggml/src/pipeline.cpp`：`handle_whisper_result()` 中 align
- 对齐实现：`diarization-ggml/src/aligner.cpp`（建议在写作前提取其核心对齐规则与边界情况）

### B5. “系统章节”里必须诚实列出的开销点（支撑学术讨论）

建议从代码提取并在论文中讨论：

- **GGML segmentation 输出转置**：GGML 输出为 class-major，需要转成 frame-major
  - 证据位置：`diarization-ggml/src/streaming.cpp` 中 `tmp` + 双层 for-loop 转置
- **masking fbank 的 CPU memset**：按说话人掩码逐帧将 80 bins 清零
  - 证据位置：`diarization-ggml/src/streaming.cpp` 中对 `masked_fbank` 的 `memset`
- **Whisper 与 diarization 的同步边界**：offline 先 whisper_full 再 diarize；streaming 通过队列与 in-flight 控制
  - 证据位置：`diarization-ggml/src/offline_pipeline.cpp`、`diarization-ggml/src/pipeline.cpp`（`whisper_in_flight` + queue）

---

## C. 建议你额外从代码里“摘录成论文表格/伪代码”的项目（写作效率很高）

这些材料非常适合直接变成论文中的表/算法框。

- **表：segmentation/embedding 的 I/O 形状、dtype、后端选项**
  - 来源：`streaming.cpp` 常量 + coreml bridge 输入输出 + ggml build_graph 输入输出

- **伪代码：offline_transcribe 的 5-step pipeline**
  - 来源：`diarization-ggml/src/offline_pipeline.cpp`

- **伪代码：streaming pipeline 的 push-loop 与触发条件**
  - 来源：`diarization-ggml/src/pipeline.cpp` + `diarization-ggml/src/streaming.cpp`

- **算法框：CUDA cooperative BiLSTM recurrence（每步 grid.sync）**
  - 来源：`whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`（建议抽取 step 循环与状态 ping-pong）
