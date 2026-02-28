# 将 Whisper 转写与 Pyannote 流式说话人日志集成

所有流水线均在 **过滤后的时间轴 (filtered timeline)** 上运行 —— 在静音过滤阶段之后，除了 VAD 过滤器用于测量 5 秒静音刷新的真实时间轴 (true-timeline) 之外，不再使用原始时间轴。

## 流水线概览

```
原始音频 → VAD 静音过滤器 → 过滤后的音频
                  │                      │
                  │         ┌────────────┬┘
                  │         ▼            ▼
                  │      音频缓冲区     Pyannote 流式处理
                  │         │            │
                  │         │      VAD 预测帧
                  │         │            │
                  │         │     分段结束检测器 ←────────┐
                  │         │            │                 │
                  │         ├────────────┘                 │
                  │         ▼                              │
                  │    Whisper 转写器 ←────────────────────┘
                  │         │                    (5秒静音强制刷新)
                  │      单词级标记
                  │         │
                  │         ▼
                  └→   对齐 + 重新聚类
                            │
                            ▼
                      回调 (带标签的分段 + 单词)
```

---

## 1. VAD 静音过滤器

将音频中任何位置的静音限制在最多 2 秒，同时保留自然的语音边界（无突发性的开始/结束）。

### 输入

音频帧以可变大小的块到达（例如，一次 512 帧，下一次 2000 帧）。VAD 以 512 帧为单位进行处理，因此输入的帧会在内部缓冲，直到累积 ≥512 帧。每个 512 帧的批次处理后，音频和 VAD 结果将传递到下一阶段。

### 静音缓冲区

| 缓冲区 | 类型 | 大小 | 用途 |
|--------|------|------|---------|
| `endSilence` | 循环/FIFO | 1 秒 | 最近静音的滚动窗口 —— 在下一次语音开始前置，以实现平滑过渡 |

初始化时缓冲区为空。

### 逻辑

**当静音帧到达时：**
- 如果 `silence_started` 不存在 → 设置为当前帧号，通过帧
- 如果 `silence_started` 已存在：
  - 如果自 `silence_started` 以来经过的时间 ≤ 1 秒 → 通过帧（允许自然的尾部静音）
  - 如果经过的时间 > 1 秒 → 将帧写入 `endSilence` 缓冲区（FIFO，1秒容量 —— 最旧的帧将被覆盖）
    - 追踪 `consecutive_discarded_frames`（每当写入循环缓冲区的帧覆盖现有帧时递增，即缓冲区填满后）
    - 如果 `consecutive_discarded_frames` 达到 5 秒对应的帧数（16kHz 下为 80,000 帧）→ 调用分段结束检测器的 **刷新函数 (flush function)**（见 §3），然后重置计数器

**当语音帧到达时：**
- 如果 `silence_started` 存在 → 设置为 null，重置 `consecutive_discarded_frames` 为 0
  - 如果 `endSilence` 中有帧 → 连接 `endSilence` 内容 + 语音帧，清空缓冲区，通过连接后的帧
  - 如果 `endSilence` 为空 → 通过语音帧
- 如果 `silence_started` 不存在 → 通过语音帧

**效果：** 语音结束后，最多允许 1 秒的静音通过。超过此限度的静音将被吸收。当语音恢复时，最近 1 秒的静音（来自 `endSilence`）会被添加在前面，使过渡听起来自然。大于 2 秒的静音间隙被压缩至约 2 秒（1s 尾部 + 1s 头部）。如果静音持续足够长（5 秒丢弃帧 ≈ 6 秒总静音），下游流水线将被刷新，以避免无限期持有音频。

---

## 2. 音频缓冲区

持有过滤后的音频，直到它可以被分块并发送给 Whisper。

### 状态

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `buffer` | FIFO | 累积过滤后的音频帧 |
| `dequeued_frames` | int64 | 总共出队的帧数 —— 用于时间戳计算的偏移量 |

### 时间戳计算

对于缓冲区中位置 `i` 的任何帧：
```
absolute_frame = dequeued_frames + i
time = absolute_frame / sample_rate
```

### 操作

- **入队 (Enqueue)**：追加来自静音过滤器的输入帧
- **出队至指定时间戳 (Dequeue up to timestamp)**：将过滤时间轴的时间戳转换为采样位置（`round(time × sample_rate)`），从前端移除直到该位置的帧，递增 `dequeued_frames`
- **读取范围 (Read range)**：提取两个绝对帧位置之间的音频（用于发送给 Whisper）

---

## 3. 分段 (分段结束检测)

使用 pyannote 的流式 VAD 输出找到“语音→静音”的转换，以便将音频分块给 Whisper。

### Pyannote 集成

- 每当过滤后的音频到达时，通过 `streaming_push`（零延迟模式）将其推入 pyannote
- Pyannote 内部缓冲直到拥有 16,000 个采样（1 秒），然后处理一个块
- 单次推送 N 秒音频可能会产生 N 个 VADChunks —— 在循环中处理所有块
- 每个块包含 `num_frames` 个预测帧（约 59 或 60 帧，可变）
- 帧之间完美平铺，没有重叠或间隙
- 帧计时：`time = frame_number × 0.016875`

### 帧计数

维持一个 `current_prediction_frame` 计数器（从 0 开始）。每当 VADChunk 到达时，将其 `num_frames` 加到计数器中。这提供了精确的帧→时间对应关系：

```
segment_end_time = frame_number × 0.016875
```

589帧:10秒的比例是底层真相，但我们不需要计算它 —— 累加帧计数会自动处理 59/60 交替的模式。

### 分段结束检测逻辑

维持 `last_frame_was_speech`（布尔值，初始为 false）。

当 VADChunk 到达时：
1. **检查跨边界转换**：如果 `last_frame_was_speech == true` 且 `vad[0] == 0.0`：
   - 在 `(current_prediction_frame) × 0.016875` 处检测到分段结束（新块的第一帧）
   - 将此时间戳发送到转写阶段
2. **块内扫描**：在块的 `vad` 数组中搜索第一个“语音 → 非语音”转换：
   - 找到第一个索引 `i`，使得 `vad[i] == 1.0` 且其后紧跟 `vad[i+1] == 0.0`（或任何 `1.0` 后的第一个 `0.0`）
   - 如果找到：分段结束于 `(current_prediction_frame + i + 1) × 0.016875`
   - 将此时间戳发送到转写阶段
   - 注意：每个块仅检测第一个转换。同一 ~1s 窗口内的后续转换将在后续块中捕获。这是故意的 —— 每秒一个切割点对于 Whisper 分块已足够。
3. **更新状态**：设置 `last_frame_was_speech = (vad[num_frames - 1] != 0.0)`
4. **递增计数器**：`current_prediction_frame += num_frames`

### 刷新函数 (由 VAD 过滤器调用)

暴露给 VAD 静音过滤器，用于 5 秒静音超时。调用时：
- 向转写阶段 (§4) 发送 **强制刷新信号 (force-flush signal)**，绕过 20 秒最小阈值
- 不涉及分段结束时间戳 —— 转写阶段直接发送音频缓冲区中的任何内容
- 如果音频缓冲区为空，则不执行任何操作

---

## 4. 转写 (Whisper)

接收来自分段阶段的分段结束时间戳，并决定何时将音频发送给 Whisper。

### 状态

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `buffer_start_time` | double | 当前音频缓冲区中第一帧的过滤时间轴时间戳 |

### 触发条件

当满足以下 **任一** 条件时，音频被发送给 Whisper：

1. **分段结束 + 最小长度**：检测到分段结束 且 `segment_end_time - buffer_start_time ≥ 20 秒`
   - 将分段结束时间转换为采样位置：`cut_sample = round(segment_end_time × sample_rate)`
   - 提取从缓冲区开始到 `cut_sample` 的音频
   - 将缓冲区出队直到 `cut_sample`
   - 发送音频给 Whisper

2. **静音刷新**（来自 VAD 过滤器，经分段结束检测器）：收到强制刷新信号
   - 提取缓冲区中当前所有的音频
   - 将整个缓冲区出队
   - 发送音频给 Whisper
   - 如果缓冲区为空，不执行操作

3. **结束 (Finalize)**（流结束）：无论长度如何，刷新剩余音频（见 §6）

### 输出

Whisper 返回 **相对于发送的音频块起点** 的单词级时间戳。在传递给对齐阶段之前，转换为过滤时间轴的绝对时间戳：

```
absolute_start = buffer_start_time + whisper_relative_start
absolute_end   = buffer_start_time + whisper_relative_end
```

传递给对齐阶段的标记 (Tokens)：

```
struct Token {
    std::string text;
    double start;    // 过滤时间轴 (绝对)
    double end;      // 过滤时间轴 (绝对)
};
```

---

## 5. 对齐 (说话人日志 + 标记匹配)

结合 pyannote 说话人标签与 Whisper 单词时间戳，遵循 WhisperX 的方法（参考 [m-bain/whisperX](https://github.com/m-bain/whisperX) 的 `assign_word_speakers`）。

### 核心算法：最大相交分配 (Maximum Intersection Assignment)

对于每个单词，找到与其 `[start, end]` 范围重叠时间最长的说话人分段：

```
对于每个带有 [start, end] 的单词：
    找到所有与 [start, end] 重叠的日志分段
    对于每个重叠的分段：
        intersection = min(seg_end, word_end) - max(seg_start, word_start)
        按说话人累加 intersection
    word.speaker = 拥有最大总重叠时间的说话人
```

- **单词跨越两个说话人**：重叠时间更长的说话人胜出
- **单词处于间隙中**（无重叠的日志分段）：按中点距离分配给最近的日志分段（`(word_start + word_end) / 2` 与分段中点比较）
- **无需预分块**：单词通过时间重叠逐个匹配。Whisper 自然的分段边界（大致为句子级）不用于说话人分配 —— 每个单词都有自己的查找逻辑

为了处理长音频的效率，对日志分段使用区间树（排序数组 + 二分查找），实现每个单词 O(log n) 的重叠查询。

### 数据存储

#### 标记存储 (Token Storage - 工作集)

由 Whisper 产生但尚未分配给已完成分段的所有标记。当 Whisper 返回结果时添加标记，当匹配到已完成的分段时移除。

#### 已完成分段存储 (Finished Segment Storage - 永久)

存储 `(segment_boundaries, list_of_tokens)` 元组，用于 **结束点在 10 秒前** 的日志分段。这些分段的二值活动边界（开始/结束时间）是稳定的，在未来的重新聚类中不会改变。

**已完成分段中的说话人标签可能会改变** —— 只有边界是稳定的。标签在每次重新聚类时都会被重新映射。

### 对齐流程

当 Whisper 返回单词级标记时：

1. **存储标记**：将所有新标记添加到“标记存储”

2. **重新聚类**：在 pyannote 状态上调用 `streaming_recluster()` 以获取当前的 `DiarizationResult`（带有全局说话人标签的分段）

3. **从头重建完整映射**：基于新的日志分段构建区间树。对于每一个标记（包括“标记存储”和“已完成分段存储”中的标记）：
   - 查询区间树以寻找重叠的日志分段
   - 将标记分配给拥有最大相交时长的说话人
   - 如果不存在重叠，按中点距离分配给最近的分段
   - 不与之前的重新聚类输出做差异对比 —— 直接覆盖所有分配

4. **提升为已完成**：对于结束时间早于当前音频位置 10 秒以上的已匹配日志分段：
   - 将 `(segment_boundaries, token_list)` 移入“已完成分段存储”
   - 从“标记存储”中移除这些标记

5. **发出输出**：收集所有日志分段（包括已完成和进行中的）及其标记和当前说话人标签。传递给回调函数。

---

## 6. 结束 (Finalize - 流结束)

当音频流结束时，按顺序刷新整个流水线：

1. **静音过滤器**：刷新 `endSilence` 缓冲区 —— 将任何剩余帧同时传递给音频缓冲区和通过 `streaming_push` 传递给 pyannote
2. **Pyannote**：调用 `streaming_finalize()` —— 处理剩余的部分块（补零），运行最终的重新聚类。返回完整的 `DiarizationResult`
3. **Whisper**：将缓冲区中所有剩余音频发送给 Whisper（无论长度或分段结束检测结果如何）
4. **对齐**：根据最终的日志结果进行最终的标记匹配。所有分段现在都已完成。通过回调发出最终输出。

结束后，流水线完成。所有资源均可释放。

---

## 7. 回调

```
callback(segments: [{
    speaker: string,       // "SPEAKER_00", "SPEAKER_01", ...
    start: double,         // 过滤时间轴秒数
    duration: double,      // 秒数
    tokens: [{
        text: string,
        start: double,     // 过滤时间轴秒数
        end: double,       // 过滤时间轴秒数
    }]
}])
```

在每次对齐步骤 (§5.5) 后调用，包含当前的完整状态 —— 既有已完成的分段，也有进行中的分段。消费者随着更多音频的处理接收增量更新。

---

## 常量参考

| 常量 | 数值 | 来源 |
|----------|-------|--------|
| 采样率 | 16,000 Hz | 输入音频 |
| VAD 处理大小 | 512 帧 | VAD 模型要求 |
| 静音限制 | 2 秒 | 设计选择 (1s 尾部 + 1s 头部) |
| 静音刷新阈值 | 5 秒丢弃帧 | 真实时间轴；通过 VAD 过滤器触发 Whisper 刷新 |
| Pyannote 推送大小 | 可变 (内部缓冲至 16,000) | 流式 API 处理任何输入大小 |
| 每次推送的预测帧 | ~59-60 (可变) | SincNet 架构 |
| 帧时间步长 | 0.016875s | 270 采样 / 16kHz |
| 每 10s 块的帧数 | 589 | SincNet 3阶段卷积+池化 |
| Whisper 最小发送长度 | 20 秒 | 设计选择 |
| 分段稳定性范围 | 10 秒 | Pyannote 重叠窗口 |
| Whisper 最大输入 | ~30 秒 | Whisper 架构限制 |
