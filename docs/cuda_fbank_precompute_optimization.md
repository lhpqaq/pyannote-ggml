# CUDA 路径 Fbank 全局预计算优化

本文档记录将 CoreML 路径中的 fbank 全局预计算优化移植到 GGML/CUDA 路径的改动。

## 1. 动机与瓶颈

### 1.1 问题

在 `extract_embeddings` 函数中，每个 chunk 独立调用 `compute_fbank(cropped, CHUNK_SAMPLES, SAMPLE_RATE)`。由于滑动窗口步长为 1s、窗口长度为 10s，相邻 chunk 有 **90% 音频重叠**，导致 fbank 帧被重复计算约 **10 次**。

对于 mulpeo.wav（221 chunks），逐 chunk 计算需要 221 × 998 = 220,558 帧的 FFT + mel 滤波，而全局预计算只需 22,998 帧。

### 1.2 CoreML 路径已有方案

CoreML 路径的 `pipeline_parallel_seg_emb` 函数已实现此优化（代码位于 `diarization.cpp` 第 392-432 行），正确性已通过 RTTM 一致性验证。

### 1.3 GGML/CUDA 路径缺失

`extract_embeddings` 函数（同时服务 CoreML 非并行路径和 GGML/CUDA 路径）仍使用逐 chunk 计算模式。

## 2. 实现方案

### 2.1 核心改动

在 `extract_embeddings` 入口处一次性计算全局 fbank，替代循环内的逐 chunk 计算：

```cpp
// 全局 fbank 预计算（无 CMN）
const int padded_len = (num_chunks - 1) * STEP_SAMPLES + CHUNK_SAMPLES;
std::vector<float> padded_audio(padded_len, 0.0f);
// ... copy audio with zero-padding ...
embedding::fbank_result global_fbank =
    embedding::compute_fbank(padded_audio.data(), padded_len, SAMPLE_RATE, false);

// 每个 chunk 切片 + per-chunk CMN
for (int c = 0; c < num_chunks; c++) {
    const int frame_offset = c * FBANK_STEP;  // FBANK_STEP = 100
    const int nf = std::min(nf_per_chunk, global_fbank.num_frames - frame_offset);
    std::memcpy(chunk_fbank.data(),
                global_fbank.data.data() + frame_offset * FBANK_NUM_BINS,
                nf * FBANK_NUM_BINS * sizeof(float));
    embedding::apply_cmn(chunk_fbank.data(), nf, FBANK_NUM_BINS);
    // ... masking + inference as before ...
}
```

### 2.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FBANK_FRAME_SHIFT | 160 samples | 10ms at 16kHz |
| FBANK_STEP | 100 frames | = STEP_SAMPLES / FBANK_FRAME_SHIFT |
| nf_per_chunk | 998 frames | = (160000 - 400) / 160 + 1 |

### 2.3 两阶段计算

1. **全局 raw fbank**（一次性，`apply_cmn=false`）：对 zero-padded 全长音频计算
2. **Per-chunk 切片 + CMN**（每个 chunk 独立）：从全局 fbank 中按 `frame_offset = c * 100` 提取 998 帧切片，应用 per-chunk CMN

## 3. 正确性保证

### 3.1 数学等价性

fbank 是 per-frame 的局部运算（每帧只依赖 25ms 窗口内的音频）。对连续音频计算 fbank 后再切片，与先切片音频再计算 fbank，结果完全相同。CMN 在相同的 998 帧切片上计算，因此也完全相同。

### 3.2 验证方式

- 优化前后 RTTM 输出逐字节一致
- 此方案已在 CoreML 并行路径中验证通过

## 4. 计算量对比

| 指标 | 优化前 | 优化后 |
|------|-------:|-------:|
| FFT + mel 滤波帧数 (mulpeo.wav) | 220,558 | 22,998 |
| CMN 计算次数 | 221 次 (998帧) | 221 次 (998帧) |
| memcpy 切片 | 0 | 221 次 (998×80×4B) |
| **理论计算量** | 1× | **~0.1×** |

## 5. 修改文件

| 文件 | 改动 |
|------|------|
| `diarization-ggml/src/diarization.cpp` | `extract_embeddings` 函数重构为全局 fbank 预计算 + per-chunk 切片 |

## 6. 实测结果（Tesla T4, sample.wav 30s, 21 chunks）

| 指标 | 基线 | 优化后 | 变化 |
|------|-----:|-------:|-----:|
| Segmentation | 185 ms | 185 ms | 0% |
| Embeddings | 1105 ms | 992 ms | **-10.2%** |
| Total | 1486 ms | 1379 ms | **-7.2%** |

- 输出正确性：优化前后 RTTM 逐字节一致
- 分割阶段不受影响（预期中）
- 嵌入阶段节省 ~113ms，主要来自 fbank FFT/mel 帧冗余消除
- 对长音频收益更大（更多 chunk = 更多冗余消除）

## 7. 适用范围

- 对所有后端路径（CPU、CUDA、CoreML 非并行）生效
- 零风险：不改变任何推理路径的数学表达
