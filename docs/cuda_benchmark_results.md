# CUDA 优化基准测试结果

测试环境：Tesla T4 (Turing, 7.5), Linux, CUDA BiLSTM coop kernels enabled

测试样本：`samples/sample.wav` (30s, 480000 samples, 21 chunks)

## 环境变量

```bash
DIARIZATION_SEG_LSTM_COOP=1
DIARIZATION_SEG_LSTM_COOP_WARP=1
DIARIZATION_SEG_LSTM_COOP_WARPS=4
DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=1
DIARIZATION_SEG_LSTM_COOP_BIDIR=1
```

## 基线 vs 优化版对比（热启动，5 次取中位数）

| 指标 | 基线 | 优化版 | 变化 |
|------|-----:|-------:|-----:|
| Segmentation | 185 ms | 185 ms | 0% |
| Embeddings | 1105 ms | 990 ms | **-10.4%** |
| Total | 1486 ms | 1377 ms | **-7.3%** |
| 输出正确性 | 14 段 2 人 | 14 段 2 人 | RTTM 完全一致 |

## 生效的优化

### Opt-1: Fbank 全局预计算（`extract_embeddings`）

- **效果**：Embedding 阶段减少 ~115ms（10.4%）
- **原理**：相邻 chunk 90% 音频重叠，逐 chunk 计算 fbank 有 ~10x 冗余。改为全局计算一次后 per-chunk 切片 + CMN
- **风险**：零风险，数学完全等价
- **适用**：所有后端路径（CPU、CUDA、Metal）

### Opt-4: Embedding 正确性诊断工具

- **效果**：无性能影响（仅在设置 `DIARIZATION_EMB_DEBUG=1` 时激活）
- **功能**：per-embedding NaN/Inf/L2 检查 + 推理前汇总统计

## 未生效 / 已回退的优化

### Opt-2: Seg/Emb 管线并行（已保留但默认禁用）

- **实测**：单 GPU 下 seg 和 emb 竞争相同 GPU 资源，总耗时无收益
- **状态**：opt-in（`DIARIZATION_PARALLEL=1`），适用于 CPU-only 或多 GPU 场景

### Opt-3: Segmentation 计算图缓存（已回退）

- **问题**：`ggml_backend_sched_reset()` 清除 tensor→backend 映射后，缓存图的 GPU 分配丢失，导致输出全零
- **状态**：完全回退，需深入研究 `ggml_backend_sched` 的生命周期管理

## 基线数据（原始记录）

```
BASELINE (5 runs, warm runs 2-5):
  Run 2: Seg 185ms, Emb 1100ms, Total 1479ms
  Run 3: Seg 184ms, Emb 1108ms, Total 1489ms
  Run 4: Seg 185ms, Emb 1103ms, Total 1486ms
  Run 5: Seg 185ms, Emb 1108ms, Total 1488ms
  Median: Seg 185ms, Emb 1105ms, Total 1486ms

OPTIMIZED (5 runs, warm runs 2-5):
  Run 2: Seg 187ms, Emb  990ms, Total 1387ms
  Run 3: Seg 184ms, Emb  990ms, Total 1372ms
  Run 4: Seg 188ms, Emb  994ms, Total 1379ms
  Run 5: Seg 184ms, Emb  993ms, Total 1377ms
  Median: Seg 185ms, Emb  990ms, Total 1377ms
```
