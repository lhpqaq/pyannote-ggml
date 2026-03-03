# Context Compression: Segmentation CUDA Bottleneck and Decisions

This document is a compressed context snapshot for collaborators working on `pyannote-ggml` segmentation performance.

It captures what was measured, what was concluded, what was tried, and what the current optimization direction is.

## 1. Repository / Components

Key directories:

- `diarization-ggml/`: end-to-end diarization pipeline binary.
- `models/segmentation-ggml/`: segmentation model graph (SincNet + BiLSTM + MLP + classifier).
- `whisper.cpp/ggml/src/ggml-cuda/`: CUDA backend used by this repo.

CoreML-related components (Apple):

- `models/segmentation-ggml/convert_coreml.py`: PyTorch -> CoreML conversion for segmentation
- `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`: runtime bridge
- diarization flags: `--seg-coreml` (segmentation) and `--coreml` (embedding)

Important note:

- The diarization binary links against the ggml implementation under `whisper.cpp/ggml/`.

## 2. How We Isolated Segmentation Performance

To avoid confounding effects from embedding extraction, filtering, PLDA scoring, clustering, etc., we used a segmentation-only configuration:

- diarization binary with `--bypass-embeddings`
- CUDA backend enabled

Example:

```bash
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda \
  --bypass-embeddings \
  --plda diarization-ggml/plda.gguf \
  -o /tmp/out.rttm
```

## 3. Profiling Tools and Reports

Profiling used:

- `nsys profile` for CUDA timeline
- `nsys stats` to summarize kernel and API time

Reports used:

- `cuda_gpu_kern_sum`
- `cuda_api_sum`

## 4. Core Findings (Stable)

### 4.1. The bottleneck is the custom BiLSTM CUDA recurrence kernel

Across multiple runs, segmentation GPU time is overwhelmingly concentrated in two kernels:

- `k_pyannote_seg_lstm_dir<reverse=true,  half, float>`
- `k_pyannote_seg_lstm_dir<reverse=false, half, float>`

Combined, they typically account for ~99%+ of GPU kernel time.

Kernel call counts match the model structure:

- 4-layer BiLSTM
- typical audio produces 21 chunks
- expected per-direction call count: `21 * 4 = 84` (forward direction) and 84 (reverse direction)

This alignment provides strong confidence that the observed kernels correspond to the BiLSTM recurrence.

### 4.2. CPU time is mostly GPU waiting

`cuda_api_sum` is dominated by `cudaStreamSynchronize`.

Interpretation:

- the pipeline launches kernels and synchronizes to obtain results
- synchronization time reflects the long-running recurrence kernels

### 4.3. Graph build / alloc / reset is not the primary cause

Caching graph metadata and allocator state (to avoid per-chunk rebuild/alloc overhead) does not materially change segmentation runtime.

Therefore:

- focus on recurrence kernel performance, not graph plumbing

## 5. Decisions and Discarded Paths

### 5.1. CUDA Graphs

CUDA Graph capture was tested previously; on Tesla T4 it was not usable/beneficial in the tested configuration (runtime disabled by arch check or slower when forced).

### 5.2. Batch multiple chunks at once

Batching multiple chunks (batch > 1) was explored as a way to reduce host synchronization and amortize overhead.

Outcome:

- batch > 1 caused correctness issues (e.g., SincNet outputs differ from per-sample inference; in some attempts classifier outputs became numerically invalid).
- user decided to remove batch-related changes and not pursue this route.

Status:

- batching is not the current plan.

## 6. Current Main Optimization Direction

### 6.1. Parallelize the LSTM recurrence kernel

The optimization that produced clear gains is to increase GPU parallelism inside the custom LSTM recurrence by splitting the hidden dimension across blocks and synchronizing per time step.

Characteristics:

- keep `W_ih * x` as cuBLAS GEMM (already fast)
- parallelize `W_hh * h_{t-1}` recurrence over hidden indices
- use cooperative grid synchronization to preserve time-step correctness

Measured outcome:

- segmentation time reduced from ~4.7s to ~1.0s (representative T4 results)

## 6.3. What "correctness" means in this work

There are multiple notions of correctness relevant to GPU kernel optimization:

- Bitwise identical logits: generally unrealistic once arithmetic order changes due to parallelism.
- Logits agreement within tolerance: practical and measurable.
- Downstream diarization equivalence (DER/RTTM): the product metric, but sensitive to thresholding and non-local effects.

In this work, the primary kernel-level validation was logits comparison (max/mean abs diff and argmax agreement), followed by downstream sanity checks.

### 6.2. Correctness validation approach

Avoid using RTTM/DER alone to validate kernel equivalence because post-processing can amplify small numerical differences.

Preferred validation:

- compare per-frame segmentation logits/log-probabilities
- report max/mean abs diff and argmax agreement

Python reference (pyannote-audio) can be used, but note that C++ vs Python are not bitwise identical even without kernel changes.

## 7. Next Work Items (Non-batch)

- Further optimize recurrence kernel throughput:
  - vectorization (half2)
  - fuse gate math + activations + state update
  - explore WMMA/CUTLASS for dot products where viable

- Reduce end-to-end synchronization overhead without batching:
  - avoid unnecessary device-host transfers
  - ensure scheduler does not force extra sync points
  - consider limited graph-level capture only if architecture allows

## 8. Minimal command set (copy/paste)

Segmentation-only run:

```bash
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
  -o /tmp/out.rttm
```

Profile with nsys:

```bash
mkdir -p /tmp/nsys
nsys profile -o /tmp/nsys/seg_only --force-overwrite=true --trace=cuda,nvtx,osrt \
  ./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
    models/segmentation-ggml/segmentation.gguf \
    models/embedding-ggml/embedding.gguf \
    samples/sample.wav \
    --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
    -o /tmp/out.rttm

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /tmp/nsys/seg_only.nsys-rep
```

## 9. Paper-ready storyline (one paragraph)

We profiled a GPU diarization segmentation pipeline and found that nearly all GPU time is spent in a custom BiLSTM recurrence kernel, while CPU time is dominated by CUDA stream synchronization (waiting for that kernel). Caching graph construction and allocator state did not change runtime, confirming the recurrence kernel as the true bottleneck. We then redesigned the recurrence kernel to parallelize across hidden units within each time step and enforced time-step correctness via a grid-level barrier. This structural change improved GPU occupancy and reduced recurrence kernel latency, resulting in a multi-x end-to-end segmentation speedup on a Tesla T4. Correctness was evaluated primarily using logits-level comparisons (max/mean diff and argmax agreement) rather than downstream RTTM alone.
