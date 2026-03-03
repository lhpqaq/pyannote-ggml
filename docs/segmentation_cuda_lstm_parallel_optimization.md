# Segmentation BiLSTM CUDA Kernel Parallelization (Paper-Oriented Notes)

This document records the motivation, evidence, algorithmic changes, validation methodology, and measured outcomes for optimizing the segmentation model's custom BiLSTM CUDA kernel in `pyannote-ggml`.

The intended use is as source material for a future academic paper. It is written to be self-contained and to preserve the profiling-to-optimization-to-validation narrative.

## 1. Problem Statement

The segmentation component (speaker activity / overlap detection) in `pyannote-ggml` is dominated by a 4-layer bidirectional LSTM (BiLSTM). In the CUDA backend, the LSTM recurrence is implemented via a custom CUDA kernel.

Empirically, segmentation on a single audio file can take multiple seconds on a Tesla T4 GPU, even though segmentation is expected to be near real-time or sub-second for short inputs.

Goal:

- Reduce segmentation wall time on CUDA substantially while keeping inference behavior correct (within an acceptable tolerance for the diarization pipeline).

Non-goals:

- Replacing the model architecture.
- Large refactors of `ggml` internals (vendored) unless unavoidable.

## 2. System Under Test

Repository structure (relevant parts):

- `models/segmentation-ggml/`: segmentation model graph and inference wrapper.
- `whisper.cpp/ggml/src/ggml-cuda/`: CUDA backend implementation used by the pipeline.

Model structure (high level):

- SincNet front-end (convs + pooling + InstanceNorm + activation)
- 4-layer BiLSTM (hidden size H=128)
- 2-layer MLP (linear + LeakyReLU)
- classifier producing powerset log-probabilities (7 classes) per frame

Inference chunking:

- The diarization pipeline runs segmentation on sliding chunks.
- For the typical configuration used during profiling:
  - chunk duration ~10 seconds
  - step ~1 second
  - each chunk produces ~589 frames
  - an example 30-second audio produces 21 chunks

## 3. Profiling Methodology

### 3.1. Segmentation-only execution

To remove embedding / clustering effects and isolate segmentation, run diarization with embeddings bypassed:

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

This configuration is used purely to measure segmentation runtime and CUDA behavior.

### 3.2. nsys capture

Collect a CUDA timeline and aggregated stats:

```bash
mkdir -p /tmp/nsys
nsys profile -o /tmp/nsys/seg_only \
  --force-overwrite=true \
  --capture-range=none \
  --trace=cuda,nvtx,osrt \
  ./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
    models/segmentation-ggml/segmentation.gguf \
    models/embedding-ggml/embedding.gguf \
    samples/sample.wav \
    --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
    -o /tmp/out.rttm

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /tmp/nsys/seg_only.nsys-rep
```

The key reports used were:

- `cuda_gpu_kern_sum` (GPU kernel time distribution)
- `cuda_api_sum` (CPU-side CUDA API time distribution)

## 4. Baseline Findings (Evidence)

### 4.1. GPU time concentrated in custom LSTM kernels

The segmentation graph contains many operations (conv, pooling, norm, softmax, log, etc.). However, on CUDA nearly all GPU time is spent in two kernels corresponding to the custom LSTM recurrence:

- `k_pyannote_seg_lstm_dir<reverse=true,  __half, float>`
- `k_pyannote_seg_lstm_dir<reverse=false, __half, float>`

In typical runs, these two kernels account for ~99% of the total GPU kernel time.

The number of kernel calls matches the model and chunking structure:

- 21 chunks
- 4 LSTM layers
- for each chunk and each layer, forward and reverse directions are executed
- therefore: `21 * 4 = 84` calls per direction, 168 calls total

This consistent call count provides a strong sanity check that the observed kernels correspond exactly to the model's BiLSTM structure.

### 4.2. CPU time dominated by synchronization (waiting for GPU)

`cuda_api_sum` shows that a large fraction of CPU time is spent in `cudaStreamSynchronize`, often dominating total CPU time.

Interpretation:

- The pipeline launches kernels then synchronizes to obtain results.
- The cost is not the synchronization itself; it reflects that kernels (primarily the LSTM kernels) are long-running.

### 4.3. Not the graph build / alloc / reset overhead

Graph build / alloc / reset overhead was tested by caching graph metadata and allocator state in the segmentation wrapper. The segmentation runtime changed negligibly (~4.6s remained ~4.6s).

Therefore, optimization effort should target the LSTM recurrence kernel itself.

## 5. Baseline Kernel Structure and Why It Is Slow

At a high level, an LSTM recurrence step for each hidden index `h` requires computing four gates:

- `i_t(h), f_t(h), g_t(h), o_t(h)`

and then updating:

- `c_t(h) = f_t(h) * c_{t-1}(h) + i_t(h) * g_t(h)`
- `h_t(h) = o_t(h) * tanh(c_t(h))`

The baseline kernel effectively performs, for each direction, for each chunk:

- Precompute `ih_all = W_ih * x` (done efficiently via cuBLAS GEMM)
- For each time step `t`:
  - For each hidden index `h`:
    - compute dot products `W_hh[:, gate(h)] · h_{t-1}`
    - apply activations
    - update `c_t(h)` and `h_t(h)`

The performance bottleneck is the recurrence dot products `W_hh * h_{t-1}`:

- Hidden size is H=128, so each `h` does a dot product of length 128.
- Sequence length is ~589 per chunk.
- The baseline implementation uses insufficient thread-block level parallelism (effectively one block per direction), limiting occupancy and throughput.

## 6. Optimization Approach: Parallelize the Recurrence

The core design constraint is that LSTM recurrence is sequential across time steps `t`, but parallel across hidden indices `h` within each time step.

### 6.1. Keep GEMM for input contribution

We keep:

- `ih_all = W_ih * x` as cuBLAS GEMM.

This part is already fast and not the bottleneck.

### 6.2. Parallelize over hidden indices with multiple blocks

We restructure the recurrence kernel to launch multiple blocks, each responsible for a subset of hidden indices.

Example conceptual mapping:

- blockIdx.x selects a tile of hidden indices
- threadIdx.x selects one hidden index within the tile

Each thread computes the gate dot products and state updates for its own `h`.

### 6.3. Maintain time-step correctness via grid synchronization

Because blocks write different portions of `h_t` and `c_t`, all blocks must finish step `t` before any block starts step `t+1`.

We use a cooperative grid synchronization mechanism:

- launch via `cudaLaunchCooperativeKernel`
- within the kernel, use `cooperative_groups::this_grid().sync()` at the end of each time step

This provides a global barrier across blocks for each time step.

### 6.4. Controlled rollout

The optimized kernel is gated by an environment variable so it can be disabled for comparison or fallback:

- `DIARIZATION_SEG_LSTM_COOP=1` enables the cooperative parallel kernel
- `DIARIZATION_SEG_LSTM_COOP=0` uses the legacy kernel

## 7. Implementation Notes (Engineering Details)

### 7.1. Weight and tensor layouts

The custom op expects ggml tensor layouts that differ from conventional cuDNN formats:

- `w_ih` and `w_hh` are stored in ggml layout with specific `ne[]` ordering.
- Care must be taken to interpret and pack weights correctly.

This optimization keeps the existing weight interpretation and only changes parallelization strategy of the recurrence stage.

### 7.2. Memory for recurrent states

The recurrence uses global memory buffers for `h` and `c` state arrays. A double-buffering scheme is used to avoid read-after-write hazards across time steps.

The cooperative barrier ensures all writes complete before the next step reads from the updated buffer.

## 7.3. Why this is not "just parallelism"

The key algorithmic change is not merely increasing thread count; it changes the mapping between the mathematical recurrence and the GPU execution model:

- The baseline implementation is effectively a one-block recurrence and is limited by low occupancy.
- The optimized implementation exploits independence across hidden indices within each time step.
- Correctness is preserved by introducing an explicit global synchronization per step (grid barrier), matching sequential recurrence semantics.

This is a structural optimization: it re-expresses the recurrence to match GPU parallel hierarchy.

## 8. Validation Methodology

### 8.1. Why RTTM is not a sufficient correctness test

RTTM output is produced after thresholding and postprocessing. Small differences in segmentation logits can change decisions around thresholds and cause large differences downstream (including speaker count changes) even if model outputs are nearly identical.

Therefore kernel validation must be done at the logits level.

### 8.2. Logits dumping and comparison

We used a logits-level comparison approach:

- dump raw segmentation powerset log-probabilities (or logits) per chunk
- compare elementwise statistics and argmax agreement

Metrics:

- `max_abs_diff`
- `mean_abs_diff`
- `argmax agreement` per frame

### 8.3. Reference implementation

Python `pyannote-audio` provides a reference path. For matching the C++ pipeline behavior:

- run sliding-window inference
- `skip_aggregation=True` (no overlap-add)
- `skip_conversion=True` (keep powerset classes)

Note: even between Python and C++ the outputs are not bitwise identical; acceptable correctness must be defined accordingly.

## 9. Results

On Tesla T4 (representative):

- Legacy kernel segmentation time for 21 chunks: ~4.6-4.8s
- Cooperative parallel kernel segmentation time: ~0.9-1.2s

This corresponds to an end-to-end segmentation speedup of roughly 4-6x.

Kernel-level stats typically show reduction of per-call recurrence kernel latency from ~27ms down to ~5-6ms.

## 9.1. nsys tables to include in a paper

For publication-quality evidence, the following are typically sufficient:

1) GPU kernel time summary (`cuda_gpu_kern_sum`):

- baseline: two LSTM recurrence kernels dominate GPU time
- optimized: the same kernels dominate but with reduced average latency

2) CUDA API summary (`cuda_api_sum`):

- baseline: `cudaStreamSynchronize` dominates CPU time
- optimized: total `cudaStreamSynchronize` time drops roughly proportionally with kernel latency

## 9.2. Simple scaling model

Let:

- `C` = number of chunks
- `L` = number of LSTM layers (4)
- `D` = number of directions (2)

Approximate recurrence kernel launches:

- `N_launch = C * L * D`

If average per-launch latency is `t_k`, then recurrence time is approximately:

- `T_recur ~ N_launch * t_k`

This predicts the observed baseline and optimized timings and provides a compact explanation linking nsys stats to end-to-end wall time.

## 10. Limitations and Future Work

### 10.1. Cooperative launch constraints

Cooperative kernels require:

- GPU support for cooperative launch
- sufficient resources (registers/shared memory) for all blocks to be resident

This can limit portability or require careful tuning of block sizes.

### 10.2. Remaining bottleneck

Even after parallelization, the recurrence computation remains the dominant cost.

Potential next steps:

1) Increase arithmetic throughput:
   - use half2 vectorization
   - use WMMA/CUTLASS for gate dot products
   - fuse gate computation + activation + state update

2) Reduce synchronization overhead:
   - reduce host-side synchronizations
   - evaluate CUDA Graph capture where supported and stable

3) Improve numerical stability & downstream robustness:
- diarization pipelines can be sensitive to tiny changes; evaluation should include DER/RTTM and include tolerance design.

## Appendix A. Code pointers

Key code regions involved in the optimization:

- CUDA custom op for pyannote segmentation LSTM:
  - `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`

- Segmentation model graph definition and tensor naming:
  - `models/segmentation-ggml/src/model.cpp`

- Pipeline chunk loop invoking segmentation:
  - `diarization-ggml/src/diarization.cpp`

## 11. Reproducibility Checklist

- GPU: Tesla T4 (sm_75)
- Build: CUDA backend enabled
- Run segmentation-only via `--bypass-embeddings`
- Profile with `nsys`, confirm recurrence kernel dominance
- Enable optimized kernel via `DIARIZATION_SEG_LSTM_COOP=1`
- Validate logits-level similarity, then evaluate downstream diarization metrics
