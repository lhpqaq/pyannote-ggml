# Segmentation BiLSTM CUDA Kernel Parallelization (Paper-Oriented Notes)

This document records the motivation, evidence, algorithmic changes, validation methodology, and measured outcomes for optimizing the segmentation model's custom BiLSTM CUDA kernel in `pyannote-ggml`.

The intended use is as source material for a future academic paper. It is written to be self-contained and to preserve the profiling-to-optimization-to-validation narrative.

If you want to convert this into a paper, the sections are already aligned to a common structure:

- Motivation / Problem -> Section 1-2
- Measurement methodology -> Section 3-4
- Method (kernel design) -> Section 6-7
- Evaluation -> Section 8-9
- Discussion -> Section 10

## 1. Problem Statement

The segmentation component (speaker activity / overlap detection) in `pyannote-ggml` is dominated by a 4-layer bidirectional LSTM (BiLSTM). In the CUDA backend, the LSTM recurrence is implemented via a custom CUDA kernel.

Empirically, segmentation on a single audio file can take multiple seconds on a Tesla T4 GPU, even though segmentation is expected to be near real-time or sub-second for short inputs.

Goal:

- Reduce segmentation wall time on CUDA substantially while keeping inference behavior correct (within an acceptable tolerance for the diarization pipeline).

Additional practical goals (paper-relevant):

- Keep the runtime memory footprint low compared to typical PyTorch inference.
- Provide operator-level implementation details that explain *why* performance differs from PyTorch and where optimizations apply.

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

## 2.1. Notation and Shapes (for paper)

We adopt the following notation for a single LSTM layer and a single direction.

Symbols:

- `T`: number of time steps per chunk (e.g. 589)
- `H`: LSTM hidden size (e.g. 128)
- `I`: input feature size to the layer (layer0: 60; higher layers: 2H)
- `G`: gate size, `G = 4H`

State:

- `h_t` in R^H: hidden state at time `t`
- `c_t` in R^H: cell state at time `t`

Weights (single direction):

- `W_ih` in R^(G x I)
- `W_hh` in R^(G x H)
- `b_ih` in R^G
- `b_hh` in R^G

Input and precomputation:

- `x_t` in R^I: input vector at time `t`
- `ih_t = W_ih x_t` in R^G

The code path precomputes `ih_t` for all `t` using GEMM:

- `IH = W_ih * X` where `X` stacks `x_t`

## 2.2. LSTM Equations (for paper)

For each `t` and elementwise over hidden index `h`:

- `z_t = ih_t + W_hh h_{t-1} + b_ih + b_hh`  (vector in R^G)

Split `z_t` into gates `i, f, g, o` in R^H:

- `i_t = sigmoid(z_t[i])`
- `f_t = sigmoid(z_t[f])`
- `g_t = tanh(z_t[g])`
- `o_t = sigmoid(z_t[o])`

Update:

- `c_t = f_t * c_{t-1} + i_t * g_t`
- `h_t = o_t * tanh(c_t)`

The recurrence is sequential in `t` but embarrassingly parallel in hidden index `h` within each `t`.

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

## 6.5. Pseudocode (paper-friendly)

Below is a simplified view of the optimized recurrence kernel. It omits details of memory layout and data types.

```text
Inputs:
  IH[g, t]  = precomputed W_ih * x for all gates g in [0..4H)
  W_hh[g, k] (k in [0..H))
  b_ih[g], b_hh[g]

State:
  h_prev[H], c_prev[H]
  h_next[H], c_next[H]

Parallelization:
  grid tiles hidden indices h across blocks
  each thread owns one h

for step = 0..T-1:
  t = (reverse ? T-1-step : step)
  for each hidden index h in parallel:
    for each gate group (i,f,g,o):
      dot = sum_{k=0..H-1} W_hh[gate(h), k] * h_prev[k]
      gate_pre = IH[gate(h), t] + dot + b_ih[gate(h)] + b_hh[gate(h)]
    i = sigmoid(gate_pre_i)
    f = sigmoid(gate_pre_f)
    g = tanh(gate_pre_g)
    o = sigmoid(gate_pre_o)
    c_next[h] = f * c_prev[h] + i * g
    h_next[h] = o * tanh(c_next[h])
    output[h, t] = h_next[h]
  grid_barrier_sync()
  swap(h_prev, h_next); swap(c_prev, c_next)
```

Key point: `grid_barrier_sync()` is required to preserve recurrence semantics across blocks.

### 6.4. Controlled rollout

The optimized kernel is gated by an environment variable so it can be disabled for comparison or fallback:

- `DIARIZATION_SEG_LSTM_COOP=1` enables the cooperative parallel kernel
- `DIARIZATION_SEG_LSTM_COOP=0` uses the legacy kernel

## 6.6. Correctness argument (informal)

The LSTM recurrence is a deterministic sequential program over time steps `t`:

1) compute all gates at time `t` from `h_{t-1}` and `x_t`
2) update `c_t` and `h_t`
3) advance to `t+1`

Within a fixed `t`, each hidden coordinate `h` depends on the entire previous vector `h_{t-1}[0..H)` but does not depend on the *current* values of other `h'` at the same `t`.

Therefore, for a fixed `t`, it is correct to compute all `h_t[h]` in parallel as long as:

- all reads use `h_{t-1}` (a stable buffer)
- all writes go to a separate `h_t` buffer (or are otherwise race-free)
- no thread advances to using `h_t` as the next step's previous state until all `h_t[h]` are written

The optimized kernel satisfies these by:

- using double-buffered `h_prev/h_next` and `c_prev/c_next`
- inserting a grid-wide barrier after finishing step `t` so that the swap to the next step is globally consistent

This yields equivalence to the sequential recurrence, up to floating-point arithmetic differences from instruction selection and execution order.

## 6.7. Numerical differences: expected sources

Even when the mathematical program is equivalent, GPU implementation changes can introduce small numerical differences:

- Different instruction sequences for sigmoid/tanh (fast math vs precise)
- Different intermediate precision (e.g., half -> float conversion placement)
- Different accumulation order (if dot products are parallel-reduced)
- Different use of fused multiply-add

In this particular optimization, the recurrence dot products are still accumulated in a fixed loop per hidden index in many implementations, but parallel scheduling and barrier placement can still affect rounding and timing.

Because downstream diarization includes thresholding and discrete decisions, small logit differences can become larger downstream differences.

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

## 7.4. Cooperative launch constraints (practical)

Cooperative launches can fail or silently fall back depending on hardware and resource usage.

Practical considerations to document in the paper:

- GPU must support cooperative launch.
- Kernel resource usage (registers, shared memory) constrains the maximum active blocks per SM.
- `grid.sync()` requires all participating blocks to be resident; therefore the launch grid size must be chosen conservatively.

In practice, the implementation should:

- detect cooperative support at runtime
- provide a fallback path
- allow benchmarking toggles

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

Additional optional metrics (useful in a paper):

- KL divergence per frame between the two distributions (requires converting log-probs to probs)
- Agreement of top-2 classes, or agreement after merging certain powerset classes
- Distribution of per-frame max abs diff (e.g., 50/90/99 percentiles)

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

## 9.5. Comparing against PyTorch (speed vs memory)

This project is not only a speed exercise. A key empirical observation is that the C++/ggml stack uses substantially less memory than PyTorch while still providing reasonable throughput.

Typical (example) footprint differences observed in practice:

- PyTorch CUDA inference: ~2-3 GB GPU memory usage for the segmentation model in a full environment.
- C++/ggml CUDA inference: ~200 MB total process footprint (order of magnitude), with small GGML buffers and lightweight runtime.

Important paper framing:

- Current CUDA C++ implementation may still be slower than PyTorch CUDA on the same GPU.
- However, the ggml-based implementation offers a drastically smaller memory footprint, which can be critical for deployment on constrained GPUs or multi-tenant servers.

### 9.5.1. How to measure memory consistently

Recommended measurements to report:

- Host RSS (peak): from `/proc/self/status` (VmHWM) on Linux.
- GPU memory: from `nvidia-smi` sampled while inference is running.

Example commands:

```bash
# GPU memory snapshot
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Host peak RSS (after run)
grep -n "VmHWM" /proc/$(pidof diarization-ggml)/status || true
```

For PyTorch, report:

- `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`
- plus `nvidia-smi` for external confirmation.

### 9.5.2. Template table (paper)

| Backend | Runtime | Model precision | Memory (GPU) | Memory (Host) | Notes |
| --- | --- | --- | ---: | ---: | --- |
| PyTorch CUDA | <fill> | fp32/fp16 | 2-3 GB | <fill> | cuDNN/fused kernels |
| ggml CUDA (legacy) | <fill> | f16 weights + f32 activations | ~<fill> | ~<fill> | custom LSTM kernel |
| ggml CUDA (optimized) | <fill> | same | ~<fill> | ~<fill> | cooperative recurrence |

## 9.6. Why PyTorch CUDA is faster today (hypotheses)

This section is useful for explaining performance gaps without overstating claims.

Common reasons PyTorch CUDA can outperform a custom kernel stack:

- Mature fused RNN kernels (e.g., cuDNN) with extensive tuning.
- Better kernel fusion across elementwise ops.
- Better use of Tensor Cores and mixed-precision pathways.
- Reduced launch overhead via internal graph optimizations.

For the ggml stack, the primary opportunity is to bring the recurrence kernel closer to these properties while preserving the low-memory design.

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

## 9.3. Measurement protocol (recommended)

To make results publication-grade:

- Fix GPU clocks if possible (or at least record power/temperature).
- Use a warmup phase to amortize one-time allocations and JIT effects.
- Repeat runs (e.g., >= 10) and report median and dispersion (IQR or std).
- For profiling runs, note that `nsys` introduces overhead; use it for attribution, not for final performance numbers.

Suggested reporting:

- End-to-end segmentation time (ms): median over N runs.
- Recurrence kernel avg latency (ms): from `cuda_gpu_kern_sum`.
- Total CPU sync time (ms): from `cuda_api_sum`.

## 9.4. How to collect profiling artifacts

Recommended files to archive with the paper (or as supplemental material):

- `*.nsys-rep` capture for baseline
- `*.nsys-rep` capture for optimized
- extracted `nsys stats` text outputs
- a script or exact command lines used to reproduce

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

### 10.3. Toward "instant" segmentation

Parallelizing recurrence improves occupancy, but the dominant work remains `W_hh * h_prev` dot products and nonlinearities.

To push further toward "instant" runtime on T4-class GPUs:

- Use half2 and vectorized loads for `W_hh` and `h_prev`.
- Increase arithmetic intensity by fusing gate computation and state update.
- Consider reformulating gate dot products as a small GEMM and leveraging Tensor Cores, while carefully managing the per-step launch overhead.
- Reduce host synchronizations by minimizing tensor downloads and deferring synchronization to a single boundary.

### 10.5. Roadmap: operator-level innovations

To close the remaining speed gap to PyTorch while keeping the memory footprint small, the most promising directions are operator-level:

1) Recurrence dot-product optimization
   - half2 vectorization for `W_hh` and `h_prev` loads
   - register tiling and unrolling for `H=128`
   - reduce global memory traffic using shared/L2-friendly layouts

2) Gate fusion
   - fuse gate computation + activations + state updates into a single kernel
   - reduce intermediate writes (especially `ih_all` materialization if feasible)

3) Precision engineering
   - keep weights in fp16, accumulate in fp32
   - optionally enable fast-math modes as an explicit trade-off with documented impact

4) Launch and scheduling
   - reduce kernel launch count where possible
   - minimize host-device synchronization points in the pipeline

## 10.4. Discussion: why sync shows up as a bottleneck

In `cuda_api_sum`, `cudaStreamSynchronize` often dominates. This is expected when the host waits for GPU completion between pipeline stages.

Important framing for a paper:

- A large `cudaStreamSynchronize` percentage does not mean synchronization itself is expensive.
- It means the host is idle while the GPU runs long kernels.
- Therefore, reducing the dominant kernel latency reduces the observed synchronization time.

This links the API summary directly to kernel attribution.

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

## Appendix B. Experimental Tables (templates)

### B.1. Hardware / Software Setup

Fill the following table for the paper:

| Item | Value |
| --- | --- |
| GPU | Tesla T4 (sm_75) |
| Driver | <driver_version> |
| CUDA toolkit | <cuda_version> |
| cuBLAS | <cublas_version> |
| OS | Linux <distro> |
| Compiler | <gcc/clang version> |
| Build flags | `-DGGML_CUDA=ON` ... |

### B.2. Runtime Configuration

| Parameter | Value |
| --- | --- |
| chunk duration | 10 s |
| chunk step | 1 s |
| frames/chunk | 589 |
| #chunks (sample.wav) | 21 |
| LSTM layers | 4 |
| hidden size | 128 |
| env toggles | `DIARIZATION_SEG_LSTM_COOP=0/1` |

### B.3. End-to-end Timing Summary

| Variant | Segmentation time (ms) | Total time (ms) | Speedup |
| --- | ---: | ---: | ---: |
| legacy | <fill> | <fill> | 1.0x |
| optimized | <fill> | <fill> | <fill> |

### B.4. nsys Kernel Summary (top kernels)

| Variant | Kernel | Calls | Avg (ms) | Total GPU (ms) | % |
| --- | --- | ---: | ---: | ---: | ---: |
| legacy | k_pyannote_seg_lstm_dir (fwd) | 84 | <fill> | <fill> | <fill> |
| legacy | k_pyannote_seg_lstm_dir (rev) | 84 | <fill> | <fill> | <fill> |
| optimized | k_pyannote_seg_lstm_dir_coop (fwd) | 84 | <fill> | <fill> | <fill> |
| optimized | k_pyannote_seg_lstm_dir_coop (rev) | 84 | <fill> | <fill> | <fill> |

### B.5. Logits-level Similarity

| Comparison | max_abs_diff | mean_abs_diff | argmax_agreement |
| --- | ---: | ---: | ---: |
| C++ legacy vs C++ optimized | <fill> | <fill> | <fill> |
| Python ref vs C++ legacy | <fill> | <fill> | <fill> |
| Python ref vs C++ optimized | <fill> | <fill> | <fill> |

## Appendix C. Reproducible evaluation protocol (paper-ready)

This appendix describes a minimal but publication-grade protocol.

### C.1. Separate profiling from benchmarking

- Use `nsys` runs for attribution (which kernel / which API dominates).
- Use plain runs (no profiler) for final reported timing numbers.

### C.2. Warmup and repetition

Recommended procedure for each variant (legacy vs optimized):

1) Warmup: run once to load models and allocate buffers.
2) Timed repetitions: run N times (e.g., N=10) and record segmentation time.
3) Report: median and interquartile range (IQR) for segmentation time.

### C.3. Commands to run both variants

Legacy:

```bash
unset DIARIZATION_SEG_LSTM_COOP
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
  -o /tmp/out_legacy.rttm
```

Optimized:

```bash
export DIARIZATION_SEG_LSTM_COOP=1
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
  -o /tmp/out_opt.rttm
```

Note: keep the input audio and chunking parameters identical.

### C.4. Capturing nsys artifacts

Baseline:

```bash
mkdir -p /tmp/nsys
unset DIARIZATION_SEG_LSTM_COOP
nsys profile -o /tmp/nsys/seg_legacy --force-overwrite=true --trace=cuda,nvtx,osrt \
  ./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
    models/segmentation-ggml/segmentation.gguf \
    models/embedding-ggml/embedding.gguf \
    samples/sample.wav \
    --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
    -o /tmp/out_legacy.rttm
```

Optimized:

```bash
mkdir -p /tmp/nsys
export DIARIZATION_SEG_LSTM_COOP=1
nsys profile -o /tmp/nsys/seg_opt --force-overwrite=true --trace=cuda,nvtx,osrt \
  ./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
    models/segmentation-ggml/segmentation.gguf \
    models/embedding-ggml/embedding.gguf \
    samples/sample.wav \
    --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
    -o /tmp/out_opt.rttm
```

Then summarize:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /tmp/nsys/seg_legacy.nsys-rep > /tmp/nsys/seg_legacy.stats.txt
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /tmp/nsys/seg_opt.nsys-rep    > /tmp/nsys/seg_opt.stats.txt
```

### C.5. Extracting kernel numbers into a table

`nsys stats` output is text. For a paper, it is often easiest to copy the top rows for the LSTM kernels into a markdown table.

If you need automation, use `nsys stats --format csv` and parse the CSV.

### C.6. Logits-level similarity reporting

Recommended reporting in the paper:

- Provide a table with max/mean abs diff and argmax agreement.
- Also provide a short qualitative statement: "Most frames preserve the same top-1 class; differences are concentrated around low-margin frames near decision boundaries."

## Appendix D. Complexity and throughput (sketch)

For each time step `t`, for each hidden index `h`, the dominant work is the dot product over `k in [0..H)` for four gates.

Per direction per step approximate multiply-adds:

- `4 * H * H` (four gates, each dot length H for each output h)

Per direction per chunk:

- `T * 4 * H * H`

For `T=589, H=128`, this is substantial. The baseline kernel under-utilizes GPU parallelism; the optimized kernel increases concurrency across `h`.

This appendix can be expanded into a full roofline-style argument in the paper (compute-bound vs memory-bound) once per-kernel memory traffic is quantified.

## Appendix E. Roofline-style analysis plan (how to quantify bottlenecks)

This appendix describes a practical plan to classify the optimized recurrence kernel as compute-bound or memory-bound.

### E.1. What to measure

The recurrence step does:

- read `h_prev[0..H)` and `c_prev[h]`
- read a slice of `W_hh` corresponding to four gates for one `h`
- read precomputed `IH[4H, t]` for this time step
- write `h_next[h]`, `c_next[h]`, and the output activation

At a high level, the ratio of floating-point operations to bytes moved (arithmetic intensity) will determine whether the kernel is compute- or memory-limited.

### E.2. Back-of-the-envelope arithmetic intensity

Per hidden coordinate `h` per time step:

- gate dot products: 4 dot products of length `H`.
- each dot contributes `H` multiply-adds, i.e., roughly `2H` FLOPs if counted as mul+add.

Approx FLOPs per `h` per step:

- `F_h ~ 4 * 2H = 8H` FLOPs

For `H=128`:

- `F_h ~ 1024` FLOPs per `h` per step

Bytes per `h` per step (rough approximation, float32 states):

- read `h_prev[H]`: `4H` bytes
- read `W_hh[4H, H]` row-slices for this `h`: `4H` weights, but each weight is used in the dot with all `k`, so total is `4H` weights *but each dot requires H weights; for one `h`, it reads `4 * H` weights (one weight per k per gate) -> `4H` weights -> `4 * H * sizeof(weight)`.
- read `IH[4]` gate inputs for this `h` at time `t`: 4 floats
- write `h_next[h]`, `c_next[h]`: 2 floats

This shows the dominant bandwidth term is `W_hh` and `h_prev` reads. In reality caching and reuse across threads will matter.

Paper note:

- A rigorous roofline requires measuring actual DRAM transactions or at least L2 hit rates.

### E.3. How to measure memory traffic

Two practical options:

1) Nsight Compute (preferred for roofline):

- capture kernel metrics such as DRAM bytes, L2 bytes, FLOP counts, achieved occupancy.
- report achieved FLOP/s and achieved bandwidth.

2) nsys + derived estimates:

- nsys alone is excellent for attribution and timing.
- it is not ideal for byte-level roofline; use it to isolate kernels, then switch to Nsight Compute.

If Nsight Compute is not available, a paper can still present:

- kernel time
- inferred FLOPs (from `T`, `H`, and gate equations)
- implied FLOP/s = FLOPs / time

### E.4. What to include in the paper

Minimal roofline table (one row per kernel variant):

| Variant | Kernel avg (ms) | Approx FLOPs (per call) | Approx TFLOP/s |
| --- | ---: | ---: | ---: |
| legacy | <fill> | <fill> | <fill> |
| optimized | <fill> | <fill> | <fill> |

Optionally add bandwidth if measured:

| Variant | DRAM GB/s | L2 hit rate | Occupancy |
| --- | ---: | ---: | ---: |
| legacy | <fill> | <fill> | <fill> |
| optimized | <fill> | <fill> | <fill> |

## Appendix F. Numerical stability and equivalence checklist

This appendix is a checklist for writing the "numerical equivalence" part of the paper.

### F.1. What is compared

- Per-frame powerset log-probabilities: shape `[chunks, frames, 7]`
- Optionally compare intermediate tensors (e.g., LSTM layer outputs) if available

### F.2. What metrics to report

Required:

- `max_abs_diff`
- `mean_abs_diff`
- `argmax_agreement`

Recommended additional metrics:

- 50/90/99 percentile of abs diff
- per-frame margin agreement (difference between top-1 and top-2 classes)
- KL divergence per frame (if values are true log-probabilities)

### F.3. How to interpret differences

- Large `max_abs_diff` can occur at rare frames without changing `argmax`.
- `argmax_agreement` close to 100% indicates most discrete decisions are preserved.
- Differences near decision boundaries are expected and can be amplified by thresholding.

### F.4. Threats to validity

When writing the paper, explicitly list threats:

- Profiler overhead (nsys changes timing)
- GPU frequency scaling and thermal throttling
- Non-determinism from GPU math modes (fast math, TF32, etc.)
- Model conversion differences between Python and C++ implementations
- Sensitivity of downstream diarization to small segmentation changes

Mitigations to mention:

- separate profiling from benchmarking
- repeat runs and report dispersion
- validate at logits level and also sanity-check downstream outputs

## Appendix G. Related-work positioning (non-citation version)

This work is positioned as a systems-style optimization:

- Identify bottlenecks via profiling.
- Preserve model semantics.
- Improve GPU utilization by restructuring the kernel.

It is complementary to:

- algorithmic improvements to diarization/segmentation models
- framework-level optimizations (cuDNN RNNs, XLA, TensorRT)

The novelty here is in optimizing a custom inference stack (ggml-based) where standard fused RNN kernels are not directly applicable.
