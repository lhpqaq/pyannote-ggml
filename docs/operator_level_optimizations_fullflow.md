#+ Operator-Level Optimizations (CUDA + CoreML) for Diarization (Seg + Emb)

This note is an engineering-focused collection of operator/kernels/layout optimizations implemented in this repository's diarization pipeline.

Scope:

- Segmentation full flow: SincNet -> BiLSTM -> linear -> classifier -> powerset mapping
- Embedding full flow: fbank -> ResNet34 -> TSTP pooling -> linear embedding
- Backends:
  - CUDA (ggml-cuda custom operators + cuBLAS)
  - CoreML (conversion wrappers + runtime bridge)

Reference framing:

- PyTorch is treated as the reference implementation for correctness.
- CUDA and CoreML are deployment backends; the goal here is to record implementation details and non-trivial optimizations.

Key code entry points:

- Offline pipeline: `diarization-ggml/src/diarization.cpp`
- Streaming pipeline: `diarization-ggml/src/streaming.cpp`
- Segmentation model (GGML graph): `models/segmentation-ggml/src/model.cpp`, `models/segmentation-ggml/src/sincnet.cpp`, `models/segmentation-ggml/src/lstm.cpp`
- Embedding model (GGML graph): `models/embedding-ggml/src/model.cpp`, `models/embedding-ggml/src/fbank.cpp`
- CUDA custom BiLSTM fast-path: `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`
- CoreML bridges:
  - Segmentation: `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`
  - Embedding: `models/embedding-ggml/src/coreml/coreml_bridge.mm`

---

## 1. End-to-end pipeline: where operators matter

The offline diarization path (`diarization-ggml/src/diarization.cpp`) processes audio in sliding 10s chunks (160000 samples) with a 1s step. For each chunk:

- Segmentation produces powerset log-probabilities for 589 frames and 7 classes.
- A deterministic powerset-to-multilabel mapping (`diarization-ggml/src/powerset.cpp`) turns 7-class argmax into 3 local speaker activity tracks.
- Each local speaker track masks fbank frames and runs embedding inference.
- Downstream clustering (AHC + VBx) is CPU-side linear algebra over embeddings; it is not the core operator focus here.

From an operator perspective, the performance-critical neural portions are:

- Segmentation: the BiLSTM recurrence dominates CUDA time.
- Embedding: 2D convolutions and reduction-heavy TSTP pooling dominate operator count.

There are also important layout/copy steps (often as expensive as small operators on CPU-bound paths):

- Segmentation output layout mismatch (GGML vs expected frame-major) currently requires a transpose copy.
- Embedding input fbank layout mismatch (row-major frames) requires a transpose copy before ggml inference.

---

## 2. Segmentation (GGML) operator decomposition

### 2.1 SincNet: Conv1d + MaxPool1d + InstanceNorm1d

Implementation: `models/segmentation-ggml/src/sincnet.cpp`.

Operator-level details and optimizations:

- `DIARIZATION_SEG_SINCNET_2D=1` exists as an optional path to map some 1D ops onto 2D ops. This document does not expand on that route (per current project focus).

- **InstanceNorm1d via `ggml_norm`**: `ggml_instance_norm_1d()` uses `ggml_norm(ctx, x, eps)` and then applies affine weight/bias broadcast.
  - This leverages ggml's internal normalization primitive instead of materializing mean/var explicitly.

### 2.2 BiLSTM: represented as GGML_OP_CUSTOM

Implementation: `models/segmentation-ggml/src/lstm.cpp`.

The BiLSTM forward for each layer is built as a custom op node:

- `lstm_layer_bidirectional()` builds `ggml_custom_4d(..., args=9, n_tasks=2)`.
- The 9 sources are stable: `input + {w_ih, w_hh, b_ih, b_hh} * 2 directions`.

Why a custom op is valuable:

- It avoids decomposing LSTM into dozens of generic ggml ops (matmul, add, sigmoid/tanh, elementwise updates) which would increase graph node count and create intermediate tensors.
- It enables backend-specific implementations, especially CUDA.

CPU-side custom op optimization:

- When weights are on host memory, `lstm_init_weight_cache()` preconverts FP16 weights to FP32 and preallocates an `ih_all` buffer to avoid per-inference allocations.
- The custom op uses BLAS (`cblas_sgemm` / `cblas_sgemv`) for the input projection and recurrence dot products.

CUDA-side custom op implementation is handled in ggml-cuda (next section).

### 2.3 Linear + Classifier: layout-aware matmul

Implementation: `models/segmentation-ggml/src/model.cpp`.

Notable operator mechanics:

- Both linear layers and classifier reshape/permute input to 2D and call `ggml_mul_mat`.
- Explicit `ggml_cont()` calls are used after permutes/transposes to enforce contiguous layout, which is often required for backend kernels.
- Classifier applies `soft_max` + `log` inside ggml graph to produce log-probabilities.

---

## 3. Segmentation CUDA custom operator: BiLSTM fast-path

Implementation: `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`.

### 3.1 Recent git history (subproject)

The local `whisper.cpp` subproject contains a dedicated commit introducing the PyAnnote segmentation BiLSTM CUDA fast-path:

- Commit: `105f119d` (message: "pyannote")
- File touched: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Diff size: +612 lines, no deletions

This commit adds:

- A pattern matcher for a specific `GGML_OP_CUSTOM` shape/name signature.
- A CUDA implementation that uses cuBLAS for the input projection GEMM and custom CUDA kernels for the recurrent step.
- A cooperative-groups variant for parallelizing hidden dimension recurrence.

### 3.2 Dispatch mechanism: selective override for GGML_OP_CUSTOM

The CUDA backend's op dispatcher checks custom ops:

- In `case GGML_OP_CUSTOM`: if `ggml_cuda_is_pyannote_seg_lstm_custom(dst)` returns true, it executes `ggml_cuda_pyannote_seg_lstm_custom(ctx, dst)`.

The matcher enforces a strict contract:

- Types: `x` and output are FP32; weights are FP16; biases can be FP16 or FP32 (but consistent across bias tensors).
- Shapes: expects `x` as `[T, in, B, 1]` and output as `[T, 2H, B, 1]`.
- Names: uses stable tensor naming (`lstm.weight_ih_l*`, `lstm.weight_hh_l*`, and reverse names contain `_reverse`).

This signature-based dispatch is an explicit engineering tradeoff:

- Pro: no API changes needed in ggml; works with existing custom-op graph.
- Con: couples the CUDA implementation to the naming/shape conventions of this repository.

### 3.3 Kernel decomposition: GEMM + recurrent step

The CUDA path decomposes a direction as:

- **Input projection** (cuBLAS): `IH = W_ih^T @ X^T` producing a `[G, T]` float buffer (G = 4H).
- **Recurrence** (custom CUDA kernel): for each time step, computes the `W_hh` dot products, adds biases and `IH[:, t]`, applies sigmoid/tanh, updates `(c_t, h_t)`, and writes `h_t` to output.

Precision and data layout choices:

- Weights are FP16 to reduce bandwidth.
- Activations/state are FP32.
- GEMM computes FP32 output.
- A dedicated kernel `k_f32_to_f16` casts input X from FP32 to FP16 before GEMM.
  - Rationale in code: "cuBLAS does not reliably support (F16 x F32) GEMMEx across all builds".

Batch support:

- For `B > 1`, it uses `cublasGemmStridedBatchedEx` to compute each sequence's `IH` in one call.
- For `B == 1`, it uses `cublasGemmEx` and separate `IH` buffers for forward/reverse.

### 3.4 Cooperative recurrence route (the intended CUDA path)

This repository's intended CUDA execution route for segmentation BiLSTM is the cooperative-groups kernel, enabled via:

```bash
export DIARIZATION_SEG_LSTM_COOP=1
```

Activation conditions in code (`whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`):

- GPU supports cooperative launch: `ggml_cuda_info().devices[id].supports_cooperative_launch`
- env var `DIARIZATION_SEG_LSTM_COOP` is set and not equal to "0"
- batch size `B == 1`

The cooperative kernel `k_pyannote_seg_lstm_dir_coop<REVERSE, W_T, B_T, COOP_THREADS>`:

- Parallelization strategy: tile the hidden dimension `H` across blocks.
  - `COOP_THREADS` is currently `8`.
  - Launch: `grid = (ceil(H/COOP_THREADS), B, 1)`, `block = (COOP_THREADS, 1, 1)`.
- Synchronization: grid-level barrier `grid.sync()` once per time step.
- State storage: hidden/cell state stored in global memory buffers `h_state` / `c_state`, sized `[B, 2, H]` floats.
  - The kernel double-buffers state across steps by ping-ponging between the two slices (`step & 1`).

The legacy single-block recurrence kernel remains as a fallback when cooperative launch is unavailable.

---

## 4. Segmentation CoreML implementation details

Conversion: `models/segmentation-ggml/convert_coreml.py`.

- Uses a trace-friendly PyTorch wrapper that:
  - Freezes SincNet parametric sinc filters into a concrete `nn.Conv1d` weight tensor.
  - Replaces `einops.rearrange` with `.permute()`.
  - Includes `log_softmax` inside the model so CoreML output matches PyTorch log-probabilities.
- Converts to `mlprogram` targeting macOS 13, Float32 compute.

Runtime bridge: `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`.

Operator-level engineering choices:

- **Zero-copy input**: wraps caller-provided `audio_data` as an `MLMultiArray` using `initWithDataPointer` with explicit shape/strides.
- **Stride-aware output extraction**: reads `out_array.shape` and `out_array.strides` and copies output respecting layout.
  - This avoids relying on CoreML output contiguity assumptions; it is a robustness optimization (prevents silent wrong layout).
- **Float16 output supported**: the bridge includes a manual FP16->FP32 conversion path.

---

## 5. Embedding (GGML) operator decomposition

Implementation: `models/embedding-ggml/src/model.cpp`.

### 5.1 BatchNorm fusion at graph level

The embedding model implements inference-time batch norm as an explicit *scale + shift* fusion:

- `scale = weight / sqrt(running_var + eps)`
- `shift = bias - running_mean * scale`
- `y = x * scale + shift`

This is encoded in `batch_norm_2d()` and avoids a more expensive decomposition that would require multiple passes or explicit broadcasting tensors at every spatial location.

Backend implications:

- The fused form reduces op count and intermediate tensors in the ggml graph.
- It also aligns well with GPU kernels (elementwise mul/add with broadcast).

### 5.2 ResNet BasicBlock: conv2d + BN + ReLU + residual add

`basic_block()` builds a standard ResNet block using ggml ops:

- `ggml_conv_2d` for convs
- `batch_norm_2d` for BN
- `ggml_relu` for activations
- `ggml_add` for skip connection

This is a faithful operator mapping of the PyTorch reference forward.

### 5.3 TSTP pooling: reduction via matmul with ones

Temporal Statistics Pooling (TSTP) is implemented without a dedicated reduction op across time.

Instead, it constructs reductions using matmul:

- Flatten `[T/8, 10, 256]` into `feat_dim=2560`.
- Reorder to shape `[T8, feat_dim]` and compute:
  - `sum = (perm_t)^T @ ones`
  - `mean = sum / T8`
  - `sum_sq = (perm_sq_t)^T @ ones`
  - `var = E[x^2] - E[x]^2`

Engineering tradeoffs:

- Pro: leverages existing `ggml_mul_mat` kernels (which are typically well-optimized on GPU).
- Con: introduces extra permute/cont steps and allocates helper inputs (`ones`, `tstp_t8`, `tstp_t8m1`, eps scalars).

Numerical stability measures:

- Applies Bessel correction for unbiased variance (to match PyTorch default std behavior): multiplies by `T8/(T8-1)`.
- Uses `ggml_clamp(var_unbiased, 0, INF)` prior to sqrt to avoid tiny negative values from FP roundoff (noted as a CUDA-sensitive issue).

### 5.4 Input layout transpose: fbank row-major -> ggml layout

In `embedding::model_infer()`, the caller supplies fbank as `(T, 80)` row-major (frame-major contiguous).

The ggml input tensor is created as `[T, 80, 1, 1]` where `ne[0]=T` is the contiguous dimension. That requires a transpose copy:

- `state.fbank_transposed[b * T + t] = fbank_data[t * 80 + b]`.

This is correct but is a non-trivial per-chunk memory copy. It is a natural target for further operator-level optimization:

- Add a backend kernel for transposing fbank on GPU.
- Or restructure the ggml graph input layout to accept frame-major row-major directly (if backend supports non-contiguous strides).

---

## 6. Embedding CoreML implementation details

Conversion: `models/embedding-ggml/convert_coreml.py`.

- Wraps the PyTorch model to be trace-friendly:
  - Replaces `einops` with `.permute()`.
  - Uses `std(unbiased=False)` explicitly for CoreML trace compatibility.
- Converts to `mlprogram`, macOS 13, Float16 compute.
- Supports variable-length frame axis `T in [100, 2000]` via `RangeDim`.

Runtime bridge: `models/embedding-ggml/src/coreml/coreml_bridge.mm`.

- **Zero-copy input**: wraps `fbank_data` into `MLMultiArray` with shape `[1, T, 80]` and strides `[T*80, 80, 1]`.
- Output conversion supports FP16 and FP32.

Known robustness gap (engineering note / future work):

- Unlike segmentation, embedding bridge currently assumes output data is contiguous and uses `output.count` / `offset` heuristics.
- A stride-aware output copy (similar to segmentation) would make it robust to non-standard CoreML layouts.

---

## 7. Cross-cutting engineering optimizations

### 7.1 Weight backend allocation and small-memory posture

Both models follow the whisper.cpp pattern:

- Load GGUF metadata with `no_alloc=true`.
- Allocate weight buffers on a chosen backend (CPU or CUDA).
- Load weights into the backend buffer.

This explicitly controls memory residency and helps keep peak RSS/VRAM low compared to a typical PyTorch runtime.

### 7.2 Backend scheduler: multi-backend graph execution

Both segmentation and embedding use `ggml_backend_sched_*` APIs to:

- Pre-allocate compute buffers for a representative graph.
- Assign supported ops to preferred GPU backend.
- Compute the graph and then reset the scheduler state.

This reduces repeated allocation overheads and enables partial GPU offload (where some ops may still run on CPU depending on backend support).

---

## 8. Concrete operator-level optimization ideas (next steps)

These are *implementation-level* extensions consistent with the existing code structure.

Segmentation BiLSTM (CUDA):

- Fuse more of `W_hh * h_prev` into tensor-core friendly fragments (e.g., vectorize weight loads, reuse `h_prev` across gates).
- Explore half2 / vectorized activation approximations with explicit numerical validation vs PyTorch.
- Reduce global-memory traffic for cooperative kernel state by keeping per-block state in shared memory and using a structured handoff per step.

Embedding TSTP pooling:

- Introduce a dedicated reduction kernel (mean and variance) instead of matmul-with-ones to reduce permute/cont overhead.
- Move fbank transpose into a backend kernel to avoid CPU-side copies.

CoreML bridges:

- Make embedding output extraction stride-aware (mirror segmentation bridge logic).
- Consider using CoreML multiarray views that match the model's expected layout to avoid implicit transposes.
