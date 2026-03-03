# Segmentation CUDA BiLSTM (DIARIZATION_SEG_LSTM_COOP=1) Implementation Notes

This document records the exact CUDA operator implementation used for the segmentation model's BiLSTM when running via ggml-cuda.

This is *not* a high-level overview. It is a collection of details that are easy to lose (shapes, types, kernel launch configs, dispatch rules, and the precise decomposition).

Primary code: `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`.

## 0. What this optimizes

The segmentation model in GGML expresses each BiLSTM layer as `GGML_OP_CUSTOM` (see `models/segmentation-ggml/src/lstm.cpp`).

On CUDA, this repository adds a fast-path implementation for that custom op that:

- Uses cuBLAS for the input projection (the `W_ih * X` part).
- Uses a cooperative-groups recurrence kernel to parallelize across hidden units while preserving time-step dependency.

Enable the intended route:

```bash
export DIARIZATION_SEG_LSTM_COOP=1
```

## 1. Dispatch / pattern matching (how CUDA decides to run this)

Entry points:

- Matcher: `ggml_cuda_is_pyannote_seg_lstm_custom(const ggml_tensor * op)`
- Executor: `ggml_cuda_pyannote_seg_lstm_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- Dispatcher hook: in the CUDA op switch `case GGML_OP_CUSTOM`.

Matcher contract (must all hold):

- `op->op == GGML_OP_CUSTOM`.
- Exactly 9 source tensors are present (`src[0..8]`).
- Types:
  - input `x`: `GGML_TYPE_F32`
  - output `dst`: `GGML_TYPE_F32`
  - weights `w_ih`, `w_hh`, `w_ih_r`, `w_hh_r`: `GGML_TYPE_F16`
  - biases: either F16 or F32, but consistent across all bias tensors.
- Shapes:
  - `x` has layout `[T, in, B, 1]` in ggml `ne[]` terms (i.e. `ne[3] == 1`).
  - output `dst` has `ne[0] == T`, `ne[1] == 2H`, `ne[2] == B`, `ne[3] == 1`.
  - recurrent weight `w_hh` is `[H, 4H]` (ggml: `ne[0]=H`, `ne[1]=4H`).
  - bias vectors are `[4H]`.
- Naming:
  - `w_ih->name` prefix must match `"lstm.weight_ih_l"`
  - `w_hh->name` prefix must match `"lstm.weight_hh_l"`
  - reverse weights must include `"_reverse"`

This is intentionally strict. It ensures:

- The CUDA implementation can assume exact memory layout and shape arithmetic.
- The route only triggers for the intended segmentation BiLSTM custom op, not arbitrary user custom ops.

## 2. High-level decomposition inside ggml_cuda_pyannote_seg_lstm_custom

Given:

- `T = x->ne[0]` (time steps; segmentation uses 589)
- `in = x->ne[1]` (input feature dimension)
- `B = x->ne[2]` (batch; diarization uses B=1)
- `H = w_hh->ne[0]` (hidden size; segmentation uses 128)
- `G = 4*H` (gate size)

The implementation computes, per direction (fwd and reverse):

1) Input projection (GEMM):

- Compute `IH[t] = W_ih * X[t]` for all time steps.
- Materialize `IH` as a float32 buffer shaped `[G, T]` (per batch element).

2) Recurrent step (custom kernel):

- For each step t, compute:
  - dot products `W_hh * h_prev` for each of the 4 gates
  - add `IH[:, t]` and biases
  - apply sigmoid/tanh
  - update `(c_t, h_t)`
  - write `h_t` into output at the appropriate offset for direction.

The two directions write into a single output tensor `dst` in a packed format:

- forward writes to `dir_off = 0` (output hidden indices `[0..H)`)
- reverse writes to `dir_off = H` (output hidden indices `[H..2H)`).

## 3. Data types and why X is cast to FP16

Weights in GGUF are FP16. Input `x` is FP32.

The code explicitly casts `x` from FP32 to FP16 using:

- kernel `k_f32_to_f16(n, x_f32, x_f16)`

Then cuBLAS runs an FP16 x FP16 -> FP32 GEMM.

Rationale (captured as a comment in code):

- "cuBLAS does not reliably support (F16 x F32) GEMMEx across all builds. Cast X to F16 and use tensor-op GEMM (F16 x F16 -> F32)."

Implications:

- This is a deployment-robustness optimization, not just a speed tweak.
- It introduces a small numerical difference vs a pure FP32 input projection, but keeps recurrence in FP32.

## 4. GEMM details (fwd and reverse)

Pointers:

- `x_d`: FP32 input pointer (`x->data`)
- `x_h`: FP16 temporary buffer in CUDA pool
- `wih_f_d`, `wih_r_d`: FP16 weight pointers for forward/reverse

Two modes:

- If `B > 1`: use `cublasGemmStridedBatchedEx` with strides:
  - `stride_b = T * in` (one sequence worth of X)
  - `stride_c = G * T` (one sequence worth of IH)
  - `stride_a = 0` (same weights for all batch entries)
- If `B == 1`: use `cublasGemmEx` twice (fwd and rev) into separate buffers `ih_fwd` and `ih_rev`.

The IH buffer layout expected by recurrence kernels is:

- `ih_all[g + t*G]` for gate index `g` at time step `t`.

## 5. Cooperative recurrence kernel: k_pyannote_seg_lstm_dir_coop

This is the intended segmentation CUDA route when:

- `supports_cooperative_launch == true`
- `DIARIZATION_SEG_LSTM_COOP` is set and not "0"
- `B == 1`

Launch config in code:

- `COOP_THREADS = 8`
- `block = (8, 1, 1)`
- `grid  = (ceil(H/8), B, 1)`

Thread mapping:

- `b = blockIdx.y`
- `h = blockIdx.x * COOP_THREADS + threadIdx.x`
- `active = (h < H)`

State buffers:

- `h_state` and `c_state` allocated from ggml CUDA pool as float buffers sized `B*(2*H)`.
- They are zeroed with `cudaMemsetAsync` before each direction launch.

Ping-pong (double buffering):

- Two slices are used for prev/next state: (`h0/h1`, `c0/c1`).
- For step `step`, prev is `(step & 1) ? h1 : h0`, next is the other slice.

Per-step algorithm (for each active h):

- Compute four dot products over k=0..H-1 using the exact loop order:
  - `dot_i += w_hh[col_i + k] * hp[k]` and similarly for f/g/o.
- Load biases and `ih_all` contributions.
- Compute gates and nonlinearities:
  - sigmoid for i,f,o and tanh for g.
- Update:
  - `c_new = f_val * cp[h] + i_val * g_val`
  - `h_new = o_val * tanh(c_new)`
- Store state to `cn[h]` / `hn[h]`.
- Store output:
  - `dst[(dir_off + h) * T + t] = h_new`

Synchronization:

- `grid.sync()` occurs once per time step.
- This is required so that all threads see a fully updated `hn/cn` before advancing to the next step.

Why this is non-trivial / innovative in practice:

- LSTM recurrence has a strict dependency along time; naive CUDA ports often serialize everything in one block.
- Using cooperative launch plus a grid barrier allows distributing hidden units across multiple blocks (more SMs), while preserving the recurrence ordering.

## 6. Legacy recurrence kernel (fallback)

`k_pyannote_seg_lstm_dir<REVERSE, ...>` remains as fallback.

Key properties:

- One CUDA block per batch element, fixed `legacy_threads = 128`.
- Hidden state and cell state are stored in shared memory.
- Synchronization uses `__syncthreads()` per step.

This fallback is not the intended route for this repository when cooperative launch is available.

## 7. Notes for diarization usage

In `diarization-ggml/src/diarization.cpp` segmentation is executed chunk-by-chunk, i.e. batch size `B == 1`.

Therefore the cooperative route is designed to cover the dominant deployment scenario.
