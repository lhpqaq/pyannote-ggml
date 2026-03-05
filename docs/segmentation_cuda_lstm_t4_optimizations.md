# Segmentation CUDA BiLSTM Optimizations (T4-First)

This note documents the CUDA-side optimizations implemented for the PyanNet segmentation BiLSTM recurrence.
The focus is Tesla T4 (sm_75) first, while keeping the approach portable across NVIDIA GPUs.

Code lives in the `whisper.cpp` submodule:

- `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`

The optimizations are controlled via environment variables and are integrated into the segmentation custom-op
(`GGML_OP_CUSTOM` BiLSTM fast-path).

## Baseline Problem (Why LSTM Was Slow)

The segmentation model uses a 4-layer BiLSTM with fixed hidden size `H=128` and sequence length `T=589`.
In the original cooperative-kernel recurrence (`k_pyannote_seg_lstm_dir_coop`), each hidden unit `h` was computed
by one thread, and each thread iterated over all `k=0..H-1` to accumulate four dot-products (i/f/g/o gates).

Key bottleneck:

- `w_hh` is stored in ggml layout with `ne0 = H`, `ne1 = 4H` and treated as **column-major** with `ld = H`.
- In the 1-thread-per-hidden mapping, a warp’s lanes read `w_hh[col + k]` with `k` changing per thread in a
  strided way (stride `H`), which produces **poor memory coalescing**.
- The recurrence is step-serialized (must compute `h_t` from `h_{t-1}`), and in cooperative mode it executes
  `grid.sync()` each timestep. That makes “work per step” extremely important: if each step is slow, the barrier
  overhead becomes dominant.

## Optimization 1: Warp-Per-Hidden (Coalesced Weight Loads)

### High-level algorithm

We changed the recurrence mapping from *1 thread -> 1 hidden unit* to *1 warp -> 1 hidden unit*.

For each direction (forward/reverse), for each timestep:

1. A block cooperatively loads the previous hidden state vector `h_{t-1}` into shared memory (`sh_hp[k]`).
2. Each warp is assigned a hidden index `h`.
3. Within that warp, lanes split the dot-product over `k`:

   - lane computes partial sums for `k = lane, lane+32, lane+64, lane+96` (for `H=128`).
   - This makes `w_hh[col + k]` accesses **contiguous across lanes** for a fixed `col`, improving coalescing.

4. Partial sums for the four gates are reduced using warp shuffle (`__shfl_down_sync`).
5. Lane 0 computes sigmoid/tanh and updates `c_t` and `h_t`, writes `dst`.
6. `grid.sync()` enforces timestep order across blocks.

### Kernel

- `k_pyannote_seg_lstm_dir_coop_warp<REVERSE, W_T, B_T, WARPS_PER_BLOCK>`

### Tuning parameter

- `DIARIZATION_SEG_LSTM_COOP_WARPS` controls `WARPS_PER_BLOCK`.

On T4, `WARPS_PER_BLOCK=4` is the sweet spot.

### Expected numerical effect

The reduction order changes (warp-parallel partial sums + shuffle reduction). That can produce small FP32
rounding differences versus the baseline kernel.

## Optimization 2: H=128 Unroll (Reduce Loop Overhead)

For the warp-per-hidden kernel, we added a specialization for `H==128`:

- Replace the generic loop (`for (k = lane; k < H; k += 32)`) with a fixed 4-iteration unrolled form.
- Use `fmaf()` accumulation to keep the instruction sequence tight.

This tends to give a small but measurable speedup on T4.

## Optimization 3: Half2 Vectorization for `w_hh` (H=128)

To reduce instruction count and half->float conversion overhead further, we added a FP16-specialized warp kernel
that uses `half2` loads for `w_hh` when `H==128`.

### Kernel

- `k_pyannote_seg_lstm_dir_coop_warp_h2<REVERSE, B_T, WARPS_PER_BLOCK>`

### Inner loop strategy (H=128)

Each lane processes two `k` values at once:

- `k0 = 2*lane + 64*it` and `k1 = k0+1` for `it in {0,1}`.
- Load gate weights with `half2` (`wi2[k0/2]`, etc) and convert with `__half22float2`.
- Accumulate two FMAs per gate per iteration.

This reduces the number of weight loads and conversion instructions in the hot loop.

## Optimization 4: Remove Per-Step Shared `hp` Cache (Optional)

The warp kernels originally cached the previous hidden vector `hp` into shared memory each timestep.
This reduces repeated global reads of `hp`, but it adds per-step overhead:

- a block-wide loop that writes `H` floats into shared
- a `__syncthreads()` per step

On T4, for this specific kernel shape, reading `hp` directly from global memory can be faster.

### Kernel

- `k_pyannote_seg_lstm_dir_coop_warp_h2_nosh<REVERSE, B_T, WARPS_PER_BLOCK>`

### Enable

- `DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=1`

This keeps `grid.sync()` (timestep barrier) but removes the shared-memory staging and the per-step block
barrier.

### Numerical effect

Half2 itself is just a load/convert strategy here; the main numerical difference still comes from the parallel
reduction order in the warp mapping.

## Environment Variables / How To Use

These are read by the CUDA custom-op:

- `DIARIZATION_SEG_LSTM_COOP=1`
  - Enables cooperative recurrence (requires cooperative launch support and `B==1`).

- `DIARIZATION_SEG_LSTM_COOP_WARP=1`
  - Switches to warp-per-hidden cooperative kernel.
  - When enabled, the implementation currently uses the half2 specialized kernel for recurrent weights.

- `DIARIZATION_SEG_LSTM_COOP_WARPS={2|4|8|16}`
  - Number of warps per block.
  - T4 recommendation: `4`.

- `DIARIZATION_SEG_LSTM_COOP_WARP_NOSH=1`
  - Use the no-shared `hp` variant (direct global reads).
  - T4 observation: often faster than shared `hp` staging.

## Optimization 5: Fused Bidirectional Kernel (Optional)

The baseline implementation runs forward and reverse directions as separate cooperative launches.
Each direction kernel includes a `grid.sync()` per timestep, so the total barrier count is effectively doubled.

The fused bidirectional kernel computes both directions in a single cooperative launch:

- One cooperative kernel per layer per chunk instead of two
- One `grid.sync()` per timestep for both directions combined

### Kernel

- `k_pyannote_seg_lstm_bidir_coop_warp_h2_nosh<B_T, WARPS_PER_BLOCK>`

### Enable

- `DIARIZATION_SEG_LSTM_COOP_BIDIR=1`

This mode currently applies only to the warp-per-hidden cooperative path.

### Notes

- Fused bidirectional mode computes the reverse `ih` GEMM before launching the fused kernel.
- Correctness can be validated via tensor dumps; the fused and separate modes should match.

### Script defaults

`tools/run-diarization.sh` is set up to default to the faster warp-per-hidden kernel:

- `DIARIZATION_SEG_LSTM_COOP=1`
- `DIARIZATION_SEG_LSTM_COOP_WARP=1`
- `DIARIZATION_SEG_LSTM_COOP_WARPS=4`

Override by setting env vars before invoking the script.

## Profiling: Recommended Method (nsys)

Nsight Compute (`ncu`) may fail in restricted environments due to missing permission for GPU performance counters
(`ERR_NVGPUCTRPERM`). Nsight Systems (`nsys`) still works for timing and kernel attribution.

### Segmentation-only command

Use diarization with embeddings bypassed to isolate segmentation:

```bash
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
  -o /tmp/out.rttm
```

### nsys capture (baseline vs warp)

```bash
mkdir -p /tmp/nsys

# Baseline cooperative kernel
export DIARIZATION_SEG_LSTM_COOP=1
export DIARIZATION_SEG_LSTM_COOP_WARP=0
nsys profile -o /tmp/nsys/seg_base --force-overwrite=true --trace=cuda,nvtx,osrt \
  ./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
    models/segmentation-ggml/segmentation.gguf \
    models/embedding-ggml/embedding.gguf \
    samples/sample.wav \
    --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
    -o /tmp/out_base.rttm

# Warp-per-hidden (recommended)
export DIARIZATION_SEG_LSTM_COOP_WARP=1
export DIARIZATION_SEG_LSTM_COOP_WARPS=4
nsys profile -o /tmp/nsys/seg_warp --force-overwrite=true --trace=cuda,nvtx,osrt \
  ./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
    models/segmentation-ggml/segmentation.gguf \
    models/embedding-ggml/embedding.gguf \
    samples/sample.wav \
    --backend cuda --bypass-embeddings --plda diarization-ggml/plda.gguf \
    -o /tmp/out_warp.rttm

nsys stats --report cuda_gpu_kern_sum /tmp/nsys/seg_base.nsys-rep
nsys stats --report cuda_gpu_kern_sum /tmp/nsys/seg_warp.nsys-rep
```

## Correctness / Regression Checks (Tensor Dumps)

Segmentation dumps are emitted by `models/segmentation-ggml` when configured via env vars:

- `DIARIZATION_SEG_DEBUG_DUMP_DIR=/tmp/seg-dump`
- `DIARIZATION_SEG_DEBUG_DUMP_MAX=1`

The dump files use the pattern:

- `seg_<tensor_name>_<infer_index>.bin`

Relevant tensors:

- `lstm_out_cont`
- `linear1_mm`
- `linear2_mm`
- `classifier_mm`
- `classifier_out`

Compare dumps:

```bash
python3 tools/compare-seg-dumps.py /tmp/seg-a /tmp/seg-b
```

Notes:

- CUDA baseline vs CUDA warp should be very close, but not bit-identical.
- CPU vs CUDA can differ materially (existing behavior).

## Observed Results (T4, indicative)

On Tesla T4 with `--bypass-embeddings`:

- Baseline coop LSTM kernels dominate runtime (two directions, many invocations).
- Warp-per-hidden cuts the LSTM kernel time by ~2-3x.
- Half2 specialization provides an additional single-digit percent improvement.

The best-performing setting in sweeps so far is:

- `DIARIZATION_SEG_LSTM_COOP_WARPS=4`

## Known Limitations / Future Work

- Cooperative grid barriers (`grid.sync()`) are still executed per timestep; further improvements likely require
  increasing per-step efficiency (done so far) or rethinking synchronization strategy.
- Weight packing (reordering `w_hh` into a layout designed for warp access) may provide additional gains but
  increases complexity and memory usage.
- Nsight Compute requires GPU performance counter permissions; if unavailable, only `nsys` timing attribution is
  possible.
