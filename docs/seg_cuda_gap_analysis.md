# Segmentation (CUDA) vs PyTorch Gap Analysis And Optimization Notes

This document analyzes why the segmentation model's CUDA path in this repo is still far slower than the PyTorch reference on the same GPU, and records the optimizations attempted here.

Scope:

- Hardware used for measurements: Tesla T4 (SM 7.5)
- Repo binary used: `diarization-ggml/build-x86-cuda/bin/diarization-ggml`
- Run mode: segmentation-only path via `--bypass-embeddings`
- Intended CUDA kernel route: `DIARIZATION_SEG_LSTM_COOP=1`

## 1. Baseline Measurements

### 1.1 PyTorch reference (CUDA)

In the `conda activate diar` environment, a rough microbenchmark of a 4-layer bidirectional LSTM with matching sequence length and hidden size:

- Shape: input `(B=1, T=589, in=60)`, hidden `H=128`, 4 layers, bidirectional
- Result on T4: ~12.6 ms per forward (LSTM only)

This is not the full segmentation model (SincNet + LSTM + MLP), but it sets an upper bound expectation: the LSTM component can be ~O(10ms) with cuDNN.

### 1.2 This repo (GGML + CUDA backend)

Command (segmentation-only, 30s audio => 21 chunks):

```bash
export DIARIZATION_SEG_LSTM_COOP=1
diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --plda diarization-ggml/plda.gguf \
  --backend cuda --gpu-device 0 \
  --bypass-embeddings \
  -o /tmp/out.rttm
```

Observed segmentation time (typical): ~0.9s for 21 chunks (~43 ms per chunk).

This is orders of magnitude slower than a "near-instant" PyTorch forward, hence the gap.

## 2. nsys Attribution: Where Time Goes

Profiling with `nsys` shows that GPU kernel time is dominated by the custom BiLSTM recurrence kernels:

- `k_pyannote_seg_lstm_dir_coop<forward>`
- `k_pyannote_seg_lstm_dir_coop<reverse>`

Each is launched 84 times in the 21-chunk run, corresponding to:

- 4 layers x 2 directions x 21 chunks = 168 launches
- The count 84 per direction matches exactly.

The CUDA API summary also shows large time in `cudaStreamSynchronize`, indicating that per-graph scheduling and synchronization overhead is significant and that the pipeline is not fully overlapped.

## 3. Primary Root Cause

### 3.1 The recurrence kernel is not cuDNN

PyTorch uses cuDNN's fused RNN kernels for LSTM. Those kernels are heavily optimized:

- gate fusion
- tensor-core / HMMA paths
- internal persistent kernels and layout packing
- minimal synchronization in the recurrent loop

This repo's BiLSTM path decomposes the computation into:

- cuBLAS GEMM: `W_ih * X` (input projection)
- custom CUDA kernel: recurrence loop over time steps with explicit `grid.sync()` per step

The recurrence loop is the critical bottleneck.

### 3.2 Cooperative-groups `grid.sync()` is expensive at T=589

Even when per-step math is small, a grid-level barrier 589 times per kernel launch produces heavy overhead, especially when blocks-per-grid is not large.

On T4, with H=128, splitting H across blocks yields at most 16 blocks (when threads-per-block=8) per direction. That limits occupancy and amplifies barrier overhead.

## 4. Optimizations Implemented / Tested Here

### 4.1 Fast exp for sigmoid

Change:

- In `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`, replaced `expf` with `__expf` inside sigmoid.

Motivation:

- `__expf` is significantly faster on NVIDIA GPUs and aligns with typical "fast math" choices.

### 4.2 Tune cooperative threads per block (and why it was reverted)

Attempt:

- Increased cooperative threads-per-block from 8 to 32.

Result:

- Kernel runtime increased substantially (nsys showed ~8.8ms median vs ~4.7ms median at 8 threads).

Conclusion:

- For this kernel shape on T4, the default 8 threads per block performed best.
- The optimal setting is hardware-sensitive; it should be runtime-tunable, but any change must be benchmarked.

Implementation outcome:

- Added env var `DIARIZATION_SEG_LSTM_COOP_THREADS` (allowed values: 8/16/32), defaulting to 8.

### 4.3 Shared-memory cache of previous hidden state (hp)

Attempt:

- In the cooperative kernel, cached the previous hidden state `hp[0..H-1]` into shared memory per block per time step.

Result:

- GPU kernel time improved modestly in isolation (a few percent), but overall end-to-end effect was dominated by other overheads.

### 4.4 Graph caching experiment (reverted)

Attempt:

- Tried to cache the fixed-shape graph in `models/segmentation-ggml/src/model.cpp` across chunk calls.

Outcome:

- Caused backend buffer issues (`GGML_ASSERT(buf != NULL && "tensor buffer not set")`) and/or wrong outputs.
- Root cause is likely that `ggml_backend_sched_alloc_graph()` and the scheduler internals assume a graph lifetime aligned with the build/alloc/compute/reset cycle.

Decision:

- Reverted graph caching. Correctness is non-negotiable.

## 5. Recommendations / Next Steps

If the goal is to match PyTorch speed on CUDA, the realistic path is:

1. cuDNN-based LSTM forward inference for the BiLSTM custom op.
   - This repo previously experimented with it (see `notes-cudnn-experiment.md`), but it was archived as a patch.
   - Re-introducing it in a controlled manner, with strong correctness tests, is the highest leverage.

2. Reduce CPU-side synchronization overhead.
   - `cudaStreamSynchronize` dominates API time in profiling outputs.
   - Investigate where sync happens in ggml scheduler, and whether the segmentation path can be executed with fewer mandatory sync points.

3. Improve recurrence kernel arithmetic intensity.
   - A more aggressive kernel would compute multiple hidden units per thread (vectorize) and use warp-level primitives.
   - However, the presence of grid.sync() is still a fundamental limiter.

## 6. Artifacts

- Example nsys captures were saved under `/tmp/nsys/` during experimentation.
- To reproduce:

```bash
mkdir -p /tmp/nsys
export DIARIZATION_SEG_LSTM_COOP=1
nsys profile -o /tmp/nsys/seg_only --force-overwrite=true --trace=cuda,nvtx,osrt \
  diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --plda diarization-ggml/plda.gguf \
  --backend cuda --gpu-device 0 --bypass-embeddings \
  -o /tmp/out.rttm

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /tmp/nsys/seg_only.nsys-rep
```
