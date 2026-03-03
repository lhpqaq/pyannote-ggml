# Operator-Level Optimizations for Segmentation (Paper Notes)

This document collects operator-level implementation details and optimization ideas for the segmentation model.

Scope:

- LSTM recurrence custom CUDA operator (primary bottleneck)
- Supporting operators that matter once LSTM is optimized

This is intended to support a paper section describing concrete engineering innovations.

## 1. Where the time goes

nsys profiling indicates segmentation GPU time is dominated by the custom BiLSTM recurrence kernels.

Therefore, operator-level work should prioritize the recurrence operator.

## 2. Custom LSTM operator decomposition

For each chunk, layer, and direction:

1) Precompute input contribution:

- `IH = W_ih * X` (GEMM)

2) Recurrent contribution:

- `W_hh * h_prev` for four gates
- apply nonlinearities
- update `c_t` and `h_t`

### 2.1. Data types and precision

Typical arrangement:

- weights: fp16
- activations/state: fp32
- accumulation: fp32

Rationale:

- fp16 weights reduce memory footprint and bandwidth
- fp32 accumulation improves stability

## 3. Baseline kernel pattern

The baseline recurrence kernel shows low occupancy and high per-launch latency. Its structure (conceptually):

- insufficient parallelism across hidden indices
- substantial sequential work per thread-block

Optimization focus: increase concurrency over hidden indices while preserving time-step ordering.

## 4. Cooperative recurrence kernel pattern

The optimized version tiles the hidden dimension across blocks and uses a grid-level barrier per time step.

Key benefits:

- increases SM utilization
- reduces per-call kernel latency

Key costs:

- cooperative launch constraints
- barrier overhead per time step

## 5. Next-step optimizations (paper-friendly list)

### 5.1. Half2 vectorization

Goal: increase memory throughput and arithmetic density.

Ideas:

- pack `h_prev` into `half2` where possible
- load `W_hh` in vectorized form
- accumulate in fp32

Risks:

- numerical differences
- alignment and layout constraints

### 5.2. Register tiling and unrolling for H=128

When `H` is fixed (128), specialize:

- unroll the dot product loop
- tile `h_prev` into registers or shared memory per block
- reduce repeated global reads

### 5.3. Gate fusion

Fuse:

- dot products + bias + nonlinearity + state update

Benefits:

- fewer intermediate arrays
- fewer global memory writes
- fewer kernel launches

### 5.4. Reduce IH materialization

Currently `IH` is materialized as a full `G x T` float buffer.

Potential improvement:

- compute `W_ih * x_t` on the fly per step
- or compute in tiles and stream into recurrence

Trade-off:

- may increase compute, reduce memory

### 5.5. Precision modes

Explicitly parameterize:

- precise vs fast-math sigmoid/tanh
- TF32 enable/disable (if used)

Document impact:

- speed vs logits similarity

## 6. How to present innovations in a paper

Suggested structure:

1) Baseline operator design and measured bottleneck.
2) Structural change (parallelize hidden indices + step barrier).
3) Micro-optimizations roadmap (half2, tiling, fusion).
4) Trade-offs: portability, numerical equivalence, deployment constraints.

Include:

- a kernel launch diagram
- a table of per-kernel avg latency
- a table of similarity metrics vs PyTorch
