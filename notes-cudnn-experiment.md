# cuDNN LSTM Experiment (Archived)

This repo's diarization pipeline uses a pyannote segmentation model that
contains a 4-layer BiLSTM. The CUDA custom-op implementation for that BiLSTM
(`GGML_OP_CUSTOM`) was profiled and found to dominate GPU time on T4.

This document summarizes an experiment that attempted to replace that custom
CUDA kernel path with cuDNN RNN forward inference.

## Why

- Profiling (`nsys`) showed segmentation time was dominated by the custom LSTM
  recurrence CUDA kernels, not by graph build/alloc.
- Goal: use cuDNN's RNN implementation to improve throughput.

## What Was Implemented (Patch-Only)

All changes were made inside the `whisper.cpp` submodule (specifically the vendored
`ggml` CUDA backend). Because we do not commit to the submodule, the work is
archived as a patch file in the main project:

- `patches/whisper.cpp/0003-ggml-cuda-cudnn-lstm-experiment.patch`

The patch contains:

- Build-time option `GGML_CUDA_CUDNN` (default OFF) added to `ggml-cuda` to locate
  `cudnn.h` and link `libcudnn`.
- A cuDNN-based forward inference path for the pyannote segmentation BiLSTM
  custom op, guarded by compile-time `GGML_CUDA_USE_CUDNN`.
- Runtime env controls:
  - `GGML_CUDA_PYANNOTE_LSTM_CUDNN=1` enable cuDNN path
  - `GGML_CUDA_PYANNOTE_LSTM_CUDNN_FORCE=1` abort if cuDNN fails (avoid silent fallback)
  - `GGML_CUDA_PYANNOTE_LSTM_CUDNN_VERBOSE=1` extra logs
  - `GGML_CUDA_PYANNOTE_LSTM_CUDNN_BIAS=1` experimental bias packing toggle

## Approach

- Use `cudnnSetRNNDescriptor_v8` + `cudnnRNNForward` for inference.
- Pack GGML-format weights into cuDNN `weightSpace` via `cudnnGetRNNWeightParams`.
- Convert input/output layouts between GGML's tensor layout and cuDNN's
  seq-major unpacked layout.

## Validation Used During Development

- Correctness checks (segmentation logits):
  - Run `segmentation-ggml` with a fixed 10s input and compare saved logits.
- Performance checks:
  - `nsys profile` and `nsys stats` to confirm the kernel time shifted away from
    the custom `k_pyannote_seg_lstm_dir` kernels.

## Outcome

- The experiment successfully executed cuDNN RNN kernels and reduced segmentation
  compute time for the LSTM component.
- However, even small numerical differences in segmentation outputs can be
  amplified by downstream steps (e.g. strict floating comparisons in embedding
  filtering), potentially changing end-to-end diarization output.
- Decision: archive the experiment as a patch and revert the repo to the
  non-cuDNN implementation.
