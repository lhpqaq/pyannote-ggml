# Segmentation / Embedding CUDA Debug Env Vars

This document lists the environment variables currently used in this repo for
CUDA scheduling/debugging during diarization experiments.

## Quick Answer

- To see **which ops are GPU-supported vs not supported** in segmentation:
  - `DIARIZATION_SEG_OP_GAP=1`
- To see **actual assigned GPU/CPU ratio at runtime**:
  - `DIARIZATION_DEBUG_BACKEND_ASSIGN_RATIO=1`

Use both together for full visibility.

## Runtime Coverage / Support

- `DIARIZATION_SEG_OP_GAP=1`
  - Prints segmentation per-op inventory:
    - `total`
    - `gpu_supported`
    - `missing`
    - support `%`
  - Source: `models/segmentation-ggml/src/model.cpp`

- `DIARIZATION_DEBUG_BACKEND_ASSIGN_RATIO=1`
  - Prints actual backend assignment ratio (GPU/CPU/other) for both
    segmentation and embedding graphs.
  - Source:
    - `models/segmentation-ggml/src/model.cpp`
    - `models/embedding-ggml/src/model.cpp`

## Segmentation Partition / Scheduling

- `DIARIZATION_SEG_GPU_PARTITION_MODE={classifier|linear|all}`
  - Enables segmentation experimental partition mode.
  - `classifier`: try offloading classifier matmul only.
  - `linear`: classifier + linear head matmuls.
  - `all`: allow broader offload path.
  - Usually set via wrapper env `SEG_GPU_PARTITION_MODE` in
    `tools/run-diarization.sh`.

- `DIARIZATION_SEG_BACKEND_STATS=1`
  - Forces segmentation backend stats/split behavior.
  - Primarily for reproducing scheduler split-related issues.

## Segmentation Debug Dumps / Metadata

- `DIARIZATION_SEG_DEBUG_DUMP_DIR=/tmp/seg-*`
  - Enables segmentation tensor binary dumps.

- `DIARIZATION_SEG_DEBUG_DUMP_MAX=<N>`
  - Max infer calls/chunks to dump.

- `DIARIZATION_SEG_DEBUG_TENSOR_META=1`
  - Prints tensor type/shape/stride/contiguous metadata for selected tensors.

- `DIARIZATION_SEG_DEBUG_ASSIGN=1`
  - Prints per-node backend assignment decisions.

- `DIARIZATION_SEG_DEBUG_BACKEND_MAP=1`
  - Prints backend map around classifier neighborhood nodes.

## SincNet / LSTM Experimental Paths

- `DIARIZATION_SEG_SINCNET_2D=1`
  - Uses SincNet conv/pool via 2D ops (`conv_2d`/`pool_2d`) instead of 1D path.

- `DIARIZATION_SEG_LSTM_NOCUSTOM_DEBUG=1`
  - Switches LSTM to non-custom debug graph path (experimental).
  - Not parity-safe by default; for debugging operator pathways.

## Force Backend for Isolation (Debug Only)

These are isolation switches in segmentation scheduler and are not intended for
production inference:

- `DIARIZATION_SEG_FORCE_ALL_GPU=1`
- `DIARIZATION_SEG_FORCE_CLASSIFIER_STAGE_CPU=1`
- `DIARIZATION_SEG_FORCE_CPU_IM2COL=1`
- `DIARIZATION_SEG_FORCE_CPU_MUL=1`
- `DIARIZATION_SEG_FORCE_CPU_ADD=1`
- `DIARIZATION_SEG_FORCE_CPU_SET=1`
- `DIARIZATION_SEG_FORCE_CPU_UNARY=1`
- `DIARIZATION_SEG_FORCE_CPU_VIEW=1`
- `DIARIZATION_SEG_FORCE_CPU_RESHAPE=1`
- `DIARIZATION_SEG_FORCE_CPU_SUB=1`
- `DIARIZATION_SEG_GPU_ONLY_FIRST_MUL=<N>`

## Practical Recipes

### 1) Show unsupported ops + actual assignment ratio

```bash
DIARIZATION_SEG_OP_GAP=1 \
DIARIZATION_DEBUG_BACKEND_ASSIGN_RATIO=1 \
./tools/run-diarization.sh samples/sample.wav cuda /tmp/out.rttm
```

### 2) Reproduce full-GPU stress for segmentation debug

```bash
CUDA_LAUNCH_BLOCKING=1 \
DIARIZATION_SEG_LSTM_NOCUSTOM_DEBUG=1 \
DIARIZATION_SEG_SINCNET_2D=1 \
DIARIZATION_SEG_FORCE_ALL_GPU=1 \
./tools/run-diarization.sh samples/sample.wav cuda /tmp/out_fullgpu.rttm
```

## Notes

- Some variables are temporary experimental knobs added for diagnosis and may be
  removed once the CUDA path is stabilized.
- `tools/run-diarization.sh` currently exports
  `DIARIZATION_SEG_LSTM_NOCUSTOM_DEBUG=1` by default in this branch state.
  If you want default model behavior, unset it before running or remove that
  export in the script.
