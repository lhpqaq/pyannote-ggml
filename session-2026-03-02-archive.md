# Session Archive (2026-03-02)

This file freezes the current working state before cleanup.

## Repo State

- Superproject: `pyannote-ggml`
- Current branch at time of archive: `cuda`
- Intent: segmentation runs full-GPU on CUDA backend (small graph) by executing the pyannote BiLSTM `GGML_OP_CUSTOM` on CUDA.

## Key Implementation Notes

- `whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`
  - CUDA fast-path for pyannote segmentation BiLSTM custom op.
  - cuBLAS precompute + CUDA recurrence kernel.

## Downstream Patch

- Because this repo uses `whisper.cpp` as a submodule and downstream users may not have
  submodule push permissions, the corresponding upstream patch is saved here:
  - Base whisper.cpp commit: `9c807c8056f8b3da89ede02015cef942f77de01b`
  - Apply in order:
    - `patches/whisper.cpp/0001-whisper-ggml-cuda-pool1d-and-binbcast-fix.patch`
    - `patches/whisper.cpp/0002-whisper-ggml-cuda-pyannote-seg-custom-op.patch`

## Repro Commands (example)

Build:

```bash
cmake --build diarization-ggml/build-x86-cuda -j8
```

Run (CUDA backend):

```bash
./diarization-ggml/build-x86-cuda/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --plda diarization-ggml/plda.gguf \
  --backend cuda \
  -o /tmp/out.rttm
```

## Cleanup Plan (post-archive)

- Remove debug-only logs, env toggles, and historical experimental paths.
- Keep only the final correct behavior and minimal user-facing configuration.
