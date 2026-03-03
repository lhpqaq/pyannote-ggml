# CoreML Segmentation Implementation Notes (Paper Chapter Draft)

This document describes the CoreML-based implementation path for the segmentation model in `pyannote-ggml`.

It is intended as material for a small paper chapter/section complementing the CUDA optimization chapter.

## 1. Motivation

The segmentation model (PyanNet-style: SincNet -> BiLSTM -> MLP -> classifier) can be executed via multiple backends:

- GGML CPU
- GGML CUDA
- CoreML (Apple platforms)

CoreML is relevant because it provides a high-performance deployment path on Apple hardware (GPU and/or Neural Engine), and serves as a useful reference implementation when discussing system-level tradeoffs.

## 2. Model I/O Contract

The CoreML segmentation model is packaged as a `.mlpackage`.

Input:

- name: `waveform`
- shape: `(1, 1, 160000)`
- dtype: float32
- semantics: 10 seconds of mono waveform at 16 kHz

Output:

- name: `log_probabilities`
- shape: `(1, 589, 7)`
- dtype: float32 or float16 (depending on conversion settings)
- semantics: per-frame powerset log-probabilities (frame-major)

The diarization pipeline expects segmentation output as frame-major `[589][7]` for a single chunk.

## 3. CoreML Conversion Pipeline

Conversion script:

- `models/segmentation-ggml/convert_coreml.py`

Key design choices in the converter:

- A trace-friendly wrapper is constructed around the PyTorch model:
  - replaces `einops.rearrange` patterns with `.permute()`
  - freezes SincNet parametric sinc filters into a fixed `Conv1d` weight tensor
  - includes `log_softmax` in the forward pass so the CoreML model emits log-probabilities directly

High-level wrapper behavior (simplified):

1) waveform instance norm
2) SincNet conv stages (conv -> abs (stage0 only) -> pool -> norm -> leaky_relu)
3) permute to time-major for LSTM
4) LSTM
5) MLP (linear + leaky_relu)
6) classifier + log_softmax

Conversion:

- uses `coremltools.convert(..., convert_to="mlprogram")`
- targets macOS 13+ in the script
- uses float32 compute precision in the script (accuracy-first)

Paper note:

- For speed-first deployment, experimenting with float16 precision may increase throughput on Apple accelerators, but can affect numerical parity.

## 4. Runtime Bridge (C/ObjC++)

Bridge implementation:

- Header: `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.h`
- Implementation: `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`

Public API:

- `segmentation_coreml_init(path)`
- `segmentation_coreml_infer(ctx, audio_data, n_samples, output, output_size)`
- `segmentation_coreml_free(ctx)`

Runtime behavior:

1) Compile `.mlpackage` using `MLModel compileModelAtURL`.
2) Load model using `MLModel modelWithContentsOfURL`.
3) For inference, wrap `audio_data` as an `MLMultiArray` without copying.
4) Run `predictionFromFeatures` and extract `log_probabilities`.
5) Copy output into the caller-provided buffer.

### 4.1. Output layout and strides

CoreML returns an `MLMultiArray` with shape and strides. For correctness, the bridge should interpret the multi-array using its strides rather than assuming a contiguous layout.

This matters for:

- correctness across different CoreML runtime versions
- avoiding silent layout mismatches
- paper reproducibility (explicitly documenting assumptions)

## 5. Pipeline Integration

The diarization pipeline accepts a segmentation CoreML model via:

- CLI flag: `--seg-coreml <path/to/segmentation.mlpackage>`

When enabled and compiled with `SEGMENTATION_USE_COREML`, segmentation inference uses the CoreML path for each chunk.

Relevant call sites:

- `diarization-ggml/src/diarization.cpp`
- `diarization-ggml/src/streaming.cpp`

## 6. Validation and Reporting (paper)

Recommended validation strategy:

- Compare logits/log-probabilities at the segmentation output level:
  - CoreML vs PyTorch wrapper (converter already provides a sanity check)
  - CoreML vs PyTorch original model (where feasible)

Paper framing:

- Treat PyTorch as the reference implementation.
- CoreML is a deployment backend; numerical differences should be quantified relative to PyTorch, not relative to CUDA.

Recommended metrics:

- max/mean abs diff
- argmax agreement
- logsumexp sanity check for log-probabilities (should be ~0)

Paper-friendly reporting:

- Provide a small table comparing runtime and numerical similarity against PyTorch:

| Backend | Time / chunk | Time / file | max_abs_diff vs PyTorch | argmax agreement |
| --- | ---: | ---: | ---: | ---: |
| CoreML | <fill> | <fill> | <fill> | <fill> |
| PyTorch (reference) | <fill> | <fill> | 0 | 100% |

## 7. Chapter Outline (drop-in)

If this is included as a small paper chapter, a minimal structure is:

1) Motivation: why CoreML matters (deployment on Apple devices)
2) Conversion: wrapper + tracing + coremltools settings
3) Bridge: compile/load/predict, input/output layout
4) Integration: pipeline flag and usage
5) Results: runtime and similarity comparisons

Optional sub-section:

- Precision and deployment trade-offs:
  - float32 vs float16 conversion
  - compute units selection (CPU/GPU/ANE)
  - observed effect on similarity vs PyTorch
