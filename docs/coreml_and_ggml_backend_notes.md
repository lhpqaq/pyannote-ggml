# CoreML and GGML Backend Implementation Notes (Segmentation + Embedding)

This document collects concrete implementation details for the CoreML and GGML (CPU/CUDA) backends in this repository, based on code inspection.

The intent is to preserve non-obvious engineering decisions: tensor shapes/layouts, conversion wrapper choices, and runtime bridge behavior.

## 1. Segmentation (CoreML)

### 1.1 Conversion script

File: `models/segmentation-ggml/convert_coreml.py`

What it does:

- Loads PyTorch model `pyannote/segmentation-3.0`.
- Wraps it in `PyanNetWrapper` to be trace-friendly:
  - Freezes SincNet parametric sinc filterbank into a normal `nn.Conv1d` weight.
  - Uses `.permute()` instead of `einops.rearrange`.
  - Includes `F.log_softmax(..., dim=-1)` in-graph so output matches log-probabilities.
- Traces via `torch.jit.trace`.
- Converts to CoreML `mlprogram`.

I/O contract (as converted):

- Input: `waveform` shape `(1, 1, 160000)` float32 (fixed 10s chunk at 16kHz)
- Output: `log_probabilities` shape `(1, 589, 7)`

### 1.2 Runtime bridge

File: `models/segmentation-ggml/src/coreml/segmentation_coreml_bridge.mm`

Key behaviors:

- Compiles `.mlpackage` at runtime using `+[MLModel compileModelAtURL:error:]`.
- Loads `MLModel` with `MLComputeUnitsAll`.

Zero-copy input:

- Wraps caller-owned `audio_data` via `initWithDataPointer`.
- Explicitly provides shape `[1, 1, n_samples]` and strides `[n_samples, n_samples, 1]`.
- The deallocator is a no-op (caller retains ownership).

Stride-aware output extraction:

- Reads `out_array.shape` and `out_array.strides`.
- Copies batch 0 into caller buffer as frame-major `[T][C]` (`output[t*C + c]`).
- Supports output dtype float16 and float32.
  - float16 path performs manual FP16->FP32 conversion.

Why stride-awareness matters:

- CoreML may return `MLMultiArray` with non-contiguous strides.
- Reading with assumed contiguous indexing can silently permute data.

## 2. Segmentation (GGML, CPU/CUDA)

### 2.1 Graph construction and scheduler

File: `models/segmentation-ggml/src/model.cpp`

- Builds a ggml graph per inference using a metadata-only context backed by `state.graph_meta`.
- Uses `ggml_backend_sched_*` to allocate and compute the graph.
- Tries to place supported ops on GPU via `prefer_gpu_for_supported_nodes()`.

Important detail:

- LSTM is expressed as `GGML_OP_CUSTOM` in `models/segmentation-ggml/src/lstm.cpp`.
- The CUDA backend has a special-case implementation for this custom op (see `docs/segmentation_cuda_lstm_coop_route.md`).

### 2.2 Output layout in diarization pipeline

File: `diarization-ggml/src/diarization.cpp`

- CoreML segmentation output is already `[589][7]` frame-major.
- GGML segmentation output tensor is stored class-major and is explicitly transposed into frame-major before calling `powerset_to_multilabel()`.

This transpose is currently implemented as a temporary `std::vector<float> tmp` + nested loops.

## 3. Embedding (CoreML)

### 3.1 Conversion script

File: `models/embedding-ggml/convert_coreml.py`

What it does:

- Loads PyTorch model `pyannote/wespeaker-voxceleb-resnet34-LM`.
- Wraps resnet forward in `ResNetEmbeddingWrapper`:
  - Replaces `einops` with `.permute()`.
  - Uses `std(unbiased=False)` for trace compatibility.
- Converts to CoreML `mlprogram` with Float16 compute.
- Uses a variable-length time dimension `T` via `RangeDim(lower=100, upper=2000, default=998)`.

I/O contract:

- Input: `fbank_features` shape `(1, T, 80)` float32
- Output: `embedding` shape `(1, 256)` (float16 or float32 depending on CoreML execution).

### 3.2 Runtime bridge

File: `models/embedding-ggml/src/coreml/coreml_bridge.mm`

Key behaviors:

- Zero-copy input via `initWithDataPointer` with shape `[1, T, 80]` and strides `[T*80, 80, 1]`.
- Output conversion supports float16 and float32.

Robustness note:

- Output extraction currently uses `output.count` and an `offset` heuristic and assumes contiguous indexing.
- Unlike segmentation, it does not read `output.strides`.
- If CoreML returns a non-standard stride layout, this could misread output.

## 4. Embedding (GGML, CPU/CUDA)

### 4.1 Fbank feature extraction

File: `models/embedding-ggml/src/fbank.cpp`

- Uses `kaldi-native-fbank`.
- Applies waveform scaling by 32768 (WeSpeaker convention).
- Applies global CMN (subtract mean per mel bin over all frames).

### 4.2 Graph construction details

File: `models/embedding-ggml/src/model.cpp`

BN fusion (inference form):

- Implements BN as `y = x * scale + shift` where:
  - `scale = weight / sqrt(running_var + eps)`
  - `shift = bias - running_mean * scale`

TSTP pooling (stats pooling):

- Flattens `[T/8, 10, 256]` into `feat_dim=2560`.
- Computes mean and variance over time using matmul against a ones vector.
- Applies Bessel correction and clamps variance to non-negative before sqrt.

Input layout transpose:

- Caller provides fbank as `(T,80)` row-major.
- GGML input is `[T,80,1,1]` with `ne[0]=T` contiguous.
- Implementation transposes into `state.fbank_transposed` before `ggml_backend_tensor_set()`.
