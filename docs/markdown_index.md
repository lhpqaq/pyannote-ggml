# Markdown Index (Excluding whisper.cpp)

Generated from the repository `pyannote-ggml`.

- Scope: all `*.md` files excluding the `whisper.cpp/` subtree
- Notes:
  - Files prefixed with `session-` are usually historical conversation logs / snapshots.
  - `.sisyphus/` contains plans and engineering notepads (often useful, but not always authoritative).

## A. Project Entry Points And Conventions

- `README.md`
  - Project overview: native C++ speaker diarization pipeline (pyannote port) with GGML + CoreML, plus optional Whisper transcription integration.
  - Quick-start build/model-conversion/run commands; documents offline and streaming APIs and key constants (10s chunk, 1s hop, 589 frames, 3 local speakers, 256-d embeddings).

- `AGENTS.md`
  - Coding agent guidance, repo map, build/test commands, and style constraints.

- `DEPLOYMENT_ZH.md`
  - Chinese deployment guide: macOS/Apple Silicon environment setup, Python/coremltools version notes, HF token requirements, model conversion, and runtime examples.

## B. Backend / Platform Runbooks

- `METAL_RUNBOOK.md`
  - Metal build/run recipe and behavioral notes (fail-fast when Metal not compiled in).

- `CUDA_CONTEXT.md`
  - CUDA x86 bring-up context: correctness-first stance, known failures/guardrails, build dirs, and next diagnostic steps.

## C. Whisper + Diarization Integration (Design Docs)

- `INTEGRATION_PLAN.md`
  - Detailed streaming integration design: silence filter (filtered timeline), audio buffer timestamping, segment-end detection from pyannote VAD frames, Whisper chunking rules, and WhisperX-style alignment.

- `INTEGRATION_PLAN_ZH.md`
  - Chinese version of `INTEGRATION_PLAN.md`.

- `diarization_whisper_integration.md`
  - Earlier English design sketch for the same general pipeline.

- `DIARIZATION_WHISPER_INTEGRATION_ZH.md`
  - Earlier Chinese design sketch for the same general pipeline.

## D. Streaming Diarization Design And Tools

- `diarization-ggml/STREAMING_DESIGN.md`
  - Streaming diarization design with periodic reclustering; explains state growth requirements (embeddings + binarized must retain full history for offline-identical reclustering).

- `tools/streaming-viewer/README.md`
  - Flask-based viewer that visualizes streaming diarization events emitted by `streaming_test --json-stream`.

## E. Model Subprojects (Segmentation / Embedding)

- `models/segmentation-ggml/README.md`
  - Segmentation model GGML implementation notes: status, accuracy checks vs PyTorch, build/convert/run/test entry points.

- `models/segmentation-ggml/docs/architecture.md`
  - Long, layer-by-layer architecture doc with shapes/params for PyanNet (SincNet + 4-layer BiLSTM + MLP + 7-class powerset).

- `models/embedding-ggml/kaldi-native-fbank/README.md`
  - Upstream dependency readme: Kaldi-compatible online fbank feature extraction.

- `models/embedding-ggml/kaldi-native-fbank/cmake/Modules/README.md`
  - Upstream note: FetchContent module source origin.

- `models/embedding-ggml/kaldi-native-fbank/toolchains/README.md`
  - Upstream note: ARM toolchain download/extract instructions.

## F. Engineering / Paper Materials Under `docs/`

- `docs/coreml_and_ggml_backend_notes.md`
  - Code-inspection notes: CoreML conversion + runtime bridges, GGML CPU/CUDA graph construction, layout/transpose quirks, and scheduler behavior.

- `docs/coreml_segmentation_implementation.md`
  - Paper-style chapter draft: CoreML segmentation I/O contract, conversion wrapper decisions, runtime bridge behavior (zero-copy + stride-aware output), and validation recommendations.

- `docs/operator_level_optimizations.md`
  - Segmentation operator-level optimization notes (LSTM recurrence bottleneck + next-step ideas).

- `docs/operator_level_optimizations_fullflow.md`
  - Full-flow operator/layout notes for segmentation + embedding + CUDA + CoreML, including the contract-based CUDA custom-op dispatch for segmentation BiLSTM.

- `docs/segmentation_cuda_lstm_coop_route.md`
  - Implementation-contract doc for the CUDA cooperative BiLSTM route (env toggles, shape/type/name matching, GEMM + recurrence decomposition, kernel launch, grid.sync semantics).

- `docs/segmentation_cuda_lstm_parallel_optimization.md`
  - Paper-oriented narrative: profiling evidence -> kernel redesign -> validation methodology -> representative outcomes.

- `docs/context_segmentation_cuda_bottleneck_summary.md`
  - Compressed context snapshot: stable findings, discarded paths, current main direction, minimal repro/profiling commands.

- `docs/thesis_structure.md`
  - Chinese thesis structure guidance: how to frame the work academically (graph translation, semantic equivalence, backend contracts, etc.).

- `docs/thesis_ch4_ch5_materials_to_extract.md`
  - Checklist of code materials to extract for thesis chapters 4/5 with concrete file/function pointers.

## G. Academic Drafts

- `CHAPTER4_ACADEMIC.md`
  - Chinese chapter draft: PyTorch->GGML graph translation, operator mapping, memory pool/lifecycle, PLDA+AHC+VBx complexity framing, etc.

## H. Archived Experiments

- `notes-cudnn-experiment.md`
  - Archived attempt to replace the CUDA LSTM custom path with cuDNN RNN inference; kept as a patch due to end-to-end sensitivity to small numerical diffs.

## I. Session / Historical Logs (Likely Historical Context)

- `session-2026-03-02-archive.md`
  - Snapshot of a CUDA-focused working state and downstream whisper.cpp patch references.

- `session-changshigpu.md`
  - Conversation log around forcing segmentation onto CUDA and subsequent rollbacks/experiments.

- `session-ses_3572_huijia.md`
  - Conversation log including seg dump comparison tooling experiments.

- `session-ses_35b4.md`
  - Older conversation log; includes environment-specific noise (paths/LSP).

- `session-论文.md`
  - Session stub / compaction residue; low information density.

## J. Node Bindings

- `bindings/node/packages/pyannote-cpp-node/README.md`
  - Node.js native addon API: shared model cache, offline/one-shot/streaming session modes, event semantics, and TypeScript types.

## K. Third-Party / Build-Artifact READMEs (Low Priority)

- `diarization-ggml/build/_deps/kissfft-src/README.md`
- `diarization-ggml/build-linux/_deps/kissfft-src/README.md`
- `diarization-ggml/build-x86-cuda/_deps/kissfft-src/README.md`
  - Upstream KISS FFT README pulled via build dependencies.

---

# Compressed Context (Rich Summary)

This section is a purpose-built, high-signal context capsule intended to be pasted into future tasks.

## System Goal

- Provide a native C++ diarization pipeline equivalent to `pyannote/speaker-diarization-community-1`, expressed via GGML graphs and runnable on multiple backends (CPU/CUDA/Metal/CoreML).
- Optionally integrate Whisper transcription with streaming diarization to output speaker-labeled transcripts with word/segment timestamps.

## Core Constants / Data Contracts

- Audio: 16 kHz mono float samples.
- Segmentation:
  - Windowing: 10s chunks (`160000` samples) with 1s hop (`16000` samples).
  - Output: `589` frames per 10s chunk.
  - Powerset: 7-class log-probabilities -> deterministic mapping to 3 local speaker activity tracks.
- Embedding:
  - Features: fbank (80 bins) + ResNet34 + TSTP pooling.
  - Output: 256-d speaker embeddings (NaN used to mark invalid/silent tracks).
- Clustering:
  - Filter embeddings based on "clean" activity ratio.
  - PLDA transform (256 -> 128) + AHC init + VBx refinement.
  - Constrained assignment per chunk uses Hungarian mapping from 3 local speakers to K global clusters.

## Streaming Diarization State Invariants

- To allow periodic reclustering that can retroactively change earlier speaker labels, streaming must retain full history:
  - embeddings (growing forever)
  - binarized per-frame local activity (growing forever)
- A sliding audio buffer is sufficient for segmentation inference (only recent 10s window needed), but not for recluster.

## High-Risk / Most Important Engineering Contracts

- Layout/stride correctness is critical:
  - Post-processing (powerset decoding) expects frame-major contiguous access to 7 classes per frame.
  - Past failures were caused by layout mismatches (mistaken as "precision" issues).
- CoreML bridge robustness:
  - Segmentation bridge is stride-aware when reading `MLMultiArray` outputs (prevents silent layout bugs).
  - Embedding bridge currently assumes output contiguity more strongly (potential robustness gap / future work).
- Input layout copies are hot spots:
  - Embedding fbank input often requires a transpose copy to match GGML layout expectations.

## CUDA Reality And Optimization Narrative

- On CUDA, segmentation time is dominated by the BiLSTM recurrence custom operator.
- The repo documents a CUDA fast-path:
  - contract-based dispatch for a specific `GGML_OP_CUSTOM` signature
  - cuBLAS for input projection (`W_ih * X`)
  - cooperative-groups recurrence kernel with per-step `grid.sync()` to parallelize hidden units while preserving time-step dependencies
  - gated via `DIARIZATION_SEG_LSTM_COOP=1`
- Correctness is best validated at the logits/log-probabilities level (max/mean abs diff + argmax agreement), because downstream RTTM/DER is sensitive to thresholding.
- CUDA end-to-end correctness for embedding/segmentation can be brittle depending on weight placement/scheduler splits; `CUDA_CONTEXT.md` captures conservative guardrails and next diagnostic steps.

## Whisper + Diarization Integration (Streaming)

- Use a "filtered timeline" after a VAD-based silence filter that compresses long silences to ~2s (keeps transitions natural).
- Segment-end detection uses pyannote streaming VAD frames with accurate time mapping via cumulative frame counting (handles the 59/60-frame alternation).
- Whisper chunking rules:
  - segment-end triggers transcription only if buffered audio >= 20s
  - silence flush and finalize force transcription of whatever is buffered
- Alignment assigns speaker labels to words using WhisperX-style maximum time intersection, with a nearest-segment fallback for gaps.

## Node Bindings

- Node addon `pyannote-cpp-node` wraps the integrated pipeline:
  - shared model cache loaded once
  - offline / one-shot / incremental session APIs
  - streaming session emits cumulative `segments` events
