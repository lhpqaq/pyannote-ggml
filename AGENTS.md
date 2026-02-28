# AGENTS.md

Guidance for coding agents working in `pyannote-ggml`.

## Scope And Priority

- This file applies to the whole repository.
- Follow this file first, then nearby module conventions.
- Preserve existing architecture and behavior unless the task requests change.

## Repository Map (High-Level)

- `diarization-ggml/`: main C++ diarization + transcription pipeline.
- `models/segmentation-ggml/`: segmentation model code + conversion/tests.
- `models/embedding-ggml/`: embedding model code + CoreML bridge.
- `ggml/`: upstream tensor library (treat as vendored unless asked).
- `whisper.cpp/`: local copy of whisper.cpp with project-specific integration.

## Cursor / Copilot Rules

- `.cursorrules`: not found.
- `.cursor/rules/`: not found.
- `.github/copilot-instructions.md`: not found.
- Therefore, no additional Cursor/Copilot policy files are currently enforced.

## Build Commands

### Full Pipeline (most common)

```bash
cmake -S diarization-ggml -B diarization-ggml/build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
cmake --build diarization-ggml/build -j
```

### With Whisper CoreML

```bash
cmake -S diarization-ggml -B diarization-ggml/build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON
cmake --build diarization-ggml/build -j
```

### Build Individual Model Projects

```bash
cmake -S models/segmentation-ggml -B models/segmentation-ggml/build
cmake --build models/segmentation-ggml/build -j

cmake -S models/embedding-ggml -B models/embedding-ggml/build
cmake --build models/embedding-ggml/build -j
```

### Build A Single C++ Test Target

```bash
cmake --build diarization-ggml/build --target test_aligner -j
```

## Test Commands

### Run All Important Local Checks (manual sequence)

```bash
# C++ unit/integration-style binaries
./diarization-ggml/build/bin/test_aligner
./diarization-ggml/build/bin/test_segment_detector
./diarization-ggml/build/bin/test_audio_buffer
./diarization-ggml/build/bin/test_silence_filter
./diarization-ggml/build/bin/test_transcriber
./diarization-ggml/build/bin/test_pipeline
./diarization-ggml/build/bin/test_offline_progress

# Streaming parity script
python3 diarization-ggml/tests/test_streaming.py

# Segmentation numerical parity
python3 models/segmentation-ggml/tests/test_accuracy.py
```

### Run A Single Test (explicit patterns)

```bash
# Single C++ test binary
./diarization-ggml/build/bin/test_aligner

# Single Python test script
python3 diarization-ggml/tests/test_integration.py --help

# Build + run one C++ target
cmake --build diarization-ggml/build --target test_audio_buffer -j && ./diarization-ggml/build/bin/test_audio_buffer
```

### DER Regression Check

```bash
./diarization-ggml/build/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --plda diarization-ggml/plda.gguf \
  --coreml models/embedding-ggml/embedding.mlpackage \
  --seg-coreml models/segmentation-ggml/segmentation.mlpackage \
  -o /tmp/test.rttm

python3 diarization-ggml/tests/compare_rttm.py /tmp/test.rttm /tmp/py_reference.rttm --threshold 1.0
```

## Lint / Formatting

- No top-level mandatory lint command is configured in this repo.
- No repo-wide `.clang-format` exists at root (one exists only in nested third-party code).
- For C++ edits, match surrounding style exactly; do not mass-reformat unrelated lines.
- If you must format touched C++ files, use `clang-format` conservatively and only on changed regions.
- For Python scripts, follow PEP 8-like style already used in tests/tools.

## Code Style Conventions

### Language Standards

- `diarization-ggml` and `embedding-ggml` use C++17.
- `segmentation-ggml` currently declares C++11; keep compatibility there unless requested.

### Includes / Imports

- Keep project headers first, then standard library headers.
- Maintain existing local include patterns (some files use explicit relative includes intentionally).
- In Python, keep stdlib imports first, third-party next, local imports last.

### Naming

- Types/structs/classes: `PascalCase` (`DiarizationConfig`, `PipelineState`).
- Functions: `snake_case` (`streaming_init`, `align_segments`).
- Constants: `UPPER_SNAKE_CASE` (`CHUNK_SAMPLES`, `FRAME_STEP_SEC`).
- Member/variable names: `snake_case`.

### Formatting And Structure

- Prefer small, single-purpose helper functions for pipeline stages.
- Use `static constexpr` for hard-coded model/pipeline constants.
- Keep hot-path code straightforward and allocation-aware.
- Avoid introducing new dependencies when existing modules already provide utilities.

### Types And Numeric Safety

- Preserve float/double intent: many clustering/scoring paths depend on precision.
- Use explicit casts where type conversion is non-trivial or cross-width.
- Keep tensor/layout assumptions explicit (shape/order comments where needed).
- Be careful with index math between sample, frame, and chunk coordinate spaces.

### Error Handling

- C++ APIs primarily signal failure via `bool` / `nullptr`; follow existing patterns.
- Emit clear `fprintf(stderr, ...)` diagnostics for user-visible failures.
- On initialization failure, free already-acquired resources before returning.
- Favor early returns for invalid arguments and missing resources.

### Memory / Resource Management

- Respect ownership contracts (borrowed vs owned contexts in pipeline/cache code).
- When adding cleanup logic, mirror existing init-order teardown behavior.
- Avoid leaks in mixed C/C++ boundaries (CoreML, whisper, GGML contexts).

### Testing Expectations For Changes

- If touching diarization, clustering, streaming, or alignment logic: run relevant C++ tests plus `diarization-ggml/tests/test_streaming.py`.
- If touching segmentation model math or conversion: run `models/segmentation-ggml/tests/test_accuracy.py`.
- For CLI/output behavior changes, verify at least one real command from `README.md` end-to-end.

## Agent Workflow Tips

- Before editing, inspect neighboring files for local conventions.
- Keep diffs tight: no drive-by refactors in unrelated modules.
- Do not modify vendored `ggml/` or `whisper.cpp/` unless the task explicitly targets them.
- Prefer incremental, verifiable changes; include exact commands used for validation in your report.
