# Metal Support Runbook

## Build (macOS)

```bash
cmake -S diarization-ggml -B diarization-ggml/build-metal \
  -DGGML_METAL=ON \
  -DEMBEDDING_COREML=ON \
  -DSEGMENTATION_COREML=ON
cmake --build diarization-ggml/build-metal -j8
```

## Run

```bash
tools/run-diarization.sh samples/sample.wav metal /tmp/metal_out.rttm
```

## Notes

- `--backend metal` now fails fast with a clear error if binary is built without Metal.
- Wrapper script automatically uses `diarization-ggml/build-metal` for Metal backend.
- On non-macOS systems, Metal backend is rejected with an explicit message.
