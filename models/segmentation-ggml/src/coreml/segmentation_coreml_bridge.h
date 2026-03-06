#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct segmentation_coreml_context;

struct segmentation_coreml_context * segmentation_coreml_init(const char * path_model);

void segmentation_coreml_free(struct segmentation_coreml_context * ctx);

// Run inference: raw waveform → log-probabilities
// audio_data: float array of 160000 samples (10s at 16kHz mono)
// output: float array of 589 * 7 = 4123 floats (frame-major: [589][7])
void segmentation_coreml_infer(
    const struct segmentation_coreml_context * ctx,
    float * audio_data,
    int32_t n_samples,
    float * output,
    int32_t output_size);

// Batch inference: process multiple waveforms in one call.
// Requires a batch-capable model (exported with --batch).
// audio_batch: [batch_size * n_samples] contiguous
// output_batch: [batch_size * output_per_sample] contiguous
void segmentation_coreml_infer_batch(
    const struct segmentation_coreml_context * ctx,
    float * audio_batch,
    int32_t batch_size,
    int32_t n_samples,
    float * output_batch,
    int32_t output_per_sample);

#ifdef __cplusplus
}
#endif
