#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct embedding_coreml_context;

// Initialize CoreML model from .mlpackage path
struct embedding_coreml_context * embedding_coreml_init(const char * path_model);

// Free CoreML context
void embedding_coreml_free(struct embedding_coreml_context * ctx);

// Run inference: fbank features → 256-dim embedding
// fbank_data: row-major float array of shape (num_frames, 80)
// embedding_out: output buffer for 256 floats
void embedding_coreml_encode(
    const struct embedding_coreml_context * ctx,
    int64_t num_frames,
    float * fbank_data,
    float * embedding_out);

// Batch inference: process multiple fbank inputs in one call.
// Requires a batch-capable model (exported with --batch).
// All items must have the same num_frames.
// fbank_batch: [batch_size * num_frames * 80] contiguous
// embedding_batch: [batch_size * 256] contiguous
void embedding_coreml_encode_batch(
    const struct embedding_coreml_context * ctx,
    int64_t num_frames,
    int32_t batch_size,
    float * fbank_batch,
    float * embedding_batch);

#ifdef __cplusplus
}
#endif
