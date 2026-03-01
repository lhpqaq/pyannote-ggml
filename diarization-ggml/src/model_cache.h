#pragma once

#include "plda.h"         // for diarization::PLDAModel
#include "transcriber.h"  // for TranscriberConfig

#include "../../models/segmentation-ggml/src/model.h"
#include "../../models/embedding-ggml/src/model.h"

#include <string>

// Forward declarations
struct whisper_context;
struct whisper_vad_context;
struct segmentation_coreml_context;
struct embedding_coreml_context;

struct ModelCacheConfig {
    // Diarization models
    std::string seg_model_path;
    std::string emb_model_path;
    std::string plda_path;
    std::string seg_coreml_path;
    std::string coreml_path;  // embedding CoreML
    std::string ggml_backend = "auto"; // auto | cpu | metal | cuda
    int ggml_gpu_device = 0;

    // Whisper
    TranscriberConfig transcriber;  // contains whisper_model_path + context params

    // VAD (optional)
    const char* vad_model_path = nullptr;
};

struct ModelCache {
    // Diarization models (CoreML)
    segmentation_coreml_context* seg_coreml_ctx = nullptr;
    embedding_coreml_context* emb_coreml_ctx = nullptr;
    diarization::PLDAModel plda;
    bool plda_loaded = false;

    // Diarization models (GGML)
    segmentation::segmentation_model seg_model = {};
    segmentation::segmentation_state seg_state = {};
    bool seg_ggml_loaded = false;
    embedding::embedding_model emb_model = {};
    embedding::embedding_state emb_state = {};
    bool emb_ggml_loaded = false;

    // Whisper
    whisper_context* whisper_ctx = nullptr;

    // VAD (optional)
    whisper_vad_context* vad_ctx = nullptr;
};

// Load all models. Returns nullptr on failure.
ModelCache* model_cache_load(const ModelCacheConfig& config);

// Free all models.
void model_cache_free(ModelCache* cache);

// Reload whisper context with updated config (e.g., different use_coreml setting).
// Frees existing whisper_ctx and loads a new one. Returns true on success.
// On failure, whisper_ctx is set to nullptr.
bool model_cache_reload_whisper(ModelCache* cache, const TranscriberConfig& config);
