#include "model_cache.h"
#include "whisper.h"

#ifdef SEGMENTATION_USE_COREML
#include "segmentation_coreml_bridge.h"
#endif
#ifdef EMBEDDING_USE_COREML
#include "coreml_bridge.h"
#endif

#include <cstdio>

static void free_ggml_diarization_models(ModelCache* cache) {
    if (!cache) {
        return;
    }

    if (cache->emb_ggml_loaded) {
        embedding::state_free(cache->emb_state);
        embedding::model_free(cache->emb_model);
        cache->emb_ggml_loaded = false;
    }
    if (cache->seg_ggml_loaded) {
        segmentation::state_free(cache->seg_state);
        segmentation::model_free(cache->seg_model);
        cache->seg_ggml_loaded = false;
    }
}

ModelCache* model_cache_load(const ModelCacheConfig& config) {
    auto* cache = new ModelCache();

    // Load GGML segmentation model/state (optional fallback when CoreML is unavailable)
    if (!config.seg_model_path.empty()) {
        if (!segmentation::model_load(config.seg_model_path, cache->seg_model, false)) {
            fprintf(stderr, "model_cache_load: failed to load segmentation GGML model '%s'\n",
                    config.seg_model_path.c_str());
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
        if (!segmentation::state_init(cache->seg_state, cache->seg_model, false)) {
            fprintf(stderr, "model_cache_load: failed to init segmentation GGML state\n");
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
        cache->seg_ggml_loaded = true;
    }

    // Load GGML embedding model/state (optional fallback when CoreML is unavailable)
    if (!config.emb_model_path.empty()) {
        if (!embedding::model_load(config.emb_model_path, cache->emb_model, false)) {
            fprintf(stderr, "model_cache_load: failed to load embedding GGML model '%s'\n",
                    config.emb_model_path.c_str());
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
        if (!embedding::state_init(cache->emb_state, cache->emb_model, false)) {
            fprintf(stderr, "model_cache_load: failed to init embedding GGML state\n");
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
        cache->emb_ggml_loaded = true;
    }

    // Step 1: Load segmentation CoreML model
#ifdef SEGMENTATION_USE_COREML
    if (!config.seg_coreml_path.empty()) {
        cache->seg_coreml_ctx = segmentation_coreml_init(config.seg_coreml_path.c_str());
        if (!cache->seg_coreml_ctx) {
            fprintf(stderr, "model_cache_load: failed to load CoreML segmentation model '%s'\n",
                    config.seg_coreml_path.c_str());
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
    }
#endif

    // Step 2: Load embedding CoreML model
#ifdef EMBEDDING_USE_COREML
    if (!config.coreml_path.empty()) {
        cache->emb_coreml_ctx = embedding_coreml_init(config.coreml_path.c_str());
        if (!cache->emb_coreml_ctx) {
            fprintf(stderr, "model_cache_load: failed to load CoreML embedding model '%s'\n",
                    config.coreml_path.c_str());
#ifdef SEGMENTATION_USE_COREML
            if (cache->seg_coreml_ctx) segmentation_coreml_free(cache->seg_coreml_ctx);
#endif
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
    }
#endif

    // Step 3: Load PLDA model
    if (!config.plda_path.empty()) {
        if (!diarization::plda_load(config.plda_path, cache->plda)) {
            fprintf(stderr, "model_cache_load: failed to load PLDA model '%s'\n",
                    config.plda_path.c_str());
#ifdef EMBEDDING_USE_COREML
            if (cache->emb_coreml_ctx) embedding_coreml_free(cache->emb_coreml_ctx);
#endif
#ifdef SEGMENTATION_USE_COREML
            if (cache->seg_coreml_ctx) segmentation_coreml_free(cache->seg_coreml_ctx);
#endif
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
        cache->plda_loaded = true;
    }

    // Step 4: Load Whisper model
    if (config.transcriber.whisper_model_path) {
        auto cparams = whisper_context_default_params();
        cparams.use_gpu    = config.transcriber.use_gpu;
        cparams.flash_attn = config.transcriber.flash_attn;
        cparams.gpu_device = config.transcriber.gpu_device;
        cparams.use_coreml = config.transcriber.use_coreml;

        if (config.transcriber.no_prints) {
            whisper_log_set([](enum ggml_log_level, const char*, void*){}, nullptr);
        }

        cache->whisper_ctx = whisper_init_from_file_with_params(
            config.transcriber.whisper_model_path, cparams);
        if (!cache->whisper_ctx) {
            fprintf(stderr, "model_cache_load: failed to load Whisper model '%s'\n",
                    config.transcriber.whisper_model_path);
            if (cache->plda_loaded) cache->plda = {};
#ifdef EMBEDDING_USE_COREML
            if (cache->emb_coreml_ctx) embedding_coreml_free(cache->emb_coreml_ctx);
#endif
#ifdef SEGMENTATION_USE_COREML
            if (cache->seg_coreml_ctx) segmentation_coreml_free(cache->seg_coreml_ctx);
#endif
            free_ggml_diarization_models(cache);
            delete cache;
            return nullptr;
        }
    }

    // Step 5: Load VAD model (optional)
    if (config.vad_model_path) {
        cache->vad_ctx = whisper_vad_init_from_file_with_params(
            config.vad_model_path, whisper_vad_default_context_params());
        if (!cache->vad_ctx) {
            fprintf(stderr, "model_cache_load: failed to load VAD model '%s'\n",
                    config.vad_model_path);
            if (cache->whisper_ctx) whisper_free(cache->whisper_ctx);
            if (cache->plda_loaded) cache->plda = {};
#ifdef EMBEDDING_USE_COREML
            if (cache->emb_coreml_ctx) embedding_coreml_free(cache->emb_coreml_ctx);
#endif
#ifdef SEGMENTATION_USE_COREML
            if (cache->seg_coreml_ctx) segmentation_coreml_free(cache->seg_coreml_ctx);
#endif
            delete cache;
            return nullptr;
        }
    }

    return cache;
}

void model_cache_free(ModelCache* cache) {
    if (!cache) return;

    // Free in reverse order of loading

    if (cache->vad_ctx) {
        whisper_vad_free(cache->vad_ctx);
        cache->vad_ctx = nullptr;
    }

    if (cache->whisper_ctx) {
        whisper_free(cache->whisper_ctx);
        cache->whisper_ctx = nullptr;
    }

    cache->plda = {};
    cache->plda_loaded = false;

#ifdef EMBEDDING_USE_COREML
    if (cache->emb_coreml_ctx) {
        embedding_coreml_free(cache->emb_coreml_ctx);
        cache->emb_coreml_ctx = nullptr;
    }
#endif

#ifdef SEGMENTATION_USE_COREML
    if (cache->seg_coreml_ctx) {
        segmentation_coreml_free(cache->seg_coreml_ctx);
        cache->seg_coreml_ctx = nullptr;
    }
#endif

    free_ggml_diarization_models(cache);

    delete cache;
}

bool model_cache_reload_whisper(ModelCache* cache, const TranscriberConfig& config) {
    if (!cache) return false;

    // Free existing whisper context
    if (cache->whisper_ctx) {
        whisper_free(cache->whisper_ctx);
        cache->whisper_ctx = nullptr;
    }

    // Build new context params from config (same pattern as model_cache_load step 4)
    auto cparams = whisper_context_default_params();
    cparams.use_gpu    = config.use_gpu;
    cparams.flash_attn = config.flash_attn;
    cparams.gpu_device = config.gpu_device;
    cparams.use_coreml = config.use_coreml;

    if (config.no_prints) {
        whisper_log_set([](enum ggml_log_level, const char*, void*){}, nullptr);
    }

    cache->whisper_ctx = whisper_init_from_file_with_params(
        config.whisper_model_path, cparams);
    if (!cache->whisper_ctx) {
        fprintf(stderr, "model_cache_reload_whisper: failed to load Whisper model '%s'\n",
                config.whisper_model_path);
        return false;
    }

    return true;
}
