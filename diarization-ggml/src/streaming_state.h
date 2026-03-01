#pragma once
#include <string>
#include <vector>
#include "plda.h"  // for PLDAModel

#include "../../models/segmentation-ggml/src/model.h"
#include "../../models/embedding-ggml/src/model.h"

// Forward declarations for CoreML contexts
struct embedding_coreml_context;
struct segmentation_coreml_context;

struct StreamingConfig {
    std::string seg_model_path;
    std::string emb_model_path;
    std::string plda_path;
    std::string coreml_path;
    std::string seg_coreml_path;
    std::string ggml_backend = "auto"; // auto | cpu | metal | cuda
    int ggml_gpu_device = 0;
    bool zero_latency = false;
};

struct StreamingState {
    // Configuration
    StreamingConfig config;
    
    // Accumulated audio buffer (grows over time)
    std::vector<float> audio_buffer;
    
    // Accumulated embeddings [N × 256]
    std::vector<float> embeddings;
    
    // Tracking which chunk/speaker each embedding came from
    std::vector<int> chunk_idx;
    std::vector<int> local_speaker_idx;
    
    // Binarized segmentation [num_chunks × 589 × 3]
    std::vector<float> binarized;
    
    // Provisional clustering state
    std::vector<float> centroids;      // [K × 256] provisional centroids
    std::vector<int> centroid_counts;  // How many embeddings contributed to each centroid
    int num_provisional_speakers = 0;
    
    int num_speakers = 0;
    
    // Bookkeeping
    int chunks_processed = 0;
    int last_recluster_chunk = 0;
    double audio_time_processed = 0.0;
    bool finalized = false;
    int samples_trimmed = 0;
    int silence_frames_offset = 0;
    
    // Model contexts (owned unless borrowed via streaming_init_with_models)
    bool owns_models = true;
    struct segmentation_coreml_context* seg_coreml_ctx = nullptr;
    struct embedding_coreml_context* emb_coreml_ctx = nullptr;

    // GGML models/states (used when CoreML is unavailable or not requested)
    bool use_seg_ggml = false;
    bool use_emb_ggml = false;
    segmentation::segmentation_model seg_model = {};
    segmentation::segmentation_state seg_state = {};
    embedding::embedding_model emb_model = {};
    embedding::embedding_state emb_state = {};
    
    // PLDA model
    diarization::PLDAModel plda;
};
