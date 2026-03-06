#include "diarization.h"
#include "aggregation.h"
#include "clustering.h"
#include "plda.h"
#include "powerset.h"
#include "rttm.h"
#include "vbx.h"

// Both model.h files have unique include guards (SEGMENTATION_GGML_MODEL_H, EMBEDDING_GGML_MODEL_H).
// Use explicit relative paths since both libraries export "model.h" in their PUBLIC includes.
#include "../../models/segmentation-ggml/src/model.h"
#include "../../models/embedding-ggml/src/model.h"
#include "fbank.h"

#ifdef EMBEDDING_USE_COREML
#include "coreml_bridge.h"
#endif

#ifdef SEGMENTATION_USE_COREML
#include "segmentation_coreml_bridge.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Constants — hardcoded pipeline parameters for community-1
// ============================================================================

static constexpr int SAMPLE_RATE             = 16000;
static constexpr int CHUNK_SAMPLES           = 160000;   // 10s at 16kHz
static constexpr int STEP_SAMPLES            = 16000;    // 1s step at 16kHz
static constexpr int FRAMES_PER_CHUNK        = 589;      // segmentation frames per chunk
static constexpr int NUM_POWERSET_CLASSES    = 7;        // powerset output classes
static constexpr int NUM_LOCAL_SPEAKERS      = 3;        // speakers after powerset->multilabel
static constexpr int EMBEDDING_DIM           = 256;
static constexpr int PLDA_DIM                = 128;
static constexpr int FBANK_NUM_BINS          = 80;

static constexpr double AHC_THRESHOLD        = 0.6;
static constexpr double VBX_FA               = 0.07;
static constexpr double VBX_FB               = 0.8;
static constexpr int    VBX_MAX_ITERS        = 20;

static constexpr double FRAME_DURATION       = 0.0619375; // model receptive field duration (NOT the step)
static constexpr double FRAME_STEP           = 0.016875;  // seconds
static constexpr double CHUNK_DURATION       = 10.0;      // seconds
static constexpr double CHUNK_STEP           = 1.0;       // seconds

// ============================================================================
// WAV loading — adapted from embedding-ggml/src/main.cpp
// ============================================================================

struct wav_header {
    char     riff[4];
    uint32_t file_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

struct wav_data_chunk {
    char     id[4];
    uint32_t size;
};

static bool load_wav_file(const std::string& path, std::vector<float>& samples,
                          uint32_t& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "ERROR: Failed to open WAV file: %s\n", path.c_str());
        return false;
    }

    wav_header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(wav_header));

    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Invalid WAV file format\n");
        return false;
    }

    if (header.audio_format != 1) {
        fprintf(stderr, "ERROR: Only PCM format supported (got format %d)\n",
                header.audio_format);
        return false;
    }

    if (header.num_channels != 1) {
        fprintf(stderr, "ERROR: Only mono audio supported (got %d channels)\n",
                header.num_channels);
        return false;
    }

    if (header.bits_per_sample != 16) {
        fprintf(stderr, "ERROR: Only 16-bit audio supported (got %d bits)\n",
                header.bits_per_sample);
        return false;
    }

    wav_data_chunk data_chunk;
    file.read(reinterpret_cast<char*>(&data_chunk), sizeof(wav_data_chunk));

    while (std::strncmp(data_chunk.id, "data", 4) != 0) {
        file.seekg(data_chunk.size, std::ios::cur);
        if (!file.read(reinterpret_cast<char*>(&data_chunk), sizeof(wav_data_chunk))) {
            fprintf(stderr, "ERROR: Data chunk not found\n");
            return false;
        }
    }

    uint32_t num_samples = data_chunk.size / (header.bits_per_sample / 8);
    samples.resize(num_samples);

    std::vector<int16_t> pcm_data(num_samples);
    file.read(reinterpret_cast<char*>(pcm_data.data()), data_chunk.size);

    for (size_t i = 0; i < num_samples; i++) {
        samples[i] = static_cast<float>(pcm_data[i]) / 32768.0f;
    }

    sample_rate = header.sample_rate;
    file.close();
    return true;
}

// ============================================================================
// Embedding extraction (uses either CoreML or GGML backend)
// ============================================================================

bool extract_embeddings(
    const float* audio,
    int          num_samples,
    const float* binarized_segmentations,
    int          num_chunks,
    int          num_frames_per_chunk,
    int          num_speakers,
    struct embedding_coreml_context* coreml_ctx,
    embedding::embedding_model* emb_model,
    embedding::embedding_state* emb_state,
    float*       embeddings_out)
{
    std::vector<float> cropped(CHUNK_SAMPLES, 0.0f);

    int backend_nodes_total = 0;
    int backend_nodes_gpu = 0;
    int backend_nodes_cpu = 0;

    for (int c = 0; c < num_chunks; c++) {
        const int chunk_start = c * STEP_SAMPLES;
        int copy_len = num_samples - chunk_start;
        if (copy_len > CHUNK_SAMPLES) copy_len = CHUNK_SAMPLES;
        if (copy_len < 0) copy_len = 0;

        std::fill(cropped.begin(), cropped.end(), 0.0f);
        if (copy_len > 0) {
            std::memcpy(cropped.data(), audio + chunk_start,
                        static_cast<size_t>(copy_len) * sizeof(float));
        }

        embedding::fbank_result fbank =
            embedding::compute_fbank(cropped.data(), CHUNK_SAMPLES, SAMPLE_RATE);
        const int num_fbank_frames = fbank.num_frames;

        // Optional scratch for CoreML path (row-major masked fbank).
        std::vector<float> masked_fbank_rowmajor;
        if (coreml_ctx) {
            masked_fbank_rowmajor.resize(fbank.data.size());
        }

        // seg layout: [num_chunks, num_frames_per_chunk, num_speakers] row-major
        const float* seg_chunk =
            binarized_segmentations + c * num_frames_per_chunk * num_speakers;

        // Preserve original behavior: if speaker s is all-zero in this chunk, write NaNs and skip.
        bool speaker_all_zero[16] = {};
        for (int s = 0; s < num_speakers; s++) {
            speaker_all_zero[s] = true;
        }
        for (int f = 0; f < num_frames_per_chunk; f++) {
            const float * frame = seg_chunk + f * num_speakers;
            for (int s = 0; s < num_speakers; s++) {
                if (frame[s] != 0.0f) {
                    speaker_all_zero[s] = false;
                }
            }
        }



        for (int s = 0; s < num_speakers; s++) {
            float* emb_out =
                embeddings_out + (c * num_speakers + s) * EMBEDDING_DIM;

            if (speaker_all_zero[s]) {
                const float nan_val = std::nanf("");
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    emb_out[d] = nan_val;
                }
                continue;
            }

            if (coreml_ctx) {
                // CoreML expects row-major [T][80].
                for (int ft = 0; ft < num_fbank_frames; ft++) {
                    int seg_frame = (int) ((long long) ft * (long long) num_frames_per_chunk / (long long) num_fbank_frames);
                    if (seg_frame >= num_frames_per_chunk) {
                        seg_frame = num_frames_per_chunk - 1;
                    }

                    const float mask_val = seg_chunk[seg_frame * num_speakers + s];
                    const float * src_row = fbank.data.data() + (size_t) ft * FBANK_NUM_BINS;
                    float * dst_row = masked_fbank_rowmajor.data() + (size_t) ft * FBANK_NUM_BINS;

                    if (mask_val == 0.0f) {
                        std::memset(dst_row, 0, FBANK_NUM_BINS * sizeof(float));
                    } else {
                        std::memcpy(dst_row, src_row, FBANK_NUM_BINS * sizeof(float));
                    }
                }
            } else if (emb_state) {
                // GGML path expects column-major [80][T] (ne0=T contiguous), so build masked+transposed directly.
                emb_state->fbank_transposed.resize((size_t) num_fbank_frames * FBANK_NUM_BINS);
                for (int ft = 0; ft < num_fbank_frames; ft++) {
                    int seg_frame = (int) ((long long) ft * (long long) num_frames_per_chunk / (long long) num_fbank_frames);
                    if (seg_frame >= num_frames_per_chunk) {
                        seg_frame = num_frames_per_chunk - 1;
                    }

                    const float mask_val = seg_chunk[seg_frame * num_speakers + s];
                    const float * src_row = fbank.data.data() + (size_t) ft * FBANK_NUM_BINS;
                    if (mask_val == 0.0f) {
                        for (int b = 0; b < FBANK_NUM_BINS; b++) {
                            emb_state->fbank_transposed[(size_t) b * (size_t) num_fbank_frames + (size_t) ft] = 0.0f;
                        }
                    } else {
                        for (int b = 0; b < FBANK_NUM_BINS; b++) {
                            emb_state->fbank_transposed[(size_t) b * (size_t) num_fbank_frames + (size_t) ft] = src_row[b];
                        }
                    }
                }
            }

            if (coreml_ctx) {
#ifdef EMBEDDING_USE_COREML
                embedding_coreml_encode(coreml_ctx,
                                         static_cast<int64_t>(num_fbank_frames),
                                         masked_fbank_rowmajor.data(),
                                         emb_out);
#else
                fprintf(stderr, "Error: CoreML embedding requested but EMBEDDING_USE_COREML is disabled\n");
                return false;
#endif
            } else if (emb_model && emb_state) {
                // Use the transposed buffer directly (avoid per-call transpose).
                if (!embedding::model_infer_transposed(*emb_model, *emb_state,
                                                       emb_state->fbank_transposed.data(), num_fbank_frames,
                                                       emb_out, EMBEDDING_DIM)) {
                    fprintf(stderr, "Error: GGML embedding failed at chunk %d speaker %d\n", c, s);
                    const float nan_val = std::nanf("");
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        emb_out[d] = nan_val;
                    }
                }

                if (std::getenv("DIARIZATION_EMB_DEBUG_NAN") != nullptr) {
                    int nan_count = 0;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        if (std::isnan(emb_out[d])) {
                            nan_count++;
                        }
                    }
                    if (nan_count > 0) {
                        fprintf(stderr, "[emb] NaNs in embedding: chunk=%d speaker=%d nan=%d/%d\n", c, s, nan_count, EMBEDDING_DIM);
                    }
                }
                backend_nodes_total += emb_state->last_nodes_total;
                backend_nodes_gpu += emb_state->last_nodes_gpu;
                backend_nodes_cpu += emb_state->last_nodes_cpu;
            } else {
                fprintf(stderr, "Error: no embedding backend available\n");
                return false;
            }
        }
    }

    if (emb_model && emb_state && emb_state->backend_stats && backend_nodes_total > 0) {
        fprintf(stderr,
                "[backend] embedding nodes total=%d gpu=%d cpu=%d gpu_ratio=%.1f%%\n",
                backend_nodes_total,
                backend_nodes_gpu,
                backend_nodes_cpu,
                100.0 * (double) backend_nodes_gpu / (double) backend_nodes_total);
    }

    return true;
}

// ============================================================================
// Parallel seg+emb pipeline (CoreML only)
// ============================================================================

#if defined(SEGMENTATION_USE_COREML) && defined(EMBEDDING_USE_COREML)
static bool pipeline_parallel_seg_emb(
    struct segmentation_coreml_context* seg_ctx,
    struct embedding_coreml_context* emb_ctx,
    const float* audio, int n_samples, int num_chunks,
    float* binarized_out,
    float* embeddings_out,
    double& t_seg_ms, double& t_emb_ms)
{
    using Clock = std::chrono::high_resolution_clock;

    std::atomic<int> chunks_segmented{0};
    std::mutex mtx;
    std::condition_variable cv;
    double seg_time = 0.0;

    std::vector<float> seg_logits(
        static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);

    fprintf(stderr, "Segmentation+Embedding (parallel): %d chunks... ", num_chunks);
    fflush(stderr);

    // Producer: segmentation + per-chunk powerset
    std::thread seg_thread([&] {
        auto t0 = Clock::now();
        std::vector<float> cropped(CHUNK_SAMPLES, 0.0f);

        for (int c = 0; c < num_chunks; c++) {
            const int chunk_start = c * STEP_SAMPLES;
            int copy_len = n_samples - chunk_start;
            if (copy_len > CHUNK_SAMPLES) copy_len = CHUNK_SAMPLES;
            if (copy_len < 0) copy_len = 0;

            std::fill(cropped.begin(), cropped.end(), 0.0f);
            if (copy_len > 0) {
                std::memcpy(cropped.data(), audio + chunk_start,
                            static_cast<size_t>(copy_len) * sizeof(float));
            }

            float* logits_out = seg_logits.data() +
                static_cast<size_t>(c) * FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES;

            segmentation_coreml_infer(seg_ctx,
                                      cropped.data(), CHUNK_SAMPLES,
                                      logits_out,
                                      FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);

            float* bin_out = binarized_out +
                static_cast<size_t>(c) * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;
            diarization::powerset_to_multilabel(logits_out, 1, FRAMES_PER_CHUNK, bin_out);

            chunks_segmented.store(c + 1, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lk(mtx);
            }
            cv.notify_one();

            if ((c + 1) % 50 == 0 || c + 1 == num_chunks) {
                fprintf(stderr, "%d/%d chunks", c + 1, num_chunks);
                fflush(stderr);
            }
        }

        seg_time = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    });

    // Consumer: precompute global fbank, then masked embedding extraction
    auto t_emb_start = Clock::now();
    {
        // Precompute global fbank (without CMN) on zero-padded audio.
        // Chunks overlap 90% (10s windows, 1s step), so computing once
        // avoids ~10x redundant FFT/mel work.
        const int padded_len = (num_chunks - 1) * STEP_SAMPLES + CHUNK_SAMPLES;
        std::vector<float> padded_audio(padded_len, 0.0f);
        const int copy_len = std::min(padded_len, n_samples);
        if (copy_len > 0) {
            std::memcpy(padded_audio.data(), audio, static_cast<size_t>(copy_len) * sizeof(float));
        }

        embedding::fbank_result global_fbank =
            embedding::compute_fbank(padded_audio.data(), padded_len, SAMPLE_RATE, false);

        // Each chunk c maps to global fbank frames [c*FBANK_STEP .. c*FBANK_STEP + nf_per_chunk - 1]
        // where FBANK_STEP = STEP_SAMPLES / frame_shift_samples = 16000 / 160 = 100
        static constexpr int FBANK_STEP = STEP_SAMPLES / 160;  // 100 frames per chunk step

        // Compute nf_per_chunk from the first chunk's frame count
        const int nf_per_chunk = std::min(
            static_cast<int>((CHUNK_SAMPLES - 400) / 160) + 1,  // 998 for 160000 samples
            global_fbank.num_frames);

        std::vector<float> chunk_fbank(static_cast<size_t>(nf_per_chunk) * FBANK_NUM_BINS);
        std::vector<float> masked_fbank(static_cast<size_t>(nf_per_chunk) * FBANK_NUM_BINS);

        for (int c = 0; c < num_chunks; c++) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] {
                    return chunks_segmented.load(std::memory_order_acquire) > c;
                });
            }

            // Extract this chunk's fbank frames from the global fbank and apply per-chunk CMN
            const int frame_offset = c * FBANK_STEP;
            const int nf = std::min(nf_per_chunk, global_fbank.num_frames - frame_offset);
            std::memcpy(chunk_fbank.data(),
                        global_fbank.data.data() + static_cast<size_t>(frame_offset) * FBANK_NUM_BINS,
                        static_cast<size_t>(nf) * FBANK_NUM_BINS * sizeof(float));
            embedding::apply_cmn(chunk_fbank.data(), nf, FBANK_NUM_BINS);

            const float* seg_chunk = binarized_out +
                static_cast<size_t>(c) * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;

            bool speaker_zero[NUM_LOCAL_SPEAKERS];
            for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) speaker_zero[s] = true;
            for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                const float* frame = seg_chunk + f * NUM_LOCAL_SPEAKERS;
                for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                    if (frame[s] != 0.0f) speaker_zero[s] = false;
                }
            }

            for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                float* emb_out = embeddings_out +
                    (c * NUM_LOCAL_SPEAKERS + s) * EMBEDDING_DIM;

                if (speaker_zero[s]) {
                    const float nan_val = std::nanf("");
                    for (int d = 0; d < EMBEDDING_DIM; d++) emb_out[d] = nan_val;
                    continue;
                }

                for (int ft = 0; ft < nf; ft++) {
                    int seg_frame = static_cast<int>(
                        static_cast<long long>(ft) * FRAMES_PER_CHUNK / nf);
                    if (seg_frame >= FRAMES_PER_CHUNK) seg_frame = FRAMES_PER_CHUNK - 1;

                    const float mask = seg_chunk[seg_frame * NUM_LOCAL_SPEAKERS + s];
                    const float* src = chunk_fbank.data() + static_cast<size_t>(ft) * FBANK_NUM_BINS;
                    float* dst = masked_fbank.data() + static_cast<size_t>(ft) * FBANK_NUM_BINS;

                    if (mask == 0.0f) {
                        std::memset(dst, 0, FBANK_NUM_BINS * sizeof(float));
                    } else {
                        std::memcpy(dst, src, FBANK_NUM_BINS * sizeof(float));
                    }
                }

                embedding_coreml_encode(emb_ctx,
                                         static_cast<int64_t>(nf),
                                         masked_fbank.data(),
                                         emb_out);
            }
        }
    }
    auto t_emb_end = Clock::now();

    seg_thread.join();

    t_seg_ms = seg_time;
    t_emb_ms = std::chrono::duration<double, std::milli>(t_emb_end - t_emb_start).count();

    fprintf(stderr, "done [seg %.0f ms | emb %.0f ms | parallel]\n", t_seg_ms, t_emb_ms);
    return true;
}
#endif

// ============================================================================
// URI extraction — filename without extension
// ============================================================================

static std::string extract_uri(const std::string& audio_path) {
    std::string uri = audio_path;
    size_t last_slash = uri.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        uri = uri.substr(last_slash + 1);
    }
    size_t last_dot = uri.find_last_of('.');
    if (last_dot != std::string::npos) {
        uri = uri.substr(0, last_dot);
    }
    return uri;
}

// ============================================================================
// Buffer-based pipeline — core implementation used by both file and buffer APIs
// ============================================================================

bool diarize_from_samples(const DiarizationConfig& config, const float* audio, int n_samples, DiarizationResult& result) {
    using Clock = std::chrono::high_resolution_clock;
    
    // Timing variables for each stage
    double t_load_audio_ms = 0.0;
    double t_load_seg_model_ms = 0.0;
    double t_load_emb_model_ms = 0.0;
    double t_load_plda_ms = 0.0;
    double t_segmentation_ms = 0.0;
    double t_powerset_ms = 0.0;
    double t_speaker_count_ms = 0.0;
    double t_embeddings_ms = 0.0;
    double t_filter_plda_ms = 0.0;
    double t_clustering_ms = 0.0;
    double t_postprocess_ms = 0.0;
    
    auto t_total_start = Clock::now();
    auto t_stage_start = Clock::now();
    auto t_stage_end   = Clock::now();

    // Audio parameters derived from buffer
    const double audio_duration = static_cast<double>(n_samples) / SAMPLE_RATE;
    
    int audio_mins = static_cast<int>(audio_duration) / 60;
    int audio_secs = static_cast<int>(audio_duration) % 60;
    fprintf(stderr, "Audio: %.2fs (%d:%02d), %d samples\n", 
            audio_duration, audio_mins, audio_secs, n_samples);

    // ====================================================================
    // Step 2: Load segmentation model (CoreML or GGML)
    // ====================================================================

    t_stage_start = Clock::now();
#ifdef SEGMENTATION_USE_COREML
    struct segmentation_coreml_context* seg_coreml_ctx = nullptr;
    bool use_seg_coreml = !config.seg_coreml_path.empty();
#else
    bool use_seg_coreml = false;
#endif

    segmentation::segmentation_model seg_model = {};
    segmentation::segmentation_state seg_state = {};

#ifdef SEGMENTATION_USE_COREML
    if (use_seg_coreml) {
        seg_coreml_ctx = segmentation_coreml_init(config.seg_coreml_path.c_str());
        if (!seg_coreml_ctx) {
            fprintf(stderr, "Error: failed to load CoreML segmentation model '%s'\n",
                    config.seg_coreml_path.c_str());
            return false;
        }
    } else
#endif
    {
        const std::string seg_weight_backend = config.ggml_backend;

        if (!segmentation::model_load(config.seg_model_path,
                                      seg_model,
                                      seg_weight_backend,
                                      config.ggml_gpu_device,
                                      false)) {
            fprintf(stderr, "Error: failed to load segmentation model '%s'\n",
                    config.seg_model_path.c_str());
            return false;
        }
        if (!segmentation::state_init(seg_state, seg_model,
                                      config.ggml_backend,
                                      config.ggml_gpu_device,
                                      false)) {
            fprintf(stderr, "Error: failed to initialize segmentation state\n");
            segmentation::model_free(seg_model);
            return false;
        }
        segmentation::state_set_backend_stats(seg_state, (config.ggml_backend != "cuda"));
    }
    
    t_stage_end = Clock::now();
    t_load_seg_model_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();

    // ====================================================================
    // Step 3: Load embedding model (CoreML or GGML)
    // ====================================================================

    t_stage_start = Clock::now();
    struct embedding_coreml_context* coreml_ctx = nullptr;
    bool use_emb_coreml = false;
    embedding::embedding_model emb_model = {};
    embedding::embedding_state emb_state = {};

#ifdef EMBEDDING_USE_COREML
    use_emb_coreml = !config.coreml_path.empty();
    if (use_emb_coreml) {
        coreml_ctx = embedding_coreml_init(config.coreml_path.c_str());
        if (!coreml_ctx) {
            fprintf(stderr, "Error: failed to load CoreML embedding model '%s'\n",
                    config.coreml_path.c_str());
            if (seg_state.sched) segmentation::state_free(seg_state);
            if (seg_model.ctx) segmentation::model_free(seg_model);
            return false;
        }
    }
#endif

    if (!use_emb_coreml) {
        const std::string emb_weight_backend = config.ggml_backend;

        if (!embedding::model_load(config.emb_model_path,
                                   emb_model,
                                   emb_weight_backend,
                                   config.ggml_gpu_device,
                                   false)) {
            fprintf(stderr, "Error: failed to load embedding model '%s'\n",
                    config.emb_model_path.c_str());
            if (seg_state.sched) segmentation::state_free(seg_state);
            if (seg_model.ctx) segmentation::model_free(seg_model);
            return false;
        }
        if (!embedding::state_init(emb_state, emb_model,
                                   config.ggml_backend,
                                   config.ggml_gpu_device,
                                   false)) {
            fprintf(stderr, "Error: failed to initialize embedding state\n");
            embedding::model_free(emb_model);
            if (seg_state.sched) segmentation::state_free(seg_state);
            if (seg_model.ctx) segmentation::model_free(seg_model);
            return false;
        }
        const bool enable_emb_backend_stats = (config.ggml_backend != "cuda");
        embedding::state_set_backend_stats(emb_state, enable_emb_backend_stats);
    }
    
    t_stage_end = Clock::now();
    t_load_emb_model_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();

    // ====================================================================
    // Step 4: Load PLDA model
    // ====================================================================

    t_stage_start = Clock::now();
    diarization::PLDAModel plda;
    if (config.plda_path.empty()) {
        fprintf(stderr, "Error: --plda path is required for VBx clustering\n");
#ifdef EMBEDDING_USE_COREML
        embedding_coreml_free(coreml_ctx);
#else
        embedding::state_free(emb_state);
        embedding::model_free(emb_model);
#endif
        if (seg_state.sched) segmentation::state_free(seg_state);
        if (seg_model.ctx) segmentation::model_free(seg_model);
        return false;
    }
    if (!diarization::plda_load(config.plda_path, plda)) {
        fprintf(stderr, "Error: failed to load PLDA model '%s'\n",
                config.plda_path.c_str());
#ifdef EMBEDDING_USE_COREML
        embedding_coreml_free(coreml_ctx);
#else
        embedding::state_free(emb_state);
        embedding::model_free(emb_model);
#endif
        if (seg_state.sched) segmentation::state_free(seg_state);
        if (seg_model.ctx) segmentation::model_free(seg_model);
        return false;
    }
    
    t_stage_end = Clock::now();
    t_load_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();

    // ====================================================================
    // Step 5: Sliding window segmentation + embedding
    // ====================================================================

    int num_chunks = std::max(1,
        1 + static_cast<int>(std::ceil((audio_duration - CHUNK_DURATION) / CHUNK_STEP)));

    // Outer-scope buffers shared by parallel and sequential paths
    std::vector<float> seg_logits;
    std::vector<float> binarized(
        static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS);
    std::vector<float> embeddings;
    bool parallel_done = false;

#if defined(SEGMENTATION_USE_COREML) && defined(EMBEDDING_USE_COREML)
    if (use_seg_coreml && use_emb_coreml && !config.bypass_embeddings) {
        t_stage_start = Clock::now();
        embeddings.resize(
            static_cast<size_t>(num_chunks) * NUM_LOCAL_SPEAKERS * EMBEDDING_DIM);

        if (!pipeline_parallel_seg_emb(seg_coreml_ctx, coreml_ctx,
                                       audio, n_samples, num_chunks,
                                       binarized.data(), embeddings.data(),
                                       t_segmentation_ms, t_embeddings_ms)) {
            goto cleanup;
        }
        t_powerset_ms = 0.0;

        segmentation_coreml_free(seg_coreml_ctx);
        seg_coreml_ctx = nullptr;
        embedding_coreml_free(coreml_ctx);
        coreml_ctx = nullptr;

        parallel_done = true;
    }
#endif

    if (!parallel_done) {
    // --- Sequential segmentation ---
    t_stage_start = Clock::now();
    fprintf(stderr, "Segmentation: %d chunks... ", num_chunks);
    fflush(stderr);

    seg_logits.resize(
        static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);

    {
        std::vector<float> cropped(CHUNK_SAMPLES, 0.0f);
        for (int c = 0; c < num_chunks; c++) {
            const int chunk_start = c * STEP_SAMPLES;
            int copy_len = n_samples - chunk_start;
            if (copy_len > CHUNK_SAMPLES) copy_len = CHUNK_SAMPLES;
            if (copy_len < 0) copy_len = 0;

            std::fill(cropped.begin(), cropped.end(), 0.0f);
            if (copy_len > 0) {
                std::memcpy(cropped.data(), audio + chunk_start,
                            static_cast<size_t>(copy_len) * sizeof(float));
            }

            float* output = seg_logits.data() +
                static_cast<size_t>(c) * FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES;

#ifdef SEGMENTATION_USE_COREML
            if (use_seg_coreml) {
                segmentation_coreml_infer(seg_coreml_ctx,
                                          cropped.data(), CHUNK_SAMPLES,
                                          output,
                                          FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);
            } else
#endif
            {
                if (!segmentation::model_infer(seg_model, seg_state,
                                               cropped.data(), CHUNK_SAMPLES,
                                               output,
                                               FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES)) {
                    fprintf(stderr, "Error: segmentation failed at chunk %d/%d\n",
                            c + 1, num_chunks);
                    goto cleanup;
                }

                // Transpose GGML output from [class, frame] to [frame, class] layout
                {
                    std::vector<float> tmp(FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);
                    std::memcpy(tmp.data(), output, tmp.size() * sizeof(float));
                    for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                        for (int k = 0; k < NUM_POWERSET_CLASSES; k++) {
                            output[f * NUM_POWERSET_CLASSES + k] = tmp[k * FRAMES_PER_CHUNK + f];
                        }
                    }
                }
            }

            if ((c + 1) % 50 == 0 || c + 1 == num_chunks) {
                fprintf(stderr, "%d/%d chunks\r", c + 1, num_chunks);
                fflush(stderr);
            }
        }
    }

#ifdef SEGMENTATION_USE_COREML
    if (use_seg_coreml) {
        segmentation_coreml_free(seg_coreml_ctx);
        seg_coreml_ctx = nullptr;
    } else
#endif
    {
        segmentation::state_free(seg_state);
        segmentation::model_free(seg_model);
        seg_state.sched = nullptr;
        seg_model.ctx = nullptr;
    }
    
    t_stage_end = Clock::now();
    t_segmentation_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
    fprintf(stderr, "done [%.0f ms]\n", t_segmentation_ms);

    // --- Sequential powerset ---
    t_stage_start = Clock::now();
    diarization::powerset_to_multilabel(
        seg_logits.data(), num_chunks, FRAMES_PER_CHUNK, binarized.data());
    seg_logits.clear();
    seg_logits.shrink_to_fit();
    t_stage_end = Clock::now();
    t_powerset_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
    fprintf(stderr, "Powerset: done [%.0f ms]\n", t_powerset_ms);

    } // end if (!parallel_done) — sequential seg + powerset

    {

        // ================================================================
        // Step 7: Compute frame-level speaker count
        // ================================================================

        t_stage_start = Clock::now();
        diarization::SlidingWindowParams chunk_window = {0.0, CHUNK_DURATION, CHUNK_STEP};
        diarization::SlidingWindowParams frame_window = {chunk_window.start, FRAME_DURATION, FRAME_STEP};

        std::vector<int> count;
        int total_frames = 0;
        diarization::compute_speaker_count(
            binarized.data(), num_chunks, FRAMES_PER_CHUNK, NUM_LOCAL_SPEAKERS,
            chunk_window, frame_window, count, total_frames);

        // Early exit if no speaker is ever active
        int max_count = 0;
        for (int i = 0; i < total_frames; i++) {
            if (count[i] > max_count) max_count = count[i];
        }
        
        t_stage_end = Clock::now();
        t_speaker_count_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
        
        if (max_count == 0) {
            fprintf(stderr, "Speaker count: no speakers detected\n");
            result.segments.clear();
            std::string uri = config.audio_path.empty() ? "audio" : extract_uri(config.audio_path);
            std::vector<diarization::RTTMSegment> empty;
            if (config.output_path.empty()) {
                diarization::write_rttm_stdout(empty, uri);
            } else {
                diarization::write_rttm(empty, uri, config.output_path);
            }
            goto cleanup_emb;
        }

        fprintf(stderr, "Speaker count: max %d speakers, %d frames [%.0f ms]\n",
                max_count, total_frames, t_speaker_count_ms);

        std::vector<int> hard_clusters(
            static_cast<size_t>(num_chunks) * NUM_LOCAL_SPEAKERS, 0);
        int num_clusters = 1;

        if (config.bypass_embeddings) {
            fprintf(stderr, "Embeddings: bypassed (segmentation-only assignment)\n");
            num_clusters = NUM_LOCAL_SPEAKERS;
            for (int c = 0; c < num_chunks; c++) {
                for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                    hard_clusters[c * NUM_LOCAL_SPEAKERS + s] = s;
                }
            }
        } else {
            // ================================================================
            // Step 8: Extract embeddings
            // (num_chunks, 3, 256)
            // ================================================================

            if (!parallel_done) {
            t_stage_start = Clock::now();
            embeddings.resize(
                static_cast<size_t>(num_chunks) * NUM_LOCAL_SPEAKERS * EMBEDDING_DIM);

            fprintf(stderr, "Embeddings: %d chunks... ", num_chunks);
            fflush(stderr);
            if (!extract_embeddings(
                    audio, n_samples,
                    binarized.data(), num_chunks, FRAMES_PER_CHUNK, NUM_LOCAL_SPEAKERS,
                    coreml_ctx,
                    use_emb_coreml ? nullptr : &emb_model,
                    use_emb_coreml ? nullptr : &emb_state,
                    embeddings.data())) {
                fprintf(stderr, "Error: embedding extraction failed\n");
                goto cleanup;
            }

            t_stage_end = Clock::now();
            t_embeddings_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            fprintf(stderr, "done [%.0f ms]\n", t_embeddings_ms);

            // Free embedding model — no longer needed
            if (coreml_ctx) {
#ifdef EMBEDDING_USE_COREML
                embedding_coreml_free(coreml_ctx);
                coreml_ctx = nullptr;
#endif
            } else {
                embedding::state_free(emb_state);
                embedding::model_free(emb_model);
                emb_state.sched = nullptr;
                emb_model.ctx = nullptr;
            }
            } // end if (!parallel_done) — sequential embeddings

            // ================================================================
            // Step 9: Filter embeddings
            // ================================================================

            t_stage_start = Clock::now();
            std::vector<float> filtered_emb;
            std::vector<int> filt_chunk_idx, filt_speaker_idx;
            diarization::filter_embeddings(
                embeddings.data(), num_chunks, NUM_LOCAL_SPEAKERS, EMBEDDING_DIM,
                binarized.data(), FRAMES_PER_CHUNK,
                filtered_emb, filt_chunk_idx, filt_speaker_idx);

            const int num_filtered = static_cast<int>(filt_chunk_idx.size());
            fprintf(stderr, "Filter: %d embeddings ", num_filtered);
            fflush(stderr);

            // ================================================================
            // Steps 10-17: Clustering + assignment
            // ================================================================

            if (num_filtered < 2) {
            t_stage_end = Clock::now();
            t_filter_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_filter_plda_ms);
            
            // Too few embeddings — assign all to single cluster
            fprintf(stderr, "Too few embeddings for clustering, using single cluster\n");

            } else {
            // Step 11: L2-normalize filtered embeddings (for AHC)
            std::vector<double> filtered_normed(
                static_cast<size_t>(num_filtered) * EMBEDDING_DIM);
            for (int i = 0; i < num_filtered; i++) {
                const float* src = filtered_emb.data() + i * EMBEDDING_DIM;
                double* dst = filtered_normed.data() + i * EMBEDDING_DIM;
                double norm = 0.0;
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    dst[d] = static_cast<double>(src[d]);
                    norm += dst[d] * dst[d];
                }
                norm = std::sqrt(norm);
                if (norm > 0.0) {
                    const double inv = 1.0 / norm;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        dst[d] *= inv;
                    }
                }
            }

            // Step 12: PLDA transform (on original filtered embeddings, NOT normalized)
            std::vector<double> filtered_double(
                static_cast<size_t>(num_filtered) * EMBEDDING_DIM);
            for (int i = 0; i < num_filtered * EMBEDDING_DIM; i++) {
                filtered_double[i] = static_cast<double>(filtered_emb[i]);
            }

            std::vector<double> plda_features(
                static_cast<size_t>(num_filtered) * PLDA_DIM);
            diarization::plda_transform(
                plda, filtered_double.data(), num_filtered, plda_features.data());
            
            t_stage_end = Clock::now();
            t_filter_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_filter_plda_ms);
            
            // Step 13: AHC cluster (on normalized embeddings)
            t_stage_start = Clock::now();
            std::vector<int> ahc_clusters;
            diarization::ahc_cluster(
                filtered_normed.data(), num_filtered, EMBEDDING_DIM,
                AHC_THRESHOLD, ahc_clusters);

            int num_ahc_clusters = 0;
            for (int i = 0; i < num_filtered; i++) {
                if (ahc_clusters[i] + 1 > num_ahc_clusters)
                    num_ahc_clusters = ahc_clusters[i] + 1;
            }
            fprintf(stderr, "PLDA: done ");
            fflush(stderr);

            // Free intermediates
            filtered_normed.clear();
            filtered_normed.shrink_to_fit();
            filtered_double.clear();
            filtered_double.shrink_to_fit();

            fprintf(stderr, "AHC: %d clusters ", num_ahc_clusters);
            fflush(stderr);
            
            auto t_ahc_end = Clock::now();
            double t_ahc_ms = std::chrono::duration<double, std::milli>(t_ahc_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_ahc_ms);

            // Step 14: VBx clustering
            t_stage_start = Clock::now();
            diarization::VBxResult vbx_result;
            if (!diarization::vbx_cluster(
                    ahc_clusters.data(), num_filtered, num_ahc_clusters,
                    plda_features.data(), PLDA_DIM,
                    plda.plda_psi.data(), VBX_FA, VBX_FB, VBX_MAX_ITERS,
                    vbx_result)) {
                fprintf(stderr, "Error: VBx clustering failed\n");
                result.segments.clear();
                return false;
            }

            // Step 15: Compute centroids from VBx soft assignments
            // W = gamma[:, pi > 1e-7], centroids = W.T @ train_emb / W.sum(0).T
            const int vbx_S = vbx_result.num_speakers;
            const int vbx_T = vbx_result.num_frames;

            std::vector<int> sig_speakers;
            for (int s = 0; s < vbx_S; s++) {
                if (vbx_result.pi[s] > 1e-7) {
                    sig_speakers.push_back(s);
                }
            }
            num_clusters = static_cast<int>(sig_speakers.size());
            if (num_clusters == 0) num_clusters = 1;  // fallback

            t_stage_end = Clock::now();
            double t_vbx_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            t_clustering_ms = t_ahc_ms + t_vbx_ms;
            fprintf(stderr, "VBx: %d speakers [%.0f ms]\n", num_clusters, t_vbx_ms);

            // centroids: (num_clusters, EMBEDDING_DIM)
            std::vector<float> centroids(
                static_cast<size_t>(num_clusters) * EMBEDDING_DIM, 0.0f);
            std::vector<double> w_col_sum(num_clusters, 0.0);

            for (int k = 0; k < num_clusters; k++) {
                const int s = sig_speakers[k];
                for (int t = 0; t < vbx_T; t++) {
                    const double w = vbx_result.gamma[t * vbx_S + s];
                    w_col_sum[k] += w;
                    const float* emb = filtered_emb.data() + t * EMBEDDING_DIM;
                    float* cent = centroids.data() + k * EMBEDDING_DIM;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        cent[d] += static_cast<float>(w * static_cast<double>(emb[d]));
                    }
                }
            }
            for (int k = 0; k < num_clusters; k++) {
                if (w_col_sum[k] > 0.0) {
                    float* cent = centroids.data() + k * EMBEDDING_DIM;
                    const float inv = static_cast<float>(1.0 / w_col_sum[k]);
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        cent[d] *= inv;
                    }
                }
            }

            // Step 16: Compute soft clusters and assign all embeddings
            // cosine distance: all embeddings vs centroids
            // soft_clusters = 2 - cosine_distance (similarity, range [0, 2])
            const int total_emb = num_chunks * NUM_LOCAL_SPEAKERS;
            std::vector<float> soft_clusters(
                static_cast<size_t>(total_emb) * num_clusters);

            for (int e = 0; e < total_emb; e++) {
                const float* emb = embeddings.data() + e * EMBEDDING_DIM;
                for (int k = 0; k < num_clusters; k++) {
                    const float* cent = centroids.data() + k * EMBEDDING_DIM;
                    double dist = diarization::cosine_distance(emb, cent, EMBEDDING_DIM);
                    soft_clusters[e * num_clusters + k] =
                        static_cast<float>(2.0 - dist);
                }
            }

            // Constrained argmax (Hungarian per chunk)
            diarization::constrained_argmax(
                soft_clusters.data(), num_chunks, NUM_LOCAL_SPEAKERS, num_clusters,
                hard_clusters);
            }
        }

        // Step 17: Mark inactive speakers as -2
        // inactive = sum(binarized[c, :, s]) == 0
        t_stage_start = Clock::now();
        for (int c = 0; c < num_chunks; c++) {
            const float* seg_chunk =
                binarized.data() + c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;
            for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                float sum = 0.0f;
                for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                    sum += seg_chunk[f * NUM_LOCAL_SPEAKERS + s];
                }
                if (sum == 0.0f) {
                    hard_clusters[c * NUM_LOCAL_SPEAKERS + s] = -2;
                }
            }
        }

        // ================================================================
        // Step 18: Reconstruct — build clustered segmentations
        // (num_chunks, 589, num_clusters) from binarized + hard_clusters
        // ================================================================

        std::vector<float> clustered_seg(
            static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * num_clusters);
        {
            const float nan_val = std::nanf("");
            std::fill(clustered_seg.begin(), clustered_seg.end(), nan_val);
        }

        for (int c = 0; c < num_chunks; c++) {
            const int* chunk_clusters = hard_clusters.data() + c * NUM_LOCAL_SPEAKERS;
            const float* seg_chunk =
                binarized.data() + c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;

            for (int k = 0; k < num_clusters; k++) {
                // Find all local speakers assigned to cluster k in this chunk
                for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                    if (chunk_clusters[s] != k) continue;

                    for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                        const float val =
                            seg_chunk[f * NUM_LOCAL_SPEAKERS + s];
                        float& out =
                            clustered_seg[
                                (c * FRAMES_PER_CHUNK + f) * num_clusters + k];
                        if (std::isnan(out)) {
                            out = val;
                        } else {
                            out = std::max(out, val);
                        }
                    }
                }
            }
        }

        // ================================================================
        // Step 19: to_diarization — aggregate + select top-count speakers
        // ================================================================

        std::vector<float> discrete_diarization;
        diarization::to_diarization(
            clustered_seg.data(), num_chunks, FRAMES_PER_CHUNK, num_clusters,
            count.data(), total_frames,
            chunk_window, frame_window,
            discrete_diarization);

        // Free intermediates
        clustered_seg.clear();
        clustered_seg.shrink_to_fit();

        // ================================================================
        // Step 20: Convert to RTTM segments
        // Each contiguous run of 1.0 in a speaker column -> one segment
        // ================================================================

        const int out_frames =
            static_cast<int>(discrete_diarization.size()) / num_clusters;

        std::vector<diarization::RTTMSegment> rttm_segments;

        for (int k = 0; k < num_clusters; k++) {
            char speaker_label[16];
            snprintf(speaker_label, sizeof(speaker_label), "SPEAKER_%02d", k);

            bool in_segment = false;
            int seg_start_frame = 0;

            for (int f = 0; f <= out_frames; f++) {
                bool active = false;
                if (f < out_frames) {
                    active = (discrete_diarization[f * num_clusters + k] == 1.0f);
                }

                if (active && !in_segment) {
                    seg_start_frame = f;
                    in_segment = true;
                } else if (!active && in_segment) {
                    // Segment: [seg_start_frame, f) frames
                    // Use frame midpoint (matches Python Binarize: timestamps = [frames[i].middle for i in ...])
                    const double start_time =
                        chunk_window.start + seg_start_frame * FRAME_STEP + 0.5 * FRAME_DURATION;
                    const double duration =
                        (f - seg_start_frame) * FRAME_STEP;
                    if (duration > 0.0) {
                        rttm_segments.push_back(
                            {start_time, duration, speaker_label});
                    }
                    in_segment = false;
                }
            }
        }

        // Sort segments by start time
        std::sort(rttm_segments.begin(), rttm_segments.end(),
                  [](const diarization::RTTMSegment& a,
                     const diarization::RTTMSegment& b) {
                      return a.start < b.start;
                  });

        // ================================================================
        // Step 21: Write RTTM output
        // ================================================================

        std::string uri = config.audio_path.empty() ? "audio" : extract_uri(config.audio_path);

        if (config.output_path.empty()) {
            diarization::write_rttm_stdout(rttm_segments, uri);
        } else {
            diarization::write_rttm(rttm_segments, uri, config.output_path);
        }

        // Populate result struct
        result.segments.clear();
        result.segments.reserve(rttm_segments.size());
        for (const auto& seg : rttm_segments) {
            result.segments.push_back({seg.start, seg.duration, seg.speaker});
        }

        t_stage_end = Clock::now();
        t_postprocess_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
        fprintf(stderr, "Assignment + reconstruction: done [%.0f ms]\n", t_postprocess_ms);
        fprintf(stderr, "RTTM: %zu segments [0 ms]\n", rttm_segments.size());
        
        auto t_total_end = Clock::now();
        double t_total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
        
        fprintf(stderr, "\n=== Timing Summary ===\n");
        fprintf(stderr, "  Load audio:      %6.0f ms\n", t_load_audio_ms);
        fprintf(stderr, "  Load seg model:  %6.0f ms\n", t_load_seg_model_ms);
        fprintf(stderr, "  Load emb model:  %6.0f ms\n", t_load_emb_model_ms);
        fprintf(stderr, "  Load PLDA:       %6.0f ms\n", t_load_plda_ms);
        fprintf(stderr, "  Segmentation:    %6.0f ms  (%.1f%%)\n", 
                t_segmentation_ms, 100.0 * t_segmentation_ms / t_total_ms);
        fprintf(stderr, "  Powerset:        %6.0f ms\n", t_powerset_ms);
        fprintf(stderr, "  Speaker count:   %6.0f ms\n", t_speaker_count_ms);
        fprintf(stderr, "  Embeddings:      %6.0f ms  (%.1f%%)\n", 
                t_embeddings_ms, 100.0 * t_embeddings_ms / t_total_ms);
        fprintf(stderr, "  Filter+PLDA:     %6.0f ms\n", t_filter_plda_ms);
        fprintf(stderr, "  AHC+VBx:         %6.0f ms\n", t_clustering_ms);
        fprintf(stderr, "  Post-process:    %6.0f ms\n", t_postprocess_ms);
        fprintf(stderr, "  ─────────────────────────\n");
        
        int total_mins = static_cast<int>(t_total_ms / 1000.0) / 60;
        double total_secs = (t_total_ms / 1000.0) - (total_mins * 60);
        fprintf(stderr, "  Total:           %6.0f ms  (%d:%.1f)\n", 
                t_total_ms, total_mins, total_secs);
        
        fprintf(stderr, "\nDiarization complete: %zu segments, %d speakers\n",
                rttm_segments.size(), num_clusters);
        return true;
    }

    // --- cleanup labels for error/early-exit paths ---
cleanup_emb:
    if (coreml_ctx) {
#ifdef EMBEDDING_USE_COREML
        embedding_coreml_free(coreml_ctx);
#endif
    } else {
        if (emb_state.sched) embedding::state_free(emb_state);
        if (emb_model.ctx) embedding::model_free(emb_model);
    }
    return true;

cleanup:
    if (coreml_ctx) {
#ifdef EMBEDDING_USE_COREML
        embedding_coreml_free(coreml_ctx);
#endif
    } else {
        if (emb_state.sched) embedding::state_free(emb_state);
        if (emb_model.ctx) embedding::model_free(emb_model);
    }
#ifdef SEGMENTATION_USE_COREML
    if (seg_coreml_ctx) segmentation_coreml_free(seg_coreml_ctx);
#endif
    if (seg_state.sched) segmentation::state_free(seg_state);
    if (seg_model.ctx) segmentation::model_free(seg_model);
    return false;
}

// ============================================================================
// File-based pipeline — loads WAV and delegates to buffer-based pipeline
// ============================================================================

bool diarize(const DiarizationConfig& config, DiarizationResult& result) {
    std::vector<float> audio_samples;
    uint32_t sample_rate = 0;
    if (!load_wav_file(config.audio_path, audio_samples, sample_rate)) {
        return false;
    }
    if (sample_rate != SAMPLE_RATE) {
        fprintf(stderr, "Error: expected %d Hz audio, got %u Hz\n",
                SAMPLE_RATE, sample_rate);
        return false;
    }
    return diarize_from_samples(config, audio_samples.data(),
                                static_cast<int>(audio_samples.size()), result);
}

#if defined(SEGMENTATION_USE_COREML) && defined(EMBEDDING_USE_COREML)
// ============================================================================
// Buffer-based pipeline with pre-loaded models — CoreML only
// ============================================================================

bool diarize_from_samples_with_models(
    const DiarizationConfig& config,
    const float* audio, int n_samples,
    struct segmentation_coreml_context* seg_ctx,
    struct embedding_coreml_context* emb_ctx,
    const diarization::PLDAModel& plda,
    DiarizationResult& result)
{
    using Clock = std::chrono::high_resolution_clock;

    double t_segmentation_ms = 0.0;
    double t_powerset_ms = 0.0;
    double t_speaker_count_ms = 0.0;
    double t_embeddings_ms = 0.0;
    double t_filter_plda_ms = 0.0;
    double t_clustering_ms = 0.0;
    double t_postprocess_ms = 0.0;

    auto t_total_start = Clock::now();
    auto t_stage_start = Clock::now();
    auto t_stage_end   = Clock::now();

    const double audio_duration = static_cast<double>(n_samples) / SAMPLE_RATE;

    int audio_mins = static_cast<int>(audio_duration) / 60;
    int audio_secs = static_cast<int>(audio_duration) % 60;
    fprintf(stderr, "Audio: %.2fs (%d:%02d), %d samples\n",
            audio_duration, audio_mins, audio_secs, n_samples);

    // ====================================================================
    // Step 1: Sliding window segmentation + embedding (parallel)
    // ====================================================================

    int num_chunks = std::max(1,
        1 + static_cast<int>(std::ceil((audio_duration - CHUNK_DURATION) / CHUNK_STEP)));

    std::vector<float> binarized(
        static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS);
    std::vector<float> embeddings(
        static_cast<size_t>(num_chunks) * NUM_LOCAL_SPEAKERS * EMBEDDING_DIM);

    t_stage_start = Clock::now();
    pipeline_parallel_seg_emb(seg_ctx, emb_ctx,
                              audio, n_samples, num_chunks,
                              binarized.data(), embeddings.data(),
                              t_segmentation_ms, t_embeddings_ms);
    t_powerset_ms = 0.0;

    {
        // ================================================================
        // Step 2: Compute frame-level speaker count
        // ================================================================

        t_stage_start = Clock::now();
        diarization::SlidingWindowParams chunk_window = {0.0, CHUNK_DURATION, CHUNK_STEP};
        diarization::SlidingWindowParams frame_window = {chunk_window.start, FRAME_DURATION, FRAME_STEP};

        std::vector<int> count;
        int total_frames = 0;
        diarization::compute_speaker_count(
            binarized.data(), num_chunks, FRAMES_PER_CHUNK, NUM_LOCAL_SPEAKERS,
            chunk_window, frame_window, count, total_frames);

        int max_count = 0;
        for (int i = 0; i < total_frames; i++) {
            if (count[i] > max_count) max_count = count[i];
        }

        t_stage_end = Clock::now();
        t_speaker_count_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();

        if (max_count == 0) {
            fprintf(stderr, "Speaker count: no speakers detected\n");
            result.segments.clear();
            std::string uri = config.audio_path.empty() ? "audio" : extract_uri(config.audio_path);
            std::vector<diarization::RTTMSegment> empty;
            if (config.output_path.empty()) {
                diarization::write_rttm_stdout(empty, uri);
            } else {
                diarization::write_rttm(empty, uri, config.output_path);
            }
            return true;
        }

        fprintf(stderr, "Speaker count: max %d speakers, %d frames [%.0f ms]\n",
                max_count, total_frames, t_speaker_count_ms);

        // ================================================================
        // Step 5: Filter embeddings + PLDA + Clustering
        // ================================================================

        t_stage_start = Clock::now();
        std::vector<float> filtered_emb;
        std::vector<int> filt_chunk_idx, filt_speaker_idx;
        diarization::filter_embeddings(
            embeddings.data(), num_chunks, NUM_LOCAL_SPEAKERS, EMBEDDING_DIM,
            binarized.data(), FRAMES_PER_CHUNK,
            filtered_emb, filt_chunk_idx, filt_speaker_idx);

        const int num_filtered = static_cast<int>(filt_chunk_idx.size());
        fprintf(stderr, "Filter: %d embeddings ", num_filtered);
        fflush(stderr);

        std::vector<int> hard_clusters(
            static_cast<size_t>(num_chunks) * NUM_LOCAL_SPEAKERS, 0);
        int num_clusters = 1;

        if (num_filtered < 2) {
            t_stage_end = Clock::now();
            t_filter_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_filter_plda_ms);
            fprintf(stderr, "Too few embeddings for clustering, using single cluster\n");
        } else {
            // L2-normalize filtered embeddings (for AHC)
            std::vector<double> filtered_normed(
                static_cast<size_t>(num_filtered) * EMBEDDING_DIM);
            for (int i = 0; i < num_filtered; i++) {
                const float* src = filtered_emb.data() + i * EMBEDDING_DIM;
                double* dst = filtered_normed.data() + i * EMBEDDING_DIM;
                double norm = 0.0;
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    dst[d] = static_cast<double>(src[d]);
                    norm += dst[d] * dst[d];
                }
                norm = std::sqrt(norm);
                if (norm > 0.0) {
                    const double inv = 1.0 / norm;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        dst[d] *= inv;
                    }
                }
            }

            // PLDA transform (on original filtered embeddings, NOT normalized)
            std::vector<double> filtered_double(
                static_cast<size_t>(num_filtered) * EMBEDDING_DIM);
            for (int i = 0; i < num_filtered * EMBEDDING_DIM; i++) {
                filtered_double[i] = static_cast<double>(filtered_emb[i]);
            }

            std::vector<double> plda_features(
                static_cast<size_t>(num_filtered) * PLDA_DIM);
            diarization::plda_transform(
                plda, filtered_double.data(), num_filtered, plda_features.data());

            t_stage_end = Clock::now();
            t_filter_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_filter_plda_ms);

            // AHC cluster (on normalized embeddings)
            t_stage_start = Clock::now();
            std::vector<int> ahc_clusters;
            diarization::ahc_cluster(
                filtered_normed.data(), num_filtered, EMBEDDING_DIM,
                AHC_THRESHOLD, ahc_clusters);

            int num_ahc_clusters = 0;
            for (int i = 0; i < num_filtered; i++) {
                if (ahc_clusters[i] + 1 > num_ahc_clusters)
                    num_ahc_clusters = ahc_clusters[i] + 1;
            }
            fprintf(stderr, "PLDA: done ");
            fflush(stderr);

            filtered_normed.clear();
            filtered_normed.shrink_to_fit();
            filtered_double.clear();
            filtered_double.shrink_to_fit();

            fprintf(stderr, "AHC: %d clusters ", num_ahc_clusters);
            fflush(stderr);

            auto t_ahc_end = Clock::now();
            double t_ahc_ms = std::chrono::duration<double, std::milli>(t_ahc_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_ahc_ms);

            // VBx clustering
            t_stage_start = Clock::now();
            diarization::VBxResult vbx_result;
            if (!diarization::vbx_cluster(
                    ahc_clusters.data(), num_filtered, num_ahc_clusters,
                    plda_features.data(), PLDA_DIM,
                    plda.plda_psi.data(), VBX_FA, VBX_FB, VBX_MAX_ITERS,
                    vbx_result)) {
                fprintf(stderr, "Error: VBx clustering failed\n");
                result.segments.clear();
                return false;
            }

            const int vbx_S = vbx_result.num_speakers;
            const int vbx_T = vbx_result.num_frames;

            std::vector<int> sig_speakers;
            for (int s = 0; s < vbx_S; s++) {
                if (vbx_result.pi[s] > 1e-7) {
                    sig_speakers.push_back(s);
                }
            }
            num_clusters = static_cast<int>(sig_speakers.size());
            if (num_clusters == 0) num_clusters = 1;

            t_stage_end = Clock::now();
            double t_vbx_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            t_clustering_ms = t_ahc_ms + t_vbx_ms;
            fprintf(stderr, "VBx: %d speakers [%.0f ms]\n", num_clusters, t_vbx_ms);

            // Compute centroids
            std::vector<float> centroids(
                static_cast<size_t>(num_clusters) * EMBEDDING_DIM, 0.0f);
            std::vector<double> w_col_sum(num_clusters, 0.0);

            for (int k = 0; k < num_clusters; k++) {
                const int s = sig_speakers[k];
                for (int t = 0; t < vbx_T; t++) {
                    const double w = vbx_result.gamma[t * vbx_S + s];
                    w_col_sum[k] += w;
                    const float* emb = filtered_emb.data() + t * EMBEDDING_DIM;
                    float* cent = centroids.data() + k * EMBEDDING_DIM;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        cent[d] += static_cast<float>(w * static_cast<double>(emb[d]));
                    }
                }
            }
            for (int k = 0; k < num_clusters; k++) {
                if (w_col_sum[k] > 0.0) {
                    float* cent = centroids.data() + k * EMBEDDING_DIM;
                    const float inv = static_cast<float>(1.0 / w_col_sum[k]);
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        cent[d] *= inv;
                    }
                }
            }

            // Soft clusters + constrained argmax
            const int total_emb = num_chunks * NUM_LOCAL_SPEAKERS;
            std::vector<float> soft_clusters(
                static_cast<size_t>(total_emb) * num_clusters);

            for (int e = 0; e < total_emb; e++) {
                const float* emb = embeddings.data() + e * EMBEDDING_DIM;
                for (int k = 0; k < num_clusters; k++) {
                    const float* cent = centroids.data() + k * EMBEDDING_DIM;
                    double dist = diarization::cosine_distance(emb, cent, EMBEDDING_DIM);
                    soft_clusters[e * num_clusters + k] =
                        static_cast<float>(2.0 - dist);
                }
            }

            diarization::constrained_argmax(
                soft_clusters.data(), num_chunks, NUM_LOCAL_SPEAKERS, num_clusters,
                hard_clusters);
        }

        // Mark inactive speakers as -2
        t_stage_start = Clock::now();
        for (int c = 0; c < num_chunks; c++) {
            const float* seg_chunk =
                binarized.data() + c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;
            for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                float sum = 0.0f;
                for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                    sum += seg_chunk[f * NUM_LOCAL_SPEAKERS + s];
                }
                if (sum == 0.0f) {
                    hard_clusters[c * NUM_LOCAL_SPEAKERS + s] = -2;
                }
            }
        }

        // ================================================================
        // Step 6: Reconstruct — build clustered segmentations
        // ================================================================

        std::vector<float> clustered_seg(
            static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * num_clusters);
        {
            const float nan_val = std::nanf("");
            std::fill(clustered_seg.begin(), clustered_seg.end(), nan_val);
        }

        for (int c = 0; c < num_chunks; c++) {
            const int* chunk_clusters = hard_clusters.data() + c * NUM_LOCAL_SPEAKERS;
            const float* seg_chunk =
                binarized.data() + c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;

            for (int k = 0; k < num_clusters; k++) {
                for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                    if (chunk_clusters[s] != k) continue;

                    for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                        const float val =
                            seg_chunk[f * NUM_LOCAL_SPEAKERS + s];
                        float& out =
                            clustered_seg[
                                (c * FRAMES_PER_CHUNK + f) * num_clusters + k];
                        if (std::isnan(out)) {
                            out = val;
                        } else {
                            out = std::max(out, val);
                        }
                    }
                }
            }
        }

        // ================================================================
        // Step 7: to_diarization — aggregate + select top-count speakers
        // ================================================================

        std::vector<float> discrete_diarization;
        diarization::to_diarization(
            clustered_seg.data(), num_chunks, FRAMES_PER_CHUNK, num_clusters,
            count.data(), total_frames,
            chunk_window, frame_window,
            discrete_diarization);

        clustered_seg.clear();
        clustered_seg.shrink_to_fit();

        // ================================================================
        // Step 8: Convert to RTTM segments
        // ================================================================

        const int out_frames =
            static_cast<int>(discrete_diarization.size()) / num_clusters;

        std::vector<diarization::RTTMSegment> rttm_segments;

        for (int k = 0; k < num_clusters; k++) {
            char speaker_label[16];
            snprintf(speaker_label, sizeof(speaker_label), "SPEAKER_%02d", k);

            bool in_segment = false;
            int seg_start_frame = 0;

            for (int f = 0; f <= out_frames; f++) {
                bool active = false;
                if (f < out_frames) {
                    active = (discrete_diarization[f * num_clusters + k] == 1.0f);
                }

                if (active && !in_segment) {
                    seg_start_frame = f;
                    in_segment = true;
                } else if (!active && in_segment) {
                    const double start_time =
                        chunk_window.start + seg_start_frame * FRAME_STEP + 0.5 * FRAME_DURATION;
                    const double duration =
                        (f - seg_start_frame) * FRAME_STEP;
                    if (duration > 0.0) {
                        rttm_segments.push_back(
                            {start_time, duration, speaker_label});
                    }
                    in_segment = false;
                }
            }
        }

        std::sort(rttm_segments.begin(), rttm_segments.end(),
                  [](const diarization::RTTMSegment& a,
                     const diarization::RTTMSegment& b) {
                      return a.start < b.start;
                  });

        // ================================================================
        // Step 9: Write RTTM output
        // ================================================================

        std::string uri = config.audio_path.empty() ? "audio" : extract_uri(config.audio_path);

        if (config.output_path.empty()) {
            diarization::write_rttm_stdout(rttm_segments, uri);
        } else {
            diarization::write_rttm(rttm_segments, uri, config.output_path);
        }

        // Populate result struct
        result.segments.clear();
        result.segments.reserve(rttm_segments.size());
        for (const auto& seg : rttm_segments) {
            result.segments.push_back({seg.start, seg.duration, seg.speaker});
        }

        t_stage_end = Clock::now();
        t_postprocess_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
        fprintf(stderr, "Assignment + reconstruction: done [%.0f ms]\n", t_postprocess_ms);

        auto t_total_end = Clock::now();
        double t_total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

        fprintf(stderr, "\n=== Timing Summary (with_models) ===\n");
        fprintf(stderr, "  Segmentation:    %6.0f ms  (%.1f%%)\n",
                t_segmentation_ms, 100.0 * t_segmentation_ms / t_total_ms);
        fprintf(stderr, "  Powerset:        %6.0f ms\n", t_powerset_ms);
        fprintf(stderr, "  Speaker count:   %6.0f ms\n", t_speaker_count_ms);
        fprintf(stderr, "  Embeddings:      %6.0f ms  (%.1f%%)\n",
                t_embeddings_ms, 100.0 * t_embeddings_ms / t_total_ms);
        fprintf(stderr, "  Filter+PLDA:     %6.0f ms\n", t_filter_plda_ms);
        fprintf(stderr, "  AHC+VBx:         %6.0f ms\n", t_clustering_ms);
        fprintf(stderr, "  Post-process:    %6.0f ms\n", t_postprocess_ms);
        fprintf(stderr, "  ─────────────────────────\n");
        fprintf(stderr, "  Total:           %6.0f ms\n", t_total_ms);

        fprintf(stderr, "\nDiarization complete: %zu segments, %d speakers\n",
                rttm_segments.size(), num_clusters);
        return true;
    }

    return true;
}
#endif  // SEGMENTATION_USE_COREML && EMBEDDING_USE_COREML
