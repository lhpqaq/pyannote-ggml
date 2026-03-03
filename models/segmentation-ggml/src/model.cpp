#include "model.h"
#include "sincnet.h"
#include "lstm.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ggml-alloc.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

namespace segmentation {

// ============================================================================
// Helper Functions
// ============================================================================

static struct ggml_tensor* get_tensor(struct ggml_context* ctx, const char* name, bool required = true) {
    struct ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor && required) {
        fprintf(stderr, "ERROR: Required tensor '%s' not found\n", name);
    }
    return tensor;
}

static uint32_t get_u32_meta(struct gguf_context* ctx, const char* key, uint32_t default_val) {
    int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        fprintf(stderr, "WARNING: Metadata key '%s' not found, using default %u\n", key, default_val);
        return default_val;
    }
    return gguf_get_val_u32(ctx, key_id);
}

static void print_tensor_info(const char* name, struct ggml_tensor* tensor) {
    if (!tensor) {
        printf("  %-45s: (null)\n", name);
        return;
    }
    
    char shape_str[128];
    int n_dims = ggml_n_dims(tensor);
    int offset = 0;
    offset += snprintf(shape_str + offset, sizeof(shape_str) - offset, "[");
    for (int i = n_dims - 1; i >= 0; i--) {
        offset += snprintf(shape_str + offset, sizeof(shape_str) - offset, 
                          "%lld%s", (long long)tensor->ne[i], i > 0 ? ", " : "");
    }
    snprintf(shape_str + offset, sizeof(shape_str) - offset, "]");
    
    printf("  %-45s: %-20s %s\n", name, shape_str, ggml_type_name(tensor->type));
}

static bool verify_tensor_shape(struct ggml_tensor* tensor, const char* name,
                                int64_t ne0, int64_t ne1 = 0, int64_t ne2 = 0, int64_t ne3 = 0) {
    if (!tensor) {
        fprintf(stderr, "ERROR: Tensor '%s' is null\n", name);
        return false;
    }
    
    bool match = true;
    if (tensor->ne[0] != ne0) match = false;
    if (ne1 > 0 && tensor->ne[1] != ne1) match = false;
    if (ne2 > 0 && tensor->ne[2] != ne2) match = false;
    if (ne3 > 0 && tensor->ne[3] != ne3) match = false;
    
    if (!match) {
        fprintf(stderr, "ERROR: Shape mismatch for '%s': expected [%lld, %lld, %lld, %lld], got [%lld, %lld, %lld, %lld]\n",
                name, (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3,
                (long long)tensor->ne[0], (long long)tensor->ne[1], 
                (long long)tensor->ne[2], (long long)tensor->ne[3]);
    }
    
    return match;
}

static int prefer_gpu_for_supported_nodes(segmentation_state& state, struct ggml_cgraph* graph) {
    if (!state.sched || !state.preferred_gpu || !graph) {
        return 0;
    }
    int assigned = 0;
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) {
            continue;
        }

        if (ggml_backend_supports_op(state.preferred_gpu, node)) {
            ggml_backend_sched_set_tensor_backend(state.sched, node, state.preferred_gpu);
            assigned++;
        }
    }

    return assigned;
}

static void dump_tensor_binary(struct ggml_cgraph* graph, const char* tensor_name, const char* path_prefix, int infer_index) {
    if (!graph || !tensor_name || !path_prefix) {
        return;
    }

    struct ggml_tensor* tensor = ggml_graph_get_tensor(graph, tensor_name);
    if (!tensor) {
        return;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    if (nbytes == 0) {
        return;
    }

    std::vector<uint8_t> bytes(nbytes);
    ggml_backend_tensor_get(tensor, bytes.data(), 0, nbytes);

    char dump_path[1024];
    std::snprintf(dump_path, sizeof(dump_path), "%s/seg_%s_%04d.bin", path_prefix, tensor_name, infer_index);

    std::ofstream out(dump_path, std::ios::binary);
    if (!out.good()) {
        return;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)nbytes);
}

static void report_gpu_offload_coverage_once(segmentation_state& state, struct ggml_cgraph* graph) {
    if (!state.preferred_gpu || !graph || state.printed_gpu_coverage) {
        return;
    }

    int preferred_total = 0;
    int preferred_supported = 0;
    const int n_nodes = ggml_graph_n_nodes(graph);

    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_CONV_2D:
            case GGML_OP_LEAKY_RELU:
            case GGML_OP_SOFT_MAX:
            case GGML_OP_LOG:
                preferred_total++;
                if (ggml_backend_supports_op(state.preferred_gpu, node)) {
                    preferred_supported++;
                }
                break;
            default:
                break;
        }
    }

    if (preferred_total > 0) {
        fprintf(stderr,
                "[backend] segmentation gpu-op support %d/%d (%.1f%%)\n",
                preferred_supported,
                preferred_total,
                100.0 * (double) preferred_supported / (double) preferred_total);
    }

    state.printed_gpu_coverage = true;
}

// ============================================================================
// Model Loading (Phase 2: Backend Allocation Pattern)
// ============================================================================

bool model_load(const std::string& fname,
                segmentation_model& model,
                const std::string& weight_backend,
                int gpu_device,
                bool verbose) {
    if (verbose) printf("Loading model from: %s\n", fname.c_str());
    
    // Step 1: Open GGUF with no_alloc=true (metadata only, no data allocation)
    struct gguf_init_params gguf_params = {
        .no_alloc = true,
        .ctx = &model.ctx,
    };
    
    model.gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!model.gguf_ctx) {
        fprintf(stderr, "ERROR: Failed to open GGUF file: %s\n", fname.c_str());
        return false;
    }
    
    if (verbose) {
        printf("\nGGUF File Info:\n");
        printf("  Version: %u\n", gguf_get_version(model.gguf_ctx));
        printf("  Alignment: %zu\n", gguf_get_alignment(model.gguf_ctx));
        printf("  Tensors: %lld\n", (long long)gguf_get_n_tensors(model.gguf_ctx));
        printf("  Metadata keys: %lld\n", (long long)gguf_get_n_kv(model.gguf_ctx));
    }
    
    // Step 2: Load hyperparameters from metadata
    if (verbose) printf("\nLoading hyperparameters...\n");
    model.hparams.sample_rate    = get_u32_meta(model.gguf_ctx, "pyannet.sample_rate", 16000);
    model.hparams.num_classes    = get_u32_meta(model.gguf_ctx, "pyannet.num_classes", 7);
    model.hparams.lstm_layers    = get_u32_meta(model.gguf_ctx, "pyannet.lstm_layers", 4);
    model.hparams.lstm_hidden    = get_u32_meta(model.gguf_ctx, "pyannet.lstm_hidden", 128);
    model.hparams.sincnet_kernel = get_u32_meta(model.gguf_ctx, "pyannet.sincnet_kernel_size", 251);
    model.hparams.sincnet_stride = get_u32_meta(model.gguf_ctx, "pyannet.sincnet_stride", 10);
    
    if (verbose) {
        printf("  sample_rate: %d\n", model.hparams.sample_rate);
        printf("  num_classes: %d\n", model.hparams.num_classes);
        printf("  lstm_layers: %d\n", model.hparams.lstm_layers);
        printf("  lstm_hidden: %d\n", model.hparams.lstm_hidden);
        printf("  sincnet_kernel: %d\n", model.hparams.sincnet_kernel);
        printf("  sincnet_stride: %d\n", model.hparams.sincnet_stride);
    }
    
    if (!model.ctx) {
        fprintf(stderr, "ERROR: Failed to create ggml context\n");
        gguf_free(model.gguf_ctx);
        model.gguf_ctx = nullptr;
        return false;
    }
    
    // Step 3: Map tensor pointers (metadata only, no data yet)
    if (verbose) printf("\nMapping tensor pointers...\n");
    
    model.wav_norm_weight = get_tensor(model.ctx, "sincnet.wav_norm.weight", false);
    model.wav_norm_bias   = get_tensor(model.ctx, "sincnet.wav_norm.bias", false);
    
    model.sincnet_conv_weight[0] = get_tensor(model.ctx, "sincnet.0.conv.weight");
    model.sincnet_conv_weight[1] = get_tensor(model.ctx, "sincnet.1.conv.weight");
    model.sincnet_conv_weight[2] = get_tensor(model.ctx, "sincnet.2.conv.weight");
    
    model.sincnet_conv_bias[0] = nullptr;
    model.sincnet_conv_bias[1] = get_tensor(model.ctx, "sincnet.1.conv.bias");
    model.sincnet_conv_bias[2] = get_tensor(model.ctx, "sincnet.2.conv.bias");
    
    for (int i = 0; i < SINCNET_STAGES; i++) {
        char weight_name[64], bias_name[64];
        snprintf(weight_name, sizeof(weight_name), "sincnet.%d.norm.weight", i);
        snprintf(bias_name, sizeof(bias_name), "sincnet.%d.norm.bias", i);
        
        model.sincnet_norm_weight[i] = get_tensor(model.ctx, weight_name);
        model.sincnet_norm_bias[i]   = get_tensor(model.ctx, bias_name);
    }
    
    for (int layer = 0; layer < LSTM_LAYERS; layer++) {
        char name[64];
        
        snprintf(name, sizeof(name), "lstm.weight_ih_l%d", layer);
        model.lstm_weight_ih[layer] = get_tensor(model.ctx, name);
        
        snprintf(name, sizeof(name), "lstm.weight_hh_l%d", layer);
        model.lstm_weight_hh[layer] = get_tensor(model.ctx, name);
        
        snprintf(name, sizeof(name), "lstm.bias_ih_l%d", layer);
        model.lstm_bias_ih[layer] = get_tensor(model.ctx, name);
        
        snprintf(name, sizeof(name), "lstm.bias_hh_l%d", layer);
        model.lstm_bias_hh[layer] = get_tensor(model.ctx, name);
        
        snprintf(name, sizeof(name), "lstm.weight_ih_l%d_reverse", layer);
        model.lstm_weight_ih_reverse[layer] = get_tensor(model.ctx, name);
        
        snprintf(name, sizeof(name), "lstm.weight_hh_l%d_reverse", layer);
        model.lstm_weight_hh_reverse[layer] = get_tensor(model.ctx, name);
        
        snprintf(name, sizeof(name), "lstm.bias_ih_l%d_reverse", layer);
        model.lstm_bias_ih_reverse[layer] = get_tensor(model.ctx, name);
        
        snprintf(name, sizeof(name), "lstm.bias_hh_l%d_reverse", layer);
        model.lstm_bias_hh_reverse[layer] = get_tensor(model.ctx, name);
    }
    
    for (int layer = 0; layer < LINEAR_LAYERS; layer++) {
        char weight_name[64], bias_name[64];
        snprintf(weight_name, sizeof(weight_name), "linear.%d.weight", layer);
        snprintf(bias_name, sizeof(bias_name), "linear.%d.bias", layer);
        
        model.linear_weight[layer] = get_tensor(model.ctx, weight_name);
        model.linear_bias[layer]   = get_tensor(model.ctx, bias_name);
    }
    
    model.classifier_weight = get_tensor(model.ctx, "classifier.weight");
    model.classifier_bias   = get_tensor(model.ctx, "classifier.bias");
    
    // Step 4: Allocate weight buffers on selected backend
    if (verbose) printf("\nAllocating weight buffers...\n");

    ggml_backend_t weight_backend_handle = nullptr;
    if (weight_backend == "cuda") {
        char device_desc[32];
        std::snprintf(device_desc, sizeof(device_desc), "CUDA%d", gpu_device);
        weight_backend_handle = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, device_desc);
        if (!weight_backend_handle) {
            fprintf(stderr,
                    "WARNING: Failed to initialize CUDA weight backend '%s', falling back to CPU\n",
                    device_desc);
        }
    }

    if (!weight_backend_handle) {
        weight_backend_handle = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (!weight_backend_handle) {
        fprintf(stderr, "ERROR: Failed to initialize backend for weights\n");
        return false;
    }
    if (verbose) printf("  Using backend for weights: %s\n", ggml_backend_name(weight_backend_handle));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(model.ctx, weight_backend_handle);
    if (!buf) {
        fprintf(stderr, "ERROR: Failed to allocate weight buffer\n");
        ggml_backend_free(weight_backend_handle);
        return false;
    }
    
    model.weight_buffers.push_back(buf);
    model.weight_backends.push_back(weight_backend_handle);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    
    if (verbose) printf("  Weight buffer: %.2f MB (%s)\n",
           ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0,
           ggml_backend_buffer_name(buf));
    
    // Step 5: Load weight data from GGUF file
    if (verbose) printf("\nLoading weight data from file...\n");
    
    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open file for weight loading: %s\n", fname.c_str());
        return false;
    }
    
    int n_tensors = gguf_get_n_tensors(model.gguf_ctx);
    std::vector<uint8_t> read_buf;
    
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(model.gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model.ctx, name);
        
        if (!tensor) {
            fprintf(stderr, "WARNING: Tensor '%s' not found in context\n", name);
            continue;
        }
        
        size_t offset = gguf_get_data_offset(model.gguf_ctx) + gguf_get_tensor_offset(model.gguf_ctx, i);
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
        fseek(f, (long)offset, SEEK_SET);
        size_t nread = fread(read_buf.data(), 1, nbytes, f);
        if (nread != nbytes) {
            fprintf(stderr, "ERROR: Failed to read tensor '%s' data (read %zu of %zu bytes)\n",
                    name, nread, nbytes);
            fclose(f);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    
    if (verbose) printf("  Loaded %d tensors\n", n_tensors);
    
    return true;
}

bool model_load(const std::string& fname, segmentation_model& model, bool verbose) {
    return model_load(fname, model, "cpu", 0, verbose);
}

void model_free(segmentation_model& model) {
    for (size_t i = 0; i < model.weight_buffers.size(); i++) {
        ggml_backend_buffer_free(model.weight_buffers[i]);
    }
    model.weight_buffers.clear();

    for (size_t i = 0; i < model.weight_backends.size(); i++) {
        ggml_backend_free(model.weight_backends[i]);
    }
    model.weight_backends.clear();
    
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.gguf_ctx) {
        gguf_free(model.gguf_ctx);
        model.gguf_ctx = nullptr;
    }
}

void model_print_info(const segmentation_model& model) {
    printf("\n");
    printf("============================================================\n");
    printf("Model Tensor Summary\n");
    printf("============================================================\n");
    
    printf("\nSincNet Tensors:\n");
    print_tensor_info("sincnet.wav_norm.weight", model.wav_norm_weight);
    print_tensor_info("sincnet.wav_norm.bias", model.wav_norm_bias);
    
    for (int i = 0; i < SINCNET_STAGES; i++) {
        char name[64];
        snprintf(name, sizeof(name), "sincnet.%d.conv.weight", i);
        print_tensor_info(name, model.sincnet_conv_weight[i]);
        
        snprintf(name, sizeof(name), "sincnet.%d.conv.bias", i);
        print_tensor_info(name, model.sincnet_conv_bias[i]);
        
        snprintf(name, sizeof(name), "sincnet.%d.norm.weight", i);
        print_tensor_info(name, model.sincnet_norm_weight[i]);
        
        snprintf(name, sizeof(name), "sincnet.%d.norm.bias", i);
        print_tensor_info(name, model.sincnet_norm_bias[i]);
    }
    
    printf("\nLSTM Tensors:\n");
    for (int layer = 0; layer < LSTM_LAYERS; layer++) {
        char name[64];
        
        snprintf(name, sizeof(name), "lstm.weight_ih_l%d", layer);
        print_tensor_info(name, model.lstm_weight_ih[layer]);
        
        snprintf(name, sizeof(name), "lstm.weight_hh_l%d", layer);
        print_tensor_info(name, model.lstm_weight_hh[layer]);
        
        snprintf(name, sizeof(name), "lstm.bias_ih_l%d", layer);
        print_tensor_info(name, model.lstm_bias_ih[layer]);
        
        snprintf(name, sizeof(name), "lstm.bias_hh_l%d", layer);
        print_tensor_info(name, model.lstm_bias_hh[layer]);
        
        snprintf(name, sizeof(name), "lstm.weight_ih_l%d_reverse", layer);
        print_tensor_info(name, model.lstm_weight_ih_reverse[layer]);
        
        snprintf(name, sizeof(name), "lstm.weight_hh_l%d_reverse", layer);
        print_tensor_info(name, model.lstm_weight_hh_reverse[layer]);
        
        snprintf(name, sizeof(name), "lstm.bias_ih_l%d_reverse", layer);
        print_tensor_info(name, model.lstm_bias_ih_reverse[layer]);
        
        snprintf(name, sizeof(name), "lstm.bias_hh_l%d_reverse", layer);
        print_tensor_info(name, model.lstm_bias_hh_reverse[layer]);
    }
    
    printf("\nLinear Tensors:\n");
    for (int layer = 0; layer < LINEAR_LAYERS; layer++) {
        char name[64];
        snprintf(name, sizeof(name), "linear.%d.weight", layer);
        print_tensor_info(name, model.linear_weight[layer]);
        
        snprintf(name, sizeof(name), "linear.%d.bias", layer);
        print_tensor_info(name, model.linear_bias[layer]);
    }
    
    printf("\nClassifier Tensors:\n");
    print_tensor_info("classifier.weight", model.classifier_weight);
    print_tensor_info("classifier.bias", model.classifier_bias);
    
    printf("\n============================================================\n");
}

void model_print_memory_usage(const segmentation_model& model, const segmentation_state& state) {
    printf("\n");
    printf("============================================================\n");
    printf("Memory Usage\n");
    printf("============================================================\n");

    // Weight buffer size
    size_t weight_size = 0;
    for (size_t i = 0; i < model.weight_buffers.size(); i++) {
        if (model.weight_buffers[i]) {
            weight_size += ggml_backend_buffer_get_size(model.weight_buffers[i]);
        }
    }
    printf("  Weight buffer:  %8.2f MB\n", weight_size / (1024.0 * 1024.0));

    // Compute buffer size from scheduler
    size_t compute_size = 0;
    if (state.sched) {
        int n_backends = ggml_backend_sched_get_n_backends(state.sched);
        for (int i = 0; i < n_backends; i++) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(state.sched, i);
            if (backend) {
                compute_size += ggml_backend_sched_get_buffer_size(state.sched, backend);
            }
        }
    }
    printf("  Compute buffer: %8.2f MB\n", compute_size / (1024.0 * 1024.0));

    // Graph metadata
    size_t meta_size = state.graph_meta.size();
    printf("  Graph metadata: %8.2f MB\n", meta_size / (1024.0 * 1024.0));

    // Total GGML
    size_t total_ggml = weight_size + compute_size + meta_size;
    printf("  Total GGML:     %8.2f MB\n", total_ggml / (1024.0 * 1024.0));

    // Peak RSS (process memory)
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        printf("  Peak RSS:       %8.2f MB\n", info.resident_size_max / (1024.0 * 1024.0));
    }
#elif defined(__linux__)
    // On Linux, read from /proc/self/status
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmHWM:", 6) == 0) {
                long peak_kb = 0;
                sscanf(line + 6, "%ld", &peak_kb);
                printf("  Peak RSS:       %8.2f MB\n", peak_kb / 1024.0);
                break;
            }
        }
        fclose(f);
    }
#endif

    printf("============================================================\n");
}

bool model_verify(const segmentation_model& model) {
    bool all_valid = true;
    int null_count = 0;
    int tensor_count = 0;
    
    printf("\nVerifying model tensors...\n");
    
    #define CHECK_TENSOR(tensor, name, ...) do { \
        tensor_count++; \
        if (!(tensor)) { \
            null_count++; \
            fprintf(stderr, "MISSING: %s\n", name); \
            all_valid = false; \
        } else if (!verify_tensor_shape(tensor, name, __VA_ARGS__)) { \
            all_valid = false; \
        } \
    } while(0)
    
    CHECK_TENSOR(model.sincnet_conv_weight[0], "sincnet.0.conv.weight", 251, 1, 80);
    CHECK_TENSOR(model.sincnet_conv_weight[1], "sincnet.1.conv.weight", 5, 80, 60);
    CHECK_TENSOR(model.sincnet_conv_weight[2], "sincnet.2.conv.weight", 5, 60, 60);
    
    CHECK_TENSOR(model.sincnet_conv_bias[1], "sincnet.1.conv.bias", 60);
    CHECK_TENSOR(model.sincnet_conv_bias[2], "sincnet.2.conv.bias", 60);
    
    CHECK_TENSOR(model.sincnet_norm_weight[0], "sincnet.0.norm.weight", 80);
    CHECK_TENSOR(model.sincnet_norm_bias[0], "sincnet.0.norm.bias", 80);
    CHECK_TENSOR(model.sincnet_norm_weight[1], "sincnet.1.norm.weight", 60);
    CHECK_TENSOR(model.sincnet_norm_bias[1], "sincnet.1.norm.bias", 60);
    CHECK_TENSOR(model.sincnet_norm_weight[2], "sincnet.2.norm.weight", 60);
    CHECK_TENSOR(model.sincnet_norm_bias[2], "sincnet.2.norm.bias", 60);
    
    int lstm_hidden = model.hparams.lstm_hidden;
    int gate_size = 4 * lstm_hidden;
    
    for (int layer = 0; layer < LSTM_LAYERS; layer++) {
        int input_size = (layer == 0) ? 60 : (2 * lstm_hidden);
        char name[64];
        
        snprintf(name, sizeof(name), "lstm.weight_ih_l%d", layer);
        CHECK_TENSOR(model.lstm_weight_ih[layer], name, input_size, gate_size);
        
        snprintf(name, sizeof(name), "lstm.weight_hh_l%d", layer);
        CHECK_TENSOR(model.lstm_weight_hh[layer], name, lstm_hidden, gate_size);
        
        snprintf(name, sizeof(name), "lstm.bias_ih_l%d", layer);
        CHECK_TENSOR(model.lstm_bias_ih[layer], name, gate_size);
        
        snprintf(name, sizeof(name), "lstm.bias_hh_l%d", layer);
        CHECK_TENSOR(model.lstm_bias_hh[layer], name, gate_size);
        
        snprintf(name, sizeof(name), "lstm.weight_ih_l%d_reverse", layer);
        CHECK_TENSOR(model.lstm_weight_ih_reverse[layer], name, input_size, gate_size);
        
        snprintf(name, sizeof(name), "lstm.weight_hh_l%d_reverse", layer);
        CHECK_TENSOR(model.lstm_weight_hh_reverse[layer], name, lstm_hidden, gate_size);
        
        snprintf(name, sizeof(name), "lstm.bias_ih_l%d_reverse", layer);
        CHECK_TENSOR(model.lstm_bias_ih_reverse[layer], name, gate_size);
        
        snprintf(name, sizeof(name), "lstm.bias_hh_l%d_reverse", layer);
        CHECK_TENSOR(model.lstm_bias_hh_reverse[layer], name, gate_size);
    }
    
    CHECK_TENSOR(model.linear_weight[0], "linear.0.weight", 256, 128);
    CHECK_TENSOR(model.linear_bias[0], "linear.0.bias", 128);
    CHECK_TENSOR(model.linear_weight[1], "linear.1.weight", 128, 128);
    CHECK_TENSOR(model.linear_bias[1], "linear.1.bias", 128);
    
    int num_classes = model.hparams.num_classes;
    CHECK_TENSOR(model.classifier_weight, "classifier.weight", 128, num_classes);
    CHECK_TENSOR(model.classifier_bias, "classifier.bias", num_classes);
    
    #undef CHECK_TENSOR
    
    printf("\nVerification complete: %d/%d tensors loaded\n", 
           tensor_count - null_count, tensor_count);
    
    if (all_valid) {
        printf("All tensor shapes verified successfully!\n");
    } else {
        fprintf(stderr, "WARNING: Some tensors missing or have incorrect shapes\n");
    }
    
    return all_valid;
}

// ============================================================================
// State Management (Phase 3: Backend Scheduler)
// ============================================================================

bool state_init(segmentation_state& state, segmentation_model& model, bool verbose) {
    return state_init(state, model, "auto", 0, verbose);
}

bool state_init(segmentation_state& state,
                segmentation_model& model,
                const std::string& backend,
                int gpu_device,
                bool verbose) {
    if (verbose) printf("\nInitializing inference state...\n");

    state.backends.clear();

    const bool want_cpu = backend == "cpu" || backend == "auto";
    const bool want_metal = backend == "metal";
    const bool want_cuda = backend == "cuda";

#ifdef GGML_USE_METAL
    constexpr bool metal_compiled = true;
#else
    constexpr bool metal_compiled = false;
#endif

    if (backend == "metal" && !metal_compiled) {
        fprintf(stderr, "ERROR: Metal backend requested but this binary was built without Metal support (GGML_USE_METAL)\n");
        return false;
    }

    ggml_backend_t cpu_primary = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_primary) {
        fprintf(stderr, "ERROR: Failed to initialize CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu_primary, 4);
    state.backends.push_back(cpu_primary);
    if (verbose) printf("  Using CPU backend: %s (4 threads)\n", ggml_backend_name(cpu_primary));

    if (want_metal) {
        ggml_backend_t metal_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, "Metal");
        if (metal_backend) {
            state.backends.push_back(metal_backend);
            if (verbose) printf("  Using GPU backend: %s\n", ggml_backend_name(metal_backend));
        } else if (backend == "metal") {
            fprintf(stderr, "ERROR: Failed to initialize Metal backend (check macOS runtime and GGML_METAL build option)\n");
            return false;
        }
    }

    if (want_cuda) {
        char device_desc[32];
        std::snprintf(device_desc, sizeof(device_desc), "CUDA%d", gpu_device);
        ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, device_desc);
        if (cuda_backend) {
            state.backends.push_back(cuda_backend);
            if (verbose) printf("  Using GPU backend: %s\n", ggml_backend_name(cuda_backend));
        } else if (backend == "cuda") {
            fprintf(stderr, "ERROR: Failed to initialize CUDA backend for device %d\n", gpu_device);
            return false;
        }
    }

    if (state.backends.size() > 1 || want_cpu) {
        ggml_backend_t cpu_fallback = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!cpu_fallback) {
            fprintf(stderr, "ERROR: Failed to initialize CPU fallback backend\n");
            return false;
        }
        ggml_backend_cpu_set_n_threads(cpu_fallback, 4);
        state.backends.push_back(cpu_fallback);
        if (verbose) printf("  Using CPU fallback backend: %s (4 threads)\n", ggml_backend_name(cpu_fallback));
    }

    ggml_backend_t gpu_backend = nullptr;
    for (ggml_backend_t backend_i : state.backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend_i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend_i;
            break;
        }
    }

    if (!gpu_backend && backend != "cpu" && backend != "auto") {
        fprintf(stderr, "ERROR: Requested backend '%s' unavailable\n", backend.c_str());
        return false;
    }

    if (gpu_backend && !state.backends.empty() && state.backends.front() != gpu_backend) {
        auto it = std::find(state.backends.begin(), state.backends.end(), gpu_backend);
        if (it != state.backends.end()) {
            std::rotate(state.backends.begin(), it, it + 1);
        }
    }

    state.preferred_gpu = gpu_backend;
    
    // Step 2: Allocate graph metadata buffer
    size_t meta_size = ggml_tensor_overhead() * MAX_GRAPH_NODES + ggml_graph_overhead_custom(MAX_GRAPH_NODES, false);
    state.graph_meta.resize(meta_size);
    if (verbose) printf("  Graph metadata buffer: %.2f MB\n", meta_size / 1024.0 / 1024.0);
    
    // Step 3: Create backend scheduler
    state.sched = ggml_backend_sched_new(
        state.backends.data(),
        NULL,
        (int)state.backends.size(),
        MAX_GRAPH_NODES,
        false,
        true
    );
    
    if (!state.sched) {
        fprintf(stderr, "ERROR: Failed to create backend scheduler\n");
        return false;
    }
    
    // Step 4: Pre-allocate by building and allocating graph once
    if (verbose) printf("  Pre-allocating compute buffers...\n");
    struct ggml_cgraph* graph = build_graph(model, state);
    if (!graph) {
        fprintf(stderr, "ERROR: Failed to build graph for pre-allocation\n");
        return false;
    }
    
    prefer_gpu_for_supported_nodes(state, graph);
    report_gpu_offload_coverage_once(state, graph);
    // (debug) op inventory removed in cleanup branch

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to pre-allocate compute buffers\n");
        return false;
    }
    
    ggml_backend_sched_reset(state.sched);
    
    if (verbose) printf("  Pre-converting LSTM weights (F16->F32 cache)...\n");
    lstm_init_weight_cache(state.lstm_cache, model);
    
    if (verbose) printf("  State initialized successfully\n");
    
    return true;
}

void state_free(segmentation_state& state) {
    if (state.sched) {
        ggml_backend_sched_free(state.sched);
        state.sched = nullptr;
    }
    
    for (size_t i = 0; i < state.backends.size(); i++) {
        ggml_backend_free(state.backends[i]);
    }
    state.backends.clear();
    state.graph_meta.clear();
}

// ============================================================================
// Linear and Classifier Forward Functions (Graph Building)
// ============================================================================

static struct ggml_tensor* linear_forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    const char* mm_name) {

    const int64_t seq_len = x->ne[0];
    const int64_t input_dim = x->ne[1];
    const int64_t batch = x->ne[2];
    const int64_t output_dim = weight->ne[1];

    // x:   [seq_len, input_dim, batch]
    // x_t: [input_dim, seq_len, batch]
    struct ggml_tensor* x_t = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_t = ggml_cont(ctx, x_t);

    // Fold (seq_len, batch) into columns: [input_dim, seq_len * batch]
    struct ggml_tensor* x_2d = ggml_reshape_2d(ctx, x_t, input_dim, seq_len * batch);
    x_2d = ggml_cont(ctx, x_2d);

    // y_2d: [output_dim, seq_len * batch]
    struct ggml_tensor* y_2d = ggml_mul_mat(ctx, weight, x_2d);
    if (mm_name) {
        ggml_set_name(y_2d, mm_name);
    }
    y_2d = ggml_add(ctx, y_2d, bias);
    y_2d = ggml_leaky_relu(ctx, y_2d, 0.01f, true);

    // Un-flatten: [output_dim, seq_len, batch] -> [seq_len, output_dim, batch]
    struct ggml_tensor* y_3d = ggml_reshape_3d(ctx, y_2d, output_dim, seq_len, batch);
    struct ggml_tensor* y = ggml_permute(ctx, y_3d, 1, 0, 2, 3);
    y = ggml_cont(ctx, y);
    return y;
}

static struct ggml_tensor* classifier_forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias) {

    const int64_t seq_len = x->ne[0];
    const int64_t input_dim = x->ne[1];
    const int64_t batch = x->ne[2];
    const int64_t num_classes = weight->ne[1];

    // x_t: [input_dim, seq_len, batch]
    struct ggml_tensor* x_t = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_t = ggml_cont(ctx, x_t);

    // Fold (seq_len, batch) into columns: [input_dim, seq_len * batch]
    struct ggml_tensor* x_2d = ggml_reshape_2d(ctx, x_t, input_dim, seq_len * batch);
    x_2d = ggml_cont(ctx, x_2d);

    // logits_2d: [num_classes, seq_len * batch]
    struct ggml_tensor* logits_2d = ggml_mul_mat(ctx, weight, x_2d);
    ggml_set_name(logits_2d, "classifier_mm");
    logits_2d = ggml_add(ctx, logits_2d, bias);

    // softmax over classes per column
    struct ggml_tensor* probs = ggml_soft_max(ctx, logits_2d);
    struct ggml_tensor* log_probs = ggml_log(ctx, probs);

    // Un-flatten: [num_classes, seq_len, batch] -> [seq_len, num_classes, batch]
    struct ggml_tensor* out_3d = ggml_reshape_3d(ctx, log_probs, num_classes, seq_len, batch);
    struct ggml_tensor* output = ggml_permute(ctx, out_3d, 1, 0, 2, 3);
    output = ggml_cont(ctx, output);
    return output;
}

// ============================================================================
// Model Forward Pass (Graph Building Only)
// ============================================================================

struct ggml_tensor* model_forward(
    struct ggml_context* ctx,
    const segmentation_model& model,
    struct ggml_tensor* waveform) {
    
    struct ggml_tensor* features = sincnet_forward(ctx, model, waveform);
    if (!features) {
        fprintf(stderr, "ERROR: SincNet forward pass failed\n");
        return nullptr;
    }

    // Keep a named contiguous view for debugging.
    features = ggml_cont(ctx, features);
    ggml_set_name(features, "sincnet_out_cont");
    
    struct ggml_tensor* lstm_out = lstm_forward(ctx, model, features);
    if (!lstm_out) {
        fprintf(stderr, "ERROR: LSTM forward pass failed\n");
        return nullptr;
    }

    // Keep a named contiguous view for debugging. In non-custom mode this is a no-op.
    lstm_out = ggml_cont(ctx, lstm_out);
    ggml_set_name(lstm_out, "lstm_out_cont");
    
    struct ggml_tensor* linear1_out = linear_forward(ctx, lstm_out,
                                                     model.linear_weight[0],
                                                     model.linear_bias[0],
                                                     "linear1_mm");

    struct ggml_tensor* linear2_out = linear_forward(ctx, linear1_out,
                                                     model.linear_weight[1],
                                                     model.linear_bias[1],
                                                     "linear2_mm");
    
    struct ggml_tensor* output = classifier_forward(ctx, linear2_out,
                                                    model.classifier_weight,
                                                    model.classifier_bias);
    ggml_set_name(output, "classifier_out");
    
    return output;
}

// ============================================================================
// Graph Building (Phase 4: Pure Function)
// ============================================================================

struct ggml_cgraph* build_graph(
    segmentation_model& model,
    segmentation_state& state) {
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state.graph_meta.size(),
        /*.mem_buffer =*/ state.graph_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create graph context\n");
        return nullptr;
    }
    
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, MAX_GRAPH_NODES, false);
    
    struct ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 160000, 1, 1);
    ggml_set_name(input, "waveform");
    ggml_set_input(input);
    
    struct ggml_tensor* output = model_forward(ctx, model, input);
    if (!output) {
        ggml_free(ctx);
        return nullptr;
    }
    
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    
    ggml_free(ctx);
    
    return graph;
}

static struct ggml_cgraph* build_graph_batch(
    segmentation_model& model,
    segmentation_state& state,
    int batch) {

    if (batch <= 0) {
        return nullptr;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ state.graph_meta.size(),
        /*.mem_buffer =*/ state.graph_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create graph context\n");
        return nullptr;
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, MAX_GRAPH_NODES, false);

    struct ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 160000, 1, batch);
    ggml_set_name(input, "waveform");
    ggml_set_input(input);

    struct ggml_tensor* output = model_forward(ctx, model, input);
    if (!output) {
        ggml_free(ctx);
        return nullptr;
    }

    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_free(ctx);
    return graph;
}

bool state_prepare_batch(segmentation_state& state,
                         segmentation_model& model,
                         int batch,
                         bool verbose) {
    if (batch <= 1) {
        return true;
    }
    if (!state.sched) {
        fprintf(stderr, "ERROR: state_prepare_batch requires initialized scheduler\n");
        return false;
    }
    if (verbose) {
        printf("  Preparing scheduler buffers for batch=%d...\n", batch);
    }

    struct ggml_cgraph* graph = build_graph_batch(model, state, batch);
    if (!graph) {
        fprintf(stderr, "ERROR: Failed to build batch graph for pre-allocation\n");
        return false;
    }

    prefer_gpu_for_supported_nodes(state, graph);
    report_gpu_offload_coverage_once(state, graph);

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to allocate compute buffers for batch graph\n");
        return false;
    }
    ggml_backend_sched_reset(state.sched);
    return true;
}

// ============================================================================
// Inference (Phase 5: Backend Scheduler)
// ============================================================================

void model_enable_profile(bool /* enable */) {
}

bool model_infer(
    segmentation_model& model,
    segmentation_state& state,
    const float* audio,
    size_t n_samples,
    float* output,
    size_t output_size) {
    
    auto prof_start = std::chrono::high_resolution_clock::now();
    
    struct ggml_cgraph* graph = build_graph(model, state);
    if (!graph) {
        fprintf(stderr, "ERROR: Failed to build computation graph\n");
        return false;
    }
    
    auto prof_graph_built = std::chrono::high_resolution_clock::now();
    
    prefer_gpu_for_supported_nodes(state, graph);
    report_gpu_offload_coverage_once(state, graph);

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to allocate compute buffers\n");
        return false;
    }

    auto prof_alloc_done = std::chrono::high_resolution_clock::now();
    
    struct ggml_tensor* input_tensor = ggml_graph_get_tensor(graph, "waveform");
    if (!input_tensor) {
        fprintf(stderr, "ERROR: Input tensor 'waveform' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }
    ggml_backend_tensor_set(input_tensor, audio, 0, n_samples * sizeof(float));
    
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if ((node->flags & GGML_TENSOR_FLAG_INPUT) && node != input_tensor) {
            size_t nbytes = ggml_nbytes(node);
            std::vector<uint8_t> zeros(nbytes, 0);
            ggml_backend_tensor_set(node, zeros.data(), 0, nbytes);
        }
    }
    
    auto prof_input_set = std::chrono::high_resolution_clock::now();
    
    lstm_reset_profile();
    lstm_set_active_cache(state.lstm_cache.initialized ? &state.lstm_cache : nullptr);
    
    if (ggml_backend_sched_graph_compute(state.sched, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }
    
    auto prof_compute_done = std::chrono::high_resolution_clock::now();
    
    struct ggml_tensor* output_tensor = ggml_graph_get_tensor(graph, "classifier_out");
    if (!output_tensor) {
        fprintf(stderr, "ERROR: Output tensor 'classifier_out' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }
    
    size_t output_bytes = ggml_nbytes(output_tensor);
    size_t requested_bytes = output_size * sizeof(float);
    size_t copy_bytes = (output_bytes < requested_bytes) ? output_bytes : requested_bytes;
    ggml_backend_tensor_get(output_tensor, output, 0, copy_bytes);

    ggml_backend_sched_reset(state.sched);
    
    auto prof_end = std::chrono::high_resolution_clock::now();
    
    return true;
}

bool model_infer_batch(
    segmentation_model& model,
    segmentation_state& state,
    const float* audio,
    size_t n_samples_per_chunk,
    int batch,
    float* output,
    size_t output_size) {

    if (batch <= 0) {
        return false;
    }
    if (n_samples_per_chunk != 160000) {
        fprintf(stderr, "ERROR: model_infer_batch expects 160000 samples per chunk\n");
        return false;
    }

    struct ggml_cgraph* graph = build_graph_batch(model, state, batch);
    if (!graph) {
        fprintf(stderr, "ERROR: Failed to build computation graph (batch=%d)\n", batch);
        return false;
    }

    prefer_gpu_for_supported_nodes(state, graph);
    report_gpu_offload_coverage_once(state, graph);

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to allocate compute buffers\n");
        return false;
    }

    struct ggml_tensor* input_tensor = ggml_graph_get_tensor(graph, "waveform");
    if (!input_tensor) {
        fprintf(stderr, "ERROR: Input tensor 'waveform' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    // ggml tensor memory for [ne0, ne1, ne2] is laid out with ne0 contiguous,
    // then ne1, then ne2 (i.e. waveform for each batch element is contiguous).
    // Our input buffer is also [batch][samples] contiguous, so we can copy once.
    const size_t input_bytes = (size_t) batch * n_samples_per_chunk * sizeof(float);
    ggml_backend_tensor_set(input_tensor, audio, 0, input_bytes);

    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if ((node->flags & GGML_TENSOR_FLAG_INPUT) && node != input_tensor) {
            size_t nbytes = ggml_nbytes(node);
            std::vector<uint8_t> zeros(nbytes, 0);
            ggml_backend_tensor_set(node, zeros.data(), 0, nbytes);
        }
    }

    lstm_reset_profile();
    lstm_set_active_cache(state.lstm_cache.initialized ? &state.lstm_cache : nullptr);

    if (ggml_backend_sched_graph_compute(state.sched, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    struct ggml_tensor* output_tensor = ggml_graph_get_tensor(graph, "classifier_out");
    if (!output_tensor) {
        fprintf(stderr, "ERROR: Output tensor 'classifier_out' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    // Return raw ggml output layout to match model_infer.
    // Caller will transpose from [class, frame] to [frame, class].
    const size_t out_bytes = ggml_nbytes(output_tensor);
    const size_t requested_bytes = output_size * sizeof(float);
    const size_t copy_bytes = (out_bytes < requested_bytes) ? out_bytes : requested_bytes;
    ggml_backend_tensor_get(output_tensor, output, 0, copy_bytes);

    ggml_backend_sched_reset(state.sched);
    return true;
}

static bool fetch_named_tensor_f32(struct ggml_cgraph* graph, const char* tensor_name, std::vector<float>& out, infer_tensor_info& info) {
    if (!graph || !tensor_name) {
        return false;
    }
    struct ggml_tensor* t = ggml_graph_get_tensor(graph, tensor_name);
    if (!t) {
        fprintf(stderr, "ERROR: tensor '%s' not found in graph\n", tensor_name);
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "ERROR: tensor '%s' is not F32 (%s)\n", tensor_name, ggml_type_name(t->type));
        return false;
    }
    info.ne0 = t->ne[0];
    info.ne1 = t->ne[1];
    info.ne2 = t->ne[2];
    info.ne3 = t->ne[3];
    info.type = t->type;
    const size_t nbytes = ggml_nbytes(t);
    out.resize(nbytes / sizeof(float));
    ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    return true;
}

bool model_infer_tensor(
    segmentation_model& model,
    segmentation_state& state,
    const float* audio,
    size_t n_samples,
    const char* tensor_name,
    std::vector<float>& out,
    infer_tensor_info& info) {

    struct ggml_cgraph* graph = build_graph(model, state);
    if (!graph) {
        return false;
    }

    prefer_gpu_for_supported_nodes(state, graph);
    report_gpu_offload_coverage_once(state, graph);

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to allocate compute buffers\n");
        return false;
    }

    struct ggml_tensor* input_tensor = ggml_graph_get_tensor(graph, "waveform");
    if (!input_tensor) {
        fprintf(stderr, "ERROR: Input tensor 'waveform' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }
    ggml_backend_tensor_set(input_tensor, audio, 0, n_samples * sizeof(float));

    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if ((node->flags & GGML_TENSOR_FLAG_INPUT) && node != input_tensor) {
            size_t nbytes = ggml_nbytes(node);
            std::vector<uint8_t> zeros(nbytes, 0);
            ggml_backend_tensor_set(node, zeros.data(), 0, nbytes);
        }
    }

    lstm_reset_profile();
    lstm_set_active_cache(state.lstm_cache.initialized ? &state.lstm_cache : nullptr);

    if (ggml_backend_sched_graph_compute(state.sched, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    const bool ok = fetch_named_tensor_f32(graph, tensor_name, out, info);
    ggml_backend_sched_reset(state.sched);
    return ok;
}

bool model_infer_batch_tensor(
    segmentation_model& model,
    segmentation_state& state,
    const float* audio,
    size_t n_samples_per_chunk,
    int batch,
    const char* tensor_name,
    std::vector<float>& out,
    infer_tensor_info& info) {

    if (batch <= 0) {
        return false;
    }
    if (n_samples_per_chunk != 160000) {
        fprintf(stderr, "ERROR: model_infer_batch_tensor expects 160000 samples per chunk\n");
        return false;
    }

    struct ggml_cgraph* graph = build_graph_batch(model, state, batch);
    if (!graph) {
        return false;
    }

    prefer_gpu_for_supported_nodes(state, graph);
    report_gpu_offload_coverage_once(state, graph);

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to allocate compute buffers\n");
        return false;
    }

    struct ggml_tensor* input_tensor = ggml_graph_get_tensor(graph, "waveform");
    if (!input_tensor) {
        fprintf(stderr, "ERROR: Input tensor 'waveform' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }
    const size_t input_bytes = (size_t) batch * n_samples_per_chunk * sizeof(float);
    ggml_backend_tensor_set(input_tensor, audio, 0, input_bytes);

    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if ((node->flags & GGML_TENSOR_FLAG_INPUT) && node != input_tensor) {
            size_t nbytes = ggml_nbytes(node);
            std::vector<uint8_t> zeros(nbytes, 0);
            ggml_backend_tensor_set(node, zeros.data(), 0, nbytes);
        }
    }

    lstm_reset_profile();
    lstm_set_active_cache(state.lstm_cache.initialized ? &state.lstm_cache : nullptr);

    if (ggml_backend_sched_graph_compute(state.sched, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    const bool ok = fetch_named_tensor_f32(graph, tensor_name, out, info);
    ggml_backend_sched_reset(state.sched);
    return ok;
}

void state_set_backend_stats(segmentation_state& state, bool enable) {
    (void) state;
    (void) enable;
}

// ============================================================================
// Model Class Implementation
// ============================================================================

Model::Model() {
    memset(&model_, 0, sizeof(model_));
    memset(&state_, 0, sizeof(segmentation_state));
}

Model::~Model() {
    if (loaded_) {
        state_free(state_);
        model_free(model_);
    }
}

bool Model::load(const std::string& model_path) {
    if (loaded_) {
        state_free(state_);
        model_free(model_);
        loaded_ = false;
    }
    
    if (!model_load(model_path, model_)) {
        return false;
    }
    
    model_print_info(model_);
    
    if (!model_verify(model_)) {
        fprintf(stderr, "WARNING: Model verification failed, some features may not work\n");
    }
    
    if (!state_init(state_, model_)) {
        fprintf(stderr, "ERROR: Failed to initialize inference state\n");
        model_free(model_);
        return false;
    }
    
    loaded_ = true;
    return true;
}

std::vector<float> Model::infer(const float* audio_data, int num_samples) {
    if (!loaded_) {
        return {};
    }
    
    int seq_len = 589;
    int num_classes = model_.hparams.num_classes;
    std::vector<float> output(seq_len * num_classes);
    
    if (!model_infer(model_, state_, audio_data, num_samples, output.data(), output.size())) {
        return {};
    }
    
    return output;
}

std::string Model::get_info() const {
    if (!loaded_) {
        return "Model not loaded";
    }
    
    std::ostringstream ss;
    ss << "PyanNet Segmentation Model\n"
       << "  Sample Rate: " << model_.hparams.sample_rate << " Hz\n"
       << "  Output Classes: " << model_.hparams.num_classes << "\n"
       << "  LSTM Layers: " << model_.hparams.lstm_layers << "\n"
       << "  LSTM Hidden: " << model_.hparams.lstm_hidden << "\n"
       << "  SincNet Kernel: " << model_.hparams.sincnet_kernel;
    
    return ss.str();
}

} // namespace segmentation
