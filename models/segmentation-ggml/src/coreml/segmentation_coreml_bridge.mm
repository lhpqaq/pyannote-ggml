#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC (-fobjc-arc)"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "segmentation_coreml_bridge.h"

#include <cstring>
#include <cstdio>

// Lightweight MLFeatureProvider that avoids NSDictionary + MLDictionaryFeatureProvider overhead
@interface SegCoreMLFeatureProvider : NSObject <MLFeatureProvider>
@property (nonatomic, strong) NSSet<NSString *> *featureNames;
@property (nonatomic, strong) MLMultiArray *waveform;
@end

@implementation SegCoreMLFeatureProvider
- (nullable MLFeatureValue *)featureValueForName:(NSString *)name {
    if ([name isEqualToString:@"waveform"]) {
        return [MLFeatureValue featureValueWithMultiArray:_waveform];
    }
    return nil;
}
@end

struct segmentation_coreml_context {
    const void * model;
    const void * cached_shape;      // NSArray<NSNumber*>* [1, 1, 160000]
    const void * cached_strides;    // NSArray<NSNumber*>* [160000, 160000, 1]
    const void * cached_feat_names; // NSSet<NSString*>*
    const void * cached_pred_opts;  // MLPredictionOptions* (with outputBackings on macOS 14+)
    const void * cached_out_array;  // MLMultiArray* pre-allocated output [1, 589, 7]
    bool use_output_backings;
};

static MLComputeUnits segmentation_coreml_compute_units(void) {
    const char * env = getenv("COREML_COMPUTE_UNITS");
    if (env) {
        if (strcmp(env, "cpu_only") == 0)  return MLComputeUnitsCPUOnly;
        if (strcmp(env, "cpu_gpu") == 0)   return MLComputeUnitsCPUAndGPU;
        if (strcmp(env, "cpu_ane") == 0)   return MLComputeUnitsCPUAndNeuralEngine;
    }
    return MLComputeUnitsAll;
}

struct segmentation_coreml_context * segmentation_coreml_init(const char * path_model) {
    NSString * path_str = [[NSString alloc] initWithUTF8String:path_model];
    NSURL * url = [NSURL fileURLWithPath:path_str];

    NSError * error = nil;
    NSURL * compiledURL = nil;

    // Try loading a cached compiled model (.mlmodelc) next to the .mlpackage
    NSString * cachedPath = [[path_str stringByDeletingPathExtension]
                             stringByAppendingPathExtension:@"mlmodelc"];
    NSURL * cachedURL = [NSURL fileURLWithPath:cachedPath];
    NSFileManager * fm = [NSFileManager defaultManager];

    if ([fm fileExistsAtPath:cachedPath]) {
        compiledURL = cachedURL;
        fprintf(stderr, "[CoreML-Seg] Using cached compiled model: %s\n", [cachedPath UTF8String]);
    } else {
        compiledURL = [MLModel compileModelAtURL:url error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Failed to compile model: %s\n",
                    [[error localizedDescription] UTF8String]);
            return nullptr;
        }
        // Cache the compiled model for next time
        [fm copyItemAtURL:compiledURL toURL:cachedURL error:&error];
        if (error == nil) {
            fprintf(stderr, "[CoreML-Seg] Cached compiled model to: %s\n", [cachedPath UTF8String]);
        } else {
            fprintf(stderr, "[CoreML-Seg] Note: could not cache compiled model: %s\n",
                    [[error localizedDescription] UTF8String]);
            error = nil;
        }
    }

    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = segmentation_coreml_compute_units();

    MLModel * model = [MLModel modelWithContentsOfURL:compiledURL
                                        configuration:config
                                                error:&error];
    if (error != nil) {
        fprintf(stderr, "[CoreML-Seg] Failed to load model: %s\n",
                [[error localizedDescription] UTF8String]);
        return nullptr;
    }

    auto * ctx = new segmentation_coreml_context;
    ctx->model = CFBridgingRetain(model);

    // Cache fixed input shape/strides (always [1, 1, 160000])
    static constexpr int N_SAMPLES = 160000;
    NSArray<NSNumber *> * shape = @[@1, @1, @(N_SAMPLES)];
    NSArray<NSNumber *> * strides = @[@(N_SAMPLES), @(N_SAMPLES), @1];
    ctx->cached_shape   = CFBridgingRetain(shape);
    ctx->cached_strides = CFBridgingRetain(strides);

    NSSet<NSString *> * feat_names = [NSSet setWithObject:@"waveform"];
    ctx->cached_feat_names = CFBridgingRetain(feat_names);

    // Pre-allocate output buffer and prediction options
    ctx->use_output_backings = false;
    ctx->cached_pred_opts  = nullptr;
    ctx->cached_out_array  = nullptr;

    if (@available(macOS 14.0, iOS 17.0, *)) {
        NSError * out_err = nil;
        MLMultiArray * out_array = [[MLMultiArray alloc] initWithShape:@[@1, @589, @7]
                                                             dataType:MLMultiArrayDataTypeFloat32
                                                                error:&out_err];
        if (out_array && !out_err) {
            MLPredictionOptions * opts = [[MLPredictionOptions alloc] init];
            opts.outputBackings = @{ @"log_probabilities": out_array };
            ctx->cached_pred_opts  = CFBridgingRetain(opts);
            ctx->cached_out_array  = CFBridgingRetain(out_array);
            ctx->use_output_backings = true;
        }
    }

    return ctx;
}

void segmentation_coreml_free(struct segmentation_coreml_context * ctx) {
    if (ctx) {
        CFRelease(ctx->model);
        if (ctx->cached_shape)      CFRelease(ctx->cached_shape);
        if (ctx->cached_strides)    CFRelease(ctx->cached_strides);
        if (ctx->cached_feat_names) CFRelease(ctx->cached_feat_names);
        if (ctx->cached_pred_opts)  CFRelease(ctx->cached_pred_opts);
        if (ctx->cached_out_array)  CFRelease(ctx->cached_out_array);
        delete ctx;
    }
}

void segmentation_coreml_infer(
    const struct segmentation_coreml_context * ctx,
    float * audio_data,
    int32_t n_samples,
    float * output,
    int32_t output_size) {

    @autoreleasepool {
        MLModel * model = (__bridge MLModel *)ctx->model;

        NSError * error = nil;

        // Use cached shape/strides for the input MLMultiArray
        NSArray<NSNumber *> * shape   = (__bridge NSArray<NSNumber *> *)ctx->cached_shape;
        NSArray<NSNumber *> * strides = (__bridge NSArray<NSNumber *> *)ctx->cached_strides;

        MLMultiArray * input = [[MLMultiArray alloc]
            initWithDataPointer:audio_data
                          shape:shape
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:strides
                    deallocator:^(void * _Nonnull bytes) { }
                          error:&error];

        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Failed to create input array: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        // Use lightweight custom FeatureProvider
        SegCoreMLFeatureProvider * provider = [[SegCoreMLFeatureProvider alloc] init];
        provider.featureNames = (__bridge NSSet<NSString *> *)ctx->cached_feat_names;
        provider.waveform = input;

        id<MLFeatureProvider> result = nil;

        if (ctx->use_output_backings && ctx->cached_pred_opts) {
            MLPredictionOptions * opts = (__bridge MLPredictionOptions *)ctx->cached_pred_opts;
            result = [model predictionFromFeatures:provider options:opts error:&error];
        } else {
            result = [model predictionFromFeatures:provider error:&error];
        }

        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Inference failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        // If outputBackings was used, the pre-allocated output buffer is already filled
        if (ctx->use_output_backings && ctx->cached_out_array) {
            MLMultiArray * out_backing = (__bridge MLMultiArray *)ctx->cached_out_array;
            const float * src = (const float *)out_backing.dataPointer;
            memcpy(output, src, output_size * sizeof(float));
            return;
        }

        // Fallback: read from returned output with stride-aware copy
        MLMultiArray * out_array = [[result featureValueForName:@"log_probabilities"] multiArrayValue];
        if (out_array == nil) {
            fprintf(stderr, "[CoreML-Seg] Output 'log_probabilities' not found\n");
            return;
        }

        NSArray<NSNumber *> * out_shape   = out_array.shape;
        NSArray<NSNumber *> * out_strides = out_array.strides;
        const int ndim = (int)out_shape.count;

        bool contiguous = true;
        if (ndim >= 2) {
            int64_t expected_stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (out_strides[d].longLongValue != expected_stride) {
                    contiguous = false;
                    break;
                }
                expected_stride *= out_shape[d].longLongValue;
            }
        }

        const int total = output_size;

        if (contiguous) {
            if (out_array.dataType == MLMultiArrayDataTypeFloat16) {
                const uint16_t * fp16 = (const uint16_t *)out_array.dataPointer;
                for (int i = 0; i < total; i++) {
                    uint16_t h = fp16[i];
                    uint32_t sign = (h >> 15) & 0x1;
                    uint32_t exp  = (h >> 10) & 0x1f;
                    uint32_t mant = h & 0x3ff;
                    uint32_t f;
                    if (exp == 0) {
                        if (mant == 0) { f = sign << 31; }
                        else {
                            exp = 1;
                            while (!(mant & 0x400)) { mant <<= 1; exp--; }
                            mant &= 0x3ff;
                            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                        }
                    } else if (exp == 31) {
                        f = (sign << 31) | 0x7f800000 | (mant << 13);
                    } else {
                        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                    memcpy(&output[i], &f, sizeof(float));
                }
            } else {
                const float * src = (const float *)out_array.dataPointer;
                memcpy(output, src, total * sizeof(float));
            }
        } else {
            const int frames  = (ndim >= 2) ? (int)out_shape[ndim - 2].integerValue : 1;
            const int classes = (int)out_shape[ndim - 1].integerValue;
            const int64_t s_frame = (ndim >= 2) ? out_strides[ndim - 2].longLongValue : 0;
            const int64_t s_class = out_strides[ndim - 1].longLongValue;
            int64_t batch_off = 0;

            fprintf(stderr, "[CoreML-Seg] Non-contiguous output detected, using stride-aware copy\n");

            if (out_array.dataType == MLMultiArrayDataTypeFloat16) {
                const uint16_t * fp16 = (const uint16_t *)out_array.dataPointer;
                int out_idx = 0;
                for (int fr = 0; fr < frames && out_idx < total; fr++) {
                    for (int cl = 0; cl < classes && out_idx < total; cl++) {
                        int64_t src_idx = batch_off + fr * s_frame + cl * s_class;
                        uint16_t h = fp16[src_idx];
                        uint32_t sign = (h >> 15) & 0x1;
                        uint32_t exp  = (h >> 10) & 0x1f;
                        uint32_t mant = h & 0x3ff;
                        uint32_t f;
                        if (exp == 0) {
                            if (mant == 0) { f = sign << 31; }
                            else {
                                exp = 1;
                                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                                mant &= 0x3ff;
                                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                            }
                        } else if (exp == 31) {
                            f = (sign << 31) | 0x7f800000 | (mant << 13);
                        } else {
                            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                        }
                        memcpy(&output[out_idx++], &f, sizeof(float));
                    }
                }
            } else {
                const float * src = (const float *)out_array.dataPointer;
                int out_idx = 0;
                for (int fr = 0; fr < frames && out_idx < total; fr++) {
                    for (int cl = 0; cl < classes && out_idx < total; cl++) {
                        int64_t src_idx = batch_off + fr * s_frame + cl * s_class;
                        output[out_idx++] = src[src_idx];
                    }
                }
            }
        }
    }
}

void segmentation_coreml_infer_batch(
    const struct segmentation_coreml_context * ctx,
    float * audio_batch,
    int32_t batch_size,
    int32_t n_samples,
    float * output_batch,
    int32_t output_per_sample) {

    if (batch_size <= 0) return;

    if (batch_size == 1) {
        segmentation_coreml_infer(ctx, audio_batch, n_samples, output_batch, output_per_sample);
        return;
    }

    @autoreleasepool {
        MLModel * model = (__bridge MLModel *)ctx->model;
        NSError * error = nil;

        NSArray<NSNumber *> * shape = @[@(batch_size), @1, @(n_samples)];
        NSArray<NSNumber *> * strides = @[@(n_samples), @(n_samples), @1];

        MLMultiArray * input = [[MLMultiArray alloc]
            initWithDataPointer:audio_batch
                          shape:shape
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:strides
                    deallocator:^(void * _Nonnull bytes) { }
                          error:&error];

        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Batch: failed to create input: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        SegCoreMLFeatureProvider * provider = [[SegCoreMLFeatureProvider alloc] init];
        provider.featureNames = (__bridge NSSet<NSString *> *)ctx->cached_feat_names;
        provider.waveform = input;

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Batch inference failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        MLMultiArray * out_array = [[result featureValueForName:@"log_probabilities"] multiArrayValue];
        if (out_array == nil) {
            fprintf(stderr, "[CoreML-Seg] Batch: output not found\n");
            return;
        }

        const int total = batch_size * output_per_sample;
        if (out_array.dataType == MLMultiArrayDataTypeFloat32) {
            const float * src = (const float *)out_array.dataPointer;
            memcpy(output_batch, src, total * sizeof(float));
        } else {
            const uint16_t * fp16 = (const uint16_t *)out_array.dataPointer;
            for (int i = 0; i < total; i++) {
                uint16_t h = fp16[i];
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp  = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) { f = sign << 31; }
                    else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
                           mant &= 0x3ff; f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); }
                } else if (exp == 31) { f = (sign << 31) | 0x7f800000 | (mant << 13);
                } else { f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); }
                memcpy(&output_batch[i], &f, sizeof(float));
            }
        }
    }
}
