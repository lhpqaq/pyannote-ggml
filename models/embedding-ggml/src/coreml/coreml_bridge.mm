#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC (-fobjc-arc)"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "coreml_bridge.h"

#include <cstring>
#include <cstdio>

@interface EmbCoreMLFeatureProvider : NSObject <MLFeatureProvider>
@property (nonatomic, strong) NSSet<NSString *> *featureNames;
@property (nonatomic, strong) MLMultiArray *fbankFeatures;
@end

@implementation EmbCoreMLFeatureProvider
- (nullable MLFeatureValue *)featureValueForName:(NSString *)name {
    if ([name isEqualToString:@"fbank_features"]) {
        return [MLFeatureValue featureValueWithMultiArray:_fbankFeatures];
    }
    return nil;
}
@end

struct embedding_coreml_context {
    const void * model;
    const void * cached_feat_names;  // NSSet<NSString*>*
    const void * cached_pred_opts;   // MLPredictionOptions* (macOS 14+)
    const void * cached_out_array;   // MLMultiArray* pre-allocated [1, 256]
    bool use_output_backings;
};

static MLComputeUnits embedding_coreml_compute_units(void) {
    const char * env = getenv("COREML_COMPUTE_UNITS");
    if (env) {
        if (strcmp(env, "cpu_only") == 0)  return MLComputeUnitsCPUOnly;
        if (strcmp(env, "cpu_gpu") == 0)   return MLComputeUnitsCPUAndGPU;
        if (strcmp(env, "cpu_ane") == 0)   return MLComputeUnitsCPUAndNeuralEngine;
    }
    return MLComputeUnitsAll;
}

struct embedding_coreml_context * embedding_coreml_init(const char * path_model) {
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
        fprintf(stderr, "[CoreML-Emb] Using cached compiled model: %s\n", [cachedPath UTF8String]);
    } else {
        compiledURL = [MLModel compileModelAtURL:url error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML-Emb] Failed to compile model: %s\n",
                    [[error localizedDescription] UTF8String]);
            return nullptr;
        }
        // Cache the compiled model for next time
        [fm copyItemAtURL:compiledURL toURL:cachedURL error:&error];
        if (error == nil) {
            fprintf(stderr, "[CoreML-Emb] Cached compiled model to: %s\n", [cachedPath UTF8String]);
        } else {
            fprintf(stderr, "[CoreML-Emb] Note: could not cache compiled model: %s\n",
                    [[error localizedDescription] UTF8String]);
            error = nil;
        }
    }

    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = embedding_coreml_compute_units();

    MLModel * model = [MLModel modelWithContentsOfURL:compiledURL
                                        configuration:config
                                                error:&error];
    if (error != nil) {
        fprintf(stderr, "[CoreML-Emb] Failed to load model: %s\n",
                [[error localizedDescription] UTF8String]);
        return nullptr;
    }

    auto * ctx = new embedding_coreml_context;
    ctx->model = CFBridgingRetain(model);

    NSSet<NSString *> * feat_names = [NSSet setWithObject:@"fbank_features"];
    ctx->cached_feat_names = CFBridgingRetain(feat_names);

    ctx->use_output_backings = false;
    ctx->cached_pred_opts  = nullptr;
    ctx->cached_out_array  = nullptr;

    if (@available(macOS 14.0, iOS 17.0, *)) {
        NSError * out_err = nil;
        MLMultiArray * out_array = [[MLMultiArray alloc] initWithShape:@[@1, @256]
                                                             dataType:MLMultiArrayDataTypeFloat32
                                                                error:&out_err];
        if (out_array && !out_err) {
            MLPredictionOptions * opts = [[MLPredictionOptions alloc] init];
            opts.outputBackings = @{ @"embedding": out_array };
            ctx->cached_pred_opts  = CFBridgingRetain(opts);
            ctx->cached_out_array  = CFBridgingRetain(out_array);
            ctx->use_output_backings = true;
        }
    }

    return ctx;
}

void embedding_coreml_free(struct embedding_coreml_context * ctx) {
    if (ctx) {
        CFRelease(ctx->model);
        if (ctx->cached_feat_names) CFRelease(ctx->cached_feat_names);
        if (ctx->cached_pred_opts)  CFRelease(ctx->cached_pred_opts);
        if (ctx->cached_out_array)  CFRelease(ctx->cached_out_array);
        delete ctx;
    }
}

void embedding_coreml_encode(
    const struct embedding_coreml_context * ctx,
    int64_t num_frames,
    float * fbank_data,
    float * embedding_out) {

    @autoreleasepool {
        MLModel * model = (__bridge MLModel *)ctx->model;

        NSError * error = nil;

        NSArray<NSNumber *> * shape = @[@1, @(num_frames), @80];
        NSArray<NSNumber *> * strides = @[@(num_frames * 80), @80, @1];

        MLMultiArray * input = [[MLMultiArray alloc]
            initWithDataPointer:fbank_data
                          shape:shape
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:strides
                    deallocator:^(void * _Nonnull bytes) { }
                          error:&error];

        if (error != nil) {
            fprintf(stderr, "[CoreML] Failed to create input array: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        EmbCoreMLFeatureProvider * provider = [[EmbCoreMLFeatureProvider alloc] init];
        provider.featureNames = (__bridge NSSet<NSString *> *)ctx->cached_feat_names;
        provider.fbankFeatures = input;

        id<MLFeatureProvider> result = nil;

        if (ctx->use_output_backings && ctx->cached_pred_opts) {
            MLPredictionOptions * opts = (__bridge MLPredictionOptions *)ctx->cached_pred_opts;
            result = [model predictionFromFeatures:provider options:opts error:&error];
        } else {
            result = [model predictionFromFeatures:provider error:&error];
        }

        if (error != nil) {
            fprintf(stderr, "[CoreML] Inference failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        const int emb_dim = 256;

        if (ctx->use_output_backings && ctx->cached_out_array) {
            MLMultiArray * out_backing = (__bridge MLMultiArray *)ctx->cached_out_array;
            const float * src = (const float *)out_backing.dataPointer;
            memcpy(embedding_out, src, emb_dim * sizeof(float));
            return;
        }

        // Fallback: stride-aware output extraction
        MLMultiArray * output = [[result featureValueForName:@"embedding"] multiArrayValue];
        if (output == nil) {
            fprintf(stderr, "[CoreML] Output 'embedding' not found\n");
            return;
        }

        NSArray<NSNumber *> * out_strides = output.strides;
        const int ndim = (int)output.shape.count;
        const int64_t s_emb = out_strides[ndim - 1].longLongValue;
        int64_t batch_off = 0;
        const bool emb_contiguous = (s_emb == 1);

        if (output.dataType == MLMultiArrayDataTypeFloat16) {
            const uint16_t * fp16 = (const uint16_t *)output.dataPointer;
            for (int i = 0; i < emb_dim; i++) {
                int64_t src_idx = batch_off + i * s_emb;
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
                memcpy(&embedding_out[i], &f, sizeof(float));
            }
        } else {
            const float * src = (const float *)output.dataPointer;
            if (emb_contiguous) {
                memcpy(embedding_out, src + batch_off, emb_dim * sizeof(float));
            } else {
                for (int i = 0; i < emb_dim; i++) {
                    embedding_out[i] = src[batch_off + i * s_emb];
                }
            }
        }
    }
}

void embedding_coreml_encode_batch(
    const struct embedding_coreml_context * ctx,
    int64_t num_frames,
    int32_t batch_size,
    float * fbank_batch,
    float * embedding_batch) {

    if (batch_size <= 0) return;

    if (batch_size == 1) {
        embedding_coreml_encode(ctx, num_frames, fbank_batch, embedding_batch);
        return;
    }

    @autoreleasepool {
        MLModel * model = (__bridge MLModel *)ctx->model;
        NSError * error = nil;

        const int64_t bins = 80;
        NSArray<NSNumber *> * shape = @[@(batch_size), @(num_frames), @(bins)];
        NSArray<NSNumber *> * strides = @[@(num_frames * bins), @(bins), @1];

        MLMultiArray * input = [[MLMultiArray alloc]
            initWithDataPointer:fbank_batch
                          shape:shape
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:strides
                    deallocator:^(void * _Nonnull bytes) { }
                          error:&error];

        if (error != nil) {
            fprintf(stderr, "[CoreML-Emb] Batch: failed to create input: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        EmbCoreMLFeatureProvider * provider = [[EmbCoreMLFeatureProvider alloc] init];
        provider.featureNames = (__bridge NSSet<NSString *> *)ctx->cached_feat_names;
        provider.fbankFeatures = input;

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML-Emb] Batch inference failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        MLMultiArray * output = [[result featureValueForName:@"embedding"] multiArrayValue];
        if (output == nil) {
            fprintf(stderr, "[CoreML-Emb] Batch: output not found\n");
            return;
        }

        const int emb_dim = 256;
        const int total = batch_size * emb_dim;
        if (output.dataType == MLMultiArrayDataTypeFloat32) {
            const float * src = (const float *)output.dataPointer;
            memcpy(embedding_batch, src, total * sizeof(float));
        } else {
            const uint16_t * fp16 = (const uint16_t *)output.dataPointer;
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
                memcpy(&embedding_batch[i], &f, sizeof(float));
            }
        }
    }
}
