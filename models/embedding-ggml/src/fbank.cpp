#include "fbank.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include <cstring>

namespace embedding {

fbank_result compute_fbank(const float* audio, int num_samples, int sample_rate, bool apply_cmn_flag) {
    constexpr int NUM_BINS = 80;

    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = static_cast<float>(sample_rate);
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.snip_edges = true;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.preemph_coeff = 0.97f;
    opts.mel_opts.num_bins = NUM_BINS;
    opts.use_energy = false;
    opts.use_log_fbank = true;
    opts.use_power = true;

    // Scale waveform by 32768 (WeSpeaker convention: float [-1,1] -> int16 range)
    std::vector<float> scaled(num_samples);
    for (int i = 0; i < num_samples; i++) {
        scaled[i] = audio[i] * 32768.0f;
    }

    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(static_cast<float>(sample_rate), scaled.data(), num_samples);
    fbank.InputFinished();

    int T = fbank.NumFramesReady();

    fbank_result result;
    result.num_frames = T;
    result.num_bins = NUM_BINS;
    result.data.resize(T * NUM_BINS);

    for (int t = 0; t < T; t++) {
        const float* frame = fbank.GetFrame(t);
        std::memcpy(result.data.data() + t * NUM_BINS, frame, NUM_BINS * sizeof(float));
    }

    if (apply_cmn_flag) {
        apply_cmn(result.data.data(), T, NUM_BINS);
    }

    return result;
}

void apply_cmn(float* data, int num_frames, int num_bins) {
    std::vector<float> mean(num_bins, 0.0f);
    for (int t = 0; t < num_frames; t++) {
        for (int b = 0; b < num_bins; b++) {
            mean[b] += data[t * num_bins + b];
        }
    }
    const float inv_t = 1.0f / static_cast<float>(num_frames);
    for (int b = 0; b < num_bins; b++) {
        mean[b] *= inv_t;
    }
    for (int t = 0; t < num_frames; t++) {
        for (int b = 0; b < num_bins; b++) {
            data[t * num_bins + b] -= mean[b];
        }
    }
}

} // namespace embedding
