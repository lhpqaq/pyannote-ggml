# pyannote-ggml 部署与执行指南

本项目是 `pyannote/speaker-diarization-community-1` 管道的 C++ 原生移植版本，支持 GGML 和 CoreML 加速，在 Apple Silicon (M1/M2/M3) 上可实现极高的运行效率。

## 1. 环境准备

### 硬件要求
*   macOS 设备（推荐 Apple Silicon 芯片以获取最佳性能）。

### 软件依赖
*   **CMake**: 3.20 或更高版本。
*   **Xcode Command Line Tools**: 编译 C++ 核心。
*   **Python**: 3.12 或 3.13（注意：Python 3.14 目前与 `coremltools` 存在二进制兼容性问题）。
*   **Git LFS**: 用于拉取项目中的大文件（如测试音频）。

### 网络设置（可选代理）
如果在中国大陆地区访问 Hugging Face，请先设置代理：
```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
```

---

## 2. 编译 C++ 核心组件

进入项目目录并运行以下命令进行编译：

```bash
cd diarization-ggml
# 配置并开启 CoreML 加速支持
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON -DGGML_CCACHE=OFF
# 执行编译
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

---

## 3. 模型转换与准备

### 3.1 设置 Python 虚拟环境
建议使用 Python 3.13 创建环境并安装依赖：

```bash
# 返回根目录
cd ..
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.4" "torchaudio>=2.4" "pyannote.audio" "coremltools>=8.0" "httpx[socks]" "gguf" "scipy"
```

### 3.2 转换 Segmentation (分割) 模型
由于 PyAnnote 模型是受限访问的，请确保您已在 Hugging Face 上接受了 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 的使用协议。

```bash
export HF_TOKEN=您的_HF_TOKEN
cd models/segmentation-ggml
# 转换至 GGUF 格式
python convert.py --model-path ~/.cache/huggingface/hub/models--pyannote--segmentation-3.0/snapshots/e66f3d3b9eb0873085418a7b813d3b369bf160bb/pytorch_model.bin --output segmentation.gguf
# 转换至 CoreML 格式
python convert_coreml.py
```

### 3.3 转换 Embedding (嵌入) 模型
```bash
cd ../embedding-ggml
# 执行转换脚本（脚本会自动从 HF 下载模型并生成 .gguf 和 .mlpackage）
python convert_coreml.py
```

### 3.4 准备 PLDA 模型
```bash
cd ../../diarization-ggml
# 使用脚本转换 PLDA 权重
python convert_plda.py 
  --transform-npz ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/plda/xvec_transform.npz 
  --plda-npz ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/plda/plda.npz 
  -o plda.gguf
```

### 3.5 准备 Whisper 模型
```bash
cd ../whisper.cpp/models
./download-ggml-model.sh base.en
```

---

## 4. 执行与测试

### 4.1 仅执行说话人识别 (Diarization)
该程序将输出 RTTM 格式的文件，显示每个时间段对应的发言人。

```bash
cd ../../diarization-ggml
./build/bin/diarization-ggml ../models/segmentation-ggml/segmentation.gguf ../models/embedding-ggml/embedding.gguf ../samples/sample.wav --plda plda.gguf --coreml ../models/embedding-ggml/embedding.mlpackage --seg-coreml ../models/segmentation-ggml/segmentation.mlpackage -o output.rttm
```

### 4.2 执行集成转录 (Transcription + Diarization)
该程序将结合 Whisper 输出带有说话人标签的文字。

```bash
./build/bin/transcribe ../samples/sample.wav \
  --seg-model ../models/segmentation-ggml/segmentation.gguf \
  --emb-model ../models/embedding-ggml/embedding.gguf \
  --whisper-model ../whisper.cpp/models/ggml-base.en.bin \ 
  --plda plda.gguf \
  --seg-coreml ../models/segmentation-ggml/segmentation.mlpackage \
  --emb-coreml ../models/embedding-ggml/embedding.mlpackage \
  --language en
```

---

## 5. 常见问题排查

1.  **无效的 WAV 格式**: 确保音频文件是 `16kHz, 16-bit, Mono` 的 PCM 格式。可以使用 `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav` 转换。
2.  **CoreML 转换失败**: 如果遇到 `BlobWriter not loaded` 错误，说明您的 Python 版本过新。请务必使用 Python 3.12 或 3.13。
3.  **HF 权限拒绝 (403 Error)**: 请确保您的 `HF_TOKEN` 正确，并且已经在 Hugging Face 网页端手动同意了相关模型的使用协议。
4.  **Git LFS 问题**: 如果 `sample.wav` 文件只有几百字节（文本格式），请运行 `git lfs pull` 来拉取真正的二进制数据。
