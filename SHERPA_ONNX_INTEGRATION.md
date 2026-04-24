# StyleTTS2 Lite 集成到 sherpa-onnx 方案

## 概述

本文档描述如何将 StyleTTS2 Lite 模型集成到 sherpa-onnx 框架中，使其能在 iOS/Android 设备上运行。

## 架构对比

### StyleTTS2 Lite vs Kokoro (已集成到 sherpa-onnx)

| 特性 | StyleTTS2 Lite | Kokoro |
|------|---------------|--------|
| 模型大小 | ~312 MB | ~330 MB |
| 参数量 | 82M | 82M |
| 音素化 | espeak-ng | espeak-ng |
| 采样率 | 24kHz | 24kHz |
| Voice Pack | ✅ | ✅ |
| 模块数 | 5 | 5 |

由于架构非常相似，我们可以**复用 Kokoro 的集成模式**。

## 导出 ONNX 模型

### 步骤 1: 运行导出脚本

```bash
cd /Users/naiwenxie/TTS/StyleTTS2

python export_onnx.py \
    --checkpoint Models/LJSpeech/styletts2_lite.pth \
    --config Configs/config.yml \
    --voicepack batch_output/LJ005-0254/voicepack.pt \
    --output_dir onnx_models/
```

### 导出的文件

```
onnx_models/
├── bert_encoder.onnx      # ~25 MB (PL-BERT + projection)
├── text_encoder.onnx      # ~21 MB (Text encoding)
├── predictor.onnx         # ~62 MB (Duration prediction)
├── f0_predictor.onnx      # ~1 MB  (F0/Energy prediction)
├── decoder.onnx           # ~203 MB (iSTFTNet waveform)
├── voicepack.npy          # ~100 KB (Voice style)
└── tokens.txt             # 音素到 ID 映射
```

## sherpa-onnx 集成方案

### 方案 A: 作为新的 TTS 后端 (推荐)

在 sherpa-onnx 中添加 `styletts2` 作为新的 TTS 模型类型：

#### 1. 修改 C++ 头文件

```cpp
// sherpa-onnx/csrc/offline-tts-model-config.h

struct OfflineTtsStyleTTS2Config {
  std::string bert_encoder;      // bert_encoder.onnx
  std::string text_encoder;      // text_encoder.onnx
  std::string predictor;         // predictor.onnx
  std::string f0_predictor;      // f0_predictor.onnx
  std::string decoder;           // decoder.onnx
  std::string voicepack;         // voicepack.npy
  std::string tokens;            // tokens.txt
  std::string data_dir;          // espeak-ng-data 目录
  std::string language = "en-us";
  int32_t sample_rate = 24000;
};
```

#### 2. 创建 StyleTTS2 实现

```cpp
// sherpa-onnx/csrc/offline-tts-styletts2-impl.h

class OfflineTtsStyleTTS2Impl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsStyleTTS2Impl(const OfflineTtsModelConfig &config);
  
  GeneratedAudio Generate(
      const std::string &text,
      int32_t sid = 0,
      float speed = 1.0f) const override;

 private:
  // 5 ONNX sessions
  std::unique_ptr<Ort::Session> bert_encoder_;
  std::unique_ptr<Ort::Session> text_encoder_;
  std::unique_ptr<Ort::Session> predictor_;
  std::unique_ptr<Ort::Session> f0_predictor_;
  std::unique_ptr<Ort::Session> decoder_;
  
  // Voice pack
  std::vector<float> voicepack_;
  int32_t voicepack_size_;
  
  // Phonemizer (reuse from piper)
  std::unique_ptr<PiperPhonemizeLexicon> lexicon_;
};
```

#### 3. 推理流程

```cpp
GeneratedAudio OfflineTtsStyleTTS2Impl::Generate(
    const std::string &text, int32_t sid, float speed) const {
  
  // 1. Phonemize text using espeak-ng
  auto phonemes = lexicon_->ConvertTextToTokenIds(text);
  
  // 2. Get voice pack style
  int pack_idx = std::min((int)phonemes.size(), voicepack_size_ - 1);
  auto ref_s = GetVoicePackSlice(pack_idx);  // [1, 256]
  auto style_ref = ref_s.slice(0, 128);       // acoustic
  auto style_s = ref_s.slice(128, 256);       // prosody
  
  // 3. BERT encoding
  auto d_en = RunBertEncoder(phonemes);       // [1, hidden, T]
  
  // 4. Duration prediction
  auto [d, duration] = RunPredictor(d_en, style_s);
  duration = duration / speed;
  auto pred_dur = Round(duration).clamp_min(1);
  
  // 5. Alignment (expand d to mel length)
  auto pred_aln = ComputeAlignment(pred_dur);
  auto en = MatMul(d.transpose(), pred_aln);
  
  // 6. F0/Energy prediction
  auto [F0, N] = RunF0Predictor(en, style_s);
  
  // 7. Text encoding + alignment
  auto t_en = RunTextEncoder(phonemes);
  auto asr = MatMul(t_en, pred_aln);
  
  // 8. Decode to waveform
  auto audio = RunDecoder(asr, F0, N, style_ref);
  
  return {audio, 24000};
}
```

### 方案 B: 合并为单一 ONNX 模型

将所有模块合并为一个 `styletts2.onnx`，简化部署：

```python
# export_single_onnx.py

class StyleTTS2Combined(nn.Module):
    """合并所有模块为单一网络"""
    def __init__(self, bert, bert_encoder, text_encoder, predictor, decoder):
        super().__init__()
        self.bert = bert
        self.bert_encoder = bert_encoder
        self.text_encoder = text_encoder
        self.predictor = predictor
        self.decoder = decoder
    
    def forward(self, tokens, ref_s, speed=1.0):
        # 完整推理流程
        ...
        return audio

# 导出为单一文件
torch.onnx.export(combined_model, ..., "styletts2.onnx")
```

优点：部署简单，只需要一个 ONNX 文件
缺点：无法单独优化各模块，内存占用更大

## iOS Swift 接口示例

```swift
// StyleTTS2Wrapper.swift

import SherpaOnnx

class StyleTTS2TTS {
    private var tts: SherpaOnnxOfflineTts
    
    init(modelDir: String) {
        var config = sherpaOnnxOfflineTtsModelConfig()
        
        // StyleTTS2 specific config
        config.styletts2.bertEncoder = "\(modelDir)/bert_encoder.onnx"
        config.styletts2.textEncoder = "\(modelDir)/text_encoder.onnx"
        config.styletts2.predictor = "\(modelDir)/predictor.onnx"
        config.styletts2.f0Predictor = "\(modelDir)/f0_predictor.onnx"
        config.styletts2.decoder = "\(modelDir)/decoder.onnx"
        config.styletts2.voicepack = "\(modelDir)/voicepack.npy"
        config.styletts2.tokens = "\(modelDir)/tokens.txt"
        config.styletts2.dataDir = "\(modelDir)/espeak-ng-data"
        
        var ttsConfig = sherpaOnnxOfflineTtsConfig()
        ttsConfig.model = config
        
        tts = SherpaOnnxOfflineTts(config: &ttsConfig)
    }
    
    func synthesize(text: String, speed: Float = 1.0) -> [Float] {
        let audio = tts.generate(text: text, speed: speed)
        return audio.samples
    }
}
```

## Android Kotlin 接口示例

```kotlin
// StyleTTS2TTS.kt

class StyleTTS2TTS(context: Context, modelDir: String) {
    private val tts: OfflineTts
    
    init {
        val config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                styletts2 = OfflineTtsStyleTTS2Config(
                    bertEncoder = "$modelDir/bert_encoder.onnx",
                    textEncoder = "$modelDir/text_encoder.onnx",
                    predictor = "$modelDir/predictor.onnx",
                    f0Predictor = "$modelDir/f0_predictor.onnx",
                    decoder = "$modelDir/decoder.onnx",
                    voicepack = "$modelDir/voicepack.npy",
                    tokens = "$modelDir/tokens.txt",
                    dataDir = "$modelDir/espeak-ng-data"
                )
            )
        )
        tts = OfflineTts(assetManager = context.assets, config = config)
    }
    
    fun synthesize(text: String, speed: Float = 1.0f): FloatArray {
        return tts.generate(text = text, speed = speed).samples
    }
}
```

## 集成步骤总结

1. **导出 ONNX** (本地完成)
   ```bash
   python export_onnx.py --checkpoint ... --output_dir onnx_models/
   ```

2. **准备 espeak-ng-data** (复用 sherpa-onnx 已有的)
   - 从任意 piper 模型中复制 `espeak-ng-data` 目录

3. **添加到 sherpa-onnx 源码**
   - 新增 `offline-tts-styletts2-impl.h/cc`
   - 修改 `offline-tts-model-config.h`
   - 更新 CMakeLists.txt

4. **编译测试**
   ```bash
   cd sherpa-onnx
   mkdir build && cd build
   cmake .. && make -j
   ./bin/sherpa-onnx-offline-tts --styletts2-model=... --text="Hello"
   ```

5. **移动端集成**
   - iOS: 重新运行 `build-ios.sh`
   - Android: 重新运行 `build-android.sh`

## 参考

- [Kokoro 集成 PR](https://github.com/k2-fsa/sherpa-onnx/pull/XXX)
- [VITS 实现](sherpa-onnx/csrc/offline-tts-vits-impl.h)
- [Matcha 实现](sherpa-onnx/csrc/offline-tts-matcha-impl.h)

## 预估工作量

| 任务 | 时间 |
|------|------|
| ONNX 导出调试 | 1-2 天 |
| sherpa-onnx C++ 集成 | 2-3 天 |
| iOS/Android 测试 | 1-2 天 |
| 性能优化 | 1-2 天 |
| **总计** | **5-9 天** |
