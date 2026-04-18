# StyleTTS2 Inference 指南

本文档描述了 StyleTTS2 项目中所有推理（inference）相关脚本的功能、生成 `.pt` 文件的方法以及完整的推理流程。

---

## 目录

1. [推理流程总览](#推理流程总览)
2. [生成 .pt 文件（Voice Pack / Style Cache）](#生成-pt-文件)
3. [推理脚本详解](#推理脚本详解)
4. [batch_output 目录说明](#batch_output-目录说明)
5. [快速上手](#快速上手)

---

## 推理流程总览

StyleTTS2 提供 **4 种推理方式**，从完整模型到轻量级部署逐步精简：

```
                        ┌──────────────────────────────────────────────┐
                        │         StyleTTS2 推理方式对比                │
                        ├──────────────┬───────────┬──────────┬────────┤
                        │ 方式         │ 风格来源  │ 需Diffusion│ 速度  │
                        ├──────────────┼───────────┼──────────┼────────┤
                        │ inference.py │ Diffusion │ ✅ 需要   │ 较慢  │
                        │ inference_   │ 参考音频  │ ❌ 不需要 │ 中等  │
                        │   ref.py     │ (Style    │          │       │
                        │              │  Encoder) │          │       │
                        │ inference_   │ Voice Pack│ ❌ 不需要 │ 快    │
                        │  voicepack.py│ (.pt文件) │          │       │
                        │ styletts2_   │ Voice Pack│ ❌ 不需要 │ 最快  │
                        │   lite.py    │ (.pt文件) │ (精简模型)│       │
                        └──────────────┴───────────┴──────────┴────────┘
```

### 核心推理管线

无论使用哪种方式，核心推理管线都遵循以下流程：

```
输入文本 "Hello world"
    │
    ▼
┌─────────────────────┐
│  Phonemizer (eSpeak) │  文本 → 音素序列
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  TextCleaner         │  音素 → token IDs
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  PL-BERT (bert)      │  token → 上下文文本特征
│  + bert_encoder      │  
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  风格向量获取         │  获取 ref_s [1, 256]
│  (4种不同方式)       │  ├── [:, :128] 声学风格 (音色)
│                      │  └── [:, 128:] 韵律风格 (语速/语调)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Predictor           │  预测时长 (duration)、
│                      │  基频 (F0)、能量 (energy)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Decoder (iSTFTNet)  │  生成 24kHz 波形
└─────────┬───────────┘
          ▼
    输出 WAV 音频文件
```

---

## 生成 .pt 文件

项目中有 **3 种 .pt 文件**：

### 1. Voice Pack（语音风格包）

Voice Pack 是预计算的风格向量集合，形状为 `[max_len, 1, 256]`，按音素长度索引。推理时直接查表获取风格，**无需 Diffusion 或 Style Encoder**。

#### 方法 A：从参考音频生成（Style Encoder 方式）

```bash
# 从单个参考音频
python make_voicepack.py \
    --ref_audio LJSpeech-1.1/wavs/LJ002-0254.wav \
    --output voices/lj_single.pt

# 从语料库（按音素长度分组平均）
python make_voicepack.py \
    --wavs_dir LJSpeech-1.1/wavs \
    --metadata LJSpeech-1.1/metadata.csv \
    --output voices/lj_corpus.pt \
    --max_files 500
```

**流程：**
1. 加载参考音频 → 转为 Mel 频谱
2. 通过 `style_encoder` 提取声学风格向量 [1, 128]
3. 通过 `predictor_encoder` 提取韵律风格向量 [1, 128]
4. 拼接为 [1, 256]，扩展到 [510, 1, 256]
5. 保存为 `.pt` 文件

#### 方法 B：通过 Diffusion 生成

```bash
python make_voicepack_diffusion.py \
    --output voices/lj_diffusion.pt \
    --diffusion_steps 5 \
    --embedding_scale 1.0 \
    --num_seeds 5
```

**流程：**
1. 用多段示例文本通过 Diffusion 模型采样多个风格向量
2. 按音素长度分组取平均
3. 对空缺长度进行线性插值
4. 保存为 `.pt` 文件 [510, 1, 256]

### 2. Style Cache（风格缓存）

Diffusion 推理时的缓存文件，用于避免重复计算。

```bash
python inference_cached.py \
    --text "Hello world" \
    --output out.wav \
    --cache_dir style_cache/
```

**缓存命名规则：** `{文本前30字符}_{MD5哈希}.pt`

缓存内容为 Diffusion 生成的风格预测张量 `s_pred [1, 256]`。首次运行会执行 Diffusion 并保存缓存，后续相同文本/参数直接加载。

### 3. Lite Checkpoint（精简模型权重）

从完整 checkpoint (2GB) 中提取 5 个核心推理模块：

```bash
python export_lite.py \
    --input Models/LJSpeech/epoch_2nd_00078.pth \
    --output Models/LJSpeech/styletts2_lite.pth
```

**提取的模块：**

| 模块 | 参数量 | 大小 | 作用 |
|------|--------|------|------|
| `bert` | 6.3M | 24 MB | PL-BERT 文本理解 |
| `bert_encoder` | 0.4M | 1.5 MB | BERT 输出投影 |
| `predictor` | 16.2M | 62 MB | 时长 / F0 / 能量预测 |
| `decoder` | 53.3M | 203 MB | iSTFTNet 波形生成 |
| `text_encoder` | 5.6M | 21 MB | 文本编码 |
| **合计** | **81.8M** | **312 MB** | 较完整模型减少 84% |

---

## 推理脚本详解

### `inference.py` — Diffusion 全模型推理

**适用场景：** 最高质量推理，支持长文本模式

```bash
# 单句合成
python inference.py \
    --text "Hello world" \
    --output output.wav \
    --diffusion_steps 5 \
    --embedding_scale 1.0

# 长文本合成（保持风格连贯）
python inference.py \
    --text "First sentence. Second sentence. Third sentence." \
    --output output.wav \
    --long_form \
    --alpha 0.7

# 带调试输出
python inference.py \
    --text "Hello world" \
    --output output.wav \
    --debug

# 生成 Mel 频谱图
python inference.py \
    --text "Hello world" \
    --output output.wav \
    --mel --save_mel_npy
```

**风格获取方式：** Diffusion 模型从高斯噪声出发，以 BERT 特征为条件，去噪生成 256 维风格向量。

**关键参数：**
- `--diffusion_steps`：去噪步数（5=快速，10=更多样化）
- `--embedding_scale`：CFG 引导强度（越高情感越强，建议 1~2）
- `--alpha`：长文本模式中前后句风格混合比例

---

### `inference_ref.py` — 参考音频推理

**适用场景：** 需要克隆特定说话人音色

```bash
python inference_ref.py \
    --text "Hello world" \
    --ref_audio LJSpeech-1.1/wavs/LJ001-0001.wav \
    --output output_ref.wav
```

**风格获取方式：** 
- 参考音频 → Mel 频谱 → `style_encoder` → 声学风格 [1, 128]
- 参考音频 → Mel 频谱 → `predictor_encoder` → 韵律风格 [1, 128]
- 无需 Diffusion，速度更快

---

### `inference_voicepack.py` — Voice Pack 推理

**适用场景：** 使用预制风格包的快速推理

```bash
python inference_voicepack.py \
    --text "Hello world" \
    --voicepack voices/lj.pt \
    --output output_vp.wav
```

**风格获取方式：** 从 Voice Pack 中按音素长度索引直接查表，不需要 Diffusion 也不需要 Style Encoder。

---

### `inference_cached.py` — 带缓存的 Diffusion 推理

**适用场景：** 反复合成相同文本的场景

```bash
# 首次运行（执行 Diffusion + 缓存）
python inference_cached.py \
    --text "Hello world" \
    --output out.wav \
    --cache_dir style_cache/

# 再次运行相同文本（直接读缓存，跳过 Diffusion）
python inference_cached.py \
    --text "Hello world" \
    --output out.wav \
    --cache_dir style_cache/
```

---

### `styletts2_lite.py` — 轻量级推理

**适用场景：** 部署环境，最快速度

```bash
python styletts2_lite.py \
    --text "Hello world" \
    --voicepack voices/lj.pt \
    --checkpoint Models/LJSpeech/styletts2_lite.pth \
    --speed 1.0 \
    --output output_lite.wav
```

**特点：** 
- 使用精简后的 312MB checkpoint（仅含 5 个核心模块）
- 支持语速控制（`--speed` 参数）
- 平均 RTF ≈ 0.05（GPU）

---

## batch_output 目录说明

`batch_output/` 是通过 `batch_voicepack.py` 和 `batch_lite.py` 批量生成的结果目录。

### 目录结构

```
batch_output/
├── report.txt              # 汇总报告（各样本的 RTF、风格向量 norm 等）
├── LJ005-0254/
│   ├── reference.wav       # 参考音频（从 LJSpeech 中选取）
│   ├── voicepack.pt        # 生成的 Voice Pack [510, 1, 256]
│   ├── tts_output.wav      # Voice Pack 推理生成的语音
│   └── tts_lite.wav        # Lite 模式推理生成的语音
├── LJ012-0037/
│   ├── reference.wav
│   ├── voicepack.pt
│   ├── tts_output.wav
│   └── tts_lite.wav
├── LJ020-0029/
│   └── ...
├── LJ024-0066/
│   └── ...
└── LJ026-0118/
    └── ...
```

### 生成流程

```bash
# 步骤 1：批量生成 Voice Pack + TTS 输出
python batch_voicepack.py \
    --wavs_dir LJSpeech-1.1/wavs \
    --num_refs 5 \
    --text "Hello world, this is a speech synthesis test using StyleTTS2."

# 步骤 2：用 Lite 模型对所有 Voice Pack 再做一次推理
python batch_lite.py
```

**batch_voicepack.py 处理流程：**
1. 从 LJSpeech 语料库中按章节选择 N 个参考音频（3-8秒，信噪比良好）
2. 对每个参考音频：
   - 复制 `reference.wav` 到子目录
   - 通过 Style Encoder 提取风格 → 生成 `voicepack.pt`
   - 使用风格向量推理 → 生成 `tts_output.wav`
3. 输出汇总 `report.txt`

**report.txt 示例：**

```
Text: "Hello world, this is a speech synthesis test using StyleTTS2."

Name                 Ref(s)   TTS(s)   RTF      Acoustic   Prosody   
----------------------------------------------------------------------
LJ026-0118           4.3      5.50     0.1588   2.4496     3.5566    
LJ024-0066           6.0      6.42     0.0525   2.3857     2.6287    
LJ020-0029           4.0      4.58     0.0176   2.4240     3.5562    
LJ012-0037           4.9      5.35     0.0150   2.4228     2.6799    
LJ005-0254           7.0      5.20     0.0153   2.3965     3.2181    
```

---

## 快速上手

### 环境准备

```bash
pip install -r requirements.txt
```

### 最简推理（3 步）

```bash
# 1. 导出轻量模型
python export_lite.py \
    --input Models/LJSpeech/epoch_2nd_00078.pth \
    --output Models/LJSpeech/styletts2_lite.pth

# 2. 生成 Voice Pack
python make_voicepack.py \
    --ref_audio LJSpeech-1.1/wavs/LJ002-0254.wav \
    --output voices/lj.pt

# 3. 运行推理
python styletts2_lite.py \
    --text "Hello world, this is StyleTTS2 Lite inference." \
    --voicepack voices/lj.pt \
    --checkpoint Models/LJSpeech/styletts2_lite.pth \
    --output output.wav
```

### 全模型推理

```bash
python inference.py \
    --text "Hello world" \
    --output output.wav \
    --checkpoint Models/LJSpeech/epoch_2nd_00078.pth \
    --diffusion_steps 5
```

---

## 文件清单

| 文件 | 类型 | 描述 |
|------|------|------|
| `inference.py` | 推理 | Diffusion 全模型推理（单句 + 长文本） |
| `inference_ref.py` | 推理 | 参考音频 Style Encoder 推理 |
| `inference_voicepack.py` | 推理 | 全模型 + Voice Pack 推理 |
| `inference_cached.py` | 推理 | 带 Diffusion 风格缓存的推理 |
| `styletts2_lite.py` | 推理 | 轻量级推理（5 模块 + Voice Pack） |
| `export_lite.py` | 工具 | 导出精简 checkpoint |
| `make_voicepack.py` | 工具 | 从参考音频生成 Voice Pack |
| `make_voicepack_diffusion.py` | 工具 | 通过 Diffusion 生成 Voice Pack |
| `batch_voicepack.py` | 批量 | 批量生成 Voice Pack + TTS |
| `batch_lite.py` | 批量 | 批量 Lite 推理 |
