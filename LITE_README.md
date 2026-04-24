# StyleTTS2 Lite

A lightweight inference version of [StyleTTS2](https://github.com/yl4579/StyleTTS2), inspired by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

## What is StyleTTS2 Lite?

StyleTTS2 Lite removes the Diffusion, Style Encoder, and Predictor Encoder modules from the full StyleTTS2 model, keeping only **5 core modules** needed for inference with a pre-computed Voice Pack:

| Module | Parameters | Size | Role |
|---|---|---|---|
| `bert` | 6.3M | 24 MB | PL-BERT text understanding |
| `bert_encoder` | 0.4M | 1.5 MB | BERT output projection |
| `predictor` | 16.2M | 62 MB | Duration / F0 / Energy prediction |
| `decoder` | 53.3M | 203 MB | iSTFTNet waveform generation |
| `text_encoder` | 5.6M | 21 MB | Text encoding |
| **Total** | **81.8M** | **312 MB** | — |

### Comparison

| | Full StyleTTS2 | StyleTTS2 Lite | Kokoro-82M |
|---|---|---|---|
| Model size | 2.0 GB | **312 MB** (↓84%) | ~330 MB |
| Parameters | 209M | **81.8M** | 82M |
| Modules | 13 | **5** | 5 |
| Diffusion | ✅ | ❌ | ❌ |
| Style Encoder | ✅ | ❌ | ❌ |
| Style source | Diffusion / Ref Audio | **Voice Pack** | Voice Pack |
| Speed control | ❌ | ✅ | ✅ |
| Avg RTF (GPU) | ~0.6 | **~0.05** | ~0.05 |

## Pre-trained Model

Download from HuggingFace: [sienaiwen/StyleTTS2-LJSpeech](https://huggingface.co/sienaiwen/StyleTTS2-LJSpeech)

- `epoch_2nd_00078.pth` — Full model (2.0 GB, 209M params)
- `styletts2_lite.pth` — Lite model (312 MB, 82M params)

## Quick Start

### 1. Export Lite model from full checkpoint

```bash
python export_lite.py \
    --input Models/LJSpeech/epoch_2nd_00078.pth \
    --output Models/LJSpeech/styletts2_lite.pth
```

### 2. Create a Voice Pack from reference audio

```bash
# From a single reference audio
python make_voicepack.py \
    --ref_audio LJSpeech-1.1/wavs/LJ002-0254.wav \
    --output voices/lj.pt

# From a corpus (averaged by phoneme length)
python make_voicepack.py \
    --wavs_dir LJSpeech-1.1/wavs \
    --output voices/lj_corpus.pt \
    --max_files 500
```

### 3. Run inference with Lite model + Voice Pack

```bash
python styletts2_lite.py \
    --text "Hello world, this is StyleTTS2 Lite." \
    --voicepack voices/lj.pt \
    --checkpoint Models/LJSpeech/styletts2_lite.pth \
    --speed 1.0 \
    --output output.wav
```

### 4. Batch generate for multiple voices

```bash
# First generate voice packs from multiple reference audios
python batch_voicepack.py \
    --wavs_dir LJSpeech-1.1/wavs \
    --num_refs 5 \
    --text "Hello world"

# Then run Lite inference on all of them
python batch_lite.py
```

## Files

| File | Description |
|---|---|
| `export_lite.py` | Export 5 core modules from full checkpoint → lite checkpoint |
| `make_voicepack.py` | Generate Voice Pack from reference audio(s) using Style Encoder |
| `make_voicepack_diffusion.py` | Generate Voice Pack using Diffusion (experimental) |
| `styletts2_lite.py` | **Lite inference** — Voice Pack mode, no Diffusion |
| `inference.py` | Full inference — Diffusion mode |
| `inference_ref.py` | Full inference — Reference audio (Style Encoder) mode |
| `inference_voicepack.py` | Full model + Voice Pack inference |
| `inference_cached.py` | Full inference with Diffusion style caching |
| `batch_voicepack.py` | Batch generate Voice Packs + TTS from multiple references |
| `batch_lite.py` | Batch Lite inference for all Voice Packs in batch_output/ |

## How it works

```
Text "Hello world"
    │
    ▼
[Phonemizer] → phonemes
    │
    ▼
[BERT + bert_encoder] → text features
    │
    ▼
Voice Pack (.pt) ──→ ref_s [1, 256]
    ├── ref_s[:, :128]  → acoustic style (timbre) → Decoder
    └── ref_s[:, 128:]  → prosody style (rhythm)  → Predictor
    │
    ▼
[Predictor] → duration, F0, energy
    │
    ▼
[Text Encoder] → aligned text features
    │
    ▼
[Decoder (iSTFTNet)] → 24kHz waveform
```

## License

Based on [StyleTTS2](https://github.com/yl4579/StyleTTS2) by yl4579.
