"""
用 StyleTTS2 Lite 批量为 batch_output 中的每个 voice pack 生成推理音频
"""
import os
import sys
import time
import yaml
import numpy as np
import torch
import soundfile as sf

# 复用 styletts2_lite 中的类和函数
from styletts2_lite import StyleTTS2Lite, synthesize
from text_utils import TextCleaner
from nltk.tokenize import word_tokenize
import phonemizer

TEXT = "Hello world, this is a speech synthesis test using StyleTTS2 Lite model."
BATCH_DIR = "batch_output"
CONFIG = "Configs/config.yml"
CHECKPOINT = "Models/LJSpeech/styletts2_lite.pth"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = yaml.safe_load(open(CONFIG))

    print("Building StyleTTS2 Lite...")
    model = StyleTTS2Lite(config)
    model.eval().to(device)
    print(f"Loading checkpoint: {CHECKPOINT}")
    model.load_from_checkpoint(CHECKPOINT)

    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)
    textcleaner = TextCleaner()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.1f}M params | No Diffusion, No Style Encoder")
    print(f"Text: \"{TEXT}\"\n")

    subdirs = sorted([d for d in os.listdir(BATCH_DIR)
                      if os.path.isfile(os.path.join(BATCH_DIR, d, 'voicepack.pt'))])

    results = []
    for name in subdirs:
        vp_path = os.path.join(BATCH_DIR, name, 'voicepack.pt')
        pack = torch.load(vp_path, map_location='cpu', weights_only=True)

        start = time.time()
        wav, pred_dur = synthesize(model, textcleaner, global_phonemizer, TEXT, pack, device)
        elapsed = time.time() - start
        dur = len(wav) / 24000
        rtf = elapsed / dur

        out_path = os.path.join(BATCH_DIR, name, 'tts_lite.wav')
        sf.write(out_path, wav, 24000)

        print(f"  ✅ {name:<16} → tts_lite.wav  {dur:.2f}s  RTF={rtf:.4f}")
        results.append((name, dur, rtf))

    print(f"\n{'='*50}")
    print(f"📊 Summary ({len(results)} voices)")
    print(f"{'='*50}")
    avg_rtf = np.mean([r[2] for r in results])
    for name, dur, rtf in results:
        print(f"  {name:<16}  duration={dur:.2f}s  RTF={rtf:.4f}")
    print(f"\n  Average RTF: {avg_rtf:.4f}")
    print(f"\n✅ Done! Each folder now has: reference.wav, voicepack.pt, tts_output.wav, tts_lite.wav")

if __name__ == '__main__':
    main()
