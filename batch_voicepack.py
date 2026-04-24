"""
批量生成: 从 LJSpeech 选多个参考音频 → 每个生成 voice pack + TTS 音频
用法:
    python batch_voicepack.py --wavs_dir LJSpeech-1.1/wavs --num_refs 5 --text "Hello world, this is a speech synthesis test."
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from nltk.tokenize import word_tokenize
from collections import OrderedDict

from models import *
from utils import *
from text_utils import TextCleaner

import phonemizer


# Mel 参数
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def preprocess_audio(wav_path, sr=24000):
    wave, _ = librosa.load(wav_path, sr=sr)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor, wave


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def extract_style(model, mel, device):
    mel_device = mel.to(device)
    with torch.no_grad():
        acoustic = model.style_encoder(mel_device.unsqueeze(1))
        prosody = model.predictor_encoder(mel_device.unsqueeze(1))
    return torch.cat([acoustic, prosody], dim=-1).cpu()


def make_voicepack(model, ref_audio_path, device, max_len=510):
    mel, wave = preprocess_audio(ref_audio_path)
    style = extract_style(model, mel, device)
    pack = style.unsqueeze(0).expand(max_len, -1, -1).clone()
    return pack, wave, style


def inference_with_style(model, textcleaner, global_phonemizer, text, ref_s, device):
    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref = ref_s[:, :128].to(device)
        s = ref_s[:, 128:].to(device)

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        aln_en = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        out = model.decoder(aln_en, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()


def select_references(wavs_dir, num_refs=5, sr=24000):
    """选择不同风格/长度的参考音频"""
    import random
    wav_files = sorted([f for f in os.listdir(wavs_dir) if f.endswith('.wav')])

    # 按不同章节选 (LJSpeech 的不同章节有不同录音风格)
    chapters = {}
    for f in wav_files:
        chap = f.split('-')[0]  # e.g. LJ001, LJ002, ...
        if chap not in chapters:
            chapters[chap] = []
        chapters[chap].append(f)

    # 从不同章节中各选一个合适的音频 (3-8秒)
    selected = []
    random.seed(42)
    chapter_keys = sorted(chapters.keys())
    random.shuffle(chapter_keys)

    for chap in chapter_keys:
        if len(selected) >= num_refs:
            break
        candidates = chapters[chap]
        random.shuffle(candidates)
        for f in candidates:
            path = os.path.join(wavs_dir, f)
            try:
                wave, _ = librosa.load(path, sr=sr)
                dur = len(wave) / sr
                if 3 <= dur <= 8:
                    rms = np.sqrt(np.mean(wave ** 2))
                    if rms > 0.02:  # 排除太安静的
                        selected.append((f, path, dur, rms))
                        break
            except:
                continue

    # 如果不够，随机补充
    while len(selected) < num_refs:
        f = random.choice(wav_files)
        path = os.path.join(wavs_dir, f)
        if not any(s[0] == f for s in selected):
            try:
                wave, _ = librosa.load(path, sr=sr)
                dur = len(wave) / sr
                if 2 <= dur <= 10:
                    selected.append((f, path, dur, np.sqrt(np.mean(wave ** 2))))
            except:
                continue

    return selected


def load_model(config_path, checkpoint_path, device):
    config = yaml.safe_load(open(config_path))
    text_aligner = load_ASR_models(config.get('ASR_path', False), config.get('ASR_config', False))
    pitch_extractor = load_F0_models(config.get('F0_path', False))
    from Utils.PLBERT.util import load_plbert
    plbert = load_plbert(config.get('PLBERT_dir', False))

    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['net']
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except RuntimeError:
                sd = params[key]
                new_sd = OrderedDict()
                if list(sd.keys())[0].startswith('module.'):
                    new_sd = OrderedDict((k[7:], v) for k, v in sd.items())
                else:
                    new_sd = OrderedDict(
                        (k_m, v_c) for (k_m, _), (_, v_c) in
                        zip(model[key].state_dict().items(), sd.items()))
                model[key].load_state_dict(new_sd, strict=True)
    _ = [model[key].eval() for key in model]
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Batch generate voice packs and TTS from multiple references')
    parser.add_argument('--wavs_dir', type=str, default='LJSpeech-1.1/wavs')
    parser.add_argument('--output_dir', type=str, default='batch_output')
    parser.add_argument('--num_refs', type=int, default=5)
    parser.add_argument('--text', type=str, default='Hello world, this is a speech synthesis test using StyleTTS2.')
    parser.add_argument('--config', type=str, default='Configs/config.yml')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00078.pth')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model, config = load_model(args.config, args.checkpoint, device)
    print("Model loaded ✅\n")

    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)
    textcleaner = TextCleaner()

    # 选择参考音频
    print(f"\n📂 Selecting {args.num_refs} reference audios from {args.wavs_dir}...")
    refs = select_references(args.wavs_dir, args.num_refs)

    print(f"\nSelected references:")
    for i, (fname, path, dur, rms) in enumerate(refs):
        print(f"  [{i+1}] {fname} ({dur:.1f}s, RMS={rms:.4f})")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理每个参考音频
    results = []
    for i, (fname, ref_path, dur, rms) in enumerate(refs):
        name = fname.replace('.wav', '')
        subdir = os.path.join(args.output_dir, name)
        os.makedirs(subdir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(refs)}] Processing: {fname} ({dur:.1f}s)")
        print(f"{'='*60}")

        # 1. 复制参考音频
        ref_wave, _ = librosa.load(ref_path, sr=24000)
        sf.write(os.path.join(subdir, 'reference.wav'), ref_wave, 24000)
        print(f"  📎 Reference saved: {subdir}/reference.wav")

        # 2. 生成 voice pack
        pack, _, style = make_voicepack(model, ref_path, device)
        vp_path = os.path.join(subdir, 'voicepack.pt')
        torch.save(pack, vp_path)
        print(f"  📦 Voice pack saved: {vp_path}")
        print(f"     Acoustic norm: {style[0, :128].norm():.4f}, Prosody norm: {style[0, 128:].norm():.4f}")

        # 3. 用 voice pack 生成 TTS
        start = time.time()
        wav = inference_with_style(model, textcleaner, global_phonemizer, args.text, style, device)
        elapsed = time.time() - start
        rtf = elapsed / (len(wav) / 24000)

        tts_path = os.path.join(subdir, 'tts_output.wav')
        sf.write(tts_path, wav, 24000)
        print(f"  🎤 TTS output saved: {tts_path}")
        print(f"     Duration: {len(wav)/24000:.2f}s, Time: {elapsed:.2f}s, RTF: {rtf:.4f}")

        results.append({
            'name': name,
            'ref_duration': dur,
            'tts_duration': len(wav) / 24000,
            'rtf': rtf,
            'acoustic_norm': style[0, :128].norm().item(),
            'prosody_norm': style[0, 128:].norm().item(),
        })

    # 写汇总报告
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    print(f"Text: \"{args.text}\"")
    print(f"Output dir: {args.output_dir}/")
    print()

    report = []
    report.append(f"Text: \"{args.text}\"\n")
    report.append(f"{'Name':<20} {'Ref(s)':<8} {'TTS(s)':<8} {'RTF':<8} {'Acoustic':<10} {'Prosody':<10}")
    report.append('-' * 70)

    for r in results:
        line = f"{r['name']:<20} {r['ref_duration']:<8.1f} {r['tts_duration']:<8.2f} {r['rtf']:<8.4f} {r['acoustic_norm']:<10.4f} {r['prosody_norm']:<10.4f}"
        report.append(line)
        print(f"  {line}")

    report_path = os.path.join(args.output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n📝 Report saved: {report_path}")

    print(f"\n✅ Done! {len(refs)} voice packs + TTS outputs generated in {args.output_dir}/")
    print(f"   Each folder contains: reference.wav, voicepack.pt, tts_output.wav")


if __name__ == '__main__':
    main()
