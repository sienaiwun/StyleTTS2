"""
生成 Kokoro 风格的 Voice Pack
用法:
    python make_voicepack.py --wavs_dir LJSpeech-1.1/wavs --output voices/lj.pt
    python make_voicepack.py --ref_audio LJSpeech-1.1/wavs/LJ002-0254.wav --output voices/lj_single.pt
"""

import argparse
import os
import yaml
import numpy as np
import torch
import torchaudio
import librosa
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from models import *
from utils import *
from text_utils import TextCleaner

import phonemizer
from nltk.tokenize import word_tokenize

# Mel 参数 (与训练保持一致)
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def preprocess_audio(wav_path, sr=24000):
    """加载音频并转为归一化 mel 频谱"""
    wave, _ = librosa.load(wav_path, sr=sr)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor, wave


def count_phonemes(text, global_phonemizer, textcleaner):
    """计算文本对应的音素数量"""
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textcleaner(ps)
    return len(tokens) + 1  # +1 for the leading 0 token


def load_model(config_path, checkpoint_path, device):
    """加载完整模型"""
    config = yaml.safe_load(open(config_path))

    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params_whole = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    params = params_whole['net']

    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except RuntimeError:
                state_dict = params[key]
                new_state_dict = OrderedDict()
                ckpt_keys = list(state_dict.keys())
                if len(ckpt_keys) > 0 and ckpt_keys[0].startswith('module.'):
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v
                else:
                    for (k_m, _), (_, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                        new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)

    _ = [model[key].eval() for key in model]
    return model, config


def extract_style(model, mel, device):
    """从 mel 频谱提取 256 维风格向量 [acoustic_128 | prosody_128]"""
    mel_device = mel.to(device)
    with torch.no_grad():
        acoustic = model.style_encoder(mel_device.unsqueeze(1))       # [1, 128]
        prosody = model.predictor_encoder(mel_device.unsqueeze(1))    # [1, 128]
    style = torch.cat([acoustic, prosody], dim=-1)  # [1, 256]
    return style.cpu()


def make_voicepack_from_single(model, ref_audio_path, device, max_len=510):
    """从单个参考音频生成 voice pack [max_len, 1, 256]"""
    print(f"📎 Reference: {ref_audio_path}")
    mel, wave = preprocess_audio(ref_audio_path)
    duration = len(wave) / 24000
    print(f"   Duration: {duration:.2f}s, Mel shape: {mel.shape}")

    style = extract_style(model, mel, device)  # [1, 256]
    print(f"   Style vector: shape={style.shape}, norm={style.norm():.4f}")
    print(f"   Acoustic ([:128]) norm: {style[0, :128].norm():.4f}")
    print(f"   Prosody ([128:]) norm: {style[0, 128:].norm():.4f}")

    # 扩展为 [max_len, 1, 256]，所有长度使用相同的风格向量
    pack = style.unsqueeze(0).expand(max_len, -1, -1).clone()  # [510, 1, 256]
    return pack


def make_voicepack_from_corpus(model, wavs_dir, device, global_phonemizer, textcleaner,
                                metadata_path=None, max_len=510, max_files=500):
    """从语料库生成 voice pack (按音素长度分组平均，类似 Kokoro)"""
    
    # 加载 metadata 获取文本
    texts = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    wav_name = parts[0] + '.wav' if not parts[0].endswith('.wav') else parts[0]
                    text = parts[-1] if len(parts) == 2 else parts[1]
                    texts[wav_name] = text
    
    wav_files = sorted([f for f in os.listdir(wavs_dir) if f.endswith('.wav')])[:max_files]
    print(f"Processing {len(wav_files)} audio files...")

    # 按音素长度收集风格向量
    style_by_length = defaultdict(list)
    all_styles = []

    for f in tqdm(wav_files, desc="Extracting styles"):
        path = os.path.join(wavs_dir, f)
        try:
            mel, wave = preprocess_audio(path)
            style = extract_style(model, mel, device)  # [1, 256]
            all_styles.append(style)

            # 如果有文本，按音素长度分组
            if f in texts:
                try:
                    n_phonemes = count_phonemes(texts[f], global_phonemizer, textcleaner)
                    if 1 <= n_phonemes <= max_len:
                        style_by_length[n_phonemes].append(style)
                except:
                    pass
        except Exception as e:
            print(f"  Skip {f}: {e}")
            continue

    # 全局平均风格（用于填充没有数据的长度）
    global_mean = torch.stack(all_styles).mean(dim=0)  # [1, 256]
    print(f"\nGlobal mean style norm: {global_mean.norm():.4f}")
    print(f"Phoneme lengths with data: {len(style_by_length)}")

    # 构建 voice pack
    pack = torch.zeros(max_len, 1, 256)

    # 先填入有数据的长度的平均值
    filled_lengths = {}
    for length, styles in style_by_length.items():
        idx = length - 1  # 0-indexed
        if 0 <= idx < max_len:
            avg = torch.stack(styles).mean(dim=0)
            pack[idx] = avg
            filled_lengths[idx] = avg

    # 对没有数据的长度做插值或用最近邻填充
    if filled_lengths:
        sorted_indices = sorted(filled_lengths.keys())
        for i in range(max_len):
            if i not in filled_lengths:
                # 找最近的已填充长度
                closest = min(sorted_indices, key=lambda x: abs(x - i))
                pack[i] = filled_lengths[closest]
    else:
        # 没有文本信息，全部用全局平均
        pack = global_mean.unsqueeze(0).expand(max_len, -1, -1).clone()

    return pack


def main():
    parser = argparse.ArgumentParser(description='Generate Kokoro-style Voice Pack')
    parser.add_argument('--ref_audio', type=str, default=None, help='单个参考音频路径')
    parser.add_argument('--wavs_dir', type=str, default=None, help='语料库目录 (多文件模式)')
    parser.add_argument('--metadata', type=str, default=None, help='LJSpeech metadata CSV (多文件模式)')
    parser.add_argument('--output', type=str, default='voices/lj.pt', help='输出 voice pack 路径')
    parser.add_argument('--config', type=str, default='Configs/config.yml')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00078.pth')
    parser.add_argument('--max_files', type=int, default=500, help='多文件模式最多处理的文件数')
    parser.add_argument('--max_len', type=int, default=510, help='Voice pack 最大音素长度')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model, config = load_model(args.config, args.checkpoint, device)
    print("Model loaded ✅")

    if args.ref_audio:
        # 单文件模式
        pack = make_voicepack_from_single(model, args.ref_audio, device, args.max_len)
    elif args.wavs_dir:
        # 多文件模式
        print("Loading phonemizer for corpus mode...")
        global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True)
        textcleaner = TextCleaner()
        pack = make_voicepack_from_corpus(
            model, args.wavs_dir, device, global_phonemizer, textcleaner,
            args.metadata, args.max_len, args.max_files)
    else:
        print("Error: 请指定 --ref_audio 或 --wavs_dir")
        return

    # 保存
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(pack, args.output)
    print(f"\n✅ Voice pack saved: {args.output}")
    print(f"   Shape: {pack.shape}")
    print(f"   Size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
