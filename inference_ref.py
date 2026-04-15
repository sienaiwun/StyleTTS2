"""
StyleTTS2 推理脚本 - 方法2: 使用参考音频的 Style Encoder
用法:
    python inference_ref.py --text "Hello world" --ref_audio LJSpeech-1.1/wavs/LJ001-0001.wav --output output_ref.wav
"""

import argparse
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
from nltk.tokenize import word_tokenize
from collections import OrderedDict

from models import *
from utils import *
from text_utils import TextCleaner

import phonemizer


# Mel 参数 (与训练保持一致)
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def preprocess_audio(wav_path, sr=24000):
    """加载音频并转为归一化 mel 频谱"""
    wave, orig_sr = librosa.load(wav_path, sr=sr)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def inference_with_ref(model, textcleaner, global_phonemizer, text, ref_mel, device):
    """使用参考音频的 style encoder 进行推理 (方法2, 无 diffusion)"""
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

        # Text encoding
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        # === 方法2: 从参考音频提取风格 (不用 Diffusion) ===
        ref_mel_device = ref_mel.to(device)
        
        # style_encoder → 声学风格 (音色) → 送入 Decoder
        ref = model.style_encoder(ref_mel_device.unsqueeze(1))    # [1, 128]
        
        # predictor_encoder → 韵律风格 (语速/语调) → 送入 Duration/F0 预测器
        s = model.predictor_encoder(ref_mel_device.unsqueeze(1))  # [1, 128]

        print(f"  [Style Encoder] acoustic ref: shape={ref.shape}, norm={ref.norm():.4f}")
        print(f"  [Predictor Encoder] prosody s: shape={s.shape}, norm={s.norm():.4f}")

        # Duration prediction
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_dur[-1] += 5

        # Alignment
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # F0 and energy prediction
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        # Decode waveform
        aln_en = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        out = model.decoder(aln_en, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()


def find_best_reference(wavs_dir, sr=24000, target_duration=(3, 8), num_candidates=20):
    """从 LJSpeech 中找一个合适的参考音频 (中等长度、信号质量好)"""
    import os
    import random
    
    wav_files = sorted([f for f in os.listdir(wavs_dir) if f.endswith('.wav')])
    random.seed(42)
    candidates = random.sample(wav_files, min(num_candidates, len(wav_files)))
    
    best_file = None
    best_score = -1
    
    for f in candidates:
        path = os.path.join(wavs_dir, f)
        wave, _ = librosa.load(path, sr=sr)
        duration = len(wave) / sr
        
        # 偏好 3-8 秒的音频
        if duration < target_duration[0] or duration > target_duration[1]:
            continue
        
        # 信噪比估计 (RMS能量)
        rms = np.sqrt(np.mean(wave**2))
        score = rms * (1 - abs(duration - 5) / 5)  # 偏好 5 秒左右
        
        if score > best_score:
            best_score = score
            best_file = path
    
    if best_file is None:
        # 如果没找到合适的，就用第一个候选
        best_file = os.path.join(wavs_dir, candidates[0])
    
    return best_file


def main():
    parser = argparse.ArgumentParser(description='StyleTTS2 Reference-based Inference (Method 2)')
    parser.add_argument('--text', type=str, required=True, help='要合成的文本')
    parser.add_argument('--output', type=str, default='output_ref.wav', help='输出 WAV 文件路径')
    parser.add_argument('--config', type=str, default='Configs/config.yml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00078.pth',
                        help='模型 checkpoint 路径')
    parser.add_argument('--ref_audio', type=str, default=None, 
                        help='参考音频路径 (不指定则自动选择)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载配置
    config = yaml.safe_load(open(args.config))

    # 加载 phonemizer
    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)

    textcleaner = TextCleaner()

    # 加载预训练模型
    print("Loading ASR model...")
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    print("Loading F0 model...")
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    print("Loading PLBERT model...")
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # 构建模型
    print("Building model...")
    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # 加载 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    params_whole = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    params = params_whole['net']

    for key in model:
        if key in params:
            print(f'  {key} loading...')
            try:
                model[key].load_state_dict(params[key], strict=True)
                print(f'    ✅ {key} loaded directly')
            except RuntimeError:
                state_dict = params[key]
                new_state_dict = OrderedDict()
                ckpt_keys = list(state_dict.keys())
                if len(ckpt_keys) > 0 and ckpt_keys[0].startswith('module.'):
                    print(f'    ⚠️  Removing "module." prefix for {key}')
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v
                else:
                    for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                        new_state_dict[k_m] = v_c
                try:
                    model[key].load_state_dict(new_state_dict, strict=True)
                    print(f'    ✅ {key} loaded after key remapping')
                except RuntimeError as e2:
                    print(f'    ❌ {key} FAILED to load: {e2}')

    _ = [model[key].eval() for key in model]

    # 选择参考音频
    if args.ref_audio:
        ref_path = args.ref_audio
    else:
        print("\nAuto-selecting reference audio from LJSpeech...")
        ref_path = find_best_reference('LJSpeech-1.1/wavs')
    
    print(f"\n📎 Reference audio: {ref_path}")
    ref_wave, _ = librosa.load(ref_path, sr=24000)
    print(f"   Duration: {len(ref_wave)/24000:.2f}s")
    
    # 预处理参考音频 → mel
    ref_mel = preprocess_audio(ref_path, sr=24000)
    print(f"   Mel shape: {ref_mel.shape}")

    # 合成语音
    print(f"\n🎤 Synthesizing: \"{args.text}\"")
    print(f"   Method: Style Encoder (Reference-based, NO Diffusion)")

    start = time.time()
    wav = inference_with_ref(model, textcleaner, global_phonemizer, args.text, ref_mel, device)
    elapsed = time.time() - start
    rtf = elapsed / (len(wav) / 24000)

    # 保存
    sf.write(args.output, wav, 24000)

    print(f"\n✅ Done!")
    print(f"  Output: {args.output}")
    print(f"  Duration: {len(wav)/24000:.2f}s")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  RTF: {rtf:.4f}")


if __name__ == '__main__':
    main()
