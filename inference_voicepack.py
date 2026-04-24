"""
StyleTTS2 推理脚本 - 使用预制 Voice Pack (Kokoro 风格)
用法:
    python inference_voicepack.py --text "Hello world" --voicepack voices/lj.pt --output output_vp.wav
"""

import argparse
import time
import yaml
import numpy as np
import torch
import soundfile as sf
from nltk.tokenize import word_tokenize
from collections import OrderedDict

from models import *
from utils import *
from text_utils import TextCleaner

import phonemizer


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def inference_with_voicepack(model, textcleaner, global_phonemizer, text, pack, device):
    """使用 Voice Pack 推理 (无 Diffusion, 无 Style Encoder, 直接查表)"""
    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    n_tokens = len(tokens)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    # 从 voice pack 查表获取风格向量 (Kokoro 风格)
    pack_idx = min(n_tokens - 1, len(pack) - 1)
    ref_s = pack[pack_idx].to(device)  # [1, 256]

    ref = ref_s[:, :128]   # 声学风格 → Decoder
    s = ref_s[:, 128:]     # 韵律风格 → Duration/F0

    print(f"  [Voice Pack] index={pack_idx}, n_tokens={n_tokens}")
    print(f"  [Voice Pack] acoustic ref norm: {ref.norm():.4f}")
    print(f"  [Voice Pack] prosody s norm: {s.norm():.4f}")

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        # Text encoding
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

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


def main():
    parser = argparse.ArgumentParser(description='StyleTTS2 Voice Pack Inference (Kokoro-style)')
    parser.add_argument('--text', type=str, required=True, help='要合成的文本')
    parser.add_argument('--output', type=str, default='output_vp.wav', help='输出 WAV 文件路径')
    parser.add_argument('--voicepack', type=str, required=True, help='Voice Pack .pt 文件路径')
    parser.add_argument('--config', type=str, default='Configs/config.yml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00078.pth',
                        help='模型 checkpoint 路径')
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

    # 加载预训练模型 (注意: 不需要 diffusion, style_encoder, predictor_encoder)
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

    print("Building model...")
    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # 加载 checkpoint (只需要核心模块)
    print(f"Loading checkpoint: {args.checkpoint}")
    params_whole = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    params = params_whole['net']

    # Voice Pack 推理只需要这5个模块 (与 Kokoro 相同)
    needed_modules = ['bert', 'bert_encoder', 'predictor', 'decoder', 'text_encoder']

    for key in model:
        if key in params:
            suffix = " ⭐ (core)" if key in needed_modules else ""
            try:
                model[key].load_state_dict(params[key], strict=True)
                print(f'  ✅ {key}{suffix}')
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
                try:
                    model[key].load_state_dict(new_state_dict, strict=True)
                    print(f'  ✅ {key} (remapped){suffix}')
                except RuntimeError as e2:
                    print(f'  ❌ {key} FAILED: {e2}')

    _ = [model[key].eval() for key in model]

    # 加载 Voice Pack
    print(f"\n📦 Loading voice pack: {args.voicepack}")
    pack = torch.load(args.voicepack, map_location='cpu', weights_only=True)
    print(f"   Shape: {pack.shape}")

    # 合成语音
    print(f"\n🎤 Synthesizing: \"{args.text}\"")
    print(f"   Method: Voice Pack (NO Diffusion, NO Style Encoder)")

    start = time.time()
    wav = inference_with_voicepack(model, textcleaner, global_phonemizer, args.text, pack, device)
    elapsed = time.time() - start
    rtf = elapsed / (len(wav) / 24000)

    sf.write(args.output, wav, 24000)

    print(f"\n✅ Done!")
    print(f"  Output: {args.output}")
    print(f"  Duration: {len(wav)/24000:.2f}s")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  RTF: {rtf:.4f}")


if __name__ == '__main__':
    main()
