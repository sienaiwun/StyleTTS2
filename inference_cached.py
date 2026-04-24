"""
StyleTTS2 推理 - 带 Diffusion 风格缓存
第一次运行: Diffusion 生成风格 → 缓存到文件
后续运行: 直接加载缓存, 跳过 Diffusion

用法:
    # 首次 (会运行 Diffusion 并缓存)
    python inference_cached.py --text "Hello world" --output out.wav --cache_dir style_cache/

    # 再次同样文本 (直接读缓存, 超快)
    python inference_cached.py --text "Hello world" --output out.wav --cache_dir style_cache/

    # 新文本 (会重新 Diffusion)
    python inference_cached.py --text "Good morning" --output out2.wav --cache_dir style_cache/
"""

import argparse
import hashlib
import os
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
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

import phonemizer


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def text_to_cache_key(text, seed, diffusion_steps, embedding_scale):
    """生成缓存文件名"""
    raw = f"{text}|seed={seed}|steps={diffusion_steps}|scale={embedding_scale}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    safe_name = text[:30].replace(' ', '_').replace('"', '').replace("'", '')
    return f"{safe_name}_{h}.pt"


def inference_with_cache(model, sampler, textcleaner, global_phonemizer, text, device,
                         diffusion_steps=5, embedding_scale=1.0, seed=0,
                         cache_dir=None):
    """推理: 有缓存用缓存, 无缓存跑 Diffusion 并存缓存"""

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

        # === 检查缓存 ===
        cache_file = None
        cached = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = text_to_cache_key(text, seed, diffusion_steps, embedding_scale)
            cache_file = os.path.join(cache_dir, cache_key)

            if os.path.exists(cache_file):
                s_pred = torch.load(cache_file, map_location=device, weights_only=True)
                cached = True
                print(f"  ⚡ Cache HIT: {cache_file}")
            
        if not cached:
            # === 运行 Diffusion ===
            torch.manual_seed(seed)
            noise = torch.randn(1, 1, 256).to(device)
            s_pred = sampler(noise,
                             embedding=bert_dur[0].unsqueeze(0),
                             num_steps=diffusion_steps,
                             embedding_scale=embedding_scale).squeeze(0)
            print(f"  🎲 Diffusion generated (steps={diffusion_steps}, scale={embedding_scale})")

            # 保存缓存
            if cache_file:
                torch.save(s_pred.cpu(), cache_file)
                print(f"  💾 Cached to: {cache_file}")

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        print(f"  Style norm: acoustic={ref.norm():.4f}, prosody={s.norm():.4f}")

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

        # F0 and energy
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        # Decode
        aln_en = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        out = model.decoder(aln_en, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='StyleTTS2 Inference with Diffusion Style Cache')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_cached.wav')
    parser.add_argument('--config', type=str, default='Configs/config.yml')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00078.pth')
    parser.add_argument('--cache_dir', type=str, default='style_cache/', help='风格缓存目录')
    parser.add_argument('--diffusion_steps', type=int, default=5)
    parser.add_argument('--embedding_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_cache', action='store_true', help='不使用缓存')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = yaml.safe_load(open(args.config))

    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)
    textcleaner = TextCleaner()

    print("Loading models...")
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    pitch_extractor = load_F0_models(config.get('F0_path', False))
    from Utils.PLBERT.util import load_plbert
    plbert = load_plbert(config.get('PLBERT_dir', False))

    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['net']
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

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    print(f"\n🎤 Synthesizing: \"{args.text}\"")
    cache = None if args.no_cache else args.cache_dir

    start = time.time()
    wav = inference_with_cache(
        model, sampler, textcleaner, global_phonemizer, args.text, device,
        args.diffusion_steps, args.embedding_scale, args.seed, cache)
    elapsed = time.time() - start
    rtf = elapsed / (len(wav) / 24000)

    sf.write(args.output, wav, 24000)
    print(f"\n✅ Output: {args.output} ({len(wav)/24000:.2f}s, RTF={rtf:.4f})")


if __name__ == '__main__':
    main()
