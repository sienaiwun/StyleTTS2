"""
用 Diffusion 生成 Voice Pack
- 用不同文本通过 Diffusion 生成多个风格向量
- 按音素长度分组平均，保存为 voice pack
用法:
    python make_voicepack_diffusion.py --output voices/lj_diffusion.pt
"""

import argparse
import os
import yaml
import numpy as np
import torch
from collections import OrderedDict, defaultdict
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

import phonemizer


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# 多样化的英文示例文本 (覆盖不同长度和语境)
SAMPLE_TEXTS = [
    "Hi.",
    "Hello.",
    "Yes please.",
    "Thank you.",
    "Good morning.",
    "How are you?",
    "Hello world.",
    "Nice to meet you.",
    "What time is it?",
    "I love this place.",
    "The weather is nice today.",
    "Can you help me with this?",
    "She walked along the quiet river.",
    "I think we should leave now before it rains.",
    "The quick brown fox jumps over the lazy dog.",
    "Please make sure you have the correct access rights.",
    "Artificial intelligence is transforming the way we live and work.",
    "In the beginning, there was nothing but silence and darkness everywhere.",
    "The committee has decided to postpone the meeting until further notice from the board.",
    "Scientists have discovered a new species of deep sea fish living near hydrothermal vents in the Pacific Ocean.",
    "Despite the challenges and setbacks they faced along the way, the team managed to complete the project on time and within budget.",
    "The development of large language models has revolutionized natural language processing, enabling machines to understand and generate human language with unprecedented accuracy and fluency.",
    "Throughout history, great civilizations have risen and fallen, each leaving behind a unique legacy that continues to influence our modern world in ways both subtle and profound, shaping our cultures, technologies, and ways of thinking about the universe around us.",
]


def generate_diffusion_style(model, sampler, textcleaner, global_phonemizer, text, device,
                              diffusion_steps=5, embedding_scale=1.0):
    """用 Diffusion 为给定文本生成风格向量，返回 (n_tokens, s_pred[1, 256])"""
    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    n_tokens = len(tokens)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())

        noise = torch.randn(1, 1, 256).to(device)
        s_pred = sampler(noise,
                         embedding=bert_dur[0].unsqueeze(0),
                         num_steps=diffusion_steps,
                         embedding_scale=embedding_scale).squeeze(0)  # [1, 256]

    return n_tokens, s_pred.cpu()


def main():
    parser = argparse.ArgumentParser(description='Generate Voice Pack using Diffusion')
    parser.add_argument('--output', type=str, default='voices/lj_diffusion.pt', help='输出路径')
    parser.add_argument('--config', type=str, default='Configs/config.yml')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00078.pth')
    parser.add_argument('--diffusion_steps', type=int, default=5, help='Diffusion 去噪步数')
    parser.add_argument('--embedding_scale', type=float, default=1.0, help='CFG scale')
    parser.add_argument('--num_seeds', type=int, default=5, help='每个文本用多少个不同 seed 生成')
    parser.add_argument('--max_len', type=int, default=510, help='Voice pack 最大音素长度')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载配置和模型
    config = yaml.safe_load(open(args.config))

    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)
    textcleaner = TextCleaner()

    print("Loading models...")
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

    params_whole = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    params = params_whole['net']
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except RuntimeError:
                state_dict = params[key]
                new_state_dict = OrderedDict()
                if list(state_dict.keys())[0].startswith('module.'):
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v
                else:
                    for (k_m, _), (_, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                        new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)
    _ = [model[key].eval() for key in model]
    print("Model loaded ✅")

    # 创建 Diffusion Sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    # 用 Diffusion 生成大量风格向量
    print(f"\n🎲 Generating style vectors with Diffusion...")
    print(f"   {len(SAMPLE_TEXTS)} texts × {args.num_seeds} seeds = {len(SAMPLE_TEXTS) * args.num_seeds} samples")
    print(f"   Diffusion steps: {args.diffusion_steps}, Embedding scale: {args.embedding_scale}")

    style_by_length = defaultdict(list)
    all_styles = []

    for text in SAMPLE_TEXTS:
        for seed in range(args.num_seeds):
            torch.manual_seed(seed * 1000 + hash(text) % 10000)
            try:
                n_tokens, s_pred = generate_diffusion_style(
                    model, sampler, textcleaner, global_phonemizer, text, device,
                    args.diffusion_steps, args.embedding_scale)

                style_by_length[n_tokens].append(s_pred)
                all_styles.append(s_pred)
            except Exception as e:
                print(f"  Skip '{text[:30]}...' seed={seed}: {e}")
                continue

    print(f"\n   Total samples: {len(all_styles)}")
    print(f"   Unique phoneme lengths: {len(style_by_length)}")
    print(f"   Length range: {min(style_by_length.keys())} - {max(style_by_length.keys())}")

    # 全局平均
    global_mean = torch.stack(all_styles).mean(dim=0)  # [1, 256]

    # 构建 voice pack: 按长度分组平均 + 插值填充
    pack = torch.zeros(args.max_len, 1, 256)

    # 填入有数据的长度
    filled = {}
    for length, styles in sorted(style_by_length.items()):
        idx = length - 1
        if 0 <= idx < args.max_len:
            avg = torch.stack(styles).mean(dim=0)
            pack[idx] = avg
            filled[idx] = avg
            print(f"   Length {length:3d} (idx={idx:3d}): {len(styles)} samples, norm={avg.norm():.4f}")

    # 线性插值填充空缺
    sorted_filled = sorted(filled.keys())
    if len(sorted_filled) >= 2:
        # 填充两端
        for i in range(sorted_filled[0]):
            pack[i] = filled[sorted_filled[0]]
        for i in range(sorted_filled[-1] + 1, args.max_len):
            pack[i] = filled[sorted_filled[-1]]

        # 线性插值中间空缺
        for k in range(len(sorted_filled) - 1):
            start_idx = sorted_filled[k]
            end_idx = sorted_filled[k + 1]
            if end_idx - start_idx > 1:
                for i in range(start_idx + 1, end_idx):
                    alpha = (i - start_idx) / (end_idx - start_idx)
                    pack[i] = (1 - alpha) * filled[start_idx] + alpha * filled[end_idx]
    else:
        # 只有一个长度，全部用全局平均
        pack = global_mean.unsqueeze(0).expand(args.max_len, -1, -1).clone()

    # 保存
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(pack, args.output)

    print(f"\n✅ Diffusion Voice Pack saved: {args.output}")
    print(f"   Shape: {pack.shape}")
    print(f"   Size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print(f"   Norm range: {pack.norm(dim=-1).min():.4f} - {pack.norm(dim=-1).max():.4f}")


if __name__ == '__main__':
    main()
