#!/usr/bin/env python3
"""对比 StyleTTS2 和 Kokoro 在推理阶段的参数量差异"""
import torch, yaml, sys
sys.path.insert(0, '/home/naiwen/StyleTTS2')

from munch import Munch

def count(m):
    return sum(p.numel() for p in m.parameters())

def fmt(n):
    return f"{n/1e6:.2f}M"

# ── StyleTTS2 ─────────────────────────────────────────────────
config = yaml.safe_load(open('/home/naiwen/StyleTTS2/Configs/config.yml'))
def deep_munch(d):
    if isinstance(d, dict):
        return Munch({k: deep_munch(v) for k, v in d.items()})
    return d

mp = deep_munch(config['model_params'])

from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional
from models import (TextEncoder, StyleEncoder, AdainResBlk1d,
                    ProsodyPredictor, DurationEncoder)
import torch.nn as nn

# 直接实例化各组件（不依赖 text_aligner/bert）
from Utils.PLBERT.util import load_plbert
bert = load_plbert('Utils/PLBERT')
bert_encoder = nn.Linear(bert.config.hidden_size, mp.hidden_dim)

from models import TextEncoder as TE, StyleEncoder, ProsodyPredictor
text_encoder = TE(channels=mp.hidden_dim, kernel_size=5,
                  depth=mp.n_layer, n_symbols=mp.n_token)
style_encoder = StyleEncoder(dim_in=mp.dim_in, style_dim=mp.style_dim,
                              max_conv_dim=mp.max_conv_dim)
predictor_encoder = StyleEncoder(dim_in=mp.dim_in, style_dim=mp.style_dim,
                                 max_conv_dim=mp.max_conv_dim)
predictor = ProsodyPredictor(style_dim=mp.style_dim, d_hid=mp.hidden_dim,
                             nlayers=mp.n_layer, max_dur=mp.max_dur,
                             dropout=mp.dropout)

# decoder
if mp.decoder.type == 'istftnet':
    from Modules.istftnet import Decoder as ISTFTDecoder
    decoder = ISTFTDecoder(dim_in=mp.hidden_dim, style_dim=mp.style_dim,
                           dim_out=mp.n_mels,
                           resblock_kernel_sizes=mp.decoder.resblock_kernel_sizes,
                           upsample_rates=mp.decoder.upsample_rates,
                           upsample_initial_channel=mp.decoder.upsample_initial_channel,
                           resblock_dilation_sizes=mp.decoder.resblock_dilation_sizes,
                           upsample_kernel_sizes=mp.decoder.upsample_kernel_sizes,
                           gen_istft_n_fft=mp.decoder.gen_istft_n_fft,
                           gen_istft_hop_size=mp.decoder.gen_istft_hop_size)

# diffusion
transformer = StyleTransformer1d(channels=mp.style_dim * 2,
                                 context_embedding_features=bert.config.hidden_size,
                                 context_features=mp.style_dim * 2,
                                 **mp.diffusion.transformer)
diffusion = AudioDiffusionConditional(in_channels=1,
                                     embedding_max_length=512,
                                     embedding_features=bert.config.hidden_size,
                                     embedding_mask_proba=mp.diffusion.embedding_mask_proba)
diffusion.diffusion = KDiffusion(
    net=diffusion.unet,
    sigma_distribution=LogNormalDistribution(mean=mp.diffusion.dist.mean,
                                             std=mp.diffusion.dist.std),
    sigma_data=mp.diffusion.dist.sigma_data)
diffusion.diffusion.net = transformer
diffusion.unet = transformer

s2_comps = {
    "bert (PLBERT)":        bert,
    "bert_encoder":         bert_encoder,
    "predictor":            predictor,
    "text_encoder":         text_encoder,
    "decoder (iSTFTNet)":   decoder,
    "diffusion":            diffusion,
    "style_encoder":        style_encoder,
    "predictor_encoder":    predictor_encoder,
}

print()
print("=" * 56)
print("  StyleTTS2 各组件参数量（推理时）")
print("=" * 56)

infer_no_ref   = 0   # 无参考音频路径（用 diffusion）
infer_with_ref = 0   # 有参考音频路径（用 style_encoder）
diff_n = 0

for name, mod in s2_comps.items():
    n = count(mod)
    note = ""
    if "diffusion" in name:
        diff_n = n
        note = "<-- 无参考音频时才用"
        infer_no_ref += n
    elif "style_encoder" in name or "predictor_encoder" in name:
        note = "<-- 有参考音频时才用"
        infer_with_ref += n
    else:
        infer_no_ref += n
        infer_with_ref += n
    print(f"  {name:<28} {fmt(n):>8}   {note}")

total = sum(count(m) for m in s2_comps.values())
print("-" * 56)
print(f"  合计                         {fmt(total):>8}")
print(f"  推理(无ref,用diffusion)      {fmt(infer_no_ref):>8}")
print(f"  推理(有ref,用style_encoder)  {fmt(infer_with_ref):>8}")
print(f"  Diffusion 占比               {diff_n/total*100:>7.1f}%")

print()
print("=" * 56)
print("  训练专用（推理时不加载）")
print("=" * 56)
train_only = {
    "text_aligner (ASR)":     "~12M",
    "pitch_extractor (JDC)":  "~1M",
    "WavLM discriminator":    "~94M",
    "MPD discriminator":      "~5M",
    "MSD discriminator":      "~1M",
}
for k, v in train_only.items():
    print(f"  {k:<28} {v:>8}")

print()
print("=" * 56)
print("  Kokoro 82M 各组件估算")
print("=" * 56)

kokoro_comps = {
    "CustomAlbert (PLBERT)": "~50M  (albert-base 级)",
    "bert_encoder (Linear)": "~0.3M",
    "predictor (ProsodyPred)":"~5M",
    "text_encoder":           "~3M",
    "decoder (iSTFTNet)":     "~23M",
    "voice pack lookup":      "0     (预计算，非参数)",
    "diffusion":              "0     (无！)",
    "style_encoder":          "0     (无！)",
}
kok_total = 82
for k, v in kokoro_comps.items():
    print(f"  {k:<28} {v}")
print(f"  合计                          ~{kok_total}M")
