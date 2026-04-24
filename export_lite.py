"""
将 StyleTTS2 完整 checkpoint 精简为只包含 5 个核心推理模块的轻量版
类似 Kokoro 的做法：只保留 bert, bert_encoder, predictor, decoder, text_encoder

用法:
    python export_lite.py --input Models/LJSpeech/epoch_2nd_00078.pth --output Models/LJSpeech/styletts2_lite.pth
"""

import argparse
import torch
from collections import OrderedDict


CORE_MODULES = ['bert', 'bert_encoder', 'predictor', 'decoder', 'text_encoder']


def main():
    parser = argparse.ArgumentParser(description='Export lightweight StyleTTS2 model')
    parser.add_argument('--input', type=str, required=True, help='完整 checkpoint 路径')
    parser.add_argument('--output', type=str, required=True, help='精简 checkpoint 输出路径')
    args = parser.parse_args()

    print(f"Loading full checkpoint: {args.input}")
    full_ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    full_params = full_ckpt['net']

    # 提取核心模块，并移除 "module." 前缀
    lite_params = {}
    for key in CORE_MODULES:
        if key not in full_params:
            print(f"  ⚠️ {key} not found in checkpoint!")
            continue

        state_dict = full_params[key]
        # 移除 DDP 的 "module." 前缀
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.'):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

        n_params = sum(p.numel() for p in state_dict.values())
        size_mb = sum(p.nbytes for p in state_dict.values()) / 1024 / 1024
        lite_params[key] = state_dict
        print(f"  ✅ {key:<20} {n_params:>12,} params  {size_mb:>8.1f} MB")

    # 保存精简 checkpoint
    torch.save(lite_params, args.output)

    import os
    full_size = os.path.getsize(args.input) / 1024 / 1024
    lite_size = os.path.getsize(args.output) / 1024 / 1024

    total_params = sum(sum(p.numel() for p in sd.values()) for sd in lite_params.values())
    print(f"\n📊 Summary:")
    print(f"   Full checkpoint: {full_size:.1f} MB")
    print(f"   Lite checkpoint: {lite_size:.1f} MB ({lite_size/full_size*100:.1f}%)")
    print(f"   Total params:    {total_params:,}")
    print(f"   Modules:         {', '.join(CORE_MODULES)}")
    print(f"\n✅ Saved: {args.output}")


if __name__ == '__main__':
    main()
