#!/usr/bin/env python3
"""
生成 sherpa-onnx 所需的两个辅助文件:

  voices.bin  - 将所有 voice pack 的 style 向量打包成一个 flat float32 二进制文件
                格式: N × 256 float32,  row i = style vector for token_len = i+1
                N = 最大 token 长度 (通常 512)

  tokens.txt  - 一行一个 token, 行号 = token ID
                格式与 Kokoro / Piper tokens.txt 完全相同

Usage:
  python prepare_sherpa_resources.py \\
      --voicepack_dir batch_output \\
      --max_len 512 \\
      --output_dir sherpa_resources
"""

import argparse
import os
import struct
import sys
import torch
import numpy as np
from pathlib import Path

# 导入 TextCleaner 获取 token 表
sys.path.insert(0, str(Path(__file__).parent))
from text_utils import TextCleaner


def build_voices_bin(voicepack_dir: str, max_len: int, output_path: str):
    """
    扫描 voicepack_dir 下的所有 voicepack.pt,
    将 style vector 按 token_len 索引排列后写入 voices.bin
    """
    voicepack_dir = Path(voicepack_dir)
    packs = sorted(voicepack_dir.glob("**/voicepack.pt"))
    
    if not packs:
        print(f"  ⚠️  No voicepack.pt found in {voicepack_dir}")
        print("     Generating random voice pack for testing...")
        # 生成随机 voice pack 供测试
        style_dim = 256
        voices = np.random.randn(max_len, style_dim).astype(np.float32) * 0.1
        voices.tofile(output_path)
        print(f"  ✅ Written {max_len} random voice vectors → {output_path}")
        return

    print(f"  Found {len(packs)} voice packs")
    
    # 加载第一个 pack 看结构
    first_pack = torch.load(packs[0], map_location='cpu', weights_only=True)
    print(f"  Pack shape: {first_pack.shape}  dtype: {first_pack.dtype}")
    
    # squeeze 掉多余维度，确保形状为 [N, 256]
    first_pack = first_pack.squeeze()  # [510, 256] 或 [510, 1, 256] -> [510, 256]
    if first_pack.dim() == 1:
        first_pack = first_pack.unsqueeze(0)  # edge case: single vector
    
    pack_len = first_pack.shape[0]   # 行数 = 该 pack 覆盖的最大 token 长度
    style_dim = first_pack.shape[1]  # 256
    
    # 用第一个 pack 填充整个 voices 矩阵（可后续扩展多 pack 支持）
    voices = torch.zeros(max_len, style_dim, dtype=torch.float32)
    copy_len = min(pack_len, max_len)
    voices[:copy_len] = first_pack[:copy_len]
    
    # 如果 pack 比 max_len 短，用最后一行填充剩余行
    if pack_len < max_len:
        voices[pack_len:] = first_pack[-1].unsqueeze(0).expand(max_len - pack_len, -1)
    
    # 保存为 flat float32 binary
    voices_np = voices.numpy().astype(np.float32)
    voices_np.tofile(output_path)
    
    print(f"  ✅ Written {max_len} × {style_dim} voice vectors → {output_path}")
    print(f"     File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def build_tokens_txt(output_path: str):
    """
    生成 tokens.txt，格式:
      <token_string> <token_id>
    一行一个，按 token ID 升序排列
    """
    tc = TextCleaner()
    sym2id = tc.word_index_dictionary  # dict: str -> int
    
    # 按 id 排序
    id2sym = {v: k for k, v in sym2id.items()}
    max_id = max(id2sym.keys())
    
    lines = []
    for i in range(max_id + 1):
        sym = id2sym.get(i, f"<unk_{i}>")
        lines.append(f"{sym} {i}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"  ✅ Written {len(lines)} tokens → {output_path}")
    print(f"     Token 0 = '{id2sym.get(0, '?')}' (BOS/blank)")
    print(f"     Total tokens: {len(lines)}")


def add_onnx_meta_data(onnx_path: str, sample_rate: int, num_voices: int, 
                        style_dim: int = 256, voice: str = "en-us"):
    """
    向 ONNX 模型写入 custom meta_data（供 C++ 侧读取）
    """
    import onnx
    
    model = onnx.load(onnx_path)
    
    meta = model.metadata_props
    
    def set_meta(key, value):
        # 检查是否已存在
        for prop in meta:
            if prop.key == key:
                prop.value = str(value)
                return
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = str(value)
    
    set_meta("sample_rate", sample_rate)
    set_meta("num_voices",  num_voices)
    set_meta("style_dim",   style_dim)
    set_meta("voice",       voice)
    
    onnx.save(model, onnx_path)
    print(f"  ✅ ONNX meta_data written:")
    print(f"     sample_rate = {sample_rate}")
    print(f"     num_voices  = {num_voices}")
    print(f"     style_dim   = {style_dim}")
    print(f"     voice       = {voice}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare sherpa-onnx resources for StyleTTS2 Lite')
    parser.add_argument('--voicepack_dir', type=str, default='batch_output',
                        help='Directory containing voicepack.pt files')
    parser.add_argument('--onnx_model', type=str, default='styletts2_kokoro.onnx',
                        help='Path to exported ONNX model (will add meta_data)')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum token sequence length for voice pack')
    parser.add_argument('--output_dir', type=str, default='sherpa_resources',
                        help='Output directory for sherpa-onnx resources')
    parser.add_argument('--sample_rate', type=int, default=24000)
    parser.add_argument('--voice', type=str, default='en-us',
                        help='espeak-ng voice tag')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 55)
    print("Preparing sherpa-onnx resources for StyleTTS2 Lite")
    print("=" * 55)
    
    # 1. voices.bin
    print("\n[1/3] Building voices.bin...")
    voices_path = os.path.join(args.output_dir, "voices.bin")
    build_voices_bin(args.voicepack_dir, args.max_len, voices_path)
    
    # 2. tokens.txt
    print("\n[2/3] Building tokens.txt...")
    tokens_path = os.path.join(args.output_dir, "tokens.txt")
    build_tokens_txt(tokens_path)
    
    # 3. ONNX meta_data
    if os.path.exists(args.onnx_model):
        print(f"\n[3/3] Adding meta_data to {args.onnx_model}...")
        add_onnx_meta_data(
            onnx_path=args.onnx_model,
            sample_rate=args.sample_rate,
            num_voices=args.max_len,
            style_dim=256,
            voice=args.voice
        )
    else:
        print(f"\n[3/3] ⚠️  ONNX model not found: {args.onnx_model}")
        print("     Run export_onnx_kokoro_style.py first")
    
    print("\n" + "=" * 55)
    print("✅ Resources ready!")
    print("=" * 55)
    print(f"\nGenerated files in '{args.output_dir}/':")
    for f in ['voices.bin', 'tokens.txt']:
        p = os.path.join(args.output_dir, f)
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"  {f:15s}  {size/1024:.1f} KB")
    
    print(f"\nNext steps:")
    print(f"  1. Copy {args.onnx_model} → Xcode project as 'styletts2.onnx'")
    print(f"  2. Copy {args.output_dir}/voices.bin → Xcode project")
    print(f"  3. Copy {args.output_dir}/tokens.txt → Xcode project")
    print(f"  4. Ensure espeak-ng-data/ is added as a Folder Reference in Xcode")
    print(f"  5. Rebuild the iOS XCFramework and run")


if __name__ == '__main__':
    main()
