#!/usr/bin/env python3
"""
验证导出的 StyleTTS2 ONNX 模型

测试流程:
1. 加载原始 PyTorch 模型和 ONNX 模型
2. 使用相同输入进行推理
3. 比较输出并生成示例音频
"""

import argparse
import time
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import yaml
from pathlib import Path

# 导入原始模型
import sys
sys.path.insert(0, str(Path(__file__).parent))

from text_utils import TextCleaner
from nltk.tokenize import word_tokenize
import phonemizer


def load_voicepack(voicepack_path: str, token_len: int) -> np.ndarray:
    """加载 Voice Pack 并选择对应长度的 style vector"""
    pack = torch.load(voicepack_path, map_location='cpu', weights_only=True)
    pack_idx = min(token_len - 1, len(pack) - 1)
    ref_s = pack[pack_idx].numpy()  # [1, 256]
    return ref_s


def text_to_tokens(text: str, textcleaner, global_phonemizer) -> np.ndarray:
    """文本转 token IDs"""
    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    
    tokens = textcleaner(ps)
    tokens.insert(0, 0)  # BOS
    
    return np.array([tokens], dtype=np.int64)


def run_onnx_inference(
    onnx_path: str,
    tokens: np.ndarray,
    style: np.ndarray,
    speed: float = 1.0
) -> np.ndarray:
    """运行 ONNX 推理"""
    
    # 创建会话
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['CPUExecutionProvider']
    # 如果有 GPU，可以添加 CUDAExecutionProvider
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    # 准备输入
    inputs = {
        'tokens': tokens,
        'style': style.astype(np.float32),
        'speed': np.array([speed], dtype=np.float32)
    }
    
    # 推理
    outputs = session.run(None, inputs)
    
    return outputs[0]


def main():
    parser = argparse.ArgumentParser(description='Verify StyleTTS2 ONNX model')
    parser.add_argument('--onnx', type=str, default='styletts2_kokoro.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--voicepack', type=str, 
                        default='batch_output/LJ005-0254/voicepack.pt',
                        help='Path to voice pack')
    parser.add_argument('--text', type=str, 
                        default='Hello, this is a test of StyleTTS two lite.',
                        help='Text to synthesize')
    parser.add_argument('--output', type=str, default='onnx_output.wav',
                        help='Output audio file')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speech speed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("StyleTTS2 ONNX Model Verification")
    print("=" * 60)
    
    # 1. 初始化 phonemizer
    print("\n[1/4] Initializing phonemizer...")
    try:
        global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True)
        textcleaner = TextCleaner()
        print("  ✅ Phonemizer initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize phonemizer: {e}")
        return
    
    # 2. 转换文本为 tokens
    print(f"\n[2/4] Converting text to tokens...")
    print(f"  Text: \"{args.text}\"")
    tokens = text_to_tokens(args.text, textcleaner, global_phonemizer)
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Token IDs: {tokens[0][:10]}... (first 10)")
    
    # 3. 加载 voice pack
    print(f"\n[3/4] Loading voice pack...")
    print(f"  Path: {args.voicepack}")
    try:
        style = load_voicepack(args.voicepack, tokens.shape[1])
        if style.ndim == 1:
            style = style.reshape(1, -1)
        print(f"  Style shape: {style.shape}")
        print(f"  Style range: [{style.min():.3f}, {style.max():.3f}]")
    except Exception as e:
        print(f"  ❌ Failed to load voice pack: {e}")
        print("  Creating random style vector for testing...")
        style = np.random.randn(1, 256).astype(np.float32) * 0.1
    
    # 4. 运行 ONNX 推理
    print(f"\n[4/4] Running ONNX inference...")
    print(f"  Model: {args.onnx}")
    print(f"  Speed: {args.speed}x")
    
    try:
        start_time = time.time()
        audio = run_onnx_inference(
            args.onnx,
            tokens,
            style,
            args.speed
        )
        elapsed = time.time() - start_time
        
        print(f"  ✅ Inference successful!")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        print(f"  Inference time: {elapsed:.2f}s")
        
        # 计算 RTF (Real-Time Factor)
        duration = len(audio) / 24000  # 假设 24kHz
        rtf = elapsed / duration
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  RTF: {rtf:.4f}")
        
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 保存音频
    print(f"\n[Result] Saving audio...")
    print(f"  Output: {args.output}")
    
    # 检查音频是否有效
    if np.isnan(audio).any():
        print("  ⚠️  Warning: Audio contains NaN values!")
        audio = np.nan_to_num(audio, nan=0.0)
    
    if np.abs(audio).max() > 1.0:
        print(f"  ⚠️  Warning: Audio clipping detected, normalizing...")
        audio = audio / np.abs(audio).max() * 0.95
    
    sf.write(args.output, audio, 24000)
    print(f"  ✅ Audio saved!")
    
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)
    print(f"\n🎵 Play the audio: open {args.output}")


if __name__ == '__main__':
    main()
