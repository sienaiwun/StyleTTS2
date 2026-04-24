"""
StyleTTS2 Lite - 精简推理模型 (类似 Kokoro)
只包含 5 个核心模块: bert, bert_encoder, predictor, decoder, text_encoder
配合 Voice Pack 使用, 无需 Diffusion / Style Encoder
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from typing import Optional, Union

from text_utils import TextCleaner

import phonemizer


class StyleTTS2Lite(nn.Module):
    """
    StyleTTS2 精简推理模型 - 只保留 5 个核心模块
    类似 Kokoro 的 KModel, 配合 Voice Pack 使用
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_params = config['model_params']

        # 1. BERT (PL-BERT)
        from Utils.PLBERT.util import load_plbert
        plbert = load_plbert(config.get('PLBERT_dir', False))
        self.bert = nn.Linear(plbert.config.hidden_size, model_params['hidden_dim'])
        # 实际上 bert 是完整的 PL-BERT, bert_encoder 是投影层
        # 但为了简洁, 我们在 load_from_checkpoint 里处理

        # 这里用原始的构建方式
        self._build_from_config(model_params, plbert)

    def _build_from_config(self, model_params, plbert):
        from models import build_model
        from utils import recursive_munch

        # 构建 ASR 和 F0 的 dummy (不会使用)
        ASR_config = self.config.get('ASR_config', False)
        ASR_path = self.config.get('ASR_path', False)
        F0_path = self.config.get('F0_path', False)

        from models import load_ASR_models, load_F0_models
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        pitch_extractor = load_F0_models(F0_path)

        full_model = build_model(recursive_munch(model_params), text_aligner, pitch_extractor, plbert)

        # 只保留核心模块
        self.bert = full_model['bert']
        self.bert_encoder = full_model['bert_encoder']
        self.predictor = full_model['predictor']
        self.decoder = full_model['decoder']
        self.text_encoder = full_model['text_encoder']

        # 清理引用
        del full_model

    def load_from_checkpoint(self, checkpoint_path):
        """从精简 checkpoint 加载权重"""
        params = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        for key in ['bert', 'bert_encoder', 'predictor', 'decoder', 'text_encoder']:
            if key in params:
                module = getattr(self, key)
                try:
                    module.load_state_dict(params[key], strict=True)
                    print(f'  ✅ {key}')
                except RuntimeError:
                    # 尝试移除 module. 前缀
                    sd = params[key]
                    if list(sd.keys())[0].startswith('module.'):
                        sd = OrderedDict((k[7:], v) for k, v in sd.items())
                    module.load_state_dict(sd, strict=True)
                    print(f'  ✅ {key} (remapped)')

    @torch.no_grad()
    def forward(self, tokens, ref_s, speed=1.0):
        """
        Forward pass - 类似 Kokoro 的接口
        Args:
            tokens: [1, T] LongTensor of token IDs
            ref_s:  [1, 256] FloatTensor from voice pack
            speed:  float, 语速控制 (1.0=正常, <1=慢, >1=快)
        Returns:
            audio: [samples] FloatTensor
        """
        device = tokens.device
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)

        # Text mask
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

        # Split style vector
        ref = ref_s[:, :128]   # acoustic style → decoder
        s = ref_s[:, 128:]     # prosody style → predictor

        # BERT + encoder
        bert_dur = self.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        # Text encoder
        t_en = self.text_encoder(tokens, input_lengths, text_mask)

        # Duration prediction
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # Alignment (向量化, 类似 Kokoro)
        indices = torch.repeat_interleave(
            torch.arange(tokens.shape[1], device=device), pred_dur)
        pred_aln_trg = torch.zeros(
            (tokens.shape[1], indices.shape[0]), device=device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        # F0 and energy prediction
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        # Decode waveform
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return audio.squeeze(), pred_dur


def synthesize(model, textcleaner, global_phonemizer, text, pack, device, speed=1.0):
    """高层合成接口"""
    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    n_tokens = len(tokens)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    # Voice Pack 查表
    pack_idx = min(n_tokens - 1, len(pack) - 1)
    ref_s = pack[pack_idx].to(device)

    # Forward
    audio, pred_dur = model(tokens, ref_s, speed)
    return audio.cpu().numpy(), pred_dur


def main():
    parser = argparse.ArgumentParser(description='StyleTTS2 Lite Inference (Kokoro-style)')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_lite.wav')
    parser.add_argument('--voicepack', type=str, required=True, help='Voice Pack .pt 路径')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/styletts2_lite.pth',
                        help='精简 checkpoint 路径')
    parser.add_argument('--config', type=str, default='Configs/config.yml')
    parser.add_argument('--speed', type=float, default=1.0, help='语速 (1.0=正常)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 加载配置
    config = yaml.safe_load(open(args.config))

    # 构建精简模型
    print("Building StyleTTS2 Lite model...")
    model = StyleTTS2Lite(config)
    model.eval()
    model.to(device)

    # 加载权重
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_from_checkpoint(args.checkpoint)

    # Phonemizer
    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)
    textcleaner = TextCleaner()

    # 加载 Voice Pack
    print(f"Loading voice pack: {args.voicepack}")
    pack = torch.load(args.voicepack, map_location='cpu', weights_only=True)
    print(f"  Shape: {pack.shape}")

    # 合成
    print(f"\n🎤 Text: \"{args.text}\"")
    print(f"   Speed: {args.speed}x")

    start = time.time()
    wav, pred_dur = synthesize(model, textcleaner, global_phonemizer, args.text, pack, device, args.speed)
    elapsed = time.time() - start
    rtf = elapsed / (len(wav) / 24000)

    sf.write(args.output, wav, 24000)
    print(f"\n✅ Output: {args.output}")
    print(f"   Duration: {len(wav)/24000:.2f}s")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   RTF: {rtf:.4f}")

    # 打印模型统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model stats:")
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Modules: bert, bert_encoder, predictor, decoder, text_encoder")
    print(f"   No Diffusion, No Style Encoder ✅")


if __name__ == '__main__':
    main()
