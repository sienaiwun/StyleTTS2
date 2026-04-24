#!/usr/bin/env python3
"""
StyleTTS2 Lite ONNX 导出脚本 - 借鉴 Kokoro 的方式

核心思路：
1. 创建一个 ONNX 友好的包装类，避免 pack_padded_sequence 等不可追踪操作
2. 对于推理模式（batch_size=1），直接使用 LSTM 而不需要 padding 优化
3. 使用预计算的 style vector（类似 Kokoro 的 ref_s）

输入:
  - tokens: [1, seq_len] token IDs (包含开始/结束标记)
  - style: [1, 256] 风格向量 (128 acoustic + 128 prosody)
  - speed: [1] 语速因子

输出:
  - audio: [num_samples] 音频波形
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path

# 导入原始模型定义
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import *
from utils import recursive_munch


class TextEncoderForONNX(nn.Module):
    """ONNX 兼容的 TextEncoder - 移除 pack_padded_sequence"""
    
    def __init__(self, original: nn.Module):
        super().__init__()
        self.embedding = original.embedding
        self.cnn = original.cnn
        self.lstm = original.lstm
    
    def forward(self, x, text_mask):
        """
        Args:
            x: [1, seq_len] token IDs
            text_mask: [1, seq_len] bool mask (True = masked position)
        Returns:
            [1, channels, seq_len]
        """
        x = self.embedding(x)  # [1, seq_len, emb]
        x = x.transpose(1, 2)  # [1, emb, seq_len]
        
        m = text_mask.unsqueeze(1)  # [1, 1, seq_len]
        x = x.masked_fill(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x = x.masked_fill(m, 0.0)
        
        x = x.transpose(1, 2)  # [1, seq_len, channels]
        
        # 对于 batch_size=1，直接使用 LSTM 而不需要 pack
        x, _ = self.lstm(x)
        
        x = x.transpose(1, 2)  # [1, channels, seq_len]
        x = x.masked_fill(m, 0.0)
        
        return x


class DurationEncoderForONNX(nn.Module):
    """ONNX 兼容的 DurationEncoder - 移除 pack_padded_sequence"""
    
    def __init__(self, original: nn.Module):
        super().__init__()
        self.lstms = original.lstms
        self.dropout = original.dropout
        self.d_model = original.d_model
        self.sty_dim = original.sty_dim
    
    def forward(self, x, style, text_mask):
        """
        Args:
            x: [1, d_model, seq_len] BERT encoder 输出
            style: [1, sty_dim] 风格向量
            text_mask: [1, seq_len] bool mask
        Returns:
            [1, seq_len, d_model]
        """
        masks = text_mask
        seq_len = x.shape[2]
        
        # x: [d_model, 1, seq_len] -> permute
        x = x.permute(2, 0, 1)  # [seq_len, 1, d_model]
        
        # 扩展 style
        s = style.expand(x.shape[0], x.shape[1], -1)  # [seq_len, 1, sty_dim]
        x = torch.cat([x, s], dim=-1)  # [seq_len, 1, d_model + sty_dim]
        
        # mask
        x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        
        x = x.transpose(0, 1)  # [1, seq_len, d_model + sty_dim]
        x = x.transpose(1, 2)  # [1, d_model + sty_dim, seq_len]
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
                x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(1, 2)  # [1, seq_len, channels]
                # 直接使用 LSTM，不用 pack
                x, _ = block(x)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(1, 2)  # [1, channels, seq_len]
        
        return x.transpose(1, 2)  # [1, seq_len, d_model]


class ProsodyPredictorForONNX(nn.Module):
    """ONNX 兼容的 ProsodyPredictor"""
    
    def __init__(self, original: nn.Module):
        super().__init__()
        self.text_encoder = DurationEncoderForONNX(original.text_encoder)
        self.lstm = original.lstm
        self.duration_proj = original.duration_proj
        self.shared = original.shared
        self.F0 = original.F0
        self.N = original.N
        self.F0_proj = original.F0_proj
        self.N_proj = original.N_proj
    
    def forward_duration(self, d_en, style, text_mask):
        """预测 duration"""
        d = self.text_encoder(d_en, style, text_mask)  # [1, seq_len, d_model]
        
        # 直接使用 LSTM
        x, _ = self.lstm(d)
        
        duration = self.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1)  # [1, seq_len]
        
        return duration, d
    
    def forward_f0_n(self, en, style):
        """预测 F0 和 N"""
        # en: [1, d_model, num_frames]
        x, _ = self.shared(en.transpose(1, 2))  # [1, num_frames, d_model]
        x = torch.nan_to_num(x, nan=0.0)
        
        F0 = x.transpose(1, 2)  # [1, d_model, num_frames]
        for block in self.F0:
            F0 = block(F0, style)
        F0 = self.F0_proj(F0)  # [1, 1, num_frames]
        
        N = x.transpose(1, 2)
        for block in self.N:
            N = block(N, style)
        N = self.N_proj(N)  # [1, 1, num_frames]
        
        return F0.squeeze(1), N.squeeze(1)  # [1, num_frames], [1, num_frames]


class StyleTTS2ForONNX(nn.Module):
    """
    StyleTTS2 的 ONNX 导出包装类
    
    类似 Kokoro 的 KModelForONNX，将所有子模块的前向传播整合为单一的 ONNX 可追踪流程
    
    注意: 这个版本不包含扩散采样器（需要外部预计算 style vector）
    """
    
    def __init__(self, model):
        super().__init__()
        # BERT 和编码器
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        
        # 使用 ONNX 兼容版本的编码器
        self.text_encoder = TextEncoderForONNX(model.text_encoder)
        self.predictor = ProsodyPredictorForONNX(model.predictor)
        
        # Decoder
        self.decoder = model.decoder
    
    def forward(
        self,
        tokens: torch.LongTensor,      # [1, seq_len]
        style: torch.FloatTensor,       # [1, 256]
        speed: torch.FloatTensor        # [1]
    ) -> torch.FloatTensor:
        """
        端到端推理
        
        Args:
            tokens: [1, seq_len] Token IDs (已包含 BOS/EOS)
            style: [1, 256] 预计算的风格向量 (前 128 是 acoustic，后 128 是 prosody)
            speed: [1] 语速因子 (1.0 = 正常)
        
        Returns:
            audio: [num_samples] 生成的音频波形
        """
        seq_len = tokens.shape[1]
        
        # 1. 创建 attention mask
        # 对于固定长度输入，mask 全为 False（无 padding）
        text_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=tokens.device)
        
        # 2. BERT encoding for duration prediction
        bert_dur = self.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)  # [1, hidden_dim, seq_len]
        
        # 3. 分离 style 向量
        s = style[:, 128:]   # prosody style [1, 128]
        ref = style[:, :128]  # acoustic reference [1, 128]
        
        # 4. Duration prediction
        duration, d = self.predictor.forward_duration(d_en, s, text_mask)
        duration = duration / speed  # 应用语速
        
        # 5. 生成 alignment
        pred_dur = torch.round(duration).clamp(min=1).long()  # [1, seq_len]
        pred_dur_squeezed = pred_dur.squeeze(0)  # [seq_len]
        
        # 用 cumsum 构建对齐矩阵，避免 repeat_interleave (ORT 1.17 不兼容)
        # pred_dur_squeezed: [seq_len]  每个 phoneme 对应的帧数
        total_frames = pred_dur_squeezed.sum()
        
        # 构建 [seq_len, total_frames] 的对齐矩阵
        # 每一行 i 在 [start_i, start_i + dur_i) 范围内为 1
        # 用 cumsum 计算每个 phoneme 的起始帧
        # frame_idx: [1, total_frames]
        # phoneme_idx: [seq_len, 1]
        frame_idx = torch.arange(total_frames, device=tokens.device).unsqueeze(0)  # [1, T]
        dur_cumsum = torch.cumsum(pred_dur_squeezed, dim=0)  # [seq_len]
        start = torch.cat([torch.zeros(1, dtype=torch.long, device=tokens.device),
                           dur_cumsum[:-1]])  # [seq_len]
        end = dur_cumsum  # [seq_len]
        
        start = start.unsqueeze(1)  # [seq_len, 1]
        end   = end.unsqueeze(1)    # [seq_len, 1]
        
        # [seq_len, total_frames]: 1 where start <= frame_idx < end
        pred_aln_trg = ((frame_idx >= start) & (frame_idx < end)).float()
        pred_aln_trg = pred_aln_trg.unsqueeze(0)  # [1, seq_len, total_frames]
        
        # 6. Prosody encoding
        en = d.transpose(-1, -2) @ pred_aln_trg  # [1, hidden_dim, total_frames]
        F0_pred, N_pred = self.predictor.forward_f0_n(en, s)
        
        # 7. Text encoding
        t_en = self.text_encoder(tokens, text_mask)  # [1, hidden_dim, seq_len]
        asr = t_en @ pred_aln_trg  # [1, hidden_dim, total_frames]
        
        # 8. Decoder
        audio = self.decoder(asr, F0_pred, N_pred, ref)
        
        return audio.squeeze()


def load_model(config_path: str, checkpoint_path: str, device: str = 'cpu'):
    """加载 StyleTTS2 模型"""
    from collections import OrderedDict
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 从 models.py 加载相关函数
    from models import build_model, load_ASR_models, load_F0_models
    from Utils.PLBERT.util import load_plbert
    
    print("Loading ASR model...")
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    print("Loading F0 model...")
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    print("Loading PLBERT model...")
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # 构建模型
    print("Building model...")
    model = build_model(
        recursive_munch(config['model_params']), 
        text_aligner,
        pitch_extractor,
        plbert
    )
    
    # 加载 checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    params_whole = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = params_whole.get('net', params_whole)  # 兼容不同格式
    
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
                print(f'  ✅ {key} loaded')
            except:
                # 处理 module. 前缀
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
                print(f'  ✅ {key} loaded (remapped)')
    
    # 移到目标设备并设置为评估模式
    for key in model:
        model[key].to(device)
        model[key].eval()
    
    return model, config


def export_onnx(
    config_path: str,
    checkpoint_path: str,
    output_path: str = 'styletts2.onnx',
    opset_version: int = 14
):
    """导出 ONNX 模型"""
    
    print("Loading model...")
    model_dict, config = load_model(config_path, checkpoint_path, device='cpu')
    
    # 替换 Decoder 中的 TorchSTFT 为 CustomSTFT（ONNX 兼容版本）
    print("Replacing TorchSTFT with CustomSTFT for ONNX compatibility...")
    from Modules.custom_stft import CustomSTFT
    
    decoder = model_dict['decoder']
    if hasattr(decoder, 'generator') and hasattr(decoder.generator, 'stft'):
        old_stft = decoder.generator.stft
        # 创建新的 CustomSTFT（不使用复数）
        new_stft = CustomSTFT(
            filter_length=old_stft.filter_length,
            hop_length=old_stft.hop_length,
            win_length=old_stft.win_length,
            window='hann'
        )
        decoder.generator.stft = new_stft
        print("  ✅ Replaced TorchSTFT with CustomSTFT")
    
    # 修复 InstanceNorm1d affine=False 问题
    # ONNX 不支持 affine=False 的 InstanceNorm 导出动态通道
    print("Fixing InstanceNorm1d affine=False for ONNX compatibility...")
    def fix_instance_norm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.InstanceNorm1d) and not child.affine:
                # 创建新的 InstanceNorm1d with affine=True
                new_norm = nn.InstanceNorm1d(
                    child.num_features, 
                    eps=child.eps, 
                    momentum=child.momentum,
                    affine=True,  # 关键修复
                    track_running_stats=child.track_running_stats
                )
                # 初始化 affine 参数为恒等变换
                new_norm.weight.data.fill_(1.0)
                new_norm.bias.data.fill_(0.0)
                setattr(module, name, new_norm)
            else:
                fix_instance_norm(child)
    
    for key in model_dict:
        fix_instance_norm(model_dict[key])
    print("  ✅ Fixed InstanceNorm1d affine=True for ONNX")
    
    # 创建复合模型对象
    class ModelWrapper:
        pass
    
    model = ModelWrapper()
    model.bert = model_dict['bert']
    model.bert_encoder = model_dict['bert_encoder']
    model.text_encoder = model_dict['text_encoder']
    model.predictor = model_dict['predictor']
    model.decoder = model_dict['decoder']
    
    print("Creating ONNX wrapper...")
    onnx_model = StyleTTS2ForONNX(model).eval()
    
    # 创建 dummy 输入
    batch_size = 1
    seq_len = 50  # 示例序列长度
    style_dim = 256
    
    dummy_tokens = torch.randint(0, 178, (batch_size, seq_len), dtype=torch.long)
    dummy_tokens[0, 0] = 0  # BOS
    dummy_tokens[0, -1] = 0  # EOS
    
    dummy_style = torch.randn(batch_size, style_dim, dtype=torch.float32)
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)
    
    print(f"Dummy inputs:")
    print(f"  tokens: {dummy_tokens.shape}")
    print(f"  style: {dummy_style.shape}")
    print(f"  speed: {dummy_speed.shape}")
    
    # 测试前向传播
    print("Testing forward pass...")
    with torch.no_grad():
        try:
            output = onnx_model(dummy_tokens, dummy_style, dummy_speed)
            print(f"  output: {output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 导出 ONNX - 使用 TorchScript 模式（关闭 dynamo）
    print(f"Exporting to {output_path}...")
    
    # 现在 InstanceNorm1d 已改为 affine=True，可以安全使用 dynamic_axes
    torch.onnx.export(
        onnx_model,
        (dummy_tokens, dummy_style, dummy_speed),
        output_path,
        input_names=['tokens', 'style', 'speed'],
        output_names=['audio'],
        dynamic_axes={
            'tokens': {1: 'seq_len'},    # tokens 长度可变
            'audio':  {0: 'audio_len'},  # 输出音频长度可变
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # 使用传统 TorchScript 模式
    )
    
    print("Validating ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Successfully exported to {output_path}")
    
    # 打印模型信息
    print(f"\nModel info:")
    print(f"  ONNX opset version: {opset_version}")
    print(f"  Input: tokens [1, seq_len], style [1, 256], speed [1]")
    print(f"  Output: audio [audio_len]")


def main():
    parser = argparse.ArgumentParser(description='Export StyleTTS2 to ONNX (Kokoro style)')
    parser.add_argument('--config', type=str, default='Configs/config.yml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00028.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='styletts2.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    
    args = parser.parse_args()
    
    export_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset
    )


if __name__ == '__main__':
    main()
