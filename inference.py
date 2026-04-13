"""
StyleTTS2 推理脚本
用法:
    python inference.py --text "Hello world" --output output.wav
    python inference.py --text "Hello world" --output output.wav --diffusion_steps 10 --embedding_scale 2
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
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

import phonemizer


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


debug_mode = False  # 全局调试开关

def inference(model, sampler, textcleaner, global_phonemizer, text, device,
              noise=None, diffusion_steps=5, embedding_scale=1):
    """合成单句语音"""
    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    debug = debug_mode

    if debug:
        print(f"  [DEBUG] Phonemized: {ps}")

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    if debug:
        print(f"  [DEBUG] Tokens: {tokens.shape} = {tokens.tolist()}")

    if noise is None:
        noise = torch.randn(1, 1, 256).to(device)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        if debug:
            print(f"  [DEBUG] t_en: shape={t_en.shape}, min={t_en.min():.4f}, max={t_en.max():.4f}")

        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        if debug:
            print(f"  [DEBUG] d_en: shape={d_en.shape}, min={d_en.min():.4f}, max={d_en.max():.4f}")

        s_pred = sampler(noise,
                         embedding=bert_dur[0].unsqueeze(0),
                         num_steps=diffusion_steps,
                         embedding_scale=embedding_scale).squeeze(0)

        if debug:
            print(f"  [DEBUG] s_pred: shape={s_pred.shape}, min={s_pred.min():.4f}, max={s_pred.max():.4f}")

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        if debug:
            print(f"  [DEBUG] s (style): min={s.min():.4f}, max={s.max():.4f}")
            print(f"  [DEBUG] ref (acoustic): min={ref.min():.4f}, max={ref.max():.4f}")

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        if debug:
            print(f"  [DEBUG] pred_dur: {pred_dur.tolist()}")
            print(f"  [DEBUG] total frames: {int(pred_dur.sum().data)}")

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        if debug:
            print(f"  [DEBUG] F0_pred: shape={F0_pred.shape}, min={F0_pred.min():.4f}, max={F0_pred.max():.4f}")
            print(f"  [DEBUG] N_pred: shape={N_pred.shape}, min={N_pred.min():.4f}, max={N_pred.max():.4f}")
            has_nan = torch.isnan(F0_pred).any() or torch.isnan(N_pred).any()
            print(f"  [DEBUG] F0/N has NaN: {has_nan}")

        aln_en = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        if debug:
            print(f"  [DEBUG] decoder input (aln_en): shape={aln_en.shape}, min={aln_en.min():.4f}, max={aln_en.max():.4f}")

        out = model.decoder(aln_en, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        if debug:
            out_np = out.squeeze().cpu().numpy()
            print(f"  [DEBUG] output wav: shape={out_np.shape}, min={out_np.min():.4f}, max={out_np.max():.4f}")
            has_nan_out = np.isnan(out_np).any()
            print(f"  [DEBUG] output has NaN: {has_nan_out}")

    return out.squeeze().cpu().numpy()


def LFinference(model, sampler, textcleaner, global_phonemizer, text, device,
                s_prev=None, noise=None, alpha=0.7, diffusion_steps=5, embedding_scale=1):
    """长文本合成（带风格延续）"""
    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    if noise is None:
        noise = torch.randn(1, 1, 256).to(device)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise,
                         embedding=bert_dur[0].unsqueeze(0),
                         num_steps=diffusion_steps,
                         embedding_scale=embedding_scale).squeeze(0)

        if s_prev is not None:
            s_pred = alpha * s_prev + (1 - alpha) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy(), s_pred


def main():
    parser = argparse.ArgumentParser(description='StyleTTS2 Text-to-Speech Inference')
    parser.add_argument('--text', type=str, required=True, help='要合成的文本')
    parser.add_argument('--output', type=str, default='output.wav', help='输出 WAV 文件路径')
    parser.add_argument('--config', type=str, default='Configs/config.yml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='Models/LJSpeech/epoch_2nd_00028.pth',
                        help='模型 checkpoint 路径')
    parser.add_argument('--diffusion_steps', type=int, default=5, help='扩散步数 (5=快速, 10=更多样)')
    parser.add_argument('--embedding_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale (越高情感越强, 建议 1~2)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--long_form', action='store_true', help='启用长文本模式（按句号分割并保持风格连贯）')
    parser.add_argument('--alpha', type=float, default=0.7, help='长文本模式下前后句风格混合比例')
    parser.add_argument('--mel', action='store_true', help='生成 mel 频谱图 (保存为 PNG)')
    parser.add_argument('--debug', action='store_true', help='打印每一步中间值用于调试')
    parser.add_argument('--save_mel_npy', action='store_true', help='保存 mel 频谱为 .npy 文件')
    args = parser.parse_args()

    # 设置调试模式
    global debug_mode
    debug_mode = args.debug

    # 设置随机种子
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

    # 加载文本清洗器
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
            except RuntimeError as e:
                # checkpoint 来自 DataParallel 训练，key 有 module. 前缀
                state_dict = params[key]
                new_state_dict = OrderedDict()

                # 检查是否需要去掉 module. 前缀
                ckpt_keys = list(state_dict.keys())
                model_keys = list(model[key].state_dict().keys())

                if len(ckpt_keys) > 0 and ckpt_keys[0].startswith('module.'):
                    print(f'    ⚠️  Removing "module." prefix for {key}')
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v  # remove 'module.'
                else:
                    # 尝试按位置匹配
                    print(f'    ⚠️  Key mismatch for {key}, remapping by position')
                    print(f'       model keys: {len(model_keys)}, ckpt keys: {len(ckpt_keys)}')
                    for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                        new_state_dict[k_m] = v_c

                try:
                    model[key].load_state_dict(new_state_dict, strict=True)
                    print(f'    ✅ {key} loaded after key remapping')
                except RuntimeError as e2:
                    print(f'    ❌ {key} FAILED to load: {e2}')

    _ = [model[key].eval() for key in model]

    # 创建 diffusion sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    # 合成语音
    print(f"\nSynthesizing: \"{args.text}\"")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print(f"  Embedding scale: {args.embedding_scale}")

    start = time.time()

    if args.long_form:
        # 长文本模式：按句号分割
        sentences = args.text.split('.')
        wavs = []
        s_prev = None
        for sent in sentences:
            if sent.strip() == "":
                continue
            sent = sent.strip() + '.'
            noise = torch.randn(1, 1, 256).to(device)
            wav, s_prev = LFinference(
                model, sampler, textcleaner, global_phonemizer, sent, device,
                s_prev=s_prev, noise=noise, alpha=args.alpha,
                diffusion_steps=args.diffusion_steps, embedding_scale=args.embedding_scale)
            wavs.append(wav)
        wav = np.concatenate(wavs)
    else:
        # 单句模式
        noise = torch.randn(1, 1, 256).to(device)
        wav = inference(
            model, sampler, textcleaner, global_phonemizer, args.text, device,
            noise=noise, diffusion_steps=args.diffusion_steps,
            embedding_scale=args.embedding_scale)

    elapsed = time.time() - start
    rtf = elapsed / (len(wav) / 24000)

    # 保存音频
    sf.write(args.output, wav, 24000)

    # 生成 mel 频谱图
    if args.mel:
        import matplotlib
        matplotlib.use('Agg')  # 无 GUI 后端
        import matplotlib.pyplot as plt

        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        wav_tensor = torch.from_numpy(wav).float()
        mel_spec = mel_spec_transform(wav_tensor)
        mel_spec_db = torch.log(mel_spec + 1e-5)

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(mel_spec_db.numpy(), aspect='auto', origin='lower',
                       interpolation='nearest', cmap='magma')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mel Channel')
        ax.set_title(f'Mel Spectrogram: "{args.text}"')
        fig.colorbar(im, ax=ax, format='%.1f')
        plt.tight_layout()

        mel_path = args.output.rsplit('.', 1)[0] + '_mel.png'
        fig.savefig(mel_path, dpi=150)
        plt.close(fig)
        print(f"  Mel spectrogram: {mel_path}")

    # 保存 mel 为 npy 文件（可用于外部 vocoder）
    if args.save_mel_npy:
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        wav_tensor = torch.from_numpy(wav).float()
        mel_spec = mel_spec_transform(wav_tensor)
        mel_spec_db = torch.log(mel_spec + 1e-5)
        npy_path = args.output.rsplit('.', 1)[0] + '_mel.npy'
        np.save(npy_path, mel_spec_db.numpy())
        print(f"  Mel .npy saved: {npy_path} (shape: {mel_spec_db.shape})")

    print(f"\n✅ Done!")
    print(f"  Output: {args.output}")
    print(f"  Duration: {len(wav)/24000:.2f}s")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  RTF: {rtf:.4f}")


if __name__ == '__main__':
    main()
