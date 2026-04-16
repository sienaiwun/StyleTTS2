#!/bin/bash
# ============================================================
# StyleTTS2 LibriTTS 多说话人训练 - 一键执行脚本
# 在远端 WSL 上运行: bash setup_train_libritts.sh
# ============================================================
set -e

# ---- 配置 ----
PROJECT_DIR="$HOME/StyleTTS2"
DATA_DIR="$HOME/LibriTTS"
VENV_DIR="$HOME/styletts2_env/venv"
NUM_WORKERS=4

echo "============================================"
echo " StyleTTS2 LibriTTS Multi-Speaker Training"
echo "============================================"
echo "Project: $PROJECT_DIR"
echo "Data:    $DATA_DIR"
echo ""

# ---- Step 0: 激活虚拟环境 ----
echo "[Step 0] Activating venv..."
source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---- Step 1: 下载 LibriTTS 数据集 ----
echo ""
echo "[Step 1] Downloading LibriTTS dataset..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 下载 train-clean-100 和 train-clean-360 (约 30GB)
# dev-clean 作为验证集
SUBSETS=("train-clean-100" "train-clean-360" "dev-clean")

for subset in "${SUBSETS[@]}"; do
    TAR_FILE="${subset}.tar.gz"
    if [ -d "$DATA_DIR/$subset" ] || [ -d "$DATA_DIR/LibriTTS/$subset" ]; then
        echo "  ✅ $subset already exists, skipping download"
    else
        URL="https://www.openslr.org/resources/60/${TAR_FILE}"
        echo "  📥 Downloading $subset..."
        wget -c "$URL" -O "$TAR_FILE"
        echo "  📦 Extracting $subset..."
        tar -xzf "$TAR_FILE"
        # LibriTTS 解压后在 LibriTTS/ 子目录
        if [ -d "$DATA_DIR/LibriTTS/$subset" ]; then
            mv "$DATA_DIR/LibriTTS/$subset" "$DATA_DIR/$subset"
        fi
        rm -f "$TAR_FILE"
        echo "  ✅ $subset ready"
    fi
done

# 清理空的 LibriTTS 目录
rmdir "$DATA_DIR/LibriTTS" 2>/dev/null || true

# ---- Step 2: 安装依赖 ----
echo ""
echo "[Step 2] Installing dependencies..."
pip install phonemizer nltk pandas tqdm -q
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# 确保 espeak-ng 已安装
if ! command -v espeak-ng &>/dev/null; then
    echo "  Installing espeak-ng..."
    sudo apt-get update -qq && sudo apt-get install -y -qq espeak-ng
fi
echo "  ✅ Dependencies ready"

# ---- Step 3: 预处理数据 - 生成 phoneme 标注文件 ----
echo ""
echo "[Step 3] Preprocessing LibriTTS data..."
cd "$PROJECT_DIR"

python -c "
import os
import sys
import phonemizer
from tqdm import tqdm
from pathlib import Path
import librosa
import numpy as np

DATA_DIR = '$DATA_DIR'
PROJECT_DIR = '$PROJECT_DIR'
SR = 24000

# 初始化 phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', preserve_punctuation=True, with_stress=True)

def process_subset(subset_name, wav_paths, texts, speaker_ids):
    \"\"\"处理一个子集的所有音频\"\"\"
    results = []
    for wav_path, text, spk_id in tqdm(zip(wav_paths, texts, speaker_ids), 
                                         total=len(wav_paths), desc=subset_name):
        try:
            # 检查音频时长 (过短或过长的跳过)
            info = librosa.get_duration(path=wav_path)
            if info < 1.0 or info > 15.0:
                continue

            # 文本转音素
            ps = global_phonemizer.phonemize([text])[0].strip()
            if len(ps) < 5:
                continue

            # 使用相对路径
            rel_path = os.path.relpath(wav_path, DATA_DIR)
            results.append(f'{rel_path}|{ps}|{spk_id}')
        except Exception as e:
            continue
    return results

# 收集所有 train 子集
train_subsets = ['train-clean-100', 'train-clean-360']
val_subsets = ['dev-clean']

def collect_data(subsets):
    wav_paths = []
    texts = []
    speaker_ids = []
    speaker_map = {}
    spk_counter = 0

    for subset in subsets:
        subset_dir = os.path.join(DATA_DIR, subset)
        if not os.path.exists(subset_dir):
            print(f'  ⚠️ {subset} not found, skipping')
            continue

        for speaker_dir in sorted(os.listdir(subset_dir)):
            speaker_path = os.path.join(subset_dir, speaker_dir)
            if not os.path.isdir(speaker_path):
                continue

            # 分配 speaker ID
            if speaker_dir not in speaker_map:
                speaker_map[speaker_dir] = spk_counter
                spk_counter += 1
            spk_id = speaker_map[speaker_dir]

            for chapter_dir in sorted(os.listdir(speaker_path)):
                chapter_path = os.path.join(speaker_path, chapter_dir)
                if not os.path.isdir(chapter_path):
                    continue

                # 读取转录文本
                trans_file = os.path.join(chapter_path, f'{speaker_dir}_{chapter_dir}.trans.tsv')
                if not os.path.exists(trans_file):
                    # 尝试 normalized 版本
                    trans_file = os.path.join(chapter_path, f'{speaker_dir}_{chapter_dir}.normalized.txt')

                if not os.path.exists(trans_file):
                    continue

                trans = {}
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            trans[parts[0]] = parts[1]
                        else:
                            parts = line.strip().split(' ', 1)
                            if len(parts) >= 2:
                                trans[parts[0]] = parts[1]

                for wav_file in sorted(os.listdir(chapter_path)):
                    if not wav_file.endswith('.wav'):
                        continue
                    utt_id = wav_file.replace('.wav', '')
                    if utt_id in trans:
                        wav_paths.append(os.path.join(chapter_path, wav_file))
                        texts.append(trans[utt_id])
                        speaker_ids.append(spk_id)

    return wav_paths, texts, speaker_ids, speaker_map

print('Collecting training data...')
train_wavs, train_texts, train_spks, speaker_map = collect_data(train_subsets)
print(f'  Training: {len(train_wavs)} utterances, {len(set(train_spks))} speakers')

print('Collecting validation data...')
val_wavs, val_texts, val_spks, _ = collect_data(val_subsets)
# 重新映射 val speaker IDs
val_speaker_map = {}
for i, spk in enumerate(val_spks):
    orig_spk_dir = [k for k, v in speaker_map.items() if v == spk]
    if orig_spk_dir:
        val_spks[i] = speaker_map.get(orig_spk_dir[0], spk)
print(f'  Validation: {len(val_wavs)} utterances')

print('\\nPhonemerizing training data...')
train_results = process_subset('train', train_wavs, train_texts, train_spks)

print('Phonemerizing validation data...')
val_results = process_subset('val', val_wavs, val_texts, val_spks)

# 保存
os.makedirs(os.path.join(PROJECT_DIR, 'Data'), exist_ok=True)

train_list_path = os.path.join(PROJECT_DIR, 'Data', 'train_list_libritts.txt')
val_list_path = os.path.join(PROJECT_DIR, 'Data', 'val_list_libritts.txt')

with open(train_list_path, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(train_results))

with open(val_list_path, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(val_results))

# 保存 speaker map
import json
with open(os.path.join(PROJECT_DIR, 'Data', 'speaker_map_libritts.json'), 'w') as f:
    json.dump(speaker_map, f, indent=2)

print(f'\\n✅ Done!')
print(f'  Train list: {train_list_path} ({len(train_results)} entries)')
print(f'  Val list:   {val_list_path} ({len(val_results)} entries)')
print(f'  Speakers:   {len(speaker_map)}')
"

echo "  ✅ Data preprocessing done"

# ---- Step 4: 创建 LibriTTS 训练配置 ----
echo ""
echo "[Step 4] Creating training config..."

cat > "$PROJECT_DIR/Configs/config_libritts_train.yml" << 'YAML_CONFIG'
log_dir: "Models/LibriTTS"
first_stage_path: "first_stage.pth"
save_freq: 1
log_interval: 10
device: "cuda"
epochs_1st: 50
epochs_2nd: 30
batch_size: 8
max_len: 300
pretrained_model: ""
second_stage_load_pretrained: false
load_only_params: false

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"
PLBERT_dir: 'Utils/PLBERT/'

data_params:
  train_data: "Data/train_list_libritts.txt"
  val_data: "Data/val_list_libritts.txt"
  root_path: "LIBRITTS_ROOT_PLACEHOLDER"
  OOD_data: "Data/OOD_texts.txt"
  min_length: 50

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300

model_params:
  multispeaker: true

  dim_in: 64 
  hidden_dim: 512
  max_conv_dim: 512
  n_layer: 3
  n_mels: 80

  n_token: 178
  max_dur: 50
  style_dim: 128
  
  dropout: 0.2

  decoder: 
      type: 'istftnet'
      resblock_kernel_sizes: [3,7,11]
      upsample_rates :  [10,6]
      upsample_initial_channel: 512
      resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
      upsample_kernel_sizes: [20,12]
      gen_istft_n_fft: 20
      gen_istft_hop_size: 5

  slm:
      model: 'microsoft/wavlm-base-plus'
      sr: 16000
      hidden: 768
      nlayers: 13
      initial_channel: 64
  
  diffusion:
    embedding_mask_proba: 0.1
    transformer:
      num_layers: 3
      num_heads: 8
      head_features: 64
      multiplier: 2
    dist:
      sigma_data: 0.2
      estimate_sigma_data: true
      mean: -3.0
      std: 1.0
  
loss_params:
    lambda_mel: 5.
    lambda_gen: 1.
    lambda_slm: 1.
    
    lambda_mono: 1.
    lambda_s2s: 1.
    TMA_epoch: 5

    lambda_F0: 1.
    lambda_norm: 1.
    lambda_dur: 1.
    lambda_ce: 20.
    lambda_sty: 1.
    lambda_diff: 1.
    
    diff_epoch: 10
    joint_epoch: 15

optimizer_params:
  lr: 0.0001
  bert_lr: 0.00001
  ft_lr: 0.00001
  
slmadv_params:
  min_len: 400
  max_len: 500
  batch_percentage: 0.5
  iter: 20
  thresh: 5
  scale: 0.01
  sig: 1.5
YAML_CONFIG

# 替换数据路径
sed -i "s|LIBRITTS_ROOT_PLACEHOLDER|$DATA_DIR|g" "$PROJECT_DIR/Configs/config_libritts_train.yml"

echo "  ✅ Config created: $PROJECT_DIR/Configs/config_libritts_train.yml"

# ---- Step 5: 创建模型目录 ----
echo ""
echo "[Step 5] Setting up model directory..."
mkdir -p "$PROJECT_DIR/Models/LibriTTS"

# ---- Step 6: 检查 GPU 和环境 ----
echo ""
echo "[Step 6] Environment check..."
python -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA:        {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:        {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
import os
data_dir = '$DATA_DIR'
train_list = '$PROJECT_DIR/Data/train_list_libritts.txt'
if os.path.exists(train_list):
    with open(train_list) as f:
        lines = f.readlines()
    print(f'  Train data:  {len(lines)} utterances')
val_list = '$PROJECT_DIR/Data/val_list_libritts.txt'
if os.path.exists(val_list):
    with open(val_list) as f:
        lines = f.readlines()
    print(f'  Val data:    {len(lines)} utterances')
"

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "To start FIRST STAGE training:"
echo "  cd $PROJECT_DIR"
echo "  python train_first.py -p Configs/config_libritts_train.yml"
echo ""
echo "To start SECOND STAGE training (after first stage):"
echo "  python train_second.py -p Configs/config_libritts_train.yml"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir Models/LibriTTS/tensorboard"
echo "============================================"
