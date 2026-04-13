#!/bin/bash
# =============================================================================
# StyleTTS2 Remote WSL Setup Script
# 适用于: RTX 5090 (32GB VRAM) + WSL2 Ubuntu
# 功能: 1) 创建Python venv  2) 解压/重采样LJSpeech  3) 启动iSTFTNet训练
# 用法: bash setup_remote_wsl.sh
# =============================================================================

set -e  # 遇到错误立即退出

# ---------- 颜色输出 ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# 配置区 - 按需修改
# =============================================================================
PROJECT_DIR="$HOME/StyleTTS2"                          # WSL 内项目目录
VENV_DIR="$PROJECT_DIR/venv"                           # virtualenv 路径
DATA_TAR="/mnt/e/NewStyleTTS2/StyleTTS2/Demo/LJSpeech-1.1.tar.bz2"  # Windows 侧压缩包 (WSL路径)
LJSPEECH_DEST="$PROJECT_DIR/LJSpeech-1.1"             # 解压目标目录
WAVS_RAW="$LJSPEECH_DEST/wavs"                        # 原始wav目录 (22050Hz)
WAVS_24K="$LJSPEECH_DEST/wavs"                        # 重采样后覆盖原目录(24000Hz)
TARGET_SR=24000                                         # StyleTTS2 目标采样率
PYTHON_BIN="python3.10"                                 # 推荐 Python 3.10
CUDA_VERSION="12.1"                                     # WSL CUDA 版本 (torch对应)

# =============================================================================
# STEP 0: 系统检查
# =============================================================================
echo ""
echo -e "${CYAN}========================================================${NC}"
echo -e "${CYAN}  StyleTTS2 WSL Setup for RTX 5090                     ${NC}"
echo -e "${CYAN}========================================================${NC}"
echo ""

info "检查系统环境..."

# 检查 WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    warn "未检测到 WSL 环境，仍继续（原生 Linux 也适用）"
fi

# 检查 GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    success "检测到 GPU: $GPU_NAME | 显存: $GPU_MEM"
else
    error "未检测到 nvidia-smi，请确保 WSL2 已正确安装 NVIDIA 驱动和 CUDA"
fi

# 检查数据包
if [ ! -f "$DATA_TAR" ]; then
    error "找不到数据集: $DATA_TAR\n请确认 Windows 路径 E:\\NewStyleTTS2\\StyleTTS2\\Demo\\LJSpeech-1.1.tar.bz2 存在"
fi
success "找到数据集: $DATA_TAR"

# =============================================================================
# STEP 1: 安装系统依赖
# =============================================================================
echo ""
info "STEP 1/6: 安装系统依赖..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.10 python3.10-venv python3.10-dev \
    python3-pip \
    libsndfile1 libsndfile1-dev \
    ffmpeg \
    sox \
    git \
    build-essential \
    tmux \
    wget curl \
    2>/dev/null || true

success "系统依赖安装完成"

# =============================================================================
# STEP 2: 克隆/更新项目代码
# =============================================================================
echo ""
info "STEP 2/6: 准备项目代码..."

if [ ! -d "$PROJECT_DIR" ]; then
    info "克隆 StyleTTS2 仓库..."
    git clone https://github.com/yl4579/StyleTTS2.git "$PROJECT_DIR"
    success "项目克隆完成: $PROJECT_DIR"
else
    warn "项目目录已存在，跳过克隆: $PROJECT_DIR"
    # 如需更新，取消下面注释:
    # cd "$PROJECT_DIR" && git pull
fi

cd "$PROJECT_DIR"

# =============================================================================
# STEP 3: 创建 Python venv 并安装依赖
# =============================================================================
echo ""
info "STEP 3/6: 创建 Python 虚拟环境..."

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
    success "venv 创建于: $VENV_DIR"
else
    warn "venv 已存在，跳过创建: $VENV_DIR"
fi

# 激活 venv
source "$VENV_DIR/bin/activate"
info "已激活 venv: $(which python)"

# 升级 pip
pip install --upgrade pip -q

# 安装 PyTorch (支持 CUDA 12.1, 兼容 RTX 5090)
info "安装 PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
success "PyTorch 安装完成: $(python -c 'import torch; print(torch.__version__)')"

# 验证 CUDA 可用
python -c "
import torch
if torch.cuda.is_available():
    print(f'  [OK] CUDA 可用: {torch.cuda.get_device_name(0)}')
    print(f'  [OK] CUDA 版本: {torch.version.cuda}')
    print(f'  [OK] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  [WARN] CUDA 不可用，请检查驱动')
"

# 安装其他依赖
info "安装项目依赖..."
pip install \
    SoundFile \
    munch \
    pydub \
    pyyaml \
    librosa \
    nltk \
    matplotlib \
    accelerate \
    transformers \
    einops \
    einops-exts \
    tqdm \
    typing-extensions \
    phonemizer \
    -q

# monotonic_align 需要编译
info "安装 monotonic_align..."
pip install git+https://github.com/resemble-ai/monotonic_align.git -q
success "所有依赖安装完成"

# =============================================================================
# STEP 4: 解压 LJSpeech 数据集
# =============================================================================
echo ""
info "STEP 4/6: 解压 LJSpeech 数据集..."

if [ -d "$LJSPEECH_DEST/wavs" ] && [ "$(ls -A $LJSPEECH_DEST/wavs 2>/dev/null | wc -l)" -gt 10000 ]; then
    warn "数据集已解压，跳过: $LJSPEECH_DEST ($(ls $LJSPEECH_DEST/wavs | wc -l) 个wav文件)"
else
    info "正在解压 (约600MB，请稍候)..."
    mkdir -p "$PROJECT_DIR"
    tar -xjf "$DATA_TAR" -C "$PROJECT_DIR"
    success "解压完成: $LJSPEECH_DEST"
fi

WAV_COUNT=$(ls "$WAVS_RAW"/*.wav 2>/dev/null | wc -l)
info "原始 wav 文件数量: $WAV_COUNT"

# =============================================================================
# STEP 5: 重采样到 24000 Hz
# =============================================================================
echo ""
info "STEP 5/6: 重采样音频到 ${TARGET_SR} Hz..."

# 检查是否已重采样(通过抽查第一个文件的采样率)
FIRST_WAV=$(ls "$WAVS_RAW"/*.wav | head -1)
CURRENT_SR=$(python -c "
import soundfile as sf
info = sf.info('$FIRST_WAV')
print(info.samplerate)
" 2>/dev/null || echo "0")

if [ "$CURRENT_SR" = "$TARGET_SR" ]; then
    success "音频已是 ${TARGET_SR} Hz，跳过重采样"
else
    info "当前采样率: ${CURRENT_SR} Hz → 目标: ${TARGET_SR} Hz"
    info "开始批量重采样 (${WAV_COUNT} 个文件，使用多线程)..."

    python - << 'PYEOF'
import os
import sys
import librosa
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

WAVS_DIR = os.environ.get('WAVS_24K', os.path.expanduser('~/StyleTTS2/LJSpeech-1.1/wavs'))
TARGET_SR = int(os.environ.get('TARGET_SR', 24000))

wav_files = sorted(Path(WAVS_DIR).glob('*.wav'))
print(f"  共 {len(wav_files)} 个wav文件，目标采样率: {TARGET_SR} Hz")
print(f"  使用多线程 (workers=8) 加速处理...")

def resample_file(wav_path):
    try:
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        sf.write(str(wav_path), y, TARGET_SR, subtype='PCM_16')
        return True, str(wav_path.name)
    except Exception as e:
        return False, f"{wav_path.name}: {e}"

ok_count = 0
err_count = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(resample_file, p): p for p in wav_files}
    with tqdm(total=len(wav_files), desc="Resampling", unit="file") as pbar:
        for future in as_completed(futures):
            ok, msg = future.result()
            if ok:
                ok_count += 1
            else:
                err_count += 1
                print(f"\n  [ERR] {msg}")
            pbar.update(1)

print(f"\n  重采样完成: {ok_count} 成功, {err_count} 失败")
PYEOF

    export WAVS_24K="$WAVS_24K"
    export TARGET_SR="$TARGET_SR"
    success "重采样完成"
fi

# =============================================================================
# STEP 6: 生成训练列表 & 修改配置 & 启动训练
# =============================================================================
echo ""
info "STEP 6/6: 准备训练配置..."

cd "$PROJECT_DIR"

# 生成 train_list.txt 和 val_list.txt (若不存在)
if [ ! -f "Data/train_list.txt" ] || [ ! -s "Data/train_list.txt" ]; then
    info "生成训练/验证列表..."
    python - << 'PYEOF'
import os
import random
from pathlib import Path

project_dir = os.path.expanduser('~/StyleTTS2')
wavs_dir = Path(project_dir) / 'LJSpeech-1.1' / 'wavs'
metadata_file = Path(project_dir) / 'LJSpeech-1.1' / 'metadata.csv'
data_dir = Path(project_dir) / 'Data'
data_dir.mkdir(exist_ok=True)

# 读取 metadata
entries = []
with open(metadata_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            stem = parts[0]
            text = parts[2] if len(parts) >= 3 else parts[1]
            wav_path = wavs_dir / f"{stem}.wav"
            if wav_path.exists():
                entries.append(f"{wav_path}|{text}")

random.seed(42)
random.shuffle(entries)
split = max(1, int(len(entries) * 0.02))   # 2% 作验证集
val_entries   = entries[:split]
train_entries = entries[split:]

with open(data_dir / 'train_list.txt', 'w') as f:
    f.write('\n'.join(train_entries))
with open(data_dir / 'val_list.txt', 'w') as f:
    f.write('\n'.join(val_entries))

print(f"  训练集: {len(train_entries)} 条")
print(f"  验证集: {len(val_entries)} 条")
print(f"  列表已写入 Data/train_list.txt 和 Data/val_list.txt")
PYEOF
    success "训练列表生成完成"
else
    warn "训练列表已存在，跳过生成"
fi

# 为 RTX 5090 (32GB) 生成两份分阶段配置
info "生成 Stage1 配置 (无WavLM, batch=24)..."

cat > "$PROJECT_DIR/Configs/config_5090_stage1.yml" << YAML
# ============================================================
# StyleTTS2 训练配置 - 针对 RTX 5090 32GB 优化 (iSTFTNet)
# ============================================================
log_dir: "Models/LJSpeech_5090"
first_stage_path: "first_stage.pth"
save_freq: 2
log_interval: 10
device: "cuda"

# RTX 5090 32GB: 可大幅提升 batch_size 和 max_len
epochs_1st: 200        # 第一阶段完整训练
epochs_2nd: 100        # 第二阶段完整训练
batch_size: 16         # 32GB显存可跑 batch=16
max_len: 400           # 5秒片段，充分利用显存

pretrained_model: ""
second_stage_load_pretrained: false
load_only_params: false

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"
PLBERT_dir: 'Utils/PLBERT/'

data_params:
  train_data: "Data/train_list.txt"
  val_data: "Data/val_list.txt"
  root_path: ""          # 留空: 列表中已包含完整路径
  OOD_data: "Data/OOD_texts.txt"
  min_length: 50

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300

model_params:
  multispeaker: false
  dim_in: 64
  hidden_dim: 512
  max_conv_dim: 512
  n_layer: 3
  n_mels: 80
  n_token: 178
  max_dur: 50
  style_dim: 128
  dropout: 0.2

  # iSTFTNet decoder (更快、显存更省)
  decoder:
    type: 'istftnet'
    resblock_kernel_sizes: [3,7,11]
    upsample_rates: [10, 6]
    upsample_initial_channel: 512
    resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
    upsample_kernel_sizes: [20, 12]
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
  lambda_mel: 5.0
  lambda_gen: 1.0
  lambda_slm: 1.0
  lambda_mono: 1.0
  lambda_s2s: 1.0
  TMA_epoch: 50       # 从第50 epoch 开启 TMA (第一阶段)
  TMA_CEloss: false

optimizer_params:
  lr: 0.0001
  bert_lr: 0.00001
  ft_lr: 0.0001
  wd: 1.0e-4

slmadv_params:
  min_len: 400
  max_len: 500
  batch_percentage: 0.5
  iter: 10
  thresh: 5
  scale: 0.01
  sig: 1.5
YAML

success "配置文件生成: Configs/config_5090_stage1.yml"

# ---- Stage 2 配置: 加入 WavLM 判别器，显存多用 ~10GB，减小 batch ----
info "生成 Stage2 配置 (含WavLM判别器, batch=8)..."

cat > "$PROJECT_DIR/Configs/config_5090_stage2.yml" << YAML
# ============================================================
# StyleTTS2 Stage2 配置 - RTX 5090 32GB (iSTFTNet + WavLM)
# WavLM wavlm-base-plus 约占 1.2GB 常驻显存
# 对抗训练激活时每步额外前向约 8GB，batch 需要缩小
# ============================================================
log_dir: "Models/LJSpeech_5090"
first_stage_path: "first_stage.pth"
save_freq: 2
log_interval: 10
device: "cuda"

epochs_1st: 200        # Stage2 不用 train_first，此项仅占位
epochs_2nd: 100        # Stage2 完整 100 epoch
batch_size: 8          # WavLM 判别器额外占用，batch 降为 8
max_len: 400           # 保持 5 秒片段

# Stage2 从 Stage1 最终权重续训
pretrained_model: "Models/LJSpeech_5090/first_stage.pth"
second_stage_load_pretrained: true
load_only_params: true  # 只加载参数，重置 optimizer

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"
PLBERT_dir: 'Utils/PLBERT/'

data_params:
  train_data: "Data/train_list.txt"
  val_data: "Data/val_list.txt"
  root_path: ""
  OOD_data: "Data/OOD_texts.txt"
  min_length: 50

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300

model_params:
  multispeaker: false
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
    upsample_rates: [10, 6]
    upsample_initial_channel: 512
    resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
    upsample_kernel_sizes: [20, 12]
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
  lambda_mel: 5.0
  lambda_gen: 1.0
  lambda_slm: 1.0
  lambda_mono: 1.0
  lambda_s2s: 1.0
  TMA_epoch: 0          # Stage2 从第0 epoch 就开启 TMA
  TMA_CEloss: false
  diff_epoch: 20        # 第20 epoch 后开始 diffusion 训练
  joint_epoch: 50       # 第50 epoch 后 joint 训练

optimizer_params:
  lr: 0.0001
  bert_lr: 0.00001
  ft_lr: 0.0001
  wd: 1.0e-4

slmadv_params:
  min_len: 400
  max_len: 500
  batch_percentage: 0.5
  iter: 10
  thresh: 5
  scale: 0.01
  sig: 1.5
YAML

success "配置文件生成: Configs/config_5090_stage2.yml"

# 创建模型输出目录
mkdir -p "$PROJECT_DIR/Models/LJSpeech_5090"

# =============================================================================
# 启动训练 (使用 tmux 后台运行)
# =============================================================================
echo ""
info "准备在 tmux 会话中启动训练..."

# 检查 tmux
if ! command -v tmux &>/dev/null; then
    sudo apt-get install -y tmux -q
fi

# 写入训练启动脚本
cat > "$PROJECT_DIR/run_training_5090.sh" << 'TRAINEOF'
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

# Stage1: 无WavLM判别器 → batch=24, max_len=400
CONFIG_S1="Configs/config_5090_stage1.yml"
# Stage2: 含WavLM判别器 (~10GB额外显存) → batch=8, max_len=400
CONFIG_S2="Configs/config_5090_stage2.yml"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

echo "========================================"
echo "  StyleTTS2 iSTFTNet Training (RTX 5090)"
echo "  GPU : $GPU_NAME ($GPU_MEM)"
echo "  S1  : $CONFIG_S1  [batch=24]"
echo "  S2  : $CONFIG_S2  [batch=8]"
echo "  Time: $(date)"
echo "========================================"

echo ""
echo "[STAGE 1/2] 第一阶段训练 (acoustic model, no WavLM)..."
echo "  → batch=24, max_len=400, epochs=200"
python train_first.py --config_path "$CONFIG_S1" 2>&1 | tee logs/train_first_$(date +%Y%m%d_%H%M).log

# Stage1 完成后，将最终模型复制为 first_stage.pth 供 Stage2 加载
LAST_S1=$(ls -t Models/LJSpeech_5090/epoch_1st_*.pth 2>/dev/null | head -1)
if [ -n "$LAST_S1" ]; then
    cp "$LAST_S1" Models/LJSpeech_5090/first_stage.pth
    echo "  → 已将 $LAST_S1 复制为 first_stage.pth"
fi

echo ""
echo "[STAGE 2/2] 第二阶段训练 (+ WavLM SLM判别器)..."
echo "  → batch=8, max_len=400, epochs=100"
python train_second.py --config_path "$CONFIG_S2" 2>&1 | tee logs/train_second_$(date +%Y%m%d_%H%M).log

echo ""
echo "=============================="
echo "  训练全部完成! $(date)"
echo "=============================="
TRAINEOF

chmod +x "$PROJECT_DIR/run_training_5090.sh"
mkdir -p "$PROJECT_DIR/logs"

# 启动 tmux 会话
SESSION_NAME="styletts2_train"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    warn "tmux 会话 '$SESSION_NAME' 已存在，跳过启动"
    warn "可运行: tmux attach -t $SESSION_NAME 查看进度"
else
    info "在 tmux 会话 '$SESSION_NAME' 中启动训练..."
    tmux new-session -d -s "$SESSION_NAME" \
        "cd $PROJECT_DIR && bash run_training_5090.sh; exec bash"
    success "训练已在后台启动!"
    echo ""
    echo -e "${GREEN}======================================================${NC}"
    echo -e "${GREEN}  训练已启动！查看进度:${NC}"
    echo -e "${GREEN}    tmux attach -t $SESSION_NAME${NC}"
    echo -e "${GREEN}  退出 tmux (不中断训练):${NC}"
    echo -e "${GREEN}    Ctrl+B 然后按 D${NC}"
    echo -e "${GREEN}  实时监控 GPU:${NC}"
    echo -e "${GREEN}    watch -n 2 nvidia-smi${NC}"
    echo -e "${GREEN}======================================================${NC}"
fi

echo ""
success "所有步骤完成！"
echo ""
echo "  项目目录:  $PROJECT_DIR"
echo "  数据集:    $LJSPEECH_DEST  ($(ls $WAVS_RAW/*.wav | wc -l) 个wav, ${TARGET_SR}Hz)"
echo "  虚拟环境:  $VENV_DIR"
echo ""
echo "  训练配置:"
echo "    Stage1 (batch=24): $PROJECT_DIR/Configs/config_5090_stage1.yml"
echo "    Stage2 (batch= 8): $PROJECT_DIR/Configs/config_5090_stage2.yml"
echo ""
echo "  模型输出:  $PROJECT_DIR/Models/LJSpeech_5090/"
echo "  训练日志:  $PROJECT_DIR/logs/"
echo ""
echo -e "${YELLOW}  显存预估:"
echo "    Stage1 (无WavLM): ~18-22 GB  → batch=24 安全"
echo "    Stage2 (含WavLM): ~26-30 GB  → batch=8  安全${NC}"
echo ""
