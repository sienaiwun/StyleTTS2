#!/bin/bash
# =============================================================
#  StyleTTS2 - WSL2 环境配置脚本
#  用法: bash /mnt/d/StyleTTS2/setup_wsl.sh
# =============================================================

set -e  # 任意命令出错即停止

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_info() { echo -e "${CYAN}[>>]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
log_err()  { echo -e "${RED}[错误]${NC} $1"; exit 1; }

# ── 路径配置 ─────────────────────────────────────────────────
WIN_PROJECT="/mnt/d/StyleTTS2"       # Windows 项目路径（WSL 挂载点）
WSL_PROJECT="$HOME/StyleTTS2"        # WSL 目标路径（高速内部文件系统）

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  StyleTTS2 WSL2 环境配置脚本${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
echo -e "  Windows 项目: ${YELLOW}$WIN_PROJECT${NC}"
echo -e "  WSL 目标路径: ${YELLOW}$WSL_PROJECT${NC}"
echo ""

# ── 检查 Windows 挂载点是否可访问 ────────────────────────────
log_info "检查 Windows 挂载点..."
if [ ! -d "$WIN_PROJECT" ]; then
    log_err "无法访问 $WIN_PROJECT\n       请确认：\n       1. Windows 盘符 D: 存在\n       2. 项目路径正确\n       3. WSL 已正确挂载 /mnt/d"
fi
log_ok "Windows 挂载点可访问"

# ── 1. 安装系统依赖 ───────────────────────────────────────────
echo ""
log_info "更新 apt 软件源..."
sudo apt-get update -q

log_info "安装系统依赖 (gcc, python3, espeak-ng 等)..."
sudo apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    espeak-ng-data \
    rsync

log_ok "系统依赖安装完成"

# ── 2. 验证 GPU / CUDA ───────────────────────────────────────
echo ""
log_info "检查 CUDA 支持..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    log_ok "CUDA GPU 可用"
else
    log_warn "未检测到 nvidia-smi，将使用 CPU 训练（速度很慢，不推荐）"
    log_warn "请确认：NVIDIA 驱动版本 >= 515，且已启用 WSL2 GPU 支持"
fi

# ── 3. 复制项目代码到 WSL 内部文件系统 ──────────────────────
echo ""
log_info "复制项目代码到 WSL 内部文件系统..."
log_warn "注意: 在 WSL 内部运行比跨文件系统快 5~10 倍"
echo ""

COPY_PROJECT=false

if [ -d "$WSL_PROJECT" ]; then
    echo -n "  $WSL_PROJECT 已存在，是否覆盖更新？[y/N] "
    read -r REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        COPY_PROJECT=true
    else
        log_info "跳过项目复制，使用现有目录"
    fi
else
    COPY_PROJECT=true
fi

if [ "$COPY_PROJECT" = true ]; then
    mkdir -p "$WSL_PROJECT"
    log_info "复制项目代码（排除 LJSpeech 数据集、venv、Models）..."
    rsync -ah --progress \
        --exclude="LJSpeech-1.1/" \
        --exclude="venv/" \
        --exclude="__pycache__/" \
        --exclude="*.pyc" \
        --exclude="Models/" \
        "$WIN_PROJECT/" "$WSL_PROJECT/"
    log_ok "项目代码复制完成"
fi

# ── 4. 复制 LJSpeech 数据集 ──────────────────────────────────
echo ""
WIN_DATA="$WIN_PROJECT/LJSpeech-1.1"
WSL_DATA="$WSL_PROJECT/LJSpeech-1.1"

if [ ! -d "$WIN_DATA" ]; then
    log_warn "未找到 LJSpeech-1.1 数据集: $WIN_DATA"
    log_warn "跳过数据集复制，请手动复制或下载完整数据集"
    log_warn "下载地址: https://keithito.com/LJ-Speech-Dataset/"
else
    COPY_DATA=false

    if [ -d "$WSL_DATA" ]; then
        echo -n "  LJSpeech-1.1 已存在于 WSL，是否重新复制？[y/N] "
        read -r REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            COPY_DATA=true
        else
            log_info "跳过数据集复制，使用现有目录"
        fi
    else
        COPY_DATA=true
    fi

    if [ "$COPY_DATA" = true ]; then
        WAV_COUNT=$(find "$WIN_DATA/wavs" -name "*.wav" 2>/dev/null | wc -l)
        log_info "正在复制 LJSpeech-1.1 数据集 (共 $WAV_COUNT 个 wav 文件)..."
        log_info "这可能需要几分钟，请耐心等待..."
        rsync -ah --progress "$WIN_DATA/" "$WSL_DATA/"
        log_ok "LJSpeech-1.1 数据集复制完成 -> $WSL_DATA"
    fi
fi

# ── 5. 创建 Python 虚拟环境 ──────────────────────────────────
echo ""
log_info "创建 Python 虚拟环境..."
cd "$WSL_PROJECT"

VENV_DIR="$WSL_PROJECT/venv"
if [ -d "$VENV_DIR" ]; then
    log_info "venv 已存在，跳过创建"
else
    python3 -m venv venv
    log_ok "虚拟环境创建完成: $VENV_DIR"
fi

source venv/bin/activate
log_ok "虚拟环境已激活"

# ── 6. 升级 pip ───────────────────────────────────────────────
pip install --upgrade pip -q
log_ok "pip 已升级"

# ── 7. 安装 PyTorch (CUDA 11.8) ───────────────────────────────
echo ""
log_info "安装 PyTorch 2.2.2 + torchaudio (CUDA 11.8)..."
log_warn "如需其他 CUDA 版本，修改下方 URL："
log_warn "  CUDA 12.1 -> https://download.pytorch.org/whl/cu121"
log_warn "  CUDA 12.4 -> https://download.pytorch.org/whl/cu124"
log_warn "  仅 CPU    -> https://download.pytorch.org/whl/cpu"
echo ""

pip install torch==2.2.2+cu118 torchaudio==2.2.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

log_ok "PyTorch 安装完成"

# ── 8. 安装 Python 依赖 ───────────────────────────────────────
echo ""
log_info "安装 StyleTTS2 所需依赖包..."

pip install \
    SoundFile \
    munch \
    pydub \
    pyyaml \
    librosa \
    click \
    "numpy==1.26.4" \
    scipy \
    pandas \
    tensorboard \
    tqdm \
    phonemizer \
    matplotlib \
    accelerate \
    transformers \
    einops \
    einops-exts \
    nltk \
    typing_extensions

log_ok "依赖包安装完成"

# ── 9. 安装 monotonic_align ───────────────────────────────────
echo ""
log_info "安装 monotonic_align (需要 gcc 编译)..."
pip install git+https://github.com/resemble-ai/monotonic_align.git
log_ok "monotonic_align 安装完成"

# ── 10. 预处理：生成训练/验证列表 ────────────────────────────
echo ""
log_info "检查并生成 LJSpeech 训练数据列表..."

PREPROCESS_SCRIPT="$WSL_PROJECT/prepare_lj_data.py"

# 写出预处理脚本
cat > "$PREPROCESS_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
为 StyleTTS2 准备 LJSpeech 训练/验证数据列表
格式: filename.wav|IPA音素|说话人ID(0)

依赖: phonemizer
"""
import os
import sys
import random
from pathlib import Path

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
except ImportError:
    print("[错误] 请先安装 phonemizer: pip install phonemizer")
    sys.exit(1)

# ── 配置 ──────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
LJ_DIR      = SCRIPT_DIR / "LJSpeech-1.1"
METADATA    = LJ_DIR / "metadata.csv"
DATA_DIR    = SCRIPT_DIR / "Data"
TRAIN_LIST  = DATA_DIR / "train_list.txt"
VAL_LIST    = DATA_DIR / "val_list.txt"
VAL_SIZE    = 100         # 验证集大小
RANDOM_SEED = 42

if not METADATA.exists():
    print(f"[错误] 未找到 {METADATA}")
    print("请确认 LJSpeech-1.1 数据集已复制到 WSL 项目目录下")
    sys.exit(1)

DATA_DIR.mkdir(exist_ok=True)

print(f"[>>] 读取 metadata.csv ...")
entries = []
with open(METADATA, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            wav_name = parts[0] + ".wav"
            text     = parts[1]          # 原始文本（非标准化）
            wav_path = LJ_DIR / "wavs" / wav_name
            if wav_path.exists():
                entries.append((wav_name, text))

print(f"[>>] 共找到 {len(entries)} 个有效音频")

print(f"[>>] 用 espeak-ng 进行 IPA 音素转换（可能需要几分钟）...")
texts = [e[1] for e in entries]

backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)
ipa_list = backend.phonemize(texts, njobs=4)

print(f"[>>] 音素转换完成，写出训练列表...")

# 打乱顺序并拆分
random.seed(RANDOM_SEED)
indices = list(range(len(entries)))
random.shuffle(indices)

val_indices   = set(indices[:VAL_SIZE])
train_indices = set(indices[VAL_SIZE:])

with open(TRAIN_LIST, "w", encoding="utf-8") as f_train, \
     open(VAL_LIST,   "w", encoding="utf-8") as f_val:
    for i, (wav_name, _) in enumerate(entries):
        ipa = ipa_list[i].strip()
        line = f"{wav_name}|{ipa}|0\n"
        if i in val_indices:
            f_val.write(line)
        else:
            f_train.write(line)

print(f"[OK] 训练集: {len(train_indices)} 条 -> {TRAIN_LIST}")
print(f"[OK] 验证集: {VAL_SIZE} 条           -> {VAL_LIST}")
print("[OK] 数据准备完成！")
PYEOF

log_ok "预处理脚本已生成: $PREPROCESS_SCRIPT"

# ── 11. 修改 config.yml 中的数据路径为 WSL 本地路径 ──────────
echo ""
log_info "更新 config.yml 的数据集路径为 WSL 本地路径..."

CONFIG="$WSL_PROJECT/Configs/config.yml"
if [ -f "$CONFIG" ]; then
    # 替换 root_path 为 WSL 本地路径
    sed -i "s|root_path:.*|root_path: \"$WSL_PROJECT/LJSpeech-1.1/wavs\"|" "$CONFIG"
    log_ok "config.yml 路径已更新: root_path -> $WSL_PROJECT/LJSpeech-1.1/wavs"
else
    log_warn "未找到 $CONFIG，请手动修改 root_path"
fi

# ── 12. 创建训练启动脚本 train.sh ─────────────────────────────
echo ""
log_info "创建 train.sh 训练启动脚本..."

cat > "$WSL_PROJECT/train.sh" << TRAINEOF
#!/bin/bash
# StyleTTS2 LJSpeech 训练脚本
# 用法: bash train.sh [first|second|both]
# ─────────────────────────────────────────
set -e

STAGE=\${1:-both}  # 默认训练两个阶段
PROJECT_DIR="\$HOME/StyleTTS2"
VENV_DIR="\$PROJECT_DIR/venv"

cd "\$PROJECT_DIR"
source "\$VENV_DIR/bin/activate"

echo ""
echo "============================================"
echo "  StyleTTS2 训练启动器"
echo "  阶段: \$STAGE"
echo "============================================"
echo ""

# 检查 GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU (无GPU)\"}')"
echo ""

# 检查数据列表是否存在
if [ ! -f "Data/train_list.txt" ] || [ ! -s "Data/train_list.txt" ]; then
    echo "[!!] 训练列表不存在，先运行数据预处理..."
    python prepare_lj_data.py
fi

# ── 第一阶段：预训练（fp16 混合精度，单 GPU）──────────────────
if [ "\$STAGE" = "first" ] || [ "\$STAGE" = "both" ]; then
    echo "[>>] 开始第一阶段训练 (epochs_1st=60, fp16混合精度)..."
    echo "     日志目录: Models/LJSpeech"
    echo "     RTX 4080 预计耗时: ~10~12 小时"
    echo ""
    accelerate launch \
        --mixed_precision=fp16 \
        --num_processes=1 \
        train_first.py --config_path ./Configs/config.yml
    echo "[OK] 第一阶段训练完成"
fi

# ── 第二阶段：联合训练（DP，单进程）──────────────────────────
if [ "\$STAGE" = "second" ] || [ "\$STAGE" = "both" ]; then
    echo "[>>] 开始第二阶段训练 (epochs_2nd=30, diff_epoch=6, joint_epoch=15)..."
    echo "     注意: 第二阶段使用 DP（非 DDP），单进程运行"
    echo "     RTX 4080 预计耗时: ~9~12 小时"
    echo ""
    python train_second.py --config_path ./Configs/config.yml
    echo "[OK] 第二阶段训练完成"
fi

echo ""
echo "============================================"
echo "  训练完成！模型已保存到 Models/LJSpeech/"
echo "============================================"
TRAINEOF

chmod +x "$WSL_PROJECT/train.sh"
log_ok "train.sh 已创建"

# ── 13. 验证安装 ──────────────────────────────────────────────
echo ""
log_info "验证关键包安装..."

python - <<'EOF'
import sys
results = []

# PyTorch
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    results.append(f"  torch {torch.__version__} | CUDA={cuda_ok} | GPU={gpu_name}")
except Exception as e:
    results.append(f"  [FAIL] torch: {e}")

# torchaudio
try:
    import torchaudio
    results.append(f"  torchaudio {torchaudio.__version__}")
except Exception as e:
    results.append(f"  [FAIL] torchaudio: {e}")

# librosa
try:
    import librosa
    results.append(f"  librosa {librosa.__version__}")
except Exception as e:
    results.append(f"  [FAIL] librosa: {e}")

# soundfile
try:
    import soundfile
    results.append(f"  soundfile OK")
except Exception as e:
    results.append(f"  [FAIL] soundfile: {e}")

# monotonic_align
try:
    import monotonic_align
    results.append(f"  monotonic_align OK")
except Exception as e:
    results.append(f"  [FAIL] monotonic_align: {e}")

# phonemizer
try:
    import phonemizer
    results.append(f"  phonemizer {phonemizer.__version__}")
except Exception as e:
    results.append(f"  [FAIL] phonemizer: {e}")

# accelerate
try:
    import accelerate
    results.append(f"  accelerate {accelerate.__version__}")
except Exception as e:
    results.append(f"  [FAIL] accelerate: {e}")

# transformers
try:
    import transformers
    results.append(f"  transformers {transformers.__version__}")
except Exception as e:
    results.append(f"  [FAIL] transformers: {e}")

for r in results:
    print(r)
EOF

# ── 14. 写入 ~/.bashrc 快捷命令 ───────────────────────────────
echo ""
log_info "写入 ~/.bashrc 快捷命令..."

BASHRC_ENTRY="
# StyleTTS2 快捷命令
alias styletts2='cd $WSL_PROJECT && source venv/bin/activate'
export STYLETTS2_HOME=$WSL_PROJECT
"

if ! grep -q "StyleTTS2 快捷命令" ~/.bashrc; then
    echo "$BASHRC_ENTRY" >> ~/.bashrc
    log_ok "已添加 'styletts2' 快捷命令到 ~/.bashrc"
else
    log_info "快捷命令已存在，跳过"
fi

# ── 完成 ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}============================================${NC}"
echo -e "${BOLD}${GREEN}  环境配置完成！${NC}"
echo -e "${BOLD}${GREEN}============================================${NC}"
echo ""
echo -e "  项目路径: ${YELLOW}$WSL_PROJECT${NC}"
echo ""
echo -e "${BOLD}  后续步骤:${NC}"
echo ""
echo -e "  1. 准备训练数据列表（音素转换）:"
echo -e "     ${YELLOW}cd $WSL_PROJECT${NC}"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo -e "     ${YELLOW}python prepare_lj_data.py${NC}"
echo ""
echo -e "  2. 开始训练（两个阶段）:"
echo -e "     ${YELLOW}bash train.sh${NC}"
echo ""
echo -e "  3. 只训练某一阶段:"
echo -e "     ${YELLOW}bash train.sh first${NC}   # 仅第一阶段（预训练）"
echo -e "     ${YELLOW}bash train.sh second${NC}  # 仅第二阶段（联合训练）"
echo ""
echo -e "  4. 使用快捷命令（重开终端后生效）:"
echo -e "     ${YELLOW}styletts2${NC}              # 自动 cd + 激活 venv"
echo ""
echo -e "  5. 查看 TensorBoard 日志:"
echo -e "     ${YELLOW}tensorboard --logdir $WSL_PROJECT/Models/LJSpeech${NC}"
echo ""
