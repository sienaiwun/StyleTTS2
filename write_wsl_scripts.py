"""
在 WSL 内部生成 prepare_lj_data.py 和 train.sh
用法: python3 /mnt/d/StyleTTS2/write_wsl_scripts.py
"""
from pathlib import Path

HOME = Path.home()
P = HOME / "StyleTTS2"
P.mkdir(exist_ok=True)

# ── prepare_lj_data.py ───────────────────────────────────────
(P / "prepare_lj_data.py").write_text(
r"""#!/usr/bin/env python3
import os, sys, random
from pathlib import Path
try:
    from phonemizer.backend import EspeakBackend
except ImportError:
    print("[错误] pip install phonemizer"); sys.exit(1)

SCRIPT_DIR  = Path(__file__).parent
LJ_DIR      = SCRIPT_DIR / "LJSpeech-1.1"
METADATA    = LJ_DIR / "metadata.csv"
DATA_DIR    = SCRIPT_DIR / "Data"
TRAIN_LIST  = DATA_DIR / "train_list.txt"
VAL_LIST    = DATA_DIR / "val_list.txt"
VAL_SIZE    = 100
RANDOM_SEED = 42

if not METADATA.exists():
    print(f"[错误] 未找到 {METADATA}"); sys.exit(1)

DATA_DIR.mkdir(exist_ok=True)
print("[>>] 读取 metadata.csv ...")
entries = []
with open(METADATA, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            wav_name = parts[0] + ".wav"
            text = parts[1]
            if (LJ_DIR / "wavs" / wav_name).exists():
                entries.append((wav_name, text))

print(f"[>>] 共找到 {len(entries)} 个有效音频")
print("[>>] IPA 音素转换（约需 5~15 分钟）...")
texts = [e[1] for e in entries]
backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)
ipa_list = backend.phonemize(texts, njobs=4)

print("[>>] 写出训练列表...")
random.seed(RANDOM_SEED)
indices = list(range(len(entries)))
random.shuffle(indices)
val_idx = set(indices[:VAL_SIZE])

with open(TRAIN_LIST, "w", encoding="utf-8") as ft, \
     open(VAL_LIST,   "w", encoding="utf-8") as fv:
    for i, (wav_name, _) in enumerate(entries):
        line = f"{wav_name}|{ipa_list[i].strip()}|0\n"
        (fv if i in val_idx else ft).write(line)

print(f"[OK] 训练集: {len(entries)-VAL_SIZE} 条 -> {TRAIN_LIST}")
print(f"[OK] 验证集: {VAL_SIZE} 条 -> {VAL_LIST}")
print("[OK] 数据准备完成！")
""", encoding="utf-8")
print("[OK] prepare_lj_data.py 已写入")

# ── train.sh ─────────────────────────────────────────────────
train_sh = r"""#!/bin/bash
# StyleTTS2 训练启动器  用法: bash train.sh [first|second|both]
set -e
STAGE=${1:-both}
PROJECT_DIR="$HOME/StyleTTS2"
cd "$PROJECT_DIR"
source venv/bin/activate

echo ""
echo "============================================"
echo "  StyleTTS2 训练启动器  阶段: $STAGE"
echo "============================================"
echo ""

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

if [ ! -f "Data/train_list.txt" ] || [ ! -s "Data/train_list.txt" ]; then
    echo "[!!] 训练列表缺失，先运行数据预处理..."
    python prepare_lj_data.py
fi

if [ "$STAGE" = "first" ] || [ "$STAGE" = "both" ]; then
    echo "[>>] Stage 1: epochs_1st=60，预计 10~12 小时"
    python3 train_first.py --config_path ./Configs/config.yml
    echo "[OK] Stage 1 完成"
fi

if [ "$STAGE" = "second" ] || [ "$STAGE" = "both" ]; then
    echo "[>>] Stage 2: epochs_2nd=30, diff_epoch=6, joint_epoch=15，预计 9~12 小时"
    python train_second.py --config_path ./Configs/config.yml
    echo "[OK] Stage 2 完成"
fi

echo ""
echo "============================================"
echo "  训练完成！模型 -> Models/LJSpeech/"
echo "============================================"
"""
train_sh_path = P / "train.sh"
train_sh_path.write_text(train_sh, encoding="utf-8")
train_sh_path.chmod(0o755)
print("[OK] train.sh 已写入")

# ── 更新 config.yml root_path ────────────────────────────────
import re
cfg_path = P / "Configs/config.yml"
if cfg_path.exists():
    txt = cfg_path.read_text()
    correct_path = str(P / "LJSpeech-1.1/wavs")
    txt = re.sub(r'root_path:.*', f'root_path: "{correct_path}"', txt)
    cfg_path.write_text(txt)
    print(f"[OK] config.yml root_path -> {correct_path}")

print("\n[完成] 所有脚本已生成！")
