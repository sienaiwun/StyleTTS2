#!/bin/bash
cd ~/StyleTTS2
source venv/bin/activate

CONFIG="Configs/config_5090.yml"

# 确保 config 有必要字段
grep -q "diff_epoch" "$CONFIG" || sed -i '/TMA_CEloss: false/a\  diff_epoch: 20\n  joint_epoch: 50' "$CONFIG"

echo "========================================"
echo "  Stage 1: train_first.py"
echo "  Start: $(date)"
echo "========================================"
python train_first.py --config_path "$CONFIG"

echo "========================================"
echo "  Stage 1 done. Copying weights..."
echo "========================================"
LAST=$(ls -t Models/LJSpeech_5090/epoch_1st_*.pth 2>/dev/null | head -1)
if [ -n "$LAST" ]; then
    cp "$LAST" Models/LJSpeech_5090/first_stage.pth
    echo "Copied: $LAST → Models/LJSpeech_5090/first_stage.pth"
else
    echo "[ERROR] 找不到 epoch_1st_*.pth，请检查 Models/LJSpeech_5090/ 目录"
    ls -lh Models/LJSpeech_5090/
    exit 1
fi

echo "========================================"
echo "  Stage 2: train_second.py"
echo "  Start: $(date)"
echo "========================================"
python train_second.py --config_path "$CONFIG"

echo "========================================"
echo "  ALL DONE: $(date)"
echo "========================================"
