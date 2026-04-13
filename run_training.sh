#!/bin/bash
# 训练全流程启动（在独立 WSL 窗口中运行）
cd ~/StyleTTS2
source venv/bin/activate

echo "========================================"
echo " StyleTTS2 Training Started"
echo " Log: ~/StyleTTS2/train_stage1.log"
echo "========================================"

# Stage 1
    accelerate launch \
        --mixed_precision=no \
        --num_processes=1 \
        train_first.py --config_path ./Configs/config.yml > ~/StyleTTS2/train_stage1.log 2>&1
echo "[$(date)] Stage 1 DONE" >> ~/StyleTTS2/train_stage1.log

# Stage 2
bash train.sh second > ~/StyleTTS2/train_stage2.log 2>&1
echo "[$(date)] Stage 2 DONE" >> ~/StyleTTS2/train_stage2.log

echo "ALL DONE"
read -p "Press Enter to close..."