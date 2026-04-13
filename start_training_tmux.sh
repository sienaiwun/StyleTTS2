#!/bin/bash
# 在 tmux 会话中重启训练（防止终端关闭中断）
tmux kill-session -t train 2>/dev/null || true

tmux new-session -d -s train \
  "cd ~/StyleTTS2 && source venv/bin/activate && bash train.sh first 2>&1 | tee ~/StyleTTS2/train_stage1.log; echo 'Stage1 done'; bash train.sh second 2>&1 | tee ~/StyleTTS2/train_stage2.log; echo 'ALL DONE'; read"

echo ""
echo "✅ 训练已在 tmux 会话 'train' 中启动"
echo ""
echo "查看进度:   tmux attach -t train"
echo "脱离会话:   Ctrl+B 然后 D"
echo "查看日志:   tail -f ~/StyleTTS2/train_stage1.log"
echo "查看GPU:    nvidia-smi"
