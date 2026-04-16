#!/bin/bash
# ============================================================
# 启动 StyleTTS2 LibriTTS 训练 (tmux 后台运行)
# 用法:
#   bash start_train_libritts.sh first   # 第一阶段
#   bash start_train_libritts.sh second  # 第二阶段
# ============================================================

PROJECT_DIR="$HOME/StyleTTS2"
VENV_DIR="$HOME/styletts2_env/venv"
CONFIG="Configs/config_libritts_train.yml"
SESSION_NAME="styletts2_train"

STAGE=${1:-first}

echo "============================================"
echo " Starting StyleTTS2 LibriTTS Training"
echo " Stage: $STAGE"
echo "============================================"

# 检查 tmux
if ! command -v tmux &>/dev/null; then
    echo "Installing tmux..."
    sudo apt-get install -y tmux
fi

# 构建训练命令
if [ "$STAGE" = "first" ]; then
    TRAIN_CMD="cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python train_first.py -p $CONFIG 2>&1 | tee Models/LibriTTS/train_first.log"
elif [ "$STAGE" = "second" ]; then
    TRAIN_CMD="cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python train_second.py -p $CONFIG 2>&1 | tee Models/LibriTTS/train_second.log"
else
    echo "Usage: bash start_train_libritts.sh [first|second]"
    exit 1
fi

# 杀掉旧的 session (如果存在)
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# 在 tmux 中启动训练
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_CMD"

echo ""
echo "✅ Training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME     # 查看训练进度"
echo "  tmux kill-session -t $SESSION_NAME  # 停止训练"
echo "  tail -f $PROJECT_DIR/Models/LibriTTS/train_${STAGE}.log  # 查看日志"
echo ""
echo "Detach from tmux: Ctrl+B then D"
