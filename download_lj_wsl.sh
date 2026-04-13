#!/bin/bash
# =============================================================
#  LJSpeech-1.1 完整数据集下载脚本（直接下载到 WSL 内部文件系统）
#  用法: wsl bash /mnt/d/StyleTTS2/download_lj_wsl.sh
#  或在 WSL 内: bash /mnt/d/StyleTTS2/download_lj_wsl.sh
# =============================================================

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_info() { echo -e "${CYAN}[>>]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
log_err()  { echo -e "${RED}[错误]${NC} $1"; exit 1; }

# ── 路径配置 ─────────────────────────────────────────────────
WSL_PROJECT="$HOME/StyleTTS2"                        # WSL 项目根目录
WSL_DATA="$WSL_PROJECT/LJSpeech-1.1"                # 数据集目标目录
TMP_DIR="$HOME/tmp_lj"                               # 下载临时目录
LJ_URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJ_TAR="$TMP_DIR/LJSpeech-1.1.tar.bz2"
EXPECTED_WAV_COUNT=13100                             # 完整数据集应有 13100 个 wav

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  LJSpeech-1.1 完整数据集下载器${NC}"
echo -e "${BOLD}  直接下载到 WSL 内部文件系统（高速 I/O）${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
echo -e "  下载源:   ${YELLOW}$LJ_URL${NC}"
echo -e "  目标路径: ${YELLOW}$WSL_DATA${NC}"
echo -e "  压缩包大小约 2.6 GB，解压后约 5.5 GB"
echo ""

# ── 检查磁盘空间（需要至少 9 GB：2.6 压缩 + 5.5 解压 + 余量）──
log_info "检查 WSL 磁盘空间..."
AVAIL_KB=$(df "$HOME" | awk 'NR==2 {print $4}')
AVAIL_GB=$(echo "scale=1; $AVAIL_KB / 1024 / 1024" | bc)
echo -e "  当前可用空间: ${YELLOW}${AVAIL_GB} GB${NC}"
if [ "$AVAIL_KB" -lt 9437184 ]; then  # 9 GB
    log_err "磁盘空间不足！至少需要 9 GB，当前仅有 ${AVAIL_GB} GB"
fi
log_ok "磁盘空间充足"

# ── 检查当前数据集状态 ────────────────────────────────────────
echo ""
if [ -d "$WSL_DATA/wavs" ]; then
    CURRENT_COUNT=$(find "$WSL_DATA/wavs" -name "*.wav" 2>/dev/null | wc -l)
    echo -e "  当前 WSL 内数据集: ${YELLOW}$CURRENT_COUNT 个 wav${NC}（完整应有 $EXPECTED_WAV_COUNT 个）"

    if [ "$CURRENT_COUNT" -ge "$EXPECTED_WAV_COUNT" ]; then
        log_ok "数据集已完整，无需重新下载"
        echo -n "  是否强制重新下载并覆盖？[y/N] "
        read -r FORCE
        if [[ ! "$FORCE" =~ ^[Yy]$ ]]; then
            log_info "跳过下载，数据集已就绪"
            exit 0
        fi
    else
        log_warn "当前数据集不完整（$CURRENT_COUNT / $EXPECTED_WAV_COUNT），将下载完整版本覆盖"
    fi
fi

# ── 确保项目目录存在 ──────────────────────────────────────────
mkdir -p "$WSL_PROJECT"
mkdir -p "$TMP_DIR"

# ── 下载数据集 ────────────────────────────────────────────────
echo ""
log_info "开始下载 LJSpeech-1.1.tar.bz2 (~2.6 GB)..."
log_info "支持断点续传，如中断后重新运行可继续"
echo ""

# 用 wget 带断点续传（-c）和进度条
wget -c "$LJ_URL" -O "$LJ_TAR" \
    --progress=bar:force \
    --show-progress \
    2>&1 | tail -c 5000

log_ok "下载完成: $LJ_TAR"

# ── 校验文件大小（至少 2.5 GB）────────────────────────────────
TAR_SIZE=$(stat -c%s "$LJ_TAR" 2>/dev/null || echo "0")
TAR_SIZE_MB=$((TAR_SIZE / 1024 / 1024))
echo -e "  文件大小: ${YELLOW}${TAR_SIZE_MB} MB${NC}"
if [ "$TAR_SIZE_MB" -lt 2400 ]; then
    log_err "下载文件过小（${TAR_SIZE_MB} MB），可能下载不完整，请检查网络后重试"
fi
log_ok "文件大小校验通过"

# ── 备份旧数据集（如存在）────────────────────────────────────
echo ""
if [ -d "$WSL_DATA" ]; then
    log_info "删除旧的不完整数据集..."
    rm -rf "$WSL_DATA"
    log_ok "旧数据集已清除"
fi

# ── 解压数据集 ────────────────────────────────────────────────
echo ""
log_info "解压数据集到 $WSL_PROJECT ..."
log_info "解压约需 2~5 分钟，请耐心等待..."

tar -xjf "$LJ_TAR" -C "$WSL_PROJECT" --checkpoint=1000 \
    --checkpoint-action=echo='  已解压 %{r}T ...'

log_ok "解压完成"

# ── 验证解压结果 ──────────────────────────────────────────────
echo ""
log_info "验证数据集完整性..."
FINAL_COUNT=$(find "$WSL_DATA/wavs" -name "*.wav" 2>/dev/null | wc -l)
echo -e "  wav 文件数量: ${YELLOW}$FINAL_COUNT / $EXPECTED_WAV_COUNT${NC}"

if [ "$FINAL_COUNT" -lt "$EXPECTED_WAV_COUNT" ]; then
    log_warn "解压后文件数量 ($FINAL_COUNT) 少于预期 ($EXPECTED_WAV_COUNT)"
    log_warn "数据集可能不完整，但仍可继续训练"
else
    log_ok "数据集完整性验证通过！"
fi

# 验证 metadata.csv 存在
if [ -f "$WSL_DATA/metadata.csv" ]; then
    META_LINES=$(wc -l < "$WSL_DATA/metadata.csv")
    log_ok "metadata.csv 存在 ($META_LINES 行)"
else
    log_warn "未找到 metadata.csv，数据集可能有问题"
fi

# ── 清理临时文件 ──────────────────────────────────────────────
echo ""
echo -n "  是否删除下载的压缩包以释放空间（2.6 GB）？[Y/n] "
read -r CLEAN
if [[ ! "$CLEAN" =~ ^[Nn]$ ]]; then
    rm -f "$LJ_TAR"
    rmdir "$TMP_DIR" 2>/dev/null || true
    log_ok "压缩包已删除"
else
    log_info "压缩包保留在: $LJ_TAR"
fi

# ── 同步到 Windows（可选）────────────────────────────────────
echo ""
WIN_DATA="/mnt/d/StyleTTS2/LJSpeech-1.1"
log_info "是否同时更新 Windows 目录 $WIN_DATA ？（非必须，训练在 WSL 运行）"
echo -n "  同步到 Windows D:\\StyleTTS2\\LJSpeech-1.1？[y/N] "
read -r SYNC_WIN
if [[ "$SYNC_WIN" =~ ^[Yy]$ ]]; then
    log_info "同步到 Windows 路径（跨文件系统，速度较慢）..."
    rsync -ah --progress "$WSL_DATA/" "$WIN_DATA/"
    log_ok "Windows 路径已同步"
fi

# ── 更新 config.yml 路径 ──────────────────────────────────────
echo ""
CONFIG="$WSL_PROJECT/Configs/config.yml"
if [ -f "$CONFIG" ]; then
    sed -i "s|root_path:.*|root_path: \"$WSL_DATA/wavs\"|" "$CONFIG"
    log_ok "config.yml root_path 已确认: $WSL_DATA/wavs"
fi

# ── 完成 ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}============================================${NC}"
echo -e "${BOLD}${GREEN}  LJSpeech-1.1 数据集准备完成！${NC}"
echo -e "${BOLD}${GREEN}============================================${NC}"
echo ""
echo -e "  数据集路径: ${YELLOW}$WSL_DATA${NC}"
echo -e "  wav 文件数: ${YELLOW}$FINAL_COUNT 个${NC}"
echo ""
echo -e "${BOLD}  下一步：${NC}"
echo -e "  ${YELLOW}cd $WSL_PROJECT${NC}"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  ${YELLOW}python prepare_lj_data.py   # 生成 IPA 音素训练列表${NC}"
echo -e "  ${YELLOW}bash train.sh               # 开始训练${NC}"
echo ""
