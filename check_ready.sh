#!/bin/bash
# StyleTTS2 训练就绪检查脚本
cd ~/StyleTTS2 2>/dev/null || { echo "ERROR: ~/StyleTTS2 不存在"; exit 1; }
source venv/bin/activate 2>/dev/null || { echo "ERROR: venv 不存在"; exit 1; }

python3 << 'PYEOF'
import sys
from pathlib import Path

P = Path.home() / "StyleTTS2"
ok = True

def chk(label, cond, val=""):
    global ok
    mark = "OK  " if cond else "FAIL"
    if not cond: ok = False
    print(f"  [{mark}] {label}: {val}")

print("\n========== StyleTTS2 训练就绪检查 ==========\n")

# ── Python 环境 ─────────────────────────────────────────────
print("── Python 依赖 ──")
try:
    import torch
    cuda = torch.cuda.is_available()
    gpu  = torch.cuda.get_device_name(0) if cuda else "无GPU"
    chk("PyTorch", True, f"{torch.__version__} | CUDA={cuda} | {gpu}")
except Exception as e:
    chk("PyTorch", False, str(e))

for pkg in ["torchaudio","librosa","soundfile","phonemizer",
            "accelerate","transformers","munch","monotonic_align","numpy"]:
    try:
        m = __import__(pkg)
        chk(pkg, True, getattr(m,"__version__","installed"))
    except Exception as e:
        chk(pkg, False, f"未安装: {e}")

# ── 数据集 ──────────────────────────────────────────────────
print("\n── 数据集 ──")
wav_dir = P / "LJSpeech-1.1" / "wavs"
if wav_dir.exists():
    n = len(list(wav_dir.glob("*.wav")))
    chk("LJSpeech wavs", n >= 13000, f"{n} / 13100 个文件")
else:
    chk("LJSpeech wavs", False, "目录不存在")

meta = P / "LJSpeech-1.1" / "metadata.csv"
chk("metadata.csv", meta.exists(), "OK" if meta.exists() else "缺失")

# ── 训练数据列表 ─────────────────────────────────────────────
print("\n── 训练列表 ──")
for fname in ["Data/train_list.txt", "Data/val_list.txt"]:
    fp = P / fname
    if fp.exists() and fp.stat().st_size > 0:
        lines = sum(1 for _ in open(fp))
        chk(fname, True, f"{lines} 条")
    else:
        chk(fname, False, "缺失或为空 → 需运行 prepare_lj_data.py")

# ── 配置 & 脚本 ──────────────────────────────────────────────
print("\n── 配置 & 训练脚本 ──")
files = [
    "Configs/config.yml",
    "train_first.py",
    "train_second.py",
    "prepare_lj_data.py",
    "train.sh",
]
for f in files:
    fp = P / f
    chk(f, fp.exists(), "OK" if fp.exists() else "缺失")

# ── 预训练模型权重 ────────────────────────────────────────────
print("\n── 预训练权重 ──")
weights = {
    "Utils/JDC/bst.t7":           "F0 提取器",
    "Utils/ASR/epoch_00080.pth":  "文本对齐器",
    "Utils/ASR/config.yml":       "ASR 配置",
    "Utils/PLBERT/config.yml":    "PL-BERT",
}
for path, desc in weights.items():
    fp = P / path
    chk(f"{desc} ({path})", fp.exists(), "OK" if fp.exists() else "缺失")

# ── config.yml root_path ─────────────────────────────────────
print("\n── config.yml 关键参数 ──")
cfg = P / "Configs/config.yml"
if cfg.exists():
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    root_path = c.get("data_params",{}).get("root_path","")
    path_ok = Path(root_path).exists()
    chk("root_path 存在", path_ok, root_path)
    chk("epochs_1st", True, str(c.get("epochs_1st","")))
    chk("epochs_2nd", True, str(c.get("epochs_2nd","")))
    chk("TMA_epoch",  True, str(c.get("loss_params",{}).get("TMA_epoch","")))
    chk("diff_epoch", True, str(c.get("loss_params",{}).get("diff_epoch","")))
    chk("joint_epoch",True, str(c.get("loss_params",{}).get("joint_epoch","")))

# ── 总结 ─────────────────────────────────────────────────────
print("\n============================================")
if ok:
    print("  ✅ 全部就绪！可以开始训练：")
    print("     bash ~/StyleTTS2/train.sh")
else:
    print("  ❌ 有缺失项，请按上方 [FAIL] 提示修复")
    # 给出具体建议
    tl = P / "Data/train_list.txt"
    if not (tl.exists() and tl.stat().st_size > 0):
        print("\n  修复数据列表：")
        print("     python3 ~/StyleTTS2/prepare_lj_data.py")
print("============================================\n")
PYEOF
