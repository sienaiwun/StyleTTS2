#!/usr/bin/env python3
"""
修复 LibriTTS 训练/验证集分割问题
—— 按 speaker 分割，确保验证集使用不同的说话人

用法: python fix_libritts_split.py
"""
import os
import random
from pathlib import Path
from collections import defaultdict

# ── 配置 ──────────────────────────────────────────────
PROJECT_DIR = Path(os.path.expanduser("~/StyleTTS2"))
DATA_DIR    = PROJECT_DIR / "Data"
TRAIN_LIST  = DATA_DIR / "train_list.txt"
VAL_LIST    = DATA_DIR / "val_list.txt"
VAL_SPEAKER_RATIO = 0.10   # 10% 的说话人用于验证
RANDOM_SEED = 42

# ── 读取现有的 train_list.txt（里面包含所有数据）──────
all_entries = []
src_file = TRAIN_LIST if TRAIN_LIST.exists() else VAL_LIST

if not src_file.exists():
    print(f"[错误] 找不到 {TRAIN_LIST} 或 {VAL_LIST}")
    exit(1)

print(f"[>>] 读取 {src_file} ...")
with open(src_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_entries.append(line)

print(f"[>>] 总条目数: {len(all_entries)}")

# ── 按 speaker 分组 ──────────────────────────────────
# LibriTTS 格式: /path/to/SPEAKER_ID/BOOK_ID/SPEAKER_BOOK_UTTER.wav|text|speaker_id
# 或者: path|phonemes|speaker_id
speaker_to_entries = defaultdict(list)
for entry in all_entries:
    parts = entry.split("|")
    if len(parts) >= 3:
        speaker_id = parts[-1].strip()  # 最后一个字段是 speaker_id
    elif len(parts) >= 1:
        # 从路径中提取 speaker_id
        wav_path = parts[0]
        # LibriTTS 路径通常: .../SPEAKER_ID/BOOK_ID/xxx.wav
        path_parts = Path(wav_path).parts
        speaker_id = "unknown"
        for p in path_parts:
            if p.isdigit():
                speaker_id = p
                break
    else:
        speaker_id = "unknown"
    speaker_to_entries[speaker_id].append(entry)

speakers = sorted(speaker_to_entries.keys())
print(f"[>>] 说话人数量: {len(speakers)}")

# ── 按说话人分割 ──────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(speakers)

n_val_speakers = max(1, int(len(speakers) * VAL_SPEAKER_RATIO))
val_speakers = set(speakers[:n_val_speakers])
train_speakers = set(speakers[n_val_speakers:])

train_entries = []
val_entries = []

for spk in train_speakers:
    train_entries.extend(speaker_to_entries[spk])

for spk in val_speakers:
    val_entries.extend(speaker_to_entries[spk])

# 打乱顺序
random.shuffle(train_entries)
random.shuffle(val_entries)

print(f"[>>] 训练集: {len(train_entries)} 条, {len(train_speakers)} 个说话人")
print(f"[>>] 验证集: {len(val_entries)} 条, {len(val_speakers)} 个说话人")

# ── 备份旧文件 ────────────────────────────────────────
for f in [TRAIN_LIST, VAL_LIST]:
    if f.exists():
        backup = f.with_suffix(".txt.bak")
        print(f"[>>] 备份 {f.name} -> {backup.name}")
        f.rename(backup)

# ── 写入新文件 ────────────────────────────────────────
with open(TRAIN_LIST, "w", encoding="utf-8") as f:
    f.write("\n".join(train_entries) + "\n")

with open(VAL_LIST, "w", encoding="utf-8") as f:
    f.write("\n".join(val_entries) + "\n")

print(f"\n[OK] 训练集: {len(train_entries)} 条 -> {TRAIN_LIST}")
print(f"[OK] 验证集: {len(val_entries)} 条 -> {VAL_LIST}")
print("[OK] 分割完成！训练集和验证集使用不同的说话人。")
