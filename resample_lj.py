#!/usr/bin/env python3
"""
预处理 LJSpeech：将所有 wav 从 22050 Hz 重采样到 24000 Hz
并保存到新目录，训练时直接读取，跳过实时重采样
用法: python3 resample_lj.py
预计耗时: ~5~10 分钟（CPU 多进程）
"""
import os
import sys
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

try:
    import soundfile as sf
    import librosa
    from tqdm import tqdm
except ImportError:
    print("[错误] pip install soundfile librosa tqdm")
    sys.exit(1)

# ── 配置 ──────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
SRC_DIR     = SCRIPT_DIR / "LJSpeech-1.1" / "wavs"       # 原始 22050 Hz
DST_DIR     = SCRIPT_DIR / "LJSpeech-1.1" / "wavs_24k"   # 目标 24000 Hz
TARGET_SR   = 24000
NUM_WORKERS = max(4, os.cpu_count() // 2)  # 使用一半 CPU 核心

def resample_one(args):
    """处理单个 wav 文件"""
    src_path, dst_path = args
    if dst_path.exists():
        return dst_path.name, True, "skip"

    try:
        wave, sr = sf.read(str(src_path))
        if wave.ndim == 2:          # 立体声转单声道
            wave = wave[:, 0]
        if sr != TARGET_SR:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=TARGET_SR)
        sf.write(str(dst_path), wave, TARGET_SR)
        return dst_path.name, True, "done"
    except Exception as e:
        return dst_path.name, False, str(e)


def main():
    if not SRC_DIR.exists():
        print(f"[错误] 未找到 {SRC_DIR}")
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(SRC_DIR.glob("*.wav"))
    print(f"[>>] 共找到 {len(wav_files)} 个 wav 文件")
    print(f"[>>] 源目录: {SRC_DIR}")
    print(f"[>>] 目标目录: {DST_DIR}")
    print(f"[>>] 使用 {NUM_WORKERS} 个 CPU 进程")
    print()

    args_list = [(src, DST_DIR / src.name) for src in wav_files]

    done = skip = fail = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(resample_one, a): a for a in args_list}
        with tqdm(total=len(args_list), unit="file", desc="重采样") as pbar:
            for future in as_completed(futures):
                name, ok, status = future.result()
                if status == "skip":
                    skip += 1
                elif ok:
                    done += 1
                else:
                    fail += 1
                    print(f"\n[FAIL] {name}: {status}")
                pbar.set_postfix(done=done, skip=skip, fail=fail)
                pbar.update(1)

    print(f"\n[OK] 完成: {done} 个，跳过(已存在): {skip} 个，失败: {fail} 个")
    print(f"[OK] 24kHz wav 已保存到: {DST_DIR}")
    print()

    # ── 提示更新 config.yml ────────────────────────────────────
    print("[>>] 请更新 Configs/config.yml 的 root_path:")
    print(f'     root_path: "{DST_DIR}"')


if __name__ == "__main__":
    main()
