"""修复 WSL .bashrc 并追加正确的配置"""
from pathlib import Path

bashrc = Path.home() / ".bashrc"
txt = bashrc.read_text()

# ── 清理所有之前错误写入的 StyleTTS2 / pyenv 重复行 ──────────
bad_patterns = [
    'source /home/naiwen/.bashrc',
    '# StyleTTS2 快捷命令',
    "alias styletts2=",
    "export STYLETTS2_HOME",
]
lines = txt.splitlines()
clean_lines = []
skip_next = False
for line in lines:
    if any(p in line for p in bad_patterns):
        skip_next = False
        continue
    clean_lines.append(line)

txt_clean = "\n".join(clean_lines)

# ── 追加正确的配置 ────────────────────────────────────────────
append = """
# pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true
eval "$(pyenv virtualenv-init -)" 2>/dev/null || true

# StyleTTS2 快捷命令
alias styletts2='cd $HOME/StyleTTS2 && source venv/bin/activate'
export STYLETTS2_HOME="$HOME/StyleTTS2"
"""

bashrc.write_text(txt_clean + append)
print("[OK] .bashrc 已修复")
print("[OK] 追加内容:")
print(append)
