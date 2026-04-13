#!/bin/bash
cd ~/StyleTTS2
source venv/bin/activate
python3 compare_models.py 2>&1 | grep -v UserWarning | grep -v "warnings.warn"
