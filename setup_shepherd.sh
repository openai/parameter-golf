#!/bin/bash
cd /workspace
if [ -d "parameter-golf" ]; then cd parameter-golf && git pull; else git clone https://github.com/Unwindology/parameter-golf.git && cd parameter-golf; fi
pip install sentencepiece huggingface_hub -q
python data/cached_challenge_fineweb.py
echo "Setup complete - ready to train"
