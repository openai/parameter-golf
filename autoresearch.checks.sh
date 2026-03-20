#!/bin/bash
set -euo pipefail
python3 -m py_compile modal_app.py train_gpt.py
bash -n autoresearch.sh
