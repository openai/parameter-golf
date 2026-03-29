"""Shared constants and paths."""

from __future__ import annotations

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(
    PROJECT_DIR,
    "records/track_10min_16mb/2026-03-29_FullStack_TTT_Ngram_KNN_TurboQuant/train_gpt.py",
)
RESULTS_FILE = os.path.join(PROJECT_DIR, "validation_results.jsonl")
PENALTY = 10.0
