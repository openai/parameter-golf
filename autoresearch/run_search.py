#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

ARTIFACT_LIMIT_BYTES = 16_000_000
MAX_CANDIDATE_ATTEMPTS = int(os.environ.get("AR_MAX_CANDIDATE_ATTEMPTS", 200))
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "autoresearch"
TRIALS_DIR = LOG_DIR / "trials"
WORKBENCH_DIR = LOG_DIR / "workbench"
RESULTS_TSV = LOG_DIR / "results.tsv"
BEST_JSON = LOG_DIR / "best_config.json"
TRAIN_CUDA = ROOT / "train_gpt.py"
TRAIN_MLX = ROOT / "train_gpt_mlx.py"

VAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact.*?val_loss:\s*(?P<val_loss>[-+0-9.eE]+).*?val_bpb:\s*(?P<val_bpb>[-+0-9.eE]+)",
    flags=re.IGNORECASE,
)
CUDA_MODEL_SIZE_RE = re.compile(r"serialized\s+model\s+int8\+zlib:\s*(?P<bytes>\d+)\s*bytes", flags=re.IGNORECASE)
CUDA_TOTAL_SIZE_RE = re.compile(r"total\s+submission\s+size\s+int8\+zlib:\s*(?P<bytes>\d+)\s*bytes", flags=re.IGNORECASE)
MLX_MODEL_SIZE_RE = re.compile(r"serialized_model_int8_zlib:\s*(?P<bytes>\d+)\s*bytes", flags=re.IGNORECASE)
PARAM_RE = re.compile(r"model_params:\s*(?P<params>\d+)", flags=re.IGNORECASE)

CONFIG_KEYS = [
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "MODEL_DIM",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MLP_MULT",
    "TRAIN_SEQ_LEN",
    "TRAIN_BATCH_TOKENS",
    "VAL_BATCH_SIZE",
    "ITERATIONS",
    "MAX_WALLCLOCK_SECONDS",
    "WARMUP_STEPS",
    "VAL_EVAL_MAX_SEQS",
    "TIED_EMBED_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "MUON_MOMENTUM",
    "MUON_BACKEND_STEPS",
    "QK_GAIN_INIT",
    "LOGIT_SOFTCAP",
    "WARMDOWN_ITERS",
    "TIED_EMBED_INIT_STD",
    "GRAD_ACCUM_STEPS",
    "MLX_MAX_MICROBATCH_TOKENS",
    "SPARSE_FFN_TOPK_RATIO",
    "GRAD_CLIP_NORM",
]

SEARCH_CHOICES: dict[str, dict[str, list[str]]] = {
    "cuda": {
        "NUM_LAYERS": ["8", "9", "10", "11", "12"],
        "MODEL_DIM": ["384", "448", "512", "576", "640"],
        "NUM_HEADS": ["6", "7", "8", "9", "10"],
        "NUM_KV_HEADS": ["1", "2", "4", "5"],
        "MLP_MULT": ["2", "3"],
        "TRAIN_SEQ_LEN": ["512", "768", "1024", "2048", "4096"],
        "TRAIN_BATCH_TOKENS": ["131072", "262144", "393216", "524288", "786432"],
        "TIED_EMBED_LR": ["0.03", "0.04", "0.05", "0.06", "0.07"],
        "MATRIX_LR": ["0.02", "0.025", "0.03", "0.04", "0.05"],
        "SCALAR_LR": ["0.02", "0.025", "0.03", "0.04", "0.05"],
        "MUON_MOMENTUM": ["0.92", "0.95", "0.97", "0.99"],
        "MUON_BACKEND_STEPS": ["4", "5", "6"],
        "QK_GAIN_INIT": ["1.0", "1.25", "1.5", "1.75", "2.0"],
        "LOGIT_SOFTCAP": ["20.0", "30.0", "40.0"],
        "WARMDOWN_ITERS": ["600", "800", "1200", "1600", "3000", "4000"],
        "TIED_EMBED_INIT_STD": ["0.003", "0.005", "0.007", "0.01"],
        "SPARSE_FFN_TOPK_RATIO": ["0.25", "0.5", "1.0"],
        "GRAD_CLIP_NORM": ["0.0", "0.3", "0.5", "1.0"],
    },
    "mlx": {
        "NUM_LAYERS": ["8", "9", "10", "12"],
        "MODEL_DIM": ["384", "448", "512", "576"],
        "NUM_HEADS": ["6", "7", "8", "9"],
        "NUM_KV_HEADS": ["1", "2", "4"],
        "MLP_MULT": ["2", "3"],
        "TRAIN_SEQ_LEN": ["512", "1024"],
        "TRAIN_BATCH_TOKENS": ["8192", "16384", "32768", "65536"],
        "TIED_EMBED_LR": ["0.03", "0.04", "0.05", "0.06", "0.07"],
        "MATRIX_LR": ["0.02", "0.025", "0.03", "0.04", "0.05"],
        "SCALAR_LR": ["0.02", "0.025", "0.03", "0.04", "0.05"],
        "MUON_MOMENTUM": ["0.92", "0.95", "0.97"],
        "MUON_BACKEND_STEPS": ["4", "5", "6"],
        "QK_GAIN_INIT": ["1.0", "1.25", "1.5", "1.75", "2.0"],
        "LOGIT_SOFTCAP": ["20.0", "30.0", "40.0"],
        "WARMDOWN_ITERS": ["600", "800", "1200", "1600"],
        "TIED_EMBED_INIT_STD": ["0.003", "0.005", "0.007", "0.01"],
        "SPARSE_FFN_TOPK_RATIO": ["0.25", "0.5", "1.0"],
        "GRAD_CLIP_NORM": ["0.0", "0.3", "0.5", "1.0"],
    },
}

PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "cuda": {
        "baseline": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "524288",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "1200",
            "TIED_EMBED_INIT_STD": "0.005",
        },
        "depth_first": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "12",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "393216",
            "VAL_BATCH_SIZE": "393216",
            "ITERATIONS": "22000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.04",
            "MATRIX_LR": "0.03",
            "SCALAR_LR": "0.03",
            "MUON_MOMENTUM": "0.97",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.25",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "1600",
            "TIED_EMBED_INIT_STD": "0.005",
        },
        "width_first": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "8",
            "MODEL_DIM": "640",
            "NUM_HEADS": "10",
            "NUM_KV_HEADS": "2",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "768",
            "TRAIN_BATCH_TOKENS": "262144",
            "VAL_BATCH_SIZE": "262144",
            "ITERATIONS": "18000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.04",
            "MATRIX_LR": "0.025",
            "SCALAR_LR": "0.03",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "4",
            "QK_GAIN_INIT": "1.75",
            "LOGIT_SOFTCAP": "20.0",
            "WARMDOWN_ITERS": "1200",
            "TIED_EMBED_INIT_STD": "0.003",
        },
        "compact_context": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "10",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "2",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "512",
            "TRAIN_BATCH_TOKENS": "786432",
            "VAL_BATCH_SIZE": "786432",
            "ITERATIONS": "24000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.06",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.92",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "40.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.007",
        },
        # Strategy: small batch + deep (dominant lever from autoresearch experiments)
        "small_batch_deep": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "12",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "512",
            "TRAIN_BATCH_TOKENS": "131072",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "25000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.06",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.92",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "600",
            "TIED_EMBED_INIT_STD": "0.005",
        },
        # Strategy: wide + shallow + single KV head
        "wide_shallow_gqa": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "8",
            "MODEL_DIM": "640",
            "NUM_HEADS": "10",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "131072",
            "VAL_BATCH_SIZE": "131072",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.03",
            "SCALAR_LR": "0.03",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.75",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.003",
        },
        # Strategy: high LR + fast convergence + aggressive warmdown
        "high_lr_fast": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "2",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "262144",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "22000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.07",
            "MATRIX_LR": "0.05",
            "SCALAR_LR": "0.05",
            "MUON_MOMENTUM": "0.97",
            "MUON_BACKEND_STEPS": "6",
            "QK_GAIN_INIT": "2.0",
            "LOGIT_SOFTCAP": "40.0",
            "WARMDOWN_ITERS": "600",
            "TIED_EMBED_INIT_STD": "0.007",
        },
        # Strategy: compact context, many more steps
        "compact_many_steps": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "10",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "256",
            "TRAIN_BATCH_TOKENS": "131072",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "30000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.06",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.92",
            "MUON_BACKEND_STEPS": "4",
            "QK_GAIN_INIT": "1.25",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.005",
        },
        # Strategy: 3x MLP expansion (richer feature space)
        "wide_mlp": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "3",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "262144",
            "VAL_BATCH_SIZE": "262144",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.03",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.005",
        },
        # Strategy: chase SOTA (9L 3xMLP fits under 16MB with int8, seq2048, low LR)
        "sota_chase": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "3",
            "TRAIN_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "131072",
            "VAL_BATCH_SIZE": "131072",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.03",
            "MATRIX_LR": "0.02",
            "SCALAR_LR": "0.02",
            "MUON_MOMENTUM": "0.99",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "3000",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_CLIP_NORM": "0.3",
        },
        # Strategy: aggressive — seq4096, reduced batch for A10G, long warmdown
        "sota_aggressive": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "3",
            "TRAIN_SEQ_LEN": "4096",
            "TRAIN_BATCH_TOKENS": "131072",
            "VAL_BATCH_SIZE": "131072",
            "ITERATIONS": "25000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.03",
            "MATRIX_LR": "0.02",
            "SCALAR_LR": "0.02",
            "MUON_MOMENTUM": "0.99",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "4000",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_CLIP_NORM": "0.3",
        },
        # Strategy: SWA-ready baseline — high momentum + long warmdown for weight averaging
        "swa_baseline": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "524288",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.99",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "3000",
            "TIED_EMBED_INIT_STD": "0.005",
        },
        # PR #332 inspired: 12L with 2xMLP (fits int8; PR uses int6+3xMLP)
        "pr332_12l_xsa": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "12",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "524288",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.03",
            "MATRIX_LR": "0.02",
            "SCALAR_LR": "0.02",
            "MUON_MOMENTUM": "0.99",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "3000",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_CLIP_NORM": "0.3",
        },
        # PR #338 inspired: 11L with 2xMLP (fits int8; PR uses int6+3xMLP+TTT)
        "pr338_11l_ttt": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "524288",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.03",
            "MATRIX_LR": "0.02",
            "SCALAR_LR": "0.02",
            "MUON_MOMENTUM": "0.99",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "3000",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_CLIP_NORM": "0.3",
        },
        # Blockchain-inspired: BFT trimmed-mean ensemble + high momentum convergence
        "bft_ensemble": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "3",
            "TRAIN_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "393216",
            "VAL_BATCH_SIZE": "393216",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.03",
            "MATRIX_LR": "0.02",
            "SCALAR_LR": "0.02",
            "MUON_MOMENTUM": "0.99",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "3000",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_CLIP_NORM": "0.3",
        },
        # Blockchain-inspired: difficulty-adjusted search — short seq, many steps, tight LR
        "difficulty_adjusted": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "10",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "2",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "262144",
            "VAL_BATCH_SIZE": "524288",
            "ITERATIONS": "25000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.04",
            "MATRIX_LR": "0.025",
            "SCALAR_LR": "0.025",
            "MUON_MOMENTUM": "0.97",
            "MUON_BACKEND_STEPS": "6",
            "QK_GAIN_INIT": "1.75",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "4000",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_CLIP_NORM": "0.5",
        },
        # Partial RoPE + per-head temp (from PR #327) — baseline arch, novel attention
        "partial_rope_headtemp": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "393216",
            "VAL_BATCH_SIZE": "393216",
            "ITERATIONS": "20000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TIED_EMBED_LR": "0.03",
            "MATRIX_LR": "0.02",
            "SCALAR_LR": "0.02",
            "MUON_MOMENTUM": "0.99",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "3000",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_CLIP_NORM": "0.3",
        },
    },
    "mlx": {
        "baseline": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "ITERATIONS": "2000",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "1200",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_ACCUM_STEPS": "1",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
        },
        # Strategy: aggressive batch reduction (biggest lever per autoresearch findings)
        "small_batch_deep": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "12",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "512",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "32768",
            "ITERATIONS": "2000",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.06",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.92",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "600",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_ACCUM_STEPS": "1",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
        },
        # Strategy: wider model, fewer layers, aggressive GQA (1 KV head)
        "wide_shallow_gqa": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "8",
            "MODEL_DIM": "576",
            "NUM_HEADS": "9",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "16384",
            "VAL_BATCH_SIZE": "32768",
            "ITERATIONS": "1500",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.03",
            "SCALAR_LR": "0.03",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.75",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.003",
            "GRAD_ACCUM_STEPS": "4",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
        },
        # Strategy: high-LR fast convergence with aggressive warmdown
        "high_lr_fast": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "2",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "16384",
            "VAL_BATCH_SIZE": "32768",
            "ITERATIONS": "1500",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.07",
            "MATRIX_LR": "0.05",
            "SCALAR_LR": "0.05",
            "MUON_MOMENTUM": "0.97",
            "MUON_BACKEND_STEPS": "6",
            "QK_GAIN_INIT": "2.0",
            "LOGIT_SOFTCAP": "40.0",
            "WARMDOWN_ITERS": "600",
            "TIED_EMBED_INIT_STD": "0.007",
            "GRAD_ACCUM_STEPS": "4",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
        },
        # Strategy: compact context with more iterations (higher throughput)
        "compact_many_steps": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "10",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "256",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "32768",
            "ITERATIONS": "3000",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.06",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.92",
            "MUON_BACKEND_STEPS": "4",
            "QK_GAIN_INIT": "1.25",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_ACCUM_STEPS": "1",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
        },
        # Strategy: 3x MLP expansion for richer feature space
        "wide_mlp": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "3",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "16384",
            "VAL_BATCH_SIZE": "32768",
            "ITERATIONS": "1200",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.03",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_ACCUM_STEPS": "4",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
        },
        # Strategy: sparse FFN with top-k activation (keep 25% of activations)
        "sparse_ffn": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "2",
            "MLP_MULT": "3",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_BATCH_TOKENS": "16384",
            "VAL_BATCH_SIZE": "32768",
            "ITERATIONS": "1500",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.05",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_ACCUM_STEPS": "4",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
            "SPARSE_FFN_TOPK_RATIO": "0.25",
        },
        "small_fast": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "8",
            "MODEL_DIM": "384",
            "NUM_HEADS": "6",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "512",
            "TRAIN_BATCH_TOKENS": "4096",
            "VAL_BATCH_SIZE": "8192",
            "ITERATIONS": "2000",
            "MAX_WALLCLOCK_SECONDS": "120",
            "TIED_EMBED_LR": "0.06",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.92",
            "MUON_BACKEND_STEPS": "4",
            "QK_GAIN_INIT": "1.25",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.007",
            "GRAD_ACCUM_STEPS": "8",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
        },
        "micro_smoke": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "2",
            "MODEL_DIM": "128",
            "NUM_HEADS": "2",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "64",
            "TRAIN_BATCH_TOKENS": "64",
            "VAL_BATCH_SIZE": "64",
            "GRAD_ACCUM_STEPS": "1",
            "ITERATIONS": "400",
            "MAX_WALLCLOCK_SECONDS": "8",
            "WARMUP_STEPS": "2",
            "VAL_EVAL_MAX_SEQS": "256",
            "TIED_EMBED_LR": "0.06",
            "MATRIX_LR": "0.04",
            "SCALAR_LR": "0.04",
            "MUON_MOMENTUM": "0.92",
            "MUON_BACKEND_STEPS": "4",
            "QK_GAIN_INIT": "1.25",
            "LOGIT_SOFTCAP": "30.0",
            "WARMDOWN_ITERS": "800",
            "TIED_EMBED_INIT_STD": "0.007",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
        },
        "balanced": {
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "10",
            "MODEL_DIM": "448",
            "NUM_HEADS": "7",
            "NUM_KV_HEADS": "1",
            "MLP_MULT": "2",
            "TRAIN_SEQ_LEN": "768",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "ITERATIONS": "1200",
            "MAX_WALLCLOCK_SECONDS": "180",
            "TIED_EMBED_LR": "0.04",
            "MATRIX_LR": "0.03",
            "SCALAR_LR": "0.03",
            "MUON_MOMENTUM": "0.95",
            "MUON_BACKEND_STEPS": "5",
            "QK_GAIN_INIT": "1.5",
            "LOGIT_SOFTCAP": "20.0",
            "WARMDOWN_ITERS": "1200",
            "TIED_EMBED_INIT_STD": "0.005",
            "GRAD_ACCUM_STEPS": "8",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
        },
    },
}

CODE_MUTATIONS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "cuda": {
        "gelu_mlp": [
            (
                "        x = torch.relu(self.fc(x))\n        return self.proj(x.square())",
                "        x = F.gelu(self.fc(x))\n        return self.proj(x)",
            ),
        ],
        "silu_mlp": [
            (
                "        x = torch.relu(self.fc(x))\n        return self.proj(x.square())",
                "        x = F.silu(self.fc(x))\n        return self.proj(x)",
            ),
        ],
        "plain_logits": [
            (
                "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)",
                "        logits = logits_proj",
            ),
        ],
        "identity_resid_mix": [
            (
                "        mix = self.resid_mix.to(dtype=x.dtype)\n        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0",
                "        x = x",
            ),
        ],
    },
    "mlx": {
        "gelu_mlp": [
            (
                "        x = nn.relu(self.fc(x))\n        x = x * x\n        if self.topk_ratio >= 1.0:\n            return self.proj(x)\n\n        # Keep only top-k hidden activations per token to reduce FFN compute.\n        k = max(1, int(self.hidden_dim * self.topk_ratio))\n        flat = x.reshape(-1, self.hidden_dim)\n        # Use a per-row threshold mask to avoid non-compiled scatter paths.\n        row_sorted = mx.sort(flat, axis=-1)\n        threshold = row_sorted[:, self.hidden_dim - k][:, None]\n        keep_mask = (flat >= threshold).astype(flat.dtype)\n        sparse_flat = flat * keep_mask\n        proj_w_t = self.proj.weight.astype(sparse_flat.dtype).T\n        return (sparse_flat @ proj_w_t).reshape(*x.shape[:-1], self.proj.weight.shape[0])",
                "        x = nn.gelu(self.fc(x))\n        return self.proj(x)",
            ),
        ],
        "silu_mlp": [
            (
                "        x = nn.relu(self.fc(x))\n        x = x * x\n        if self.topk_ratio >= 1.0:\n            return self.proj(x)\n\n        # Keep only top-k hidden activations per token to reduce FFN compute.\n        k = max(1, int(self.hidden_dim * self.topk_ratio))\n        flat = x.reshape(-1, self.hidden_dim)\n        # Use a per-row threshold mask to avoid non-compiled scatter paths.\n        row_sorted = mx.sort(flat, axis=-1)\n        threshold = row_sorted[:, self.hidden_dim - k][:, None]\n        keep_mask = (flat >= threshold).astype(flat.dtype)\n        sparse_flat = flat * keep_mask\n        proj_w_t = self.proj.weight.astype(sparse_flat.dtype).T\n        return (sparse_flat @ proj_w_t).reshape(*x.shape[:-1], self.proj.weight.shape[0])",
                "        x = nn.silu(self.fc(x))\n        return self.proj(x)",
            ),
        ],
        "plain_logits": [
            (
                "            logits = self.softcap(logits_proj)\n            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction=\"mean\")",
                "            logits = logits_proj\n            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction=\"mean\")",
            ),
            (
                "            logits = self.softcap(logits_proj)\n            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction=\"sum\")",
                "            logits = logits_proj\n            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction=\"sum\")",
            ),
        ],
        "identity_resid_mix": [
            (
                "        x = self.resid_lambda * x + self.x0_lambda * x0",
                "        x = x",
            ),
        ],
    },
}


@dataclass
class TrialResult:
    run_id: str
    backend: str
    mode: str
    status: str
    val_bpb: float
    val_loss: float
    total_bytes: int
    train_script_path: str
    train_script_bytes: int
    quantized_model_bytes: int
    model_params: int
    elapsed_seconds: float
    log_path: str
    preset: str
    code_mutation: str
    parents: list[str]
    config: dict[str, Any]
    description: str = ""


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    WORKBENCH_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "run_id\tbackend\tmode\tval_bpb\tval_loss\ttotal_bytes\tstatus\tpreset\tcode_mutation\tparents\ttrain_script_path\tdescription\n",
            encoding="utf-8",
        )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def script_for_backend(backend: str) -> Path:
    return TRAIN_CUDA if backend == "cuda" else TRAIN_MLX


def script_bytes(path: Path) -> int:
    return len(path.read_bytes())


def normalize_config(backend: str, cfg: dict[str, Any]) -> dict[str, str]:
    base = dict(PRESETS[backend]["baseline"])
    for key, value in cfg.items():
        if key in CONFIG_KEYS:
            base[key] = str(value)
    if backend == "cuda":
        base.pop("GRAD_ACCUM_STEPS", None)
        base.pop("MLX_MAX_MICROBATCH_TOKENS", None)
    return base


def estimate_params(cfg: dict[str, Any]) -> int:
    dim = int(cfg["MODEL_DIM"])
    layers = int(cfg["NUM_LAYERS"])
    vocab = int(cfg["VOCAB_SIZE"])
    heads = int(cfg["NUM_HEADS"])
    kv_heads = int(cfg["NUM_KV_HEADS"])
    mlp_mult = int(cfg["MLP_MULT"])
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim
    hidden = dim * mlp_mult
    embed = vocab * dim
    block = (dim * dim) + (dim * kv_dim) + (dim * kv_dim) + (dim * dim) + (dim * hidden) + (hidden * dim)
    controls = layers * (7 * dim) + (layers // 2) * dim
    return embed + layers * block + controls


def estimated_total_bytes(cfg: dict[str, Any], script_path: Path) -> int:
    # Estimate int8-quantized + zlib-compressed size.  Empirically the
    # compressed artifact is ~0.55-0.65 bytes per parameter (int8 = 1 byte,
    # zlib typically achieves ~1.8-2x compression on weight tensors, plus
    # pickle overhead).  We use 0.70 as a conservative upper bound so we
    # don't reject configs that would actually fit.
    quantized = int(estimate_params(cfg) * 0.70)
    return quantized + script_bytes(script_path)


def find_under_limit_candidate(
    *,
    backend: str,
    mode: str,
    base_cfg: dict[str, str],
    script_path: Path,
    build_candidate: Callable[[], dict[str, str]],
    max_attempts: int = MAX_CANDIDATE_ATTEMPTS,
) -> dict[str, str]:
    for attempt in range(1, max_attempts + 1):
        candidate = build_candidate()
        if estimated_total_bytes(candidate, script_path) < ARTIFACT_LIMIT_BYTES:
            return candidate
    raise RuntimeError(
        "unable to generate under-limit candidate "
        f"backend={backend} mode={mode} attempts={max_attempts} "
        f"limit={ARTIFACT_LIMIT_BYTES} base_model_dim={base_cfg.get('MODEL_DIM')} "
        f"base_layers={base_cfg.get('NUM_LAYERS')} base_seq={base_cfg.get('TRAIN_SEQ_LEN')}"
    )


def find_under_limit_code_candidate(
    *,
    backend: str,
    mode: str,
    base_cfg: dict[str, str],
    code_mutation: str,
    index: int,
    rng: random.Random,
    max_attempts: int = MAX_CANDIDATE_ATTEMPTS,
) -> dict[str, str]:
    for attempt in range(1, max_attempts + 1):
        candidate = mutate_config(base_cfg, backend, rng, intensity=rng.randint(1, 4))
        estimate_script = create_candidate_script(f"estimate_{backend}_{index}_{attempt}", backend, candidate, code_mutation)
        if estimated_total_bytes(candidate, estimate_script) < ARTIFACT_LIMIT_BYTES:
            return candidate
    raise RuntimeError(
        "unable to generate under-limit code candidate "
        f"backend={backend} mode={mode} mutation={code_mutation} attempts={max_attempts} "
        f"limit={ARTIFACT_LIMIT_BYTES} base_model_dim={base_cfg.get('MODEL_DIM')} "
        f"base_layers={base_cfg.get('NUM_LAYERS')} base_seq={base_cfg.get('TRAIN_SEQ_LEN')}"
    )


def valid_shape(cfg: dict[str, str]) -> bool:
    dim = int(cfg["MODEL_DIM"])
    heads = int(cfg["NUM_HEADS"])
    kv_heads = int(cfg["NUM_KV_HEADS"])
    seq_len = int(cfg["TRAIN_SEQ_LEN"])
    batch_tokens = int(cfg["TRAIN_BATCH_TOKENS"])
    return (
        dim % heads == 0
        and heads % kv_heads == 0
        and (dim // heads) % 2 == 0
        and batch_tokens >= seq_len
        and batch_tokens % seq_len == 0
    )


def mutate_config(base: dict[str, str], backend: str, rng: random.Random, intensity: int = 4) -> dict[str, str]:
    cfg = dict(base)
    choices = SEARCH_CHOICES[backend]
    keys = list(choices.keys())
    for key in rng.sample(keys, k=min(intensity, len(keys))):
        cfg[key] = rng.choice(choices[key])
    if backend == "mlx":
        cfg["TRAIN_BATCH_TOKENS"] = rng.choice(["8192", "16384", "32768", "65536"])
        cfg["VAL_BATCH_SIZE"] = cfg["TRAIN_BATCH_TOKENS"]
        cfg["ITERATIONS"] = str(rng.choice([400, 800, 1200, 1600]))
        cfg["MAX_WALLCLOCK_SECONDS"] = str(rng.choice([120, 180, 240]))
        cfg["GRAD_ACCUM_STEPS"] = "8"
        cfg["MLX_MAX_MICROBATCH_TOKENS"] = rng.choice(["4096", "8192", "16384"])
    else:
        cfg["VAL_BATCH_SIZE"] = cfg["TRAIN_BATCH_TOKENS"]
        cfg["ITERATIONS"] = str(rng.choice([12000, 16000, 20000, 24000]))
        cfg["MAX_WALLCLOCK_SECONDS"] = "600"
    while not valid_shape(cfg):
        cfg["MODEL_DIM"] = rng.choice(choices["MODEL_DIM"])
        cfg["NUM_HEADS"] = rng.choice([item for item in choices["NUM_HEADS"] if int(cfg["MODEL_DIM"]) % int(item) == 0])
        cfg["NUM_KV_HEADS"] = rng.choice([item for item in choices["NUM_KV_HEADS"] if int(cfg["NUM_HEADS"]) % int(item) == 0])
        cfg["TRAIN_BATCH_TOKENS"] = rng.choice([item for item in choices["TRAIN_BATCH_TOKENS"] if int(item) % int(cfg["TRAIN_SEQ_LEN"]) == 0])
        cfg["VAL_BATCH_SIZE"] = cfg["TRAIN_BATCH_TOKENS"]
    return cfg


def crossover_config(a: dict[str, str], b: dict[str, str], backend: str, rng: random.Random) -> dict[str, str]:
    merged = dict(a)
    for key in CONFIG_KEYS:
        if key in b and rng.random() < 0.5:
            merged[key] = str(b[key])
    merged = normalize_config(backend, merged)
    while not valid_shape(merged):
        merged = mutate_config(merged, backend, rng, intensity=3)
    return merged


def build_command(backend: str, script_path: Path, nproc: int) -> list[str]:
    if backend == "cuda":
        return ["uv", "run", "torchrun", "--standalone", f"--nproc_per_node={nproc}", str(script_path)]
    return ["uv", "run", "python3", "-u", str(script_path)]


def parse_metrics(log_text: str, backend: str, script_path: Path) -> tuple[float, float, int, int, int]:
    val_match = list(VAL_RE.finditer(log_text))
    if not val_match:
        raise ValueError("missing final validation metrics")
    val_loss = float(val_match[-1].group("val_loss"))
    val_bpb = float(val_match[-1].group("val_bpb"))
    if backend == "cuda":
        total_match = list(CUDA_TOTAL_SIZE_RE.finditer(log_text))
        model_match = list(CUDA_MODEL_SIZE_RE.finditer(log_text))
        if not total_match:
            raise ValueError("missing cuda total submission size metric")
        total_bytes = int(total_match[-1].group("bytes"))
        quantized_model_bytes = int(model_match[-1].group("bytes")) if model_match else max(total_bytes - script_bytes(script_path), 0)
    else:
        model_match = list(MLX_MODEL_SIZE_RE.finditer(log_text))
        if not model_match:
            raise ValueError("missing mlx serialized model size metric")
        quantized_model_bytes = int(model_match[-1].group("bytes"))
        total_bytes = quantized_model_bytes + script_bytes(script_path)
    param_match = list(PARAM_RE.finditer(log_text))
    model_params = int(param_match[-1].group("params")) if param_match else 0
    return val_loss, val_bpb, total_bytes, quantized_model_bytes, model_params


def append_result(result: TrialResult) -> None:
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{result.run_id}\t{result.backend}\t{result.mode}\t{result.val_bpb:.8f}\t{result.val_loss:.8f}\t{result.total_bytes}\t{result.status}\t{result.preset}\t{result.code_mutation}\t{','.join(result.parents)}\t{result.train_script_path}\t{result.description}\n"
        )
    (TRIALS_DIR / f"{result.run_id}.json").write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")


def coerce_trial_result(payload: dict[str, Any], source: Path | None = None) -> TrialResult:
    backend = str(payload.get("backend", "mlx"))
    if backend not in PRESETS:
        raise ValueError(f"unsupported backend in trial artifact: {backend}")

    config_payload = payload.get("config")
    config = normalize_config(backend, config_payload if isinstance(config_payload, dict) else PRESETS[backend]["baseline"])

    default_script_path = ROOT / ("train_gpt.py" if backend == "cuda" else "train_gpt_mlx.py")
    train_script_path = str(payload.get("train_script_path") or default_script_path.relative_to(ROOT))
    resolved_script_path = ROOT / train_script_path
    if not resolved_script_path.exists():
        resolved_script_path = default_script_path
        train_script_path = str(resolved_script_path.relative_to(ROOT))

    parents_payload = payload.get("parents", [])
    if isinstance(parents_payload, str):
        parents = [item for item in parents_payload.split(",") if item]
    elif isinstance(parents_payload, list):
        parents = [str(item) for item in parents_payload]
    else:
        parents = []

    log_path = str(payload.get("log_path") or (LOG_DIR / f"{payload.get('run_id', 'legacy')}.log").relative_to(ROOT))
    quantized_model_bytes = int(payload.get("quantized_model_bytes", 0))
    train_script_bytes = int(payload.get("train_script_bytes", script_bytes(resolved_script_path)))
    total_bytes = int(payload.get("total_bytes", quantized_model_bytes + train_script_bytes))

    return TrialResult(
        run_id=str(payload.get("run_id", source.stem if source is not None else "legacy")),
        backend=backend,
        mode=str(payload.get("mode", "")),
        status=str(payload.get("status", "ok")),
        val_bpb=float(payload.get("val_bpb", 0.0)),
        val_loss=float(payload.get("val_loss", 0.0)),
        total_bytes=total_bytes,
        train_script_path=train_script_path,
        train_script_bytes=train_script_bytes,
        quantized_model_bytes=quantized_model_bytes,
        model_params=int(payload.get("model_params", 0)),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        log_path=log_path,
        preset=str(payload.get("preset", "")),
        code_mutation=str(payload.get("code_mutation", "")),
        parents=parents,
        config=config,
        description=str(payload.get("description", "")),
    )


def load_trial_result(path: Path) -> TrialResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return coerce_trial_result(payload, source=path)
    except (TypeError, ValueError):
        return None


def load_best() -> TrialResult | None:
    if not BEST_JSON.exists():
        return None
    return load_trial_result(BEST_JSON)


def save_best(best: TrialResult) -> None:
    BEST_JSON.write_text(json.dumps(asdict(best), indent=2, sort_keys=True), encoding="utf-8")


def load_population(backend: str) -> list[TrialResult]:
    population: list[TrialResult] = []
    for path in sorted(TRIALS_DIR.glob("*.json")):
        result = load_trial_result(path)
        if result is None:
            continue
        if result.backend == backend and result.status == "ok":
            population.append(result)
    population.sort(key=lambda item: item.val_bpb)
    return population


def better(candidate: TrialResult, incumbent: TrialResult | None) -> bool:
    return candidate.status == "ok" and (incumbent is None or candidate.val_bpb < incumbent.val_bpb)


def describe_delta(base: dict[str, str], candidate: dict[str, str]) -> str:
    changed = [f"{key}={candidate[key]}" for key in sorted(candidate) if base.get(key) != candidate[key]]
    return ", ".join(changed) if changed else "baseline"


def apply_config_defaults(script_text: str, cfg: dict[str, str]) -> str:
    updated = script_text
    for key, value in cfg.items():
        pattern = re.compile(rf'os\.environ\.get\("{re.escape(key)}",\s*[^)]*\)')
        replacement = f'os.environ.get("{key}", {json.dumps(str(value))})'
        updated = pattern.sub(replacement, updated)
    return updated


def apply_code_mutation(script_text: str, backend: str, mutation: str) -> str:
    updated = script_text
    for old, new in CODE_MUTATIONS[backend][mutation]:
        if old not in updated:
            raise ValueError(f"missing mutation target for {mutation}")
        updated = updated.replace(old, new)
    return updated


def create_candidate_script(run_id: str, backend: str, cfg: dict[str, str], code_mutation: str) -> Path:
    source = script_for_backend(backend)
    target = WORKBENCH_DIR / f"{run_id}_{source.name}"
    text = source.read_text(encoding="utf-8")
    text = apply_config_defaults(text, cfg)
    if code_mutation:
        text = apply_code_mutation(text, backend, code_mutation)
    target.write_text(text, encoding="utf-8")
    return target


def make_result(
    run_id: str,
    backend: str,
    mode: str,
    status: str,
    val_bpb: float,
    val_loss: float,
    total_bytes: int,
    script_path: Path,
    quantized_model_bytes: int,
    model_params: int,
    elapsed_seconds: float,
    log_path: Path,
    preset: str,
    code_mutation: str,
    parents: list[str],
    config: dict[str, str],
    description: str,
) -> TrialResult:
    return TrialResult(
        run_id=run_id,
        backend=backend,
        mode=mode,
        status=status,
        val_bpb=val_bpb,
        val_loss=val_loss,
        total_bytes=total_bytes,
        train_script_path=str(script_path.relative_to(ROOT)),
        train_script_bytes=script_bytes(script_path),
        quantized_model_bytes=quantized_model_bytes,
        model_params=model_params,
        elapsed_seconds=elapsed_seconds,
        log_path=str(log_path.relative_to(ROOT)),
        preset=preset,
        code_mutation=code_mutation,
        parents=parents,
        config=config,
        description=description,
    )


def run_trial(
    index: int,
    backend: str,
    mode: str,
    cfg: dict[str, str],
    nproc: int,
    description: str,
    preset: str = "",
    code_mutation: str = "",
    parents: list[str] | None = None,
) -> TrialResult:
    run_id = f"ar_{backend}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}_{index:03d}"
    script_path = create_candidate_script(run_id, backend, cfg, code_mutation) if mode == "code" else script_for_backend(backend)
    env = os.environ.copy()
    env.update(cfg)
    env["RUN_ID"] = run_id
    env["PYTHONUNBUFFERED"] = "1"
    log_path = LOG_DIR / f"{run_id}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(build_command(backend, script_path, nproc), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    text = read_text(log_path)
    status = "ok" if proc.returncode == 0 else "crash"
    val_loss = 0.0
    val_bpb = 0.0
    total_bytes = 0
    quantized_model_bytes = 0
    model_params = 0
    if status == "ok":
        try:
            val_loss, val_bpb, total_bytes, quantized_model_bytes, model_params = parse_metrics(text, backend, script_path)
            if total_bytes > ARTIFACT_LIMIT_BYTES:
                status = "over_limit"
        except Exception as exc:
            status = f"parse_error:{type(exc).__name__}"
    result = make_result(
        run_id,
        backend,
        mode,
        status,
        val_bpb,
        val_loss,
        total_bytes,
        script_path,
        quantized_model_bytes,
        model_params,
        elapsed,
        log_path,
        preset,
        code_mutation,
        parents or [],
        cfg,
        description,
    )
    append_result(result)
    return result


def run_preset_mode(args: argparse.Namespace, best: TrialResult | None) -> TrialResult | None:
    rng = random.Random(args.seed)
    preset_names = [args.preset] if args.preset else list(PRESETS[args.backend].keys())
    for index in range(args.trials):
        preset_name = preset_names[index % len(preset_names)] if args.preset else rng.choice(preset_names)
        cfg = normalize_config(args.backend, PRESETS[args.backend][preset_name])
        if estimated_total_bytes(cfg, script_for_backend(args.backend)) >= ARTIFACT_LIMIT_BYTES:
            continue
        result = run_trial(index, args.backend, "preset", cfg, args.nproc, f"preset:{preset_name}", preset=preset_name)
        if better(result, best):
            best = result
            save_best(best)
            print(f"new best: {best.val_bpb:.8f} {best.run_id}")
        else:
            print(f"kept best: {best.val_bpb:.8f} ({best.run_id}) status={result.status}" if best else f"no valid best yet after {result.run_id} status={result.status}")
    return best


def run_random_mode(args: argparse.Namespace, best: TrialResult | None) -> TrialResult | None:
    rng = random.Random(args.seed)
    seed_cfg = normalize_config(args.backend, best.config) if best is not None else normalize_config(args.backend, PRESETS[args.backend]["baseline"])
    for index in range(args.trials):
        candidate = find_under_limit_candidate(
            backend=args.backend,
            mode="random",
            base_cfg=seed_cfg,
            script_path=script_for_backend(args.backend),
            build_candidate=lambda: mutate_config(seed_cfg, args.backend, rng, intensity=rng.randint(2, 5)),
        )
        result = run_trial(index, args.backend, "random", candidate, args.nproc, describe_delta(seed_cfg, candidate))
        if better(result, best):
            best = result
            seed_cfg = normalize_config(args.backend, result.config)
            save_best(best)
            print(f"new best: {best.val_bpb:.8f} {best.run_id}")
        else:
            print(f"kept best: {best.val_bpb:.8f} ({best.run_id}) status={result.status}" if best else f"no valid best yet after {result.run_id} status={result.status}")
    return best


def run_evolution_mode(args: argparse.Namespace, best: TrialResult | None) -> TrialResult | None:
    rng = random.Random(args.seed)
    population = load_population(args.backend)
    pool = population[: max(2, args.population)]
    if not pool:
        for preset_name, preset_cfg in PRESETS[args.backend].items():
            pool.append(
                make_result(
                    f"seed_{preset_name}",
                    args.backend,
                    "seed",
                    "ok",
                    float("inf"),
                    float("inf"),
                    estimated_total_bytes(preset_cfg, script_for_backend(args.backend)),
                    script_for_backend(args.backend),
                    0,
                    estimate_params(preset_cfg),
                    0.0,
                    LOG_DIR / "seed.log",
                    preset_name,
                    "",
                    [],
                    normalize_config(args.backend, preset_cfg),
                    f"seed:{preset_name}",
                )
            )
    for index in range(args.trials):
        parents = rng.sample(pool, k=2 if len(pool) > 1 else 1)
        base_a = normalize_config(args.backend, parents[0].config)
        if len(parents) == 1:
            candidate = find_under_limit_candidate(
                backend=args.backend,
                mode="evolution",
                base_cfg=base_a,
                script_path=script_for_backend(args.backend),
                build_candidate=lambda: mutate_config(base_a, args.backend, rng, intensity=rng.randint(2, 4)),
            )
        else:
            base_b = normalize_config(args.backend, parents[1].config)
            candidate = find_under_limit_candidate(
                backend=args.backend,
                mode="evolution",
                base_cfg=base_a,
                script_path=script_for_backend(args.backend),
                build_candidate=lambda: mutate_config(crossover_config(base_a, base_b, args.backend, rng), args.backend, rng, intensity=rng.randint(1, 3)),
            )
        result = run_trial(index, args.backend, "evolution", candidate, args.nproc, describe_delta(base_a, candidate), parents=[item.run_id for item in parents])
        if result.status == "ok":
            pool.append(result)
            pool.sort(key=lambda item: item.val_bpb)
            pool = pool[: max(2, args.population)]
        if better(result, best):
            best = result
            save_best(best)
            print(f"new best: {best.val_bpb:.8f} {best.run_id}")
        else:
            print(f"kept best: {best.val_bpb:.8f} ({best.run_id}) status={result.status}" if best else f"no valid best yet after {result.run_id} status={result.status}")
    return best


def run_code_mode(args: argparse.Namespace, best: TrialResult | None) -> TrialResult | None:
    rng = random.Random(args.seed)
    seed_cfg = normalize_config(args.backend, best.config) if best is not None else normalize_config(args.backend, PRESETS[args.backend]["baseline"])
    mutation_names = [args.code_mutation] if args.code_mutation else list(CODE_MUTATIONS[args.backend].keys())
    for index in range(args.trials):
        mutation_name = mutation_names[index % len(mutation_names)] if args.code_mutation else rng.choice(mutation_names)
        candidate = find_under_limit_code_candidate(
            backend=args.backend,
            mode="code",
            base_cfg=seed_cfg,
            code_mutation=mutation_name,
            index=index,
            rng=rng,
        )
        result = run_trial(index, args.backend, "code", candidate, args.nproc, f"mutation:{mutation_name}; {describe_delta(seed_cfg, candidate)}", code_mutation=mutation_name)
        if better(result, best):
            best = result
            seed_cfg = normalize_config(args.backend, result.config)
            save_best(best)
            print(f"new best: {best.val_bpb:.8f} {best.run_id}")
        else:
            print(f"kept best: {best.val_bpb:.8f} ({best.run_id}) status={result.status}" if best else f"no valid best yet after {result.run_id} status={result.status}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mlx", "cuda"], default="mlx")
    parser.add_argument("--mode", choices=["random", "preset", "evolution", "code"], default="random")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-first", action="store_true")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--code-mutation", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    best = load_best() if args.resume else None
    if best is not None and best.backend != args.backend:
        best = None

    if args.baseline_first and best is None:
        baseline_cfg = normalize_config(args.backend, PRESETS[args.backend]["baseline"])
        if estimated_total_bytes(baseline_cfg, script_for_backend(args.backend)) < ARTIFACT_LIMIT_BYTES:
            baseline = run_trial(0, args.backend, "preset", baseline_cfg, args.nproc, "preset:baseline", preset="baseline")
            if better(baseline, best):
                best = baseline
                save_best(best)

    if args.mode == "preset":
        best = run_preset_mode(args, best)
    elif args.mode == "evolution":
        best = run_evolution_mode(args, best)
    elif args.mode == "code":
        best = run_code_mode(args, best)
    else:
        best = run_random_mode(args, best)

    if best is None:
        print("no valid runs found", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(asdict(best), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
