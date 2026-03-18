from __future__ import annotations

import importlib.util
import platform
from pathlib import Path
from typing import Any

from .common import REPO_ROOT, VENV_PYTHON


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def available_profiles() -> dict[str, dict[str, Any]]:
    data_path = str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")
    tokenizer_path = str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")
    profiles: dict[str, dict[str, Any]] = {
        "mlx_smoke": {
            "description": "Local Apple Silicon smoke loop with train_gpt_mlx.py",
            "track": "local-smoke",
            "launcher": "python",
            "script": str(REPO_ROOT / "train_gpt_mlx.py"),
            "supports_autoloop": platform.system() == "Darwin" and (VENV_PYTHON.is_file() or _has_module("mlx")),
            "require_challenge_ready": False,
            "run_timeout_seconds": 900,
            "idle_timeout_seconds": 180,
            "required_modules": ["mlx", "sentencepiece"],
            "base_env": {
                "DATA_PATH": data_path,
                "TOKENIZER_PATH": tokenizer_path,
                "VOCAB_SIZE": "1024",
                "ITERATIONS": "200",
                "WARMUP_STEPS": "20",
                "WARMDOWN_ITERS": "1200",
                "TRAIN_BATCH_TOKENS": "8192",
                "TRAIN_SEQ_LEN": "1024",
                "VAL_LOSS_EVERY": "0",
                "VAL_BATCH_SIZE": "8192",
                "TRAIN_LOG_EVERY": "25",
                "GRAD_ACCUM_STEPS": "8",
                "QK_GAIN_INIT": "1.5",
                "TIED_EMBED_LR": "0.05",
                "MATRIX_LR": "0.04",
                "SCALAR_LR": "0.04",
                "GRAD_CLIP_NORM": "0.0",
                "OUT_DIR": ".",
            },
        },
        "torch_single_gpu_smoke": {
            "description": "CUDA smoke loop with train_gpt.py on one GPU",
            "track": "local-cuda-smoke",
            "launcher": "torchrun",
            "script": str(REPO_ROOT / "train_gpt.py"),
            "nproc_per_node": 1,
            "supports_autoloop": True,
            "require_challenge_ready": False,
            "run_timeout_seconds": 900,
            "idle_timeout_seconds": 180,
            "required_modules": ["torch", "sentencepiece"],
            "required_gpus": 1,
            "base_env": {
                "DATA_PATH": data_path,
                "TOKENIZER_PATH": tokenizer_path,
                "VOCAB_SIZE": "1024",
                "ITERATIONS": "200",
                "WARMUP_STEPS": "20",
                "WARMDOWN_ITERS": "1200",
                "TRAIN_BATCH_TOKENS": "8192",
                "TRAIN_SEQ_LEN": "1024",
                "VAL_LOSS_EVERY": "0",
                "VAL_BATCH_SIZE": "8192",
                "TRAIN_LOG_EVERY": "25",
                "QK_GAIN_INIT": "1.5",
                "TIED_EMBED_LR": "0.05",
                "MATRIX_LR": "0.04",
                "SCALAR_LR": "0.04",
                "GRAD_CLIP_NORM": "0.0",
                "MAX_WALLCLOCK_SECONDS": "0",
            },
        },
        "torch_record_8gpu": {
            "description": "8xH100-style record-track profile using train_gpt.py",
            "track": "record-10min-16mb",
            "launcher": "torchrun",
            "script": str(REPO_ROOT / "train_gpt.py"),
            "nproc_per_node": 8,
            "supports_autoloop": True,
            "require_challenge_ready": True,
            "run_timeout_seconds": 1800,
            "idle_timeout_seconds": 180,
            "required_modules": ["torch", "sentencepiece"],
            "required_gpus": 8,
            "base_env": {
                "DATA_PATH": data_path,
                "TOKENIZER_PATH": tokenizer_path,
                "VOCAB_SIZE": "1024",
                "MAX_WALLCLOCK_SECONDS": "600",
                "WARMUP_STEPS": "20",
                "WARMDOWN_ITERS": "1200",
                "TRAIN_BATCH_TOKENS": "524288",
                "TRAIN_SEQ_LEN": "1024",
                "TRAIN_LOG_EVERY": "50",
                "VAL_LOSS_EVERY": "200",
                "QK_GAIN_INIT": "1.5",
                "TIED_EMBED_LR": "0.05",
                "MATRIX_LR": "0.04",
                "SCALAR_LR": "0.04",
                "GRAD_CLIP_NORM": "0.0",
            },
        },
        "torch_nonrecord_8gpu": {
            "description": "8-GPU non-record profile for longer exploratory training",
            "track": "nonrecord-16mb",
            "launcher": "torchrun",
            "script": str(REPO_ROOT / "train_gpt.py"),
            "nproc_per_node": 8,
            "supports_autoloop": True,
            "require_challenge_ready": True,
            "run_timeout_seconds": 21600,
            "idle_timeout_seconds": 300,
            "required_modules": ["torch", "sentencepiece"],
            "required_gpus": 8,
            "base_env": {
                "DATA_PATH": data_path,
                "TOKENIZER_PATH": tokenizer_path,
                "VOCAB_SIZE": "1024",
                "ITERATIONS": "500000",
                "WARMUP_STEPS": "20",
                "WARMDOWN_ITERS": "1200",
                "MAX_WALLCLOCK_SECONDS": "14400",
                "TRAIN_BATCH_TOKENS": "524288",
                "TRAIN_SEQ_LEN": "1024",
                "TRAIN_LOG_EVERY": "200",
                "VAL_LOSS_EVERY": "20000",
                "QK_GAIN_INIT": "1.5",
                "TIED_EMBED_LR": "0.05",
                "MATRIX_LR": "0.04",
                "SCALAR_LR": "0.04",
                "GRAD_CLIP_NORM": "0.0",
            },
        },
    }
    return profiles


def default_profile_name() -> str:
    profiles = available_profiles()
    if profiles["mlx_smoke"]["supports_autoloop"]:
        return "mlx_smoke"
    return "torch_single_gpu_smoke"


def resolve_profile(name: str | None) -> tuple[str, dict[str, Any]]:
    profiles = available_profiles()
    profile_name = name or default_profile_name()
    if profile_name not in profiles:
        known = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile {profile_name!r}. Known profiles: {known}")
    return profile_name, profiles[profile_name]
