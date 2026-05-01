"""PR1812+ART Attempt 1: triple ART RoPE-fix full 8x evaluation recipe."""

from __future__ import annotations

import collections
import copy
import datetime
import glob
import io
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

_SMOKE_TEST: bool = bool(int(os.environ.get("SMOKE_TEST", "0")))
try:
    if _SMOKE_TEST:
        raise ImportError("SMOKE_TEST: forcing SDPA fallback")
    from flash_attn_interface import flash_attn_func as _fa3_impl

    _USE_FA3 = True

    def flash_attn_3_func(q, k, v, causal: bool = True):
        return _fa3_impl(q, k, v, causal=causal)

except ImportError:
    _USE_FA3 = False

    def flash_attn_3_func(q, k, v, causal: bool = True):
        B, T, Hq, D = q.shape
        Hkv = k.shape[2]
        q_ = q.permute(0, 2, 1, 3).contiguous()
        k_ = k.permute(0, 2, 1, 3).contiguous()
        v_ = v.permute(0, 2, 1, 3).contiguous()
        if Hkv != Hq:
            repeat = Hq // Hkv
            k_ = k_.repeat_interleave(repeat, dim=1)
            v_ = v_.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal)
        return out.permute(0, 2, 1, 3).contiguous()


_SCRIPT_PATH = Path(__file__).resolve()


def _find_repo_root(start: Path) -> Path:
    for candidate in (start.parent, *start.parents):
        if (candidate / "experiments").exists() and (candidate / "data").exists():
            return candidate
    return start.parent


_REPO_ROOT = _find_repo_root(_SCRIPT_PATH)


def _unique_paths(paths) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _is_safe_recursive_search_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if resolved == resolved.parent:
        return False
    return str(resolved) not in {"/proc", "/sys", "/dev"}


def _with_vocab_substitution(raw_path: str, vocab_size: int | None) -> list[str]:
    paths = [raw_path]
    if vocab_size is not None:
        replaced = re.sub(r"fineweb10B_sp\d+", f"fineweb10B_sp{vocab_size}", raw_path)
        replaced = re.sub(
            r"fineweb_\d+_bpe\.model", f"fineweb_{vocab_size}_bpe.model", replaced
        )
        paths.insert(0, replaced)
    return list(dict.fromkeys(paths))


def _expand_path_candidates(raw_path: str, vocab_size: int | None = None) -> list[Path]:
    candidates = []
    for candidate_path in _with_vocab_substitution(raw_path, vocab_size):
        path = Path(candidate_path).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append((Path.cwd() / path).resolve())
            candidates.append((_REPO_ROOT / path).resolve())
    return _unique_paths(candidates)


def _data_root_candidates_from_path(path: Path) -> list[Path]:
    text = str(path)
    base = path.parent if any(ch in text for ch in "*?[") else path
    candidates = []
    if base.name.startswith("fineweb10B_sp") and base.parent.name == "datasets":
        candidates.append(base.parent.parent)
    if base.name == "datasets":
        candidates.append(base.parent)
    if base.name == "tokenizers":
        candidates.append(base.parent)
    if base.parent.name == "tokenizers":
        candidates.append(base.parent.parent)
    candidates.append(base)
    return _unique_paths(candidates)


def _path_mentions_other_vocab(path: Path, vocab_size: int) -> bool:
    text = str(path)
    for pattern in (r"fineweb10B_sp(\d+)", r"fineweb_(\d+)_bpe\.model"):
        for match in re.finditer(pattern, text):
            if int(match.group(1)) != vocab_size:
                return True
    return False


def _tokenizer_matches_vocab(path: Path, vocab_size: int) -> bool:
    if not path.exists():
        return False
    try:
        sp = spm.SentencePieceProcessor(model_file=str(path))
    except Exception:
        return False
    return int(sp.vocab_size()) == vocab_size


def _candidate_search_roots(
    explicit_data_dir: str | None, vocab_size: int | None = None
) -> list[Path]:
    candidates = []
    if explicit_data_dir:
        for path in _expand_path_candidates(explicit_data_dir, vocab_size=vocab_size):
            candidates.extend([path, *path.parents])
    candidates.extend([_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend(
        [
            _REPO_ROOT,
            Path("/workspace"),
            Path("/workspace/parameter-golf"),
            Path("/workspace/caseops_data"),
            Path("/workspace/caseops_data/datasets"),
        ]
    )
    return _unique_paths(
        [path for path in candidates if _is_safe_recursive_search_root(path)]
    )


def _candidate_data_roots(
    explicit_data_dir: str | None, vocab_size: int | None = None
) -> list[Path]:
    candidates = []
    if explicit_data_dir:
        for path in _expand_path_candidates(explicit_data_dir, vocab_size=vocab_size):
            candidates.extend(_data_root_candidates_from_path(path))
    candidates.extend(
        [
            _REPO_ROOT / "data",
            Path.cwd() / "data",
            Path("/workspace/parameter-golf/data"),
            Path("/workspace/caseops_data/datasets"),
        ]
    )
    return _unique_paths(candidates)


def _resolve_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_data_dir(
    explicit_data_dir: str | None, vocab_size: int | None = None
) -> str:
    candidates = _candidate_data_roots(explicit_data_dir, vocab_size=vocab_size)
    resolved = _resolve_existing_path(candidates)
    return str(resolved if resolved is not None else candidates[0])


def _resolve_glob_candidates(
    patterns: list[Path], vocab_size: int | None = None
) -> Path | None:
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(str(pattern))]
        if vocab_size is not None:
            matches = [
                path
                for path in matches
                if not _path_mentions_other_vocab(path, vocab_size)
            ]
        if matches:
            return pattern
    return None


def _search_dataset_pattern(
    split: str, vocab_size: int, roots: list[Path]
) -> Path | None:
    target_dir = f"fineweb10B_sp{vocab_size}"
    filename = f"fineweb_{split}_*.bin"
    direct_patterns = []
    for root in roots:
        direct_patterns.extend(
            [
                root / "data" / "datasets" / target_dir / filename,
                root / "datasets" / target_dir / filename,
                root / "dashboard_data" / "smoke_data" / target_dir / filename,
                root / filename,
            ]
        )
    resolved = _resolve_glob_candidates(
        _unique_paths(direct_patterns), vocab_size=vocab_size
    )
    if resolved is not None:
        return resolved

    for root in roots:
        try:
            recursive = sorted(root.glob(f"**/{target_dir}/{filename}"))
        except (PermissionError, OSError):
            continue
        usable = [path for path in recursive if "_bytes_" not in path.name]
        if usable:
            return usable[0].parent / filename
    return None


def _resolve_data_pattern(
    split: str,
    vocab_size: int,
    explicit_pattern: str | None,
    explicit_data_dir: str | None,
) -> str:
    if explicit_pattern:
        explicit_candidates = [
            path
            for path in _expand_path_candidates(explicit_pattern, vocab_size=vocab_size)
            if not _path_mentions_other_vocab(path, vocab_size)
        ]
        resolved = _resolve_glob_candidates(explicit_candidates, vocab_size=vocab_size)
        if resolved is not None:
            return str(resolved)
        if explicit_candidates:
            return str(explicit_candidates[0])

    search_roots = _candidate_search_roots(explicit_data_dir, vocab_size=vocab_size)
    resolved = _search_dataset_pattern(split, vocab_size, search_roots)
    if resolved is not None:
        return str(resolved)

    candidates = [
        root / "datasets" / f"fineweb10B_sp{vocab_size}" / f"fineweb_{split}_*.bin"
        for root in _candidate_data_roots(explicit_data_dir, vocab_size=vocab_size)
    ]
    for candidate in candidates:
        if glob.glob(str(candidate)):
            return str(candidate)
    return str(candidates[0])


def _resolve_tokenizer_path(
    vocab_size: int, explicit_tokenizer_path: str | None, explicit_data_dir: str | None
) -> str:
    if explicit_tokenizer_path:
        explicit_candidates = [
            path
            for path in _expand_path_candidates(
                explicit_tokenizer_path, vocab_size=vocab_size
            )
            if not _path_mentions_other_vocab(path, vocab_size)
        ]
        for candidate in explicit_candidates:
            if _tokenizer_matches_vocab(candidate, vocab_size):
                return str(candidate)
    filename = f"fineweb_{vocab_size}_bpe.model"
    search_roots = _candidate_search_roots(explicit_data_dir, vocab_size=vocab_size)
    candidates = []
    for root in search_roots:
        candidates.extend(
            [
                root / "data" / "tokenizers" / filename,
                root / "tokenizers" / filename,
            ]
        )
    for candidate in _unique_paths(candidates):
        if _tokenizer_matches_vocab(candidate, vocab_size):
            return str(candidate)

    for root in search_roots:
        try:
            recursive = sorted(root.glob(f"**/{filename}"))
        except (PermissionError, OSError):
            continue
        for candidate in recursive:
            if _tokenizer_matches_vocab(candidate, vocab_size):
                return str(candidate)

    candidates = [
        root / "tokenizers" / filename
        for root in _candidate_data_roots(explicit_data_dir, vocab_size=vocab_size)
    ]
    for candidate in candidates:
        if _tokenizer_matches_vocab(candidate, vocab_size):
            return str(candidate)
    return str(_unique_paths(candidates)[0])


class Hyperparameters:
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _short = uuid.uuid4().hex[:8]
    run_id = os.environ.get("RUN_ID", f"{_ts}_{_short}")
    seed = int(os.environ.get("SEED", 1337))

    requested_vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    vocab_size = 8192
    data_root_hint = os.environ.get("DATA_DIR") or os.environ.get("DATA_PATH")
    data_dir = _resolve_data_dir(data_root_hint, vocab_size=vocab_size)
    datasets_dir = os.path.dirname(
        _resolve_data_pattern(
            "train",
            vocab_size,
            os.environ.get("TRAIN_FILES"),
            data_root_hint,
        )
    )
    train_files = _resolve_data_pattern(
        "train",
        vocab_size,
        os.environ.get("TRAIN_FILES"),
        data_root_hint,
    )
    val_files = _resolve_data_pattern(
        "val",
        vocab_size,
        os.environ.get("VAL_FILES"),
        data_root_hint,
    )
    tokenizer_path = _resolve_tokenizer_path(
        vocab_size,
        os.environ.get("TOKENIZER_PATH"),
        data_root_hint,
    )

    iterations = int(os.environ.get("ITERATIONS", 50000))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    recurrence_probe_every = int(os.environ.get("RECURRENCE_PROBE_EVERY", 10))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524_288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    sliding_window_enabled = bool(int(os.environ.get("SLIDING_WINDOW_ENABLED", "0")))

    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", 512))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 2048))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))

    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.15))
    art_halt_enabled = bool(int(os.environ.get("ART_HALT_ENABLED", "1")))
    art_halt_enable_at = float(os.environ.get("ART_HALT_ENABLE_AT", 0.30))
    art_router_hidden_dim = int(os.environ.get("ART_ROUTER_HIDDEN_DIM", 64))
    art_router_lr = float(os.environ.get("ART_ROUTER_LR", 5e-4))
    art_router_wd = float(os.environ.get("ART_ROUTER_WD", 0.01))
    art_router_entropy_start = float(os.environ.get("ART_ROUTER_ENTROPY_START", 0.05))
    art_router_entropy_end = float(os.environ.get("ART_ROUTER_ENTROPY_END", 0.0))
    art_cycle_penalty = float(os.environ.get("ART_CYCLE_PENALTY", 0.002))
    art_route_group_size = int(os.environ.get("ART_ROUTE_GROUP_SIZE", 1))
    art_shard_coherent_batches = bool(
        int(os.environ.get("ART_SHARD_COHERENT_BATCHES", "0"))
    )
    _art_routed_compile_default = (
        "0" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "1"
    )
    art_routed_compile_enabled = bool(
        int(os.environ.get("ART_ROUTED_COMPILE_ENABLED", _art_routed_compile_default))
    )
    art_gptq_calibrate_routed = bool(
        int(os.environ.get("ART_GPTQ_CALIBRATE_ROUTED", "0"))
    )
    art_quant_router_fp32 = bool(int(os.environ.get("ART_QUANT_ROUTER_FP32", "0")))
    art_eval_route_stats = bool(int(os.environ.get("ART_EVAL_ROUTE_STATS", "1")))
    art_eval_sample_routes = bool(int(os.environ.get("ART_EVAL_SAMPLE_ROUTES", "1")))
    art_eval_route_threshold = float(os.environ.get("ART_EVAL_ROUTE_THRESHOLD", 0.5))
    art_quant_diag_enabled = bool(int(os.environ.get("ART_QUANT_DIAG_ENABLED", "0")))
    art_quant_diag_modes = os.environ.get(
        "ART_QUANT_DIAG_MODES", "sample,argmax,force0,force1,force2,force3"
    )
    raw_roundtrip_check_enabled = bool(
        int(os.environ.get("RAW_ROUNDTRIP_CHECK_ENABLED", "0"))
    )
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))

    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    muon_momentum_warmup_fraction = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_FRACTION", 0.22)
    )
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    adam_wd = float(os.environ.get("ADAM_WD", 0.005))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    muon_wd_mlp = float(os.environ.get("MUON_WD_MLP", 0.115))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    ema_reset_on_looping = bool(int(os.environ.get("EMA_RESET_ON_LOOPING", "1")))
    ema_reset_on_art = bool(int(os.environ.get("EMA_RESET_ON_ART", "1")))

    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 4))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 65536))
    ttt_log_every_chunks = int(os.environ.get("TTT_LOG_EVERY_CHUNKS", 1))

    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))
    lowbit_layers = os.environ.get("LOWBIT_LAYERS", "")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = max(1, 8 // world_size)

    logfile = f"logs/{run_id}.txt"
    os.makedirs("ckpt", exist_ok=True)
    model_path = "ckpt/final_model.pt"
    quantized_model_path = "ckpt/final_model.int6.ptz"


if _SMOKE_TEST:
    Hyperparameters.iterations = int(os.environ.get("ITERATIONS", "30"))
    Hyperparameters.train_batch_tokens = int(
        os.environ.get("TRAIN_BATCH_TOKENS", "32768")
    )
    Hyperparameters.train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "512"))
    Hyperparameters.eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", "512"))
    Hyperparameters.max_wallclock_seconds = float(
        os.environ.get("MAX_WALLCLOCK_SECONDS", "120.0")
    )
    Hyperparameters.val_batch_tokens = int(
        os.environ.get("VAL_BATCH_TOKENS", "2097152")
    )
    Hyperparameters.warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", "0.2"))
    Hyperparameters.val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", "0"))
    Hyperparameters.train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", "5"))
    Hyperparameters.warmup_steps = int(os.environ.get("WARMUP_STEPS", "0"))
    Hyperparameters.ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    Hyperparameters.gptq_calibration_batches = int(
        os.environ.get("GPTQ_CALIBRATION_BATCHES", "2")
    )

_logger_hparams = None


def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id("▁") != sp.unk_id(), (
        "Tokenizer must have '▁' as a token for BPB byte counting"
    )
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


class ValidationData:
    def __init__(self, h, device):
        if not Path(h.tokenizer_path).exists():
            raise FileNotFoundError(
                f"Tokenizer model not found: {h.tokenizer_path}. "
                "Set TOKENIZER_PATH or DATA_DIR/DATA_PATH to a location containing tokenizers/."
            )
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = (
            build_sentencepiece_luts(self.sp, h.vocab_size, device)
        )


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        self.shard_coherent_batches = getattr(h, "art_shard_coherent_batches", False)
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank :: h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        self._phase_epoch = [0] * len(self.files)
        self._phase_order = [self.rng.permutation(8).tolist() for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(
            self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1)
        )
        if max_phase > 0:
            k = self._phase_order[si][self._phase_epoch[si] % 8]
            self._phase_epoch[si] += 1
            phase = min(k * (max_phase + 1) // 8, max_phase)
        else:
            phase = 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        num_shards = len(self.files)
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        total = remaining.sum()
        if total <= 0:
            for si in range(num_shards):
                self._reset_shard(si)
            remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
            total = remaining.sum()
        target = device_batch_size * remaining / total
        quotas = np.floor(target).astype(np.int64)
        leftover = device_batch_size - int(quotas.sum())
        if leftover > 0:
            order = np.argsort(-(target - quotas))
            for i in order[:leftover]:
                quotas[int(i)] += 1
        quotas = np.minimum(quotas, remaining.astype(np.int64))
        shortfall = device_batch_size - int(quotas.sum())
        if shortfall > 0:
            extra = remaining.astype(np.int64) - quotas
            order = np.argsort(-extra)
            fill = 0
            while shortfall > 0 and fill < num_shards:
                j = int(order[fill])
                if extra[j] > 0:
                    quotas[j] += 1
                    extra[j] -= 1
                    shortfall -= 1
                else:
                    fill += 1
        shard_plan = np.concatenate(
            [np.full(int(q), s, dtype=np.int64) for s, q in enumerate(quotas)]
        )
        if self.shard_coherent_batches:
            if shard_plan.size > 1:
                shard_plan = np.roll(
                    shard_plan, int(self.rng.integers(0, shard_plan.size))
                )
        else:
            self.rng.shuffle(shard_plan)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            si = int(shard_plan[bi])
            if not self.start_inds[si]:
                for si2 in range(num_shards):
                    self._reset_shard(si2)
            start_ind = self.start_inds[si].pop()
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(
                np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64)
            )
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            base = self.base
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                base = self.base * scale ** (rd / (rd - 2))
            # Recompute in fp32. The non-persistent buffer is affected by
            # module-wide bf16 conversion and made fresh reloads non-equivalent.
            inv_freq = 1.0 / base ** (
                torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd
            )
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads, 2), qk_gain_init, dtype=torch.float32)
        )
        self.attn_out_gate_width = 12
        self.attn_out_gate_w = nn.Parameter(
            torch.zeros(num_heads, self.attn_out_gate_width, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        rd = self.rope_dims
        gain = self.q_gain.to(dtype=q.dtype)
        if 0 < rd < q.size(-1):
            q_rope = q[..., :rd] * gain[None, None, :, 0:1]
            q_nope = q[..., rd:] * gain[None, None, :, 1:2]
            q = torch.cat((q_rope, q_nope), dim=-1)
        else:
            q = q * gain[None, None, :, 0:1]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        gate_w = self.attn_out_gate_w.to(dtype=x.dtype)
        g = 2.0 * torch.sigmoid(F.linear(x[..., : self.attn_out_gate_width], gate_w))
        y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.silu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        train_seq_len,
        layer_idx=0,
        ln_scale=False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            x_out = (
                x_in
                + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
                + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
            )
        else:
            x_out = (
                x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            )
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[
                None, None, :
            ] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


class ARTHaltRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(F.gelu(self.fc(features.float())))


class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        self.embed_scale = nn.Parameter(
            torch.ones(h.embedding_dim, dtype=torch.float32)
        )
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                )
                for i in range(h.num_layers)
            ]
        )
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim,
                    base=h.rope_base,
                    train_seq_len=h.rope_train_seq_len,
                    rope_dims=h.rope_dims,
                )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if h.tie_embeddings
            else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        if h.parallel_residual_start >= 0:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True
        if h.num_loops > 0 and h.ln_scale:
            loop_scale = 1.0 / math.sqrt(h.num_loops + 1)
            for i in range(h.loop_start, h.loop_end + 1):
                self.blocks[i].ln_scale_factor = (
                    self.blocks[i].ln_scale_factor * loop_scale
                )

        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))

        self.art_halt_enabled = h.art_halt_enabled
        self.art_halt_runtime_active = False
        self.art_router_entropy_bonus = h.art_router_entropy_start
        self.art_cycle_penalty = h.art_cycle_penalty
        self.art_router_first = ARTHaltRouter(2 * h.model_dim, h.art_router_hidden_dim)
        self.art_router_early = ARTHaltRouter(2 * h.model_dim, h.art_router_hidden_dim)
        self.art_router = ARTHaltRouter(2 * h.model_dim, h.art_router_hidden_dim)
        self.art_router_continue_idx = 0
        self.art_router_stop_idx = 1
        self.art_route_group_size = h.art_route_group_size
        self.art_eval_sample_routes = h.art_eval_sample_routes
        self.art_eval_route_mode = "sample" if h.art_eval_sample_routes else "argmax"
        self.art_eval_route_threshold = h.art_eval_route_threshold
        self.art_eval_force_depth = -1
        self._art_eval_gate_index = 0
        self.use_routed_compiled_blocks = False
        self._routed_compiled_blocks = None
        self._compiled_art_prefix_dual = None
        self._compiled_art_tail_dual = None
        self.last_art_avg_cycles = 3.0
        self.last_art_continue_rate = 1.0
        self.last_art_router_entropy = 0.0
        self.last_art_router_loss = 0.0
        self.last_art_policy_loss = 0.0
        self.art_gate_names = ("cycle1", "cycle2", "cycle3")
        self._last_art_log_probs: Tensor | None = None
        self._last_art_entropies: Tensor | None = None
        self._last_art_cycles: Tensor | None = None
        self._last_art_gate_continue_stats: Tensor | None = None
        self._last_art_avg_cycles_stat: Tensor | None = None
        self._last_art_continue_rate_stat: Tensor | None = None
        self._last_art_router_entropy_stat: Tensor | None = None
        self._last_art_router_loss_stat: Tensor | None = None
        self._last_art_policy_loss_stat: Tensor | None = None
        self._art_early_cycle_boundary_encoder_pos = 5
        self._art_early_encoder_cycle_positions = (6, 7)
        self._art_late_cycle_boundary_decoder_pos = 0
        self._art_late_decoder_cycle_positions = (1, 2, 3)

        if self.art_halt_enabled:
            if (h.num_loops, h.loop_start, h.loop_end) != (2, 3, 5):
                raise ValueError(
                    "Triple ART halt requires NUM_LOOPS=2, LOOP_START=3, LOOP_END=5"
                )
            expected_encoder = [0, 1, 2, 3, 4, 5, 3, 4]
            if self.encoder_indices != expected_encoder:
                raise ValueError(
                    f"Triple ART halt expects encoder {expected_encoder}, got {self.encoder_indices}"
                )
            expected_decoder = [5, 3, 4, 5]
            if self.decoder_indices[:4] != expected_decoder:
                raise ValueError(
                    f"Triple ART halt expects decoder prefix {expected_decoder}, got {self.decoder_indices[:4]}"
                )

        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
            )
            if h.skip_gates_enabled
            else None
        )
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _art_router_features(self, x: Tensor) -> Tensor:
        last_token = x[:, -1, :]
        pooled = x.amax(dim=1)
        return torch.cat((last_token, pooled), dim=-1)

    def _next_art_eval_gate_index(self) -> int:
        gate_idx = int(self._art_eval_gate_index)
        self._art_eval_gate_index = gate_idx + 1
        return gate_idx

    def _art_eval_action_ids(self, probs: Tensor) -> Tensor:
        mode = self.art_eval_route_mode
        if mode == "sample":
            return torch.multinomial(probs, num_samples=1).squeeze(1)
        if mode == "threshold":
            continue_probs = probs[:, self.art_router_continue_idx]
            return torch.where(
                continue_probs >= self.art_eval_route_threshold,
                torch.full(
                    (probs.size(0),),
                    self.art_router_continue_idx,
                    device=probs.device,
                    dtype=torch.long,
                ),
                torch.full(
                    (probs.size(0),),
                    self.art_router_stop_idx,
                    device=probs.device,
                    dtype=torch.long,
                ),
            )
        if mode == "force":
            continue_gate = self._next_art_eval_gate_index() < self.art_eval_force_depth
            action_id = (
                self.art_router_continue_idx
                if continue_gate
                else self.art_router_stop_idx
            )
            return torch.full(
                (probs.size(0),), action_id, device=probs.device, dtype=torch.long
            )
        return probs.argmax(dim=-1)

    def _art_router_decision(
        self, router: ARTHaltRouter, x: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        features = self._art_router_features(x)
        if self.art_route_group_size != 1 and features.size(0) > 1:
            return self._art_router_decision_grouped(router, features)
        logits = router(features)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-9))
        entropies = -(probs * log_probs).sum(dim=1)
        if self.training and torch.is_grad_enabled():
            action_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            selected_log_probs = log_probs.gather(1, action_ids.unsqueeze(1)).squeeze(1)
        else:
            action_ids = self._art_eval_action_ids(probs)
            selected_log_probs = None
        continue_mask = action_ids == self.art_router_continue_idx
        return continue_mask, selected_log_probs, entropies

    def _art_router_decision_grouped(
        self, router: ARTHaltRouter, features: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        batch_size = features.size(0)
        group_size = (
            batch_size
            if self.art_route_group_size <= 0
            else min(self.art_route_group_size, batch_size)
        )
        num_groups = (batch_size + group_size - 1) // group_size
        group_ids = torch.arange(batch_size, device=features.device) // group_size
        if batch_size % group_size == 0:
            group_features = features.reshape(num_groups, group_size, -1).mean(dim=1)
        else:
            group_features = features.new_zeros(num_groups, features.size(-1))
            group_features.index_add_(0, group_ids, features)
            group_counts = torch.bincount(group_ids, minlength=num_groups).to(
                dtype=features.dtype
            )
            group_features = group_features / group_counts.clamp_min(1.0).unsqueeze(1)

        logits = router(group_features)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-9))
        group_entropies = -(probs * log_probs).sum(dim=1)
        if self.training and torch.is_grad_enabled():
            group_action_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            group_selected_log_probs = log_probs.gather(
                1, group_action_ids.unsqueeze(1)
            ).squeeze(1)
        else:
            group_action_ids = self._art_eval_action_ids(probs)
            group_selected_log_probs = None

        action_ids = group_action_ids.index_select(0, group_ids)
        selected_log_probs = (
            group_selected_log_probs.index_select(0, group_ids)
            if group_selected_log_probs is not None
            else None
        )
        entropies = group_entropies.index_select(0, group_ids)
        continue_mask = action_ids == self.art_router_continue_idx
        return continue_mask, selected_log_probs, entropies

    def _art_router_anchor(self, ref: Tensor) -> Tensor:
        anchor = torch.zeros((), device=ref.device, dtype=torch.float32)
        for param in self.art_router_first.parameters():
            anchor = anchor + param.reshape(-1)[0].float() * 0.0
        for param in self.art_router_early.parameters():
            anchor = anchor + param.reshape(-1)[0].float() * 0.0
        for param in self.art_router.parameters():
            anchor = anchor + param.reshape(-1)[0].float() * 0.0
        return anchor.to(dtype=ref.dtype)

    def _art_single_router_anchor(self, router: ARTHaltRouter, ref: Tensor) -> Tensor:
        anchor = torch.zeros((), device=ref.device, dtype=torch.float32)
        for param in router.parameters():
            anchor = anchor + param.reshape(-1)[0].float() * 0.0
        return anchor.to(dtype=ref.dtype)

    def _block_param_anchor(
        self, block_indices: tuple[int, ...], ref: Tensor
    ) -> Tensor:
        anchor = torch.zeros((), device=ref.device, dtype=torch.float32)
        for block_idx in block_indices:
            for param in self.blocks[block_idx].parameters():
                anchor = anchor + param.reshape(-1)[0].float() * 0.0
        return anchor.to(dtype=ref.dtype)

    def prepare_routed_compiled_blocks(self) -> None:
        if _SMOKE_TEST:
            return
        if self._routed_compiled_blocks is None:
            self._routed_compiled_blocks = [
                torch.compile(block, dynamic=True, fullgraph=True)
                for block in self.blocks
            ]

    def _run_block(self, x: Tensor, x0: Tensor, block_idx: int) -> Tensor:
        if self.use_routed_compiled_blocks and self._routed_compiled_blocks is not None:
            return self._routed_compiled_blocks[block_idx](x, x0)
        return self.blocks[block_idx](x, x0)

    def warm_routed_compiled_blocks(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if _SMOKE_TEST:
            return
        self.prepare_routed_compiled_blocks()
        previous_enabled = self.use_routed_compiled_blocks
        previous_training = self.training
        self.use_routed_compiled_blocks = True
        self.train()
        dim = int(self.skip_weights.size(1))
        block_indices = sorted(set(self.encoder_indices + self.decoder_indices))
        for block_idx in block_indices:
            x = torch.zeros(
                batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True
            )
            x0 = torch.zeros(
                batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True
            )
            loss = self._run_block(x, x0, block_idx).float().mean()
            loss.backward()
            self.zero_grad(set_to_none=True)
        self.use_routed_compiled_blocks = previous_enabled
        self.train(previous_training)

    def _project_logits(self, x: Tensor) -> Tensor:
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _forward_art_prefix_triple(self, input_ids: Tensor):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x * self.embed_scale.to(x.dtype), (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        for i in self.encoder_indices[:3]:
            x = self._run_block(x, x0, i)
            skips.append(x)
        return x, x0, skips[0], skips[1], skips[2]

    def _forward_art_tail_triple(
        self,
        x: Tensor,
        x0: Tensor,
        s0: Tensor,
        s1: Tensor,
        s2: Tensor,
        first_continue_idx: Tensor,
        active_s3: Tensor | None,
    ) -> Tensor:
        if first_continue_idx.numel() > 0 and active_s3 is not None:
            scaled_s3 = (
                self.skip_weights[4].to(dtype=x.dtype)[None, None, :] * active_s3
            )
            if first_continue_idx.numel() == x.size(0):
                x = self._apply_skip(x, scaled_s3, 4)
            else:
                full_scaled_s3 = torch.zeros_like(x)
                full_scaled_s3.index_copy_(0, first_continue_idx, scaled_s3)
                x = self._apply_skip_subset(x, full_scaled_s3, 4, first_continue_idx)
        x = self._run_block(x, x0, self.decoder_indices[4])
        for skip_idx, block_idx, skip in (
            (5, self.decoder_indices[5], s2),
            (6, self.decoder_indices[6], s1),
            (7, self.decoder_indices[7], s0),
        ):
            scaled_skip = (
                self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skip
            )
            x = self._apply_skip(x, scaled_skip, skip_idx)
            x = self._run_block(x, x0, block_idx)
        x = self._run_block(x, x0, self.decoder_indices[8])
        return self._project_logits(x)

    def _forward_logits_trunk_only(self, input_ids: Tensor) -> Tensor:
        x, x0, s0, s1, s2 = self._forward_art_prefix_triple(input_ids)
        empty_idx = torch.empty(0, device=x.device, dtype=torch.long)
        if self.training and torch.is_grad_enabled():
            x = x + self._art_router_anchor(x).view(1, 1, 1)
            x = x + self._block_param_anchor((3, 4, 5), x).view(1, 1, 1)
        return self._forward_art_tail_triple(x, x0, s0, s1, s2, empty_idx, None)

    def _forward_logits_art_triple_segmented(self, input_ids: Tensor) -> Tensor:
        x, x0, s0, s1, s2 = self._forward_art_prefix_triple(input_ids)
        (
            first_continue_mask,
            first_log_probs,
            first_entropies,
        ) = self._art_router_decision(self.art_router_first, x)
        first_continue_idx = first_continue_mask.nonzero(as_tuple=False).flatten()
        art_cycles = first_continue_mask.to(dtype=torch.float32)
        art_log_prob_accum = first_log_probs
        art_entropy_accum = first_entropies
        active_s3_for_tail = None
        early_continue_rate = x.new_zeros((), dtype=torch.float32)
        late_continue_rate = x.new_zeros((), dtype=torch.float32)

        if first_continue_idx.numel() > 0:
            if first_continue_idx.numel() == x.size(0):
                active_x = x
                active_x0 = x0
            else:
                active_x = x.index_select(0, first_continue_idx)
                active_x0 = x0.index_select(0, first_continue_idx)

            active_x, active_s3, active_s4, active_s5 = (
                self._run_art_first_cycle_active(active_x, active_x0)
            )
            active_s3_for_tail = active_s3
            (
                early_continue_subset,
                early_log_probs,
                early_entropies,
            ) = self._art_router_decision(self.art_router_early, active_x)
            early_continue_mask = torch.zeros_like(first_continue_mask)
            early_continue_mask.index_copy_(
                0, first_continue_idx, early_continue_subset
            )
            early_continue_subset_idx = early_continue_subset.nonzero(
                as_tuple=False
            ).flatten()
            early_continue_rate = early_continue_subset.to(dtype=torch.float32).mean()
            art_cycles = art_cycles + early_continue_mask.to(dtype=torch.float32)
            if early_log_probs is not None:
                if art_log_prob_accum is None:
                    art_log_prob_accum = x.new_zeros(x.size(0), dtype=torch.float32)
                art_log_prob_accum = art_log_prob_accum.index_copy(
                    0,
                    first_continue_idx,
                    art_log_prob_accum.index_select(0, first_continue_idx)
                    + early_log_probs,
                )
            art_entropy_accum = art_entropy_accum.index_copy(
                0,
                first_continue_idx,
                art_entropy_accum.index_select(0, first_continue_idx) + early_entropies,
            )

            if early_continue_subset_idx.numel() > 0:
                second_x = active_x.index_select(0, early_continue_subset_idx)
                second_x0 = active_x0.index_select(0, early_continue_subset_idx)
                second_s5 = active_s5.index_select(0, early_continue_subset_idx)
                second_s4 = active_s4.index_select(0, early_continue_subset_idx)
                second_x, second_s6 = self._run_art_second_cycle_active(
                    second_x, second_x0
                )
                (
                    late_continue_subset,
                    late_log_probs,
                    late_entropies,
                ) = self._art_router_decision(self.art_router, second_x)
                second_continue_idx = first_continue_idx.index_select(
                    0, early_continue_subset_idx
                )
                late_continue_mask = torch.zeros_like(first_continue_mask)
                late_continue_mask.index_copy_(
                    0, second_continue_idx, late_continue_subset
                )
                late_continue_subset_idx = late_continue_subset.nonzero(
                    as_tuple=False
                ).flatten()
                late_continue_rate = late_continue_subset.to(dtype=torch.float32).mean()
                art_cycles = art_cycles + late_continue_mask.to(dtype=torch.float32)
                if late_log_probs is not None:
                    if art_log_prob_accum is None:
                        art_log_prob_accum = x.new_zeros(x.size(0), dtype=torch.float32)
                    art_log_prob_accum = art_log_prob_accum.index_copy(
                        0,
                        second_continue_idx,
                        art_log_prob_accum.index_select(0, second_continue_idx)
                        + late_log_probs,
                    )
                art_entropy_accum = art_entropy_accum.index_copy(
                    0,
                    second_continue_idx,
                    art_entropy_accum.index_select(0, second_continue_idx)
                    + late_entropies,
                )

                if late_continue_subset_idx.numel() > 0:
                    if late_continue_subset_idx.numel() == second_x.size(0):
                        second_x = self._run_art_third_cycle_active(
                            second_x, second_x0, second_s6, second_s5, second_s4
                        )
                    else:
                        late_x = second_x.index_select(0, late_continue_subset_idx)
                        late_x0 = second_x0.index_select(0, late_continue_subset_idx)
                        late_s6 = second_s6.index_select(0, late_continue_subset_idx)
                        late_s5 = second_s5.index_select(0, late_continue_subset_idx)
                        late_s4 = second_s4.index_select(0, late_continue_subset_idx)
                        late_x = self._run_art_third_cycle_active(
                            late_x, late_x0, late_s6, late_s5, late_s4
                        )
                        second_x = second_x.index_copy(
                            0, late_continue_subset_idx, late_x
                        )
                active_x = active_x.index_copy(0, early_continue_subset_idx, second_x)
            elif self.training and torch.is_grad_enabled():
                x = x + self._art_single_router_anchor(self.art_router, x).view(1, 1, 1)
            x = self._scatter_subset(x, first_continue_idx, active_x)
        else:
            if self.training and torch.is_grad_enabled():
                x = x + self._art_single_router_anchor(self.art_router_early, x).view(
                    1, 1, 1
                )
                x = x + self._art_single_router_anchor(self.art_router, x).view(1, 1, 1)

        self._last_art_log_probs = art_log_prob_accum
        self._last_art_entropies = art_entropy_accum
        self._last_art_cycles = art_cycles
        self._last_art_gate_continue_stats = torch.stack(
            (
                first_continue_mask.to(dtype=torch.float32).mean(),
                early_continue_rate,
                late_continue_rate,
            )
        ).detach()
        self._last_art_continue_rate_stat = (art_cycles / 3.0).mean().detach()
        self._last_art_avg_cycles_stat = art_cycles.mean().detach()
        self._last_art_router_entropy_stat = art_entropy_accum.mean().detach()

        return self._forward_art_tail_triple(
            x, x0, s0, s1, s2, first_continue_idx, active_s3_for_tail
        )

    def sync_art_logging_stats(self) -> None:
        if self._last_art_avg_cycles_stat is not None:
            self.last_art_avg_cycles = float(self._last_art_avg_cycles_stat.item())
        if self._last_art_continue_rate_stat is not None:
            self.last_art_continue_rate = float(
                self._last_art_continue_rate_stat.item()
            )
        if self._last_art_router_entropy_stat is not None:
            self.last_art_router_entropy = float(
                self._last_art_router_entropy_stat.item()
            )
        if self._last_art_router_loss_stat is not None:
            self.last_art_router_loss = float(self._last_art_router_loss_stat.item())
        if self._last_art_policy_loss_stat is not None:
            self.last_art_policy_loss = float(self._last_art_policy_loss_stat.item())

    def _apply_skip(self, x: Tensor, scaled_skip: Tensor, skip_idx: int) -> Tensor:
        if self.skip_gates is not None:
            g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[
                None, None, :
            ]
            return torch.lerp(scaled_skip, x, g)
        return x + scaled_skip

    def _apply_skip_subset(
        self,
        x: Tensor,
        scaled_skip: Tensor,
        skip_idx: int,
        active_idx: Tensor,
    ) -> Tensor:
        if active_idx.numel() == 0:
            return x
        if active_idx.numel() == x.size(0):
            return self._apply_skip(x, scaled_skip, skip_idx)
        next_x = x.clone()
        next_x.index_copy_(
            0,
            active_idx,
            self._apply_skip(
                x.index_select(0, active_idx),
                scaled_skip.index_select(0, active_idx),
                skip_idx,
            ),
        )
        return next_x

    def _scatter_subset(
        self, x: Tensor, active_idx: Tensor, active_x: Tensor
    ) -> Tensor:
        if active_idx.numel() == 0:
            return x
        if active_idx.numel() == x.size(0):
            return active_x
        return x.index_copy(0, active_idx, active_x)

    def _run_block_subset(
        self,
        x: Tensor,
        x0: Tensor,
        block_idx: int,
        active_idx: Tensor,
    ) -> Tensor:
        if active_idx.numel() == 0:
            return x
        if active_idx.numel() == x.size(0):
            return self._run_block(x, x0, block_idx)
        next_x = x.clone()
        next_x.index_copy_(
            0,
            active_idx,
            self._run_block(
                x.index_select(0, active_idx),
                x0.index_select(0, active_idx),
                block_idx,
            ),
        )
        return next_x

    def _run_art_first_cycle_active(
        self, active_x: Tensor, active_x0: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        active_x = self._run_block(active_x, active_x0, self.encoder_indices[3])
        active_s3 = active_x
        active_x = self._run_block(active_x, active_x0, self.encoder_indices[4])
        active_s4 = active_x
        active_x = self._run_block(active_x, active_x0, self.encoder_indices[5])
        active_s5 = active_x
        return active_x, active_s3, active_s4, active_s5

    def _run_art_second_cycle_active(
        self, active_x: Tensor, active_x0: Tensor
    ) -> tuple[Tensor, Tensor]:
        active_x = self._run_block(active_x, active_x0, self.encoder_indices[6])
        active_s6 = active_x
        active_x = self._run_block(active_x, active_x0, self.encoder_indices[7])
        scaled_skip = (
            self.skip_weights[0].to(dtype=active_x.dtype)[None, None, :] * active_x
        )
        active_x = self._apply_skip(active_x, scaled_skip, 0)
        active_x = self._run_block(active_x, active_x0, self.decoder_indices[0])
        return active_x, active_s6

    def _run_art_third_cycle_active(
        self,
        active_x: Tensor,
        active_x0: Tensor,
        active_s6: Tensor,
        active_s5: Tensor,
        active_s4: Tensor,
    ) -> Tensor:
        for skip_idx, block_idx, skip in (
            (1, self.decoder_indices[1], active_s6),
            (2, self.decoder_indices[2], active_s5),
            (3, self.decoder_indices[3], active_s4),
        ):
            scaled_skip = (
                self.skip_weights[skip_idx].to(dtype=active_x.dtype)[None, None, :]
                * skip
            )
            active_x = self._apply_skip(active_x, scaled_skip, skip_idx)
            active_x = self._run_block(active_x, active_x0, block_idx)
        return active_x

    def forward_logits(
        self, input_ids: Tensor, art_halt_active: bool | None = None
    ) -> Tensor:
        art_active = (
            self.art_halt_enabled
            and self.looping_active
            and (
                self.art_halt_runtime_active
                if art_halt_active is None
                else art_halt_active
            )
        )
        art_early_continue_mask: Tensor | None = None
        art_early_continue_idx: Tensor | None = None
        art_late_continue_mask: Tensor | None = None
        art_late_continue_idx: Tensor | None = None
        art_cycles: Tensor | None = None
        art_log_prob_accum: Tensor | None = None
        art_entropy_accum: Tensor | None = None
        art_early_decision_made = False
        art_late_decision_made = False
        self._last_art_log_probs = None
        self._last_art_entropies = None
        self._last_art_cycles = None
        self._last_art_gate_continue_stats = None
        self._last_art_avg_cycles_stat = None
        self._last_art_continue_rate_stat = None
        self._last_art_router_entropy_stat = None
        self._last_art_router_loss_stat = None
        self._last_art_policy_loss_stat = None
        self.last_art_policy_loss = 0.0
        self.last_art_router_loss = 0.0
        self.last_art_router_entropy = 0.0
        self.last_art_continue_rate = 1.0
        self.last_art_avg_cycles = 3.0 if self.looping_active else 0.0
        if not (self.training and torch.is_grad_enabled()):
            self._art_eval_gate_index = 0
        if art_active:
            return self._forward_logits_art_triple_segmented(input_ids)
        if not self.looping_active:
            return self._forward_logits_trunk_only(input_ids)

        x = self.tok_emb(input_ids)
        x = F.rms_norm(x * self.embed_scale.to(x.dtype), (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else (0, 1, 2)
        dec_iter = self.decoder_indices if self.looping_active else (6, 7, 8, 9, 10)
        for enc_pos, i in enumerate(enc_iter):
            in_early_cycle = (
                art_active
                and art_early_decision_made
                and enc_pos in self._art_early_encoder_cycle_positions
                and art_early_continue_idx is not None
            )
            if in_early_cycle:
                x = self._run_block_subset(x, x0, i, art_early_continue_idx)
            else:
                x = self._run_block(x, x0, i)
            skips.append(x)
            if (
                art_active
                and not art_early_decision_made
                and enc_pos == self._art_early_cycle_boundary_encoder_pos
            ):
                (
                    art_early_continue_mask,
                    selected_log_probs,
                    entropies,
                ) = self._art_router_decision(self.art_router_early, x)
                art_early_continue_idx = art_early_continue_mask.nonzero(
                    as_tuple=False
                ).flatten()
                art_cycles = 1.0 + art_early_continue_mask.to(dtype=torch.float32)
                if selected_log_probs is not None:
                    art_log_prob_accum = selected_log_probs
                art_entropy_accum = entropies
                self._last_art_log_probs = art_log_prob_accum
                self._last_art_entropies = art_entropy_accum
                self._last_art_cycles = art_cycles
                self._last_art_continue_rate_stat = (
                    ((art_cycles - 1.0) / 2.0).mean().detach()
                )
                self._last_art_avg_cycles_stat = art_cycles.mean().detach()
                self._last_art_router_entropy_stat = entropies.mean().detach()
                art_early_decision_made = True
        for skip_idx, i in enumerate(dec_iter):
            in_second_cycle = (
                art_active
                and art_early_decision_made
                and skip_idx == self._art_late_cycle_boundary_decoder_pos
                and art_early_continue_idx is not None
            )
            in_late_cycle = (
                art_active
                and art_late_decision_made
                and skip_idx in self._art_late_decoder_cycle_positions
                and art_late_continue_idx is not None
            )
            art_slot_idx = None
            if in_second_cycle:
                art_slot_idx = art_early_continue_idx
            elif in_late_cycle:
                art_slot_idx = art_late_continue_idx
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = (
                    self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
                if art_slot_idx is not None:
                    x = self._apply_skip_subset(x, scaled_skip, skip_idx, art_slot_idx)
                else:
                    x = self._apply_skip(x, scaled_skip, skip_idx)
            if art_slot_idx is not None:
                x = self._run_block_subset(x, x0, i, art_slot_idx)
            else:
                x = self._run_block(x, x0, i)
            if (
                art_active
                and art_early_decision_made
                and not art_late_decision_made
                and skip_idx == self._art_late_cycle_boundary_decoder_pos
                and art_early_continue_idx is not None
                and art_cycles is not None
            ):
                active_idx = art_early_continue_idx
                art_late_continue_mask = torch.zeros_like(art_early_continue_mask)
                if active_idx.numel() > 0:
                    (
                        late_continue_subset,
                        selected_log_probs,
                        entropies,
                    ) = self._art_router_decision(
                        self.art_router, x.index_select(0, active_idx)
                    )
                    art_late_continue_mask.index_copy_(
                        0, active_idx, late_continue_subset
                    )
                    late_continue_subset_idx = late_continue_subset.nonzero(
                        as_tuple=False
                    ).flatten()
                    art_late_continue_idx = active_idx.index_select(
                        0, late_continue_subset_idx
                    )
                    art_cycles = art_cycles + art_late_continue_mask.to(
                        dtype=torch.float32
                    )
                    if selected_log_probs is not None:
                        if art_log_prob_accum is None:
                            art_log_prob_accum = x.new_zeros(
                                x.size(0), dtype=torch.float32
                            )
                        art_log_prob_accum = art_log_prob_accum.index_copy(
                            0,
                            active_idx,
                            art_log_prob_accum.index_select(0, active_idx)
                            + selected_log_probs,
                        )
                    if art_entropy_accum is None:
                        art_entropy_accum = x.new_zeros(x.size(0), dtype=torch.float32)
                    art_entropy_accum = art_entropy_accum.index_copy(
                        0,
                        active_idx,
                        art_entropy_accum.index_select(0, active_idx) + entropies,
                    )
                self._last_art_log_probs = art_log_prob_accum
                self._last_art_entropies = art_entropy_accum
                self._last_art_cycles = art_cycles
                self._last_art_continue_rate_stat = (
                    ((art_cycles - 1.0) / 2.0).mean().detach()
                )
                self._last_art_avg_cycles_stat = art_cycles.mean().detach()
                if art_entropy_accum is not None:
                    self._last_art_router_entropy_stat = (
                        art_entropy_accum.mean().detach()
                    )
                art_late_decision_made = True

        x = self.final_norm(x)
        if (
            not art_early_decision_made
            and not art_late_decision_made
            and self.training
            and torch.is_grad_enabled()
        ):
            x = x + self._art_router_anchor(x).view(1, 1, 1)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        art_halt_active: bool | None = None,
    ) -> Tensor:
        logits = self.forward_logits(input_ids, art_halt_active=art_halt_active)
        art_loss_active = (
            self.art_halt_enabled
            and self.looping_active
            and (
                self.art_halt_runtime_active
                if art_halt_active is None
                else art_halt_active
            )
            and self.training
            and torch.is_grad_enabled()
        )
        if not art_loss_active:
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                target_ids.reshape(-1),
                reduction="mean",
            )
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        seq_losses = token_losses.mean(dim=1)
        main_loss = seq_losses.mean()
        total_loss = main_loss
        router_policy_loss = main_loss.new_zeros(())
        router_loss = main_loss.new_zeros(())
        if (
            self.art_halt_enabled
            and self.art_halt_runtime_active
            and self.training
            and torch.is_grad_enabled()
            and self._last_art_log_probs is not None
            and self._last_art_entropies is not None
            and self._last_art_cycles is not None
        ):
            router_costs = (
                seq_losses.detach() + self.art_cycle_penalty * self._last_art_cycles
            )
            advantages = router_costs - router_costs.mean()
            router_policy_loss = (advantages * self._last_art_log_probs).mean()
            router_mean_entropy = self._last_art_entropies.mean()
            router_loss = (
                router_policy_loss - self.art_router_entropy_bonus * router_mean_entropy
            )
            total_loss = total_loss + router_loss
        self._last_art_policy_loss_stat = router_policy_loss.detach()
        self._last_art_router_loss_stat = router_loss.detach()
        if (
            self._last_art_entropies is not None
            and self._last_art_entropies.numel() > 0
        ):
            self._last_art_router_entropy_stat = (
                self._last_art_entropies.mean().detach()
            )
        return total_loss


def classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
        row_normalize=False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if group.get("row_normalize", False):
                        row_norms = (
                            g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                        )
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,attn_out_gate",
    ).split(",")
    if pattern
)


class Optimizers:
    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_attn_params = []
        matrix_mlp_params = []
        for name, p in block_named_params:
            if p.ndim != 2 or any(
                pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS
            ):
                continue
            if ".mlp." in name:
                matrix_mlp_params.append(p)
            else:
                matrix_attn_params.append(p)
        scalar_params = [
            p
            for (name, p) in block_named_params
            if p.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if hasattr(base_model, "embed_scale"):
            scalar_params.append(base_model.embed_scale)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [
            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
        ]
        self.optimizer_tok = torch.optim.AdamW(
            tok_params,
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=True,
        )
        self.optimizer_muon = Muon(
            [
                {"params": matrix_attn_params, "weight_decay": h.muon_wd},
                {"params": matrix_mlp_params, "weight_decay": h.muon_wd_mlp},
            ],
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [
            self.optimizer_tok,
            self.optimizer_muon,
            self.optimizer_scalar,
        ]
        if h.art_halt_enabled:
            router_params = (
                list(base_model.art_router_first.parameters())
                + list(base_model.art_router_early.parameters())
                + list(base_model.art_router.parameters())
            )
            self.optimizer_art_router = torch.optim.AdamW(
                [
                    {
                        "params": router_params,
                        "lr": h.art_router_lr,
                        "base_lr": h.art_router_lr,
                    }
                ],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                weight_decay=h.art_router_wd,
                fused=True,
            )
            self.optimizers.append(self.optimizer_art_router)
        else:
            self.optimizer_art_router = None
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [
                    {
                        "params": [base_model.lm_head.weight],
                        "lr": h.head_lr,
                        "base_lr": h.head_lr,
                    }
                ],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, (CastedLinear, ARTHaltRouter)):
            module.float()
    for name, param in model.named_parameters():
        if (
            param.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ) and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hessian_divisors = {}
    expected_hessian_names = []
    hooks = []
    capture_names = None
    capture_divisor = max(int(n_calibration_batches), 1)
    previous_routed_blocks = getattr(model, "use_routed_compiled_blocks", None)
    previous_looping_active = getattr(model, "looping_active", None)
    previous_art_runtime_active = getattr(model, "art_halt_runtime_active", None)
    previous_training = model.training

    def make_hook(name):
        def hook_fn(module, inp, out):
            if capture_names is not None and name not in capture_names:
                return
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
                hessian_divisors[name] = capture_divisor
            hessians[name].addmm_(x.T, x)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hessian_name = name + ".weight"
                expected_hessian_names.append(hessian_name)
                hooks.append(module.register_forward_hook(make_hook(hessian_name)))
    if model.tie_embeddings:
        hook_module = (
            model.head_proj if model.head_proj is not None else model.final_norm
        )

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                if capture_names is not None and name not in capture_names:
                    return
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                    hessian_divisors[name] = capture_divisor
                hessians[name].addmm_(x.T, x)

            return hook_fn

        expected_hessian_names.append("tok_emb.weight")
        hooks.append(
            hook_module.register_forward_hook(make_output_hook("tok_emb.weight"))
        )

    def run_calibration_pass(num_batches, routed_art):
        nonlocal capture_divisor
        capture_divisor = max(int(num_batches), 1)
        if previous_routed_blocks is not None:
            model.use_routed_compiled_blocks = False
        if previous_looping_active is not None:
            model.looping_active = True
        if previous_art_runtime_active is not None:
            model.art_halt_runtime_active = bool(routed_art)
        with torch.no_grad():
            for _ in range(num_batches):
                x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
                model.forward_logits(x)

    routed_calibration = bool(
        h.art_halt_enabled and getattr(h, "art_gptq_calibrate_routed", True)
    )
    model.eval()
    try:
        if routed_calibration:
            log("GPTQ:calibration path routed_art")
            run_calibration_pass(n_calibration_batches, routed_art=True)
            missing = sorted(set(expected_hessian_names) - set(hessians))
            if missing:
                fallback_batches = max(4, min(n_calibration_batches, 16))
                log(
                    "GPTQ:routed calibration missed "
                    f"{len(missing)} tensors; fallback fixed_full batches={fallback_batches}"
                )
                capture_names = set(missing)
                run_calibration_pass(fallback_batches, routed_art=False)
                capture_names = None
        else:
            log("GPTQ:calibration path fixed_full")
            run_calibration_pass(n_calibration_batches, routed_art=False)
    finally:
        capture_names = None
        if previous_routed_blocks is not None:
            model.use_routed_compiled_blocks = previous_routed_blocks
        if previous_looping_active is not None:
            model.looping_active = previous_looping_active
        if previous_art_runtime_active is not None:
            model.art_halt_runtime_active = previous_art_runtime_active
        model.train(previous_training)
        for hook in hooks:
            hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / max(float(hessian_divisors[name]), 1.0)
    return hessians


def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=32):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H.diagonal().mul_(1.01)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


def _parse_lowbit_map(spec):
    m = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if item:
            pat, b = item.rsplit(":", 1)
            m[pat.strip()] = int(b)
    return m


def _is_art_router_tensor(name):
    return name.startswith("art_router")


def gptq_mixed_quantize(state_dict, hessians, h):
    lowbit_map = _parse_lowbit_map(getattr(h, "lowbit_layers", ""))
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            if (
                t.is_floating_point()
                and getattr(h, "art_quant_router_fp32", True)
                and _is_art_router_tensor(name)
            ):
                result[name] = t.to(torch.float32)
                meta[name] = "passthrough (float32)"
            else:
                result[name] = t.to(torch.float16) if t.is_floating_point() else t
                meta[name] = "passthrough (float16)"
            continue
        cs = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        if "tok_emb" in name:
            bits = h.embed_bits
        else:
            bits = h.matrix_bits
            for pat, b in lowbit_map.items():
                if pat in name:
                    bits = b
                    break
        if name not in hessians:
            raise KeyError(
                f"Missing GPTQ Hessian for {name}. "
                "Routed calibration should collect this or fill it via fixed-full fallback."
            )
        q, s = gptq_quantize_weight(
            t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1
        )
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub(r"\.\d+$", "", re.sub(r"blocks\.\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (
                q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
            ).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


_BSHF_MAGIC = b"BSHF"


def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off : dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off : src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == "brotli":
        import brotli

        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _decompress(data, compressor):
    if compressor == "brotli":
        import brotli

        return _byte_unshuffle(brotli.decompress(data))
    raise ValueError(f"Unknown compressor: {compressor!r}")


def serialize(h, base_model):
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device("cuda", h.local_rank)
    log("GPTQ:collecting Hessians from calibration data...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
    return quant_file_bytes


def _move_state_to_device(state, device):
    return {
        name: tensor.to(device=device, non_blocking=True)
        if isinstance(tensor, torch.Tensor)
        else tensor
        for name, tensor in state.items()
    }


def load_state_exact(model, state, device):
    state = _move_state_to_device(state, device)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state, strict=True)
    restore_fp32_params(model)


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu"
    )
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    load_state_exact(eval_model, deq_state, device)
    return eval_model


def deserialize_raw_checkpoint(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    raw_state = torch.load(h.model_path, map_location="cpu")
    load_state_exact(eval_model, raw_state, device)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def compile_target(target, allow_graph_breaks: bool):
    if _SMOKE_TEST:
        return target
    if allow_graph_breaks:
        # ART routing uses dynamic subset execution. Letting Dynamo graph-break there
        # recompiles helper frames for per-layer constants until it hits the limit.
        return target
    return torch.compile(target, dynamic=False, fullgraph=not allow_graph_breaks)


def configure_art_eval_runtime(model, h):
    if h.num_loops > 0:
        model.looping_active = True
    if h.art_halt_enabled:
        model.art_halt_runtime_active = True
        model.art_router_entropy_bonus = h.art_router_entropy_end
        model.art_eval_sample_routes = h.art_eval_sample_routes
        model.art_eval_route_mode = "sample" if h.art_eval_sample_routes else "argmax"
        model.art_eval_route_threshold = h.art_eval_route_threshold
        model.art_eval_force_depth = -1
        # Keep post-serialization eval eager for triple ART. Per-block compiled
        # routing recompiles across block-local constants and can hit Dynamo's limit.
        model.use_routed_compiled_blocks = False


def compare_model_states(label, reference_model, candidate_model):
    reference_sd = reference_model.state_dict()
    candidate_sd = candidate_model.state_dict()
    missing = sorted(set(reference_sd) ^ set(candidate_sd))
    dtype_mismatches = 0
    max_abs = 0.0
    for name, ref in reference_sd.items():
        if name not in candidate_sd:
            continue
        cand = candidate_sd[name]
        if ref.dtype != cand.dtype:
            dtype_mismatches += 1
        if ref.is_floating_point() or cand.is_floating_point():
            diff = (
                ref.detach().float()
                - cand.detach().to(device=ref.device, dtype=torch.float32)
            ).abs()
            max_abs = max(max_abs, float(diff.max().item()) if diff.numel() else 0.0)
        else:
            mismatch = ref.detach() != cand.detach().to(device=ref.device)
            max_abs = max(max_abs, float(mismatch.any().item()))
    log(
        f"{label}_state_compare:tensors:{len(reference_sd)} "
        f"missing:{len(missing)} dtype_mismatches:{dtype_mismatches} "
        f"max_abs:{max_abs:.9g}"
    )


def format_art_probe_summary(
    step: int,
    total_steps: int,
    window_steps: int,
    depth_sum: float,
    depth_count: float,
    continue_sum: float,
    entropy_sum: float,
) -> str:
    if depth_count <= 0:
        return (
            f"step:{step}/{total_steps} art_probe "
            f"window_steps:{window_steps} avg_cycles:none extra_cycle_frac:none "
            f"avg_stop:none router_entropy:none"
        )
    avg_depth = depth_sum / depth_count
    avg_continue = continue_sum / depth_count
    avg_stop = 1.0 - avg_continue
    avg_entropy = entropy_sum / depth_count
    return (
        f"step:{step}/{total_steps} art_probe "
        f"window_steps:{window_steps} avg_cycles:{avg_depth:.3f} "
        f"extra_cycle_frac:{avg_continue:.3f} avg_stop:{avg_stop:.3f} "
        f"router_entropy:{avg_entropy:.4f}"
    )


def unwrap_model_for_stats(model):
    unwrapped = model
    if hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return getattr(unwrapped, "_orig_mod", unwrapped)


def format_eval_art_route_stats(label, stats):
    gates = stats.get("gates") or {}
    gate_text = ",".join(f"{name}={value:.3f}" for name, value in gates.items())
    if not gate_text:
        gate_text = "none"
    return (
        f"{label}_art_routes avg_cycles:{stats['avg_cycles']:.3f} "
        f"extra_cycle_frac:{stats['extra_cycle_frac']:.3f} "
        f"art_entropy:{stats['entropy']:.4f} art_continue:{gate_text}"
    )


def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_TOKENS must provide at least one sequence per rank; got "
            f"VAL_BATCH_TOKENS={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, "
            f"GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    route_depth_sum = torch.zeros((), device=device, dtype=torch.float64)
    route_continue_sum = torch.zeros((), device=device, dtype=torch.float64)
    route_entropy_sum = torch.zeros((), device=device, dtype=torch.float64)
    route_count = torch.zeros((), device=device, dtype=torch.float64)
    route_gate_sum = None
    route_gate_names = None
    stats_model = unwrap_model_for_stats(model)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            if getattr(h, "art_eval_route_stats", True):
                art_avg_cycles = getattr(stats_model, "_last_art_avg_cycles_stat", None)
                art_continue = getattr(
                    stats_model, "_last_art_continue_rate_stat", None
                )
                art_entropy = getattr(
                    stats_model, "_last_art_router_entropy_stat", None
                )
                if (
                    art_avg_cycles is not None
                    and art_continue is not None
                    and art_entropy is not None
                ):
                    batch_seq_count = float(x.size(0))
                    route_depth_sum += (
                        art_avg_cycles.detach().to(device=device, dtype=torch.float64)
                        * batch_seq_count
                    )
                    route_continue_sum += (
                        art_continue.detach().to(device=device, dtype=torch.float64)
                        * batch_seq_count
                    )
                    route_entropy_sum += (
                        art_entropy.detach().to(device=device, dtype=torch.float64)
                        * batch_seq_count
                    )
                    route_count += batch_seq_count
                    gate_stats = getattr(
                        stats_model, "_last_art_gate_continue_stats", None
                    )
                    if gate_stats is not None:
                        gate_stats = gate_stats.detach().to(
                            device=device, dtype=torch.float64
                        )
                        if route_gate_sum is None:
                            route_gate_sum = torch.zeros_like(gate_stats)
                            route_gate_names = getattr(
                                stats_model, "art_gate_names", None
                            )
                        route_gate_sum += gate_stats * batch_seq_count
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                val_data.has_leading_space_lut[tgt_ids]
                & ~val_data.is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(route_depth_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(route_continue_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(route_entropy_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(route_count, op=dist.ReduceOp.SUM)
        if route_gate_sum is not None:
            dist.all_reduce(route_gate_sum, op=dist.ReduceOp.SUM)
    h._last_eval_art_route_stats = None
    if route_count.item() > 0:
        gates = {}
        if route_gate_sum is not None:
            gate_values = (route_gate_sum / route_count.clamp_min(1.0)).detach().cpu()
            names = route_gate_names or tuple(
                f"gate{i + 1}" for i in range(int(gate_values.numel()))
            )
            gates = {
                name: float(value)
                for name, value in zip(names, gate_values.tolist(), strict=False)
            }
        h._last_eval_art_route_stats = {
            "avg_cycles": float((route_depth_sum / route_count).item()),
            "extra_cycle_frac": float((route_continue_sum / route_count).item()),
            "entropy": float((route_entropy_sum / route_count).item()),
            "gates": gates,
        }
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = compile_target(
        base_model.forward_logits, allow_graph_breaks=h.art_halt_enabled
    )
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [
        ws
        for ws in range(0, total_tokens, h.eval_stride)
        if ws + context_size < total_tokens
    ]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi : bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws : we + 1].to(
                    dtype=torch.int64, device=device
                )
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (
                    val_data.has_leading_space_lut[tgt]
                    & ~val_data.is_boundary_token_lut[prev]
                ).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [
        ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens
    ]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    total_windows = len(window_starts)
    log(
        f"ttt:start chunks={num_chunks} windows={total_windows} "
        f"ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}"
    )
    compiled_logits = compile_target(
        base_model.forward_logits, allow_graph_breaks=h.art_halt_enabled
    )
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    _all_ttt_candidates = [p for p in base_model.parameters()]
    ttt_params = [p for p in _all_ttt_candidates if p.numel() >= 10000]
    for p in _all_ttt_candidates:
        p.requires_grad_(p.numel() >= 10000)
    optimizer = torch.optim.SGD(
        ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum, nesterov=True
    )
    ttt_start_time = time.perf_counter()
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_should_log = h.ttt_log_every_chunks > 0 and (
            ci == 0 or (ci + 1) % h.ttt_log_every_chunks == 0 or ci == num_chunks - 1
        )
        chunk_t0 = time.perf_counter()
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        if chunk_should_log:
            log(
                f"ttt:chunk {ci + 1}/{num_chunks} score_start "
                f"windows={len(windows)} local_windows={len(my_windows)}"
            )
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi : bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws : we + 1].to(
                        dtype=torch.int64, device=device
                    )
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (
                        val_data.has_leading_space_lut[tgt]
                        & ~val_data.is_boundary_token_lut[prev]
                    ).to(torch.float64)
                    byte_count += tb.sum()
        if chunk_should_log:
            log(
                f"ttt:chunk {ci + 1}/{num_chunks} score_done "
                f"elapsed:{time.perf_counter() - chunk_t0:.1f}s"
            )
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = (
                    h.ttt_lr
                    * 0.5
                    * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                if chunk_should_log:
                    log(
                        f"ttt:chunk {ci + 1}/{num_chunks} adapt_start "
                        f"seqs={chunk_seqs} local_seqs={my_chunk_seqs} lr={cos_lr:.6g}"
                    )
                for _ep in range(h.ttt_epochs):
                    epoch_t0 = time.perf_counter()
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(
                            device=device, dtype=torch.int64
                        )
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _lf = base_model.forward_logits(x).reshape(-1, 8192).float()
                            loss = (
                                F.cross_entropy(_lf, y.reshape(-1))
                                + 1e-5 * torch.logsumexp(_lf, dim=-1).pow(2).mean()
                            )
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()
                    if chunk_should_log:
                        log(
                            f"ttt:chunk {ci + 1}/{num_chunks} "
                            f"epoch:{_ep + 1}/{h.ttt_epochs} "
                            f"elapsed:{time.perf_counter() - epoch_t0:.1f}s"
                        )
        if chunk_should_log:
            log(
                f"ttt:chunk {ci + 1}/{num_chunks} done "
                f"total_elapsed:{time.perf_counter() - ttt_start_time:.1f}s"
            )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label, fn, *args, **kwargs):
    h = args[0] if args else kwargs.get("h")
    if h is not None:
        h._last_eval_art_route_stats = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(
        f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms"
    )
    stats = getattr(h, "_last_eval_art_route_stats", None) if h is not None else None
    if stats is not None:
        log(format_eval_art_route_stats(label, stats))
    return val_loss, val_bpb


def parse_art_eval_mode(mode_spec: str):
    token = mode_spec.strip().lower()
    if token in ("sample", "sampled", "stochastic"):
        return "sample", None, "sample"
    if token in ("argmax", "greedy", "deterministic"):
        return "argmax", None, "argmax"
    if token.startswith("force"):
        depth_text = token[len("force") :].lstrip("_:= ")
        if not depth_text:
            raise ValueError(f"Missing force depth in ART diag mode {mode_spec!r}")
        depth = int(depth_text)
        if depth < 0 or depth > 3:
            raise ValueError(f"ART force depth must be in [0, 3], got {depth}")
        return "force", depth, f"force{depth}"
    for prefix in ("threshold", "thresh", "thr"):
        if token.startswith(prefix):
            value_text = token[len(prefix) :].lstrip("_:= ")
            if not value_text:
                raise ValueError(
                    f"Missing threshold value in ART diag mode {mode_spec!r}"
                )
            threshold = float(value_text)
            return "threshold", threshold, f"threshold{threshold:g}".replace(".", "p")
    raise ValueError(f"Unknown ART diag mode {mode_spec!r}")


def set_art_eval_mode(model, mode_spec: str):
    base_model = unwrap_model_for_stats(model)
    old_state = (
        base_model.art_eval_route_mode,
        base_model.art_eval_route_threshold,
        base_model.art_eval_force_depth,
        base_model.art_eval_sample_routes,
    )
    mode, value, label = parse_art_eval_mode(mode_spec)
    base_model.art_eval_route_mode = mode
    if mode == "threshold":
        base_model.art_eval_route_threshold = float(value)
        base_model.art_eval_force_depth = -1
        base_model.art_eval_sample_routes = False
    elif mode == "force":
        base_model.art_eval_force_depth = int(value)
        base_model.art_eval_sample_routes = False
    else:
        base_model.art_eval_force_depth = -1
        base_model.art_eval_sample_routes = mode == "sample"
    return label, old_state


def restore_art_eval_mode(model, old_state):
    base_model = unwrap_model_for_stats(model)
    (
        base_model.art_eval_route_mode,
        base_model.art_eval_route_threshold,
        base_model.art_eval_force_depth,
        base_model.art_eval_sample_routes,
    ) = old_state


def seed_art_diag_eval(h, label):
    seed = int(h.seed) + 100_000 + sum(ord(ch) for ch in label)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_art_quant_diagnostics(stage, h, device, val_data, model):
    if not (h.art_halt_enabled and h.art_quant_diag_enabled):
        return
    modes = [m.strip() for m in h.art_quant_diag_modes.split(",") if m.strip()]
    if not modes:
        return
    log(f"{stage}_quant_diag:start modes={','.join(modes)}")
    results = getattr(h, "_art_quant_diag_results", {})
    for mode_spec in modes:
        label, old_state = set_art_eval_mode(model, mode_spec)
        try:
            seed_art_diag_eval(h, label)
            _, val_bpb = timed_eval(
                f"{stage}_diag_{label}", eval_val, h, device, val_data, model
            )
        finally:
            restore_art_eval_mode(model, old_state)
        if stage == "prequant":
            results[label] = val_bpb
        elif label in results:
            pre_bpb = results[label]
            log(
                f"quant_diag_delta mode:{label} "
                f"pre_bpb:{pre_bpb:.8f} post_bpb:{val_bpb:.8f} "
                f"delta_bpb:{val_bpb - pre_bpb:.8f}"
            )
    h._art_quant_diag_results = results
    log(f"{stage}_quant_diag:done")


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = compile_target(base_model, allow_graph_breaks=False)
    if _SMOKE_TEST:
        log("smoke_test: torch.compile disabled (eager mode)")
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    log(
        "router_params:"
        f"{sum(p.numel() for p in base_model.art_router_first.parameters()) + sum(p.numel() for p in base_model.art_router_early.parameters()) + sum(p.numel() for p in base_model.art_router.parameters())}"
    )
    if h.art_halt_enabled:
        log(
            "triple_art_halt:training starts on trunk-only 0,1,2,6,7,8,9,10; "
            "fixed triple recurrence activates at ENABLE_LOOPING_AT; "
            "routed path uses packed-active token-level ART after ART_HALT_ENABLE_AT; "
            f"route_group_size:{h.art_route_group_size} "
            f"cycle_penalty:{h.art_cycle_penalty:g} "
            f"shard_coherent_batches:{int(h.art_shard_coherent_batches)}"
        )
        log(
            "triple_art_ropefix_quant:"
            f"gptq_calibrate_routed:{int(h.art_gptq_calibrate_routed)} "
            f"router_fp32:{int(h.art_quant_router_fp32)} "
            f"eval_route_stats:{int(h.art_eval_route_stats)} "
            f"eval_sample_routes:{int(h.art_eval_sample_routes)} "
            f"routed_compile:{int(h.art_routed_compile_enabled)} "
            "reload_diag:raw_checkpoint_state_and_eval "
            f"ema_reset_on_looping:{int(h.ema_reset_on_looping)} "
            f"ema_reset_on_art:{int(h.ema_reset_on_art)} "
            f"entropy_start:{h.art_router_entropy_start:g} "
            f"entropy_end:{h.art_router_entropy_end:g}"
        )
    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    probe_window_size = max(h.recurrence_probe_every, 1)
    recent_depth_sums = collections.deque(maxlen=probe_window_size)
    recent_depth_counts = collections.deque(maxlen=probe_window_size)
    recent_continue_sums = collections.deque(maxlen=probe_window_size)
    recent_entropy_sums = collections.deque(maxlen=probe_window_size)
    max_wallclock_ms = (
        1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    )

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def muon_momentum_frac(elapsed_ms):
        if h.muon_momentum_warmup_fraction <= 0 or max_wallclock_ms is None:
            return 1.0
        warmup_ms = max_wallclock_ms * h.muon_momentum_warmup_fraction
        return min(elapsed_ms / max(warmup_ms, 1e-9), 1.0)

    def art_router_entropy_value(frac):
        clamped = min(max(frac, 0.0), 1.0)
        return (
            1.0 - clamped
        ) * h.art_router_entropy_start + clamped * h.art_router_entropy_end

    def step_fn(step, lr_scale, elapsed_ms):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        step_depth_sum = torch.zeros((), device=device, dtype=torch.float64)
        step_continue_sum = torch.zeros((), device=device, dtype=torch.float64)
        step_entropy_sum = torch.zeros((), device=device, dtype=torch.float64)
        step_gate_continue_sum = None
        step_art_count = 0.0
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            if (
                base_model.art_halt_runtime_active
                and base_model._last_art_avg_cycles_stat is not None
                and base_model._last_art_continue_rate_stat is not None
                and base_model._last_art_router_entropy_stat is not None
            ):
                step_depth_sum = (
                    step_depth_sum
                    + base_model._last_art_avg_cycles_stat.detach().to(
                        dtype=torch.float64
                    )
                )
                step_continue_sum = (
                    step_continue_sum
                    + base_model._last_art_continue_rate_stat.detach().to(
                        dtype=torch.float64
                    )
                )
                step_entropy_sum = (
                    step_entropy_sum
                    + base_model._last_art_router_entropy_stat.detach().to(
                        dtype=torch.float64
                    )
                )
                gate_stats = getattr(base_model, "_last_art_gate_continue_stats", None)
                if gate_stats is not None:
                    gate_stats = gate_stats.detach().to(
                        device=device, dtype=torch.float64
                    )
                    if step_gate_continue_sum is None:
                        step_gate_continue_sum = torch.zeros_like(gate_stats)
                    step_gate_continue_sum = step_gate_continue_sum + gate_stats
                step_art_count += 1.0
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac_mm = muon_momentum_frac(elapsed_ms)
        muon_momentum = (
            1 - frac_mm
        ) * h.muon_momentum_warmup_start + frac_mm * h.muon_momentum
        mm_blend = max(lr_scale, 0.25)
        muon_momentum = (
            mm_blend * muon_momentum + (1.0 - mm_blend) * h.muon_momentum_warmup_start
        )
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return (
            train_loss,
            step_depth_sum,
            step_continue_sum,
            step_entropy_sum,
            step_gate_continue_sum,
            step_art_count,
        )

    def sum_recent_stats(values):
        total = torch.zeros((), device=device, dtype=torch.float64)
        for value in values:
            if isinstance(value, torch.Tensor):
                total = total + value.detach().to(device=device, dtype=torch.float64)
            else:
                total = total + torch.tensor(
                    float(value), device=device, dtype=torch.float64
                )
        return total

    if h.warmup_steps > 0:
        initial_model_state = {
            name: t.detach().cpu().clone()
            for name, t in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0, 0.0)
            if (
                warmup_step <= 5
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == h.warmup_steps
            ):
                log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0, 0.0)
                if (
                    warmup_step <= 5
                    or (warmup_step + 1) % 10 == 0
                    or warmup_step + 1 == h.warmup_steps
                ):
                    log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    if h.art_halt_enabled and h.art_routed_compile_enabled:
        log("art_halt:warming routed compiled blocks")
        warm_batch_size = max(
            1,
            h.train_batch_tokens
            // (h.world_size * h.grad_accum_steps * h.train_seq_len),
        )
        base_model.warm_routed_compiled_blocks(
            warm_batch_size, h.train_seq_len, device, torch.bfloat16
        )
        optimizers.zero_grad_all()
        log("art_halt:warmed routed compiled blocks")
    elif h.art_halt_enabled:
        log("art_halt:routed compiled blocks disabled for distributed stability")

    ema_state = {
        name: t.detach().float().clone() for name, t in base_model.state_dict().items()
    }
    ema_decay = h.ema_decay

    def reset_ema_state(reason):
        nonlocal ema_state
        ema_state = {
            name: t.detach().float().clone()
            for name, t in base_model.state_dict().items()
        }
        log(f"ema:reset reason:{reason}")

    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == h.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )
        should_validate = last_step or (
            h.val_loss_every > 0 and step % h.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            if _SMOKE_TEST and last_step:
                break
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(
                f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms "
                    f"step: {step}/{h.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        enable_loop_now = (
            h.num_loops > 0
            and not base_model.looping_active
            and frac >= h.enable_looping_at
        )
        if h.distributed and h.num_loops > 0 and not base_model.looping_active:
            flag = torch.tensor(int(enable_loop_now), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            enable_loop_now = bool(flag.item())
        if enable_loop_now:
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} "
                f"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            if h.ema_reset_on_looping:
                reset_ema_state("looping_enabled")
        if h.art_halt_enabled:
            base_model.art_router_entropy_bonus = art_router_entropy_value(frac)
        enable_art_now = (
            h.art_halt_enabled
            and base_model.looping_active
            and not base_model.art_halt_runtime_active
            and frac >= h.art_halt_enable_at
        )
        if (
            h.distributed
            and h.art_halt_enabled
            and base_model.looping_active
            and not base_model.art_halt_runtime_active
        ):
            flag = torch.tensor(int(enable_art_now), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            enable_art_now = bool(flag.item())
        if enable_art_now:
            base_model.art_halt_runtime_active = True
            if compiled_model is not base_model:
                del model
                if h.art_routed_compile_enabled:
                    base_model.prepare_routed_compiled_blocks()
                    base_model.use_routed_compiled_blocks = True
                else:
                    base_model.use_routed_compiled_blocks = False
                compiled_model = base_model
                if h.distributed:
                    model = DDP(
                        base_model, device_ids=[h.local_rank], broadcast_buffers=False
                    )
                else:
                    model = base_model
                model.train()
            log(
                f"art_halt:enabled step:{step} frac:{frac:.3f} "
                f"entropy_bonus:{base_model.art_router_entropy_bonus:.4f}"
            )
            log(
                "art_halt:training switched to eager routers "
                f"routed_compile:{int(h.art_routed_compile_enabled)}"
            )
            if h.ema_reset_on_art:
                reset_ema_state("art_halt_enabled")
        (
            train_loss,
            local_step_depth_sum,
            local_step_continue_sum,
            local_step_entropy_sum,
            local_step_gate_continue_sum,
            local_step_art_count,
        ) = step_fn(step, scale, elapsed_ms)
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(
                    t.detach().float(), alpha=1.0 - ema_decay
                )
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        if local_step_art_count > 0:
            recent_depth_sums.append(local_step_depth_sum)
            recent_depth_counts.append(local_step_art_count)
            recent_continue_sums.append(local_step_continue_sum)
            recent_entropy_sums.append(local_step_entropy_sum)
        should_probe_recurrence = (
            h.art_halt_enabled
            and base_model.art_halt_runtime_active
            and h.recurrence_probe_every > 0
            and step % h.recurrence_probe_every == 0
        )
        should_log_train = h.train_log_every > 0 and (
            step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        step_art_avg_depth = 0.0
        step_art_avg_continue = 0.0
        step_art_avg_entropy = 0.0
        step_art_gate_continue = None
        if (should_probe_recurrence or should_log_train) and local_step_art_count > 0:
            step_stats = torch.stack(
                (
                    local_step_depth_sum.detach().to(
                        device=device, dtype=torch.float64
                    ),
                    local_step_continue_sum.detach().to(
                        device=device, dtype=torch.float64
                    ),
                    local_step_entropy_sum.detach().to(
                        device=device, dtype=torch.float64
                    ),
                    torch.tensor(
                        local_step_art_count, device=device, dtype=torch.float64
                    ),
                )
            )
            if h.distributed:
                dist.all_reduce(step_stats, op=dist.ReduceOp.SUM)
            step_art_avg_depth = float(
                (step_stats[0] / step_stats[3].clamp_min(1.0)).item()
            )
            step_art_avg_continue = float(
                (step_stats[1] / step_stats[3].clamp_min(1.0)).item()
            )
            step_art_avg_entropy = float(
                (step_stats[2] / step_stats[3].clamp_min(1.0)).item()
            )
            if local_step_gate_continue_sum is not None:
                gate_stats = local_step_gate_continue_sum.detach().to(
                    device=device, dtype=torch.float64
                )
                if h.distributed:
                    dist.all_reduce(gate_stats, op=dist.ReduceOp.SUM)
                step_art_gate_continue = (
                    (gate_stats / step_stats[3].clamp_min(1.0)).detach().cpu()
                )
        if should_probe_recurrence:
            window_stats = torch.stack(
                (
                    sum_recent_stats(recent_depth_sums),
                    sum_recent_stats(recent_depth_counts),
                    sum_recent_stats(recent_continue_sums),
                    sum_recent_stats(recent_entropy_sums),
                )
            )
            if h.distributed:
                dist.all_reduce(window_stats, op=dist.ReduceOp.SUM)
        else:
            window_stats = None
        if should_log_train:
            step_avg_ms = approx_training_time_ms / max(step, 1)
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            art_suffix = ""
            if h.art_halt_enabled and base_model.art_halt_runtime_active:
                base_model.sync_art_logging_stats()
                gate_suffix = ""
                if step_art_gate_continue is not None:
                    gate_suffix = " art_continue:" + ",".join(
                        f"{name}={float(value):.3f}"
                        for name, value in zip(
                            base_model.art_gate_names,
                            step_art_gate_continue.tolist(),
                            strict=False,
                        )
                    )
                art_suffix = (
                    f" art_cycles:{step_art_avg_depth:.3f}"
                    f" extra_cycle_frac:{step_art_avg_continue:.3f}"
                    f"{gate_suffix}"
                    f" art_entropy:{step_art_avg_entropy:.4f}"
                    f" art_router_loss:{base_model.last_art_router_loss:.4f}"
                )
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} "
                f"train_time: {approx_training_time_ms / 60000:.1f}m "
                f"step_avg: {step_avg_ms:.2f}ms tok/s: {tok_per_sec:.0f}"
                f"{art_suffix}"
            )
        if (
            should_probe_recurrence
            and (not h.distributed or h.rank == 0)
            and window_stats is not None
        ):
            log(
                format_art_probe_summary(
                    step,
                    h.iterations,
                    len(recent_depth_sums),
                    float(window_stats[0].item()),
                    float(window_stats[1].item()),
                    float(window_stats[2].item()),
                    float(window_stats[3].item()),
                )
            )
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {
        name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(
        f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}"
    )
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")
    base_model, compiled_model = train_model(h, device, val_data)
    if _SMOKE_TEST:
        log("smoke_test: training complete — running GPTQ+brotli pack for size check")
        serialize(h, base_model)
        if h.is_main_process:
            import sys as _sys
            from pathlib import Path as _Path

            _proj = _Path(__file__).resolve().parent
            if str(_proj) not in _sys.path:
                _sys.path.insert(0, str(_proj))
            from pack_submission import pack_code as _pack_code

            code_b = len(_pack_code(_Path(__file__).read_text(encoding="utf-8")))
            model_b = os.path.getsize(h.quantized_model_path)
            total = code_b + model_b
            log(f"smoke_pack_bytes: code={code_b} model={model_b} total={total}")
        log(
            "smoke_test:complete (code ran successfully; val_bpb not computed in smoke mode)"
        )
        return
    torch._dynamo.reset()
    timed_eval(
        "pre-quantization post-ema", eval_val, h, device, val_data, compiled_model
    )
    serialize(h, base_model)
    if h.distributed:
        dist.barrier()
    if h.raw_roundtrip_check_enabled:
        raw_eval_model = deserialize_raw_checkpoint(h, device)
        configure_art_eval_runtime(raw_eval_model, h)
        compare_model_states("raw_checkpoint_roundtrip", base_model, raw_eval_model)
        timed_eval(
            "raw_checkpoint_roundtrip",
            eval_val,
            h,
            device,
            val_data,
            raw_eval_model,
        )
        del raw_eval_model
        torch.cuda.empty_cache()
    eval_model = deserialize(h, device)
    configure_art_eval_runtime(eval_model, h)
    compiled_model = compile_target(eval_model, allow_graph_breaks=h.art_halt_enabled)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        timed_eval(
            "quantized_sliding_window",
            eval_val_sliding,
            h,
            device,
            val_data,
            eval_model,
        )
    if h.ttt_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        configure_art_eval_runtime(ttt_model, h)
        log("post_quant_ttt:enabled using RoPE reload fix and routed ART eval runtime")
        timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, ttt_model)
        del ttt_model
    else:
        log("post_quant_ttt:disabled via TTT_ENABLED=0")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        if h.requested_vocab_size != h.vocab_size:
            log(
                f"sp8192_override: requested_vocab_size={h.requested_vocab_size} "
                f"forcing_vocab_size={h.vocab_size}",
                console=True,
            )
        log(
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            ).stdout,
            console=False,
        )
        log("=" * 100, console=False)
    if _SMOKE_TEST:
        log(f"[SMOKE_TEST] attention_backend=sdpa_fallback  FA3=False  smoke_test=True")
        log(f"[SMOKE_TEST] val_bpb from this run is NOT comparable to proxy/full runs")
    log(
        f"attention_backend:{'flash_attn_3' if _USE_FA3 else 'sdpa_fallback(smoke)'} smoke_test:{_SMOKE_TEST}"
    )
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
