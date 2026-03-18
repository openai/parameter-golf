#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUM = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"

PATTERNS = {
    "seed": re.compile(rf"^seed:(?P<seed>\d+)$"),
    "train_loader": re.compile(r"^train_loader:dataset:(?P<dataset>\S+) train_shards:(?P<train_shards>\d+)$"),
    "world_size": re.compile(r"^world_size:(?P<world_size>\d+) grad_accum_steps:(?P<grad_accum_steps>\d+)$"),
    "attention": re.compile(r"^attention_mode:gqa num_heads:(?P<num_heads>\d+) num_kv_heads:(?P<num_kv_heads>\d+)$"),
    "optimizer": re.compile(
        rf"^tie_embeddings:(?P<tie_embeddings>\S+) embed_lr:(?P<embed_lr>{NUM}) "
        rf"head_lr:(?P<head_lr>{NUM}) matrix_lr:(?P<matrix_lr>{NUM}) scalar_lr:(?P<scalar_lr>{NUM})$"
    ),
    "schedule": re.compile(
        rf"^train_batch_tokens:(?P<train_batch_tokens>\d+) train_seq_len:(?P<train_seq_len>\d+) "
        rf"iterations:(?P<iterations>\d+) warmup_steps:(?P<warmup_steps>\d+) "
        rf"max_wallclock_seconds:(?P<max_wallclock_seconds>{NUM})$"
    ),
    "train_step": re.compile(
        rf"^step:(?P<step>\d+)/(?P<iterations>\d+) train_loss:(?P<train_loss>{NUM}) .* step_avg:(?P<step_avg>{NUM})ms$"
    ),
    "stop": re.compile(r"^stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<iterations>\d+)$"),
    "peak_mem": re.compile(r"^peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB$"),
    "total_int8": re.compile(r"^Total submission size int8\+zlib: (?P<bytes>\d+) bytes$"),
    "final_exact": re.compile(
        rf"^final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>{NUM}) val_bpb:(?P<val_bpb>{NUM})$"
    ),
}

ENV_KEY_MAP = {
    "RUN_ID": ("run_id", str),
    "GIT_SHA": ("git_sha", str),
    "GIT_BRANCH": ("git_branch", str),
    "LAUNCH_ID": ("launch_id", str),
    "DATA_PATH": ("data_path", str),
    "TOKENIZER_PATH": ("tokenizer_path", str),
    "VOCAB_SIZE": ("vocab_size", int),
    "NUM_LAYERS": ("num_layers", int),
    "MODEL_DIM": ("model_dim", int),
    "NUM_HEADS": ("num_heads", int),
    "NUM_KV_HEADS": ("num_kv_heads", int),
    "MLP_MULT": ("mlp_mult", int),
    "TIE_EMBEDDINGS": ("tie_embeddings", lambda value: bool(int(value))),
    "TRAIN_SEQ_LEN": ("train_seq_len", int),
    "TRAIN_BATCH_TOKENS": ("train_batch_tokens", int),
    "VAL_BATCH_SIZE": ("val_batch_size", int),
    "ITERATIONS": ("iterations", int),
    "WARMUP_STEPS": ("warmup_steps", int),
    "WARMDOWN_ITERS": ("warmdown_iters", int),
    "TRAIN_LOG_EVERY": ("train_log_every", int),
    "VAL_LOSS_EVERY": ("val_loss_every", int),
    "MAX_WALLCLOCK_SECONDS": ("max_wallclock_seconds", float),
    "SEED": ("seed", int),
    "TIED_EMBED_LR": ("tied_embed_lr", float),
    "MATRIX_LR": ("matrix_lr", float),
    "SCALAR_LR": ("scalar_lr", float),
    "MUON_MOMENTUM": ("muon_momentum", float),
    "MUON_BACKEND_STEPS": ("muon_backend_steps", int),
    "MUON_MOMENTUM_WARMUP_START": ("muon_momentum_warmup_start", float),
    "MUON_MOMENTUM_WARMUP_STEPS": ("muon_momentum_warmup_steps", int),
    "BETA1": ("beta1", float),
    "BETA2": ("beta2", float),
    "ADAM_EPS": ("adam_eps", float),
    "GRAD_CLIP_NORM": ("grad_clip_norm", float),
    "QK_GAIN_INIT": ("qk_gain_init", float),
    "LOGIT_SOFTCAP": ("logit_softcap", float),
    "ROPE_BASE": ("rope_base", float),
    "WORLD_SIZE": ("world_size", int),
}


def parse_env_file(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        target = ENV_KEY_MAP.get(key)
        if target is None:
            continue
        field_name, caster = target
        try:
            result[field_name] = caster(value)
        except ValueError:
            result[field_name] = value
    return result


def parse_log(path: Path) -> dict[str, object]:
    env_path = path.parent.parent / "env.txt"
    result: dict[str, object] = {
        "log_path": str(path),
        "run_dir": str(path.parent.parent),
        "run_id": path.stem,
    }
    result.update(parse_env_file(env_path))
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for key, pattern in PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            groups = match.groupdict()
            if key == "seed":
                result["seed"] = int(groups["seed"])
            elif key == "train_loader":
                result["dataset"] = groups["dataset"]
                result["train_shards"] = int(groups["train_shards"])
            elif key == "world_size":
                result["world_size"] = int(groups["world_size"])
                result["grad_accum_steps"] = int(groups["grad_accum_steps"])
            elif key == "attention":
                result["num_heads"] = int(groups["num_heads"])
                result["num_kv_heads"] = int(groups["num_kv_heads"])
            elif key == "optimizer":
                result["tie_embeddings"] = groups["tie_embeddings"] == "True"
                result["tied_embed_lr"] = float(groups["embed_lr"])
                result["head_lr"] = float(groups["head_lr"])
                result["matrix_lr"] = float(groups["matrix_lr"])
                result["scalar_lr"] = float(groups["scalar_lr"])
            elif key == "schedule":
                result["train_batch_tokens"] = int(groups["train_batch_tokens"])
                result["train_seq_len"] = int(groups["train_seq_len"])
                result["iterations"] = int(groups["iterations"])
                result["warmup_steps"] = int(groups["warmup_steps"])
                result["max_wallclock_seconds"] = float(groups["max_wallclock_seconds"])
            elif key == "train_step":
                result["last_logged_step"] = int(groups["step"])
                result["last_train_loss"] = float(groups["train_loss"])
                result["step_avg_ms"] = float(groups["step_avg"])
            elif key == "stop":
                result["step_stop"] = int(groups["step"])
                result["train_time_ms"] = int(groups["train_time_ms"])
            elif key == "peak_mem":
                result["peak_mem_allocated_mib"] = int(groups["allocated"])
                result["peak_mem_reserved_mib"] = int(groups["reserved"])
            elif key == "total_int8":
                result["bytes_total_int8_zlib"] = int(groups["bytes"])
            elif key == "final_exact":
                result["final_int8_zlib_roundtrip_exact_val_loss"] = float(groups["val_loss"])
                result["final_int8_zlib_roundtrip_exact_val_bpb"] = float(groups["val_bpb"])
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: extract_run_metrics.py <log.txt> [<log.txt> ...]", file=sys.stderr)
        return 2

    for arg in argv[1:]:
        path = Path(arg)
        if path.is_dir():
            candidates = sorted(path.glob("logs/*.txt"))
            for candidate in candidates:
                print(json.dumps(parse_log(candidate), sort_keys=True))
            continue
        print(json.dumps(parse_log(path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
