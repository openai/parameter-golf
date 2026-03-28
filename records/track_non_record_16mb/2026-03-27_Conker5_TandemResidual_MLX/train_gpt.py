#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from conker5.conker3 import ConkerThreeConfig
from conker5.conker4b import ConkerFourBConfig, ConkerFourBModel, scale_config
from conker5.golf_data import _load_golf_shard, build_parameter_golf_dataset
from conker5.quantize import (
    bits_per_token_from_loss,
    dequantize_packed_params,
    pack_trainable_params,
    serialize_packed_params_zlib,
)


def _e(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if cast is bool:
        return bool(int(raw))
    return cast(raw)


class Hyperparameters:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = _e("DATA_PATH", str(repo_root / "data" / "datasets" / "fineweb10B_sp1024"))
    run_id = _e("RUN_ID", "conker5_tandem_nonrecord")
    seed = _e("SEED", 43, int)
    seq_len = _e("SEQ_LEN", 256, int)
    batch_size = _e("BATCH_SIZE", 16, int)
    steps = _e("ITERATIONS", 1000, int)
    learning_rate = _e("LEARNING_RATE", 5e-4, float)
    weight_decay = _e("WEIGHT_DECAY", 1e-5, float)
    grad_clip = _e("GRAD_CLIP", 1.0, float)
    log_every = _e("TRAIN_LOG_EVERY", 100, int)
    eval_batches = _e("EVAL_BATCHES", 50, int)
    scale = _e("SCALE", 10.0, float)
    linear_modes = _e("LINEAR_MODES", 256, int)
    vocab_size = _e("VOCAB_SIZE", 1024, int)
    variant = _e("VARIANT", "window4")
    quant_bits = _e("QUANT_BITS", 6, int)
    artifact_level = _e("ARTIFACT_ZLIB_LEVEL", 9, int)
    author = _e("AUTHOR", "asuramaya")
    github_id = _e("GITHUB_ID", "asuramaya")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random_module = getattr(mx, "random", None)
    if random_module is not None and hasattr(random_module, "seed"):
        random_module.seed(seed)


def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    batch_size, timesteps, vocab_size = logits.shape
    return mx.mean(
        nn.losses.cross_entropy(
            logits.reshape(batch_size * timesteps, vocab_size),
            y.reshape(batch_size * timesteps),
        )
    )


def train_loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    custom_loss = getattr(model, "supervised_loss", None)
    if callable(custom_loss):
        return custom_loss(x, y)
    return loss_fn(model, x, y)


def evaluate_slice(model: nn.Module, dataset, args: Hyperparameters, split: str) -> float:
    total = 0.0
    compiled_loss = mx.compile(lambda x, y: loss_fn(model, x, y), inputs=model.state, outputs=model.state)
    warm_x, warm_y = dataset.batch(split, args.batch_size, args.seq_len)
    warm_loss = compiled_loss(warm_x, warm_y)
    mx.eval(warm_loss)
    for _ in range(args.eval_batches):
        x, y = dataset.batch(split, args.batch_size, args.seq_len)
        loss = compiled_loss(x, y)
        mx.eval(loss)
        total += float(loss.item())
    return total / args.eval_batches


def load_full_split_tokens(dataset, split: str, seq_len: int) -> np.ndarray:
    files = dataset.train_files if split == "train" else dataset.test_files
    tokens = np.ascontiguousarray(np.concatenate([_load_golf_shard(path) for path in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"{split} split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def evaluate_full_split(model: nn.Module, dataset, args: Hyperparameters, split: str) -> tuple[float, float | None, int]:
    tokens = load_full_split_tokens(dataset, split, args.seq_len)
    compiled_loss = mx.compile(lambda x, y: loss_fn(model, x, y), inputs=model.state, outputs=model.state)
    warm_chunk = tokens[: args.batch_size * args.seq_len + 1]
    warm_x = mx.array(warm_chunk[:-1].reshape(-1, args.seq_len), dtype=mx.int32)
    warm_y = mx.array(warm_chunk[1:].reshape(-1, args.seq_len), dtype=mx.int32)
    warm_loss = compiled_loss(warm_x, warm_y)
    mx.eval(warm_loss)

    total_loss_sum = 0.0
    total_tokens = 0
    total_seqs = (tokens.size - 1) // args.seq_len
    for seq_start in range(0, total_seqs, args.batch_size):
        seq_end = min(seq_start + args.batch_size, total_seqs)
        raw_start = seq_start * args.seq_len
        raw_end = seq_end * args.seq_len + 1
        chunk = tokens[raw_start:raw_end]
        x = mx.array(chunk[:-1].reshape(-1, args.seq_len), dtype=mx.int32)
        y = mx.array(chunk[1:].reshape(-1, args.seq_len), dtype=mx.int32)
        loss = compiled_loss(x, y)
        mx.eval(loss)
        token_count = int(y.size)
        total_loss_sum += float(loss.item()) * token_count
        total_tokens += token_count
    eval_loss = total_loss_sum / max(total_tokens, 1)
    eval_bpb = None
    if split == "test" and dataset.test_tokens_per_byte is not None:
        eval_bpb = bits_per_token_from_loss(eval_loss) * dataset.test_tokens_per_byte
    return eval_loss, eval_bpb, total_tokens


def count_code_bytes(folder: Path) -> int:
    total = 0
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".py", ".sh"} or path.name == "requirements.txt":
            total += path.stat().st_size
    return total


def build_model(args: Hyperparameters, dataset) -> ConkerFourBModel:
    base_cfg = ConkerThreeConfig(
        max_seq_len=args.seq_len,
        linear_modes=args.linear_modes,
        local_window=4,
        linear_half_life_max=16.0,
        oscillatory_frac=0.875,
        oscillatory_period_min=4.0,
        oscillatory_period_max=64.0,
        static_bank_gate=True,
    )
    cfg = scale_config(
        ConkerFourBConfig(
            base_config=base_cfg,
            freeze_base=False,
            enable_exact1=True,
            enable_exact2=True,
            enable_exact3=True,
            enable_special2=True,
            enable_number2=True,
            enable_markup2=True,
            enable_attr2=True,
            enable_delim2=True,
            enable_recency=False,
            tokenizer_vocab_path=str(Path(dataset.tokenizer_path).with_suffix(".vocab")) if dataset.tokenizer_path else None,
            dynamic_support_gates=True,
            gate_only_mode=True,
            exact1_opens_mask=False,
            delim2_opens_mask=False,
        ),
        args.scale,
    )
    return ConkerFourBModel(vocab_size=dataset.vocab_size, config=cfg)


def main() -> None:
    args = Hyperparameters()
    record_dir = Path(__file__).resolve().parent
    log_path = record_dir / "train.log"
    artifact_path = record_dir / f"{args.run_id}.int{args.quant_bits}.ptz"
    submission_path = record_dir / "submission.json"
    results_path = record_dir / "results.json"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    log_path.write_text("", encoding="utf-8")
    log(f"run_id:{args.run_id}")
    log(f"data_path:{args.data_path}")
    log(f"seed:{args.seed} seq_len:{args.seq_len} batch_size:{args.batch_size} steps:{args.steps} lr:{args.learning_rate:g}")

    dataset = build_parameter_golf_dataset(args.data_path, vocab_size=args.vocab_size)
    seed_everything(args.seed)
    model = build_model(args, dataset)
    params = sum(value.size for _, value in nn.utils.tree_flatten(model.parameters()))
    log(f"params:{params}")

    value_and_grad = nn.value_and_grad(model, train_loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    start = time.time()
    losses: list[float] = []
    best = float("inf")
    for step in range(1, args.steps + 1):
        x, y = dataset.batch("train", args.batch_size, args.seq_len)
        loss, grads = value_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, max_norm=args.grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        current = float(loss.item())
        losses.append(current)
        best = min(best, current)
        if step % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            tok_s = (step * args.batch_size * args.seq_len) / max(time.time() - start, 1e-9)
            log(f"step:{step}/{args.steps} train_loss:{recent:.4f} best:{best:.4f} tok_s:{tok_s:.0f}")
    train_time = time.time() - start

    train_eval = evaluate_slice(model, dataset, args, "train")
    test_eval = evaluate_slice(model, dataset, args, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    log(
        f"slice_eval train_loss:{train_eval:.4f} test_loss:{test_eval:.4f} "
        f"test_bpb:{test_bpb:.4f} train_time:{train_time:.1f}s"
    )

    full_pre_loss, full_pre_bpb, full_tokens = evaluate_full_split(model, dataset, args, "test")
    log(f"full_eval_pre loss:{full_pre_loss:.4f} bpb:{full_pre_bpb:.4f} tokens:{full_tokens}")

    full_state = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    packed_state, packed_stats = pack_trainable_params(full_state, trainable_names, args.quant_bits)
    artifact_blob, artifact_raw_bytes = serialize_packed_params_zlib(packed_state, level=args.artifact_level)
    artifact_path.write_bytes(artifact_blob)
    quant_state = dequantize_packed_params(packed_state)
    model.update(nn.utils.tree_unflatten(list(quant_state.items())))
    full_q_loss, full_q_bpb, full_q_tokens = evaluate_full_split(model, dataset, args, "test")
    log(
        f"full_eval_int{args.quant_bits} loss:{full_q_loss:.4f} bpb:{full_q_bpb:.4f} "
        f"tokens:{full_q_tokens} artifact_bytes:{artifact_path.stat().st_size}"
    )

    code_bytes = count_code_bytes(record_dir)
    bytes_total = artifact_path.stat().st_size + code_bytes
    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": "Conker-5 Tandem Residual Exact Experts (MLX, non-record)",
        "blurb": (
            "Non-record unlimited-compute submission. Tandem-trained Conker-3 + sparse exact residual experts "
            "(exact1/2/3, delim2, special2, number2, markup2, attr2) with gate-only learned selection, "
            f"trained {args.steps} steps on MLX and packaged as int{args.quant_bits}+zlib."
        ),
        "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "val_loss": round(float(full_q_loss), 8),
        "val_bpb": round(float(full_q_bpb), 8),
        "pre_quant_val_loss": round(float(full_pre_loss), 8),
        "pre_quant_val_bpb": round(float(full_pre_bpb), 8),
        f"int{args.quant_bits}_zlib_val_loss": round(float(full_q_loss), 8),
        f"int{args.quant_bits}_zlib_val_bpb": round(float(full_q_bpb), 8),
        "bytes_total": int(bytes_total),
        f"bytes_model_int{args.quant_bits}_zlib": int(artifact_path.stat().st_size),
        "bytes_code": int(code_bytes),
    }
    submission_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")

    results = {
        "seed": args.seed,
        "params": params,
        "train_time_sec": train_time,
        "slice_train_loss": train_eval,
        "slice_test_loss": test_eval,
        "slice_test_bpb": test_bpb,
        "full_pre_loss": full_pre_loss,
        "full_pre_bpb": full_pre_bpb,
        "full_pre_tokens": full_tokens,
        "full_quant_loss": full_q_loss,
        "full_quant_bpb": full_q_bpb,
        "full_quant_tokens": full_q_tokens,
        "packed_stats": packed_stats,
        "artifact_bytes_raw": artifact_raw_bytes,
        "artifact_bytes_zlib": artifact_path.stat().st_size,
        "bytes_code": code_bytes,
        "bytes_total": bytes_total,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote:{submission_path.name} {results_path.name} {artifact_path.name}")


if __name__ == "__main__":
    main()
