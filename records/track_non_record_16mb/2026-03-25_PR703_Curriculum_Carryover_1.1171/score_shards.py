#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HEADER_WORDS = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1


def load_data_shard(path: Path, vocab_size: int) -> torch.Tensor:
    header = np.fromfile(path, dtype="<i4", count=HEADER_WORDS)
    if header.size != HEADER_WORDS or int(header[0]) != HEADER_MAGIC or int(header[1]) != HEADER_VERSION:
        raise ValueError(f"unexpected shard header: {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(
        path,
        dtype="<u2",
        count=num_tokens,
        offset=HEADER_WORDS * np.dtype("<i4").itemsize,
    )
    if tokens.size != num_tokens:
        raise ValueError(f"short read for {path}")
    return torch.from_numpy(np.clip(tokens.astype(np.int64, copy=False), 0, vocab_size - 1))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, num_layers: int, num_heads: int, mlp_mult: float):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=int(model_dim * mlp_mult),
                    batch_first=True,
                    norm_first=True,
                    dropout=0.0,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        hidden = self.tok_emb(x)
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask, is_causal=True)
        return self.head(self.norm(hidden))


def score_shard(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    seq_len: int,
    max_batches: int,
    batch_size: int,
) -> float:
    model.eval()
    num_sequences = len(tokens) // (seq_len + 1)
    if num_sequences == 0:
        return float("inf")
    step = max(1, num_sequences // max(max_batches * batch_size, 1))
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_index in range(0, min(num_sequences, max_batches * batch_size), batch_size):
            starts = [
                ((batch_index + offset) * step) * (seq_len + 1)
                for offset in range(batch_size)
                if (batch_index + offset) * step < num_sequences
            ]
            if not starts:
                break
            x = torch.stack([tokens[start : start + seq_len].to(device) for start in starts])
            y = torch.stack([tokens[start + 1 : start + seq_len + 1].to(device) for start in starts])
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(y.numel())
    return total_loss / max(total_tokens, 1)


def train_steps(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    *,
    steps: int,
    seq_len: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_sequences = len(tokens) // (seq_len + 1)
    for step in range(steps):
        start_index = (step * batch_size) % max(num_sequences, 1)
        starts = [
            (start_index + offset) * (seq_len + 1)
            for offset in range(batch_size)
            if start_index + offset < num_sequences
        ]
        if not starts:
            continue
        x = torch.stack([tokens[start : start + seq_len].to(device) for start in starts])
        y = torch.stack([tokens[start + 1 : start + seq_len + 1].to(device) for start in starts])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"train_step:{step + 1}/{steps} loss:{loss.item():.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank training shards by remaining loss after a short warmup.")
    parser.add_argument("--data-dir", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-mult", type=float, default=3.0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        default="shard_order.json",
        help="JSON file to write in the current directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    train_files = sorted(glob.glob(str(Path(args.data_dir) / "fineweb_train_*.bin")))
    if not train_files:
        raise FileNotFoundError(f"no training shards found under {args.data_dir}")

    model = MiniGPT(args.vocab_size, args.model_dim, args.layers, args.heads, args.mlp_mult).to(device)
    print(f"train_shards:{len(train_files)} model_params:{sum(p.numel() for p in model.parameters()):,}")

    random_scores: dict[int, float] = {}
    for idx, shard_path in enumerate(train_files):
        tokens = load_data_shard(Path(shard_path), args.vocab_size)
        loss = score_shard(model, tokens, device, args.seq_len, args.max_batches, args.batch_size)
        random_scores[idx] = loss
        print(f"random_score:{idx} loss:{loss:.6f}")

    first_shard = load_data_shard(Path(train_files[0]), args.vocab_size)
    train_steps(
        model,
        first_shard,
        device,
        steps=args.train_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trained_scores: dict[int, float] = {}
    for idx, shard_path in enumerate(train_files):
        tokens = load_data_shard(Path(shard_path), args.vocab_size)
        loss = score_shard(model, tokens, device, args.seq_len, args.max_batches, args.batch_size)
        trained_scores[idx] = loss
        print(f"trained_score:{idx} loss:{loss:.6f}")

    ranking = [
        {
            "shard_index": idx,
            "random_loss": random_scores[idx],
            "trained_loss": trained_scores[idx],
            "learned_delta": random_scores[idx] - trained_scores[idx],
        }
        for idx in range(len(train_files))
    ]
    ranking.sort(key=lambda item: item["trained_loss"], reverse=True)
    order = [item["shard_index"] for item in ranking]
    summary = {
        "order": order,
        "train_steps": args.train_steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "remaining_loss_min": ranking[-1]["trained_loss"],
        "remaining_loss_max": ranking[0]["trained_loss"],
        "remaining_loss_std": float(np.std([item["trained_loss"] for item in ranking])),
    }
    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps({"summary": summary, "ranking": ranking}, indent=2), encoding="utf-8")
    print(f"recommended_order:{','.join(str(idx) for idx in order)}")
    print(f"wrote:{output_path}")


if __name__ == "__main__":
    main()
