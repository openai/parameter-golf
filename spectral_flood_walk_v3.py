#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_flood_walk_v0 import maybe_reset_cuda_peak_memory, maybe_sync_cuda
from spectral_flood_walk_v2a import batch_from_starts, build_lm_starts


def parse_int_csv(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return ()
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def maybe_autocast(device: torch.device) -> Any:
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def maybe_cache_tokens_on_device(tokens: torch.Tensor, *, device: torch.device, enabled: bool) -> torch.Tensor:
    if enabled and device.type == "cuda" and tokens.device != device:
        return tokens.to(device=device, dtype=torch.long)
    return tokens


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class V3Config:
    enwik8_path: str
    output_json: str | None = None
    device: str = "auto"
    seed: int = 1337

    vocab_size: int = 256
    seq_len: int = 128
    stride: int = 64
    batch_size: int = 4
    train_steps: int = 16
    eval_batches: int = 8
    report_every: int = 4
    cache_dataset_on_device: bool = True

    recurrent_dim: int = 64
    recurrent_heads: int = 4
    num_distinct_blocks: int = 1
    view_count: int = 2
    view_combination: str = "average"
    cross_token_mode: str = "floor"
    block_has_residual: bool = True
    block_nonlinearity: str = "gelu"
    recurrence_step_size: float = 1.0
    train_recurrence_steps: int = 16
    eval_recurrence_steps: int = 32
    train_floor_interval: int = 8
    floor_min_interval: int = 4
    floor_max_interval: int = 16
    floor_threshold: float = 0.05
    kernel_feature_map: str = "elu_plus_1"
    accumulator_decay: float = 0.999
    quantization: str = "ternary"

    base_lr: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def activation_fn(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "swish":
        return F.silu(x)
    raise ValueError(f"unsupported activation: {name}")


class DeepFloorRecurrentBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        step_size: float,
        has_residual: bool,
        nonlinearity: str,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = nn.Linear(dim * 2, dim, bias=False)
        self.step_size = step_size
        self.has_residual = has_residual
        self.nonlinearity = nonlinearity

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normed = self.norm(state)
        gate = torch.sigmoid(self.gate_proj(normed))
        candidate = self.down_proj(activation_fn(self.nonlinearity, self.up_proj(normed)))
        update = gate * candidate
        if self.has_residual:
            return state + self.step_size * update
        return self.step_size * update


class DeepFloorAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("recurrent_dim must be divisible by recurrent_heads")
        self.norm = RMSNorm()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.out_norm = RMSNorm()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normed = self.norm(state)
        mixed, _ = self.attn(normed, normed, normed, need_weights=False)
        return self.out_norm(state + mixed)


class DeepFloorFusedMixer(nn.Module):
    def __init__(self, dim: int, feature_map: str, decay: float) -> None:
        super().__init__()
        self.norm = RMSNorm()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.feature_map = feature_map
        self.decay = decay

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu_plus_1":
            return F.elu(x) + 1.0
        if self.feature_map == "identity":
            return x
        raise ValueError(f"unsupported kernel feature map: {self.feature_map}")

    def forward(self, state: torch.Tensor, accumulator: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(state)
        q = self._phi(self.q_proj(normed))
        k = self._phi(self.k_proj(normed))
        v = self.v_proj(normed)
        step_accumulator = torch.matmul(k.transpose(1, 2), v) / max(state.size(1), 1)
        if accumulator is None:
            accumulator = step_accumulator
        else:
            accumulator = self.decay * accumulator + step_accumulator
        mixed = torch.matmul(q, accumulator) / math.sqrt(state.size(-1))
        return state + self.out_proj(mixed), accumulator


class DeepFloorModel(nn.Module):
    def __init__(self, cfg: V3Config) -> None:
        super().__init__()
        if cfg.view_count <= 0:
            raise ValueError("view_count must be positive")
        if cfg.num_distinct_blocks <= 0:
            raise ValueError("num_distinct_blocks must be positive")
        self.cfg = cfg
        self.view_embeddings = nn.ModuleList(
            nn.Embedding(cfg.vocab_size, cfg.recurrent_dim) for _ in range(cfg.view_count)
        )
        self.blocks = nn.ModuleList(
            DeepFloorRecurrentBlock(
                cfg.recurrent_dim,
                step_size=cfg.recurrence_step_size,
                has_residual=cfg.block_has_residual,
                nonlinearity=cfg.block_nonlinearity,
            )
            for _ in range(cfg.num_distinct_blocks)
        )
        self.floor_attention = DeepFloorAttentionBlock(cfg.recurrent_dim, cfg.recurrent_heads)
        self.fused_mixer = DeepFloorFusedMixer(cfg.recurrent_dim, cfg.kernel_feature_map, cfg.accumulator_decay)
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(cfg.recurrent_dim, cfg.vocab_size, bias=False)
        if cfg.view_combination == "weighted":
            self.view_weights = nn.Parameter(torch.zeros((cfg.view_count,), dtype=torch.float32))
        else:
            self.register_parameter("view_weights", None)

    def _should_fire_floor(
        self,
        *,
        step_idx: int,
        last_floor_step: int,
        state: torch.Tensor,
        previous_state: torch.Tensor,
        adaptive: bool,
    ) -> bool:
        interval = step_idx + 1 - last_floor_step
        if interval < self.cfg.floor_min_interval:
            return False
        if interval >= self.cfg.floor_max_interval:
            return True
        if not adaptive:
            return (step_idx + 1) % self.cfg.train_floor_interval == 0
        delta = torch.linalg.vector_norm((state - previous_state).reshape(-1, state.size(-1)), dim=-1).mean()
        baseline = torch.linalg.vector_norm(previous_state.reshape(-1, previous_state.size(-1)), dim=-1).mean()
        relative_delta = float((delta / baseline.clamp_min(1e-6)).detach().cpu())
        return relative_delta >= self.cfg.floor_threshold

    def _combine_views(self, view_states: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(view_states, dim=0)
        if self.view_weights is None:
            return stacked.mean(dim=0)
        weights = F.softmax(self.view_weights, dim=0).view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        adaptive_floor: bool,
        recurrence_steps: int,
        return_metadata: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        view_states: list[torch.Tensor] = []
        metadata: dict[str, Any] = {
            "floor_steps": [],
            "floor_counts": [],
        }
        for view_idx, embedding in enumerate(self.view_embeddings):
            state = embedding(inputs)
            previous_state = state
            accumulator: torch.Tensor | None = None
            floor_steps: list[int] = []
            last_floor_step = 0
            for step_idx in range(recurrence_steps):
                block = self.blocks[step_idx % len(self.blocks)]
                state = block(state)
                if self.cfg.cross_token_mode == "floor":
                    if self._should_fire_floor(
                        step_idx=step_idx,
                        last_floor_step=last_floor_step,
                        state=state,
                        previous_state=previous_state,
                        adaptive=adaptive_floor,
                    ):
                        state = self.floor_attention(state)
                        last_floor_step = step_idx + 1
                        floor_steps.append(last_floor_step)
                elif self.cfg.cross_token_mode == "fused":
                    state, accumulator = self.fused_mixer(state, accumulator)
                else:
                    raise ValueError(f"unsupported cross_token_mode: {self.cfg.cross_token_mode}")
                previous_state = state
            view_states.append(state)
            metadata["floor_steps"].append(floor_steps)
            metadata["floor_counts"].append(len(floor_steps))
        combined = self._combine_views(view_states)
        logits = self.lm_head(self.final_norm(combined))
        if return_metadata:
            return logits, metadata
        return logits

    def estimate_artifact_bytes(self) -> int:
        bits_per_value = {
            "ternary": 2,
            "int4": 4,
            "int6": 6,
            "fp16": 16,
        }.get(self.cfg.quantization)
        if bits_per_value is None:
            raise ValueError(f"unsupported quantization: {self.cfg.quantization}")
        params = sum(parameter.numel() for parameter in self.parameters())
        return math.ceil(params * bits_per_value / 8)


def prepare_byte_enwik8_splits(
    enwik8_path: Path,
    *,
    device: torch.device,
    cache_on_device: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw = np.frombuffer(enwik8_path.read_bytes(), dtype=np.uint8)
    if raw.size < 1024:
        raise ValueError("enwik8 fixture is too small")
    train_end = max(int(raw.size * 0.95), 1)
    val_end = max(train_end + int(raw.size * 0.025), train_end + 1)
    val_end = min(val_end, raw.size - 1)
    train_tokens = torch.tensor(raw[:train_end].astype(np.int64), dtype=torch.long)
    val_tokens = torch.tensor(raw[train_end:val_end].astype(np.int64), dtype=torch.long)
    test_tokens = torch.tensor(raw[val_end:].astype(np.int64), dtype=torch.long)
    train_tokens = maybe_cache_tokens_on_device(train_tokens, device=device, enabled=cache_on_device)
    val_tokens = maybe_cache_tokens_on_device(val_tokens, device=device, enabled=cache_on_device)
    test_tokens = maybe_cache_tokens_on_device(test_tokens, device=device, enabled=cache_on_device)
    return train_tokens, val_tokens, test_tokens, {
        "raw_bytes": int(raw.size),
        "tokenization_mode": "bytes",
        "tokenizer_name": "bytes",
        "tokenizer_vocab_size": 256,
        "residency": "cuda" if cache_on_device and device.type == "cuda" else "cpu",
    }


def choose_eval_starts(starts: list[int], *, batch_size: int, eval_batches: int, seed: int) -> list[int]:
    if not starts:
        return []
    rng = random.Random(seed)
    needed = min(len(starts), batch_size * eval_batches)
    if needed >= len(starts):
        return starts[:needed]
    return rng.sample(starts, needed)


def evaluate_model(
    model: DeepFloorModel,
    tokens: torch.Tensor,
    *,
    starts: list[int],
    cfg: V3Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    total_tokens = 0
    with torch.no_grad():
        with maybe_autocast(device):
            for offset in range(0, len(starts), cfg.batch_size):
                batch_starts = starts[offset : offset + cfg.batch_size]
                inputs, targets = batch_from_starts(tokens, batch_starts, cfg.seq_len, device)
                logits, _ = model(
                    inputs,
                    adaptive_floor=True,
                    recurrence_steps=cfg.eval_recurrence_steps,
                    return_metadata=True,
                )
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                losses.append(float(loss.detach().cpu()))
                total_tokens += int(targets.numel())
    mean_loss = float(sum(losses) / max(len(losses), 1))
    return {
        "loss": mean_loss,
        "bpb": mean_loss / math.log(2.0),
        "tokens": float(total_tokens),
    }


def train_and_evaluate(cfg: V3Config) -> dict[str, Any]:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    enwik8_path = Path(cfg.enwik8_path)
    train_tokens, val_tokens, test_tokens, dataset_meta = prepare_byte_enwik8_splits(
        enwik8_path,
        device=device,
        cache_on_device=cfg.cache_dataset_on_device,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
    if not train_starts or not val_starts or not test_starts:
        raise ValueError("dataset split is too small for the requested seq_len/stride")

    model = DeepFloorModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, Any]] = []
    maybe_reset_cuda_peak_memory(device)
    train_rng = random.Random(cfg.seed + 17)
    train_start_time = time.perf_counter()

    for step in range(cfg.train_steps):
        batch_starts = train_rng.sample(train_starts, min(cfg.batch_size, len(train_starts)))
        inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device):
            logits, metadata = model(
                inputs,
                adaptive_floor=False,
                recurrence_steps=cfg.train_recurrence_steps,
                return_metadata=True,
            )
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm))
        optimizer.step()
        record = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "bpb": float(loss.detach().cpu()) / math.log(2.0),
            "grad_norm": grad_norm,
            "floor_counts": metadata["floor_counts"],
        }
        history.append(record)
        if cfg.report_every > 0 and ((step + 1) % cfg.report_every == 0 or step == cfg.train_steps - 1):
            maybe_sync_cuda(device)

    val_eval = evaluate_model(
        model,
        val_tokens,
        starts=choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=cfg.eval_batches, seed=cfg.seed + 101),
        cfg=cfg,
        device=device,
    )
    test_eval = evaluate_model(
        model,
        test_tokens,
        starts=choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=cfg.eval_batches, seed=cfg.seed + 202),
        cfg=cfg,
        device=device,
    )
    elapsed = time.perf_counter() - train_start_time
    result = {
        "config": asdict(cfg),
        "dataset": {
            "enwik8_path": str(enwik8_path),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
        "artifact": {
            "estimated_bytes": int(model.estimate_artifact_bytes()),
            "estimated_mb": float(model.estimate_artifact_bytes() / (1024.0 * 1024.0)),
            "quantization": cfg.quantization,
        },
        "train": {
            "steps": int(cfg.train_steps),
            "elapsed_seconds": float(elapsed),
            "history": history,
            "final_loss": float(history[-1]["loss"]) if history else float("nan"),
        },
        "val": val_eval,
        "test": test_eval,
    }
    if cfg.output_json is not None:
        path = Path(cfg.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepFloor recurrent multi-view language model")
    parser.add_argument("--enwik8-path", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--device", default=V3Config.device)
    parser.add_argument("--seed", type=int, default=V3Config.seed)
    parser.add_argument("--seq-len", type=int, default=V3Config.seq_len)
    parser.add_argument("--stride", type=int, default=V3Config.stride)
    parser.add_argument("--batch-size", type=int, default=V3Config.batch_size)
    parser.add_argument("--train-steps", type=int, default=V3Config.train_steps)
    parser.add_argument("--eval-batches", type=int, default=V3Config.eval_batches)
    parser.add_argument("--report-every", type=int, default=V3Config.report_every)
    parser.add_argument("--recurrent-dim", type=int, default=V3Config.recurrent_dim)
    parser.add_argument("--recurrent-heads", type=int, default=V3Config.recurrent_heads)
    parser.add_argument("--num-distinct-blocks", type=int, default=V3Config.num_distinct_blocks)
    parser.add_argument("--view-count", type=int, default=V3Config.view_count)
    parser.add_argument("--view-combination", choices=("average", "weighted"), default=V3Config.view_combination)
    parser.add_argument("--cross-token-mode", choices=("floor", "fused"), default=V3Config.cross_token_mode)
    parser.add_argument("--block-has-residual", action=argparse.BooleanOptionalAction, default=V3Config.block_has_residual)
    parser.add_argument("--block-nonlinearity", choices=("relu", "gelu", "swish"), default=V3Config.block_nonlinearity)
    parser.add_argument("--recurrence-step-size", type=float, default=V3Config.recurrence_step_size)
    parser.add_argument("--train-recurrence-steps", type=int, default=V3Config.train_recurrence_steps)
    parser.add_argument("--eval-recurrence-steps", type=int, default=V3Config.eval_recurrence_steps)
    parser.add_argument("--train-floor-interval", type=int, default=V3Config.train_floor_interval)
    parser.add_argument("--floor-min-interval", type=int, default=V3Config.floor_min_interval)
    parser.add_argument("--floor-max-interval", type=int, default=V3Config.floor_max_interval)
    parser.add_argument("--floor-threshold", type=float, default=V3Config.floor_threshold)
    parser.add_argument("--kernel-feature-map", choices=("elu_plus_1", "identity"), default=V3Config.kernel_feature_map)
    parser.add_argument("--accumulator-decay", type=float, default=V3Config.accumulator_decay)
    parser.add_argument("--quantization", choices=("ternary", "int4", "int6", "fp16"), default=V3Config.quantization)
    parser.add_argument("--base-lr", type=float, default=V3Config.base_lr)
    parser.add_argument("--weight-decay", type=float, default=V3Config.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=V3Config.grad_clip_norm)
    parser.add_argument(
        "--cache-dataset-on-device",
        action=argparse.BooleanOptionalAction,
        default=V3Config.cache_dataset_on_device,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> V3Config:
    return V3Config(
        enwik8_path=args.enwik8_path,
        output_json=args.output_json,
        device=args.device,
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        report_every=args.report_every,
        recurrent_dim=args.recurrent_dim,
        recurrent_heads=args.recurrent_heads,
        num_distinct_blocks=args.num_distinct_blocks,
        view_count=args.view_count,
        view_combination=args.view_combination,
        cross_token_mode=args.cross_token_mode,
        block_has_residual=args.block_has_residual,
        block_nonlinearity=args.block_nonlinearity,
        recurrence_step_size=args.recurrence_step_size,
        train_recurrence_steps=args.train_recurrence_steps,
        eval_recurrence_steps=args.eval_recurrence_steps,
        train_floor_interval=args.train_floor_interval,
        floor_min_interval=args.floor_min_interval,
        floor_max_interval=args.floor_max_interval,
        floor_threshold=args.floor_threshold,
        kernel_feature_map=args.kernel_feature_map,
        accumulator_decay=args.accumulator_decay,
        quantization=args.quantization,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        cache_dataset_on_device=args.cache_dataset_on_device,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = train_and_evaluate(config_from_args(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
