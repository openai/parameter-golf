#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from spectral_flood_walk_v1_sizes import estimate_v1a_sizes as estimate_v1a_size_values, parse_int_csv
from spectral_flood_walk_v0 import (
    build_sentencepiece_luts,
    cleanup_distributed_if_needed,
    distributed_env,
    export_quantized_model_npz,
    get_cuda_memory_stats,
    init_distributed_if_needed,
    is_rank0,
    load_data_shard_prefix,
    load_vocab_size,
    maybe_all_reduce_mean,
    maybe_reset_cuda_peak_memory,
    maybe_sync_cuda,
    resolve_device,
    set_seed,
    setup_auto_logging,
    summarize_history,
)


def build_lm_starts(num_tokens: int, seq_len: int, stride: int) -> list[int]:
    stop = num_tokens - seq_len - 1
    if stop <= 0:
        return []
    return list(range(0, stop, stride))


def batch_from_starts(
    tokens: torch.Tensor,
    starts: list[int],
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = [tokens[start : start + seq_len + 1] for start in starts]
    batch = torch.stack(windows).to(device)
    return batch[:, :-1], batch[:, 1:]


@dataclass
class V1Config:
    data_path: str = "data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    output_json: str | None = None
    auto_log_path: str | None = None
    model_artifact_path: str | None = None
    device: str = "auto"
    train_tokens: int = 65_536
    val_tokens: int = 16_384
    seq_len: int = 128
    stride: int = 64
    batch_size: int = 8
    train_steps: int = 48
    eval_batches: int = 64
    report_every: int = 4
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_mult: int = 4
    pos_buckets: int = 256
    semantic_layers: str = "2,4"
    use_semantic_memory: bool = True
    pk_num_subkeys: int = 64
    pk_key_dim: int = 16
    pk_topk_sub: int = 4
    pk_topk_final: int = 8
    pk_code_dim: int = 64
    lr: float = 2e-3
    min_lr_scale: float = 0.10
    warmup_steps: int = 4
    cooldown_start_frac: float = 0.75
    weight_decay: float = 1e-2
    seed: int = 1337

    @property
    def semantic_layer_ids(self) -> tuple[int, ...]:
        return parse_int_csv(self.semantic_layers)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int) -> None:
        super().__init__()
        hidden = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProductKeyMemory(nn.Module):
    def __init__(
        self,
        dim: int,
        num_subkeys: int,
        key_dim: int,
        code_dim: int,
        topk_sub: int,
        topk_final: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_subkeys = num_subkeys
        self.key_dim = key_dim
        self.topk_sub = topk_sub
        self.topk_final = topk_final
        self.query = nn.Linear(dim, 2 * key_dim, bias=False)
        self.key_a = nn.Parameter(torch.randn(num_subkeys, key_dim) * 0.02)
        self.key_b = nn.Parameter(torch.randn(num_subkeys, key_dim) * 0.02)
        self.codes = nn.Parameter(torch.randn(num_subkeys * num_subkeys, code_dim) * 0.02)
        self.expand = nn.Sequential(
            nn.Linear(code_dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def compact_bytes(self) -> int:
        return int(
            self.key_a.numel()
            + self.key_b.numel()
            + self.codes.numel()
            + sum(param.numel() for param in self.query.parameters())
            + sum(param.numel() for param in self.expand.parameters())
        )

    def expanded_value_bytes(self, dtype_bytes: int = 2) -> int:
        return int(self.codes.size(0) * self.dim * dtype_bytes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq, _ = x.shape
        q = self.query(x).view(batch, seq, 2, self.key_dim)
        q_a = q[:, :, 0]
        q_b = q[:, :, 1]

        scores_a = torch.einsum("btd,kd->btk", q_a, self.key_a)
        scores_b = torch.einsum("btd,kd->btk", q_b, self.key_b)
        top_a_scores, top_a_idx = scores_a.topk(min(self.topk_sub, self.num_subkeys), dim=-1)
        top_b_scores, top_b_idx = scores_b.topk(min(self.topk_sub, self.num_subkeys), dim=-1)

        cand_scores = top_a_scores.unsqueeze(-1) + top_b_scores.unsqueeze(-2)
        cand_indices = top_a_idx.unsqueeze(-1) * self.num_subkeys + top_b_idx.unsqueeze(-2)

        cand_scores = cand_scores.reshape(batch, seq, -1)
        cand_indices = cand_indices.reshape(batch, seq, -1)
        final_k = min(self.topk_final, cand_scores.size(-1))
        final_scores, final_pos = cand_scores.topk(final_k, dim=-1)
        final_indices = cand_indices.gather(-1, final_pos)

        codes = self.codes[final_indices]
        weights = torch.softmax(final_scores / math.sqrt(2.0 * self.key_dim), dim=-1)
        mixed_codes = (weights.unsqueeze(-1) * codes).sum(dim=-2)
        out = self.expand(mixed_codes)
        stats = {
            "selected_score_mean": final_scores.mean().detach(),
            "selected_score_max": final_scores.max().detach(),
        }
        return out, stats


class TransformerBlock(nn.Module):
    def __init__(self, cfg: V1Config, *, semantic: bool) -> None:
        super().__init__()
        self.semantic = semantic
        self.norm1 = RMSNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg.embed_dim, cfg.num_heads)
        self.norm2 = RMSNorm(cfg.embed_dim)
        if semantic and cfg.use_semantic_memory:
            self.ff = ProductKeyMemory(
                dim=cfg.embed_dim,
                num_subkeys=cfg.pk_num_subkeys,
                key_dim=cfg.pk_key_dim,
                code_dim=cfg.pk_code_dim,
                topk_sub=cfg.pk_topk_sub,
                topk_final=cfg.pk_topk_final,
            )
        else:
            self.ff = FeedForward(cfg.embed_dim, cfg.ff_mult)

    def compact_bytes(self) -> int:
        if isinstance(self.ff, ProductKeyMemory):
            ff_bytes = self.ff.compact_bytes()
        else:
            ff_bytes = sum(param.numel() for param in self.ff.parameters())
        return int(
            sum(param.numel() for param in self.norm1.parameters())
            + sum(param.numel() for param in self.attn.parameters())
            + sum(param.numel() for param in self.norm2.parameters())
            + ff_bytes
        )

    def expanded_value_bytes(self, dtype_bytes: int = 2) -> int:
        if isinstance(self.ff, ProductKeyMemory):
            return self.ff.expanded_value_bytes(dtype_bytes=dtype_bytes)
        return 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        x = x + self.attn(self.norm1(x))
        ff_in = self.norm2(x)
        if isinstance(self.ff, ProductKeyMemory):
            ff_out, stats = self.ff(ff_in)
        else:
            ff_out = self.ff(ff_in)
            stats = None
        return x + ff_out, stats


class SpectralFloodWalkV1A(nn.Module):
    def __init__(self, cfg: V1Config, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.pos_buckets, cfg.embed_dim)
        semantic_layers = set(cfg.semantic_layer_ids if cfg.use_semantic_memory else ())
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, semantic=layer_idx in semantic_layers) for layer_idx in range(cfg.num_layers)]
        )
        self.norm_f = RMSNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def compact_bytes(self) -> int:
        return int(sum(block.compact_bytes() for block in self.blocks) + self.token_embed.weight.numel() + self.pos_embed.weight.numel())

    def expanded_semantic_bytes(self, dtype_bytes: int = 2) -> int:
        return int(sum(block.expanded_value_bytes(dtype_bytes=dtype_bytes) for block in self.blocks))

    def encode_hidden(self, tokens: torch.Tensor) -> dict[str, torch.Tensor | None]:
        batch, seq = tokens.shape
        pos = torch.arange(seq, device=tokens.device).clamp(max=self.cfg.pos_buckets - 1)
        x = self.token_embed(tokens) + self.pos_embed(pos).unsqueeze(0)
        semantic_means: list[torch.Tensor] = []
        semantic_maxes: list[torch.Tensor] = []
        for block in self.blocks:
            x, stats = block(x)
            if stats is not None:
                semantic_means.append(stats["selected_score_mean"])
                semantic_maxes.append(stats["selected_score_max"])
        x = self.norm_f(x)
        semantic_score_mean = None
        semantic_score_max = None
        if semantic_means:
            semantic_score_mean = torch.stack(semantic_means).mean()
            semantic_score_max = torch.stack(semantic_maxes).max()
        return {
            "hidden": x,
            "semantic_score_mean": semantic_score_mean,
            "semantic_score_max": semantic_score_max,
        }

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        encoded = self.encode_hidden(tokens)
        hidden = encoded["hidden"]
        assert hidden is not None
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return {
            "hidden": hidden,
            "logits": logits,
            "loss": loss,
            "semantic_score_mean": encoded["semantic_score_mean"],
            "semantic_score_max": encoded["semantic_score_max"],
        }


def estimate_v1a_sizes(cfg: V1Config, vocab_size: int) -> dict[str, float]:
    return estimate_v1a_size_values(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        ff_mult=cfg.ff_mult,
        pos_buckets=cfg.pos_buckets,
        semantic_layers=cfg.semantic_layers,
        use_semantic_memory=cfg.use_semantic_memory,
        pk_num_subkeys=cfg.pk_num_subkeys,
        pk_key_dim=cfg.pk_key_dim,
        pk_code_dim=cfg.pk_code_dim,
    )
def lr_scale_for_step(step: int, cfg: V1Config) -> float:
    if cfg.train_steps <= 1:
        return 1.0
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return float(step + 1) / float(cfg.warmup_steps)
    cooldown_start = max(cfg.warmup_steps, int(cfg.train_steps * cfg.cooldown_start_frac))
    if step < cooldown_start:
        return 1.0
    if cooldown_start >= cfg.train_steps - 1:
        return 1.0
    progress = (step - cooldown_start) / max(cfg.train_steps - cooldown_start - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine


def evaluate_v1a(
    model: SpectralFloodWalkV1A,
    tokens: torch.Tensor,
    starts: list[int],
    cfg: V1Config,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    total_correct = 0
    semantic_score_means: list[float] = []
    start_time = time.perf_counter()
    eval_starts = starts[: cfg.eval_batches * cfg.batch_size]
    with torch.no_grad():
        for idx in range(0, len(eval_starts), cfg.batch_size):
            batch_starts = eval_starts[idx : idx + cfg.batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, cfg.seq_len, device)
            out = model(inputs, targets)
            logits = out["logits"]
            loss = out["loss"]
            total_loss += float(loss.item()) * targets.numel()
            total_tokens += int(targets.numel())
            total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
            prev_ids = inputs.reshape(-1)
            tgt_ids = targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            total_bytes += float(token_bytes.to(torch.float64).sum().item())
            if out["semantic_score_mean"] is not None:
                semantic_score_means.append(float(out["semantic_score_mean"].item()))
    maybe_sync_cuda(device)
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    bits_per_token = avg_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / max(total_bytes, 1.0))
    result = {
        "loss": float(avg_loss),
        "val_bpb": float(val_bpb),
        "accuracy": float(total_correct / max(total_tokens, 1)),
        "tokens": float(total_tokens),
        "bytes": float(total_bytes),
        "elapsed_s": float(elapsed),
        "tokens_per_s": float(total_tokens / max(elapsed, 1e-6)),
    }
    if semantic_score_means:
        result["semantic_score_mean"] = float(sum(semantic_score_means) / len(semantic_score_means))
    cuda_stats = get_cuda_memory_stats(device)
    if cuda_stats is not None:
        result["cuda_memory"] = cuda_stats
    model.train()
    return result


def run_v1a(cfg: V1Config) -> dict[str, object]:
    set_seed(cfg.seed)
    rank, local_rank, world_size = distributed_env()
    device = resolve_device(cfg.device, local_rank)
    ddp_enabled = init_distributed_if_needed(device)

    tokenizer_path = Path(cfg.tokenizer_path)
    vocab_size = load_vocab_size(tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        tokenizer_path, vocab_size, device
    )

    train_path = Path(cfg.data_path) / "fineweb_train_000000.bin"
    val_path = Path(cfg.data_path) / "fineweb_val_000000.bin"
    train_tokens = load_data_shard_prefix(train_path, cfg.train_tokens)
    val_tokens = load_data_shard_prefix(val_path, cfg.val_tokens)
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    if len(train_starts) < cfg.batch_size or len(val_starts) < cfg.batch_size:
        raise ValueError("not enough tokens for the requested V1a config")

    model = SpectralFloodWalkV1A(cfg, vocab_size=vocab_size).to(device)
    core_model: SpectralFloodWalkV1A = model
    if ddp_enabled:
        ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        model_for_train: SpectralFloodWalkV1A | DDP = ddp_model
        core_model = ddp_model.module  # type: ignore[assignment]
    else:
        model_for_train = model

    optimizer = torch.optim.AdamW(core_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed + rank + 17)
    history: list[dict[str, float]] = []
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    train_start = time.perf_counter()

    try:
        for step in range(cfg.train_steps):
            step_start = time.perf_counter()
            lr_scale = lr_scale_for_step(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_scale
            batch_starts = rng.sample(train_starts, cfg.batch_size)
            inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
            out = model_for_train(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
            optimizer.step()

            if is_rank0():
                step_metrics = {
                    "step": float(step),
                    "loss": maybe_all_reduce_mean(float(out["loss"].item()), device, ddp_enabled),
                    "lr_scale": float(lr_scale),
                    "semantic_score_mean": maybe_all_reduce_mean(
                        float(out["semantic_score_mean"].item()) if out["semantic_score_mean"] is not None else 0.0,
                        device,
                        ddp_enabled,
                    ),
                    "semantic_memory_mb_estimate": float(core_model.expanded_semantic_bytes() / (1024.0 * 1024.0)),
                    "step_ms": float((time.perf_counter() - step_start) * 1000.0),
                }
                cuda_stats = get_cuda_memory_stats(device)
                if cuda_stats is not None:
                    step_metrics["cuda_allocated_mb"] = float(cuda_stats["allocated_bytes"] / (1024.0 * 1024.0))
                    step_metrics["cuda_reserved_mb"] = float(cuda_stats["reserved_bytes"] / (1024.0 * 1024.0))
                    step_metrics["cuda_peak_allocated_mb"] = float(cuda_stats["max_allocated_bytes"] / (1024.0 * 1024.0))
                    step_metrics["cuda_peak_reserved_mb"] = float(cuda_stats["max_reserved_bytes"] / (1024.0 * 1024.0))
                if cfg.report_every <= 1 or step % cfg.report_every == 0 or step == cfg.train_steps - 1:
                    history.append(step_metrics)
                    cuda_fragment = ""
                    if cuda_stats is not None:
                        cuda_fragment = (
                            f" vram={step_metrics['cuda_allocated_mb']:.1f}/{step_metrics['cuda_reserved_mb']:.1f}MB"
                            f" peak={step_metrics['cuda_peak_allocated_mb']:.1f}MB"
                        )
                    semantic_fragment = ""
                    if cfg.use_semantic_memory and cfg.semantic_layer_ids:
                        semantic_fragment = (
                            f" sem_score={step_metrics['semantic_score_mean']:.3f}"
                            f" sem_mb_est={step_metrics['semantic_memory_mb_estimate']:.2f}"
                        )
                    print(
                        "[train] "
                        f"step={step}/{cfg.train_steps - 1} "
                        f"loss={step_metrics['loss']:.4f} "
                        f"lr_scale={step_metrics['lr_scale']:.3f} "
                        f"step_ms={step_metrics['step_ms']:.1f}"
                        f"{semantic_fragment}{cuda_fragment}"
                    )
            elif ddp_enabled:
                _ = maybe_all_reduce_mean(float(out["loss"].item()), device, ddp_enabled)
                semantic_value = float(out["semantic_score_mean"].item()) if out["semantic_score_mean"] is not None else 0.0
                _ = maybe_all_reduce_mean(semantic_value, device, ddp_enabled)

        maybe_sync_cuda(device)
        train_elapsed = time.perf_counter() - train_start
        result: dict[str, object] = {}
        if is_rank0():
            train_cuda_memory = get_cuda_memory_stats(device)
            model_artifact_path = Path(cfg.model_artifact_path) if cfg.model_artifact_path is not None else None
            if model_artifact_path is None and cfg.output_json is not None:
                model_artifact_path = Path(cfg.output_json).with_name("model_int8.npz")
            model_artifact_bytes = None
            if model_artifact_path is not None:
                model_artifact_bytes = export_quantized_model_npz(core_model, model_artifact_path)
            eval_metrics = evaluate_v1a(
                core_model,
                val_tokens,
                val_starts,
                cfg,
                device,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            result = {
                "config": {**asdict(cfg), "resolved_device": str(device), "world_size": world_size},
                "train_history": history,
                "train_summary": summarize_history(history),
                "train_elapsed_s": float(train_elapsed),
                "train_tokens_per_s": float((cfg.train_steps * cfg.batch_size * cfg.seq_len * world_size) / max(train_elapsed, 1e-6)),
                "size_estimate": estimate_v1a_sizes(cfg, vocab_size),
                "train_cuda_memory": train_cuda_memory,
                "model_artifact_path": str(model_artifact_path) if model_artifact_path is not None else None,
                "model_artifact_bytes": float(model_artifact_bytes) if model_artifact_bytes is not None else None,
                "eval": eval_metrics,
                "vocab_size": vocab_size,
            }
            if cfg.output_json is not None:
                output_path = Path(cfg.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2) + "\n")
        if ddp_enabled:
            torch.distributed.barrier()
        return result
    finally:
        cleanup_distributed_if_needed()


def parse_args() -> V1Config:
    parser = argparse.ArgumentParser(description="Spectral Flood Walk V1a semantic-memory runner")
    parser.add_argument("--data-path", default=V1Config.data_path)
    parser.add_argument("--tokenizer-path", default=V1Config.tokenizer_path)
    parser.add_argument("--output-json", default=V1Config.output_json)
    parser.add_argument("--auto-log-path", default=V1Config.auto_log_path)
    parser.add_argument("--model-artifact-path", default=V1Config.model_artifact_path)
    parser.add_argument("--device", default=V1Config.device)
    parser.add_argument("--train-tokens", type=int, default=V1Config.train_tokens)
    parser.add_argument("--val-tokens", type=int, default=V1Config.val_tokens)
    parser.add_argument("--seq-len", type=int, default=V1Config.seq_len)
    parser.add_argument("--stride", type=int, default=V1Config.stride)
    parser.add_argument("--batch-size", type=int, default=V1Config.batch_size)
    parser.add_argument("--train-steps", type=int, default=V1Config.train_steps)
    parser.add_argument("--eval-batches", type=int, default=V1Config.eval_batches)
    parser.add_argument("--report-every", type=int, default=V1Config.report_every)
    parser.add_argument("--embed-dim", type=int, default=V1Config.embed_dim)
    parser.add_argument("--num-layers", type=int, default=V1Config.num_layers)
    parser.add_argument("--num-heads", type=int, default=V1Config.num_heads)
    parser.add_argument("--ff-mult", type=int, default=V1Config.ff_mult)
    parser.add_argument("--pos-buckets", type=int, default=V1Config.pos_buckets)
    parser.add_argument("--semantic-layers", default=V1Config.semantic_layers)
    parser.add_argument("--use-semantic-memory", action=argparse.BooleanOptionalAction, default=V1Config.use_semantic_memory)
    parser.add_argument("--pk-num-subkeys", type=int, default=V1Config.pk_num_subkeys)
    parser.add_argument("--pk-key-dim", type=int, default=V1Config.pk_key_dim)
    parser.add_argument("--pk-topk-sub", type=int, default=V1Config.pk_topk_sub)
    parser.add_argument("--pk-topk-final", type=int, default=V1Config.pk_topk_final)
    parser.add_argument("--pk-code-dim", type=int, default=V1Config.pk_code_dim)
    parser.add_argument("--lr", type=float, default=V1Config.lr)
    parser.add_argument("--min-lr-scale", type=float, default=V1Config.min_lr_scale)
    parser.add_argument("--warmup-steps", type=int, default=V1Config.warmup_steps)
    parser.add_argument("--cooldown-start-frac", type=float, default=V1Config.cooldown_start_frac)
    parser.add_argument("--weight-decay", type=float, default=V1Config.weight_decay)
    parser.add_argument("--seed", type=int, default=V1Config.seed)
    args = parser.parse_args()
    return V1Config(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        output_json=args.output_json,
        auto_log_path=args.auto_log_path,
        model_artifact_path=args.model_artifact_path,
        device=args.device,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        report_every=args.report_every,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        pos_buckets=args.pos_buckets,
        semantic_layers=args.semantic_layers,
        use_semantic_memory=args.use_semantic_memory,
        pk_num_subkeys=args.pk_num_subkeys,
        pk_key_dim=args.pk_key_dim,
        pk_topk_sub=args.pk_topk_sub,
        pk_topk_final=args.pk_topk_final,
        pk_code_dim=args.pk_code_dim,
        lr=args.lr,
        min_lr_scale=args.min_lr_scale,
        warmup_steps=args.warmup_steps,
        cooldown_start_frac=args.cooldown_start_frac,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    setup_auto_logging(cfg.auto_log_path)
    result = run_v1a(cfg)
    if is_rank0():
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
