#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def load_data_shard_prefix(path: Path, max_tokens: int | None = None) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    read_tokens = num_tokens if max_tokens is None else min(num_tokens, max_tokens)
    tokens = np.fromfile(path, dtype="<u2", count=read_tokens, offset=header_bytes)
    if tokens.size != read_tokens:
        raise ValueError(f"Short read for {path}")
    return torch.from_numpy(tokens.astype(np.int64, copy=False))


def fake_quantize_unit_int8(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    x = F.normalize(x.float(), dim=-1, eps=eps)
    q = torch.clamp(torch.round(x * 127.0), -127, 127)
    dequant = q / 127.0
    x_st = x + (dequant - x).detach()
    return x_st, q.to(torch.int8)


@dataclass
class V0Config:
    data_path: str = "data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    output_json: str | None = None
    auto_log_path: str | None = None
    seed_pool_path: str | None = None
    seed_pool_load_path: str | None = None
    model_artifact_path: str | None = None
    device: str = "auto"
    train_tokens: int = 32_768
    val_tokens: int = 8_192
    prefix_len: int = 64
    chunk_size: int = 8
    stride: int = 8
    batch_size: int = 16
    train_steps: int = 24
    eval_samples: int = 128
    eval_online_append: bool = False
    eval_append_writes_per_batch: int = 0
    bank_capacity: int = 1_024
    bank_min_entries: int = 128
    bank_writes_per_step: int = 8
    warmup_steps: int = 2
    retrieval_dropout_start: float = 0.75
    retrieval_dropout_end: float = 0.15
    report_every: int = 1
    bank_store_mode: str = "codes"
    bank_runtime_dtype: str = "fp16"
    num_experts: int = 8
    embed_dim: int = 64
    expert_hidden: int = 64
    expert_rank: int = 8
    fused_dim: int = 128
    runtime_dim: int = 128
    code_dim: int = 16
    query_dim: int = 32
    reader_heads: int = 4
    topk: int = 64
    read_mass_ema: float = 0.90
    read_mass_weight: float = 0.50
    candidate_surprise_weight: float = 0.70
    candidate_novelty_weight: float = 0.20
    candidate_gate_weight: float = 0.10
    lr: float = 2e-3
    min_lr_scale: float = 0.10
    cooldown_start_frac: float = 0.75
    weight_decay: float = 1e-2
    seed: int = 1337


def resolve_bank_runtime_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported bank runtime dtype: {name}")
    return mapping[name]


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def maybe_sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def maybe_reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_memory_stats(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "allocated_bytes": float(torch.cuda.memory_allocated(index)),
        "reserved_bytes": float(torch.cuda.memory_reserved(index)),
        "max_allocated_bytes": float(torch.cuda.max_memory_allocated(index)),
        "max_reserved_bytes": float(torch.cuda.max_memory_reserved(index)),
        "free_bytes": float(free_bytes),
        "total_bytes": float(total_bytes),
    }


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def is_rank0() -> bool:
    rank, _, _ = distributed_env()
    return rank == 0


def setup_auto_logging(auto_log_path: str | None) -> None:
    if auto_log_path is None or not is_rank0():
        return
    log_path = Path(auto_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, handle)
    sys.stderr = TeeStream(sys.stderr, handle)


def resolve_device(device_spec: str, local_rank: int) -> torch.device:
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_spec)


def init_distributed_if_needed(device: torch.device) -> bool:
    _, _, world_size = distributed_env()
    if world_size <= 1:
        return False
    if dist.is_initialized():
        return True
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return True


def cleanup_distributed_if_needed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def maybe_all_reduce_mean(value: float, device: torch.device, enabled: bool) -> float:
    if not enabled:
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


class AffineExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.w_a = nn.Linear(input_dim, hidden_dim)
        self.w_g = nn.Linear(input_dim, rank)
        self.w_u = nn.Linear(input_dim, hidden_dim)
        self.u = nn.Parameter(torch.randn(hidden_dim, rank) * 0.02)
        self.v = nn.Parameter(torch.randn(hidden_dim, rank) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        state = torch.zeros(batch, self.w_a.out_features, device=x.device, dtype=x.dtype)
        for i in range(seq):
            x_t = x[:, i]
            a_t = torch.sigmoid(self.w_a(x_t))
            g_t = self.w_g(x_t)
            u_t = self.w_u(x_t)
            low_rank = (state @ self.v) * g_t
            state = a_t * state + u_t + low_rank @ self.u.T
        return state


class SpectralFloodWalkV0(nn.Module):
    def __init__(self, cfg: V0Config, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, cfg.embed_dim)
        self.experts = nn.ModuleList(
            [AffineExpert(cfg.embed_dim, cfg.expert_hidden, cfg.expert_rank) for _ in range(cfg.num_experts)]
        )
        self.expert_proj = nn.ModuleList([nn.Linear(cfg.expert_hidden, cfg.fused_dim) for _ in range(cfg.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.num_experts),
        )
        self.query_proj = nn.Linear(cfg.fused_dim, cfg.query_dim)
        self.target_encoder = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.runtime_dim),
            nn.SiLU(),
            nn.Linear(cfg.runtime_dim, cfg.runtime_dim),
        )
        self.key_proj = nn.Linear(cfg.runtime_dim, cfg.query_dim)
        self.code_proj = nn.Linear(cfg.runtime_dim, cfg.code_dim)
        self.expansion_basis = nn.Linear(cfg.code_dim, cfg.runtime_dim, bias=False)
        self.write_code_proj = nn.Linear(cfg.fused_dim, cfg.code_dim)
        self.reader_q = nn.Linear(cfg.fused_dim, cfg.fused_dim)
        self.reader_k = nn.Linear(cfg.runtime_dim + 16, cfg.fused_dim)
        self.reader_v = nn.Linear(cfg.runtime_dim + 16, cfg.fused_dim)
        self.reader = nn.MultiheadAttention(cfg.fused_dim, cfg.reader_heads, batch_first=True)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(cfg.fused_dim * 2, cfg.fused_dim),
            nn.SiLU(),
            nn.Linear(cfg.fused_dim, cfg.fused_dim),
        )
        self.meta_expert = nn.Embedding(cfg.num_experts, 8)
        self.meta_pos = nn.Embedding(512, 8)
        self.chunk_pos = nn.Embedding(cfg.chunk_size, cfg.fused_dim)
        self.chunk_in = nn.Linear(cfg.embed_dim + cfg.fused_dim, cfg.fused_dim)
        self.chunk_mlp = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.fused_dim),
            nn.SiLU(),
            nn.Linear(cfg.fused_dim, cfg.fused_dim),
        )
        self.lm_head = nn.Linear(cfg.fused_dim, vocab_size)

    @staticmethod
    def normalize_runtime_state(state: torch.Tensor) -> torch.Tensor:
        return F.normalize(state, dim=-1, eps=1e-6)

    def encode_prefix(self, prefix_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_emb = self.token_embed(prefix_tokens)
        summary = prefix_emb.mean(dim=1)
        gate_probs = torch.softmax(self.gate(summary), dim=-1)
        expert_states = []
        for expert, proj in zip(self.experts, self.expert_proj):
            state = proj(expert(prefix_emb))
            expert_states.append(state)
        stacked = torch.stack(expert_states, dim=1)
        fused = torch.sum(stacked * gate_probs.unsqueeze(-1), dim=1)
        return fused, gate_probs

    def encode_target(self, chunk_tokens: torch.Tensor, next_chunk_tokens: torch.Tensor) -> torch.Tensor:
        chunk_emb = self.token_embed(chunk_tokens).mean(dim=1)
        next_emb = self.token_embed(next_chunk_tokens).mean(dim=1)
        target = self.target_encoder(torch.cat([chunk_emb, next_emb], dim=-1))
        return self.normalize_runtime_state(target)

    def query_from_fused(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return fake_quantize_unit_int8(self.query_proj(fused))

    def key_from_target(self, target_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return fake_quantize_unit_int8(self.key_proj(target_state))

    def code_from_target(self, target_state: torch.Tensor) -> torch.Tensor:
        return self.code_proj(target_state)

    def expand_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.normalize_runtime_state(self.expansion_basis(codes))

    def write_from_ctx(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        write_codes = self.write_code_proj(ctx)
        write_state = self.expand_codes(write_codes)
        write_k_st, write_k_int8 = self.key_from_target(write_state)
        return write_codes, write_state, write_k_st, write_k_int8

    def read_context(
        self,
        fused: torch.Tensor,
        retrieved_states: torch.Tensor | None,
        retrieved_expert_ids: torch.Tensor | None,
        retrieved_pos: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if retrieved_states is None or retrieved_states.numel() == 0:
            zero_read = torch.zeros_like(fused)
            ctx = self.ctx_mlp(torch.cat([fused, zero_read], dim=-1))
            return ctx, None
        meta = torch.cat(
            [
                self.meta_expert(retrieved_expert_ids),
                self.meta_pos(retrieved_pos.clamp(max=self.meta_pos.num_embeddings - 1)),
            ],
            dim=-1,
        )
        kv_in = torch.cat([retrieved_states, meta], dim=-1)
        q = self.reader_q(fused).unsqueeze(1)
        k = self.reader_k(kv_in)
        v = self.reader_v(kv_in)
        read, weights = self.reader(q, k, v, need_weights=True, average_attn_weights=True)
        read = read.squeeze(1)
        ctx = self.ctx_mlp(torch.cat([fused, read], dim=-1))
        return ctx, weights.squeeze(1)

    def decode_chunk(self, ctx: torch.Tensor, chunk_inputs: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(chunk_inputs)
        pos_ids = torch.arange(self.cfg.chunk_size, device=tok.device).unsqueeze(0)
        pos = self.chunk_pos(pos_ids).expand(tok.size(0), -1, -1)
        ctx_rep = ctx.unsqueeze(1).expand(tok.size(0), self.cfg.chunk_size, ctx.size(-1))
        hidden = self.chunk_in(torch.cat([tok, ctx_rep], dim=-1))
        hidden = hidden + pos
        hidden = self.chunk_mlp(hidden)
        return self.lm_head(hidden)

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        chunk_inputs: torch.Tensor,
        chunk_targets: torch.Tensor,
        next_chunk_tokens: torch.Tensor,
        bank: "SeedBank" | None,
        retrieval_enabled: bool,
    ) -> dict[str, torch.Tensor | None]:
        fused, gate_probs = self.encode_prefix(prefix_tokens)
        q_st, q_int8 = self.query_from_fused(fused)
        target_state = self.encode_target(chunk_targets, next_chunk_tokens)
        target_k_st, target_k_int8 = self.key_from_target(target_state)

        retrieved_indices = None
        if retrieval_enabled and bank is not None and bank.size > 0:
            retrieved_states, retrieved_expert_ids, retrieved_pos, retrieved_indices = bank.retrieve(q_st, self.cfg.topk)
        else:
            retrieved_states = retrieved_expert_ids = retrieved_pos = None

        ctx, attn = self.read_context(fused, retrieved_states, retrieved_expert_ids, retrieved_pos)
        write_codes, write_state, write_k_st, write_k_int8 = self.write_from_ctx(ctx)
        logits = self.decode_chunk(ctx, chunk_inputs)
        token_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            chunk_targets.reshape(-1),
            reduction="none",
        ).reshape(chunk_targets.size(0), chunk_targets.size(1))
        loss_nll = token_loss.mean()

        sim = q_st @ target_k_st.T / math.sqrt(self.cfg.query_dim)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss_ret = F.cross_entropy(sim, labels)
        loss_recon = F.mse_loss(write_state, target_state.detach())
        gate_mean = gate_probs.mean(dim=0)
        loss_gate = ((gate_mean - gate_mean.mean()) ** 2).mean()
        loss = loss_nll + 0.10 * loss_ret + 0.05 * loss_recon + 0.01 * loss_gate

        return {
            "loss": loss,
            "loss_nll": loss_nll,
            "loss_ret": loss_ret,
            "loss_recon": loss_recon,
            "loss_gate": loss_gate,
            "example_nll": token_loss.mean(dim=-1).detach(),
            "logits": logits,
            "q_st": q_st.detach(),
            "q_int8": q_int8.detach(),
            "target_k_st": target_k_st.detach(),
            "target_k_int8": target_k_int8.detach(),
            "write_k_st": write_k_st.detach(),
            "write_k_int8": write_k_int8.detach(),
            "write_codes": write_codes.detach(),
            "write_state": write_state.detach(),
            "gate_probs": gate_probs.detach(),
            "read_attn": attn.detach() if attn is not None else None,
            "retrieved_indices": retrieved_indices.detach() if retrieved_indices is not None else None,
        }

    def forward_batch(
        self,
        prefix_tokens: torch.Tensor,
        chunk_inputs: torch.Tensor,
        chunk_targets: torch.Tensor,
        next_chunk_tokens: torch.Tensor,
        bank: "SeedBank" | None,
        retrieval_enabled: bool,
    ) -> dict[str, torch.Tensor | None]:
        return self(
            prefix_tokens=prefix_tokens,
            chunk_inputs=chunk_inputs,
            chunk_targets=chunk_targets,
            next_chunk_tokens=next_chunk_tokens,
            bank=bank,
            retrieval_enabled=retrieval_enabled,
        )


class SeedBank:
    def __init__(
        self,
        *,
        model: SpectralFloodWalkV0,
        capacity: int | None = None,
        device: torch.device | None = None,
        read_mass_weight: float = 0.5,
        store_mode: str | None = None,
        runtime_state_dtype: torch.dtype | None = None,
        keys: torch.Tensor | None = None,
        codes: torch.Tensor | None = None,
        runtime_states: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        pos_buckets: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
        read_mass: torch.Tensor | None = None,
        ages: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.device = device if device is not None else model.token_embed.weight.device
        self.capacity = capacity if capacity is not None else (int(keys.size(0)) if keys is not None else 0)
        self.read_mass_weight = float(read_mass_weight)
        self.store_mode = store_mode if store_mode is not None else model.cfg.bank_store_mode
        if self.store_mode not in {"codes", "runtime"}:
            raise ValueError(f"Unsupported bank store mode: {self.store_mode}")
        self.runtime_state_dtype = (
            runtime_state_dtype if runtime_state_dtype is not None else resolve_bank_runtime_dtype(model.cfg.bank_runtime_dtype)
        )
        qdim = model.cfg.query_dim
        cdim = model.cfg.code_dim
        rdim = model.cfg.runtime_dim
        self.keys = torch.empty((0, qdim), dtype=torch.int8, device=self.device) if keys is None else keys.to(self.device)
        initial_size = int(self.keys.size(0))
        self.codes = (
            torch.empty((0, cdim), dtype=torch.float32, device=self.device)
            if codes is None
            else codes.float().to(self.device)
        )
        self.runtime_states = (
            torch.empty((0, rdim), dtype=self.runtime_state_dtype, device=self.device)
            if runtime_states is None
            else runtime_states.to(device=self.device, dtype=self.runtime_state_dtype)
        )
        self.expert_ids = (
            torch.zeros((initial_size,), dtype=torch.long, device=self.device)
            if expert_ids is None
            else expert_ids.to(self.device)
        )
        self.pos_buckets = (
            torch.zeros((initial_size,), dtype=torch.long, device=self.device)
            if pos_buckets is None
            else pos_buckets.to(self.device)
        )
        self.scores = (
            torch.zeros((initial_size,), dtype=torch.float32, device=self.device)
            if scores is None
            else scores.float().to(self.device)
        )
        if read_mass is None:
            self.read_mass = torch.zeros((initial_size,), dtype=torch.float32, device=self.device)
        else:
            self.read_mass = read_mass.float().to(self.device)
        self.ages = (
            torch.zeros((initial_size,), dtype=torch.long, device=self.device)
            if ages is None
            else ages.to(self.device)
        )

    @property
    def size(self) -> int:
        return int(self.keys.size(0))

    def priority(self) -> torch.Tensor:
        if self.size == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        return self.scores + self.read_mass_weight * self.read_mass + self.ages.float() * 1e-6

    def summary(self) -> dict[str, float]:
        if self.size == 0:
            return {
                "size": 0.0,
                "avg_score": 0.0,
                "avg_read_mass": 0.0,
                "max_score": 0.0,
                "max_read_mass": 0.0,
                "resident_bytes": float(self.resident_bytes()),
            }
        return {
            "size": float(self.size),
            "avg_score": float(self.scores.mean().item()),
            "avg_read_mass": float(self.read_mass.mean().item()),
            "max_score": float(self.scores.max().item()),
            "max_read_mass": float(self.read_mass.max().item()),
            "resident_bytes": float(self.resident_bytes()),
        }

    def resident_bytes(self) -> int:
        total = 0
        for tensor in (
            self.keys,
            self.codes,
            self.runtime_states,
            self.expert_ids,
            self.pos_buckets,
            self.scores,
            self.read_mass,
            self.ages,
        ):
            total += tensor_bytes(tensor)
        return total

    def clone(self) -> "SeedBank":
        return SeedBank(
            model=self.model,
            capacity=self.capacity,
            device=self.device,
            read_mass_weight=self.read_mass_weight,
            store_mode=self.store_mode,
            runtime_state_dtype=self.runtime_state_dtype,
            keys=self.keys.clone(),
            codes=self.codes.clone(),
            runtime_states=self.runtime_states.clone(),
            expert_ids=self.expert_ids.clone(),
            pos_buckets=self.pos_buckets.clone(),
            scores=self.scores.clone(),
            read_mass=self.read_mass.clone(),
            ages=self.ages.clone(),
        )

    def retrieve(self, q_st: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise ValueError("bank is empty")
        k = min(topk, self.size)
        key_f = self.keys.float() / 127.0
        scores = q_st @ key_f.T
        idx = scores.topk(k, dim=-1).indices
        flat = idx.reshape(-1)
        if self.store_mode == "runtime":
            states = self.runtime_states[flat].to(dtype=self.model.token_embed.weight.dtype)
        else:
            states = self.model.expand_codes(self.codes[flat].to(dtype=self.model.token_embed.weight.dtype))
        states = states.reshape(idx.size(0), idx.size(1), -1)
        expert_ids = self.expert_ids[flat].reshape(idx.size(0), idx.size(1))
        pos = self.pos_buckets[flat].reshape(idx.size(0), idx.size(1))
        return states, expert_ids, pos, idx

    def max_similarity(self, query: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return torch.zeros(query.size(0), device=query.device, dtype=query.dtype)
        key_f = self.keys.float() / 127.0
        return (query @ key_f.T).max(dim=-1).values

    def note_reads(self, idx: torch.Tensor | None, attn: torch.Tensor | None, ema: float) -> None:
        if idx is None or attn is None or self.size == 0:
            return
        flat_idx = idx.reshape(-1)
        flat_attn = attn.reshape(-1).float()
        updates = torch.zeros(self.size, device=self.device, dtype=torch.float32)
        counts = torch.zeros(self.size, device=self.device, dtype=torch.float32)
        ones = torch.ones_like(flat_attn)
        updates.scatter_add_(0, flat_idx, flat_attn)
        counts.scatter_add_(0, flat_idx, ones)
        mask = counts > 0
        if mask.any():
            avg = updates[mask] / counts[mask]
            self.read_mass[mask] = self.read_mass[mask] * ema + avg * (1.0 - ema)

    def add_entries(
        self,
        keys: torch.Tensor,
        codes: torch.Tensor,
        runtime_states: torch.Tensor | None,
        expert_ids: torch.Tensor,
        pos_buckets: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        max_new_entries: int | None = None,
    ) -> int:
        if keys.numel() == 0 or self.capacity <= 0:
            return 0
        keys = keys.to(self.device)
        codes = codes.to(self.device)
        if runtime_states is not None:
            runtime_states = runtime_states.to(device=self.device, dtype=self.runtime_state_dtype)
        expert_ids = expert_ids.to(self.device)
        pos_buckets = pos_buckets.to(self.device)
        scores = scores.float().to(self.device)

        if max_new_entries is not None and scores.numel() > max_new_entries:
            keep = scores.topk(max_new_entries).indices
            keys = keys[keep]
            codes = codes[keep]
            if runtime_states is not None:
                runtime_states = runtime_states[keep]
            expert_ids = expert_ids[keep]
            pos_buckets = pos_buckets[keep]
            scores = scores[keep]

        new_read_mass = torch.zeros_like(scores)
        new_ages = torch.full_like(expert_ids, fill_value=step, dtype=torch.long)

        keys_all = torch.cat([self.keys, keys], dim=0)
        if self.store_mode == "runtime":
            if runtime_states is None:
                raise ValueError("runtime bank mode requires runtime_states on add_entries")
            runtime_states_all = torch.cat([self.runtime_states, runtime_states], dim=0)
            codes_all = self.codes
        else:
            runtime_states_all = self.runtime_states
            codes_all = torch.cat([self.codes, codes], dim=0)
        expert_all = torch.cat([self.expert_ids, expert_ids], dim=0)
        pos_all = torch.cat([self.pos_buckets, pos_buckets], dim=0)
        score_all = torch.cat([self.scores, scores], dim=0)
        read_all = torch.cat([self.read_mass, new_read_mass], dim=0)
        ages_all = torch.cat([self.ages, new_ages], dim=0)

        priority = score_all + self.read_mass_weight * read_all + ages_all.float() * 1e-6
        keep = priority.topk(min(self.capacity, priority.numel())).indices

        self.keys = keys_all[keep]
        if self.store_mode == "runtime":
            self.runtime_states = runtime_states_all[keep]
        else:
            self.codes = codes_all[keep]
        self.expert_ids = expert_all[keep]
        self.pos_buckets = pos_all[keep]
        self.scores = score_all[keep]
        self.read_mass = read_all[keep]
        self.ages = ages_all[keep]
        return int(keys.size(0))

    def export_npz(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "store_mode": np.array([self.store_mode]),
            "keys": self.keys.detach().cpu().numpy(),
            "expert_ids": self.expert_ids.detach().cpu().numpy().astype(np.uint8, copy=False),
            "pos_buckets": self.pos_buckets.detach().cpu().numpy().astype(np.uint16, copy=False),
            "scores": self.scores.detach().cpu().numpy().astype(np.float16, copy=False),
            "read_mass": self.read_mass.detach().cpu().numpy().astype(np.float16, copy=False),
            "ages": self.ages.detach().cpu().numpy().astype(np.int32, copy=False),
        }
        if self.store_mode == "runtime":
            payload["runtime_states"] = self.runtime_states.detach().cpu().numpy().astype(np.float16, copy=False)
        else:
            payload["codes"] = self.codes.detach().cpu().numpy().astype(np.float16, copy=False)
        np.savez_compressed(path, **payload)
        return int(path.stat().st_size)

    @classmethod
    def load_npz(
        cls,
        path: Path,
        *,
        model: SpectralFloodWalkV0,
        device: torch.device | None = None,
        read_mass_weight: float | None = None,
        capacity: int | None = None,
    ) -> "SeedBank":
        payload = np.load(path, allow_pickle=True)
        store_mode = str(payload["store_mode"][0]) if "store_mode" in payload else "codes"
        bank_device = device if device is not None else model.token_embed.weight.device
        keys = torch.from_numpy(payload["keys"]).to(device=bank_device, dtype=torch.int8)
        expert_ids = torch.from_numpy(payload["expert_ids"].astype(np.int64, copy=False)).to(bank_device)
        pos_buckets = torch.from_numpy(payload["pos_buckets"].astype(np.int64, copy=False)).to(bank_device)
        scores = torch.from_numpy(payload["scores"].astype(np.float32, copy=False)).to(bank_device)
        read_mass = torch.from_numpy(payload["read_mass"].astype(np.float32, copy=False)).to(bank_device)
        ages = torch.from_numpy(payload["ages"].astype(np.int64, copy=False)).to(bank_device)
        bank_kwargs: dict[str, object] = {}
        if "codes" in payload:
            bank_kwargs["codes"] = torch.from_numpy(payload["codes"].astype(np.float32, copy=False)).to(bank_device)
        if "runtime_states" in payload:
            bank_kwargs["runtime_states"] = torch.from_numpy(payload["runtime_states"]).to(
                device=bank_device,
                dtype=resolve_bank_runtime_dtype(model.cfg.bank_runtime_dtype),
            )
        effective_capacity = capacity if capacity is not None else int(keys.size(0))
        return cls(
            model=model,
            capacity=effective_capacity,
            device=bank_device,
            read_mass_weight=read_mass_weight if read_mass_weight is not None else model.cfg.read_mass_weight,
            store_mode=store_mode,
            runtime_state_dtype=resolve_bank_runtime_dtype(model.cfg.bank_runtime_dtype),
            keys=keys,
            expert_ids=expert_ids,
            pos_buckets=pos_buckets,
            scores=scores,
            read_mass=read_mass,
            ages=ages,
            **bank_kwargs,
        )


def build_positions(num_tokens: int, prefix_len: int, chunk_size: int, stride: int) -> list[int]:
    stop = num_tokens - 2 * chunk_size
    return list(range(prefix_len, stop, stride))


def batch_from_positions(
    tokens: torch.Tensor,
    positions: list[int],
    prefix_len: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    prefix = []
    chunk_inputs = []
    chunk_targets = []
    next_chunk = []
    pos_buckets = []
    for start in positions:
        prefix_tokens = tokens[start - prefix_len : start]
        chunk = tokens[start : start + chunk_size]
        nxt = tokens[start + chunk_size : start + 2 * chunk_size]
        chunk_in = torch.cat([prefix_tokens[-1:].clone(), chunk[:-1].clone()])
        prefix.append(prefix_tokens)
        chunk_inputs.append(chunk_in)
        chunk_targets.append(chunk)
        next_chunk.append(nxt)
        pos_buckets.append(min(start // chunk_size, 511))
    return (
        torch.stack(prefix).to(device),
        torch.stack(chunk_inputs).to(device),
        torch.stack(chunk_targets).to(device),
        torch.stack(next_chunk).to(device),
        torch.tensor(pos_buckets, device=device, dtype=torch.long),
    )


def load_vocab_size(tokenizer_path: Path) -> int:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return int(sp.get_piece_size())


def build_sentencepiece_luts(
    tokenizer_path: Path,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    sp_vocab_size = int(sp.vocab_size())
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def retrieval_dropout(step: int, cfg: V0Config) -> float:
    if cfg.train_steps <= 1:
        return cfg.retrieval_dropout_end
    progress = step / max(cfg.train_steps - 1, 1)
    return cfg.retrieval_dropout_start + (cfg.retrieval_dropout_end - cfg.retrieval_dropout_start) * progress


def compute_candidate_scores(
    bank: SeedBank,
    k_st: torch.Tensor,
    example_nll: torch.Tensor,
    gate_probs: torch.Tensor,
    cfg: V0Config,
    vocab_size: int,
) -> torch.Tensor:
    surprise = torch.clamp(example_nll.float() / max(math.log(vocab_size), 1e-6), 0.0, 4.0)
    novelty = 0.5 * (1.0 - bank.max_similarity(k_st).clamp(-1.0, 1.0)) if bank.size > 0 else torch.ones_like(surprise)
    gate_peak = gate_probs.max(dim=-1).values.float()
    return (
        cfg.candidate_surprise_weight * surprise
        + cfg.candidate_novelty_weight * novelty
        + cfg.candidate_gate_weight * gate_peak
    )


def lr_scale_for_step(step: int, cfg: V0Config) -> float:
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


def summarize_history(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {}
    keys = [
        "loss",
        "loss_nll",
        "loss_ret",
        "loss_recon",
        "loss_gate",
        "step_ms",
        "dropout",
        "lr_scale",
        "bank_size",
        "bank_resident_mb",
        "candidates_written",
        "avg_candidate_score",
        "avg_read_mass",
        "retrieval_used",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in history if key in item]
        if values:
            summary[f"mean_{key}"] = float(sum(values) / len(values))
    summary["num_records"] = float(len(history))
    return summary


def export_quantized_model_npz(model: SpectralFloodWalkV0, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    manifest: list[dict[str, object]] = []
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy()
        key = name.replace(".", "__")
        entry: dict[str, object] = {"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
            scale = max(max_abs / 127.0, 1e-8)
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8, copy=False)
            payload[f"{key}__q"] = q
            payload[f"{key}__scale"] = np.array([scale], dtype=np.float32)
            entry["quantized"] = True
            entry["scale"] = scale
        else:
            payload[f"{key}__raw"] = arr
            entry["quantized"] = False
        manifest.append(entry)
    payload["_manifest"] = np.array([json.dumps(manifest)], dtype=object)
    np.savez_compressed(path, **payload)
    return int(path.stat().st_size)


def evaluate(
    model: SpectralFloodWalkV0,
    tokens: torch.Tensor,
    positions: list[int],
    cfg: V0Config,
    device: torch.device,
    bank: SeedBank | None,
    retrieval_enabled: bool,
    online_append: bool,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> dict[str, object]:
    model.eval()
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    active_bank = bank.clone() if (online_append and bank is not None) else bank
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    total_correct = 0
    total_appended = 0
    eval_positions = positions[: cfg.eval_samples]
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(eval_positions), cfg.batch_size):
            batch_pos = eval_positions[i : i + cfg.batch_size]
            prefix, chunk_inputs, chunk_targets, next_chunk, pos_buckets = batch_from_positions(
                tokens, batch_pos, cfg.prefix_len, cfg.chunk_size, device
            )
            out = model(
                prefix_tokens=prefix,
                chunk_inputs=chunk_inputs,
                chunk_targets=chunk_targets,
                next_chunk_tokens=next_chunk,
                bank=active_bank,
                retrieval_enabled=retrieval_enabled,
            )
            logits = out["logits"]
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), chunk_targets.reshape(-1))
            total_loss += float(loss.item()) * chunk_targets.numel()
            total_tokens += int(chunk_targets.numel())
            total_correct += int((logits.argmax(dim=-1) == chunk_targets).sum().item())
            prev_ids = chunk_inputs.reshape(-1)
            tgt_ids = chunk_targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            total_bytes += float(token_bytes.to(torch.float64).sum().item())
            if active_bank is not None:
                active_bank.note_reads(out["retrieved_indices"], out["read_attn"], cfg.read_mass_ema)
            if online_append and active_bank is not None:
                candidate_scores = compute_candidate_scores(
                    bank=active_bank,
                    k_st=out["write_k_st"],
                    example_nll=out["example_nll"],
                    gate_probs=out["gate_probs"],
                    cfg=cfg,
                    vocab_size=model.vocab_size,
                )
                max_new_entries = None if cfg.eval_append_writes_per_batch <= 0 else min(
                    cfg.eval_append_writes_per_batch, chunk_targets.size(0)
                )
                total_appended += active_bank.add_entries(
                    keys=out["write_k_int8"],
                    codes=out["write_codes"],
                    runtime_states=out["write_state"],
                    expert_ids=torch.argmax(out["gate_probs"], dim=-1),
                    pos_buckets=pos_buckets,
                    scores=candidate_scores,
                    step=i // max(cfg.batch_size, 1),
                    max_new_entries=max_new_entries,
                )
    maybe_sync_cuda(device)
    elapsed = time.perf_counter() - start
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    bits_per_token = avg_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / max(total_bytes, 1.0))
    result = {
        "loss": avg_loss,
        "val_bpb": float(val_bpb),
        "accuracy": total_correct / max(total_tokens, 1),
        "tokens": float(total_tokens),
        "bytes": float(total_bytes),
        "retrieval_enabled": float(retrieval_enabled),
        "perplexity": float(math.exp(min(avg_loss, 20.0))),
        "elapsed_s": float(elapsed),
        "tokens_per_s": float(total_tokens / max(elapsed, 1e-6)),
    }
    if active_bank is not None:
        result["bank_resident_bytes"] = float(active_bank.resident_bytes())
        result["bank_final_size"] = float(active_bank.size)
    if bank is not None:
        result["bank_initial_size"] = float(bank.size)
    result["online_append"] = float(online_append)
    result["appended_entries"] = float(total_appended)
    cuda_stats = get_cuda_memory_stats(device)
    if cuda_stats is not None:
        result["cuda_memory"] = cuda_stats
    return result


def run_smoke(cfg: V0Config) -> dict[str, object]:
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
    train_positions = build_positions(int(train_tokens.numel()), cfg.prefix_len, cfg.chunk_size, cfg.stride)
    val_positions = build_positions(int(val_tokens.numel()), cfg.prefix_len, cfg.chunk_size, cfg.stride)
    if len(train_positions) < cfg.batch_size or len(val_positions) < cfg.batch_size:
        raise ValueError("not enough tokens for the requested smoke config")

    model = SpectralFloodWalkV0(cfg, vocab_size=vocab_size).to(device)
    core_model: SpectralFloodWalkV0 = model
    if ddp_enabled:
        ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        model_for_train: SpectralFloodWalkV0 | DDP = ddp_model
        core_model = ddp_model.module  # type: ignore[assignment]
    else:
        model_for_train = model
    optimizer = torch.optim.AdamW(core_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.seed_pool_load_path is not None:
        bank = SeedBank.load_npz(
            Path(cfg.seed_pool_load_path),
            model=core_model,
            device=device,
            read_mass_weight=cfg.read_mass_weight,
            capacity=cfg.bank_capacity,
        )
    else:
        bank = SeedBank(
            model=core_model,
            capacity=cfg.bank_capacity,
            device=device,
            read_mass_weight=cfg.read_mass_weight,
            store_mode=cfg.bank_store_mode,
            runtime_state_dtype=resolve_bank_runtime_dtype(cfg.bank_runtime_dtype),
        )
    rng = random.Random(cfg.seed + 1 + rank)
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
            batch_pos = rng.sample(train_positions, cfg.batch_size)
            prefix, chunk_inputs, chunk_targets, next_chunk, pos_buckets = batch_from_positions(
                train_tokens, batch_pos, cfg.prefix_len, cfg.chunk_size, device
            )
            drop = 1.0 if step < cfg.warmup_steps else retrieval_dropout(step, cfg)
            retrieval_ready = bank.size >= cfg.bank_min_entries and step >= cfg.warmup_steps
            use_retrieval = retrieval_ready and rng.random() > drop
            out = model_for_train(
                prefix_tokens=prefix,
                chunk_inputs=chunk_inputs,
                chunk_targets=chunk_targets,
                next_chunk_tokens=next_chunk,
                bank=bank,
                retrieval_enabled=use_retrieval,
            )

            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
            optimizer.step()

            bank.note_reads(out["retrieved_indices"], out["read_attn"], cfg.read_mass_ema)

            gate_probs = out["gate_probs"]
            candidate_scores = compute_candidate_scores(
                bank=bank,
                k_st=out["write_k_st"],
                example_nll=out["example_nll"],
                gate_probs=gate_probs,
                cfg=cfg,
                vocab_size=vocab_size,
            )
            written = bank.add_entries(
                keys=out["write_k_int8"],
                codes=out["write_codes"],
                runtime_states=out["write_state"],
                expert_ids=torch.argmax(gate_probs, dim=-1),
                pos_buckets=pos_buckets,
                scores=candidate_scores,
                step=step,
                max_new_entries=min(cfg.bank_writes_per_step, cfg.batch_size),
            )

            bank_stats = bank.summary()
            if is_rank0():
                step_metrics = {
                    "step": float(step),
                    "loss": maybe_all_reduce_mean(float(out["loss"].item()), device, ddp_enabled),
                    "loss_nll": maybe_all_reduce_mean(float(out["loss_nll"].item()), device, ddp_enabled),
                    "loss_ret": maybe_all_reduce_mean(float(out["loss_ret"].item()), device, ddp_enabled),
                    "loss_recon": maybe_all_reduce_mean(float(out["loss_recon"].item()), device, ddp_enabled),
                    "loss_gate": maybe_all_reduce_mean(float(out["loss_gate"].item()), device, ddp_enabled),
                    "retrieval_used": maybe_all_reduce_mean(float(use_retrieval), device, ddp_enabled),
                    "dropout": float(drop),
                    "lr_scale": float(lr_scale),
                    "bank_size": float(bank.size),
                    "candidates_written": float(written),
                    "avg_candidate_score": float(candidate_scores.mean().item()),
                    "avg_read_mass": float(bank_stats["avg_read_mass"]),
                    "bank_resident_mb": float(bank.resident_bytes() / (1024.0 * 1024.0)),
                }
                if cfg.report_every <= 1 or step % cfg.report_every == 0 or step == cfg.train_steps - 1:
                    maybe_sync_cuda(device)
                step_metrics["step_ms"] = float((time.perf_counter() - step_start) * 1000.0)
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
                            f" bank_mb={step_metrics['bank_resident_mb']:.2f}"
                        )
                    print(
                        "[train] "
                        f"step={step}/{cfg.train_steps - 1} "
                        f"loss={step_metrics['loss']:.4f} "
                        f"nll={step_metrics['loss_nll']:.4f} "
                        f"ret={step_metrics['loss_ret']:.4f} "
                        f"recon={step_metrics['loss_recon']:.4f} "
                        f"bank={int(step_metrics['bank_size'])} "
                        f"drop={step_metrics['dropout']:.3f} "
                        f"used={int(step_metrics['retrieval_used'])} "
                        f"step_ms={step_metrics['step_ms']:.1f}"
                        f"{cuda_fragment}"
                    )
            elif ddp_enabled:
                # Keep all ranks aligned with the logging reductions triggered on rank 0.
                for value in (
                    float(out["loss"].item()),
                    float(out["loss_nll"].item()),
                    float(out["loss_ret"].item()),
                    float(out["loss_recon"].item()),
                    float(out["loss_gate"].item()),
                    float(use_retrieval),
                ):
                    _ = maybe_all_reduce_mean(value, device, ddp_enabled)

        maybe_sync_cuda(device)
        train_elapsed = time.perf_counter() - train_start
        result: dict[str, object] = {}
        if is_rank0():
            train_cuda_memory = get_cuda_memory_stats(device)
            seed_pool_path = Path(cfg.seed_pool_path) if cfg.seed_pool_path is not None else None
            if seed_pool_path is None and cfg.output_json is not None:
                seed_pool_path = Path(cfg.output_json).with_name("seed_pool.npz")
            model_artifact_path = Path(cfg.model_artifact_path) if cfg.model_artifact_path is not None else None
            if model_artifact_path is None and cfg.output_json is not None:
                model_artifact_path = Path(cfg.output_json).with_name("model_int8.npz")
            seed_pool_bytes = None
            if seed_pool_path is not None:
                seed_pool_bytes = bank.export_npz(seed_pool_path)
            model_artifact_bytes = None
            if model_artifact_path is not None:
                model_artifact_bytes = export_quantized_model_npz(core_model, model_artifact_path)
            eval_static = evaluate(
                core_model,
                val_tokens,
                val_positions,
                cfg,
                device,
                bank,
                True,
                False,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            eval_with = eval_static
            if cfg.eval_online_append:
                eval_with = evaluate(
                    core_model,
                    val_tokens,
                    val_positions,
                    cfg,
                    device,
                    bank,
                    True,
                    True,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
            eval_without = evaluate(
                core_model,
                val_tokens,
                val_positions,
                cfg,
                device,
                None,
                False,
                False,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            result = {
                "config": {**asdict(cfg), "resolved_device": str(device), "world_size": world_size},
                "train_history": history,
                "train_summary": summarize_history(history),
                "train_elapsed_s": float(train_elapsed),
                "train_tokens_per_s": float((cfg.train_steps * cfg.batch_size * cfg.chunk_size * world_size) / max(train_elapsed, 1e-6)),
                "bank_summary": bank.summary(),
                "train_cuda_memory": train_cuda_memory,
                "seed_pool_path": str(seed_pool_path) if seed_pool_path is not None else None,
                "seed_pool_load_path": cfg.seed_pool_load_path,
                "seed_pool_bytes": float(seed_pool_bytes) if seed_pool_bytes is not None else None,
                "model_artifact_path": str(model_artifact_path) if model_artifact_path is not None else None,
                "model_artifact_bytes": float(model_artifact_bytes) if model_artifact_bytes is not None else None,
                "export_bytes_total": float(
                    (seed_pool_bytes or 0) + (model_artifact_bytes or 0)
                ),
                "eval_with_retrieval_static": eval_static,
                "eval_with_retrieval": eval_with,
                "eval_without_retrieval": eval_without,
                "eval_delta_loss": float(eval_with["loss"] - eval_without["loss"]),
                "eval_delta_bpb": float(eval_with["val_bpb"] - eval_without["val_bpb"]),
                "eval_delta_accuracy": float(eval_with["accuracy"] - eval_without["accuracy"]),
                "eval_static_delta_loss": float(eval_static["loss"] - eval_without["loss"]),
                "eval_static_delta_bpb": float(eval_static["val_bpb"] - eval_without["val_bpb"]),
                "eval_static_delta_accuracy": float(eval_static["accuracy"] - eval_without["accuracy"]),
                "vocab_size": vocab_size,
            }
            if cfg.output_json is not None:
                output_path = Path(cfg.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2) + "\n")
        if ddp_enabled:
            dist.barrier()
        return result
    finally:
        cleanup_distributed_if_needed()


def parse_args() -> V0Config:
    parser = argparse.ArgumentParser(description="Smoke-scale Spectral Flood Walk LM v0 runner")
    parser.add_argument("--data-path", default=V0Config.data_path)
    parser.add_argument("--tokenizer-path", default=V0Config.tokenizer_path)
    parser.add_argument("--output-json", default=V0Config.output_json)
    parser.add_argument("--auto-log-path", default=V0Config.auto_log_path)
    parser.add_argument("--seed-pool-path", default=V0Config.seed_pool_path)
    parser.add_argument("--seed-pool-load-path", default=V0Config.seed_pool_load_path)
    parser.add_argument("--model-artifact-path", default=V0Config.model_artifact_path)
    parser.add_argument("--device", default=V0Config.device)
    parser.add_argument("--train-tokens", type=int, default=V0Config.train_tokens)
    parser.add_argument("--val-tokens", type=int, default=V0Config.val_tokens)
    parser.add_argument("--prefix-len", type=int, default=V0Config.prefix_len)
    parser.add_argument("--chunk-size", type=int, default=V0Config.chunk_size)
    parser.add_argument("--stride", type=int, default=V0Config.stride)
    parser.add_argument("--batch-size", type=int, default=V0Config.batch_size)
    parser.add_argument("--train-steps", type=int, default=V0Config.train_steps)
    parser.add_argument("--eval-samples", type=int, default=V0Config.eval_samples)
    parser.add_argument("--eval-online-append", action=argparse.BooleanOptionalAction, default=V0Config.eval_online_append)
    parser.add_argument("--eval-append-writes-per-batch", type=int, default=V0Config.eval_append_writes_per_batch)
    parser.add_argument("--bank-capacity", type=int, default=V0Config.bank_capacity)
    parser.add_argument("--bank-min-entries", type=int, default=V0Config.bank_min_entries)
    parser.add_argument("--bank-writes-per-step", type=int, default=V0Config.bank_writes_per_step)
    parser.add_argument("--bank-store-mode", choices=("codes", "runtime"), default=V0Config.bank_store_mode)
    parser.add_argument("--bank-runtime-dtype", choices=("fp16", "bf16", "fp32"), default=V0Config.bank_runtime_dtype)
    parser.add_argument("--warmup-steps", type=int, default=V0Config.warmup_steps)
    parser.add_argument("--retrieval-dropout-start", type=float, default=V0Config.retrieval_dropout_start)
    parser.add_argument("--retrieval-dropout-end", type=float, default=V0Config.retrieval_dropout_end)
    parser.add_argument("--report-every", type=int, default=V0Config.report_every)
    parser.add_argument("--num-experts", type=int, default=V0Config.num_experts)
    parser.add_argument("--embed-dim", type=int, default=V0Config.embed_dim)
    parser.add_argument("--expert-hidden", type=int, default=V0Config.expert_hidden)
    parser.add_argument("--expert-rank", type=int, default=V0Config.expert_rank)
    parser.add_argument("--fused-dim", type=int, default=V0Config.fused_dim)
    parser.add_argument("--runtime-dim", type=int, default=V0Config.runtime_dim)
    parser.add_argument("--code-dim", type=int, default=V0Config.code_dim)
    parser.add_argument("--query-dim", type=int, default=V0Config.query_dim)
    parser.add_argument("--reader-heads", type=int, default=V0Config.reader_heads)
    parser.add_argument("--topk", type=int, default=V0Config.topk)
    parser.add_argument("--read-mass-ema", type=float, default=V0Config.read_mass_ema)
    parser.add_argument("--read-mass-weight", type=float, default=V0Config.read_mass_weight)
    parser.add_argument("--candidate-surprise-weight", type=float, default=V0Config.candidate_surprise_weight)
    parser.add_argument("--candidate-novelty-weight", type=float, default=V0Config.candidate_novelty_weight)
    parser.add_argument("--candidate-gate-weight", type=float, default=V0Config.candidate_gate_weight)
    parser.add_argument("--lr", type=float, default=V0Config.lr)
    parser.add_argument("--weight-decay", type=float, default=V0Config.weight_decay)
    parser.add_argument("--min-lr-scale", type=float, default=V0Config.min_lr_scale)
    parser.add_argument("--cooldown-start-frac", type=float, default=V0Config.cooldown_start_frac)
    parser.add_argument("--seed", type=int, default=V0Config.seed)
    args = parser.parse_args()
    return V0Config(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        output_json=args.output_json,
        auto_log_path=args.auto_log_path,
        seed_pool_path=args.seed_pool_path,
        seed_pool_load_path=args.seed_pool_load_path,
        model_artifact_path=args.model_artifact_path,
        device=args.device,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        prefix_len=args.prefix_len,
        chunk_size=args.chunk_size,
        stride=args.stride,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_samples=args.eval_samples,
        eval_online_append=args.eval_online_append,
        eval_append_writes_per_batch=args.eval_append_writes_per_batch,
        bank_capacity=args.bank_capacity,
        bank_min_entries=args.bank_min_entries,
        bank_writes_per_step=args.bank_writes_per_step,
        bank_store_mode=args.bank_store_mode,
        bank_runtime_dtype=args.bank_runtime_dtype,
        warmup_steps=args.warmup_steps,
        retrieval_dropout_start=args.retrieval_dropout_start,
        retrieval_dropout_end=args.retrieval_dropout_end,
        report_every=args.report_every,
        num_experts=args.num_experts,
        embed_dim=args.embed_dim,
        expert_hidden=args.expert_hidden,
        expert_rank=args.expert_rank,
        fused_dim=args.fused_dim,
        runtime_dim=args.runtime_dim,
        code_dim=args.code_dim,
        query_dim=args.query_dim,
        reader_heads=args.reader_heads,
        topk=args.topk,
        read_mass_ema=args.read_mass_ema,
        read_mass_weight=args.read_mass_weight,
        candidate_surprise_weight=args.candidate_surprise_weight,
        candidate_novelty_weight=args.candidate_novelty_weight,
        candidate_gate_weight=args.candidate_gate_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr_scale=args.min_lr_scale,
        cooldown_start_frac=args.cooldown_start_frac,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    setup_auto_logging(cfg.auto_log_path)
    result = run_smoke(cfg)
    if is_rank0():
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
