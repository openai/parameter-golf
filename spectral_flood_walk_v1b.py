#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

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
from spectral_flood_walk_v1 import (
    V1Config,
    SpectralFloodWalkV1A,
    batch_from_starts,
    build_lm_starts,
    estimate_v1a_sizes,
    lr_scale_for_step,
)


def parse_name_csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


@dataclass
class V1BConfig:
    data_path: str = "data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    output_json: str | None = None
    auto_log_path: str | None = None
    model_artifact_path: str | None = None
    device: str = "auto"
    train_tokens: int = 262_144
    val_tokens: int = 65_536
    seq_len: int = 128
    stride: int = 64
    batch_size: int = 8
    train_steps: int = 400
    eval_batches: int = 128
    report_every: int = 10
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_mult: int = 4
    pos_buckets: int = 256
    use_semantic_memory: bool = False
    semantic_layers: str = ""
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
    eval_modes: str = "controller,raw,refined"
    episodic_bucket_count: int = 512
    episodic_max_entries: int = 65_536
    episodic_topk: int = 16
    episodic_read_alpha: float = 0.20
    episodic_route_tokens: int = 16
    episodic_write_every: int = 1
    maintenance_every: int = 16
    maintenance_budget_buckets: int = 16
    maintenance_source_limit: int = 64
    summary_per_bucket: int = 4
    merge_similarity: float = 0.94
    surprise_weight: float = 0.5

    @property
    def eval_mode_names(self) -> tuple[str, ...]:
        names = parse_name_csv(self.eval_modes)
        if not names:
            return ("controller", "raw", "refined")
        return names

    def controller_config(self) -> V1Config:
        return V1Config(
            data_path=self.data_path,
            tokenizer_path=self.tokenizer_path,
            output_json=None,
            auto_log_path=None,
            model_artifact_path=None,
            device=self.device,
            train_tokens=self.train_tokens,
            val_tokens=self.val_tokens,
            seq_len=self.seq_len,
            stride=self.stride,
            batch_size=self.batch_size,
            train_steps=self.train_steps,
            eval_batches=self.eval_batches,
            report_every=self.report_every,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_mult=self.ff_mult,
            pos_buckets=self.pos_buckets,
            semantic_layers=self.semantic_layers,
            use_semantic_memory=self.use_semantic_memory,
            pk_num_subkeys=self.pk_num_subkeys,
            pk_key_dim=self.pk_key_dim,
            pk_topk_sub=self.pk_topk_sub,
            pk_topk_final=self.pk_topk_final,
            pk_code_dim=self.pk_code_dim,
            lr=self.lr,
            min_lr_scale=self.min_lr_scale,
            warmup_steps=self.warmup_steps,
            cooldown_start_frac=self.cooldown_start_frac,
            weight_decay=self.weight_decay,
            seed=self.seed,
        )


class LocalEpisodicMemory:
    def __init__(self, cfg: V1BConfig, device: torch.device, vocab_size: int) -> None:
        self.cfg = cfg
        self.device = device
        self.dim = cfg.embed_dim
        self.bucket_count = cfg.episodic_bucket_count
        self.max_entries = cfg.episodic_max_entries
        self.topk = cfg.episodic_topk
        self.route_tokens = max(1, min(cfg.episodic_route_tokens, cfg.seq_len))
        self.summary_per_bucket = cfg.summary_per_bucket
        self.storage_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.sim_dtype = torch.float32

        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed + 2718)
        token_hash = torch.randint(
            1,
            2**62,
            (vocab_size,),
            generator=gen,
            dtype=torch.int64,
        )
        pos_hash = torch.randint(
            1,
            2**62,
            (self.route_tokens,),
            generator=gen,
            dtype=torch.int64,
        )
        self.token_hash = token_hash.to(device=device)
        self.pos_hash = pos_hash.to(device=device)

        self.keys = torch.zeros((self.max_entries, self.dim), dtype=self.storage_dtype, device=device)
        self.values = torch.zeros((self.max_entries, self.dim), dtype=self.storage_dtype, device=device)
        self.hits = torch.zeros(self.max_entries, dtype=self.sim_dtype, device=device)
        self.surprise = torch.zeros(self.max_entries, dtype=self.sim_dtype, device=device)
        self.ages = torch.zeros(self.max_entries, dtype=torch.int32, device=device)
        self.bucket_ids = torch.full((self.max_entries,), -1, dtype=torch.int32, device=device)
        self.bucket_entries: list[list[int]] = [[] for _ in range(self.bucket_count)]
        self.summary_keys = torch.zeros(
            (self.bucket_count, self.summary_per_bucket, self.dim), dtype=self.storage_dtype, device=device
        )
        self.summary_values = torch.zeros_like(self.summary_keys)
        self.summary_weights = torch.zeros((self.bucket_count, self.summary_per_bucket), dtype=self.sim_dtype, device=device)
        self.summary_counts = torch.zeros(self.bucket_count, dtype=torch.int32, device=device)
        self.dirty_queue: list[int] = []
        self.dirty_flags = torch.zeros(self.bucket_count, dtype=torch.bool, device=device)
        self.size = 0
        self.insert_cursor = 0
        self.total_appends = 0
        self.total_hit_mass = 0.0
        self.total_raw_candidates = 0
        self.total_summary_candidates = 0
        self.total_queries = 0
        self.total_maintenance_s = 0.0
        self.total_refined_buckets = 0

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.to(dtype=self.sim_dtype), dim=-1, eps=1e-6)

    def bucket_ids_for_keys(self, keys: torch.Tensor) -> torch.Tensor:
        scores = self._normalize(keys)
        fingerprint = torch.round(scores * 127.0).to(torch.int64)
        mixed = torch.zeros(scores.size(0), dtype=torch.int64, device=self.device)
        for idx in range(min(8, fingerprint.size(1))):
            mixed ^= (fingerprint[:, idx].abs() + 17 * (idx + 1))
        return torch.remainder(mixed, self.bucket_count).to(torch.int64)

    def bucket_ids_for_tokens(self, token_windows: torch.Tensor) -> torch.Tensor:
        tokens = token_windows.to(device=self.device, dtype=torch.long)
        if tokens.size(1) > self.route_tokens:
            tokens = tokens[:, -self.route_tokens :]
        mixed = torch.zeros(tokens.size(0), dtype=torch.int64, device=self.device)
        pos_hash = self.pos_hash[: tokens.size(1)]
        for idx in range(tokens.size(1)):
            mixed ^= self.token_hash[tokens[:, idx]] ^ pos_hash[idx]
        return torch.remainder(mixed.abs(), self.bucket_count).to(torch.int64)

    def _mark_dirty(self, bucket_id: int) -> None:
        if not bool(self.dirty_flags[bucket_id].item()):
            self.dirty_queue.append(bucket_id)
            self.dirty_flags[bucket_id] = True

    def _evict_slot(self, slot: int) -> None:
        old_bucket = int(self.bucket_ids[slot].item())
        if old_bucket >= 0:
            entries = self.bucket_entries[old_bucket]
            try:
                entries.remove(slot)
            except ValueError:
                pass

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        surprise: torch.Tensor,
        age: int,
        *,
        bucket_ids: torch.Tensor | None = None,
    ) -> int:
        keys = self._normalize(keys).to(dtype=self.storage_dtype)
        values = self._normalize(values).to(dtype=self.storage_dtype)
        surprise = surprise.to(device=self.device, dtype=self.sim_dtype)
        if bucket_ids is None:
            bucket_ids = self.bucket_ids_for_keys(keys)
        else:
            bucket_ids = bucket_ids.to(device=self.device, dtype=torch.int64)
        appended = 0
        for row in range(keys.size(0)):
            if self.size < self.max_entries:
                slot = self.size
                self.size += 1
            else:
                slot = self.insert_cursor
                self._evict_slot(slot)
            self.insert_cursor = (slot + 1) % self.max_entries
            bucket_id = int(bucket_ids[row].item())
            self.keys[slot] = keys[row]
            self.values[slot] = values[row]
            self.hits[slot] = 0.0
            self.surprise[slot] = surprise[row]
            self.ages[slot] = int(age)
            self.bucket_ids[slot] = bucket_id
            self.bucket_entries[bucket_id].append(slot)
            self._mark_dirty(bucket_id)
            appended += 1
        self.total_appends += appended
        return appended

    def retrieve(
        self,
        queries: torch.Tensor,
        *,
        bucket_ids: torch.Tensor,
        use_summaries: bool,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], dict[str, float]]:
        queries = self._normalize(queries)
        contexts = torch.zeros_like(queries)
        updates: list[tuple[torch.Tensor, torch.Tensor]] = []
        raw_candidates = 0
        summary_candidates = 0
        hit_mass = 0.0
        for row in range(queries.size(0)):
            query = queries[row]
            bucket_id = int(bucket_ids[row].item())
            raw_slots = self.bucket_entries[bucket_id]
            raw_candidates += len(raw_slots)
            cand_values: list[torch.Tensor] = []
            cand_scores: list[torch.Tensor] = []
            cand_slots: list[int] = []
            if raw_slots:
                slot_tensor = torch.tensor(raw_slots, device=self.device, dtype=torch.long)
                raw_keys = self._normalize(self.keys[slot_tensor])
                raw_scores = raw_keys @ query
                cand_values.append(self._normalize(self.values[slot_tensor]))
                cand_scores.append(raw_scores)
                cand_slots.extend(raw_slots)
            summary_count = int(self.summary_counts[bucket_id].item()) if use_summaries else 0
            summary_candidates += summary_count
            if summary_count > 0:
                sum_keys = self._normalize(self.summary_keys[bucket_id, :summary_count])
                sum_scores = sum_keys @ query
                cand_values.append(self._normalize(self.summary_values[bucket_id, :summary_count]))
                cand_scores.append(sum_scores)
                cand_slots.extend([-1 - slot for slot in range(summary_count)])
            if not cand_values:
                updates.append((torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=self.sim_dtype, device=self.device)))
                continue
            all_values = torch.cat(cand_values, dim=0)
            all_scores = torch.cat(cand_scores, dim=0)
            k = min(self.topk, all_scores.numel())
            top_scores, top_pos = all_scores.topk(k)
            weights = torch.softmax(top_scores / math.sqrt(self.dim), dim=0)
            contexts[row] = self._normalize((weights.unsqueeze(-1) * all_values[top_pos]).sum(dim=0)).to(dtype=queries.dtype)

            raw_hit_slots = []
            raw_hit_weights = []
            for pos, weight in zip(top_pos.tolist(), weights.tolist()):
                slot = cand_slots[pos]
                if slot >= 0:
                    raw_hit_slots.append(slot)
                    raw_hit_weights.append(weight)
            if raw_hit_slots:
                updates.append(
                    (
                        torch.tensor(raw_hit_slots, device=self.device, dtype=torch.long),
                        torch.tensor(raw_hit_weights, device=self.device, dtype=self.sim_dtype),
                    )
                )
                hit_mass += float(sum(raw_hit_weights))
            else:
                updates.append(
                    (
                        torch.empty(0, dtype=torch.long, device=self.device),
                        torch.empty(0, dtype=self.sim_dtype, device=self.device),
                    )
                )
        self.total_queries += int(queries.size(0))
        self.total_raw_candidates += raw_candidates
        self.total_summary_candidates += summary_candidates
        self.total_hit_mass += hit_mass
        stats = {
            "raw_candidates_per_query": float(raw_candidates / max(queries.size(0), 1)),
            "summary_candidates_per_query": float(summary_candidates / max(queries.size(0), 1)),
            "hit_mass_per_query": float(hit_mass / max(queries.size(0), 1)),
        }
        return contexts, updates, stats

    def apply_hit_updates(self, updates: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for slots, weights in updates:
            if slots.numel() == 0:
                continue
            self.hits.index_add_(0, slots, weights)

    def refine_dirty(self) -> dict[str, float]:
        if self.cfg.maintenance_budget_buckets <= 0 or not self.dirty_queue:
            return {"processed_buckets": 0.0, "summary_count_delta": 0.0, "maintenance_ms": 0.0}
        start = time.perf_counter()
        processed = 0
        total_summaries = 0
        while self.dirty_queue and processed < self.cfg.maintenance_budget_buckets:
            bucket_id = self.dirty_queue.pop(0)
            self.dirty_flags[bucket_id] = False
            processed += 1
            entries = self.bucket_entries[bucket_id]
            if not entries:
                self.summary_counts[bucket_id] = 0
                self.summary_weights[bucket_id].zero_()
                continue
            slot_tensor = torch.tensor(entries, device=self.device, dtype=torch.long)
            keys = self._normalize(self.keys[slot_tensor])
            values = self._normalize(self.values[slot_tensor])
            priorities = 1.0 + self.hits[slot_tensor] + self.cfg.surprise_weight * self.surprise[slot_tensor]
            source_limit = min(self.cfg.maintenance_source_limit, priorities.numel())
            source_pos = priorities.topk(source_limit).indices
            keys = keys[source_pos]
            values = values[source_pos]
            priorities = priorities[source_pos]
            used = torch.zeros(source_limit, dtype=torch.bool, device=self.device)
            summary_keys: list[torch.Tensor] = []
            summary_values: list[torch.Tensor] = []
            summary_weights: list[torch.Tensor] = []
            while len(summary_keys) < self.summary_per_bucket:
                remaining = (~used).nonzero(as_tuple=False).flatten()
                if remaining.numel() == 0:
                    break
                anchor = int(remaining[0].item())
                sim = keys @ keys[anchor]
                members = (sim >= self.cfg.merge_similarity) & (~used)
                if not bool(members.any().item()):
                    members[anchor] = True
                weights = priorities[members]
                summary_keys.append(self._normalize((weights.unsqueeze(-1) * keys[members]).sum(dim=0)))
                summary_values.append(self._normalize((weights.unsqueeze(-1) * values[members]).sum(dim=0)))
                summary_weights.append(weights.sum())
                used |= members
            count = len(summary_keys)
            self.summary_counts[bucket_id] = count
            if count > 0:
                stacked_keys = torch.stack(summary_keys).to(dtype=self.storage_dtype)
                stacked_values = torch.stack(summary_values).to(dtype=self.storage_dtype)
                stacked_weights = torch.stack(summary_weights).to(dtype=self.sim_dtype)
                self.summary_keys[bucket_id].zero_()
                self.summary_values[bucket_id].zero_()
                self.summary_weights[bucket_id].zero_()
                self.summary_keys[bucket_id, :count] = stacked_keys
                self.summary_values[bucket_id, :count] = stacked_values
                self.summary_weights[bucket_id, :count] = stacked_weights
            total_summaries += count
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.total_maintenance_s += elapsed_ms / 1000.0
        self.total_refined_buckets += processed
        return {
            "processed_buckets": float(processed),
            "summary_count_delta": float(total_summaries),
            "maintenance_ms": float(elapsed_ms),
        }

    def snapshot_stats(self) -> dict[str, float]:
        occupancy = [len(entries) for entries in self.bucket_entries]
        non_empty = [count for count in occupancy if count > 0]
        summary_total = int(self.summary_counts.sum().item())
        bytes_per_entry = self.dim * torch.tensor([], dtype=self.storage_dtype).element_size() * 2
        raw_bytes = float(self.size * bytes_per_entry)
        summary_bytes = float(summary_total * bytes_per_entry)
        return {
            "raw_entry_count": float(self.size),
            "summary_entry_count": float(summary_total),
            "bucket_non_empty": float(len(non_empty)),
            "bucket_mean_occupancy": float(sum(non_empty) / max(len(non_empty), 1)),
            "bucket_max_occupancy": float(max(non_empty) if non_empty else 0),
            "raw_memory_mb_estimate": raw_bytes / (1024.0 * 1024.0),
            "summary_memory_mb_estimate": summary_bytes / (1024.0 * 1024.0),
            "avg_raw_candidates_per_query": float(self.total_raw_candidates / max(self.total_queries, 1)),
            "avg_summary_candidates_per_query": float(self.total_summary_candidates / max(self.total_queries, 1)),
            "avg_hit_mass_per_query": float(self.total_hit_mass / max(self.total_queries, 1)),
            "total_maintenance_s": float(self.total_maintenance_s),
            "total_refined_buckets": float(self.total_refined_buckets),
        }


def per_sequence_surprise(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    return losses.view(targets.size(0), targets.size(1)).mean(dim=-1)


def evaluate_mode(
    model: SpectralFloodWalkV1A,
    tokens: torch.Tensor,
    starts: list[int],
    cfg: V1BConfig,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    mode: str,
) -> dict[str, float | dict[str, float]]:
    model.eval()
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    total_correct = 0
    semantic_score_means: list[float] = []
    retrieval_stats: list[dict[str, float]] = []
    episodic: LocalEpisodicMemory | None = None
    if mode in {"raw", "refined"}:
        episodic = LocalEpisodicMemory(cfg, device, model.vocab_size)
    start_time = time.perf_counter()
    eval_starts = starts[: cfg.eval_batches * cfg.batch_size]
    with torch.no_grad():
        for batch_idx in range(0, len(eval_starts), cfg.batch_size):
            batch_starts = eval_starts[batch_idx : batch_idx + cfg.batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, cfg.seq_len, device)
            out = model(inputs, targets=None)
            hidden = out["hidden"]
            assert hidden is not None
            hidden = hidden.to(dtype=model.token_embed.weight.dtype)
            pooled = F.normalize(hidden.mean(dim=1), dim=-1, eps=1e-6)
            route_bucket_ids = episodic.bucket_ids_for_tokens(inputs[:, -episodic.route_tokens :]) if episodic is not None else None
            retrieval_context = torch.zeros_like(pooled)
            pending_updates: list[tuple[torch.Tensor, torch.Tensor]] = []
            if episodic is not None:
                retrieval_context, pending_updates, stats = episodic.retrieve(
                    pooled,
                    bucket_ids=route_bucket_ids,
                    use_summaries=(mode == "refined"),
                )
                retrieval_stats.append(stats)
            fused_hidden = hidden
            if episodic is not None:
                fused_hidden = F.normalize(hidden + cfg.episodic_read_alpha * retrieval_context.unsqueeze(1), dim=-1, eps=1e-6)
            logits = model.lm_head(fused_hidden)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
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
            if episodic is not None:
                episodic.apply_hit_updates(pending_updates)
                if (batch_idx // cfg.batch_size) % max(cfg.episodic_write_every, 1) == 0:
                    surprise = per_sequence_surprise(logits, targets)
                    episodic.append(
                        keys=pooled,
                        values=pooled,
                        surprise=surprise,
                        age=batch_idx // cfg.batch_size,
                        bucket_ids=route_bucket_ids,
                    )
                if mode == "refined" and (batch_idx // cfg.batch_size + 1) % max(cfg.maintenance_every, 1) == 0:
                    episodic.refine_dirty()
    maybe_sync_cuda(device)
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    bits_per_token = avg_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / max(total_bytes, 1.0))
    result: dict[str, float | dict[str, float]] = {
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
    if retrieval_stats:
        result["retrieval"] = {
            "raw_candidates_per_query": float(
                sum(item["raw_candidates_per_query"] for item in retrieval_stats) / len(retrieval_stats)
            ),
            "summary_candidates_per_query": float(
                sum(item["summary_candidates_per_query"] for item in retrieval_stats) / len(retrieval_stats)
            ),
            "hit_mass_per_query": float(sum(item["hit_mass_per_query"] for item in retrieval_stats) / len(retrieval_stats)),
        }
    if episodic is not None:
        result["memory"] = episodic.snapshot_stats()
    cuda_stats = get_cuda_memory_stats(device)
    if cuda_stats is not None:
        result["cuda_memory"] = cuda_stats
    model.train()
    return result


def run_v1b(cfg: V1BConfig) -> dict[str, object]:
    set_seed(cfg.seed)
    rank, local_rank, world_size = distributed_env()
    device = resolve_device(cfg.device, local_rank)
    ddp_enabled = init_distributed_if_needed(device)
    controller_cfg = cfg.controller_config()

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
        raise ValueError("not enough tokens for the requested V1b config")

    model = SpectralFloodWalkV1A(controller_cfg, vocab_size=vocab_size).to(device)
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
            lr_scale = lr_scale_for_step(step, controller_cfg)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_scale
            batch_starts = rng.sample(train_starts, cfg.batch_size)
            inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
            out = model_for_train(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            assert out["loss"] is not None
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
                    if cfg.use_semantic_memory and controller_cfg.semantic_layer_ids:
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
            eval_results: dict[str, dict[str, float | dict[str, float]]] = {}
            for mode in cfg.eval_mode_names:
                eval_results[mode] = evaluate_mode(
                    core_model,
                    val_tokens,
                    val_starts,
                    cfg,
                    device,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    mode,
                )
            result = {
                "config": {**asdict(cfg), "resolved_device": str(device), "world_size": world_size},
                "train_history": history,
                "train_summary": summarize_history(history),
                "train_elapsed_s": float(train_elapsed),
                "train_tokens_per_s": float((cfg.train_steps * cfg.batch_size * cfg.seq_len * world_size) / max(train_elapsed, 1e-6)),
                "size_estimate": estimate_v1a_sizes(controller_cfg, vocab_size),
                "train_cuda_memory": train_cuda_memory,
                "model_artifact_path": str(model_artifact_path) if model_artifact_path is not None else None,
                "model_artifact_bytes": float(model_artifact_bytes) if model_artifact_bytes is not None else None,
                "eval_modes": eval_results,
                "vocab_size": vocab_size,
            }
            if "controller" in eval_results:
                controller_bpb = float(eval_results["controller"]["val_bpb"])
                for mode, metrics in eval_results.items():
                    if mode == "controller":
                        continue
                    metrics["delta_vs_controller_bpb"] = float(metrics["val_bpb"]) - controller_bpb
            if cfg.output_json is not None:
                output_path = Path(cfg.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2) + "\n")
        if ddp_enabled:
            torch.distributed.barrier()
        return result
    finally:
        cleanup_distributed_if_needed()


def parse_args() -> V1BConfig:
    parser = argparse.ArgumentParser(description="Spectral Flood Walk V1b episodic-memory runner")
    parser.add_argument("--data-path", default=V1BConfig.data_path)
    parser.add_argument("--tokenizer-path", default=V1BConfig.tokenizer_path)
    parser.add_argument("--output-json", default=V1BConfig.output_json)
    parser.add_argument("--auto-log-path", default=V1BConfig.auto_log_path)
    parser.add_argument("--model-artifact-path", default=V1BConfig.model_artifact_path)
    parser.add_argument("--device", default=V1BConfig.device)
    parser.add_argument("--train-tokens", type=int, default=V1BConfig.train_tokens)
    parser.add_argument("--val-tokens", type=int, default=V1BConfig.val_tokens)
    parser.add_argument("--seq-len", type=int, default=V1BConfig.seq_len)
    parser.add_argument("--stride", type=int, default=V1BConfig.stride)
    parser.add_argument("--batch-size", type=int, default=V1BConfig.batch_size)
    parser.add_argument("--train-steps", type=int, default=V1BConfig.train_steps)
    parser.add_argument("--eval-batches", type=int, default=V1BConfig.eval_batches)
    parser.add_argument("--report-every", type=int, default=V1BConfig.report_every)
    parser.add_argument("--embed-dim", type=int, default=V1BConfig.embed_dim)
    parser.add_argument("--num-layers", type=int, default=V1BConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=V1BConfig.num_heads)
    parser.add_argument("--ff-mult", type=int, default=V1BConfig.ff_mult)
    parser.add_argument("--pos-buckets", type=int, default=V1BConfig.pos_buckets)
    parser.add_argument("--use-semantic-memory", action=argparse.BooleanOptionalAction, default=V1BConfig.use_semantic_memory)
    parser.add_argument("--semantic-layers", default=V1BConfig.semantic_layers)
    parser.add_argument("--pk-num-subkeys", type=int, default=V1BConfig.pk_num_subkeys)
    parser.add_argument("--pk-key-dim", type=int, default=V1BConfig.pk_key_dim)
    parser.add_argument("--pk-topk-sub", type=int, default=V1BConfig.pk_topk_sub)
    parser.add_argument("--pk-topk-final", type=int, default=V1BConfig.pk_topk_final)
    parser.add_argument("--pk-code-dim", type=int, default=V1BConfig.pk_code_dim)
    parser.add_argument("--lr", type=float, default=V1BConfig.lr)
    parser.add_argument("--min-lr-scale", type=float, default=V1BConfig.min_lr_scale)
    parser.add_argument("--warmup-steps", type=int, default=V1BConfig.warmup_steps)
    parser.add_argument("--cooldown-start-frac", type=float, default=V1BConfig.cooldown_start_frac)
    parser.add_argument("--weight-decay", type=float, default=V1BConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=V1BConfig.seed)
    parser.add_argument("--eval-modes", default=V1BConfig.eval_modes)
    parser.add_argument("--episodic-bucket-count", type=int, default=V1BConfig.episodic_bucket_count)
    parser.add_argument("--episodic-max-entries", type=int, default=V1BConfig.episodic_max_entries)
    parser.add_argument("--episodic-topk", type=int, default=V1BConfig.episodic_topk)
    parser.add_argument("--episodic-read-alpha", type=float, default=V1BConfig.episodic_read_alpha)
    parser.add_argument("--episodic-write-every", type=int, default=V1BConfig.episodic_write_every)
    parser.add_argument("--maintenance-every", type=int, default=V1BConfig.maintenance_every)
    parser.add_argument("--maintenance-budget-buckets", type=int, default=V1BConfig.maintenance_budget_buckets)
    parser.add_argument("--maintenance-source-limit", type=int, default=V1BConfig.maintenance_source_limit)
    parser.add_argument("--summary-per-bucket", type=int, default=V1BConfig.summary_per_bucket)
    parser.add_argument("--merge-similarity", type=float, default=V1BConfig.merge_similarity)
    parser.add_argument("--surprise-weight", type=float, default=V1BConfig.surprise_weight)
    args = parser.parse_args()
    return V1BConfig(
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
        use_semantic_memory=args.use_semantic_memory,
        semantic_layers=args.semantic_layers,
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
        eval_modes=args.eval_modes,
        episodic_bucket_count=args.episodic_bucket_count,
        episodic_max_entries=args.episodic_max_entries,
        episodic_topk=args.episodic_topk,
        episodic_read_alpha=args.episodic_read_alpha,
        episodic_write_every=args.episodic_write_every,
        maintenance_every=args.maintenance_every,
        maintenance_budget_buckets=args.maintenance_budget_buckets,
        maintenance_source_limit=args.maintenance_source_limit,
        summary_per_bucket=args.summary_per_bucket,
        merge_similarity=args.merge_similarity,
        surprise_weight=args.surprise_weight,
    )


def main() -> None:
    cfg = parse_args()
    setup_auto_logging(cfg.auto_log_path)
    result = run_v1b(cfg)
    if is_rank0():
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
