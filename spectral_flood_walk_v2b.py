#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

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
    maybe_reset_cuda_peak_memory,
    maybe_sync_cuda,
    resolve_device,
    set_seed,
    setup_auto_logging,
)
from spectral_flood_walk_v2a import (
    ResidualRouter,
    StrongTransformerLM,
    batch_from_starts,
    build_eval_score_mask,
    build_lm_starts,
    summarize_history,
    train_base_model,
)


def parse_int_csv(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return ()
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def parse_float_csv(spec: str) -> tuple[float, ...]:
    if not spec.strip():
        return ()
    return tuple(float(part.strip()) for part in spec.split(",") if part.strip())


def clip_rows_to_norm_(rows: torch.Tensor, max_norm: float) -> None:
    if max_norm <= 0.0 or rows.numel() == 0:
        return
    norms = rows.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rows.mul_((max_norm / norms).clamp(max=1.0))


def hidden_cross_entropy_gradient(
    model: StrongTransformerLM,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    flat_grad = probs.reshape(-1, probs.size(-1))
    flat_targets = targets.reshape(-1)
    flat_grad[torch.arange(flat_grad.size(0), device=flat_grad.device), flat_targets] -= 1.0

    if model.logit_softcap > 0.0:
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        softcap_grad = 1.0 - (flat_logits / model.logit_softcap).square()
        flat_grad = flat_grad * softcap_grad

    if model.tie_embeddings:
        weight = model.tok_emb.weight.float()
    else:
        if model.lm_head is None:
            raise RuntimeError("lm_head is required when tie_embeddings=False")
        weight = model.lm_head.weight.float()
    hidden_grad = flat_grad @ weight
    return hidden_grad.reshape(logits.size(0), logits.size(1), weight.size(1))


@dataclass
class V2bConfig:
    data_path: str = "data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    output_json: str | None = None
    auto_log_path: str | None = None
    model_artifact_path: str | None = None
    device: str = "auto"
    train_tokens: int = 262_144
    val_tokens: int = 65_536
    seq_len: int = 256
    stride: int = 64
    batch_size: int = 8
    train_steps: int = 128
    eval_batches: int = 128
    report_every: int = 8
    vocab_size_override: int = 0
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 3
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    rope_base: float = 10_000.0
    logit_softcap: float = 30.0
    qk_gain_init: float = 1.5
    spine_variant: str = "plain"
    xsa_last_n: int = 0
    base_lr: float = 2e-3
    min_lr_scale: float = 0.10
    warmup_steps: int = 8
    cooldown_start_frac: float = 0.75
    weight_decay: float = 1e-2
    seed: int = 1337

    memory_orders: str = "1,2,3,4"
    memory_order_scales: str = ""
    memory_table_size: int = 65_536
    memory_update_lr: float = 0.05
    memory_decay: float = 0.999
    memory_ema_decay: float = 0.95
    memory_read_scale: float = 1.0
    memory_min_read_count: float = 2.0
    memory_max_delta_norm: float = 4.0

    maintenance_passes: int = 1
    maintenance_blend: float = 0.25
    maintenance_max_slots: int = 128
    maintenance_metric: str = "counts"
    maintenance_use_grad: bool = True

    @property
    def memory_order_ids(self) -> tuple[int, ...]:
        return parse_int_csv(self.memory_orders)

    @property
    def memory_order_scale_values(self) -> tuple[float, ...]:
        values = parse_float_csv(self.memory_order_scales)
        if not values:
            return tuple(1.0 for _ in self.memory_order_ids)
        if len(values) != len(self.memory_order_ids):
            raise ValueError("memory_order_scales length must match memory_orders length")
        return values


class PersistentHiddenMemory:
    def __init__(
        self,
        *,
        orders: tuple[int, ...],
        table_size: int,
        dim: int,
        device: torch.device,
        order_scales: tuple[float, ...],
        decay: float,
        ema_decay: float,
        read_scale: float,
        min_read_count: float,
        max_delta_norm: float,
        maintenance_passes: int,
        maintenance_blend: float,
        maintenance_max_slots: int,
        maintenance_metric: str,
        maintenance_use_grad: bool,
    ) -> None:
        if len(order_scales) != len(orders):
            raise ValueError("order_scales length must match number of orders")
        self.orders = orders
        self.table_size = table_size
        self.dim = dim
        self.device = device
        self.order_scales = torch.tensor(order_scales, dtype=torch.float32, device=device)
        self.decay = float(decay)
        self.ema_decay = float(ema_decay)
        self.read_scale = float(read_scale)
        self.min_read_count = float(min_read_count)
        self.max_delta_norm = float(max_delta_norm)
        self.maintenance_passes = int(maintenance_passes)
        self.maintenance_blend = float(maintenance_blend)
        self.maintenance_max_slots = int(maintenance_max_slots)
        self.maintenance_metric = maintenance_metric
        self.maintenance_use_grad = bool(maintenance_use_grad)

        shape = (len(orders), table_size, dim)
        self.delta = torch.zeros(shape, dtype=torch.float32, device=device)
        self.grad_ema = torch.zeros(shape, dtype=torch.float32, device=device)
        self.counts = torch.zeros((len(orders), table_size), dtype=torch.float32, device=device)
        self.hits = torch.zeros((len(orders), table_size), dtype=torch.float32, device=device)

    def clone(self) -> "PersistentHiddenMemory":
        cloned = PersistentHiddenMemory(
            orders=self.orders,
            table_size=self.table_size,
            dim=self.dim,
            device=self.device,
            order_scales=tuple(float(x) for x in self.order_scales.detach().cpu().tolist()),
            decay=self.decay,
            ema_decay=self.ema_decay,
            read_scale=self.read_scale,
            min_read_count=self.min_read_count,
            max_delta_norm=self.max_delta_norm,
            maintenance_passes=self.maintenance_passes,
            maintenance_blend=self.maintenance_blend,
            maintenance_max_slots=self.maintenance_max_slots,
            maintenance_metric=self.maintenance_metric,
            maintenance_use_grad=self.maintenance_use_grad,
        )
        cloned.delta = self.delta.clone()
        cloned.grad_ema = self.grad_ema.clone()
        cloned.counts = self.counts.clone()
        cloned.hits = self.hits.clone()
        return cloned

    def resident_bytes(self) -> int:
        total = 0
        for tensor in (self.delta, self.grad_ema, self.counts, self.hits):
            total += int(tensor.numel() * tensor.element_size())
        return total

    def stats(self) -> dict[str, float]:
        non_empty = (self.counts > 0).float()
        readable = (self.counts >= self.min_read_count).float()
        delta_norm = self.delta.norm(dim=-1)
        grad_norm = self.grad_ema.norm(dim=-1)
        return {
            "resident_mb": float(self.resident_bytes() / (1024.0 * 1024.0)),
            "non_empty_fraction": float(non_empty.mean().item()),
            "readable_fraction": float(readable.mean().item()),
            "mean_non_empty_slots": float(non_empty.sum(dim=1).mean().item()),
            "mean_delta_norm": float(delta_norm.mean().item()),
            "mean_grad_ema_norm": float(grad_norm.mean().item()),
            "mean_hits": float(self.hits.mean().item()),
            "mean_counts": float(self.counts.mean().item()),
        }

    def lookup(self, context_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], int]:
        if context_ids.numel() == 0:
            empty = torch.zeros((*context_ids.shape[:2], self.dim), dtype=torch.float32, device=self.device)
            return empty, {"active_slots": 0.0, "readable_slots": 0.0, "mean_delta_norm": 0.0}, 0

        gathered: list[torch.Tensor] = []
        active_slots = 0.0
        readable_slots = 0.0
        delta_norms: list[float] = []
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx]
            delta = self.delta[order_idx, ids]
            counts = self.counts[order_idx, ids]
            readable_mask = counts >= self.min_read_count if self.min_read_count > 0.0 else counts > 0.0
            scaled = delta * readable_mask.unsqueeze(-1).to(dtype=delta.dtype)
            scaled = scaled * self.order_scales[order_idx]
            gathered.append(scaled)
            active_slots += float((counts > 0).float().mean().item())
            readable_slots += float(readable_mask.float().mean().item())
            delta_norms.append(float(scaled.norm(dim=-1).mean().item()))

        combined = torch.stack(gathered, dim=2).sum(dim=2)
        combined = combined * self.read_scale
        flops = int(context_ids.size(0) * context_ids.size(1) * max(context_ids.size(2), 1) * self.dim * 2)
        stats = {
            "active_slots": active_slots / max(context_ids.size(2), 1),
            "readable_slots": readable_slots / max(context_ids.size(2), 1),
            "mean_delta_norm": sum(delta_norms) / max(len(delta_norms), 1),
        }
        return combined, stats, flops

    @torch.no_grad()
    def note_reads(self, context_ids: torch.Tensor) -> None:
        if context_ids.numel() == 0:
            return
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx].reshape(-1)
            self.hits[order_idx].scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def _select_maintenance_ids(self, order_idx: int, slot_ids: torch.Tensor) -> torch.Tensor:
        if slot_ids.numel() == 0:
            return slot_ids
        if self.maintenance_max_slots <= 0 or slot_ids.numel() <= self.maintenance_max_slots:
            return slot_ids
        source = self.counts if self.maintenance_metric == "counts" else self.hits
        values = source[order_idx, slot_ids]
        topk = torch.topk(values, k=self.maintenance_max_slots, largest=True).indices
        return slot_ids[topk]

    @torch.no_grad()
    def _run_maintenance(self, touched: list[tuple[int, torch.Tensor]]) -> tuple[int, int]:
        if self.maintenance_passes <= 0:
            return 0, 0
        total_flops = 0
        total_slots = 0
        scale = math.sqrt(float(self.dim))
        for order_idx, slot_ids in touched:
            work_ids = self._select_maintenance_ids(order_idx, slot_ids)
            if work_ids.numel() == 0:
                continue
            state = self.delta[order_idx, work_ids]
            grad_state = self.grad_ema[order_idx, work_ids]
            slots = int(work_ids.numel())
            for _ in range(self.maintenance_passes):
                query = state + grad_state if self.maintenance_use_grad else state
                scores = (query @ query.transpose(0, 1)) / scale
                attn = torch.softmax(scores.float(), dim=-1).to(dtype=state.dtype)
                mixed = attn @ state
                state = torch.lerp(state, mixed, self.maintenance_blend)
                clip_rows_to_norm_(state, self.max_delta_norm)
                total_flops += int(4 * slots * slots * self.dim + 2 * slots * self.dim)
            self.delta[order_idx, work_ids] = state
            total_slots += slots
        return total_flops, total_slots

    @torch.no_grad()
    def update(
        self,
        context_ids: torch.Tensor,
        hidden_grad: torch.Tensor,
        step_size: float,
        score_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if context_ids.numel() == 0:
            return {"updated_slots": 0.0, "update_flops": 0.0, "maintenance_flops": 0.0, "maintenance_slots": 0.0}

        flat_grad = hidden_grad.to(device=self.device, dtype=torch.float32).reshape(-1, self.dim)
        if score_mask is None:
            active_weights = torch.ones((flat_grad.size(0),), dtype=torch.float32, device=self.device)
        else:
            active_weights = score_mask.to(device=self.device, dtype=torch.float32).reshape(-1)
        active_mask = active_weights > 0
        if not bool(active_mask.any().item()):
            return {"updated_slots": 0.0, "update_flops": 0.0, "maintenance_flops": 0.0, "maintenance_slots": 0.0}

        total_update_flops = 0
        touched: list[tuple[int, torch.Tensor]] = []
        updated_slots = 0

        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx].reshape(-1)[active_mask]
            grads = flat_grad[active_mask] * self.order_scales[order_idx]
            weights = active_weights[active_mask]
            grads = grads * weights.unsqueeze(-1)

            unique_ids, inverse = torch.unique(ids, return_inverse=True)
            accum = torch.zeros((unique_ids.numel(), self.dim), dtype=torch.float32, device=self.device)
            accum.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, self.dim), grads)
            slot_weight = torch.zeros((unique_ids.numel(),), dtype=torch.float32, device=self.device)
            slot_weight.scatter_add_(0, inverse, weights)
            accum = accum / slot_weight.clamp_min(1.0).unsqueeze(-1)

            ema = self.grad_ema[order_idx, unique_ids]
            ema = self.ema_decay * ema + (1.0 - self.ema_decay) * accum
            delta = self.delta[order_idx, unique_ids]
            delta = self.decay * delta - step_size * ema
            clip_rows_to_norm_(delta, self.max_delta_norm)

            self.grad_ema[order_idx, unique_ids] = ema
            self.delta[order_idx, unique_ids] = delta
            self.counts[order_idx].scatter_add_(0, ids, weights)

            updated_slots += int(unique_ids.numel())
            touched.append((order_idx, unique_ids))
            total_update_flops += int(ids.numel() * self.dim * 2 + unique_ids.numel() * self.dim * 6)

        maintenance_flops, maintenance_slots = self._run_maintenance(touched)
        return {
            "updated_slots": float(updated_slots),
            "update_flops": float(total_update_flops),
            "maintenance_flops": float(maintenance_flops),
            "maintenance_slots": float(maintenance_slots),
        }


def evaluate_mode(
    *,
    mode: str,
    model: StrongTransformerLM,
    memory: PersistentHiddenMemory | None,
    router: ResidualRouter | None,
    tokens: torch.Tensor,
    starts: list[int],
    cfg: V2bConfig,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> dict[str, object]:
    model.eval()
    active_memory = memory.clone() if mode == "online_persistent_hidden" and memory is not None else memory
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    total_correct = 0
    active_slot_means: list[float] = []
    readable_slot_means: list[float] = []
    delta_norm_means: list[float] = []
    memory_lookup_flops = 0.0
    memory_update_flops = 0.0
    maintenance_flops = 0.0
    maintenance_slots = 0.0
    eval_starts = starts[: cfg.eval_batches * cfg.batch_size]
    end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    wall_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if wall_start is not None:
        wall_start.record()
    else:
        cpu_start = time.perf_counter()

    with torch.no_grad():
        for idx in range(0, len(eval_starts), cfg.batch_size):
            batch_starts = eval_starts[idx : idx + cfg.batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, cfg.seq_len, device)
            score_mask = build_eval_score_mask(batch_starts, cfg.seq_len, cfg.stride, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(inputs)
                hidden = out["hidden"]
                logits = out["logits"]
                assert hidden is not None
                assert logits is not None
                context_ids = None
                adapted_hidden = hidden
                if mode != "context" and active_memory is not None and router is not None:
                    context_ids = router.context_ids(inputs)
                    delta_h, memory_stats, lookup_flops = active_memory.lookup(context_ids)
                    adapted_hidden = hidden + delta_h.to(dtype=hidden.dtype)
                    logits = model.logits_from_hidden(adapted_hidden)
                    active_slot_means.append(float(memory_stats["active_slots"]))
                    readable_slot_means.append(float(memory_stats["readable_slots"]))
                    delta_norm_means.append(float(memory_stats["mean_delta_norm"]))
                    memory_lookup_flops += float(lookup_flops)

            flat_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            total_loss += float(flat_loss[score_mask].sum().item())
            total_tokens += int(score_mask.sum().item())
            total_correct += int(((logits.argmax(dim=-1) == targets) & score_mask).sum().item())

            prev_ids = inputs.reshape(-1)
            tgt_ids = targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            total_bytes += float(token_bytes.reshape_as(targets)[score_mask].to(torch.float64).sum().item())

            if mode == "online_persistent_hidden" and active_memory is not None and router is not None and context_ids is not None:
                hidden_grad = hidden_cross_entropy_gradient(model, logits, targets)
                update_stats = active_memory.update(
                    context_ids,
                    hidden_grad,
                    cfg.memory_update_lr,
                    score_mask=score_mask,
                )
                active_memory.note_reads(context_ids)
                memory_update_flops += float(update_stats["update_flops"])
                maintenance_flops += float(update_stats["maintenance_flops"])
                maintenance_slots += float(update_stats["maintenance_slots"])

    if end_time is not None and wall_start is not None:
        end_time.record()
        maybe_sync_cuda(device)
        elapsed = float(wall_start.elapsed_time(end_time) / 1000.0)
    else:
        elapsed = float(time.perf_counter() - cpu_start)

    avg_loss = total_loss / max(total_tokens, 1)
    bits_per_token = avg_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / max(total_bytes, 1.0))
    result: dict[str, object] = {
        "loss": float(avg_loss),
        "val_bpb": float(val_bpb),
        "accuracy": float(total_correct / max(total_tokens, 1)),
        "tokens": float(total_tokens),
        "bytes": float(total_bytes),
        "elapsed_s": float(elapsed),
        "tokens_per_s": float(total_tokens / max(elapsed, 1e-6)),
        "memory_lookup_flops_estimate": float(memory_lookup_flops),
        "memory_update_flops_estimate": float(memory_update_flops),
        "memory_maintenance_flops_estimate": float(maintenance_flops),
        "memory_total_flops_estimate": float(memory_lookup_flops + memory_update_flops + maintenance_flops),
        "maintenance_slots": float(maintenance_slots),
    }
    if active_slot_means:
        result["active_slots_mean"] = float(sum(active_slot_means) / len(active_slot_means))
        result["readable_slots_mean"] = float(sum(readable_slot_means) / len(readable_slot_means))
        result["delta_norm_mean"] = float(sum(delta_norm_means) / len(delta_norm_means))
    if active_memory is not None:
        result["persistent_memory"] = active_memory.stats()
    cuda_stats = get_cuda_memory_stats(device)
    if cuda_stats is not None:
        result["cuda_memory"] = cuda_stats
    return result


def run_v2b(cfg: V2bConfig) -> dict[str, object]:
    set_seed(cfg.seed)
    _, local_rank, world_size = distributed_env()
    device = resolve_device(cfg.device, local_rank)
    ddp_enabled = init_distributed_if_needed(device)
    tokenizer_path = Path(cfg.tokenizer_path)
    vocab_size = cfg.vocab_size_override or load_vocab_size(tokenizer_path)
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
        raise ValueError("not enough tokens for the requested V2b config")

    model = StrongTransformerLM(cfg, vocab_size=vocab_size).to(device)
    train_history, train_elapsed = train_base_model(model, cfg, train_tokens, train_starts, device, ddp_enabled)

    router = ResidualRouter(cfg.memory_order_ids, cfg.memory_table_size, device)
    memory = PersistentHiddenMemory(
        orders=cfg.memory_order_ids,
        table_size=cfg.memory_table_size,
        dim=cfg.model_dim,
        device=device,
        order_scales=cfg.memory_order_scale_values,
        decay=cfg.memory_decay,
        ema_decay=cfg.memory_ema_decay,
        read_scale=cfg.memory_read_scale,
        min_read_count=cfg.memory_min_read_count,
        max_delta_norm=cfg.memory_max_delta_norm,
        maintenance_passes=cfg.maintenance_passes,
        maintenance_blend=cfg.maintenance_blend,
        maintenance_max_slots=cfg.maintenance_max_slots,
        maintenance_metric=cfg.maintenance_metric,
        maintenance_use_grad=cfg.maintenance_use_grad,
    )

    result: dict[str, object] = {}
    if is_rank0():
        train_cuda_memory = get_cuda_memory_stats(device)
        model_artifact_path = Path(cfg.model_artifact_path) if cfg.model_artifact_path is not None else None
        if model_artifact_path is None and cfg.output_json is not None:
            model_artifact_path = Path(cfg.output_json).with_name("model_int8.npz")
        model_artifact_bytes = None
        if model_artifact_path is not None:
            model_artifact_bytes = export_quantized_model_npz(model, model_artifact_path)

        eval_context = evaluate_mode(
            mode="context",
            model=model,
            memory=None,
            router=None,
            tokens=val_tokens,
            starts=val_starts,
            cfg=cfg,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        eval_online = evaluate_mode(
            mode="online_persistent_hidden",
            model=model,
            memory=memory,
            router=router,
            tokens=val_tokens,
            starts=val_starts,
            cfg=cfg,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )

        result = {
            "config": {**asdict(cfg), "resolved_device": str(device), "world_size": world_size, "vocab_size": vocab_size},
            "train_history": train_history,
            "train_summary": summarize_history(train_history),
            "train_elapsed_s": float(train_elapsed),
            "train_tokens_per_s": float((cfg.train_steps * cfg.batch_size * cfg.seq_len * world_size) / max(train_elapsed, 1e-6)),
            "train_cuda_memory": train_cuda_memory,
            "base_model_param_mb_estimate": float(model.compact_bytes() / (1024.0 * 1024.0)),
            "persistent_memory_resident_mb_estimate": float(memory.resident_bytes() / (1024.0 * 1024.0)),
            "model_artifact_path": str(model_artifact_path) if model_artifact_path is not None else None,
            "model_artifact_bytes": float(model_artifact_bytes) if model_artifact_bytes is not None else None,
            "eval_context": eval_context,
            "eval_online_persistent_hidden": eval_online,
            "eval_delta_online_bpb": float(eval_online["val_bpb"] - eval_context["val_bpb"]),
        }
        if cfg.output_json is not None:
            output_path = Path(cfg.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n")

    if ddp_enabled:
        torch.distributed.barrier()
    cleanup_distributed_if_needed()
    return result


def parse_args() -> V2bConfig:
    parser = argparse.ArgumentParser(description="Spectral Flood Walk V2b runner")
    parser.add_argument("--data-path", default=V2bConfig.data_path)
    parser.add_argument("--tokenizer-path", default=V2bConfig.tokenizer_path)
    parser.add_argument("--output-json", default=V2bConfig.output_json)
    parser.add_argument("--auto-log-path", default=V2bConfig.auto_log_path)
    parser.add_argument("--model-artifact-path", default=V2bConfig.model_artifact_path)
    parser.add_argument("--device", default=V2bConfig.device)
    parser.add_argument("--train-tokens", type=int, default=V2bConfig.train_tokens)
    parser.add_argument("--val-tokens", type=int, default=V2bConfig.val_tokens)
    parser.add_argument("--seq-len", type=int, default=V2bConfig.seq_len)
    parser.add_argument("--stride", type=int, default=V2bConfig.stride)
    parser.add_argument("--batch-size", type=int, default=V2bConfig.batch_size)
    parser.add_argument("--train-steps", type=int, default=V2bConfig.train_steps)
    parser.add_argument("--eval-batches", type=int, default=V2bConfig.eval_batches)
    parser.add_argument("--report-every", type=int, default=V2bConfig.report_every)
    parser.add_argument("--vocab-size-override", type=int, default=V2bConfig.vocab_size_override)
    parser.add_argument("--model-dim", type=int, default=V2bConfig.model_dim)
    parser.add_argument("--num-layers", type=int, default=V2bConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=V2bConfig.num_heads)
    parser.add_argument("--num-kv-heads", type=int, default=V2bConfig.num_kv_heads)
    parser.add_argument("--mlp-mult", type=int, default=V2bConfig.mlp_mult)
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=V2bConfig.tie_embeddings)
    parser.add_argument("--tied-embed-init-std", type=float, default=V2bConfig.tied_embed_init_std)
    parser.add_argument("--rope-base", type=float, default=V2bConfig.rope_base)
    parser.add_argument("--logit-softcap", type=float, default=V2bConfig.logit_softcap)
    parser.add_argument("--qk-gain-init", type=float, default=V2bConfig.qk_gain_init)
    parser.add_argument("--spine-variant", choices=("plain", "xsa"), default=V2bConfig.spine_variant)
    parser.add_argument("--xsa-last-n", type=int, default=V2bConfig.xsa_last_n)
    parser.add_argument("--base-lr", type=float, default=V2bConfig.base_lr)
    parser.add_argument("--min-lr-scale", type=float, default=V2bConfig.min_lr_scale)
    parser.add_argument("--warmup-steps", type=int, default=V2bConfig.warmup_steps)
    parser.add_argument("--cooldown-start-frac", type=float, default=V2bConfig.cooldown_start_frac)
    parser.add_argument("--weight-decay", type=float, default=V2bConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=V2bConfig.seed)

    parser.add_argument("--memory-orders", default=V2bConfig.memory_orders)
    parser.add_argument("--memory-order-scales", default=V2bConfig.memory_order_scales)
    parser.add_argument("--memory-table-size", type=int, default=V2bConfig.memory_table_size)
    parser.add_argument("--memory-update-lr", type=float, default=V2bConfig.memory_update_lr)
    parser.add_argument("--memory-decay", type=float, default=V2bConfig.memory_decay)
    parser.add_argument("--memory-ema-decay", type=float, default=V2bConfig.memory_ema_decay)
    parser.add_argument("--memory-read-scale", type=float, default=V2bConfig.memory_read_scale)
    parser.add_argument("--memory-min-read-count", type=float, default=V2bConfig.memory_min_read_count)
    parser.add_argument("--memory-max-delta-norm", type=float, default=V2bConfig.memory_max_delta_norm)

    parser.add_argument("--maintenance-passes", type=int, default=V2bConfig.maintenance_passes)
    parser.add_argument("--maintenance-blend", type=float, default=V2bConfig.maintenance_blend)
    parser.add_argument("--maintenance-max-slots", type=int, default=V2bConfig.maintenance_max_slots)
    parser.add_argument("--maintenance-metric", choices=("counts", "hits"), default=V2bConfig.maintenance_metric)
    parser.add_argument("--maintenance-use-grad", action=argparse.BooleanOptionalAction, default=V2bConfig.maintenance_use_grad)

    args = parser.parse_args()
    return V2bConfig(
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
        vocab_size_override=args.vocab_size_override,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base,
        logit_softcap=args.logit_softcap,
        qk_gain_init=args.qk_gain_init,
        spine_variant=args.spine_variant,
        xsa_last_n=args.xsa_last_n,
        base_lr=args.base_lr,
        min_lr_scale=args.min_lr_scale,
        warmup_steps=args.warmup_steps,
        cooldown_start_frac=args.cooldown_start_frac,
        weight_decay=args.weight_decay,
        seed=args.seed,
        memory_orders=args.memory_orders,
        memory_order_scales=args.memory_order_scales,
        memory_table_size=args.memory_table_size,
        memory_update_lr=args.memory_update_lr,
        memory_decay=args.memory_decay,
        memory_ema_decay=args.memory_ema_decay,
        memory_read_scale=args.memory_read_scale,
        memory_min_read_count=args.memory_min_read_count,
        memory_max_delta_norm=args.memory_max_delta_norm,
        maintenance_passes=args.maintenance_passes,
        maintenance_blend=args.maintenance_blend,
        maintenance_max_slots=args.maintenance_max_slots,
        maintenance_metric=args.maintenance_metric,
        maintenance_use_grad=args.maintenance_use_grad,
    )


def main() -> None:
    cfg = parse_args()
    setup_auto_logging(cfg.auto_log_path)
    result = run_v2b(cfg)
    if is_rank0():
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
