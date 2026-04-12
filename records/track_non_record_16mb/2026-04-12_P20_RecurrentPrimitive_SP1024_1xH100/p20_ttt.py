from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


P20_TTT_MODES = frozenset({"off", "residual_scalar", "residual_vector"})


@dataclass(frozen=True)
class P20TTTStats:
    doc_count: int
    chunk_count: int
    update_count: int
    token_count: float
    byte_count: float


@dataclass(frozen=True)
class P20TTTPairedResult:
    baseline_val_loss: float
    baseline_val_bpb: float
    ttt_val_loss: float
    ttt_val_bpb: float
    baseline_stats: P20TTTStats
    ttt_stats: P20TTTStats
    paired_stats: dict[str, float | int | None]
    per_doc: list[dict[str, float | int]]


def normalize_p20_ttt_mode(raw_mode: str) -> str:
    mode = raw_mode.strip().lower()
    if mode not in P20_TTT_MODES:
        raise ValueError(f"P20_TTT_MODE must be one of {sorted(P20_TTT_MODES)}, got {raw_mode!r}")
    return mode


def _iter_p20_blocks(model: nn.Module) -> Iterable[nn.Module]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return
    for block in blocks:
        if (
            hasattr(block, "primitive")
            and hasattr(block, "readout_projection")
            and hasattr(block, "residual_scale")
        ):
            yield block


def _doc_ranges_from_bos(val_tokens: Tensor, bos_token_id: int) -> list[tuple[int, int]]:
    if val_tokens.device.type != "cpu":
        val_tokens = val_tokens.cpu()
    starts = (val_tokens == bos_token_id).nonzero(as_tuple=False).flatten().tolist()
    if not starts:
        raise ValueError(f"no BOS token id {bos_token_id} found in validation token stream")
    if starts[0] != 0:
        starts.insert(0, 0)
    ends = starts[1:] + [int(val_tokens.numel())]
    return [(start, end) for start, end in zip(starts, ends, strict=True) if end - start >= 2]


def _reset_adapter_params(params: list[Tensor], optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for param in params:
            param.zero_()
    optimizer.state.clear()


def _set_p20_ttt_gate_params(model: nn.Module, mode: str, device: torch.device) -> tuple[list[nn.Module], list[Tensor]]:
    blocks = list(_iter_p20_blocks(model))
    if not blocks:
        raise ValueError("P20_TTT_MODE requires at least one P20 primitive block")

    params: list[Tensor] = []
    for block in blocks:
        residual_scale = getattr(block, "residual_scale")
        if mode == "residual_scalar":
            shape = (1,)
        elif mode == "residual_vector":
            shape = tuple(residual_scale.shape)
        else:
            raise ValueError(f"unsupported P20 TTT mode {mode!r}")
        gate = torch.zeros(shape, device=device, dtype=torch.float32, requires_grad=True)
        setattr(block, "p20_ttt_residual_gate", gate)
        params.append(gate)
    return blocks, params


def _clear_p20_ttt_gate_params(blocks: list[nn.Module]) -> None:
    for block in blocks:
        setattr(block, "p20_ttt_residual_gate", None)


def _restore_requires_grad(params: list[nn.Parameter], flags: list[bool]) -> None:
    for param, flag in zip(params, flags, strict=True):
        param.requires_grad_(flag)


def _next_chunk_end(position: int, target_tokens: int, requested_chunk_size: int, scan_chunk_size: int) -> int:
    remaining = target_tokens - position
    if remaining <= requested_chunk_size:
        if scan_chunk_size > 0 and remaining > scan_chunk_size:
            rounded = (remaining // scan_chunk_size) * scan_chunk_size
            if rounded > 0:
                return position + rounded
        return position + remaining

    chunk = requested_chunk_size
    if scan_chunk_size > 0 and chunk > scan_chunk_size:
        chunk = (chunk // scan_chunk_size) * scan_chunk_size
    return position + max(chunk, 1)


def _resolve_context_size(args) -> int:
    context_size = int(getattr(args, "p20_ttt_context_size", 0) or 0)
    if context_size <= 0:
        context_size = int(args.train_seq_len)
    if context_size < args.p20_ttt_chunk_size:
        raise ValueError(
            "P20_TTT_CONTEXT_SIZE must be 0 or at least P20_TTT_CHUNK_SIZE; "
            f"got context={context_size} chunk={args.p20_ttt_chunk_size}"
        )
    return context_size


def _chunk_tensors(
    val_tokens: Tensor,
    doc_start_token: int,
    position: int,
    next_position: int,
    context_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    context_start = max(0, next_position - context_size)
    if context_start > position:
        raise RuntimeError(
            f"context start {context_start} passed score position {position}; "
            "increase P20_TTT_CONTEXT_SIZE or reduce P20_TTT_CHUNK_SIZE"
        )
    x_cpu = val_tokens[doc_start_token + context_start : doc_start_token + next_position]
    y_cpu = val_tokens[doc_start_token + position + 1 : doc_start_token + next_position + 1]
    prev_cpu = val_tokens[doc_start_token + position : doc_start_token + next_position]
    x = x_cpu.to(device=device, dtype=torch.int64, non_blocking=True).unsqueeze(0)
    y = y_cpu.to(device=device, dtype=torch.int64, non_blocking=True).unsqueeze(0)
    prev = prev_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
    score_offset = position - context_start
    return x, y, prev, score_offset


def _token_byte_count(
    previous_ids: Tensor,
    target_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    token_bytes = base_bytes_lut[target_ids].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[target_ids] & ~is_boundary_token_lut[previous_ids]).to(dtype=torch.int16)
    return token_bytes.to(torch.float64).sum()


def _bpb_from_loss_and_counts(loss_sum: float, token_count: float, byte_count: float) -> float:
    if token_count <= 0 or byte_count <= 0:
        return float("nan")
    return (loss_sum / token_count) / math.log(2.0) * (token_count / byte_count)


def _summarize_paired_records(
    records: list[dict[str, float | int]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float | int | None]:
    if not records:
        return {
            "doc_count": 0,
            "delta_bpb": None,
            "mean_doc_delta_bpb": None,
            "median_doc_delta_bpb": None,
            "stdev_doc_delta_bpb": None,
            "sem_doc_delta_bpb": None,
            "bootstrap_samples": 0,
            "bootstrap_ci95_low_bpb": None,
            "bootstrap_ci95_high_bpb": None,
            "bootstrap_p_two_sided": None,
            "improved_doc_count": 0,
            "worsened_doc_count": 0,
            "unchanged_doc_count": 0,
        }

    n_docs = len(records)
    baseline_loss_sum = sum(float(record["baseline_loss_sum"]) for record in records)
    ttt_loss_sum = sum(float(record["ttt_loss_sum"]) for record in records)
    token_count = sum(float(record["token_count"]) for record in records)
    byte_count = sum(float(record["byte_count"]) for record in records)
    baseline_bpb = _bpb_from_loss_and_counts(baseline_loss_sum, token_count, byte_count)
    ttt_bpb = _bpb_from_loss_and_counts(ttt_loss_sum, token_count, byte_count)
    deltas = [float(record["delta_bpb"]) for record in records]
    mean_delta = sum(deltas) / n_docs
    median_delta = sorted(deltas)[n_docs // 2] if n_docs % 2 else 0.5 * (
        sorted(deltas)[n_docs // 2 - 1] + sorted(deltas)[n_docs // 2]
    )
    if n_docs > 1:
        variance = sum((delta - mean_delta) ** 2 for delta in deltas) / (n_docs - 1)
        stdev_delta = variance**0.5
        sem_delta = stdev_delta / (n_docs**0.5)
    else:
        stdev_delta = 0.0
        sem_delta = 0.0

    bootstrap_values: list[float] = []
    requested_samples = max(0, int(bootstrap_samples))
    if requested_samples > 0:
        rng = random.Random(bootstrap_seed)
        for _ in range(requested_samples):
            sample_baseline_loss = 0.0
            sample_ttt_loss = 0.0
            sample_tokens = 0.0
            sample_bytes = 0.0
            for _doc in range(n_docs):
                record = records[rng.randrange(n_docs)]
                sample_baseline_loss += float(record["baseline_loss_sum"])
                sample_ttt_loss += float(record["ttt_loss_sum"])
                sample_tokens += float(record["token_count"])
                sample_bytes += float(record["byte_count"])
            sample_baseline_bpb = _bpb_from_loss_and_counts(sample_baseline_loss, sample_tokens, sample_bytes)
            sample_ttt_bpb = _bpb_from_loss_and_counts(sample_ttt_loss, sample_tokens, sample_bytes)
            bootstrap_values.append(sample_ttt_bpb - sample_baseline_bpb)

    if bootstrap_values:
        ordered = sorted(bootstrap_values)
        low_index = min(len(ordered) - 1, max(0, int(0.025 * len(ordered))))
        high_index = min(len(ordered) - 1, max(0, int(0.975 * len(ordered))))
        ci_low = ordered[low_index]
        ci_high = ordered[high_index]
        p_ge_zero = (sum(1 for value in bootstrap_values if value >= 0.0) + 1) / (len(bootstrap_values) + 1)
        p_le_zero = (sum(1 for value in bootstrap_values if value <= 0.0) + 1) / (len(bootstrap_values) + 1)
        p_two_sided = min(1.0, 2.0 * min(p_ge_zero, p_le_zero))
    else:
        ci_low = None
        ci_high = None
        p_two_sided = None

    improved = sum(1 for delta in deltas if delta < 0.0)
    worsened = sum(1 for delta in deltas if delta > 0.0)
    unchanged = n_docs - improved - worsened
    return {
        "doc_count": n_docs,
        "baseline_bpb": baseline_bpb,
        "ttt_bpb": ttt_bpb,
        "delta_bpb": ttt_bpb - baseline_bpb,
        "mean_doc_delta_bpb": mean_delta,
        "median_doc_delta_bpb": median_delta,
        "stdev_doc_delta_bpb": stdev_delta,
        "sem_doc_delta_bpb": sem_delta,
        "bootstrap_samples": len(bootstrap_values),
        "bootstrap_ci95_low_bpb": ci_low,
        "bootstrap_ci95_high_bpb": ci_high,
        "bootstrap_p_two_sided": p_two_sided,
        "improved_doc_count": improved,
        "worsened_doc_count": worsened,
        "unchanged_doc_count": unchanged,
    }


def _score_chunk(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    score_offset: int,
    *,
    device_type: str,
    autocast_enabled: bool,
    reduction: str,
) -> Tensor:
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
        logits = model.forward_logits(x)
        scored_logits = logits[:, score_offset : score_offset + y.size(1), :]
        return F.cross_entropy(
            scored_logits.reshape(-1, scored_logits.size(-1)).float(),
            y.reshape(-1),
            reduction=reduction,
        )


def eval_val_doc_chunks(
    args,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    *,
    bos_token_id: int,
) -> tuple[float, float, P20TTTStats]:
    if args.p20_ttt_chunk_size <= 0:
        raise ValueError(f"P20_TTT_CHUNK_SIZE must be positive, got {args.p20_ttt_chunk_size}")

    doc_ranges = _doc_ranges_from_bos(val_tokens, bos_token_id)
    if args.p20_ttt_max_docs > 0:
        doc_ranges = doc_ranges[: args.p20_ttt_max_docs]
    doc_start = (len(doc_ranges) * rank) // world_size
    doc_end = (len(doc_ranges) * (rank + 1)) // world_size
    local_doc_ranges = doc_ranges[doc_start:doc_end]

    was_training = model.training
    model.eval()
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    chunk_count = 0
    processed_docs = 0

    context_size = _resolve_context_size(args)
    autocast_enabled = device.type == "cuda"
    scan_chunk_size = int(getattr(args, "p20_scan_chunk_size", 0) or 0)
    with torch.no_grad():
        for doc_start_token, doc_end_token in local_doc_ranges:
            target_tokens = doc_end_token - doc_start_token - 1
            if target_tokens <= 0:
                continue
            processed_docs += 1
            position = 0
            while position < target_tokens:
                next_position = _next_chunk_end(
                    position,
                    target_tokens,
                    args.p20_ttt_chunk_size,
                    scan_chunk_size,
                )
                x, y, previous_ids, score_offset = _chunk_tensors(
                    val_tokens,
                    doc_start_token,
                    position,
                    next_position,
                    context_size,
                    device,
                )
                score_loss = _score_chunk(
                    model,
                    x,
                    y,
                    score_offset,
                    device_type=device.type,
                    autocast_enabled=autocast_enabled,
                    reduction="sum",
                )
                val_loss_sum += score_loss.to(torch.float64)
                val_token_count += float(y.numel())
                val_byte_count += _token_byte_count(
                    previous_ids,
                    y.reshape(-1),
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                chunk_count += 1
                position = next_position

    model.train(was_training)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
        counts = torch.tensor([processed_docs, chunk_count], device=device, dtype=torch.float64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        processed_docs = int(counts[0].item())
        chunk_count = int(counts[1].item())

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    stats = P20TTTStats(
        doc_count=processed_docs,
        chunk_count=chunk_count,
        update_count=0,
        token_count=float(val_token_count.item()),
        byte_count=float(val_byte_count.item()),
    )
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte), stats


def eval_val_p20_gate_ttt_paired(
    args,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    *,
    bos_token_id: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> P20TTTPairedResult:
    mode = normalize_p20_ttt_mode(args.p20_ttt_mode)
    if mode == "off":
        raise ValueError("eval_val_p20_gate_ttt_paired called with P20_TTT_MODE=off")
    if args.p20_ttt_chunk_size <= 0:
        raise ValueError(f"P20_TTT_CHUNK_SIZE must be positive, got {args.p20_ttt_chunk_size}")

    doc_ranges = _doc_ranges_from_bos(val_tokens, bos_token_id)
    if args.p20_ttt_max_docs > 0:
        doc_ranges = doc_ranges[: args.p20_ttt_max_docs]
    doc_start = (len(doc_ranges) * rank) // world_size
    doc_end = (len(doc_ranges) * (rank + 1)) // world_size
    local_doc_ranges = doc_ranges[doc_start:doc_end]

    was_training = model.training
    model.eval()
    base_params = list(model.parameters())
    base_requires_grad = [param.requires_grad for param in base_params]
    for param in base_params:
        param.requires_grad_(False)

    blocks: list[nn.Module] = []
    adapter_params: list[Tensor] = []
    records: list[dict[str, float | int]] = []
    baseline_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    ttt_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_count_sum = torch.zeros((), device=device, dtype=torch.float64)
    chunk_count = 0
    update_count = 0
    processed_docs = 0

    try:
        blocks, adapter_params = _set_p20_ttt_gate_params(model, mode, device)
        optimizer = torch.optim.Adam(
            adapter_params,
            lr=args.p20_ttt_lr,
            betas=(args.p20_ttt_beta1, args.p20_ttt_beta2),
            weight_decay=args.p20_ttt_weight_decay,
        )
        context_size = _resolve_context_size(args)
        autocast_enabled = device.type == "cuda"
        scan_chunk_size = int(getattr(args, "p20_scan_chunk_size", 0) or 0)

        for global_doc_index, (doc_start_token, doc_end_token) in enumerate(
            doc_ranges[doc_start:doc_end],
            start=doc_start,
        ):
            target_tokens = doc_end_token - doc_start_token - 1
            if target_tokens <= 0:
                continue
            processed_docs += 1
            doc_baseline_loss = 0.0
            doc_ttt_loss = 0.0
            doc_tokens = 0.0
            doc_bytes = 0.0
            doc_chunks = 0
            doc_updates = 0
            chunk_specs: list[tuple[int, int]] = []
            position = 0
            while position < target_tokens:
                next_position = _next_chunk_end(
                    position,
                    target_tokens,
                    args.p20_ttt_chunk_size,
                    scan_chunk_size,
                )
                chunk_specs.append((position, next_position))
                position = next_position

            _reset_adapter_params(adapter_params, optimizer)
            with torch.no_grad():
                for position, next_position in chunk_specs:
                    x, y, previous_ids, score_offset = _chunk_tensors(
                        val_tokens,
                        doc_start_token,
                        position,
                        next_position,
                        context_size,
                        device,
                    )
                    score_loss = _score_chunk(
                        model,
                        x,
                        y,
                        score_offset,
                        device_type=device.type,
                        autocast_enabled=autocast_enabled,
                        reduction="sum",
                    )
                    doc_baseline_loss += float(score_loss.item())
                    doc_tokens += float(y.numel())
                    doc_bytes += float(
                        _token_byte_count(
                            previous_ids,
                            y.reshape(-1),
                            base_bytes_lut,
                            has_leading_space_lut,
                            is_boundary_token_lut,
                        ).item()
                    )
                    doc_chunks += 1

            _reset_adapter_params(adapter_params, optimizer)
            for position, next_position in chunk_specs:
                x, y, _previous_ids, score_offset = _chunk_tensors(
                    val_tokens,
                    doc_start_token,
                    position,
                    next_position,
                    context_size,
                    device,
                )
                with torch.no_grad():
                    score_loss = _score_chunk(
                        model,
                        x,
                        y,
                        score_offset,
                        device_type=device.type,
                        autocast_enabled=autocast_enabled,
                        reduction="sum",
                    )
                doc_ttt_loss += float(score_loss.item())

                if next_position < target_tokens:
                    optimizer.zero_grad(set_to_none=True)
                    update_loss = _score_chunk(
                        model,
                        x,
                        y,
                        score_offset,
                        device_type=device.type,
                        autocast_enabled=autocast_enabled,
                        reduction="mean",
                    )
                    update_loss.backward()
                    if args.p20_ttt_grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(adapter_params, args.p20_ttt_grad_clip_norm)
                    optimizer.step()
                    doc_updates += 1

            baseline_loss_sum += doc_baseline_loss
            ttt_loss_sum += doc_ttt_loss
            token_count_sum += doc_tokens
            byte_count_sum += doc_bytes
            chunk_count += doc_chunks
            update_count += doc_updates
            baseline_bpb = _bpb_from_loss_and_counts(doc_baseline_loss, doc_tokens, doc_bytes)
            ttt_bpb = _bpb_from_loss_and_counts(doc_ttt_loss, doc_tokens, doc_bytes)
            records.append(
                {
                    "doc_index": global_doc_index,
                    "doc_start_token": doc_start_token,
                    "doc_end_token": doc_end_token,
                    "token_count": doc_tokens,
                    "byte_count": doc_bytes,
                    "chunk_count": doc_chunks,
                    "update_count": doc_updates,
                    "baseline_loss_sum": doc_baseline_loss,
                    "ttt_loss_sum": doc_ttt_loss,
                    "baseline_bpb": baseline_bpb,
                    "ttt_bpb": ttt_bpb,
                    "delta_bpb": ttt_bpb - baseline_bpb,
                }
            )
    finally:
        if blocks:
            _clear_p20_ttt_gate_params(blocks)
        _restore_requires_grad(base_params, base_requires_grad)
        model.train(was_training)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(baseline_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(ttt_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count_sum, op=dist.ReduceOp.SUM)
        counts = torch.tensor([processed_docs, chunk_count, update_count], device=device, dtype=torch.float64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        processed_docs = int(counts[0].item())
        chunk_count = int(counts[1].item())
        update_count = int(counts[2].item())
        gathered_records: list[list[dict[str, float | int]]] = [list() for _ in range(world_size)]
        dist.all_gather_object(gathered_records, records)
        records = [record for shard in gathered_records for record in shard]
        records.sort(key=lambda record: int(record["doc_index"]))

    baseline_val_loss = float((baseline_loss_sum / token_count_sum).item())
    ttt_val_loss = float((ttt_loss_sum / token_count_sum).item())
    baseline_val_bpb = _bpb_from_loss_and_counts(
        float(baseline_loss_sum.item()),
        float(token_count_sum.item()),
        float(byte_count_sum.item()),
    )
    ttt_val_bpb = _bpb_from_loss_and_counts(
        float(ttt_loss_sum.item()),
        float(token_count_sum.item()),
        float(byte_count_sum.item()),
    )
    baseline_stats = P20TTTStats(
        doc_count=processed_docs,
        chunk_count=chunk_count,
        update_count=0,
        token_count=float(token_count_sum.item()),
        byte_count=float(byte_count_sum.item()),
    )
    ttt_stats = P20TTTStats(
        doc_count=processed_docs,
        chunk_count=chunk_count,
        update_count=update_count,
        token_count=float(token_count_sum.item()),
        byte_count=float(byte_count_sum.item()),
    )
    paired_stats = _summarize_paired_records(
        records,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return P20TTTPairedResult(
        baseline_val_loss=baseline_val_loss,
        baseline_val_bpb=baseline_val_bpb,
        ttt_val_loss=ttt_val_loss,
        ttt_val_bpb=ttt_val_bpb,
        baseline_stats=baseline_stats,
        ttt_stats=ttt_stats,
        paired_stats=paired_stats,
        per_doc=records,
    )


def eval_val_p20_gate_ttt(
    args,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    *,
    bos_token_id: int,
) -> tuple[float, float, P20TTTStats]:
    mode = normalize_p20_ttt_mode(args.p20_ttt_mode)
    if mode == "off":
        raise ValueError("eval_val_p20_gate_ttt called with P20_TTT_MODE=off")
    if args.p20_ttt_chunk_size <= 0:
        raise ValueError(f"P20_TTT_CHUNK_SIZE must be positive, got {args.p20_ttt_chunk_size}")

    doc_ranges = _doc_ranges_from_bos(val_tokens, bos_token_id)
    if args.p20_ttt_max_docs > 0:
        doc_ranges = doc_ranges[: args.p20_ttt_max_docs]
    doc_start = (len(doc_ranges) * rank) // world_size
    doc_end = (len(doc_ranges) * (rank + 1)) // world_size
    local_doc_ranges = doc_ranges[doc_start:doc_end]

    was_training = model.training
    model.eval()
    base_params = list(model.parameters())
    base_requires_grad = [param.requires_grad for param in base_params]
    for param in base_params:
        param.requires_grad_(False)

    blocks: list[nn.Module] = []
    adapter_params: list[Tensor] = []
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    chunk_count = 0
    update_count = 0
    processed_docs = 0

    try:
        blocks, adapter_params = _set_p20_ttt_gate_params(model, mode, device)
        optimizer = torch.optim.Adam(
            adapter_params,
            lr=args.p20_ttt_lr,
            betas=(args.p20_ttt_beta1, args.p20_ttt_beta2),
            weight_decay=args.p20_ttt_weight_decay,
        )

        context_size = _resolve_context_size(args)
        autocast_enabled = device.type == "cuda"
        scan_chunk_size = int(getattr(args, "p20_scan_chunk_size", 0) or 0)
        for doc_start_token, doc_end_token in local_doc_ranges:
            _reset_adapter_params(adapter_params, optimizer)
            target_tokens = doc_end_token - doc_start_token - 1
            if target_tokens <= 0:
                continue
            processed_docs += 1
            position = 0
            while position < target_tokens:
                next_position = _next_chunk_end(
                    position,
                    target_tokens,
                    args.p20_ttt_chunk_size,
                    scan_chunk_size,
                )
                x, y, previous_ids, score_offset = _chunk_tensors(
                    val_tokens,
                    doc_start_token,
                    position,
                    next_position,
                    context_size,
                    device,
                )

                with torch.no_grad():
                    score_loss = _score_chunk(
                        model,
                        x,
                        y,
                        score_offset,
                        device_type=device.type,
                        autocast_enabled=autocast_enabled,
                        reduction="sum",
                    )
                val_loss_sum += score_loss.to(torch.float64)
                val_token_count += float(y.numel())
                val_byte_count += _token_byte_count(
                    previous_ids,
                    y.reshape(-1),
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                chunk_count += 1

                if next_position < target_tokens:
                    optimizer.zero_grad(set_to_none=True)
                    update_loss = _score_chunk(
                        model,
                        x,
                        y,
                        score_offset,
                        device_type=device.type,
                        autocast_enabled=autocast_enabled,
                        reduction="mean",
                    )
                    update_loss.backward()
                    if args.p20_ttt_grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(adapter_params, args.p20_ttt_grad_clip_norm)
                    optimizer.step()
                    update_count += 1

                position = next_position
    finally:
        if blocks:
            _clear_p20_ttt_gate_params(blocks)
        _restore_requires_grad(base_params, base_requires_grad)
        model.train(was_training)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
        counts = torch.tensor([processed_docs, chunk_count, update_count], device=device, dtype=torch.float64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        processed_docs = int(counts[0].item())
        chunk_count = int(counts[1].item())
        update_count = int(counts[2].item())

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    stats = P20TTTStats(
        doc_count=processed_docs,
        chunk_count=chunk_count,
        update_count=update_count,
        token_count=float(val_token_count.item()),
        byte_count=float(val_byte_count.item()),
    )
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte), stats
