#!/usr/bin/env python3
"""
Parameter Golf training script (direct LM training, no distillation).

Run examples:
  Single GPU:
    python param_golf.py --max_steps 2000 --batch_size 16

  Multi-GPU (Runpod / 8xH100):
    torchrun --standalone --nproc_per_node=8 param_golf.py --max_steps 8000 --batch_size 8 --grad_accum 8
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset


BYTE_VOCAB_SIZE = 257  # 0-255 byte values + EOS=256
EOS_TOKEN_ID = 256


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main_process() -> bool:
    return dist_rank() == 0


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        if world_size > 1:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class EmbeddingSemanticConv(nn.Module):
    """
    Convolutional compressor to approximate embedding semantics while shrinking
    channel dimension before attention.
    """

    def __init__(self, d_model: int, d_attn: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_model,
            bias=False,
        )
        self.pointwise = nn.Conv1d(d_model, d_attn, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        y = x.transpose(1, 2)  # [B, D, T]
        y = self.depthwise(y)
        y = self.pointwise(y)
        return y.transpose(1, 2)  # [B, T, D_attn]


class CausalCompressedAttention(nn.Module):
    def __init__(self, d_attn: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_attn % n_heads == 0, "d_attn must be divisible by n_heads"
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.head_dim = d_attn // n_heads
        self.q_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.k_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.v_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.o_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_attn)
        return self.o_proj(y)


class ResidualApproxNode(nn.Module):
    """
    One residual approximation node:
    conv-compress embeddings -> low-parameter attention -> expand -> residual MLP.
    """

    def __init__(
        self,
        d_model: int,
        d_attn: int,
        n_heads: int,
        ffn_hidden: int,
        conv_kernel_size: int,
        conv_dilation: int,
        dropout: float,
        residual_scale: float,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.pre_norm = RMSNorm(d_model)
        self.compressor = EmbeddingSemanticConv(d_model, d_attn, conv_kernel_size, conv_dilation)
        self.attn = CausalCompressedAttention(d_attn, n_heads, dropout)
        self.expand = nn.Linear(d_attn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_hidden, bias=False)
        self.ffn_out = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x)
        h = self.compressor(h)
        h = self.attn(h)
        h = self.expand(h)
        x = x + self.dropout(h) * self.residual_scale

        m = self.ffn_norm(x)
        m = self.ffn_in(m)
        m = F.silu(m)
        m = self.ffn_out(m)
        x = x + self.dropout(m) * self.residual_scale
        return x


class NodeFusion(nn.Module):
    """
    Multiple nodes + learned linear fusion (softmax-normalized scalar weights).
    """

    def __init__(self, nodes: list[nn.Module]):
        super().__init__()
        if len(nodes) < 1:
            raise ValueError("NodeFusion requires at least one node")
        self.nodes = nn.ModuleList(nodes)
        self.fusion_logits = nn.Parameter(torch.zeros(len(nodes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.fusion_logits, dim=0)
        out = None
        for i, node in enumerate(self.nodes):
            y = node(x)
            out = y.mul(weights[i]) if out is None else out.add(y.mul(weights[i]))
        return out


@dataclass
class ModelConfig:
    vocab_size: int = BYTE_VOCAB_SIZE
    seq_len: int = 512
    d_model: int = 384
    d_attn: int = 128
    n_heads: int = 4
    ffn_hidden: int = 512
    n_blocks: int = 3
    n_nodes: int = 4
    recurrence: int = 2
    dropout: float = 0.0


class ParameterGolfLM(nn.Module):
    """
    Parameter-efficient causal LM using multiple fused-node blocks.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.logit_scale = 1.0 / math.sqrt(cfg.d_model)
        nn.init.normal_(self.pos_emb, std=0.01)

        kernels = [3, 5, 7, 9]
        dilations = [1, 1, 2, 2]
        total_depth = max(1, cfg.n_blocks * cfg.recurrence)
        residual_scale = 1.0 / math.sqrt(2.0 * total_depth)
        self.blocks = nn.ModuleList()
        for block_idx in range(cfg.n_blocks):
            nodes: list[nn.Module] = []
            for node_idx in range(cfg.n_nodes):
                pattern_idx = (block_idx + node_idx) % len(kernels)
                nodes.append(
                    ResidualApproxNode(
                        d_model=cfg.d_model,
                        d_attn=cfg.d_attn,
                        n_heads=cfg.n_heads,
                        ffn_hidden=cfg.ffn_hidden,
                        conv_kernel_size=kernels[pattern_idx],
                        conv_dilation=dilations[pattern_idx],
                        dropout=cfg.dropout,
                        residual_scale=residual_scale,
                    )
                )
            self.blocks.append(NodeFusion(nodes))
        self.final_norm = RMSNorm(cfg.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        if seq_len > self.cfg.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model max {self.cfg.seq_len}")

        x = self.tok_emb(idx) + self.pos_emb[:, :seq_len, :]
        for block in self.blocks:
            for _ in range(self.cfg.recurrence):
                x = block(x)
        x = self.final_norm(x)
        # Weight tying: output projection uses token embedding weights.
        logits = F.linear(x, self.tok_emb.weight) * self.logit_scale
        return logits


class FineWebByteChunks(IterableDataset):
    """
    Streams FineWeb text and emits fixed-length byte token chunks.
    """

    def __init__(self, hf_iterable: Iterable[Dict], seq_len: int):
        super().__init__()
        self.hf_iterable = hf_iterable
        self.seq_len = seq_len

    def __iter__(self):
        buf: list[int] = []
        cursor = 0
        chunk_len = self.seq_len + 1
        trim_threshold = 1_000_000

        for ex in self.hf_iterable:
            text = ex.get("text", "")
            if not text:
                continue
            raw = text.encode("utf-8", errors="ignore")
            if not raw:
                continue
            buf.extend(raw)
            buf.append(EOS_TOKEN_ID)

            while (len(buf) - cursor) >= chunk_len:
                chunk = buf[cursor : cursor + chunk_len]
                cursor += chunk_len
                yield torch.tensor(chunk, dtype=torch.long)

            if cursor > trim_threshold:
                buf = buf[cursor:]
                cursor = 0


def collate_stack(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_fp16_artifact_mb(param_count: int) -> float:
    return (param_count * 2) / (1024 * 1024)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def bpb_stats_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (nll_bits_sum, byte_count) for tokenizer-agnostic BPB.
    In this byte-level setup, each non-EOS target token represents one byte.
    """
    nll_nats = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    byte_mask = (targets != EOS_TOKEN_ID).to(nll_nats.dtype)
    nll_bits_sum = (nll_nats * byte_mask).sum() / math.log(2.0)
    byte_count = byte_mask.sum()
    return nll_bits_sum, byte_count


@torch.no_grad()
def evaluate_bpb(
    model: nn.Module,
    data_iter: Iterable[torch.Tensor],
    device: torch.device,
    eval_batches: int,
    use_bf16: bool,
) -> float:
    was_training = model.training
    model.eval()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bits_sum = torch.zeros(1, device=device, dtype=torch.float64)
    bytes_sum = torch.zeros(1, device=device, dtype=torch.float64)

    for _ in range(eval_batches):
        batch = next(data_iter).to(device, non_blocking=True)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(inp)
        nll_bits, byte_count = bpb_stats_from_logits(logits.float(), tgt)
        bits_sum += nll_bits.to(torch.float64)
        bytes_sum += byte_count.to(torch.float64)

    if is_dist():
        dist.all_reduce(bits_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bytes_sum, op=dist.ReduceOp.SUM)

    bpb = (bits_sum / bytes_sum.clamp(min=1.0)).item()
    if was_training:
        model.train()
    return bpb


def maybe_load_fineweb(name: str, config_name: Optional[str], split: str, seed: int, shuffle_buffer: int, rank: int, world_size: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from exc

    kwargs = dict(split=split, streaming=True)
    if config_name:
        ds = load_dataset(name, name=config_name, **kwargs)
    else:
        ds = load_dataset(name, **kwargs)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)
    return ds


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parameter Golf direct-training script")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_config", type=str, default="sample-10BT")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--shuffle_buffer", type=int, default=10000)

    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--d_attn", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ffn_hidden", type=int, default=512)
    p.add_argument("--n_blocks", type=int, default=3)
    p.add_argument("--n_nodes", type=int, default=4)
    p.add_argument("--recurrence", type=int, default=2, help="Number of repeated passes per block.")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=8000, help="Micro-steps (before grad_accum reduction).")
    p.add_argument("--additional_steps", type=int, default=0, help="If >0, run this many extra micro-steps from current/resumed step.")
    p.add_argument("--time_limit_minutes", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every_updates", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--save_best", action="store_true", default=True, help="Save checkpoint when val_bpb improves.")
    p.add_argument("--no_save_best", action="store_false", dest="save_best")

    p.add_argument("--out_path", type=str, default="param_golf_ckpt.pt")
    p.add_argument("--resume_path", type=str, default="", help="Path to a saved checkpoint (.pt) to resume model weights/steps from.")
    p.add_argument("--resume_reset_steps", action="store_true", help="When resuming, ignore saved step counters.")
    p.add_argument("--strict_16mb", action="store_true", help="Fail if saved artifact exceeds 16MB.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    seed_everything(args.seed + rank)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = ModelConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        seq_len=args.seq_len,
        d_model=args.d_model,
        d_attn=args.d_attn,
        n_heads=args.n_heads,
        ffn_hidden=args.ffn_hidden,
        n_blocks=args.n_blocks,
        n_nodes=args.n_nodes,
        recurrence=args.recurrence,
        dropout=args.dropout,
    )
    model = ParameterGolfLM(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    param_count = count_parameters(model.module if isinstance(model, DDP) else model)
    est_fp16_mb = estimate_fp16_artifact_mb(param_count)
    if is_main_process():
        print(f"params={param_count:,} est_fp16_artifact={est_fp16_mb:.2f}MB")

    train_stream = maybe_load_fineweb(
        args.dataset_name,
        args.dataset_config if args.dataset_config else None,
        args.train_split,
        args.seed + rank,
        args.shuffle_buffer,
        rank,
        world_size,
    )
    train_ds = FineWebByteChunks(train_stream, seq_len=args.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_stack,
    )
    val_iter = None
    if args.eval_every_updates > 0 and args.eval_batches > 0:
        try:
            val_stream = maybe_load_fineweb(
                args.dataset_name,
                args.dataset_config if args.dataset_config else None,
                args.val_split,
                args.seed + 11_111 + rank,
                max(1000, args.shuffle_buffer // 10),
                rank,
                world_size,
            )
            val_ds = FineWebByteChunks(val_stream, seq_len=args.seq_len)
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                collate_fn=collate_stack,
            )
            val_iter = iter(val_loader)
        except Exception as exc:
            try:
                if is_main_process():
                    print(f"Validation split='{args.val_split}' unavailable ({exc}); falling back to train split.")
                val_stream = maybe_load_fineweb(
                    args.dataset_name,
                    args.dataset_config if args.dataset_config else None,
                    args.train_split,
                    args.seed + 22_222 + rank,
                    max(1000, args.shuffle_buffer // 10),
                    rank,
                    world_size,
                )
                val_ds = FineWebByteChunks(val_stream, seq_len=args.seq_len)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    num_workers=0,
                    pin_memory=(device.type == "cuda"),
                    collate_fn=collate_stack,
                )
                val_iter = iter(val_loader)
            except Exception as exc2:
                if is_main_process():
                    print(f"Validation disabled after fallback attempt: {exc2}")

    fused_ok = device.type == "cuda"
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=fused_ok,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    global_step = 0
    update_step = 0
    best_val_bpb = float("inf")
    best_update_step = 0
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location="cpu")
        ckpt_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        target_model = model.module if isinstance(model, DDP) else model
        missing, unexpected = target_model.load_state_dict(ckpt_sd, strict=False)
        if (not args.resume_reset_steps) and isinstance(ckpt, dict):
            global_step = int(ckpt.get("global_step", 0))
            update_step = int(ckpt.get("update_step", 0))
            best_val_bpb = float(ckpt.get("best_val_bpb", float("inf")))
            best_update_step = int(ckpt.get("best_update_step", 0))
        if is_main_process():
            print(
                f"resumed from {args.resume_path} "
                f"(missing={len(missing)} unexpected={len(unexpected)} "
                f"global_step={global_step} update_step={update_step})"
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_global_step = global_step
    if args.additional_steps > 0:
        target_global_step = global_step + args.additional_steps
    else:
        target_global_step = args.max_steps
    run_update_budget = max(1, (max(0, target_global_step - start_global_step)) // max(1, args.grad_accum))
    run_update_step = 0

    if is_main_process():
        print(
            f"training window: start_g={start_global_step} target_g={target_global_step} "
            f"run_update_budget={run_update_budget}"
        )

    start_time = time.time()
    time_limit_sec = args.time_limit_minutes * 60.0

    # Dataloader for iterable datasets does not naturally end; stop by max_steps/time.
    data_iter = iter(train_loader)
    log_bits_sum = 0.0
    log_bytes_sum = 0.0
    log_token_loss_sum = 0.0
    log_updates = 0
    while global_step < target_global_step:
        elapsed = time.time() - start_time
        if elapsed >= time_limit_sec:
            if is_main_process():
                print(f"Stopping due to time limit at micro_step={global_step}, update_step={update_step}")
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = batch.to(device, non_blocking=True)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]

        use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss = loss / args.grad_accum
        with torch.no_grad():
            nll_bits, byte_count = bpb_stats_from_logits(logits.float(), tgt)
            log_bits_sum += nll_bits.item()
            log_bytes_sum += byte_count.item()
            log_token_loss_sum += (loss.item() * args.grad_accum)

        loss.backward()
        global_step += 1

        if global_step % args.grad_accum == 0:
            lr = cosine_lr(run_update_step, run_update_budget, args.warmup_steps, args.lr, args.min_lr)
            set_lr(optimizer, lr)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            run_update_step += 1
            log_updates += 1

            should_log = (update_step % args.log_every == 0 or update_step == 1)
            if should_log:
                tok_per_step = args.batch_size * args.seq_len * args.grad_accum * world_size
                if is_dist():
                    reduced = torch.tensor(
                        [log_bits_sum, log_bytes_sum, log_token_loss_sum, float(log_updates)],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    bits_global = reduced[0].item()
                    bytes_global = reduced[1].item()
                    token_loss_global = reduced[2].item()
                    updates_global = max(1.0, reduced[3].item())
                else:
                    bits_global = log_bits_sum
                    bytes_global = log_bytes_sum
                    token_loss_global = log_token_loss_sum
                    updates_global = float(max(1, log_updates))

                avg_token_loss = token_loss_global / max(1.0, updates_global * args.grad_accum)
                avg_train_bpb = bits_global / max(1.0, bytes_global)
                if is_main_process():
                    print(
                        f"u={update_step} g={global_step} "
                        f"loss={avg_token_loss:.4f} train_bpb={avg_train_bpb:.4f} "
                        f"lr={lr:.6g} tok/step={tok_per_step}"
                    )
                log_bits_sum = 0.0
                log_bytes_sum = 0.0
                log_token_loss_sum = 0.0
                log_updates = 0

            if val_iter is not None and (update_step % args.eval_every_updates == 0):
                try:
                    val_bpb = evaluate_bpb(
                        model=model,
                        data_iter=val_iter,
                        device=device,
                        eval_batches=args.eval_batches,
                        use_bf16=use_bf16,
                    )
                    if is_main_process():
                        print(f"eval u={update_step} val_bpb={val_bpb:.4f} batches={args.eval_batches}")
                        if args.save_best and (val_bpb < best_val_bpb):
                            best_val_bpb = val_bpb
                            best_update_step = update_step
                            best_model = model.module if isinstance(model, DDP) else model
                            best_path = f"{args.out_path}.best.pt"
                            best_obj = {
                                "model_config": asdict(cfg),
                                "state_dict": {k: v.detach().cpu().half() for k, v in best_model.state_dict().items()},
                                "global_step": global_step,
                                "update_step": update_step,
                                "best_val_bpb": best_val_bpb,
                                "args": vars(args),
                            }
                            torch.save(best_obj, best_path)
                            best_mb = os.path.getsize(best_path) / (1024 * 1024)
                            print(
                                f"saved best checkpoint: {best_path} ({best_mb:.2f} MB, val_bpb={best_val_bpb:.4f})"
                            )
                except StopIteration:
                    # Recreate iterator if backend exhausts unexpectedly.
                    val_iter = iter(val_loader)  # type: ignore[name-defined]

    if is_dist():
        dist.barrier()

    unwrapped = model.module if isinstance(model, DDP) else model
    if is_main_process():
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        save_obj = {
            "model_config": asdict(cfg),
            "state_dict": {k: v.detach().cpu().half() for k, v in unwrapped.state_dict().items()},
            "global_step": global_step,
            "update_step": update_step,
            "args": vars(args),
        }
        torch.save(save_obj, args.out_path)
        artifact_mb = os.path.getsize(args.out_path) / (1024 * 1024)
        print(f"Saved checkpoint: {args.out_path} ({artifact_mb:.2f} MB)")
        if args.strict_16mb and artifact_mb > 16.0:
            raise RuntimeError(f"Artifact is {artifact_mb:.2f}MB, exceeds 16MB limit.")

        meta_path = f"{args.out_path}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "param_count": param_count,
                    "estimated_fp16_mb": est_fp16_mb,
                    "actual_ckpt_mb": artifact_mb,
                    "global_step": global_step,
                    "update_step": update_step,
                    "best_val_bpb": None if best_val_bpb == float("inf") else best_val_bpb,
                    "best_update_step": best_update_step,
                },
                f,
                indent=2,
            )
        print(f"Wrote metadata: {meta_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
