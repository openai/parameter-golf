#!/usr/bin/env python3
"""
Parameter Golf training script (direct LM training, no distillation).

Run examples:
  Single GPU:
    python train_gpt.py --max_steps 2000 --batch_size 16

  Multi-GPU (Runpod / 8xH100):
    torchrun --standalone --nproc_per_node=8 train_gpt.py --max_steps 8000 --batch_size 8 --grad_accum 8
"""

import argparse
import glob
import json
import math
import os
import random
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
DEFAULT_VOCAB_SIZE = 1024


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
    vocab_size: int = DEFAULT_VOCAB_SIZE
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


def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = batch_size * seq_len
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(batch_size, seq_len)
        y = local[1:].reshape(batch_size, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


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


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def load_validation_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


@torch.no_grad()
def evaluate_bpb(
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    seq_len: int,
    val_batch_tokens: int,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    use_bf16: bool,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    local_batch_tokens = val_batch_tokens // world_size
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_TOKENS too small: {val_batch_tokens} for world_size={world_size}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    nll_nats_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
        batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
        inp = local[:-1].reshape(-1, seq_len)
        tgt = local[1:].reshape(-1, seq_len)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(inp)
        nll_nats = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="none",
        )
        nll_nats_sum += nll_nats.to(torch.float64).sum()
        token_count += float(tgt.numel())

        prev_ids = inp.reshape(-1)
        tgt_ids = tgt.reshape(-1)
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        byte_count += token_bytes.to(torch.float64).sum()

    if is_dist():
        dist.all_reduce(nll_nats_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (nll_nats_sum / token_count.clamp(min=1.0)).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = (token_count / byte_count.clamp(min=1.0)).item()
    bpb = bits_per_token * tokens_per_byte
    if was_training:
        model.train()
    return float(val_loss), float(bpb)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parameter Golf direct-training script")
    p.add_argument("--data_path", type=str, default="./data/datasets/fineweb10B_sp1024")
    p.add_argument(
        "--train_files_pattern",
        type=str,
        default="",
        help="Optional explicit glob for train token shards. Defaults to <data_path>/fineweb_train_*.bin",
    )
    p.add_argument(
        "--val_files_pattern",
        type=str,
        default="",
        help="Optional explicit glob for val token shards. Defaults to <data_path>/fineweb_val_*.bin",
    )
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default="./data/tokenizers/fineweb_1024_bpe.model",
        help="SentencePiece model used to compute tokenizer-agnostic BPB.",
    )
    p.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)

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
    p.add_argument(
        "--eval_every_updates",
        type=int,
        default=0,
        help="Deprecated: periodic validation is disabled; one full validation pass runs at the end.",
    )
    p.add_argument("--val_batch_tokens", type=int, default=524_288)
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
        vocab_size=args.vocab_size,
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

    train_pattern = args.train_files_pattern or os.path.join(args.data_path, "fineweb_train_*.bin")
    val_pattern = args.val_files_pattern or os.path.join(args.data_path, "fineweb_val_*.bin")
    train_loader = DistributedTokenLoader(train_pattern, rank=rank, world_size=world_size, device=device)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"--vocab_size={args.vocab_size} does not match tokenizer vocab size {int(sp.vocab_size())}"
        )
    val_tokens = load_validation_tokens(val_pattern, args.seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    if is_main_process():
        print(f"train_loader pattern={train_pattern}")
        print(f"val_loader pattern={val_pattern} tokens={val_tokens.numel() - 1}")

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

    log_loss_sum = 0.0
    log_updates = 0
    while global_step < target_global_step:
        elapsed = time.time() - start_time
        if elapsed >= time_limit_sec:
            if is_main_process():
                print(f"Stopping due to time limit at micro_step={global_step}, update_step={update_step}")
            break

        inp, tgt = train_loader.next_batch(batch_size=args.batch_size, seq_len=args.seq_len)

        use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss = loss / args.grad_accum
        with torch.no_grad():
            log_loss_sum += loss.item() * args.grad_accum

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
                        [log_loss_sum, float(log_updates)],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    loss_global = reduced[0].item()
                    updates_global = max(1.0, reduced[1].item())
                else:
                    loss_global = log_loss_sum
                    updates_global = float(max(1, log_updates))

                avg_token_loss = loss_global / max(1.0, updates_global)
                if is_main_process():
                    print(
                        f"u={update_step} g={global_step} "
                        f"loss={avg_token_loss:.4f} "
                        f"lr={lr:.6g} tok/step={tok_per_step}"
                    )
                log_loss_sum = 0.0
                log_updates = 0

    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    val_loss, val_bpb = evaluate_bpb(
        model=model,
        rank=rank,
        world_size=world_size,
        device=device,
        seq_len=args.seq_len,
        val_batch_tokens=args.val_batch_tokens,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        use_bf16=use_bf16,
    )
    if is_main_process():
        print(f"final_eval u={update_step} g={global_step} val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
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
            print(f"saved best checkpoint: {best_path} ({best_mb:.2f} MB, val_bpb={best_val_bpb:.4f})")

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
