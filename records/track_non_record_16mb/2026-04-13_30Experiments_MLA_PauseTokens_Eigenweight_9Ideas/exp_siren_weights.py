"""
Experiment: SIREN Weight Generator (Idea 4 - Developmental/CPPN)
A tiny SIREN MLP maps (row, col, layer) coordinates to weight values.
The SIREN IS the model — evaluate it at every coordinate to get all weights.

Core insight: Biology encodes a brain (~1PB connectivity) in a genome (~750MB).
The genome stores rules for generating connections from geometry, not the
connections themselves. SIREN's periodic activations excel at representing
structured signals like weight matrices.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # SIREN config
    siren_hidden = int(os.environ.get("SIREN_HIDDEN", 256))
    siren_layers = int(os.environ.get("SIREN_LAYERS", 4))
    siren_omega = float(os.environ.get("SIREN_OMEGA", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    siren_lr = float(os.environ.get("SIREN_LR", 1e-4))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))


# --- BPB eval helpers (same as other experiments) ---
def build_sentencepiece_luts(sp, vocab_size, device):
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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files for {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos+k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start+per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    model.train()
    return float(val_loss.item()), float(val_loss.item() / math.log(2.0) * val_token_count.item() / val_byte_count.item())


# --- SIREN Weight Generator ---
class SIRENLayer(nn.Module):
    """Single SIREN layer: linear + sin activation with omega scaling."""
    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SIRENWeightGenerator(nn.Module):
    """
    Maps (row_norm, col_norm, layer_norm, weight_type) -> weight_value.
    This tiny network IS the stored model. All transformer weights are
    generated by evaluating it at every coordinate.
    """
    def __init__(self, hidden_dim=256, num_layers=4, omega=30.0):
        super().__init__()
        # Input: (row_norm, col_norm, layer_norm, weight_type_embed) = 4 + 8 = 12 dims
        self.type_embed_dim = 8
        self.type_embedding = nn.Embedding(16, self.type_embed_dim)  # 16 weight types max
        input_dim = 3 + self.type_embed_dim  # row, col, layer + type embedding

        layers = [SIRENLayer(input_dim, hidden_dim, omega=omega, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega=omega))
        # Final layer: linear (no sin) to output weight value
        final = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords, weight_type_ids):
        """
        coords: [N, 3] — (row_norm, col_norm, layer_norm) in [0, 1]
        weight_type_ids: [N] — integer type ID for each coordinate
        Returns: [N] — weight values
        """
        type_emb = self.type_embedding(weight_type_ids)  # [N, type_embed_dim]
        x = torch.cat([coords, type_emb], dim=-1)  # [N, 3 + type_embed_dim]
        return self.net(x).squeeze(-1)  # [N]


def generate_weight_coords(out_features, in_features, layer_idx, num_layers, weight_type_id, device):
    """Generate normalized coordinates for every element in a weight matrix."""
    rows = torch.arange(out_features, device=device, dtype=torch.float32) / max(out_features - 1, 1)
    cols = torch.arange(in_features, device=device, dtype=torch.float32) / max(in_features - 1, 1)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
    n = out_features * in_features
    coords = torch.stack([
        row_grid.reshape(-1),
        col_grid.reshape(-1),
        torch.full((n,), layer_idx / max(num_layers - 1, 1), device=device),
    ], dim=-1)
    type_ids = torch.full((n,), weight_type_id, device=device, dtype=torch.long)
    return coords, type_ids


# Weight type IDs for different parts of the transformer
WEIGHT_TYPES = {
    'c_q': 0, 'c_k': 1, 'c_v': 2, 'proj': 3,
    'mlp_fc': 4, 'mlp_proj': 5,
}


class SIRENTransformer(nn.Module):
    """
    A transformer whose weights are generated by a SIREN network.
    The SIREN parameters are the ONLY stored parameters (plus embeddings and norms).
    """
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, siren_hidden, siren_layers, siren_omega):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.mlp_hidden = mlp_mult * model_dim
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.rope_base = rope_base

        # The SIREN weight generator — main stored parameters
        self.siren = SIRENWeightGenerator(siren_hidden, siren_layers, siren_omega)

        # These are stored directly (small, hard to generate from coordinates)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

        # Per-layer scalar parameters (small, stored directly)
        self.q_gains = nn.ParameterList([
            nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
            for _ in range(num_layers)
        ])
        self.attn_scales = nn.ParameterList([
            nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
            for _ in range(num_layers)
        ])
        self.mlp_scales = nn.ParameterList([
            nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
            for _ in range(num_layers)
        ])

        # RoPE
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute coordinate grids for each weight matrix (fixed, not stored)
        self._precompute_coords()

    def _precompute_coords(self):
        """Precompute all coordinate grids. These are deterministic buffers, not parameters."""
        self._weight_specs = []
        for layer_idx in range(self.num_layers):
            # Q projection: model_dim x model_dim
            self._weight_specs.append(('c_q', layer_idx, self.model_dim, self.model_dim))
            # K projection: kv_dim x model_dim
            self._weight_specs.append(('c_k', layer_idx, self.kv_dim, self.model_dim))
            # V projection: kv_dim x model_dim
            self._weight_specs.append(('c_v', layer_idx, self.kv_dim, self.model_dim))
            # Output projection: model_dim x model_dim
            self._weight_specs.append(('proj', layer_idx, self.model_dim, self.model_dim))
            # MLP fc: mlp_hidden x model_dim
            self._weight_specs.append(('mlp_fc', layer_idx, self.mlp_hidden, self.model_dim))
            # MLP proj: model_dim x mlp_hidden
            self._weight_specs.append(('mlp_proj', layer_idx, self.model_dim, self.mlp_hidden))

    def _generate_weight(self, name, layer_idx, out_features, in_features, device):
        """Generate a single weight matrix from the SIREN."""
        coords, type_ids = generate_weight_coords(
            out_features, in_features, layer_idx, self.num_layers,
            WEIGHT_TYPES[name], device
        )
        values = self.siren(coords, type_ids)
        return values.reshape(out_features, in_features)

    def _get_rope(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :].to(dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype)
        return cos, sin

    def forward(self, input_ids, target_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        dtype = torch.bfloat16

        # Embedding
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # RoPE
        cos, sin = self._get_rope(seqlen, device, dtype)

        # Generate all weights and run transformer
        for layer_idx in range(self.num_layers):
            # Generate weights for this layer
            W_q = self._generate_weight('c_q', layer_idx, self.model_dim, self.model_dim, device).to(dtype)
            W_k = self._generate_weight('c_k', layer_idx, self.kv_dim, self.model_dim, device).to(dtype)
            W_v = self._generate_weight('c_v', layer_idx, self.kv_dim, self.model_dim, device).to(dtype)
            W_o = self._generate_weight('proj', layer_idx, self.model_dim, self.model_dim, device).to(dtype)
            W_fc = self._generate_weight('mlp_fc', layer_idx, self.mlp_hidden, self.model_dim, device).to(dtype)
            W_proj = self._generate_weight('mlp_proj', layer_idx, self.model_dim, self.mlp_hidden, device).to(dtype)

            # Attention
            h = F.rms_norm(x, (x.size(-1),))
            q = F.linear(h, W_q).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = F.linear(h, W_k).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = F.linear(h, W_v).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

            # RoPE
            half = self.head_dim // 2
            q1, q2 = q[..., :half], q[..., half:]
            q = torch.cat((q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos), dim=-1)
            k1, k2 = k[..., :half], k[..., half:]
            k = torch.cat((k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos), dim=-1)

            q = q * self.q_gains[layer_idx].to(dtype=dtype)[None, :, None, None]

            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                        enable_gqa=(self.num_kv_heads != self.num_heads))
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, seqlen, self.model_dim)
            attn_out = F.linear(attn_out, W_o)

            x = x + self.attn_scales[layer_idx].to(dtype=dtype)[None, None, :] * attn_out

            # MLP (relu^2)
            h = F.rms_norm(x, (x.size(-1),))
            mlp_out = F.linear(torch.relu(F.linear(h, W_fc)).square(), W_proj)
            x = x + self.mlp_scales[layer_idx].to(dtype=dtype)[None, None, :] * mlp_out

        # Output
        x = F.rms_norm(x, (x.size(-1),)).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# --- MAIN ---
def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(f"=== SIREN WEIGHT GENERATOR EXPERIMENT === hidden={args.siren_hidden} layers={args.siren_layers} omega={args.siren_omega}")
    log0(code, console=False)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # NOTE: SIREN model is NOT compiled because weight generation is dynamic
    base_model = SIRENTransformer(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        siren_hidden=args.siren_hidden, siren_layers=args.siren_layers, siren_omega=args.siren_omega,
    ).to(device)

    siren_params = list(base_model.siren.parameters())
    other_params = [p for n, p in base_model.named_parameters() if not n.startswith('siren.')]

    n_siren = sum(p.numel() for p in siren_params)
    n_other = sum(p.numel() for p in other_params)
    n_total = n_siren + n_other
    log0(f"siren_params:{n_siren} other_params:{n_other} total_stored:{n_total}")

    # Count how many weight values the SIREN generates
    generated = 0
    for name, layer_idx, out_f, in_f in base_model._weight_specs:
        generated += out_f * in_f
    log0(f"generated_weights:{generated} expansion_ratio:{generated/max(n_siren,1):.1f}x")

    model = base_model  # No compile for SIREN (dynamic graph)

    # Optimizer: Adam for everything (Muon doesn't apply well to SIREN)
    optimizer = torch.optim.Adam([
        {"params": siren_params, "lr": args.siren_lr, "base_lr": args.siren_lr},
        {"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr},
        {"params": [p for p in other_params if p is not base_model.tok_emb.weight],
         "lr": args.scalar_lr, "base_lr": args.scalar_lr},
    ], betas=(args.beta1, args.beta2), eps=args.adam_eps)

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None: return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Training loop (no warmup — SIREN doesn't benefit from compile warmup)
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        optimizer.step()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    log0(f"Code size: {len(code.encode('utf-8'))} bytes")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
