from __future__ import annotations

import glob
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class Config:
    # Required paths / vocab.
    vocab_size: int = 1024
    dataset_dir: Path = Path("./parameter-golf/data/datasets/fineweb10B_sp1024")
    tokenizer_path: Path = Path("./parameter-golf/data/tokenizers/fineweb_1024_bpe.model")

    # Requested model shape.
    model_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    mlp_mult: int = 3

    # Required training geometry.
    batch_tokens: int = 4096
    seq_len: int = 512

    # Requested training schedule.
    final_step: int = 300
    ternary_enable_step: int = 251
    pre_ternary_eval_step: int = 250
    post_ternary_eval_step: int = 300

    # Stability defaults for smoke test.
    seed: int = 1337
    lr: float = 3e-4
    weight_decay: float = 0.01

    # Keep validation bounded for a quick smoke run.
    eval_max_seqs: int = 128
    eval_batch_seqs: int = 8


CONFIG = Config()


def load_data_shard(file: Path) -> Tensor:
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


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
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


def load_validation_tokens(pattern: str, seq_len: int, max_seqs: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    target_tokens = max_seqs * seq_len + 1
    chunks: list[Tensor] = []
    loaded = 0
    for file in files:
        shard = load_data_shard(file)
        chunks.append(shard)
        loaded += shard.numel()
        if loaded >= target_tokens:
            break
    tokens = torch.cat(chunks).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    usable = min(usable, max_seqs * seq_len)
    return tokens[: usable + 1]


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

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
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


class SingleGpuTokenLoader:
    def __init__(self, pattern: str, device: torch.device):
        self.stream = TokenStream(pattern)
        self.device = device

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        if batch_tokens % seq_len != 0:
            raise ValueError(f"batch_tokens must be divisible by seq_len, got {batch_tokens}/{seq_len}")
        local = self.stream.take(batch_tokens + 1).to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True)
        y = local[1:].reshape(-1, seq_len).to(self.device, non_blocking=True)
        return x, y


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class TernaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("_ternary_enabled", torch.tensor(0.0), persistent=False)

    def set_ternary_enabled(self, enabled: bool) -> None:
        self._ternary_enabled.fill_(1.0 if enabled else 0.0)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        alpha = weight.abs().mean(dim=1, keepdim=True)
        w_q = (weight / (alpha + 1e-8)).round().clamp(-1, 1)
        w_ste = weight + (w_q - weight).detach()
        ternary_weight = w_ste * alpha
        enabled = self._ternary_enabled.to(dtype=weight.dtype)
        effective_weight = weight + enabled * (ternary_weight - weight)
        return F.linear(x, effective_weight, self.bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = TernaryLinear(dim, hidden, bias=False)
        self.proj = TernaryLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def enable_ternary(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            module.set_ternary_enabled(True)
            count += 1
    return count


def evaluate_bpb(
    model: nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    eval_batch_seqs: int,
    max_eval_seqs: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    total_seqs = (val_tokens.numel() - 1) // seq_len
    eval_seqs = min(total_seqs, max_eval_seqs)
    if eval_seqs <= 0:
        raise ValueError("No validation sequences available")

    val_loss_sum = 0.0
    val_token_count = 0.0
    val_byte_count = 0.0

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(0, eval_seqs, eval_batch_seqs):
            batch_seq_end = min(batch_seq_start + eval_batch_seqs, eval_seqs)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += float(batch_loss.item()) * batch_token_count
            val_token_count += batch_token_count

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += float(token_bytes.to(torch.float32).sum().item())

    model.train()
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / val_byte_count
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def main() -> None:
    cfg = CONFIG

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    train_glob = str(cfg.dataset_dir / "fineweb_train_*.bin")
    val_glob = str(cfg.dataset_dir / "fineweb_val_*.bin")
    if not cfg.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {cfg.dataset_dir}")
    if not cfg.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {cfg.tokenizer_path}")

    train_files = sorted(glob.glob(train_glob))
    val_files = sorted(glob.glob(val_glob))
    if not train_files:
        raise FileNotFoundError(f"No train shards found for pattern: {train_glob}")
    if not val_files:
        raise FileNotFoundError(f"No val shards found for pattern: {val_glob}")

    sp = spm.SentencePieceProcessor(model_file=str(cfg.tokenizer_path))
    if int(sp.vocab_size()) != cfg.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={cfg.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, cfg.vocab_size, device
    )

    val_tokens = load_validation_tokens(
        val_glob,
        seq_len=cfg.seq_len,
        max_seqs=cfg.eval_max_seqs,
    )
    train_loader = SingleGpuTokenLoader(train_glob, device=device)

    base_model = GPT(
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        mlp_mult=cfg.mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    ).to(device)
    compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "aot_eager")
    model = torch.compile(base_model, dynamic=False, fullgraph=True, backend=compile_backend)
    print(f"compile_backend:{compile_backend}")
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    print(
        "config:"
        f" model_dim={cfg.model_dim} num_layers={cfg.num_layers} num_heads={cfg.num_heads}"
        f" num_kv_heads={cfg.num_kv_heads} mlp_mult={cfg.mlp_mult}"
        f" batch_tokens={cfg.batch_tokens} seq_len={cfg.seq_len}"
    )
    print(f"train_shards={len(train_files)} val_shards={len(val_files)}")

    first_train_loss: float | None = None
    final_train_loss: float | None = None
    val_bpb_before = math.nan
    val_bpb_after = math.nan
    diverged = False
    ternary_enabled_layers = 0

    for step in range(cfg.final_step + 1):
        if step == cfg.ternary_enable_step:
            ternary_enabled_layers = enable_ternary(model)
            if ternary_enabled_layers == 0:
                ternary_enabled_layers = enable_ternary(base_model)
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            print(
                f"step:{step} ternary_enabled=True ternary_layers:{ternary_enabled_layers} "
                f"lr:{optimizer.param_groups[0]['lr']:.6g}"
            )

        optimizer.zero_grad(set_to_none=True)
        x, y = train_loader.next_batch(cfg.batch_tokens, cfg.seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        loss_value = float(loss.detach().item())
        if first_train_loss is None:
            first_train_loss = loss_value
        final_train_loss = loss_value

        if not math.isfinite(loss_value):
            diverged = True
            print(f"step:{step} train_loss:NaN_or_Inf")
            break

        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step in (cfg.pre_ternary_eval_step, cfg.post_ternary_eval_step):
            print(f"step:{step} train_loss:{loss_value:.6f}")

        if step == cfg.pre_ternary_eval_step:
            _, val_bpb_before = evaluate_bpb(
                model=model,
                val_tokens=val_tokens,
                seq_len=cfg.seq_len,
                eval_batch_seqs=cfg.eval_batch_seqs,
                max_eval_seqs=cfg.eval_max_seqs,
                device=device,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            if not math.isfinite(val_bpb_before):
                diverged = True
            print(f"step:{step} val_bpb:{val_bpb_before:.6f}")

        if step == cfg.post_ternary_eval_step:
            _, val_bpb_after = evaluate_bpb(
                model=model,
                val_tokens=val_tokens,
                seq_len=cfg.seq_len,
                eval_batch_seqs=cfg.eval_batch_seqs,
                max_eval_seqs=cfg.eval_max_seqs,
                device=device,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            if not math.isfinite(val_bpb_after):
                diverged = True
            print(f"step:{step} val_bpb:{val_bpb_after:.6f}")

    if first_train_loss is None or final_train_loss is None:
        raise RuntimeError("No training steps executed.")

    loss_went_down = final_train_loss < first_train_loss
    ternary_gap = val_bpb_after - val_bpb_before
    ternary_gap_abs = abs(ternary_gap) if math.isfinite(ternary_gap) else math.nan
    ternary_within_threshold = math.isfinite(ternary_gap_abs) and ternary_gap_abs <= 0.1

    fail_reasons: list[str] = []
    if not loss_went_down:
        fail_reasons.append(
            f"loss did not decrease overall (start={first_train_loss:.6f}, end={final_train_loss:.6f})"
        )
    if diverged:
        fail_reasons.append("training diverged (NaN/Inf detected)")
    if not ternary_within_threshold:
        if not math.isfinite(ternary_gap):
            fail_reasons.append("ternary bpb comparison unavailable (non-finite pre/post value)")
        else:
            fail_reasons.append(
                f"ternary bpb drift too high (abs_delta={ternary_gap_abs:.6f}, allowed<=0.100000)"
            )

    passed = len(fail_reasons) == 0
    status = "PASS" if passed else "FAIL"

    print(f"val_bpb BEFORE ternary enabled (step 250): {val_bpb_before:.6f}")
    print(f"val_bpb AFTER ternary enabled (step 300): {val_bpb_after:.6f}")
    print(f"training diverged (NaN check): {diverged}")
    if passed:
        print("PASS: loss went down overall, no NaN, and ternary bpb is within 0.1 of pre-ternary bpb.")
    else:
        print(f"FAIL: {'; '.join(fail_reasons)}")

    results = {
        "status": status,
        "pass": passed,
        "model_shape": {
            "model_dim": cfg.model_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "mlp_mult": cfg.mlp_mult,
            "vocab_size": cfg.vocab_size,
        },
        "schedule": {
            "normal_steps": [0, 250],
            "ternary_steps": [251, 300],
            "ternary_enable_step": cfg.ternary_enable_step,
            "lr_reduction_factor_on_enable": 0.5,
            "ternary_layers_enabled": ternary_enabled_layers,
        },
        "metrics": {
            "first_train_loss": first_train_loss,
            "final_train_loss": final_train_loss,
            "loss_went_down_overall": loss_went_down,
            "val_bpb_before_step_250": val_bpb_before,
            "val_bpb_after_step_300": val_bpb_after,
            "ternary_bpb_delta": ternary_gap,
            "ternary_bpb_abs_delta": ternary_gap_abs,
            "diverged": diverged,
        },
        "fail_reasons": fail_reasons,
    }

    results_path = Path("./results/experiment_a.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved_results:{results_path.resolve()}")


if __name__ == "__main__":
    main()
