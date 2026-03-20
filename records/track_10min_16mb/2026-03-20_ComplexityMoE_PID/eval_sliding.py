"""
Sliding Window Evaluation for Parameter Golf — Complexity-ML
Loads quantized model, evaluates with overlapping windows for better BPB.
Stride=64, window=2048 (configurable via env vars).
"""
from __future__ import annotations
import glob, io, math, os, sys, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try: import zstandard as zstd; USE_ZSTD = True
except ImportError: USE_ZSTD = False

# Import model + quantization from train script
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt import (
    Hyperparameters, GPT, build_sentencepiece_luts, load_data_shard,
    dequantize_state_dict_int8, restore_low_dim_params_to_fp32,
    CONTROL_TENSOR_NAME_PATTERNS, CastedLinear,
)

def sliding_window_eval(
    model: nn.Module,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    window: int = 2048,
    stride: int = 64,
    device: torch.device = torch.device("cuda"),
) -> tuple[float, float]:
    """Evaluate with overlapping sliding windows. Only score the last `stride` tokens of each window."""
    total_tokens = val_tokens.numel() - 1
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        pos = 0
        while pos < total_tokens:
            end = min(pos + window, total_tokens)
            start = max(end - window, 0)
            chunk = val_tokens[start:end + 1].to(device=device, dtype=torch.int64)
            x = chunk[:-1].unsqueeze(0)  # [1, win]
            y = chunk[1:].unsqueeze(0)   # [1, win]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_raw = model.tok_emb(x)
                # Full forward to get loss per token
                logits = _forward_logits(model, x)

            # Cross-entropy per token
            log_probs = F.log_softmax(logits.float(), dim=-1)  # [1, win, V]
            per_token_loss = -log_probs[0, :, :].gather(1, y[0, :].unsqueeze(-1)).squeeze(-1)  # [win]

            # Only score the last `stride` tokens (or all if first window)
            score_start = 0 if pos == 0 else (end - start) - stride
            score_end = end - start
            if score_start >= score_end:
                pos += stride
                continue

            scored_loss = per_token_loss[score_start:score_end]
            scored_x = x[0, score_start:score_end]
            scored_y = y[0, score_start:score_end]

            loss_sum += scored_loss.to(torch.float64).sum()
            token_count += scored_loss.numel()

            # BPB: byte count
            tok_bytes = base_bytes_lut[scored_y].to(torch.float64)
            if score_start > 0:
                prev_ids = x[0, score_start - 1:score_end - 1] if score_start > 0 else scored_x
            else:
                prev_ids = scored_x
            tok_bytes += (has_leading_space_lut[scored_y] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            byte_sum += tok_bytes.sum()

            pos += stride
            if pos == 0:
                pos = stride  # avoid infinite loop

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb


def _forward_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Forward pass returning raw logits (no loss computation)."""
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    expert_ids = model.token_to_expert[input_ids.clamp(0, model.vocab_size - 1)]
    vel = torch.zeros_like(x)
    skips = []
    for i in range(model.num_encoder_layers):
        x, vel = model.blocks[i](x, x0, vel, expert_ids)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        bi = model.num_encoder_layers + i
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x, vel = model.blocks[bi](x, x0, vel, expert_ids)
    x = model.final_norm(x)
    logits = F.linear(x, model.tok_emb.weight) if model.tie_embeddings else model.lm_head(x)
    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
    return logits


def main():
    args = Hyperparameters()
    window = int(os.environ.get("EVAL_WINDOW", "2048"))
    stride = int(os.environ.get("EVAL_STRIDE", "64"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Sliding Window Eval: window={window}, stride={stride}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load validation tokens
    files = [Path(p) for p in sorted(glob.glob(args.val_files))]
    val_tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    print(f"Val tokens: {val_tokens.numel():,}")

    # Build model
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        num_experts=args.num_experts, moe_activation=args.moe_activation,
        moe_routing=args.moe_routing,
        pid_alpha=args.pid_alpha, pid_beta=args.pid_beta, pid_gate=args.pid_gate,
        pid_dt=args.pid_dt, pid_mu_min=args.pid_mu_min, pid_mu_max=args.pid_mu_max,
        pid_velocity_max=args.pid_velocity_max,
    ).to(device)

    # Load quantized weights
    model_path = os.environ.get("MODEL_PATH", "final_model.int8.ptz")
    print(f"Loading: {model_path}")
    with open(model_path, "rb") as f:
        blob = f.read()
    raw = zstd.ZstdDecompressor().decompress(blob) if USE_ZSTD else zlib.decompress(blob)
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    model = model.to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(model)

    # Evaluate
    import time
    t0 = time.perf_counter()
    val_loss, val_bpb = sliding_window_eval(
        model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        window=window, stride=stride, device=device,
    )
    elapsed = time.perf_counter() - t0
    print(f"val_loss: {val_loss:.4f}")
    print(f"val_bpb:  {val_bpb:.4f}")
    print(f"eval_time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
