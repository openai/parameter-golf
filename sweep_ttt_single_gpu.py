"""
TTT Hyperparameter Sweep — Single GPU
======================================
Loads a quantized .ptz artifact, dequantizes, then runs the TTT sliding-window
eval with a grid of hyperparameters. Reports per-chunk BPB trace + final BPB
for each configuration.

Usage (on pod/vast, single GPU):
    python sweep_ttt_single_gpu.py --ptz final_model.int6.ptz [--grid untested5]

Phases:
    untested5 — 5 targeted configs that haven't been tested yet (~25 min)
    phase1    — sweep max_train_chunks [20,30,40,50,60,80], EMA=0 (6 runs)
    phase2    — sweep lr [0.001,0.0015,0.002,0.003,0.005], EMA=0 (5 runs)
    phase3    — sweep epochs [1,2,3,5], EMA=0 (4 runs)

Reuses model code from train_gpt_v7_submit.py via import.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.nn.functional as F
from torch import Tensor

# Import model + utilities from the submission script
import train_gpt_v7_submit as base

@dataclass
class TTTConfig:
    lr: float = 0.002
    momentum: float = 0.9
    epochs: int = 3
    max_train_chunks: int = 200
    ema_decay: float = 0.0  # 0 = disabled
    freeze_blocks: int = 2
    freeze_embed: bool = True
    grad_clip: float = 1.0
    optim: str = "sgd"  # "sgd" or "adamw"

    def tag(self) -> str:
        return (f"lr{self.lr}_ep{self.epochs}_ch{self.max_train_chunks}"
                f"_ema{self.ema_decay}_{self.optim}")


def build_grid(phase: str, base_cfg: TTTConfig) -> list[TTTConfig]:
    """Build sweep grid for a given phase."""
    configs = []
    if phase == "untested5":
        # ============================================================
        # 5 UNTESTED configs — avoids duplicating prior experiments:
        #   - ShortTTT (chunks=50, EMA=0, freeze=0) => 1.1207
        #   - Baseline (chunks=200, EMA=0.995, freeze=2) => 1.1206
        # Key difference: all add freeze=2 (ShortTTT had freeze=0)
        # ============================================================
        # A: chunks=40, EMA=0, freeze=2 — shorter than ShortTTT, with freezing
        c = copy.copy(base_cfg); c.max_train_chunks = 40; c.ema_decay = 0.0; c.freeze_blocks = 2
        configs.append(c)
        # B: chunks=50, EMA=0, freeze=2 — same window as ShortTTT but WITH freezing
        c = copy.copy(base_cfg); c.max_train_chunks = 50; c.ema_decay = 0.0; c.freeze_blocks = 2
        configs.append(c)
        # C: chunks=40, EMA=0.9 (light), freeze=2 — smooth without washing
        c = copy.copy(base_cfg); c.max_train_chunks = 40; c.ema_decay = 0.9; c.freeze_blocks = 2
        configs.append(c)
        # D: chunks=30, EMA=0, freeze=2 — most val scored at peak
        c = copy.copy(base_cfg); c.max_train_chunks = 30; c.ema_decay = 0.0; c.freeze_blocks = 2
        configs.append(c)
        # E: chunks=50, EMA=0, freeze=3 — heavy freeze, maximum stability
        c = copy.copy(base_cfg); c.max_train_chunks = 50; c.ema_decay = 0.0; c.freeze_blocks = 3
        configs.append(c)
    elif phase == "phase1":
        # Sweep max_train_chunks, EMA off
        for chunks in [20, 30, 40, 50, 60, 80]:
            c = copy.copy(base_cfg)
            c.max_train_chunks = chunks
            c.ema_decay = 0.0
            configs.append(c)
    elif phase == "phase2":
        # Sweep LR (set max_train_chunks via env TTT_BEST_CHUNKS or default 40)
        best_chunks = int(os.environ.get("TTT_BEST_CHUNKS", "40"))
        for lr in [0.001, 0.0015, 0.002, 0.003, 0.005]:
            c = copy.copy(base_cfg)
            c.max_train_chunks = best_chunks
            c.ema_decay = 0.0
            c.lr = lr
            configs.append(c)
    elif phase == "phase3":
        # Sweep epochs
        best_chunks = int(os.environ.get("TTT_BEST_CHUNKS", "40"))
        best_lr = float(os.environ.get("TTT_BEST_LR", "0.002"))
        for ep in [1, 2, 3, 5]:
            c = copy.copy(base_cfg)
            c.max_train_chunks = best_chunks
            c.ema_decay = 0.0
            c.lr = best_lr
            c.epochs = ep
            configs.append(c)
    elif phase == "phase_adamw":
        # Same as phase1 but with AdamW
        for chunks in [20, 30, 40, 50, 60, 80]:
            c = copy.copy(base_cfg)
            c.max_train_chunks = chunks
            c.ema_decay = 0.0
            c.optim = "adamw"
            c.lr = 0.0001  # AdamW typically needs lower LR
            configs.append(c)
    elif phase == "all_quick":
        # Quick comparison: SGD vs AdamW at a few key points
        for optim, lr in [("sgd", 0.002), ("adamw", 0.0001), ("adamw", 0.0003)]:
            for chunks in [30, 40, 50]:
                c = copy.copy(base_cfg)
                c.max_train_chunks = chunks
                c.ema_decay = 0.0
                c.optim = optim
                c.lr = lr
                configs.append(c)
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return configs


def run_ttt_sweep_single(
    model: torch.nn.Module,
    initial_state: dict[str, Tensor],
    cfg: TTTConfig,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
    seq_len: int = 2048,
    ttt_chunk_tokens: int = 32768,
    stride: int = 64,
    batch_seqs: int = 32,
) -> dict:
    """Run a single TTT configuration, return results dict."""

    # Restore model to initial (dequantized) state
    model.load_state_dict(initial_state, strict=True)
    model.to(device)

    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk_tokens - 1) // ttt_chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        chunk_windows[min((ws + s) // ttt_chunk_tokens, num_chunks - 1)].append(ws)

    print(f"\n{'='*70}")
    print(f"TTT CONFIG: {cfg.tag()}")
    print(f"  chunks={num_chunks} windows={len(window_starts)} "
          f"lr={cfg.lr} epochs={cfg.epochs} freeze={cfg.freeze_blocks} "
          f"ema={cfg.ema_decay} optim={cfg.optim} max_train={cfg.max_train_chunks}")
    print(f"{'='*70}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze blocks
    frozen_ids = set(range(min(cfg.freeze_blocks, len(model.blocks))))
    embed_names = {"tok_emb", "bigram", "ve_shared"} if cfg.freeze_embed else set()
    ttt_params = []
    for name, p in model.named_parameters():
        if any(f"blocks.{bi}." in name for bi in frozen_ids):
            p.requires_grad_(False)
        elif any(en in name for en in embed_names):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    unfrozen_count = sum(p.numel() for p in ttt_params)
    print(f"  unfrozen={unfrozen_count} freeze_embed={cfg.freeze_embed}")

    # Optimizer
    if cfg.optim == "adamw":
        optimizer = torch.optim.AdamW(ttt_params, lr=cfg.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.SGD(ttt_params, lr=cfg.lr, momentum=cfg.momentum)

    # EMA setup
    ema_state = None
    raw_state = None
    if cfg.ema_decay > 0:
        ema_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        raw_state = {n: torch.empty_like(p.data) for n, p in model.named_parameters() if n in ema_state}
        print(f"  ema_decay={cfg.ema_decay}")

    t0 = time.perf_counter()
    cur_lr = cfg.lr
    chunk_bpbs = []  # per-chunk running BPB trace

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        # Swap to EMA for scoring if enabled
        if ema_state is not None and ci > 0:
            for n, p in model.named_parameters():
                if n in ema_state:
                    raw_state[n].copy_(p.data)
                    p.data.copy_(ema_state[n])

        # === SCORE this chunk (inference only) ===
        model.eval()
        with torch.inference_mode():
            for bi in range(0, len(windows), batch_seqs):
                batch_ws = windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    wlen = min(ws + seq_len, total_tokens) - ws
                    wlens.append(wlen)
                    ct = val_tokens[ws:ws + wlen + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = ct[:-1]
                    y_batch[i, :wlen] = ct[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    loss_sum += nll[i, s:wlen].to(torch.float64).sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = (base_bytes_lut[tgt].to(torch.float64)
                          + (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64))
                    byte_count += tb.sum()

        # Restore raw weights after scoring
        if ema_state is not None and ci > 0:
            for n, p in model.named_parameters():
                if n in raw_state:
                    p.data.copy_(raw_state[n])

        # Record running BPB
        if token_count.item() > 0:
            rl = loss_sum.item() / token_count.item()
            cur_bpb = rl / math.log(2) * (token_count.item() / max(byte_count.item(), 1))
            chunk_bpbs.append((ci + 1, cur_bpb))

        # === TRAIN on this chunk (score-first = legal) ===
        if ci < num_chunks - 1 and ci < cfg.max_train_chunks and cfg.epochs > 0:
            model.train()
            chunk_start = ci * ttt_chunk_tokens
            chunk_end = min((ci + 1) * ttt_chunk_tokens, total_tokens)
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cur_lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(cfg.max_train_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                for _ep in range(cfg.epochs):
                    for bs in range(0, chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, chunk_seqs)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = model(x, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(ttt_params, cfg.grad_clip)
                        optimizer.step()
                # EMA update
                if ema_state is not None:
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if n in ema_state:
                                ema_state[n].mul_(cfg.ema_decay).add_(p.data, alpha=1.0 - cfg.ema_decay)

        # Load EMA permanently when training stops
        if ema_state is not None and ci == cfg.max_train_chunks:
            print(f"  ttt:loading EMA weights permanently at chunk {ci}")
            for n, p in model.named_parameters():
                if n in ema_state:
                    p.data.copy_(ema_state[n])
            ema_state = None
            raw_state = None

        # Print progress every 5 chunks
        if ci % 5 == 0 or ci == num_chunks - 1:
            rl = loss_sum.item() / max(token_count.item(), 1)
            cur_bpb = rl / math.log(2) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0
            lr_str = f" lr={cur_lr:.6f}" if ci < cfg.max_train_chunks else " lr=done"
            elapsed = time.perf_counter() - t0
            print(f"  ttt[{ci+1}/{num_chunks}] bpb={cur_bpb:.6f}{lr_str} t={elapsed:.0f}s")

    # Restore all params to trainable for next run
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()

    elapsed = time.perf_counter() - t0
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    # Find best running BPB (the "floor")
    best_chunk, best_bpb = min(chunk_bpbs, key=lambda x: x[1])

    result = {
        "tag": cfg.tag(),
        "final_bpb": val_bpb,
        "final_loss": val_loss,
        "best_running_bpb": best_bpb,
        "best_at_chunk": best_chunk,
        "elapsed_s": elapsed,
        "config": {
            "lr": cfg.lr, "epochs": cfg.epochs,
            "max_train_chunks": cfg.max_train_chunks,
            "ema_decay": cfg.ema_decay, "optim": cfg.optim,
            "freeze_blocks": cfg.freeze_blocks,
        },
        "trace": [(c, round(b, 6)) for c, b in chunk_bpbs if c % 5 == 0 or c <= 10],
    }

    print(f"\n  RESULT: final_bpb={val_bpb:.6f}  best_running={best_bpb:.6f}@chunk{best_chunk}  time={elapsed:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="TTT Hyperparameter Sweep")
    parser.add_argument("--ptz", required=True, help="Path to quantized .ptz file")
    parser.add_argument("--grid", default="untested5", help="Sweep phase: untested5, phase1, phase2, phase3, phase_adamw, all_quick")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--output", default="sweep_ttt_results.json", help="Output JSON file")
    cli_args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model config from submission defaults
    args = base.Hyperparameters()

    # Load tokenizer + LUTs
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=cli_args.tokenizer)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load val tokens
    import glob as glob_mod
    val_pattern = os.path.join(cli_args.data_path, "fineweb_val_*.bin")
    val_tokens = base.load_validation_tokens(val_pattern, args.train_seq_len)
    print(f"val_tokens: {val_tokens.numel()-1:,} tokens")

    # Load quantized model
    print(f"Loading {cli_args.ptz}...")
    import zstandard
    with open(cli_args.ptz, "rb") as f:
        raw = f.read()
    quant_state = torch.load(
        __import__("io").BytesIO(zstandard.ZstdDecompressor().decompress(raw)),
        map_location="cpu",
    )

    # Build template model to get state dict shapes
    template_model = base.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    )
    sd_cpu = template_model.state_dict()

    # Dequantize
    deq_state = base.dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)

    # Build eval model
    eval_model = base.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, base.CastedLinear):
            m.float()
    base.restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    # Save initial state for resetting between runs
    initial_state = copy.deepcopy(eval_model.state_dict())

    # Don't compile — we reload weights each run so compilation cache won't help
    print(f"Model loaded. Running sweep: {cli_args.grid}")

    # Build sweep grid
    base_cfg = TTTConfig()
    grid = build_grid(cli_args.grid, base_cfg)
    print(f"Grid has {len(grid)} configurations")

    all_results = []
    for i, cfg in enumerate(grid):
        print(f"\n{'#'*70}")
        print(f"# RUN {i+1}/{len(grid)}")
        print(f"{'#'*70}")
        result = run_ttt_sweep_single(
            model=eval_model,
            initial_state=initial_state,
            cfg=cfg,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            seq_len=args.train_seq_len,
            stride=args.eval_stride if args.eval_stride > 0 else 64,
        )
        all_results.append(result)

        # Save after each run (in case we stop early)
        with open(cli_args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*90}")
    print(f"{'CONFIG':<45} {'FINAL BPB':>10} {'BEST BPB':>10} {'BEST@':>6} {'TIME':>6}")
    print(f"{'='*90}")
    for r in sorted(all_results, key=lambda x: x["final_bpb"]):
        print(f"{r['tag']:<45} {r['final_bpb']:>10.6f} {r['best_running_bpb']:>10.6f} "
              f"{r['best_at_chunk']:>6} {r['elapsed_s']:>5.0f}s")
    print(f"{'='*90}")

    best = min(all_results, key=lambda x: x["final_bpb"])
    print(f"\nBEST: {best['tag']}  final_bpb={best['final_bpb']:.6f}")
    print(f"Results saved to {cli_args.output}")


if __name__ == "__main__":
    main()
