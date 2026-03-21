#!/usr/bin/env python3
"""
Diagnostic: compare sliding window BPB at different strides on first 20K tokens.
Should take ~30 seconds. Prints per-window loss/bytes to find the bug.
"""
from __future__ import annotations
import io, math, os, time, zlib
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

from train_gpt import (
    CastedLinear, GPT, Hyperparameters, Rotary,
    build_sentencepiece_luts, dequantize_state_dict_int8,
    load_validation_tokens, restore_low_dim_params_to_fp32,
)

try:
    import zstandard as zstd_mod
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


def main():
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = os.environ.get("CHECKPOINT", "final_model.int6.ptz")
    seq_len = args.train_seq_len  # 1024

    # Load model
    model = GPT(
        vocab_size=args.vocab_size,
        num_unique_blocks=args.num_unique_blocks,
        num_loops=args.num_loops,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_smear_gate=args.use_smear_gate,
        use_bigram_hash=args.use_bigram_hash,
        bigram_hash_buckets=args.bigram_hash_buckets,
        bigram_hash_dim=args.bigram_hash_dim,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)

    with open(checkpoint, "rb") as f:
        blob = f.read()
    try:
        if HAS_ZSTD:
            dctx = zstd_mod.ZstdDecompressor()
            raw = dctx.decompress(blob)
        else:
            raw = zlib.decompress(blob)
    except Exception:
        raw = zlib.decompress(blob)
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    state.pop("__correction_table__", None)
    state.pop("__correction_table_v2__", None)
    model.load_state_dict(dequantize_state_dict_int8(state), strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load val tokens and byte LUTs
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Test on first N tokens
    N = 20_000
    test_tokens = val_tokens[:N + 1]  # +1 for target of last position
    print(f"Testing on first {N:,} tokens\n")

    for stride in [1024, 256, 64]:
        print(f"{'='*60}")
        print(f"  STRIDE = {stride}")
        print(f"{'='*60}")

        total_loss = 0.0
        total_scored = 0
        total_bytes = 0.0
        starts = list(range(0, N - seq_len, stride))
        if not starts:
            starts = [0]

        with torch.inference_mode():
            for win_idx, start in enumerate(starts):
                end = start + seq_len + 1
                if end > len(test_tokens):
                    break

                chunk = test_tokens[start:end].to(device=device, dtype=torch.int64)
                x = chunk[:-1].reshape(1, seq_len)
                y = chunk[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model.forward_logits(x)

                score_start = 0 if start == 0 else seq_len - stride
                score_logits = logits[score_start:].float()
                score_targets = y[score_start:].to(device)

                # Shapes check
                n_scored = seq_len - score_start

                per_token_ce = F.cross_entropy(
                    score_logits, score_targets, reduction="sum"
                )

                # Byte counting
                prev_ids = x.reshape(-1)[score_start:]
                tgt_ids = score_targets
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (
                    has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                win_bytes = token_bytes.to(torch.float64).sum().item()

                total_loss += per_token_ce.item()
                total_scored += n_scored
                total_bytes += win_bytes

                # Print first 10 windows + every 20th after
                if win_idx < 10 or win_idx % 20 == 0:
                    avg_loss = per_token_ce.item() / n_scored
                    win_bpb = (avg_loss / math.log(2)) * (n_scored / win_bytes)
                    print(
                        f"  win={win_idx:4d} start={start:6d} "
                        f"score_start={score_start:4d} n_scored={n_scored:4d} "
                        f"logits_shape={list(score_logits.shape)} "
                        f"loss={per_token_ce.item():.2f} bytes={win_bytes:.0f} "
                        f"avg_loss={avg_loss:.4f} win_bpb={win_bpb:.4f}"
                    )

        # Final stats
        avg_loss = total_loss / total_scored
        bpb = (avg_loss / math.log(2)) * (total_scored / total_bytes)
        print(f"\n  TOTAL: scored={total_scored:,} bytes={total_bytes:.0f} "
              f"avg_loss={avg_loss:.4f} BPB={bpb:.4f}")
        print()


if __name__ == "__main__":
    main()
