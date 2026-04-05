#!/usr/bin/env python3
"""Score training sequences by difficulty and rewrite shards in sorted order.

Scores every seq_len-token sequence using a reference model's cross-entropy loss,
sorts by difficulty (easy first), and rewrites the shard binary files in sorted order.
The resulting data directory can be used with the standard training script by setting
DATA_PATH to the output directory — no loader code changes needed.

Usage (CUDA):
  python scripts/score_and_reorder_data.py \\
    --checkpoint final_model.pt \\
    --data-dir data/datasets/fineweb10B_sp1024 \\
    --output-dir data/datasets/fineweb10B_sp1024_curriculum \\
    --seq-len 1024

References:
  - Rho-1 (arXiv 2404.07965, NeurIPS 2024) — token-level selection via reference model
  - Beyond Random Sampling (arXiv 2506.11300, 2025) — easy-to-hard curriculum, 18-40% step reduction
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

# Reuse data loading from the training script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def load_data_shard(file):
    """Load a binary shard file (same format as train_gpt.py)."""
    header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = HEADER_INTS * np.dtype("<i4").itemsize
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


def write_data_shard(file, tokens_np):
    """Write a binary shard file (same format as train_gpt.py)."""
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens_np)
    with open(file, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.astype("<u2").tobytes())


def score_sequences(model, tokens, seq_len, device, batch_size=16):
    """Score all non-overlapping sequences in a token array."""
    num_seqs = (tokens.numel() - 1) // seq_len
    scores = []
    for batch_start in range(0, num_seqs, batch_size):
        batch_end = min(batch_start + batch_size, num_seqs)
        xs, ys = [], []
        for i in range(batch_start, batch_end):
            offset = i * seq_len
            xs.append(tokens[offset: offset + seq_len])
            ys.append(tokens[offset + 1: offset + seq_len + 1])
        x = torch.stack(xs).to(device=device, dtype=torch.int64)
        y = torch.stack(ys).to(device=device, dtype=torch.int64)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # Per-sequence mean loss
            per_pos = torch.nn.functional.cross_entropy(
                model_forward_logits(model, x), y.reshape(-1), reduction="none"
            ).reshape(x.shape[0], -1)
            seq_losses = per_pos.mean(dim=1)
        for i, loss_val in enumerate(seq_losses.cpu().tolist()):
            scores.append((batch_start + i, loss_val))
    return scores


def model_forward_logits(model, x):
    """Run model forward and return logits (not loss).

    This is a simplified version that works with our GPT class.
    We need logits for per-sequence scoring, but GPT.forward() returns loss.
    So we extract the logit computation inline.
    """
    import torch.nn.functional as F
    # Run through the model's layers manually to get logits
    # This avoids modifying GPT.forward() just for scoring
    with torch.no_grad():
        emb = model.tok_emb(x)
        if hasattr(model, 'bigram') and model.bigram is not None:
            emb = emb + model.bigram(x)
        h = F.rms_norm(emb, (emb.size(-1),))
        x0 = h
        v0 = None
        skips = []
        for i in range(model.num_encoder_layers):
            h, raw_v = model.blocks[i](h, x0, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(h)
        for i in range(model.num_decoder_layers):
            if skips:
                h = h + model.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips.pop()
            h, _ = model.blocks[model.num_encoder_layers + i](h, x0, v0=v0)
        h = model.final_norm(h).reshape(-1, h.size(-1))
        if model.tie_embeddings:
            logits = F.linear(h, model.tok_emb.weight)
        else:
            logits = model.lm_head(h)
        logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
    return logits


def main():
    parser = argparse.ArgumentParser(description="Score and reorder training data by difficulty")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir", required=True, help="Input data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for reordered shards")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16, help="Sequences per scoring batch")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    # Import GPT class from training script
    import importlib.util
    spec = importlib.util.spec_from_file_location("r2", "train_gpt_r2.py")
    r2 = importlib.util.module_from_spec(spec)
    r2.__name__ = "r2"
    spec.loader.exec_module(r2)

    # Build model with default hyperparameters, load checkpoint
    hyper = r2.Hyperparameters()
    model = r2.GPT(
        vocab_size=hyper.vocab_size, num_layers=hyper.num_layers,
        model_dim=hyper.model_dim, num_heads=hyper.num_heads,
        num_kv_heads=hyper.num_kv_heads, mlp_mult=hyper.mlp_mult,
        tie_embeddings=hyper.tie_embeddings, tied_embed_init_std=hyper.tied_embed_init_std,
        logit_softcap=hyper.logit_softcap, rope_base=hyper.rope_base,
        qk_gain_init=hyper.qk_gain_init,
        bigram_vocab_size=hyper.bigram_vocab_size, bigram_dim=hyper.bigram_dim,
        xsa_last_n=hyper.xsa_last_n, value_residual=hyper.value_residual,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Phase 1: Score all sequences
    shard_files = sorted(glob.glob(os.path.join(args.data_dir, "fineweb_train_*.bin")))
    print(f"Found {len(shard_files)} training shards")

    all_scored = []  # list of (loss, token_array)
    total_tokens_in = 0
    t0 = time.time()

    for shard_idx, shard_file in enumerate(shard_files):
        tokens = load_data_shard(shard_file)
        total_tokens_in += tokens.numel()
        num_seqs = (tokens.numel() - 1) // args.seq_len
        scores = score_sequences(model, tokens, args.seq_len, device, args.batch_size)
        losses = [s[1] for s in scores]
        for seq_idx, loss_val in scores:
            offset = seq_idx * args.seq_len
            # Store seq_len + 1 tokens (for input/target overlap)
            seq_tokens = tokens[offset: offset + args.seq_len + 1].numpy().copy()
            if len(seq_tokens) == args.seq_len + 1:
                all_scored.append((loss_val, seq_tokens))

        elapsed = time.time() - t0
        print(f"  Shard {shard_idx + 1}/{len(shard_files)}: {shard_file} | "
              f"{num_seqs} seqs | median_loss={np.median(losses):.4f} | {elapsed:.0f}s elapsed")

    print(f"\nTotal sequences scored: {len(all_scored):,}")
    print(f"Total tokens in: {total_tokens_in:,}")

    # Phase 2: Sort by difficulty (ascending = easy first)
    all_scored.sort(key=lambda s: s[0])
    print(f"Sorted. Easy (loss={all_scored[0][0]:.4f}) → Hard (loss={all_scored[-1][0]:.4f})")

    # Phase 3: Rewrite shards in sorted order
    os.makedirs(args.output_dir, exist_ok=True)
    seqs_per_shard = 100_000_000 // args.seq_len  # ~97K sequences per shard

    total_tokens_out = 0
    shard_count = 0
    for shard_start in range(0, len(all_scored), seqs_per_shard):
        chunk = all_scored[shard_start: shard_start + seqs_per_shard]
        # Concatenate token sequences (each is seq_len+1, overlap by 1)
        # For contiguous streaming: just pack seq_len tokens per sequence, then add 1 final token
        shard_tokens = []
        for _, seq_tokens in chunk:
            shard_tokens.append(seq_tokens[:args.seq_len])  # first seq_len tokens
        # Add the final target token from the last sequence
        shard_tokens.append(chunk[-1][1][-1:])
        shard_np = np.concatenate(shard_tokens).astype("<u2")
        total_tokens_out += len(shard_np)

        out_file = os.path.join(args.output_dir, f"fineweb_train_{shard_count:06d}.bin")
        write_data_shard(out_file, shard_np)
        shard_count += 1
        print(f"  Wrote {out_file}: {len(shard_np):,} tokens")

    # Symlink validation shards
    for val_file in sorted(glob.glob(os.path.join(args.data_dir, "fineweb_val_*.bin"))):
        link = os.path.join(args.output_dir, os.path.basename(val_file))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(val_file), link)

    # Symlink tokenizer files
    for tok_file in sorted(glob.glob(os.path.join(args.data_dir, "../tokenizers/*"))):
        # Tokenizers are in a sibling directory — just note the path
        pass

    # Save metadata
    metadata = {
        "total_sequences": len(all_scored),
        "total_shards": shard_count,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "source_dir": os.path.abspath(args.data_dir),
        "checkpoint": args.checkpoint,
        "easy_loss": all_scored[0][0],
        "hard_loss": all_scored[-1][0],
        "median_loss": all_scored[len(all_scored) // 2][0],
    }
    with open(os.path.join(args.output_dir, "curriculum_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! {len(all_scored):,} sequences → {shard_count} shards in {args.output_dir}")
    print(f"Total tokens in: {total_tokens_in:,}, out: {total_tokens_out:,}")
    print(f"To use: DATA_PATH={args.output_dir} torchrun ... train_gpt_r2.py")


if __name__ == "__main__":
    main()
