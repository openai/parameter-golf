#!/usr/bin/env python3
"""
Tier 6 utility: dump per-token negative log2 probabilities from a trained
neural model checkpoint. Runs on GPU (RunPod).

Produces an NPZ file with per-token -log2(p) values for the validation set,
plus target/prev token IDs for byte accounting in Stage 2a.

Design: uses the model's own forward pass to compute logits, avoiding any
hardcoded architecture assumptions. The checkpoint's state_dict defines
the model shape.

Usage (on RunPod after training):
  python3 experiments/tier6/dump_neural_logprobs.py \
      --trainer train_gpt.py \
      --checkpoint final_model.int8.ptz \
      --output experiments/tier6/neural_logprobs.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import math
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def import_trainer(trainer_path: str):
    """Import the trainer module from an arbitrary path so we use the same
    model definition that produced the checkpoint."""
    spec = importlib.util.spec_from_file_location("trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trainer from {trainer_path}")
    mod = importlib.util.module_from_spec(spec)
    # Prevent the trainer's if __name__=="__main__" from firing
    mod.__name__ = "trainer"
    spec.loader.exec_module(mod)
    return mod


def get_logits_from_model(model, input_ids: Tensor) -> Tensor:
    """Extract logits by running the model's forward pass up to the logit
    projection, using the model's own architecture (including any feature
    paths like kdistill, uts, ccpr, ncc).

    We monkey-patch F.cross_entropy to intercept the logits right before
    loss computation. This guarantees we use the exact same forward path
    the model was trained/evaluated with."""
    captured = {}
    original_ce = F.cross_entropy

    def capture_ce(logits, targets, **kwargs):
        captured["logits"] = logits
        return original_ce(logits, targets, **kwargs)

    F.cross_entropy = capture_ce
    try:
        # Create dummy targets (same shape as input_ids)
        dummy_targets = input_ids.clone()
        model(input_ids, dummy_targets)
    finally:
        F.cross_entropy = original_ce

    if "logits" not in captured:
        raise RuntimeError("Failed to capture logits from model forward pass")
    return captured["logits"]


def infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Infer GPT constructor args from checkpoint keys and tensor shapes,
    so we don't rely on Hyperparameters() defaults."""
    config = {}

    # tok_emb.weight -> (vocab_size, model_dim)
    tok_emb = state_dict.get("tok_emb.weight")
    if tok_emb is not None:
        config["vocab_size"] = tok_emb.shape[0]
        config["model_dim"] = tok_emb.shape[1]

    # Count blocks
    block_ids = set()
    for key in state_dict:
        if key.startswith("blocks."):
            idx = int(key.split(".")[1])
            block_ids.add(idx)
    config["num_layers"] = len(block_ids) if block_ids else 0

    # Attention heads from q projection: blocks.0.attn.c_q.weight -> (dim, dim)
    # KV heads from k projection: blocks.0.attn.c_k.weight -> (kv_dim, dim)
    c_q = state_dict.get("blocks.0.attn.c_q.weight")
    c_k = state_dict.get("blocks.0.attn.c_k.weight")
    if c_q is not None and c_k is not None:
        dim = c_q.shape[0]
        kv_dim = c_k.shape[0]
        # head_dim must divide both; try common sizes
        for hd in [64, 128, 32, 96, 48, 16]:
            if dim % hd == 0 and kv_dim % hd == 0:
                config["num_heads"] = dim // hd
                config["num_kv_heads"] = kv_dim // hd
                break

    # MLP mult from fc weight: blocks.0.mlp.fc.weight -> (hidden, dim)
    fc = state_dict.get("blocks.0.mlp.fc.weight")
    if fc is not None and "model_dim" in config:
        config["mlp_mult"] = fc.shape[0] // config["model_dim"]

    # Tie embeddings: no lm_head means tied
    config["tie_embeddings"] = "lm_head.weight" not in state_dict

    return config


def main():
    parser = argparse.ArgumentParser(description="Dump per-token neural log-probs")
    parser.add_argument("--trainer", default="train_gpt.py",
                        help="Path to the trainer .py file that defines the model")
    parser.add_argument("--checkpoint", default="final_model.int8.ptz",
                        help="Path to int8+zlib compressed checkpoint")
    parser.add_argument("--raw-checkpoint", default="",
                        help="Path to raw .pt checkpoint (alternative to int8+zlib)")
    parser.add_argument("--output", default="experiments/tier6/neural_logprobs.npz")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-seqs", type=int, default=32,
                        help="Sequences per forward pass")
    parser.add_argument("--max-seqs", type=int, default=0,
                        help="Limit to first N sequences (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Import the trainer that produced the checkpoint
    trainer_path = str(Path(args.trainer).resolve())
    print(f"Trainer: {trainer_path}")
    trainer = import_trainer(trainer_path)

    # Load checkpoint state_dict
    print(f"Loading checkpoint: {args.checkpoint or args.raw_checkpoint}")
    if args.raw_checkpoint:
        raw_state = torch.load(args.raw_checkpoint, map_location="cpu")
    else:
        with open(args.checkpoint, "rb") as f:
            quant_blob = f.read()
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")
        raw_state = trainer.dequantize_state_dict_int8(quant_state)

    # Infer model config from checkpoint
    config = infer_model_config_from_state_dict(raw_state)
    print(f"Inferred config: {config}")

    # Fill remaining config from trainer defaults (for non-structural params)
    hparams = trainer.Hyperparameters()
    model = trainer.GPT(
        vocab_size=config.get("vocab_size", hparams.vocab_size),
        num_layers=config.get("num_layers", hparams.num_layers),
        model_dim=config.get("model_dim", hparams.model_dim),
        num_heads=config.get("num_heads", hparams.num_heads),
        num_kv_heads=config.get("num_kv_heads", hparams.num_kv_heads),
        mlp_mult=config.get("mlp_mult", hparams.mlp_mult),
        tie_embeddings=config.get("tie_embeddings", hparams.tie_embeddings),
        tied_embed_init_std=hparams.tied_embed_init_std,
        logit_softcap=hparams.logit_softcap,
        rope_base=hparams.rope_base,
        qk_gain_init=hparams.qk_gain_init,
    )

    model.load_state_dict(raw_state, strict=True)
    model = model.to(device).bfloat16()
    if hasattr(trainer, "CastedLinear"):
        for m in model.modules():
            if isinstance(m, trainer.CastedLinear):
                m.float()
    if hasattr(trainer, "restore_low_dim_params_to_fp32"):
        trainer.restore_low_dim_params_to_fp32(model)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params} params")

    # Load validation data
    val_pattern = f"{args.data_path}/fineweb_val_*.bin"
    val_tokens = trainer.load_validation_tokens(val_pattern, args.seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.seq_len
    if args.max_seqs > 0:
        total_seqs = min(args.max_seqs, total_seqs)
    print(f"Validation: {val_tokens.numel()} tokens, evaluating {total_seqs} sequences")

    # Dump per-token log-probs in chunks to limit memory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    chunk_size = 1000  # sequences per chunk
    all_chunks_logp = []
    all_chunks_tgt = []
    all_chunks_prev = []

    with torch.inference_mode():
        for chunk_start in range(0, total_seqs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_seqs)
            chunk_neg_log2_p = []
            chunk_target_ids = []
            chunk_prev_ids = []

            for batch_start in range(chunk_start, chunk_end, args.batch_seqs):
                batch_end = min(batch_start + args.batch_seqs, chunk_end)
                raw_start = batch_start * args.seq_len
                raw_end = batch_end * args.seq_len + 1

                local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
                x = local[:-1].reshape(-1, args.seq_len)
                y = local[1:].reshape(-1, args.seq_len)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    logits = get_logits_from_model(model, x)

                # logits is flattened: [batch*seq_len, vocab]
                log_probs = F.log_softmax(logits, dim=-1)
                y_flat = y.reshape(-1)
                target_log_probs = log_probs.gather(1, y_flat.unsqueeze(1)).squeeze(1)
                neg_log2_p = -target_log_probs.float() / math.log(2.0)

                chunk_neg_log2_p.append(neg_log2_p.cpu().numpy())
                chunk_target_ids.append(y.reshape(-1).cpu().numpy())
                chunk_prev_ids.append(x.reshape(-1).cpu().numpy())

            all_chunks_logp.append(np.concatenate(chunk_neg_log2_p))
            all_chunks_tgt.append(np.concatenate(chunk_target_ids))
            all_chunks_prev.append(np.concatenate(chunk_prev_ids))

            done = chunk_end
            print(f"  {done}/{total_seqs} seqs dumped")

    neg_log2_p = np.concatenate(all_chunks_logp).astype(np.float32)
    target_ids = np.concatenate(all_chunks_tgt).astype(np.uint16)
    prev_ids = np.concatenate(all_chunks_prev).astype(np.uint16)

    np.savez_compressed(
        args.output,
        neg_log2_p=neg_log2_p,
        target_ids=target_ids,
        prev_ids=prev_ids,
        seq_len=np.array(args.seq_len, dtype=np.int32),
    )
    print(f"Saved {len(neg_log2_p)} token log-probs to {args.output}")
    print(f"Neural BPB check: mean neg_log2_p = {neg_log2_p.mean():.4f} bits/token")


if __name__ == "__main__":
    main()
