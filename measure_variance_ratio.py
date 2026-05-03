"""
Measure per-layer variance ratio r_ℓ = Var(p_t - p_{t-1}) / Var(p_t)
for trained standard and DG Attention models.

Usage:
    python measure_variance_ratio.py --model-type standard --checkpoint final_model.pt --script train_gpt.py
    python measure_variance_ratio.py --model-type dg --checkpoint final_model.pt --script records/.../train_gpt.py
"""

import argparse
import importlib.util
import sys
import json
import torch
import numpy as np
import sentencepiece as spm
from pathlib import Path


def load_module_from_path(script_path: str, module_name: str = "train_mod"):
    """Dynamically import a train_gpt.py so we can reuse its model classes."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent it from running main
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(mod, device, model_type):
    """Build a model using the module's Hyperparameters and GPT class.

    Inspects the GPT constructor to pass any extra kwargs the script version
    accepts (e.g. attn_variant, bigram_vocab_size, latent_kv_dim).
    """
    import inspect
    args = mod.Hyperparameters

    # Base kwargs every GPT version needs
    kwargs = dict(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=getattr(args, 'tied_embed_init_std', 0.005),
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=getattr(args, 'qk_gain_init', 1.5),
    )

    # Check what extra params the GPT constructor accepts and pass them
    sig = inspect.signature(mod.GPT.__init__)
    extra_params = {
        'attn_variant': model_type if model_type == 'dg' else 'standard',
        'bigram_vocab_size': getattr(args, 'bigram_vocab_size', 0),
        'bigram_dim': getattr(args, 'bigram_dim', 128),
        'latent_kv_dim': getattr(args, 'latent_kv_dim', 64),
    }
    for name, value in extra_params.items():
        if name in sig.parameters:
            kwargs[name] = value

    model = mod.GPT(**kwargs).to(device)
    return model


def measure_variance_ratio(model, data_tokens, seq_len, device, model_type):
    """
    Hook into each attention layer's value/payload projection,
    measure Var(p_t - p_{t-1}) / Var(p_t) per layer.
    """
    model.eval()

    # Collect projection outputs per layer
    projections = {}

    def make_hook(layer_idx, proj_name):
        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, kv_dim)
            key = layer_idx
            if key not in projections:
                projections[key] = []
            projections[key].append(output.detach().float())
        return hook_fn

    # Register hooks on the value/payload projection in each block
    # Auto-detect: DG models have c_payload, standard models have c_v
    hooks = []
    for i, block in enumerate(model.blocks):
        attn = block.attn
        if hasattr(attn, 'c_payload'):
            proj = attn.c_payload
        elif hasattr(attn, 'c_v'):
            proj = attn.c_v
        else:
            raise AttributeError(f"Block {i} attention has neither c_payload nor c_v")
        h = proj.register_forward_hook(make_hook(i, type(attn).__name__))
        hooks.append(h)

    # Forward pass over data in chunks
    num_tokens = len(data_tokens)
    num_seqs = num_tokens // seq_len
    if num_seqs == 0:
        raise ValueError(f"Not enough tokens ({num_tokens}) for seq_len={seq_len}")

    # Use up to 4096 tokens worth of sequences
    max_seqs = min(num_seqs, 4096 // seq_len + 1)
    tokens = torch.tensor(data_tokens[: max_seqs * seq_len], dtype=torch.long, device=device)
    tokens = tokens.reshape(max_seqs, seq_len)

    # We need input_ids and target_ids for the model's forward
    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:]

    with torch.no_grad():
        # Process in small batches to avoid OOM
        batch_size = min(4, max_seqs)
        for start in range(0, max_seqs, batch_size):
            end = min(start + batch_size, max_seqs)
            model(input_ids[start:end], target_ids[start:end])

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute variance ratios
    results = {}
    num_layers = len(model.blocks)
    for layer_idx in sorted(projections.keys()):
        # Concatenate all batches: shape (total_seqs, seq_len-1, kv_dim)
        p = torch.cat(projections[layer_idx], dim=0)  # (B, T, D)

        # p_t and p_{t-1}
        p_curr = p[:, 1:, :]  # positions 1..T-1
        p_prev = p[:, :-1, :]  # positions 0..T-2
        diff = p_curr - p_prev

        var_diff = diff.var().item()
        var_raw = p_curr.var().item()
        r_ell = var_diff / var_raw if var_raw > 0 else float('nan')

        # Compute alpha for this layer (DG schedule)
        alpha = layer_idx / max(num_layers - 1, 1)

        results[layer_idx] = {
            "layer": layer_idx,
            "alpha": round(alpha, 2),
            "var_diff": var_diff,
            "var_raw": var_raw,
            "r_ell": round(r_ell, 4),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=["standard", "dg"])
    parser.add_argument("--checkpoint", required=True, help="Path to final_model.pt")
    parser.add_argument("--script", required=True, help="Path to train_gpt.py that defines the model")
    parser.add_argument("--data-dir", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the training script as a module
    mod = load_module_from_path(args.script)

    # Build model and load checkpoint
    model = build_model(mod, device, args.model_type)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded {args.model_type} model from {args.checkpoint}")

    # Load validation data (first shard)
    import glob
    val_files = sorted(glob.glob(f"{args.data_dir}/fineweb_val_*.bin"))
    if not val_files:
        raise FileNotFoundError(f"No val files in {args.data_dir}")
    data = np.fromfile(val_files[0], dtype=np.uint16).astype(np.int32)
    # Use first 8192 tokens, clamp to vocab size
    data = data[:8192]
    vocab_size = mod.Hyperparameters.vocab_size
    data = np.clip(data, 0, vocab_size - 1)
    print(f"Loaded {len(data)} tokens from {val_files[0]} (clamped to vocab_size={vocab_size})")

    # Measure
    results = measure_variance_ratio(model, data, args.seq_len, device, args.model_type)

    # Print table
    print(f"\n{'Layer':>5} {'α':>5} {'r_ℓ':>8} {'Var(diff)':>12} {'Var(raw)':>12}")
    print("-" * 50)
    for idx in sorted(results.keys()):
        r = results[idx]
        print(f"{r['layer']:>5} {r['alpha']:>5.2f} {r['r_ell']:>8.4f} {r['var_diff']:>12.6f} {r['var_raw']:>12.6f}")

    # Gradient
    layers = sorted(results.keys())
    if len(layers) >= 2:
        gradient = results[layers[0]]['r_ell'] - results[layers[-1]]['r_ell']
        print(f"\nGradient (r_0 - r_last): {gradient:+.4f}")

    # Save JSON
    output_path = args.output or f"variance_ratio_{args.model_type}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
