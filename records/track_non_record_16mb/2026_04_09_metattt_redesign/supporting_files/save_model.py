#!/usr/bin/env python3
"""Save trained model checkpoint for exp106_metasgd-crosschunk-delta_from_exp101.

Copies final_model.pt and final_model.int6.ptz into a versioned checkpoint
directory alongside a config.json derived from the training hyperparameters.

Note: meta_sgd_{qo,kv,up,down} parameters are EXCLUDED from final_model.pt
(filtered during export). They are not saved here; the checkpoint represents
only the inference-time weights.

Usage (run from repo root or experiment directory):
    python3 records/phase3/exp106_metasgd-crosschunk-delta_from_exp101/save_model.py \
        --model-pt final_model.pt \
        --model-ptz final_model.int6.ptz \
        --output-dir records/phase3/exp106_metasgd-crosschunk-delta_from_exp101/checkpoint
"""

import argparse
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pt",  type=str, default="final_model.pt")
    parser.add_argument("--model-ptz", type=str, default="final_model.int6.ptz")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "checkpoint"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, exp_dir)
    import train_gpt as tg
    sys.path.pop(0)

    hp = tg.Hyperparameters()

    config = {
        "exp_name":              "exp106_metasgd-crosschunk-delta_from_exp101",
        "parent":                "exp101_poscond-bigram-trigram_from_exp95",
        # Meta-TTT redesign flags
        "meta_ttt_enabled":      True,
        "meta_ttt_split":        "batch",        # (A) cross-chunk
        "meta_ttt_delta_weight": 0.3,            # (B) delta-loss
        "meta_sgd_enabled":      True,           # (C) MetaSGD scales
        "meta_sgd_params":       66,             # excluded from export
        # Architecture (unchanged from exp101)
        "vocab_size":            hp.vocab_size,
        "num_layers":            hp.num_layers,
        "model_dim":             hp.model_dim,
        "num_heads":             hp.num_heads,
        "num_kv_heads":          hp.num_kv_heads,
        "mlp_mult":              hp.mlp_mult,
        "tie_embeddings":        hp.tie_embeddings,
        "logit_softcap":         hp.logit_softcap,
        "rope_base":             hp.rope_base,
        "qk_gain_init":          hp.qk_gain_init,
        "bigram_vocab_size":     hp.bigram_vocab_size,
        "bigram_dim":            hp.bigram_dim,
        "unique_layers":         hp.unique_layers,
        "train_seq_len":         hp.train_seq_len,
        # Results
        "steps_completed":       "6686/7500 (wall-clock cap)",
        "pre_quant_val_bpb":     1.1377,
        "int6_val_bpb":          1.1416,
        "float_ttt_bpb":         1.1147,
        "float_baseline_bpb":    1.1377,
        "float_ttt_delta_bpb":   -0.0230,
        "int6_ttt_partial_80pct": 1.1180,
    }

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")

    for src, name in [
        (args.model_pt,  "model.pt"),
        (args.model_ptz, "model.int6.ptz"),
    ]:
        if os.path.exists(src):
            dst = os.path.join(args.output_dir, name)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / 1e6
            print(f"Copied {src} → {dst}  ({size_mb:.2f} MB)")
        else:
            print(f"[skip] not found: {src}")

    print(f"\nCheckpoint saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
