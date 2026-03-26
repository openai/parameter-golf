#!/usr/bin/env python3
"""Save a trained model checkpoint for exp06_gptq_from-exp05."""

import argparse
import json
import os
import sys
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pt", type=str, default="final_model.pt")
    parser.add_argument("--output-dir", type=str, default="model_checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_gpt as tg

    hp = tg.Hyperparameters()
    config = {
        "vocab_size": hp.vocab_size,
        "num_layers": hp.num_layers,
        "model_dim": hp.model_dim,
        "num_heads": hp.num_heads,
        "num_kv_heads": hp.num_kv_heads,
        "mlp_mult": hp.mlp_mult,
        "tie_embeddings": hp.tie_embeddings,
        "tied_embed_init_std": hp.tied_embed_init_std,
        "logit_softcap": hp.logit_softcap,
        "rope_base": hp.rope_base,
        "qk_gain_init": hp.qk_gain_init,
        "bigram_vocab_size": hp.bigram_vocab_size,
        "bigram_dim": hp.bigram_dim,
        "unique_layers": hp.unique_layers,
        "rope_frac": hp.rope_frac,
        "ema_decay": hp.ema_decay,
        "xsa_last_n": hp.xsa_last_n,
        "qat_threshold": hp.qat_threshold,
        "use_gptq": hp.use_gptq,
        "train_seq_len": hp.train_seq_len,
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if os.path.exists(args.model_pt):
        shutil.copy2(args.model_pt, os.path.join(args.output_dir, "model.pt"))
    quant_path = args.model_pt.replace(".pt", ".int8.ptz")
    if os.path.exists(quant_path):
        shutil.copy2(quant_path, os.path.join(args.output_dir, "model_quant.ptz"))

    print(f"Checkpoint saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
