import sys
from unittest.mock import MagicMock

# Mock dependencies
mock_mod = MagicMock()
sys.modules["flash_attn_interface"] = mock_mod
sys.modules["flash_attn"] = mock_mod
sys.modules["sentencepiece"] = mock_mod

import torch
import torch.nn as nn
# Mock CUDA
torch.cuda.is_available = lambda: False 
torch.cuda.device_count = lambda: 0

import os

# Create a hacky way to import the baseline script
import importlib.util
spec = importlib.util.spec_from_file_location("baseline", "records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py")
baseline = importlib.util.module_from_spec(spec)
# Add dummy globals that the script expects
baseline.code = "" 
spec.loader.exec_module(baseline)

def audit():
    args = baseline.Hyperparameters()
    model = baseline.GPT(
        vocab_size=args.vocab_size, 
        num_layers=args.num_layers, 
        model_dim=args.model_dim,
        num_heads=args.num_heads, 
        num_kv_heads=args.num_kv_heads, 
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, 
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, 
        rope_base=args.rope_base, 
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, 
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"--- Baseline Audit ---")
    print(f"Layers: {args.num_layers}")
    print(f"Total Parameters: {total_params:,}")

if __name__ == "__main__":
    audit()
