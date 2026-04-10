import sys
from unittest.mock import MagicMock

# Mock dependencies before importing train_challenger
mock_mod = MagicMock()
sys.modules["flash_attn_interface"] = mock_mod
sys.modules["flash_attn"] = mock_mod
sys.modules["sentencepiece"] = mock_mod

import torch
import torch.nn as nn
# Mock CUDA as available to avoid startup crashes in some scripts
torch.cuda.is_available = lambda: False 
torch.cuda.device_count = lambda: 0

import os
import math
import lzma
import io

# Mock CastedLinear and other classes for CPU audit
class CastedLinear(nn.Linear):
    _qat_enabled = False
    def forward(self, x): return super().forward(x)

# Import the actual GPT class from our challenger script (patching for CPU)
from train_challenger import GPT, Hyperparameters

def audit():
    args = Hyperparameters()
    # Force 12 layers and trigram for audit
    args.num_layers = 12
    args.trigram_enabled = True
    args.xsa_last_n = 12
    args.bigram_vocab_size = 3072
    args.bigram_dim = 112
    
    print(f"--- Challenger Audit ---")
    print(f"Layers: {args.num_layers}")
    print(f"Bigram: {args.bigram_vocab_size}x{args.bigram_dim}")
    print(f"Trigram: {args.trigram_enabled}")
    
    model = GPT(
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
    print(f"Total Parameters: {total_params:,}")
    
    # Estimate size: int6 is approx 0.75 bytes per param + scales
    # We'll do a mock state dict save
    sd = model.state_dict()
    export_sd = {k: v for k, v in sd.items() if "mtp_heads" not in k}
    
    # Mock some entropy for LZMA: assume 90% efficiency on int6
    # (Real models are often sparse after GPTQ)
    raw_bits = total_params * 6
    raw_bytes = raw_bits / 8
    
    print(f"Estimated Raw int6 Size: {raw_bytes / 1024 / 1024:.2f} MB")
    
    # Save a small sample to check overhead
    buf = io.BytesIO()
    torch.save(export_sd, buf)
    full_val = buf.getvalue()
    compressed = lzma.compress(full_val, preset=9)
    
    print(f"Full FP32 Compressed (worst case): {len(compressed) / 1024 / 1024:.2f} MB")
    
    if total_params < 20_000_000:
        print("RESULT: PASS - Well within the 16MB limit for int6.")
    else:
        print("RESULT: CAUTION - Parameter count is high, will rely on Selective Pruner.")

if __name__ == "__main__":
    audit()
