import torch
import io
import os
import zlib
from train_gpt import Hyperparameters, GPT, quantize_state_dict_int8

def verify_size():
    args = Hyperparameters()
    # Initialize the real competition-spec model
    model = GPT(
        vocab_size=1024, 
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init
    )

    # 1. Quantize
    quant_obj, stats = quantize_state_dict_int8(model.state_dict())
    
    # 2. Serialize to bytes
    buffer = io.BytesIO()
    torch.save(quant_obj, buffer)
    raw_bytes = buffer.getvalue()
    
    # 3. Compress (Level 9 is what the script uses)
    compressed_blob = zlib.compress(raw_bytes, level=9)
    
    final_size_mb = len(compressed_blob) / (1024 * 1024)
    code_size_kb = os.path.getsize("train_gpt.py") / 1024
    
    print(f"--- FINAL VERIFICATION ---")
    print(f"Total Parameters: {stats['param_count']:,}")
    print(f"Compressed Model Size: {final_size_mb:.2f} MB")
    print(f"Code Size: {code_size_kb:.2f} KB")
    print(f"TOTAL SUBMISSION: {final_size_mb + (code_size_kb/1024):.2f} MB")
    
    if final_size_mb + (code_size_kb/1024) < 16.0:
        print("✅ LEGAL: You are under the 16MB limit.")
    else:
        print("❌ ILLEGAL: You will be disqualified. Reduce model_dim or num_layers.")

if __name__ == "__main__":
    verify_size()