#!/usr/bin/env python3
"""
Verify that clean_train_132_cudagraph.py produces the same model outputs
as clean_train_113.py (the original) when CUDA graphs are disabled.

This tests that the pre-allocated buffer refactoring doesn't change model behavior.

Usage (single GPU, no DDP):
    python experiments/verify_cudagraph.py
"""
import sys
import os
import torch
import torch.nn.functional as F
from torch import Tensor

# We need to import both model classes. Since they're in scripts with main(),
# we'll extract just the model code by importing the modules.

def load_model_class(script_path, class_prefix=""):
    """Import a script and return its GPT class."""
    import importlib.util
    module_name = f"model_{class_prefix}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent the script from running main()
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set required env vars
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Import both modules
    print("Loading original model...")
    orig_mod = load_model_class(
        os.path.join(base_dir, "checkpoints", "clean_train_113.py"), "orig"
    )

    print("Loading cudagraph model...")
    cg_mod = load_model_class(
        os.path.join(base_dir, "experiments", "clean_train_132_cudagraph.py"), "cg"
    )

    # Create models with same config
    torch.manual_seed(42)
    kwargs = dict(
        vocab_size=1024,
        num_layers=12,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=3,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=50000.0,
        qk_gain_init=1.5,
        mtp_num_heads=0,
        bigram_vocab_size=16384,
        bigram_dim=64,
    )

    orig_mod.MLP._activation = "leaky2"
    torch.manual_seed(42)
    orig_model = orig_mod.GPT(**kwargs).to(device).bfloat16()

    cg_mod.MLP._activation = "leaky2"
    torch.manual_seed(42)
    cg_kwargs = {**kwargs, "max_batch_seqs": 4, "max_seq_len": 64}
    cg_model = cg_mod.GPT(**cg_kwargs).to(device).bfloat16()

    # Copy weights from original to cudagraph model to ensure identical params
    cg_model.load_state_dict(orig_model.state_dict(), strict=False)

    # Create test input
    torch.manual_seed(123)
    bsz, seq_len = 4, 64
    input_ids = torch.randint(0, 1024, (bsz, seq_len), device=device)
    target_ids = torch.randint(0, 1024, (bsz, seq_len), device=device)

    # Forward pass (no compile, no CUDA graphs — just raw model)
    orig_model.eval()
    cg_model.eval()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        orig_loss = orig_model(input_ids, target_ids)
        cg_loss = cg_model(input_ids, target_ids)

    print(f"Original loss:   {orig_loss.item():.8f}")
    print(f"CudaGraph loss:  {cg_loss.item():.8f}")
    print(f"Difference:      {abs(orig_loss.item() - cg_loss.item()):.2e}")

    # Test forward_logits too
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        orig_logits = orig_model.forward_logits(input_ids)
        cg_logits = cg_model.forward_logits(input_ids)

    max_diff = (orig_logits - cg_logits).abs().max().item()
    mean_diff = (orig_logits - cg_logits).abs().mean().item()
    print(f"\nLogits max diff:  {max_diff:.2e}")
    print(f"Logits mean diff: {mean_diff:.2e}")

    # Verify
    if max_diff < 1e-3 and abs(orig_loss.item() - cg_loss.item()) < 1e-5:
        print("\n✅ PASS: Models produce identical outputs!")
    else:
        print("\n❌ FAIL: Models produce different outputs!")
        sys.exit(1)

    # Also test training mode (with gradients)
    orig_model.train()
    cg_model.train()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        orig_train_loss = orig_model(input_ids, target_ids)
        cg_train_loss = cg_model(input_ids, target_ids)

    print(f"\nTraining mode:")
    print(f"Original loss:   {orig_train_loss.item():.8f}")
    print(f"CudaGraph loss:  {cg_train_loss.item():.8f}")
    print(f"Difference:      {abs(orig_train_loss.item() - cg_train_loss.item()):.2e}")

    if abs(orig_train_loss.item() - cg_train_loss.item()) < 1e-5:
        print("\n✅ PASS: Training mode also matches!")
    else:
        print("\n❌ FAIL: Training mode differs!")
        sys.exit(1)

if __name__ == "__main__":
    main()
