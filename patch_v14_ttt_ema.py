"""
patch_v14_ttt_ema.py
====================
Patches AjAnubolu's PR #1735 train_gpt.py to add TTT weights EMA.

Innovation: Instead of using the LAST epoch's weights from pre-quant TTT,
we maintain an exponential moving average across epochs and use the EMA
weights as the final pre-quant model. This is a standard ML technique that:

1. Reduces noise from late-epoch AdamW oscillation
2. Effectively averages multiple "good" model snapshots
3. Adds <1 second of compute and 0 bytes to the artifact
4. Is unambiguously legal (no val-loss-based selection)

Two new env vars:
- TTT_EMA_ENABLED (default 1): toggle the EMA wrapper
- TTT_EMA_DECAY   (default 0.7): EMA decay factor (0.7 = effective last-5-epochs window)

Usage on RunPod:
    cd /workspace/parameter-golf
    python3 patch_v14_ttt_ema.py
"""

import os
import re
import sys

PATH = "records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/train_gpt.py"

if not os.path.exists(PATH):
    # Try alternative path for local testing
    alt = "E:/parameter/parameter-golf/" + PATH
    if os.path.exists(alt):
        PATH = alt
    else:
        print(f"ERROR: train_gpt.py not found at {PATH}")
        sys.exit(1)

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

original_size = len(src)
print(f"Loaded {PATH} ({original_size} bytes)")

# ============================================================================
# PATCH 1: Add env vars for TTT EMA at the bottom of Hyperparameters defaults
# ============================================================================

# Find the prequant_ttt_grad_clip line and add EMA env vars after it
hp_old = 'prequant_ttt_grad_clip = float(os.environ.get("PREQUANT_TTT_GRAD_CLIP", 1.0))'
hp_new = (
    'prequant_ttt_grad_clip = float(os.environ.get("PREQUANT_TTT_GRAD_CLIP", 1.0))\n'
    '    ttt_ema_enabled = bool(int(os.environ.get("TTT_EMA_ENABLED", "1")))\n'
    '    ttt_ema_decay = float(os.environ.get("TTT_EMA_DECAY", 0.7))'
)

if hp_old not in src:
    print("ERROR: Patch 1 anchor not found")
    sys.exit(1)
if "ttt_ema_enabled" in src:
    print("WARN: Patch 1 already applied, skipping")
else:
    src = src.replace(hp_old, hp_new, 1)
    print("Patch 1 applied: added TTT_EMA_ENABLED and TTT_EMA_DECAY env vars")

# ============================================================================
# PATCH 2: Initialize EMA state before the epoch loop
# ============================================================================

ema_init_anchor = "    base_model.train()\n    batch_seqs = h.ttt_batch_seqs\n\n    for epoch in range(h.prequant_ttt_epochs):"
ema_init_replacement = (
    "    base_model.train()\n"
    "    batch_seqs = h.ttt_batch_seqs\n\n"
    "    # TTT EMA state (v14 innovation): maintain EMA of trainable params across epochs\n"
    "    ttt_ema_state = {}\n"
    "    if h.ttt_ema_enabled:\n"
    "        for n, p in base_model.named_parameters():\n"
    "            if p.requires_grad:\n"
    "                ttt_ema_state[n] = p.data.detach().clone()\n"
    "        log(f'ttt_ema:initialized decay={h.ttt_ema_decay} params={len(ttt_ema_state)}')\n\n"
    "    for epoch in range(h.prequant_ttt_epochs):"
)

if ema_init_anchor not in src:
    print("ERROR: Patch 2 anchor not found")
    sys.exit(1)
if "ttt_ema_state = {}" in src:
    print("WARN: Patch 2 already applied, skipping")
else:
    src = src.replace(ema_init_anchor, ema_init_replacement, 1)
    print("Patch 2 applied: added EMA state initialization")

# ============================================================================
# PATCH 3: Update EMA after each epoch's all_reduce sync
# ============================================================================

ema_update_anchor = "        # Sync: average all trainable parameters across ranks after each epoch\n        if distributed:\n            for p in base_model.parameters():\n                if p.requires_grad:\n                    dist.all_reduce(p.data, op=dist.ReduceOp.AVG)"
ema_update_replacement = (
    "        # Sync: average all trainable parameters across ranks after each epoch\n"
    "        if distributed:\n"
    "            for p in base_model.parameters():\n"
    "                if p.requires_grad:\n"
    "                    dist.all_reduce(p.data, op=dist.ReduceOp.AVG)\n\n"
    "        # TTT EMA update (v14): blend current weights into EMA state\n"
    "        if h.ttt_ema_enabled:\n"
    "            with torch.no_grad():\n"
    "                for n, p in base_model.named_parameters():\n"
    "                    if n in ttt_ema_state:\n"
    "                        ttt_ema_state[n].mul_(h.ttt_ema_decay).add_(p.data, alpha=1.0 - h.ttt_ema_decay)"
)

if ema_update_anchor not in src:
    print("ERROR: Patch 3 anchor not found")
    sys.exit(1)
if "TTT EMA update (v14)" in src:
    print("WARN: Patch 3 already applied, skipping")
else:
    src = src.replace(ema_update_anchor, ema_update_replacement, 1)
    print("Patch 3 applied: added EMA update after each epoch")

# ============================================================================
# PATCH 4: Load EMA weights into model after the epoch loop
# ============================================================================

ema_load_anchor = "    # Unfreeze all parameters\n    for p in base_model.parameters():\n        p.requires_grad_(True)\n    base_model.eval()"
ema_load_replacement = (
    "    # TTT EMA: replace final weights with EMA-averaged weights (v14 innovation)\n"
    "    if h.ttt_ema_enabled and ttt_ema_state:\n"
    "        with torch.no_grad():\n"
    "            for n, p in base_model.named_parameters():\n"
    "                if n in ttt_ema_state:\n"
    "                    p.data.copy_(ttt_ema_state[n])\n"
    "        log(f'ttt_ema:loaded final EMA weights into model')\n"
    "        # Diagnostic: eval with EMA weights\n"
    "        base_model.eval()\n"
    "        with torch.no_grad():\n"
    "            ema_loss, ema_bpb = eval_val(h, device, val_data, base_model)\n"
    "        log(f'ttt_ema:final val_bpb={ema_bpb:.6f} (vs last-epoch above)')\n"
    "        base_model.train()\n\n"
    "    # Unfreeze all parameters\n"
    "    for p in base_model.parameters():\n"
    "        p.requires_grad_(True)\n"
    "    base_model.eval()"
)

if ema_load_anchor not in src:
    print("ERROR: Patch 4 anchor not found")
    sys.exit(1)
if "TTT EMA: replace final weights" in src:
    print("WARN: Patch 4 already applied, skipping")
else:
    src = src.replace(ema_load_anchor, ema_load_replacement, 1)
    print("Patch 4 applied: load EMA weights as final pre-quant model")

# ============================================================================
# Write back and verify
# ============================================================================

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

new_size = len(src)
print(f"\nPatched train_gpt.py: {original_size} -> {new_size} bytes (+{new_size - original_size})")

# Syntax check
import ast
try:
    ast.parse(src)
    print("PASS: Python syntax valid")
except SyntaxError as e:
    print(f"FAIL: SyntaxError at line {e.lineno}: {e.msg}")
    sys.exit(1)

# Final verification: count expected new code markers
markers = [
    "ttt_ema_enabled",
    "ttt_ema_decay",
    "ttt_ema_state = {}",
    "TTT EMA update (v14)",
    "TTT EMA: replace final weights",
]
print("\nVerification markers:")
for m in markers:
    count = src.count(m)
    status = "OK" if count >= 1 else "MISSING"
    print(f"  [{status}] '{m}': {count} occurrences")

print("\n" + "=" * 60)
print("PATCH COMPLETE - Ready to train")
print("=" * 60)
print("\nRun on RunPod (after data download):")
print("  cd records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/")
print("  SEED=1337 TTT_EMA_ENABLED=1 TTT_EMA_DECAY=0.7 \\")
print("    torchrun --standalone --nproc_per_node=8 train_gpt.py")
print("\nExpected output during eval:")
print("  prequant_ttt:epoch 21/21 val_bpb=1.034 ...")
print("  ttt_ema:loaded final EMA weights into model")
print("  ttt_ema:final val_bpb=1.032 (target: better than last-epoch)")
print("  Final 3-seed mean BPB: target 1.040-1.042 (vs base 1.0429)")
