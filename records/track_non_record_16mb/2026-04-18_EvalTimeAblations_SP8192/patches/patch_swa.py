"""Add Sliding-Window Attention via SWA_WINDOW_SIZE env var.

When SWA_WINDOW_SIZE > 0, each query attends to last W tokens (inclusive of self).
When SWA_WINDOW_SIZE = 0 (default), full causal attention (baseline).

Modifies a single line in the CausalSelfAttention forward call to pass
window_size to flash_attn_3_func.
"""
import re, os
F = 'train_gpt_stacked_v2_fixed.py'
src = open(F).read()

# Add SWA_WINDOW_SIZE module-level constant after the Hyperparameters line
# Keep it simple: module global read from env
old_marker = "from flash_attn_interface import flash_attn_func as flash_attn_3_func"
new_marker = ("from flash_attn_interface import flash_attn_func as flash_attn_3_func\n"
              "_SWA_WINDOW_SIZE = int(os.environ.get('SWA_WINDOW_SIZE', '0'))\n"
              "def _swa_window_arg():\n"
              "\treturn (-1, -1) if _SWA_WINDOW_SIZE <= 0 else (_SWA_WINDOW_SIZE - 1, 0)")

if "_SWA_WINDOW_SIZE" not in src:
    src = src.replace(old_marker, new_marker, 1)
    print("Added SWA_WINDOW_SIZE global")
else:
    print("SWA_WINDOW_SIZE already present")

# Modify the flash_attn_3_func call to pass window_size
old_call = "y=flash_attn_3_func(q,k,v,causal=True)"
new_call = "y=flash_attn_3_func(q,k,v,causal=True,window_size=_swa_window_arg())"
if old_call in src:
    src = src.replace(old_call, new_call, 1)
    print("Modified flash_attn_3_func call to pass window_size")
else:
    if new_call in src:
        print("Already patched")
    else:
        print("ERR: target call pattern not found")
        raise SystemExit(1)

open(F, 'w').write(src)
import py_compile
py_compile.compile(F, doraise=True)
print(f"Patched: {len(src)} bytes, syntax OK")
