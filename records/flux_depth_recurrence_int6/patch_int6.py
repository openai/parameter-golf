#!/usr/bin/env python3
"""
Patch train_gpt.py to add int6 quantization for middle transformer layers.
Run: python3 patch_int6.py

Int6 stores values in [-31, 31] range as int8. Zlib compresses this
much better than full int8 [-127, 127] because fewer unique values.
Middle layers (not first or last) get int6, saving ~25% on those layers.
"""

text = open('train_gpt.py').read()

# Add INT6 env vars after INT8_CLIP_Q line
old_clip = "INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0"
new_clip = """INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

# Int6 quantization for middle layers: uses [-31, 31] range stored as int8.
# Zlib compresses much better due to fewer unique values (~25% size savings).
INT6_ENABLED = bool(int(os.environ.get("INT6_MIDDLE_LAYERS", "0")))
INT6_RANGE = 31  # 6-bit: 2^5 - 1 = 31"""
text = text.replace(old_clip, new_clip)

# Replace quantize_float_tensor to support int6
old_quant = '''def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale'''

new_quant = '''def quantize_float_tensor(t, use_int6=False):
    t32 = t.float()
    qrange = INT6_RANGE if use_int6 else 127
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(qrange)).clamp_min(1.0 / float(qrange))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qrange, qrange).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(qrange) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qrange, qrange).to(torch.int8).contiguous()
    return q, scale'''
text = text.replace(old_quant, new_quant)

# Update quantize_state_dict_int8 to use int6 for middle blocks
old_sd = '''        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)'''

new_sd = '''        stats["num_float_tensors"] += 1
        # Use int6 for middle block layers if enabled
        use_int6 = False
        if INT6_ENABLED and "blocks." in name:
            # Extract block index
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts) and parts[i+1].isdigit():
                    block_idx = int(parts[i+1])
                    total_blocks = len([k for k in state_dict.keys() if k.startswith("blocks.") and ".attn.c_q.weight" in k])
                    if total_blocks > 2 and 0 < block_idx < total_blocks - 1:
                        use_int6 = True
                    break
        q, s = quantize_float_tensor(t, use_int6=use_int6)'''
text = text.replace(old_sd, new_sd)

open('train_gpt.py', 'w').write(text)
print("Patched! Int6 quantization added for middle layers.")
print("Enable with: INT6_MIDDLE_LAYERS=1")
print("")
print("Run command:")
print("INT6_MIDDLE_LAYERS=1 NUM_UNIQUE_BLOCKS=4 NUM_PASSES=3 MODEL_DIM=832 MLP_MULT=3 \\")
print("USE_SWIGLU=1 TRAIN_SEQ_LEN=2048 EVAL_STRIDE=64 \\")
print("torchrun --standalone --nproc_per_node=8 train_gpt.py")
