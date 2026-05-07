"""
patch_v15_caseops.py
====================
V15 = PR #1735 base + CaseOps tokenizer support + TTT EMA disabled

Adds byte sidecar loading to support CaseOps lossless-case tokenizer (PR #1729).
The sidecar (fineweb_val_bytes_*.bin) provides per-token raw UTF-8 byte counts,
which is required for honest BPB computation when tokenizer applies a transform.

V15 changes vs V14:
1. TTT_EMA_ENABLED default 1 -> 0 (V14 showed EMA hurts monotonic-decrease TTT)
2. Add load_validation_token_bytes() function
3. Add val_token_bytes field to ValidationData
4. Modify eval_val() to use sidecar when available (raw_start/raw_end available)
5. Modify eval_val_sliding() to use sidecar (compute via window absolute positions)
6. Modify eval_val_ttt() to use sidecar similarly
"""

import os
import re
import sys

PATH = "records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/train_gpt.py"
if not os.path.exists(PATH):
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
# PATCH 1: Disable TTT EMA by default (V14 lesson - EMA hurts here)
# ============================================================================

old1 = 'ttt_ema_enabled = bool(int(os.environ.get("TTT_EMA_ENABLED", "1")))'
new1 = 'ttt_ema_enabled = bool(int(os.environ.get("TTT_EMA_ENABLED", "0")))  # V15: disabled by default'

if old1 not in src:
    print("WARN: Patch 1 anchor not found (EMA already disabled?)")
else:
    src = src.replace(old1, new1, 1)
    print("Patch 1 applied: TTT_EMA_ENABLED default 1 -> 0")

# ============================================================================
# PATCH 2: Add load_validation_token_bytes function (after load_validation_tokens)
# ============================================================================

# Find load_validation_tokens function definition and add our function after it
patch2_anchor = "def load_validation_tokens"
if patch2_anchor not in src:
    print("ERROR: Cannot find load_validation_tokens function")
    sys.exit(1)

# Insert AFTER the load_validation_tokens function ends.
# We find it by looking for next 'def ' or 'class ' after it.
idx = src.find(patch2_anchor)
# Find end of this function: next 'def ' or 'class ' at column 0
search_start = idx + len(patch2_anchor)
next_def = re.search(r'\n(def |class )', src[search_start:])
if next_def is None:
    print("ERROR: Cannot find end of load_validation_tokens")
    sys.exit(1)
insert_pos = search_start + next_def.start() + 1  # +1 for the newline before next def

new_function = '''
def load_validation_token_bytes(pattern, expected_len):
    """V15: Load byte sidecar for CaseOps tokenizer compliance.

    For tokenizers that apply transforms (e.g., CaseOps), per-token byte counts
    cannot be derived from the SentencePiece vocab alone. The sidecar file
    (fineweb_val_bytes_*.bin) records raw original UTF-8 byte counts per token,
    enabling honest BPB computation.

    Returns None if no sidecar exists (fall back to LUT-based counting).
    """
    bytes_pattern = pattern.replace("fineweb_val_", "fineweb_val_bytes_")
    if bytes_pattern == pattern:
        return None
    files = [Path(p) for p in sorted(glob.glob(bytes_pattern))]
    if not files:
        return None
    token_bytes = torch.cat([load_data_shard(file) for file in files]).to(torch.int32).contiguous()
    if token_bytes.numel() < expected_len:
        raise ValueError(
            f"Validation byte sidecar is too short: expected at least {expected_len}, got {token_bytes.numel()}"
        )
    if token_bytes.numel() > expected_len:
        token_bytes = token_bytes[:expected_len]
    return token_bytes


'''

if "def load_validation_token_bytes" in src:
    print("WARN: Patch 2 already applied")
else:
    src = src[:insert_pos] + new_function + src[insert_pos:]
    print(f"Patch 2 applied: added load_validation_token_bytes() ({len(new_function)} bytes)")

# ============================================================================
# PATCH 3: ValidationData class - load byte sidecar
# ============================================================================

old3 = """        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = ("""

new3 = """        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        # V15: Load byte sidecar for CaseOps compliance (None if no sidecar exists)
        self.val_token_bytes = load_validation_token_bytes(h.val_files, self.val_tokens.numel())
        if h.is_main_process:
            log(f"val_bpb:byte_sidecar:{'enabled' if self.val_token_bytes is not None else 'disabled'}")
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = ("""

if old3 not in src:
    print("ERROR: Patch 3 anchor not found")
    sys.exit(1)
if "self.val_token_bytes" in src:
    print("WARN: Patch 3 already applied")
else:
    src = src.replace(old3, new3, 1)
    print("Patch 3 applied: ValidationData loads byte sidecar")

# ============================================================================
# PATCH 4: eval_val() - use sidecar when available
# ============================================================================

old4 = """            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()"""

new4 = """            # V15: Prefer byte sidecar (CaseOps compliance) when available
            if val_data.val_token_bytes is not None:
                token_bytes = val_data.val_token_bytes[raw_start + 1 : raw_end].to(
                    device=device, dtype=torch.float64, non_blocking=True
                )
                val_byte_count += token_bytes.sum()
            else:
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (
                    val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()"""

if old4 not in src:
    print("ERROR: Patch 4 anchor not found")
    sys.exit(1)
if "V15: Prefer byte sidecar (CaseOps compliance) when available" in src:
    print("WARN: Patch 4 already applied")
else:
    src = src.replace(old4, new4, 1)
    print("Patch 4 applied: eval_val() uses sidecar")

# ============================================================================
# PATCH 5: eval_val_sliding() - use sidecar
# ============================================================================

old5 = """                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)

def eval_val_ttt"""

new5 = """                # V15: Prefer byte sidecar (CaseOps compliance)
                if val_data.val_token_bytes is not None:
                    abs_start = ws + s
                    abs_end = ws + wlen
                    tb = val_data.val_token_bytes[abs_start + 1 : abs_end + 1].to(
                        device=device, dtype=torch.float64, non_blocking=True
                    )
                    byte_count += tb.sum()
                else:
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)

def eval_val_ttt"""

if old5 not in src:
    print("ERROR: Patch 5 anchor not found")
    sys.exit(1)
if "V15: Prefer byte sidecar (CaseOps compliance)" in src:
    print("WARN: Patch 5 already applied")
else:
    src = src.replace(old5, new5, 1)
    print("Patch 5 applied: eval_val_sliding() uses sidecar")

# ============================================================================
# PATCH 6: eval_val_ttt() - use sidecar (same pattern as eval_val_sliding)
# ============================================================================

# In eval_val_ttt, the byte counting block is similar but inside scoring loop
# Look for the 'tb = val_data.base_bytes_lut[tgt]' pattern that follows the
# scored_nll computation in eval_val_ttt
old6 = """                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1"""

new6 = """                    # V15: Prefer byte sidecar (CaseOps compliance)
                    if val_data.val_token_bytes is not None:
                        abs_start = ws + s
                        abs_end = ws + wlen
                        tb = val_data.val_token_bytes[abs_start + 1 : abs_end + 1].to(
                            device=device, dtype=torch.float64, non_blocking=True
                        )
                        byte_count += tb.sum()
                    else:
                        tgt = y_batch[i, s:wlen]
                        prev = x_batch[i, s:wlen]
                        tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                        tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                        byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1"""

if old6 not in src:
    print("WARN: Patch 6 anchor not found (eval_val_ttt may have different pattern)")
else:
    if "V15: Prefer byte sidecar (CaseOps compliance)" in src and src.count("V15: Prefer byte sidecar (CaseOps compliance)") >= 3:
        print("WARN: Patch 6 already applied")
    else:
        src = src.replace(old6, new6, 1)
        print("Patch 6 applied: eval_val_ttt() uses sidecar")

# ============================================================================
# Write back and verify
# ============================================================================

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

new_size = len(src)
print(f"\nPatched train_gpt.py: {original_size} -> {new_size} bytes (+{new_size - original_size})")

import ast
try:
    ast.parse(src)
    print("PASS: Python syntax valid")
except SyntaxError as e:
    print(f"FAIL: SyntaxError at line {e.lineno}: {e.msg}")
    sys.exit(1)

print("\n=== Verification markers ===")
markers = [
    ("def load_validation_token_bytes", 1),
    ("self.val_token_bytes", 4),
    ("V15: Prefer byte sidecar", 3),
    ('TTT_EMA_ENABLED", "0"', 1),
]
for marker, expected_min in markers:
    count = src.count(marker)
    status = "OK" if count >= expected_min else "MISSING"
    print(f"  [{status}] '{marker}': {count} occurrences (expected >= {expected_min})")

print("\n" + "=" * 60)
print("V15 PATCH COMPLETE")
print("=" * 60)
print("\nUsage on RunPod:")
print("  # Need to download CaseOps dataset first:")
print("  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c \"")
print("  from huggingface_hub import snapshot_download")
print("  snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1',")
print("    repo_type='dataset', local_dir='/workspace/caseops_data')")
print("  \"")
print("")
print("  # Then run with CaseOps paths:")
print("  cd records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/")
print("  SEED=1337 \\")
print("    DATASETS_DIR=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \\")
print("    TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \\")
print("    TTT_EMA_ENABLED=0 \\")
print("    PREQUANT_TTT_ENABLED=1 PREQUANT_TTT_EPOCHS=21 \\")
print("    torchrun --standalone --nproc_per_node=8 train_gpt.py")
