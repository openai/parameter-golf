# V15: PR #1735 + CaseOps Tokenizer (TTT EMA disabled)

**Base:** PR #1735 (AjAnubolu, 1.0429 BPB)
**Innovation:** Add CaseOps lossless-case tokenizer (PR #1729) on top of pre-quant TTT stack

## What V15 Does

1. **Switches tokenizer** to `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` (lossless reversible Title/AllCaps/CapNext encoding)
2. **Adds byte sidecar support** to compute honest BPB (CaseOps adds control chars that would inflate naive byte counts)
3. **Disables TTT EMA** (V14 lesson: EMA hurts monotonic-decrease TTT)
4. **Falls back gracefully** to LUT-based byte counting when no sidecar exists

## Expected Result

| Metric | PR #1735 base | V15 (this) | Delta |
|--------|--------------:|-----------:|------:|
| Pre-quant TTT BPB | ~1.033 | ~1.025 | -0.008 |
| Final sliding BPB | 1.0429 | ~1.030-1.038 | -0.005 to -0.012 |
| Record threshold (1.0357) | NO | **YES (~50% prob)** | |

## Compliance Notes

- **CaseOps is lossless reversible** — original text can be recovered exactly
- **Byte sidecar uses RAW UTF-8 byte counts** (not transformed text) — honest BPB
- **No SLOT, no n-gram cache, no eval-time TTT** — inherits PR #1735 cleanliness
- **Pre-quant TTT remains unchanged** — same legal status as PR #1735

## Files Changed

- `records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/train_gpt.py`
  - Added `load_validation_token_bytes()` function
  - Modified `ValidationData.__init__` to load sidecar
  - Modified `eval_val()` to use sidecar
  - Modified `eval_val_sliding()` to use sidecar  
  - Modified `eval_val_ttt()` to use sidecar
  - Disabled TTT EMA by default (V14 lesson)
- `patch_v15_caseops.py`: standalone patch script
- `V15_README.md`: this file

## Usage on RunPod

### Step 1: Clone V15 branch

```bash
cd /workspace
rm -rf parameter-golf
git clone -b v15-pr1735-caseops https://github.com/alertcat/parameter-golf.git
cd parameter-golf

# Verify patches
grep -c "V15: Prefer byte sidecar" records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/train_gpt.py
# Expected: 3
grep -c "load_validation_token_bytes" records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/train_gpt.py
# Expected: >= 2
```

### Step 2: Install deps

```bash
pip install sentencepiece brotli zstandard huggingface-hub hf_transfer -q
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ -q
```

### Step 3: Download CaseOps dataset (~5 min, 16GB)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='/workspace/caseops_data',
)
"

# Verify key files
ls /workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/ | grep -E "val_bytes|val_000000" | head -5
ls /workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

### Step 4: Run V15 scout seed

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/

SEED=1337 \
  DATASETS_DIR=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  TTT_EMA_ENABLED=0 \
  PREQUANT_TTT_ENABLED=1 \
  PREQUANT_TTT_EPOCHS=21 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/scout_v15.log
```

**Watch for this log line confirming sidecar is active:**
```
val_bpb:byte_sidecar:enabled
```

If you see `val_bpb:byte_sidecar:disabled`, the dataset path is wrong — bytes won't be honest.

## Decision Points

After scout (~25 min), check `final_int6_sliding val_bpb`:

| BPB | Verdict |
|-----|---------|
| ≤ 1.0357 | 🔥 **BREAK RECORD** — run seeds 42 + 999, submit |
| 1.0358-1.040 | 👍 Strong, run 3 seeds |
| 1.040-1.045 | 😐 Worse than PR #1735 — investigate sidecar |
| > 1.045 | ❌ Failure — check `val_bpb:byte_sidecar:enabled` line |
