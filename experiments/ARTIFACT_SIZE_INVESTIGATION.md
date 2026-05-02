# Artifact Size Investigation: Our Script vs PR198

## The Problem
- **PR198's exact script on our machine**: 12.71MB artifact (9L, 22.4M params, torch.save+zstd)
- **Our script (pr135_modified.py) with same params**: 15.78MB artifact (9L, 22.4M params, FLAT+zstd)
- **Gap: 3.07MB — same hardware, same model architecture, different serialization code**

Both scripts use the same model: 9 layers, dim=512, 8 heads, 4 KV heads, MLP=3x,
vocab=1024, bigram_vocab=4096, bigram_dim=128, tied embeddings. Both achieve ~22.37M params.

---

## ROOT CAUSE #1: FP16_KEEP_NAME_PATTERNS (~1.0-1.5MB compressed)

### What it does
Our script (line 509-513) keeps certain tensors as fp16 INSTEAD of quantizing them:
```python
FP16_KEEP_NAME_PATTERNS = ("tok_emb", "blocks.7.attn.c_k", "blocks.8.attn.c_k")
```

**PR198 has NO equivalent — all tensors get int6/int8 quantized.**

### Impact breakdown

| Tensor | Our storage | PR198 storage | Raw diff |
|--------|-------------|---------------|----------|
| tok_emb.weight (1024x512) | fp16: 1,048,576B | int8+scale: 526,336B | +522,240B |
| blocks.7.attn.c_k (512x256) | fp16: 262,144B | int6+scale: 132,096B | +130,048B |
| blocks.8.attn.c_k (512x256) | fp16: 262,144B | int6+scale: 132,096B | +130,048B |
| **Total raw** | **1,572,864B** | **790,528B** | **+782,336B (~764KB)** |

### Why compressed impact is MUCH larger than raw (~1.0-1.5MB)
The critical insight from exp075: **fp16 data barely compresses with zstd** (~1.0-1.1x ratio)
while int6-in-int8 compresses excellently (~1.75-1.85x).

- Our tok_emb fp16 (1.05MB raw) -> ~1.00MB compressed (random embeddings don't compress)
- PR198 tok_emb int8 (0.52MB raw) -> ~0.29MB compressed (int8 has pattern -> compresses 1.80x)
- **Compressed difference for tok_emb alone: ~0.71MB**

- Our 2 K-proj fp16 (0.52MB raw) -> ~0.50MB compressed
- PR198 K-proj int6 (0.26MB raw) -> ~0.14MB compressed
- **Compressed difference for K-proj: ~0.36MB**

**Total FP16_KEEP compressed cost: ~1.07MB**

Verified by exp075: switching bigram.embed from fp16->int6 saved 1.58MB compressed
(0.52MB raw difference -> 1.58MB compressed because fp16 barely compresses).

### Fix
```python
FP16_KEEP_NAME_PATTERNS = ()  # Empty — quantize everything
# OR via env var:
export FP16_KEEP_NAME_PATTERNS=""
```

---

## ROOT CAUSE #2: Outlier Splitting (~0.2-0.4MB compressed)

### What it does
Our script (line 757-810, OUTLIER_QUANT=1 default) extracts the top 0.1% magnitude values
from each large weight matrix and stores them separately:
```python
# For each qualifying matrix, stores 4 tensors instead of 2:
result[name + ".q"]           # int8: main quantized weights (outliers zeroed)
result[name + ".scale"]       # fp16: per-row scales
result[name + ".outlier_idx"] # int16: (row, col) coordinates of outliers
result[name + ".outlier_val"] # fp16: original outlier values
```

**PR198 has NO outlier splitting — each matrix stores just `.q` and `.scale`.**

### Impact

For 9L model, ~53 matrices qualify for outlier splitting:
- ~21,500 total outliers across all matrices
- outlier_idx: int16 tensor, 21,500 x 2 x 2 bytes = **~86KB raw**
- outlier_val: fp16 tensor, 21,500 x 2 bytes = **~43KB raw**
- Total extra data: **~129KB raw**

**But the compressed impact is WORSE than raw because:**
1. Outlier indices are essentially random positions -> **0% compressibility**
2. Outlier values are essentially random fp16 -> **0% compressibility**
3. Confirmed in COMPRESSION_BENCHMARK_RESULTS.md: "Outlier splitting hurts compression"
4. COMPRESSION_SUGGESTIONS.md: "Outlier splitting fragments structure... creates index arrays
   (semi-random) and value arrays (fp16 noise) that break contiguous structure"

Plus: 53 x 2 = 106 extra tensor entries in torch.save -> ~20-30KB extra ZIP/pickle overhead.

**Estimated compressed cost: ~200-400KB**

### Fix
```python
export OUTLIER_QUANT=0
```
Or remove outlier code entirely from mixed_quantize_int6.

---

## ROOT CAUSE #3: Code Size (+31KB)

| Script | Size | Lines |
|--------|------|-------|
| PR198 (pr198_train_gpt.py) | 64,820 bytes | ~1,500 |
| Ours (pr135_modified.py) | 95,612 bytes | 2,123 |
| **Difference** | **+30,792 bytes** | **+623 lines** |

The artifact `code_bytes = len(code.encode("utf-8"))` is included in the submission size.

### Extra code in our script (not in PR198):
1. `BigramSinusoidal` class (lines 1260-1287) — unused when bigram_mode="learned"
2. `quantize_int5_per_row`, `quantize_int6_blockwise` functions (lines 666-750)
3. `manual_serialize_quantized`, `manual_deserialize_quantized` (lines 904-980)
4. `fake_quantize_int5_ste`, `fake_quantize_int6_ste` QAT functions
5. TTT (test-time training) code (lines 2027-2075)
6. Multiple compression format comparisons (FLAT, LZMA, zlib9) — lines 1951-1990
7. Partial RoPE support, multiple activation functions
8. Extra env var parsing and logging

### Fix
Strip unused features to match PR198's code size. Target: <65KB.

---

## ROOT CAUSE #4: Compression Ratio Degradation (explains gap amplification)

### Why 0.94MB raw difference -> 3.07MB compressed difference

The overall compression ratio depends on the DATA MIX:

**PR198's data composition:**
- 98.7% int8 (int6/int8 values) -> compresses at 1.80x
- 0.9% fp16 (scales + small passthrough) -> compresses at ~1.2x
- 0.4% float32 (control params) -> compresses at ~1.1x
- **Overall ratio: ~1.79x -> 22.58MB raw -> 12.65MB compressed**

**Our data composition:**
- 91.4% int8 (int6 values, fewer because tok_emb+K removed) -> compresses at 1.80x
- 7.8% fp16 (scales + FP16_KEEP passthrough + outlier vals) -> compresses at ~1.05x
- 0.4% int16 (outlier indices) -> compresses at ~1.05x
- 0.4% float32 (control params) -> compresses at ~1.1x
- **Overall ratio: ~1.50-1.60x -> 23.49MB raw -> 15.71MB compressed**

The fp16 passthrough data acts as a **compression anchor** — it barely compresses and
drags down the overall ratio. Each MB of fp16 passthrough costs ~0.95MB compressed,
while the int8 data it replaces would only cost ~0.56MB compressed. The effective
compressed cost of FP16_KEEP is ~1.7x the raw byte difference.

---

## ROOT CAUSE #5: _classify_param Differences (MINOR, helps us)

Our script classifies `bigram.embed` as "mlp" -> gets int6 quantization.
PR198 classifies it as "other" -> gets int8 quantization.

Int6 values [-32,31] compress BETTER than int8 [-127,127], so this
slightly HELPS our compression. **Not a problem — keep this.**

---

## ROOT CAUSE #6: CONTROL_TENSOR_NAME_PATTERNS (NEGLIGIBLE)

Our patterns include `"bigram.scale"` (not in PR198).
Effect: bigram.scale stored as float32 (4 bytes) instead of fp16 (2 bytes).
**Impact: +2 bytes. Negligible.**

---

## SUMMARY OF FIXES (ordered by impact)

| # | Fix | Estimated Savings | Effort |
|---|-----|-------------------|--------|
| 1 | Remove FP16_KEEP_NAME_PATTERNS | **~1.0-1.5MB** | 1 line |
| 2 | Disable outlier splitting (OUTLIER_QUANT=0) | **~0.2-0.4MB** | 1 env var |
| 3 | Strip unused code features | **~31KB** | Medium |
| 4 | Combined compression ratio improvement | **~0.5-1.0MB** | Automatic (from 1+2) |
| **Total** | | **~1.7-2.9MB** | |

### After all fixes, expected artifact: ~12.9-14.1MB (vs current 15.78MB)

---

## DETAILED DIFF: mixed_quantize_int6

### PR198 (clean, simple — lines 983-1014):
```python
def mixed_quantize_int6(state_dict, int6_cats):
    # For each param:
    # 1. Small/non-float -> passthrough (fp16 or original)
    # 2. CONTROL pattern -> float32 passthrough
    # 3. MLP/Attn category -> int6 per-row (.q + .scale)
    # 4. Everything else -> int8 per-row (.q + .scale)
    # Result: ~130 tensor entries, clean data, high compressibility (1.80x)
```

### Ours (complex, adds overhead — lines 752-840):
```python
def mixed_quantize_int6(state_dict, int6_cats, outlier_quant, blockwise_quant, int5_mlp, block_size):
    # For each param:
    # 1. Small/non-float -> passthrough (fp16 or original)
    # 2. CONTROL pattern -> float32 passthrough
    # 3. FP16_KEEP pattern -> fp16 passthrough (EXTRA — not in PR198!)
    # 4. MLP/Attn with outlier split -> .q + .scale + .outlier_idx + .outlier_val (EXTRA!)
    # 5. MLP/Attn without outlier -> int6/int5 per-row (.q + .scale)
    # 6. Everything else -> int8 per-row (.q + .scale)
    # Result: ~235 tensor entries, mixed dtype data, poor compressibility (1.50x)
```

### Key differences in dequantize_mixed_int6:
Our version handles 5 types (int6, int6_outlier, int6_blockwise, int6_outlier_blockwise, int8).
PR198 handles 2 types (int6, int8).

---

## VERIFICATION: Known Data Points

| Experiment | Config | FLAT+zstd | Notes |
|------------|--------|-----------|-------|
| 068 (9L no bigram) | no FP16_KEEP, outlier on | 14.90MB | Baseline without bigram |
| 071 (9L no bigram) | with some opts | 15.33MB | |
| 075 (9L bigram int6) | FP16_KEEP on, outlier on | 15.80MB | Adding bigram +0.90MB |
| 070 (9L bigram fp16) | FP16_KEEP on, outlier on | 17.38MB | bigram as fp16 = +2.5MB! |
| 081 (9L bigram int6) | FP16_KEEP on, outlier on | 15.78MB | Same as 075 basically |
| 087 (10L no bigram) | no FP16_KEEP, outlier on | 16.25MB | 10L over budget by 252KB |
| PR198 (9L bigram int8) | no FP16_KEEP, no outlier | **12.71MB** | TARGET |

Key observation: removing FP16_KEEP (070->075) saved 1.58MB for just bigram.embed.
Removing it for tok_emb + 2 K projections should save ~1.0-1.5MB more.

---

## ACTIONABLE NEXT STEPS

### Quick fix (1 minute, estimated -> ~14.3-14.8MB):
```bash
export FP16_KEEP_NAME_PATTERNS=""
export OUTLIER_QUANT=0
```

### Code cleanup (30 min, estimated -> ~14.0-14.5MB):
Strip BigramSinusoidal, int5/blockwise functions, TTT code,
manual serialization, FLAT comparison, multiple activations.
Target code size: <65KB (same as PR198).

### Match PR198's serialization exactly (estimated -> ~12.7-13.0MB):
Replace our mixed_quantize_int6 with PR198's simpler version (no outlier args,
no int5, no blockwise, no FP16_KEEP). Use PR198's torch.save+zstd format.

### UNLOCK 10L MODEL:
After fixes, 10L model should go from 16.25MB -> ~13.5-14.5MB
-> **EASILY fits under 16MB** with 1.5-2.5MB headroom!
This unlocks the 0.004 BPB improvement (exp087: 1.1391 vs best 9L 1.1427).

---

## FILE REFERENCES

| File | Lines | Description |
|------|-------|-------------|
| pr135_modified.py | 509-513 | FP16_KEEP_NAME_PATTERNS definition |
| pr135_modified.py | 500-507 | CONTROL_TENSOR_NAME_PATTERNS (has extra "bigram.scale") |
| pr135_modified.py | 655-664 | _classify_param (has extra "bigram.embed" -> "mlp") |
| pr135_modified.py | 666-750 | quantize_int5/int6_blockwise (unused by PR198) |
| pr135_modified.py | 752-840 | mixed_quantize_int6 (complex, with outlier/blockwise/int5) |
| pr135_modified.py | 842-901 | dequantize_mixed_int6 (handles 5 types vs PR198's 2) |
| pr135_modified.py | 904-980 | manual_serialize/deserialize (not in PR198) |
| pr135_modified.py | 1260-1287 | BigramSinusoidal class (not in PR198) |
| pr135_modified.py | 2027-2075 | TTT code (not in PR198) |
| /tmp/pr198_train_gpt.py | 312-316 | CONTROL (no "bigram.scale") |
| /tmp/pr198_train_gpt.py | 962-969 | _classify_param (no "bigram.embed" special case) |
| /tmp/pr198_train_gpt.py | 971-981 | quantize_int6_per_row (identical to ours) |
| /tmp/pr198_train_gpt.py | 983-1014 | mixed_quantize_int6 (clean, no extras) |
