# PR198 vs pr135_modified.py — Detailed Artifact Efficiency Comparison

## 1. Code Size (affects artifact total)

| Metric | PR198 | Ours (pr135_modified) | Delta |
|--------|-------|----------------------|-------|
| Bytes  | 64,820 | 95,612 | **+30,792 bytes** |
| Lines  | 1,528  | 2,123  | +595 lines |

**Impact**: Our code is ~30KB larger. Code bytes are part of the 16MB artifact budget.
This wastes ~30KB of headroom that PR198 can use for model weights.

### Dead code in our file NOT in PR198:
- `class NorMuon` optimizer (~85 lines, ~3KB)
- `class BigramSinusoidal` (~30 lines, ~1KB) 
- `def quantize_int5_per_row` (~13 lines)
- `def quantize_int6_blockwise` (~20 lines)
- `def pack_lowbit_tensor` / `def unpack_lowbit_tensor` (~35 lines)
- `def manual_serialize_quantized` / `def manual_deserialize_quantized` (~50 lines)
- `class _FakeQuantizeInt6STE` / `class _FakeQuantizeInt5STE` (~30 lines)
- Wandb integration (~40 lines)
- Magnitude pruning code (~15 lines)
- TTT (Test-Time Training) code (~40 lines)
- LZMA/zlib9/flat compression comparison code (~30 lines)
- int8 quantization path (`quantize_state_dict_int8`, `dequantize_state_dict_int8`) (~65 lines)
- Multiple MLP activation options (abs2, leaky2, softplus2) in `MLP.forward`
- Partial RoPE code in `CausalSelfAttention.forward`
- `_MLP_ACTIVATION` / `_ROPE_FRACTION` globals

**CHECKLIST ITEM 1**: Strip ALL dead code. Target ≤65KB. Remove everything not on the eval path.

---

## 2. CONTROL_TENSOR_NAME_PATTERNS (affects what gets stored as fp32 passthrough)

| Pattern | PR198 | Ours |
|---------|-------|------|
| attn_scale | ✅ | ✅ |
| attn_scales | ✅ | ✅ |
| mlp_scale | ✅ | ✅ |
| mlp_scales | ✅ | ✅ |
| resid_mix | ✅ | ✅ |
| resid_mixes | ✅ | ✅ |
| q_gain | ✅ | ✅ |
| skip_weight | ✅ | ✅ |
| skip_weights | ✅ | ✅ |
| smear | ✅ | ✅ |
| **bigram.scale** | ❌ | ✅ |

**Impact**: Our `bigram.scale` in CONTROL_TENSOR_NAME_PATTERNS means it's stored as **fp32** (4 bytes for 1 scalar).
In PR198, `bigram.scale` is NOT in CONTROL_TENSOR_NAME_PATTERNS. It's a 1-element tensor, so `t.numel() <= 65536` catches it and stores as **fp16** (2 bytes).

Wait — actually in PR198's `mixed_quantize_int6`, the check is:
```python
if not t.is_floating_point() or t.numel() <= 65536:
    result[name] = t.to(torch.float16) if t.is_floating_point() else t
```
And then:
```python
if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
    result[name] = t.float()  # fp32!
```

**The passthrough check comes FIRST**. So `bigram.scale` (numel=1 < 65536) hits the passthrough branch and becomes fp16 in BOTH scripts. Our extra `bigram.scale` in CONTROL_TENSOR_NAME_PATTERNS is only hit by things >65536 elements, and `bigram.scale` is 1 element. **No real difference here.**

**CHECKLIST ITEM 2**: No action needed for bigram.scale, but clean up the extra pattern for clarity.

---

## 3. FP16_KEEP_NAME_PATTERNS (affects tok_emb and late-K layers)

| Feature | PR198 | Ours |
|---------|-------|------|
| FP16_KEEP_NAME_PATTERNS | **Does not exist** | `"tok_emb,blocks.7.attn.c_k,blocks.8.attn.c_k"` |

**Critical difference**: Our script has `FP16_KEEP_NAME_PATTERNS` that keeps `tok_emb.weight` and the last two K-projection weights as **fp16 passthrough** (no quantization).

PR198 does NOT have this. In PR198's `mixed_quantize_int6`:
- `tok_emb.weight` is classified as `"embed"` category
- `"embed"` is NOT in `int6_cats` (which is `{"mlp", "attn"}`)
- So tok_emb falls to the ELSE branch: `quantize_float_tensor(t)` → **int8 quantization** (per-row, 99.99984th percentile clipping)

So:
- **PR198**: `tok_emb.weight` (1024×512 = 524,288 params) → int8 quantized (524K bytes q + 1K bytes scale = ~525KB)
- **Ours**: `tok_emb.weight` → fp16 passthrough (1,048,576 bytes = ~1MB)
- **Delta**: Our tok_emb costs **~523KB MORE** than PR198's

Similarly for `blocks.7.attn.c_k.weight` and `blocks.8.attn.c_k.weight`:
- Each is 256×512 = 131,072 params
- **PR198**: int6 quantized (~131KB q + ~0.5KB scale ≈ 132KB each)
- **Ours**: fp16 passthrough (~262KB each)
- **Delta**: Each late-K layer costs ~130KB MORE → total ~260KB MORE for both

**Total FP16 overhead in our script: ~783KB extra** vs PR198.

**CHECKLIST ITEM 3**: REMOVE `FP16_KEEP_NAME_PATTERNS` entirely. Let tok_emb go through int8/int6 quantization like PR198. Let late-K layers go through int6 quantization. This saves ~783KB.

---

## 4. OUTLIER_QUANT (adds extra tensors per weight matrix)

| Feature | PR198 | Ours |
|---------|-------|------|
| outlier_quant | **Does not exist** | Default=1 (ON) |
| Extra tensors per matrix | 0 | 2 (`.outlier_idx` int16, `.outlier_val` fp16) |

When `OUTLIER_QUANT=1` (our default), for EVERY large 2D weight matrix in "mlp"/"attn" categories, we add:
- `name.outlier_idx`: int16 tensor of shape (n_outliers, 2) — row,col indices
- `name.outlier_val`: fp16 tensor of shape (n_outliers,) — outlier values

For 9 layers with 6 weight matrices each (c_q, c_k, c_v, proj, mlp.fc, mlp.proj) + bigram.proj = ~55 matrices:
- ~0.1% of each matrix = ~262 outliers per 262K-param matrix
- Each outlier: 4 bytes (idx) + 2 bytes (val) = 6 bytes
- Per matrix: ~1.6KB
- Total: **55 × 1.6KB ≈ 88KB of extra outlier data + torch.save overhead for extra tensor entries**

The torch.save pickle/zip overhead for storing ~110 extra tensors (55 idx + 55 val) could add **another 50-100KB** of metadata/alignment padding.

PR198 has ZERO outlier tensors — just `.q` and `.scale` per matrix.

**CHECKLIST ITEM 4**: Set `OUTLIER_QUANT=0` (disable outlier splitting). It adds ~150-200KB of artifact overhead for marginal quality improvement.

---

## 5. torch.save Dict Structure

### PR198:
```python
torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
```
Where `quant_result` dict contains:
- `name` → fp16/fp32 tensor (for passthrough)
- `name.q` → int8 tensor (for quantized)
- `name.scale` → fp16/fp32 tensor (for quantized)

And `quant_meta` dict contains:
- `name` → `"passthrough"` or `"passthrough_ctrl"` or `{"type": "int6"}` or `{"type": "int8"}`

### Ours (with OUTLIER_QUANT=1):
Same structure PLUS:
- `name.outlier_idx` → int16 tensor 
- `name.outlier_val` → fp16 tensor
And meta entries like `{"type": "int6_outlier", "n_outliers": N}` or `{"type": "int6_blockwise", ...}`

**Impact**: More tensors = more torch.save zip/pickle overhead = worse compression ratio.

torch.save overhead per tensor is typically 200-400 bytes (pickle entry + zip local file header + alignment). With outlier splitting, we add ~110 extra tensors → **~22-44KB overhead**.

**CHECKLIST ITEM 5**: After disabling outliers, our dict structure matches PR198's.

---

## 6. Quantization Scale Clamping

### PR198 `quantize_int6_per_row`:
```python
scale = (row_max / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
```

### Ours `quantize_int6_per_row`:
```python
scale = (row_max / 31.0).clamp_min(1e-12).to(torch.float16)
scale = scale.clamp_min(torch.finfo(torch.float16).tiny)  # 6.1e-05
```

**Impact**: PR198 clamps at `1/31 ≈ 0.0323`, ours at `6.1e-05`. PR198's larger minimum scale means zero-ish rows produce scale=0.0323 instead of scale=0.000061, which results in all quantized values being 0 (since round(tiny_val / 0.0323) = 0). This may produce slightly different quantization for near-zero rows but the artifact size difference is negligible.

**CHECKLIST ITEM 6**: Match PR198's `clamp_min(1.0/31.0)` for consistency. Minor.

---

## 7. _classify_param differences

### PR198:
```python
def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    return "other"
```

### Ours:
Same as PR198 PLUS:
```python
    if "bigram.embed" in name: return "mlp"  # extra line
```

**Impact**: In PR198, `bigram.embed.weight` is classified as `"other"`. Since `"other"` is NOT in `int6_cats`, it falls to the ELSE branch → `quantize_float_tensor` → **int8 quantization**.

In ours, `bigram.embed.weight` is classified as `"mlp"` → **int6 quantization**.

For bigram_vocab=4096, bigram_dim=128 → 524,288 params:
- int8: 524K bytes (q) + 4K bytes (per-row scale) ≈ 528KB
- int6-in-int8: 524K bytes (q) + 4K bytes (per-row scale) ≈ 528KB raw, BUT int6 values have 2 zero high bits in int8 → better zstd compression (~1.3x vs ~1.15x)
- **Savings from int6 on bigram.embed: ~70-80KB better compression**

Our approach (int6 for bigram.embed) is BETTER for compression. But it depends on whether the quality loss from int6 vs int8 on bigram.embed matters.

**CHECKLIST ITEM 7**: KEEP our `bigram.embed → mlp` classification. It helps compression.

---

## 8. Model Architecture Differences

| Feature | PR198 | Ours |
|---------|-------|------|
| MLP activation | relu² only | relu², abs², leaky², softplus² |
| `mlp_hidden` param | ❌ | ✅ (allows manual override) |
| Partial RoPE | ❌ | ✅ (`rope_fraction`) |
| BigramSinusoidal | ❌ | ✅ (alternative mode) |
| MTP heads | ✅ (but MTP_NUM_HEADS=0 default) | ❌ (removed?) |
| Flash Attention 3 | ✅ (`_HAS_FA3` check) | ❌ (always SDPA) |
| NTK RoPE in Rotary class | ✅ (`train_seq_len` arg) | ❌ (no auto NTK) |
| `eval_seq_len` separate param | ✅ | ❌ (uses train_seq_len) |
| Attention layout for FA3 | (B,S,H,D) no transpose | (B,H,S,D) with transpose |

**Impact on artifact**: Architecture differences don't change artifact size IF defaults match. But PR198 has MTP heads that get EXCLUDED from the export:
```python
export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
```

**CHECKLIST ITEM 8**: 
- Add `eval_seq_len` hyperparameter (separate from train_seq_len) for NTK RoPE eval
- Add NTK-aware RoPE in Rotary class (auto-scales base when seq_len > train_seq_len)
- Add Flash Attention 3 support for speed
- Remove unused activation modes, partial RoPE, BigramSinusoidal for code size

---

## 9. Training Hyperparameter Defaults

| Parameter | PR198 | Ours | Impact |
|-----------|-------|------|--------|
| warmdown_iters | **1200** | **3000** | Ours trains less at peak LR |
| tied_embed_lr | **0.05** | **0.03** | |
| matrix_lr | **0.04** | **0.02** | |
| scalar_lr | **0.04** | **0.02** | |
| muon_momentum | **0.95** | **0.99** | |
| muon_momentum_warmup_start | **0.85** | **0.92** | |
| muon_momentum_warmup_steps | **500** | **1500** | |
| muon_wd | **0.02** | **0.0** (via muon_weight_decay) | |
| adam_wd | **0.01** | **0.01** | Same |
| logit_softcap | **30.0** | **30.0** | Same |
| adam_eps | **1e-8** | **1e-8** | Same |
| val_loss_every | **1000** | **500** | Ours evals 2x more often |
| train_log_every | **200** | **100** | Ours logs 2x more often |

**CHECKLIST ITEM 9**: These are training-quality params, not artifact-size params. But the final val_bpb depends on training quality. PR198's defaults are the baseline's defaults (conservative). Ours have been tuned.

---

## 10. Serialization Pipeline

### PR198:
```
base_model.state_dict() → exclude mtp_heads → mixed_quantize_int6({"mlp","attn"})
→ torch.save({"w": result, "m": meta}) → zstd-22 compress → .int6.ptz
```

### Ours:
```
base_model.state_dict() → mixed_quantize_int6({"mlp","attn"}, outlier_quant=True) 
→ TWO paths:
  1. torch.save({"w": result, "m": meta}) → zstd-22 → .int6.ptz
  2. manual_serialize_quantized(result, meta) → zstd-22 → .int6.manual.ptz
→ pick smaller, also try LZMA, zlib9, flat concatenation for comparison
```

**Impact**: 
- Our dual-path serialization is good for comparison but the manual_serialize path adds code bloat
- The comparison code (LZMA, zlib9, flat) is dead weight in the submission
- The core serialization matches (torch.save + zstd-22)

**CHECKLIST ITEM 10**: For submission, keep ONLY the torch.save + zstd-22 path. Remove manual serialization, LZMA, zlib9, flat comparison code.

---

## 11. Tensor Count Comparison (9 layers, default config)

### PR198 state_dict (before quantization):
Per block (×9):
- `blocks.{i}.attn_norm` — 0 params (RMSNorm has no weight param)
- `blocks.{i}.mlp_norm` — 0 params
- `blocks.{i}.attn.c_q.weight` — 512×512 = 262,144 → int6 (.q + .scale = 2 tensors)
- `blocks.{i}.attn.c_k.weight` — 256×512 = 131,072 → int6 (.q + .scale = 2 tensors)
- `blocks.{i}.attn.c_v.weight` — 256×512 = 131,072 → int6 (.q + .scale = 2 tensors)
- `blocks.{i}.attn.proj.weight` — 512×512 = 262,144 → int6 (.q + .scale = 2 tensors)
- `blocks.{i}.attn.q_gain` — 8 params → passthrough_ctrl (fp32)
- `blocks.{i}.mlp.fc.weight` — 1536×512 = 786,432 → int6 (.q + .scale = 2 tensors)
- `blocks.{i}.mlp.proj.weight` — 512×1536 = 786,432 → int6 (.q + .scale = 2 tensors)
- `blocks.{i}.attn_scale` — 512 params → passthrough_ctrl (fp32)
- `blocks.{i}.mlp_scale` — 512 params → passthrough_ctrl (fp32)
- `blocks.{i}.resid_mix` — 2×512 = 1024 params → passthrough_ctrl (fp32)

Per block quantized entries: 6 × 2 = 12 (.q + .scale) + 4 passthrough_ctrl = 16 entries
For 9 blocks: 144 entries

Global:
- `tok_emb.weight` — 1024×512 = 524,288 → **int8** (.q + .scale = 2 entries)
- `skip_weights` — 4×512 = 2,048 → passthrough (fp16, numel ≤ 65536)
- `smear.gate` — 512 → passthrough_ctrl (fp32)
- `final_norm` — 0 params (no weight)
- `bigram.embed.weight` — 4096×128 = 524,288 → **int8** (.q + .scale = 2 entries)
- `bigram.proj.weight` — 512×128 = 65,536 → passthrough (fp16, numel ≤ 65536)
- `bigram.scale` — 1 → passthrough (fp16)

Global entries: 2 + 1 + 1 + 0 + 2 + 1 + 1 = 8

**PR198 total dict entries: ~152 in quant_result + ~78 in quant_meta**

### Ours (with OUTLIER_QUANT=1):
Same base as above, BUT:
- Each int6-quantized 2D matrix ALSO gets `.outlier_idx` (int16) + `.outlier_val` (fp16) = 2 extra entries
- 54 int6 matrices × 2 extra = 108 extra entries
- PLUS `tok_emb.weight` → fp16 passthrough (1 entry, no .q/.scale)
- PLUS `blocks.7.attn.c_k.weight` → fp16 passthrough (1 entry)
- PLUS `blocks.8.attn.c_k.weight` → fp16 passthrough (1 entry)

**Our total dict entries: ~152 + 108 outlier entries = ~260 in quant_result**

**CHECKLIST ITEM 11**: Disabling OUTLIER_QUANT drops ~108 extra tensor entries from the artifact.

---

## 12. Compression Efficiency

| Factor | PR198 | Ours | Impact |
|--------|-------|------|--------|
| Compressor | zstd-22 | zstd-22 | Same |
| int6 value range | [-32, 31] | [-32, 31] | Same |
| Number of tensors in torch.save | ~152 | ~260 | More pickle/zip overhead |
| tok_emb storage | int8 (525KB) | fp16 (1MB) | +~500KB |
| Late-K layers | int6 | fp16 | +~260KB |
| Outlier tensors | None | ~88KB data + overhead | +~150KB |
| Code size | 65KB | 96KB | +31KB |

**Estimated total overhead of our approach vs PR198: ~940KB larger artifact**

---

## FINAL CHECKLIST — Changes to Match PR198 Artifact Efficiency

### HIGH IMPACT (saves ~800KB+):
- [ ] **Remove FP16_KEEP_NAME_PATTERNS** — let tok_emb and late-K layers get quantized normally (~783KB)
- [ ] **Set OUTLIER_QUANT=0** as default or remove entirely (~150KB)
- [ ] **Strip dead code** to match PR198's ~65KB code size (~31KB)

### MEDIUM IMPACT (saves ~50-100KB):
- [ ] Remove manual serialization (manual_serialize/deserialize, ~50 lines)
- [ ] Remove LZMA/zlib9/flat compression comparison code (~30 lines)
- [ ] Remove int8 quantization path (quantize_state_dict_int8, dequantize_state_dict_int8)
- [ ] Remove NorMuon class
- [ ] Remove BigramSinusoidal class
- [ ] Remove pack_lowbit_tensor / unpack_lowbit_tensor
- [ ] Remove _FakeQuantizeInt6STE / _FakeQuantizeInt5STE (if QAT disabled)
- [ ] Remove quantize_int5_per_row / quantize_int6_blockwise
- [ ] Remove TTT code
- [ ] Remove magnitude pruning code
- [ ] Remove wandb integration
- [ ] Remove unused MLP activation modes
- [ ] Remove partial RoPE support

### LOW IMPACT (correctness/consistency):
- [ ] Match PR198's scale clamp: `clamp_min(1.0/31.0)` instead of `clamp_min(1e-12)`
- [ ] Add `eval_seq_len` as separate hyperparameter
- [ ] Add NTK-aware RoPE in Rotary class
- [ ] Add Flash Attention 3 support
- [ ] Keep `bigram.embed → "mlp"` classification (our approach is better)

### DO NOT CHANGE:
- ✅ Keep zstd-22 compression (matches PR198)
- ✅ Keep torch.save({"w": ..., "m": ...}) format (matches PR198)
- ✅ Keep int6 for mlp+attn categories (matches PR198)
- ✅ Keep CONTROL_TENSOR_NAME_PATTERNS (matches PR198 for all functional patterns)
- ✅ Keep bigram.embed classified as "mlp" for int6 (better than PR198's int8)

### TOTAL ESTIMATED SAVINGS: ~940KB → comfortably under 16MB with headroom for larger model
