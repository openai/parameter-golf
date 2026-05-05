# PR219 vs PR198: Serialization & Artifact Size Reduction Deep Dive

## Summary
PR219 (WarmdownQuantization, samuellarson) achieves 1.1574 BPB in 15.98MB.
PR198 achieves 1.1318 BPB in 15.7MB.
Our best FLAT+zstd benchmark: 14.90MB for same model.

---

## PR219: "Warmdown-Quantization: Training for Compression"
**Record:** \`records/track_10min_16mb/2026-03-19_WarmdownQuantization/\`
**Result:** val_bpb=1.1574 (sliding), artifact=15,977,717 bytes (15.98MB)

### Quantization Approach: Uniform Int6 for ALL Weights
PR219 uses a **single unified** \`quantize_float_tensor(t, bits=6)\` function for ALL weight matrices.
No mixed-precision per-category — everything large enough gets int6.

\`\`\`python
def quantize_float_tensor(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    max_val = 127 if bits == 8 else (2 ** (bits - 1)) - 1  # int6: 31, int8: 127
    # ... per-row quantization with percentile clipping (99.99984%)
\`\`\`

At line 398-400:
\`\`\`python
# Everything else: int6 quantization (saves ~25% vs int8)
q, s = quantize_float_tensor(t, bits=6)
\`\`\`

**Key insight:** Int6 values (range [-32,31]) stored in int8 containers have 3 zero high bits,
which zlib/zstd compresses extremely well. This enables fitting a 3× MLP expansion model.

### FP16 Passthrough Tensors
PR219 keeps THREE classes of tensors in fp16 instead of quantizing:

1. **tok_emb.weight** (tied embedding, line 374-380):
   - Serves as BOTH input embed AND output projection head
   - Int8/int6 quantization errors compound on both pathways
   - fp16 reduces post-quant BPB penalty from 0.005 → ~0.001 BPP

2. **Last 2 layers' c_k.weight** ("Late-K passthrough", lines 382-389):
   - Key projections in final layers are most sensitive
   - Borrowed from PR #99

3. **Small tensors ≤65,536 elements** (line 392):
   - Control tensors (attn_scale, skip_weights, etc.) → fp32
   - Other small tensors → fp16

### Serialization Pipeline
\`\`\`
Full model (bf16/fp32)     →  torch.save("final_model.pt")      → 86.1 MB
                           →  quantize_state_dict_int8()          → 22.69 MB raw
                           →  torch.save(quant_obj, BytesIO)      → 22.73 MB (torch overhead)
                           →  zlib.compress(level=9)              → 15.92 MB
                           +  code (~53.6KB)                      → 15.98 MB total
\`\`\`

Standard \`torch.save\` + \`zlib.compress(level=9)\`. No manual serialization, no byte-grouping,
no FLAT format. The simplest possible pipeline.

### The Warmdown Innovation (Training FOR Compression)
The biggest contribution: \`WARMDOWN_ITERS=20000\` (far beyond actual ~7,199 steps).
This means LR decays linearly across the ENTIRE training run.

**Effect:** Produces tighter weight distributions with fewer outliers → int6 quantization
maps weights to the grid with much less damage.
- Post-quant penalty: 0.014 BPP (default WD=1200) → 0.005 BPP (WD=20000)
- Combined with fp16 embed: → ~0.001 BPP

### Architecture (for context)
- 3× MLP expansion (mlp_mult=3.0, hidden=992 reduced from 1024 to fit fp16 embed)
- NTK-RoPE extrapolation at eval_seq_len=2048 (1.375× train length)
- Sliding window eval stride=256
- Code size: ~53.6KB

---

## PR198: Mixed Int6/Int8 with zstd-22
**Result:** val_bpb=1.1318 (sliding), artifact=~15.7MB
**Script:** \`experiments/pr198_train_gpt.py\` (1516 lines, 64,415 bytes)

### Quantization Approach: Category-Based Mixed Precision
PR198 uses a **category-based** system that assigns different quantization levels per tensor type.

\`\`\`python
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name: return "attn"
    return "other"
\`\`\`

The \`mixed_quantize_int6()\` function (lines 971-1002) takes an \`int6_cats\` set parameter.
Called at line 1422 as:
\`\`\`python
quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
\`\`\`

This means:
- **MLP weights → int6** (range [-32,31], via dedicated \`quantize_int6_per_row\`)
- **Attention weights → int6**
- **Embedding weights → int8** (falls through to standard \`quantize_float_tensor\`)
- **Small tensors ≤65536 → fp16 passthrough**
- **Control tensors → fp32 passthrough**

### FP16 Passthrough: NONE for Large Tensors!
Despite the script's docstring mentioning "fp16 embed + late-K passthrough", the actual code
does NOT implement either:

1. **tok_emb.weight**: Classified as "embed" → falls to int8 via \`quantize_float_tensor\`
   (line 991 comment: "tok_emb.weight falls through to int8 via 'embed' category")
2. **Late-K layers**: Variable \`late_k_layers\` is COMPUTED (line 976) but NEVER USED
   in the quantization loop. Dead code.

This is a 770KB savings vs PR219 (no fp16 passthrough for large tensors).

### Dedicated Int6 Function (not parameterized)
Unlike PR219's generic \`quantize_float_tensor(t, bits=6)\`, PR198 has a **separate** function:
\`\`\`python
def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    # Uses abs().amax() instead of quantile clipping
    row_max = t32.abs().amax(dim=1)
    scale = (row_max / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
\`\`\`

Key difference from PR219: Uses **abs max** instead of **percentile clipping (99.99984%)**.
No outlier clipping → simpler but potentially worse for outlier-heavy distributions.

### Serialization Pipeline
\`\`\`
Full model (bf16/fp32)     →  torch.save("final_model.pt")      → ~80+ MB
                           →  mixed_quantize_int6(sd, {"mlp","attn"})
                           →  torch.save({"w": result, "m": meta}, BytesIO)
                           →  zstd.compress(level=22)             → ~15.7 MB
                           +  code (~64KB)                        → ~15.7 MB total
\`\`\`

**Critical difference:** Uses **zstandard.ZstdCompressor(level=22)** instead of zlib-9.
zstd-22 achieves ~5-10% better compression than zlib-9 on this data, explaining part
of the 280KB artifact size advantage despite having more code bytes.

Falls back to zlib-9 if zstandard is not installed:
\`\`\`python
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
\`\`\`

### Hyperparameters (for context)
- WARMDOWN_ITERS=1200 (default, NOT the aggressive 20000 from PR219)
- MLP_MULT=3.0 (3× expansion), BIGRAM_VOCAB_SIZE=4096, BIGRAM_DIM=128
- SWA_ENABLED=1, SWA_EVERY=200, QAT_ENABLED=0 (despite docstring)
- MUON_WD=0.02, ADAM_WD=0.01
- EVAL_STRIDE=64 (sliding window)
- TRAIN_SEQ_LEN=2048, EVAL_SEQ_LEN=2048
- Code size: 64,415 bytes (~63KB)

---

## Comparison Table

| Feature | PR219 | PR198 | Our Main Script |
|---------|-------|-------|-----------------|
| **val_bpb** | 1.1574 | 1.1318 | ~1.14-1.17 |
| **Artifact size** | 15.98MB | ~15.7MB | ~16-18MB |
| **Quant precision** | Int6 (ALL weights) | Int6 (MLP+attn) / Int8 (embed) | Int8 (ALL) |
| **FP16 tok_emb** | ✅ Yes | ❌ No (int8) | ❌ No |
| **FP16 late-K** | ✅ Yes (last 2 c_k) | ❌ Dead code | ❌ No |
| **Percentile clip** | 99.99984% | None (abs max) | 99.99984% |
| **Compressor** | zlib level 9 | zstd level 22 | zlib level 9 |
| **Serialization** | torch.save (standard) | torch.save (custom dict) | torch.save (standard) |
| **Warmdown trick** | WD=20000 (full training) | WD=1200 (default) | WD=1200 |
| **Code size** | ~53.6KB | ~64KB | ~45KB |
| **Manual serial** | ❌ No | ❌ No | ❌ No |
| **FLAT format** | ❌ No | ❌ No | Tested (750KB win) |

---

## Key Findings

### 1. zstd-22 beats zlib-9 by ~750KB
Our compression benchmarks (COMPRESSION_BENCHMARK_RESULTS.md) confirm:
- torch.save + zstd: 15.65MB
- torch.save + zlib: ~16.4MB (estimated from ratios)
- FLAT + zstd: **14.90MB** (best)
PR198 uses zstd-22, PR219 uses zlib-9. This 750KB difference accounts for most of the
artifact size gap between them.

### 2. FP16 passthrough is a TRADEOFF, not always a win
PR219 spends ~500KB on fp16 tok_emb + ~130KB on fp16 late-K = ~630KB of artifact budget
in exchange for 0.004 BPP less quantization damage. PR198 skips this entirely, using that
space for more model parameters or better compression headroom. At int6 with aggressive
warmdown, the quantization damage is already so low that fp16 passthrough may not be worth it.

### 3. Neither uses FLAT serialization (our biggest opportunity)
FLAT serialization (concatenate all int8 bytes together, then all fp16, etc.) gives 750KB
savings over torch.save+zstd. Neither PR implements it. This is our unique advantage.

### 4. Percentile clipping vs abs max is marginal
PR219 uses percentile clipping at 99.99984% to suppress outliers. PR198 uses simple abs max.
At int6 precision (only 64 levels), outlier clipping matters more because each level covers
a wider range. But PR198's better BPP suggests the training innovations matter more than
quantization precision details.

### 5. PR198's mixed int6/int8 saves bytes over uniform int6
By keeping embeddings at int8 (127 levels) and only using int6 for MLP+attn, PR198 preserves
more precision for the most sensitive tensor (tied embedding) without wasting bytes on fp16.
This is a middle ground between PR219's approaches (all int6 + fp16 embed).

---

## Actionable Recommendations

### Immediate wins (can implement now):
1. **Switch to zstd-22** compression — 750KB savings, zero quality impact
2. **Add FLAT serialization** — additional 750KB savings on top of zstd
3. **Combined FLAT+zstd: ~14.9MB** — this gives 1.1MB headroom for bigger model

### Worth testing:
4. **Remove fp16 passthrough** (match PR198) — saves 630KB, test quality impact
5. **Category-based mixed int6/int8** — int6 for MLP+attn, int8 for embed
   (middle ground that avoids fp16 embed cost while keeping embed precision)
6. **Aggressive warmdown** (WD=20000) — trains for compression, 0.009 BPP savings
7. **Combined FLAT+zstd + no-fp16 + mixed quant**: Could fit ~13.5MB → room for 11 layers

### Best possible pipeline:
\`\`\`
Model weights
→ mixed_quantize (int6 MLP+attn, int8 embed)
→ FLAT serialization (dtype-grouped contiguous bytes + JSON header)
→ zstd level 22 compression
→ ~13.5-14.5MB artifact (with 1.5-2.5MB headroom!)
\`\`\`

This would let us fit 11 layers comfortably, or add larger bigram, or wider model.
