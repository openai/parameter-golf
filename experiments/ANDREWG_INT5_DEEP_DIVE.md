# andrewgcodes/parameter-golf PR #4 Deep Dive: Int5 MLP, Quantization, Serialization

**File analyzed**: `records/track_10min_16mb/2026-03-20_OptimizedArchitecture/train_gpt.py`
**Commit**: `2e47666a76cfa09662b653e7893c9270f94e60a7`
**Claimed results**: val_bpb=1.1385 (submission.json), 15.87MB total artifact

---

## 1. CastedLinear & STE Fake Quantization During Training

### Location: Lines 488-504

```python
INT6_QUANT_RANGE = 31         # constant used for STE
INT6_CLIP_Q = 0.9999984
_INT6_STE_ENABLED = bool(int(os.environ.get("INT6_STE", "0")))  # OFF BY DEFAULT

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if _INT6_STE_ENABLED and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = w.float()
                clip_abs = torch.quantile(w32.abs(), INT6_CLIP_Q, dim=1).clamp_min(1e-8)
                scale = clip_abs / INT6_QUANT_RANGE  # scale = row_max / 31
                w_clipped = torch.clamp(w32, -clip_abs[:, None], clip_abs[:, None])
                w_q = (torch.round(w_clipped / scale[:, None]) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # STE: forward uses quantized, backward uses original
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
```

### Key findings:

1. **STE is DISABLED by default** (`INT6_STE=0`). The submission README does NOT set `INT6_STE=1` in the env vars. So **the winning 1.1385 BPB run did NOT use STE during training**.

2. **When enabled, CastedLinear only knows about int6 (clip_range=31)**. It does NOT distinguish between MLP and attention weights. There is NO int5 awareness during training. The STE always quantizes to [-32, 31] range.

3. **The STE uses per-row quantile clipping** (`INT6_CLIP_Q = 0.9999984`), not simple max. This matches the post-training quantization's `quantize_float_tensor` which uses `INT8_CLIP_PERCENTILE = 99.99984`.

4. **STE implementation**: Standard straight-through estimator: `w = w + (w_q - w).detach()`. Forward pass sees quantized weights, backward pass gets gradients through original weights.

**BOTTOM LINE: Int5 is purely a post-training quantization choice. CastedLinear has no int5 awareness during training. The STE (when enabled) only simulates int6.**

---

## 2. mixed_quantize_int6: Int5 for MLP vs Int6 for Attention

### Location: Lines 335-386

### Parameter classification (line 335):
```python
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
```

### The quantization logic (lines 357-386):
```python
def mixed_quantize_int6(state_dict, int6_cats):
    for name, tensor in state_dict.items():
        # Priority 1: Small tensors (<=65536 elements) → fp16 passthrough
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough"
            continue

        # Priority 2: Control tensors (scales, gains, etc.) → float32 passthrough
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue

        # Priority 3: FP16 keep patterns → fp16 passthrough
        # DEFAULT: "tok_emb,blocks.8.attn.c_k"
        if any(pattern in name for pattern in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(dtype=torch.float16)
            meta[name] = "passthrough_fp16"
            continue

        # Priority 4: The actual mixed quantization
        if cat in int6_cats and t.ndim >= 1:
            clip = 15 if cat == "mlp" else 31  # ← THIS IS THE KEY LINE
            q, s = quantize_intN_per_row(t, clip_range=clip)
            result[name + ".q"] = q      # int8 tensor with values in [-16,15] or [-32,31]
            result[name + ".scale"] = s   # fp16 per-row scale
            meta[name] = {"type": f"int{5 if cat == 'mlp' else 6}"}
```

### The quantize_intN_per_row function (lines 344-355):
```python
def quantize_intN_per_row(t, clip_range=31):
    # For 2D tensors (weight matrices):
    row_max = t32.abs().amax(dim=1)
    scale = (row_max / clip_range).clamp_min(1e-12).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale[:, None]), -(clip_range+1), clip_range).to(torch.int8)
    return q, scale
```

### Key findings:

1. **Int5 MLP clip range**: `clip = 15` → values clipped to `[-16, 15]` → stored as `torch.int8`
   - The int8 byte stores values like: -16, -15, ..., -1, 0, 1, ..., 14, 15
   - In binary, these values only use 5 bits of the 8-bit representation
   - The 3 unused high bits are always 0 (for positive) or 1 (for negative, due to two's complement)
   - This pattern is highly compressible by zstd/zlib

2. **Int6 attention clip range**: `clip = 31` → values clipped to `[-32, 31]` → stored as `torch.int8`
   - Values use 6 of 8 bits → 2 unused bits → less compressible than int5 but still better than full int8

3. **FP16 passthrough for specific tensors**: `tok_emb` (tied embedding) and `blocks.8.attn.c_k` are stored as fp16, NOT quantized at all. The tied embedding serves as both input embedding and output head — keeping it fp16 eliminates the most sensitive quantization bottleneck (PR #42's finding).

4. **Scale storage**: Per-row scales stored as `torch.float16` — adds 2 bytes per row of the weight matrix.

5. **The "other" category**: Parameters that don't match mlp/attn/embed get classified as "other" and are included in `int6_cats={"mlp", "attn", "other"}`, so they get int6 quantization (clip=31 since they're not "mlp").

### Why int5 for MLP specifically?
- MLP matrices are the LARGEST tensors (512×1536 = 786K params each, 2 per layer)
- They constitute ~60% of total model parameters
- Reducing from int6→int5 on MLPs alone saves significant artifact space
- MLP weights are reportedly more tolerant of aggressive quantization than attention weights
  (attention Q/K projections are more sensitive because small errors get amplified by softmax)

---

## 3. Serialization & Compression Pipeline

### Location: Lines 1253-1300

### Pipeline:
```
1. Save fp32 model checkpoint (torch.save) → "final_model.pt"
2. Magnitude pruning: zero out smallest 2% of weights
3. mixed_quantize_int6: int5 for MLP, int6 for attn, fp16 for tok_emb
4. torch.save({"w": quant_result, "m": quant_meta}, BytesIO) → raw bytes
5. zstd level 22 compress (preferred) OR zlib level 9 (fallback)
6. Write compressed blob → "final_model.int8.ptz"
```

### Step-by-step:

#### Step 2: Magnitude Pruning (lines 1262-1272)
```python
prune_frac = args.prune_frac  # default: 0.02 (2%)
for name, param in base_model.named_parameters():
    if param.ndim == 2 and param.numel() > 65536:
        threshold = torch.quantile(param.abs().float().flatten(), prune_frac)
        mask = param.abs() < threshold
        param.masked_fill_(mask, 0.0)
```
- Zeroes out the smallest 2% of weights by magnitude
- Only applies to large 2D tensors (weight matrices)
- More zeros → more identical bytes → better compression
- Applied BEFORE quantization so the zeros survive the quantize/dequantize round-trip

#### Step 3-5: Quantization + Serialization (lines 1274-1290)
```python
sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn", "other"})
quant_buf = io.BytesIO()
torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
quant_raw = quant_buf.getvalue()
if _COMPRESSOR == "zstd":
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
else:
    quant_blob = zlib.compress(quant_raw, 9)
```

### Key findings:

1. **Standard `torch.save` format** — no custom serialization, no byte grouping, no manual struct packing. They use the vanilla `torch.save` which produces a ZIP+pickle file, then compress the whole thing with zstd/zlib.

2. **zstd level 22** (maximum quality, slow) is the preferred compressor. Falls back to zlib level 9 if `zstandard` package is not installed.

3. **The quantized dict structure**:
   ```python
   {"w": {
       "tok_emb.weight": fp16_tensor,              # passthrough fp16
       "blocks.0.attn_scale": fp32_tensor,          # passthrough control
       "blocks.0.attn.c_q.weight.q": int8_tensor,   # int6 quantized
       "blocks.0.attn.c_q.weight.scale": fp16_tensor, # per-row scale
       "blocks.0.mlp.fc.weight.q": int8_tensor,     # int5 quantized
       "blocks.0.mlp.fc.weight.scale": fp16_tensor,  # per-row scale
       ...
   },
    "m": {
       "tok_emb.weight": "passthrough_fp16",
       "blocks.0.attn.c_q.weight": {"type": "int6"},
       "blocks.0.mlp.fc.weight": {"type": "int5"},
       ...
    }}
   ```

4. **No bit-packing**: Int5 values stay in int8 containers. The compression ratio comes entirely from zstd's ability to exploit the redundant high bits.

5. **Decompression/dequantization** at eval time (lines 1294-1302):
   ```python
   decompressed = zstd.decompress(quant_blob_disk)
   quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
   deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
   base_model.load_state_dict(deq_state, strict=True)
   ```
   Straightforward: decompress → torch.load → reconstruct fp32/bf16 weights from int5/int6 + scales.

---

## 4. How They Fit in 15.87MB

### The actual winning config (from submission.json blurb):
| Param | Value |
|-------|-------|
| NUM_LAYERS | 10 |
| MODEL_DIM | 512 |
| NUM_HEADS | 8 |
| NUM_KV_HEADS | 4 |
| MLP_MULT | 3.0 |
| BIGRAM_VOCAB_SIZE | 16384 |
| BIGRAM_DIM | 64 |
| VOCAB_SIZE | 1024 |
| Compression | zstd level 22 |
| Pruning | 2% magnitude |

**Note**: The code defaults are different (9 layers, bigram 2048). The final run used env var overrides.

### Parameter budget estimate (10 layers, dim=512, MLP 3×):

| Component | Params | Quant | Storage estimate |
|-----------|--------|-------|-----------------|
| tok_emb.weight | 524K | fp16 | 1.05MB raw |
| bigram.embed | 16384×64 = 1.05M | fp16 (<=65536? No, 1M > 65K) | int5/int6 ~0.5MB compressed |
| bigram.proj | 64×512 = 32K | passthrough (<=65536) | 0.06MB |
| Per-layer attn (Q,K,V,proj) | ~786K | int6 (clip=31) | ~0.52MB compressed per layer |
| Per-layer MLP (fc,proj) | ~1.57M | int5 (clip=15) | ~0.84MB compressed per layer |
| Per-layer scalars | ~3K | float32/fp16 | negligible |
| blocks.8.attn.c_k | 131K | fp16 | 0.26MB |
| skip_weights, smear, etc. | ~10K | float32 | negligible |

Per 10 layers: ~5.2MB attn (int6 compressed) + ~8.4MB MLP (int5 compressed) = ~13.6MB
Plus: ~1.05MB tok_emb + ~0.5MB bigram + overhead = ~15.2MB
Plus: torch.save zip/pickle overhead + metadata = ~15.8MB

**The key space savings vs int8 baseline (which would be ~17MB for 10 layers):**
1. **Int5 MLP** saves ~25% on MLP weights (5/8 bits effective → 1.88× compression vs int8's ~1.5×)
2. **Int6 attention** saves ~12.5% on attn weights (6/8 bits effective)
3. **2% magnitude pruning** adds ~2-5% more compression from increased zero frequency
4. **zstd-22 vs zlib-9**: zstd is ~10-15% better at this kind of data

---

## 5. Full Technique Stack Summary

The submission stacks ALL of these for the 1.1385 BPB result:

### Training-time techniques:
| Technique | Config | Lines |
|-----------|--------|-------|
| Muon optimizer | lr=0.02, momentum=0.99, warmup 0.92→0.99/1500 steps | 55-79, 1055-1061 |
| Muon weight decay | WD=0.08 (vs baseline 0.0) | 85, 1060 |
| Adam weight decay | WD=0.02 | 84, 1052 |
| Grad clip | 0.3 | 83, 1200-1201 |
| SmearGate | Learned adjacent-token blending | 607-616, 695, 727 |
| BigramHash embedding | 16384 vocab × 64 dim | 618-643, 690 |
| Attention gate | Per-head sigmoid from first 12 residual dims, zero-init | 564-568, 586-589 |
| 3× MLP expansion | MLP_MULT=3.0 (vs baseline 2.0) | 65, 597 |
| QK gain init | 1.5 (vs baseline 1.0 typical) | 58, 562, 580 |
| Orthogonal weight init | gain=1.0, proj scaled by 1/√(2*num_layers) | 708-720 |
| Warmdown | 3000 iters (vs baseline 1200) | 53, 1105-1114 |
| Batch tokens | 786K (vs baseline 524K) | 55 |
| Train seq len | 2048 (vs baseline 1024) | 56 |
| SWA | Start at lr<50%, every 50 steps | 95-97, 1209-1218, 1243-1251 |

### Post-training techniques:
| Technique | Config | Lines |
|-----------|--------|-------|
| Magnitude pruning | 2% of weights zeroed | 98, 1262-1272 |
| Int5 MLP quantization | clip_range=15, stored as int8 | 99, 376, 344-355 |
| Int6 attention quantization | clip_range=31, stored as int8 | 376, 344-355 |
| FP16 tok_emb passthrough | No quantization on tied embedding | 295-298, 371-373 |
| FP16 blocks.8.attn.c_k | One specific K matrix kept fp16 | 297 |
| zstd-22 compression | Maximum quality compression | 1280-1281 |

### Eval-time techniques:
| Technique | Config | Lines |
|-----------|--------|-------|
| TTT (Test-Time Training) | 5 epochs SGD, lr=0.003, momentum=0.9, freeze 4 layers | 101-106, 771-835, 1304-1317 |
| Sliding window eval | stride=64, batch_seqs=32 | 87-88, 838-915, 1322-1328 |

---

## 6. Critical Observations for Our Implementation

### What to copy immediately:
1. **Int5 MLP quantization** — just change clip_range from 31→15 for MLP params. Zero training cost.
2. **Magnitude pruning 2%** — 6 lines of code. Free compression improvement.
3. **zstd-22** — pip install zstandard, ~10-15% better than zlib-9.
4. **FP16 tok_emb passthrough** — skip quantizing the tied embedding. Costs ~500KB artifact space but recovers most quantization BPB loss.

### What's NOT transferable without more work:
1. **10 layers with 3× MLP** — different architecture, need to verify it fits our budget
2. **BigramHash 16384×64** — adds ~1.05M params, needs int5/int6 quantization to fit
3. **TTT** — requires modifying eval function, 5 epochs SGD at eval time
4. **Sliding window eval** — requires modifying eval to use overlapping windows
5. **SmearGate + Attention gate** — architecture changes

### STE is NOT needed:
The winning config has `INT6_STE=0` (disabled). The int5/int6 quantization noise is handled purely at post-training time. This simplifies our implementation — we just need the post-training `mixed_quantize_int6` function.

### The code defaults vs actual submission:
The code in the PR has conservative defaults (9 layers, bigram 2048, TTT 2 epochs). The actual winning run used env var overrides for 10 layers, bigram 16384, TTT 5 epochs. The submission.json is the source of truth:
- **val_bpb = 1.1385** (not the README's 1.1497)
- **10 layers** (not the code's default 9)
- **BiggramHash 16384×64** (not the code's default 2048×64)
- **TTT 5 epochs at lr=0.003** (not the code's default 2 at 0.004)
- **SWA start 30% every 20 steps** (not the code's default 50% every 50)
