# Record: AttnOutGate + SmearGate + Softcap 15 — val_bpb 1.07750 (3-seed mean)

**val_bpb: 1.07750** (3-seed mean, std 0.0006) | **~15.99 MB** | 8×H100 SXM

Beats current SOTA (PR #1493, 1.0810) by **0.00350 BPB** with std 0.0006 → t-statistic ≈ 5.5, p < 0.001 across 3 seeds. Comparable in magnitude to recent record gaps on the leaderboard (e.g., #2→#1 was 0.0012, #3→#2 was 0.0006).

Three additive zero-cost modifications, all fully precedented and reproducible.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | Pre-quant BPB | Quantized BPB | Sliding BPB | TTT BPB | Artifact |
|------|-------|---------------|---------------|-------------|---------|----------|
| 1337 | 4457 | 1.08396 | 1.09737 | 1.07817 | **1.07693** | 15,994,840 |
| 42   | 4459 | 1.08491 | 1.09617 | 1.07934 | **1.07805** | 15,996,097 |
| 2025 | 4450 | 1.08458 | 1.09555 | 1.07873 | **1.07753** | 15,992,597 |
| **Mean** | **4455** | **1.08449** | **1.09636** | **1.07875** | **1.07750** | **15,994,511** |

## Key Changes vs Our Previous Submission (PR #1876, 1.08008 BPB)

Three additive zero-cost modifications:

### 1. AttnOutGate (PR #1667/#1693)
Per-head data-dependent gate on SDPA output, before `out_proj`:
```
out = W_o @ ( SDPA(x) ⊙ 2σ(W_g · x[:, :12]) )
```
- `W_g`: (12 × 8) per layer, zero-init → 2σ(0) = 1 (transparent at init)
- 8 heads × 12 width × 11 layers = **1,056 extra params** (~1KB at fp16)
- Lets each head dynamically suppress noise per-token
- Routes through scalar AdamW (added `attn_gate` to CONTROL_TENSOR_NAME_PATTERNS)

### 2. SmearGate (PR #1667 + PR #1851 BOS-fix)
Forward-1-token residual mixer at embedding lane:
```
x_t ← x_t + λ · σ(W · x_t[:12]) · x_{t-1}     (for t ≥ 1, identity at t=0)
```
- `W`: (12 × 1) and `λ`: scalar — both zero-init
- **Total: 13 extra params** (~26 bytes)
- BOS-fix prevents cross-document leakage during packed training: gate is masked to 0 where `input_ids == BOS_TOKEN_ID` (default 1)
- Routes through scalar AdamW

### 3. Lower logit softcap 30 → 15 (Modded-NanoGPT record #18)
Single hyperparameter change:
```
logits = 15 * tanh(logits / 15)    (was 30 * tanh(logits / 30))
```
- Tighter cap engages tanh's saturating region
- Smoother loss landscape, prevents extreme overconfidence
- **Single-line change, no params**

## Architecture (unchanged from previous submission)

- SP8192 BPE tokenizer
- 11 layers, dim=512, 8 heads, 4 KV heads (GQA)
- Depth recurrence: layers 3-5 looped 3× (17 virtual layers), enabled at 35%
- XSA on all 11 layers, parallel residuals from layer 7+
- U-Net skip connections with learnable gates
- Tied embeddings, MLP 4× LeakyReLU(0.5)²
- Coprime-stride multi-shard data loader

## Training (unchanged)
- Muon optimizer (5-step NS) for matrices, AdamW for embeds/scalars
- EMA decay 0.9965, 72% warmdown, 20-step warmup + 20-step loop warmup
- Gradient clipping 0.3
- Brotli-11 compression + byte shuffling
- Score-first TTT (SGD, momentum 0.9, LR 0.005, 3 epochs, 32K chunks)
- Full Hessian GPTQ with Cholesky error compensation + actorder
- LZMA code compression (53KB → 19KB)

## What We Tried That Did Not Help

| Technique | Result | Why it failed |
|---|---|---|
| LoRA on recurrence (rank 2/4) | Worse | 10% step loss, artifact over 16MB |
| MTP (Multi-Token Prediction) | Worse | 10.5% step loss, no quality gain |
| QAT weight-snapping during warmdown | Catastrophic | Disrupted Muon's update dynamics |
| Hessian-Aware SDClip (PR #1412) | No change | Per-row Hessian importance too noisy |
| Per-group clip allocation | No change | Group traces are stable but didn't translate |
| Asymmetric sigmoid logit rescale | Worse (+0.001) | Tanh form was already well-tuned |
| nGPT normalization | Excluded after research | Speedup only at 0.5B+ params and 200k+ steps |
| GatedDeltaNet/linear attention | Excluded after research | All "frontier" PRs had byte-accounting bugs |
| Value embeddings | Excluded | Don't fit in 5KB artifact headroom |

## Compliance (Issue #1017 conditions)

### Condition 1 (Strict Causal Dependence)
Causal attention via `flash_attn_func(causal=True)`. AttnOutGate uses position-local input `x_t[:12]` (no leakage). SmearGate is strictly backward-looking (`x_{t-1}`), with BOS-mask preventing cross-document leakage. TTT only incorporates tokens from already-scored chunks.

### Condition 2 (Full Normalized Distribution)
`F.cross_entropy` over full vocab_size logits. Softcap is monotonic (does not mask).

### Condition 3 (Score-Before-Update)
Each TTT chunk scored under `torch.no_grad()` BEFORE any training on it. Model weights at scoring reflect only prior chunks.

### Condition 4 (Single Left-to-Right Pass)
Single `for ci in range(num_chunks)` loop. Each token scored exactly once.

## Credits
- Current SOTA base (PR #1493): @bigbag
- AttnOutGate: @MarioPaerle (PR #1667), @dexhunter (PR #1693)
- SmearGate + BOS-fix: @KoszarskyB / @classiclarryd (modded-nanogpt), @cocohearts + @aquariouseworkman (PR #1851)
- Logit softcap 15: @KoszarskyB (Modded-NanoGPT record #18)
- SP8192 + GPTQ + SDClip: @clarkkev (PR #1394)
- Depth recurrence: @dexhunter (PR #1331, #1437)
- Parallel residuals: @Robby955 (PR #1412), @msisovic (PR #1204)
- Score-first TTT: @abaybektursun (PR #549), @Christopher-Lee-McClendon (PR #461)
- Coprime-stride loader: PR #726 style
- LZMA code compression: PR #1394
