# Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT + CaseOps Tokenizer — val_bpb 1.07462

**val_bpb = 1.07462** (3-seed mean, std 0.00043) | **8×H100 SXM** | max artifact 15,991,629 bytes

## 3-Seed Results

| Seed | Pre-quant EMA | Quantized | Sliding (Track A) | **TTT (Track B)** | Artifact bytes |
|------|---------------|-----------|-------------------|-------------------|----------------|
| 42   | 1.08393       | 1.09482   | 1.07605           | **1.07447**       | 15,991,629     |
| 314  | 1.08467       | 1.09589   | 1.07711           | **1.07521**       | 15,990,248     |
| 999  | 1.08384       | 1.09437   | 1.07552           | **1.07418**       | 15,989,091     |
| **Mean** | **1.08415** | **1.09503** | **1.07623** | **1.07462** | — |
| **Std**  | 0.00037 | 0.00064 | 0.00066 | 0.00043 | — |

Merged SOTA (PR #1493 @bigbag): **1.08100 BPB**. Delta: **−0.00638 BPB = −0.01402 nats/token**.

### Statistical significance

- Our mean: 1.07462, SE = 0.00025 (std / √3)
- Merged SOTA: 1.0810, SE = 0.00012 (from reported std 0.0002 / √3)
- Combined SE: 0.00028
- **z ≈ 22.8, p ≪ 0.0001** ✓ clears p<0.01 threshold

## Contribution

This submission combines two previously-separate directions:

1. **Legal score-first TTT** (merged, PR #1493 @bigbag): 3-layer depth recurrence + parallel residuals + QK-Gain 5.25 + SGD TTT with score-before-update ordering. Fully compliant with Issue #1017 Conditions 1–4.

2. **Lossless CaseOps tokenizer with byte sidecar** (pending, PR #1729 @romeerp): bijective case-folding tokenization (TITLE / ALLCAPS / CAPNEXT reserved tokens) plus a companion `fineweb_val_bytes_*.bin` sidecar that reports original UTF-8 byte counts per token. This enables honest BPB accounting against raw bytes even when the tokenizer inserts control symbols.

**What's novel in this submission:**
- Integrated #1729's CaseOps tokenizer onto #1493's merged legal-TTT stack via a ~25-line byte-sidecar patch to `ValidationData` and the three eval functions (`eval_val`, `eval_val_sliding`, `eval_val_ttt`).
- **Deliberately excluded** the pre-quant TTT component of PR #1735/#1738, which has been community-flagged (see PR #1416 review by @MatoTeziTanka and @dexhunter) as a Condition-3 violation: multi-epoch training on val_tokens without score-first discipline.
- Fixed `load_validation_tokens` to exclude `_bytes_*.bin` files from glob match (prevents double-counting the token stream).

## Compliance (Issue #1017 — all 4 conditions)

- **C1 (Strict causal)**: `flash_attn_3_func(..., causal=True)`; sliding-window eval uses strict prefix only; byte sidecar is pre-computed data (shipped as `fineweb_val_bytes_*.bin`), not runtime state from val tokens.
- **C2 (Full normalized distribution)**: standard softmax over full 8192-vocab Σ; logit softcap `30·tanh(x/30)` applied uniformly to all logits, independent of x_t.
- **C3 (Score-before-update)**: Legal score-first TTT from PR #1493 unchanged. Per chunk: `base_model.eval(); with torch.no_grad(): loss_sum += scored_nll.sum()` → THEN `base_model.train(); loss.backward(); optimizer.step()`. Updates only affect subsequent chunks.
- **C4 (Single pass)**: Each window scored exactly once; no rescoring, no second pass.

Byte-sidecar accounting (from PR #1729) is pre-computed from training data and shipped alongside the val tokens; it is **reference data, not runtime state from val tokens**, so it does not affect C1 or C3.

Additional compliance (no banned mechanisms):
- No SLOT (any variant), no ETLB, no n-gram cache, no pre-quant TTT
- No validation tokens used for adaptation before being scored
- Tokenizer transform is **fully reversible** (see PR #1729 `lossless_caps.py`)
- BPB computed against **original UTF-8 bytes** via sidecar, not transformed token length

## Budget

- **Training**: 588 s (wallclock-capped at 600 s) on 8×H100 SXM
- **Evaluation**: ~497 s total per seed (pre-quant EMA: 7 s + GPTQ: 13 s + quantized eval: 9 s + sliding: 106 s + TTT: 380 s)
- **Artifact**: max 15,991,629 bytes (LZMA-packed code: 16,831 bytes + Brotli-compressed quantized model: 15,971,683 bytes). Under 16,000,000 decimal limit on all 3 seeds.

## Architecture

Inherited from PR #1493:
- 11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap 30
- Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5 activated at step ~2016, frac=0.35)
- Parallel residuals from layer 7: attention and MLP operate on same pre-residual input
- Skip gates (sigmoid-gated U-Net connections)
- MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps), AdamW for embeddings/scalars
- Full-Hessian GPTQ with SDClip: k=12.85 (int6 matrices), k=20.0 (int8 embeddings)
- Byte-shuffle + Brotli-11 compression
- EMA decay 0.9965, weight decay (Muon 0.095, embed 0.085, Adam 0.02), warmdown frac 0.72

Changes from #1493:
- **Tokenizer**: `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved` from `romeerp/parameter-golf-caseops-v1` (instead of default SP8192)
- **Byte counting**: reads `fineweb_val_bytes_*.bin` sidecar when present, uses it for BPB computation instead of LUT-based accounting (~25-line patch)
- **Data-loading filter**: `load_validation_tokens` now excludes `_bytes_` filenames from the glob match

## Reproduction

### 1. Install

```bash
pip install torch flash-attn sentencepiece brotli huggingface-hub numpy tqdm
```

### 2. Download CaseOps data + tokenizer

```bash
cd parameter-golf
# Uses PR #1729's modified downloader which accepts suffixed variant names;
# or apply the one-line patch to data/cached_challenge_fineweb.py:
#   if name.startswith("sp"): return f"fineweb10B_{name}"
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
  python3 data/cached_challenge_fineweb.py \
    --variant sp8192_lossless_caps_caseops_v1_reserved \
    --train-shards 80
```

### 3. Rename / symlink to expected paths

```bash
cd data/datasets
mv fineweb10B_sp8192_lossless_caps_caseops_v1_reserved fineweb10B_sp8192
cd ../tokenizers
mv fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model fineweb_8192_bpe.model
mv fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.vocab fineweb_8192_bpe.vocab
```

### 4. Run

```bash
SEED=42 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_LegalTTT_CaseOps/train_gpt.py
```

Repeat with SEED=314 and SEED=999 for 3-seed validation.

## Attribution

- **@bigbag** — PR #1493 (merged SOTA): the entire base stack — SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal score-first TTT
- **@romeerp** — PR #1729 (pending): CaseOps lossless-case tokenizer and byte sidecar design. This submission adopts the tokenizer + sidecar components only.
- **@clarkkev** — PR #1394: SP8192 base stack, GPTQ SDClip, int6 matrices / int8 embeddings, MuonEq-R, SP8192 tokenizer
- **@dexhunter** — PR #1331, #1437, #1413: 3-layer depth recurrence, QK-Gain variants
- **@Robby955** — PR #1412: parallel residuals (Hessian-aware SDClip lineage)
- **@msisovic** — PR #1204: mini depth recurrence precursor
- **@abaybektursun** — PR #549, #1019: legal score-first TTT precedent, GPTQ-XSA lineage
- **@Christopher-Lee-McClendon** — PR #461: LoRA TTT framework
- **@stukenov** — PR #1364 (pending): pre-quant AdamW TTT concept (deliberately not adopted, see compliance discussion in PR #1416)
- **@X-Abhishek-X** — PR #1445: hyperparameter tuning
- **@MatoTeziTanka**, **@dexhunter** — PR #1416 review: the compliance analysis that guided our decision to exclude pre-quant TTT from this submission
