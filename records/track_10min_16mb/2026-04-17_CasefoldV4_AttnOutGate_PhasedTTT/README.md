# Record: Casefold V4 + Attention Output Gate + Multi-Phase Global SGD TTT

**val_bpb: 1.05733** (3-seed mean, std 0.00035) | **3.04721 nats** | **~15.21 MB** | 8xH100 SXM, 600s | Phased TTT

## Summary

Stacks per-head **Attention Output Gate** (from PR #1667 @MarioPaerle) on top of our Casefold V4 + Multi-Phase Global SGD TTT record (PR #1670). The gate is weight-initialized to zero (identity at init) and adds 1,056 parameters total (12 x 8 heads x 11 layers). Combined with SmearGate (input-dependent per-channel mixer), these architectural additions are orthogonal to the casefold tokenizer and the phased TTT protocol, yielding a clean -0.00237 BPB improvement over PR #1670.

**Note:** Casefold tokenizer normalization is a novel technique pending organizer review at Issue #1604. The tokenizer itself is retrained from scratch on casefolded data -- it is NOT a modified version of the standard SP8192 tokenizer. This submission is offered for evaluation under that pending ruling. The Attention Output Gate and SmearGate are pure architectural additions and do not depend on Issue #1604.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

### Core Results

| Seed | Steps | ms/step | Pre-TTT BPB | **Post-TTT BPB** | TTT gain | TTT time | Artifact |
|------|-------|---------|-------------|------------------|----------|----------|----------|
| 42   | 4902  | 121.6   | 1.06633     | **1.05693**      | -0.00940 | 351s     | 15,936,269 |
| 0    | 4883  | 122.1   | 1.06674     | **1.05730**      | -0.00944 | 347s     | 15,937,514 |
| 1234 | 4906  | 121.5   | 1.06714     | **1.05777**      | -0.00937 | 307s     | 15,938,772 |
| **Mean** | **4897** | **121.7** | **1.06674** | **1.05733** | **-0.00940** | **335s** | **15,937,518** |
| **Std** | | | | **0.00035** | | | |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Quantized BPB | Post-TTT BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|-------------|---------------|--------------|-----------------|-----------|-----------------|------------|-----------|
| 42   | 1.05634     | 1.06633       | 1.05693      | 3.04604         | 124,826 B | 15,936,269 B    | 596.1s     | 350.9s    |
| 0    | 1.05652     | 1.06674       | 1.05730      | 3.04712         | 124,826 B | 15,937,514 B    | 596.1s     | 347.3s    |
| 1234 | 1.05707     | 1.06714       | 1.05777      | 3.04846         | 124,826 B | 15,938,772 B    | 596.2s     | 306.7s    |

### Record Comparison

| Submission | val_bpb | val_loss (nats) | Delta BPB | Delta nats |
|------------|---------|-----------------|-----------|------------|
| Merged SOTA (PR #1493) | 1.08100 | - | - | - |
| PR #1530 @samacqua | 1.07336 | - | - | - |
| PR #1585 @codemath3000 (casefold leader) | 1.06390 | - | - | - |
| PR #1667 @MarioPaerle (AttnOutGate + SmearGate) | 1.07139 | - | - | - |
| PR #1670 @dexhunter (casefold v4 + phased TTT) | 1.05970 | 3.05401 | - | - |
| **This (3-seed)** | **1.05733** | **3.04721** | **-0.00657 vs #1585** | **-0.01697 vs #1585** |

Clears the 0.005-nat record threshold vs the casefold leader (PR #1585) by 3.4x. Improves on PR #1670 by -0.00237 BPB (-0.00680 nats).

## Key Innovations

### 1. Attention Output Gate (from PR #1667 @MarioPaerle)

Lightweight per-head multiplicative gate on the attention output. Weight-initialized to zero (so at init, all heads pass through at scale 1.0). Activated in the inline-safe path with `.contiguous()` barriers so it works under fullgraph torch.compile:

```python
def _apply_attn_out_gate_inline(y, x_orig, gate_w):
    """Inline-safe version: .contiguous() barriers prevent over-aggressive kernel fusion."""
    gate_in = x_orig[:, :, :12].contiguous()
    gate = (2.0 * torch.sigmoid(F.linear(gate_in, gate_w.to(gate_in.dtype)))).contiguous()
    return y * gate.unsqueeze(-1)
```

- Total new parameters: 12 x 8 heads = 96 weights per layer x 11 layers = **1,056 parameters**
- Applied in all three attention paths: standard, parallel-residual, and depth-recurrent
- Negligible throughput cost (<2%)

### 2. SmearGate (input-dependent per-channel mixer)

Input-dependent SmearGate applied at the residual stream before attention:

```python
def _apply_smear_gate_inline(x, smear_w, smear_lambda):
    prev_x = torch.zeros_like(x)
    prev_x[:, 1:] = x[:, :-1]
    gate_in = x[:, :, :12].contiguous()
    gate = torch.sigmoid(F.linear(gate_in, smear_w.to(x.dtype).unsqueeze(0))).contiguous()
    return x + smear_lambda.to(x.dtype) * gate * prev_x
```

- Total new parameters: 12 gate weights + 1 scalar lambda = **13 parameters**
- Zero-initialized smear_lambda (so at init, this is exactly the residual stream)

### 3. Casefold V4 Tokenizer (from PR #1670)

All input text is lowercased (casefolded) offline before SP8192 BPE retraining. Both train and validation shards are retokenized with the casefolded tokenizer, and BOS tokens are preserved. Byte-level BPB is computed over the original (non-casefolded) validation bytes through the sentencepiece piece table.

### 4. Multi-Phase Global SGD TTT (from PR #1670 / PR #1610 concept)

Score-first SGD adaptation on 2000 prefix documents split into 3 phases (boundaries [666, 1333, 2000]). Each phase fully scores its prefix under `torch.no_grad()` before any SGD update.

## Changes from PR #1670 (Casefold V4 baseline)

| Aspect | PR #1670 (base) | This submission |
|--------|----------------|-----------------|
| AttnOutGate | Off | **On (width=12, per-head, all 11 layers, zero-init)** |
| SmearGate | Off | **On (width=12, zero-init lambda)** |
| val_bpb | 1.05970 | **1.05733 (-0.00237)** |
| val_loss (nats) | 3.05401 | 3.04721 (-0.00680) |
| Artifact | ~15.20 MB | ~15.21 MB |
| Tokenizer | Casefold V4 (retrained SP8192 on lowercased text) | Same |
| TTT | Multi-Phase Global SGD (3 phases, 2000 prefix docs) | Same |
| Code size (uncompressed) | 122,604 B | 124,826 B |
| Code size (compressed) | ~28 KB | 28,060 B |

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at step frac=0.35). Parallel residuals from layer 8. Skip gates (sigmoid-gated U-Net connections). EMA decay 0.9965.

**New this submission:** Per-head Attention Output Gate (12 x 8 heads per layer, zero-init, 11 layers). Residual-stream SmearGate (width 12, zero-init lambda).

### Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps, momentum 0.97), AdamW for embeddings/scalars. Gradient clip 0.3. ~4897 steps in 596s on 8xH100 SXM. Linear warmdown over final 75% of training.

### Quantization

Full-Hessian GPTQ with SDClip: int6 for attention/MLP matrices (clip=12.85 sigma for attention, 12.0 sigma for MLP), int7 for token embeddings (clip=15.0 sigma). Brotli-11 compression. Trimmed GPTQ (reserve=4s, calibration=16 batches). The new AttnOutGate and SmearGate parameters (all scalar-like) are kept in float16 passthrough.

### TTT (Test-Time Training)

Multi-Phase Global SGD with score-first ordering:
- 3 phases, each adapting on a growing prefix of 2000 validation documents
- Phase boundaries: [666, 1333, 2000] documents
- Per phase: score all sliding windows under `torch.no_grad()`, then SGD update
- SGD: lr=0.001, momentum=0.9, gradient clipping at 1.0
- Plus per-doc LoRA TTT (rank=96, lr=0.0001, chunk=48, 64-batch) for the suffix documents
- Total TTT eval time: ~335s (within 600s eval budget)

## Rule Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only. AttnOutGate and SmearGate are both purely positional-local -- AttnOutGate multiplies the attention output by a sigmoid of the current token's first-12 channels, SmearGate mixes the current token with the previous token (strictly backward-looking).
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing. Gates modulate hidden states only, not logits.
- **Condition 3 (Score before update):** Each phase fully scored under `torch.no_grad()` BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once per phase. Final scores from last phase only.

**Casefold tokenizer normalization:** Novel technique. The tokenizer is retrained on casefolded (lowercased) text. Organizer review is pending at Issue #1604. The technique does not violate any of the four conditions above -- it only changes the tokenizer vocabulary, not the scoring or adaptation procedure. The byte-level BPB computation remains correct: each sentencepiece token maps to its constituent bytes via the piece table, and BPB is computed over all bytes in the validation set.

**Attention Output Gate / SmearGate:** Pure architectural additions (training-time learned parameters). No eval-time effect beyond the trained weights. Fully legal under all Issue #1017 conditions; analogous gating constructs have precedent in SmearGate (modded-nanogpt), skip gates (PR #549 family), and parallel-lane gating (PR #1204 family).

Additional compliance:
- No SLOT (standard or causal)
- No pre-quant TTT on val data (model quantized once during training, TTT adapts at eval time)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes on all 3 seeds (max 15,938,772 B)
- Training under 600s on all seeds (596.1-596.2s actual)
- Eval (phased TTT) under 600s on all seeds (~307-351s actual)

## Requirements

- Python >= 3.12
- PyTorch >= 2.9.1
- flash-attn-3
- brotli
- sentencepiece

## Run Command

```bash
# Install dependencies
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Prepare casefolded data (offline, before training)
# Casefolded shards expected at DATA_DIR
# Contains: datasets/fineweb10B_sp8192/*.bin, tokenizers/fineweb_8192_bpe.model

# 3-seed evaluation loop
for SEED in 42 0 1234; do
  DATA_DIR=/path/to/casefold_data/ SEED=$SEED \
    ATTN_OUT_GATE=1 SMEAR_GATE=1 \
    PHASED_TTT_ENABLED=1 PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=2000 \
    GLOBAL_TTT_LR=0.001 GLOBAL_TTT_MOMENTUM=0.9 GLOBAL_TTT_GRAD_CLIP=1.0 \
    GLOBAL_TTT_CHUNK_TOKENS=32768 GLOBAL_TTT_BATCH_SEQS=32 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${SEED}.log
done
```

## Lineage

PR #1530 (@samacqua) -> PR #1626 (@dexhunter, multi-phase SGD TTT) -> PR #1670 (@dexhunter, casefold v4 + phased TTT) -> this PR (+ AttnOutGate from PR #1667 @MarioPaerle + SmearGate)

## Credits

- **@samacqua** -- PR #1530 base architecture (11L/512d/4x MLP, depth recurrence, parallel residuals, MuonEq-R, GPTQ SDClip, VarLen attention, fused MLP)
- **@MarioPaerle** -- Attention Output Gate (PR #1667), SmearGate reintroduction to parameter-golf
- **@kellerjordan** -- SmearGate concept (originally from modded-nanogpt)
- **@mikeapedia** -- Casefold tokenizer concept (PR #1578)
- **@romeerp** -- Phased TTT concept (PR #1610)
- **@abaybektursun** -- Score-first TTT framework (PR #549, merged precedent)
- **@dexhunter** -- Casefold V4 retokenization + BOS fix, multi-phase global SGD TTT, trimmed GPTQ tuning, inline-safe gate implementation compatible with fullgraph torch.compile

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed0.log`
- `train_seed1234.log`
