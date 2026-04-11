# Non-record: Pre-Quant AdamW TTT (Compiled) + SP8192 + Depth Recurrence

> **Compliance note:** This submission violates Condition 3 of Issue #1017 (score-before-update). Pre-quant TTT fine-tunes on val tokens before scoring them. Submitted as a technique study, not a leaderboard claim.

**val_bpb = 1.0587** (3-seed mean, std 0.0004) | **~15.5 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | Roundtrip BPB | Artifact |
|------|-------------|---------------|----------|
| 42   | 1.05840     | 1.06847       | 15,477,275 |
| 1337 | 1.05856     | 1.06904       | 15,439,370 |
| 2024 | 1.05912     | 1.06921       | 15,480,770 |
| **Mean** | **1.05869** | **1.06891** | **15,465,805** |
| **Std** | **0.00038** | **0.00037** | |

## Why this is a useful non-record

### 1. Quantifying the Condition 3 boundary

This submission provides a controlled measurement of how much BPB improvement comes from violating Condition 3:

| Configuration | BPB | Source | Measured? |
|---|---:|---|---|
| Post-EMA (no TTT, no GPTQ) | 1.1028 | This submission | Yes |
| **Post-GPTQ sliding (illegal 6-epoch TTT)** | **1.0587** | This submission | Yes |
| Post-GPTQ sliding (no TTT) | ~1.106 | Estimated: post-EMA + ~0.003 GPTQ gap | No |
| Post-GPTQ sliding (legal score-first TTT) | ~1.104 | Estimated from PR #1493 delta (-0.002) | No |

The two measured points bound the illegal TTT contribution at **-0.044 BPB** (post-EMA 1.103 → post-GPTQ sliding 1.059). For comparison, the legal score-first TTT in merged PR #1493 contributes approximately -0.002 BPB (sliding 1.083 → TTT 1.081). This is not an apples-to-apples comparison — the illegal variant uses AdamW for 6 full epochs while the legal variant uses SGD for 3 epochs per chunk, on a different base model — but the order-of-magnitude gap illustrates why Condition 3 is load-bearing.

**On the theoretical ceiling:** Issue #1017 states: *"Corpus-level TTT has a ceiling of approximately 0.0003 bits"* — this refers specifically to the gain from closing the train-val distribution gap, which the author measured as negligible for FineWeb. However, the author also notes that *"a model that undertrained on the training distribution can still benefit from additional learning at test time."* This means legal TTT can legitimately exceed the 0.0003 ceiling if the model hasn't fully converged during training (our 600s-capped model is certainly in this regime). The merged #1493's legal TTT gain of -0.002 BPB is consistent with this — it reflects real undertraining compensation, not memorization.

Our illegal TTT's -0.044 BPB gain, however, is 22x larger than legal TTT on a similar architecture. This magnitude is not explainable by undertraining compensation alone and is consistent with memorization of the validation set. A per-epoch ablation (not performed in this submission) would strengthen this argument: if the gain scales roughly linearly with epoch count rather than saturating quickly, that would be a direct memorization signature.

### 2. Compiled TTT: torch.compile for test-time training

We demonstrate that `torch.compile(dynamic=False, fullgraph=True)` can be applied to TTT models for a **2x speedup** (860s → 426s for 6 epochs). This is safe because:

- TTT operates in train mode with `torch.autocast`
- No `torch.inference_mode()` — avoids rotary cache poisoning (a $60+ lesson from our development)
- Fresh model instance created before TTT (deletes compiled training model, resets dynamo)
- Compilation overhead (~20s) amortized over multiple epochs

This technique applies equally to legal score-first TTT and would reduce eval-time TTT costs.

### 3. Artifact budget engineering under 16MB

With SP8192, fitting under 16MB required careful component analysis:

| Component | Compressed Cost | BPB Benefit | Decision |
|---|---:|---|---|
| BigramHash 2048×128 | +109KB | ~0.001 at SP8192 | **Dropped** — marginal at large vocab |
| VE dim=128 → dim=44 | -340KB | -0.001 | **Shrunk** — optimal via EV analysis |
| VE dim=44 → dim=0 | -150KB | -0.001 | Kept — positive expected value |

We optimized VE dimension by sweeping dims 0-128, measuring compressed artifact size at each, computing pruning probability vs BPB tradeoff, and selecting the dimension that minimized expected BPB accounting for pruning risk. dim=44 gives 0% pruning risk with 39KB margin.

## Compliance Statement

**This submission violates Condition 3 of Issue #1017.** Pre-quant TTT (lines 2417-2455 of `train_gpt.py`) runs 6 AdamW epochs on the full val stream before GPTQ quantization. The same tokens are then scored via sliding window evaluation. No score-before-adapt discipline is implemented. This pattern is structurally identical to the closed PR #1376 and the withdrawn PR #1485 (@ndokutovich acknowledged the violation).

## Key Techniques

1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0) (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence** (L3-5, 14 virtual from 11 physical) (PR #1493 @bigbag)
3. **Parallel Residuals** (L7+, GPT-J style) (PR #1412 @Robby955, PR #1204 @msisovic)
4. **Pre-Quant AdamW TTT** — 6 epochs, `torch.compile` 2x speedup, freeze 2 blocks (PR #1485 @ndokutovich)
5. **QK-Gain 5.25** + MuonEq-R (Polar Express) + EMA 0.9965 + warmdown 72% (PR #1493 @bigbag)

## Architecture

11L × 512d × 8H/4KV, MLP 4× (2048), LeakyReLU(0.5)², Partial RoPE (16/64), LN scale, tied embeddings, softcap=30. Depth recurrence [0,1,2,3,4,5,3,4,5,6,7,8,9,10] = 14 virtual layers. Parallel residuals L7+. XSA all layers. VE dim=44 L9-10. SmearGate.

## Training

MuonEq-R (Polar Express, 4 NS steps) + AdamW. ~5160 steps in 600s on 8×H100 SXM. Linear warmdown to 0 over final 72%. EMA 0.9965. Late QAT at LR scale < 15%.

## Pre-Quant AdamW TTT (VIOLATES CONDITION 3)

Fine-tunes the EMA model on the full validation token stream before GPTQ:

- `torch.compile(dynamic=False, fullgraph=True)` for 2x speedup (426s vs 860s)
- AdamW, lr=0.0005, weight_decay=0.0, cosine decay to lr×0.1
- 6 epochs, freeze first 2 transformer blocks
- Batch: 32 sequences × 2048 tokens, grad clip 1.0
- Fresh model instance (avoids inference_mode rotary cache poisoning)

## Quantization

GPTQ int6 SDClip (k=12.85) + int8 embeddings (k=20.0). 32 AR self-gen calibration sequences. Brotli-11 compression. Zero pruning on all seeds.

## Reproduction

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
VOCAB_SIZE=8192 BIGRAM_VOCAB_SIZE=0 VE_DIM=44 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1394 @clarkkev, PR #1493 @bigbag, PR #1485 @ndokutovich, PR #1412 @Robby955, PR #1204 @msisovic, PR #1285 @dexhunter, PR #549 @abaybektursun
