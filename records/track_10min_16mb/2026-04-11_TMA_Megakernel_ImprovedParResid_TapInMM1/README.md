# Record: TMA Megakernel + Improved Parallel Residuals + Tap-In min_match=1

**val_bpb = 1.07636** (3-seed mean, std 0.0006) | **~15.97 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | **TTT BPB** | val_loss (nats) | Artifact |
|------|-------------|-------------|-----------------|----------|
| 42   | 1.07856 | **1.07703** | 2.78208 | 15,961,726 |
| 1337 | 1.07727 | **1.07586** | 2.77907 | 15,964,616 |
| 2024 | 1.07833 | **1.07619** | 2.77990 | 15,970,213 |
| **Mean** | **1.07805** | **1.07636** | **2.78035** | **15,965,518** |

Merged SOTA (PR #1493): **1.0810 BPP** / **2.78932 nats**. Delta: **-0.00897 nats**. Clears 0.005-nat threshold by 80%.

## Novel Contributions

1. **TMA Megakernel** — Triton Hopper TMA fused MLP forward: `fc -> leaky_relu(0.5) -> square` in a single persistent kernel with async G->S transfers. Avoids materializing the 403MB intermediate activation. +10.5% training throughput = ~200 extra training steps in 600s.

2. **Tap-In min_match=1 (Unigram Matching)** — Lowered Tap-In V4/V6's minimum match length from 3 to 1 token. Fires at 21% of positions (vs 1.7% at min_match=3). Derived from a local CPU loss analysis showing 77% of tokens are repeats and the entropy 3-5 bucket (51.6% of NLL) is where the model is uncertain but the answer exists in local context. The simplest possible retrieval — "this token appeared before, predict what followed it" — is the most powerful. Nobody else in the competition uses min_match=1.

3. **Improved Parallel Residuals** — Ported from msisovic's PR #1529. Fuller parallel residual formulation with two-lane decoder routing, lane0-only skip connections, and final-lane mean aggregation.

## Architecture

SP8192, 11L/512d/8H/4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: layers 3-5 looped 2x (17 virtual layers from 11 physical, activated at frac=0.35). Improved parallel residuals from layer 8. Skip gates (sigmoid-gated U-Net connections). Per-pass loop embeddings (3 zero-init learned vectors). Parameter banking (contiguous weight banks for batched Muon).

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps, momentum=0.97), AdamW for embeddings/scalars (WD=0.095). ~4700 steps in 587s on 8xH100 SXM. Linear warmdown to LR=0 over final 66.7% of training. EMA decay 0.997. QK-Gain 5.0.

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)` (k=12.85). int6 for attention/MLP matrices, int8 for token embeddings. Brotli compression. All artifacts under 16,000,000 bytes.

## Eval-Time Stack

1. **Sliding Window** (stride=64): causal overlapping windows for extended context.
2. **Score-First TTT**: SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, cosine LR decay. All GPUs iterate every chunk, TTT grads all-reduced before each step, last chunk skipped. ~400-460s.
3. **Tap-In V4/V6** (min_match=1): C++ phrase matcher with unigram matching, top-K=1000 gating, match-length scaling, vote-count scaling, cross-window matching (w=0.06). ~135s.
4. **Eval Hash Embedding**: Zero-init 16384x512 bigram hash embedding trained through TTT loop with 10x LR multiplier.

## Compliance (Track B)

Per Issue #1017:
- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Tap-In matcher uses strict prefix (`p+1 < t`). Hash embedding uses `prev_token * 2039 + curr_token` (prefix only).
- **Condition 2 (Normalized distribution):** Tap-In probability mixing sums to 1 by construction: `p_new(k) = (1-w)*p(k) + w*1[k==hint]`. Standard softmax for TTT.
- **Condition 3 (Score before update):** Each TTT chunk fully scored under `torch.no_grad()` before any SGD update. Tap-In linked list updated AFTER scoring. Last TTT chunk never trained on.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:
- No SLOT, no pre-quant TTT, no n-gram caches
- GPTQ Hessians collected from `train_loader.next_batch()` (zero val-data exposure)
- Model deserialized from `.int6.ptz` before TTT/Tap-In touch anything
- All artifacts under 16,000,000 bytes on all 3 seeds
- Training under 600s on all 3 seeds (~587s)
- Eval (sliding + TTT + Tap-In) under 600s on all 3 seeds

## Reproduction

```bash
pip install brotli sentencepiece ninja
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 TTT_ENABLED=1 VOCAB_SIZE=8192 RECORD_MODE=1 \
  MUON_MOMENTUM=0.97 LOOP_PASS_EMB_ENABLED=1 \
  TAPIN_ENABLED=1 TAPIN_CROSS_ENABLED=1 TAPIN_MIN_MATCH=1 TAPIN_MIX_W=0.01 \
  TAPIN_TOP_K=1000 TAPIN_BOOST_PER_MATCH=0.02 TAPIN_CROSS_W=0.06 \
  EVAL_HASH_ENABLED=1 TTT_LR=0.005 \
  torchrun --standalone --nproc_per_node=8 train_gpt_mega.py
```

## Credits

- **@msisovic** — Improved parallel residuals (PR #1529), depth recurrence (PR #1204)
- **@abaybektursun** — Tap-In V4/V6 framework (PR #1518, #1420), triple loop + parallel residuals (PR #1420), score-first TTT (PR #549)
- **@clarkkev** — SP8192 + SDClip + MuonEq-R (PR #1394)
- **@dexhunter** — Legal score-first TTT on SP8192 (PR #1413), 3-layer depth recurrence (PR #1331)
- **@EthanYangTW** — Parameter banking + batched Muon (PR #1523)
- **@resouer** — Eval-time hash embedding (PR #1460)
- **@bigbag** — QK-Gain tuning, hyperparameter bundle (PR #1493)
- **@Robby955** — Hessian-aware SDClip analysis (PR #1412)
