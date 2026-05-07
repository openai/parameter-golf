# Notable: SP8192 + 3-Layer Recurrence + Parallel Residuals - 5-Seed Quantization Reference and SDClip Ablations

**val_bpb = 1.08181** (5-seed mean, std 0.00052) | **~15.99 MB** | 8xH100 SXM

## Summary

This submission does not claim a new SOTA technique; it packages a clean 5-seed reference for the current SP8192 / 3-layer recurrence / parallel residual stack near the leaderboard frontier (merged SOTA is PR #1493 @bigbag at 1.0810; our 5-seed mean is 1.08181). Across all five seeds, the post-quantized artifact improves over the pre-quant checkpoint, which we report as an empirical observation rather than a causal claim. We also include unsuccessful SDClip variants (per-family and depth-split schedules) so future submissions have a documented negative baseline for quantization tuning under the 16 MB constraint.

## 5-Seed Results

| Seed | Pre-Quant BPB | Post-Quant BPB | Delta (post - pre) | Artifact bytes |
|------|---------------|----------------|---------------------|----------------|
| 42   | 1.08684927    | 1.08221297     | **-0.00463630**     | 15,991,025     |
| 314  | 1.08746825    | 1.08207213     | **-0.00539612**     | 15,992,260     |
| 999  | 1.08807118    | 1.08233139     | **-0.00573979**     | 15,988,191     |
| 1234 | 1.08649034    | 1.08093918     | **-0.00555116**     | 15,988,803     |
| 2025 | 1.08741814    | 1.08148221     | **-0.00593593**     | 15,986,880     |
| **Mean** | **1.08725944** | **1.08180758** | **-0.00545186** | **15,989,432** |
| **Std (pop)** | 0.000545 | **0.000523** | 0.000446 | 1,949 |

All 5 seeds satisfy `post_quant < pre_quant` and all 5 artifacts clear the 16,000,000 byte decimal cap with 7.7 kB to 13.1 kB of headroom per seed.

## 1. Post-Quantization Dynamics (Empirical Observation)

Across all five validation seeds, post-quantization BPB comes out below pre-quantization BPB by a mean of **0.00545 BPB** (range 0.00464 to 0.00594), with delta standard deviation **0.00045 BPB**. The int6 GPTQ step with per-family SDClip sigmas (`MLP=12.85`, `ATTN=13.0`, `EMBED=20.0`) does not introduce the usual quantization penalty on this stack.

We report this as an empirical observation, not a causal mechanism claim. It is directionally consistent with the following prior work, which we cite for context rather than as a theoretical basis we derive from:

- **QReg** (Bhatt et al., arXiv [2206.12372](https://arxiv.org/abs/2206.12372)) — quantization noise as a regularizer on loss landscape curvature.
- **QGen** (arXiv [2404.11769](https://arxiv.org/abs/2404.11769)) — QAT loss-landscape flattening that improves generalization.
- **Can Less Precise Be More Reliable?** (arXiv [2509.21173](https://arxiv.org/abs/2509.21173)) — post-training-quantization study reporting post-quant val_loss improvements in a sizeable fraction of configurations.

Section 3 below includes a negative-result single-seed run (Run 7c Deep40): loosening SDClip on the deepest 40 percent of MLP layers (sigma 12.0 -> 14.0) inverts the post - pre sign to +0.01102 BPB and degrades total BPB by +0.01556 vs the 5-seed mean. Readers should treat this as evidence that the observed sign depends on specific clip choices, not that tight clipping is guaranteed to produce improvement on other stacks.

**BPB accounting defense**: all numbers above are evaluated strictly via canonical SP8192 utf-8 byte accounting against raw source text. This submission is entirely unaffected by bugs observed in recent alternate-architecture submissions (e.g., the ~+0.18 BPB offset in PR #1687).

## 2. Headline Metrics

- `val_bpb` 5-seed mean: **1.08180758**
- `val_bpb_std` (population): **0.00052320**
- Best seed (1234): **1.08093918**
- Worst seed (42): **1.08221297**
- Mean artifact size: **15,989,432** bytes (decimal; all seeds clear 16,000,000 cap)
- Training time: under 600s on all 5 seeds (8xH100 SXM)
- Eval time: under 600s on all 5 seeds (sliding-window only; **no TTT**)
- PyTorch 2.11.0+cu130, FlashAttention 3, Brotli-11 compression

## 3. Adaptive Clipping Ablations

The submission recipe (Run 5 stack) was selected after a sequence of single-seed per-layer GPTQ clip-sigma experiments. All ablations were run on the same underlying training stack and seed (1337) unless noted. The decision was to ship the uniform SDClip stack (this submission) rather than per-layer variants, because the per-layer variants either failed the 16,000,000 byte cap or regressed quality.

| Run  | Clip configuration                                           | Single-seed post-quant BPB | Artifact bytes | Verdict         |
|------|--------------------------------------------------------------|-----------------------------|----------------|-----------------|
| 5    | MLP uniform 12.85, ATTN 13.0, EMBED 20.0, EMBED_BITS=8       | 1.08089856                  | 15,988,676     | Submitted (Run 5 stack) |
| 6    | Run 5 + EMBED_BITS=7, EMBED 15.0                             | 1.08220197                  | 15,685,954     | Quality regressed |
| 7    | Per-family per-layer clip: MLP 12.0, ATTN 13.0, EMBED 20.0   | 1.07938099                  | 16,227,236     | Failed size cap by 227,236 bytes |
| 7b   | Uniform relax: MLP 12.3, ATTN 13.0, EMBED 20.0               | 1.08050593                  | 16,144,357     | Failed size cap by 144,357 bytes |
| 7c   | Split MLP: shallow 12.0 + deepest 40 percent at 14.0         | 1.09736695                  | 16,005,384     | Failed both gates; regularizer inverted (see Section 1) |
| 8    | Run 7 (per-family) + EMBED_BITS=7, EMBED 20.0                | 1.08368365                  | 15,697,038     | Size recovered, quality non-additively worse (+0.00279) |

Attribution for this ablation direction: **SpinQuant** (Liu et al., arXiv [2405.16406](https://arxiv.org/abs/2405.16406)) for motivating per-family adaptive clipping variance. Our results clarify that, without Hadamard rotation, tightening MLP clipping below roughly sigma 12.85 pushes the quantized weight distribution into a regime where brotli compression blows past the 16,000,000 decimal cap. This clarifies the size-limit breaches encountered by other submissions (e.g., PR #1695) when clip sigmas tuned on non-rotated weights are reused against rotated distributions.

## 4. Architecture and Training (Lineage from PR #1394 @clarkkev)

**Architecture (inherited, unchanged from PR #1394):** 11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. **3-layer depth recurrence** loops layers 3-5 (encoder [0,1,2,3,4,5,3,4], decoder [5,3,4,5,6,7,8,9,10]; activated at step fraction 0.35). Yields 17 virtual layers from 11 physical. **Parallel residuals from layer 7** (GPT-J style: attention and MLP share the pre-residual input).

**Training (our deltas on top of PR #1394 stack):**

- `QK_GAIN_INIT=5.25` (learnable per-head query scaling; PR #1413 @dexhunter and community ablations)
- `MATRIX_LR=0.026` (up from 0.022)
- `WARMDOWN_FRAC=0.75` (up from 0.72)
- `MUON_WD=0.095`
- `EMA_DECAY=0.9965`
- `PARALLEL_RESIDUAL_START=7`
- `LOOP_START=3`, `ENABLE_LOOPING_AT=0.35`
- No TTT, no SpinQuant, no MP-SGD-TTT, no Casefold, no SLOT, no n-gram cache

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps) for matrices, AdamW for embeddings and scalars. 4550-ish training steps in under 600s on 8xH100 SXM. Linear warmdown to LR=0 over the final 75 percent of training. EMA decay 0.9965.

## 5. Quantization Recipe

Standard GPTQ + SDClip pipeline inherited from PR #1394:

- Full-Hessian GPTQ with SDClip per-family sigmas: `clip = k * std(row)`.
- Sigmas: MLP `k=12.85`, attention `k=13.0`, token embeddings `k=20.0`.
- int6 for attention and MLP matrices, int8 for token embeddings.
- Byte-shuffle permutation + Brotli-11 compression.
- Zero selective pruning; the model fits natively under the 16,000,000 byte decimal cap on every seed.

## Compliance

Per Issue #1017 and repo guidance:

- **Train under 600s on 8xH100 SXM**: satisfied on all 5 seeds.
- **Eval under 600s**: satisfied on all 5 seeds. Eval is pure sliding-window over the full validation set; no TTT-style adaptation is invoked.
- **Artifact under 16,000,000 bytes (decimal)**: satisfied on all 5 seeds, verified by the `total_submission_size_quantized_brotli_bytes` field in each per-seed `summary.json` (model bytes + code bytes); the largest seed measured 15,992,260 bytes, leaving 7,740 bytes of headroom.
- **No SLOT** (standard or causal).
- **No pre-quantization TTT** on validation data.
- **No eval-time TTT / MP-SGD-TTT / score-first TTT / LoRA TTT**.
- **No ETLB** (eval-time logit bias).
- **No n-gram cache or tilt.**
- **No casefold, no custom text normalization beyond the default SentencePiece nmt_nfkc**.
- **5-seed run across seeds 42, 314, 999, 1234, 2025**; `three_seeds=true` flag is retained in `submission.json` for schema compatibility with PR #1493 despite the 5-seed execution (the flag indicates "at least 3 seeds reported", not "exactly 3 seeds").

## Reproduction

```bash
# from repo root
REPO_DIR=$PWD
RECORD_DIR=$REPO_DIR/records/track_10min_16mb/2026-04-19_SP8192_3LayerRecur_ParResid_QK525_NoTTT

pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu130 torch
pip install --no-cache-dir "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
pip install brotli
pip install -r requirements.txt

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

cd "$RECORD_DIR"
for SEED in 42 314 999 1234 2025; do
  DATA_DIR="$REPO_DIR/data" \
  SEED=$SEED \
  QK_GAIN_INIT=5.25 MATRIX_LR=0.026 WARMDOWN_FRAC=0.75 \
  MUON_WD=0.095 EMA_DECAY=0.9965 \
  PARALLEL_RESIDUAL_START=7 LOOP_START=3 ENABLE_LOOPING_AT=0.35 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py \
    > "train_seed${SEED}.log" 2>&1
done
```

## Credits

- **@clarkkev** — SP8192 + GPTQ SDClip + MuonEq-R + depth recurrence baseline (PR #1394).
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437); validated QK-gain 5 direction (PR #1413).
- **@Robby955** — parallel residuals on SP8192 (PR #1412).
- **@msisovic** — parallel residuals concept (PR #1204).
- **@X-Abhishek-X** — hyperparameter tuning direction (PR #1445).
- Public merged SOTA reference: **@bigbag** (PR #1493, 1.0810 BPB with Legal TTT).

## Acknowledgements

Thanks to OpenAI's Advanced Competitor program ($500 compute credit via RunPod), which funded the Phase 1-4 reproduction, the adaptive clipping ablations, and the 5-seed validation reported here.
