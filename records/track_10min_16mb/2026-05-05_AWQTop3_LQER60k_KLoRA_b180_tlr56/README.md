# Record candidate: PR #1797 + AWQ-lite top3 + LQER 60k on b180-tlr56 — 3-seed mean 1.06082

**Status update (3-seed): does not clear the 0.005-nat record-acceptance threshold.** Marking this PR as draft.

original single-seed (SEED=0) headline was 1.06043 / 2.32062 nats — promising. The full 3-seed result is 1.06082 BPB mean / 2.32147 nats mean, an improvement of -0.00026 BPB / -0.00056 nats over PR #1855's 3-seed mean (1.06108, 2.32203). that is ~9× below the README's 0.005-nat threshold.

Honest assessment: SEED=0 was a favourable draw from the seed-lottery distribution, not a recipe-level win over PR #1855. The single-seed value was within the noise envelope of either set; the 3-seed mean settles into the noise floor of #1855.

The current SOTA candidate is PR #2135 (3-seed mean 1.05651, -0.00457 BPB / -0.01000 nats vs PR #1855), which clears the threshold by 2× — driven primarily by the token-only n-gram tilt (PR #1145 lineage) which this submission does not include. **Recommend evaluating PR #2135 ahead of this one.**

This PR remains as a draft for documentation of the AWQ-lite + LQER + drop-M LoRA stacking experiment and as the missing H100 multi-seed continuation of PR #1935.

## Results (3-seed, fixed seed set {0, 314, 1234})

| Seed | Steps | ms/step | Train ms | Pre-quant BPB | Quant BPB | Post-TTT BPB | TTT eval s | Artifact bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0    | 4831 | 123.4 | 596,153 | 1.06496396 | 1.07344835 | 1.06043138 | 599.3 | 15,947,372 |
| 314  | 4819 | 123.7 | 596,078 | 1.06524848 | 1.07365670 | 1.06070308 | 572.3 | 15,947,099 |
| 1234 | 4821 | 123.6 | 596,087 | 1.06562877 | 1.07432284 | 1.06132461 | 563.8 | 15,947,670 |
| **Mean** | **4824** | **123.6** | **596,106** | **1.06528** | **1.07381** | **1.06082** | **578.5** | **15,947,380** |

3-seed sample std: 0.00046 BPB / 0.00100 nats.

All seeds within the 16 MB cap (max total 15,982,480) and 600s train + 600s eval lane caps (max train 596.2s, max eval 599.3s).

## Statistical comparison vs PR #1855

| Test | t-stat | df | p (one-tailed) | Verdict at p<0.25 | Clears 0.005-nat? |
|---|---|---|---|---|---|
| Welch's t-test (independent samples, unbiased σ) | -0.41 | ~2 | 0.36 | Does not pass | No |

Improvement of -0.00056 nats is well within the noise envelope of either side (PR #1855 std 0.00090 BPB; ours 0.00046 BPB).

## Recipe (vs PR #1855)

| Lever | PR #1855 | This | Source |
|---|---|---|---|
| QK_GAIN_INIT | 5.25 | 6.0 | b180 lineage tuning |
| TTT_LORA_RANK | 80 | 56 | rank ablation in PR #1935 (inverted-U at 56) |
| TTT_MLP_LORA / K / O | 1 / 1 / 1 | 0 / 1 / 1 | drop M only |
| LQER_BUDGET_BYTES | 80000 | 60000 | uses cap margin freed by AWQ |
| AWQ_LITE_ENABLED | off | 1 | PR #1908 lineage |
| AWQ_LITE_GROUP_TOP_K | n/a | 3 | top-3 saliency 64-col groups → INT8 |
| AWQ_LITE_SKIP_EMBED | n/a | 1 | tok_emb stays at INT7 |
| COMPRESSOR | pergroup | pergroup_lrzip | helper port from PR #1855 lineage |

All other knobs identical to PR #1855.

## What we learned

The recipe stack DID deliver — just not as much as the SEED=0 single-seed implied:

1. AWQ-lite top_k=3 → ~-0.0001 to -0.0002 BPB at this stack on H100, smaller than the ~-0.00052 LUMI claim from the source experiment. The +0.00089 FIXED_SEQ_COMPILE shift assumed for LUMI→H100 transfer did not hold under DYNAMIC eval mode.

2. LQER 80k → 60k → ~-0.0001 BPB with AWQ. Margin freed (~20 KB) was useful but the BPB return was small.

3. Disk inductor cache hits partially across seeds. saved ~57s on TTT compile warmup and ~20s on phase 1 in-timer. cache invalidates more than expected — likely deserialized-weight identity affecting compile graph hashes.

4. PR #2135's n-gram tilt is the dominant lever we don't have. their per-seed pre-TTT BPBs are in the same neighborhood as ours pre-quant; the closed-form per-token logit boost (PR #1145 / PR #1514 lineage) is essentially additive over a stack like this one — porting it would be the single highest-leverage next step.

## Compliance

Inherits from PR #1855 / PR #1797 lineage. AWQ-lite runs in the same calibration-data pass as GPTQ — training shards only — and writes its INT8-promoted column metadata at quantize time. No validation data is touched.

C1 causality, C2 normalization, C3 score-first TTT, C4 single L→R pass: all preserved. CaseOps byte sidecar accounting (PR #1729 / #1736) preserved; `ZERO_PUE_MARKERS=1` ships.

## Hardware

8x H100 80GB HBM3 SXM (RunPod), `vimetoivonen/pgolf:b180-tlr64` image, torch 2.9.1+cu129, FA3 via `flash_attn_interface`. All three seeds within 596.2s train, max 599.3s eval.

## Lineage and credits

- PR #1855 (merged) — base recipe and 9-hparam stack
- PR #1797 — SparseAttnGate + LQER asymmetric rank-r
- PR #1935 (closed) — QK_GAIN=6.0 + TTT_LORA_RANK=56 + drop-M LoRA — this PR is the H100 multi-seed continuation
- PR #1908 — AWQ-lite mixed-precision GPTQ
- PR #1145 / PR #1514 — token-only n-gram tilt (NOT used in this PR; the dominant gap to SOTA)
- PR #2135 — current SOTA candidate (1.05651 3-seed mean); recommend reviewing ahead of this PR
- PR #1729 / PR #1736 — CaseOps lossless tokenizer + byte sidecar

## Files

- `train_gpt.py` — full training/eval script, includes per-group lrzip serialize/deserialize ports from PR #1855 lineage helpers (PGRP magic + LRZI byte autodetect on read), AWQ-lite encoder/decoder paths
- `submission.json` — recipe + per-seed metadata + statistical analysis
- `train_seed0.log`, `train_seed314.log`, `train_seed1234.log` — full per-seed training + TTT eval logs
- `lossless_caps.py`, `prepare_caseops_data.py`, `requirements.txt`, `tokenizers/`
