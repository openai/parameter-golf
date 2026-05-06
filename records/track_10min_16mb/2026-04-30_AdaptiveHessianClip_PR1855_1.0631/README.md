# Record candidate: PR #1855 stack + Adaptive Hessian-Sensitivity GPTQ Clip + TTT_LORA_RANK=56

**val_bpb: 1.06310** (3-seed mean, std 0.00102) | ~15.9 MB | 8×H100 SXM, 600s wallclock | phased TTT eval

**Comparison vs current leaderboard (PR #1855, 1.06108):** **+0.00203 BPB / +0.00444 nats**.

This is a follow-up to **PR [#1689](https://github.com/openai/parameter-golf/pull/1689)**
by the same author, which introduced adaptive Hessian-sensitivity GPTQ
clipping at 1.0822 BPB on the PR #1394 base. The leaderboard has since
advanced through ~20 PRs to PR #1855 at 1.06108. This submission ports the
adaptive-clip technique onto PR #1855's stack to test whether the
sensitivity-driven per-tensor approach holds up against LQER asymmetric
quantization, the heavily-tuned 9-hparam greedy stack, and the rest of
the modern pipeline.

The adaptive Hessian-sensitivity GPTQ clipping technique **eliminates three
hand-tuned hyperparameters** (`MLP_CLIP_SIGMAS=11.5`, `ATTN_CLIP_SIGMAS=13.0`,
`MATRIX_CLIP_SIGMAS=12.85`) from the search space and replaces them with one
automated per-tensor selection driven by the Hessian diagonal magnitude. On
PR #1855's heavily-tuned base it reproduces the hand-tuned result within ~2σ
(t≈2.0 across our 3 seeds vs theirs) at +0.00203 BPB; on a stack that has not
had per-group sigmas tuned, it saves both the search cost and the risk of
overfitting the validation distribution.

A **second technique** — Hessian-sensitivity-driven mixed-precision GPTQ
(int5/int6/int7) — was tested in the same codebase under `MIXED_PRECISION_HESSIAN=1`
and is documented below as a negative result.

## Results

| Seed     | Steps     | ms/step   | Pre-quant val_bpb | Post-quant val_bpb | **Post-TTT val_bpb** | Artifact bytes | Eval time |
|----------|-----------|-----------|-------------------|--------------------|----------------------|----------------|-----------|
| 42       | 4,835     | 123.3     | 1.06498           | 1.07480            | **1.06214**          | 15,905,000     | 592.9 s   |
| 1337     | 4,807     | 124.0     | 1.06711           | 1.07696            | **1.06417**          | 15,918,827     | 521.1 s   |
| 999      | 4,805     | 124.1     | 1.06611           | 1.07570            | **1.06300**          | 15,901,152     | 456.5 s   |
| **Mean** | **4,816** | **123.8** | 1.06607           | 1.07582            | **1.06310**          | 15,908,326     | 523.5 s   |

3-seed std: 0.00102 BPB / 0.00224 nats. All artifacts under 16,000,000 bytes;
all runs stopped at the 600 s wallclock cap.

## Architecture

Inherited from PR #1855 unless marked **new**.

| Component                         | Setting                                                 | Source                                                                                                                                                                                                                                                                         |
|-----------------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Layers                            | 11 (512d, 8 GQA heads, 4 KV heads)                      | Baseline                                                                                                                                                                                                                                                                       |
| MLP                               | 4× (2048) with LeakyReLU(0.5)²                          | [#493](https://github.com/openai/parameter-golf/pull/493)                                                                                                                                                                                                                      |
| Fused MLP kernel                  | LeakyReLU-square Triton                                 | [#1530](https://github.com/openai/parameter-golf/pull/1530)                                                                                                                                                                                                                    |
| Attention                         | Standard FA3, GQA 2:1                                   | Baseline                                                                                                                                                                                                                                                                       |
| XSA                               | All 11 layers                                           | [#478](https://github.com/openai/parameter-golf/pull/478)                                                                                                                                                                                                                      |
| RoPE                              | Partial (16/64 dims) + YaRN                             | [#315](https://github.com/openai/parameter-golf/pull/315)                                                                                                                                                                                                                      |
| LN Scale                          | 1/√(layer+1)                                            | [#315](https://github.com/openai/parameter-golf/pull/315)                                                                                                                                                                                                                      |
| QK Gain init                      | 5.0 (per-head learned)                                  | [#1276](https://github.com/openai/parameter-golf/pull/1276)                                                                                                                                                                                                                    |
| U-Net skips                       | Encoder-decoder skip + skip gates                       | [#289](https://github.com/openai/parameter-golf/pull/289)                                                                                                                                                                                                                      |
| Parallel decoder                  | 2-lane parallel from layer 8+, lane mix learned         | [#1530](https://github.com/openai/parameter-golf/pull/1530)                                                                                                                                                                                                                    |
| Depth recurrence                  | Loop layers 3–5, run 3× once frac ≥ 0.35                | [#1344](https://github.com/openai/parameter-golf/pull/1344)                                                                                                                                                                                                                    |
| Logit softcap                     | 30                                                      | Gemma2-style                                                                                                                                                                                                                                                                   |
| Sparse attention gate             | Narrow head-output gate, gate_window=12                 | [#1787](https://github.com/openai/parameter-golf/pull/1787)                                                                                                                                                                                                                    |
| SmearGate (BOS-fixed)             | Position-mixing gate with `not_bos` mask                | [#1667](https://github.com/openai/parameter-golf/pull/1667) + [#1851](https://github.com/openai/parameter-golf/pull/1851)                                                                                                                                                      |
| Polar-Express Newton-Schulz       | Muon, 5 steps, per-iter minimax tuples                  | [#1344](https://github.com/openai/parameter-golf/pull/1344) → [#1787](https://github.com/openai/parameter-golf/pull/1787)                                                                                                                                                      |
| MIN_LR floor                      | 0.10 (warmdown LR floor)                                | [#1787](https://github.com/openai/parameter-golf/pull/1787)                                                                                                                                                                                                                    |
| Fused softcapped CE Triton kernel | Single-pass training-only                               | [#1787](https://github.com/openai/parameter-golf/pull/1787)                                                                                                                                                                                                                    |
| LQER asymmetric int4              | Rank-4 quant-error correction on top-3 tensors          | [#1797](https://github.com/openai/parameter-golf/pull/1797)                                                                                                                                                                                                                    |
| Per-group compression             | lrzip zpaq + L1 simsort + brotli remainder              | [#1855](https://github.com/openai/parameter-golf/pull/1855)                                                                                                                                                                                                                    |
| Quantization base                 | GPTQ int6 + int7 embed + int8-per-row attn-gate         | [#1394](https://github.com/openai/parameter-golf/pull/1394), [#1586](https://github.com/openai/parameter-golf/pull/1586), [#1736](https://github.com/openai/parameter-golf/pull/1736)                                                                                          |
| Phased TTT                        | 3 cumulative phases, LoRA rank 80→**56**, per-doc reset | concept [#1610](https://github.com/openai/parameter-golf/pull/1610) → [#1626](https://github.com/openai/parameter-golf/pull/1626) → [#1736](https://github.com/openai/parameter-golf/pull/1736); rank 56 from open [#1935](https://github.com/openai/parameter-golf/pull/1935) |
| **GPTQ clip selection**           | **Per-tensor adaptive σ ∈ [6, 24] from H_diag·row_var** | **[#1689](https://github.com/openai/parameter-golf/pull/1689) (this submitter), ported to PR #1855 stack**                                                                                                                                                                    |
| Tokenizer                         | sp8192 lossless caps caseops v1 reserved                | [#1729](https://github.com/openai/parameter-golf/pull/1729)                                                                                                                                                                                                                    |

## Adaptive Hessian-Sensitivity GPTQ Clipping

**Setting up the problem.** GPTQ rounds floating-point weights to a small
integer range (here `bits=6`, range `[-31, 31]`). Each row of each matrix is
scaled by a single fp16 factor `s_row` such that `q_row = round(W_row / s_row)`.
The choice of `s_row` is a tradeoff: tight scales waste bits on outlier rows
(more clipping); loose scales dilute precision in the bulk (more roundoff).
The standard heuristic — used by SDClip in PR #1394 and tuned per-group in PR
#1855 — is `s_row = (clip_sigmas · std(W_row)) / clip_range`, where
`clip_sigmas` is a hyperparameter usually in [3, 20].

PR #1855's 9-hparam greedy search settled on **three** per-group `clip_sigmas`
values:

| Group          | clip_sigmas | Tensor count                              |
|----------------|-------------|-------------------------------------------|
| MLP            | 11.5        | 22 (mlp.fc + mlp.proj)                    |
| Attention      | 13.0        | 44 (c_q, c_k, c_v, proj across 11 layers) |
| Matrix (other) | 12.85       | a few non-bank tensors                    |

These three numbers are themselves an averaging of the per-tensor optimum
under a constrained search. Different stacks (different recurrence patterns,
different LR schedules, different Hessian distributions from training data)
would shift the optimum.

**Adaptive replacement.** This submission replaces the three constants with one
**per-tensor** clip-sigma chosen from each tensor's Hessian sensitivity:

```
sens(name) = max( H_diag(name).mean() · row_var(W(name)),  ε )
log_raw(name) = -0.15 · log(sens(name))
σ(name) = clamp( exp(log_raw(name) + offset),  6.0,  24.0 )
```

The `offset` is determined by binary search such that the numel-weighted
log-average of σ across all matrix tensors equals PR #1855's hand-tuned
log-average:

```
target = (Σ_n  numel(n) · log(base_cs(n)))  /  Σ_n  numel(n)
```

where `base_cs(n)` is the appropriate hand-tuned per-group sigma for tensor
`n`. This guarantees that **the overall compression budget is preserved** —
the average sigma across the model is unchanged from PR #1855, only the
per-tensor distribution shifts according to sensitivity.

**Why `H_diag.mean() × row_var`?** GPTQ minimises the layer-wise output error
`||(W − Ŵ) X||_F²` where `X` is calibration activations. To leading order
this scales with `tr(H · ΔW · ΔW^T)` for a tensor with Hessian `H = X X^T`.
Allocating clip-aggressiveness inversely to `H_diag.mean() · row_var` (hence
the `-0.15 · log` above; the 0.15 coefficient is empirical, originally tuned
in PR [#1689](https://github.com/openai/parameter-golf/pull/1689)) approximates
the optimal sensitivity-budgeted clip allocation. Tensors with high effective sensitivity (high Hessian magnitude,
high weight variance) get smaller σ — finer integer granularity at the cost
of more outlier clipping. Tensors with low sensitivity get larger σ — accept
roundoff in the bulk to preserve outliers.

**Per-tensor allocation (seed 42).** All 66 matrix tensors, sorted by canonical
order:

```
                  c_q     c_k     c_v    proj    fc     proj
blocks.0:        8.74    8.60    8.37   11.39   8.39    9.25
blocks.1:        9.73    9.46    9.19   14.52   9.24   11.26
blocks.2:       10.37   10.17   10.00   14.48   9.80   12.28
blocks.3:        9.21    8.94    8.79   13.82   8.75   11.08
blocks.4:        9.55    9.39    9.17   12.94   9.02   12.22
blocks.5:        9.86    9.48    9.35   14.52   9.27   12.41
blocks.6:       11.75   11.47   11.21   18.13  11.11   14.83
blocks.7:       12.02   11.66   11.51   19.61  11.35   16.79
blocks.8:       12.31   12.04   11.57   15.51  11.60   17.67
blocks.9:       12.66   12.18   11.55   24.00  11.80   17.89
blocks.10:      12.75   12.36   11.75   24.00  12.09   19.58
```

The pattern reflects two robust intuitions:

1. **Output projections (`attn.proj`, `mlp.proj`) systematically receive larger σ**
   than input projections at the same depth — their inputs are post-attention /
   post-activation residual streams with smaller variance, so their per-row
   Hessians scale smaller, so the sensitivity-divided budget allocates them
   more clip-room.
2. **Sigmas widen with depth**. Deeper layers' attn.proj and mlp.proj weights
   are processing increasingly low-variance signals (after many residual
   additions); their Hessian diagonals are small in absolute terms; they get
   the most permissive clipping. Layers 9-10 attn.proj hit the upper clamp at
   24.00 — the technique would prefer even less precision there but is bounded.

The pattern is **stable across seeds** (per-tensor σ matches to within 0.05 across
all three runs), confirming the technique is responding to the model architecture
and training-data distribution, not seed-specific noise.

**Composability with LQER.** PR #1855's LQER asymmetric int4 picks the top-3
highest-error tensors after GPTQ for rank-4 correction. Across all three
seeds, our adaptive-clip pipeline converges on the same LQER selection as
PR #1855's hand-tuned pipeline (top-3 are mlp.fc layers, plus tok_emb at
int7 by default). The adaptive technique does not destabilise LQER's
selection.

**Composability with phased TTT.** TTT recovery dynamics under the adaptive
clip pipeline match PR #1855's hand-tuned pipeline exactly (-0.01272 mean
across all three seeds in both submissions, four decimals identical). The
technique only modifies the GPTQ clip-sigma stage; the phased TTT eval
sees an identically-shaped quantization error distribution and recovers
the same amount.

## Mixed-precision GPTQ ablation (gated, negative result)

A second technique was implemented in the same codebase under
`MIXED_PRECISION_HESSIAN=1` (off by default). Same Hessian-sensitivity ranking
is used to allocate **bit width** rather than clip σ:

- Bottom 25% by sensitivity → int5
- Middle 50% → int6
- Top 25% → int7

Average bits-per-weight is conserved at 6.0 (same as all-int6).

Single-seed test on the same PR #1855 stack:

| Configuration                                | Pre-quant | Post-quant | Quant penalty |
|----------------------------------------------|-----------|------------|---------------|
| PR #1855 baseline (all int6 + LQER)          | 1.06396   | 1.07254    | +0.00858      |
| This work (adaptive σ + all int6 + LQER)     | 1.06498   | 1.07480    | +0.00982      |
| **+ Hessian mixed-precision (5/6/7) + LQER** | 1.07155   | 1.08465    | **+0.01310**  |

The 16 int5 tensors lose more precision than the 16 int7 tensors gain. Net
+0.0045 quant penalty vs all-int6. Different from PR #1908's AWQ-lite mixed
precision (which uses **activation-magnitude** sensitivity rather than
Hessian-diagonal sensitivity), but the negative result is symmetric: an
aggressive int5 floor on a heavily-tuned stack with LQER on top is too
costly. Disabled in this submission. Reported as an honest ablation; the
codebase ships with the gate so reviewers can reproduce.

## Hyperparameter stack

PR #1855's full 9-hparam greedy stack is preserved with one change:

| hparam                 | value  | source            |
|------------------------|--------|-------------------|
| MLP_CLIP_SIGMAS        | 11.5   | PR #1855          |
| EMBED_CLIP_SIGMAS      | 14.0   | PR #1855          |
| ATTN_CLIP_SIGMAS       | 13.0   | PR #1855          |
| WARMDOWN_FRAC          | 0.85   | PR #1855          |
| BETA2                  | 0.99   | PR #1855          |
| TTT_BETA2              | 0.99   | PR #1855          |
| TTT_WEIGHT_DECAY       | 0.5    | PR #1855          |
| **TTT_LORA_RANK**      | **56** | **open PR #1935** |
| SPARSE_ATTN_GATE_SCALE | 0.5    | PR #1855          |
| PHASED_TTT_PREFIX_DOCS | 2500   | PR #1855          |

The CLIP_SIGMAS values still appear in the env-var list because they define
the **target compression budget** that the adaptive offset binary search
matches. They no longer directly determine per-tensor σ; the adaptive
mapping does.

## Training & evaluation

|                  |                                                                                                                                                    |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Training         | 4,816 ± 14 steps in 600 s on 8×H100 SXM (123.8 ms/step mean), warmup=20, warmdown_frac=0.85, MIN_LR=0.10, MATRIX_LR=0.026, GRAD_CLIP_NORM=0.3      |
| Optimizer        | Polar-Express Muon (5 steps) on matrix params; Adam (β₁=0.9, β₂=0.99) on tied embeddings (lr=0.03) and scalars (lr=0.02)                           |
| EMA              | decay = 0.9965                                                                                                                                     |
| Quantization     | GPTQ int6 (matrix, adaptive σ) + int7 (tied embed, fixed σ=14) + int8-per-row attn-gate, with LQER asym int4 rank-4 correction on top-3 tensors    |
| Compression      | per-group lrzip ZPAQ + L1 simsort on hot tensors + brotli on remainder + brotli code wrapper                                                       |
| TTT              | Phased, 3 cumulative phases at doc-boundaries 833 / 1666 / 2500; LoRA rank=56 on Q/K/V/O/MLP + lm_head, per-doc reset, lr=5e-5 cosine within phase |
| Eval time (mean) | 523.5 s of 600 s budget                                                                                                                            |

## Reproduction

```bash
# Deps (RunPod runpod/parameter-golf:latest base; PEP-668 image)
pip install --break-system-packages brotli huggingface_hub python-minifier sentencepiece
pip install --break-system-packages --no-deps flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
apt-get update && apt-get install -y lrzip

# CaseOps tokenized FineWeb (HF, ~16 GB; set HF_TOKEN to skip rate limits)
export HF_TOKEN=<your_hf_read_token>
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('romeerp/parameter-golf-caseops-v1',
                  repo_type='dataset',
                  local_dir='/workspace/parameter-golf/data/datasets')
"

# 3-seed run
for SEED in 42 1337 999; do
  SEED=$SEED MAX_WALLCLOCK_SECONDS=600 \
    ADAPTIVE_HESSIAN_CLIP=1 MIXED_PRECISION_HESSIAN=0 TTT_LORA_RANK=56 \
    CASEOPS_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SMEAR_GATE_ENABLED=1 \
    LQER_ENABLED=1 LQER_ASYM_ENABLED=1 \
    MIN_LR=0.1 PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=2500 \
    MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 EMBED_BITS=7 \
    WARMDOWN_FRAC=0.85 BETA2=0.99 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 \
    SPARSE_ATTN_GATE_SCALE=0.5 \
    COMPRESSOR=pergroup NCCL_NET=Socket \
    DATA_PATH=/workspace/parameter-golf/data/datasets/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 \
    | tee train_seed${SEED}.log
done
```

To reproduce **with** the mixed-precision ablation (negative result), set
`MIXED_PRECISION_HESSIAN=1`. To reproduce **without** the adaptive technique
(i.e. PR #1855 verbatim), set `ADAPTIVE_HESSIAN_CLIP=0`.

## Files

- `train_gpt.py` — full training script (~3,750 lines, identical to PR #1855
  except for ~80 lines of new env-var-gated code: adaptive clip computation
  in `gptq_mixed_quantize` and the optional mixed-precision bit-allocation
  block).
- `train_seed42.log`, `train_seed1337.log`, `train_seed999.log` — full
  per-seed run logs including the adaptive σ allocation table for each seed.
- `submission.json` — structured metadata (per-seed val_bpb, std, comparison,
  technique summary).

## Compliance (track_10min_16mb)

- **Training under 600 s:** ✅ All seeds stopped at the wallclock cap (596 s mean).
- **Artifact under 16,000,000 bytes:** ✅ All seeds 15.90–15.92 MB.
- **Eval under 600 s:** ✅ Seeds 456–593 s (median 521 s).
- **No pre-quant TTT:** ✅ TTT runs post-quantization only.
- **Score-first TTT:** ✅ Phased TTT scores before each update.
- **No SLOT / no ETLB / no n-gram cache:** ✅
- **3 seeds:** ✅ Seeds 42, 1337, 999.

## Credits

The novel contribution — adaptive Hessian-sensitivity GPTQ clipping — was
first introduced in **PR [#1689](https://github.com/openai/parameter-golf/pull/1689)**
by this submitter on a different base. This submission ports it to PR #1855's
stack and shows the technique generalises across architectures.

Implementation lineage of the inherited components stacks decisions from the
established community PR chain. Direct ancestors:

- **PR [#1855](https://github.com/openai/parameter-golf/pull/1855)** by
  @codemath3000 — the entire architectural and quantization stack used here:
  11L XSA, LQER asymmetric, SparseAttnGate, BOS-fixed SmearGate, Polar-Express
  Muon, MIN_LR, FusedCE, per-group lrzip, 9-hparam greedy. This submission
  is PR #1855 with the three hand-tuned per-group clip sigmas replaced by
  one Hessian-driven adaptive selection. Everything else is preserved.
- **PR [#1851](https://github.com/openai/parameter-golf/pull/1851)** by
  @aquariouseworkman — SmearGate BOS leak fix, LQER asymmetric int4, 3-phase
  phased TTT framework that PR #1855 builds on.
- **PR [#1797](https://github.com/openai/parameter-golf/pull/1797)** by
  @dexhunter — SmearGate + LQER asymmetric implementation.
- **PR [#1787](https://github.com/openai/parameter-golf/pull/1787)** by
  @nprime06 — Polar-Express Newton-Schulz Muon, MIN_LR floor, sparse attention
  gate, fused softcapped CE.
- **PR [#1736](https://github.com/openai/parameter-golf/pull/1736)** —
  CaseOps + GatedAttn + QuantGate + Loop3-5 + phased TTT integration.
- **PR [#1729](https://github.com/openai/parameter-golf/pull/1729)** by
  @romeerp — sp8192 lossless caps caseops v1 reserved tokenizer + dataset
  hosting on HuggingFace as `romeerp/parameter-golf-caseops-v1`.
- **PR [#1626](https://github.com/openai/parameter-golf/pull/1626)** by
  @dexhunter — Multi-phase global SGD phased-TTT.
- **PR [#1530](https://github.com/openai/parameter-golf/pull/1530)** by
  @samacqua — Variable-length attention, fused LeakyReLU² MLP Triton kernel,
  parallel residuals, doc-based LoRA TTT.
- **PR [#1493](https://github.com/openai/parameter-golf/pull/1493)** by
  @bigbag — 3-layer recurrence, parallel residuals, QK-Gain 5.25 on PR #1394.
- **PR [#1394](https://github.com/openai/parameter-golf/pull/1394)** by
  @clarkkev — SP8192 + GPTQ embeddings + SDClip base. The original GPTQ
  pipeline our adaptive σ replaces.
- **PR [#1344](https://github.com/openai/parameter-golf/pull/1344)** —
  Polar-Express Newton-Schulz coefficients + depth recurrence.
- **PR [#1276](https://github.com/openai/parameter-golf/pull/1276)** —
  QK-Gain 5.0.
- **PR [#1586](https://github.com/openai/parameter-golf/pull/1586)** —
  Per-Layer Adaptive GPTQ Clip + int7 embeddings + MATRIX_LR=0.026.
- Open **PR [#1935](https://github.com/openai/parameter-golf/pull/1935)**
  by @vimeto — `TTT_LORA_RANK=56` tweak adopted here.
- **PR [#493](https://github.com/openai/parameter-golf/pull/493)** —
  LeakyReLU² activation.
- **PR [#478](https://github.com/openai/parameter-golf/pull/478)** by
  @gowtham0992 — XSA on all layers.
- **PR [#315](https://github.com/openai/parameter-golf/pull/315)** —
  Partial RoPE + LN Scale.
- **PR [#289](https://github.com/openai/parameter-golf/pull/289)** —
  U-Net skip connections.

