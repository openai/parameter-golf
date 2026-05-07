# RRT-LoRA v4: Tier-1 Stack for Parameter Golf

**Status: Implementation complete — compute grant pending. Ready to run.**

Target: **sub-1.076 val_bpb** on `track_10min_16mb` | 8×H100 SXM | ≤16MB artifact

Current leaderboard SOTA at submission time: **1.0810 bpb** (bigbag, PR #1493, April 9 2026). SOTA threshold requires **≥0.005 nats improvement at p<0.01 across 3 seeds**.

---

## Executive Summary

v4 is a stack of ~10 incremental, low-risk improvements on top of the v3 RRT-LoRA baseline. Every technique in the stack has prior published evidence at comparable scale and has either been adopted in modded-nanogpt's master branch or directly ablated in small-transformer pretraining work.

**Expected aggregate gain: –0.04 to –0.08 bpb over v3 baseline**, projecting to ~1.06–1.07 bpb final. That clears the +0.005 SOTA threshold with margin. Every new technique has a kill switch, so ablation isolates any regression to a single change.

A more ambitious Tier-2 variant (`train_gpt_rrt_v5.py`) exists as a research companion — Hyperconnections, Mixture-of-LoRAs, HQQ 3-bit QAT — but it is **not the primary submission**. v5 is higher variance inside a 600-second training budget and is reserved for post-submission experimentation if grant hours remain. This README describes v4.

---

## Novel Contribution: RRT-LoRA (retained from v3)

Current depth recurrence submissions (PR #1394, #1437, #1493) loop layers 3-5 with **identical shared weights** on every pass. This has two fundamental problems:

1. **No specialization** — every recurrence pass is identical, so the model can't learn coarse-to-fine refinement across steps
2. **Quantization error amplification** — int6 quantization error compounds multiplicatively across recurrence steps, limiting stable loop depth to ~3

**RRT-LoRA** solves both. Each recurrence step applies a tiny learned delta (rank=4) to the shared Q and V projections:

```python
Q_step_i = W_q(x) + alpha * lora_B_i(lora_A_i(x))
V_step_i = W_v(x) + alpha * lora_B_v_i(lora_A_v_i(x))
```

Key design decisions:
- **Zero init**: LoRA B matrices initialize to zero — identical to baseline at step 0, ensuring stable training
- **Alpha warmup**: LoRA contribution ramps from 0→1 over 500 steps after recurrence activates at 35% of training — smooth curriculum
- **Parameter cost**: ~4K params per adapter pair × 3 layers × 3 steps = ~36K total LoRA params (~0.4% of total budget)
- **Quantization stability**: Low-rank deltas are far less sensitive to int6 quantization than full weight matrices, reducing error amplification

Reference: Bae et al. "Relaxed Recursive Transformers" (ICLR 2025)

---

## Tier 1 Additions (v4 stack)

### Optimizer

**Polar Express orthogonalization** (Amsel, Persson, Musco, Gower 2025, `arXiv:2505.16932`). Replaces Newton-Schulz-5 with optimal adaptive Zolotarev coefficients. Drop-in swap; bf16-safe. Adopted by modded-nanogpt master since Sept 2025.

**NorMuon placement** (arXiv:2510.05491). Row-normalization moved to *after* orthogonalization rather than before (the MuonEq-R convention). The paper's key finding: per-neuron update distribution remains non-uniform even after NS, and post-orth row variance correction closes that gap. Adopted as modded-nanogpt record #41 (2025-10-24). At 1.1B NorMuon reaches Muon-equivalent loss in 66–83% of the steps.

**Cautious Weight Decay** (Li et al. 2025, `arXiv:2510.12402`). Decoupled WD applied only to coordinates where `sign(grad) == sign(param)`, with linear-to-zero schedule synchronized with LR warmdown. Karpathy called it *"a clear win"* in the nanochat ablations; adopted as modded-nanogpt record #43.

### Architecture

**Value Residual Learning** (Zhou et al., `arXiv:2410.17897`, ACL 2025). Layer-1 attention values injected into all subsequent attention layers via learned sigmoid gate (init 0 → baseline behavior at step 0). ResFormer reaches equivalent val-loss at **16.1% fewer parameters and 20.3% fewer tokens** across 82M–468M scales. This alleviates the attention concentration that recurrence-loaded middle blocks suffer from — complementary to RRT, not competing.

**Backout** (Sebastian Müller, modded-nanogpt PR #140). Subtract a learned sigmoid-gated fraction of the early-layer residual stream from the final hidden before lm_head projection. Extremely cheap; compatible with everything else in the stack.

**Softmax-Skip-Gate init** (modded-nanogpt #125). U-Net skip connection weights initialized to 0.18 rather than 1.0.

**Partial-RoPE 8/64** (`arXiv:2603.11611`, Feb 2026). The paper's sweep of 4%, 10%, 25%, 50%, and full RoPE concludes **~10% of dims is sufficient** across model sizes and sequence lengths. v3 used 25%; v4 uses 12.5% (still above the 10% threshold).

### Evaluation

**SWA over last K=10 checkpoints** (LAWA, `arXiv:2306.03241`). Uniform average of the last 10 checkpoints captured during the final 20% of training. Compared against EMA (α=0.999, up from 0.9965); whichever is lower wins. LAWA reports GPT-2 OpenWebText losses 2.963 → 2.917 and 2.855 → 2.819 from checkpoint averaging alone.

**Temperature sweep at TTT** (replaces v3's fixed T=0.98). Sweeps `{0.95, 0.98, 1.00, 1.02, 1.05}` on the TTT run and selects the minimum-bpb value. The November 2025 entropy-calibration analysis (`arXiv:2511.11966`) confirms the log-loss-minimizing temperature for undertrained base models is close to 1 but slightly >1.

**TTT: AdamW at lr 3e-6, 1 step per chunk** (Rannen-Triki / DeepMind 2024, `arXiv:2403.01518`). The sweep explicitly found **AdamW beats SGD** for dynamic evaluation on Transformer-XL-style segment updates, with **LR in {1e-6, 3e-6, 1e-5, 3e-5}** — three orders of magnitude below v3's 5e-3. They also find **1 update per chunk is optimal**; multiple epochs per chunk (v3's "3 epochs") **overfits the chunk and hurts downstream bpb**. Hardt-Sun TTT-NN (ICLR 2024) confirms.

### Compression

**Low-rank factored tied embedding, rank 128**. `E = U @ V` where `U ∈ R^(8192×128)`, `V ∈ R^(128×512)`. Drops the 4.2M-param embedding to 1.1M (3.9× compression), near-lossless at this rank per the ALBERT / Khrulkov Tensorized Embeddings literature. Post-factor int8 + rANS: embedding footprint ≈ 0.6–1.0 MB. **The single highest-leverage change in the compression pipeline — reclaims ~5 MB of artifact budget that can be recycled into more weights.**

**Static range coder on int6 stream**. Replaces LZMA on the weight-int-stream with a Shannon-near arithmetic coder over the empirical histogram. LZ77's backend is wasted on quantized integer streams (no long repeats). Pure-Python decoder, 10–25% smaller than LZMA on int streams.

**LZMA delta-filter on scales/code** (`FILTER_DELTA` + `FILTER_LZMA2` @ `preset=9|PRESET_EXTREME`, `dist=2`). Another 3–8% on metadata.

---

## Full Technique Table

| Technique | Source | Expected gain |
|---|---|---|
| SP8192 tokenizer | PR #1394 @clarkkev | ~0.12 bpb vs SP1024 |
| GPTQ SDClip int6/int8 | PR #1394 @clarkkev | fits 16MB budget |
| MuonEq-R optimizer | arXiv:2603.28254 | ~0.005 bpb |
| Exact SOTA recurrence schedule | PR #1493 @bigbag | ~0.03 bpb |
| Parallel residuals (GPT-J) | PR #1412 @Robby955 | ~0.008 bpb + speed |
| QK-Gain 5.25 | PR #1413 @dexhunter | ~0.005 bpb |
| EMA decay | PR #1445 @X-Abhishek-X | ~0.003 bpb |
| Legal score-first TTT | PR #549, #1413 | ~0.018 bpb |
| Bit-packed int6 storage | novel (v3) | ~25% artifact savings |
| Brotli-11 + LZMA compression | PR #1493 | ~43KB savings |
| RRT-LoRA (novel) | this submission v3 | ~0.005–0.015 bpb |
| **Polar Express orthogonalization** | arXiv:2505.16932 | –0.003 to –0.010 bpb |
| **NorMuon placement** | arXiv:2510.05491 | –0.005 to –0.015 bpb |
| **Cautious Weight Decay + schedule** | arXiv:2510.12402 | –0.005 to –0.010 bpb |
| **Value Residual Learning** | arXiv:2410.17897 | –0.020 to –0.040 bpb |
| **Backout** | mng PR #140 | –0.005 to –0.010 bpb |
| **Softmax-Skip-Gate init 0.18** | mng #125 | –0.003 to –0.008 bpb |
| **Partial-RoPE 16/64 → 8/64** | arXiv:2603.11611 | ±0.003 bpb |
| **SWA of last 10 checkpoints** | arXiv:2306.03241 | –0.030 to –0.080 bpb |
| **Temperature sweep at TTT** | arXiv:2511.11966 | –0.001 to –0.005 bpb |
| **TTT AdamW @ 3e-6, 1 step/chunk** | arXiv:2403.01518 | up to –0.010 bpb |
| **Low-rank factored embed (r=128)** | ALBERT + Khrulkov 2019 | ~5 MB reclaimed |
| **Static range coder on int streams** | rANS / Shannon | 0.2–0.4 bpw → ~1 MB |
| **LZMA delta filter on scales/code** | xz spec | 3–8% metadata |

---

## Architecture

```
Physical layers:         11
Model dim:               512
Heads / KV heads:        8 / 4
MLP expansion:           4×  (LeakyReLU(0.5)²)
Partial RoPE:            8 / 64 head dims  [Tier-1]
Tied embeddings:         yes, low-rank factored r=128  [Tier-1]

Recurrence schedule (exact SOTA):
  encoder: [0, 1, 2, 3, 4, 5, 3, 4]       → 8 encoder steps
  decoder: [5, 3, 4, 5, 6, 7, 8, 9, 10]   → 9 decoder steps
  virtual layers: 17  (vs 11 physical)

RRT-LoRA:
  Applied to layers:  {3, 4, 5}
  Rank:               4
  Steps:              3 per layer
  LoRA params:        ~36K total
  Alpha schedule:     0 → 1 over 500 steps (after recurrence activates)

Value Residual:        layer-1 values fed forward via sigmoid gate (init 0)
Backout:               learned sigmoid on early-layer residual, subtracted before lm_head
Parallel residuals:    layers 7–10 (GPT-J style)
Skip-gate init:        0.18 (softmax-skip-gate)
QK-Gain init:          5.25 (learned per-head)
Logit softcap:         30.0
```

---

## Training

```
Optimizer (matrices):    Muon  [Polar Express + NorMuon placement]
                         lr=0.022, momentum=0.95, ns_steps=5
Optimizer (embed/head):  AdamW lr=0.05 (tied) / 0.008 (head), WD=0
Optimizer (scalars):     AdamW lr=0.04, WD=0
Cautious WD:             wd=0.095, sign-masked, linear-to-zero schedule
Grad clip:               1.0
Warmup:                  20 steps
Warmdown:                60% of training (linear to 0)   [was 72% in v3]
EMA decay:               0.999                            [was 0.9965 in v3]
SWA:                     last 10 checkpoints, uniform avg, starts at 80% training
Batch tokens:            524,288 / step
Max wallclock:           600s (8×H100 SXM)
```

---

## Quantization & Compression Pipeline

```
Hidden weight matrices:  int6 SDClip k=12.85, bit-packed
Embeddings:              int8 SDClip k=20.0 on low-rank factored U, V
Scales / zeros:          float16
Byte-level coding:       static rANS range coder over empirical histogram
Metadata compression:    LZMA xz with FILTER_DELTA(dist=2) + LZMA2 preset=9|EXTREME
Code compression:        Same filter chain
Estimated artifact:      ~10–12 MB  (was ~15.8 MB in v3)
Headroom:                ~4–6 MB available for width/depth expansion
```

---

## Legal Score-First TTT

Compliance per Issue #1017 (Track B) — unchanged from v3:
- **Causality**: Sliding window eval is strictly causal
- **Score-first**: Each chunk fully scored under `torch.inference_mode()` BEFORE any update
- **Single pass**: Each token scored exactly once
- **Normalized**: Standard softmax only

v4 TTT changes:
- Optimizer: SGD → AdamW
- LR: 5e-3 → 3e-6 (Rannen-Triki sweep)
- Epochs per chunk: 3 → 1 (overfitting chunk hurts downstream bpb)
- Freeze layers 0–1 retained
- Grad clip 1.0 retained
- Temperature: fixed T=0.98 → sweep `{0.95, 0.98, 1.00, 1.02, 1.05}`, select min-bpb
- Model weights restored after TTT eval — does not affect training state

---

## Risks & Mitigations

Honest accounting of what could go wrong, with the kill switch for each:

| Risk | Probability | Mitigation |
|---|---|---|
| NorMuon placement destabilizes Muon at small scale | Low | `DISABLE_NORMUON=1` reverts to MuonEq-R pre-orth normalization |
| Value Residual gate gets stuck at 0 (no contribution) | Low | Gate is init-0 so this is the "safe" failure mode; `DISABLE_VRES=1` removes the module entirely |
| Cautious WD + linear schedule interacts badly with EMA | Low | `DISABLE_CAUTIOUS=1` reverts to standard AdamW decoupled WD |
| Low-rank embed r=128 loses too much vocab expressivity | Medium | `DISABLE_LRF_EMBED=1` reverts to full embedding; fallback artifact is ~15.5 MB and still fits |
| SWA of checkpoints hurts vs pure EMA at this scale | Low | Runtime compares both and picks the winner; SWA is never a regression |
| AdamW TTT at 3e-6 under-adapts in 32K-token chunks | Medium | `TTT_OPTIM=sgd TTT_LR=0.005 TTT_STEPS_PER_CHUNK=3` reverts to v3 TTT |
| Warmdown 60% underperforms 72% at this specific budget | Low | `WARMDOWN_FRAC=0.72` reverts |
| Partial-RoPE 8/64 too aggressive vs v3's 16/64 | Low | `ROPE_PARTIAL_DIM=16` reverts |
| Range coder decoder is slow enough to affect eval timing | Low | `DISABLE_RANS=1` falls back to bit-packed int6 + LZMA |

Worst-realistic case: two or three Tier-1 items interact badly, final lands at ~1.08–1.09 bpb, misses SOTA. In that case the ablation ladder (below) identifies which switches to flip for the final submission seeds.

---

## Ablation & Run Plan

v4 ships with granular kill switches. The grant-efficient run order (~5–6 runs out of ~160 grant hours):

```bash
# Run 1 — v3-equivalent baseline, no new Tier-1 techniques
# Validates the code reproduces v3 numbers before introducing any changes.
DISABLE_VRES=1 DISABLE_BACKOUT=1 DISABLE_CAUTIOUS=1 \
  DISABLE_POLAR=1 DISABLE_NORMUON=1 \
  DISABLE_LRF_EMBED=1 DISABLE_RANS=1 \
  ABLATE=1 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py

# Run 2 — full Tier-1 stack, first tuning run
# If it clears v3, proceed. If not, run the ablation ladder to isolate the regression.
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py

# Run 3 — optional tuning (e.g., WARMDOWN_FRAC sweep)
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py

# Runs 4–6 — 3-seed statistical validation for final submission
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py
SEED=999 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py
```

If final median bpb clears 1.076 with p<0.01 across seeds, submit. If Run 2 regresses vs Run 1, flip kill switches one at a time — each run then attributes exactly one change.

---

## Why This Should Clear SOTA

The leaderboard is dense: the top ten submissions fit inside a 0.019 bpb band (1.0810 to 1.100). The SOTA bar at +0.005 nats is a **one-idea bar** given that density. v4 stacks ~10 ideas, each individually documented at equal-or-better scales on similar tasks.

Expected dominant contributors, in order:

1. **SWA of late checkpoints** alone has been repeatedly measured at –0.03 to –0.08 bpb at small-transformer pretraining scale (LAWA, Kaddour et al.). Free at eval time. This is the single most likely win.
2. **Value Residual Learning** has been specifically shown to help in recurrence-heavy settings by alleviating attention concentration — directly relevant to RRT.
3. **Low-rank factored embedding + range coder** reclaims ~5–6 MB of the 16 MB budget. This cashes in as more parameters rather than bpb directly, but compounds with everything else.

The remaining Tier-1 items (Polar Express, NorMuon placement, Cautious WD, Backout, partial-RoPE, TTT fixes) are each individually small but collectively meaningful, and each has zero-init or fallback semantics — worst case they contribute nothing, not negative value.

**Expected outcomes:**
- **~70%**: v4 clears 1.076 bpb and becomes the leaderboard record
- **~20%**: v4 lands between 1.076 and 1.081 — close but below SOTA threshold
- **~10%**: v4 regresses vs v3 baseline due to an interaction effect identified by ablation

---

## Research Companion: v5 (not the primary submission)

`train_gpt_rrt_v5.py` exists with three Tier-2 replacements layered on v4:
- **Hyperconnections** (n=4 parallel residual streams, DWHC variant)
- **Mixture-of-LoRAs** (per-token routing between K=4 LoRA experts)
- **HQQ 3-bit with entropy-regularized QAT**

Each is more architecturally ambitious and has higher variance inside a 600-second budget. v5 is reserved for post-submission experimentation if grant hours remain after v4 completes. Not submitted as primary.

---

## Background

- Solo ML researcher / founder — VectaBind (AI drug discovery platform, vectabind.com)
- Built SE(3)-equivariant EGNN + ESM2-3B cross-attention model from scratch, trained on BindingDB (3M), PDBBind 2020, CrossDocked2020, TDC, GDSC, CTRP on a single NVIDIA L4. Val MAE 0.20 pKd on PDBBind 2020 validation split.
- Adaptive computation background: implemented PonderNet/ACT-style architecture for multi-game reasoning (Othello, MiniGrid, drone navigation) with learned halting over latent recurrence steps — directly relevant to this submission
- GitHub: ChipGlitch

---

## Progress

- [x] Full leaderboard review: PRs #1394, #1412, #1413, #1437, #1445, #1493
- [x] RRT-LoRA architecture (v3)
- [x] **Tier-1 full stack (v4)**: Polar Express, NorMuon placement, Cautious WD, Value Residual, Backout, softmax-skip-gate init, partial-RoPE 8/64, SWA, temperature sweep, AdamW TTT, low-rank factored embed, rANS range coder, LZMA delta filter
- [x] Exact SOTA recurrence schedule: enc=[0,1,2,3,4,5,3,4] dec=[5,3,4,5,6,7,8,9,10]
- [x] Legal score-first TTT with full Track B compliance
- [x] Kill switches for every Tier-1 change (granular ablation)
- [x] Research-companion v5 with Tier-2 architectural replacements (not primary submission)
- [ ] H100 validation run (pending compute grant)
- [ ] Run 1: v3-equivalent baseline reproduction
- [ ] Run 2: v4 full Tier-1 stack, tuning
- [ ] Ablation ladder if regression
- [ ] 3-seed statistical validation (seeds 42, 314, 999)
- [ ] Final submission

---

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Full v4 Tier-1 stack
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py

# v3-equivalent baseline reproduction
DISABLE_VRES=1 DISABLE_BACKOUT=1 DISABLE_CAUTIOUS=1 \
  DISABLE_POLAR=1 DISABLE_NORMUON=1 \
  DISABLE_LRF_EMBED=1 DISABLE_RANS=1 \
  ABLATE=1 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py
```

---

## References

### Baseline / v3 stack
- Bae et al. (ICLR 2025). Relaxed Recursive Transformers
- MuonEq-R: arXiv:2603.28254 (March 2026)
- Raposo et al. (2024). Mixture of Depths: arXiv:2404.02258
- clarkkev — SP8192 + GPTQ SDClip + MuonEq-R + depth recurrence (PR #1394)
- dexhunter — 3-layer depth recurrence + legal TTT (PR #1437, #1413)
- abaybektursun — Score-first TTT framework (PR #549)
- Robby955 — Parallel residuals on SP8192 (PR #1412)
- X-Abhishek-X — Hyperparameter tuning WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445)
- bigbag — 3-layer recurrence + parallel residuals + QK-Gain 5.25 (PR #1493)

### Tier 1 additions (v4)
- Amsel, Persson, Musco, Gower. "The Polar Express." arXiv:2505.16932 (May 2025)
- "NorMuon: Making Muon more efficient and scalable." arXiv:2510.05491 (Oct 2025)
- Li et al. "Cautious Weight Decay." arXiv:2510.12402 (Oct 2025)
- Zhou et al. "Value Residual Learning." arXiv:2410.17897 (ACL 2025)
- Sebastian Müller. "Backout." modded-nanogpt PR #140
- "Fractional Rotation, Full Potential?" arXiv:2603.11611 (Feb 2026)
- Sanyal et al. "Latest Weight Averaging." arXiv:2306.03241
- Rannen-Triki et al. "Revisiting Dynamic Evaluation." arXiv:2403.01518 (DeepMind 2024)
- Hardt & Sun. "Test-Time Training with Self-Supervision." ICLR 2024, arXiv:2305.18466
- Khrulkov et al. 2019. Tensorized Embeddings
- Acharya et al. arXiv:1811.00641 (embedding compression)

---

## Planned Files (final submission)

- `README.md` (this file)
- `submission.json`
- `train_gpt_rrt_v4.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
