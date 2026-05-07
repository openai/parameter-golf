# Record-track: Trajectory-State Readout + Muon 0.98 + Legal TTT (1.0788)

**val_bpb = 1.0788** (3-seed mean, std 0.0003) | **~15.99 MB** | 8xH100 SXM

## Summary

Depth-recurrent blocks form a dynamical system traversing a trajectory through
hidden-state space. PR #1493 uses only the final state. This submission adds a
32-learned-parameter grouped trajectory readout to the PR #1493 family,
alongside a step-based recurrence schedule and adopted Muon/GPTQ settings from
PR #1518. The resulting 3-seed mean is 1.0788 BPB.

## 3-Seed Results

| Seed | Sliding BPB | **TTT BPB** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0804      | **1.0787**  | 15,990,973 |
| 2025 | 1.0804      | **1.0791**  | 15,989,784 |
| 1234 | 1.0799      | **1.0785**  | 15,990,480 |
| **Mean** | **1.0802** | **1.0788** | |
| **Std** | **0.0003** | **0.0003** | |

Base (PR #1493): **1.0810 BPB**. Delta: **-0.0022 BPB**. On the fixed FineWeb validation set, this corresponds to an approximate val_loss improvement of ~0.006 nats, above the leaderboard's stated 0.005-nat magnitude threshold.

## What's New vs PR #1493

1. **Trajectory-state readout** (32 learned params) — blends block-5 outputs from all
   three recurrence passes instead of discarding the first two.
2. **Step-based loop activation** — depth recurrence activates at step 2000
   instead of wallclock fraction 0.35, eliminating a cross-machine confound
   that contaminated prior ablations.
3. **QK_GAIN_INIT=5.0** — PR #1493 used 5.25; these runs default to 5.0.
   This is an unintentional recipe difference from PR #1493 caused by the code
   default; its effect was not isolated separately.

## Adopted from PR #1518

- **MUON_MOMENTUM=0.98** — adopted from PR #1518; in our ablations, this
  setting consistently outperformed 0.99.
- **GPTQ_RESERVE_SECONDS=4, GPTQ_CALIBRATION_BATCHES=16** — tighter quantization
  budget frees ~8 more training seconds.

## Recipe Differences vs PR #1493

Full list of intentional and unintentional changes from PR #1493's recipe:
- Trajectory-state readout (32 learned parameters, new)
- Step-based loop activation at step 2000 (was wallclock fraction 0.35)
- MUON_MOMENTUM=0.98 (was 0.99, adopted from PR #1518)
- GPTQ_RESERVE_SECONDS=4 (was 12, adopted from PR #1518)
- GPTQ_CALIBRATION_BATCHES=16 (was 64, adopted from PR #1518)
- QK_GAIN_INIT=5.0 (was 5.25, unintentional — code default)

The trajectory readout is the only novel architectural contribution. The
remaining differences are hyperparameter adoption and a curriculum fix.

## Negative Results

Five other approaches were tested before converging on trajectory readout. All
failed or showed negligible effect. Documenting them here to save future effort.

**Birkhoff contraction with asymmetric init.** Replaced the linear `resid_mix`
blend with `sigmoid(logit) * x + (1-sigmoid(logit)) * x0` to make the
recurrence a contraction. Asymmetric init (entry alpha=0.984, mid=0.993,
exit=0.997) biased toward re-injection at entry, preservation at exit. Looked
promising in short ablations (-0.0002 to -0.0005 BPB at 2000 steps), but fell
apart at full scale — runs WITHOUT Birkhoff outperformed runs WITH it once
trajectory readout was added. The contraction appeared to reduce state diversity across passes in our
experiments, leaving less for the readout to recover. Code retained behind
`USE_BIRKHOFF=1` for ablation reproducibility.

**FiLM gamma on MLP output.** Per-pass channel-wise scaling:
`mlp_out *= (1 + gamma[pass_idx])` with [3,512] learned parameters. The
~-0.00034 post-EMA signal turned out to be an EMA crossover artifact, not a
real per-pass effect. The shared MLP appeared to absorb the per-pass gamma signal during training,
possibly due to conflicting gradient directions.

**Per-pass embeddings, hidden gates, output scales.** Three flavors of explicit
per-pass conditioning. Zero measurable effect after 2000 steps. Pre-norm
absorbs additive per-pass signals; shared weights simply can't exploit per-pass
identity when it's just an additive bias.

**Controls-only TTT (from PR #1518).** Adapting only ~30K control parameters
(skip_weights, attn_scale, mlp_scale, etc.) instead of all 35.9M. On this
architecture it regressed by +0.00199 BPB, turning TTT's -0.0017 gain into a
+0.002 loss. Full-param SGD remains necessary.

**parallel_residual_start=8 (from PR #1518).** Neutral without their two-lane
routing mechanism. Kept at default 7.

**Key takeaway:** In our experiments, explicit per-pass conditioning consistently
failed on this architecture. The shared blocks already see an evolving hidden
state — that evolution *is* the pass-specific signal. The approach that worked
was leaving the dynamics alone and focusing on better readout.

## Core Mechanism

The looped blocks (3,4,5) execute three passes. Block 5 produces output at
three trajectory landmarks:

- **p0**: encoder block-5 (first pass)
- **p1**: decoder first block-5 (second pass)
- **p2**: decoder second block-5 (third pass — the only output PR #1493 uses)

Trajectory readout learns per-group correction coefficients:

```
a = scale * tanh(readout_delta)              # shape [2, G]
a_full = a.repeat_interleave(D // G, dim=1)  # broadcast each group to channels
x = p2 + a_full[0] * (p0 - p2) + a_full[1] * (p1 - p2)
```

Channels are split into G equal groups of D/G each. Every group learns its own
readout weight, so different parts of the representation can recover different
amounts of early-pass information.

This submission uses G=16, scale=0.5, D=512 (32 channels per group, 32 total
parameters). At init, `readout_delta = zeros(2, 16)` so `a = 0` and `x = p2`
— identity, exactly PR #1493 behavior. Cost: 32 learned parameters, 64 bytes FP16
passthrough; negligible artifact overhead. Negligible runtime overhead — a tanh, broadcast, and two elementwise
operations per forward pass.

Conceptually analogous to reservoir computing: the depth-recurrent core
produces a trajectory of states, and a tiny learned readout recovers
information from earlier trajectory points. Unlike classical RC, the reservoir
here is trained end-to-end; the analogy motivates the readout design, not the
training procedure.

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims),
layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence:
encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5,
activated at step 2000). Parallel residuals from layer 7. Sigmoid-gated U-Net
skip connections.

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps, momentum=0.98),
AdamW for embeddings/scalars. ~4623-4659 steps in ~596s on 8xH100 SXM. Linear
warmdown to LR=0 over final 72%. EMA decay 0.9965. GPTQ int6 with SDClip
(k=12.85), int8 embeddings (k=20.0), Brotli-11 compression.

## TTT

Score-first chunk-based SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token
chunk, cosine LR decay. Gradient clipping at 1.0. Full-parameter adaptation
(35.9M params). Total TTT eval time: ~370s.

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

TTT_ENABLED=1 MUON_MOMENTUM=0.98 \
    GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
    ENABLE_LOOPING_AT_STEP=2000 \
    USE_PASS_READOUT=1 READOUT_GROUPS=16 READOUT_SCALE=0.5 \
    SEED=42 RUN_ID=train_seed42 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All readout env vars default to the winning recipe; the explicit flags above
are for documentation clarity. `TTT_ENABLED=1` and `MUON_MOMENTUM=0.98` must
be set as they differ from the base code defaults.

## Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):

- **Causality:** Sliding-window eval is strictly causal.
- **Normalized distribution:** Standard softmax over full vocab.
- **Score before update:** Each chunk fully scored before any SGD update.
- **Single pass:** Each token scored exactly once.
- **TTT state not persisted:** TTT runs once at eval time after quantization; no TTT-adapted weights are included in the artifact.
- No SLOT, no pre-quant TTT, no ETLB, no n-gram cache.
- Artifacts: ~15.99 MB across seeds (15,989,784–15,990,973 bytes), all under 16,000,000 bytes.
- Training under 600s (~596s actual). Eval under 600s.

## Credits

- **PR #1493** (@bigbag) — base submission: SP8192 + 3-layer depth recurrence + parallel residuals + QK-Gain 5.25 + legal TTT (1.0810 BPB)
- **PR #1518** (@abaybektursun) — MUON_MOMENTUM=0.98, GPTQ timing optimizations
- **PR #1394** (@clarkkev) — SP8192 + GPTQ SDClip + MuonEq-R
- **PR #1413** (@dexhunter) — legal TTT on SP8192
- **PR #1412** (@Robby955) — parallel residuals on SP8192
- **PR #1204** (@msisovic) — parallel residuals concept
- **PR #549** (@abaybektursun) — score-first TTT framework
- **PR #1445, #1471** (@X-Abhishek-X) — hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965
- **PR #1331** (@dexhunter) — depth recurrence exploration
- **Reservoir computing framing** — Jaeger (2001), Lukoševičius & Jaeger (2009);
  RingFormer (Heo et al. 2025, arXiv:2502.13181) for recurrence-aware control
  signals
- OpenAI Advanced Competitor grant ($500 RunPod credit)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_gpt_dev.py`
- `train_seed42.log`
- `train_seed2025.log`
- `train_seed1234.log`
