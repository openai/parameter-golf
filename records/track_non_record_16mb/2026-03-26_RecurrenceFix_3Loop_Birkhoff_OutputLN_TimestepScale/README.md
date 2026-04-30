# Non-Record: First Viable 3-Loop Recurrence — Birkhoff + Output-LN + Timestep Scaling

## Result

**1.2659 BPB** post-quantization (10.7 MB model + 57KB code). Config: 1 prelude + 4 shared × 3 loops + 1 coda = **14 effective layers from 6 unique blocks**. Q-gap +0.0076. Depth recurrence at 3 loops has never worked in this competition — prior 3-loop attempts failed catastrophically (PR #363 measured ~900× quantization amplification over 3 cycles). This submission brings Q-gap down to +0.0076 using three new techniques.

## Technique Summary

| Technique | What it does | Delta | Verdict |
|-----------|-------------|-------|---------|
| **Output-LN** | Moves RMSNorm from MLP input to output, letting shared weights see different magnitudes per iteration | −0.007 BPB (screening) | Critical — nothing works without it |
| **Birkhoff mixing** | Constrains residual mixing to convex combination (spectral norm ≤ 1), preventing signal blowup across loops | Enables 3-loop stability (Q-gap +0.0076 vs prior catastrophic 3-loop failure) | Required for 3-loop stability, but hurts alone |
| **Timestep scaling** | Per-iteration learned scale vectors (capped ±4.0), stored as float16 passthrough | Q-gap −26–30% | Helps quantization, not training |
| **Prelude-coda** | Unique first/last layers, shared middle blocks (Huginn-style) | −0.016 BPB (screening) | Biggest single BPB win |
| **LeakyReLU(0.5)²** | Preserves negative signal through quadratic activation | Adopted from SOTA | Necessary with Output-LN to avoid dead neurons |
| **FiLM bias** | Per-iteration shift vectors (β), completing scale+shift FiLM conditioning | −0.003 BPB | Clean free win, additive with gammas |

## Key Findings

- **Timestep scaling helps quantization, not training.** Pre-quant BPB is unchanged (1.2578 vs 1.2580), but Q-gap drops 26–30%. The gammas are float16 passthrough params that bypass int8 quantization entirely.
- **Birkhoff alone hurts.** Run C' (Birkhoff only) is +0.002 BPB worse than bare recurrence. It only helps when paired with Output-LN.
- **Q-gap scales with training duration.** Screening (2000 steps) shows Q-gap +0.0019. Full-scale (10k steps) shows +0.0076–0.0126. Screening underestimates the quantization problem by 4–7×.
- **Output-LN is the critical technique.** Without it, mixing alphas collapse to ~0.48 (uniform) and MLP scale drops to 0.2–0.3. With it, alphas learn a meaningful gradient (0.37→0.70 across layers).
- **Prelude-coda gives the biggest single improvement** (−0.016 BPB at screening). Boundary layers need unique parameters.
- **3 loops are viable for the first time.** Q-gap +0.0076 at 3 loops, vs prior 3-loop attempts that failed catastrophically (PR #363 measured ~900× quantization amplification over 3 cycles).
- **Attention-only sharing (shared attention, unique MLPs per iteration) gives −0.026 post-Q BPB** — the largest improvement found — but costs +3.9MB artifact (14.65MB total), leaving insufficient room for SOTA features.
- **FiLM bias adds −0.003 BPB at both 2 and 3 loops** with zero artifact/throughput cost. Additive with timestep scaling gammas.
- **ALBERT (Lan et al., 2020) found attention sharing is nearly free while FFN sharing causes most degradation.** s3_L confirms: the model needs per-iteration MLP differentiation, not per-iteration attention differentiation.
- **Learned depth embeddings and unique input norms both hurt BPB despite reducing Q-gap.** The throughput overhead (6–15%) costs training steps that outweigh any specialization benefit. Learned depth embeddings remained near zero even after full training (RMS 0.006–0.010), suggesting they need far more steps to become useful. FiLM bias alone (s3_O: 1.2625) remains the best full-sharing config.
- **The 0.026 BPB gap between full sharing and unique MLPs (s3_L) cannot be closed by cheap per-iteration input controls.** The MLP genuinely needs different weights per iteration, not just different inputs.
- **Sinusoidal depth encoding (UT-style) is neutral on BPB (1.2624 vs 1.2625 without) but strictly better than learned depth embeddings (1.2624 vs 1.2639)** due to zero throughput overhead. Keep it on since it's truly free — zero params, zero artifact cost, zero throughput penalty, marginal Q-gap benefit (0.0073 vs 0.0078).

## Techniques Applicable to Non-Recurrent Submissions

Output-LN could benefit any submission using quadratic activations (relu², leaky_relu²) — it lets the MLP see unnormalized inputs while bounding its output, which may improve gradient flow in deeper networks. Birkhoff mixing is a drop-in replacement for learned residual mixing with fewer parameters and bounded spectral norm. Per-layer scaling vectors (the non-recurrent version of timestep scaling) add ~4KB of float16 params that survive quantization and could reduce Q-gap on any deep submission.

## Screening Results (2000 steps, 1×H100)

| Run | Config | Post-Q BPB | Q-Gap | Δ vs B' (bare) |
|-----|--------|-----------|-------|-----------------|
| B' | 4×2 bare recurrence | 1.3637 | +0.0024 | — |
| C' | 4×2 + birkhoff only | 1.3660 | +0.0024 | +0.002 |
| C | 4×2 + peri + birkhoff | 1.3587 | +0.0020 | −0.005 |
| D | 4×2 + peri + birk + timestep | 1.3584 | +0.0019 | −0.005 |
| E | 1+3×2+1 all fixes | 1.3428 | +0.0019 | −0.021 |
| F | 1+2×3+1 all (3 loops) | 1.3622 | +0.0019 | −0.002 |

## Full-Scale Results (600s, 8×H100)

| Run | Config | Eff. Layers | Pre-Q BPB | Post-Q BPB | Q-Gap |
|-----|--------|-------------|-----------|------------|-------|
| H | 1+4×2+1 peri+birk | 10 | 1.2578 | 1.2704 | +0.0126 |
| I | 1+4×2+1 peri+birk+ts(cap4) | 10 | 1.2580 | 1.2668 | +0.0088 |
| J | 1+4×3+1 peri+birk (3 loops) | 14 | 1.2567 | 1.2670 | +0.0103 |
| **K** | **1+4×3+1 peri+birk+ts(cap4)** | **14** | **1.2583** | **1.2659** | **+0.0076** |

## Series 3: Follow-up Ablations (600s, 8×H100)

| Run | Config | Eff. Layers | Pre-Q BPB | Post-Q BPB | Q-Gap |
|-----|--------|-------------|-----------|------------|-------|
| L | 1+4×2+1 shared-attn + unique MLPs | 10 | 1.2333 | 1.2406 | +0.0073 |
| N | 1+4×2+1 full share + FiLM bias | 10 | 1.2555 | 1.2641 | +0.0086 |
| O | 1+4×3+1 full share + FiLM bias | 14 | 1.2547 | 1.2625 | +0.0078 |

Run M (1+4×3+1 attn-only, 3 loops) crashed during torch.compile with 12 UniqueMLP modules. Works without compile (verified via smoke test).

## Series 4: Depth Embeddings + Unique Norms (600s, 8×H100)

| Run | Config | Eff. Layers | Pre-Q BPB | Post-Q BPB | Q-Gap |
|-----|--------|-------------|-----------|------------|-------|
| P | 1+4×2+1 learned depth+norms+bias | 10 | 1.2579 | 1.2663 | +0.0084 |
| Q | 1+4×3+1 learned depth+norms+bias | 14 | 1.2574 | 1.2643 | +0.0069 |
| R | 1+4×3+1 learned depth only+bias | 14 | 1.2566 | 1.2639 | +0.0073 |
| S | 1+4×3+1 norms only+bias | 14 | 1.2560 | 1.2629 | +0.0069 |

## Series 5: Sinusoidal Depth Encoding (600s, 8×H100)

| Run | Config | Eff. Layers | Pre-Q BPB | Post-Q BPB | Q-Gap |
|-----|--------|-------------|-----------|------------|-------|
| T | 1+4×3+1 sinusoidal depth+bias | 14 | 1.2551 | 1.2624 | +0.0073 |

## Next Direction

The depth encoding question is resolved: sinusoidal depth encoding (UT-style) is free and marginally helpful — keep it on. The best full-sharing config is now s5_T (1.2624 post-Q) with the complete stack: Output-LN + Birkhoff mixing + FiLM scale+shift (gammas+betas) + sinusoidal depth encoding.

The remaining path is **grafting the validated technique stack onto the SOTA stack** for a competitive submission.

## How to Reproduce

The submitted run (Run K) from repo root:

```bash
# 8×H100 (600s wallclock cap)
SEED=1337 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=200 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  NUM_LAYERS=14 NUM_PRELUDE=1 NUM_SHARED=4 NUM_LOOPS=3 NUM_CODA=1 \
  USE_PERI_NORM=1 USE_BIRKHOFF_MIX=1 USE_TIMESTEP_SCALE=1 TIMESTEP_GAMMA_MAX=4.0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Full ablation via bundled scripts:

```bash
# Screening (7 runs, on 1xH100)
bash scripts/run_screening.sh

# Full-scale (5 runs, on 8×H100)
bash scripts/run_fullscale.sh

# Follow-up ablations (4 runs, on 8×H100)
bash scripts/run_fullscale2.sh
```

### Environment Variables

| Variable | Run K Value | Description |
|----------|-------------|-------------|
| `NUM_PRELUDE` | 1 | Unique prefix layers |
| `NUM_CODA` | 1 | Unique suffix layers |
| `NUM_SHARED` | 4 | Shared blocks in the recurrent loop |
| `NUM_LOOPS` | 3 | Loop iterations over shared blocks |
| `USE_PERI_NORM` | 1 | Output-LN (norm on MLP output, not input) |
| `USE_BIRKHOFF_MIX` | 1 | Sigmoid-constrained residual mixing |
| `USE_TIMESTEP_SCALE` | 1 | Per-iteration learned scale vectors |
| `TIMESTEP_GAMMA_MAX` | 4.0 | Cap for timestep gammas (0 = uncapped) |
| `LEAKY_RELU_SLOPE` | 0.5 | Negative slope for leaky relu² (0.0 = relu²) |

## Files

```
├── train_gpt.py           # Modified training script
├── train_log.txt           # Run K log (primary submission)
├── submission.json         # Competition metadata
├── research_notes.md       # Theory + citations
├── ablation2_report.md     # Full comparison report (Series 1–3)
├── logs/                   # All run logs
│   ├── s1_Ap–F.txt         #   Screening (7 runs)
│   ├── s2_G–K.txt          #   Full-scale (5 runs)
│   ├── s3_L–O.txt          #   Follow-up ablations (4 runs)
│   ├── s4_P–S.txt          #   Depth embeddings + unique norms (4 runs)
│   └── s5_T.txt            #   Sinusoidal depth encoding (1 run)
└── scripts/
    ├── run_screening.sh    # Series 1 screening
    ├── run_fullscale.sh    # Series 2 full-scale
    ├── run_fullscale2.sh   # Series 3 follow-up ablations
    ├── run_fullscale3.sh   # Series 4 depth embeddings + unique norms
    └── run_fullscale4.sh   # Series 5 sinusoidal depth encoding
```

## Links

See [research_notes.md](research_notes.md) for theory, citations, and detailed technique descriptions.
