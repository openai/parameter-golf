# Stage 3.3 Hypothesis Discovery

Date: 2026-03-25

## Abstract Problem

Fixed wallclock (600s), maximize deployed val_bpb (post-quant roundtrip).
The training loop has ~20 hyperparameters, all static constants or fixed schedules.
The optimization landscape changes dramatically across phases (bulk → warmdown → QAT).
Static hyperparams are optimal for at most one phase.

The hidden trade: hyperparameter quality vs. implementation complexity.
Each dynamic hyperparam adds a control law that could help or hurt.

## Attack Surface Map

### Currently static (never changes during training)

| Hyperparam | Value | Causal role | Patch surface |
|---|---|---|---|
| `grad_clip_norm` | 0.3 | Gradient safety | training loop, 1 line |
| `muon_wd` | 0.04 | Regularization, param magnitudes → quant damage | Muon optimizer step |
| `adam_wd` | 0.04 | Same for embeddings/scalars | Adam optimizer step |
| `muon_backend_steps` | 5 | Newton-Schulz iterations, update quality vs throughput | Muon.step() |
| `train_seq_len` | 2048 | Context window, attention cost is O(n²) | data loader call |
| `ema_decay` | 0.997 | Averaging weight for exported checkpoint | EMA update block |
| `swa_every` | 50 | SWA collection frequency | SWA block |
| `logit_softcap` | 30.0 | Output clipping | model forward |

### Currently has a fixed schedule

| Hyperparam | Schedule | What's missing |
|---|---|---|
| LR | Cosine warmdown from `scale=1.0` to `0.0` | Only dynamic thing in the loop. Schedule is wallclock-based, not loss-based. |
| Muon momentum | Linear warmup 0.92→0.95 over 1500 steps, then fixed | No warmdown adaptation. |
| QAT | Binary on at `scale < 0.15` | No gradual transition (stage3_1 H6 addresses this). |

### Available training-state signals (already computed, zero cost)

| Signal | Where | What it tells you |
|---|---|---|
| `train_loss` | Every step | Current learning rate |
| `scale` | Every step | Position in warmdown (1.0=bulk, 0.0=end) |
| `step` | Every step | Absolute progress |
| `elapsed_ms` | Every step | Wallclock budget remaining |
| `val_loss`, `val_bpb` | Every 200 steps | Generalization quality |

### Signals computable cheaply but not currently tracked

| Signal | Cost | What it tells you |
|---|---|---|
| Gradient norm (pre-clip) | ~free, already computed for clipping | Whether gradients are stable, exploding, or vanishing |
| Loss velocity (EMA of loss deltas) | ~free, 1 line | Whether the model is still learning fast or has plateaued |
| Update-to-weight ratio | cheap, per param group | Whether updates are meaningful relative to param magnitudes |

## Operator-Generated Hypotheses

### `stage` — Different rules by phase

**H1: Warmdown weight decay ramp**
- Currently: WD=0.04 constant
- Change: WD ramps from 0.04 to 0.10 during warmdown (WD = 0.04 + 0.06 * (1 - scale))
- Why: Higher WD during warmdown shrinks weight magnitudes → less dynamic range → less quantization damage. The model should "compress itself" before export.
- Bottleneck attacked: quant damage

**H2: Warmdown momentum drop**
- Currently: momentum=0.95 fixed after warmup
- Change: momentum = 0.95 * scale + 0.80 * (1 - scale) during warmdown, i.e. decays from 0.95 to 0.80
- Why: High momentum carries history from a landscape that's deforming under decreasing LR + QAT activation. Lower momentum gives fresher gradient information for navigating the quantization-aware landscape.
- Bottleneck attacked: optimization quality during warmdown

### `reallocate` — Move budget between subsystems

**H3: Throughput-aware Muon NS steps**
- Currently: backend_steps=5 constant
- Change: backend_steps=7 during bulk (scale>=0.5), backend_steps=3 during deep warmdown (scale<0.5)
- Why: During warmdown, LR is already small so update direction quality matters less. Fewer NS iterations = faster steps = more gradient updates in remaining wallclock. During bulk, higher-quality updates are worth the cost.
- Bottleneck attacked: throughput/quality tradeoff
- Key signal: step_avg_ms should drop during warmdown phase

**H4: Phase-aware sequence length**
- Currently: train_seq_len=2048 constant
- Change: train_seq_len=1024 during bulk (scale>=0.5), switch to 2048 for warmdown (scale<0.5)
- Why: Attention is O(n²). At seq_len=1024 with same total tokens per batch, attention is 4x cheaper per token. More updates per wallclock during bulk when the model doesn't yet need long-range context. Switch to full context during warmdown for eval-aligned convergence.
- Bottleneck attacked: throughput during bulk
- Expected: 15-25% faster steps during bulk → 15-25% more total gradient updates

### `factorize` — Different rules for different param families

**H5: Per-family warmdown decay rates**
- Currently: All param families decay at the same rate (scale)
- Change: embed_lr *= scale^1.5 (faster decay), matrix_lr *= scale (normal), scalar_lr *= sqrt(scale) (slower decay)
- Why: Embeddings are the most quantization-sensitive and should converge first. Scalars (LN, gates) stay in fp32 and benefit from continued fine-tuning. Matrix weights are in between.
- Bottleneck attacked: per-family convergence quality

### `internalize` — Pull deploy signal into training

**H6: Loss-velocity warmdown gating**
- Currently: Warmdown starts at a fixed wallclock fraction
- Change: Track EMA of loss velocity (d(loss)/d(step)). Delay warmdown until velocity drops below threshold. If model is still improving fast, keep training at full LR.
- Why: Fixed wallclock-based warmdown may start too early, wasting productive bulk training steps. Loss velocity is a direct signal of "the model is done learning fast."
- Bottleneck attacked: bulk/warmdown time allocation
- Key signal: warmdown_start_step should be later than the default schedule when the model is still learning

### `invert` — Challenge the dominant assumption

**H7: Warmdown grad clip tightening**
- Currently: grad_clip=0.3 constant
- Change: grad_clip = 0.3 during bulk, tighten to 0.1 during warmdown (clip = 0.3 * scale + 0.1 * (1-scale))
- Why: During warmdown, the optimizer is trying to settle into a good basin. Large gradient spikes from occasional hard batches knock it out of the basin. Tighter clipping during warmdown is analogous to reducing temperature in annealing — let the system settle rather than jump.
- Bottleneck attacked: convergence stability during warmdown

### `relax` — Remove unnecessary precision

**H8: EMA decay annealing**
- Currently: ema_decay=0.997 constant
- Change: ema_decay = 0.995 during bulk, ramp to 0.9995 during warmdown
- Why: During bulk, the model changes rapidly — EMA should track closely (lower decay). During warmdown, updates are small and noisy — EMA should average more aggressively (higher decay) to smooth out noise. The exported checkpoint is the EMA state, so this directly affects deployed quality.
- Bottleneck attacked: export checkpoint quality

## Coverage Matrix

| Family | train_quality | quant_damage | throughput | per_family | deploy_alignment | process_control |
|---|---|---|---|---|---|---|
| H1 WD ramp | secondary | **primary** | none | none | secondary | none |
| H2 momentum drop | **primary** | secondary | none | none | none | none |
| H3 NS steps | secondary | none | **primary** | none | none | none |
| H4 seq length | **primary** | none | **primary** | none | none | secondary |
| H5 per-family decay | **primary** | secondary | none | **primary** | none | none |
| H6 velocity gating | **primary** | none | secondary | none | none | **primary** |
| H7 clip tightening | secondary | secondary | none | none | none | **primary** |
| H8 EMA annealing | none | secondary | none | none | **primary** | none |

### Coverage gaps

- **Data/context split**: H4 partially covers this (sequence length by phase)
- **Checkpoint selection**: Covered in stage3_1 H8, not duplicated here
- **Export policy**: Not in scope (stage3_3 is training-side only)

### Set-level assessment

- 6 distinct bottleneck families covered
- No two hypotheses have the same causal story
- H4 (seq length) and H3 (NS steps) both attack throughput but via orthogonal mechanisms
- H1 (WD ramp) and H7 (clip tightening) both affect warmdown convergence but through different control surfaces

## Scoring

| Family | Distinctness | Upside | Observability | Stackability | Bar pass? |
|---|---|---|---|---|---|
| H4 seq length | high | **large** (0.005-0.015) | screen (step_avg_ms) | high | **YES — lead** |
| H6 velocity gating | high | **medium** (0.003-0.008) | screen (warmdown_start_step) | high | **YES — lead** |
| H5 per-family decay | high | **medium** (0.003-0.008) | screen (per-family val) | medium | **YES — lead** |
| H3 NS steps | high | **medium** (0.003-0.008) | screen (step_avg_ms) | high | **YES — lead** |
| H1 WD ramp | medium | **small-med** (0.002-0.005) | screen (param magnitudes) | high | **YES — lead** |
| H2 momentum drop | medium | **small** (0.001-0.003) | screen | high | YES — support |
| H7 clip tightening | medium | **small** (0.001-0.003) | screen | high | YES — support |
| H8 EMA annealing | low | **small** (0.001-0.003) | only long run | medium | YES — support |

## Shortlist for Pack Design

**Lead hypotheses (6):**
1. H4 — phase-aware sequence length (throughput reallocation)
2. H6 — loss-velocity warmdown gating (process control)
3. H5 — per-family warmdown decay rates (factorize)
4. H3 — throughput-aware Muon NS steps (throughput reallocation)
5. H1 — warmdown weight decay ramp (internalize quant damage)
6. H7 — warmdown grad clip tightening (convergence stability)

**Support hypotheses (2):**
7. H2 — warmdown momentum drop
8. H8 — EMA decay annealing

**Controls (2):**
- R0A — baseline control
- R0B — baseline control (different seed)

## Negative Knowledge

Do NOT regenerate:
- Fixed-schedule retunes (e.g. "try WD=0.03 instead of 0.04") — these are coefficient scans, not mechanisms
- Batch size changes that require changing grad_accum_steps mid-run (DDP sync pattern complications)
- Gradient noise estimation across microbatches (too expensive, unreliable at small batch)
- PID controller on gradient norms (adds complexity for likely tiny gain)
- Any hypothesis that cannot state what training-state signal drives the adaptation
