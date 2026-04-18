# Spec 003 — BigramHash paired from-scratch screen

**Slug:** `bigram-hash-screen`
**Created:** 2026-04-19
**Links to idea:** `research/ideas/bigram-hash.md`

## Hypothesis
Adding BigramHash (3072 hash buckets × 112 dim ≈ 344K params) to our SOTA stack improves pre-quant val_bpb. We don't test the 16MB fit here — the artifact will be oversized (~16.2 MB). Spec 003 is a **signal screen**: does BigramHash actually help on our architecture? Budget-fit engineering is deferred to spec 004 (only if this wins).

Comparison is primarily via **training-loss trajectory at matched steps**, not just post-training eval. BigramHash activates from step 1 (`train_gpt_sota.py:474, 553-554`, zero-init → no disruption), so a paired run with same seed gives a deterministic train_loss comparison all the way through.

## Baseline
Control run within this same spec (paired). Nominal reference: Exp 24 (2×H100, 40-min, SOTA replication, 1.0867 post-quant). Control's trajectory here should match Exp 24's shape.

## Expected Δ
+0.002 to +0.005 bpb at end-of-training pre-quant (per original BigramHash submission's claim). Train_loss curves should diverge from ~step 500 onward if the signal is real.

## Accept criteria
- **Validity:** Control run's end-of-training pre-quant val_bpb is within ±0.003 of Exp 24's 1.0985 pre-quant. (Loose tolerance because pod-to-pod throughput varies; both our control and variant sit on the same pod, so their Δ is still clean.)
- **Signal:** variant ≤ control at ≥3 of last 4 logged train_loss milestones AND end-of-training pre-quant Δ ≤ **−0.002** vs control.

## Config diff
**Hyperparam-only. Two paired runs on the same pod.**

| Env var | Control | Variant |
|---|---|---|
| `BIGRAM_VOCAB_SIZE` | 0 | 3072 |
| `BIGRAM_DIM` | (n/a) | 112 |
| `QK_GAIN_INIT` | 5.25 | 5.25 |
| `TTT_ENABLED` | 1 | 1 |
| `SEED` | 42 | 42 |
| `TRAIN_LOG_EVERY` | 200 (tightened from 500 default for fine-grained comparison) | 200 |

**Same seed is critical** — identical data ordering is what makes matched-step comparison meaningful.

## Code changes
- Branch: `research`
- Commit: `b44c34e`
- Diff: **none.** BigramHash is already implemented in `train_gpt_sota.py` (L96-98 config, L432-460 class, L474 instantiation, L553-554 forward).

## Hardware ladder
- [ ] 2×H100 NA-1 — **only rung**. Both runs sequential on the **same pod** (no re-provisioning between them — critical for hardware-controlled Δ).
- [ ] 8×H100 — not used. This is a screen, not a submission attempt.

## Seed plan
Single seed (42), but **both runs use the same seed.** This is intentional — we want identical data ordering so that train_loss divergence can be attributed to the architecture, not to data variance.

## Inputs
- Data: `/workspace/data/datasets/fineweb10B_sp8192/`
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`
- Hotstart: **none** — both runs are from-scratch.
- Base repo commit: pin current `research` HEAD.

## Checkpoints to emit
**None retained.** These are screen models, not submission candidates, and we won't hotstart from either trajectory (spec 000's checkpoints already cover equivalent ground).

## Execution protocol
Sequential on the same pod:

```bash
# Run 1: control
BIGRAM_VOCAB_SIZE=0 BIGRAM_DIM=112 \
QK_GAIN_INIT=5.25 TTT_ENABLED=1 SEED=42 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt_sota.py \
  > /workspace/runs/003-bigram-hash-screen/control_train.log 2>&1

# Wait for completion. If early-kill triggered per gates below, skip variant.

# Run 2: variant
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
QK_GAIN_INIT=5.25 TTT_ENABLED=1 SEED=42 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt_sota.py \
  > /workspace/runs/003-bigram-hash-screen/variant_train.log 2>&1
```

Both runs with the same 40-minute wall-clock cap (default in Hyperparameters). Step count per run is hardware-dependent (~2000-3000 on 2×H100 in 40 min based on Exp 24).

## Stop-early criteria
**Applied to the variant run only (control is always run to end as reference):**

| After step N | Gate | Action |
|---|---|---|
| 500 | Variant within ±0.05 of control train_loss | Continue — noise dominates early |
| 1000 | Variant train_loss > control + 0.01 | **Kill variant.** BigramHash clearly hurting. Save ~25 min pod time. |
| 1500 | Variant train_loss > control − 0.005 | Weak signal; continue but flag for ambiguous-case handling |
| 2000+ | Variant ≤ control − 0.01 | Strong signal; run to end |

Standard: NaN / step-time > 2× expected / divergence → kill and mark failed.

## Cost estimate
- 2×H100 NA-1 at ~$6/hr.
- Two 40-min runs sequential = 80 min pod time = **~$8**.
- With early-kill at step 1000 (variant clearly bad): **~$5**.

## Extra artifacts
- `runs/003-bigram-hash-screen/control_train.log` — full stdout (train_loss at every 200 steps).
- `runs/003-bigram-hash-screen/variant_train.log` — full stdout.
- `runs/003-bigram-hash-screen/final.json` — both runs' final metrics: `{control: {pre_quant_bpb, quantized_bpb, sliding_bpb, post_ttt_bpb, step_count, seconds}, variant: {same fields}, delta: {all four stages}}`.
- `runs/003-bigram-hash-screen/loss_compare.md` — **primary artifact.** Matched-step train_loss table with columns: step, control_loss, variant_loss, Δ. Must include ALL logged milestones, not just final.
- `runs/003-bigram-hash-screen/notes.md` — execution narrative.

## Open questions for interview
- Confirm no re-provisioning between the two runs. **Critical.** Hardware-controlled Δ fails if pods differ.
- Confirm execution's interpretation of the early-termination gates — if unclear, err toward **running both to completion** and defer kill-or-promote to research. The extra $3 is cheap insurance.
- Confirm the 40-min wall cap matches `MAX_WALLCLOCK_SECONDS=2400` (Exp 24 harness). If Hyperparameters default differs, explicitly set `MAX_WALLCLOCK_SECONDS=2400` in the launch env.
- Confirm TRAIN_LOG_EVERY=200 doesn't materially slow down training (it shouldn't — just more log-line prints).
