# Spec 013 — BigramHash auxiliary embedding (port from #1716)

**Slug:** `bigram-hash`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/bigram-hash-embed.md`.

## Hypothesis

Adding an explicit 16384-bucket × 32-dim hash-keyed bigram embedding table, additively fused with `tok_emb(input_ids)` before block 0, gives the model an "input-layer" bigram prior without making attention/MLP rediscover first-order token co-occurrence. Most competitive submissions have a BigramHash; #1736 does not.

## Baseline

Spec 008's seed-42 val_bpb (`runs/008-1736-reproduction/seed_42/final.json`). Comparison is Δ vs that number.

## Expected Δ

**−0.001 to −0.003 bpb** on top of #1736. Rough: #1716 reported −0.00218 vs #1493; discounted by TTT-absorption and by #1736's already-capable stack (parallel residuals, SmearGate, AttnOutGate may already partially learn first-order bigrams implicitly).

## Accept criteria

- Training completes without NaN / divergence.
- Artifact **< 16 MB** (hard gate — this is the first added-params spec).
- Post-quant post-TTT val_bpb measured.
- **Decision criterion:**
  - Δ ≤ −0.002 → promote, run 3-seed confirmation.
  - Δ ∈ (−0.002, −0.0005] → promote cautiously; stack with spec 011 winner if that lands.
  - Δ ∈ (−0.0005, +0.001) → null; shelve, reclaim the ~400KB of artifact for something else.
  - Δ > +0.001 → regression; verify the hash/init isn't buggy, try smaller dim=16 if curious.

## Config diff vs spec 008

```
BIGRAM_HASH_ENABLED=1
BIGRAM_HASH_BUCKETS=16384    # default; matches #1716
BIGRAM_HASH_DIM=32           # default; matches #1716
```

All other env vars unchanged. Defaults for PRIME_A/PRIME_B match #1716.

## Code changes

- **Branch:** `exp/bigram-hash` (worktree at `worktrees/bigram-hash/`).
- **Commit:** `66e57bf`.
- **Patch target:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`.
- **Patch scope:** 110 LOC pure addition (no existing lines altered). Adds:
  - `BigramHashEmbedding(nn.Module)` class with `embed` (nn.Embedding) + zero-init `proj` (CastedLinear).
  - 5 new `Hyperparameters` fields (all default to no-op when unset).
  - `GPT.__init__`: creates `self.bigram_embed` when enabled, else None.
  - `forward_logits` + `forward_ttt`: additive merge after `tok_emb`, before SmearGate.
  - `Optimizers`: `embed.weight` → AdamW with `embed_wd`; `proj.weight` → Muon `matrix_params`.
  - GPTQ hessian hooks on both submodules.
- **Default-off invariant:** with `BIGRAM_HASH_ENABLED` unset, forward + state_dict + optimizer param list are byte-identical to baseline. All new code is attr-gated on `bigram_embed is None` or `bigram_hash_enabled`.

## Hardware ladder

- [x] **Artifact dry-run** (before any pod spend): confirm spec 008 artifact has room. Target: spec 008's current `final.int6.ptz` size + ~425 KB added (393 KB int6 embed + 32 KB fp16 proj). If spec 008 is ≤ 15.55 MB, we fit; if larger, need to trim before running.
- [x] **Smoke test: 2×H100, short (~5 min, ~$1).** Purpose: crash/NaN/shape check. `ITERATIONS=500 BIGRAM_HASH_ENABLED=1 torchrun --nproc_per_node=2 train_gpt.py`. Don't read val_bpb.
- [x] **8×H100 full training run, seed 42.** ~$20. Read post-TTT val_bpb from `final.json`.

### Early-stop guidance

Same protocol as spec 011: executor + user monitor `train.log` via `tail -f`. Compare train_loss vs spec 008 at matched step. Kill on NaN, step-time blow-up, or consistently worse trend (joint judgment). Default to finish when ambiguous.

### Pre-registered expectations (what the curve should look like)

**Important context:** the projection is zero-init. At step 0 the bigram contribution is literally zero and grows with training as `proj` and `embed` learn useful values. Effect on train_loss grows *quadratically* with `||proj||`, so expect **no early signal and late-training divergence**.

Mapped to #1736's ~4500 total steps:

| Step range | Expected behavior | Interpretation |
|---|---|---|
| **0–300** (startup) | train_loss within noise of spec 008 | projection ≈0, bigram contributes nothing |
| **300–1000** (warmup) | still near-zero Δ; <0.001 nats | gradient just starting to concentrate on high-frequency buckets |
| **1000–2500** (mid) | **first meaningful check**; divergence 0.002–0.005 nats if it works | bigram table has learned common-pair patterns |
| **2500–4500** (settled) | gap widens to 0.005–0.015 nats final train_loss Δ | fully warmed up bigram, gain plateaus |
| **Post-TTT val_bpb** | −0.001 to −0.003 bpb | TTT absorbs some upstream gain (spec 010 finding); reported Δ smaller than train_loss Δ |

**Decision thresholds at specific steps:**

- **Step 1500–2000 is the first real check.** If no visible divergence (bigram run ≡ spec 008 ±0.001 nats), the bigram table isn't gaining traction. Two possibilities: (a) projection learning too slowly (second-run LR tweak candidate); (b) #1736's existing capacity (SmearGate, parallel residuals) already captures first-order bigram signal → null result. Either way, **don't kill here** — let it finish to confirm the null.
- **Step 2500+ is decisive.** Consistently worse than spec 008 = kill candidate (discuss). Consistently better or matched = let it finish.

**Would be surprising (investigate immediately):**

- **Very early divergence (step <500).** Zero-init projection can't produce fast effects; early divergence indicates a bug (wrong init, optimizer hit, hash collision collapse, shape mismatch finding a new path).
- **Train_loss lower but val_bpb higher post-eval.** Bigram overfitting to training bigrams that don't generalize. Unusual in a 600s run.
- **Step time >> spec 008.** The bigram forward is a cheap lookup + linear; it should not add visible step-time overhead. If it does, something's off (e.g., CPU-GPU transfer, unfused kernel).

## Seed plan

Single seed (42) for screen. 3-seed only if it lands Δ ≤ −0.002.

## Inputs

- **Data:** same CaseOps dataset as spec 008.
- **Tokenizer:** bundled with #1736 submission dir. No retokenization needed; bigram hash is a runtime computation on token IDs.
- **Hotstart:** none, full from-scratch training.

## Execution protocol

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/013-bigram-hash/seed_42

NCCL_NET=Socket DATA_DIR=./data \
ARTIFACT_DIR=/workspace/runs/013-bigram-hash/seed_42 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
BIGRAM_HASH_ENABLED=1 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/013-bigram-hash/seed_42/train.log 2>&1
```

Expected startup log:
- `bigram_hash: enabled=True buckets=16384 dim=32 primes=(36313,27191)`

## Checkpoints to emit

- `final_model.pt` (pre-GPTQ FP).
- Submission artifact + `final.json`.

## Stop-early criteria

- NaN in train_loss → halt.
- Step time > 2× spec 008 → halt.
- Artifact > 16 MB → halt, flag (this is the first added-params spec; this is the real risk).

## Cost estimate

| Item | Cost |
|---|---|
| Artifact dry-run | $0 (local check) |
| 2×H100 smoke | ~$1 |
| 8×H100 full run | ~$20 |
| **Total** | **~$21** |

## Open questions for interview

1. If artifact dry-run shows we're over 16MB with bigram at int6 + full 8-bit tok_emb, options: (a) drop tok_emb to int7, (b) drop bigram to int4, (c) reduce buckets to 8192 or dim to 16. Pick one before running; default pick is (a) since tok_emb int7 is already used on similar-tier submissions (#1586).
2. Does the hash's CaseOps collision profile matter enough to verify before running? Plan: skip verification for this pass. If results are puzzling, run a quick CPU script to check collision rates vs SP8192-plain.
3. Run in parallel with spec 011 (separate pod) or sequential? Plan: parallel — two pods, clean attribution, ~$40 total.

## What this spec does NOT do

- Does not change the tokenizer.
- Does not change any non-embedding hyperparameter.
- Does not stack with spec 011 (WD + GradPower) in this run — if both land, stacking is a follow-up spec.
- Does not run 3-seed — single-seed screen only.
- Does not try smaller buckets (8192) or larger dim (64) — single config matching #1716.
