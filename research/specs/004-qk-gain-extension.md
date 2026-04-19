# Spec 004 — QK-gain extension screen (short 8×H100 training)

**Slug:** `qk-gain-extension`
**Created:** 2026-04-19
**Links to idea:** (new — incremental extension of SOTA's `QK_GAIN_INIT=5.25`)

## Hypothesis
QK-gain has monotonically improved across recent records (4.0 → 5.0 → 5.25), and no submission has tested values above 5.25. Higher qk_gain makes attention more peaked at initialization, which apparently helps our architecture. This is a **triage screen**: run two short 8×H100 training runs at QK_GAIN_INIT 6.0 and 5.5, compare `train_loss` at matched step milestones against spec 000's log, decide whether either value deserves a full-training follow-up.

## Baseline
**Spec 000's train.log** is the control. Same seed (42), same code, same everything except QK_GAIN_INIT=5.25. At matched step numbers with identical seed, data ordering is identical, so any train_loss difference is caused by the QK_gain change alone.

### Spec 000 reference milestones (from `runs/000-sota-replication/train.log`)

| step | train_loss |
|---|---|
| 500 | 3.3098 |
| 1000 | 3.2487 |
| 1500 | 3.0884 |
| 2000 | 2.9431 |

(Logged every 500 steps in spec 000; we match at these milestones.)

## Expected Δ
**Unclear.** QK=5.25 was a monotonic extension of the 4.0→5.0→5.25 trend, but extrapolation can hit a ceiling. Possible outcomes:
- QK=6.0 is too sharp, attention becomes near-one-hot at init, gradients vanish, training diverges or plateaus.
- QK=6.0 is fine but gives negligible improvement.
- QK=5.5 or 6.0 gives small positive Δ (optimistic: continuing the monotonic trend).

## Accept criteria
This is a triage screen, not a record attempt. Criteria are coarse:

- **Promising:** variant train_loss ≤ spec 000's at ≥3 of 4 matched milestones (steps 500, 1000, 1500, 2000) by a visible margin (≥0.005). Follow-up with a full 8×H100 retrain.
- **Kill:** variant train_loss > spec 000's + 0.05 at any milestone, OR NaN/explicit divergence.
- **Inconclusive:** within ±0.02 of spec 000 at all milestones → record finding, decide whether to commit to a full retrain based on budget priorities.

## Config diff vs spec 000
Only **one env var differs per run**:

| Env var | Spec 000 | Run A | Run B |
|---|---|---|---|
| `QK_GAIN_INIT` | 5.25 | **6.0** | **5.5** |
| `BIGRAM_VOCAB_SIZE` | 0 | 0 | 0 |
| `TTT_ENABLED` | 1 | 1 | 1 |
| `SEED` | 42 | 42 | 42 |
| `TRAIN_LOG_EVERY` | 500 (default) | **200** (finer own-trajectory data) | 200 |
| `MAX_WALLCLOCK_SECONDS` | 600 | **300** (5 min cap) | 300 |

Note: `TRAIN_LOG_EVERY=200` gives extra data points for our own trajectory, but comparison with spec 000 only happens at the 500-step milestones where spec 000 logged.

Note: `TTT_ENABLED=1` matches spec 000 to keep the forward-pass config identical. TTT is post-training only, so doesn't affect training dynamics — but we never reach TTT in a 5-min run anyway.

## Code changes
- Branch: `research`
- Commit: `feaf45e`
- Diff: **none.** QK_GAIN_INIT is already an env var (`train_gpt_sota.py` around the qk_gain construction). Pure hyperparam spec.

## Hardware ladder
- [ ] 8×H100 NA-1 — **only rung**. Matches spec 000's setup so same-seed comparison is clean.
- [ ] 2×H100 / 1×H100 — not used. Would reduce throughput and change the pod-shape from spec 000.

Both runs on **the same pod sequentially** (don't re-provision). Pod shape changes shouldn't matter for matched-step-number comparison, but keeping runs on one pod is simpler and avoids two provisioning cycles.

## Seed plan
Single seed (42) per run, matching spec 000. Deterministic data ordering.

## Inputs
- Data: `/workspace/data/datasets/fineweb10B_sp8192/`
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`
- Hotstart: **none** — both runs from scratch.
- Reference log: `runs/000-sota-replication/train.log` (in-repo). Already committed.
- Base repo commit: `feaf45e` on `research`.

## Execution protocol
Sequential on same pod, 8×H100 NA-1:

```bash
# Run A: QK_GAIN_INIT=6.0 (more dramatic first, per user preference)
BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=6.0 TTT_ENABLED=1 SEED=42 \
TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py \
  > /workspace/runs/004-qk-gain-extension/qk_6.0_train.log 2>&1

# Wait for completion. If kill gate triggered (divergence / NaN), SKIP Run B.

# Run B: QK_GAIN_INIT=5.5
BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.5 TTT_ENABLED=1 SEED=42 \
TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py \
  > /workspace/runs/004-qk-gain-extension/qk_5.5_train.log 2>&1
```

Execution can skip post-training evals (quant, sliding, TTT) — we're only watching train_loss. If the model script doesn't support "skip post-training eval," `MAX_WALLCLOCK_SECONDS=300` naturally caps the training phase; subsequent post-training steps are short enough that they complete anyway (~1 min extra). Either way, ignore post-training numbers in this spec.

## Stop-early criteria
**For Run A (QK=6.0), applied live:**

| At step | Condition | Action |
|---|---|---|
| 200 | variant train_loss > 3.5 | **Kill** — divergence likely |
| 500 | variant > spec 000's 3.3098 + 0.05 (= 3.36) | **Kill** — clearly worse than baseline |
| 1000 | variant > spec 000's 3.2487 + 0.05 (= 3.30) | **Kill** |
| Any step | NaN / step_time > 2× | **Kill** |

If Run A is killed, **skip Run B** (5.5 is less extreme but if 6.0 diverges we've already learned the direction).

**For Run B (QK=5.5):** same-shape gates (compare to spec 000 milestones + 0.05).

## Cost estimate
- 8×H100 NA-1 at ~$23.92/hr ($0.40/min).
- Run A: ~5 min training + ~1 min post-training overhead = ~6 min ≈ **$2.40**.
- Run B: same ≈ $2.40.
- Pod provisioning overhead: ~5 min = **$2**.
- **Total: ~$7** for both runs. **~$4.50** if Run A kills and we skip Run B.

## Extra artifacts
- `runs/004-qk-gain-extension/qk_6.0_train.log` — full stdout for Run A.
- `runs/004-qk-gain-extension/qk_5.5_train.log` — full stdout for Run B (if run).
- `runs/004-qk-gain-extension/compare.md` — matched-step table: step, spec_000_loss, qk_6.0_loss, qk_5.5_loss, Δ columns. **Primary artifact.**
- `runs/004-qk-gain-extension/notes.md` — execution narrative (was Run B skipped? did any gate fire?).

No `.pt` checkpoints retained (no hotstart will be done from these trajectories). No `.ptz` artifacts.

## Open questions for interview
- Confirm 8×H100 NA-1 availability. If not, fallback to 2×H100 40min? **No** — that changes the pod shape from spec 000 and makes the seed-matched comparison noisy. Wait for 8×H100.
- Confirm `spec 000`'s train.log is readable on the pod (it's in-repo, should be).
- Confirm the script supports `MAX_WALLCLOCK_SECONDS=300` cap — default is 600 (10 min); 300 is below that, should just cap training early.
- **Don't re-provision between Run A and Run B.** If the first run is interrupted by kill gate, tear down the variant's stale process but keep the pod alive for Run B (if applicable).

## What this spec does NOT do
- Does not train to completion — we stop at ~5 min, well before post-training phases.
- Does not produce a `.ptz` or final bpb — we only look at `train_loss`.
- Does not commit to a full retrain — that's a follow-up spec (spec 005 would be "best-QK full retrain") if either value looks promising.
- Does not provide submission-quality bpb — triage only.
