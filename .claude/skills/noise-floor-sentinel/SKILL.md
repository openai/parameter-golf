---
name: noise-floor-sentinel
description: Invoke before treating any Δ as signal on a new SSM architecture family. Runs 3 same-config seeds to characterize the empirical noise floor for THIS architecture class. Distinct from program.md "Regression sentinel" (harness-drift check) — this is architecture-family variance characterization, which Mamba's LR cliffs (primer §4.2) make even more important than for transformers.
---

# Noise-Floor Sentinel

## Hard rule
**No SSM-family experiment is promoted before this skill completes for that family.** Until you have the family's measured σ, treat any apparent win as informational only — do not invoke `promote`. The previous transformer session piled up single-seed direct-promote-zone wins before cross-seed confirms (see archived methodology-debt entries); Mamba's sharp LR cliffs make freak-good first-seed runs more likely than in the transformer regime, so the same anti-pattern is more dangerous here.

## When to invoke
Concrete gate: invoke only after a config that meets ALL of:
- Forward pass produces no NaN/Inf
- Step-1 train_loss ≈ ln(vocab) (≈ 6.93 at sp1024); step 2 within ~2× of step 1; monotonic descent thereafter (per `launch-and-await` trajectory checks, plus the SSM late-NaN gate at step 100)
- val_bpb_post_quant < 2.521 (beats the naive baseline 0001) — characterizing variance of a fundamentally broken config wastes 3× experiments and the σ doesn't generalize

Then invoke:
- After your first stable SSM block in a new architecture family
- When you change the SSM block class (S4D → Mamba-1 → Mamba-2/SSD)
- When you change the SSM placement (single layer replaced → all layers SSM → parallel attn+SSM heads)
- NOT for hyperparameter changes within an already-characterized config (those are signal, not new variance)

## Why it matters
The previous transformer session established a 0.0024–0.003 cross-seed noise floor; program.md's promote thresholds (Δ ≥ +0.010 advance, [+0.005, +0.010] judgment-call) are tuned to it — that's roughly [2σ, 4σ] empirically. SSMs may have a different floor. Per primer §4.2, Mamba has sharp LR cliffs and is more sensitive than transformers to init/hyperparameter perturbation. A "freak good" or "freak bad" single-seed run is more likely. Without knowing the floor, you'll either promote noise or discard real signal.

## Procedure
1. Pick a stable working config of the new architecture family (the one that meets the concrete gate above — it will be your "baseline" for this family). Note its experiment id (e.g., `0073`).
2. Run 3 experiments forking from that stable config — **NOT from canonical**. Use `./new_experiment.sh <slug> <stable_ssm_exp_id>` so the fork inherits the SSM block code, env.sh, and any train_gpt.py edits. Then change ONLY the SEED:
   ```
   ./new_experiment.sh <family>_sentinel_seed42 <stable_ssm_exp_id>
   # in the new experiment's env.sh, set SEED=42 (override the parent's seed)
   # fill plan.md (Question="noise-floor sentinel <family>", Hypothesis="cross-seed Δ within ±2σ", Change="SEED only", Disconfirming="bimodal or wide spread")
   cd experiments/NNNN_<family>_sentinel_seed42 && ../../run_experiment.sh
   ```
   Repeat for 2 more seeds. **Pick seeds disjoint from your stable config's seed.** If the stable config used the canonical `SEED=1337`, run `SEED=42`, `SEED=2024`, `SEED=31337`. (`SEED=2024` and `SEED=31337` haven't been used in this codebase before; they're plain torch RNG seeds and should work — a one-line sanity check that step-1 train_loss differs across the three is cheap insurance against accidentally running duplicates.)
3. Each experiment: status `sentinel`, description "noise-floor sentinel for <family> seed <N>".
4. After all 3 complete, journal with the entry header `## YYYY-MM-DD · <family> noise floor (3 seeds)`:
   - Three val_bpb_post_quant values
   - Mean and standard deviation
   - Cross-seed Δ (max - min)
   - Verdict: noise floor for this family is σ ≈ X
5. **If the three seeds are bimodal** (one freak run + two tight, or two-and-two split — i.e., the spread looks like LR-cliff behavior rather than a smooth gaussian), run a fourth seed before concluding. Mamba's LR cliffs (primer §4.2) make this regime exactly where a 3-seed sample is too small. The fourth seed disambiguates "freak event in tail" vs "true bimodal config."
6. Update journal.md Current threads with the new floor, using thresholds anchored to the measured σ. Use the same formula structure as the transformer rule (judgment_low = advance/2, the empirical 0.5 ratio):
   - **Advance**: Δ ≥ Y where Y is the family's advance threshold. Suggested Y = 3σ (standard "p < 0.003" significance heuristic). Y = 4σ mirrors the transformer's empirical anchor (transformer's advance=0.010 ≈ 4.17σ at σ=0.0024); pick whichever you can defend.
   - **Judgment-call** (re-run with an extra seed; advance only if Δ holds): Δ ∈ [Y/2, Y]
   - **Noise**: Δ < Y/2
   Format: `**<family> noise floor**: σ=X measured exp NNNN-NNNN; advance Δ ≥ Y=3σ=<value>; judgment [Y/2, Y] = [<value>, <value>].`

## Cost
3 experiments (4 if bimodal). Cost depends on architecture family:
- S4D-Lin (FFT-conv, transformer-speed step time): ~15 min total for 3 seeds; ~20 min if bimodal
- Mamba-1 sequential `selective_scan` (3-6× slower per primer §4.1): ~45-75 min for 3 seeds; ~60-100 min if bimodal

Required investment, not optional overhead.

## Distinct from regression-sentinel
| Sentinel | Asks | Cadence |
|---|---|---|
| Regression sentinel (program.md "Regression sentinel") | Is the harness still bit-reproducing 0001? | Every 10 experiments |
| Noise-floor sentinel (this skill) | What is THIS architecture family's cross-seed variance? | Once per architecture-family change |

## After the sentinel
With the floor measured, return to the experiment loop with calibrated thresholds. If the SSM noise floor is e.g. σ=0.008 (3× the transformer floor), then promote-skill `Δ ≥ +0.010` becomes `Δ ≥ +0.024` for THIS family. Document in journal.md Current threads.
