---
name: promote
description: Invoke when an experiment's val_bpb_post_quant beats the current best in `winners/`. Carries the keep/judgment-call thresholds, the cp/journal/results.tsv/git ritual, and the heading-craft rules so the new winner is grep-findable later. Always journal even on direct-promote — the previous session left wins without journal entries and that broke search.
---

# Promote

A new winner just landed. This skill carries the full ritual so promotions are consistent, journaled, and grep-findable.

## 1. Decide if it's actually a win

**Hard rule for SSM-family experiments**: do NOT promote any SSM-family win before the `noise-floor-sentinel` skill has completed for that architecture family. Treat apparent wins as informational only until you have the family's measured σ. The thresholds below were calibrated to the transformer noise floor (~0.0024 cross-seed; advance=0.010 ≈ 4.17σ); for SSM, replace them with σ-anchored thresholds using the same shape (advance Δ ≥ Y where Y = ≥3σ; judgment-call [Y/2, Y]; noise <Y/2) once the sentinel completes. The judgment-low = advance/2 ratio matches the transformer rule. Document the adjusted Y and σ in journal.md Current threads.

Compare the new `val_bpb_post_quant` against the current best in `winners/` (lower is better). The Δ rules:

- **Δ ≥ +0.010** → likely real, advance.
- **Δ ∈ [+0.005, +0.010]** → judgment call. Re-run with `SEED=42` first; advance only if Δ holds across both seeds.
- **Δ ≥ +0.050** → suspiciously large. Re-run with `SEED=42` before promoting, no exceptions.
- **Δ < +0.005** → noise, don't promote.

For wins in [+0.010, +0.050] you may direct-promote without SEED=42 — but plan to run the SEED=42 confirm within the next 5 experiments. Direct-promotes inflate Δ by ~10–20% on average; the confirm reins it back.

## 2. Promote ritual

```bash
mkdir -p winners
DEST="winners/$(date +%Y-%m-%d)_<descriptive_slug>"
cp -r experiments/NNNN_<slug> "$DEST"
rm -f "$DEST"/final_model.pt        # too big to commit; .int8.ptz stays
```

Use a slug that describes the *delta from canonical*, not just the experiment id. Example: `2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_24k_matrix_lr_045_init_05_muon_backend_15`. The slug is how future agents recognize this winner without opening the env.sh.

## 3. Journal — always, even direct-promote

The previous session direct-promoted exps 0021, 0023, 0024, 0051 without journal entries; the resulting "no heading found" gap broke `grep "^## "` discovery. Don't repeat it. Even a one-paragraph entry is enough:

```markdown
## YYYY-MM-DD · exp NNNN · short-title-with-finding-in-it

**Question**: …
**Setup**: …
**Prediction** [CONFIDENCE_TAG]: …
**Disconfirming**: …
**Result**: val_bpb post X (was Y), Δ=+Z. Pre-quant Δ=…, quant_tax=…
**Conclusion** [CONFIDENCE_TAG]: …
```

**Heading craft** — the heading is the search target, write it accordingly:
- Surface the *finding*: "MUON_BACKEND_STEPS=15 wins clean +0.013" beats "MUON sweep continued".
- Use the env-var name or technique slug a future agent would grep.
- Disambiguate cousins: "NUM_LAYERS=11 ceiling" not bare "depth ceiling".

Also update **Current threads** in `journal.md` at the top — change "Best so far" to the new winner. Move the displaced winner to "Prior winner" line.

## 4. results.tsv

Find this experiment's row, change `status` from TODO to `keep`, fill description (6–10 words + transfer tag):
- `[transfer:high]` — robust scaling/architectural; expect to hold at H100 20k-step.
- `[transfer:med]` — hyperparameter tuning; transfer depends on training-length dynamics.
- `[transfer:low]` — exploits early-training behavior; may not survive longer schedules.

## 5. Commit

```bash
git add winners/ journal.md results.tsv
git commit -m "Promote NNNN_<slug>: val_bpb X (was Y)"
```

No need to ask the human. Promote, commit, continue.

## 6. The next experiment is already half-formed

A promote means a new architecture is now the strongest base — which immediately opens new directions on top of it. Before leaving this skill, name at least one concrete next experiment that builds on the new winner: a different config of the same architecture, a stacked technique you haven't tried, the SEED=42 confirm if direct-promoted, a sweep that was previously cap-blocked but isn't anymore. Write its plan.md or queue it in `scratch/parking_lot.md` immediately. The ritual ends; the work doesn't.

If direct-promoted (Δ ∈ [+0.010, +0.050] without SEED=42), the SEED=42 confirm is the natural first follow-up — schedule it within the next 5 experiments and journal the cross-seed Δ when it lands.
