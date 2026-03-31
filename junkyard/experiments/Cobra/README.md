# COBRA: Base-Quality Racecar Harness (10-Min Timer)

## Mission
Optimize **base model quality only** for the 10-minute training budget.

- Primary metric: `final_int6_sliding_window_exact val_bpb`
- Secondary metric: `DIAGNOSTIC post_ema val_bpb` (fallback if run exits early)
- Budget target: `MAX_WALLCLOCK_SECONDS=600`
- Scope: model quality before any n-gram/mixer boost

## Why Cobra
Recent in-repo logs show the base model cluster is tight (~`1.1190` to `1.1206` BPB), so we need a disciplined, low-noise harness.

Known anchors:
- A-WING GREEN_1 reference base: `1.11947678` (`logs/awing_green1_s1337_SOTA_0.3200_20260326.log`)
- Best observed base in local logs: `1.11901519` (`logs/f1_car02_iso_var_t2_rope24_ngram5_s1337_20260325_025620.log`)

## H100 Stability Standards Applied
Cobra bakes in the edge-case guardrails from the H100 research:

1. Keep tensor-core-friendly shapes and alignment (no odd/prime architectural pivots in critical dims).
2. Avoid varlen attention path surprises during base training/eval (uniform training shape).
3. Keep toolchain conservative (`CUDA 12.8` recommended for Hopper FA3 performance consistency).
4. Use a fixed evaluation target (`final_int6_sliding_window_exact`) for rankability.

## Files
- `profiles/green1_reference.env`: faithful baseline profile from `A_wing/green_1`
- `profiles/cobra_base_quality.env`: base-quality profile (n-gram eval disabled)
- `candidates.json`: candidate override matrix for ablations
- `cobra_harness.py`: plan/run/summarize harness
- `run_plan.sh`: prints commands and race plan (no training launch)
- `RACECAR_PLAN.md`: execution playbook
- `HYPOTHESIS.md`: compact experiment hypothesis and risk map

## Quick Start (plan only)
```bash
bash experiments/Cobra/run_plan.sh
```

## Optional: summarize existing Cobra logs
```bash
python3 experiments/Cobra/cobra_harness.py summarize --glob "logs/cobra_*.log"
```
