# F1 SOTA Garage

This garage is a clean three-car workspace cloned from the latest F1 gold-standard profile:

- source profile: `concepts/f1/run_legal_lb.sh`
- source trainer: `concepts/f1/train_gpt.py`
- gold reference run (Mar 24, 2026): `legal_ttt_exact val_bpb: 1.11951975`

## Cars

- `car01_gold_reference`: exact gold baseline copy (control car)
- `car02_speed_lane`: clone for speed-focused experiments
- `car03_quality_lane`: clone for quality-focused experiments

All three start from the same legal SOTA baseline so you can compare changes apples-to-apples.

## Research Tagging

Each car folder carries a `HYPOTHESIS.md` with explicit question, prediction,
status, and verdict so runs remain auditable.

## Quick Run

```bash
SEED=1337 bash concepts/f1_sota_garage/car01_gold_reference/run.sh
SEED=2025 bash concepts/f1_sota_garage/car02_speed_lane/run.sh
SEED=7    bash concepts/f1_sota_garage/car03_quality_lane/run.sh
```
