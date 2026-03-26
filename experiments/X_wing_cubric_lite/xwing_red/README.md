# xwing_red

Pod-ready test lane for PR779-style script validation.

## Source
- Base lane: `records/track_10min_16mb/2026-03-25_PodracerIII_cubric_lite_8xH100`
- Test script copied from: `experiments/pr779_asap_test/train_gpt.py`

## Goal
- Launch a clean pod test quickly with reproducible env defaults.

## Launch
From repo root:

```bash
bash experiments/X_wing_cubric_lite/xwing_red/run.sh
```
