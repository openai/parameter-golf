# SP1024 + Shared-V(last3) 3-seed non-record submission

This is a stable non-record 16MB submission based on the official SP1024 tokenizer and a compact transformer with structured skip fusion.

## Summary

This submission uses:

- official `fineweb_1024_bpe.model`
- standard FineWeb SP1024 dataset
- structured skip fusion (`BIFPN2_MODE=1`)
- XSA on the last 4 layers
- 2-gram scaffold with fade-out
- shared V across the last 3 layers

This submission is intended as a stable, rule-compliant baseline submission rather than a leaderboard-top attempt.

## Representative run

Representative seed: **2027**

Representative exact roundtrip BPB: **1.27717259**

Submission size: **15973626 bytes**

## 3-seed results

| seed | last_val_bpb | roundtrip_exact_val_bpb | submission_bytes |
|------|--------------|-------------------------|------------------|
| 1337 | 1.2791 | 1.28079096 | 15972114 |
| 2027 | 1.2755 | 1.27717259 | 15973626 |
| 3407 | 1.2779 | 1.27952108 | 15975453 |

3-seed mean exact roundtrip BPB: **1.27916154**

## Files

- `submission.json`: metadata for this submission
- `train.log`: representative training log
- `train_gpt.py`: training script snapshot used for this submission
- `config.json`: resolved config for the representative run
- `seed_runs.csv`: all 3 seed results
- `requirements.txt`: minimal environment dependencies

## Main configuration

Key settings:

- tokenizer: SP1024
- `BIFPN2_MODE=1`
- `XSA_ENABLED=1`
- `XSA_LAST_N_LAYERS=4`
- `NGRAM_MAX_N=2`
- `NGRAM_FADE_ENABLE=1`
- `CROSS_LAYER_KV_SHARING_ENABLED=1`
- `CROSS_LAYER_KV_SHARE_V=1`
- `CROSS_LAYER_KV_PAIRWISE=0`
- `CROSS_LAYER_KV_PARTIAL_HEAD=0`

## Notes

- This submission does **not** modify the tokenizer or dataset.
- This is a reproducibility-focused non-record submission under the 16MB artifact limit.
- The representative run uses seed 2027 because it was the best run among the 3 submission seeds.

## Reproduction

Typical command pattern:

```bash
python launchv3.py config_submission_sharev3_3seed.json \
  --train-script mytrain_gpt_v2_1.py \
  --output output/submission_sharev3_3seed \
  --stop-mode steps \
  --max-steps 3000 \
  --only submission_seed2027
'''
