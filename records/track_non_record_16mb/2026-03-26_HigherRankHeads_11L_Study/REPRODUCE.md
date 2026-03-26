# Reproduction

The study runner is self-contained:
- it uses the `train_gpt.py` included in this folder
- it ensures the full `fineweb10B_sp1024` dataset is present (`80` training shards)
- it installs or reuses a Hopper-only FA3 wheel before training
- it runs the full 7-variant family on the intended fast path
- it copies the fresh per-run console logs for the rerun into `logs/` inside this folder

## Run The Full Family

```bash
bash records/track_non_record_16mb/2026-03-26_HigherRankHeads_11L_Study/run_higher_rank_heads_study.sh
```

This reruns the full family:
- `H0`: standard tied head
- `H1`: factorized `r=64`
- `H2`: factorized `r=128`
- `H3`: mixture-softmax `K=2`, `r=64`
- `H4`: mixture-softmax `K=4`, `r=64`
- `H5`: mixture-softmax `K=4`, `r=128`
- `H6`: simplex `128`

Expected budget:
- `7` runs
- about `70` minutes total on `8xH100`
