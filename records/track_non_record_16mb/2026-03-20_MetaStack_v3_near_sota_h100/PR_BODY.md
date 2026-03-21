## Summary

This PR adds a near-SOTA non-record submission for Parameter Golf:

- best confirmed sliding-window result: `1.1762` BPB
- artifact size under cap: `12,109,181` bytes total
- two confirmation seeds at `1.1762` and `1.1792`
- stride-32 ablation included as a negative result (`1.1788`)

## Why non-record

The strongest runs were close to the main leaderboard cutoff but stopped slightly over the strict `600s` training limit (`600073ms` and `600090ms`). Because of that timing caveat, this PR is submitted to the non-record track rather than claiming a strict leaderboard record.

## Included files

- `train_gpt.py` used for the H100 runs
- `train_seed42.log` best confirmed run
- `train_seed1337.log` confirmation run
- `stride32_eval.log` negative ablation
- `README.md`
- `submission.json`

## Key metrics

- best sliding-window score: `1.1762`
- pre-quant terminal BPB: `1.1796`
- int8 roundtrip BPB: `1.1819`
- int6 roundtrip BPB: `1.2110`
- eval config: `seq_len=1024`, `stride=64`
