# Gate Report

Formal local pre-submit check:

```text
runs
seed    train_ms    train_step    prequant_bpb    sliding_bpb    sliding_eval_ms    artifact_bytes    code_size_bytes    sentinel
42      600024      4702          1.09071600      1.08802944     89275              15633824          19689              1
1337    600117      4699          1.09073547      1.08783878     89174              15630505          19689              1
2024    600093      4702          1.09049411      1.08760994     89318              15630862          19689              1

checks
artifact<=16000000          1
train<=600000ms             0
train_step>=4000            1
final_sliding_eval<=600000ms 1
sentinel_present_all        1
required_seeds_present      1
mean_sliding_bpb            1.08782605
record_track_gate           0

verdict FAIL
reason training wallclock gate failed
reason record-track mean gate failed
```

The logs should be rerun with a slightly lower training cap if this candidate is intended for a strict 10-minute record-track package.
