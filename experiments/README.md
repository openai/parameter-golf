# Experiments

Minimal tracking. One line per completed run goes into `registry.jsonl`.

## Schema

```json
{"id":"20260414_loop_aware_v1","date":"2026-04-14","hardware":"2xH200","seed":42,"steps":2227,"val_bpb":1.0812,"quantized_bpb":1.0856,"wallclock_s":4832,"commit":"abc1234","notes":"int8 on v_proj blocks 4-5, int6 elsewhere. +0.003 vs baseline."}
```

Fields are whatever you want — `id`, `date`, `val_bpb`, `commit`, `notes` are the minimum. Add anything else useful (optimizer config, base PR it stacks on, etc.).

## Workflow

1. Start a branch: `git checkout -b exp/<axis>-<name>-v<N>`
2. Run your experiment, save logs.
3. Append one line to `registry.jsonl` with the result.
4. Write `<id>.md` if the run taught you something worth remembering.
5. Commit + push. Concurrent appends will auto-merge.
