# Best Results

## Current Best Known Run

- `budget_twice_eval2048_ttt1024`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.38414876`
- `final_int8_ttt_lora val_bpb: 1.3962`
- `Total submission size int8+zlib: 11280566 bytes`

This is the current baseline to protect. New experiment packs should be judged against this run first.

## Restore Point

Code branch:

- `codex/runpod-2026-03-20-checkpoint`

Runpod sync:

```bash
cd /workspace/parameter-golf
git fetch myfork
git reset --hard myfork/codex/runpod-2026-03-20-checkpoint
```

Current best log:

```bash
tail -n 20 logs/budget_twice_eval2048_ttt1024.txt
```
