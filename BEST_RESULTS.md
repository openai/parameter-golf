# Best Results

## Current Best Known Run

- `top10_winner_wd04`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.37843732`
- `final_int8_ttt_lora val_bpb: 1.3908`
- `Total submission size int8+zlib: 11219666 bytes`

This is the current baseline to protect. New experiment packs should be judged against this run first.

## Key Takeaway

- `MUON_WEIGHT_DECAY=0.04` is the strongest winner-adjacent change tested so far.
- Broad moonshots underperformed this branch.
- The next search should stay close to `winner_wd04`.

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
tail -n 20 logs/top10_winner_wd04.txt
```
