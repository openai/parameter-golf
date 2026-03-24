Experiment 2 Results Summary

┌─────────────────────────┬───────────────────────┬────────────────────────┐
│ Metric │ Exp 1 (LoRA TTT SOTA) │ Exp 2 (Register Token) │
├─────────────────────────┼───────────────────────┼────────────────────────┤
│ Final val_bpb (sliding) │ 1.1540 │ 1.1617 │
├─────────────────────────┼───────────────────────┼────────────────────────┤
│ Training steps │ 4917 │ 4259 (13% fewer) │
├─────────────────────────┼───────────────────────┼────────────────────────┤
│ Step avg │ ~122ms │ ~141ms │
├─────────────────────────┼───────────────────────┼────────────────────────┤
│ Artifact size │ 16.9 MB (zlib) │ 17.13 MB (zlib) │
├─────────────────────────┼───────────────────────┼────────────────────────┤
│ Peak VRAM │ ~20 GB │ 23.7 GB │
└─────────────────────────┴───────────────────────┴────────────────────────┘

Interpretation

Per-step, the register mechanism is actually helping. At equal step counts:

- Step 2000: Exp 2 = 1.2555 vs Exp 1 = 1.2620 (Exp 2 is 0.0065 better)
- Step 2500: Exp 2 = 1.2320 (Exp 1 was ~1.2400)

But the register adds ~19ms/step overhead (141ms vs 122ms), costing ~660
training steps in the 10-min window. That lost training time erases the
per-step gain.

Three things to fix for the next run

1. bf16 cumsum (already in your local file) — drops step time from 141ms to
   ~127ms → ~4724 steps instead of 4259
2. Artifact size — 17.13 MB > 16 MB limit. The code used zlib despite zstd
   being installed. Need to verify the import works at runtime
3. Eval speed — ran with eval_batch_seqs=32 (old default) instead of 128+.
   Eval took 233s (3.9 min). With batch_seqs=256 + stride=128, it would take ~15s

Want me to apply all three fixes to the local file and prepare a re-run? Or
shall we pivot to Experiment 3 (Hermeneutic Depth Recurrence) which could be
more impactful?

rsync -avz --no-perms --no-owner --no-group \
-e "ssh -p 22044 -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no" \
root@63.141.33.73:/workspace/parameter-golf/records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/final_model.int8.ptz \
~/Documents/YerevaNN/openai-parameter-golf/parameter-golf/records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/
