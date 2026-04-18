# Checkpoints — spec 000 run

All checkpoints live on the NA-1 network volume `hvpdph5i3g` at `/workspace/runs/000-sota-replication/checkpoints/`, **not in git**. Each is ~300 MB (model + optimizer states + EMA). Total: ~2.7 GB.

| Step | File | Event |
|---|---|---|
| 455 | `ckpt_event_step455.pt` | periodic (CKPT_STEPS) |
| 1048 | `ckpt_warmdown_start_step1048.pt` | warmdown activated (frac > 1 − 0.72 = 0.28, step 1048/3849 ≈ 0.272 of real run, 0.052 of 20000 max) |
| 1137 | `ckpt_event_step1137.pt` | periodic (CKPT_STEPS) |
| 1378 | `ckpt_pre_recurrence_step1378.pt` | just before depth recurrence activated (frac ≥ 0.35) |
| 1500 | `ckpt_event_step1500.pt` | muon momentum warmup end (auto) |
| 2275 | `ckpt_event_step2275.pt` | periodic (CKPT_STEPS) |
| 3412 | `ckpt_event_step3412.pt` | periodic (CKPT_STEPS) |
| 3849 | `ckpt_final_pre_ema_step3849.pt` | end of training loop, before EMA |
| 3849 | `ckpt_final_post_ema_step3849.pt` | after EMA applied |

Note: the SOTA planning doc expected CKPT_STEPS fractions of 4550; this run only reached 3849 due to ~15% throughput deficit vs SOTA pod. The periodic steps (455, 1137, 2275, 3412) still land as specified, but represent 12/30/59/89% of the run instead of 10/25/50/75%.
