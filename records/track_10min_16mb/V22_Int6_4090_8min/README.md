# V22 Int6 Model (RTX 4090, ~8min)

## Overview
Lightweight 16MB-constrained language model optimized for fast convergence.

## Specs
- GPU: RTX 4090 (single)
- Training time: ~8 minutes (1000 steps)
- Model size: 11.39MB
- Val bpb: ~2.05

## Key Features
- Int6 quantization
- Patchin architecture
- Fast convergence under strict constraints

## Notes
Designed to reach reasonable performance quickly rather than maximizing final accuracy.

Using device: cuda:0
🏁 Training start (Target: 1000 steps, ~8-9 mins)
step    1/1000 | loss 0.7010 | lr 3.00e-05 | time 1s
step   50/1000 | loss 6.0079 | lr 1.50e-03 | time 24s
step  100/1000 | loss 5.0116 | lr 3.00e-03 | time 47s
step  150/1000 | loss 4.3609 | lr 2.98e-03 | time 71s
step  200/1000 | loss 4.3035 | lr 2.91e-03 | time 95s
step  250/1000 | loss 4.2361 | lr 2.81e-03 | time 118s
step  300/1000 | loss 4.1687 | lr 2.66e-03 | time 142s
step  350/1000 | loss 4.1514 | lr 2.48e-03 | time 166s
step  400/1000 | loss 4.0403 | lr 2.28e-03 | time 190s
step  450/1000 | loss 3.8124 | lr 2.05e-03 | time 214s
step  500/1000 | loss 3.6194 | lr 1.80e-03 | time 237s
step  550/1000 | loss 3.5323 | lr 1.55e-03 | time 261s
step  600/1000 | loss 3.4364 | lr 1.30e-03 | time 285s
step  650/1000 | loss 3.3566 | lr 1.05e-03 | time 309s
step  700/1000 | loss 3.2829 | lr 8.25e-04 | time 333s
step  750/1000 | loss 3.2876 | lr 6.18e-04 | time 357s
step  800/1000 | loss 3.2433 | lr 4.39e-04 | time 380s
step  850/1000 | loss 3.2307 | lr 2.94e-04 | time 404s
step  900/1000 | loss 3.2049 | lr 1.87e-04 | time 428s
step  950/1000 | loss 3.1943 | lr 1.22e-04 | time 452s
step 1000/1000 | loss 3.1987 | lr 1.00e-04 | time 476s
--- 評価中 (Eval) ---
--- step 1000 | val_loss 4.5614 | val_bpb 2.0505

--- 圧縮 & 保存 ---
✅ 完成！ サイズ: 11.39 MB (target <=16)
📂 保存先: /workspace/parameter-golf/submission.int6.ptz
