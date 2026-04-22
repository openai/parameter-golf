# Spec 024 seed 42 — checkpoint pointers

All checkpoints live on NA-1 volume `hvpdph5i3g`. Not in git.

| file | step | size | state |
|---|---|---|---|
| `/workspace/runs/024-learnable-alpha-detached-lerp/seed_42/final_model.pt` | 4975 (end) | 135,593,212 bytes | post-EMA, pre-GPTQ |
| `/workspace/runs/024-learnable-alpha-detached-lerp/seed_42/final_model.int6.ptz` | 4975 (end) | 15,948,713 bytes | GPTQ int6 quantized + brotli (submission payload) |

Submission total (incl. code + tokenizer): 15,979,588 bytes (under 16MB cap).
