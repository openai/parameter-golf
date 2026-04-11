# Better Baseline Params

This record starts from [2026-03-17_LoRA_TTT](../2026-03-17_LoRA_TTT). The code path is unchanged from that record: [train_gpt.py](train_gpt.py) differs from [2026-03-17_LoRA_TTT/train_gpt.py](../2026-03-17_LoRA_TTT/train_gpt.py) only inside `Hyperparameters`, so the per-document LoRA TTT evaluation is the same mechanism as the 2026-03-17 base.

## Diff vs [2026-03-17_LoRA_TTT](../2026-03-17_LoRA_TTT)

- `TRAIN_BATCH_TOKENS`: `524288 -> 262144`
- `TRAIN_SEQ_LEN`: `1024 -> 2048`
- `NUM_LAYERS`: `9 -> 8`
- `MODEL_DIM`: `512 -> 768`
- `LOGIT_SOFTCAP`: `30.0 -> 10.0`
- `TIED_EMBED_LR`: `0.05 -> 0.03`
- `MATRIX_LR`: `0.04 -> 0.02`
- `SCALAR_LR`: `0.04 -> 0.02`
- `BETA1`: `0.9 -> 0.70`

