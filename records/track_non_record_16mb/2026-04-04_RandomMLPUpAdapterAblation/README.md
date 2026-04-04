# Non record: Random MLP Up Adapter Ablation

This experiment starts from the current root `train_gpt.py` and adds one narrow change only.

Selected MLP up projections are replaced with seeded frozen QR random feature maps plus:

1. learned per feature gain
2. learned rank 16 low rank correction

Attention remains fully learned. MLP down projections remain fully learned. The goal is to isolate whether random feature expansion is useful when routing stays learned.

## Configs

`baseline_12l`

1. 12 layers
2. 512 model dim
3. 8 heads
4. 4 KV heads
5. MLP mult 2
6. no random MLP up layers

`random_up_12l_5layers_rank16`

1. same 12 layer stack
2. layers `0,1,2,3,4` use frozen random MLP up projections
3. each frozen up projection has learned gain plus rank 16 low rank correction

## Extra Eval

The trainer keeps the existing final roundtrip eval and adds a final sliding window eval controlled by:

1. `FINAL_SLIDING_EVAL`
2. `EVAL_STRIDE`
3. `EVAL_SEQ_LEN`

Both exact metrics are logged separately.

## Reproduce

```bash
bash run.sh baseline_12l
bash run.sh random_up_12l_5layers_rank16
```
