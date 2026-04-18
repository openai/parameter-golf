# Non record: Random MLP Up Adapter Ablations

This experiment starts from the current root `train_gpt.py` and adds narrow random MLP up ablations only.

Selected MLP up projections are replaced with seeded frozen QR random feature maps plus:

1. learned per feature gain
2. learned rank 16 low rank correction
3. optional routed multi-basis expert gating

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

`random_up_moe_12l_5layers_e2`

1. same 12 layer stack
2. layers `0,1,2,3,4` use frozen random MLP up projections
3. each selected layer is split into 2 routed random expert subspaces behind a tiny token router
4. the heavy up projection stays single pass by concatenating expert bases into one random weight
5. this config disables the low rank correction path to isolate the routed multi-basis variant

`random_up_moe_12l_5layers_e2_rank8`

1. same 12 layer stack
2. same 2 expert routed random basis construction as `random_up_moe_12l_5layers_e2`
3. adds a small rank 8 learned correction on top of the routed expert path
4. intended as the next budget-conscious comparison against pure routed experts

## Extra Eval

The trainer keeps the existing final roundtrip eval and adds a final sliding window eval controlled by:

1. `FINAL_SLIDING_EVAL`
2. `EVAL_STRIDE`
3. `EVAL_SEQ_LEN`

Both exact metrics are logged separately.

## Early Smoke Results

Short single GPU smoke runs were used to sanity check learning dynamics before a full length comparison.

`random_up_moe_12l_5layers_e2`

1. run with `TRAIN_BATCH_TOKENS=65536` and `MAX_WALLCLOCK_SECONDS=180`
2. stopped at step `768`
3. reached `train_loss=2.5663` at step `750`
4. finished with `val_loss=2.6259` and `val_bpb=1.5552`
5. averaged about `234.6 ms` per optimizer step

This is materially stronger than the random guess starting point for a `1024` token vocabulary, which is about `ln(1024) ~= 6.93`. Even without completing the full schedule, the model clearly learns useful token structure and exits the near uniform regime quickly. Admittedly it is far below other competitors and it is questionalbe whether or not the compute and latency costs are worthwhile with gains. Probably looking forward need to look at efficiency. 

`random_up_moe_12l_5layers_e2_rank8`

1. matched the same short wall clock smoke setup
2. reached `train_loss=3.7947` at step `100`
3. reached `train_loss=2.5555` at step `750`
4. averaged about `235.8 ms` per optimizer step

The rank `8` correction path did not show a clear early training advantage over the pure routed `e2` variant in this short run. The train curves were very close, so the smoke result should be read as roughly neutral rather than a win or loss for the added correction path.

Overall, these partial runs suggest the random MLP up construction is viable enough to train a competent model under the existing recipe, but they do not yet show a decisive benefit from the small low rank correction. A longer run with scheduled intermediate validation is still needed for a confident ranking across variants.

## Reproduce

```bash
bash run.sh baseline_12l
bash run.sh random_up_12l_5layers_rank16
bash run.sh random_up_moe_12l_5layers_e2
bash run.sh random_up_moe_12l_5layers_e2_rank8
```
