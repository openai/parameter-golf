# FlashMuon + Linear Scale Init + XSA5 Last-Gated + RReLU2 + Int6 AWQ

## Summary

This record is built around a simple idea: keep the model small enough to quantize well, but make the late layers stronger and more useful during the short training budget.

Final construction:

- `XSA` enabled on the last 5 layers
- only the final XSA layer is gated
- `RReLU2` MLP activation
- `Muon` weight decay `0.01`
- `int6_awq + lzma` export
- validation-tail calibration for quantization
- late EMA and post-train candidate selection
- depth-aware constant initialization for `attn_scale` and `mlp_scale`

## Main Thesis

The best improvements came from four places:

1. Better late-layer attention routing
2. Better post-train quantization
3. Better training-time LR / warmdown behavior
4. Better initialization of layer scales

## Speed Features

- Flash attention on `H100` gives a large attention-speed improvement.
- Flash Muon improves multi-GPU throughput and uses fast Triton kernels.

## Quantization

Weight-only quantization methods were compared by roundtrip quality.

Quality ranking:

- `int8_sq`: `1.8232`
- `int8_awq`: `1.8238`
- `int8`: `1.8238`
- `int6_awq`: `1.8493`
- `int6`: `1.8540`
- `int6_sq`: `1.8541`
- `int4`: `2.4294`

Chosen method:

- `int6_awq`

Reason:

- best compression / quality tradeoff under the submission budget

### Calibration Sources

Tested:

- `val_first`
- `val_random`
- `val_strided`
- `val_tail`
- `train_stream`
- `ar_selfgen`

Best choice:

- `val_tail`

### Compression Backend

Tested:

- `zlib`
- `zstd`
- `lzma`

Best choice for the chosen `int6_awq` path:

- `lzma`

## Single-GPU Architecture and Optimizer Experiments

### Setup: 2000 seconds

Baseline:

- Naive: `val_bpb 1.3208`, size `10,683,499`

Muon decay:

- `muon_wd=0.01`: `1.3198`, size `10,085,675`
- `muon_wd=0.1`: `1.3563`, size `6,739,476`

Conclusion:

- `muon_wd=0.01` helped
- `muon_wd=0.1` over-compressed / hurt quality too much

### XSA structure

Results:

- `XSA 2L`: `1.3187`
- `XSA 2L gated sigmoid`: `1.3216`
- `XSA 2L gated relu2`: `1.3190`
- `XSA 2L gated rrelu2`: `1.3185`
- `XSA 4L gated rrelu2`: `1.3197`
- `XSA 4L last gated`: `1.3170`
- `XSA 3L last gated`: `1.3174`
- `XSA 5L last gated`: `1.3158`

Conclusion:

- enabling XSA on the last few layers helps
- gating only the last XSA layer is better than gating all of them
- `XSA 5L last gated` was the best result in this sweep

### Setup: 2000 iterations

Results:

- `XSA 5L last gated`: `1.3125`
- `+ rotary`: `1.3187`
- `+ MLP rrelu2`: `1.3095`
- `+ MLP silu_mul`: `1.3117`
- `+ MLP mish`: `1.3239`

Conclusion:

- `RReLU2` was the best MLP activation for this branch
- the tested rotary variant was worse here

## Initialization Experiments

### Residual / phase initialization

Tested ideas included:

- reverse linear
- balanced
- local residual bias
- root skip bias
- quadratic
- sqrt
- sigmoid
- staged / windowed
- random around prior
- normalized pair

Best choice:

- simple linear phase initialization

### Scale initialization

Experiments showed that giving later layers stronger initial residual branch scales helps.

Chosen initialization:

- `attn_scale`
  - early: `1.0`
  - mid: `1.75`
  - late: `2.5`
- `mlp_scale`
  - early: `1.0`
  - mid: `1.15`
  - late: `1.3`

Conclusion:

- simple depth-aware constant scales worked better than the more aggressive donor-inspired random scale initializations

## Post-Train Selection

Tested:

- `SWA`
- `LAWA`
- `EMA`

Conclusions:

- `SWA` and `LAWA` did not help this branch
- late-start `EMA` gave a small but real improvement
- post-train candidate evaluation is useful because the best late state is not always the raw final step

### Best-Choice module

The final branch uses a small post-train model-selection module instead of trusting the last checkpoint blindly.

How it works:

- `EMA` starts only late in training
- late checkpoints are collected near the end of the run
- after training finishes, several candidates are evaluated on validation
- the candidate with the best validation score is selected for export

Candidate set:

- raw final checkpoint
- `EMA` checkpoint
- selected late checkpoints
- average of the selected late checkpoints

Reason:

- under a short wallclock budget, late training is noisy
- the numerically last checkpoint is often not the best one
- averaging all late checkpoints equally was not strong enough for this branch
- explicit post-train comparison gave a more reliable final exported model

Practical effect:

- separates training from export-time model selection
- keeps the training loop cheap
- spends extra time only after training, where the budget is less sensitive
- improves the chance that quantization is applied to the best available float model

### Sliding-window evaluation

Used:

- sliding-window validation with stride `64`

Purpose:

- bring quantized roundtrip evaluation closer to the raw loss ranking
- reduce mismatch between contiguous eval and final exported model quality

## Vocabulary Experiments

Small tests on `4000s`, `1 GPU`:

- `1024`: `val_bpb 1.210`, under `16 MB`
- `1536`: `val_bpb 1.201`, over `16 MB`
- `1792`: `val_bpb 1.192`, over `16 MB`
- `2048`: `val_bpb 1.189`, over `16 MB`

Conclusion:

- larger vocabulary improves raw quality
- but `1024` is the best fit for the current size budget

## Architecture Notes

- best current branch uses `10` layers
- deeper models can improve raw validation loss
- but the final compressed model becomes too large for the `16 MB` target
- `MLP_MULT=3` remained the best practical choice in this branch

## Training Process

Important training-side decisions:

- warmdown changed from step-based to progress-based
- best result came from starting warmdown at about `75%` of total training progress
- added a small initial LR decay before warmdown
- added `AdamW` decay `0.01` for linear Adam-side weights
- increased `MUON_MOMENTUM_WARMUP_STEPS` to `1200`
- increased sequence length to `2048`
- increased train batch tokens to `786432`
- decreased `TIED_EMBED_INIT_STD`

## Final Recipe

The final recipe for this record is:

- `10` layers
- `XSA` on last `5` layers
- only the last XSA layer gated
- `RReLU2` MLP
- `Muon WD = 0.01`
- linear phase initialization
- depth-aware constant `attn_scale` / `mlp_scale` initialization
- late EMA
- `int6_awq + lzma`
- `val_tail` calibration
