# Baseline run (local)

## Summary

###best val_bpb = 1.3529 @ step 4200###
Ran the parameter-golf NaiveBaseline on a single local GPU and observed stable training with convergence.

## Environment

- GPU: single GPU (local)
- train_shards: 1
- train_seq_len: 1024
- grad_accum_steps: 8

## Training setup

- model_params: 17,059,912
- iterations: 20000 (stopped early at 5000)
- warmup_steps: 20

## Results

- step 0: val_bpb = 4.1077
- step 1000: val_bpb = 1.3944
- step 2000: val_bpb = 1.3651
- step 2800: val_bpb = 1.3548
- step 4200: val_bpb = 1.3529 ← best
- step 5000: val_bpb = 1.3576

## Key observation

- Validation BPB improves rapidly in early steps
- Converges around ~1.35
- After ~4200 steps, performance stops improving and fluctuates

## Conclusion

Baseline training successfully reproduced on local hardware.  
Best observed performance: **val_bpb = 1.3529 @ step 4200**

This run will be used as the reference baseline for further experiments.
