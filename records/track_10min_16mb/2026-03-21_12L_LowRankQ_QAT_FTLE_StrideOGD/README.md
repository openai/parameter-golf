# 12L + Low-Rank Q + QAT + FTLE + Stride-OGD

**Status: In progress (1xH100 development)**

## Key Results (1xH100, 7900 steps)

- **Pre-quant val_bpb: 1.2035**
- Post-quant (FTLE int6 + sliding window, OGD): eval incomplete (OGD too slow)
- Projected post-quant (uniform int6 + sliding window): **~1.19 bpb**
- Projected post-quant (uniform int7 + sliding window, if fits): **~1.17-1.18 bpb**
- Artifact size: 15.5MB (FTLE int6) / 15.2MB (uniform int6)

## Techniques

1. **Low-Rank Q factorization (rank=128)**: Q projection factored as dim->128->dim, saving ~50% Q params per layer. Enables 12 layers within speed budget.
2. **12 transformer layers** (up from 10): ~20.9M params. Step time ~616ms on 1xH100 (est. 77ms on 8xH100).
3. **QAT with STE (int7)**: Fake quantization during training (activated at 10% of training). 6% step time overhead.
4. **FTLE-guided per-row precision**: Tested but **does not help** — uniform quantization is strictly better in both RMSE and compressed size. See EXPERIMENT_LOG.md for ablation.
5. **Stride-OGD at eval**: Online gradient descent on vocab bias during sliding window eval. Implemented but too slow (~30-60 min). Needs optimization.
6. **Sliding window eval (stride=64)**: Free ~0.03 bpb improvement.

## What Works

- 12L + Low-Rank Q architecture trains well, reaches 1.2035 pre-quant at 7900 steps
- QAT integrates cleanly with modest overhead
- Sliding window eval gives free bpb gain
- All inherited SOTA techniques (Muon WD, overtone init, phase-transition resid_mix) carry over

## What Doesn't Work

- **FTLE per-row precision**: Uniform int-N is strictly better than FTLE-guided mixed precision at matched average bit widths
- **Stride-OGD**: Gradient tracking through [batch, 1024, 1024] logits tensors is too slow for practical eval

## Next Steps

- Drop FTLE, switch to uniform quantization
- Try to fit uniform int7 (16.9MB — 0.9MB over, needs code/model size reduction)
- Fix OGD eval speed or drop it
- Full 8xH100 run for final submission

## Command

```bash
RUN_ID=full_12L_v4_7900 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
WARMDOWN_ITERS=2000 \
OGD_ENABLED=0 \
EVAL_STRIDE=64 \
QAT_BITS=7 \
QAT_START_FRAC=0.1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
