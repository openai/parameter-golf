# Experiment 009: Full 8xH100 Production Run — Baseline + Softcap15 + Eps1e-10

## Status: SETTING UP (8xH100 production instance provisioning)

## Hypothesis
Our best config (softcap=15, eps=1e-10) showed marginal improvement at 2K steps but the
softcap change dramatically helps early training dynamics (loss 5.42 vs 5.99 at step 10).
At 13K+ steps on 8xH100 (the actual competition scale), this could compound to a meaningful
BPB improvement over the baseline's 1.2244.

This is the REAL TEST — everything so far was 2K-step screening on 1 GPU.

**Prediction**: val_bpb < 1.220 (beating baseline 1.2244 by >= 0.004).

## Configuration
- **Architecture**: SAME as baseline (9 blocks, dim=512, 8 heads, 4 KV heads)
- **Changes**: LOGIT_SOFTCAP=15, ADAM_EPS=1e-10
- **Script**: train_gpt.py (with wandb, NO MTP)
- **Infrastructure**: 8xH100 production mode, torchrun --nproc_per_node=8
- **Training**: 20K iters cap, 600s wallclock cap, 524K tokens/step
- **Data**: Full 80 training shards
- **wandb run**: exp009_8xh100_softcap15

## Cost
~$19.92/hr for 8xH100 production. 10-min run + setup = ~$5-7 total.
