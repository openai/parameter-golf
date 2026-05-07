# PP12: Bayesian Posterior Packets + Selective Gating
val_bpb: 1.1261 | ~15.99 MB | 8×H100 SXM

## Results (8×H100 80GB SXM)

| Metric | Value |
|--------|-------|
| Steps | 5,645 |
| Step avg | 107ms |
| Pre-TTT bpb | 1.1280 |
| Post-TTT bpb | 1.1267 |
| Post-TTT+Packet bpb | **1.1261** |
| Packet gain | -0.0006 |
| TTT gain | -0.0013 |
| TTT time | 421s |
| Packet eval time | 110s |
| Artifact | 15,950,000 |

## Key Innovation: Bayesian Posterior Packets

Distill bigram posteriors from training data into a compact packet store. At eval, use conjugate Bayesian updating to combine training priors with online counts from scored tokens:

```
p(y|h) = (online_count(y|h) + tau * prior(y|h)) / (online_total(h) + tau)
```

Selective gating ensures packets only intervene when the packet posterior is more confident than the neural model:

```python
pp_top1 = pp.max(dim=-1).values
nn_top1 = p_nn.max(dim=-1).values
pp_better = (pp_top1 > nn_top1 + 0.05).float()
has_data = ((tau + online_tot) > 20.0).float()
alpha = 0.2 * pp_better * has_data * pp_top1
```

This prevents the quality degradation seen with naive probability mixing (which consistently hurt BPB in our experiments by +0.006 to +0.22).

## Packet Store

| Component | Details |
|-----------|---------|
| Source | 5M training tokens (12s build time) |
| Content | Smoothed bigram posteriors + confidence tau |
| Online update | Scatter-add bigram counts from scored chunks |
| Mixing | Selective gate: packet top-1 must exceed neural top-1 by 0.05 |
| Size | ~24 KB compressed (negligible artifact cost) |

## Observation: TTT Drift

Early TTT chunks achieve **1.109 BPB** (below current SOTA) but drift to 1.126 by the end. Periodic weight reset every 200 chunks is implemented but not yet validated on 8xH100.

## Base Stack

PR #549 (LeakyReLU(0.5)^2 + Legal Score-First TTT + Parallel Muon + Parameter Banking + GPTQ-lite int6 + EMA + XSA4 + Partial RoPE + SmearGate + BigramHash + Value Embedding).

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| BigramHash | 3072 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |
| TTT | SGD(0.002), momentum 0.9, 3 epochs, all blocks, grad clip 1.0 |

## Run Command

```bash
RUN_ID=pp12 TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 TTT_EPOCHS=3 TTT_LR=0.002 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

Base model + TTT: PR #549 by @abaybektursun (LeakyReLU^2 + Legal TTT + Parallel Muon on PR #414 stack)
Posterior packet system: this submission
