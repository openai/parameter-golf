# Ablation 3: Bifurcated vs Log-Uniform A_log Initialization

## Hypothesis

The log-uniform A_log init (`linspace(-4.5, 0.5, nheads) + noise`) introduced in iter-005.5 is causing the BPB regression from 1.600 (iter-003.5) to 1.98 (iter-005.5).

Log-uniform spreads heads evenly across all timescales, but this dilutes specialization. Most heads end up at intermediate decay rates where they are neither good at local pattern matching (fast decay) nor long-range retrieval (slow decay). The optimizer must then move all heads to useful positions, wasting training budget on "discovery".

The bifurcated init from iter-003.5 creates strong prior specialization:
- **25% induction heads** (A_log ~ -4.0, A ~ -0.018): near-infinite memory, handle copy/retrieval tasks
- **75% local heads** (A_log in 0.3-0.6, A in [-1.82, -1.35]): fast decay, handle bigram/trigram/clause patterns

This matches the empirical distribution of useful timescales in language: most signal is local, with a minority of long-range dependencies.

## What Changed

Single change in `SSDMixer.__init__`:

**iter-005.5 (log-uniform):**
```python
A_log_init = torch.linspace(-4.5, 0.5, self.nheads)
A_log_init = A_log_init + torch.randn(self.nheads) * 0.1
```

**This ablation (bifurcated, from iter-003.1):**
```python
A_log_init = torch.rand(self.nheads) * 0.3 + 0.3        # 75% local heads
num_induction = self.nheads // 4
A_log_init[:num_induction] = torch.rand(num_induction) * 0.1 - 4.0  # 25% induction
```

No other changes. Same architecture, same hyperparameters, same optimizer config.

## Expected Outcome

If the A_log init is the primary regression driver, we should see val_bpb drop significantly (toward 1.6-1.7 range). If the regression persists, it rules out A_log init as the cause and points to other iter-005.5 changes (compilation, vertical state carry, etc.).

## Run Command

```bash
# 1xH100 smoke test (10 min)
RUN_ID=ablation_3_bifurcated_alog \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Baseline Comparison

| Run | A_log Init | val_bpb | Notes |
|-----|-----------|---------|-------|
| iter-003.5 | Bifurcated (25/75) | 1.600 | Best result to date |
| iter-005.5 | Log-uniform linspace | 1.98 | 2x throughput but worse BPB |
| ablation-3 | Bifurcated (25/75) on iter-005.5 code | ??? | This test |
