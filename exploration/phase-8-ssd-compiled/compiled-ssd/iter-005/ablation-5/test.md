# Ablation 5: Trapezoidal vs Euler Discretization

## Hypothesis

The trapezoidal discretization introduced in iter-005.x is a regression factor compared to iter-003.5 (val_bpb=1.600). iter-005.5 gets val_bpb=1.98 despite 2x throughput.

Trapezoidal discretization computes:
```python
lambda_t = sigmoid(lambda_proj(x))    # learned blend gate
gamma_t = lambda_t * dt
beta_t  = (1 - lambda_t) * dt * exp(A_disc)
X = x_heads * gamma_t + pad(x_heads[:-1] * beta_t[1:])  # blend curr + prev
```

Standard Mamba-2 Euler discretization computes:
```python
X = x_heads * dt
```

Potential problems with trapezoidal:
1. **Extra parameters**: lambda_proj is QATLinear(dim, nheads) = 32,768 params (dim=1024, nheads=32). Small in absolute terms but it's a full forward pass per token.
2. **Convergence noise**: The learned blend gate starts at sigmoid(0) = 0.5, meaning half the signal comes from a shifted-by-1 copy of x. This is a random initialization that may confuse early training.
3. **Gradient complexity**: beta_t includes exp(A_disc) which couples the discretization to the decay parameter, creating a harder optimization landscape.
4. **Causality subtlety**: The F.pad shifts x_heads by 1 position. While technically causal, it mixes adjacent tokens in the input space before the SSM scan even begins, which may interfere with the SSM's own sequential mixing.

Note: iter-003.5 also had trapezoidal and got 1.600, so this alone does not explain the full regression. But iter-005.5 has many simultaneous changes (torch.compile, different hyperparams, etc.) and trapezoidal may interact badly with those changes. This ablation isolates the discretization method.

## What Changed

- Removed `self.lambda_proj = QATLinear(dim, self.nheads)` from SSDMixer.__init__
- Removed `lambda_proj.weight` from QAT_WEIGHT_SUFFIXES
- Replaced trapezoidal block (8 lines) with standard Euler: `X = x_heads * dt.unsqueeze(-1)`
- Net effect: -32,768 parameters, -1 QATLinear forward/backward per token, simpler gradient flow

## Expected Outcome

If trapezoidal is a regression factor: val_bpb improves (lower is better) on a matched smoke test.
If trapezoidal is neutral or beneficial: val_bpb stays the same or worsens, confirming the regression comes from elsewhere.

## How to Run

```bash
# 1xH100 smoke test (5 min)
RUN_ID=ablation_5_euler_disc \
MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Comparison Points

| Run | Discretization | lambda_proj | val_bpb | Notes |
|-----|---------------|-------------|---------|-------|
| iter-003.5 | Trapezoidal | Yes | 1.600 | Best result (1xH100, 10min) |
| iter-005.5 | Trapezoidal | Yes | 1.98 | Current code (1xH100, 10min) |
| ablation-5 | Euler | No | ??? | This test |
