# Non-Record Submission: NorMuon Baseline (1xH100)

This is a non-record submission which replaces the Muon optimizer with [NorMuon (Neuron-wise Normalized Muon)](https://arxiv.org/pdf/2510.05491).

Uses a modified implementation based on the original [NorMuon implementation](https://github.com/zichongli5/NorMuon/blob/main/normuon.py).

## Motivation

Adam optimizes parameters individually without taking advantage of the structure of 2D weight matrices.  Empirically, updates produced by Adam have a large condition number (max singular value / min singular value). The matrix almost becomes low-rank, with only a few directions (with large singular values) dominating neuron updates.

Muon tackles this by projecting the gradient update matrix to the nearest orthogonal matrix so that all the singular values become 1. This amplifies rare directions that Adam would neglect resulting in a condition number of 1.

In practice, computing the exact orthogonalization is expensive, so Muon uses Newton-Schulz to approximate the orthogonal matrix. Therefore, the singular values are not exactly 1.
Empirically we observe that this approximation leads to highly non-uniform per-neuron norms, which would have otherwise had an update norm of 1 if the orthogonalization had been exact (for wide matrices with rows <= cols, see NorMuon paper).

NorMuon addresses this by normalizing per-neuron update magnitudes.
It maintains a second-order momentum buffer which is an EMA of the mean squared updates per neuron (per row).
Each neuron's update is then divided by the square root of this second-order momentum term.
After row-normalization the resulting update has much larger norm, so this is then normalized to match the original norm.

NorMuon has been shown to have higher training efficiency than Muon and is also used in the nanogpt speedrun.

## NorMuon Implementation

We add the following NorMuon update after the Newton-Schulz, but before the Muon scale correction:

```python
buf2 = state["second_momentum_buffer"]
vnorm = g.norm()
v_mean = torch.mean(g * g, dim=-1, keepdim=True)
buf2.lerp_(v_mean.float(), 1 - beta2)
step_size = buf2.clamp_min(1e-10).rsqrt()
g.mul_(step_size)
vnorm_new = g.norm()
g.mul_(vnorm / (vnorm_new.clamp_min(1e-10)))
```

We use the default `NORMUON_BETA2=0.95` as in the original NorMuon implementation, but we have not yet swept this hyperparameter.

### Changes to [original NorMuon implementation](https://github.com/zichongli5/NorMuon/blob/main/normuon.py).

- We need `v_mean.float()` as buf2 is float32 but g is bfloat16. We convert to float32 after the `g * g` for speed.
- We use `clamp_min(1e-10)` instead of `add_(1e-10)`
- We use `.rsqrt()` instead of `1 / sqrt`

## Configuration

Uses the default baseline configuration, with an added NORMUON_BETA2=0.95 hyperparameter.

## Results

Baseline (1xH100)
mean val_loss: 2.2777
mean val_bpb: 1.3490

NorMuon (1xH100)
mean val_loss: 2.2724 (-0.0053)
mean val_bpb: 1.3458 (-0.003)

Still need to tune the hyperparameters for NorMuon and do a full
training run with 8xH100.

## Hardware Note

All experiments ran on a single H100 NVL with a 10-minute wallclock cap.
The next step is to validate this on 8xH100.

## Included Files

- `train_gpt.py`
- `results.tsv`
- `submission.json`
- `baseline_logs` - Running the baseline on 1xH100
- `normuon_logs` - Running the NorMuon baseline on 1xH100
