# Hybrid XSA-SSM — Non-Record 16 MB Track

**val_bpb: 1.255216** (post-quantization) | **1.2146 pre-quantization** | 14,915,938 bytes | Stopped at step 3723 (600s wallclock cap)

Built on PR #549. 8×H100 SXM | PyTorch 2.4.0-py3.11-cuda12.4.1

---

## SSM Hybrid Layer

The core novelty of this submission is replacing the middle transformer layer (layer 5, i.e. `num_layers // 2`) with a [Mamba](https://github.com/state-spaces/mamba) SSM block instead of a standard attention block:

```python
mamba = Mamba(d_model=512, d_state=64, d_conv=4, expand=2)
```

It is inserted as a residual contribution:

```python
if isinstance(block, Mamba):
    x_out = x + run_mamba(block, x.to(torch.bfloat16)).to(x.dtype)
```

Placing it at the midpoint of the U-Net-style encoder–decoder stack lets the SSM act as a bottleneck that compresses long-range sequence state before the decoder layers unpack it. The surrounding transformer layers handle local attention patterns as usual.

The Mamba block's internal parameters (`A_log`, `dt_bias`, `D`) are routed to the AdamW scalar optimizer rather than Muon, since they don't benefit from Newton-Schulz orthogonalization. The SSM block contributes ~1.84M of the model's ~25.95M total parameters.

`@torch._dynamo.disable` is applied to the Mamba forward call to avoid TorchDynamo tracing issues with the SSM's custom CUDA kernels.

---

## Results

| Metric | Value |
|---|---|
| Pre-quantization val_bpb | 1.2146 |
| Post-quantization val_bpb | 1.255216 |
| val_loss (nats) | 2.119375 |
| Steps completed | 3723 / 20000 |
| Wallclock | 600s |
| Artifact size | 14,915,938 bytes |