# LongContext 4096 + Int4 Bank QAT 16L + Full SOTA Stack (Risky)

**val_bpb: TBD** (3-seed mean, post int4+lzma, sliding window stride=64 + TTT)

## Summary

**Risky experiment**: True full-model int4 QAT. Based on `LongContext4096_Int4_16L_FullSOTA` (fixed),
with one critical addition: bank weights also receive int4 STE fake-quant during training.

### What's new vs prior int4 experiments

All prior int4 scripts only QAT'd `CastedLinear` weights (bigram.proj, ve.proj, lm_head) — a tiny
fraction of parameters. The bulk of the model — `qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`
(~95% of all weights) — was exported to int4 with zero QAT preparation.

This script adds `_fake_quant_int4_bank()`: a helper that applies int4 STE fake-quant to every
2D bank slice during the training forward pass, using the same `CastedLinear._qat_enabled` flag.

```
_fake_quant_int4_bank(self.qo_bank[i], self.training)   # Q weights
_fake_quant_int4_bank(self.kv_bank[i], self.training)   # K weights
_fake_quant_int4_bank(self.kv_bank[n+i], self.training) # V weights
_fake_quant_int4_bank(self.qo_bank[n+i], self.training) # Out weights
_fake_quant_int4_bank(self.mlp_up_bank[i], self.training)
_fake_quant_int4_bank(self.mlp_down_bank[i], self.training)
```

## Risk

Int4 QAT on all parameters is aggressive. The STE gradient through a `clamp(-8, 7)` grid is zero
outside the range, which can stall learning for large-magnitude weights. The late-QAT trigger
(`late_qat_threshold=0.15`) limits exposure to the final ~15% of training to mitigate this.

## Run Command

```bash
SEED=1337 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42   TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2025 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

*(Pending H100 runs)*

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|-------------|--------------|-----------------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
