# Multi-Token Prediction (MTP) for Parameter Golf

## Goal

Add multi-token prediction auxiliary heads to improve representation quality during training without increasing artifact size. Target: reduce val_bpb by 0.005-0.01 from our current 1.0783.

## Design Decisions

- **Approach:** DeepSeek-V3 style — 2 aux heads predicting +2 and +3 tokens from final hidden states
- **Head architecture:** `Linear(512, 512)` zero-init transform → shared `tok_emb.weight` projection → logit softcap → CE loss
- **Loss weight:** λ=0.3 total (0.15 per head), annealed linearly to 0 over the last 30% of training
- **Artifact impact:** Zero — aux heads are training-only, dropped before quantization/serialization

## Architecture

### Aux Head Module

```python
class MTPHead(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.transform = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.zeros_(self.transform.weight)  # identity-like at init

    def forward(self, hidden: Tensor, tok_emb_weight: Tensor, softcap: float) -> Tensor:
        h = self.transform(hidden) + hidden  # residual connection
        logits = F.linear(h, tok_emb_weight)
        return softcap * torch.tanh(logits / softcap)
```

Key details:
- Zero-init weight + residual connection means at init this is identical to the main head
- Shares `tok_emb.weight` for vocabulary projection (no extra vocab params)
- Same logit softcap as main head

### Integration into GPT.forward()

```python
def forward(self, input_ids, target_ids, mtp_lambda=0.0):
    # ... existing trunk unchanged ...
    x = self.final_norm(x)

    # Main head (unchanged)
    x_flat = x.reshape(-1, x.size(-1))
    targets = target_ids.reshape(-1)
    logits = self._project_logits(x_flat)
    main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

    if mtp_lambda > 0.0 and self.training:
        aux_loss = 0.0
        for k, head in enumerate(self.mtp_heads):
            shift = k + 2  # predict +2, +3
            # Shift: use hidden[:-shift] to predict target[shift:]
            h = x[:, :-shift, :].reshape(-1, x.size(-1))
            t = target_ids[:, shift:].reshape(-1)
            aux_logits = head(h, self.tok_emb.weight, self.logit_softcap)
            aux_loss += F.cross_entropy(aux_logits.float(), t, reduction="mean")
        return main_loss + mtp_lambda * aux_loss / len(self.mtp_heads)

    return main_loss
```

### Lambda Schedule

```python
mtp_lambda_start = 0.3
mtp_anneal_start = 0.7  # start annealing at 70% of training
# At each step:
progress = step / total_steps
if progress >= mtp_anneal_start:
    mtp_lambda = mtp_lambda_start * (1.0 - (progress - mtp_anneal_start) / (1.0 - mtp_anneal_start))
else:
    mtp_lambda = mtp_lambda_start
```

### Artifact Serialization

MTP heads are excluded from the saved model:
```python
state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("mtp_")}
```

No changes needed to GPTQ, quantization, or eval code — they never see the aux heads.

## Parameter Budget

| Component | Params | Artifact Impact |
|-----------|--------|-----------------|
| mtp_head_2 transform | 262,144 | None (training-only) |
| mtp_head_3 transform | 262,144 | None (training-only) |
| **Total** | **524,288** | **0 bytes** |

## Timing Budget

| Component | Estimated Cost | Impact |
|-----------|---------------|--------|
| 2x Linear(512,512) forward | ~1ms | |
| 2x tok_emb projection | ~1ms | |
| 2x CE loss + backward | ~2ms | |
| **Total per step** | **~4ms** | ~90ms → ~94ms/step |
| **Steps lost in 600s** | ~280 fewer | ~6740 → ~6380 steps |

The representation improvement must compensate for ~360 lost steps. DeepSeek-V3 showed MTP gains are substantial even at smaller scales, so this is a favorable bet.

## Env Vars

| Var | Default | Description |
|-----|---------|-------------|
| `MTP_ENABLED` | `0` | Enable multi-token prediction aux heads |
| `MTP_LAMBDA` | `0.3` | Initial aux loss weight |
| `MTP_ANNEAL_START` | `0.7` | Fraction of training where lambda annealing begins |
| `MTP_NUM_HEADS` | `2` | Number of aux heads (predict +2 through +N+1) |

## Optimizer Groups

MTP head parameters go into the AdamW group (not Muon) since they're small linear layers:
- lr: same as `scalar_lr` or `head_lr`
- wd: same as `adam_wd`

## Success Criteria

- val_bpb improves by >=0.003 over baseline (same seed, same step count adjusted for overhead)
- No artifact size increase
- Training completes within 600s wallclock

## Risks

1. **Overhead too high:** If per-step cost exceeds ~5ms, the lost steps may not be worth it. Mitigation: MTP_ENABLED=0 fallback.
2. **Destabilizes training:** Aux gradients could interfere with main loss convergence. Mitigation: zero-init + residual means aux heads start as identity; λ annealing removes their influence before final convergence.
3. **No effect at small scale:** MTP gains were demonstrated at 100B+ scale. At 36M params the benefit may be smaller. This is the core research risk.
