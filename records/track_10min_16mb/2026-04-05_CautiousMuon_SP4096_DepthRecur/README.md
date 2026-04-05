# Cautious Muon + SP4096 + Depth Recurrence + Parallel Residuals

**val_bpb = 1.1604** (3-seed mean, std = 0.0033)

## Results

| Seed | val_bpb | val_loss | Artifact Size |
|------|---------|----------|---------------|
| 42   | 1.1568  | 2.6619   | 15,179,504 B  |
| 314  | 1.1611  | 2.6717   | 15,173,470 B  |
| 999  | 1.1634  | 2.6770   | 15,159,223 B  |
| **Mean** | **1.1604** | **2.6702** | **15,170,732 B** |

## Key Technique: Cautious Muon (arXiv:2411.16085)

The primary modification is applying the Cautious optimizer principle to the Muon optimizer. After Newton-Schulz orthogonalization and MuonEq-R row normalization, the update is masked to only apply where the orthogonalized direction agrees with the raw gradient sign:

```python
caution_mask = (g * raw_grad > 0).to(g.dtype)
g = g * caution_mask / caution_mask.mean().clamp_min(1e-3)
```

This filters out "stale" momentum directions that disagree with the current gradient, providing ~1.47x effective convergence per step with zero parameter overhead and no impact on artifact size.

## Full Architecture Stack

Built on PR #1334 (aryanbhosale) with:
- **SP4096 BPE tokenizer** (from PR #1218, @clarkkev)
- **Depth recurrence** layers 4,5 (13 virtual layers from 11 physical, activated at step 3000)
- **Parallel residuals** from layer 7 (separate attn/MLP lanes with learnable merge)
- **MuonEq-R** row normalization before Newton-Schulz (arXiv:2603.28254)
- **QK-Gain 5.0** per-head query-key scaling
- **EMA 0.997** weight averaging
- **Full GPTQ INT6** quantization with selective +-1 pruning
- **Brotli compression**

## Non-matrix parameters

Token embeddings, scalar parameters, and head use standard `torch.optim.AdamW`. Cautious masking is applied only inside Muon for matrix parameters.

## Compliance

- Track A fixed predictor -- no TTT, no SLOT, no eval-time adaptation
- All predictions are causal and normalized via softmax (F.cross_entropy)
- Artifact under 16MB limit (max 15,179,504 bytes)
- Training completes within 600s wallclock on 8xH100 SXM

## Reproduction

```bash
cd /workspace/parameter-golf
# Download SP4096 data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096
# Run
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1334 (@aryanbhosale) -- base architecture (SP4096, depth recurrence, parallel residuals, MuonEq-R)
- PR #1218 (@clarkkev) -- SP4096 tokenizer
- Liang et al. (arXiv:2411.16085) -- Cautious Optimizers
