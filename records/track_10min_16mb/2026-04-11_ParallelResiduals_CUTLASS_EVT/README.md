# Record: Parallel Residuals + CUTLASS EVT Fusion

**val_bpb: 1.07438392** (3-seed mean, std 0.0003) | **2.77524526 nats** | **~15.96 MB** | 8xH100 SXM, 600s | Legal TTT

This submission starts from [PR #1523](https://github.com/openai/parameter-golf/pull/1523). Most of the newer submissions moved away from my fuller parallel-residual formulation and settled on a simpler GPT-J-style split-lane decoder. This version keeps the strong parts of that newer baseline and reintroduces the useful parts of my parallel residual implementation.

The key architectural change relative to PR #1523 is in the decoder after the split point. Attention and MLP read from different lanes, but neither sublayer writes back immediately. Instead, both outputs are accumulated into the two lanes together at the end of the block:

```python
next_lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
next_lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
```

That keeps the GPT-J-style parallel-in-time update, while restoring the richer learned routing between the attention and MLP lanes. The other important part is that decoder U-Net skips are still written only into `lane0`, which preserves the cheaper and more stable skip path from the newer baseline. Attention reads the mixed `lane0/x0` path, while MLP reads raw `lane1`. Final output uses the mean of the two lanes.

In practice, that is pretty much the only modeling change here versus PR #1523, together with moving `PARALLEL_RESIDUAL_START` from the baseline's `7` to `8`. I ablated that start-layer change separately on top of the plain PR #1523 baseline, without my fuller parallel residual routing changes, and it gave a mild regression on its own. The other notable requirement is that I needed the `cutlass_evt_fusion` path to recover the full throughput. PR #1523's logged runs were run with that path available, but it was not included in the submission folder itself. Without it, the wallclock cap gives up too many steps and the gain disappears.

## Results (8xH100 80GB SXM, 600s, legal TTT)

| Seed | Steps | ms/step | Post-EMA BPB | Legal TTT BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|----------------|-----------------|----------|
| 1337 | 4,685 | 125.53 | 1.0829 | **1.0748** | 2.7764 | 15,958,373 |
| 2024 | 4,734 | 124.25 | 1.0824 | **1.0743** | 2.7750 | 15,956,287 |
| 42 | 4,733 | 124.26 | 1.0821 | **1.0740** | 2.7743 | 15,959,005 |
| **Mean** | **4717.33** | **124.68** | **1.0825** | **1.07438392** | **2.77524526** | **15957888** |

## Reproducibility

```bash
pip install brotli sentencepiece
pip install ./cutlass_evt_fusion
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
for SEED in 1337 2024 42; do
    SEED=$SEED TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.01
  MUON_MOMENTUM=0.97 PARALLEL_RESIDUAL_START=8 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```
