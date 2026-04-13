# Record: Improved Parallel Residuals

**val_bpb: 1.07578747** (3-seed mean, std 0.0007) | **2.77887078 nats** | **~15.98 MB** | 8xH100 SXM, 600s | Legal TTT

This submission starts from [PR #1523](https://github.com/openai/parameter-golf/pull/1523). Most of the newer submissions moved away from my fuller parallel-residual formulation and settled on a simpler GPT-J-style split-lane decoder. This version keeps the strong parts of that newer baseline and reintroduces the useful parts of my parallel residual implementation.

The key architectural change relative to PR #1523 is in the decoder after the split point. Attention and MLP read from different lanes, but neither sublayer writes back immediately. Instead, both outputs are accumulated into the two lanes together at the end of the block:

```python
next_lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
next_lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
```

That keeps the GPT-J-style parallel-in-time update, while restoring the richer learned routing between the attention and MLP lanes. The other important part is that decoder U-Net skips are still written only into `lane0`, which preserves the cheaper and more stable skip path from the newer baseline. Attention reads the mixed `lane0/x0` path, while MLP reads raw `lane1`. Final output uses the mean of the two lanes.

In practice, that is pretty much the only modeling change here versus PR #1523, together with moving `PARALLEL_RESIDUAL_START` from the baseline's `7` to `8`. I ablated that start-layer change separately on top of the plain PR #1523 baseline, without my fuller parallel residual routing changes, and it gave a mild regression on its own. The other notable requirement is that I needed the CUTLASS EVT path to recover the full throughput. In this iteration the CUDA/C++ source is inlined into the training script itself and built against a standard `/opt/cutlass` checkout rather than shipping a separate prebuilt `.so`.

## Results (8xH100 80GB SXM, 600s)

| Seed | Steps | ms/step | Post-EMA BPB | Legal TTT BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|----------------|-----------------|----------|
| 1337 | 4,655 | 126.13 | 1.0830 | **1.0751** | 2.7770 | 15,983,095 |
| 2024 | 4,689 | 125.20 | 1.0843 | **1.0765** | 2.7806 | 15,987,382 |
| 42 | 4,696 | 125.04 | 1.0837 | **1.0759** | 2.7790 | 15,982,563 |
| **Mean** | **4680.00** | **125.46** | **1.0837** | **1.07578747** | **2.77887078** | **15984347** |

## Reproducibility

```bash
pip install brotli sentencepiece
git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
cd /opt/cutlass
git checkout 08185b9c3e90510ee2b656662ed0d53b06d28157
cd -
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
for SEED in 1337 2024 42; do
    SEED=$SEED TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.01 MUON_MOMENTUM=0.97 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```
