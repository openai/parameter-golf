# Record: Improved Parallel Residuals

**val_bpb: 1.07531639** (3-seed mean, std 0.0006) | **2.77765390 nats** | **~15.96 MB** | 8xH100 SXM, 600s | Legal TTT

This submission starts from [PR #1523](https://github.com/openai/parameter-golf/pull/1523). Most of the newer submissions moved away from my fuller parallel-residual formulation and settled on a simpler GPT-J-style split-lane decoder. This version keeps the strong parts of that newer baseline and reintroduces the useful parts of my parallel residual implementation.

The key architectural change relative to PR #1523 is in the decoder after the split point. Attention and MLP read from different lanes, but neither sublayer writes back immediately. Instead, both outputs are accumulated into the two lanes together at the end of the block:

```python
next_lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
next_lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
```

That keeps the GPT-J-style parallel-in-time update, while restoring the richer learned routing between the attention and MLP lanes. The other important part is that decoder U-Net skips are still written only into `lane0`, which preserves the cheaper and more stable skip path from the newer baseline. Attention reads the mixed `lane0/x0` path, while MLP reads raw `lane1`. Final output uses the mean of the two lanes.

In practice, that is pretty much the only modeling change here versus PR #1523, together with moving `PARALLEL_RESIDUAL_START` from the baseline's `7` to `8`. I ablated that start-layer change separately on top of the plain PR #1523 baseline, without my fuller parallel residual routing changes, and it gave a mild regression on its own. The other notable requirement is that I needed the `cutlass_evt_fusion` path to recover the full throughput. PR #1523's logged runs were run with that path available, but it was not included in the submission folder itself. Without it, the wallclock cap gives up too many steps and the gain disappears.

## Results (8xH100 80GB SXM, 600s)

| Seed | Steps | ms/step | Post-EMA BPB | Legal TTT BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|----------------|-----------------|----------|
| 1337 | 4,698 | 125.00 | 1.0827 | **1.0746** | 2.7758 | 15,956,086 |
| 2024 | 4,746 | 123.72 | 1.0836 | **1.0760** | 2.7794 | 15,959,760 |
| 42 | 4,736 | 123.97 | 1.0832 | **1.0754** | 2.7778 | 15,954,783 |
| **Mean** | **4726.67** | **124.23** | **1.0832** | **1.07531639** | **2.77765390** | **15956876** |

## Reproducibility

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
for SEED in 1337 2024 42; do
    SEED=$SEED TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.01 MUON_MOMENTUM=0.97 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

The `cutlass_evt_fusion/` directory should live alongside `train_gpt.py` in the directory you run from.

## CUTLASS EVT Build

I include the prebuilt `.so` in `cutlass_evt_fusion/` only as a convenience for matching environments. If needed for verification, it can be rebuilt from source with:

```bash
git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
cd /opt/cutlass
git checkout 08185b9c3e90510ee2b656662ed0d53b06d28157
cd -
pip install --no-build-isolation ./cutlass_evt_fusion
```

## Artifact Size Note

The reported artifact sizes above follow the challenge's usual accounting of `train_gpt.py` code bytes plus compressed model bytes. If I also count the custom `cutlass_evt_fusion` source files that are shipped here for reproducibility, specifically `csrc/gemm_act_grad.cu`, `csrc/torch_binding.cpp`, `__init__.py`, and `setup.py`, that adds 8,579 bytes. Under that stricter accounting, the mean artifact size would be 15,965,455 bytes instead of 15,956,876.
