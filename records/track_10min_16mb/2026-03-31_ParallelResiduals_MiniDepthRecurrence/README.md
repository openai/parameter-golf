# Record: Parallel Residuals + Mini Depth Recurrence

**val_bpb: 1.1063** (3-seed mean, std 0.0017) | **1.8679 nats** | **~15.94 MB** | 8×H100 SXM, 600s | No TTT

I started this submission from [PR #1179](https://github.com/openai/parameter-golf/pull/1179), which gave me the base training stack I wanted to iterate on here. On top of that, I ported over the mixed-quantization and autoregressive GPTQ path from [PR #1105](https://github.com/openai/parameter-golf/pull/1105). That was partly a modeling choice and partly a practical one: AR self-generated GPTQ calibration was already a known acceptable path for this challenge, and it let me avoid having the quantization step depend on last-minute training-data access in a way that makes the 10-minute budget awkward to manage.

## Results (8×H100 80GB SXM, 600s, no TTT)

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 1337 | 6,242 | 96.1 | 1.1232 | **1.1066** | 1.8684 | 15,942,395 |
| 42 | 6,248 | 96.0 | 1.1235 | **1.1077** | 1.8704 | 15,919,617 |
| 2024 | 6,240 | 96.2 | 1.1216 | **1.1044** | 1.8648 | 15,946,657 |
| **Mean** | **6,243** | **96.1** | **1.1228** | **1.1063** | **1.8679** | **15,936,223** |

Comparison baseline [PR #1179](https://github.com/openai/parameter-golf/pull/1179): **1.11053346 BPB** (**1.87508426 nats**).
This run's exact 3-seed mean: **1.10625353 BPB** (**1.86785780 nats**).
Delta vs PR #1179: **-0.00722646 nats** (**-0.00427993 BPB**).

Current merged SOTA ([2026-03-25 AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md)): **1.11473509 BPB** (**1.88217853 nats**).
Delta vs current merged SOTA: **-0.01432073 nats** (**-0.00848156 BPB**).

## Parallel residuals

I took this idea from my modded-nanogpt record in [KellerJordan/modded-nanogpt PR #230](https://github.com/KellerJordan/modded-nanogpt/pull/230) and adapted it to this codebase.

Chronologically, this change actually came last. I am putting it first here because it ended up being the single biggest gain on top of the base + mini-depth-recurrence stack: relative to the under-budget mini-DR baseline (`1.8705` val loss / `1.1078` BPB in sliding-window eval), it improved things by roughly another `0.0037` nats and `0.0022` BPB, landing around `1.8668` / `1.1056`. But this is still a one-sample observation, so I do not want to overstate the precision of that delta.

Starting from layer 7, attention and MLP read from different residual lanes, and each sublayer learns how strongly to write back into both lanes.

One interesting pattern is that the learned routing is quite asymmetric, which is also what I saw in the modded-nanogpt run: MLP barely writes back into attention's residual stream, especially in the deeper partitioned layers.

| Virtual layer | Physical layer | `attn_to_attn` | `attn_to_mlp` | `mlp_to_attn` | `mlp_to_mlp` |
|---|---:|---:|---:|---:|---:|
| 9 | 7 | 1.3030 | 0.8484 | 0.3851 | 1.3043 |
| 10 | 8 | 2.0972 | 0.8114 | 0.0557 | 1.7884 |
| 11 | 9 | 0.4523 | 0.9251 | 0.0098 | 0.2692 |
| 12 | 10 | 1.0153 | -0.0160 | 0.0844 | 0.0844 |

Despite that pattern, I also tried the followup optimization from [modded-nanogpt PR #241](https://github.com/KellerJordan/modded-nanogpt/pull/241), where MLP simply does not write to the attention lane at all in order to get a speedup. In this repo that brought a slight regression, so I kept the original parallel-residual formulation instead.

## Mini Depth Recurrence

Note: Most of the recurrence sweeps under this section were run on an older baseline, and I later transferred the final recipe over to the newer baseline used for this submission.

After some early failed attempts at full recurrence, I backed off to a much smaller version of the idea: instead of recurring the whole stack, I only repeated a couple of middle layers. I had already convinced myself from over-budget probes that extra depth was real, so the question became how much of that gain I could recover with minimal weight sharing.

The main sweeps were simple but informative. Repeating one layer helped, repeating two consecutive layers helped more, and repeating three was already losing to the step-time penalty. I also swept the position of the repeated pair and found a clear sweet spot at layers `4,5`, right around the U-Net hinge point. So the useful regime here was not “add recurrence everywhere”, it was “reuse a very small part of the middle of the stack.”

The next improvement was to turn recurrence on only mid training. Since repeated layers slow every step down, I trained the cheaper non-recurrent model first and only activated recurrence later. In the earlier sweep, always-on recurrence reached about `1.1163` BPB post-TTT, while delayed recurrence improved that to about `1.1153`, with `RECUR_START_STEP=3000` working well.

Finally, because mixed precision left me some parameter budget headroom, I found that the best place to spend it was untying the repeated MLPs while leaving the rest of the recurrent block shared. That gave another small but real improvement. Roughly speaking, mini depth recurrence was worth about `0.003-0.004` nats and `0.002-0.003` BPB over the best under-budget non-recurrent depth probe I had at the time.

## Reproducibility

The main training runs for this submission used the following command:

```bash
SEED=$SEED POST_GPTQ_EVAL_ONLY=0 BIGRAM_DIM=112 MIXED_QUANT=1 N_INT6_LAYERS=32 NUM_LAYERS=11 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 REPEAT_UNTIE_MLP=full REPEAT_UNTIE_MLP_LAYERS=4,5 DISABLE_LAYER0_ATTN=1 PARALLEL_RESIDUAL=1 PARALLEL_START_LAYER=7 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`brotli` also needs to be installed for the final artifact path. It is included in the copied [requirements.txt](/root/parameter-golf/records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/requirements.txt).
