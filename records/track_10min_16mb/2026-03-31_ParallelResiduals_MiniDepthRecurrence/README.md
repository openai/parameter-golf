I started this submission from [PR #1179](https://github.com/openai/parameter-golf/pull/1179), which gave me the base training stack I wanted to iterate on here. On top of that, I ported over the mixed-quantization and autoregressive GPTQ path from [PR #1105](https://github.com/openai/parameter-golf/pull/1105). That was partly a modeling choice and partly a practical one: AR self-generated GPTQ calibration was already a known acceptable path for this challenge, and it let me avoid having the quantization step depend on last-minute training-data access in a way that makes the 10-minute budget awkward to manage.

From there, I ended up pursuing two main ideas:

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

After some early unsuccessful attempts at full recurrence, I took a step back and asked a more basic question: if I had extra parameters to spend, should they go into width or depth? I ran matched over-budget probes in both directions and found that both had promise, with broadly comparable headline metrics.

| Probe | Change | Post-EMA val_bpb | Step avg | Steps in 1200s | Params | Size |
|---|---|---:|---:|---:|---:|---:|
| Width | `11L x 576` | `1.1277` | `~214ms` | `5,609` | `34.0M` | `19.5 MB` |
| Depth | `12L x 512` | `1.1307` | `~174ms` | `6,878` | `29.4M` | `17.3 MB` |

The results suggested that both width and depth were plausible directions. Since some of my earlier failed recurrence attempts had already explored the width side of the space, I decided to push on depth this time. That made the next question straightforward: how much of the benefit of a deeper model could I recover by reusing layers instead of paying for fully independent ones?
