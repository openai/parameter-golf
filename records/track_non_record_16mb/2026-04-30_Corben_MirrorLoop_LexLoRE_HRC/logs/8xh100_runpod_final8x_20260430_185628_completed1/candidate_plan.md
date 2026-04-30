# 16MB Vocab-MoE Matrix

Generated: 2026-04-30 18:56:28
Iterations: 1000000
Validation tokens: 131072

| Candidate | Route | Vocab MoE | Notes |
|---|---|---|---|
| `final8x_196k_r2_d704e832_w2200_wd02_lqer8t16_vocabmoe_qk55` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Best-known legal 1x package moved to 8x while preserving the winning 24k tokens/rank optimizer rhythm. |
| `final8x_196k_r2_d704e832_w2200_wd02_lqer9t18_vocabmoe_qk55` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Near-cap legalizer between the known-safe LQER8/T16 row and the slightly-over 1x LQER10/T20 row. |
| `final8x_262k_r2_d704e832_w2200_lqer10t20_vocabmoe_qk55` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | 32k tokens/rank row. This is the strongest legal middle point between local more-step training and the official 524k batch. |
| `final8x_524k_r2_d704e832_w3500_lqer10t20_vocabmoe_qk55` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Official-style 8x global batch. Tests whether the cluster's main gift is cleaner gradients and far more training tokens per update. |
| `final8x_196k_r3_d704e768_w3000_wd02_lqer8t16_lidx_vocabmoe_qk55` | `unique=8 depth=21 start=3 repeats=3` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Only deeper-loop probe in the first paid hour: r3 plus loop index, legalized through e768/LQER8 while preserving 24k tokens/rank. |
