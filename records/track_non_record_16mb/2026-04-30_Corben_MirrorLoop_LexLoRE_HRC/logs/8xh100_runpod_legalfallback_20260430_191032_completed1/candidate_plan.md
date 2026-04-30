# 16MB Vocab-MoE Matrix

Generated: 2026-04-30 19:10:32
Iterations: 1000000
Validation tokens: 131072

| Candidate | Route | Vocab MoE | Notes |
|---|---|---|---|
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer8t16_vocabmoe_qk55` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Direct legalizer for the over-cap e832/LQER8 row: keep the 24k/rank rhythm and trim only the factored embedding rank. |
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Extra-safe size variant. Tests whether a smaller LQER payload closes the export/artifact risk without a large quality hit. |
| `final8x_legal_262k_r2_d704e768_w2500_wd02_lqer8t16_vocabmoe_qk55` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | 32k/rank middle point with e768 legalization; tests whether a larger global batch improves export BPB on 8x. |
| `final8x_legal_196k_r3_d704e768_w3000_wd02_lqer8t16_lidx_vocabmoe_qk55` | `unique=8 depth=21 start=3 repeats=3` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Single deeper recurrent sanity row, legalized through e768 and loop-indexed so repeated passes can specialize. |
