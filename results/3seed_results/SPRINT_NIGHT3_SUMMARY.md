# Night 3 Sprint Summary (April 28-29, 2026)

## Best results (single seed 314)
| Run | Config | Pre-quant | Post-quant | Post-TTT |
|---|---|---|---|---|
| Baseline (PR1797 port) | dexhunter port | 1.06565 | 1.07460 | 1.06181 |
| S5 | K+O ablation + EMA 0.9975 | 1.06625 | 1.07437 | 1.06094 |
| S13 | K+O + #1855 TTT bundle (interfered) | 1.06708 | 1.07510 | 1.06177 |
| S14 (BEST) | #1855 TTT bundle, MLP_LORA on, NO BOS fix | 1.06456 | 1.07354 | 1.06067 |
| S15 | S14 + BOS fix (REGRESSED) | 1.06570 | 1.07473 | 1.06196 |

## Key findings
- K+O ablation (TTT_MLP_LORA=0) helps on baseline (-0.00087 in S5) but does NOT compose with #1855 TTT bundle (S13 worse than S14)
- #1855 TTT bundle (rank 80, BETA2 0.99, WD 0.5, prefix 2500) transfers in our port (S14 = 1.06067)
- S14 single-seed result already beats #1855's reported 1.06108 3-seed mean
- BOS SmearGate fix REGRESSED our stack by 0.00129 (S14 -> S15). Likely #1855's other hparams compensate for the leak; without their full stack, removing the leak is net-negative.
- DocStartSequenceLoader class added but not yet activated (gated by GPTQ_CALIBRATION_MODE=doc_start env var)

## Best config (S14 - to validate or improve)
PHASED_TTT_PREFIX_DOCS=2500
TTT_LORA_RANK=80
TTT_BETA2=0.99
TTT_WEIGHT_DECAY=0.5
TTT_K_LORA=1, TTT_O_LORA=1, TTT_MLP_LORA=1
EMA_DECAY=0.9965
SEED=314 (single seed result: 1.06067)
NO BOS fix in code path (BOS fix is on disk in train_gpt_human.py but we should NOT use it without compensating hparams)

## Tomorrow's plan
1. First run: doc-start GPTQ calibration ON TOP OF S14 (no BOS fix in TTT path means we may need to revert that, see below)
2. Second run if 1 helps: LQER_TOP_K=4 isolated on top
3. Third run if needed: full #1855 train-side hparams (BETA2=0.99, WARMDOWN_FRAC=0.85, SPARSE_ATTN_GATE_SCALE=0.5, MLP_CLIP_SIGMAS=11.5, EMBED_CLIP_SIGMAS=14.0) WITH BOS fix (this is when BOS fix should help)

## Critical: BOS fix in code is currently always-on
The code patch always applies the BOS fix. If we want to test S14-style runs (no BOS fix), we need either:
(a) Revert the BOS fix in code, OR
(b) Add an env var gate around it (BOS_FIX_ENABLED=1/0)

Option (b) is cleaner for ablation. Tomorrow morning: add an env-var gate before any new runs.

## Win bar
- If #1855 accepted: bar = 1.05914 3-seed mean. Need ~1.0584 single seed.
- If #1855 rejected (lrzip violation): bar = 1.05963 3-seed mean. Need ~1.0592 single seed.
- S14 (1.06067) is 0.0015 above optimistic bar. Need to find ~0.0015-0.0023 with one lever.

## Branch / commit state
- Repo: github.com/TanishGudise/parameter-golf
- Branch: sp8192-rebase
- Top commits: 6289aa8 (summary), 0064565 (sweep logs), 47f8b1d (BOS fix + DocStartSequenceLoader)
- File: records/track_10min_16mb/2026-04-24_PR1797Base_QKGainSched_OptRot_AdamHD_LaCT/train_gpt_human.py (3799 lines)

## Pod recovery if migration fails
1. New RunPod 8xH100 SXM
2. ulimit -n 65535
3. git clone https://github.com/TanishGudise/parameter-golf.git
4. cd parameter-golf && git checkout sp8192-rebase
5. Install deps as needed
6. Re-download shards from HF: TanishGudise/param-golf-shards (1499 train shards, 29GB)
7. Token to use: revoke and recreate post-sprint
