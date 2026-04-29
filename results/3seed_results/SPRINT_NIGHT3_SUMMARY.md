# Night 3 Sprint Summary (April 28-29, 2026)

## Best results (single seed 314)
| Run | Config | Pre-quant | Post-quant | Post-TTT |
|---|---|---|---|---|
| Baseline (PR1797 port) | dexhunter port | 1.06565 | 1.07460 | 1.06181 |
| S5 | K+O ablation + EMA 0.9975 | 1.06625 | 1.07437 | 1.06094 |
| S13 | K+O + #1855 TTT bundle (interfered) | 1.06708 | 1.07510 | 1.06177 |
| S14 | #1855 TTT bundle, MLP_LORA on, NO BOS fix | 1.06456 | 1.07354 | 1.06067 |
| S15 | S14 + BOS fix | TBD | TBD | TBD |

## Key findings
- K+O ablation (TTT_MLP_LORA=0) helps on baseline (-0.00087 in S5) but does NOT compose with #1855 TTT bundle (S13 worse than S14)
- #1855 TTT bundle (rank 80, BETA2 0.99, WD 0.5, prefix 2500) transfers in our port (S14 = 1.06067)
- S14 single-seed result already beats #1855's reported 1.06108 3-seed mean
- BOS SmearGate fix applied at lines 1414-1415 and 1512-1513 (both _forward_hidden and forward_ttt)
- DocStartSequenceLoader class added but not yet activated (gated by GPTQ_CALIBRATION_MODE=doc_start env var)

## Tomorrow's queue (ordered by S15 outcome)
1. If S15 ≤1.0588 single seed: validate seeds 42 + 1234 immediately
2. If S15 1.0589-1.0602: add doc-start GPTQ calibration (GPTQ_CALIBRATION_MODE=doc_start), then validate
3. If S15 1.0603-1.0611: run full #1855 train-side hparams (BETA2=0.99, WARMDOWN_FRAC=0.85, SPARSE_ATTN_GATE_SCALE=0.5, MLP_CLIP_SIGMAS=11.5, EMBED_CLIP_SIGMAS=14.0)
4. If S15 ≥1.0612: investigate port mismatch or ship floor

## Win bar
- If #1855 accepted (3-seed 1.06108): need ≤1.05914 3-seed mean = ~1.0584 single seed for confidence
- If #1855 rejected (lrzip violation): bar stays at dexhunter 1.06157 = need ≤1.05963 3-seed = ~1.0592 single seed

## Branch / commit state
- Repo: github.com/TanishGudise/parameter-golf
- Branch: sp8192-rebase  
- Top commit: 47f8b1d (BOS SmearGate fix + DocStartSequenceLoader)
- File: records/track_10min_16mb/2026-04-24_PR1797Base_QKGainSched_OptRot_AdamHD_LaCT/train_gpt_human.py (3799 lines)

## Pod recovery if migration fails
1. New RunPod 8xH100 SXM
2. ulimit -n 65535
3. git clone https://github.com/TanishGudise/parameter-golf.git
4. cd parameter-golf && git checkout sp8192-rebase
5. pip install -r requirements.txt (or whatever the install script is)
6. Re-download shards from HF: TanishGudise/param-golf-shards (1499 train shards, 29GB)
7. Token to use: [revoke and recreate post-sprint]
