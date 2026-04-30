# Record: SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + Legal TTT — val_bpb 1.0896 (3-seed mean)

**val_bpb = 1.0896** (3-seed mean, std 0.0008) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Sliding BPB | **TTT BPB** | TTT gain | Artifact |
|------|-------------|-------------|----------|----------|
| 42   | 1.0896      | **1.0889**  | -0.0007  | 15,999,165 |
| 314  | 1.0915      | **1.0906**  | -0.0010  | 15,974,112 |
| 999  | 1.0901      | **1.0894**  | -0.0007  | 15,996,001 |
| **Mean** | | **1.0896** | **-0.0008** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0251 BPB**.

## Key Techniques

1. **4096-Vocab + MLP 4x + WD 0.090** — PR #1218 @clarkkev, PR #1285 @dexhunter
2. **Depth Recurrence (layers 4,5)** — PR #1204 @msisovic, PR #1260 @dexhunter
3. **Parallel Residuals (from layer 7)** — PR #1204 @msisovic, PR #1289 @MatoTeziTanka
4. **MuonEq-R** — arXiv:2603.28254, PR #1260 @dexhunter
5. **QK-Gain 5.0** — PR #1217 @bigbag
6. **Legal Score-First TTT** — score each 32K-token chunk under torch.no_grad before SGD training. Compiled scoring for correctness. PR #461 @Christopher-Lee-McClendon
7. **Full GPTQ int6 + Brotli + Compressed Wrapper**

## TTT Compliance

Legal score-first per PR #461 framework:
- Every token scored BEFORE any weight update (enforced by torch.no_grad + compiled scoring)
- No training data access during evaluation
- No multi-epoch scoring — each chunk scored exactly once
- Total eval time: ~600s (sliding ~100s + TTT ~300s)

## Compliance

- Legal score-first TTT (backward-looking only)
- No SLOT, no n-gram cache
- GPTQ calibration within training budget
- All four conditions from Issue #1017 satisfied

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
SEED=42 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1218 @clarkkev, PR #1285 @dexhunter, PR #1204 @msisovic, PR #1289 @MatoTeziTanka, PR #1260 @dexhunter, PR #1019 @abaybektursun, PR #1287 @dentity007, PR #1217 @bigbag, PR #493 @parinzee, PR #461 @Christopher-Lee-McClendon
