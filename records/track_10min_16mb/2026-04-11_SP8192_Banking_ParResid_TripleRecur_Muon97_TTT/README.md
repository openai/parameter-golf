# Record: SP8192 + Banking + Triple Recurrence + Parallel Residuals + Muon 0.97 + TTT — val_bpb 1.0790 (5-seed mean)

**val_bpb = 1.0790** (5-seed mean, std 0.0003) | **~15.99 MB** | 8xH100 SXM

## 5-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | **TTT BPB** | val_loss (nats) | Artifact |
|------|-------------|-----------------|----------|
| 42   | **1.0788**  | 2.7866          | 15,988,830 |
| 314  | **1.0789**  | 2.7868          | 15,983,617 |
| 1337 | **1.0788**  | 2.7867          | 15,985,310 |
| 7    | **1.0793**  | 2.7880          | 15,986,416 |
| 999  | **1.0795**  | 2.7884          | 15,986,416 |
| **Mean** | **1.0790** | **2.7873** | |

Merged SOTA (PR #1493): **1.0810 BPB / 2.7920 nats**. Delta: **-0.0047 nats** (5-seed), **-0.0020 BPB**.

## Stack

PR #1523 base (@abaybektursun) with hash embedding disabled and Triton fused MLP removed (standard MLP used instead). Key components:

1. **SP8192** vocab with GPTQ embeddings and SDClip quantization
2. **Parameter Banking** — batched Newton-Schulz optimizer step
3. **Triple Depth Recurrence** (L3-5, 17 virtual layers from 11 physical)
4. **Parallel Residuals** (L7+, GPT-J style)
5. **Muon 0.97** momentum (from PR #1514 @dexhunter)
6. **QK-Gain 5.25**
7. **Score-First TTT** (3 epochs, SGD lr=0.005, PR #461 framework)
8. **EMA 0.9965, WD 0.095, warmdown 0.72**

## Compliance (Track B — Score-First TTT)

- Score-first TTT: each chunk scored under `torch.no_grad()` BEFORE SGD weight update
- No SLOT, no hash embedding, no pre-quant TTT, no n-gram cache, no ETLB
- All four conditions from Issue #1017 satisfied
- All artifacts < 16MB, train < 600s, eval < 600s

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
SEED=42 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1523 @abaybektursun (base: banking + triple recurrence + parallel residuals), PR #1394 @clarkkev (SP8192 + SDClip), PR #1514 @dexhunter (Muon 0.97), PR #1493 @bigbag (merged #1 hyperparameters), PR #1204 @msisovic (parallel residuals concept)
