# Record: SP8192 + Triple Recurrence + Banking + Fused MLP + Muon 0.97 — val_bpb 1.0783 (3-seed mean)

**val_bpb = 1.0783** (3-seed mean, std 0.0004) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Pre-quant BPP | Sliding BPP | **TTT BPP** | Artifact |
|------|---------------|-------------|-------------|----------|
| 1337 | 1.0859 | 1.0798 | **1.0782** | 15,986,623 |
| 42 | 1.0856 | 1.0793 | **1.0781** | 15,983,529 |
| 2024 | 1.0862 | 1.0800 | **1.0788** | 15,986,767 |
| **Mean** | 1.0859 | 1.0797 | **1.0783** | |

## Architecture

```
SP8192 tokenizer, 11 physical / 17 virtual layers
512 dim, MLP 4x (2048 hidden), GQA 8Q/4KV, head_dim=64
Parallel residuals L7+, QK-Gain 5.0, XSA all 11 layers
LeakyReLU(0.5)², skip gates, logit softcap 30
MuonEq-R (lr=0.022, wd=0.095, momentum=0.97) + AdamW
EMA 0.997, warmdown 66.7%, loop at 35%
SDClip GPTQ int6 (k=12.85) + int8 embed (k=20) + brotli
Score-first TTT: SGD lr=0.01, mom=0.9, 3ep, 32K chunks
Hash embedding: 16384x512, zero-init, trained in TTT
~36M params, ~15.99MB artifact
```

## Compliance (Track B — Score-First TTT)

Per Issue #1017:
- **Condition 1:** Hash key uses prefix tokens only
- **Condition 2:** Full normalized softmax distribution
- **Condition 3:** Each chunk scored under no_grad() before TTT update
- **Condition 4:** Single left-to-right pass, no rescoring

No SLOT, no pre-quant TTT, no n-gram caches, no Tap-In.

## Reproduction

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
SEED=1337 TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.01 MUON_MOMENTUM=0.97 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1420 @abaybektursun (triple loop + fused kernels), PR #1394 @clarkkev (SP8192 + SDClip), PR #1471 @X-Abhishek-X (3-layer recurrence), PR #1477 @aryanbhosale (parallel residuals + score-first TTT), PR #1460 @resouer (eval-time hash embedding), PR #399 @abaybektursun (parameter banking concept), PR #1514 @dexhunter (Muon 0.97)
