# Record: QK-Gain 5.5 + SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT

**val_bpb = 1.0810** (3-seed mean, std 0.0005) | **< 16 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | **TTT BPB** | Artifact |
|------|-------------|----------|
| 42   | **1.0804**  | 15,994,470 |
| 314  | **1.0812**  | 15,993,777 |
| 999  | **1.0814**  | 15,991,277 |
| **Mean** | **1.0810** | |
| **Std** | **0.0005** | |

## Key Change

**QK_GAIN_INIT=5.5** (up from 5.25). The monotonic improvement trend in query-key gain scaling continues past 5.25. This extends the finding from PR #1394 (@clarkkev) which documented improvement from 4.0 to 5.25.

## Base Architecture

Built on the SOTA foundation from:
- **@clarkkev** — SP8192 + GPTQ SDClip + MuonEq-R + depth recurrence (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning (PR #1445, #1471)

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: layers 3-5 loop (num_loops=2, activated at frac=0.35). Parallel residuals from layer 7. Skip gates (sigmoid-gated U-Net connections). XSA on all layers.

## Training

~4600 steps in ~588s on 8xH100 SXM. EMA decay 0.9965. Warmdown frac 0.72. WD=0.095. MuonEq-R (row-normalized, Newton-Schulz 5 steps).

## Quantization

Full-Hessian GPTQ with SDClip: int6 for attention/MLP matrices, int8 for token embeddings. Brotli-11 compression.

## TTT (Test-Time Training)

Legal score-first TTT: SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, cosine LR decay. Each chunk scored under `torch.no_grad()` before any SGD update. Each token scored exactly once.

## Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):
- Condition 1 (Causality): Sliding-window eval is strictly causal
- Condition 2 (Normalized distribution): Standard softmax over full vocab
- Condition 3 (Score before update): Each chunk scored under torch.no_grad() before SGD update
- Condition 4 (Single pass): Each token scored exactly once
- All artifacts under 16,000,000 bytes on all 3 seeds
- Training under 600s on all 3 seeds (~588s actual)
- Eval (sliding + TTT) under 600s on all 3 seeds

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
