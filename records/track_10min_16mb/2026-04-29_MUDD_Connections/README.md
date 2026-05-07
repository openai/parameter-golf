# Record: MUDD Connections + SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

**val_bpb = 1.0769** (3-seed mean, std 0.0004) | **~15.99 MB** | 8xH100 HBM3

## 3-Seed Results

| Seed | Sliding BPP | **TTT BPP** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0788      | **1.0774**  | 15996370 |
| 423  | 1.0787      | **1.0767**  | 15998081 |
| 424  | 1.0777      | **1.0765**  | 15996509 |
| **Mean** | **1.0784** | **1.0769** | **15996987** |
| **Std** | **0.0005** | **0.0004** | |

Based on SOTA (PR #1493): **1.0810 BPP**. Delta: **-0.0041 BPP ~= -0.0107 nats**. Meets the 0.005-nat threshold.

## Key Techniques

### My Contribution

**MUDD Connections**
I introduce a lite version of **MU**ltiway **D**ynamic **D**ense (MUDD) Connections ([**MUDDFormer**](https://arxiv.org/abs/2502.12170)), and remove sigmoid-gated U-Net connections and residual mixing with x0, both of which can be seen as special cases of MUDD Connections. Although MUDD Connections is more comprehensive, the full version brings more overhead than performance gain. To reduce the overhead of MUDD Connections, I restrict connections from the following three aspects:
- **Query side** roughly a 2-stride query dilation; layer indices to make layer-aggregation: [2,4,6,8,10,12,15,16]
- **Key side** adopt interleaved local/global windows along layer-dimension and use more global windows for upper layers; corresponding window sizes: [2, None, 2, None, 2, None, None, None] (None represents global window)
- **Num of ways** keep V-stream and R-stream (the two more important streams in Q/K/V/R) for layer 12, 15, and retain only R-stream for other layers; corresponding number of ways: [1, 1, 1, 1, 1, 2, 2, 1]

In addition, MUDD Connections prefers wide V-stream, so I switch GQA back to MHA. 

### Previous Credits
1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0), zero selective pruning (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence** (layers 3,4,5, activate at frac=0.35) — 17 virtual layers from 11 physical (PR #1331 @dexhunter, PR #1437 @dexhunter)
3. **Parallel Residuals** (layers 7+) — GPT-J style, attention and MLP read from same input (PR #1412 @Robby955, PR #1204 @msisovic)
4. **QK-Gain 5.25** — learnable per-head query scaling, monotonic improvement from 4.0 to 5.25
5. **Legal Score-First TTT** — SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, cosine LR decay. Score-before-update ordering. (PR #549 @abaybektursun, PR #1413 @dexhunter)
6. **Tuned Hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1445 @X-Abhishek-X)
7. **LZMA code wrapper** — save bytes for code

## Architecture

11L x 512d x 8H / 8KV, MLP 3.5x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at step ~2016). Parallel residuals from layer 7: attention and MLP operate on same pre-residual input. Replace sigmoid-gated U-Net connections and residual mixing with x0 by **MUDD Connections**.


## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps), AdamW for embeddings/scalars. 4367 steps in 588s on 8xH100 HBM3. Linear warmdown to LR=0 over final 72% of training. EMA decay 0.9965.

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)` for principled rate-distortion. int6 for attention/MLP matrices and part of dynamic dense matrices, int8 for token embeddings. Byte-shuffle + Brotli-11 compression. Zero selective pruning needed -- model fits natively under 16MB.

## TTT (Test-Time Training)

Score-first, chunk-based SGD adaptation at eval time:
- Chunk val tokens into 32K-token chunks
- For each chunk: (1) score all sliding windows under `torch.no_grad()`, (2) train model on scored chunk tokens with SGD
- 3 epochs per chunk, cosine LR decay across chunks
- Gradient clipping at 1.0, distributed all-reduce for multi-GPU
- Total TTT eval time: ~371s (within 600s eval budget)

## Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:
- No SLOT (standard or causal)
- No pre-quant TTT on val data (model quantized once during training, TTT adapts at eval time)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes on all 3 seeds
- Training under 600s on all 3 seeds (~588s actual)
- Eval (sliding + TTT) under 600s on all 3 seeds (~500s actual)

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

NCCL_NET=Socket NCCL_DEBUG=WARN SEED=423 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_mudd_seed42 USE_MUDD=1 KEEP_UNET=0 MLP_MULT=3.5 NUM_KV_HEADS=80 WARMUP_STEPS=150 TENSORBOARD_DIR=''  \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@bigbigbag** - SOTA Baseline 04-09 
- **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549, merged precedent)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, #1471)

## Acknowledgements

Thanks to **ColorfulClouds Tech** for providing compute. Thanks to **@Lisennlp** and **@xiaoda99** for valuable discussions on reducing the overhead of MUDD Connections.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed423.log`
- `train_seed424.log`
