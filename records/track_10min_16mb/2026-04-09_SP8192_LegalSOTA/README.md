# Record: SP8192 + Triple Recurrence + Banking + Fused MLP + Muon 0.97 — val_bpb 1.0778 (3-seed mean)

**val_bpb = 1.0778** (3-seed mean, std 0.0008) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Pre-quant BPP | Sliding BPP | **TTT BPP** |
|------|---------------|-------------|-------------|
| 1337 | 1.0848 | 1.0786 | **1.0771** |
| 42 | 1.0856 | 1.0792 | **1.0776** |
| 2024 | 1.0862 | 1.0798 | **1.0787** |
| **Mean** | 1.0855 | 1.0792 | **1.0778** |

Merged SOTA (PR #1493): **1.0810 BPP**. Delta: **-0.0032 BPP**.

## Contributions

### 1. Parameter Banking with Parallel Muon (systems)

Restructures 66 separate weight matrices into 4 contiguous 3D parameter banks (qo, kv, mlp_up, mlp_down). Replaces DDP with manual reduce_scatter → batched Newton-Schulz → all_gather. Reduces optimizer step from 19.7ms to 1.3ms (15x faster). Critical fix: restored MuonEq-R row normalization that the refactor had dropped. Combined: **+3.8% training throughput**.

### 2. Fused MLP Triton TMA Kernel (systems)

Fuses `fc(x) → LeakyReLU(0.5) → square` into a single Hopper TMA kernel. The 384MB MLP intermediate never touches HBM. With CUTLASS EVT backward fusion for `(grad @ proj_w) * act_grad`. Combined with banking: **+5.2% total throughput**.

### 3. Muon Momentum 0.97 (training)

Reduced Muon optimizer momentum from default 0.99 to 0.97. Lower momentum provides less smoothing but faster adaptation to the depth-recurrent architecture. **-0.0004 BPP** improvement.

### 4. Triple Depth Recurrence (architecture)

17 virtual layers from 11 physical. Layers 3,4,5 looped 3x total (NUM_LOOPS=2), activated at 35% training. First legal Track B submission with triple recurrence.

### 5. Eval-Time Hash Embedding (eval)

Zero-initialized nn.Embedding(16384, 512) created at eval time, trained through score-first TTT. Bigram hash `h = (prev_token * 2039 + curr_token) % 16384` adds learned residual before RMSNorm.

### 6. TTT LR=0.01 (eval)

Optimized TTT learning rate from default 0.005 to 0.01. **-0.0003 BPP** free improvement.

## Full Architecture

```
SP8192 tokenizer, 11 physical / 17 virtual layers
512 dim, MLP 4x (2048 hidden), GQA 8Q/4KV, head_dim=64
Parallel residuals L7+, QK-Gain 5.0, XSA all 11 layers
LeakyReLU(0.5)², skip gates, logit softcap 30
MuonEq-R (lr=0.022, wd=0.095, momentum=0.97) + AdamW
EMA 0.997, warmdown 66.7%, loop at 35%
SDClip GPTQ int6 (k=12.85) + int8 embed (k=20) + brotli
Score-first TTT: SGD lr=0.01, mom=0.9, 3ep, 32K chunks
Hash embedding: 16384×512, zero-init, trained in TTT
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

Requires: CUTLASS 3.x for EVT backward fusion (optional, falls back to standard PyTorch).

## Credits

PR #1420 @abaybektursun (triple loop + fused kernels), PR #1394 @clarkkev (SP8192 + SDClip), PR #1471 @X-Abhishek-X (3-layer recurrence), PR #1477 @aryanbhosale (parallel residuals + score-first TTT), PR #1460 @resouer (eval-time hash embedding), PR #399 @abaybektursun (parameter banking concept), PR #1514 @dexhunter (Muon 0.97)
