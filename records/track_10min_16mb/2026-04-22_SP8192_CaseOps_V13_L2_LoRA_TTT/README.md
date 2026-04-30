# Record: SP8192 CaseOps + V13 Curriculum + SmearGate + LoRA-TTT Improvements

**val_bpb = 1.06513** (3-seed mean, std 0.00055) | **~15.98 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | **TTT BPB** | val_loss (nats/tok) | Artifact |
|------|-------------|-------------|---------------------|----------|
| 42   | 1.07767     | **1.06449** | 2.32950             | 15,975,592 |
| 314  | 1.07856     | **1.06543** | 2.33156             | 15,976,709 |
| 999  | 1.07866     | **1.06547** | 2.33162             | 15,976,693 |
| **Mean** | **1.07830** | **1.06513** | **2.33089**     | **15,976,331** |
| **Std**  | **0.00055** | **0.00055** |                 | |

Our previous submission (PR #1541): **1.07785 BPP**. Delta: **-0.01272 BPP**.
Current merged SOTA (PR #1493): **1.0810 BPP**. Delta: **-0.01587 BPP**.

**Note:** val_bpb computed via standard sentencepiece LUT byte counting (consistent with PR #1769 methodology). Train logs report sidecar-based BPB due to CaseOps byte sidecar loading; val_loss (nats/token) is the ground truth cross-entropy and is sidecar-independent.

## Key Techniques

1. **SP8192 CaseOps Tokenizer** -- Lossless reversible case normalization via 4 reserved operator tokens (TITLE/ALLCAPS/CAPNEXT/ESC). Reduces vocabulary fragmentation from case variation. Based on `romeerp/parameter-golf-caseops-v1` HF dataset. Legality pending issue #1604.

2. **Recurrence Depth Curriculum** (from PR #1756 @romeerp) -- Phased training depth schedule: depth 1 (0-33%), depth 3 (33-67%), depth 4 (67-100%). Pre-warms torch.compile kernels for depths 3 and 4. Eval at depth 4. Produces gate weight distributions that survive TTT SGD.

3. **SmearGate (L2)** (from modded-nanogpt @classiclarryd) -- Per-layer learned smoothing gate that blends adjacent token representations. Stacks additively with GatedAttn on the V13 curriculum base (-0.00063 vs V13 alone on seed 42).

4. **GatedAttn + QuantGate** (from PR #1736 @dexhunter) -- Full-dim attention output gate with small-std init (0.005). QuantGate provides int8 passthrough for gate weights during GPTQ.

5. **LoRA-TTT Improvements** (from PR #1767 @renqianluo) -- Four composable changes to BatchedLinearLoRA:
   - Alpha/rank output scaling (`output *= alpha/rank = 144/96 = 1.5x`)
   - Warm-start LoRA A across batches (A persists, only B resets)
   - TTT weight decay 0.5 -> 1.0 (counteracts A accumulation)
   - Alpha 96 -> 144 (more adaptation strength)

6. **Phased Score-First TTT** -- 3-phase legal TTT with 2000 prefix docs. AdamW optimizer (lr=0.0001, WD=1.0). Score tokens first under inference_mode, then train on scored tokens. Never re-score.

7. **GPTQ SDClip + Brotli-11** -- int6 attention/MLP, int7 embeddings, fp16 passthrough for gates/scales/smear params. MATRIX_LR=0.026.

## Architecture

11L x 512D x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence with curriculum (1->3->4 phases). Improved parallel residuals. QK-Gain 5.0.

## Rule Compliance

- Score-first phased TTT (no re-scoring of already-trained tokens)
- No pre-quant TTT on validation data
- No n-gram cache
- All hard gates pass: artifact <= 16 MB (decimal, max 15,976,709 B), train <= 600s, eval <= 600s
- No validation data accessed during training
- CaseOps legality pending issue #1604

## Training Configuration

```
SEED={42,314,999}
SMEAR_GATE_ENABLED=1
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
MATRIX_LR=0.026
TRAIN_LOOP_PHASE_DEPTHS=1,3,4 TRAIN_LOOP_PREWARM_DEPTHS=3,4 EVAL_LOOP_DEPTH=4
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15 MLP_CLIP_SIGMAS=12 ATTN_CLIP_SIGMAS=13
TTT_ENABLED=1 PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3
TTT_LORA_ALPHA=144 TTT_WARM_START_A=1 TTT_WEIGHT_DECAY=1.0
```

## Acknowledgements

Built on techniques from PR #1756 (@romeerp), PR #1767 (@renqianluo), PR #1736 (@dexhunter), PR #1667 (@MarioPaerle), and modded-nanogpt (@classiclarryd).

Thanks to OpenAI's Advanced Competitor grant ($500 compute credit via RunPod).
