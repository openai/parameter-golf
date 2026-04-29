# Record: S9 + SmearGate + Sparse Attention Gate + LQER

**val_bpb = 1.0705** (single seed) | **~15.92 MB** | 8xH100 SXM

## Results

| Seed | Pre-Quant BPB | Quant BPB | **Quant+TTT BPB** | Artifact |
|------|---------------|-----------|---------------------|----------|
| 42   | 1.0720        | 1.0816    | **1.0705**          | 15,918,882 |

Built on PR #1867 (train-gpt-0427 stack). Current merged SOTA (PR #1493): **1.0810 BPP**. Delta: **-0.0105 BPP**.

> Note: This is a single-seed result. Additional seeds needed for full statistical significance per submission guidelines.

## Key Techniques

1. **SmearGate** -- learned gating that mixes neighboring token representations via a sliding window (gate_window=12), enabling local context blending without attention overhead
2. **Sparse Attention Gate** -- per-head learned gates that allow the model to selectively prune attention patterns, reducing effective computation while preserving important interactions
3. **LQER (Low-Rank Quantization Error Reduction)** -- rank-4 asymmetric correction factors (int4, group_size=64) applied to top-3 highest-error layers post-GPTQ, recovering quantization quality
4. **Embed int7** -- 7-bit embedding quantization (clip_sigmas=15.0) instead of int8, saving ~1 MB in artifact size
5. **Phased TTT** -- 3-phase test-time training with Adam optimizer (lr=0.001, cosine decay), respecting document boundaries, with 2000 prefix docs for calibration
6. **S9 Base Stack** -- 11L x 512d x 8H/4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64), layerwise LN scale, tied embeddings, logit softcap=30.0, depth recurrence (layers 3-5, 2 loops), parallel residuals (layer 8+), skip gates, QK-Gain 5.0

## Architecture

11L x 512d x 8H / 4KV, MLP 4x (2048 hidden), LeakyReLU(0.5)^2 activation. Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at frac=0.35). Parallel residuals from layer 8. Skip gates (sigmoid-gated U-Net connections). SmearGate with BOS mask. Sparse attention gates.

Model params: 35,945,671

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps), AdamW for embeddings/scalars. 4700 steps in ~600s on 8xH100 SXM. Linear warmdown to LR=0 over final 75% of training. EMA decay 0.9965. Muon WD=0.095, matrix LR=0.026, embed LR=0.6.

Throughput: ~6,200-7,850 tok/s (8-GPU), peak memory 40.2 GiB.

## Quantization

Full-Hessian GPTQ with SDClip:
- **int6**: attention (c_q, c_k, c_v, proj) and MLP (fc, proj) matrices, clip_sigmas=12.85
- **int7 + LQER asymmetric**: token embeddings, clip_sigmas=15.0
- **int6 + LQER asymmetric**: MLP fc weights (top-3 layers by error)
- **float16 passthrough**: gates, scales, lambdas, smear_gate weights
- Byte-shuffle + Brotli-11 compression

LQER details: rank-4, int4 factor quantization, asymmetric grouping (group_size=64), applied to top-3 highest-error weight matrices.

Total artifact: 15,918,882 bytes (under 16,000,000 limit).

## TTT (Test-Time Training)

Phased score-first TTT with Adam optimizer:
- 3 phases with 2000 prefix documents for calibration
- Phase boundaries: [666, 1333, 2000] graded documents
- Adam (lr=0.001, beta1=0, beta2=0.999, wd=1.0), cosine LR decay per phase
- LoRA adaptation: rank=96, alpha=144, applied to K, O, and MLP projections
- Chunk size: 48 tokens, batch size: 64 sequences
- Document-boundary-respecting chunking
- Total TTT eval time: ~554s (within 600s eval budget)

## Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab
- **Condition 3 (Score before update):** Each phase fully scored before any gradient update
- **Condition 4 (Single pass):** Each token scored exactly once

Additional:
- No SLOT, no pre-quant TTT, no ETLB, no n-gram cache
- Artifact under 16,000,000 bytes
- Training under 600s
- Eval (sliding + TTT) under 600s (~554s actual)

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
