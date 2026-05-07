# Record: SP8192 + Full Stack + Headwise Gated Attention + Legal TTT

**val_bpb = 1.0805** (3-seed mean, std 0.0012) | **~15.74 MB** | 8xH100 SXM

**Non-record submission** documenting a novel architecture modification (headwise gated attention) and systematic ablation study across 40+ experiments.

## 3-Seed Results

| Seed | Sliding BPB | **TTT BPB** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0834      | **1.0818**  | 15,697,552 |
| 1337 | 1.0810      | **1.0794**  | 15,694,065 |
| 2025 | 1.0820      | **1.0804**  | 15,693,855 |
| **Mean** | **1.0821** | **1.0805** | **15,695,157** |
| **Std** | **0.0012** | **0.0012** | |

## Author & Research Approach

**An Thien Vo** (James Emerson Vo) — Georgia Tech, CS 7643 Deep Learning.

This submission is the result of a systematic research effort to identify which language model training techniques transfer to the extreme compression regime of Parameter Golf (36M params, 16 MB artifact, 10-minute wall clock on 8×H100).

I surveyed **29+ papers** from NeurIPS 2024-2025, ICML 2025, ICLR 2025, and ACL 2025 — covering attention modifications, normalization strategies, optimizer scheduling, data selection, structured layers, and compression techniques. Each candidate technique was:

1. **Assessed for PG feasibility** — does it fit within the 16 MB / 10-min constraints?
2. **Tested individually on 2×H100** — isolated A/B against the rank 1 baseline
3. **Validated for stacking** — confirmed no interference with other techniques before combining
4. **Scaled to 8×H100** — final verification at competition scale with 3-seed reproducibility

Over **40+ experiments** across 2×H100 and 8×H100, I identified that most techniques published for 125M+ parameter models **do not transfer** to the 36M regime — 5 of 10 tested papers produced negative results.

## Novel Contributions

1. **Headwise Gated Attention** — Original architecture modification: post-attention sigmoid gate applied per-head after FA3+XSA. Q projection widened by `gate_dim`, gate modulates each head's contribution before output projection. Consistent -0.0005 BPB across scales. Inspired by NeurIPS 2025 Best Paper ([arxiv:2505.06708](https://arxiv.org/abs/2505.06708)).

2. **29-Paper Systematic Survey** — Surveyed NeurIPS 2024-2025, ICML 2025, ICLR 2025, and ACL 2025 papers to identify which techniques are applicable to the 16 MB / 10-min / 36M-param regime. Mapped each paper to PG leaderboard presence and feasibility. Found that most techniques published for 125M+ models **do not transfer** — 5 of 10 tested papers produced negative results.

3. **EMA Decay Scaling Law at Short Training Durations** — Discovered that optimal EMA decay shifts dramatically lower when training steps are limited (~1,000-3,000 steps). Default 0.9965 -> optimal **0.990**, with gains monotonically increasing as decay decreases: 0.995 (-0.006), 0.993 (-0.0096), 0.990 (-0.0117 BPB). This submission uses the default 0.9965; the EMA finding is documented as a research contribution.

4. **Negative Results at 36M Scale** — Systematic ablation showing 5 papers fail to transfer: SLM/Rho-1 (NeurIPS 2024), ResFormer (ACL 2025), LR Warmup (NeurIPS 2024), Structured FFN (NeurIPS 2024), and Peri-LN (ICML 2025). Documents **why** each fails — providing guidance for future small-model compression research.

## Key Technique

| Technique | Source | Phase | Impact (2xH100) |
|-----------|--------|-------|-----------------|
| **Headwise Gated Attention** | Inspired by [NeurIPS 2025 Best Paper](https://arxiv.org/abs/2505.06708) | Architecture | -0.0005 BPB |

### Headwise Gated Attention (Novel Contribution)

Post-attention sigmoid gate applied per-head, after FlashAttention-3 + XSA compute the attention output. A learned gate modulates each head's contribution before the output projection:

- Q projection widened by `gate_dim` extra dimensions
- Gate signal extracted from extra Q dims, passed through sigmoid
- Applied elementwise per-head: `attn_out *= gate.unsqueeze(-1)`
- ~50K extra parameters, zero inference latency cost
- Consistent -0.0005 BPP improvement across 2xH100 and 8xH100 scales

Inspired by NeurIPS 2025 Best Paper ([arxiv:2505.06708](https://arxiv.org/abs/2505.06708)).

## Base Stack (from @bigbag, PR #1493)

Our submission builds on @bigbag's base stack:

1. **SP8192 vocabulary** — 8192-token SentencePiece BPE ([PR #1394](https://github.com/openai/parameter-golf/pull/1394) @clarkkev)
2. **11L x 512d x 8H/4KV** — 11 encoder layers, 512 model dim, GQA (8 heads, 4 KV heads)
3. **4x MLP** with LeakyReLU(0.5)^2 activation
4. **3-Layer Depth Recurrence** — layers 3,4,5 looped 2x, 17 virtual layers from 11 physical ([PR #1331](https://github.com/openai/parameter-golf/pull/1331), [#1437](https://github.com/openai/parameter-golf/pull/1437) @dexhunter)
5. **Parallel Residuals** (layers 7+) — GPT-J style ([PR #1412](https://github.com/openai/parameter-golf/pull/1412) @Robby955, [PR #1204](https://github.com/openai/parameter-golf/pull/1204) @msisovic)
6. **Sigmoid Skip Gates** — learned encoder-decoder bridging
7. **Partial RoPE** (16/64 dims) with layerwise LN scale 1/sqrt(layer+1)
8. **XSA (Exclusive Self-Attention)** on all 11 layers — attention orthogonal to self-value vector
9. **QK-Gain 5.25** — learnable per-head query scaling
10. **Logit softcap 30.0** — soft capping on output logits

## Techniques That Failed

Tested on V2 rank 1 stack. All produced negative results at the 36M-parameter scale.

| # | Technique | Paper | Result | Why It Failed |
|---|-----------|-------|--------|---------------|
| 1 | SLM / Rho-1 | [NeurIPS 2024](https://arxiv.org/abs/2404.07965) | ALL ratios worse (+0.002 to +0.155 BPB) | 17M model needs every gradient signal; paper tested at 1B+ |
| 2 | ResFormer (Value Residual) | [ACL 2025](https://arxiv.org/abs/2410.17897) | +0.0022 BPB on 8xH100 | Parallel residuals already provide the gradient highway ResFormer tries to create |
| 3 | LR Warmup | [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95431) | +0.0024 to +0.0066 (monotonically worse) | MuonEq-R has its own momentum warmup; extra LR ramp wastes steps |
| 4 | Structured FFN | [NeurIPS 2024](https://arxiv.org/abs/2406.16450) | +0.04 to +0.05 BPB | Low-rank + block-diagonal too lossy at 36M; paper tested at 125M+ |
| 5 | Peri-LN | [ICML 2025](https://arxiv.org/abs/2502.02732) | Immediate NaN | Output norms conflict with existing attn_scale/mlp_scale + ln_scale_factor |

**Takeaway:** Most techniques from large-scale papers (125M+) do not transfer to the extreme compression regime. The 36M-parameter constraint changes which optimizations matter.

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at frac=0.35). Parallel residuals from layer 7. Skip gates (sigmoid-gated U-Net connections). Headwise gated attention: Q widened by gate_dim, sigmoid gate per-head after FA3+XSA.

Total parameters: ~35.99M.

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps) for matrix params, AdamW for embeddings and scalars. `GRAD_ACCUM_STEPS=1` (8 GPUs), `TRAIN_BATCH_TOKENS=786,432` — ~4,469 steps in ~588s on 8xH100 SXM (PyTorch 2.11, CUDA 13.0). Linear warmdown to LR=0 over final 72% of training. **EMA decay 0.9965** (rank 1 default). Weight decay: Muon WD=0.095, Embed WD=0.085, Adam WD=0.02.

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)` for principled rate-distortion.
- int6 for attention/MLP matrices (`MATRIX_CLIP_SIGMAS=12.85`)
- int7 for token embeddings (`EMBED_BITS=7`, `EMBED_CLIP_SIGMAS=15.0`)
- Byte-shuffle + Brotli-11 compression
- 64 calibration batches from training data

## Evaluation

**Sliding-window causal eval** with stride 64 across the full validation set.

**Score-first TTT** (test-time training) — chunk-based SGD adaptation at eval time:
- Chunk validation tokens into 32K-token segments
- For each chunk: (1) score all sliding windows under `torch.no_grad()`, (2) train model on scored tokens with SGD
- 3 epochs per chunk, lr=0.005, momentum=0.9, cosine LR decay across chunks
- Gradient clipping at 1.0, distributed all-reduce for multi-GPU
- Total eval time: ~395s (within 600s budget)

## Compliance

Per [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) (Track B — legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass.

Additional:
- No SLOT (standard or causal)
- **No Pre-Quantization TTT** — fully legal, no val-data fine-tuning before GPTQ
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes on all 3 seeds
- Training under 600s on all 3 seeds
- Eval (sliding + TTT) under 600s on all 3 seeds

## Reproduction

```bash
pip install --upgrade torch
pip install brotli sentencepiece numpy
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 GATED_ATTN=headwise EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

This submission builds on the work of many contributors to the Parameter Golf community:

- **@bigbag** — Base stack: 3-layer depth recurrence, parallel residuals, sigmoid skip gates, QK-Gain 5.25, LeakyReLU^2, LN scale, legal TTT ([PR #1493](https://github.com/openai/parameter-golf/pull/1493))
- **@clarkkev** (Kevin Clark) — SP8192 vocabulary, GPTQ with SDClip, MuonEq-R optimizer, embedding GPTQ ([PR #1394](https://github.com/openai/parameter-golf/pull/1394))
- **@dexhunter** — Depth recurrence on SP8192 ([PR #1331](https://github.com/openai/parameter-golf/pull/1331), [#1437](https://github.com/openai/parameter-golf/pull/1437)), legal score-first TTT on SP8192 ([PR #1413](https://github.com/openai/parameter-golf/pull/1413))
- **@abaybektursun** — Score-first TTT framework and legality analysis ([PR #549](https://github.com/openai/parameter-golf/pull/549))
- **@Robby955** — Parallel residuals on SP8192 ([PR #1412](https://github.com/openai/parameter-golf/pull/1412))
- **@msisovic** — Parallel residuals concept ([PR #1204](https://github.com/openai/parameter-golf/pull/1204))
- **@X-Abhishek-X** — Hyperparameter tuning and optimizer experiments ([PR #1445](https://github.com/openai/parameter-golf/pull/1445), [#1471](https://github.com/openai/parameter-golf/pull/1471))
- **@aryanbhosale** — Parallel residuals + score-first TTT stack ([PR #1517](https://github.com/openai/parameter-golf/pull/1517))
- **An Thien Vo** (James Emerson Vo) — Headwise gated attention (novel contribution), 29-paper literature survey, 40+ experiment ablation study

## Acknowledgements

- **OpenAI** — for hosting the Parameter Golf challenge and the development grant
- **RunPod** — for compute credits supporting our 2xH100 and 8xH100 experiments
- **Georgia Tech PACE** — for supplementary compute resources
- **@sranganath02** (Sid Ranganathan) — for collaborating on nanochat research and tokenizer investigation as part of our CS 7643 Deep Learning team project
- **CS 7643 Deep Learning** at Georgia Tech, taught by Dr. Zsolt Kira — course context for this research

Total compute cost: ~$280+ across 40+ experiments on RunPod (2xH100 and 8xH100).

In memory of Moomoo, my cat.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `requirements.txt`
- `train_seed42.log`
- `train_seed1337.log`
- `train_seed2025.log`
