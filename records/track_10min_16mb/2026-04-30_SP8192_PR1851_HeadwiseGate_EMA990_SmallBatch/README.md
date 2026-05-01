# Record: SP8192 + PR#1851 Fork + Headwise Gate + EMA 0.990 + Small Batch + Emb6

**val_bpb = 1.0066** (3-seed mean, std 0.0009) | **~15.97 MB** | 8xH100 SXM

**Record submission** — beats previous SOTA (1.0611, PR #1855) by 0.0545 nats.

## 3-Seed Results

| Seed | Pre-Q BPB | Quant BPB | **TTT BPB** | Artifact |
|------|-----------|-----------|-------------|----------|
| 42   | 1.0025    | 1.0205    | **1.0069**  | 15,975,827 |
| 1337 | 1.0017    | 1.0190    | **1.0057**  | 15,973,108 |
| 2025 | 1.0030    | 1.0206    | **1.0073**  | 15,973,714 |
| **Mean** | **1.0024** | **1.0200** | **1.0066** | **15,974,216** |
| **Std** | **0.0007** | **0.0009** | **0.0009** | |

## Author & Research Approach

**An Thien Vo** (James Emerson Vo) — Georgia Tech, CS 7643 Deep Learning.

This submission forks PR #1851 (@aquariouseworkman) and adds 4 novel contributions discovered through a systematic research effort: 29+ papers surveyed, 40+ experiments across 2×H100 and 8×H100, and careful ablation to identify which techniques transfer to the extreme compression regime of Parameter Golf.

## Novel Contributions

1. **Headwise Gated Attention** — Post-attention sigmoid gate applied per-head after FA3+XSA. Q projection widened by `gate_dim`, gate modulates each head's contribution before output projection. ~50K extra parameters, zero inference cost, consistent -0.0005 BPP improvement across scales. Inspired by NeurIPS 2025 Best Paper ([arxiv:2505.06708](https://arxiv.org/abs/2505.06708)).

2. **EMA Decay = 0.990** — Discovered optimal EMA decay shifts dramatically lower when training steps are limited. Default 0.9965 → optimal **0.990** on 8×H100: more aggressive weight averaging captures better training signal when the training window is fixed at 10 minutes.

3. **Small Batch (ga=1, 196K tokens)** — Reducing effective batch size from 786K to 196K tokens yields 3.3× more optimizer steps in the same wall clock. On 8×H100, this enables 12,382 steps vs ~4,500 with default batch size, giving the optimizer more fine-grained updates.

4. **6-bit Embedding Quantization** — Reducing `EMBED_BITS` from 8 to 6 saves ~1 MB on the compressed artifact, enabling headwise gated attention's extra parameters to fit under the 16 MB budget. Costs ~0.013 BPB in quantization gap but enables the complete technique stack.

## Key Techniques

| Technique | Source | Impact |
|-----------|--------|--------|
| **Headwise Gated Attention** | James Vo (novel) | -0.0005 BPB |
| **EMA Decay = 0.990** | James Vo (novel finding) | Enables better weight averaging |
| **Small Batch (196K tokens)** | James Vo (novel finding) | 3.3× more optimizer steps |
| **6-bit Embedding Quant** | James Vo (novel) | -1 MB artifact size |

## Base Stack

### PR #1851 (@aquariouseworkman)

Extends @bigbag's PR #1493 with:
- **LQER** (asymmetric int4 error correction) — `lqer_enabled=True`, `lqer_asym_enabled=True`
- **Phased TTT** (multi-phase test-time training with LoRA)
- **Fused softcapped cross-entropy kernel** (Triton) — `fused_ce_enabled=True`
- **Brotli compression**
- SmearGate (available but **disabled** in our runs: `smear_gate_enabled=False`)
- Sparse Attention Gate (available but **replaced** by our headwise gate: `sparse_attn_gate_enabled=False`)
- CaseOps tokenizer (**active via symlinked data** — env var was `caseops_enabled=False` but pod data paths pointed to CaseOps-tokenized shards)

### Upstream: @bigbag PR #1493

- SP8192 vocabulary (8192-token SentencePiece BPE, CaseOps variant via symlinked data)
- 11L × 512d × 8H/4KV, MLP 4×, LeakyReLU(0.5)²
- 3-layer depth recurrence (layers 3-4-5 looped 2×, 17 virtual from 11 physical)
- Parallel residuals (layers 7+), sigmoid skip gates
- Partial RoPE (16/64 dims), layerwise LN scale
- XSA on all 11 layers, QK-Gain 5.0, logit softcap 30.0
- MuonEq-R optimizer, GPTQ int6+brotli, score-first TTT

## Techniques That Failed

Tested on the V2 rank 1 stack. All produced negative results at the 36M-parameter scale.

| # | Technique | Paper | Result | Why It Failed |
|---|-----------|-------|--------|---------------|
| 1 | SLM / Rho-1 | [NeurIPS 2024](https://arxiv.org/abs/2404.07965) | All ratios worse | 17M model needs every gradient signal; paper tested at 1B+ |
| 2 | ResFormer | [ACL 2025](https://arxiv.org/abs/2410.17897) | +0.0022 BPB | Parallel residuals already provide the gradient highway |
| 3 | LR Warmup | [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95431) | +0.0024 to +0.0066 | MuonEq-R has its own momentum warmup |
| 4 | Structured FFN | [NeurIPS 2024](https://arxiv.org/abs/2406.16450) | +0.04 to +0.05 BPB | Low-rank too lossy at 36M; tested at 125M+ |
| 5 | Peri-LN | [ICML 2025](https://arxiv.org/abs/2502.02732) | Immediate NaN | Conflicts with existing normalization stack |
| 6 | Differential Attention | [ICLR 2025 Oral](https://arxiv.org/abs/2410.05258) | +0.0138 BPB | 2× FA3 calls reduces throughput 22% |
| 7 | HybridNorm | [NeurIPS 2025](https://arxiv.org/abs/2501.01422) | +0.011 BPB | Normalization axis already saturated |
| 8 | GPTQ Sequential/Embed | Frantar et al. | +0.19 to +0.66 | Sequential Hessians through dequantized blocks are inferior |

## Architecture

11L × 512d × 8H/4KV, MLP 4×, LeakyReLU(0.5)², partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (enabled at frac=0.35). Parallel residuals from layer 8. Skip gates. Headwise gated attention: Q widened by gate_dim, sigmoid gate per-head after FA3+XSA. LQER asymmetric int4 error correction from PR #1851. Fused softcapped CE kernel (Triton).

Total parameters: ~35.99M.

## Training

MuonEq-R optimizer for matrix params, AdamW for embeddings/scalars. `GRAD_ACCUM_STEPS=1` (8 GPUs), `TRAIN_BATCH_TOKENS=196,608` (small batch), ~12,382 steps in ~596s on 8×H100 SXM (PyTorch 2.11, CUDA 13.0). Linear warmdown over final 75%. **EMA decay 0.990**. Weight decay: Muon WD=0.095, Embed WD=0.085, Adam WD=0.02.

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)`.
- int6 for attention/MLP matrices (`MATRIX_CLIP_SIGMAS=12.85`, `ATTN_CLIP_SIGMAS=13.0`)
- **int6 for token embeddings** (`EMBED_BITS=6`, `EMBED_CLIP_SIGMAS=20.0`)
- Byte-shuffle + Brotli-11 compression
- 16 calibration batches from training data

## Evaluation

**Phased TTT** (from PR #1851) — multi-phase test-time training with LoRA adaptation:
- LoRA rank 96, applied to K/MLP/O projections
- Adam optimizer, lr=0.0001, weight decay=1.0
- Eval seq len 2048, chunk-based scoring
- Score-first: tokens scored under `torch.no_grad()` BEFORE gradient updates
- Total eval time: ~353-389s (within 600s budget)

## Compliance

Per [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) (Track B):

- **C1 (Causality):** Causal eval only — each position scored from prefix tokens.
- **C2 (Normalized):** Standard softmax over full vocab.
- **C3 (Score before update):** Each chunk fully scored before any LoRA update.
- **C4 (Single pass):** Each token scored exactly once.
- **No SLOT, No PreQuantTTT, No ETLB, No n-gram cache.**
- All artifacts under 16,000,000 bytes on all 3 seeds.
- Training under 600s on all 3 seeds.
- Eval under 600s on all 3 seeds.

## Reproduction

```bash
git clone https://github.com/jamesEmerson112/DL-Team-Proposal.git
cd DL-Team-Proposal && git checkout James-experiment
cd parameter-golf

pip install --upgrade torch
pip install brotli sentencepiece numpy python-minifier
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"

# Step 1: Download regular SP8192 data
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80

# Step 2: Download CaseOps-tokenized data
# Due to a technical issue (regular sp8192 and CaseOps sp8192 datasets conflict
# on disk), CaseOps data was symlinked into the standard paths on our pod.
# The env var caseops_enabled=False, but training used CaseOps-tokenized shards.
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
  MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
  python3 data/cached_challenge_fineweb.py \
    --variant sp8192_lossless_caps_caseops_v1_reserved \
    --train-shards 80

# Step 3: Symlink CaseOps data into standard paths
mv data/datasets/fineweb10B_sp8192 data/datasets/fineweb10B_sp8192_regular 2>/dev/null || true
mv data/tokenizers/fineweb_8192_bpe.model data/tokenizers/fineweb_8192_bpe.model.regular 2>/dev/null || true
ln -s fineweb10B_sp8192_lossless_caps_caseops_v1_reserved data/datasets/fineweb10B_sp8192
ln -s fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model data/tokenizers/fineweb_8192_bpe.model

# Step 4: Train (CASEOPS_ENABLED=0 — byte sidecar not used, but data is CaseOps-tokenized)
SEED=42 GATED_ATTN_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=0 \
  EMA_DECAY=0.990 GRAD_ACCUM_STEPS=1 TRAIN_BATCH_TOKENS=196608 EMBED_BITS=6 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=1337` and `SEED=2025` for 3-seed verification.

## Credits

- **@aquariouseworkman** — PR #1851 base: LQER, phased TTT, fused CE kernel, brotli compression ([PR #1851](https://github.com/openai/parameter-golf/pull/1851))
- **@bigbag** — Upstream stack: depth recurrence, parallel residuals, skip gates, QK-Gain 5.0, LeakyReLU², XSA, MuonEq-R, EMA, legal TTT ([PR #1493](https://github.com/openai/parameter-golf/pull/1493))
- **@clarkkev** (Kevin Clark) — SP8192 vocabulary, GPTQ with SDClip, MuonEq-R ([PR #1394](https://github.com/openai/parameter-golf/pull/1394))
- **@dexhunter** — Depth recurrence ([PR #1331](https://github.com/openai/parameter-golf/pull/1331), [#1437](https://github.com/openai/parameter-golf/pull/1437)), legal score-first TTT ([PR #1413](https://github.com/openai/parameter-golf/pull/1413))
- **@abaybektursun** — Score-first TTT framework ([PR #549](https://github.com/openai/parameter-golf/pull/549))
- **@Robby955** — Parallel residuals ([PR #1412](https://github.com/openai/parameter-golf/pull/1412))
- **@msisovic** — Parallel residuals concept ([PR #1204](https://github.com/openai/parameter-golf/pull/1204))
- **An Thien Vo** (James Emerson Vo) — Headwise gated attention, EMA=0.990, small batch, 6-bit embedding quant, 29-paper survey, 40+ experiment ablation

## Acknowledgements

- **OpenAI** — for hosting the Parameter Golf challenge and the development grant
- **RunPod** — for compute credits supporting our 2×H100 and 8×H100 experiments
- **Georgia Tech PACE** — for supplementary compute resources
- **CS 7643 Deep Learning** at Georgia Tech, taught by Dr. Zsolt Kira
- **@sranganath2** (Sid Ranganathan) — contributed to discussion about tokenizer investigation, nanochat research, fused CE kernel insights, and research papers
- **@Ashray14** — contributed to discussion about research papers
- **@ialeksic3** — contributed to discussion about research papers

Total personal compute cost: ~$1,165 ($640 out-of-pocket + $525 OpenAI development grant) across 130+ experiments on RunPod.

In memory of Moomoo, my cat.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed1337.log`
- `train_seed2025.log`
