# SP8192 + PR #1790 Base + Polar Express NS + MIN_LR + LQER Asym Rank-4

**val_bpb 1.06766** (3-seed mean, std 0.00076) | 8×H100 SXM, 600s train / 600s eval

## What this is

PR #1790's verbatim non-CaseOps stack with three orthogonal techniques layered on:

1. **Polar Express Newton-Schulz coefficients** (from PR #1344): per-iteration minimax-tuned NS-5 tuples replace Muon's fixed `(3.4445, −4.775, 2.0315) × 5`.
2. **MIN_LR=0.10 warmdown floor** (from PR #1787): LR floors at 10% of max instead of decaying to 0.
3. **LQER asymmetric rank-4** (from PR #1797): rank-4 SVD of GPTQ residual on top-3 layers, packed as asymmetric int4 per-group.

## Results (8×H100 80GB SXM, phased TTT)

| Seed | Steps | Pre-quant post-EMA | Quantized | **Post-TTT** | Artifact (bytes) | train_time | eval_time |
|------|------:|-------------------:|----------:|-------------:|-----------------:|-----------:|----------:|
| 1337 | 4954  | 1.06842            | 1.07813   | **1.06699**  | 15,953,831       | 596.15s    | 456.6s    |
| 42   | 4954  | 1.06903            | 1.07856   | **1.06751**  | 15,950,901       | 596.12s    | 455.2s    |
| 2025 | 4953  | 1.06994            | 1.07955   | **1.06849**  | 15,948,627       | 596.13s    | 394.4s    |
| **Mean** | **4954** | **1.06913** | **1.07875** | **1.06766** | **15,951,120** | **596.13s** | **435.4s** |
| **Std**  |          | 0.00076             | 0.00072   | **0.00076**  | 2,634             | 0.02s      | 35.5s     |

## Reproducing

### Environment

Same as PR #1790: PyTorch 2.11.0+cu128, FlashAttention 3 (Hopper), 8×H100 80GB SXM.

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install huggingface_hub tiktoken blobfile tqdm sentencepiece brotli zstandard einops
git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention/hopper && pip install .
cp /tmp/flash-attention/hopper/flash_attn_config.py /opt/conda/lib/python3.11/site-packages/
```

### Data

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python data/cached_challenge_fineweb.py --variant sp8192
```

### Training

```bash
SEED=1337 \
QK_GAIN_INIT=5.25 \
SMEAR_GATE=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_WIDTH=24 \
GPTQ_RESERVE_SECONDS=4 \
GPTQ_CALIBRATION_BATCHES=16 \
POLAR_EXPRESS_NS=1 \
MIN_LR=0.10 \
LQER_ENABLED=1 \
LQER_RANK=4 \
LQER_TOP_K=3 \
LQER_GROUP_SIZE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Replace `SEED=1337` with `42` or `2025` for the other two seeds. All other hyperparameters use code defaults inherited from PR #1790.

## Comparison to base (PR #1790)

PR #1790 reports 1.06991 (3-seed mean, std 0.00061) on seeds {42, 1337, 0}. This PR achieves 1.06766 on seeds {42, 1337, 2025}. The matched seed 1337 head-to-head is 1.06699 (this PR) vs 1.06986 (PR #1790) = **−0.00287 BPP**.

The combined pre-quant + quant gain over PR #1790 is small (~0.0003 BPP); the bulk of the improvement comes through TTT amplification of the post-quant edge that LQER preserves. LQER coverage is saturated at top-K=3 on this stack (verified by single-seed ablation at top-K=12: neutral within noise).

## Files

- `train_gpt.py` — modified PR #1790 train script with the three additions toggleable by env var.
- `train_seed1337.log` — full training + eval log, seed 1337 (final BPB 1.06699).
- `train_seed42.log` — full training + eval log, seed 42 (final BPB 1.06751).
- `train_seed2025.log` — full training + eval log, seed 2025 (final BPB 1.06849).
- `submission.json` — structured metadata for organizer review.

## Rule compliance

Same as PR #1790 (Issue #1017 Track B): strict causal dependence, full normalized distribution over SP8192, score-before-update per-chunk, single left-to-right pass. Artifact, train, and eval all under their respective caps. No CaseOps, no casefold, no preprocessing — BPB measured on original UTF-8 bytes throughout.

## Attribution

Inherits all attributions from PR #1790. New additions:
- Polar Express NS coefficients: PR #1344
- MIN_LR warmdown floor: PR #1787
- LQER asymmetric rank-4: PR #1797
