# Non-Record: Text Diffusion (Masked Diffusion) — SP1024

This is a non-record, unlimited-compute submission exploring **masked text diffusion** as an alternative to autoregressive language modeling under the 16MB artifact constraint. It is **not** intended to satisfy the 10-minute cutoff for the main leaderboard.

## Approach

This submission replaces the standard autoregressive (causal) next-token prediction objective with a **masked diffusion** training scheme, inspired by discrete diffusion models (e.g. MDLM, SEDD).

### Key changes from the AR baseline

- **Training objective**: Instead of predicting the next token causally, the model is trained to denoise randomly masked sequences. At each training step, a corruption rate `t ~ Uniform(0, 1)` is sampled, and each token is independently replaced with a special `[MASK]` token (`vocab_size` id) with probability `t`. The model then predicts all original tokens at masked positions via cross-entropy.
- **Non-causal attention**: Since diffusion models are bidirectional by nature, `flash_attn_3_func` is called with `causal=False`, allowing full attention over the entire sequence.
- **Validation**: BPB is estimated by averaging the loss over multiple fixed noise levels `t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`, normalized by `1/t` to approximate the ELBO, then converted to bits-per-byte.
- **Architecture**: Same SP1024 tokenizer, 11 layers, 512 dim, 8 heads, 4 KV heads, MLP mult 3.0 as the current baseline stack.

### Motivation

The "Text Diffusion" item was explicitly listed as a requested direction in the challenge README. This submission is a first sign-of-life attempt to validate whether the masked diffusion objective can train meaningfully under the 16MB artifact budget and 1024-vocab constraint.

### Results

| Seed | Steps | Pre-quant BPB | Post-quant BPB | Artifact (bytes) |
|------|-------|---------------|----------------|------------------|
| 1337 | TBD   | TBD           | TBD            | TBD              |

> **Note**: Results to be filled after run completion.

## Requirements

All dependencies are covered by the challenge `requirements.txt`. No additional packages are needed beyond what is pre-installed in the RunPod template.

Optionally install `zstandard` for better compression (falls back to `zlib` if absent):

```bash
pip install zstandard
```

## Run Command

```bash
RUN_ID=textdiffusion_sp1024 \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
VAL_LOSS_EVERY=4000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Observations

- Masked diffusion with non-causal attention is a fundamentally different inductive bias compared to AR models: the model sees the full (partially masked) context at train time, which may help or hurt under a tight parameter budget.
- The BPB estimation via ELBO approximation at fixed `t` values is noisier than standard AR cross-entropy; more `t` samples per eval step would improve reliability at the cost of eval time.
- The quantization pipeline (int6 GPTQ + LZMA) is unchanged from the AR baseline, so compression behavior should be comparable.
- This experiment was not optimized for speed or score — the goal was to validate feasibility of the diffusion objective within the challenge framework.