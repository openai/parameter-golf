# Record Candidate: Long-Context no-QV TTT with Rank-56 / Prefix-3000 Compute Reallocation

**val_bpb = 1.05874877** (3-seed mean, population std 0.00091680) | **max artifact 15,980,110 bytes** | **8x H100 SXM** | **strict 600s train / 600s eval logs**

This submission is a clean, score-first Track B refinement of the late-April CaseOps / LQER / SparseAttnGate / phased-TTT stack. It keeps the long-context `no_qv` TTT setup from PR #1953 and reallocates the TTT eval budget by reducing LoRA rank to 56 while increasing the phased global-TTT prefix to 3000 documents.

Leaderboard context at submission time:

- vs merged PR #1855 (`1.06108`): **-0.00233123 BPB**, approximately **-0.00510 nats**.
- vs open PR #1953 (`1.05855370`): **+0.00019507 BPB** worse. This PR is submitted as a clean record candidate against the merged leaderboard and as a reproducible ablation on the #1953 lineage, not as a claim to beat #1953 if that PR is accepted first.

## 3-Seed Results

| Seed | Stop step | Train ms | Pre-quant BPB | Quant no-TTT BPB | Final TTT BPB | Eval ms | Artifact bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4900 | 596051 | 1.06108950 | 1.06949683 | **1.05780842** | 519467 | 15,975,989 |
| 0 | 4879 | 596086 | 1.06200352 | 1.07034149 | **1.05844590** | 425873 | 15,976,674 |
| 1234 | 4880 | 596146 | 1.06319088 | 1.07176584 | **1.05999198** | 400555 | 15,980,110 |
| **Mean** | **4886** | **596094** | **1.06209463** | **1.07053472** | **1.05874877** | **448632** | **15,977,591** |

All three seeds satisfy:

- artifact < 16,000,000 bytes
- training wallclock < 600,000 ms
- final TTT eval wallclock < 600,000 ms
- full validation set with CaseOps byte-sidecar BPB accounting

## What Changed

Relative to the PR #1953 long-context `no_qv` stack:

```bash
TTT_LORA_RANK=56              # PR #1953 used rank 80 in its reproduction command
PHASED_TTT_PREFIX_DOCS=3000   # PR #1953 used 2500
PHASED_TTT_NUM_PHASES=3
EVAL_SEQ_LEN=2560
TTT_EVAL_SEQ_LEN=2560
TTT_MASK=no_qv
TTT_Q_LORA=0
TTT_V_LORA=0
TTT_LOCAL_LR_MULT=0.75
QK_GAIN_INIT=5.25
```

The intent is a simple eval-budget tradeoff: smaller per-document LoRA state with a longer score-first global-TTT prefix. The result is seed-sensitive: it improves seeds 42 and 0 relative to the public #1953 numbers, but seed 1234 is weaker, so the 3-seed mean lands behind #1953 while still clearing the merged #1855 record by the 0.005-nat threshold.

## Key Inherited Components

This submission inherits the core stack from the public #1953 / #1945 / #1908 / #1855 lineage:

- CaseOps lossless tokenizer with validation byte sidecar
- BOS-fixed SmearGate and SparseAttnGate
- LQER asymmetric rank-4 quantization
- AWQ-lite mixed-precision GPTQ
- asymmetric logit rescale
- fused softcapped CE
- long-context score-first phased TTT with `no_qv` LoRA mask
- per-group lrzip artifact compression

## Compliance Against Issue #1017 Conditions

- **C1 strict causal dependence:** every token is scored from the submitted artifact plus strict prefix state only. CaseOps is a deterministic lossless transform; the byte sidecar supplies denominator accounting and does not affect predictions.
- **C2 full normalized distribution:** the neural model emits a standard full-vocabulary distribution over the 8192 CaseOps token ids. No byte-level scorer, entropy expert, PPM mixture, n-gram cache, or target-conditioned gate is used.
- **C3 score-before-update:** phased TTT scores chunks/documents before any LoRA or global update can use those scored tokens. Updated state is used only for later tokens/documents.
- **C4 single left-to-right pass:** every validation token is scored once. No rescoring, no multi-pass selection, and no oracle best-of-k over validation outcomes.

Additional exclusions:

- no SLOT
- no byte-PPM or token-PPM mixer
- no pre-quant TTT on validation data
- no eval-time logit bias selected from the observed target
- no validation-token leakage into training weights before scoring

## Reproduction

The CaseOps dataset/tokenizer are the same public data path used by the inherited stack (`romeerp/parameter-golf-caseops-v1`):

```bash
pip install brotli sentencepiece huggingface-hub
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Fetch / prepare the CaseOps SP8192 dataset used by the inherited stack.
# DATA_PATH should contain fineweb_train_*.bin, fineweb_val_*.bin, and fineweb_val_bytes_*.bin.

export DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export CASEOPS_ENABLED=1
export VOCAB_SIZE=8192
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=0
export TTT_ENABLED=1
export PHASED_TTT_ENABLED=1
export PHASED_TTT_NUM_PHASES=3
export PHASED_TTT_PREFIX_DOCS=3000
export TTT_LORA_RANK=56
export TTT_MASK=no_qv
export TTT_Q_LORA=0
export TTT_V_LORA=0
export TTT_LOCAL_LR_MULT=0.75
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_CHUNK_SIZE=48
export EVAL_SEQ_LEN=2560
export TTT_EVAL_SEQ_LEN=2560
export QK_GAIN_INIT=5.25
export MATRIX_LR=0.026
export MIN_LR=0.1
export WARMDOWN_FRAC=0.85
export BETA2=0.99
export EMBED_BITS=7
export MATRIX_CLIP_SIGMAS=12.85
export ATTN_CLIP_SIGMAS=13.0
export MLP_CLIP_SIGMAS=11.5
export EMBED_CLIP_SIGMAS=14.0
export GRAD_CLIP_NORM=0.3
export FUSED_CE_ENABLED=1
export SMEAR_GATE_ENABLED=1
export GATE_WINDOW=12
export SPARSE_ATTN_GATE_ENABLED=1
export SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_QUANT_GATE=1
export LQER_ENABLED=1
export LQER_RANK=4
export LQER_TOP_K=3
export LQER_GROUP_SIZE=64
export LQER_ASYM_ENABLED=1
export LQER_ASYM_GROUP=64
export AWQ_LITE_ENABLED=1
export ASYM_LOGIT_RESCALE=1
export GPTQ_RESERVE_SECONDS=4.0
export GPTQ_CALIBRATION_BATCHES=16
export COMPRESSOR=pergroup
export NCCL_NET=Socket

SEED=42   RUN_ID=rank56_prefix3000_seed42   ARTIFACT_DIR=./artifacts/seed42   torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=0    RUN_ID=rank56_prefix3000_seed0    ARTIFACT_DIR=./artifacts/seed0    torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=1234 RUN_ID=rank56_prefix3000_seed1234 ARTIFACT_DIR=./artifacts/seed1234 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed0.log`
- `train_seed1234.log`

## Credits

This is a small compute-allocation refinement on a broad community stack. Credit belongs primarily to the authors of the inherited lineage:

- @codemath3000 for PR #1855
- @alertcat, @romeerp, @jorge-asenjo, and @aquariouseworkman for PR #1945 / #1908 / #1923 components
- @nprime06, @dexhunter, @samacqua, and earlier VarLen/phased-TTT contributors
- @romeerp and @dexhunter for the CaseOps tokenizer and byte-sidecar precedent

The contribution here is the rank-56 / prefix-3000 score-first TTT ablation and the fully logged 3-seed rerun under the standard constraints.
