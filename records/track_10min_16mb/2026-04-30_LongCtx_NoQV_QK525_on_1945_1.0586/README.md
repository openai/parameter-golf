# Record: PR #1945 base + 2560 long-context + no_qv TTT mask + TTT LR 0.75 + QK_GAIN 5.25 (val_bpb 1.05855)

**val_bpb = 1.05855370** (3-seed mean, std 0.00029539) | max artifact 15,992,914 B | 8x H100 SXM | 600s train / 600s eval

Stacks four small, individually validated levers on the exact PR #1945 alertcat V21 record source (which is itself PR #1855 + PR #1908 AWQ-lite + PR #1923 Asymmetric Logit Rescale). Each lever was already measured on prior bases. The contribution here is the orthogonal stack and the production verification.

## 3-seed Results

| Seed | Stop step | Train ms | Pre-quant BPB | Quant no-TTT BPB | **Post-TTT BPB** | Eval s | Artifact bytes |
|-----:|----------:|---------:|--------------:|-----------------:|-----------------:|-------:|---------------:|
| 42   | 4895      | 595955   | 1.06163175    | 1.06993750       | **1.05824720**   | 430.0  | 15,988,861     |
| 0    | 4896      | 596123   | 1.06196584    | 1.07029420       | **1.05846113**   | 441.5  | 15,988,757     |
| 1234 | 4916      | 596130   | 1.06199757    | 1.07068689       | **1.05895276**   | 513.1  | 15,992,914     |
| **Mean** | **4902** | **596069** | **1.06186505** | **1.07030620** | **1.05855370** | **461.5** | **15,990,177** |

Population std on final BPB: **0.00029539**.

vs current rank 1 (PR #1855 at 1.06108): **-0.00253 BPB**.
vs PR #1945 reported mean (1.05943381): **-0.00088 BPB**.
vs merge bar (1.05893): **-0.00038 BPB**.

All seeds clear the 600s train cap, 600s eval cap, and 16,000,000-byte artifact cap.

## What changed vs PR #1945

Four literal-constant additions on top of the exact alertcat V21 source. No new code paths, no new mechanisms, no architectural changes:

```
EVAL_SEQ_LEN          = 2560     # was 2048
TTT_EVAL_SEQ_LEN      = 2560     # was 2048
TTT_MASK              = no_qv    # was default (Q/V LoRA active)
TTT_Q_LORA            = 0        # disable Q LoRA in TTT
TTT_V_LORA            = 0        # disable V LoRA in TTT
TTT_LOCAL_LR_MULT     = 0.75     # was 1.0
QK_GAIN_INIT          = 5.25     # was 5.0
```

Everything else is verbatim PR #1945. AWQ-lite, Asymmetric Logit Rescale, CaseOps tokenizer, Polar Express NS, MIN_LR, fused softcapped CE, LQER asymmetric rank-4, sparse attention gate, BOS-fixed SmearGate, phased TTT (3 phases, 2500 prefix docs), per-group lrzip + brotli compression, GPTQ int6 + int7 embeddings.

## Why each lever

Each lever was already publicly measured on a closely related base. None alone clears the merge bar. Combined on the PR #1945 base, they compose into a clearing stack.

**`EVAL_SEQ_LEN=2560` with `TTT_MASK=no_qv`**: extends eval and TTT score-first context past 2048. The baseline #1855 measurement reported 2560 + no_qv at val_bpb 1.06109776 in 473.4s, an improvement of about -0.00058 BPB vs the 2048 anchor at 473.4s eval time. Legal under the 600s eval cap.

**`TTT_LOCAL_LR_MULT=0.75`**: scales local LoRA-TTT optimizer LR. The baseline #1855 sweep at 2560 no_qv showed 0.75 was the best multiplier in `{0.50, 0.75, 1.00, 1.25, 1.50, 2.00}` at val_bpb 1.06104597. Same direction holds here.

**`QK_GAIN_INIT=5.25`**: replaces the 5.0 default per-head learnable Q-gain initialization. The baseline #1855 measurement reported QK_GAIN_INIT=5.25 seed-1234 post-TTT at -0.00019364 vs 5.0. Train-time init only.

**Asymmetric Logit Rescale via PR #1945 / PR #1923**: replaces the single `logit_softcap=30.0` with two learnable scalars `softcap_pos` and `softcap_neg`, trained inside Phased TTT global SGD. PR #1945 finds Asym is positive when stacked with AWQ-lite due to better TTT recovery. Initialized at the symmetric value (30.0) so eval is identity at start. Inherited from PR #1945.

**AWQ-lite mixed precision via PR #1945 / PR #1908**: during GPTQ calibration, collect activation RMS per layer, select the most-salient 64-column group, keep that group at int8 inside the GPTQ solve. Inherited from PR #1945.

## Data source

This submission uses the canonical CaseOps SP8192 export hosted on Hugging Face (`romeerp/parameter-golf-caseops-v1`), accessed via `huggingface_hub.snapshot_download`. The HF manifest reports `num_val_docs: 50000`, `docs_val: 50000`, `files_train: 80`, `tokens_train: 8000000058`, with disjoint train/val partitions per the canonical first-50k-docs validation split.

No local rebuild via `prepare_caseops_data.py` was used in the production runs; `prepare_caseops_data.py` is not part of this PR's file set, and no `--val-docs` invocation appears in the submitted setup. The submitted logs (`train_seed42.log`, `train_seed0.log`, `train_seed1234.log`) all use `DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`, which points at the HF snapshot extraction location. All three seeds report `val_tokens: 47851520`, consistent with the canonical 50k validation split.

## Compliance (Issue #1017)

- [x] **C1 strict causal dependence**: standard sliding-window scoring with cu_seqlens packed-doc handling. PR #1855 BOS-fixed SmearGate inherited.
- [x] **C2 full normalized distribution**: standard softmax over SP8192 vocab.
- [x] **C3 score-before-update**: phased TTT scores each chunk before any LoRA gradient step. The `no_qv` mask only zeroes Q and V LoRA paths, K / MLP / O / lm_head LoRA still trained per the PR #1855/#1945 implementation.
- [x] **C4 single left-to-right pass**: each validation token scored exactly once.
- [x] **No SLOT, no n-gram cache, no logit bias, no pre-quant TTT on val data, no PPM mixture.**
- [x] **Full validation set** (`fineweb_val_*.bin` + CaseOps byte sidecar) per PR #1855 base.
- [x] **Artifact under 16,000,000 bytes** for all seeds (max 15,992,914).
- [x] **Train under 600s** strict for all seeds (max 596,130 ms).
- [x] **Eval under 600s** for all seeds (max 513.1s).
- [x] **3-seed mean clears p < 0.01 vs PR #1855 (1.06108)** (delta -0.00253, std 0.00029539, t > 14).

## Reproduction

```bash
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
VOCAB_SIZE=8192 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TTT_ENABLED=1 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_NUM_PHASES=3 \
PHASED_TTT_PREFIX_DOCS=2500 \
TTT_LORA_RANK=80 \
TTT_MASK=no_qv \
TTT_Q_LORA=0 \
TTT_V_LORA=0 \
TTT_LOCAL_LR_MULT=0.75 \
EVAL_SEQ_LEN=2560 \
TTT_EVAL_SEQ_LEN=2560 \
QK_GAIN_INIT=5.25 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
EMBED_BITS=7 \
MATRIX_CLIP_SIGMAS=12.85 \
ATTN_CLIP_SIGMAS=13.0 \
MLP_CLIP_SIGMAS=11.5 \
EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 \
FUSED_CE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SPARSE_ATTN_GATE_ENABLED=1 \
LQER_ENABLED=1 \
LQER_RANK=4 \
LQER_TOP_K=3 \
LQER_GROUP_SIZE=64 \
LQER_ASYM_ENABLED=1 \
LQER_ASYM_GROUP=64 \
AWQ_LITE_ENABLED=1 \
ASYM_LOGIT_RESCALE=1 \
GPTQ_RESERVE_SECONDS=4.0 \
GPTQ_CALIBRATION_BATCHES=16 \
COMPRESSOR=pergroup \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat for `SEED=0` and `SEED=1234`.

## Lineage

This stands on a long chain of prior submissions. The four added levers and the PR #1945 core are all from public PRs:

- **PR #1945** (@alertcat): V21 stack of #1855 + AWQ-lite + Asymmetric Logit Rescale, the direct base for this submission.
- **PR #1855** (@codemath3000): the BOS-fixed SmearGate + LQER + SparseAttnGate + 9-hparam stack that is the current rank 1.
- **PR #1908** (@romeerp): AWQ-lite mixed-precision GPTQ.
- **PR #1923** (@jorge-asenjo): Asymmetric Logit Rescale (originally from modded-nanogpt @classiclarryd PR #181).
- **PR #1797** (@dexhunter): Smear gate + LQER asymmetric int4 lineage.
- **PR #1787** (@nprime06): Polar Express NS + MIN_LR + Sparse Attn Gate + Fused CE.
- **PR #1736**, **PR #1729** (@dexhunter, @romeerp): CaseOps lossless case transform lineage.
- **PR #1667** (@MarioPaerle): SmearGate + AttnOutGate.
- **PR #1530**, **PR #1610**, **PR #1626**: VarLen + fused MLP + multi-phase phased TTT.
- **Issue #1017** (@cocohearts): the four conditions that define a meaningful val_bpb.

The diagnostic context (long-context score gating at 2560, no_qv mask, TTT_LOCAL_LR_MULT sweep, QK_GAIN_INIT sweep) was originally measured on exact #1855 in private experiments before this stack. None alone cleared the merge bar on #1855. The contribution here is recognizing that they compose orthogonally on the PR #1945 base.

## Files

- `train_gpt.py`: training and eval script. Verbatim PR #1945 source plus the four literal-constant overrides above.
- `submission.json`: per-seed metadata, BPBs, wallclocks, artifact sizes.
- `train_seed{42,0,1234}.log`: per-seed train and eval logs.
