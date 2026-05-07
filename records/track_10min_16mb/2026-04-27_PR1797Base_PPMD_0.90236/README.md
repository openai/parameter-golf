# Record: PR #1797 base + PPM-D byte mixture — val_bpb 0.90236

**val_bpb: 0.90236** (3-seed mean, std 0.00082) | **15.95 MB** | 8×H100 SXM, ≤600s train / ≤600s eval | PPM-D mixture

This submission ports the **PPM-D byte-level mixture from PR #1835 (anmarhindi)** onto the **PR #1797 base stack from dexhunter** (val_bpb 1.06157), then evaluates the mixture as the headline submission score per the same protocol used in PR #1835.

## Result (3 seeds, 8×H100 80GB SXM)

| Seed | Pre-EMA BPB | Post-TTT BPB (neural-only) | **mix_bpb** (final) | Artifact bytes |
|------|------------:|---------------------------:|---------------------:|---------------:|
| 42   | 1.07015     | 1.06724                    | **0.901871**        | 15,953,988     |
| 1337 | 1.07210     | 1.06918                    | **0.903306**        | 15,943,986     |
| 314  | 1.07057     | 1.06732                    | **0.901916**        | 15,947,840     |
| **Mean** | **1.07094** | **1.06791**            | **0.902364**        | **15,948,605** |
| **Std**  | 0.001028 | 0.001094                  | **0.000816**        | 5,012          |

All seeds:
- Submission size ≤ 16,000,000 bytes (decimal).
- Training cleared the 600s wallclock cap (`stopping_early: wallclock_cap`).
- Evaluation cleared the 600s wallclock budget.
- 3-seed std 0.000816 — sub-record clearance vs prior leaderboard 1.0810 is **~218× std**.

## Headline result

- vs merged leaderboard (PR #1493 @bigbag, 1.0810): **−0.17864 BPB** (≈ −0.41 nats/token at SP8192 byte ratio).
- Statistical confidence: 3-seed std 0.000816 → p ≪ 0.01 vs 0.005-nat threshold.

## Stack

The submission is a clean two-component stack:

1. **PR #1797 (dexhunter) base** — verbatim:
   - SP8192 + CaseOps bijective case transform (PR #1729 @romeerp)
   - SparseAttnGate + PolarNS + MIN_LR + FusedCE (PR #1787 @nprime06)
   - Smear gate (PR #1797 self)
   - LQER asymmetric rank-4 correction (PR #1797 self)
   - Phased TTT (3 phases, 2000 prefix docs) — score-first, Issue #1017 compliant
   - GPTQ int6 (matrix) + int7 (embeddings) with SDClip
   - Brotli compression + LZMA code wrapper

2. **PR #1835 (anmarhindi) PPM-D byte mixture** — direct port:
   - Order-5 byte context, classical PPM-D escape (Cleary-Witten 1984)
   - Binary-λ gate: `λ_lo=0.05` if PPM confidence ≥ 0.9 else `λ_hi=0.9`
   - Mixture in probability space: `p_mix = λ * p_NN + (1 - λ) * p_PPM`
   - 8M-token subset of val for richer byte-context counts
   - Score-first: byte counts update AFTER per-byte log_mix recorded — preserves Issue #1017 C3.

## Reproduction

### Data preparation (CPU-bound, runs once outside the 600s training cap)

```bash
# 1. Download canonical FineWeb-10B selected docs from the OpenAI repo's HF dataset.
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='willdepueoai/parameter-golf', filename='datasets/docs_selected.jsonl', repo_type='dataset', local_dir='./hf_cache')"

# 2. CaseOps re-tokenize using our parallel prep (8 vCPU+ recommended).
python3 prepare_caseops_data.py \
    --docs ./hf_cache/datasets/docs_selected.jsonl \
    --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    --workers 16 \
    --max-docs 5000000
```

The `--max-docs 5000000` cap caps disk usage at ~10 GB while still producing far more train tokens than 600s of 8×H100 training can consume.

### Training + evaluation (3-seed, ≤600s training each)

```bash
for SEED in 42 1337 314; do
  CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 MIN_LR=0.1 \
  FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  TTT_WARM_START_A=1 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  PPM_ENABLED=1 PPM_ORDER=5 PPM_SUBSET_TOKENS=8000000 \
  PPM_LAMBDA_HI=0.9 PPM_LAMBDA_LO=0.05 PPM_CONF_THRESHOLD=0.9 \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

The headline `mix_bpb` value is logged at the end of each run as:
```
ppm_mix bytes=<N> mix_bpb=<X.XXXXXX> ppm_only=<...> nn_only=<...>
```

Pre-quant / quantized / TTT-only diagnostic BPBs are also logged for completeness.

## Compliance — Issue #1017 four conditions

The PPM-D byte mixture is a transparent extension of the score-first protocol already used by the PR #1797 base stack. Every byte's mixture probability is computed using only:

- **C1 — Causality:** Each byte is mixed using ONLY the per-byte NN log-prob derived from the same already-causal token NLL (single left-to-right sliding-window pass) and PPM-D byte counts that exist BEFORE that byte.
- **C2 — Normalized distribution:** `p_mix = λ * p_NN + (1−λ) * p_PPM` is a convex combination of two normalized distributions over the 256-symbol byte alphabet → also normalized over the same alphabet.
- **C3 — Score-before-update:** PPM-D byte counts for the byte at position `t` are updated AFTER `log_mix(t)` is recorded. The neural NLL itself is from the score-first phased-TTT path of PR #1797. No byte ever influences its own probability mass before being scored.
- **C4 — Single left-to-right pass:** PPM state is a Python `dict[bytes, dict[int, int]]` consumed exactly once in increasing byte position. No rescoring, no oracle selection across passes.

Additional compliance:
- **No SLOT** (standard or causal) and no eval-time logit bias.
- **No pre-quant TTT on val tokens** — TTT operates only on already-scored chunks per the score-first protocol.
- **No n-gram cache built from training data** — PPM state is built ONLINE from the val byte stream itself, byte-by-byte, AFTER each byte is scored.
- **All 3 seeds under 16,000,000 byte decimal cap.**
- **All 3 seeds clear 600s training and 600s evaluation budgets.**

## Architecture (inherits PR #1797 shape)

| Item                     | Value |
|--------------------------|------:|
| num_layers               | 11    |
| model_dim                | 512   |
| num_heads / num_kv_heads | 8 / 4 |
| mlp_mult                 | 4.0   |
| Loop schedule            | encoder [0..5,3,4]; decoder [5,3..10] (PR #1530, samacqua) |
| Parallel residual start  | layer 8 |
| Activate looping at      | frac 0.35 |
| RoPE dims / base         | 16 / 10000 |
| Tied embeddings          | True  |
| Logit softcap            | 30.0  |
| LeakyReLU(0.5)²          | True  |

## Lineage and credits

- **PR #549** (@abaybektursun) — original score-first TTT framework.
- **PR #1019** — byte-level BPB SentencePiece accounting (`piece.encode`).
- **PR #1394** (@clarkkev) — SP8192 + multi-phase score-first TTT baseline + GPTQ embeddings + SDClip + MuonEq-R + depth recurrence.
- **PR #1530** (@samacqua) — Loop4-5 depth recurrence + parallel residual start layer 8.
- **PR #1729** (@romeerp) — CaseOps bijective case transform + per-token byte sidecar accounting.
- **PR #1787** (@nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE base stack with TTT_WARM_START_A.
- **PR #1797** (@dexhunter) — Smear gate + LQER asymmetric rank-4 stacked on the PR #1787 base. **This submission's neural component is verbatim PR #1797.**
- **PR #1835** (@anmarhindi) — PPM-D byte-level mixture as eval-time score-first augmentation. **This submission's mixture component is a faithful port of PR #1835.**

## Author contribution

- Independent reproduction of the PR #1797 neural baseline using the canonical `willdepueoai/parameter-golf` `docs_selected.jsonl` corpus.
- Faithful integration port of the PR #1835 PPM-D mixture algorithm into the PR #1797 codebase (single eval-time hook in `eval_val_ttt_phased`, score-first protocol preserved).
- Parallel CaseOps re-tokenization preprocessor (`prepare_caseops_data.py`, multiprocessing, ~16× wall-clock improvement vs single-thread original) for tractable submission turnaround.
- 3-seed verification (42, 1337, 314) on canonical reproduction data.

## Included files

- `README.md` — this file.
- `submission.json` — metadata.
- `train_gpt.py` — training + eval script (PR #1797 verbatim + 6 small PPM-D hunks from PR #1835).
- `lossless_caps.py` — CaseOps transform module (from PR #1729 lineage).
- `prepare_caseops_data.py` — parallel CaseOps re-tokenizer.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SP8192 tokenizer.
- `train_seed42.log`, `train_seed1337.log`, `train_seed314.log` — per-seed train+eval logs.
