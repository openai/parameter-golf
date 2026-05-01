# Record: Gated XSA + LQER top-1 + strict token-only n-gram TTT (val_bpb: 1.047)

**val_bpb: 1.04722074** (3-seed mean, std 0.00104816) | **max artifact: 15,996,490 bytes** | 8xH100 SXM | strict in-timer TTT eval

**Improvement vs merged PR #1855 SOTA (1.06107587 BPB):** **-0.01385513 BPB / -0.00960 nats per byte**, clearing the README's 0.005-nats record threshold by about 1.92x.

| Metric | Seed 42 | Seed 1337 | Seed 2026 | 3-seed |
|---|---:|---:|---:|---:|
| Stop step | 4,914 | 4,926 | 4,916 | 4,918.7 mean |
| Train time | 596.127 s | 596.167 s | 596.080 s | 596.125 s mean |
| Pre-quant BPB | 1.04930686 | 1.05124428 | 1.05029930 | 1.05028348 mean |
| Quantized BPB | 1.05773513 | 1.05990331 | 1.05886641 | 1.05883495 mean |
| **Post-TTT BPB** | **1.04616727** | **1.04826351** | **1.04723144** | **1.04722074 mean** |
| Eval time | 471.457 s | 465.480 s | 463.281 s | 466.739 s mean |
| Artifact bytes | 15,995,574 | 15,992,746 | 15,996,490 | 15,996,490 max |

All reported eval time above includes the n-gram hint precompute inside the measured TTT eval timer (`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`).

## Summary

This submission picks up on the PR #1967 / CaseOps lineage and then applies a training-time attention change plus a conservative eval-time n-gram path:

1. **Gated XSA.** Each attention layer gets a learned per-head scalar `xsa_alpha`; the existing XSA subtraction coefficient is multiplied by `tanh(xsa_alpha)`. The gate is zero-initialized, so the model starts as a strict superset of the base stack.
2. **LQER top-1.** `LQER_TOP_K=1` keeps the best LQER correction tensor. This saves artifact bytes versus the top-3 setting and was a favorable knob in the PR #1948 lineage.
3. **Strict token-only n-gram tilt.** In response to the current-token class-routing concern, this update adopts the conservative PR #1514 workaround: disable the within-word and word-level experts and retain only the token-16 expert. The token hint is emitted from `token_context_hash(st)` over prefix state before the current token is pushed into the online state.
4. **In-timer hint precompute.** The n-gram hint pass is included in the final eval timer (`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`). A token-only native fast path keeps the full eval under the 10-minute cap.
5. **Cheaper phased TTT.** The final eval uses one score-first global TTT phase over a 1,000-document prefix, then scores the remaining stream with the adapted global model plus per-document LoRA TTT.

What did not work: Skylight/NorMuon was tested but is disabled in this submission (`SKYLIGHT_MUON=0`) because it destabilized this stack.

## Compliance notes

- **Artifact size:** max artifact is 15,996,490 bytes, under the decimal 16,000,000-byte cap.
- **Training budget:** all three seeds stop on the 600-second wallclock cap at about 596.1 s.
- **Eval budget:** all three token-only final TTT evals are under 600 s. The n-gram hint precompute is included in that timer.
- **Score-first TTT:** the phased TTT path scores validation tokens before using them for global or LoRA updates. The global phase only trains on already-scored prefix documents.
- **Token-only n-gram tilt:** the tilt applies a closed-form renormalized one-token boost, `p'(a) = exp(beta * 1[a=h]) p(a) / Z`, where `Z = 1 + p(h)(exp(beta)-1)`. Hints are generated left-to-right from prefix token state.
- **No within-word or word-level experts:** the final logs show `token_gate=628130 within_gate=0 word_gate=0 agree2plus=0` for every seed.
- **Gate population diagnostic:** `token_only_fast_evals/token_only_gate_population.json` reproduces the production hint pass and reports the same `token_gate=628130`, with `within_gate=0` and `word_gate=0`.
- **Dataset/tokenizer:** uses the CaseOps SP8192 lossless-caps tokenizer and byte-sidecar BPB accounting from the CaseOps lineage. The 80 training shards match the merged CaseOps leader's `prepare_caseops_data.py` default `val_docs=10000` output byte-for-byte. Evaluation uses the full CaseOps validation shard/sidecar reported by the leaderboard lineage (`val_tokens: 47851520`). See `DATASET_AUDIT.md`.

## Key settings

| Setting | Value |
|---|---|
| Base stack | PR #1967 V21 + LeakyReLU 0.3 + n-gram tilt lineage |
| Model | 11 layers, 512 dim, 8 heads / 4 KV heads |
| Tokenizer | SP8192 lossless-caps CaseOps v1 reserved |
| Eval sequence length | 2560 |
| TTT mask | `no_qv` |
| TTT LoRA rank | 80 |
| TTT local LR mult | 0.75 |
| QK gain init | 5.25 |
| Matrix LR | 0.026 |
| Min LR | 0.1 |
| LQER | rank 4, asymmetric, top-1 |
| N-gram precompute | inside timer (`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`) |
| N-gram expert | token-16 only |
| Within/word experts | disabled (`WITHIN_BOOST=0`, `WORD_BOOST=0`) |
| Phased TTT | 1 phase, 1,000 prefix docs |
| Gated XSA | enabled |
| Skylight Muon | disabled |

## Reproducing

Install Python dependencies from `requirements.txt`, install FlashAttention 3 as described there, and install the `lrzip` system package before launching the run. The script itself does not install packages or make network calls during training/evaluation.

```bash
SEED=42 \
NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 VOCAB_SIZE=8192 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
TTT_ENABLED=1 PHASED_TTT_ENABLED=1 PHASED_TTT_NUM_PHASES=1 PHASED_TTT_PREFIX_DOCS=1000 \
TTT_LORA_RANK=80 TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0 \
TTT_LOCAL_LR_MULT=0.75 EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560 \
QK_GAIN_INIT=5.25 \
MATRIX_LR=0.026 MIN_LR=0.1 EMBED_BITS=7 GRAD_CLIP_NORM=0.3 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 \
FUSED_CE_ENABLED=1 SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
SPARSE_ATTN_GATE_ENABLED=1 LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=1 \
LQER_GROUP_SIZE=64 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
AWQ_LITE_ENABLED=1 ASYM_LOGIT_RESCALE=1 NGRAM_TILT_ENABLED=1 \
TOKEN_ORDER=16 TOKEN_THRESHOLD=0.800 TOKEN_BOOST=2.625 \
WITHIN_TAU=999 WITHIN_BOOST=0 WORD_TAU=999 WORD_BOOST=0 AGREE_ADD_BOOST=0 \
GATED_XSA=1 SKYLIGHT_MUON=0 \
GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 \
COMPRESSOR=pergroup \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py` - complete training/eval script.
- `online_ngram_tilt.py`, `online_ngram_state.c` - token-only n-gram hint/tilt helper from the PR #1967 lineage with the conservative fast path.
- `prepare_caseops_data.py`, `lossless_caps.py` - CaseOps dataset preparation helpers.
- `DATASET_AUDIT.md`, `dataset_verification/` - dataset construction audit and verification logs.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - tokenizer model.
- `train_seed42.log`, `train_seed1337.log`, `train_seed2026.log` - original full per-seed training logs for the saved artifacts.
- `token_only_fast_evals/` - eval-only replay logs from the saved artifacts using the conservative token-only n-gram path.
- `submission.json` - structured metadata for the token-only 3-seed result.

## Credits

This work is a small stack on top of a long public lineage:

- PR #1967 by `ndokutovich` for the V21 + LeakyReLU 0.3 + closed-form n-gram tilt stack.
- PR #1953 by `andrewbaggio1` for the long-context/no-QV TTT and QK-gain settings.
- PR #1945 by `alertcat` for the V21/AWQ-lite/asymmetric-logit-rescale base.
- PR #1948 by `TimS-ml` and `lijuncheng16` for the LQER-top-k sweep and LeakyReLU work.
- PR #1514 by `codemath3000` for the conservative token-only n-gram workaround precedent.
- PR #1145 by `AnirudhRahul` for the online n-gram augmentation lineage.
- The CaseOps lineage from `romeerp`, `dexhunter`, `aquariouseworkman`, `codemath3000`, and others for the SP8192 lossless-caps tokenizer, byte-sidecar BPB accounting, and score-first phased TTT.
