# Record: Gated XSA + LQER top-1 + strict in-timer n-gram TTT (val_bpb: 1.046)

**val_bpb: 1.04616683** (3-seed mean, std 0.00105608) | **max artifact: 15,996,490 bytes** | 8xH100 SXM | strict in-timer TTT eval

**Improvement vs merged PR #1855 SOTA (1.06107587 BPB):** **-0.01490904 BPB / -0.01033 nats per byte**, clearing the README's 0.005-nats record threshold by about 2.07x.

| Metric | Seed 42 | Seed 1337 | Seed 2026 | 3-seed |
|---|---:|---:|---:|---:|
| Stop step | 4,914 | 4,926 | 4,916 | 4,918.7 mean |
| Train time | 596.127 s | 596.167 s | 596.080 s | 596.125 s mean |
| Pre-quant BPB | 1.04930686 | 1.05124428 | 1.05029930 | 1.05028348 mean |
| Quantized BPB | 1.05773513 | 1.05990331 | 1.05886641 | 1.05883495 mean |
| **Post-TTT BPB** | **1.04510781** | **1.04721994** | **1.04617273** | **1.04616683 mean** |
| Eval time | 542.399 s | 548.364 s | 546.342 s | 545.702 s mean |
| Artifact bytes | 15,995,574 | 15,992,746 | 15,996,490 | 15,996,490 max |

All reported eval time above includes the n-gram hint precompute inside the measured TTT eval timer (`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`).

## Summary

This submission picks up on some of the work started in PR #1967. After conducting various TTT optimization sweeps, the final stack is a V21 + n-gram tilt stack that adds a training-time attention change:

1. **Gated XSA.** Each attention layer gets a learned per-head scalar `xsa_alpha`; the existing XSA subtraction coefficient is multiplied by `tanh(xsa_alpha)`. The gate is zero-initialized, so the model starts as a strict superset of the base stack.
2. **LQER top-1.** `LQER_TOP_K=1` keeps the best LQER correction tensor. This saves artifact bytes versus the top-3 setting and was a favorable knob in the PR #1948 lineage.
3. **Strict in-timer n-gram tilt.** The closed-form normalized n-gram tilt from the PR #1145 / PR #1967 lineage is kept, but the hint precompute is measured inside the final eval timer (`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`), so the eval time stays strictly under the 10-minute cap.
4. **Cheaper phased TTT.** The final eval uses one score-first global TTT phase over a 1,000-document prefix, then scores the remaining stream with the adapted global model plus per-document LoRA TTT.

What didn't work:Skylight/NorMuon was tested but is disabled in this submission (`SKYLIGHT_MUON=0`) because it destabilized this stack.

## Compliance notes

- **Artifact size:** seed 42 is 15,995,574 bytes, under the decimal 16,000,000-byte cap.
- **Training budget:** seed 42 stops on the 600-second wallclock cap at 596.127 s.
- **Eval budget:** seed 42 final TTT eval is 542.399 s, under the 600-second eval cap. The n-gram hint precompute is included in that timer.
- **Score-first TTT:** the phased TTT path scores validation tokens before using them for global or LoRA updates. The global phase only trains on already-scored prefix documents.
- **N-gram tilt:** the tilt applies a closed-form renormalized one-token boost, `p'(a) = exp(beta * 1[a=h]) p(a) / Z`, where `Z = 1 + p(h)(exp(beta)-1)`. Hints are generated left-to-right from prefix state.
- **Dataset/tokenizer:** uses the CaseOps SP8192 lossless-caps tokenizer and byte sidecar setup from the CaseOps lineage. `prepare_caseops_data.py` and `lossless_caps.py` are included for reproducibility.

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
GATED_XSA=1 SKYLIGHT_MUON=0 \
GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 \
COMPRESSOR=pergroup \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py` - complete training/eval script.
- `online_ngram_tilt.py`, `online_ngram_state.c` - n-gram hint/tilt helper from the PR #1967 lineage.
- `prepare_caseops_data.py`, `lossless_caps.py` - CaseOps dataset preparation helpers.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - tokenizer model.
- `train_seed42.log`, `train_seed1337.log`, `train_seed2026.log` - full per-seed logs.
- `submission.json` - structured metadata for the 3-seed run.

## Credits

This work is a small stack on top of a long public lineage:

- PR #1967 by `ndokutovich` for the V21 + LeakyReLU 0.3 + closed-form n-gram tilt stack.
- PR #1953 by `andrewbaggio1` for the long-context/no-QV TTT and QK-gain settings.
- PR #1945 by `alertcat` for the V21/AWQ-lite/asymmetric-logit-rescale base.
- PR #1948 by `TimS-ml` and `lijuncheng16` for the LQER-top-k sweep and LeakyReLU work.
- PR #1145 by `AnirudhRahul` for the online n-gram augmentation lineage.
- The CaseOps lineage from `romeerp`, `dexhunter`, `aquariouseworkman`, `codemath3000`, and others for the SP8192 lossless-caps tokenizer, byte-sidecar BPB accounting, and score-first phased TTT.
