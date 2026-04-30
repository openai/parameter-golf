# SP8192 V21 + Inside-timer N-gram TTT (no Gated XSA) — val_bpb 1.05692

**Score: 1.05692 BPB** (3-seed mean, full val partition, seeds 42 / 0 / 1234)

| Seed | val_bpb | val_loss | train wallclock | eval | artifact |
|------|--------:|---------:|----------------:|-----:|---------:|
| 42   | 1.05610 | 2.31114 | 596.058s | 592.2s | 15,977,032 B |
| 0    | 1.05736 | 2.31390 | 596.017s | 565.2s | 15,975,966 B |
| 1234 | 1.05730 | 2.31377 | 596.094s | 518.8s | 15,972,820 B |
| **mean** | **1.05692** | **2.31294** | **596.06s** | **558.7s** | **15,975,273 B** |
| std (pop) | 0.000580 | — | — | — | — |

`val_tokens: 47,851,520` on every seed (full validation partition, identical to PR #1855 reference).

vs current merged SOTA (PR #1855 1.06108): **−0.00416 BPB ≈ 0.0091 nats**, clears the 0.005-nat README threshold.

## Approach

This submission stacks the **eval-time recipe from PR #2018** (simon-marcus, val_bpb 1.04616) on top of the **PR #1967 V21 + LeakyReLU 0.3 training base** (ndokutovich), without Gated XSA:

- **PR #2018's Phased TTT shape** (`PHASED_TTT_NUM_PHASES=1`, `PHASED_TTT_PREFIX_DOCS=1000`) — replaces 3-phase / 2500-prefix with one cheaper score-first phase. Frees eval budget.
- **PR #2018's N-gram tilt INSIDE the eval timer** (`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`). Per-position hint precompute (~150-160s) is part of the 600s eval budget, not setup. Same accounting as the merged A2 record (`records/track_10min_16mb/2026-04-09_A2_Muon097_3Seed/README.md` line 106: *"Eval under 600s on all 3 seeds (~436-442 s actual: ~8 s roundtrip + ~92 s sliding + ~33 s n-gram precompute + ~330-342 s TTT)"*).
- **PR #2018's LQER top-1** (`LQER_TOP_K=1`) — saves artifact bytes vs top-3.
- **GPTQ_RESERVE_SECONDS=4.0** (PR #2018) — 4 seconds reserved before training stop for GPTQ.
- **PR #1967 V21 base + LeakyReLU 0.3** (ndokutovich) — kept verbatim; preserves `WARMDOWN_FRAC=0.85`, `BETA2=0.99`, `SPARSE_ATTN_GATE_SCALE=0.5` from #1967.
- **PR #1923 AsymLogit Rescale + AWQ-lite** — already in V21 (PR #1908 + PR #1945 lineage).

**Differences from PR #2018 (which scored 1.04616):**
- This submission **does not include Gated XSA** (PR #2018's training-time per-head attention scalar). The ~0.010 BPB gap to #2018 is the Gated XSA contribution; pre-quant post-EMA matches: this run reports 1.06117 (seed 42) vs PR #2018's 1.04930 (seed 42), a difference of 0.012 BPB.
- This submission uses PR #1967's `WARMDOWN_FRAC=0.85`, `BETA2=0.99`, `SPARSE_ATTN_GATE_SCALE=0.5` overrides; PR #2018 uses defaults (0.75 / 0.95 / 1.0).

We submit this as additional 3-seed evidence that the inside-timer N-gram + cheap-Phased-TTT eval recipe is reproducible across V21 base variants without Gated XSA.

## Compliance (Issue #1017)

- **C1 strict causal dependence:** standard varlen + per-doc `cu_seqlens`; no future-token leakage.
- **C2 full normalized distribution:** standard log-softmax over SP8192 vocab; n-gram tilt is the closed-form `p'(a) = exp(β · 1[a=h]) · p(a) / Z` with `Z = 1 + p(h)(exp(β)-1)`, Σ p'(a) = 1. AsymLogit is a deterministic monotone reshape before softmax.
- **C3 score-before-update:** Phased TTT scores each chunk before any LoRA gradient step; n-gram hints are generated left-to-right from prefix state only.
- **C4 single pass:** each val token contributes exactly one BPB term in `quantized_ttt_phased`.
- **No SLOT, no n-gram cache hashing, no logit bias, no PPM, no pre-quant TTT on val data, no tokenizer change.**
- **Compute caps:** train ≤596.094s (max), eval ≤592.2s (max), all 3 seeds. `MAX_WALLCLOCK_SECONDS=600`.
- **Artifact:** ≤15,977,032 bytes (max). Cap is 16,000,000.

## Reproduction

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
apt-get install -y lrzip build-essential

# online_ngram_state.c is auto-compiled by online_ngram_tilt.py on first import
# Set DATA_DIR and TOKENIZER_PATH appropriately or use the defaults in run.sh

for seed in 42 0 1234; do
  SEED=$seed bash run.sh 2>&1 | tee train_seed${seed}.log
done
```

## Hyperparameters (additions over PR #1967)

```bash
# === PR #2018 eval-time recipe additions ===
PHASED_TTT_NUM_PHASES=1           # vs #1967's 3
PHASED_TTT_PREFIX_DOCS=1000       # vs #1967's 2500
NGRAM_HINT_PRECOMPUTE_OUTSIDE=0   # INSIDE timer (vs default outside)
LQER_TOP_K=1                      # vs default top-K=3
GPTQ_RESERVE_SECONDS=4.0          # vs default 0.5

# === Inherited from PR #1967 V21 ===
WARMDOWN_FRAC=0.85
BETA2=0.99
SPARSE_ATTN_GATE_SCALE=0.5
TTT_LR=0.75
QK_GAIN_INIT=5.25
TTT_NO_QV_MASK=1
TTT_LORA_RANK=80
EVAL_SEQ_LEN=2560
TTT_EVAL_SEQ_LEN=2560
NGRAM_TILT_ENABLED=1
ASYM_LOGIT_RESCALE=1
AWQ_LITE_ENABLED=1
LeakyReLU squared slope = 0.3 (hardcoded in train_gpt.py)
```

See `run.sh` for the full env list.

## Hardware

8×H100 SXM 80GB (RunPod CA-MTL-1, $23.95/hr), PyTorch 2.9.1+cu128, FlashAttention 3, NVIDIA Driver 580.126.09, CUDA 13.0.

## Attribution

- **simon-marcus** — PR #2018 (inside-timer N-gram + 1-phase TTT + 1000-prefix + LQER_TOP_K=1 eval recipe, GPTQ_RESERVE_SECONDS=4.0)
- **ndokutovich** — PR #1967 (V21 + LeakyReLU 0.3 + n-gram tilt code base; `train_gpt.py` used here)
- **AnirudhRahul** — PR #1145 (closed-form n-gram tilt with Σ P=1 renormalization)
- **TimS-ml, lijuncheng16** — PR #1948 (LeakyReLU squared slope 0.3)
- **alertcat** — PR #1945 (V21 base composition)
- **andrewbaggio1** — PR #1953 (7-knob TTT/QK tuning)
- **romeerp** — PR #1908 (AWQ-lite mixed-precision GPTQ) + PR #1729 (CaseOps tokenizer)
- **classiclarryd** — modded-nanogpt #181 (Asymmetric Logit Rescale)
- **codemath3000** — PR #1855 (9-hparam greedy stack base)
- **dexhunter** — PR #1797 (LQER asym + SmearGate base)
- **nprime06** — PR #1787 (Polar Express NS + MIN_LR + Sparse Attn Gate + Fused CE)
- **MarioPaerle** — PR #1667 (SmearGate origin)
- **renqianluo** — PR #1767 (LoRA TTT improvements)
- **Jorge Asenjo** — PR #1700 (Phased Multi-Phase Global SGD TTT, foundational); PR #1923 (AsymLogit + AWQ-lite stack); this PR (inside-timer recipe transferred to PR #1967 base)
