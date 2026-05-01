# Source Map

This source map records the public PRs/issues that informed
`README.md`.  It is intentionally a map, not an adjudication of validity.

Snapshot time: 2026-05-01, after PR #2103 was visible.

## Core Validity Threads

| Item | Why it matters |
|---|---|
| Issue #1017 | Community C1-C4 framing: causal dependence, normalized distribution, score-before-update, single pass. |
| Issue #1604 | Custom tokenizer normalization and casefold/CaseOps policy remains unresolved. |
| Issue #43 | Tokenizer artifact accounting discussion. |
| Issue #897 | U+2581 / byte-fallback denominator bug for custom SentencePiece models. |
| Issue #1719 | Leading-space byte double-count bug in `build_sentencepiece_luts`. |
| Issue #1872 | Byte-level PPM-D mixture legality question under C2. |
| Issue #1988 | SmearGate cross-document BOS masking discussion. |
| Issue #2045 | Late suggestion: shell/centering embedding conditioning. |

## Clean Neural / Quantization Lineage

| PR | Public claim summarized |
|---|---|
| #1855 | Merged late SOTA: LQER, SparseAttnGate, BOS-fixed SmearGate, per-group compression, phased TTT, 1.06108 BPB. |
| #1953 | Long-context 2560, no-QV TTT mask, local LR 0.75, QK_GAIN 5.25, 1.05855 BPB. |
| #2014 | Progressive context growth on PR1855/1953 base, 1.05759 BPB. |
| #2018 | Gated XSA + LQER top-1 + strict token-only in-timer n-gram TTT, 1.04722 BPB. |
| #2041 | V21 + inside-timer n-gram TTT without Gated XSA, 1.05692 BPB. |
| #2060 | LongCtx/no-QV/AsymLogit/LQER retune, 1.05792 BPB. |
| #2101 | AWQ-lite + AsymLogit + GradCentral + LabelSmooth, 1.05845 BPB. |

## PPM / Byte-Mixer Lineage

| PR | Public claim summarized |
|---|---|
| #1991 | SP8192 byte-PPM mixer with tuned order/gate, 0.94290 BPB. |
| #2039 | Conditional byte-level PPM mixer with first-byte marginalization argument, 1.027004 BPB. |
| #2083 | SP8192 CaseOps v13 PPM tuned gate, 0.94175 BPB. |
| #2098 | PR #1873 base + tuned PPM gate, 0.80051 BPB. |
| #2103 | SP1024 Value Residual + PPM mixture, single-H100 note, 0.829467 ppm_mix_bpb. |

## Validation-Adaptation / PreQuantTTT Lineage

| PR | Public claim summarized |
|---|---|
| #1958 | PreQuantTTT lineage, later withdrawn/disputed in community discussion. |
| #1972 | SimCTG + PreQuantTTT with 21 full-pass AdamW epochs on val tokens, 1.03983 BPB claim. |

## Non-record / Methodology Context

| PR | Public claim summarized |
|---|---|
| #2011 | Cross-base regularizer transferability study. |
| #2046 | Negative results compendium. |
| #2088 | Causal bigram blending eval-time improvement on limited hardware. |
| #2102 | MoE upcycling and depth recurrence quantization-gap analysis. |

## My Companion Evidence Package

`../2026-05-01_LastDay_Frontier_Transfer_Autopsy/` contains:

- final-day 8xH100 logs,
- exact runner/patch scripts,
- pre-quant kill gates,
- CrossWS train-proxy tokenizer evidence,
- corrected Memento/copy-memory no-go,
- context-horizon and artifact-saving summaries.
