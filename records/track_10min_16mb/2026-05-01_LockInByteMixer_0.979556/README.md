# Record: Lock-In Byte Mixer (LBM) — corrected val_bpb **1.067219**

> **Originally posted as 0.979556. That headline was wrong** because the
> cond-PPM mixer divided total mix-NLL by SP-piece UTF-8 bytes (which
> include the 3-byte CaseOps sentinels U+E001..U+E004) instead of the
> canonical raw-text byte sidecar that every other CaseOps-lineage record
> charges BPB against (PR #1729 et seq). The corrected mean on the
> canonical denominator is **1.067219** (3-seed mean, std ≈ 7.6e-05). See
> the Errata section below for the algebra and source-line citations.

---

## Errata (2026-05-02)

Reported by @codemath3000 on PR #2138; thank you. The correction is
algebraic — same artifact, same per-token NLL, same logs — only the BPB
denominator changes. No re-eval was required.

### The bug

In the wrapped `train_gpt.py`'s decoded `_cond_ppm_mixture_jit` (lines
~3489 / ~3572 / ~3755 in the decoded source) and its pure-Python twin
`_cond_ppm_mixture_bpb` (lines ~3910 / ~3969), the BPB denominator
accumulates `n_token_bytes = len(piece.encode("utf-8"))` per token
(plus 1 for include_space). For the CaseOps-V1 SP8192 vocabulary that
includes the 3-byte UTF-8 encodings of the U+E001..U+E004 capitalization
sentinels, which inflate the total by 8.95 % vs the canonical raw-text
byte stream.

The sliding-window denominator (lines ~3265–3271 of the decoded source)
is correct: it pulls per-token byte counts from `val_data.val_bytes`
(`load_validation_byte_sidecar` at line ~643), which is the canonical
raw-text sidecar `fineweb_val_bytes_*.bin`. PR #1729's authoritative
metric definition: *"BPB is still charged against the original raw UTF-8
bytes through the exported validation byte sidecar, not against
transformed text length."* The cond-PPM mixer is the only point in the
record's code path where that convention is bypassed.

Per-seed forensic evidence in this record's `train_seed{42,1337,314}.log`:

```
cond_ppm tokens=47851520 bytes=164594398 cond_mix_bpb=0.979491 …   # buggy denominator (SP-piece bytes)
quantized_sliding_window val_loss:2.40982833 val_bpb:1.10119889    # correct denominator (sidecar)
```

The canonical sidecar count (151,074,309) is reverse-solvable from any
seed's sliding-window line:

```
total_NLL  = val_loss × tokens = 2.40982833 × 47851520 ≈ 115,313,958 nats
sidecar    = total_NLL / log(2) / 1.10119889 = 151,074,309 bytes
inflation  = 164,594,398 / 151,074,309 = 1.089493  (+8.95 %)
```

### The corrected numbers (algebraic)

Per-token NLL is invariant under denominator change (same artifact, same
forward pass, same PPM mixer state — none depend on the divisor). So the
corrected BPB per seed is exactly:

```
new_bpb[s]  = old_bpb[s] × 164594398 / 151074309
            = old_bpb[s] × 1.089493
```

| Seed | val_loss (nats/tok) | sliding-window BPB (canonical) | reported cond-PPM BPB (buggy denom) | **corrected cond-PPM BPB (canonical)** |
|------|---------------------:|-------------------------------:|------------------------------------:|---------------------------------------:|
| 42   | 2.33531315 | 1.10119889 | 0.97949078 | **1.067148** |
| 1337 | 2.33544783 | 1.10141497 | 0.97954725 | **1.067210** |
| 314  | 2.33564246 | 1.10142762 | 0.97962885 | **1.067299** |
| **Mean** | — | **1.101347** | 0.979556 (originally reported) | **1.067219** |
| **Std**  | — | 0.000129 | 0.000069 | ≈ 0.000076 |

### Where this leaves the submission

- **vs current SOTA (PR #1855, 1.06108):** the corrected mean is
  **+0.006 BPB worse**, so this submission is **not a SOTA result on the
  canonical denominator**. The original "−0.082 vs SOTA" claim was
  entirely the inflation ratio, not a real compression improvement.
- **vs sliding-window alone (1.101347):** LBM still gives a real
  **−0.034 BPB** improvement on the canonical denominator. The mixer
  technique itself — high-confidence sigmoid lock-in to PPM-D — is
  legitimate, just much smaller in magnitude than originally reported.
- **The C2 correctness story is unchanged.** Both mix steps remain
  convex combinations of two proper distributions over the same alphabet
  (byte alphabet for byte_0; joint byte-sequence alphabet for the
  remainder). The bug was *only* in the BPB-headline divisor, not in the
  mixer math.

### What changed in this record

This errata patches:

- `README.md` — adds Errata section, corrected 3-seed table, repositions
  the writeup as not-SOTA.
- `submission.json` — corrects `val_bpb`, `val_bpb_per_seed`, std,
  `eval_canonical_byte_count_per_seed` (164594398 → 151074309), and
  `headline_metric_description`. Adds `errata` field.
- `run.sh` — header comment notes corrected headline.

The following are **kept as-is** because they are forensic records of
what actually ran:

- `final_model.int6.ptz` — the original quantized artifact.
- `train_gpt.py` — the original wrapped source (still contains the
  buggy denominator). Decode and grep `total_canonical_bytes += n_bytes`
  to find both occurrences.
- `train_seed{42,1337,314}.log` — original eval logs. The buggy
  `cond_ppm bytes=164594398 / val_bpb=0.979…` lines remain visible
  alongside the canonical-correct `quantized_sliding_window` lines.

The fix lives on branch `cond-ppm-stack` of
`https://github.com/anmarhindi/parameter-golf-a` (commits to be tagged
once verified): the mixer now takes a per-token sidecar array and
accumulates that as the BPB denominator, falling back to SP-piece bytes
only when no sidecar is available (non-CaseOps callers).

---

## Original technique writeup (denominator-correction applied)

The **Lock-In Byte Mixer (LBM)** is a high-confidence-only byte-level
mixture of the model's softmax with a PPM-D byte-conditional state. The
mixer's sigmoid gate

```
λ = 1 - sigmoid(α · (PPM_conf - β))     with α=25, β=0.9999, order=5
```

defaults to the neural model on essentially every byte position; PPM
only "locks in" — entirely substituting for the NN — at byte positions
where its context confidence exceeds 0.9999. In practice these are
positions where the language is locally near-deterministic given recent
bytes (frequent function-word continuations, common bigram/trigram
prefixes, repeated tokens), and PPM nails them at ~zero NLL while the
NN still pays a few nats. The α=25 sharp transition keeps the
mid-confidence regime out of the mix entirely.

This is the **C2-correct member of the PPM-mix family** (PR #1835).
Where #1835's byte-level mix used `P_NN(byte) = exp(token_logp / n_bytes)`
— a per-byte value that does not sum to 1 over the byte alphabet — LBM
derives `P_NN(byte_0)` from the model's full softmax via the canonical
first-byte mask:

```
P_NN(byte_0 = b) = Σ_{T : first_byte(T) = b} P_NN(T)
```

The remainder bytes mix at the joint-byte-sequence alphabet via the NN's
chain-rule residual `P_NN_rem = P_NN(token) / P_NN(byte_0)` and the
PPM-D byte chain. **Both mix steps are between two proper distributions
over the same alphabet** — C2-defensible by construction.

The mixer's hot loop is JIT-compiled with `numba` (added to
`requirements.txt`) so the eval pass over the canonical 50K-doc
validation set (~48M scored tokens) completes inside the 600s eval cap.

**Data hygiene:** trained on canonical FineWeb docs **50,000+** — fully
disjoint from the canonical 50K-doc validation set. The default in
`prepare_caseops_data.py` was patched from the historic
`--val-docs=10000` to `--val-docs=50000` to prevent the train/val
overlap flagged by the CaseOps memory-leakage audit.

Stacked on:
* PR #1493 sliding-window eval at stride=128 (each val token scored
  from up to `seq_len-stride` tokens of strict-past context;
  stride=128 is necessary to fit the 5×-larger canonical val under
  the 600s eval cap).
* PR #1797 (@dexhunter) base + SmearGate BOS-mask fix (cocohearts review).
* `PARALLEL_START_LAYER=5`, `ENABLE_LOOPING_AT=0.85`,
  `WARMDOWN_FRAC=0.85`.

## Eval-set coverage

The cond-PPM (LBM) runs over the **full canonical CaseOps validation
set** for every seed:

* **47,851,520 tokens / 151,074,309 canonical raw-text bytes per seed.**
* The 164,594,398 figure that appears in each `train_seed*.log`'s
  `cond_ppm tokens=… bytes=…` line is the **SP-piece** byte total
  (canonical bytes + CaseOps sentinel overhead) — see Errata above.
* `cppm_chunks` are all-gathered across all 8 ranks before the
  byte-level PPM-D state is advanced on rank 0 in canonical val order
  — full-val PPM context, not per-rank.

This addresses the eval-set-coverage concern raised on PR #1835.

## Compliance — Issue #1017

* **C1 (causality):** sliding-window scoring strict-past only; cond-PPM
  byte state advances ONLY after each byte's mix log-prob is recorded.
  Marginalization at byte_0 is derived from the position's softmax,
  which sees only strict past. Mix gate weights depend on PPM context
  confidence ONLY (not the realized byte).
* **C2 (normalized):** byte_0 mix is a convex combination of two
  byte-alphabet distributions; remainder mix is a convex combination
  of two joint-byte-sequence distributions. Product is a proper
  distribution over the realized token's byte stream.
* **C3 (score-first):** both NN softmax and PPM byte conditional commit
  before observing the realized byte at each step. PPM state advances
  post-scoring.
* **C4 (single L→R pass):** each val token contributes exactly one BPB
  term.

No SLOT, no n-gram cache outside the legal byte-level PPM-D state, no
logit bias, no ETLB, no pre-quant TTT (which is C3-violating). Standard
softmax over the SP8192 alphabet at every scored position.

## Lineage

PR #1394 (clarkkev) → PR #1530 (samacqua) → PR #1729 (romeerp CaseOps)
→ PR #1787 (nprime06) → PR #1797 (dexhunter, SmearGate fixed)
→ PR #1493 sliding-window
→ PR #1835 PPM byte-mix (this submission's C2-correct successor)

## Reproduction

```bash
git clone https://github.com/anmarhindi/parameter-golf-a
cd parameter-golf-a
git checkout cond-ppm-stack
cd obliterate_0p9
bash run.sh                  # 3 seeds × (≤600s train + ≤600s eval)
bash build_submissions.sh    # produces this submission folder + tarball
```

`run.sh` invokes `_pod_setup.sh` to install deps (brotli, sentencepiece,
flash_attn_3, **numba** for the LBM hot loop) and downloads
`docs_selected.jsonl` from `willdepueoai/parameter-golf`, tokenizes via
`prepare_caseops_data.py` with `--val-docs 50000`, then trains + evals
each seed.

A re-run on `cond-ppm-stack` (post-fix) will log
`cond_ppm tokens=47851520 bytes=151074309 cond_mix_bpb=1.067148 …
sidecar=on` directly, matching the algebraically-corrected number above.

To inspect the readable source of the wrapped `train_gpt.py`:

```python
import lzma, base64, re
src = open("train_gpt.py").read()
blob = re.search(r'b85decode\("([^"]+)"\)', src).group(1)
print(lzma.decompress(base64.b85decode(blob)).decode())
```

## Files

| file | purpose |
|---|---|
| `README.md`                                | this record + Errata |
| `submission.json`                          | metadata (corrected) + `errata` field |
| `train_gpt.py`                             | original wrapped source (lzma+base85). Contains the buggy denominator; preserved as forensic record. |
| `final_model.int6.ptz`                     | original quantized + brotli artifact. Unchanged. |
| `lossless_caps.py`                         | CaseOps transform module (PR #1729 lineage) |
| `prepare_caseops_data.py`                  | parallel CaseOps tokenizer (data prep helper) |
| `tokenizers/...caseops_v1_reserved.model`  | SP8192 CaseOps tokenizer |
| `train_seed{42,1337,314}.log`              | original per-seed train + eval logs (each contains both the buggy `cond_ppm bytes=164594398` line and the canonical-correct `quantized_sliding_window val_bpb` line) |
| `requirements.txt`                         | unchanged |
| `run.sh`                                   | reproduction launcher (header comment notes errata) |
