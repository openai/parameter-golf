# Record: Lock-In Byte Mixer — val_bpb 0.979556

**val_bpb: 0.979556** (3-seed mean, std 0.000069) | 15.08 MB | 8×H100 SXM, ≤600s train / ≤600s eval

The **Lock-In Byte Mixer (LBM)** is a high-confidence-only byte-level mixture
of the model's softmax with a PPM-D byte-conditional state. The mixer's
sigmoid gate

```
λ = 1 - sigmoid(α · (PPM_conf - β))     with α=25, β=0.9999, order=5
```

defaults to the neural model on essentially every byte position; PPM only
"locks in" — entirely substituting for the NN — at byte positions where its
context confidence exceeds 0.9999. In practice these are the byte positions
where the language is locally near-deterministic given recent bytes (frequent
function-word continuations, common bigram/trigram prefixes, repeated tokens),
and PPM nails them at ~zero NLL while the NN still pays a few nats. The α=25
sharp transition keeps the mid-confidence regime out of the mix entirely.

This is the **C2-correct member of the PPM-mix family** (PR #1835). Where
#1835's byte-level mix used `P_NN(byte) = exp(token_logp / n_bytes)` — a
per-byte value that does not sum to 1 over the byte alphabet — LBM derives
`P_NN(byte_0)` from the model's full softmax via the canonical first-byte
mask:

```
P_NN(byte_0 = b) = Σ_{T : first_byte(T) = b} P_NN(T)
```

The remainder bytes mix at the joint-byte-sequence alphabet via the NN's
chain-rule residual `P_NN_rem = P_NN(token) / P_NN(byte_0)` and the PPM-D
byte chain. **Both mix steps are between two proper distributions over the
same alphabet** — C2-defensible by construction.

The mixer's hot loop is JIT-compiled with `numba` (added to
`requirements.txt`) so the eval pass over the canonical 50K-doc validation
set (~48M scored tokens) completes inside the 600s eval cap.

**Data hygiene:** trained on canonical FineWeb docs **50,000+** — fully
disjoint from the canonical 50K-doc validation set. The default in
`prepare_caseops_data.py` was patched from the historic `--val-docs=10000`
to `--val-docs=50000` to prevent the train/val overlap flagged by the
CaseOps memory-leakage audit.

Stacked on:
* PR #1493 sliding-window eval at stride=128 (each val token scored from up
  to `seq_len-stride` tokens of strict-past context; stride=128 is necessary
  to fit the 5×-larger canonical val under the 600s eval cap).
* PR #1797 (@dexhunter) base + SmearGate BOS-mask fix (cocohearts review).
* `PARALLEL_START_LAYER=5`, `ENABLE_LOOPING_AT=0.85`, `WARMDOWN_FRAC=0.85`.

## 3-seed result (8×H100 SXM)

| Seed | sliding-window BPB | cond-PPM BPB |
|------|-------------------:|-------------:|
| 42   | 1.10119889 | 0.97949078 |
| 1337 | 1.10141497 | 0.97954725 |
| 314  | 1.10142762 | 0.97962885 |
| **Mean** | **1.101347** | **0.979556** |
| **Std**  | 0.000129 | 0.000069 |

Diagnostics (3-seed means): pre-EMA 1.080004, quantized 1.092764,
sliding-window 1.101347, **cond-PPM 0.979556** (headline).

All 3 seeds: artifact + wrapped code ≤ 16,000,000 bytes; training ≤ 600s; eval ≤ 600s.

## Eval-set coverage

The cond-PPM (Lock-In Byte Mixer) runs over the **full canonical CaseOps validation set**
for every seed:

* **47,851,520 tokens / 164,594,398 canonical bytes per seed.**
* `cppm_chunks` are all-gathered across all 8 ranks before the byte-level PPM-D state
  is advanced on rank 0 in canonical val order — full-val PPM context, not per-rank.

Forensic evidence: each seed's `train_seed*.log` contains a line of the form
`cond_ppm tokens=47851520 bytes=164594398 cond_mix_bpb=...` from the mixer.
`submission.json` records `eval_full_val_verified: true`,
`eval_token_count_per_seed: 47851520`, and
`eval_canonical_byte_count_per_seed: 164594398`.

This addresses the eval-set-coverage concern raised on PR #1835.

## Compliance — Issue #1017

* **C1 (causality):** sliding-window scoring strict-past only; cond-PPM byte
  state advances ONLY after each byte's mix log-prob is recorded. Marginalization
  at byte_0 is derived from the position's softmax, which sees only strict past.
  Mix gate weights depend on PPM context confidence ONLY (not the realized byte).
* **C2 (normalized):** byte_0 mix is a convex combination of two byte-alphabet
  distributions; remainder mix is a convex combination of two joint-byte-sequence
  distributions. Product is a proper distribution over the realized token's
  byte stream.
* **C3 (score-first):** both NN softmax and PPM byte conditional commit before
  observing the realized byte at each step. PPM state advances post-scoring.
* **C4 (single L→R pass):** each val token contributes exactly one BPB term.

No SLOT, no n-gram cache outside the legal byte-level PPM-D state, no logit bias,
no ETLB, no pre-quant TTT (which is C3-violating). Standard softmax over the
SP8192 alphabet at every scored position.

## Lineage

PR #1394 (clarkkev) → PR #1530 (samacqua) → PR #1729 (romeerp CaseOps)
→ PR #1787 (nprime06) → PR #1797 (dexhunter, SmearGate fixed)
→ PR #1493 sliding-window
→ PR #1835 PPM byte-mix (this submission's C2-correct successor)

## Relation to prior submission

This submission's predecessor on the same author's fork is the **cond-PPM byte-conditional mixture**
(val_bpb 1.015784, `submission-cond-ppm-fullval` branch). LBM is the next iteration: it sweeps the
sigmoid gate from the dense (α=15, β=0.80) regime into the high-confidence-only regime
(α=25, β=0.9999), runs over the canonical 50k-doc CaseOps val (5× larger than the 10k-doc val used
in the predecessor), and adds AWQ-Lite + EVAL_SEQ_LEN bumps from the cond-ppm-stack branch.

## Reproduction

```bash
git clone https://github.com/anmarhindi/parameter-golf-a
cd parameter-golf-a
git checkout cond-ppm-stack
cd obliterate_0p9
bash run.sh                  # 3 seeds × (≤600s train + ≤600s eval)
bash build_submissions.sh    # produces this submission folder + tarball
```

`run.sh` invokes `_pod_setup.sh` to install deps (brotli, sentencepiece, flash_attn_3,
numba for the LBM hot loop) and downloads `docs_selected.jsonl` from
`willdepueoai/parameter-golf`, tokenizes via `prepare_caseops_data.py` with `--val-docs 50000`,
then trains + evals each seed.

Headline metric is `quantized_cond_ppm val_bpb`, logged by `_cond_ppm_mixture_bpb` after
`eval_val_sliding` completes. To inspect the readable source of the wrapped `train_gpt.py`:

```python
import lzma, base64, re
src = open("train_gpt.py").read()
blob = re.search(r'b85decode\("([^"]+)"\)', src).group(1)
print(lzma.decompress(base64.b85decode(blob)).decode())
```

## Files

| file | purpose |
|---|---|
| `README.md`                                | this |
| `submission.json`                          | metadata + 3-seed numbers |
| `train_gpt.py`                             | wrapped (lzma+base85) — counts toward 16MB |
| `final_model.int6.ptz`                     | quantized + brotli artifact — counts toward 16MB |
| `lossless_caps.py`                         | CaseOps transform module (PR #1729 lineage) |
| `prepare_caseops_data.py`                  | parallel CaseOps tokenizer (data prep helper) |
| `tokenizers/...caseops_v1_reserved.model`  | SP8192 CaseOps tokenizer |
| `train_seed{42,1337,314}.log`              | per-seed train + eval logs |
