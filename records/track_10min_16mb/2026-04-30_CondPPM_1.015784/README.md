# Record: cond-PPM byte-conditional mixture — val_bpb 1.015784

**val_bpb: 1.015784** (3-seed mean, std 0.000524) | 14.86 MB | 8×H100 SXM, ≤600s train / ≤600s eval

This submission is the **C2-correct version of the PPM-mix family** (PR #1835).
PR #1835 used the byte-level approximation `P_NN(byte) = exp(token_logp / n_bytes)` —
a per-byte value that is identical for every byte in the realized token and does
not sum to 1 over the byte alphabet. This submission replaces that with a proper
marginal of the SP8192 softmax via the canonical first-byte LUT:

```
P_NN(byte_0 = b | history) = Σ_{T : canonical_first_byte(T, prev) = b} P_NN(T | history)
```

The remainder bytes mix at the joint-byte-sequence alphabet via the chain-rule
residual `P_NN_rem = P_NN(token | history) / P_NN(byte_0 | history)` and the
PPM-D byte chain. **Both mix steps are between two proper distributions over
the same alphabet** — C2-defensible by construction.

Per-position correctness for leading-space tokens: the canonical first byte of a
token T at position t depends on whether the previous token x[t] is a boundary.
If T has the SP-BPE leading-space marker AND prev is non-boundary, the realized
canonical first byte is `0x20` (a prepended space); otherwise it is the SP-piece's
natural first byte. `ValidationData.__init__` builds two first-byte masks
(`mask_no_space` for prev-IS-boundary, `mask_with_space` for prev-non-boundary)
and `eval_val_sliding` selects per position via `is_boundary_token_lut[prev]`.
This eliminates the per-token uniform-split fallback that earlier drafts of this
mixer used for the include_space case.

Stacked on:
* PR #1493 sliding-window stride-64 eval (each val token scored from up to
  `seq_len-1` tokens of strict-past context).
* PR #1797 (@dexhunter) base + SmearGate BOS-mask fix (cocohearts review).
* `PARALLEL_START_LAYER=5`, `ENABLE_LOOPING_AT=0.85`, `WARMDOWN_FRAC=0.85`,
  `STOCH_DEPTH_MAX=0.02`.

## 3-seed result (8×H100 SXM)

| Seed | sliding-window BPB | cond-PPM BPB |
|------|-------------------:|-------------:|
| 42   | 1.10620496 | 1.01519482 |
| 1337 | 1.10711439 | 1.01595810 |
| 314  | 1.10664426 | 1.01619827 |
| **Mean** | **1.106655** | **1.015784** |
| **Std**  | 0.000455 | 0.000524 |

Diagnostics (3-seed means): pre-EMA 1.087591, quantized 1.101712,
sliding-window 1.106655, **cond-PPM 1.015784** (headline).

All 3 seeds: artifact + wrapped code ≤ 16,000,000 bytes; training ≤ 600s; eval ≤ 600s.

## Eval-set coverage

The cond-PPM mixer runs over the **full validation set** for every seed:

* **9,662,464 tokens / 32,756,252 canonical bytes per seed.**
* `cppm_chunks` are all-gathered across all 8 ranks before the byte-level
  PPM-D state is advanced on rank 0 in canonical val order — so the PPM
  context built up over the full val sequence, not a per-rank slice.

Forensic evidence: each seed's `train_seed*.log` contains a line of the form
`cond_ppm tokens=9662464 bytes=32756252 cond_mix_bpb=...` from
`_cond_ppm_mixture_bpb`. `submission.json` records
`eval_full_val_verified: true`, `eval_token_count_per_seed: 9662464`,
and `eval_canonical_byte_count_per_seed: 32756252`.

## Compliance — Issue #1017

| Condition | How this submission satisfies it |
|---|---|
| **C1, causality** | Sliding-window scoring is strict-past only. Cond-PPM byte state advances only after each byte's mix log-prob is recorded. The marginalization at byte_0 derives from the position's softmax (which sees only strict past) using the appropriate first-byte mask conditional on `is_boundary_token_lut[prev]`. Mix gate weights depend on PPM context confidence only, never on the realized byte. |
| **C2, normalization** | byte_0 mix is between two byte-alphabet distributions (model marginal vs PPM-D byte conditional); remainder mix is between two joint-byte-sequence distributions over the realized token's byte-1..k-1 alphabet. Both are proper — no per-token uniform-split approximation, no fallback path. |
| **C3, score-first** | Both NN softmax and PPM byte conditional commit before observing the realized byte at each step. PPM state advances post-scoring. |
| **C4, single L→R pass** | Each validation token contributes exactly one BPB term. Sliding-window scoring overlaps but each token is scored at exactly one position. |

No SLOT, no n-gram cache outside the legal byte-level PPM-D state, no logit
bias, no ETLB, no pre-quant TTT (which would violate C3). Standard softmax over
SP8192 at every scored position.

## Lineage

PR #1394 (clarkkev) → PR #1530 (samacqua) → PR #1729 (romeerp CaseOps)
→ PR #1787 (nprime06) → PR #1797 (dexhunter, SmearGate fixed via cocohearts)
→ PR #1493 sliding-window
→ PR #1835 PPM byte-mix (this submission's C2-correct successor)

## Reproduction

```bash
git clone https://github.com/anmarhindi/parameter-golf
cd parameter-golf
git checkout cond-ppm
cd obliterate_0p9
bash run.sh                  # 3 seeds × (≤600s train + ≤600s eval)
bash build_submissions.sh    # produces this submission folder + tarball
```

`run.sh` invokes `_pod_setup.sh` to install deps (brotli, sentencepiece,
flash_attn_3 with auto-detection of the matching torch+CUDA wheel) and download
`docs_selected.jsonl` from `willdepueoai/parameter-golf`, then tokenizes via
`prepare_caseops_data.py`, then trains + evals each seed via `torchrun
--standalone --nproc_per_node=8 train_gpt.py` with the canonical env-var stack
embedded in `run.sh`.

Headline metric is `quantized_cond_ppm val_bpb`, logged by
`_cond_ppm_mixture_bpb` after `eval_val_sliding` completes. Inspect the
unwrapped source of the included `train_gpt.py`:

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
| `submission.json`                          | metadata + 3-seed numbers + full-val attestations |
| `train_gpt.py`                             | wrapped (lzma+base85) — counts toward 16 MB |
| `final_model.int6.ptz`                     | int6 GPTQ + brotli artifact — counts toward 16 MB |
| `lossless_caps.py`                         | CaseOps transform module (PR #1729 lineage) |
| `prepare_caseops_data.py`                  | parallel CaseOps tokenizer (data prep helper) |
| `tokenizers/...caseops_v1_reserved.model`  | SP8192 CaseOps tokenizer |
| `train_seed{42,1337,314}.log`              | per-seed train + eval logs (training and eval are unified in this run) |
