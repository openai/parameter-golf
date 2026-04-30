# Record: V21 Stack + N-gram Tilt + LeakyReLU 0.3 — val_bpb 1.05851 (3-seed)

**val_bpb = 1.05851479** (3-seed mean, std 0.000762, seeds 42 / 0 / 1234) on `track_10min_16mb`.

## Per-seed

| seed | val_bpb     | eval ops ms | artifact bytes |
|-----:|------------:|------------:|---------------:|
|  42  | 1.05764263  | 575,915     | 15,949,305     |
|   0  | 1.05886205  | 553,279     | ~15,943,000    |
| 1234 | 1.05903968  | 554,723     | ~15,945,000    |
| **mean** | **1.05851479** | — | — |
| **std**  | **0.000762**   | — | — |

## Stack

1. **PR #1945** (@alertcat): V21 base = PR #1908 + AWQ-Lite mixed-precision GPTQ + Asymmetric Logit Rescale.
2. **PR #1953** (@andrewbaggio1): TTT/QK env knobs — `TTT_LR=0.75`, `QK_GAIN_INIT=5.25`, `TTT_NO_QV_MASK=1`. (2560 long-context dropped due to OOM during global-SGD allreduce on this 8×H100 80GB SXM provisioning; remaining 7 knobs preserved.)
3. **PR #1948** (@TimS-ml, @lijuncheng16): LeakyReLU squared slope 0.3 patch (4-point sweep min identified by PR #1948).
4. **PR #1145** (@AnirudhRahul, valerio-endorsed): closed-form n-gram tilt with three causal experts (token order 16, within-doc, word order 4) and Σ P=1 closed-form Z renormalization.

The static n-gram hint table is built in a single L→R causal pass over val tokens during `validate()` setup (env flag `NGRAM_HINT_PRECOMPUTE_OUTSIDE=1`, default). Setting the flag to 0 reproduces the inline build path with identical val_bpb.

## Compliance

- Train ≤ 600,000 ms, eval ops ≤ 600,000 ms, artifact ≤ 16,000,000 bytes per seed.
- Standard log-softmax over the SP8192 alphabet at every scored position; tilt is closed-form `p'(a) = exp(β·1[a=h]) · p(a) / Z`, `Z = 1 + q · (e^β − 1)`, Σ p'(a) = 1 over vocab.
- Single-pass: each val token contributes exactly one BPB term in `quantized_ttt_phased`.
- N-gram hints are strictly causal: hint at position t depends only on tokens [0..t−1].
- No SLOT, no n-gram cache hash table, no logit bias, no ETLB, no Pre-Quant TTT.

## Δ vs neighbors (3-seed)

| Submission | val_bpb | Δ vs ours |
|------------|--------:|----------:|
| **This submission** | **1.05851** | — |
| PR #1953 (andrewbaggio1) | 1.05855 | +0.00004 |
| PR #1945 (alertcat) | 1.05943 | +0.00092 |
| PR #1934 (liujshi) | 1.05993 | +0.00142 |
| PR #1956 (AayushBaniya2006) | 1.06044 | +0.00193 |
| PR #1908 (romeerp) | 1.06081 | +0.00230 |

## System dependencies

- gcc + lrzip (`apt-get install -y build-essential lrzip` on Debian/Ubuntu).
- Python: `torch==2.9.1`, triton (bundled), Flash Attention 3, numpy, sentencepiece, tiktoken, kernels, datasets, huggingface-hub, typing-extensions==4.15.0. See `requirements.txt`.
- 8× H100 80GB SXM.
- CASEOPS-preprocessed FineWeb10B data (run `prepare_caseops_data.py` once).

## Reproduction

```
bash setup.sh                   # apt + pip + Flash Attn 3
python prepare_caseops_data.py  # one-time, ~10-20 min CPU
SEED=42   bash run.sh
SEED=0    bash run.sh
SEED=1234 bash run.sh
```

## Credits

- **PR #1145** (@AnirudhRahul): closed-form n-gram tilt with Σ P=1 Z_t renormalization, three causal experts.
- **PR #1948** (@TimS-ml, @lijuncheng16): LeakyReLU squared slope 0.3 sweep finding.
- **PR #1953** (@andrewbaggio1): 7-knob TTT/QK tuning on V21 base.
- **PR #1945** (@alertcat): V21 stack composition.
- **PR #1908** (@romeerp): activation-aware GPTQ mixed precision.
- **PR #1923** (@jorge-asenjo): Asymmetric Logit Rescale.
- **PR #1855** (@codemath3000): SP8192 CaseOps + 9-hyperparameter greedy stack base.
- **PR #1493** (@dexhunter et al.): score-first TTT framework foundation.
- **PR #549** (@abaybektursun): original score-first TTT.

## Files

- `train_gpt.py`, `online_ngram_tilt.py`, `online_ngram_state.c`, `lossless_caps.py`, `prepare_caseops_data.py`
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- `setup.sh`, `run.sh`, `requirements.txt`
- `train_seed{42,0,1234}.log`
