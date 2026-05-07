# Record: PR #1797 base + PPM-D byte mixture (v2) — full-val coverage parity

**val_bpb (mix): 0.901886** (3-seed mean, std 0.000803, PPM_SUBSET_TOKENS=8,000,000) | **val_bpb (neural-only quantized_ttt_phased): 1.062106** (3-seed mean, std 0.001166, full 47.85M val) | **~15.95 MB** | 8×H100 SXM | 599.6s train / 576.7s eval

This is the v2 of our PR #1854 with the data-coverage correction described below. It pairs a byte-level PPM-D mixture (technique class first introduced in PR #1795 on 2026-04-23; ported here from anmarhindi PR #1835 on 2026-04-25) on top of the dexhunter PR #1797 verbatim base. The headline `mix_bpb` is comparable to PR #1854 (both use the 8M PPM subset); the neural-only `quantized_ttt_phased` now sits on full-val parity with dexhunter PR #1797.

## Why a v2 — explicit data-coverage correction vs PR #1854

Our PR #1854 inherited dexhunter's `prepare_caseops_data.py` and was launched from his README invocation, which relies on the argparse default `--val-docs 10000`. dexhunter's own seed log shows he silently invoked `--val-docs 50000` (47.85M val tokens, 50K docs). Our v1 therefore measured on `val_tokens=9,662,464` — ~17% of the leaderboard's full val coverage.

dexhunter publicly flagged the same 8M-subset issue on PR #1858 (G3sparky), naming PR #1854 by direct comparison. His comment is a methodological signal we owe a correction to.

This v2 retreps from raw `docs_selected.jsonl` with the explicit invocation `--val-docs 50000` (matching dexhunter's reference seed log), runs the same v1 stack on the corrected corpus, and measures the same headline metric (`mix_bpb` on `PPM_SUBSET_TOKENS=8,000,000`) plus the full-val neural diagnostic.

| Metric | PR #1854 (v1) | This (v2) |
|---|---|---|
| `val_tokens` | 9,662,464 | **47,853,344** |
| `total_docs` | 10,000 | **50,000** |
| Reference parity (vs dexhunter PR #1797 47,851,520 / 50,000) | 79.8% / 20% | **100.0% / 100.0%** |
| Headline `mix_bpb` 3-seed mean | 0.90236 | **0.901886** |
| Neural-only `quantized_ttt_phased` 3-seed mean | 1.06791 (on 9.66M) | **1.062106** (on 47.85M) |
| `quantized_ttt_phased` parity vs dexhunter PR #1797 (1.06157 on identical 47.85M) | not comparable | within 0.0006 BPB (shared seeds 42 = 1.06181 vs 1.06181, 314 = 1.06083 vs 1.06112) |

The v2 `mix_bpb` is structurally on the same `PPM_SUBSET_TOKENS=8,000,000` as v1 — see "PPM coverage disclosure" below.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, brotli)

### Core table — headline `mix_bpb` (PPM mix on 8M subset)

| Seed | Steps | `mix_bpb` | `nn_only_bpb` (on 8M) | `ppm_only_bpb` (on 8M) | `ppm_mix_time` | Artifact (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 5018 | **0.901534** | 0.972497 | 2.076814 | 109.9s | 15,951,305 |
| 1337 | 5017 | **0.902805** | 0.974119 | 2.076814 | 109.4s | 15,950,213 |
| 314 | 5016 | **0.901319** | 0.972196 | 2.076814 | 107.6s | 15,953,505 |
| **Mean** | **5017** | **0.901886** | 0.972937 | 2.076814 | 108.97s | **15,951,674** |
| **Std** |   | **0.000803** | 0.000840 | — | 1.10s | 1,355 |

### Full-val diagnostics (47,853,344 val tokens; not subset)

| Seed | `quantized_ttt_phased` `val_bpb` | `quantized_ttt_phased` `val_loss` (nats) | `diagnostic_quantized_no_ttt` `val_bpb` | `diagnostic_pre_quant_post_ema` `val_bpb` | `total_eval_time` |
|---|---:|---:|---:|---:|---:|
| 42 | **1.061812** | 2.323635 | 1.074704 | 1.065458 | 578.3s |
| 1337 | **1.063389** | 2.327087 | 1.076163 | 1.067108 | 575.9s |
| 314 | **1.061116** | 2.322113 | 1.073869 | 1.064917 | 575.9s |
| **Mean** | **1.062106** | **2.324278** | 1.074912 | 1.065827 | **576.7s** |
| **Std** | 0.001166 | 0.002548 | 0.001151 | 0.001138 | 1.39s |

All 3 seeds clear the 600s train cap (`stopping_early: wallclock_cap` at 599.6s mean) and the 600s eval cap (576.7s mean). Artifacts max at 15,953,505 bytes (~46 KB headroom under the 16,000,000-byte decimal cap).

### Parity with dexhunter PR #1797 on shared seeds

dexhunter PR #1797 (no PPM, same val coverage) reports per seed (314, 42, 1234):

| Seed | dexhunter PR #1797 `val_bpb` | this submission `quantized_ttt_phased` | Δ |
|---|---:|---:|---:|
| 42 | 1.06181 | 1.06181 | **+0.00000** |
| 314 | 1.06083 | 1.06112 | +0.00029 |

Our third seed (1337) replaces dexhunter's (1234). Across the 2 shared seeds, our neural-only is at byte-for-byte parity with dexhunter on the same val coverage and same base stack. The PPM-D layer is the only addition.

## Stack

| Component | Origin | Role |
|---|---|---|
| CaseOps bijective case transform | PR #1729 (romeerp) / PR #1736 / PR #1797 (dexhunter) | ~1.5% token savings, full byte-level bijection (legality pending Issue #1604) |
| SparseAttnGate | PR #1787 (nprime06) | sparse per-head gate inside attention |
| Smear gate | PR #1797 (dexhunter) | causal content-conditioned gate, 1-token lookback |
| LQER asymmetric rank-4 correction | PR #1797 (dexhunter) | post-GPTQ int6 residual recovery |
| Phased TTT (score-first, 3 phases, 2000-doc prefix) | PR #1394 / PR #549 / PR #1413 lineage | per-document LoRA adapter, score-before-update |
| Int6 GPTQ + Brotli compressor | PR #1019 / PR #1530 | fits int6 model + factors + code under 16,000,000 bytes |
| **Byte-level PPM-D mixture (this addition)** | Class introduced in PR #1795 (2026-04-23); ported here from anmarhindi PR #1835 (2026-04-25) | order-5 PPM-D, binary-lambda gate `(λ_lo=0.05 when conf≥0.9 else λ_hi=0.9)`, mixed in probability space, counts updated AFTER scoring (legality pending Issue #1872) |

## PPM coverage disclosure (what dexhunter flagged on PR #1858)

`PPM_SUBSET_TOKENS=8000000` controls how many val tokens enter the PPM mix. On our val of 47,853,344 this is 16.7% coverage. The constraint is structural in this stack: at 35M PPM coverage (73% of val) the same configuration measured `total_eval_time:1041.1s`, exceeding the 600s eval cap (s42 v1, see `logs/dex_ppm_battery.log`). At 8M PPM coverage the eval fits in 576.7s mean.

The headline `mix_bpb=0.901886` is therefore **directly comparable to PR #1854 (this submission's predecessor) and to other PPM-D byte-mixture submissions using `PPM_SUBSET_TOKENS=8000000`** (PR #1835, PR #1850, PR #1858 if they re-run on the same subset). The non-PPM diagnostics (`quantized_ttt_phased`, `diagnostic_quantized_no_ttt`, `diagnostic_pre_quantization_post_ema`) are computed on the full 47,853,344 val and ARE directly comparable to dexhunter PR #1797 (1.06157) and to merged SOTA PR #1493 (1.0810) per the leaderboard's standard byte-level BPB metric.

## Rule compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL**: all 3 seeds 15,950,213–15,953,505 bytes (~46–50 KB headroom).
- **train_time ≤ 600s**: all 3 seeds 599.575–599.628s (`stopping_early: wallclock_cap`).
- **total_eval_time ≤ 600s**: all 3 seeds 575.9–578.3s.
- **Issue #1017 Condition 1 (causal dependence)**: PPM context at byte t uses bytes <t only. Phased TTT updates the per-document LoRA adapter AFTER scoring every chunk. SparseAttnGate and Smear gate causal per dexhunter PR #1797 audit.
- **Issue #1017 Condition 2 (full normalized distribution)**: token-vs-byte-alphabet question is the subject of Issue #1872 (cocohearts ruling pending). This submission is in the PPM-D cluster called out by name in #1872 (#1835/#1850/**#1854**/#1858/#1862/#1865/#1871/#1873). If the ruling concludes Σ = SP8192 token alphabet, this and the rest of the cluster fail C2. If the ruling concludes Σ = byte alphabet, both `p_NN` (bit-conserving spread) and `p_PPM` are normalized over 256 symbols and the convex combination is normalized.
- **Issue #1017 Condition 3 (score-before-update)**: Phased TTT path snapshots the pre-update per-chunk logits and scores them BEFORE the adapter SGD step. Per-document LoRA reset (`reusable_lora.reset()`) prevents cross-document leakage. PPM-D counts incremented at byte t only AFTER `−log p_mix(t)` is recorded.
- **Issue #1017 Condition 4 (single left-to-right pass)**: eval is one left-to-right pass with sliding stride 64; no rescore/selection.
- **Section V — byte-level BPB**: BPB is scored on original pre-transform UTF-8 bytes via the per-token byte sidecar (`fineweb_val_bytes_XXXXXX.bin`), parallel to the val token shards. No hardcoded bytes/token.
- **CaseOps tokenizer (Issue #1604)**: ruling pending, ~80% allowed per current discussion. Inherited verbatim from dexhunter PR #1797 base.

## Reproducibility

### 1. Data prep (one-time, ~12h on M5 Pro / 30-60 min on 16-core CPU pod)

```bash
# CRITICAL: --val-docs 50000 is required to match dexhunter PR #1797 val coverage.
# argparse default is 10000 (= ~9.66M val tokens, INCOMPATIBLE with leaderboard metric).
python3 prepare_caseops_data.py \
    --docs <path/to/docs_selected.jsonl> \
    --out <data_dir> \
    --sp tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    --val-docs 50000 \
    --workers 16
```

`docs_selected.jsonl` = `willdepueoai/parameter-golf` HF dataset.

The pre-prepared dataset is also published as `alienNiko/caseops-s11-data-v3` on Hugging Face for fast-resume:
```bash
huggingface-cli download alienNiko/caseops-s11-data-v3 --repo-type dataset --local-dir <data_dir>
```

After prep, sanity-check (REQUIRED before training):
```python
import numpy as np, glob
val_files = sorted(f for f in glob.glob('<data_dir>/fineweb_val_*.bin') if '_bytes_' not in f)
total = sum(np.fromfile(f, dtype=np.uint16, offset=1024).size for f in val_files)
assert abs(total - 47_851_520) / 47_851_520 < 0.01, f'val_tokens={total}, mismatch with reference 47,851,520'
print(f'val_tokens={total} OK')
```

### 2. Train + eval per seed (8×H100 SXM, ~20 min each)

```bash
export PYTHONUNBUFFERED=1 \
       SEED=42 \
       VOCAB_SIZE=8192 \
       NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4.0 \
       LOOP_START=3 LOOP_END=5 NUM_LOOPS=2 PARALLEL_START_LAYER=8 \
       MATRIX_BITS=6 EMBED_BITS=7 \
       SPARSE_ATTN_GATE_ENABLED=1 SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
       LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
       MIN_LR=0.1 FUSED_CE_ENABLED=1 TTT_WARM_START_A=1 \
       PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
       TTT_EPOCHS=3 TTT_LR=0.005 \
       PPM_ENABLED=1 PPM_ORDER=5 PPM_SUBSET_TOKENS=8000000 \
       LAMBDA_HI=0.9 LAMBDA_LO=0.05 PPM_CONF_THRESHOLD=0.9 \
       DATA_PATH=<data_dir>

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=1337` and `SEED=314`.

### 3. Verify

Each seed log should end with:
- `stopping_early: wallclock_cap train_time: 599xxx ms`
- `Total submission size quantized+brotli: 159xxxxx bytes`
- `ppm_mix bytes=27048621 mix_bpb=0.901xxx ppm_only=2.076814 nn_only=0.97xxxx`
- `quantized_ttt_phased val_loss:2.32xxxxxx val_bpb:1.06xxxxxx eval_time:57xxxx ms`
- `total_eval_time:57x.xs`

Numbers reproducible to ±0.001 BPB across CUDA non-determinism (verified across two independent re-runs).

## Architecture (inherits PR #1787/PR #1797 shape verbatim)

| Item | Value |
|---|---:|
| num_layers | 11 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 |
| mlp_mult | 4.0 |
| rope_base / rope_dims | 10000 / 16 |
| logit_softcap | 30.0 |
| loop_start / loop_end | 3 / 5 (NUM_LOOPS=2) |
| parallel_start_layer | 8 |
| eval_seq_len / eval_stride | 2048 / 64 |
| matrix_bits / embed_bits | 6 / 7 |
| LQER rank / top-K / A-bits / B-bits / asym group | 4 / 3 / 2 / 4 / 64 |
| smear gate window | 12 |
| TTT epochs / LR | 3 / 0.005 |
| Phased TTT prefix docs / phases | 2000 / 3 |
| PPM order / subset tokens | 5 / 8,000,000 |
| Lambda gate (PPM conf ≥ 0.9 → λ_lo=0.05; else λ_hi=0.9) | 0.05 / 0.9 |
| compressor | brotli |

## Acknowledgements

- **PR #1795** (2026-04-23) — earliest reference of the byte-level PPM-D mixture technique class.
- **anmarhindi** / PR #1835 (2026-04-25) — the specific implementation we ported from.
- **dexhunter** for the PR #1797 base stack and the PR #1858 methodology comment that motivated this v2.
- **romeerp** / **PR #1729** lineage for the CaseOps bijective tokenizer.
- All authors in the byte-level PPM-D mixture cluster (#1795, #1835, #1850, #1854, #1858, #1862, #1865, #1871, #1873) for collectively elaborating the technique while Issue #1872 awaits a ruling.

Lineage correction prompted by @OE-GOD's review note on this PR.
