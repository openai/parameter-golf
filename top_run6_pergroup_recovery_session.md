# Run 6: PR1851 + 9 hparams + wd_strong + AR + pergroup — pergroup recovery

Session 2026-04-30 (continuation). This run executes the recovery plan from
`top_run4_pergroup_recovery_runbook.md`: keep the **Run 4** training graph
(`train_top.py`, PR #1851 base) and **Run 4 hparam stack** (the 9 PR #1855
overrides + `wd_strong` + GPTQ AR), and replace the cap-busting brotli
serialization with PR #1855's `pergroup` (lrzip+brotli) compressor that we
ported into `train_top.py`.

The motivation is the Run 4 result:

```text
Run 4 q_ttt = 1.05950377   (best of session, beats PR #1855 published s42 by 0.00039)
Run 4 size  = 16,140,607 B (BUSTS 16,000,000 B cap by 140,607 B with brotli)
```

Run 4's score is invalid because the artifact is over cap. Run 5 fixed the
artifact by switching to PR #1855's full script (`train_top_1855.py` + pergroup)
but lost quality (`q_ttt = 1.06009`, +0.00059 vs Run 4). Run 6 keeps Run 4's
graph and only swaps the compressor.

## Setup

Run 6 is launched from `train_top.py` after porting the pergroup compressor
into it (commit `0209a50` — `Port PR #1855 pergroup lrzip compressor into
train_top.py`). The port adds `_serialize_pergroup` /
`_deserialize_pergroup` and routes serialize/deserialize on
`h.compressor == "pergroup"`. Verified by a synthetic 138-tensor roundtrip
before the run.

```bash
mkdir -p artifacts/top_pr1855_hparams_s42_pergroup logs

RUN_ID=top_pr1855_hparams_s42_pergroup SEED=42 \
ARTIFACT_DIR=artifacts/top_pr1855_hparams_s42_pergroup \
CASEOPS_ENABLED=1 EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
GPTQ_ALL_REDUCE=1 \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
EMBED_CLIP_SIGMAS=14.0 \
MLP_CLIP_SIGMAS=11.5 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
PHASED_TTT_PREFIX_DOCS=2500 \
COMPRESSOR=pergroup \
torchrun --standalone --nproc_per_node=8 train_top.py \
  > logs/top_pr1855_hparams_s42_pergroup.stdout 2>&1
```

This is identical to Run 4 except for the addition of `ARTIFACT_DIR=...` (so
artifacts are not overwritten by later experiments) and `COMPRESSOR=pergroup`.

### Pod state

This run was launched on a fresh 8×H100 SXM pod with no FineWeb-caseops data
locally. The session-prep steps were:

- `apt-get install -y lrzip` (`lrzip 0.651`, required by pergroup)
- `pip install brotli python-minifier`
- snapshot-download of `romeerp/parameter-golf-caseops-v1` (16 GB) for the
  caseops-tokenized FineWeb shards + the canonical
  `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` SP model
  (366,510 B). The download took 20.6 s with `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Layout match: 80 train shards + 1 val + 1 val_bytes at the path
  `data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/`,
  exactly matching `train_top.py`'s expected `_default_caseops_data` /
  `_default_caseops_tok` paths under `CASEOPS_ENABLED=1`.

The dataset is a community re-export of the canonical PR #1855 caseops shards.
The SP model file is the same canonical
`fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` referenced by the
PR #1855 submission. Token IDs should therefore be identical to Run 4 / Run 5
to the extent that the upstream export is the canonical one, and Run 6 should
be a faithful reproduction of the Run 4 training conditions.

## Results

```text
pre   = 1.06335171   (post-EMA, pre-quant)
q     = 1.07246420   (post-LQER asymmetric quantization)
q_ttt = 1.05956571   (post-phased-LoRA TTT — primary metric)
size  = 15,901,624 B (UNDER 16,000,000 B cap by 98,376 B — VALID)
  └ quantized model blob (pergroup) = 15,867,294 B
  └ compressed code wrapper         =     34,330 B
stop  = 4844/20000   (wallclock cap = 592124 ms)
total_eval_time = 517.0 s
```

Stage timings:

```text
GPTQ Hessians:                   3.6 s   (Run 4: 3.5 s)
GPTQ quantize:                  10.1 s   (Run 4: 10.1 s)
GPTQ + LQER + pergroup serialize: 129.2 s   (Run 4: 65.2 s with brotli; pergroup costs ~64 s extra)
Pergroup deserialize (each):    21.1 s   (Run 4: ~instant with brotli)
Pre-quant post-EMA val:         11.4 s
Quant val:                      61.8 s
Phased TTT (3 phases, 2500 prefix docs): 516.9 s
```

### Comparison to Run 4 — only the compressor differs

| metric | Run 4 (brotli + bshuffle) | Run 6 (pergroup) | Δ |
|---|---:|---:|---:|
| pre | 1.06330575 | 1.06335171 | +0.00005 |
| q | 1.07238835 | 1.07246420 | +0.00008 |
| q_gap (q − pre) | 0.00908 | 0.00911 | +0.00003 |
| **q_ttt** | **1.05950377** | **1.05956571** | **+0.00006** |
| Serialized fp32 model | 135,417,533 B | 135,417,533 B | identical |
| Quantized blob | 16,108,157 B | 15,867,294 B | **−240,863 B** |
| Code wrapper | 32,450 B | 34,330 B | +1,880 B |
| **Total submission** | **16,140,607 B** | **15,901,624 B** | **−238,983 B** |
| **Valid under 16 MB cap?** | **NO (+140,607 B over)** | **YES (−98,376 B under)** | — |
| stop step | 4788/20000 | 4844/20000 | +56 (more steps in same wallclock — pod variance) |
| total_eval_time | 520.0 s | 517.0 s | −3 s |

The training is **bit-equivalent within pod-to-pod noise**. All three quality
metrics (pre, q, q_ttt) drift by ≤0.00008 BPB, which is below typical 1-seed
pod variance (Run 4 vs Run 5 published s42 differ by 0.00039 even on the
"same" stack). The pergroup compressor saves **240,863 B on the model blob
and 238,983 B on the total**, which lines up with PR #1855's README claim of
"~280 KB savings vs straight brotli" (the gap is plausible — different runs
have different quantized weight distributions, so brotli/pergroup deltas
are not exactly transportable).

### Comparison to all session runs + PR #1855 published

| run | base | additions | compressor | q_ttt | total bytes | valid? |
|---|---|---|---|---:|---:|---|
| Run 0 | PR1851 | wd_strong | brotli (+bshuffle) | 1.06111 | 15,948,542 | yes |
| Run 1 | PR1851 | AR | brotli (+bshuffle) | 1.06266 | 15,956,401 | yes |
| Run 2 | PR1851 | AR + wd_default | brotli (+bshuffle) | 1.06129 | 15,950,537 | yes |
| Run 3 | PR1851 | AR + wd_strong + paired-head | brotli (+bshuffle) | 1.06136 | 15,948,493 | yes |
| Run 4 | PR1851 | 9hp + wd_strong + AR | brotli (+bshuffle) | **1.05950** | 16,140,607 | **NO, +140,607 B over** |
| Run 5 | PR1855 | wd_strong + AR | pergroup | 1.06009 | TBD | yes |
| **Run 6** | **PR1851 + ported pergroup** | **9hp + wd_strong + AR** | **pergroup** | **1.05957** | **15,901,624** | **YES** |
| PR #1855 published s42 | PR1855 | (none) | pergroup | 1.05989 | 15,897,259 | yes |
| PR #1855 3-seed mean | PR1855 | (none) | pergroup | 1.06108 | 15,901,919 | yes |

Run 6 is **the best valid-size single-seed q_ttt of the session**:

- vs Run 4 (invalid): only +0.00006 BPB worse, but valid.
- vs Run 5 (the prior valid recovery attempt on PR #1855 base): **−0.00053 BPB better**. Confirms the runbook's hypothesis that "preserve Run 4 graph + only swap compressor" outperforms "preserve compressor + retrain on PR #1855 base + apply our patches."
- vs PR #1855 published s42: **−0.00033 BPB better**, total bytes only +4,365 larger (still 98 KB under cap).
- vs PR #1855 3-seed mean (current SOTA): **−0.00152 BPB better** (~1.7σ given std=0.00090).
- vs acceptance bar (~1.0588 = SOTA − 0.005 nats): **+0.00077 BPB short**. Single seed; not a record candidate but a strong non-record submission.

## Three findings

### 1. The pergroup port is functionally identical to PR #1855's pergroup

Run 6 demonstrates that `train_top_1855.py`'s `_serialize_pergroup` /
`_deserialize_pergroup` ported into `train_top.py` produces a fully working
roundtrip on a real GPTQ + LQER quantized state dict, end-to-end through
phased TTT eval. Validation:

- Synthetic 138-tensor preflight roundtrip (all int8 tensors recover bit-exact).
- Live deserialize during phased TTT eval succeeds and produces a q val
  that matches the in-memory pre-write q val to numerical noise.
- Total artifact savings (240,863 B on model blob) align with the
  PR #1855 README's published "~280 KB" claim within tolerance.

The port is therefore safe to keep in `train_top.py` as the default
`COMPRESSOR=pergroup` path for any future PR1851-derived experiments.

### 2. Compressor swap preserves quality to within pod noise

Pre, q, and q_ttt all drift by ≤0.00008 BPB between Run 4 and Run 6 despite
the pergroup serialization replacing brotli + byte-shuffle. This is
expected — pergroup operates on the GPTQ output state dict and reorders /
re-packs / re-compresses it; the dequantized weights it produces are
bit-identical to what brotli would have produced. The tiny remaining drift
(≤8e-5 BPB) is pod-to-pod random nondeterminism (cuBLAS / flash-attn / NCCL
all-reduce ordering during training), not anything compressor-introduced.

### 3. The Run 4 → Run 6 pivot beats the Run 5 path

The earlier session pivoted from PR #1851 to PR #1855 base for Run 5 because
the runbook author thought we needed PR #1855's pergroup compressor and
their full script. Run 5 (`train_top_1855.py + wd_strong + AR + pergroup`)
landed at q_ttt = 1.06009. Run 6 (`train_top.py + 9hp + wd_strong + AR +
ported pergroup`) lands at 1.05957 — **0.00053 BPB better than Run 5**.

The mechanism is consistent with what Run 4 already showed: the 9 PR #1855
hparams compose better with our `wd_strong` on the PR #1851-derived graph
than they do on the PR #1855 graph. Whether that's because the PR #1855
graph has small training-side differences that don't like wd_strong, or
because the 9 hparams happen to be a better fit on the slightly different
PR #1851 path, we cannot disentangle without a third (PR #1855 base, no
wd_strong) run.

For the recovery question — "what's our best valid single-seed q_ttt?" —
Run 6 is the answer.

## Acceptance-bar reality check

```text
Current SOTA (PR #1855 3-seed mean) = 1.06108 BPB
Acceptance bar (SOTA − 0.005 nats / ~0.0023 BPB) ≈ 1.0588 BPB
Run 6 single-seed q_ttt              = 1.05957 BPB
  - vs SOTA mean: −0.00152 BPB (~1.7σ given std=0.00090)
  - vs acceptance bar: +0.00077 BPB short
```

A 3-seed campaign of Run 6's exact config would be needed to claim a record.
Single-seed s42 lands ~half-σ short of the bar, similar to Run 4.

The honest position:

- **Run 6 is the strongest single-seed submission on this stack so far** —
  better than Run 5 (1.06009), better than PR #1855 s42 (1.05989), worse
  only than Run 4 (1.05950, invalid).
- It is **not yet a record candidate**, because it's single-seed and ~half-σ
  short of the acceptance threshold. A 3-seed mean ≈ 1.060 ± 0.001 (Run 6's
  predicted distribution from PR #1855's variance pattern) would land on
  the acceptance line, not clearly above it.
- It is **a strong non-record submission** with documented findings
  (pergroup port works, 9hp transfer holds across compressors, Run 5 path
  was a detour).

## Files

- `train_top.py` — PR #1851-based training script with pergroup compressor
  ported from `train_top_1855.py` (commit `0209a50`)
- `artifacts/top_pr1855_hparams_s42_pergroup/final_model.pt` — FP checkpoint
- `artifacts/top_pr1855_hparams_s42_pergroup/final_model.int6.ptz` —
  GPTQ int6 + LQER + pergroup-compressed quantized blob
- `artifacts/top_pr1855_hparams_s42_pergroup/top_pr1855_hparams_s42_pergroup.txt`
  — per-rank training log
- `logs/top_pr1855_hparams_s42_pergroup.stdout` — torchrun stdout/stderr
- HuggingFace mirror: `shikhar007/parameter-golf-gram-ns` under `models/` and `logs/`
- `upload_run6_to_hf.py` — pushes the artifacts above to that HF repo
