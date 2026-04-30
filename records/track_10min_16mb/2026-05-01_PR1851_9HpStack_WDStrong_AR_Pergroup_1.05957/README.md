# Record candidate: PR1851 graph + 9-hparam stack + wd_strong + GPTQ AR + pergroup compression

**val_bpb: 1.05956571** (seed 42) | **15,901,624 bytes** | 8xH100 SXM, 600s training cap | phased LoRA TTT eval

This is the valid-size recovery of the Run 4 result. Run 4 reached
`q_ttt = 1.05950377` but produced a `16,140,607 B` brotli artifact, which
was `140,607 B` over the 16 MB cap. This submission keeps the same
PR #1851-derived training graph and hparam stack, then ports PR #1855's
`pergroup` lrzip+brotli compressor into that graph. The result is essentially
the same score, but with a valid artifact.

## Results

| Seed | Steps | ms/step | Pre-quant val_bpb | Post-quant val_bpb | **Post-TTT val_bpb** | Artifact |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4,844 | 122.2 | 1.06335171 | 1.07246420 | **1.05956571** | 15,901,624 |

The final log reports:

```text
stopping_early: wallclock_cap train_time: 592124ms step: 4844/20000
Serialized model quantized+pergroup: 15867294 bytes
Total submission size quantized+pergroup: 15901624 bytes
diagnostic pre-quantization post-ema val_loss:2.32715881 val_bpb:1.06335171 eval_time:11361ms
diagnostic quantized val_loss:2.34710161 val_bpb:1.07246420 eval_time:61777ms
quantized_ttt_phased val_loss:2.31872252 val_bpb:1.05956571 eval_time:516992ms
total_eval_time:517.0s
```

Compared with the current PR #1855 leaderboard entry:

| Comparator | val_bpb | Delta |
|---|---:|---:|
| PR #1855 published seed 42 | 1.05989454 | -0.00032883 |
| PR #1855 3-seed mean | 1.06107587 | -0.00151016 |
| This submission, seed 42 | **1.05956571** | - |

This PR is submitted as a record candidate with single-seed evidence. The
result is under the 16 MB cap and below the current best published seed-42
and 3-seed mean. Further seeds would be useful for statistical confidence,
but single-seed record-candidate PRs have been part of the challenge process.

## What changed

This run combines four pieces:

1. PR #1851-derived training graph, preserving the BOS-fixed SmearGate +
   LQER asymmetric + PR #1787 SparseAttnGate/PolarNS/FusedCE stack.
2. PR #1855's 9 greedy hparam overrides:
   `EMBED_CLIP_SIGMAS=14.0`, `MLP_CLIP_SIGMAS=11.5`,
   `WARMDOWN_FRAC=0.85`, `BETA2=0.99`, `TTT_BETA2=0.99`,
   `TTT_WEIGHT_DECAY=0.5`, `TTT_LORA_RANK=80`,
   `SPARSE_ATTN_GATE_SCALE=0.5`, and `PHASED_TTT_PREFIX_DOCS=2500`.
3. `wd_strong`: a late Muon weight-decay schedule with
   `WD_SCHEDULE_ENABLED=1`, `WD_SCHED_LOW_FACTOR=0.5`, and
   `WD_SCHED_HIGH_FACTOR=1.75`.
4. GPTQ all-rank Hessian averaging plus PR #1855's `pergroup`
   lrzip+brotli compressor ported into `train_gpt.py`.

The important distinction from PR #1855 is that this keeps the PR #1851
graph and only imports the pergroup compressor. In local experiments,
switching to the full PR #1855 script plus `wd_strong + AR` lost about
0.00053 BPB versus this path.

## Pergroup compression

The `COMPRESSOR=pergroup` path:

1. Buckets similarly-shaped quantized tensors by role.
2. Similarity-sorts rows for hot 2D groups before byte serialization.
3. Compresses group blobs with `lrzip -z -L 9`.
4. Compresses the remaining state and metadata with brotli.

On this checkpoint, replacing brotli+byte-shuffle with pergroup changed:

| Metric | Run 4 brotli | This run pergroup | Delta |
|---|---:|---:|---:|
| q_ttt | 1.05950377 | 1.05956571 | +0.00006194 |
| quantized model blob | 16,108,157 B | 15,867,294 B | -240,863 B |
| compressed code wrapper | 32,450 B | 34,330 B | +1,880 B |
| total artifact | 16,140,607 B | 15,901,624 B | -238,983 B |

So the compressor makes the previously invalid Run 4 stack fit with
`98,376 B` of margin, while preserving quality within run noise.

## Training and evaluation

| Setting | Value |
|---|---|
| Hardware | 8xH100 80GB SXM |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| Train cap | `MAX_WALLCLOCK_SECONDS=600`, `GPTQ_RESERVE_SECONDS=8.0` |
| Stop | 4,844 steps, `train_time=592124ms` |
| Quantization | GPTQ int6 matrices, int7 tied embedding, LQER asym int4 rank-4 top-k |
| Calibration | `GPTQ_CALIBRATION_BATCHES=16`, `GPTQ_ALL_REDUCE=1` |
| Compression | `COMPRESSOR=pergroup`, requires system `lrzip` binary |
| TTT | 3-phase score-first LoRA TTT, prefix docs 2500, LoRA rank 80 |
| TTT eval timer | 516.992s |

The log includes an `artifact_production_wallclock` diagnostic that spans the
script's artifact/eval path and is not the leaderboard training-loop timer.
The training timer used for the wallclock stop is the `stopping_early` line
above, matching the convention used by late CaseOps submissions.

## Requirements

See `requirements.txt`. FlashAttention 3 must be installed separately:

```bash
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

The `lrzip` system binary is required for `COMPRESSOR=pergroup`:

```bash
apt-get install -y lrzip
```

## Reproducing

```bash
RUN_ID=top_pr1855_hparams_s42_pergroup \
SEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
GPTQ_ALL_REDUCE=1 \
WD_SCHEDULE_ENABLED=1 \
WD_SCHED_LOW_FACTOR=0.5 \
WD_SCHED_HIGH_FACTOR=1.75 \
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
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The command assumes the CaseOps SP8192 dataset is prepared at the default
paths used by `train_gpt.py`. `prepare_caseops_data.py`, `lossless_caps.py`,
and the tokenizer model are included for the data-prep path.

## Files

- `train_gpt.py` - full training script with the pergroup compressor port.
- `train_seed42.log` - seed-42 training, quantization, compression, and TTT log.
- `train_seed42.stdout.log` - torchrun stdout/stderr wrapper log from the same run.
- `submission.json` - structured metadata for this run.
- `requirements.txt` - Python dependency reference.
- `lossless_caps.py` and `prepare_caseops_data.py` - CaseOps preparation support.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` -
  SentencePiece model used by the CaseOps data.

## Credits

This submission builds on the late CaseOps lineage, especially PR #1851,
PR #1855, PR #1787, PR #1736, PR #1729, PR #1667, PR #1626, PR #1610,
PR #1586, PR #1530, PR #1344, PR #493, PR #478, PR #315, and PR #289.
