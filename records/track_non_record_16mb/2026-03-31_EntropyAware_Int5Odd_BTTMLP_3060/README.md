# Entropy-Aware Int5-Odd + BTT-MLP

## Summary

This non-record submission packages the current best fully recovered run of two ideas aimed directly at Parameter Golf's artifact objective:

1. **Entropy-aware training** over a 5-bin odd quantization grid `{-2,-1,0,1,2}` aligned to the exported `int5_odd + zlib` artifact.
2. **Structured MLP matrices** using a 2-core TT/BTT-inspired `StructuredLinear` in the MLP only, leaving attention dense.
3. **Evaluation-time materialization** of the structured MLP into dense weights, so validation runs through standard `F.linear` instead of the slower TT/BTT rank loop.

The submission basis is the fully recovered `1xH100` run in [`train.log`](./train.log), which is a copy of [`logs/h100_real_r256_l16_seq1024_mb2048_materialized.txt`](./logs/h100_real_r256_l16_seq1024_mb2048_materialized.txt). This is an exploratory non-record result, not a leaderboard-comparable run, because it uses `VAL_TOKEN_LIMIT=1048576` rather than the full validation set.

A later `3300s` H100 attempt progressed further but was interrupted when RunPod exhausted the remaining credits. Because the final artifact and complete log were not recovered, that longer run is not used for the submission metrics here.

## Cloud Transition

The harness is now prepared for a single-node `torchrun` launch on `8xH100` without changing the local-tuned default path.

New training controls in [`train_gpt.py`](./train_gpt.py):

- `TRAIN_MICROBATCH_TOKENS`: per-GPU tokens per microstep
- `GRAD_ACCUM_STEPS`: optional explicit override; otherwise inferred from `TRAIN_BATCH_TOKENS`, `WORLD_SIZE`, and `TRAIN_MICROBATCH_TOKENS`
- `LR_SCALE_METHOD=none|sqrt|linear`
- `LR_REFERENCE_BATCH_TOKENS`
- `WARMUP_SCALE_METHOD=none|sqrt|linear`
- `WARMUP_REFERENCE_BATCH_TOKENS`

Recommended single-node launcher:

```bash
bash run_8xh100.sh
```

The launcher uses:

- `torchrun --standalone --nproc_per_node=8`
- `TRAIN_BATCH_TOKENS=131072`
- `TRAIN_MICROBATCH_TOKENS=8192`
- `LR_SCALE_METHOD=sqrt`
- `WARMUP_SCALE_METHOD=linear`
- `VAL_TOKEN_LIMIT=0`

This keeps DDP simple, scales the optimizer from the local `16384`-token reference batch, and increases warmup automatically when the effective global batch is larger.

### DDP data sharding

The training loader is rank-aware. Each process constructs the same contiguous shared chunk and then slices out its own disjoint span using `rank` and `world_size`, so gradients are synchronized over different token ranges rather than duplicated work.

The sharding debug path is built into [`train_gpt.py`](./train_gpt.py) via:

- `SIMULATED_WORLD_SIZE`
- `SIMULATED_RANK`
- `DEBUG_DATA_SHARDING_STEPS`
- `DRY_RUN_INIT_ONLY`

Local sharding/math sanity check:

```bash
bash run_mock_8xh100_math.sh
```

This prints:

- inferred `grad_accum_steps`
- scaled learning rates and warmup
- one simulated shared chunk split across ranks `0..7`

The checked local run is recorded in [`logs/mock_8xh100_math.txt`](./logs/mock_8xh100_math.txt).

Gradient-accumulation stress check:

```bash
bash run_extreme_accum.sh
```

The checked run in [`logs/extreme_accum.txt`](./logs/extreme_accum.txt) completed with `grad_accum_steps=64` and peaked at `4617 MiB` allocated on the 3060.

## Main Result

Submission command:

```bash
RUN_ID=h100_real_r256_l16_seq1024_mb2048_materialized \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=16 \
BTT_RANK=256 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=16384 \
TRAIN_MICROBATCH_TOKENS=2048 \
VAL_BATCH_SIZE=1048576 \
VAL_TOKEN_LIMIT=1048576 \
COMPILE_STRUCTURED_MLP=0 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
python3 train_gpt.py
```

Recovered H100 metrics on `1x H100 80GB HBM3` with `80` FineWeb train shards and `VAL_TOKEN_LIMIT=1048576`:

| Variant | Params | Step Avg | Pre-Quant val_bpb | Roundtrip val_bpb | Quantized Size | Total Size |
|---|---:|---:|---:|---:|---:|---:|
| BTT-MLP + entropy-aware `int5_odd` + eval materialization | 25,727,104 | 19468.45 ms | 5.3457 | 5.8880 | 5,184,543 B | 5,267,667 B |

Additional notes:

- Peak memory: `50290 MiB allocated / 78558 MiB reserved`
- Optimizer-step training time: `603.522 s` for `31` steps
- Final exact roundtrip metrics: `val_loss=9.82734489`, `val_bpb=5.88802157`
- Quantized eval time on the `1,048,576` token validation cap: `854 ms`
- This run is the basis for `submission.json` and `train.log`

## Evaluation Materialization Result

The key systems win in this folder is the eval-time materialization hook in [`train_gpt.py`](./train_gpt.py), which contracts the BTT cores into a dense matrix during validation and runs `F.linear` instead of the training-time structured path.

The direct benchmark is in [`logs/eval_bench_r256_l16_materialized_vb1048576.txt`](./logs/eval_bench_r256_l16_materialized_vb1048576.txt):

| Eval Path | Validation Tokens | Eval Time |
|---|---:|---:|
| Structured rank-loop path, cached | 1,048,576 | ~96.99 s |
| Materialized dense eval path | 1,048,576 | 0.851 s |

This is the change that made the structured submission operationally viable on H100 for evaluation.

## Local Result

The local 3060 smoke result is still kept in [`logs/smoke_btt_40.txt`](./logs/smoke_btt_40.txt) and remains useful as a low-cost regression test for future iterations.

## Dense Control

The dense reference run in [`dense_ablation.log`](./dense_ablation.log) was kept as a sanity check for the export path.

| Variant | Params | Step Avg | Pre-Quant val_bpb | Roundtrip val_bpb | Quantized Size | Total Size |
|---|---:|---:|---:|---:|---:|---:|
| Dense + entropy-aware `int5_odd` | 17,059,912 | 595.10 ms | 2.7479 | 3.1177 | 3,781,677 B | 3,847,186 B |

The structured path still loses too much quality relative to dense at equal training budget, so this remains exploratory rather than competitive.

## Key Debugging Findings

### 1. Safe compiled structured MLP

Naively wrapping the BTT MLP in `torch.compile(mode="reduce-overhead")` crashed under gradient accumulation because the mode enables CUDAGraphs by default in this torch build.

The fix in [`train_gpt.py`](./train_gpt.py):

- marks compile step boundaries explicitly before compiled model invocations
- compiles the structured MLP with the `reduce-overhead` option set but forces `triton.cudagraphs=False`

Measured 8-step benchmark logs:

- [`logs/compile_off_bench.txt`](./logs/compile_off_bench.txt)
- [`logs/compile_on_bench.txt`](./logs/compile_on_bench.txt)

| Compile Structured MLP | Step Avg | Roundtrip val_bpb |
|---|---:|---:|
| Off | 5037.16 ms | 6.0884 |
| On, safe non-CUDAGraph path | 1991.68 ms | 4.3758 |

The main takeaway is speed: the compiled structured path is about **2.53x faster** on the 3060.

### 2. `mup` init is slightly better than xavier

Short 12-step init ablation logs:

- [`logs/init_mup_bench.txt`](./logs/init_mup_bench.txt)
- [`logs/init_xavier_bench.txt`](./logs/init_xavier_bench.txt)

| BTT Init | Pre-Quant val_bpb | Roundtrip val_bpb |
|---|---:|---:|
| `mup` | 3.7575 | 3.9927 |
| `xavier` | 3.7596 | 4.0048 |

The gain is small but consistent, so `mup` remains the default.

### 3. Lower rate penalty is the better local default

The original entropy penalty was too aggressive for this small structured model.

Matched 40-step comparison:

- default tuned run: [`train.log`](./train.log)
- higher-penalty ablation: [`logs/lambda_high_40.txt`](./logs/lambda_high_40.txt)

| RATE_LAMBDA | Pre-Quant val_bpb | Roundtrip val_bpb | Total Size |
|---|---:|---:|---:|
| `0.00002` | 3.2636 | 3.4274 | 2,168,889 B |
| `0.002` | 3.2655 | 3.4250 | 2,159,147 B |

The short sweep favored the lower penalty, but the matched 40-step local comparison now gives a slight roundtrip edge to `0.002`. Keep both settings as live candidates before the cloud run.

The shorter sweep harness is kept in [`run_lambda_sweep.sh`](./run_lambda_sweep.sh).

### 4. The implementation can now spend the artifact budget

The early structured runs were far too small. After increasing BTT rank and depth, the same export path can reach the intended `12MB–14MB` band.

Zero-step capacity scouts:

- [`logs/R1024_L20.txt`](./logs/R1024_L20.txt)
- [`logs/R1024_L24.txt`](./logs/R1024_L24.txt)

| Config | Params | Total Size | Roundtrip val_bpb |
|---|---:|---:|---:|
| `BTT_RANK=1024 NUM_LAYERS=20` | 79,213,728 | 11,224,259 B | 3.7634 |
| `BTT_RANK=1024 NUM_LAYERS=24` | 94,951,616 | 13,435,175 B | 3.7671 |

These are **capacity scouts only**, not trained submissions, but they show the BTT stack can occupy a realistic non-record budget before moving to cloud hardware.

## Local Workflow

Helper scripts included in this folder:

- [`run_compile_bench.sh`](./run_compile_bench.sh): compile on/off benchmark for the structured MLP
- [`run_init_ablation.sh`](./run_init_ablation.sh): `mup` vs `xavier`
- [`run_lambda_sweep.sh`](./run_lambda_sweep.sh): entropy-penalty sweep with the compiled path enabled
- [`run_capacity_scout.sh`](./run_capacity_scout.sh): quick size scouts up to the 12MB–14MB regime
- [`run_8xh100.sh`](./run_8xh100.sh): single-node `8xH100` torchrun launcher with batch/LR/warmup scaling defaults
- [`run_mock_8xh100_math.sh`](./run_mock_8xh100_math.sh): local dry run for 8-GPU batch math plus rank-aware sharding inspection
- [`run_extreme_accum.sh`](./run_extreme_accum.sh): 50+ microstep accumulation stress test on one GPU
- [`run_local_tuning.sh`](./run_local_tuning.sh): runs the full local tuning sequence

## Current Limitations

- Validation is still capped locally with `VAL_TOKEN_LIMIT=1048576` by default for tractable iteration on a 3060.
- The BTT forward path still uses a rank loop to stay memory-safe during validation and export. That keeps it correct but slower than a production-quality fused implementation.
- The entropy-aware objective is a practical rate proxy aligned to this repo's serializer, not a literal reproduction of BackSlash or CERWU.
- The high-rank `12MB–14MB` configurations are not yet trained locally; they are cloud candidates.
- Validation batch sizing is now independent of gradient accumulation, which matters for high-accumulation debug runs and cloud launches.

## Future Work

The most immediate next step is to rerun the same materialized-eval stack on `1xH100` with a longer wallclock budget. A later `3300s` H100 attempt was started after the recovered `600s` run and was progressing in the right direction before RunPod exhausted the remaining credits: the last recovered checkpoint in the live log reached `step 170` at `train_time 3108679ms` with `train_loss 4.9575`, and earlier in that same run it had already improved from `train_loss 5.3368` at step `90` to `4.8696` at step `160`. Because the final artifact and complete tail of the log were not recovered, that longer run is not used in `submission.json`, but it is the clearest indicator for where to spend the next block of compute.

Concretely, the next cloud pass should:

- rerun `R256/L16` with the materialized eval path and a longer wallclock budget
- recover the full final artifact and log from that longer run
- then retest higher-capacity scouts such as `R1024/L20` and `R1024/L24` once the structured training path is fast enough to give those models meaningful token exposure
