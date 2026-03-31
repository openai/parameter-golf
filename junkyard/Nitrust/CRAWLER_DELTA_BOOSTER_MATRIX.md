# Nitrust Crawler/Delta Booster Matrix
Date: 2026-03-27
Target: `experiments/Medusa` (Crawler + DeltaNet)
Scope: architecture/external-system boosters, plus Rust integration ablations
Guardrail: NGRAM disabled for all signal tests (`NGRAM_EVAL_ORDER=0`)
Active lane: crawler-only (`DELTA_NET_HEADS=0`) until DeltaNet sandbox re-validation succeeds

Update (2026-03-29):
- DeltaNet is quarantined from the main crawler run path pending re-validation.
- Bandit is treated as current SOTA reference while crawler-only leg is rebuilt.

## Master Hypothesis Table

| ID | Area | Booster Hypothesis | Primary Knobs | Expected Win | Risk | Smoke Ready |
|---|---|---|---|---|---|---|
| CDB-01 | Quant Bridge | Loop-aware GPTQ (flat first, crawler second) beats one-shot GPTQ. | `LOOP_AWARE_GPTQ` | Better int6 roundtrip BPB | Cal cost | Yes |
| CDB-02 | Quant Bridge | Keep crawler tensors int8 while flat stays int6. | `CRAWLER_QUANT_INT8` + export policy | Less loop error compounding | Size creep | Yes |
| CDB-03 | Quant Bridge | Per-loop crawler dequant scales reduce distribution drift. | per-loop scale banks | Better loop stability | Metadata size | No |
| CDB-04 | Quant Bridge | Skip GPTQ for flat, GPTQ only crawler+delta. | selective GPTQ groups | Faster quant + similar BPB | Flat quality drop | No |
| CDB-05 | Delta Core | Delta head count has a sweet spot (under/over hurts). | `DELTA_NET_HEADS` sweep | Better quality/compute | Runtime cost | Yes |
| CDB-06 | Delta Core | Delta state precision policy impacts stability. | bf16/fp16/fp32 state | Fewer drift errors | Throughput hit | No |
| CDB-07 | Delta Core | Delta residual gate controls over-write chaos across loops. | residual gate scalar/schedule | Better convergence | Under-updating | No |
| CDB-08 | Delta Core | Delta state norm clipping prevents runaway memory. | clip threshold | Robustness | Lost signal | No |
| CDB-09 | Delta Core | Periodic delta state reset improves long-run conditioning. | reset cadence | More stable training | Loses long memory | No |
| CDB-10 | Delta Core | Head-dim tensor-core alignment boosts Delta throughput. | aligned dims / head_dim | Faster kernels | Architecture constraints | No |
| CDB-11 | Crawler Loop | Instruction bottleneck size has optimal range. | `INST_DIM` sweep | Better loop routing | Under/overfit | Yes |
| CDB-12 | Crawler Loop | Loop-specific low-rank adapters beat fully shared core. | loop LoRA rank | BPB gain at small bytes | Params grow | No |
| CDB-13 | Crawler Loop | Split sharing (shared attn, modulated MLP) improves regime handling. | attn shared + MLP gates | BPB gain | Complexity | No |
| CDB-14 | Crawler Loop | Last loop partial unsharing captures final-pass specialization. | unshare depth=1 | BPB gain with low byte cost | Param creep | No |
| CDB-15 | Crawler Loop | Dual-rate loops (heavy every 2nd loop) improve quality/compute. | heavy cadence | Better speed-quality frontier | Scheduler bugs | No |
| CDB-16 | Crawler Loop | Adaptive loop count by confidence reduces wasted compute. | short/long bucket policy | Throughput gain | Control overhead | No |
| CDB-17 | Crawler Loop | Loop state carry with explicit damping improves fixed-point stability. | carry decay | Better convergence | Slower adaptation | No |
| CDB-18 | Crawler Loop | Loop dropout/stochastic depth improves shared-block generalization. | loop drop prob | Better robustness | Instability | No |
| CDB-19 | Crawler Topology | Memory tokens across loops add persistent workspace. | memory token count | Better long context | Extra compute | No |
| CDB-20 | Crawler Topology | Latent funnel recurrence (T->T/2 core) is superior at equal bytes. | funnel ratio | Speed or BPB gain | Complexity | No |
| CDB-21 | Crawler Topology | Encoder/decoder depth rebalance improves compression frontier. | flat/crawler split | Better byte-efficiency | tuning overhead | Yes |
| CDB-22 | Crawler Topology | Add tiny per-loop channel gates for activation alignment. | gate width | Better loop reuse | Small extra params | No |
| CDB-23 | Rust Data Path | Rust mmap shard reader reduces loader stalls. | `NITRUST_ENABLE` | Step-time drop | bridge overhead | Yes |
| CDB-24 | Rust Data Path | Strict mode catches silent Rust-path regressions early. | `NITRUST_STRICT` | Safer ops | hard fail risk | Yes |
| CDB-25 | Rust Data Path | Pinned host batcher improves H2D overlap. | prefetch depth, pinned on/off | Throughput gain | Memory pressure | Partial |
| CDB-26 | Rust Eval | Rust sliding-window index engine slashes eval wallclock. | window engine on/off | Faster eval | parity bugs | No |
| CDB-27 | Rust Export | Rust quant pack pipeline accelerates `.ptz` creation. | quantpack on/off | Faster export | bit-exact risk | No |
| CDB-28 | Runtime | CUDA graph replay cuts launch overhead on static smoke shapes. | graph on/off | Step-time drop | graph fragility | No |
| CDB-29 | Runtime | NUMA/affinity pinning lowers p95 jitter on multi-GPU hosts. | affinity profile | Stability gain | host variance | No |
| CDB-30 | Runtime | Online autotune for batch/prefetch finds hidden headroom. | autotune budget | extra throughput | tune noise | No |
| CDB-31 | Scheduling | Warmdown/EMA/GPTQ ordering matters for final int6 quality. | `SKIP_EMA`, warmdown, GPTQ mode | Better end BPB | confounding effects | Yes |
| CDB-32 | Scheduling | Distill-after-loop-aware-GPTQ may recover quantization loss. | distill flags + GPTQ mode | Better final BPB | extra time | No |

## Spark Smoke Queue (v0)

| Run ID | Ablation | Delta from baseline | Status |
|---|---|---|---|
| SMK-00 | Baseline smoke | Medusa smoke config, `NITRUST_ENABLE=0` | Completed: roundtrip `6.02582801`, sliding `5.97225220` |
| SMK-01 | Rust loader ON | `NITRUST_ENABLE=1`, `NITRUST_STRICT=1` | Completed: roundtrip `6.02584613`, sliding `5.97228266` |
| SMK-02 | Delta heads OFF | `DELTA_NET_HEADS=0` + Rust ON | Completed: roundtrip `4.91216360`, sliding `4.90379569` |
| SMK-03 | Crawler int8 OFF | `CRAWLER_QUANT_INT8=0` + Rust ON | Completed: roundtrip `6.02587901`, sliding `5.97224063` |
| SMK-04 | Instruction OFF | `INST_DIM=0` + Rust ON | Completed: roundtrip `6.00549835`, sliding `5.95337039` |

## Smoke Config Contract
- Tiny dataset clone in `/tmp/nitrust_smoke_data` (header-compatible shards)
- Single Spark GPU smoke (`NPROC=1` style run)
- `VAL_LOSS_EVERY=0` to avoid known step-0 eval/autograd conflict during smoke
- Early-stop via wallclock cap + tiny iteration budget

## Initial Spark Readout
- Rust loader ON (`SMK-01`) is numerically neutral vs baseline in smoke (difference in the 1e-5 range on BPB).
- `CRAWLER_QUANT_INT8=0` (`SMK-03`) is also neutral in this tiny smoke setup.
- `INST_DIM=0` (`SMK-04`) slightly improved smoke BPB, but this is low-confidence at smoke scale.
- `DELTA_NET_HEADS=0` (`SMK-02`) changed the task dynamics substantially and ran much faster; treat as topology sanity check, not a like-for-like quality verdict.
- Artifact logs/summary captured at `results/nitrust_spark_smoke_20260327_234343/`.
