# Crawler Leg 1 Ablation Grid (Delta OFF)

| ID | Goal | Knobs | Keep Fixed | Success Signal |
|---|---|---|---|---|
| CL1-00 | Baseline | `CRAWLER_LOOPS=4`, `INST_DIM=32`, `CRAWLER_MLP_MULT=4.0` | `DELTA_NET_HEADS=0`, `SKIP_GPTQ=1`, `NGRAM_EVAL_ORDER=0` | anchor metrics |
| CL1-01 | Loop depth | `CRAWLER_LOOPS=3` | CL1-00 otherwise | speed up with small/no BPB loss |
| CL1-02 | Loop depth | `CRAWLER_LOOPS=5` | CL1-00 otherwise | BPB gain with tolerable speed cost |
| CL1-03 | Instruction off | `INST_DIM=0` | CL1-00 otherwise | detect instruction necessity |
| CL1-04 | Narrow inst | `INST_DIM=16` | CL1-00 otherwise | similar BPB at lower complexity |
| CL1-05 | Wider inst | `INST_DIM=64` | CL1-00 otherwise | improved loop specialization |
| CL1-06 | Narrow crawler MLP | `CRAWLER_MLP_MULT=3.0` | CL1-00 otherwise | speed gain with small BPB change |
| CL1-07 | Wide crawler MLP | `CRAWLER_MLP_MULT=5.0` | CL1-00 otherwise | BPB gain if width-limited |
| CL1-08 | Quant policy | `CRAWLER_QUANT_INT8=0` | CL1-00 otherwise | quality sensitivity to quant policy |
| CL1-09 | Depth split | `NUM_FLAT_LAYERS=5`, `NUM_CRAWLER_LAYERS=1` | loops/inst fixed | quality vs parameter tradeoff |
| CL1-10 | Depth split | `NUM_FLAT_LAYERS=3`, `NUM_CRAWLER_LAYERS=2` | loops/inst fixed | bottleneck recurrence strength |

## Run Command Template

```bash
SEED=1337 NPROC_PER_NODE=8 \
CRAWLER_LOOPS=4 INST_DIM=32 CRAWLER_MLP_MULT=4.0 CRAWLER_QUANT_INT8=1 \
NUM_FLAT_LAYERS=4 NUM_CRAWLER_LAYERS=1 \
bash experiments/Crawler_Leg_1/run.sh
```
