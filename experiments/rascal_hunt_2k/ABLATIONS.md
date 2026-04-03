# Rascal 2k Hunt

- `ctrl`: safepoint control
- `gptq`: turn GPTQ on
- `warm4k`: extend warmdown to 4000
- `ngram5`: causal n-gram eval order 5
- `ngram7`: causal n-gram eval order 7
- `qkgain4`: stronger QK gain init
- `bigram2816`: moderate bigram expansion
- `bigram3072`: larger bigram table
- `qk4_gptq`: QK gain plus GPTQ
- `qk4_bigram2816`: QK gain plus moderate bigram expansion
- `qk4_warm4k`: QK gain plus longer warmdown
- `qk4_ngram7`: QK gain plus stronger causal n-gram eval
- `qk4_gptq_bigram2816`: strongest near-lineage train/eval combo
- `frontier_combo`: QK gain + GPTQ + bigram2816 + warm4k

Runner:

- [run_frontier_hunt.sh](/home/frosty40/parameter-golf-lab/experiments/rascal_hunt_2k/run_frontier_hunt.sh)
- [run_signal_hunt_4gpu.sh](/home/frosty40/parameter-golf-lab/experiments/rascal_hunt_2k/run_signal_hunt_4gpu.sh)

Modes:

- `short`: `ctrl qkgain4 bigram2816 gptq qk4_gptq`
- `eval`: `ctrl ngram5 ngram7 qk4_ngram7`
- `long`: full sweep

Cheap signal runner:

- `build`: `ctrl qkgain4 bigram2816 gptq qk4_gptq qk4_bigram2816`
- `warm`: `warm1000 warm1500 warm2000`
- `all`: build block first, warmdown block after

Notes:

- `run_signal_hunt_4gpu.sh` defaults to `4` GPUs
- `ITERATIONS=2000`
- `SKIP_FINAL_EVAL=1` to save money while still logging post-EMA and serialized size
- model-build cases use `WARMDOWN_ITERS=0`
- warmdown cases are intentionally clustered at the end
- warmdown is scaled for the 2k proxy:
  - `warm200` ~= full-run `2000`
  - `warm350` ~= full-run `3500`
  - `warm500` ~= full-run `5000`
