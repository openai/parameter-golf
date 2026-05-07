# Plan C Terminal Observation

The final Plan C run was launched on an 8xH100 pod with:

```bash
BIGRAMHASH_PATCH=1 PATH_A_V3_SMALL=1 \
BIGRAM_VOCAB_SIZE=512 BIGRAM_DIM=4 BIGRAM_BITS=6 \
BASE_STACK=2018 \
  bash /workspace/pod_move39_gate_scout.sh \
  run_split 1337 12 12 planC_2018_native_bigram512d4_patha
```

The remote SSH connection closed before the run directory could be copied back.
For that reason this is recorded only as a terminal-observed negative result,
not as a complete logged artifact.

Key terminal-observed configuration lines:

```text
bigram_bits: 6
bigram_dim: 4
bigram_vocab_size: 512
path_a_v3_small: True
gate_window: 12
smear_gate_window: 12
model_params:35949858
```

Key terminal-observed pre-quant result:

```text
stopping_early: wallclock_cap train_time: 596111ms step: 4837/20000
diagnostic pre-quantization post-ema val_loss:2.33019926 val_bpb:1.06471733 eval_time:12455ms
```

The run failed the pre-quant kill gate versus PR #2018 seed-1337 reference
pre-quant BPB `1.05124428`.  I therefore did not treat it as worth further
quantized/TTT evaluation.
