# Seed 42 Terminal Evidence

This file records the relevant terminal evidence available for the non-record Variable-Rank LQER + Muon-TTT experiment.

## Reproduction Run

Command shape:

```bash
LQER_VARIABLE_RANK=0 TTT_OPTIMIZER=adam PHASED_TTT_PREFIX_DOCS=2500 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/repro_seed42.log
```

Observed terminal summary:

```text
train_time: 599601ms
artifact_bytes: 15907913
diagnostic quantized val_bpb: 1.07275682
quantized_ttt_phased val_bpb: 1.06005735
total_eval_time: 527.2s
```

Result: valid reproduction. The final BPB was inside the planned reproduction window `[1.0585, 1.0615]`, artifact size was below 16,000,000 bytes, and eval time was below 600s.

## Stack Run

Command:

```bash
LQER_VARIABLE_RANK=1 LQER_VARIABLE_RANK_CAP=8 \
TTT_OPTIMIZER=muon TTT_MUON_NS_STEPS=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
RUN_ID=stack_seed42 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/stack_seed42.log
```

Selected hyperparameters:

```text
lqer_variable_rank: True
lqer_variable_rank_cap: 8
lqer_variable_rank_floor: 0
phased_ttt_prefix_docs: 3000
ttt_muon_ns_steps: 3
ttt_optimizer: muon
seed: 42
```

Training and byte audit:

```text
[byte-audit:eval_val] caseops sidecar tokens=100000 bytes=312067 mean_bytes=3.1207 OK
4902/20000 val_loss: 2.3544 val_bpb: 1.0758
stopping_early: wallclock_cap train_time: 599546ms step: 4902/20000
```

Pre/post quant diagnostics:

```text
diagnostic pre-quantization post-ema val_loss:2.32938557 val_bpb:1.06436919 eval_time:7476ms
Serialized model quantized+pergroup: 15880995 bytes
Total submission size quantized+pergroup: 15922566 bytes
diagnostic quantized val_loss:2.34885563 val_bpb:1.07326567 eval_time:11500ms
```

Variable-rank LQER allocation:

```text
[lqer-var] selected=[('blocks.0.attn.c_q.weight', 8), ('blocks.0.attn.c_k.weight', 8), ('blocks.0.attn.c_v.weight', 8), ('blocks.3.attn.c_k.weight', 8), ('blocks.3.attn.c_v.weight', 8), ('blocks.4.attn.c_k.weight', 1), ('blocks.0.mlp.proj.weight', 8)] storage_units=55198/55494 top_traces=[403089760.0, 122168128.0, 121836376.0, 68981448.0, 59564136.0, 58912428.0, 50331208.0, 50331208.0]
```

Final result:

```text
quantized_ttt_phased val_loss:2.32412035 val_bpb:1.06203230 eval_time:627269ms
total_eval_time:627.3s
```

Result: negative. The stack regressed versus the seed 42 reproduction and exceeded the 600s eval limit.
