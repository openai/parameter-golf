# PR1493 Stacking Plan

This note documents the current stacking hypothesis after the priority runs.

## Current State

Known comparator:

```text
PR1493 seed 42, QK_GAIN_INIT=5.25:
  quantized_ttt = 1.08103358

Best tuned TTT:
  TTT_LR=0.007, TTT_EPOCHS=5
  quantized_ttt = 1.08079274

WD schedule + tuned TTT:
  quantized_ttt = 1.08029
```

So `WD_SCHEDULE_ENABLED=1` is a confirmed small win on seed 42:

```text
gain vs tuned TTT = about -0.00050 BPB
gain vs raw PR1493 = about -0.00074 BPB
```

That is enough to be numerically below `1.0810` on this seed, but it is not enough to be record-safe. The acceptance bar is closer to `0.005 nats/token`, which is approximately `0.0017-0.0020 BPB` at this tokenizer's bytes/token. We probably still need roughly another `0.0012 BPB`.

## What Is Stackable

### Keep: tuned TTT

This is already folded into all priority commands:

```bash
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5
```

There may still be a small retune after WD/paired-head Muon, but do not count the original TTT gain twice.

### Keep: WD schedule

WD improves the trained EMA model:

```text
baseline pre = 1.08757
WD pre       = 1.08650
gain         = -0.00107 BPB
```

The clean non-sliding quant gap did not improve:

```text
baseline q - pre = 1.10014 - 1.08757 = 0.01257
WD q - pre       = 1.09951 - 1.08650 = 0.01301
```

Interpretation: WD is a pre-quant / training-quality win, not a quantization win. That is good for stacking because a quant-robustness change can still add value.

### Test Next: paired-head Muon

Prior paired-head Muon results suggested little/no pre-quant gain but a lower post-quant gap:

```text
pre-quant:      roughly unchanged
INT6 sliding:   about -0.002 BPB
quant gap:      about -0.002 BPB
```

That is the kind of win WD did not provide. If even half transfers to PR1493 + WD, it could push us into a much stronger range:

```text
WD current:       1.08029
half transfer:    1.0793-ish
full transfer:    1.0783-ish
```

The new PR1493 implementation is env-gated:

```bash
PAIRED_HEAD_MUON_ENABLED=1
```

It tags only Q/K matrices for paired Newton-Schulz. V and output projection remain standard NS, matching the old banked implementation's semantics.

### Test After That: fixed IHA

The earlier IHA run failed during GPTQ because Hessian hooks were attached to `nn.Linear.forward`, but IHA bypassed those modules via inline `F.linear`.

The current fix folds trained IHA mixes into the underlying Q/K/V weights before GPTQ:

```text
q_weight <- q_mix @ q_weight_heads
k_weight <- k_mix @ k_weight_heads
v_weight <- v_mix @ v_weight_heads, only when IHA_MIX_V=1
```

Then `attn.iha_enabled=False` for the serialized training model so GPTQ Hessian collection sees normal `self.c_q(x)` / `self.c_k(x)` / `self.c_v(x)` calls again. The serialized state still contains identity mix tensors, so loading remains strict.

IHA is not a confirmed win. Treat it as a candidate only after `paired` and `wd_paired` are measured.

## What Not To Stack

### Drop: docshuffle

Result:

```text
q_ttt = 1.08279
delta vs tuned TTT = +0.001997 BPB worse
```

This is not mostly a step-count issue. Baseline stopped around `4555-4559`; docshuffle stopped at `4526`. That is less than 1% fewer steps. The likely failure is distribution mismatch from document-level reshuffling/rechunking.

### Drop: current MTP

Result:

```text
pre   = 1.11283
q_sw  = 1.11018
q_ttt = 1.09023
```

This implementation uses the same logits for both `t+1` and `t+2`, which creates contradictory supervision. Proper MTP would need separate auxiliary heads that are discarded before serialization. Do not stack this implementation.

### Still Dead: QAT, PKO, mixed precision

Previous runs already made these non-priority:

```text
QAT: net negative
PKO: breaks TTT
mixed precision: byte budget too tight
```

## Run Order

Run these in this order on seed 42:

```bash
# 1. Paired-head Muon alone
RUN_ID=pr1493_paired_s42 SEED=42 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
PAIRED_HEAD_MUON_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 2. WD + paired-head Muon
RUN_ID=pr1493_wd_paired_s42 SEED=42 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 3. Stronger WD schedule, only if default WD still looks good
RUN_ID=pr1493_wd_strong_s42 SEED=42 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.50 WD_SCHED_HIGH_FACTOR=1.75 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 4. WD + paired-head Muon + fixed IHA
RUN_ID=pr1493_wd_paired_iha_s42 SEED=42 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 IHA_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 5. WD + eval-only recurrence, if evalloop3 alone is positive
RUN_ID=pr1493_wd_evalloop3_s42 SEED=42 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 EVAL_NUM_LOOPS=3 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py
```

## How To Interpret

Primary metric:

```text
quantized_ttt val_bpb
```

Secondary metrics:

```text
pre-quantization post-ema val_bpb
quantized val_bpb
quantized_sliding_window val_bpb
clean quant gap = quantized - pre-quantization post-ema
TTT gain = quantized_sliding_window - quantized_ttt
```

Critical reads:

- If paired-head Muon improves `q_sw` but not `pre`, that is the expected useful pattern.
- If paired-head Muon improves `q_sw` but TTT gives less gain, retune TTT before rejecting it.
- If WD+paired is not better than WD, the old paired-head win did not transfer to this architecture.
- If IHA improves pre but worsens quantized/TTT, it is not stackable.
- If any result is less than `0.0002 BPB` better, rerun before trusting it.

## Submission Caveat

These are experiment scripts. Current code size is too large for final submission. Any winning stack still needs minification/packing and a fresh total-size check under `16,000,000` bytes.
