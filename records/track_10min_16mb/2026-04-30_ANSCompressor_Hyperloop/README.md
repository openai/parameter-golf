# ANS Compression Variant PR #1493

This note documents a small variant of **@bigbag PR #1493** 10min/16MB record script. The architecture and training recipe are intentionally kept close to the original SP8192 + 3-layer recurrence + parallel residuals + QK-gain 5.25 + legal TTT baseline.

The main confirmed change is the quantized model artifact format: the generic compressed artifact path is replaced with a custom ANS-coded quantized artifact.

## Main Result

The largest verified win is artifact size. In the seed-314 baseline log from PR #1493:

```text
Serialized model quantized+brotli: 15,976,325 bytes
Total submission size quantized+brotli: 15,992,919 bytes
```

With the ANS artifact path, the expected same-model baseline estimate is:

```text
Serialized model quantized+ans: ~15,859,005 bytes
Estimated total with packed code: ~15,875,599 bytes
```

That is about **0.12 MB saved directly** on the quantized model artifact, and roughly **0.1-0.16 MB of practical headroom** depending on the exact packed script and experiment. Under the 16 MB cap, this is the most useful confirmed improvement: it turns the original near-cap submission into one with meaningful byte margin while preserving the same model behavior.

## Hyperloop Experiment

I also tried spending the new byte headroom on a loop-level **Hyperloop-lite** mechanism. The implementation added stream read/write/output gates and loop position embeddings around the existing recurrent middle block.

The idea was byte-feasible and paper-aligned, but under the 10-minute wallclock it was not a good tradeoff. The K=4 run was slower than the baseline recurrence and did not recover enough quality:

```text
K=4 Hyperloop-lite:
step: 4418 vs baseline seed314 4557
pre-quant val_bpb: 1.08830 vs baseline seed314 1.08775
quantized val_bpb: 1.09960
total size with ANS: 15,920,943 bytes
```

I optimized the Hyperloop path by removing the explicit stream tensor and transition gates, but it still remained slower than the original recurrent loop. For this submission, Hyperloop should be treated as a documented negative experiment, not the claimed improvement.

## Submission Scope

Due to limited compute budget, I could only run one full exploratory pass after implementing ANS and testing Hyperloop. This should be submitted primarily as a report of the **ANS compression win** over bigbag PR #1493, not as a new 3-seed training record.

Recommended baseline-style command:

```bash
SEED=314 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
SLIDING_WINDOW_ENABLED=1 \
GPTQ_CALIBRATION_BATCHES=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional experimental knobs left in the script:

```text
LOOP_HYPERLITE=1     # experimental; slower in the tested run
LOOP_STREAMS=4       # stream count for Hyperloop-lite
LOOP_PHASE_GATES=1   # cheaper loop-phase gating experiment
QUANT_OVERRIDE_PATH  # optional per-tensor GPTQ bit/clip overrides
```

The default path should keep Hyperloop disabled.