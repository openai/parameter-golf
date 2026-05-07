# SP8192 Apex Stack Over-Cap Evidence

This is a non-record evidence submission, not a leaderboard/SOTA claim.

The run produced our best observed score, but the serialized artifact is over the 16,000,000 byte cap. I am including it because the score is useful evidence for the direction, while clearly marking it as not eligible for leaderboard placement unless a future packaging-only rescue brings the same artifact under cap.

## Result

- Final TTT score: `val_loss=2.75555988`, `val_bpb=1.06676310`
- Post-quant, pre-TTT score: `val_loss=2.78386351`, `val_bpb=1.07768558`
- Pre-quant post-EMA score: `val_loss=2.76302121`, `val_bpb=1.06961714`
- Model bytes: `16,383,814`
- Code bytes: `32,124`
- Total bytes: `16,415,938`
- Cap status: over by `415,938` bytes
- Train stop: step `4934/20000`, `599,664 ms`
- TTT eval time: `478,881 ms`

## Why This Is Non-Record

The Parameter Golf cap is decimal `16,000,000` total bytes for compressed model plus code. This run totals `16,415,938` bytes, so it is not a valid leaderboard record despite the strong BPB.

I attempted lossless packaging rescues that would preserve BPB:

- outer `xz -9e` and `zstd -19` on the existing artifact: larger than the original
- Linux `lrzip -z -L9` recompression of streams: deterministic, no material saving
- row-sorted MLP projection stream: only about `2,250` bytes saved before extra permutation bytes, so net not useful
- combined raw group stream: `16,203,310` bytes for the grouped weights, larger than the existing seven-stream layout

Because these did not close the `415,938` byte gap, this package is submitted honestly as over-cap evidence.

## Under-Cap Fallback Evidence

An export-only int5 matrix rescue was also run from the same checkpoint. It fits the cap, but it changes quality:

- Quantized score without TTT: `val_loss=2.87928093`, `val_bpb=1.11462345`
- Model bytes: `12,376,953`
- Code bytes: `32,237`
- Total bytes: `12,409,190`

That fallback is useful as evidence that the architecture can be packaged under cap, but it is not the best-BPB result and is not the result claimed here.

## Approach

This run combines an SP8192 tokenizer setup with an 11-layer 512-dim transformer, grouped-query attention, depth recurrence, sparse attention gating, smear gate, LQER asymmetric correction, mixed GPTQ quantization, per-group lrzip compression, and score-first phased TTT.

The run used cached SP8192 FineWeb data with CaseOps disabled. This avoided spending the 8xH100 run building custom data and kept the paid run focused on training, quantization, packaging, and evaluation.

## Reproduction

Run from the repository root on the official Parameter Golf 8xH100 image:

```bash
POD_ID=<runpod-pod-id> \
ACTIVE_MANIFEST=triage/promoted_runs/apex_stack_fast_sp8192_2026-04-30/apex_stack_fast_sp8192_seed42.json \
ACTIVE_RUN_ID=apex_stack_fast_sp8192_seed42_fix1 \
bash triage/promoted_runs/apex_stack_fast_sp8192_2026-04-30/run_via_runpod_apex_stack_fast_sp8192.sh
```

Important environment settings from the manifest:

```text
SEED=42
VOCAB_SIZE=8192
CASEOPS_ENABLED=0
COMPRESSOR=pergroup
MIN_LR=0.1
WARMDOWN_FRAC=0.85
BETA2=0.99
QK_GAIN_INIT=5.0
TTT_CHUNK_SIZE=48
TTT_LORA_RANK=80
PHASED_TTT_PREFIX_DOCS=2500
PHASED_TTT_NUM_PHASES=3
GPTQ_RESERVE_SECONDS=0.5
GPTQ_CALIBRATION_BATCHES=16
SPARSE_ATTN_GATE_ENABLED=1
GATE_WINDOW=12
SMEAR_GATE_ENABLED=1
LQER_ENABLED=1
LQER_ASYM_ENABLED=1
LQER_RANK=4
LQER_FACTOR_BITS=4
MAX_WALLCLOCK_SECONDS=600
```

Hardware used:

- RunPod official Parameter Golf image
- 8x NVIDIA H100 80GB HBM3
- PyTorch `2.9.1+cu128`

The exact log is included as `train.log`.
