# Non-record: StageB Scalar-Control Quantization Without TTT

This is a conservative single-seed non-record fallback. It reports the under-cap scalar/control-tensor quantized StageB artifact without claiming the later exploratory phased-TTT score.

The associated rank-128 TTT experiment reached `1.06168366` BPB, but independent review found that its phased global-update path reorders validation documents before updates. That TTT score is therefore held out of this submission. The claimed score here is the no-TTT full-validation diagnostic score from the same under-cap artifact: `1.07468466` BPB.

## Result

| Seed | val_bpb | val_loss_nats | Train timer | Eval timer | Compressed model | Counted total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.07468466 | 2.35196111 | 592.136s internal | 11.096s diagnostic | 15,962,574 bytes | 15,994,521 bytes |

Relevant evidence:

- `val_tokens: 47851520`
- training stopped at `stopping_early: wallclock_cap train_time: 592136ms step: 4844/20000`
- scalar/control requant reports `Serialized model quantized+brotli: 15962574 bytes`
- scalar/control requant reports `Total submission size quantized+brotli: 15994574 bytes`
- after replacing the non-reproducible `pyminify` CLI shell-out with the declared `python_minifier` API and defaulting this package to no TTT, local code-byte recomputation gives `31,947` bytes and a counted total of `15,994,521` bytes
- no-TTT diagnostic reports `requant diagnostic quantized val_loss:2.35196111 val_bpb:1.07468466 eval_time:11096ms`

Important caveats:

- This is not a SOTA record claim and is behind the current accepted frontier.
- It is a single-seed non-record package.
- The initial StageB training artifact was over cap before scalar/control-tensor requantization; this folder documents the under-cap reserialization path.
- The controller measured `artifact_production_wallclock: 1098.443s` around the exploratory train+artifact flow, so this should not be represented as a clean 10-minute record.
- The script still contains the held exploratory TTT implementation, but the submitted score and metadata do not depend on it.

## Comparison

| Entry | Status | BPB | Notes |
| --- | --- | ---: | --- |
| PR #1493 | stale accepted reference in local checkout | 1.0810 | older official row in this checkout |
| This package | local non-record | 1.07468466 | one seed, full validation, under cap |
| PR #1787 | accepted current frontier row | ~1.06335 | lower than this package |
| PR #1851/#1868 | accepted current frontier row | ~1.0614 | lower than this package |
| PR #1855 | accepted current top row | ~1.06108 | lower than this package |

## Reproduction

Run environment:

- Hardware: 8x NVIDIA H100 SXM 80GB
- PyTorch: 2.9.1+cu128
- Process count: `torchrun --standalone --nproc_per_node=8`
- Data: `sp8192` CaseOps cached FineWeb challenge data
- Tokenizer: `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`

Representative command for the no-TTT path:

```bash
RUN_ID=stageb_scalar_controlq_nottt_seed42 \
TRAIN_SHARDS=80 \
MAX_WALLCLOCK_SECONDS=600 \
NPROC_PER_NODE=8 \
SEED=42 \
CASEOPS_ENABLED=1 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_SCALE=0.75 \
FUSED_MLP_ENABLED=1 \
MIN_LR=0.1 \
WARMDOWN_FRAC=0.82 \
QK_GAIN_INIT=5.125 \
MATRIX_BITS=6 \
EMBED_BITS=7 \
LQER_ENABLED=1 \
LQER_RANK=4 \
LQER_TOP_K=1 \
LQER_FACTOR_BITS=4 \
LQER_ASYM_ENABLED=1 \
SCALAR_QUANT_ENABLED=1 \
GATED_ATTN_QUANT_GATE=1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=11.75 \
MATRIX_CLIP_SIGMAS=12.65 \
ATTN_CLIP_SIGMAS=13.0 \
GPTQ_CALIBRATION_BATCHES=20 \
GPTQ_RESERVE_SECONDS=8.0 \
TTT_ENABLED=0 \
NGRAM_MIX_ALPHA=0 \
COMPRESSOR=brotli \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee train_seed42.log
```

## Files

- `train_gpt.py`: exploratory script with the code-byte helper patched to use the declared `python_minifier` API instead of an undeclared `pyminify` CLI, and with no-TTT as the default for this package.
- `train_seed42.log`: StageB training and initial quantization evidence.
- `requant_nottt_seed42.txt`: no-TTT under-cap scalar/control requantization evidence.
- `scalar_scout_summary.jsonl`: machine-readable scalar scout results.
- `stageb_summary.json` and `setup.log`: controller/setup evidence.
- `requirements.txt`: declared dependency surface.

## Hashes

- `train_gpt.py`: `58f054f2194959f670ed105bbb591473bfcd41a2510cbfafd7e78ff7f78fc842`
- scored `final_model.int6.ptz`: `0204d242b4e963396ed61206f55e7bc32b4095f5bfdf43a59b55cc26661a74d3`
- `requant_nottt_seed42.txt`: `e31a8d022f0e28d6ae1640ea417a4fd0d0a978baaca09402d4c4d86abf0f0016`
- `train_seed42.log`: `ef9619d7fc179163838bd7f40f0ecd9c92e70bbc63b08c701130c450125f7c47`

## Compliance Notes

- Track label: non-record 16 MB.
- Artifact size: 15,994,521 counted bytes after local code-byte recomputation, under the decimal 16,000,000-byte cap by 5,479 bytes.
- Full validation: logs report `val_tokens: 47851520`.
- No TTT score is claimed for this package.
- No casefold or byte-level PPM is used.
- No n-gram scoring path is active: `NGRAM_MIX_ALPHA=0`.
- Evaluation does not require network access or external data downloads.

## Tracking

- Flywheel primary node: `cc0b4d0c-78c0-40e7-9b26-4083ab544461`
- Flywheel experiment node: `1450c4a2-7893-4a09-ab33-c5b4bee7e380`
- Local manifest run IDs: `stageb_scalar_scout_seed42_20260501T1444AEST`, `stageb_scalar_scout_seed42_20260501T1444AEST_stop_20260501T1457AEST`
- Local artifact evidence: `runs/stageb_train_20260501T1235AEST/local_results/scalar_lqtop1_embed150`
