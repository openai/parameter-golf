# FP16 Embed + MLP992 Sliding-Window Size-Repair Probe

This folder packages a local research run as a `track_non_record_16mb` candidate.

This is not a leaderboard-eligible `track_10min_16mb` claim. The run was executed on local `8x NVIDIA L20Z`, not the official `8xH100` evaluation environment. It is intended as a bounded, reproducible non-record submission that demonstrates a concrete under-16MB result from the current research branch.

## Key Idea

Start from the local `slide64` line, keep the tied token embedding in fp16 during the final int8+zlib export, and reduce the MLP width to `992` as a size-repair offset.

This run also used a bounded probe configuration:

- `ITERATIONS=1600`
- `MAX_WALLCLOCK_SECONDS=0`
- `VAL_LOSS_EVERY=0`
- `SKIP_TTT_EVAL=1`

So this should be read as a successful non-record research probe, not as a full leaderboard-style 10-minute H100 submission.

## Results

| Metric | Value |
|---|---:|
| Pre-quant val_loss | `2.2773` |
| Pre-quant val_bpb | `1.3488` |
| Post-quant roundtrip val_loss | `2.22232137` |
| Post-quant roundtrip val_bpb | `1.31618558` |
| TTT eval | skipped intentionally via `SKIP_TTT_EVAL=1` |
| Training stop | `1600/1600` |
| Training time | `143.948s` |
| Roundtrip eval time | `138.547s` |
| Code size | `65,628` bytes |
| Model size int8+zlib | `14,358,635` bytes |
| Total size int8+zlib | `14,424,263` bytes |
| Margin under 16MB cap | `1,575,737` bytes |

## Why This Is Interesting

- It is a clean successful run with `exit_code: 0`.
- It stays under the decimal `16,000,000` byte cap.
- It demonstrates that the `FP16 tok_emb export + MLP_HIDDEN=992` line can recover artifact size while preserving a strong sliding-window post-quant metric.
- It leaves a concrete, reproducible snapshot that can be rerun or adapted for future work.

## Exact Command

The original run was launched through the repo-local research wrappers. The equivalent standalone command for this submission snapshot is:

```bash
cd /newcpfs/user/qixuan1/research_proposal_lab/parameter-golf

RUN_ID=fp16emb_mlp992_slide64_size_repair_probe \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
TORCH_SHOW_CPP_STACKTRACES=1 \
CUDA_LAUNCH_BLOCKING=1 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=128 \
TTT_BATCH_SIZE=32 \
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight \
MLP_HIDDEN=992 \
ITERATIONS=1600 \
SKIP_TTT_EVAL=1 \
torchrun --standalone --nproc_per_node=8 records/track_non_record_16mb/2026-03-20_FP16Emb_MLP992_Slide64_SizeRepairProbe/train_gpt.py
```

For convenience, the same launch is wrapped in [run.sh](/newcpfs/user/qixuan1/research_proposal_lab/parameter-golf/records/track_non_record_16mb/2026-03-20_FP16Emb_MLP992_Slide64_SizeRepairProbe/run.sh).

## Provenance

- Source run directory:
  - [20260320T142517Z_baseline-sp1024-8gpu-sync-full-slide64-tttbs32-fp16emb-mlp992-probe1600-noval-skipttt](/newcpfs/user/qixuan1/research_proposal_lab/parameter-golf/research_agent/workspace/logs/8gpu_baseline/20260320T142517Z_baseline-sp1024-8gpu-sync-full-slide64-tttbs32-fp16emb-mlp992-probe1600-noval-skipttt)
- Source train log copied into this folder:
  - [train.log](/newcpfs/user/qixuan1/research_proposal_lab/parameter-golf/records/track_non_record_16mb/2026-03-20_FP16Emb_MLP992_Slide64_SizeRepairProbe/train.log)
- Code snapshot copied into this folder:
  - [train_gpt.py](/newcpfs/user/qixuan1/research_proposal_lab/parameter-golf/records/track_non_record_16mb/2026-03-20_FP16Emb_MLP992_Slide64_SizeRepairProbe/train_gpt.py)

## Included Files

- `train_gpt.py`: code snapshot used for this packaged submission
- `train.log`: copied exact training/eval stdout log
- `submission.json`: metadata summary
- `run.sh`: convenience launcher for this folder snapshot

## Notes

- This package is aimed at the non-record 16MB track.
- It should be described honestly in any PR as local `8xL20Z` evidence, not as official `8xH100` leaderboard evidence.
- The run deliberately skipped TTT evaluation; only the post-quant sliding-window roundtrip metric should be treated as the headline score.
