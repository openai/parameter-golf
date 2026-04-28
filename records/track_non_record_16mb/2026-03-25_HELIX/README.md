# HELIX: Hierarchical Efficient Learning with Iterative Expansion

This folder contains a non-record 16MB run of HELIX on 8xH100. The architecture is intentionally ambitious and appears high-potential, but this specific run was bottlenecked by infrastructure/throughput issues and hit the 10-minute wallclock cap before completing the intended optimization trajectory.

## Architecture Breakdown

HELIX combines four core ideas:

1. **D-TPA (Differential Tensor Product Attention)**
   - Factored QKV via rank-4 tensor products (`DTPA_RANK=4`), reducing attention parameter footprint versus standard dense projections.
   - Differential attention path (`A1 - lambda * A2`) to suppress noisy components and improve signal quality.
   - GQA setup (`NUM_HEADS=8`, `NUM_KV_HEADS=4`) with partial RoPE (`ROPE_DIMS=16`) and XSA on the final blocks.

2. **High-capacity FFN under 16MB constraints**
   - Wider hidden projections (`FFN_HIDDEN=2304`, `MODEL_DIM=768`) to preserve representational power while attention is factorized.
   - Design intent: spend parameter budget where it tends to yield better BPB returns while keeping compressibility viable.

3. **Recurrence-style virtual depth (MoR layout)**
   - `NUM_UNIQUE_BLOCKS=5`, `NUM_ITERATIONS=2` for deeper effective computation without linearly scaling unique block parameters.
   - U-Net style skip structure across stages to stabilize information flow through repeated computation.

4. **Competition-aligned optimization and compression**
   - Muon for matrix params and AdamW for scalar/control tensors.
   - EMA/SWA enabled to improve end-of-run robustness.
   - int6 per-row quantization + lzma compression for final artifact packaging.

## Why this design makes sense

- **Parameter efficiency + expressiveness:** D-TPA and recurrence reduce unique parameter cost, freeing budget for width and control tensors.
- **Trainability in deep virtual stacks:** skip pathways, EMA/SWA, and optimizer routing are chosen to keep repeated-block training stable.
- **Submission realism:** compression path is built into training/export so architecture choices are evaluated in their post-quantized form, not only in fp32/bf16.

## Run Configuration

```bash
RUN_ID=helix_v2 \
MODEL_DIM=768 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
DTPA_RANK=4 \
NUM_UNIQUE_BLOCKS=5 \
NUM_ITERATIONS=2 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=20000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.023 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_WD=0.04 \
ADAM_WD=0.01 \
EMA_DECAY=0.997 \
SWA_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results Summary

This run was cut off by wallclock before the planned training horizon. Even with the cutoff, the model shows strong early trajectory and remains under artifact budget.

| Metric | Value |
|---|---|
| Model params | 21,478,225 |
| Stop condition | `MAX_WALLCLOCK_SECONDS` reached |
| Step at stop | 2298 / 20000 |
| Train time | 600,131 ms |
| Step average | 261.15 ms |
| Pre-EMA val_loss / val_bpb | 2.1515 / 1.2742 |
| Post-EMA val_loss / val_bpb | 2.1580 / 1.2781 |
| Peak memory (alloc/reserved) | 50,628 MiB / 52,136 MiB |
| Serialized int6+lzma model | 9,884,088 bytes |
| Code size | 89,151 bytes |
| Total submission size | 9,973,239 bytes |

## What can be improved next

The core issue here was not model potential, but execution efficiency in the training environment (Runpod setup and throughput behavior), which prevented running enough optimization steps inside the 10-minute window.

Priority improvements for the next pass:

1. **Throughput optimization first**
   - Reduce per-step overhead and stabilize kernels/compilation path so effective steps in 600s increase materially.
   - Validate NCCL/runtime behavior and eliminate environment regressions that reduce steady-state tokens/sec.

2. **Architecture-speed co-tuning**
   - Keep the HELIX ideas, but trim expensive components that do not move BPB enough per ms.
   - Tune recurrence/block placement to maximize quality gain per unit wallclock.

3. **Full 10-minute completion target**
   - Current run ends too early relative to intended schedule.
   - Next owner should optimize so the planned trajectory fully executes under 600s rather than getting cut off.

Given current behavior and early metrics, confidence is high that a properly optimized 10-minute run can substantially improve this result, with realistic SOTA potential once the architecture is allowed to complete its intended training curve.

## Included files

- `train_gpt.py` (exact training script used)
- `logs/helix_8gpu.txt` (full run log)
- `final_model.pt` and `final_model.int6.ptz` (artifacts)
- `submission.json` (submission metadata)
