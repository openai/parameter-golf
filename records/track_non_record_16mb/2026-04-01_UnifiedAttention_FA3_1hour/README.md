# Non-Record: Unified Attention + FA3 + Legal TTT (1-hour training)

**val_bpb: 1.1088** | **~15.82 MB** | 8×H100 SXM | 1 hour training

Same architecture as our record submission [PR #1202](https://github.com/openai/parameter-golf/pull/1202) (val_bpb 1.1412, 10-min). This run trains for 1 hour to explore unified attention's scaling behavior with unlimited compute.

## Results

| Run | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | Training time |
|-----|----------|-------|-------------|-----------------|---------------|
| Record (PR #1202, 3-seed mean) | 49.6ms | 12,100 | 1.1643 | **1.1412** | 10 min |
| **This run (1 hour)** | 51.3ms | 72,000 | 1.1326 | **1.1088** | 60 min |
| **Improvement** | | +59,900 steps | -0.0317 | **-0.0324** | |

Beats the current unlimited compute SOTA (1.1239, 1-bit quantization, 2hr training) by 0.015 BPB in half the training time.

## What's Different From the Record Submission

Only the training schedule changes. Architecture and eval are identical:

| Parameter | Record (10 min) | This run (1 hour) |
|-----------|----------------|-------------------|
| ITERATIONS | 20,000 (wall-clock limited) | 72,000 |
| MAX_WALLCLOCK_SECONDS | 600 | 3,600 |
| WARMDOWN_ITERS | 3,500 | 10,000 |
| QAT_START_FRACTION | 0.15 | 0.85 |
| EMA_DECAY | 0.997 | 0.997 |
| Everything else | identical | identical |

The key change: with more steps, we can train clean (no QAT noise) for 85% of the run and still give QAT 10,800 steps to converge. The 10-min run needs QAT at 15% to fit enough QAT steps in the budget.

## Scaling Observation

The model plateaus at peak LR around step 48,000 (val_bpb ~1.223). The real gains come from the warmdown phase (steps 62,000-72,000) where the LR decays and the model refines. With 10,000 warmdown steps (vs ~1,100 in the 10-min run), the model has 9x more refinement steps.

Pre-warmdown base quality:
- 10-min run at step 10,000: val_bpb ~1.248
- This run at step 48,000: val_bpb ~1.223

Post-warmdown + quantization:
- 10-min run: 1.1647 mixed roundtrip
- This run: 1.1326 mixed roundtrip

The warmdown benefit scales with more steps, and unified attention benefits from longer training just as standard architectures do.

## Key Innovation: Unified Attention

Unified Attention ([Deshwal, 2026](https://github.com/ReinforceAI/yocto)) replaces separate Q/K/V projections with a single W_unified matrix. 67% fewer attention projection parameters, reallocated to the MLP. Attention is a routing mechanism; the FFN does the heavy lifting. In the 16 MB budget, we trade 2.28 MB of routing for 2.49 MB of computation.

See [PR #1202](https://github.com/openai/parameter-golf/pull/1202) for full architecture details, ablation, and negative results.

## Requirements

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280
pip install sentencepiece zstandard
```

## Run Command

```bash
RUN_ID=r1_k11_d528_fa3_1hour \
ITERATIONS=72000 \
MAX_WALLCLOCK_SECONDS=3600 \
WARMDOWN_ITERS=10000 \
QAT_START_FRACTION=0.85 \
EMA_DECAY=0.997 \
NUM_UNIQUE_LAYERS=11 MODEL_DIM=528 NUM_HEADS=4 \
VE_LAYERS=9,10 \
TRAIN_BATCH_TOKENS=524288 \
SLIDING_WINDOW_EVAL=0 \
LEGAL_TTT_EPOCHS=3 \
TTT_LORA_ATTN=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

Same as [PR #1202](https://github.com/openai/parameter-golf/pull/1202). Unified Attention and FA3 head-dim padding are this work (Viraj Deshwal, Reinforce AI). All other techniques credited in the record submission README.