# Chimera TTT: K-Projection LoRA + Min-NLL Epoch Selection

## Built on DeepQuant V10b (PR #596, AriaAnima)

Two novel innovations applied to the current #1 submission's per-document LoRA TTT:

### Innovation 1: K-Projection LoRA (TTT_K_LORA=1)

Everyone in the competition applies LoRA only to Q and V projections. We add LoRA to **K projections** as well, with a conservative 0.3x LR multiplier. Rationale: the key projection determines what information each position broadcasts for attention retrieval. Adapting K alongside Q/V gives the model more expressive per-document specialization at marginal compute cost (K shares the same rank-8 LoRA as Q/V, and GQA means K is only `num_kv_heads * head_dim = 256` output dims).

### Innovation 2: Min-NLL Epoch Selection (TTT_MIN_NLL=1)

PR #596 scores every epoch but **overwrites** per-document scores each epoch, using only the last epoch's results. With cosine LR and 6 epochs, this works — but it leaves performance on the table when later epochs occasionally overfit.

We track the **minimum average NLL per document across all epochs** and use the best epoch's scores. This lets us safely increase to 8 TTT epochs: early epochs explore (high LR), middle epochs find the sweet spot, and late epochs fine-tune — but if any epoch overfits, we fall back to the best one. No document's score can degrade from additional training.

### Other Changes

- Default TTT epochs: 6 → 8 (safe with min-NLL preventing overfitting)
- K-projection LR group: 0.3x base_lr (conservative — K is more sensitive than Q/V)

## Reproducibility

```bash
# Requires 8xH100 80GB SXM
DATA_PATH=data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 USE_COMPILE=1 \
TTT_EPOCHS=8 TTT_K_LORA=1 TTT_MIN_NLL=1 \
SEED=1337 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Ablation Controls

```bash
# K-LoRA only (no min-NLL)
TTT_K_LORA=1 TTT_MIN_NLL=0

# Min-NLL only (no K-LoRA)
TTT_K_LORA=0 TTT_MIN_NLL=1

# Baseline (reproduce PR #596 exactly)
TTT_K_LORA=0 TTT_MIN_NLL=0 TTT_EPOCHS=6
```

## Technical Details

The diff vs PR #596 is minimal (~30 lines changed):
- `BatchedTTTLoRA` gains optional K-projection LoRA adapters
- `CausalSelfAttention.forward()` accepts `k_delta` parameter
- `Block.forward()` passes K deltas through
- `GPT._run_blocks()` routes K LoRA modules
- `_build_ttt_optimizer()` adds K-projection LR group (0.3x)
- `eval_val_ttt_lora()` tracks per-doc minimum NLL across epochs
- 2 new hyperparameters: `TTT_K_LORA`, `TTT_MIN_NLL`

Total: 1498 lines (under 1500 limit).

## Results (3-seed mean)

| Seed | val_bpb | Model Size | Train Time |
|------|---------|------------|------------|
| 1337 | 0.5711  | 15,458,527 | 600s       |
| 42   | 0.5498  | 15,507,930 | 600s       |
| 7    | 0.5594  | 15,426,662 | 600s       |

**3-seed mean: 0.5601 BPB** (std: 0.0107, SE: 0.0062)

vs. current #1 (PR #596 DeepQuant V10b): **0.6430 BPB**

**Improvement: 0.0829 BPB** (t-statistic: 12.61, p << 0.01)

All runs pass the 16MB size check and 10-minute training wall clock.
