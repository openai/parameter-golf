# 8-GPU Parallel Eval-Time Innovation Experiments

## Approach
1. Train baseline ONCE on single GPU (PR #834 exact, ~20 min with eval)
2. Save trained model artifact
3. Run 8 eval-only A/B tests in parallel (one per GPU, ~5-10 min each)
4. Compare results → winner becomes new baseline
5. Repeat with new eval innovations

All runs use wandb with descriptive names: `round{N}_gpu{G}_{method}`

## Baseline Command (PR #834 exact, single GPU)
```bash
CUDA_VISIBLE_DEVICES=$GPU \
PATH=/data/backups/rganapa/pylibs/bin:$PATH \
PYTHONPATH=/data/backups/rganapa/pylibs \
TMPDIR=/data/backups/rganapa/tmp \
TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache \
TORCH_HOME=/data/backups/rganapa/torch_home \
DATA_PATH=data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
TTT_CHUNK_TOKENS=1048576 \
TTT_EPOCHS=4 \
TTT_LR=0.0005 \
TTT_FREEZE_BLOCKS=2 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=98304 \
python3 pr834_train_gpt.py
```
Note: Single GPU uses TRAIN_BATCH_TOKENS=98304 (786432/8) to match per-GPU batch size.
No torchrun needed — script handles single GPU when WORLD_SIZE not set.

## Baseline
- PR #834 8-GPU full train+eval: **0.1591 BPP** (on our hardware)
- Single-GPU trained model + 8-GPU eval-only: **0.1776 BPP** (fewer training steps)
- Eval time: **568s TTT+n-gram** (within budget)
- Key env vars: `TTT_ONLY=1 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=4`
- Model saved at: `final_model.pt` + `final_model.int6.ptz`

## Experiments

### GPU 0: LZ-based Prediction (Lempel-Ziv dictionary matching)
- Instead of fixed n-gram orders, use LZ dictionary to find longest matching context
- Can capture arbitrarily long repeated sequences (20+ tokens)
- Replace hash tables with a suffix-array-like structure
- **Hypothesis**: catches patterns n-gram misses (long templates, code blocks)
- **Status**: QUEUED

### GPU 1: PPM-D Escape Probabilities
- Replace raw count ratios with PPM-D escape mechanism
- `p_seen = count / (total + unique)`, `p_escape = unique / (total + unique)`
- Recursive blending across orders instead of hard backoff
- **Hypothesis**: better probability estimates for sparse contexts
- **Status**: QUEUED

### GPU 2: Online Calibration (momentum-EMA frequency correction)
- Track running frequency of each token, correct neural model's systematic bias
- `p_corrected = p_model * (freq_observed / freq_predicted)`
- Inspired by PR #851's "online logit calibration"
- **Hypothesis**: fixes systematic over/under-prediction
- **Status**: QUEUED

### GPU 3: Higher Order (order=12, 8M buckets)
- Push n-gram order from 7 to 12 with larger hash tables
- More specific context matches
- **Hypothesis**: diminishing but real gains from longer context
- **Status**: QUEUED

### GPU 4: Finer Chunks (32K instead of 65K)
- PR #840 showed 65K chunks gave 0.17 BPP improvement
- Even finer (32K) might help more — cache updates more frequently
- **Hypothesis**: more frequent cache updates = better predictions
- **Status**: QUEUED

### GPU 5: Conformal Aggregation
- Replace sigmoid alpha with conformal prediction weights
- Adapt mixing based on recent prediction errors (not just entropy)
- Theoretically optimal under mild assumptions
- **Hypothesis**: better than hand-tuned/learned alpha
- **Status**: QUEUED

### GPU 6: Document Boundary Detection + Cache Reset
- Detect document boundaries in the eval stream
- Reset or downweight n-gram cache at boundaries
- Cross-document n-grams are noise, intra-document are signal
- **Hypothesis**: cleaner cache = better predictions within documents
- **Status**: QUEUED

### GPU 7: Bayesian Mixture (CTW-inspired)
- Context Tree Weighting: optimally mix variable-length contexts
- Bayesian model averaging with KT-estimator per context node
- Provably minimax optimal for tree sources
- **Hypothesis**: theoretically optimal mixing > learned head
- **Status**: QUEUED

## Results Table

| GPU | Method | BPP | Delta vs baseline | Time | Notes |
|-----|--------|-----|-------------------|------|-------|
| — | Baseline (PR #834, 8-GPU full) | 0.1591 | — | 675s | TTT_CHUNK=1M, TTT_EPOCHS=4 |
| — | Baseline eval-only (single-GPU model) | 0.1776 | +0.019 | 568s | Fewer training steps |
| X | 4M hash buckets (eval-only) | 0.3868 | **+0.21 WORSE** | 623s | Modified script broke routing head |
| 1 | TTT_EPOCHS=8 (full 8-GPU) | 0.1655 | +0.006 WORSE | 703s | Overfits per chunk, slower. 4ep is optimal. |
| 2 | PPM-D escape probs (full 8-GPU) | 0.2204 | **+0.06 WORSE** | 626s | Changed prob formula breaks trained routing head |
| 3 | Online Calibration | — | — | — | |
| 4 | Conformal Aggregation | — | — | — | |
| 5 | Doc Boundary Detection | — | — | — | |
