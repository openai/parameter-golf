# Spec: M6/M7/M8 Step 2 — Warmdown + Grad Clip + EMA

## Task
Create `train_gpt_m6_step2.py`, `train_gpt_m7_step2.py`, `train_gpt_m8_step2.py` by copying each model's `_step1.py` and applying the following changes.

## Changes (apply identically to all three)

### 1. Hyperparameters (lines ~55 and ~87)
Change:
```python
warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
```
To:
```python
warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
```

Change:
```python
grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
```
To:
```python
grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
```

### 2. Add EMA (Exponential Moving Average)
Add to Hyperparameters class:
```python
ema_decay = float(os.environ.get("EMA_DECAY", "0.997"))
```

After the model is created and moved to device (but before the training loop), add:
```python
# EMA model for evaluation
import copy
ema_model = copy.deepcopy(base_model)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad_(False)
```

Inside the training loop, after the optimizer step (after `optimizer.step()` and `optimizer.zero_grad()`), add:
```python
# EMA update
with torch.no_grad():
    for p_ema, p_model in zip(ema_model.parameters(), base_model.parameters()):
        p_ema.lerp_(p_model, 1.0 - args.ema_decay)
```

For all validation evaluations AND the final int8/zlib roundtrip evaluation, use `ema_model` instead of `base_model` (or `model` if using DDP — use the underlying module). The training loop itself still uses `base_model`.

### 3. Important: DO NOT change any model architecture
- Do NOT modify any neural network classes, layers, or model construction
- Do NOT change learning rates, batch sizes, sequence lengths, or other hyperparameters
- ONLY change: warmdown_iters default, grad_clip_norm default, and add EMA

## Reference
See `train_gpt_m3_step3.py` for how the warmdown and grad_clip values are used (they're already wired — just changing defaults).
For EMA pattern, see `train_gpt_m1_step3.py` or the reference `train_gpt.py` for how EMA is integrated.

## Output
Three files in the repo root:
- `train_gpt_m6_step2.py`
- `train_gpt_m7_step2.py`
- `train_gpt_m8_step2.py`

## Verification
Each file should be a valid Python script that can run with:
```bash
RUN_ID=test DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=10 python3 train_gpt_mX_step2.py
```
