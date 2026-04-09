# NFT Quantization Field Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the NFT feedback loop as a soft-quantization field that co-evolves with model weights during training, producing less quantization damage than naive rounding.

**Architecture:** Modify CastedLinear to apply temperature-controlled soft quantization during forward pass. The sigmoid blend between grid neighbors is differentiable, so backprop naturally computes the "back-action" — which grid assignments reduce loss. Two-phase training: Phase 1 trains a parent model normally, Phase 2 continues with the quantization field active.

**Tech Stack:** MLX (Apple Silicon), Python 3.14, existing train_gpt_mlx.py infrastructure

---

### Task 1: Create the NFT training script

**Files:**
- Create: `train_nft_mlx.py` (copy of `train_gpt_mlx.py`, will be modified in subsequent tasks)

**Step 1: Copy the baseline script**

```bash
cp train_gpt_mlx.py train_nft_mlx.py
```

**Step 2: Verify the copy runs**

```bash
RUN_ID=nft_copy_test \
ITERATIONS=2 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Completes with val_loss and val_bpb printed.

**Step 3: Commit**

```bash
git add train_nft_mlx.py
git commit -m "feat: copy baseline MLX script as starting point for NFT quantizer"
```

---

### Task 2: Add NFT hyperparameters

**Files:**
- Modify: `train_nft_mlx.py` — Hyperparameters class (lines 43-97)

**Step 1: Add NFT-specific hyperparameters to the Hyperparameters class**

Add after the existing hyperparameters (after line 96):

```python
    # NFT Quantization Field
    nft_enabled: bool = bool(int(os.environ.get("NFT_ENABLED", "0")))
    nft_quant_bits: int = int(os.environ.get("NFT_QUANT_BITS", 8))  # int8 target
    nft_temp_max: float = float(os.environ.get("NFT_TEMP_MAX", 1.0))
    nft_temp_min: float = float(os.environ.get("NFT_TEMP_MIN", 0.01))
    nft_temp_power: float = float(os.environ.get("NFT_TEMP_POWER", 2.0))
    nft_checkpoint_path: str = os.environ.get("NFT_CHECKPOINT_PATH", "")
```

**Step 2: Add temperature schedule method**

Add after the `lr_mul` method:

```python
    def nft_temperature(self, step: int) -> float:
        """Temperature schedule for the quantization field.
        Starts at nft_temp_max (broad exploration), decays to nft_temp_min (commitment).
        """
        if not self.nft_enabled:
            return 0.0
        progress = min(step / max(self.iterations, 1), 1.0)
        t = self.nft_temp_max * (1.0 - progress) ** self.nft_temp_power
        return max(t, self.nft_temp_min)
```

**Step 3: Verify script still runs with NFT disabled (default)**

```bash
RUN_ID=nft_hyper_test \
ITERATIONS=2 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Same behavior as baseline.

**Step 4: Commit**

```bash
git add train_nft_mlx.py
git commit -m "feat: add NFT quantization field hyperparameters and temperature schedule"
```

---

### Task 3: Implement the soft quantization function

**Files:**
- Modify: `train_nft_mlx.py` — add `soft_quantize` function after the math helpers section (after line 193)

**Step 1: Write the soft quantization function**

```python
def soft_quantize(w: mx.array, temperature: float, bits: int = 8) -> mx.array:
    """NFT quantization field: soft-quantize weights via temperature-controlled sigmoid.

    Instead of hard rounding (naive quantization), each weight holds a differentiable
    blend between its two nearest grid neighbors. The temperature controls exploration
    vs commitment:
      - High T: blend ≈ 0.5 (superposition over grid neighbors)
      - Low T: blend → 0 or 1 (collapsed to one grid point)

    Gradients flow through the sigmoid, so backprop naturally computes which grid
    assignments reduce loss (the NFT "back-action").
    """
    if temperature <= 0.0:
        # Hard quantization (fully collapsed)
        clip_range = (1 << (bits - 1)) - 1  # 127 for int8, 31 for int6, 15 for int5
        # Per-row scale for 2D, per-tensor for 1D
        if w.ndim == 2:
            row_max = mx.max(mx.abs(w), axis=1, keepdims=True)
            scale = mx.maximum(row_max / clip_range, mx.array(1.0 / clip_range))
        else:
            tensor_max = mx.max(mx.abs(w))
            scale = mx.maximum(tensor_max / clip_range, mx.array(1.0 / clip_range))
        q = mx.round(mx.clip(w / scale, -clip_range, clip_range))
        return q * scale

    # Compute per-row (2D) or per-tensor (1D) scale
    clip_range = (1 << (bits - 1)) - 1
    if w.ndim == 2:
        row_max = mx.max(mx.abs(w), axis=1, keepdims=True)
        scale = mx.maximum(row_max / clip_range, mx.array(1.0 / clip_range))
    else:
        tensor_max = mx.max(mx.abs(w))
        scale = mx.maximum(tensor_max / clip_range, mx.array(1.0 / clip_range))

    # Normalize to grid coordinates
    w_scaled = mx.clip(w / scale, -clip_range, clip_range)

    # Grid neighbors
    grid_down = mx.floor(w_scaled)
    grid_up = grid_down + 1.0

    # Clamp grid_up to valid range
    grid_up = mx.clip(grid_up, -clip_range, clip_range)

    # Distance to grid_down in [0, 1]
    distance = w_scaled - grid_down

    # Temperature-controlled sigmoid: the "measurement basis"
    # At high T, sigmoid ≈ 0.5 (explore). At low T, sigmoid → 0 or 1 (commit).
    blend = mx.sigmoid((distance - 0.5) / temperature)

    # Soft quantized value in grid coordinates
    w_soft_scaled = (1.0 - blend) * grid_down + blend * grid_up

    # Back to weight space
    return w_soft_scaled * scale
```

**Step 2: Write a quick sanity test inline**

Add a temporary test at the bottom of the file (before `if __name__ == "__main__"`):

```python
def _test_soft_quantize():
    """Quick sanity check for soft_quantize."""
    w = mx.array([[0.347, -0.123, 0.891], [0.001, -0.999, 0.500]])
    # High temperature: output should be close to midpoints (blended)
    w_hot = soft_quantize(w, temperature=10.0, bits=8)
    # Low temperature: output should be close to hard-quantized
    w_cold = soft_quantize(w, temperature=0.001, bits=8)
    # Zero temperature: should be exactly hard-quantized
    w_hard = soft_quantize(w, temperature=0.0, bits=8)
    print(f"Original:  {w}")
    print(f"Hot (T=10): {w_hot}")
    print(f"Cold(T=.001): {w_cold}")
    print(f"Hard (T=0): {w_hard}")
    # Cold and hard should be very close
    diff = mx.max(mx.abs(w_cold - w_hard))
    print(f"Max diff cold vs hard: {diff.item():.6f}")
    assert diff.item() < 0.01, f"Cold and hard should be close, got diff={diff.item()}"
    print("soft_quantize sanity check PASSED")
```

**Step 3: Run the test**

```bash
.venv/bin/python3 -c "import train_nft_mlx; train_nft_mlx._test_soft_quantize()"
```

Expected: "soft_quantize sanity check PASSED"

**Step 4: Commit**

```bash
git add train_nft_mlx.py
git commit -m "feat: implement soft_quantize with temperature-controlled sigmoid blending"
```

---

### Task 4: Inject soft quantization into the forward pass

**Files:**
- Modify: `train_nft_mlx.py` — CastedLinear class and GPT model

**Step 1: Add a global temperature holder**

Add after the `CONTROL_TENSOR_NAME_PATTERNS` block (around line 130):

```python
# Global NFT temperature state. Updated each training step.
# Using a mutable container so compiled functions can see changes.
_NFT_STATE = {"temperature": 0.0, "enabled": False, "bits": 8}
```

**Step 2: Modify CastedLinear.__call__ to apply soft quantization**

Replace the existing `__call__` method (line 285-286):

```python
    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight.astype(x.dtype)
        if _NFT_STATE["enabled"] and _NFT_STATE["temperature"] > 0:
            w = soft_quantize(
                w,
                temperature=_NFT_STATE["temperature"],
                bits=_NFT_STATE["bits"],
            ).astype(x.dtype)
        return x @ w.T
```

**Step 3: Also apply to the logit projection in GPT.loss**

The tied embedding weight is used as the LM head. Modify the logit computation in `GPT.loss` (around line 441):

```python
        emb_w = self.tok_emb.weight.astype(x.dtype)
        if _NFT_STATE["enabled"] and _NFT_STATE["temperature"] > 0:
            emb_w = soft_quantize(
                emb_w,
                temperature=_NFT_STATE["temperature"],
                bits=_NFT_STATE["bits"],
            ).astype(x.dtype)
```

And use `emb_w` in place of `self.tok_emb.weight.astype(x.dtype)` in both the chunked and non-chunked logit paths.

**Step 4: Verify script still runs with NFT disabled**

```bash
RUN_ID=nft_inject_test \
ITERATIONS=2 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Same behavior as baseline (NFT_ENABLED defaults to 0).

**Step 5: Verify script runs WITH NFT enabled**

```bash
RUN_ID=nft_enabled_test \
NFT_ENABLED=1 \
NFT_TEMP_MAX=1.0 \
ITERATIONS=5 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Completes. Loss may be higher than baseline due to soft quantization noise — that's expected.

**Step 6: Commit**

```bash
git add train_nft_mlx.py
git commit -m "feat: inject soft quantization into CastedLinear forward pass and logit head"
```

---

### Task 5: Wire up temperature schedule in the training loop

**Files:**
- Modify: `train_nft_mlx.py` — main() training loop (around line 1000-1060)

**Step 1: Initialize NFT state in main() after model creation**

Add after the optimizer setup (around line 901):

```python
    # NFT Quantization Field setup
    _NFT_STATE["enabled"] = args.nft_enabled
    _NFT_STATE["bits"] = args.nft_quant_bits
    _NFT_STATE["temperature"] = args.nft_temp_max if args.nft_enabled else 0.0
    if args.nft_enabled:
        log(f"nft:enabled temp_max:{args.nft_temp_max} temp_min:{args.nft_temp_min} "
            f"temp_power:{args.nft_temp_power} bits:{args.nft_quant_bits}")
```

**Step 2: Update temperature each training step**

Inside the training loop, just before the forward pass (around line 1028, after `lr_mul` computation):

```python
        if args.nft_enabled:
            _NFT_STATE["temperature"] = args.nft_temperature(step)
```

**Step 3: Log temperature periodically**

In the training log line (around line 1051-1055), add temperature to the logged metrics when NFT is enabled:

```python
        nft_temp_str = f" nft_T:{_NFT_STATE['temperature']:.4f}" if args.nft_enabled else ""
```

And append `nft_temp_str` to the log message.

**Step 4: Set temperature to 0 for final quantization**

Before the final serialization section (around line 1059), collapse the field:

```python
    if args.nft_enabled:
        _NFT_STATE["temperature"] = 0.0
        log(f"nft:collapsed temperature to 0 for final quantization")
```

**Step 5: Verify full loop runs**

```bash
RUN_ID=nft_loop_test \
NFT_ENABLED=1 \
NFT_TEMP_MAX=1.0 \
ITERATIONS=10 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Completes. Log shows decreasing nft_T values. Final val_bpb printed.

**Step 6: Commit**

```bash
git add train_nft_mlx.py
git commit -m "feat: wire temperature schedule into training loop with logging"
```

---

### Task 6: Add checkpoint save/load for two-phase training

**Files:**
- Modify: `train_nft_mlx.py` — main() function

**Step 1: Add checkpoint saving at the end of Phase 1**

After the model state is saved (around line 1067), add:

```python
    # Save checkpoint for NFT Phase 2 loading
    checkpoint_path = out_dir / f"{args.run_id}_checkpoint.npz"
    mx.savez(str(checkpoint_path), **flat_state)
    log(f"checkpoint_saved:{checkpoint_path}")
```

**Step 2: Add checkpoint loading at the start of main()**

After model creation (around line 900), add:

```python
    # Load parent checkpoint if provided (Phase 2 of NFT two-phase training)
    if args.nft_checkpoint_path:
        ckpt_path = Path(args.nft_checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"NFT checkpoint not found: {ckpt_path}")
        log(f"nft:loading parent checkpoint from {ckpt_path}")
        loaded = dict(mx.load(str(ckpt_path)))
        model.update(tree_unflatten(list(loaded.items())))
        log(f"nft:loaded {len(loaded)} tensors from parent checkpoint")
```

**Step 3: Test two-phase workflow**

Phase 1 (parent):
```bash
RUN_ID=parent_test \
ITERATIONS=5 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Phase 2 (child):
```bash
RUN_ID=child_test \
NFT_ENABLED=1 \
NFT_CHECKPOINT_PATH=logs/parent_test_checkpoint.npz \
ITERATIONS=5 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Child loads parent checkpoint, trains with NFT field active, completes with val_bpb.

**Step 4: Commit**

```bash
git add train_nft_mlx.py
git commit -m "feat: add checkpoint save/load for two-phase NFT training"
```

---

### Task 7: Create the overnight run script

**Files:**
- Create: `run_nft_overnight.sh`

**Step 1: Write the script**

```bash
#!/bin/bash
# NFT Quantization Field — Overnight Mac Run
# Phase 1: Train parent model (baseline, ~2-3 hours)
# Phase 2: Train child with NFT loop (~5-6 hours)
# Compares naive quantization damage vs NFT quantization damage

set -e

VENV=".venv/bin/python3"
DATA_PATH="./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

echo "=========================================="
echo " NFT Quantization Field — Overnight Run"
echo "=========================================="

# Phase 1: Parent training (no NFT, standard baseline)
echo ""
echo "PHASE 1: Training parent model..."
echo ""

RUN_ID=nft_parent \
ITERATIONS=2000 \
WARMUP_STEPS=20 \
TRAIN_BATCH_TOKENS=65536 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=65536 \
TRAIN_LOG_EVERY=50 \
MAX_WALLCLOCK_SECONDS=7200 \
$VENV train_nft_mlx.py

echo ""
echo "Phase 1 complete. Parent checkpoint saved."
echo ""

# Phase 2: Child training with NFT quantization field
echo "PHASE 2: Training child with NFT loop..."
echo ""

RUN_ID=nft_child \
NFT_ENABLED=1 \
NFT_CHECKPOINT_PATH=logs/nft_parent_checkpoint.npz \
NFT_TEMP_MAX=1.0 \
NFT_TEMP_MIN=0.01 \
NFT_TEMP_POWER=2.0 \
NFT_QUANT_BITS=8 \
ITERATIONS=2000 \
WARMUP_STEPS=5 \
TRAIN_BATCH_TOKENS=65536 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=65536 \
TRAIN_LOG_EVERY=50 \
MAX_WALLCLOCK_SECONDS=18000 \
$VENV train_nft_mlx.py

echo ""
echo "Phase 2 complete."
echo ""

echo "=========================================="
echo " RESULTS"
echo "=========================================="
echo ""
echo "Compare these lines from the logs:"
echo "  Parent (naive int8):  grep 'final_int8_zlib_roundtrip_exact' logs/nft_parent.txt"
echo "  Child (NFT int8):     grep 'final_int8_zlib_roundtrip_exact' logs/nft_child.txt"
echo ""
echo "If child val_bpb < parent val_bpb, the NFT loop produces less quantization damage."
echo "=========================================="
```

**Step 2: Make executable**

```bash
chmod +x run_nft_overnight.sh
```

**Step 3: Commit**

```bash
git add run_nft_overnight.sh
git commit -m "feat: add overnight run script for NFT two-phase training"
```

---

### Task 8: Remove the inline test, final cleanup

**Files:**
- Modify: `train_nft_mlx.py` — remove `_test_soft_quantize` function

**Step 1: Remove the test function**

Delete the `_test_soft_quantize` function added in Task 3.

**Step 2: Final smoke test with NFT enabled**

```bash
RUN_ID=nft_final_smoke \
NFT_ENABLED=1 \
NFT_TEMP_MAX=1.0 \
ITERATIONS=3 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Completes cleanly. Log shows nft_T values decreasing.

**Step 3: Final smoke test with two-phase workflow**

Run a quick Phase 1 then Phase 2:

```bash
RUN_ID=smoke_parent \
ITERATIONS=3 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py

RUN_ID=smoke_child \
NFT_ENABLED=1 \
NFT_CHECKPOINT_PATH=logs/smoke_parent_checkpoint.npz \
ITERATIONS=3 \
WARMUP_STEPS=1 \
TRAIN_BATCH_TOKENS=8192 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
.venv/bin/python3 train_nft_mlx.py
```

Expected: Both complete. Child loads parent and trains with NFT field.

**Step 4: Commit**

```bash
git add train_nft_mlx.py
git commit -m "feat: cleanup and final smoke test for NFT quantization field"
```

---

### Task 9: Launch the overnight run

**Step 1: Start the overnight run in background**

```bash
nohup bash run_nft_overnight.sh > logs/nft_overnight.log 2>&1 &
echo $! > logs/nft_overnight.pid
```

**Step 2: Verify it started**

```bash
tail -5 logs/nft_overnight.log
```

Expected: Shows Phase 1 starting.

---

## Success Criteria

After the overnight run completes, check:

```bash
grep 'final_int8_zlib_roundtrip_exact' logs/nft_parent.txt
grep 'final_int8_zlib_roundtrip_exact' logs/nft_child.txt
```

**The NFT loop works if child val_bpb < parent val_bpb.** This means the feedback loop produced less quantization damage than naive rounding, validating the core NFT insight: co-evolving the brain and substrate yields better discrete representations than snapping independently.
