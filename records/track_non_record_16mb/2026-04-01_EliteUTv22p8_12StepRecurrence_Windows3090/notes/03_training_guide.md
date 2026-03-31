# How to Run Training — Parameter Golf (Windows)

## The Windows Wrapper: `train_gpt_windows.py`

All training must be launched through `train_gpt_windows.py` (in the repo root).
It applies 3 patches before executing `train_gpt.py`:

1. **SDP backend fix** — disables Flash SDP, enables math/cudnn/mem_efficient SDP
2. **NCCL → gloo** — swaps distributed backend for Windows compatibility
3. **No torch.compile disable** — compile is kept ON (requires `triton-windows<3.3`)

## Quick Reference: All Training Commands

All settings are passed via **environment variables** (PowerShell `$env:VAR="value"` syntax).

### Smoke Test (5 steps, no validation)
```powershell
$env:ITERATIONS="5"
$env:VAL_LOSS_EVERY="0"
$env:WARMUP_STEPS="2"
$env:TRAIN_LOG_EVERY="1"
python train_gpt_windows.py
```
> **Use this to verify the setup is working.** Completes in ~10-15 min (first run includes
> compilation cache warm-up).

---

### Short Experiment Run (500 steps, validation every 100)
```powershell
$env:ITERATIONS="500"
$env:VAL_LOSS_EVERY="100"
$env:WARMUP_STEPS="20"
$env:TRAIN_LOG_EVERY="50"
$env:MAX_WALLCLOCK_SECONDS="0"
python train_gpt_windows.py
```
> Useful for quickly testing architecture changes. Logs go to `logs/<run_id>.txt`.

---

### Baseline Run (default 10-minute wallclock cap)
```powershell
python train_gpt_windows.py
```
Defaults:
- 20,000 iterations max, or 10 minutes wallclock (whichever comes first)
- Data: `./data/datasets/fineweb10B_sp1024/`
- Tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`
- `VOCAB_SIZE=1024`, `train_seq_len=1024`, `train_batch_tokens=524288`

---

### Unlimited Time Run (no wallclock cap)
```powershell
$env:MAX_WALLCLOCK_SECONDS="0"
$env:ITERATIONS="20000"
$env:VAL_LOSS_EVERY="500"
python train_gpt_windows.py
```

---

### Custom Architecture Run
```powershell
$env:NUM_LAYERS="11"
$env:MODEL_DIM="512"
$env:NUM_HEADS="8"
$env:NUM_KV_HEADS="4"
$env:MLP_MULT="3"
$env:ITERATIONS="1000"
$env:VAL_LOSS_EVERY="200"
$env:MAX_WALLCLOCK_SECONDS="0"
python train_gpt_windows.py
```

---

## All Configurable Environment Variables

### Data

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | Dataset directory |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | SentencePiece model file |
| `VOCAB_SIZE` | `1024` | Must match tokenizer vocab size |

### Training Length

| Variable | Default | Description |
|---|---|---|
| `ITERATIONS` | `20000` | Max training steps |
| `MAX_WALLCLOCK_SECONDS` | `600.0` | Stop after N seconds (0 = no cap) |
| `WARMUP_STEPS` | `20` | Compiler warmup steps (reset before main training) |
| `WARMDOWN_ITERS` | `1200` | LR warmdown steps before end |
| `TRAIN_BATCH_TOKENS` | `524288` | Total tokens per step across all ranks |
| `TRAIN_SEQ_LEN` | `1024` | Sequence length |

### Logging

| Variable | Default | Description |
|---|---|---|
| `TRAIN_LOG_EVERY` | `200` | Log train loss every N steps |
| `VAL_LOSS_EVERY` | `1000` | Compute val loss every N steps (0 = only at end) |
| `VAL_BATCH_SIZE` | `524288` | Tokens used per validation pass |
| `RUN_ID` | `(auto UUID)` | Log file name: `logs/<RUN_ID>.txt` |
| `SEED` | `1337` | Random seed |

### Model Shape

| Variable | Default | Description |
|---|---|---|
| `NUM_LAYERS` | `9` | Number of transformer blocks |
| `MODEL_DIM` | `512` | Embedding / model width |
| `NUM_HEADS` | `8` | Attention heads (Q) |
| `NUM_KV_HEADS` | `4` | KV heads (GQA: must divide `NUM_HEADS`) |
| `MLP_MULT` | `2` | MLP hidden = `MLP_MULT * MODEL_DIM` |
| `TIE_EMBEDDINGS` | `1` | Tie input/output embeddings (1=yes, 0=no) |
| `NUM_LOOPS` | `12` | (Internal) SHARED layers in Universal Transformer |
| `ROPE_BASE` | `10000.0` | RoPE frequency base |
| `LOGIT_SOFTCAP` | `30.0` | Logit soft-capping value |

### Optimizer

| Variable | Default | Description |
|---|---|---|
| `MATRIX_LR` | `0.04` | Learning rate for weight matrices (Muon) |
| `SCALAR_LR` | `0.04` | Learning rate for vectors/scalars (Adam) |
| `EMBED_LR` | `0.6` | Embedding LR (when not tied) |
| `TIED_EMBED_LR` | `0.05` | Embedding LR (when tied) |
| `HEAD_LR` | `0.008` | Untied LM head LR |
| `MUON_MOMENTUM` | `0.95` | Muon optimizer momentum |
| `MUON_BACKEND_STEPS` | `5` | Newton-Schulz iterations in Muon |
| `BETA1` | `0.9` | Adam β₁ |
| `BETA2` | `0.95` | Adam β₂ |
| `GRAD_CLIP_NORM` | `0.0` | Gradient clipping (0 = disabled) |

---

## Output Files

After training completes, the following outputs are expected:

| File | Description |
|---|---|
| `logs/<run_id>.txt` | Full training log (losses, hyperparams) |
| `final_model.pt` | Raw bf16/fp32 PyTorch state dict |
| `final_model.int8.ptz` | Quantized (int8) + zlib-compressed model |

The competition score is `val_bpb` from the `final_int8_zlib_roundtrip` line.

---

## Understanding the Output

```
step:200/20000 train_loss:5.1234 train_time:14200ms step_avg:71.00ms
step:1000/20000 val_loss:4.8765 val_bpb:1.2300 train_time:71000ms step_avg:71.00ms
...
final_int8_zlib_roundtrip val_loss:4.1234 val_bpb:1.2244 eval_time:3200ms
final_int8_zlib_roundtrip_exact val_loss:4.12340000 val_bpb:1.22440000
```

- `val_bpb` < 1.23 = beating the naive baseline
- `val_bpb` < 1.12 = competitive with current SOTA (as of March 2026)
- The competition metric is **the `val_bpb` from the `final_int8_zlib_roundtrip` line**

---

## Expected Performance on RTX 3090 (Windows)

| Config | Step time | Iterations in 10 min |
|---|---|---|
| Baseline (default, `torch.compile` ON) | ~70–80s/step | ~7–8 steps |
| Baseline (no compile, eager) | ~150s/step | ~4 steps |

> ⚠️ The RTX 3090 is much slower than 8×H100 (~1000 steps in 10 min on the challenge
> hardware). Use local training for **quick iteration and debugging** only.
> For leaderboard submissions, use a cloud GPU (RunPod H100).

---

---

## The Architecture (16MB Protocol)

To maximize reasoning power within the 16MB zlib/int8 footprint, we use a **Universal Transformer** configuration:

### 1. Universal Transformer (Shared Weights)
- **Active Parameters**: ~12.1M (Active) / 13.5M (Total).
- **Loop Depth**: **12 shared loops**.
- **Width**: **1024-dimension** (16 heads).
- **Benefit**: Retains the depth of a 12-layer model while only "paying" for 1 layer in the parameter budget.

### 2. Optimizer: CANS (Chebyshev-Accelerated Newton-Schulz)
- **Algorithm**: Degree-7 polynomial orthogonalization.
- **Backend Steps**: **3 steps** (Optimized for speed/stability).
- **Convergence**: Provides 10-20% faster initial feature formation compared to standard Newton-Schulz (Degree 5).

### 3. Curriculum: Inverse Batch Scaling
As the sequence length curriculum triggers (256 → 512 → 1024), memory usage scales quadratically $O(L^2)$. To keep the RTX 3090 (24GB) stable, we implement **Inverse Batch Scaling**:

| Wallclock Time | Seq Len | Grad Accum Steps | Effective Tokens |
|---|---|---|---|
| 0–60s | 256 | 8 | 524,288 |
| 60–120s | 512 | 16 | 524,288 |
| 120s+ | 1024 | 32 | 524,288 |

This ensures the training footprint never exceeds ~18GB VRAM despite the transformer width.
