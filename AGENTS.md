# Local Notes

- `train_gpt.py` loads constants from `--config <path>`, and config files must expose `CONFIG` as a `TrainConstants` object from `train_gpt_constants.py`.
- `TrainConstants` intentionally only contains 8x/1x diff knobs: `target_total_gpus`, `model_num_layers`, `model_num_heads`, `model_dim`, `stage_batch_sizes`, `val_batch_size`, `optimizer_lr_scale`, `optimizer_wd_scale`, `drop_zero_length_seqlens`, `compile_model`, `empty_cache_every_steps`. Shared values (data paths, muon schedule, window schedule, val tokens, bigram vocab, etc.) stay in `train_gpt.py`.
- Config files under `configs/` are `chz`-based when available; `train_gpt_constants.py` provides a dataclass-compatible fallback for environments where `chz` is unavailable.
- When syncing to Lambda with `rsync --delete`, exclude `data/fineweb10B/` and `logs/` or shards/logs get deleted; missing shards fail training with `No files found for pattern: ./data/fineweb10B/fineweb_val_*.bin`.
- On this runtime, 1x runs can develop non-finite train gradients in the flash-attn varlen path; forcing `USE_FLASH_ATTN=0` (SDPA fallback) keeps train/val losses finite in current tests. Keep `run_1xh100.sh` launching with `USE_FLASH_ATTN=0` unless kernel behavior is re-validated.
- On 1x H100 with current nightly stack, stage batch-shape transitions can cause CUDA OOM in `FusedSoftcappedCrossEntropy`; keep `configs/train_gpt_1xh100.py` train batch fixed at `3072` tokens across stages unless kernel memory behavior changes.
- On the same 1x path, large validation batch shapes can leave too little post-eval memory headroom; keep `val_batch_size` at `4 * 1024` unless you re-validate memory behavior end-to-end.
- On this runtime, long 1x runs are more stable with `compile_model=False` in `configs/train_gpt_1xh100.py`; keeping compile enabled caused progressive memory growth and eventual OOM.
- With the current 1x profile (`model_num_layers=6`, `model_dim=512`, `model_num_heads=4`, `compile_model=False`, `empty_cache_every_steps=1`, `val_batch_size=4 * 1024`) and `USE_FLASH_ATTN=0`, train/val stay finite through at least step 500 in current tests (`val_loss=9.6280` at step 250 on Lambda).
- The flash-attn path (`USE_FLASH_ATTN=1`) still fails on this runtime with `non-finite mean train loss` (latest probe at step 174); this points to a kernel-path instability rather than a validation-only issue or LR tuning issue.
- For the same profile, per-step `torch.cuda.empty_cache()` (`empty_cache_every_steps=1`) is currently required on this host/runtime to avoid late-run OOMs from large reserved-but-unallocated CUDA blocks.

# Detailed Codebase Guide (Moved from README)

This repository contains a custom GPT training stack optimized for H100 GPUs.

Core path:
1. Prepare FineWeb GPT-2 token shards (`data/*.py`).
2. Launch distributed training (`train_gpt.py` via `run.sh` / `run_1xh100.sh`).
3. Use Triton kernels (`triton_kernels.py`) for fused loss, fused MLP activation, and optimizer math.

Both 8x and 1x paths run the same `train_gpt.py` hot path and differ by config values.

## Repository Map

### Top level
- `train_gpt.py`: main trainer used by default in `run.sh`.
- `train_gpt_medium.py`: alternate larger training config with a different schedule/optimizer orchestration.
- `train_gpt_constants.py`: `chz` config schema + config loader used by `train_gpt.py`.
- `configs/train_gpt_8xh100.py`: default 8x profile.
- `configs/train_gpt_1xh100.py`: 1x profile.
- `triton_kernels.py`: Triton kernels used directly by training.
- `run.sh`: 8-GPU launch command.
- `run_1xh100.sh`: 1-GPU launch command.
- `Dockerfile`: CUDA + Python + dependency image for reproducible runs.
- `requirements.txt`: root runtime dependencies.
- `img/`: static project images.

### `data/`
- `fineweb.py`: streams FineWeb from Hugging Face, tokenizes with GPT-2 tokenizer, writes binary shards.
- `cached_fineweb10B.py`: downloads pre-tokenized FineWeb 10B shards.
- `cached_fineweb100B.py`: downloads pre-tokenized FineWeb 100B shards.
- `cached_finewebedu10B.py`: downloads pre-tokenized FineWebEDU 10B shards.
- `requirements.txt`: data-prep-only deps (`datasets`, `tiktoken`).

## Environment and Dependencies

### Host install
```bash
pip install -r requirements.txt
```

### Data-prep install
```bash
pip install -r data/requirements.txt
```

### Docker image
The Dockerfile uses:
- Base: `nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04`
- Python: 3.12.7 (compiled from source)
- Nightly PyTorch CUDA 12.6 wheel

Build:
```bash
docker build -t n-challenge .
```

## Data Pipeline

### 1) Build shards from raw FineWeb
```bash
python data/fineweb.py --version 10B
python data/fineweb.py --version 100B
```

Behavior:
- Streams dataset split from `HuggingFaceFW/fineweb`.
- Tokenizes with `tiktoken` GPT-2 encoding.
- Prefixes each document with `<|endoftext|>`.
- Writes shards under `data/fineweb10B/` or `data/fineweb100B/`.
- First shard is validation (`fineweb_val_000000.bin`), remaining are training shards (`fineweb_train_*.bin`).

### 2) Download cached token shards instead
```bash
python data/cached_fineweb10B.py
python data/cached_fineweb100B.py
python data/cached_finewebedu10B.py
```

Optional first positional arg limits train chunk count.

### Binary shard format
Each `.bin` file is:
1. Header: 256 `int32` values.
2. Payload: token IDs as contiguous `uint16`.

Header fields used by writer:
- `header[0] = 20240520` (magic)
- `header[1] = 1` (format version)
- `header[2] = number of payload tokens`

## `train_gpt.py` Architecture and Runtime Flow

The file is intentionally monolithic: model, optimizer, scheduling, data loader, warmup, training, and validation are all in one script.

### Boot sequence
1. Reads its own source and `triton_kernels.py` source for run logging.
2. Initializes distributed NCCL from `RANK`, `WORLD_SIZE`, `LOCAL_RANK`.
3. Builds model and optimizer manager.
4. Compiles model with `torch.compile(dynamic=False, fullgraph=True)` only when `config.compile_model` is true.
5. Runs kernel warmup on selected schedule-transition steps.
6. Resets model/optimizer to pre-warmup state.
7. Runs training + periodic validation.
8. Optionally checkpoints final state on rank 0.

### Distributed assumptions
`train_gpt.py` reads constants from a config path:
- CLI: `--config configs/train_gpt_8xh100.py`
- Schema: `TrainConstants` in `train_gpt_constants.py`

Asserts enforce:
- `target_total_gpus % WORLD_SIZE == 0`
- `model_num_layers >= 4`
- `model.model_dim == model.num_heads * model.head_dim`

Gradient accumulation:
```text
grad_accum_steps = target_total_gpus // WORLD_SIZE
```

### Model structure (`GPT`)
Key components:
- Token embedding + tied output head (`lm_head` / `embed`, split later by schedule).
- Attention and MLP parameter banks for sharded optimization.
- Smear/skip gates and value embeddings.
- Bigram embedding path and scalar/gate parameter groups.
- YaRN-based RoPE objects for normal and paired-head paths.

Important hard-coded behavior:
- Layer count is now config-driven (`model_num_layers`), with assert `>= 4`.
- Attention skip on `layer 6` is preserved when that layer exists.
- Skip/backout/value-embed/window placement logic adapts to configured depth.

### Scheduling
`TrainingSchedule` is stage-driven and controls:
- Batch size
- Attention window sizes
- MTP weights
- Learning-rate multiplier
- Embedding split step
- Final post-training long-window extension

Default stage window sequence:
- `(1,3) -> (3,7) -> (5,11) -> (6,13)`
- Extension with final long-window extension to `20`

### Optimizer path
`NorMuonAndAdam` combines:
- NorMuon-like updates for large projection matrices (attention/MLP banks)
- Adam variants for scalars, gates, embeddings, and heads

Communication/update is explicitly ordered with reduce-scatter/all-reduce/all-gather steps via:
- `scatter_order`
- `work_order`

### Data feeding
`distributed_data_generator(...)`:
- Reads matching shard files.
- Aligns to BOS when requested.
- Produces rank-local flattened token buffers + cumulative lengths + bigram hashes.
- Accepts runtime schedule updates through `.send((num_tokens, max_seq_len, grad_accum_steps))`.

Validation uses the same generator with `align_to_bos=False`.

### Triton kernel usage (`triton_kernels.py`)

Used directly in the hot path:
- `XXT`, `ba_plus_cAA`: matrix ops used by optimizer math.
- `FusedLinearReLUSquareFunction`: fused MLP activation path.
- `FusedSoftcappedCrossEntropy`: fused training loss path.

More concretely:
- `XXT` / `ba_plus_cAA` are called from optimizer-side orthogonalization (`polar_express` path).
- `FusedLinearReLUSquareFunction` replaces separate MLP activation kernels in transformer blocks.
- `FusedSoftcappedCrossEntropy` is used in training loss; validation uses unfused PyTorch cross entropy.

There is no graceful fallback path if Triton is unavailable; training expects Triton kernels to import and compile.

## `train_gpt_medium.py`

`train_gpt_medium.py` is a separate trainer entrypoint with a larger model/training plan than `train_gpt.py`:
- Model: 16 layers, 8 heads, 1024 model dim.
- Sequence/window policy: longer training windows and a larger `train_max_seq_len` plan.
- Schedule length: 4700 scheduled iterations + 40 extension iterations.
- Batch schedule centers around larger long-run batches (including repeated 524,288-token phases).
- Different optimizer orchestration: separate `DistAdam` handling plus NorMuon path, with hook gating across grad-accum steps.

Use it only when you specifically want that medium-variant behavior.

## Launching Training

### Default 8x H100 path
```bash
./run.sh
```

Equivalent:
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py --config configs/train_gpt_8xh100.py
```

### 1x H100 path
```bash
./run_1xh100.sh
```

Equivalent:
```bash
torchrun --standalone --nproc_per_node=1 train_gpt.py --config configs/train_gpt_1xh100.py
```

Current stable 1x profile characteristics:
- Train batch fixed at `3072` across stages.
- Validation batch set to `4 * 1024`.
- Model set to 6 layers, width 512, heads 4.
- `compile_model=False`.
- `empty_cache_every_steps=1`.
- `USE_FLASH_ATTN=0` in launcher (`run_1xh100.sh`) to avoid non-finite train loss on this host/runtime.
- Periodic validation remains finite through at least step 250 (`val_loss=9.6280`) in current checks.

### Runtime outputs
- Text logs: `logs/<run_id>.txt`
- Optional checkpoint (when `save_checkpoint=True`):
  - `logs/<run_id>/state_stepXXXXXX.pt`

### Practical constraints
- Requires CUDA GPUs and NCCL distributed setup.
- Assumes data shards exist under `data/fineweb10B/` unless `DATA_PATH` or file patterns are changed.
- Uses strict assertions heavily; invalid scaling/configuration should fail immediately.

### Minimal customization points
- Data location: `DATA_PATH` env var and/or `Hyperparameters.train_files/val_files`
- Batch/model/schedule scaling: edit `configs/train_gpt_*.py`
- Validation cadence: `Hyperparameters.val_loss_every`
- Checkpointing: `Hyperparameters.save_checkpoint`
