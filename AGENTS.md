# Local Notes

- `train_gpt.py` loads constants from `--config <path>`, and config files must expose `CONFIG` as a `TrainConstants` object from `train_gpt_constants.py`.
- Use the exact command blocks in `README.md` as the canonical repro flow; keep README commands runnable line-for-line before/after changes.
- When validating setup on a fresh machine, run README commands exactly (no extra helper steps) before adding optional debug commands.
- Use a single dependency install command: `pip install -r requirements.txt` (no separate `data/requirements.txt`).
- Prefer `python3 -m torch.distributed.run` in launch commands instead of bare `torchrun`; some fresh hosts install torch scripts into `~/.local/bin` which is not on PATH by default.
- `TrainConstants` intentionally only contains 8x/1x diff knobs: `target_total_gpus`, `model_num_layers`, `model_num_heads`, `model_dim`, `stage_batch_sizes`, `val_batch_size`, `optimizer_lr_scale`, `optimizer_wd_scale`, `drop_zero_length_seqlens`, `compile_model`, `empty_cache_every_steps`. Shared values (data paths, muon schedule, window schedule, val tokens, tokenizer vocab, etc.) stay in `train_gpt.py`.
- Current data/tokenizer path uses SentencePiece 4k shards (`data/fineweb10B_sp4k/*`) and `train_gpt.py` aligns docs using `BOS_ID=1`; tokenizer models must preserve `bos_id=1` for BOS-aligned batching to remain valid.
- `train_gpt.py` now expects fixed local shards at `data/fineweb10B_sp4k/` and fails immediately when train/val shard globs are missing.
- Use `python3 data/cached_fineweb10B_sp4k.py 9` to download the current default 4k shards + tokenizer files in one step.
- The 4k SentencePiece trainer now uses BPE + `byte_fallback=True`; keep `vocab_size=4096` and `bos_id=1` when regenerating shards so token IDs remain uint16-safe and BOS-aligned batching remains valid.
- For this BPE setup, `character_coverage=1.0` is invalid at vocab 4096 (`required_chars` exceeds vocab when `byte_fallback=True`); keep coverage below 1.0 (currently `0.999`) unless vocab is increased.
- Use `data/build_upload_4096_bpe.py` for one-command local rebuild + upload of tokenizer/shards to HF.
- Config files under `configs/` are `chz`-based when available; `train_gpt_constants.py` provides a dataclass-compatible fallback for environments where `chz` is unavailable.
- Runpod pods currently tested here start with `torch 2.4.1+cu124`; this repo needs nightly `cu126` torch/triton (`triton.tools.tensor_descriptor`) before `train_gpt.py` will import.
- When syncing to Lambda with `rsync --delete`, exclude `data/fineweb10B_sp4k/` and `logs/` or shards/logs get deleted; missing shards fail training with `No files found for pattern: ./data/fineweb10B_sp4k/fineweb_val_*.bin`.
- On this runtime, 1x runs can develop non-finite train gradients in the flash-attn varlen path; forcing `USE_FLASH_ATTN=0` (SDPA fallback) keeps train/val losses finite in current tests.
- On 1x H100 with current nightly stack, stage batch-shape transitions can cause CUDA OOM in `FusedSoftcappedCrossEntropy`; keep `configs/train_gpt_1xh100.py` train batch fixed at `3072` tokens across stages unless kernel memory behavior changes.
- On the same 1x path, large validation batch shapes can leave too little post-eval memory headroom; keep `val_batch_size` at `4 * 1024` unless you re-validate memory behavior end-to-end.
- On this runtime, long 1x runs are more stable with `compile_model=False` in `configs/train_gpt_1xh100.py`; keeping compile enabled caused progressive memory growth and eventual OOM.
- With the current 1x profile (`model_num_layers=6`, `model_dim=512`, `model_num_heads=4`, `compile_model=False`, `empty_cache_every_steps=1`, `val_batch_size=4 * 1024`) and `USE_FLASH_ATTN=0`, train/val stay finite through at least step 500 in current tests (`val_loss=6.6203` at step 250 on Lambda).
- The flash-attn path (`USE_FLASH_ATTN=1`) still fails on this runtime with `non-finite mean train loss` (latest probe at step 174); this points to a kernel-path instability rather than a validation-only issue or LR tuning issue.
- For the same profile, per-step `torch.cuda.empty_cache()` (`empty_cache_every_steps=1`) is currently required on this host/runtime to avoid late-run OOMs from large reserved-but-unallocated CUDA blocks.

# Detailed Codebase Guide (Moved from README)

This repository contains a custom GPT training stack optimized for H100 GPUs.

Core path:
1. Prepare FineWeb SentencePiece-4k token shards (`data/*.py`).
2. Launch distributed training (`train_gpt.py` via `torchrun` + config path).
3. Use Triton kernels (`triton_kernels.py`) for fused loss, fused MLP activation, and optimizer math.

Both 8x and 1x paths run the same `train_gpt.py` hot path and differ by config values.

## Repository Map

### Top level
- `train_gpt.py`: main trainer entrypoint.
- `train_gpt_medium.py`: alternate larger training config with a different schedule/optimizer orchestration.
- `train_gpt_constants.py`: `chz` config schema + config loader used by `train_gpt.py`.
- `configs/train_gpt_8xh100.py`: default 8x profile.
- `configs/train_gpt_1xh100.py`: 1x profile.
- `triton_kernels.py`: Triton kernels used directly by training.
- `Dockerfile`: CUDA + Python + dependency image for reproducible runs.
- `requirements.txt`: root runtime dependencies.
- `img/`: static project images.

### `data/`
- `fineweb.py`: streams FineWeb from Hugging Face, tokenizes with SentencePiece, writes binary shards.
- `train_sentencepiece_4k.py`: trains a 4k SentencePiece tokenizer model used by `fineweb.py`.
- `cached_fineweb10B_sp4k.py`: downloads current default 4k-tokenized shards + tokenizer artifacts from `cocohearts/4096-bpe`.
- `cached_fineweb10B.py`, `cached_fineweb100B.py`: legacy GPT-2-tokenized shard downloaders.

## Environment and Dependencies

### Host install
```bash
pip install -r requirements.txt
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
- Tokenizes with a SentencePiece model (default `data/tokenizers/fineweb_4k.model`).
- Prefixes each document with tokenizer BOS (`bos_id=1`).
- Writes shards under `data/fineweb10B_sp4k/` or `data/fineweb100B_sp4k/`.
- First shard is validation (`fineweb_val_000000.bin`), remaining are training shards (`fineweb_train_*.bin`).

### 2) Train the 4k tokenizer model
```bash
python data/train_sentencepiece_4k.py --version 10B
python data/train_sentencepiece_4k.py --version 100B
```

### 3) Legacy: download cached GPT-2 token shards instead
```bash
python data/cached_fineweb10B.py
python data/cached_fineweb100B.py
```

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
- Scalar/gate parameter groups (residual, attention mix, smear/backout/skip controls).
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
- Produces rank-local flattened token buffers + cumulative lengths.
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
torchrun --standalone --nproc_per_node=8 train_gpt.py --config configs/train_gpt_8xh100.py
```

### 1x H100 path
```bash
USE_FLASH_ATTN=0 torchrun --standalone --nproc_per_node=1 train_gpt.py --config configs/train_gpt_1xh100.py
```

### Runpod setup + launch (1x H100)
```bash
# connect (Runpod endpoint here requires PTY)
ssh -tt -i /Users/alexzhao/.ssh/voltage-park-test <runpod-user>@ssh.runpod.io
```

```bash
# on pod, after repo is present at /workspace/N-challenge
cd /workspace/N-challenge
python3 -m pip install --upgrade pip filelock
pip3 install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python3 - <<'PY'
import importlib, torch, triton
print(torch.__version__, torch.version.cuda, triton.__version__)
importlib.import_module("triton.tools.tensor_descriptor")
print("tensor_descriptor_ok")
PY
python3 data/cached_fineweb10B_sp4k.py 9
USE_FLASH_ATTN=0 torchrun --standalone --nproc_per_node=1 train_gpt.py --config configs/train_gpt_1xh100.py
```

If `scp`/`rsync` is unavailable on this Runpod endpoint, transfer code with `runpodctl send` (local) and `runpodctl receive <code>` (pod), then move the unpacked folder to `/workspace/N-challenge`.

Current stable 1x profile characteristics:
- Train batch fixed at `3072` across stages.
- Validation batch set to `4 * 1024`.
- Model set to 6 layers, width 512, heads 4.
- `compile_model=False`.
- `empty_cache_every_steps=1`.
- `USE_FLASH_ATTN=0` in the 1x launch command to avoid non-finite train loss on this host/runtime.
- Periodic validation remains finite through at least step 250 (`val_loss=6.6203`) in current checks.

### Runtime outputs
- Text logs: `logs/<run_id>.txt`
- Optional checkpoint (when `save_checkpoint=True`):
  - `logs/<run_id>/state_stepXXXXXX.pt`

### Practical constraints
- Requires CUDA GPUs and NCCL distributed setup.
- Assumes data shards exist under `data/fineweb10B_sp4k/` unless `DATA_PATH` or file patterns are changed.
- Uses strict assertions heavily; invalid scaling/configuration should fail immediately.

### Minimal customization points
- Data location: `DATA_PATH` env var and/or `Hyperparameters.train_files/val_files`
- Batch/model/schedule scaling: edit `configs/train_gpt_*.py`
- Validation cadence: `Hyperparameters.val_loss_every`
- Checkpointing: `Hyperparameters.save_checkpoint`
