This record captures a non-record 16MB submission centered on an SP8192 BPE run with **Mamba3 SSM hybrid architecture**, trained on a single H100 for 30 minutes.

The key architecture contribution here is the SSM/attention hybrid: replacing every 4th transformer attention block with a Mamba3 state-space model layer, reducing parameter count while maintaining competitive BPB. With `ssm_every_n=4` (2 SSM blocks, 7 GQA attention blocks), the model achieves 18.31M params — saving ~2.2M params vs the all-attention variant.

Configuration:
- Track: `non-record`
- Layout: `VOCAB_SIZE=8192 MODEL_DIM=448 NUM_LAYERS=9 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- SSM: `USE_SSM=1 SSM_EVERY_N=4 SSM_IMPL=mamba3 MAMBA3_HEAD_DIM=64`
- Tokenizer: SentencePiece BPE 8192 (`fineweb_8192_bpe.model`)
- Batching: `TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024`
- Eval: sliding-window validation with `EVAL_STRIDE_FRAC=0.5`
- Opt: Muon (matrix) + Adam (scalar), `SWA_ENABLED=1`
- Quant/export: GPTQ int8 + zstd

Key metrics (from `train.log`):
- Timed training stopped at `12278/20000` steps due to 30min wallclock cap.
- Pre-quant eval at stop: `val_loss:3.2398`, `val_bpb:1.2542`
- Post-quant roundtrip eval: `val_loss:3.25624330`, `val_bpb:1.26060944`
- Train time: `1800080ms` (`step_avg:146.61ms`)
- Code size: `231880 bytes`

SSM/attention hybrid notes:
- **Mamba3 SSM** (`mamba_ssm` official CUDA extension) used as a drop-in mixer replacement
- SSM blocks use `expand=2.0, d_state=128, head_dim=64, mimo_rank=4` — comparable throughput to GQA attention on H100
- `ssm_every_n=4` means layers [2, 6] are SSM, rest are GQA attention — reduces params by ~11% vs all-attention

Dataset/tokenizer requirement:
- This package expects an **SP8192 exported dataset** at:
  - `./sp8192_data/datasets/fineweb10B_sp8192`
- And uses tokenizer assets in this folder by default:
  - `./fineweb_8192_bpe.model`
  - `./fineweb_8192_bpe.vocab`
- Build the dataset (includes mamba_ssm CUDA extension install):
  - `bash ./setup_sp8192_data.sh`

Note: `mamba-ssm` is the official Mamba CUDA extension from [state-spaces/mamba](https://github.com/state-spaces/mamba).
Install from GitHub source (requires CUDA toolkit):
```bash
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
  git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

Run command (1-GPU):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=sp8192_bpe_mamba3_d448_ssm4_1xh30m_s1337 \
DATA_PATH=./sp8192_data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
MODEL_DIM=448 \
NUM_LAYERS=9 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
USE_SWIGLU=1 \
USE_SSM=1 \
SSM_EVERY_N=4 \
MAMBA3_HEAD_DIM=64 \
TRAIN_BATCH_TOKENS=65536 \
MAX_WALLCLOCK_SECONDS=1800 \
WARMUP_STEPS=20 \
EVAL_STRIDE_FRAC=0.5 \
QUANT_SCHEME=int8 \
COMPRESSOR=zstd \
GPTQ=1 GPTQ_NSAMPLES=128 GPTQ_BLOCKSIZE=128 GPTQ_PERCDAMP=0.01 \
torchrun --standalone --nproc_per_node=1 ./train_gpt_mamba3.py
```

Included files:
- `train_gpt_mamba3.py` (code snapshot used for the run package)
- `train.log` (exact run log, source code + runtime output)
- `submission.json` (metadata)
- `reqs.txt` (dependencies)
- `fineweb_8192_bpe.model` and `fineweb_8192_bpe.vocab` (tokenizer assets)
