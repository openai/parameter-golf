# OpenAI internal prelaunch notes

These notes are for OpenAI-managed GPU boxes. Keep local edits on-device, sync with `brix`, and avoid hand-editing the remote checkout.

## Canonical local data layout

Use the same paths everywhere:

- datasets: `/root/code/parameter-golf/data/datasets/fineweb10B_<variant>/`
- tokenizers: `/root/code/parameter-golf/data/tokenizers/`

As of March 11, 2026, the internal blobstore mirror for this data is:

- `az://oaidatasets2/speedrunkits/parametergolf_fineweb`

## Sync and SSH

Run these locally:

```bash
git pull --ff-only
brix git push <pool>
brix ssh <pool>-0
```

## Python env

On the pod, use the repo-local venv. Do not rely on the system Python stack for this repo.

```bash
cd /root/code/parameter-golf
/root/.pyenv/versions/3.12.9/bin/python -m venv .venv-openai
. .venv-openai/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On March 11, 2026, the preinstalled system `torch/triton` stack failed `torch.compile` on OpenAI H100 boxes. The repo-local venv worked.

## Compile caches

Reuse explicit compile caches on the pod:

```bash
export TRITON_CACHE_DIR=/root/code/parameter-golf/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/root/code/parameter-golf/.cache/inductor
export XDG_CACHE_HOME=/root/code/parameter-golf/.cache
```

Optional sanity check:

```bash
cd /root/code/parameter-golf
. .venv-openai/bin/activate
python - <<'PY'
import os
import torch
import triton

print(torch.__version__)
print(triton.__version__)
print(os.environ["TRITON_CACHE_DIR"])
print(os.environ["TORCHINDUCTOR_CACHE_DIR"])

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
fn = torch.compile(lambda a, b: (a @ b).relu(), dynamic=False, fullgraph=True)
z = fn(x, y)
torch.cuda.synchronize()
print("compile_ok", z.shape, z.dtype, float(z[0, 0]))
PY
```

## Stage SP1024 data

For internal boxes, prefer the blobstore mirror over ad hoc manual copies from older paths.

```bash
bbb cptree az://oaidatasets2/speedrunkits/parametergolf_fineweb /root/code/parameter-golf/data/
```

If you want the public Huggingface path instead, this also lands in the same local layout:

```bash
cd /root/code/parameter-golf
. .venv-openai/bin/activate
python data/cached_challenge_fineweb.py --variant sp1024 25
```

## Smoke run

Launch the normal trainer with the canonical paths:

```bash
cd /root/code/parameter-golf
. .venv-openai/bin/activate
export TRITON_CACHE_DIR=/root/code/parameter-golf/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/root/code/parameter-golf/.cache/inductor
export XDG_CACHE_HOME=/root/code/parameter-golf/.cache

RUN_ID=openai_smoke_sp1024 \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_BATCH_TOKENS=262144 \
WARMUP_STEPS=2 \
ITERATIONS=8 \
VAL_LOSS_EVERY=4 \
VAL_TOKENS=131072 \
VAL_BATCH_SIZE=65536 \
MAX_WALLCLOCK_SECONDS=120 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```