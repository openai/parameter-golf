# OpenAI internal prelaunch notes

This file preserves the OpenAI-only setup notes that were previously added to the repo root docs. The goal is to keep the main tree aligned with commit `6523a9f4ff6ec52374c9b8ea47f5169205337866` outside this folder.

## OpenAI H100 path

- Assume a live `8xH100` box already exists.
- Pull locally before syncing:
  `git pull`
- Sync local code with `brix git push <pool>`. This includes untracked files, which matters for local scripts in this repo.
- SSH to pod `0` with `brix ssh <pool>-0`.
- On the pod, use `/root/.pyenv/versions/3.12.9/bin/python`, not `/usr/bin/python3`. The system Python on these pods does not have a working `venv/pip` flow for this project.
- Create the repo-local env with:
  `/root/.pyenv/versions/3.12.9/bin/python -m venv /root/code/parameter-golf/.venv-openai`
- Install exactly:
  `/root/code/parameter-golf/.venv-openai/bin/python -m pip install --upgrade pip`
  `/root/code/parameter-golf/.venv-openai/bin/python -m pip install -r /root/code/parameter-golf/requirements.txt`
- Do not rely on the preinstalled system `torch/triton` stack for this repo on OpenAI H100 boxes. On March 11, 2026, that stack failed `torch.compile` on H100. The repo-local venv with `requirements.txt` worked.

## Compile caches

- Reuse explicit compile caches on the pod:
  `export TRITON_CACHE_DIR=/root/code/parameter-golf/.cache/triton`
  `export TORCHINDUCTOR_CACHE_DIR=/root/code/parameter-golf/.cache/inductor`
  `export XDG_CACHE_HOME=/root/code/parameter-golf/.cache`
- Sanity check the env and cache setup with:
  `cd /root/code/parameter-golf`
  `. .venv-openai/bin/activate`
  `python - <<'PY'`
  `import os, torch, triton`
  `print(torch.__version__)`
  `print(triton.__version__)`
  `print(os.environ["TRITON_CACHE_DIR"])`
  `print(os.environ["TORCHINDUCTOR_CACHE_DIR"])`
  `x = torch.randn(1024, 1024, device="cuda")`
  `y = torch.randn(1024, 1024, device="cuda")`
  `fn = torch.compile(lambda a, b: (a @ b).relu(), dynamic=False, fullgraph=True)`
  `z = fn(x, y)`
  `torch.cuda.synchronize()`
  `print("compile_ok", z.shape, z.dtype, float(z[0, 0]))`
  `PY`

## SP1024 staging

- Stage the cached SP1024 files with `bbb`:
  `mkdir -p /root/code/parameter-golf/data/matched_10B_docs2m_seed1337/tokenizers`
  `mkdir -p /root/code/parameter-golf/data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024`
  `bbb cp az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model /root/code/parameter-golf/data/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model`
  `bbb cp az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_train_000001.bin /root/code/parameter-golf/data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_train_000001.bin`
  `bbb cp az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_val_000000.bin /root/code/parameter-golf/data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_val_000000.bin`

## Normal trainer launch

- Launch the normal trainer with:
  `cd /root/code/parameter-golf && . .venv-openai/bin/activate && export TRITON_CACHE_DIR=/root/code/parameter-golf/.cache/triton && export TORCHINDUCTOR_CACHE_DIR=/root/code/parameter-golf/.cache/inductor && export XDG_CACHE_HOME=/root/code/parameter-golf/.cache && RUN_ID=openai_smoke_sp1024 DATA_PATH=/root/code/parameter-golf/data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/root/code/parameter-golf/data/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 TRAIN_BATCH_TOKENS=262144 WARMUP_STEPS=2 ITERATIONS=8 VAL_LOSS_EVERY=4 VAL_TOKENS=131072 VAL_BATCH_SIZE=65536 MAX_WALLCLOCK_SECONDS=120 torchrun --standalone --nproc_per_node=8 train_gpt.py`
- Verified on March 11, 2026 on `pgolf-zebra-openai-0`:
  the normal `train_gpt.py` completed an `8xH100` end-to-end smoke run with compile enabled, `step:8/8 val_bpb:4.1021`, peak memory `6879 MiB`, and int8 artifact size `10,404,835` bytes including code.
- Verified cache reuse on the same box:
  the first measured train step dropped from `2911ms` on the cold run to `721ms` on the immediate rerun with the same cache dirs, and later train steps were roughly `30-35ms`.

## Other notes moved from AGENTS

- `train_gpt_openai.py` is still available for strict H100-only experiments, but the simpler path is now the normal `train_gpt.py`.
- After more local edits, repeat:
  `git pull`
  `brix git push <pool>`

## F30 notes

- On `f30`, each `gb300` pod has `4` local GPUs. Single-node runs should use `torchrun --standalone --nproc_per_node=4 ...`.
- If you want `8` GB300s on `f30`, that is a `2`-pod run. `torchrun --standalone` is not enough for that layout.
- On fresh remote boxes, run `pip install -r requirements.txt` in this repo before training. Skipping it has led to Triton/runtime failures in this project.
- On `f30` GB300 pods with the current torch build, flash SDPA is not available on device capability `10.3`; `train_gpt.py` should fall back to math SDPA there, and grouped-query cases should use the manual GQA path instead of relying on flash-only `enable_gqa`.

## Artifact budget note

- For `train_gpt.py`, log both `code_bytes + quant_raw_bytes` and the actual compressed `.ptz` size from the int8 export path. The raw blob is only a conservative upper bound; the real challenge budget check is `code_bytes + len(final_model.int8.ptz)`.
