# Cloud Setup

This is the safe setup path for RunPod-style PyTorch images and other shared CUDA images that already ship with torch.

Do not reinstall torch on top of those images unless you are intentionally rebuilding the whole stack.

## Rule

On RunPod PyTorch images:

- do not run `pip install -r requirements-local.txt`
- do not run `pip install torch ...`
- do not run any setup command that upgrades or replaces the image-provided torch unless you are intentionally rebuilding the environment

Use the image torch as the anchor, then install repo deps around it.

## Pod 2 Sequence

From a fresh SSH session:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
export PYTHONPATH=/workspace/parameter-golf${PYTHONPATH:+:$PYTHONPATH}
```

Verify the image torch stack before changing anything:

```bash
python3 scripts/check_frontier_env.py --allow-missing-flash-attn
```

Install the safe cloud deps and FlashAttention without replacing torch:

```bash
./scripts/install_cloud.sh
```

Re-check frontier readiness:

```bash
python3 scripts/check_frontier_env.py
```

Stage the small published dataset subset:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
python3 scripts/check_data.py --data-path ./data/datasets/fineweb10B_sp1024 --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model --min-train-shards 1 --seq-len 2048
```

Run the first frontier smoke test:

```bash
python3 research/run.py --preset control_verified_sota --scale smoke --run-name pod2_control_smoke --seed 1337 --nproc-per-node 1 --gpu-profile runpod_rtx3090
```

## Files

- `requirements.txt`: safe default, common deps only, no torch
- `requirements-common.txt`: repo-wide common deps, no torch
- `requirements-local.txt`: self-managed local stack, includes torch
- `requirements-cloud.txt`: shared-image stack, preserves image torch
- `scripts/install_cloud.sh`: safe cloud install helper
- `scripts/check_frontier_env.py`: frontier CUDA / FlashAttention readiness check
- `flash_attn_interface.py`: repo-local import shim so the trainers can use the installed flash-attn package consistently

## Manual Commands

If you do not want to use `./scripts/install_cloud.sh`, the manual safe sequence is:

```bash
python3 scripts/check_frontier_env.py --allow-missing-flash-attn
python3 -m pip install -r requirements-cloud.txt
python3 scripts/check_frontier_env.py --allow-missing-flash-attn
python3 -m pip install flash-attn --no-build-isolation
python3 scripts/check_frontier_env.py
```

After the `requirements-cloud.txt` step, confirm that `torch_version`, `torch_cuda_version`, and `torch_path` did not change.

## What The Frontier Check Means

`python3 scripts/check_frontier_env.py` reports:

- torch version and torch CUDA version
- whether CUDA is visible through torch
- GPU inventory
- whether `flash_attn_interface` imports
- whether the environment looks like a cloud image
- whether `nvcc` and torch CUDA versions look mismatched

If it prints `STOP`, do not spend time on training yet.

At runtime, the frontier trainers now log the selected attention backend. They
prefer FlashAttention when the actual attention tensor dtype/device are valid,
and otherwise log a fallback to `sdp_math` instead of crashing.

Common causes:

- torch was upgraded on top of the image
- `flash-attn` was not installed yet
- `flash-attn` was built against a different CUDA / torch stack
- the pod image does not actually expose CUDA correctly
