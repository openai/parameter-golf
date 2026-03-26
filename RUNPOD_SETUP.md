# RunPod 8xH100 Setup — Parameter Golf

Every time. No skipping steps. Tested against PyTorch 2.9.1+cu128 on RunPod.

## Pod Config

- GPU: 8x H100 80GB HBM3
- Template: RunPod PyTorch (2.9.x / CUDA 12.8)
- Disk: 100GB+ (data shards are ~20GB)
- Workspace: `/workspace`

## Step 1: Clone and checkout

```bash
cd /workspace
git clone https://github.com/newjordan/parameter-golf.git
cd parameter-golf
git checkout <your-branch>
```

## Step 2: Python deps

```bash
pip install sentencepiece numpy zstandard
```

## Step 3: Flash Attention 3 (the hard part)

FA3 does NOT have prebuilt wheels. You must build from source. Full build = 451 CUDA kernels = 12+ hours. Selective build = ~5 min.

### 3a. Clone FA3

```bash
cd /workspace/parameter-golf
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
```

### 3b. Create the output directory (build fails without this)

```bash
mkdir -p flash_attn_3
```

### 3c. Export ALL disable flags BEFORE building

**CRITICAL: You must `export` these. Inline `VAR=val pip install` does NOT work — pip spawns subprocesses that don't inherit inline vars.**

```bash
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
```

### 3d. Build with --no-build-isolation

**CRITICAL: Without `--no-build-isolation`, pip creates a temp venv that can't find torch and the build fails with `ModuleNotFoundError: No module named 'torch'`.**

```bash
python3 -m pip install --no-build-isolation -e .
```

This builds only ~2 kernels (bf16 + hdim64 + SM90, fwd and bwd). Takes ~5 minutes.

**How to check progress from another terminal:**
```bash
ps aux | grep nvcc | grep -v grep | wc -l
```
\>0 = still compiling. 0 = done (check build terminal).

### 3e. If the editable install doesn't register properly

Sometimes `pip install -e .` finishes but `import flash_attn_3` still fails. The nuclear option that always works:

```bash
export PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:$PYTHONPATH
```

Add this to every command that runs training. This is the reliable path.

### 3f. Verify

```bash
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

## Step 4: Verify data

```bash
cd /workspace/parameter-golf
ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l   # expect 80
ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l     # expect >0
ls data/tokenizers/fineweb_1024_bpe.model                         # must exist
```

If data is missing, it needs to be downloaded/copied from a previous pod or generated.

## Step 5: Preflight (catch errors before the 10-min run)

```bash
cd /workspace/parameter-golf
PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:$PYTHONPATH \
python3 -c "
import torch
assert torch.cuda.device_count() == 8, f'Expected 8 GPUs, got {torch.cuda.device_count()}'
from flash_attn_interface import flash_attn_func
import sentencepiece, zstandard, numpy
print(f'{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)} — all OK')
"
```

## Step 6: Run training

**CRITICAL: Always run from the repo root (`/workspace/parameter-golf`), not from a subdirectory. The data paths in the script are relative (`./data/...`). If you `cd` into a subfolder, they break.**

```bash
cd /workspace/parameter-golf
PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:$PYTHONPATH \
torchrun --nproc_per_node=8 <experiment>/train_gpt.py
```

Or if the experiment has a run.sh:
```bash
cd /workspace/parameter-golf
PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:$PYTHONPATH \
bash <experiment>/run.sh
```

## Debugging

### torchrun shows no traceback
torchrun hides Python tracebacks. Run single-GPU to see the actual error:
```bash
PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:$PYTHONPATH \
python3 <experiment>/train_gpt.py 2>&1 | head -50
```

### OMP_NUM_THREADS warning
```
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default
```
This is normal. Ignore it.

### NVIDIA_VISIBLE_DEVICES="void"
Normal RunPod thing. GPUs are still accessible via CUDA. Ignore.

### Multiple FA3 builds running
If you started a build, killed it, and started another, check for zombie nvcc:
```bash
pkill -f nvcc; pkill -f "pip install"; sleep 2
```
Then rebuild from step 3c.

## Gotchas Summary

| Gotcha | Fix |
|--------|-----|
| `pip install -e .` → `No module named 'torch'` | Add `--no-build-isolation` |
| Inline env vars not working for FA3 build | Use `export VAR=TRUE` before pip |
| `could not create 'flash_attn_3/_C.abi3.so'` | `mkdir -p flash_attn_3` before build |
| FA3 import fails after install | Use `PYTHONPATH=.../hopper:$PYTHONPATH` |
| `No such file: ./data/tokenizers/...` | Run from repo root, not experiment subdir |
| torchrun no traceback | Debug with single-GPU `python3 train_gpt.py` |
| FA3 building wrong kernels (hdim128, fp16) | Kill all, re-export flags, rebuild |
