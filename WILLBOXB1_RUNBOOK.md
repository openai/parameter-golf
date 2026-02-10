# willboxb1-0 setup + repro (install-first path)

## What was done

Date validated: 2026-02-10

1. Installed a dedicated runtime env on `willboxb1-0`:
   - Python: `3.12.9`
   - `torch==2.9.0+cu128`
   - `triton==3.5.0`
   - `flash-attn==2.8.3` (built/installed in venv)
   - `kernels==0.12.1`
2. Kept `torch.compile` and flash-attn enabled in training (no runtime fallback clamps).
3. Kept only two code compatibility changes in `train_gpt.py`:
   - Prefer local `flash_attn` import first; fallback to `get_kernel(...)` only if import fails.
   - Replace unsupported CUDA op path `torch.uint32 <<` with `torch.int32 <<` in `_cautious_wd_and_update_inplace`.

## Why this was needed

- On this devbox stack, `get_kernel('varunneal/flash-attention-3')` can fail to fetch the large `.so` via HF CAS.
- Local `flash-attn` avoids that network path.
- `torch.uint32` left-shift is not implemented for CUDA tensors on this stack, but `int32` preserves the bit pattern needed by the mantissa path.

## Devbox network note

- Devboxes are internet-restricted; `pip` must use the internal mirror.
- On the validated box, `pip config list` showed:
  - `index-url='http://nginx.pypi.svc.cluster.local/simple/'`
  - `trusted-host='nginx.pypi.svc.cluster.local'`

## Reproduce

Run these from local machine in repo root.

1. Sync code file(s) to pod:

```bash
brix scp willboxb1-0 train_gpt.py :/root/code/openai-parameter-challenge/train_gpt.py
```

2. Create/update env on pod:

```bash
brix run --pods willboxb1-0 -- bash -lc '
set -euo pipefail
python3 -m venv /root/.venvs/train29
source /root/.venvs/train29/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.9.0 kernels==0.12.1
pip install flash-attn==2.8.3
pip install -r /root/code/openai-parameter-challenge/requirements.txt
'
```

3. Verify core versions:

```bash
brix run --pods willboxb1-0 -- bash -lc '
source /root/.venvs/train29/bin/activate
python - << "PY"
import torch, triton, flash_attn, kernels
print(torch.__version__)
print(triton.__version__)
print(flash_attn.__version__)
print(kernels.__version__)
PY
'
```

4. 1x smoke test:

```bash
brix run --pods willboxb1-0 -- bash -lc '
cd /root/code/openai-parameter-challenge
source /root/.venvs/train29/bin/activate
USE_FLASH_ATTN=1 timeout 180 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --config configs/train_gpt_1xh100.py | tee /tmp/train29_1x_installfirst.log
'
```

5. 8x test:

```bash
brix run --pods willboxb1-0 -- bash -lc '
cd /root/code/openai-parameter-challenge
source /root/.venvs/train29/bin/activate
USE_FLASH_ATTN=1 timeout 540 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --config configs/train_gpt_8xh100.py | tee /tmp/train29_8x_installfirst.log
'
```

## Expected signals of success

- 1x run: progresses through warmup and prints step lines before timeout.
- 8x run: reaches final line `step:1555/1555` and prints final `val_loss`.
- No log lines like:
  - `flash attention disabled`
  - `torch.compile disabled`
  - batch-size clamp fallback messages

## Cleanup / rollback

- Unneeded prior docs tweak was removed (`README.md` restored).
- If you need to undo the code changes too:

```bash
git checkout -- train_gpt.py
```
