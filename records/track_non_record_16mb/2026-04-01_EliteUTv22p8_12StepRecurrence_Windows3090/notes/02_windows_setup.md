# Windows Setup Guide — Parameter Golf

Complete setup guide to get `train_gpt.py` running on Windows with a CUDA GPU.

---

## Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA Compute Capability ≥ 7.0 (RTX 20xx or newer)
- CUDA Toolkit 12.x installed
- Python 3.10–3.13
- Git

---

## Step 1 — Clone & Create Virtual Environment

```powershell
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## Step 2 — Install Dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt` includes:
```
numpy
tqdm
torch          ← installs PyTorch with CUDA (make sure CUDA version matches)
huggingface-hub
kernels
setuptools
typing-extensions==4.15.0
datasets
tiktoken
sentencepiece
```

> **Important:** If `torch` installs the CPU-only version, install the CUDA version manually:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

---

## Step 3 — Install `triton-windows` (critical for performance)

```powershell
pip install "triton-windows<3.3"
```

This installs the community-maintained Windows port of Triton (v3.2.x), which enables
`torch.compile`'s Inductor backend. Without this, training is **~2–3× slower**.

> Version must be `<3.3` to be compatible with PyTorch 2.6. Higher versions have an
> API mismatch (`AttrsDescriptor` import error).

---

## Step 4 — Download the Dataset

The training script needs tokenized FineWeb shards. A background download was already
started (`cached_challenge_fineweb.py --variant sp1024`).

To download a minimal subset (1 shard ≈ 100M tokens, good for local testing):
```powershell
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

For the full 80-shard dataset (8B tokens, for real training):
```powershell
python data/cached_challenge_fineweb.py --variant sp1024
```

Files land in:
```
data/datasets/fineweb10B_sp1024/
  fineweb_train_000000.bin  ← training shards (~190MB each)
  fineweb_train_000001.bin
  ...
  fineweb_val_000000.bin    ← fixed validation set
data/tokenizers/
  fineweb_1024_bpe.model    ← SentencePiece tokenizer
```

---

## Step 5 — Run Training via the Windows Wrapper

**Do NOT run `train_gpt.py` directly** — it will crash (Flash SDP/Triton issues).
Instead use `train_gpt_windows.py` which applies all patches automatically.

```powershell
python train_gpt_windows.py
```

---

## Verification — Check Your Setup

```powershell
.\venv\Scripts\python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

# Test torch.compile
fn = torch.compile(lambda x: x * 2)
out = fn(torch.randn(4,4).cuda())
print('torch.compile: OK')

# Test GQA SDPA (Windows-safe config)
from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp, enable_cudnn_sdp
enable_flash_sdp(False); enable_math_sdp(True); enable_cudnn_sdp(True); enable_mem_efficient_sdp(True)
import torch.nn.functional as F
q, k, v = [torch.randn(2,8,32,64).cuda().bfloat16() for _ in range(3)]
k = k[:, :4]; v = v[:, :4]  # GQA: 8 q heads, 4 kv heads
out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
print('GQA SDPA: OK')
"
```
