"""
CPU smoke test — verifies the Adam-TTT changes work without GPU/flash_attn.
Run: python3 smoke_test.py
Expected: prints "SMOKE TEST PASSED" at the end.
"""
import math, os, sys, types
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# ── Stub out flash_attn so we can run on CPU ──────────────────────────────────
def _fake_flash_attn(q, k, v, causal=True):
    # q/k/v: (B, T, H, D) — reshape to (B, H, T, D) for sdpa
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    q2 = q.permute(0,2,1,3)
    k2 = k.permute(0,2,1,3).repeat_interleave(H//Hkv, dim=1)
    v2 = v.permute(0,2,1,3).repeat_interleave(H//Hkv, dim=1)
    out = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    return out.permute(0,2,1,3)

stub = types.ModuleType("flash_attn_interface")
stub.flash_attn_func = _fake_flash_attn
sys.modules["flash_attn_interface"] = stub

# ── Now import our modified training code ────────────────────────────────────
import lzma, base64, importlib

train_script = open(os.path.join(os.path.dirname(__file__), "train_gpt.py")).read()
lines = train_script.strip().split('\n')
line2 = lines[1]
import re
m = re.search(r'B\.b85decode\("(.+?)"\),format=', line2)
payload = m.group(1)
code = lzma.decompress(base64.b85decode(payload), format=lzma.FORMAT_RAW,
                       filters=[{"id": lzma.FILTER_LZMA2}]).decode('utf-8')

# Execute into a fresh namespace
ns = {}
# Patch CUDA checks before exec
code = code.replace(
    "if not torch.cuda.is_available():raise RuntimeError('CUDA is required')",
    "pass  # CPU smoke test: skip CUDA check"
)
code = code.replace(
    "device=torch.device('cuda',local_rank);torch.cuda.set_device(device)",
    "device=torch.device('cpu')"
)
code = code.replace(
    "torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high');from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp;enable_cudnn_sdp(False);enable_flash_sdp(True);enable_mem_efficient_sdp(False);enable_math_sdp(False);",
    ""
)
# Disable torch.compile for CPU smoke test (avoids inductor bugs on Python 3.12)
code = code.replace("@torch.compile\n", "")
code = code.replace("@torch.compile(", "@torch.no_grad() # was compile(")
exec(code, ns)

# ── Test 1: Model forward pass ────────────────────────────────────────────────
print("Test 1: Model forward pass...")

class FakeH:
    vocab_size = 64
    num_layers = 4
    model_dim = 32
    embedding_dim = 32
    num_heads = 4
    num_kv_heads = 2
    mlp_mult = 2.0
    rope_base = 10000.0
    qk_gain_init = 5.25          # our new default
    train_seq_len = 16
    logit_softcap = 30.0
    rope_dims = 8
    rope_train_seq_len = 16
    ln_scale = True
    skip_gates_enabled = True
    tie_embeddings = True
    tied_embed_init_std = 0.005
    xsa_last_n = 0
    parallel_residual_start = 2
    num_loops = 1
    loop_start = 1
    loop_end = 2
    enable_looping_at = 0.35

h = FakeH()
GPT = ns['GPT']
model = GPT(h)
model.looping_active = True

x = torch.randint(0, 64, (2, 16))
y = torch.randint(0, 64, (2, 16))
loss = model(x, y)
assert loss.item() > 0, "Loss should be positive"
print(f"  Forward pass OK. Loss: {loss.item():.4f}, QK-gain default: {h.qk_gain_init}")

# ── Test 2: _make_ttt_opt helper ─────────────────────────────────────────────
print("Test 2: TTT optimizer selection...")

_make_ttt_opt = ns['_make_ttt_opt']

class TttH:
    ttt_optimizer = 'sgd'
    ttt_lr = 0.005
    ttt_momentum = 0.9
    ttt_adam_beta1 = 0.9
    ttt_adam_beta2 = 0.999
    ttt_adam_eps = 1e-8
    ttt_weight_decay = 0.0

params = list(model.parameters())

# SGD
th = TttH(); th.ttt_optimizer = 'sgd'
opt = _make_ttt_opt(th, params)
assert isinstance(opt, torch.optim.SGD), "Expected SGD"
print("  SGD: OK")

# Adam (our default)
th.ttt_optimizer = 'adam'
opt = _make_ttt_opt(th, params)
assert isinstance(opt, torch.optim.Adam), "Expected Adam"
print("  Adam: OK")

# AdamW
th.ttt_optimizer = 'adamw'
opt = _make_ttt_opt(th, params)
assert isinstance(opt, torch.optim.AdamW), "Expected AdamW"
print("  AdamW: OK")

# ── Test 3: Adam-TTT step works ───────────────────────────────────────────────
print("Test 3: Adam-TTT gradient step...")

th.ttt_optimizer = 'adam'
opt = _make_ttt_opt(th, params)
opt.zero_grad()
loss = model(x, y)
loss.backward()
opt.step()
print(f"  Adam step OK. Loss after step: {model(x, y).item():.4f}")

# ── Test 4: Hyperparameter class has new fields ────────────────────────────────
print("Test 4: Hyperparameter class has new fields...")
# Set env vars and check
os.environ['TTT_OPTIMIZER'] = 'adam'
os.environ['QK_GAIN_INIT'] = '5.25'
os.environ['TTT_FRESH_OPTIMIZER'] = '0'
Hyperparameters = ns['Hyperparameters']
h2 = Hyperparameters()
assert hasattr(h2, 'ttt_optimizer'), "Missing ttt_optimizer"
assert hasattr(h2, 'ttt_adam_beta1'), "Missing ttt_adam_beta1"
assert hasattr(h2, 'ttt_fresh_optimizer'), "Missing ttt_fresh_optimizer"
assert h2.ttt_optimizer == 'adam', f"Expected adam, got {h2.ttt_optimizer}"
assert h2.qk_gain_init == 5.25, f"Expected 5.25, got {h2.qk_gain_init}"
print(f"  ttt_optimizer={h2.ttt_optimizer}, qk_gain_init={h2.qk_gain_init}")

print("\nSMOKE TEST PASSED")
print("\nNext steps:")
print("  1. Apply for compute grant (OpenAI RunPod link in README)")
print("  2. On RunPod 8xH100: run Experiment 1 (Adam-TTT, QK=5.25)")
print("  3. Compare val_bpb to SOTA 1.0810")
print("  4. Run Experiments 2-5 based on results")
