#!/usr/bin/env python3
"""
CPU architecture test for SP8192 + CaseOps + RecurAlpha submission.

Stubs flash_attn_3, triton, and TensorDescriptor so the model runs
on CPU (no CUDA / Hopper required). Covers:

  1  Model instantiation with Loop345 (layers 3-5 x2)
  2  Encoder/decoder index layout matches expected U-Net pattern
  3  Forward pass with looping_active=False
  4  Forward pass with looping_active=True
  5  Recur-alpha gradient flow — recur_alpha.grad must be nonzero
  6  Nonzero alpha changes output vs zero alpha
"""
import sys, types, os
import torch
import torch.nn.functional as F

# Disable torch.compile before anything else — calling it imports triton.backends
# which breaks on CPU when triton is stubbed out.
torch.compile = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)

# ── GPU stubs (must be installed before train_gpt is imported) ───────────────

def _sdpa_func(q, k, v, causal=True):
    """Drop-in for flash_attn_3_func.  q/k/v: (B,T,H,D) → (B,T,H,D)."""
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    if Hkv != H:
        k = k.repeat_interleave(H // Hkv, dim=2)
        v = v.repeat_interleave(H // Hkv, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
    )
    return out.transpose(1, 2)

def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                 max_seqlen_q, max_seqlen_k, causal, window_size=(-1, -1)):
    """Drop-in for flash_attn_varlen_func.  q/k/v: (T,H,D) → (T,H,D)."""
    return _sdpa_func(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                      causal=causal).squeeze(0)

_fa = types.ModuleType("flash_attn_interface")
_fa.flash_attn_func = _sdpa_func
_fa.flash_attn_varlen_func = _sdpa_varlen
sys.modules["flash_attn_interface"] = _fa

# triton: @triton.jit becomes a no-op; tl.anything returns a harmless sentinel
class _Anything:
    def __getattr__(self, _): return self
    def __call__(self, *a, **kw): return a[0] if a else self
    def __getitem__(self, _): return self

_tl = _Anything()
_tl.__name__ = "triton.language"

_triton = types.ModuleType("triton")
_triton.jit = lambda f: f
_triton.cdiv = lambda a, b: (a + b - 1) // b
_triton.language = _tl

_td_mod = types.ModuleType("triton.tools.tensor_descriptor")
class _TD:
    @staticmethod
    def from_tensor(*a, **kw): return None
_td_mod.TensorDescriptor = _TD

_tools = types.ModuleType("triton.tools")
_tools.tensor_descriptor = _td_mod

sys.modules.update({
    "triton": _triton,
    "triton.language": _tl,
    "triton.tools": _tools,
    "triton.tools.tensor_descriptor": _td_mod,
})

# ── Small-model env vars (read by Hyperparameters at class-definition time) ──

os.environ.setdefault("MODEL_DIM",           "64")
os.environ.setdefault("NUM_HEADS",           "4")
os.environ.setdefault("NUM_KV_HEADS",        "2")
os.environ.setdefault("VOCAB_SIZE",          "64")
os.environ.setdefault("TRAIN_SEQ_LEN",       "32")
os.environ.setdefault("ROPE_TRAIN_SEQ_LEN",  "32")
os.environ.setdefault("XSA_LAST_N",          "0")   # disable XSA for simplicity
os.environ.setdefault("NUM_LOOPS",           "2")
os.environ.setdefault("LOOP_START",          "3")
os.environ.setdefault("LOOP_END",            "5")
os.environ.setdefault("GATED_ATTN_ENABLED",  "1")   # test with gated-attn on
os.environ.setdefault("GATED_ATTN_QUANT_GATE", "0") # no quantization on CPU

# ── Import model ─────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt as tg  # classes only — main() is guarded by __name__=="__main__"

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_model():
    h = tg.Hyperparameters()
    torch.manual_seed(0)
    model = tg.GPT(h)
    # Initialise weight banks to finite values; preserve recur_alpha=0 init
    recur_alpha_ids = {id(b.recur_alpha) for b in model.blocks}
    for p in model.parameters():
        if p.data.is_floating_point() and id(p) not in recur_alpha_ids:
            torch.nn.init.normal_(p.data, std=0.02)
    # Disable triton fused MLP; eval path uses plain PyTorch leaky_relu^2
    for b in model.blocks:
        b.mlp.use_fused = False
    return model, h

def make_ids(h, B=2):
    return torch.randint(0, h.vocab_size, (B, h.train_seq_len))

# ── Test runner ───────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag}  {name}" + (f"  [{detail}]" if not ok and detail else ""))
    if not ok:
        failures.append(name)

# ── Test 1: instantiation ────────────────────────────────────────────────────

print("\nTest 1: Model instantiation")
model, h = make_model()
check("GPT created",            model is not None)
check("11 physical blocks",     len(model.blocks) == 11)
check("recur_alpha on every block",
      all(hasattr(b, "recur_alpha") for b in model.blocks))
check("recur_alpha init=0",
      all(b.recur_alpha.item() == 0.0 for b in model.blocks))

# ── Test 2: index layout ─────────────────────────────────────────────────────

print("\nTest 2: Encoder/decoder index layout (Loop345, num_loops=2)")
# loop_seg=[3,4,5], all_indices=[0,1,2]+[3,4,5]*3+[6..10] = 17 entries
# encoder = first 8:  [0,1,2,3,4,5,3,4]
# decoder = last  9:  [5,3,4,5,6,7,8,9,10]
expected_enc = [0, 1, 2, 3, 4, 5, 3, 4]
expected_dec = [5, 3, 4, 5, 6, 7, 8, 9, 10]
check("encoder indices correct",
      model.encoder_indices == expected_enc,
      f"got {model.encoder_indices}")
check("decoder indices correct",
      model.decoder_indices == expected_dec,
      f"got {model.decoder_indices}")
# Layers 3 and 4 loop twice in encoder; layer 5 is the U-Net bottleneck
# (appears once in encoder, twice in decoder via carry)
check("layers 3,4 loop twice in encoder",
      model.encoder_indices.count(3) == 2 and model.encoder_indices.count(4) == 2)
check("layer 5 present in both encoder and decoder",
      5 in model.encoder_indices and 5 in model.decoder_indices)

# ── Test 3: forward, looping_active=False ────────────────────────────────────

print("\nTest 3: Forward pass — looping_active=False")
model.eval()
model.looping_active = False
ids = make_ids(h)
with torch.no_grad():
    loss_no_loop = model(ids, ids)
check("loss is scalar",  loss_no_loop.shape == torch.Size([]))
check("loss is finite",  loss_no_loop.isfinite().item())
print(f"         loss = {loss_no_loop.item():.4f}")

# ── Test 4: forward, looping_active=True ─────────────────────────────────────

print("\nTest 4: Forward pass — looping_active=True")
model.looping_active = True
with torch.no_grad():
    loss_with_loop = model(ids, ids)
check("loss is scalar",  loss_with_loop.shape == torch.Size([]))
check("loss is finite",  loss_with_loop.isfinite().item())
print(f"         loss = {loss_with_loop.item():.4f}")

# ── Test 5: recur_alpha gradient flow ────────────────────────────────────────

print("\nTest 5: Recur-alpha gradient flow")
model.train()
model.looping_active = True
for b in model.blocks:
    b.mlp.use_fused = False

ids_train = make_ids(h)
loss = model(ids_train, ids_train)
loss.backward()

looped = [3, 4, 5]
for i in looped:
    g = model.blocks[i].recur_alpha.grad
    check(f"blocks[{i}].recur_alpha has gradient",  g is not None)
    if g is not None:
        check(f"blocks[{i}].recur_alpha.grad is finite",
              g.isfinite().all().item())
        check(f"blocks[{i}].recur_alpha.grad != 0",
              g.abs().item() > 0.0,
              f"grad={g.item():.6e}")

# ── Test 6: nonzero alpha changes output ─────────────────────────────────────

print("\nTest 6: Nonzero alpha changes output vs zero alpha")
model.eval()
model.looping_active = True
for b in model.blocks:
    b.mlp.use_fused = False

ids_one = make_ids(h, B=1)
with torch.no_grad():
    logits_zero = model.forward_logits(ids_one).clone()
    for i in looped:
        model.blocks[i].recur_alpha.data.fill_(0.5)
    logits_half = model.forward_logits(ids_one).clone()
    for i in looped:
        model.blocks[i].recur_alpha.data.fill_(0.0)

check("alpha=0.5 changes logits vs alpha=0",
      not torch.allclose(logits_zero, logits_half),
      "outputs identical — carry path has no effect")
if not torch.allclose(logits_zero, logits_half):
    delta = (logits_half - logits_zero).abs().mean().item()
    print(f"         mean |Δlogit| = {delta:.4f}")

# ── Summary ──────────────────────────────────────────────────────────────────

print()
if not failures:
    print("All tests PASSED.\n")
else:
    print(f"{len(failures)} test(s) FAILED: {failures}\n")
    sys.exit(1)
