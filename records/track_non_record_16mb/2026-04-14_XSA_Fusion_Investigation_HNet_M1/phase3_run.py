"""Phase 3 training pilot.

Loads the FA3 baseline module, optionally patches all blocks' attention to use
fused-QKV and/or Triton-XSA, then runs `main()` for ITERATIONS steps.

Env knobs (defaults in code):
    PATCH_QKV=0|1   fuse c_q/c_k/c_v into a single c_qkv GEMM
    PATCH_XSA=0|1   swap `_xsa_efficient` to xsa_triton
    RUN_ID=...      baseline training log file name (train logs/<RUN_ID>.txt)
    ITERATIONS=200  pilot step count
    TRAIN_LOG_EVERY=20
    WARMUP_STEPS=10
    VAL_LOSS_EVERY=0  (skip in-training validation)
    MAX_WALLCLOCK_SECONDS=0  (disable wallclock cap)
"""
from __future__ import annotations
import os, sys, types, torch, torch.nn.functional as F

# -------- defaults -------------------------------------------------
os.environ.setdefault("DATA_DIR", "/workspace/parameter-golf/data/")
os.environ.setdefault("ITERATIONS", "200")
os.environ.setdefault("VAL_LOSS_EVERY", "0")
os.environ.setdefault("TRAIN_LOG_EVERY", "20")
os.environ.setdefault("WARMUP_STEPS", "10")
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "0")

PATCH_QKV = os.environ.get("PATCH_QKV", "0") == "1"
PATCH_XSA = os.environ.get("PATCH_XSA", "0") == "1"
print(f"[phase3] PATCH_QKV={PATCH_QKV}  PATCH_XSA={PATCH_XSA}")

sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton as xsa_triton_fn

# -------- load baseline module via exec ----------------------------
src = open("/workspace/work/train_gpt_baseline.py").read()
ns: dict = {"__name__": "pg_baseline", "__file__": "/workspace/work/train_gpt_baseline.py"}
exec(compile(src, "/workspace/work/train_gpt_baseline.py", "exec"), ns)

CastedLinear     = ns["CastedLinear"]
apply_rotary_emb = ns["apply_rotary_emb"]
fa3_func         = ns["flash_attn_3_func"]

# -------- patches --------------------------------------------------
def fuse_qkv_weights(attn):
    dim = attn.c_q.weight.size(1)
    q_dim = attn.num_heads * attn.head_dim
    kv_dim = attn.num_kv_heads * attn.head_dim
    c_qkv = CastedLinear(dim, q_dim + 2 * kv_dim, bias=False)
    c_qkv = c_qkv.to(attn.c_q.weight.device).to(attn.c_q.weight.dtype)
    with torch.no_grad():
        c_qkv.weight.copy_(torch.cat([attn.c_q.weight, attn.c_k.weight, attn.c_v.weight], dim=0))
    attn.c_qkv = c_qkv
    del attn.c_q, attn.c_k, attn.c_v


def make_fused_qkv_forward(use_triton_xsa):
    def forward(self, x):
        bsz, seqlen, _ = x.shape
        qkv = self.c_qkv(x)
        q_dim  = self.num_heads    * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads,    self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = fa3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = xsa_triton_fn(y, v) if use_triton_xsa else self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, y.size(-2) * y.size(-1))
        return self.proj(y)
    return forward


def patched_xsa_method(self, y, v):
    return xsa_triton_fn(y, v)


# Intercept GPT.__init__ to apply post-construction patches
original_GPT_init = ns["GPT"].__init__
def patched_GPT_init(self, *args, **kwargs):
    original_GPT_init(self, *args, **kwargs)
    n_patched = 0
    for blk in self.blocks:
        if PATCH_QKV:
            fuse_qkv_weights(blk.attn)
            blk.attn.forward = types.MethodType(make_fused_qkv_forward(PATCH_XSA), blk.attn)
        elif PATCH_XSA:
            # Only XSA patch, keep 3-linear path
            blk.attn._xsa_efficient = types.MethodType(patched_xsa_method, blk.attn)
        n_patched += 1 if (PATCH_QKV or PATCH_XSA) else 0
    print(f"[phase3] patched {n_patched} blocks (QKV={PATCH_QKV} XSA={PATCH_XSA})")
ns["GPT"].__init__ = patched_GPT_init

# -------- run main() -----------------------------------------------
ns["main"]()
