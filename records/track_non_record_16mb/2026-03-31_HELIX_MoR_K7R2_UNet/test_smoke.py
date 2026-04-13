"""Full forward/backward/optimizer smoke test on CPU for MoR SOTA."""
import sys, os, types, math, io
sys.path.insert(0, os.path.dirname(__file__))
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
import torch, torch.nn.functional as F

# Stub out GPU-only dependencies
def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)

sys.modules['flash_attn_interface'] = types.SimpleNamespace(flash_attn_func=_cpu_fa3)
sys.modules.setdefault('sentencepiece', types.SimpleNamespace(SentencePieceProcessor=None))
sys.modules.setdefault('torch.distributed', types.SimpleNamespace())
sys.modules.setdefault('torch.nn.parallel', types.SimpleNamespace(DistributedDataParallel=None))
sys.modules.setdefault('zstandard', types.SimpleNamespace())

import train_gpt as tg

# Small-scale config for CPU smoke test
V   = 512   # vocab
K   = 3     # num_unique_blocks
R   = 2     # num_iterations → 6 virtual layers
d   = 64    # model_dim
B, T = 2, 32

model = tg.GPT(
    vocab_size=V,
    num_layers=K * R,
    num_unique_blocks=K,
    num_iterations=R,
    model_dim=d,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=3,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000,
    qk_gain_init=1.0,
    mtp_num_heads=0,
    mtp_loss_weight=0.0,
    bigram_vocab_size=256,
    bigram_dim=32,
    xsa_last_n=1,
    rope_dims=8,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
    ve_dim=64,
    ve_layers="4,5",
    gated_attention=False,
    value_residual=False,
)

# ------- Forward -------
model.train()
ids = torch.randint(0, V, (B, T))
tgt = torch.randint(0, V, (B, T))
loss = model(ids, tgt)
assert torch.isfinite(loss), f"Loss not finite: {loss}"
print(f"  forward OK  loss={loss.item():.4f}")

# ------- Backward -------
loss.backward()
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is None:
        print(f"  WARN: no grad for {name}")
print("  backward OK")

# ------- Optimizer split (mirrors main()) -------
matrix_params = [
    model.qo_bank, model.kv_bank,
    model.mlp_up_bank, model.mlp_down_bank,
]
block_named = list(model.blocks.named_parameters())
scalar_params = [
    p for name, p in block_named
    if p.ndim < 2 or any(pat in name for pat in tg.CONTROL_TENSOR_NAME_PATTERNS)
]
if model.skip_weights.numel() > 0:
    scalar_params.append(model.skip_weights)
scalar_params.append(model.smear.gate)
if model.bigram is not None:
    scalar_params.append(model.bigram.scale)
# MoR extra scalars
scalar_params.extend(list(model.v_attn_scale.parameters()))
scalar_params.extend(list(model.v_mlp_scale.parameters()))
scalar_params.extend(list(model.v_resid_mix.parameters()))
if model.bigram is not None:
    if model.bigram.proj is not None:
        scalar_params.append(model.bigram.proj.weight)

tok_params = [model.tok_emb.weight]
if model.bigram is not None:
    tok_params.append(model.bigram.embed.weight)

opt = torch.optim.AdamW(scalar_params + tok_params, lr=1e-3)
opt.step()
opt.zero_grad()
print("  optimizer step OK")

# ------- Inference -------
model.eval()
with torch.no_grad():
    logits = model.forward_logits(ids)
assert logits.shape == (B, T, V), f"Wrong logits shape: {logits.shape}"
print(f"  inference OK  logits.shape={logits.shape}")

# ------- Quantization roundtrip -------
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
unbanked = tg._unbank_state_dict(sd, K)
quant_result, quant_meta = tg.mixed_quantize_int6(unbanked, {"mlp", "attn"})
deq_unbanked = tg.dequantize_mixed_int6(quant_result, quant_meta, unbanked)
deq_state = tg._rebank_state_dict(deq_unbanked, K, sd)
assert set(deq_state.keys()) == set(sd.keys()), \
    f"Key mismatch: {set(sd.keys()) - set(deq_state.keys())} missing, {set(deq_state.keys()) - set(sd.keys())} extra"
print("  quantization roundtrip OK")

n_params = sum(p.numel() for p in model.parameters())
print(f"\nPASS: Smoke test OK  params={n_params:,}  loss={loss.item():.4f}  K={K}  R={R}  virtual_layers={K*R}")
