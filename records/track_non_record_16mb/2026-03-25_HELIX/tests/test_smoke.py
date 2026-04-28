"""Full forward/backward/optimizer smoke test on CPU."""
import sys, os, types, math, io
sys.path.insert(0, os.path.dirname(__file__))
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
import torch, torch.nn.functional as F

def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
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

V, K, R, d = 1024, 3, 2, 64
B, T = 2, 32
model = tg.HELIX_GPT(
    vocab_size=V, num_unique_blocks=K, num_iterations=R,
    model_dim=d, num_heads=4, num_kv_heads=2, dtpa_rank=2,
    ffn_hidden=128, rope_dims=8, xsa_last_n=1,
    bigram_vocab_size=256, bigram_dim=32,
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
)

# Forward
model.train()
model._current_lb_weight = 0.01
ids = torch.randint(0, V, (B, T))
tgt = torch.randint(0, V, (B, T))
loss = model(ids, tgt)
assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
print(f"  forward OK  loss={loss.item():.4f}")

# Backward
loss.backward()
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is None:
        print(f"  WARN: no grad for {name}")
print("  backward OK")

# Optimizer step
all_block_named = list(model.blocks.named_parameters())
matrix_params = [
    p for n, p in all_block_named
    if p.ndim == 2
    and not any(pat in n for pat in tg.CONTROL_TENSOR_NAME_PATTERNS)
    and not any(x in n for x in ('A_q', 'A_k', 'A_v'))
]
scalar_params = [
    p for n, p in all_block_named
    if p.ndim != 2
    or any(pat in n for pat in tg.CONTROL_TENSOR_NAME_PATTERNS)
    or any(x in n for x in ('A_q', 'A_k', 'A_v'))
]
for p in model.mor_gate.parameters():
    scalar_params.append(p)
scalar_params.extend([model.smear.gate, model.bigram.scale, model.skip_weights])
if model.bigram.proj is not None:
    scalar_params.append(model.bigram.proj.weight)

opt_adam = torch.optim.AdamW(scalar_params + [model.tok_emb.weight, model.bigram.embed.weight], lr=1e-3)
opt_adam.step()
opt_adam.zero_grad()
print("  optimizer step OK")

# Inference
model.eval()
with torch.no_grad():
    logits = model.forward_logits(ids)
assert logits.shape == (B, T, V)
print(f"  inference OK  logits.shape={logits.shape}")

# Quantization roundtrip
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
quant_result, quant_meta = tg.mixed_quantize_int6(sd, {"mlp", "attn"})
deq_sd = tg.dequantize_mixed_int6(quant_result, quant_meta, sd)
assert set(deq_sd.keys()) == set(sd.keys()), "Key mismatch after quantization roundtrip"
print("  quantization roundtrip OK")

n_params = sum(p.numel() for p in model.parameters())
print(f"\nPASS: Smoke test OK  params={n_params:,}  loss={loss.item():.4f}")
