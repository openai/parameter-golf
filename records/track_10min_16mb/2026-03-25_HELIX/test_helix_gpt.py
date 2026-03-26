import sys, os, types, math
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn.functional as F
def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)
sys.modules['flash_attn_interface'] = types.SimpleNamespace(flash_attn_func=_cpu_fa3)
sys.modules.setdefault('sentencepiece', types.SimpleNamespace(SentencePieceProcessor=None))
sys.modules.setdefault('torch.distributed', types.SimpleNamespace())
sys.modules.setdefault('torch.nn.parallel', types.SimpleNamespace(DistributedDataParallel=None))
sys.modules.setdefault('zstandard', types.SimpleNamespace())
import train_gpt as tg
V, K, R, d = 1024, 5, 3, 64
n_heads, n_kv, rank = 4, 2, 2
ffn_h, rope_d = 128, 8
B, T = 2, 32
model = tg.HELIX_GPT(vocab_size=V, num_unique_blocks=K, num_iterations=R, model_dim=d, num_heads=n_heads, num_kv_heads=n_kv, dtpa_rank=rank, ffn_hidden=ffn_h, rope_dims=rope_d, xsa_last_n=2, bigram_vocab_size=512, bigram_dim=32, tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0)
assert hasattr(model, 'blocks')
assert hasattr(model, 'smear')
assert hasattr(model, 'bigram')
assert hasattr(model, 'tok_emb')
assert hasattr(model, 'skip_weights')
assert hasattr(model, 'mtp_heads')
assert model.mtp_num_heads == 0
assert hasattr(model, 'forward_logits')
assert model.lm_head is None
assert len(model.mor_gate) == R - 1
print(f"PASS: HELIX_GPT OK  n_blocks={len(model.blocks)}  n_iters={R}")
