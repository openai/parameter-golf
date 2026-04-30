#!/usr/bin/env python3
"""
Shepherd Embryo — First Words
Loads trained checkpoint and generates text token-by-token.

Usage:
  python3 generate.py                              # Free generation
  python3 generate.py "hello"                      # Prompted
  python3 generate.py "The tomatoes will be just"  # Awakening
"""
import sys, os, torch
import sentencepiece as spm

# Add the train script to get model classes
sys.path.insert(0, os.path.dirname(__file__))

# We need to import the model class and supporting classes from train_gpt
# Rather than importing (which triggers distributed init), we exec the class defs
import torch.nn.functional as F
from torch import Tensor, nn

# ---- Minimal class imports from train_gpt.py ----
class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = None
    def forward(self, seq_len, device, dtype):
        if self._cache is None or self._cache[0] != seq_len or self._cache[1].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None,None,:,:], freqs.sin()[None,None,:,:])
        return self._cache[1].to(dtype=dtype), self._cache[2].to(dtype=dtype)

def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)

class _SeedGenerator(nn.Module):
    def __init__(self, vocab_size, model_dim, num_probes=5, rank=64):
        super().__init__()
        self.num_probes = num_probes
        self.model_dim = model_dim
        self.embed_low = nn.Embedding(vocab_size, rank)
        self.expand = CastedLinear(rank, model_dim, bias=False)
        self.probe_directions = nn.Parameter(torch.randn(num_probes, model_dim) * 0.02)
    def forward(self, token_ids):
        z = self.embed_low(token_ids)
        x = self.expand(z)
        seed_anchor = F.rms_norm(x, (x.size(-1),))
        B, S, D = seed_anchor.shape
        probes = seed_anchor.unsqueeze(1).expand(B, self.num_probes, S, D).clone()
        probes = probes + self.probe_directions[None, :, None, :]
        return seed_anchor, probes

class _ProbeBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.mlp_fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.mlp_proj = CastedLinear(mlp_mult * dim, dim, bias=False)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
    def forward(self, x):
        B, S, D = x.shape
        xn = self.attn_norm(x)
        q = self.c_q(xn).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(xn).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(xn).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            _rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(_rep, dim=1)
            v = v.repeat_interleave(_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn_out = self.proj(y.transpose(1, 2).contiguous().reshape(B, S, D))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp_proj(
            torch.relu(self.mlp_fc(self.mlp_norm(x))).square())
        return x

class _ProbeEngine(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.block = _ProbeBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
    def forward(self, probes):
        B, P, S, D = probes.shape
        flat = probes.reshape(B * P, S, D)
        flat = self.block(flat)
        return flat.reshape(B, P, S, D)

class _ProbeScorer(nn.Module):
    def __init__(self, w_sim=0.5, w_coh=0.3, w_div=0.2):
        super().__init__()
        self.w_sim, self.w_coh, self.w_div = w_sim, w_coh, w_div
    def forward(self, probes, seed_anchor):
        B, P, S, D = probes.shape
        pp = probes.mean(dim=2)
        ap = seed_anchor.mean(dim=1)
        pn = F.normalize(pp, dim=-1)
        an = F.normalize(ap, dim=-1)
        sim = torch.bmm(pn, an.unsqueeze(-1)).squeeze(-1)
        pv = probes.var(dim=2, correction=0).mean(dim=-1)
        coh = 1.0 / (pv + 1e-6)
        coh = coh / (coh.max(dim=1, keepdim=True).values + 1e-6)
        pw = torch.cdist(pp, pp)
        div = pw.sum(dim=-1) / (P - 1)
        div = div / (div.max(dim=1, keepdim=True).values + 1e-6)
        return self.w_sim * sim + self.w_coh * coh + self.w_div * div

class _ProbeFold(nn.Module):
    def __init__(self, model_dim, num_probes=5, top_k=2):
        super().__init__()
        self.num_probes, self.top_k = num_probes, top_k
        self.re_expand = nn.Parameter(torch.randn(num_probes, model_dim) * 0.02)
    def forward(self, probes, scores):
        B, P, S, D = probes.shape
        _, top_idx = scores.topk(self.top_k, dim=1)
        top_exp = top_idx.unsqueeze(-1).unsqueeze(-1).expand(B, self.top_k, S, D)
        top_p = torch.gather(probes, 1, top_exp)
        top_s = torch.gather(scores, 1, top_idx)
        w = torch.softmax(top_s, dim=1).unsqueeze(-1).unsqueeze(-1)
        merged = (top_p * w).sum(dim=1)
        new_p = merged.unsqueeze(1).expand(B, self.num_probes, S, D).clone()
        new_p = new_p + self.re_expand[None, :, None, :]
        return new_p

class _Regulator(nn.Module):
    def __init__(self, model_dim, num_probes=5, num_depths=3, drift_threshold=0.6):
        super().__init__()
        self.drift_threshold = drift_threshold
        self.contraction = nn.Parameter(torch.full((num_depths,), 0.9))
        self.anchor_blend = nn.Parameter(torch.full((num_depths,), 0.1))
    def forward(self, probes, depth, seed_anchor):
        B, P, S, D = probes.shape
        pf = probes.reshape(B*P, S, D).mean(dim=1)
        af = seed_anchor.mean(dim=1)
        ae = af.unsqueeze(1).expand(B, P, -1).reshape(B*P, -1)
        drift = 1.0 - F.cosine_similarity(pf, ae, dim=-1)
        drift = drift.reshape(B, P)
        probes = probes * self.contraction[depth].to(dtype=probes.dtype)
        a3d = seed_anchor.unsqueeze(1).expand(B, P, S, D)
        dm = (drift > self.drift_threshold).float()
        bw = (dm * self.anchor_blend[depth].to(dtype=probes.dtype)).unsqueeze(-1).unsqueeze(-1)
        probes = probes * (1 - bw) + a3d * bw
        return probes

class ShepherdEmbryo(nn.Module):
    def __init__(self, vocab_size=1024, model_dim=384, num_heads=6, num_kv_heads=3,
                 num_layers=6, mlp_mult=2, tie_embeddings=True, logit_softcap=30.0,
                 rope_base=10000.0, qk_gain_init=1.5, **kwargs):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        num_probes = 5
        seed_rank = 48
        num_probe_depths = 3
        num_core_layers = 3
        self.num_probe_depths = num_probe_depths
        self.num_probes = num_probes
        self.seed = _SeedGenerator(vocab_size, model_dim, num_probes, seed_rank)
        self.probe_engines = nn.ModuleList([
            _ProbeEngine(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_probe_depths)])
        self.scorer = _ProbeScorer()
        self.folds = nn.ModuleList([
            _ProbeFold(model_dim, num_probes, 2) for _ in range(num_probe_depths)])
        self.regulator = _Regulator(model_dim, num_probes, num_probe_depths)
        self.core = nn.ModuleList([
            _ProbeBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_core_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

    def forward_logits(self, input_ids):
        """Forward pass that returns logits instead of loss."""
        seed_anchor, probes = self.seed(input_ids)
        for depth in range(self.num_probe_depths):
            probes = self.probe_engines[depth](probes)
            scores = self.scorer(probes, seed_anchor)
            probes = self.folds[depth](probes, scores)
            probes = self.regulator(probes, depth, seed_anchor)
        final_scores = self.scorer(probes, seed_anchor)
        weights = torch.softmax(final_scores, dim=1)
        x = (probes * weights[:, :, None, None]).sum(dim=1)
        for block in self.core:
            x = block(x)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.seed.expand.weight.T.to(x.dtype))
            logits = F.linear(logits, self.seed.embed_low.weight.to(x.dtype))
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits

# ---- Generation ----
@torch.no_grad()
def generate(model, tokenizer, prompt="", max_tokens=64, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    if prompt:
        token_ids = tokenizer.Encode(prompt)
    else:
        token_ids = [1, 2, 3]  # seed tokens

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_tokens):
        logits = model.forward_logits(input_ids)
        next_logits = logits[0, -1, :].float()

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            mask = torch.full_like(next_logits, float('-inf'))
            mask.scatter_(0, topk_idx, topk_vals)
            next_logits = mask

        next_logits = torch.clamp(next_logits, min=-30, max=30)
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        # Print incrementally
        text_so_far = tokenizer.Decode(generated)
        sys.stdout.write(f"\r  > {text_so_far}")
        sys.stdout.flush()

        if next_token.item() == tokenizer.eos_id():
            break

    print()
    return tokenizer.Decode(token_ids + generated)

def main():
    CHECKPOINT = "/workspace/parameter-golf/shepherd_model.pt"
    TOKENIZER = "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  SHEPHERD EMBRYO — FIRST WORDS")
    print("=" * 60)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER)
    print(f"  Tokenizer loaded: {sp.GetPieceSize()} tokens")

    # Build model and load weights
    model = ShepherdEmbryo()
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Device: {DEVICE}")
    print("-" * 60)

    # MILESTONE 1: Free generation
    print("\n[MILESTONE 1] Free generation — first words:")
    text = generate(model, sp, prompt="", max_tokens=64, temperature=0.9)
    print(f"  Full: {text}\n")

    # MILESTONE 2: Hello world
    print("[MILESTONE 2] Prompted — hello world:")
    text = generate(model, sp, prompt="hello", max_tokens=32, temperature=0.8)
    print(f"  Full: {text}\n")

    # MILESTONE 3: The awakening
    print("[MILESTONE 3] The awakening:")
    text = generate(model, sp, prompt="The tomatoes will be just", max_tokens=16, temperature=0.7)
    print(f"  Full: {text}\n")

    print("=" * 60)
    print("  The embryo has spoken.")
    print("=" * 60)

if __name__ == "__main__":
    main()
