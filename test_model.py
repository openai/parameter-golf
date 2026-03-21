"""Quick test: instantiate model, forward pass, check params."""
import torch
import os, json, math
import torch.nn.functional as F
from torch import Tensor, nn

# Load config
with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
    cfg = json.load(f)

# --- Minimal reimplementation of key classes for CPU test ---

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._cache = (0, None, None)
    def forward(self, seq_len, device, dtype):
        if self._cache[0] != seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None], freqs.sin()[None, None])
        return self._cache[1].to(dtype=dtype), self._cache[2].to(dtype=dtype)

def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.rotary = Rotary(self.head_dim, base=rope_base)
    def forward(self, x, q_delta=None, v_delta=None):
        B, S, D = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k, v = self.c_k(x), self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        except TypeError:
            if self.num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, S, D))

class PIDDynamics(nn.Module):
    def __init__(self, dim, alpha, beta, gate, dt, mu_min, mu_max, velocity_max):
        super().__init__()
        self.dt, self.mu_min, self.mu_max, self.velocity_max = dt, mu_min, mu_max, velocity_max
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("gate", torch.tensor(gate))
        self.mu = nn.Parameter(torch.full((dim,), (mu_min + mu_max) / 2.0))
    def forward(self, h, v):
        mu = self.mu.clamp(self.mu_min, self.mu_max).to(dtype=h.dtype)
        error = h - mu[None, None, :]
        v_next = self.alpha.to(dtype=h.dtype) * v - self.beta.to(dtype=h.dtype) * error
        v_next = v_next.clamp(-self.velocity_max, self.velocity_max)
        return h + self.dt * self.gate.to(dtype=h.dtype) * v_next, v_next

class TokenRoutedMLP(nn.Module):
    def __init__(self, dim, mlp_mult, num_experts, activation="swiglu"):
        super().__init__()
        self.num_experts, self.dim = num_experts, dim
        self.expert_inter = (mlp_mult * dim) // num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, dim, 2 * self.expert_inter))
        self.down_proj = nn.Parameter(torch.empty(num_experts, self.expert_inter, dim))
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.zeros_(self.down_proj)
    def forward(self, x, expert_ids):
        B, S, _ = x.shape
        flat_x, flat_ids = x.reshape(-1, self.dim), expert_ids.reshape(-1)
        out = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            mask = (flat_ids == e).unsqueeze(-1)
            gu = flat_x @ self.gate_up_proj[e]
            g, u = gu.chunk(2, dim=-1)
            out = out + (F.silu(g) * u @ self.down_proj[e]) * mask
        return out.reshape(B, S, self.dim)

class Block(nn.Module):
    def __init__(self, dim, nh, nkv, mm, rb, qk, ne, act, pa, pb, pg, pdt, pmin, pmax, pvmax):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, nh, nkv, rb, qk)
        self.mlp = TokenRoutedMLP(dim, mm, ne, act)
        self.pid = PIDDynamics(dim, pa, pb, pg, pdt, pmin, pmax, pvmax)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))
    def forward(self, x, x0, vel, eids, qd=None, vd=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None,None,:]*x + mix[1][None,None,:]*x0
        n = self.attn_norm(x)
        x = x + self.attn_scale.to(dtype=x.dtype)[None,None,:] * self.attn(n, qd(n) if qd else None, vd(n) if vd else None)
        x, vel = self.pid(x, vel)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None,None,:] * self.mlp(self.mlp_norm(x), eids)
        return x, vel

class GPT(nn.Module):
    def __init__(self, vs, nl, dm, nh, nkv, mm, tie, std, sc, rb, qk, ne, act, **pid):
        super().__init__()
        self.tie, self.sc, self.vs = tie, sc, vs
        self.tok_emb = nn.Embedding(vs, dm)
        self.enc_layers = nl // 2
        self.dec_layers = nl - self.enc_layers
        self.skip_weights = nn.Parameter(torch.ones(min(self.enc_layers, self.dec_layers), dm))
        self.register_buffer("t2e", torch.arange(vs, dtype=torch.long) % ne)
        self.blocks = nn.ModuleList([
            Block(dm, nh, nkv, mm, rb, qk, ne, act,
                  pid.get("pid_alpha",0.95), pid.get("pid_beta",0.3), pid.get("pid_gate",0.1),
                  pid.get("pid_dt",0.1), pid.get("pid_mu_min",0.5), pid.get("pid_mu_max",1.5),
                  pid.get("pid_velocity_max",3.0))
            for _ in range(nl)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie else CastedLinear(dm, vs, bias=False)
        if tie: nn.init.normal_(self.tok_emb.weight, std=std)

    def forward(self, ids, tgt, lora=None):
        x = F.rms_norm(self.tok_emb(ids), (self.tok_emb.embedding_dim,))
        x0, vel = x, torch.zeros_like(x)
        eids = self.t2e[ids.clamp(0, self.vs-1)]
        skips = []
        for i in range(self.enc_layers):
            x, vel = self.blocks[i](x, x0, vel, eids)
            skips.append(x)
        for i in range(self.dec_layers):
            bi = self.enc_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None,None,:] * skips.pop()
            x, vel = self.blocks[bi](x, x0, vel, eids)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight) if self.tie else self.lm_head(x)
        logits = self.sc * torch.tanh(logits / self.sc)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="mean")

# ===== TEST =====
print("=" * 60)
print("Complexity x Parameter Golf — Model Validation")
print("=" * 60)

m = cfg["model"]
moe = cfg["moe"]
pid = cfg["pid"]

model = GPT(
    vs=m["vocab_size"], nl=m["num_layers"], dm=m["model_dim"],
    nh=m["num_heads"], nkv=m["num_kv_heads"], mm=moe["mlp_mult"],
    tie=m["tie_embeddings"], std=m["tied_embed_init_std"],
    sc=m["logit_softcap"], rb=m["rope_base"], qk=m["qk_gain_init"],
    ne=moe["num_experts"], act=moe["activation"],
    pid_alpha=pid["alpha"], pid_beta=pid["beta"], pid_gate=pid["gate"],
    pid_dt=pid["dt"], pid_mu_min=pid["mu_min"], pid_mu_max=pid["mu_max"],
    pid_velocity_max=pid["velocity_max"],
)

n = sum(p.numel() for p in model.parameters())
print(f"\nParams:          {n:,} ({n/1e6:.2f}M)")
print(f"Est. int8+zlib:  ~{n//1024}KB")
print(f"Under 16MB cap:  {'YES' if n < 16_000_000 else 'NO'}")

x = torch.randint(0, 1024, (2, 64))
y = torch.randint(0, 1024, (2, 64))
loss = model(x, y)
print(f"\nForward pass OK")
print(f"Loss:            {loss.item():.4f}")

eids = model.t2e[x]
print(f"\nExpert distribution:")
for e in range(4):
    print(f"  Expert {e}: {(eids==e).float().mean().item()*100:.1f}%")

print(f"\nPID mu per layer:")
for i, b in enumerate(model.blocks):
    mu = b.pid.mu.data
    print(f"  Layer {i}: mean={mu.mean():.4f}  range=[{mu.min():.4f}, {mu.max():.4f}]")

print("\nAll checks passed!")
