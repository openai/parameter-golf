import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from triton_mlp import fused_relu2

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)

class RelaxedLinear(CastedLinear):
    def __init__(self, in_features, out_features, num_steps, rank, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.num_steps = num_steps
        self.rank = rank
        self.lora_A = nn.Parameter(torch.empty(num_steps, out_features, rank))
        self.lora_B = nn.Parameter(torch.empty(num_steps, rank, in_features))
        self.register_buffer("scaling", torch.tensor(1.0 / math.sqrt(rank), dtype=torch.float32))

    def forward(self, x: Tensor, step_idx: int | None = None) -> Tensor:
        y = super().forward(x)
        if step_idx is not None:
            dtype = x.dtype
            a = self.lora_A[step_idx].to(dtype)
            b = self.lora_B[step_idx].to(dtype)
            lora_y = torch.matmul(torch.matmul(x, b.transpose(-1, -2)), a.transpose(-1, -2))
            y = y + lora_y * self.scaling.to(dtype)
        return y


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float, num_steps: int, rank: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = RelaxedLinear(dim, dim, num_steps, rank, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = RelaxedLinear(dim, kv_dim, num_steps, rank, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.v_step_bias = nn.Parameter(torch.empty(num_steps, num_kv_heads, self.head_dim))
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, step_idx: int | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, step_idx).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x, step_idx).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if step_idx is not None:
            v_bias = self.v_step_bias[step_idx].to(dtype=v.dtype)
            v = v + v_bias[None, :, None, :] # Broadcast: (1, num_kv_heads, 1, head_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, num_steps: int, rank: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = RelaxedLinear(dim, hidden, num_steps, rank, bias=False)
        self.proj = RelaxedLinear(hidden, dim, num_steps, rank, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor, step_idx: int | None = None) -> Tensor:
        lora_fc_out = 0.0
        if step_idx is not None:
            dtype = x.dtype
            a_fc = self.fc.lora_A[step_idx].to(dtype)
            b_fc = self.fc.lora_B[step_idx].to(dtype)
            lora_fc_out = (x @ b_fc.t()) @ a_fc.t() * self.fc.scaling
        
        x = fused_relu2(x, self.fc.weight.t()) + lora_fc_out
        return self.proj(x, step_idx)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float, num_steps: int, rank: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, num_steps, rank)
        self.mlp = MLP(dim, mlp_mult, num_steps, rank)
        self.attn_scale = nn.Parameter(torch.full((dim,), 1e-4, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), 1e-4, dtype=torch.float32))
        self.dropout = nn.Dropout(0.15)

    def forward(self, x: Tensor, x0: Tensor, step_idx: int | None = None) -> Tensor:
        # Subtle Stochastic Depth (0.04 DropRate) for Safe-Speed Stability
        if self.training:
            mask = (torch.rand(1, device=x.device) > 0.04).to(dtype=x.dtype)
        else:
            mask = 1.0

        attn_out = self.attn(self.attn_norm(x), step_idx)
        # Apply mask to the recursive update branch
        x = x + mask * self.attn_scale[None, None, :] * self.dropout(attn_out)
        x = x + mask * self.mlp_scale[None, None, :] * self.dropout(self.mlp(self.mlp_norm(x), step_idx))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_steps: int, model_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float = 10.0, rope_base: float = 10000.0, qk_gain_init: float = 1.5, lora_rank: int = 16):
        super().__init__()
        print(f"[debug] GPT init: steps={num_steps}, dim={model_dim}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_steps = num_steps
        self.step_embeddings = nn.Parameter(torch.randn(num_steps, model_dim) * 0.002)
        
        print("[debug] GPT init: creating block...")
        self.block = Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, num_steps, lora_rank)
        
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        
        print("[debug] GPT init: casting params...")
        with torch.no_grad():
            self.step_embeddings.data = self.step_embeddings.data.to(dtype=torch.bfloat16)
            self.block.attn_scale.data = self.block.attn_scale.data.to(dtype=torch.bfloat16)
            self.block.mlp_scale.data = self.block.mlp_scale.data.to(dtype=torch.bfloat16)
        
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        print("[debug] GPT init: calling _init_weights...")
        self._init_weights()
        
        # MEGA-KERNEL OPTIMIZATION: Block-level compilation (Stabilized)
        # We compile at the Block level to avoid massive "Whole-GPT" Inductor overhead.
        # "default" is the stablest mode for Windows with checkpointing.
        print("[debug] GPT init: compiling block (mode=default)...")
        self.block = torch.compile(self.block, mode="default")
        print("[debug] GPT init: complete")

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            # Standard Linear weights and projections
            if isinstance(module, (nn.Linear, CastedLinear)):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    # Xavier Initialization for stable unit variance across recurrences
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            # Universal Transformer LoRA Adapters
            if isinstance(module, RelaxedLinear):
                for i in range(module.num_steps):
                    # Kaiming for A (input), Zero for B (output) to start as Identity
                    nn.init.kaiming_uniform_(module.lora_A[i], a=math.sqrt(5))
                    nn.init.zeros_(module.lora_B[i])
            
            # AutoResearch Value Embeddings (Initial Shortcut)
            if isinstance(module, CausalSelfAttention):
                # Small standard deviation to avoid overwhelming the initial residue logic
                nn.init.normal_(module.v_step_bias, mean=0.0, std=0.002)

    def forward_logits(self, input_ids: Tensor, use_compiled: bool = True) -> Tensor:
        x = self.tok_emb(input_ids)
        x0 = x

        # MEGA-KERNEL OPTIMIZATION: Elite Standard 9.5 (Direct Unroll)
        # We disable checkpointing to avoid the Inductor metadata mismatch.
        # Micro-batch size 128 correctly fits in 24GB VRAM for 12 steps.
        
        for i in range(self.num_steps):
            step_idx_tensor = torch.tensor(i, device=x.device, dtype=torch.int32)
            # Inject Step Embeddings before the Block as per SOTA UT spec
            x = x + self.step_embeddings[i][None, None, :]
            # Direct call to block (compiled vs eager toggle)
            block_fn = self.block if use_compiled else getattr(self.block, "_orig_mod", self.block)
            x = block_fn(x, x0, step_idx_tensor)
        
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean", label_smoothing=0.05)
