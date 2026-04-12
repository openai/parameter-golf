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


class BigramHashEmbedding(nn.Module):
    """Learned bigram-pair correction on top of token embeddings.

    Uses a compact hash table (hash_size << vocab^2) to store per-bigram
    residual embeddings. The hash maps (prev_token, cur_token) → bucket via
    a simple polynomial: (prev * vocab_size + cur) % hash_size.
    """
    def __init__(self, hash_size: int, model_dim: int, scale: float = 0.05, vocab_size: int = 1024):
        super().__init__()
        self.hash_size = hash_size
        self.scale = scale
        self.vocab_size = vocab_size
        self.table = nn.Embedding(hash_size, model_dim)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.002)

    def forward(self, input_ids: Tensor) -> Tensor:
        # Pad left with 0 (BOS sentinel) to create prev_ids
        prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
        hash_idx = (prev_ids.long() * self.vocab_size + input_ids.long()) % self.hash_size
        return self.table(hash_idx) * self.scale


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
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, num_steps: int, rank: int, lora_scope: str = "q"):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim

        # Q always has dual-Q LoRA (near-full-rank second projection)
        self.c_q = RelaxedLinear(dim, dim, num_steps, rank, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)

        # V has LoRA only in 'qv' or 'full' scope
        self.use_v_lora = lora_scope in ("qv", "full")
        if self.use_v_lora:
            self.c_v = RelaxedLinear(dim, kv_dim, num_steps, rank, bias=False)
        else:
            self.c_v = CastedLinear(dim, kv_dim, bias=False)

        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.v_step_bias = nn.Parameter(torch.empty(num_steps, num_kv_heads, self.head_dim))
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, step_idx: int | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, step_idx).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_input = self.c_v(x, step_idx) if self.use_v_lora else self.c_v(x)
        v = v_input.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if step_idx is not None:
            v_bias = self.v_step_bias[step_idx].to(dtype=v.dtype)
            v = v + v_bias[None, :, None, :]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, num_steps: int, rank: int, lora_scope: str = "q"):
        super().__init__()
        hidden = mlp_mult * dim
        self.use_lora = lora_scope == "full"
        if self.use_lora:
            self.fc = RelaxedLinear(dim, hidden, num_steps, rank, bias=False)
            self.proj = RelaxedLinear(hidden, dim, num_steps, rank, bias=False)
        else:
            self.fc = CastedLinear(dim, hidden, bias=False)
            self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor, step_idx: int | None = None) -> Tensor:
        if self.use_lora:
            lora_fc_out = 0.0
            if step_idx is not None:
                dtype = x.dtype
                a_fc = self.fc.lora_A[step_idx].to(dtype)
                b_fc = self.fc.lora_B[step_idx].to(dtype)
                lora_fc_out = (x @ b_fc.t()) @ a_fc.t() * self.fc.scaling
            x = fused_relu2(x, self.fc.weight.t()) + lora_fc_out
            return self.proj(x, step_idx)
        else:
            x = fused_relu2(x, self.fc.weight.t())
            return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, num_steps: int, rank: int,
                 lora_scope: str = "q"):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        num_steps, rank, lora_scope=lora_scope)
        self.mlp = MLP(dim, mlp_mult, num_steps, rank, lora_scope=lora_scope)
        self.attn_scale = nn.Parameter(torch.full((dim,), 1e-4, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), 1e-4, dtype=torch.float32))
        self.dropout = nn.Dropout(0.15)

    def forward(self, x: Tensor, x0: Tensor, step_idx: int | None = None) -> Tensor:
        if self.training:
            mask = (torch.rand(1, device=x.device) > 0.04).to(dtype=x.dtype)
        else:
            mask = 1.0

        attn_out = self.attn(self.attn_norm(x), step_idx)
        x = x + mask * self.attn_scale[None, None, :] * self.dropout(attn_out)
        x = x + mask * self.mlp_scale[None, None, :] * self.dropout(self.mlp(self.mlp_norm(x), step_idx))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_steps: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float = 10.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        lora_rank: int = 16,
        lora_scope: str = "q",
        bigram_hash_enabled: bool = False,
        bigram_hash_size: int = 2048,
        bigram_hash_scale: float = 0.05,
        level_signal_enabled: bool = False,
        level_rank: int | None = None,
    ):
        super().__init__()
        print(f"[debug] GPT init: steps={num_steps}, dim={model_dim}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_steps = num_steps
        self.step_embeddings = nn.Parameter(torch.randn(num_steps, model_dim) * 0.002)

        # Optional BigramHash correction on token embeddings
        self.bigram_hash = (
            BigramHashEmbedding(bigram_hash_size, model_dim, bigram_hash_scale, vocab_size)
            if bigram_hash_enabled else None
        )

        # RingFormer-style level signals (content-conditioned per-step injection)
        self.level_signal_enabled = bool(level_signal_enabled)
        if level_rank is None:
            level_rank = max(1, model_dim // 16)
        self.level_rank = int(level_rank)
        if self.level_signal_enabled:
            self.level_down = nn.Parameter(torch.empty(num_steps, model_dim, self.level_rank))
            self.level_up = nn.Parameter(torch.empty(num_steps, self.level_rank, model_dim))
            self.level_gain = nn.Parameter(torch.zeros(num_steps, dtype=torch.float32))
        else:
            self.level_down = None
            self.level_up = None
            self.level_gain = None

        print("[debug] GPT init: creating block...")
        self.block = Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                           qk_gain_init, num_steps, lora_rank, lora_scope=lora_scope)

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

        print("[debug] GPT init: casting params...")
        with torch.no_grad():
            self.step_embeddings.data = self.step_embeddings.data.to(dtype=torch.bfloat16)
            self.block.attn_scale.data = self.block.attn_scale.data.to(dtype=torch.bfloat16)
            self.block.mlp_scale.data = self.block.mlp_scale.data.to(dtype=torch.bfloat16)
            if self.level_signal_enabled:
                self.level_down.data = self.level_down.data.to(dtype=torch.bfloat16)
                self.level_up.data = self.level_up.data.to(dtype=torch.bfloat16)

        if self.lm_head is not None:
            self.lm_head._zero_init = True
        print("[debug] GPT init: calling _init_weights...")
        self._init_weights()

        print("[debug] GPT init: compiling block (mode=default)...")
        self.block = torch.compile(self.block, mode="default")
        print("[debug] GPT init: complete")

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.level_signal_enabled:
            nn.init.normal_(self.level_down, mean=0.0, std=0.002)
            nn.init.zeros_(self.level_up)
        for module in self.modules():
            if isinstance(module, (nn.Linear, CastedLinear)):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, RelaxedLinear):
                for i in range(module.num_steps):
                    nn.init.kaiming_uniform_(module.lora_A[i], a=math.sqrt(5))
                    nn.init.zeros_(module.lora_B[i])
            if isinstance(module, CausalSelfAttention):
                nn.init.normal_(module.v_step_bias, mean=0.0, std=0.002)

    def forward_logits(self, input_ids: Tensor, use_compiled: bool = True) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        x0 = x

        for i in range(self.num_steps):
            step_idx_tensor = torch.tensor(i, device=x.device, dtype=torch.int32)
            x = x + self.step_embeddings[i][None, None, :]
            if self.level_signal_enabled:
                down = self.level_down[i].to(dtype=x.dtype)
                up = self.level_up[i].to(dtype=x.dtype)
                signal = (x @ down) @ up
                gain = torch.tanh(self.level_gain[i]).to(dtype=x.dtype)
                x = x + gain * signal
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
