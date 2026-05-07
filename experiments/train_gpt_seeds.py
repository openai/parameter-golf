"""
Experiment: Seeded Random Basis + LoRA Adapters
================================================
"Learning adapters on random linear maps" — explicit wish list item in parameter-golf README.

CONCEPT:
  ALL weight matrices are generated on-the-fly from integer seeds at runtime.
  Only small rank-8/rank-4 LoRA adapters (A, B matrices) are actually stored
  in the 16MB artifact.  The ~128M random-basis model is never stored.

  W_effective = W_random(seed) + B @ A    (full rank ≈ but LoRA adapts it)

PARAMETER BUDGET (512-dim, 11 layers):
  Full baseline matrices:  11 × (4×512×512 + 512×2048 + 2048×512) ≈ 24M params
  LoRA A+B (rank 8 attn, rank 4 mlp): 11 × 4 × (512×8 + 8×512) + 11 × 2 × (512×4 + 4×512) ≈ 440K params
  = ~98% parameter reduction on weight matrices

STORAGE PLAN:
  Matrices:  0 bytes (regenerated from seeds)
  LoRA A,B:  440K × int6 ≈ 330KB
  Embeddings: vocab × dim × int8 = 8192 × 512 × 1B = 4MB
  Seeds list: 11 × 4 layers × 4 bytes = 176 bytes (negligible)
  All control vectors (gains, norms): ~tiny
  TOTAL: well under 16MB → can use higher-precision LoRA or larger rank!

FAST RANDOM MATRIX:
  Use PyTorch Generator-based structured randomness.
  We use Kronecker / Hadamard-like structure for efficient matmul
  (FastFood transform: W ≈ S·H·G·Π·H·B where H=Hadamard, others are diagonal/perm)
  This makes W@x O(n log n) instead of O(n²).

TRAINING:
  1. Generate random W from seed (on device); use @=no_grad
  2. Compute y = F.linear(x, W) + F.linear(F.linear(x, A), B)   (LoRA addition)
  3. Backprop only through A,B
  4. At quantization/compression time: only A,B,embeddings, seed list are stored

EVAL:
  Standard sliding window BPB (same eval infrastructure as main submission)

RISKS:
  - Random bases may not capture useful inductive biases (untrained)
  - Training signal may get diluted by large random component
  - FastFood approx random vs full random: different inductive theory
  - Attention QK with purely random weights may degrade attn quality

MITIGATION:
  - Add skip connections: y = x + scale * (W_random @ x + B @ A @ x)
    so gradient path always has identity
  - Use gradient checkpointing (recompute random W at backward pass)
  - Start with smaller rank-2 and see if loss decreases at all

TO RUN (1xH100, ablation mode):
  RUN_ID=seeds_smoke \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  MAX_WALLCLOCK_SECONDS=0 \
  ITERATIONS=1000 \
  LORA_RANK_ATTN=8 \
  LORA_RANK_MLP=4 \
  torchrun --standalone --nproc_per_node=1 experiments/train_gpt_seeds.py
"""

import collections, copy, glob, io, math, os
from pathlib import Path
import random, sys, time, uuid
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FLASH3 = True
except ImportError:
    HAS_FLASH3 = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
class Hyperparameters:
    data_dir                 = os.environ.get('DATA_DIR', './data/')
    seed                     = int(os.environ.get('SEED', 1337))
    run_id                   = os.environ.get('RUN_ID', str(uuid.uuid4()))
    iterations               = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac            = float(os.environ.get('WARMDOWN_FRAC', 0.72))
    warmup_steps             = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens       = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len            = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every          = int(os.environ.get('TRAIN_LOG_EVERY', 200))
    max_wallclock_seconds    = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600))
    val_batch_tokens         = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    eval_seq_len             = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every           = int(os.environ.get('VAL_LOSS_EVERY', 500))
    sliding_window_enabled   = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))
    # Model
    vocab_size               = int(os.environ.get('VOCAB_SIZE', 8192))
    model_dim                = int(os.environ.get('MODEL_DIM', 512))
    num_heads                = int(os.environ.get('NUM_HEADS', 8))
    num_kv_heads             = int(os.environ.get('NUM_KV_HEADS', 4))
    num_layers               = int(os.environ.get('NUM_LAYERS', 11))
    mlp_mult                 = float(os.environ.get('MLP_MULT', 4.0))
    rope_base                = float(os.environ.get('ROPE_BASE', 1e4))
    rope_dims                = int(os.environ.get('ROPE_DIMS', 16))
    logit_softcap            = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    qk_gain_init             = float(os.environ.get('QK_GAIN_INIT', 5.5))
    # Seed-LoRA specific
    lora_rank_attn           = int(os.environ.get('LORA_RANK_ATTN', 8))
    lora_rank_mlp            = int(os.environ.get('LORA_RANK_MLP', 4))
    # Whether to use FastFood structured random (True) or dense random (False)
    # Dense is more expressive but slower; FastFood scales O(n log n).
    use_fastfood             = bool(int(os.environ.get('USE_FASTFOOD', '0')))
    # Scale factor for random basis output (keeps activations in range)
    random_basis_scale       = float(os.environ.get('RANDOM_BASIS_SCALE', 1.0))
    # Whether the random basis has its own scale parameter (learnable)
    learn_random_scale       = bool(int(os.environ.get('LEARN_RANDOM_SCALE', '1')))
    # Optimizer
    min_lr                   = float(os.environ.get('MIN_LR', 0.0))
    tied_embed_lr            = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std      = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    lora_lr                  = float(os.environ.get('LORA_LR', 0.01))
    scalar_lr                = float(os.environ.get('SCALAR_LR', 0.02))
    beta1                    = float(os.environ.get('BETA1', 0.9))
    beta2                    = float(os.environ.get('BETA2', 0.95))
    adam_eps                 = float(os.environ.get('ADAM_EPS', 1e-8))
    adam_wd                  = float(os.environ.get('ADAM_WD', 0.02))
    embed_wd                 = float(os.environ.get('EMBED_WD', 0.085))
    grad_clip_norm           = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    ema_decay                = float(os.environ.get('EMA_DECAY', 0.9965))
    eval_stride              = int(os.environ.get('EVAL_STRIDE', 64))
    # Quantization
    compressor               = os.environ.get('COMPRESSOR', 'brotli')
    lora_bits                = int(os.environ.get('LORA_BITS', 6))
    embed_bits               = int(os.environ.get('EMBED_BITS', 8))
    lora_clip_sigmas         = float(os.environ.get('LORA_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas        = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds     = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    # Distributed
    distributed      = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank             = int(os.environ.get('RANK', '0'))
    world_size       = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank       = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process  = rank == 0
    grad_accum_steps = 8 // world_size
    # Derived
    datasets_dir         = os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files          = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files            = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path       = os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')
    logfile              = f'logs/{run_id}.txt'


_log_h = None


def set_log_h(h):
    global _log_h
    _log_h = h


def log(msg):
    if _log_h is None:
        print(msg)
        return
    if _log_h.is_main_process:
        print(msg)
        if _log_h.logfile:
            with open(_log_h.logfile, 'a') as f:
                print(msg, file=f)


# ---------------------------------------------------------------------------
# Random basis generation (stateless, from seed)
# ---------------------------------------------------------------------------
def generate_random_matrix(seed: int, out_features: int, in_features: int,
                            device, dtype) -> torch.Tensor:
    """
    Generate a random Gaussian weight matrix from an integer seed.
    This is NEVER stored — regenerated identically on every call.
    Normalized by 1/sqrt(in_features) like a standard linear init.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    W = torch.randn(out_features, in_features, dtype=dtype, device=device, generator=g)
    W = W * (1.0 / math.sqrt(in_features))
    return W


class FastFoodTransform(nn.Module):
    """
    Structured random matrix: W ≈ D2 @ H @ G @ Pi @ H @ D1
    where D are random ±1 diagonal matrices (seeded), H is Walsh-Hadamard,
    G is a random Gaussian diagonal, Pi is a random permutation.
    Matrix-vector product is O(n log n) instead of O(n²).
    Only diagonal seeds and perm are "stored" (~n integers).

    This is an approximation of a random Gaussian matrix with better
    memory locality and faster GEMM.
    """
    def __init__(self, dim: int, seed: int, device, dtype):
        super().__init__()
        assert (dim & (dim - 1)) == 0, f"FastFood requires power-of-2 dim, got {dim}"
        self.dim = dim
        g = torch.Generator()
        g.manual_seed(seed)
        d1 = (torch.randint(0, 2, (dim,), generator=g).float() * 2 - 1).to(device=device, dtype=dtype)
        g2 = torch.Generator()
        g2.manual_seed(seed + 1)
        d2 = (torch.randint(0, 2, (dim,), generator=g2).float() * 2 - 1).to(device=device, dtype=dtype)
        g3 = torch.Generator()
        g3.manual_seed(seed + 2)
        gauss_d = torch.randn(dim, generator=g3).abs().to(device=device, dtype=dtype)
        g4 = torch.Generator()
        g4.manual_seed(seed + 3)
        perm = torch.randperm(dim, generator=g4).to(device=device)
        self.register_buffer('d1', d1, persistent=False)
        self.register_buffer('d2', d2, persistent=False)
        self.register_buffer('gauss_d', gauss_d, persistent=False)
        self.register_buffer('perm', perm, persistent=False)
        # Normalization: E[||y||²] = ||x||² for this construction
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] → [..., dim]"""
        x = x * self.d1
        x = self._fwht(x)
        x = x[:, self.perm] if x.ndim == 2 else x[..., self.perm]
        x = x * self.gauss_d
        x = self._fwht(x)
        x = x * self.d2
        return x * self.scale

    @staticmethod
    def _fwht(x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh-Hadamard Transform (iterative, in-place)."""
        n = x.size(-1)
        h = 1
        while h < n:
            x = x.reshape(*x.shape[:-1], n // (2 * h), 2 * h)
            a = x[..., :h]
            b = x[..., h:]
            x = torch.cat([a + b, a - b], dim=-1)
            x = x.reshape(*x.shape[:-2], n)
            h *= 2
        return x


# ---------------------------------------------------------------------------
# Seeded LoRA Linear Layer
# ---------------------------------------------------------------------------
class SeededLoRALinear(nn.Module):
    """
    Linear layer with:
      - A random weight matrix W_random(seed) — never stored, regenerated at runtime
      - A small LoRA adaptation delta: output += lora_B @ lora_A @ input
      - Optional learnable scale for the random component

    During serialization: only lora_A, lora_B, and the seed integer are stored.
    The random basis is reconstructed from the seed during deserialization.
    """
    def __init__(self, in_features: int, out_features: int, seed: int, rank: int,
                 use_fastfood: bool = False, learn_random_scale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        self.rank = rank
        self.use_fastfood = use_fastfood
        assert in_features == out_features or not use_fastfood, \
            "FastFood requires in_features == out_features"
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        # Random basis is NOT a parameter — regenerated on every forward pass
        # (use register_buffer for fast access without grad)
        # Optional learnable scale for the random component
        if learn_random_scale:
            self.rand_scale = nn.Parameter(torch.ones(1))
        else:
            self.rand_scale = None
        # Initialize LoRA A with small values (standard LoRA init)
        nn.init.normal_(self.lora_A, std=0.02 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Random component: W_random @ x (no gradient through W_random)
        with torch.no_grad():
            W_rand = generate_random_matrix(self.seed, self.out_features, self.in_features,
                                            device=x.device, dtype=x.dtype)
        rand_out = F.linear(x, W_rand)   # Note: W_rand has no grad, but x does
        # rand_out.detach_() # WRONG: would kill attn/mlp gradient path; random component OK
        # Scale random output
        if self.rand_scale is not None:
            rand_out = rand_out * self.rand_scale.to(dtype=x.dtype)
        # LoRA component (has gradient)
        lora_out = F.linear(F.linear(x, self.lora_A.to(dtype=x.dtype)), self.lora_B.to(dtype=x.dtype))
        return rand_out + lora_out

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return (f'in={self.in_features}, out={self.out_features}, '
                f'rank={self.rank}, seed={self.seed}, fastfood={self.use_fastfood}')


# ---------------------------------------------------------------------------
# Model reusing same blocks/norms as reference but with SeededLoRA matrices
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=2048, rope_dims=0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.base = base
        self.train_seq_len = train_seq_len
        inv_freq = 1.0 / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cache = {}

    def forward(self, seq_len, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._cache:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cache[key] = (freqs.cos()[None, :, None, :].to(dtype),
                                freqs.sin()[None, :, None, :].to(dtype))
        return self._cache[key]


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        return torch.cat((torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1), x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class SeededSelfAttention(nn.Module):
    def __init__(self, h, layer_idx: int, seed_offset: int = 0):
        super().__init__()
        dim = h.model_dim
        num_heads = h.num_heads
        num_kv_heads = h.num_kv_heads
        head_dim = dim // num_heads
        kv_dim = num_kv_heads * head_dim
        rank = h.lora_rank_attn
        use_ff = h.use_fastfood and (dim == dim)  # square only
        base_seed = (layer_idx * 7 + seed_offset) * 1000 + 42
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.c_q = SeededLoRALinear(dim, dim, base_seed + 0, rank, use_fastfood=False,
                                    learn_random_scale=h.learn_random_scale)
        self.c_k = SeededLoRALinear(dim, kv_dim, base_seed + 1, rank, use_fastfood=False,
                                    learn_random_scale=h.learn_random_scale)
        self.c_v = SeededLoRALinear(dim, kv_dim, base_seed + 2, rank, use_fastfood=False,
                                    learn_random_scale=h.learn_random_scale)
        self.proj = SeededLoRALinear(dim, dim, base_seed + 3, rank, use_fastfood=False,
                                     learn_random_scale=h.learn_random_scale)
        self.q_gain = nn.Parameter(torch.full((num_heads,), h.qk_gain_init, dtype=torch.float32))
        self.rope_dims = h.rope_dims
        self.rotary = Rotary(head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FLASH3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
            v = v.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        return self.proj(y.reshape(bsz, seqlen, dim))


class SeededMLP(nn.Module):
    def __init__(self, h, layer_idx: int, seed_offset: int = 100):
        super().__init__()
        dim = h.model_dim
        hidden = int(h.mlp_mult * dim)
        rank = h.lora_rank_mlp
        base_seed = (layer_idx * 7 + seed_offset) * 1000 + 77
        self.fc = SeededLoRALinear(dim, hidden, base_seed + 0, rank, use_fastfood=False,
                                   learn_random_scale=h.learn_random_scale)
        self.proj = SeededLoRALinear(hidden, dim, base_seed + 1, rank, use_fastfood=False,
                                     learn_random_scale=h.learn_random_scale)

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class SeededBlock(nn.Module):
    def __init__(self, h, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = SeededSelfAttention(h, layer_idx)
        self.mlp = SeededMLP(h, layer_idx)
        self.attn_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))

    def forward(self, x):
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class SeededGPT(nn.Module):
    """
    GPT with seeded random basis + LoRA adapters.
    Only LoRA weights and token embeddings are stored; everything else is seeds.
    """
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.blocks = nn.ModuleList([SeededBlock(h, i) for i in range(h.num_layers)])
        self.final_norm = RMSNorm()
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=h.tied_embed_init_std)

    def _count_lora_params(self):
        count = 0
        for name, p in self.named_parameters():
            if 'lora_' in name or 'rand_scale' in name:
                count += p.numel()
        return count

    def forward_logits(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1))


# ---------------------------------------------------------------------------
# Data loading (same as main submission — reuse the same infrastructure)
# ---------------------------------------------------------------------------
_SHARD_HEADER_BYTES = 256 * np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    if key not in _SHARD_NTOKENS_CACHE:
        header = np.fromfile(file, dtype='<i4', count=256)
        _SHARD_NTOKENS_CACHE[key] = int(header[2])
    return _SHARD_NTOKENS_CACHE[key]


def _get_shard_memmap(file):
    key = str(file)
    if key not in _MMAP_CACHE:
        n = _read_num_tokens(file)
        _MMAP_CACHE[key] = np.memmap(file, mode='r', dtype='<u2',
                                     offset=_SHARD_HEADER_BYTES, shape=(n,))
    return _MMAP_CACHE[key]


def load_data_shard(file):
    header = np.fromfile(file, dtype='<i4', count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=_SHARD_HEADER_BYTES)
    return torch.from_numpy(tokens_np.astype(np.int64))


def load_validation_tokens(pattern, seq_len):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files matching {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files])
    usable = (tokens.numel() - 1) // seq_len * seq_len
    return tokens[:usable + 1]


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    table_size = max(sp_vs, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith('▁'):
            has_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode('utf-8'))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_np, dtype=torch.bool, device=device))


class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        luts = build_sentencepiece_luts(self.sp, h.vocab_size, device)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = luts


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        seq_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + seq_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (1 * grad_accum_steps)
        bsz = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((bsz, self.seq_len), dtype=torch.int64)
        y = torch.empty((bsz, self.seq_len), dtype=torch.int64)
        for bi in range(bsz):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
            probs = remaining / remaining.sum()
            si = int(self.rng.choice(len(self.files), p=probs))
            start = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[start:start + self.seq_len + 1], dtype=np.int64))
            x[bi], y[bi] = window[:-1], window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_val_sliding(h, device, val_data, model, batch_seqs=32):
    model.eval()
    seq_len = h.eval_seq_len
    context = seq_len - h.eval_stride
    total = val_data.val_tokens.numel() - 1
    starts = [ws for ws in range(0, total, h.eval_stride) if ws + context < total]
    my_starts = starts[h.rank::h.world_size]
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_starts), batch_seqs):
            ws_batch = my_starts[bi:bi + batch_seqs]
            bsz = len(ws_batch)
            x_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(ws_batch):
                we = min(ws + seq_len, total)
                wlen = we - ws
                wlens.append(wlen)
                ch = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_b[i, :wlen] = ch[:-1]
                y_b[i, :wlen] = ch[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model.forward_logits(x_b)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y_b.reshape(-1), reduction='none').reshape(bsz, seq_len)
            for i, ws in enumerate(ws_batch):
                wlen = wlens[i]
                s = 0 if ws == 0 else context
                ls += nll[i, s:wlen].to(torch.float64).sum()
                tc += float(wlen - s)
                tgt = y_b[i, s:wlen]
                prev = x_b[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                bc += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        dist.all_reduce(bc, op=dist.ReduceOp.SUM)
    val_loss = (ls / tc).item()
    val_bpb = val_loss / math.log(2.0) * (tc.item() / bc.item())
    model.train()
    return val_loss, val_bpb


# ---------------------------------------------------------------------------
# Optimizer — all parameters are LoRA + embeddings, use AdamW for simplicity
# (Can switch to Muon for lora_B matrices if needed)
# ---------------------------------------------------------------------------
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def make_optimizers(h, model):
    embed_params = [model.tok_emb.weight]
    lora_params = [p for name, p in model.named_parameters()
                   if 'lora_' in name or 'rand_scale' in name]
    scalar_params = [p for name, p in model.named_parameters()
                     if 'attn_scale' in name or 'mlp_scale' in name or 'q_gain' in name]
    opt_embed = torch.optim.AdamW(
        [{'params': embed_params, 'lr': h.tied_embed_lr, 'base_lr': h.tied_embed_lr}],
        betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
    opt_lora = torch.optim.AdamW(
        [{'params': lora_params, 'lr': h.lora_lr, 'base_lr': h.lora_lr}],
        betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
    opt_scalar = torch.optim.AdamW(
        [{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}],
        betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=0.0, fused=True)
    return [opt_embed, opt_lora, opt_scalar]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(h, device, val_data):
    base_model = SeededGPT(h).to(device).bfloat16()
    for name, p in base_model.named_parameters():
        if p.ndim < 2 or 'q_gain' in name or 'attn_scale' in name or 'mlp_scale' in name:
            p.data = p.data.float()
    compiled = torch.compile(base_model, dynamic=False)
    if h.distributed:
        model = DDP(compiled, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled
    total_params = sum(p.numel() for p in base_model.parameters())
    lora_params = base_model._count_lora_params()
    emb_params = base_model.tok_emb.weight.numel()
    log(f"total_params={total_params} lora_params={lora_params} emb_params={emb_params}")
    log(f"param_budget: LoRA={lora_params} embed={emb_params} "
        f"(random_basis: {total_params - lora_params - emb_params} params never stored)")

    optimizers = make_optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_ms is not None:
        max_ms -= h.gptq_reserve_seconds * 1e3

    def lr_scale(elapsed_ms, step):
        frac = elapsed_ms / max_ms if max_ms else step / h.iterations
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    ema = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    training_ms = 0.0
    stop_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == h.iterations or (stop_step is not None and step >= stop_step)
        if last or (h.val_loss_every > 0 and step % h.val_loss_every == 0):
            torch.cuda.synchronize()
            training_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val_sliding(h, device, val_data, base_model)
            log(f"step:{step} val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last:
            break
        elapsed = training_ms + 1e3 * (time.perf_counter() - t0)
        scale = lr_scale(elapsed, step)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for _ in range(h.grad_accum_steps):
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            if h.distributed:
                model.require_backward_grad_sync = (_ == h.grad_accum_steps - 1)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema[name].mul_(h.ema_decay).add_(t.detach().float(), alpha=1.0 - h.ema_decay)
        step += 1
        approx_ms = training_ms + 1e3 * (time.perf_counter() - t0)
        if h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0):
            tok_per_sec = step * h.train_batch_tokens / (approx_ms / 1e3)
            log(f"{step}/{h.iterations} train_loss:{train_loss.item():.4f} "
                f"time:{approx_ms / 60000:.1f}m tok/s:{tok_per_sec:.0f}")
        if stop_step is None and max_ms is not None and approx_ms >= max_ms:
            stop_step = step

    avg_state = {n: t.to(dtype=base_model.state_dict()[n].dtype) for n, t in ema.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model


def main():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    h = Hyperparameters()
    set_log_h(h)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        log(f"=== Seeded Random Basis + LoRA ===")
        log(f"lora_rank: attn={h.lora_rank_attn} mlp={h.lora_rank_mlp}")
        log(f"use_fastfood={h.use_fastfood} learn_random_scale={h.learn_random_scale}")
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    train_model(h, device, val_data)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
