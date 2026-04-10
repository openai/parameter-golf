from __future__ import annotations
import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: Flash Attention not available, using standard attention")

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
    lawa_k = int(os.environ.get("LAWA_K", 10))
    lawa_freq = int(os.environ.get("LAWA_FREQ", 100))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1536))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_lr_warmup_iters = int(os.environ.get("TTT_LR_WARMUP_ITERS", 100))
    ttt_lr_cosine_iters = int(os.environ.get("TTT_LR_COSINE_ITERS", 500))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    late_qat_enabled = bool(int(os.environ.get("LATE_QAT_ENABLED", "1")))
    target_mb = float(os.environ.get("TARGET_MB", 15.9))
    activation_type = os.environ.get("ACTIVATION", "leaky_relu_squared")
    leaky_relu_slope = float(os.environ.get("LEAKY_RELU_SLOPE", 0.5))
    compression_level = int(os.environ.get("COMPRESSION_LEVEL", 9))
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "0")))
    gptq_calibration_samples = int(os.environ.get("GPTQ_CALIBRATION_SAMPLES", 64))
    gptq_calibration_seq_len = int(os.environ.get("GPTQ_CALIBRATION_SEQ_LEN", 2048))
    gptq_temperature = float(os.environ.get("GPTQ_TEMPERATURE", 0.8))
    gptq_full_hessian = bool(int(os.environ.get("GPTQ_FULL_HESSIAN", "0")))
    smear_enabled = bool(int(os.environ.get("SMEAR_ENABLED", "1")))
    selective_pruning = bool(int(os.environ.get("SELECTIVE_PRUNING", "0")))

args = Hyperparameters()

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], 
                                     stderr=subprocess.DEVNULL).decode().strip()
    except:
        return "unknown"

def get_git_diff():
    try:
        return subprocess.check_output(["git", "diff", "--shortstat"], 
                                     stderr=subprocess.DEVNULL).decode().strip()
    except:
        return "unknown"

git_hash = get_git_hash()
git_diff = get_git_diff()

# ---------------------------------------------------------------------------
# Token loading utilities
# ---------------------------------------------------------------------------
class TokenStream:
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        self.current_file = None
        self.current_tokens = None
        self.pos = 0
        
    def _load_file(self, filepath: str):
        self.current_tokens = np.fromfile(filepath, dtype=np.uint16)
        self.pos = 0
        
    def take(self, n: int) -> Tensor:
        result = []
        remaining = n
        
        while remaining > 0:
            if self.current_file is None or self.pos >= len(self.current_tokens):
                if not self.files:
                    raise RuntimeError("Ran out of data")
                self.current_file = self.files.pop(0)
                self._load_file(self.current_file)
                
            available = len(self.current_tokens) - self.pos
            take = min(remaining, available)
            
            result.append(torch.from_numpy(self.current_tokens[self.pos:self.pos + take]))
            self.pos += take
            remaining -= take
            
        return torch.cat(result)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        
        # Use pin_memory for GPU acceleration, standard to() for CPU
        if self.device.type == "cuda":
            local = chunk[start:start + per_rank_span].pin_memory().to(self.device, non_blocking=True).to(torch.int64)
        else:
            local = chunk[start:start + per_rank_span].to(self.device).to(torch.int64)
            
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
        if eps is None:
            self.weight = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("weight", torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        if self.eps is None:
            return F.rms_norm(x, (x.size(-1),), self.weight)
        else:
            return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos[(*x.shape[:-2], None, None)]
    sin = sin[(*x.shape[:-2], None, None)]
    return x * cos + torch.cat([-x[..., x.size(-1)//2:], x[..., :x.size(-1)//2]], dim=-1) * sin

class Rotary(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0, rope_dims: int = 64):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.rope_dims = rope_dims
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
        
    def forward(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        cos = self.cos[:seqlen].to(device=device, dtype=dtype)
        sin = self.sin[:seqlen].to(device=device, dtype=dtype)
        return cos, sin

class BigramHash(nn.Module):
    def __init__(self, vocab_size: int, bigram_vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.bigram_vocab_size = bigram_vocab_size
        self.embed_dim = embed_dim
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.bigram_embed = nn.Embedding(bigram_vocab_size, embed_dim)
        self.hash = nn.Linear(vocab_size, bigram_vocab_size, bias=False)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len)
        token_emb = self.token_embed(x)
        
        # Create bigram indices
        shifted = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        bigram_indices = self.hash(shifted).argmax(dim=-1)
        bigram_emb = self.bigram_embed(bigram_indices)
        
        return token_emb + bigram_emb

class XSA(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        
        # Use Flash Attention if available, otherwise standard attention
        if FLASH_ATTN_AVAILABLE:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            # Standard attention for CPU compatibility
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)
        
        y = y.contiguous().view(B, N, C)
        return self.out_proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "leaky_relu_squared", leaky_slope: float = 0.5):
        super().__init__()
        self.activation = activation
        self.leaky_slope = leaky_slope
        
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        if self.activation == "leaky_relu_squared":
            gate = F.leaky_relu(gate, negative_slope=self.leaky_slope).square()
            up = F.leaky_relu(up, negative_slope=self.leaky_slope).square()
        elif self.activation == "swiglu":
            gate = F.silu(gate)
        else:
            gate = F.relu(gate)
            
        return self.down_proj(gate * up)

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm()
        
    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(x))
        return self.norm(x * gate + x * (1 - gate))

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, qk_gain: float = 1.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.qk_gain = qk_gain
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.rotary = Rotary(self.head_dim, rope_dims=args.rope_dims)
        
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(N, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.qk_gain
        
        # Handle GQA
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # Use Flash Attention if available, otherwise standard attention
        if FLASH_ATTN_AVAILABLE:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            # Standard attention for CPU compatibility
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)
        
        y = y.contiguous().view(B, N, C)
        return self.out_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, dim: int, num_heads: int, num_kv_heads: int, 
                 mlp_mult: float = 3.0, use_xsa: bool = False, use_smear: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_xsa = use_xsa
        self.use_smear = use_smear
        
        self.attn_norm = RMSNorm()
        self.attn = Attention(dim, num_heads, num_kv_heads)
        
        self.mlp_norm = RMSNorm()
        self.mlp = MLP(dim, int(dim * mlp_mult), args.activation_type, args.leaky_relu_slope)
        
        if use_xsa:
            self.xsa = XSA(dim, num_heads)
        if use_smear:
            self.smear_gate = SmearGate(dim)
            
        # Layer scale
        if args.ln_scale:
            self.attn_scale = nn.Parameter(1.0 / math.sqrt(layer_idx + 1))
            self.mlp_scale = nn.Parameter(1.0 / math.sqrt(layer_idx + 1))
        else:
            self.register_buffer("attn_scale", torch.tensor(1.0))
            self.register_buffer("mlp_scale", torch.tensor(1.0))
    
    def forward(self, x: Tensor) -> Tensor:
        # Attention
        h = self.attn_norm(x)
        h = self.attn(h)
        x = x + self.attn_scale * h
        
        # XSA (if enabled)
        if self.use_xsa:
            h = self.xsa(x)
            x = x + h
        
        # MLP
        h = self.mlp_norm(x)
        h = self.mlp(h)
        x = x + self.mlp_scale * h
        
        # Smear gate (if enabled)
        if self.use_smear:
            x = self.smear_gate(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_layers = args.num_layers
        self.model_dim = args.model_dim
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        
        # Embeddings
        if args.bigram_vocab_size > 0:
            self.embed = BigramHash(args.vocab_size, args.bigram_vocab_size, args.model_dim)
        else:
            self.embed = nn.Embedding(args.vocab_size, args.model_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                layer_idx=i,
                dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult,
                use_xsa=(i >= args.num_layers - args.xsa_last_n),
                use_smear=args.smear_enabled
            )
            for i in range(args.num_layers)
        ])
        
        # Final norm and output
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(args.model_dim, args.vocab_size, bias=False)
        
        # Tie embeddings
        if args.tie_embeddings:
            self.lm_head.weight = self.embed.token_embed.weight if args.bigram_vocab_size > 0 else self.embed.weight
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if module is self.embed.token_embed if hasattr(self, 'embed') and hasattr(self.embed, 'token_embed') else (module is self.embed if hasattr(self, 'embed') else False):
                torch.nn.init.normal_(module.weight, mean=0.0, std=args.tied_embed_init_std)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if args.logit_softcap > 0:
            logits = torch.tanh(logits / args.logit_softcap) * args.logit_softcap
        
        return logits

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def get_optimizer(model: nn.Module):
    matrix_params = []
    scalar_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) > 1:
                matrix_params.append(param)
            else:
                scalar_params.append(param)
    
    optimizers = []
    
    if matrix_params:
        matrix_optimizer = torch.optim.AdamW(
            matrix_params,
            lr=args.matrix_lr,
            betas=(args.beta1, args.muon_beta2),
            weight_decay=args.muon_wd,
            eps=args.adam_eps
        )
        optimizers.append(matrix_optimizer)
    
    if scalar_params:
        scalar_optimizer = torch.optim.AdamW(
            scalar_params,
            lr=args.scalar_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.adam_wd,
            eps=args.adam_eps
        )
        optimizers.append(scalar_optimizer)
    
    return optimizers

def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

def evaluate_model(model: nn.Module, val_loader: DistributedTokenLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        x, y = val_loader.next_batch(args.val_batch_size, args.eval_seq_len, 1)
        logits = model(x)
        loss = compute_loss(logits, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    
    return total_loss / total_tokens

# ---------------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------------
def quantize_model(model: nn.Module, calibration_data: Tensor | None = None) -> nn.Module:
    """Simple int6 quantization"""
    model.eval()
    quantized_model = copy.deepcopy(model)
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            scale = weight.abs().max() / 63.0
            quantized_weight = torch.round(weight / scale).clamp(-63, 63)
            module.weight.data = quantized_weight * scale
    
    return quantized_model

def compress_model(model: nn.Module) -> bytes:
    """Compress model using LZMA"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    
    if _COMPRESSOR == "zstd":
        compressor = zstandard.ZstdCompressor(level=args.compression_level)
        return compressor.compress(model_bytes)
    else:
        return lzma.compress(model_bytes, preset=args.compression_level)

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    # Set up distributed training
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Device setup for RunPod
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if torch.cuda.device_count() > 0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"Using CUDA device: {device}")
    else:
        print("Warning: CUDA not available, using CPU")
        device = torch.device("cpu")
    
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    
    master_process = rank == 0
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create model
    model = Transformer().to(device)
    
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Create data loaders
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    val_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)
    
    # Create optimizers
    optimizers = get_optimizer(model)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for step in range(args.iterations):
        # Check time limit
        if time.time() - start_time > args.max_wallclock_seconds:
            break
        
        # Get batch
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, 1)
        
        # Forward pass
        logits = model(x)
        loss = compute_loss(logits, y)
        
        # Backward pass
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        
        # Optimizer step
        for optimizer in optimizers:
            optimizer.step()
        
        # Logging
        if master_process and step % args.train_log_every == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}: loss={loss.item():.4f}, time={elapsed:.1f}s")
    
    # Final evaluation
    if master_process:
        model.eval()
        val_loss = evaluate_model(model, val_loader, device)
        val_bpb = val_loss / math.log(2)
        
        # Quantize and compress
        quantized_model = quantize_model(model)
        compressed_size = len(compress_model(quantized_model))
        
        print(f"Final val_loss: {val_loss:.4f}")
        print(f"Final val_bpb: {val_bpb:.4f}")
        print(f"Compressed size: {compressed_size} bytes ({compressed_size / 1e6:.2f} MB)")
        
        # Save submission info
        submission_info = {
            "author": "Jtss-ux",
            "github_id": "Jtss-ux",
            "val_bpb": val_bpb,
            "compressed_size_bytes": compressed_size,
            "git_hash": git_hash,
            "git_diff": git_diff,
            "hyperparameters": vars(args)
        }
        
        import json
        with open("submission.json", "w") as f:
            json.dump(submission_info, f, indent=2)

if __name__ == "__main__":
    main()
