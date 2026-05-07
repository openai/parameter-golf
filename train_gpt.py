"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.


------------------------------
MDLM Enhanced Architecture
Fusiona el pipeline de entrenamiento DDP/Muon/INT8 del Código 1 
con la arquitectura avanzada de bloques (Residual Mix, U-Net Skips, QK Norm) del Código 2.


---------------------- based on #1855   by codemath3000
the transformer of MDLM was improved based on the  transformer created by codemath3000
transformer with U-Net skips, parallel residuals, partial RoPE, Polar-Express Newton-Schulz Muon, LQER asymmetric int4 rank-4 quant correction, sparse attention head-output gate, SmearGate with cross-document leak fix on BOS positions (audit response), fused LeakyReLU-square MLP, fused softcapped CE Triton kernel, GPTQ int6 + int7 embed + per-row int8 attn-gate


"""



from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 354))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    diffusion_steps = int(os.environ.get("DIFFUSION_STEPS", 100))
    activation = os.environ.get("ACTIVATION", "leaky_relu")
    
    # NUEVOS: Hiperparámetros de la arquitectura avanzada (Extraídos del Código 2)
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0)) # Más alto que el 1.25 anterior
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))     # Escala LN por capa
    skip_connections_enabled = bool(int(os.environ.get("SKIP_CONNECTIONS", "1"))) # U-Net skips

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.026)) # Ajustado al estilo Code 2
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.97))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.90))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

HP = Hyperparameters()

# NUEVO: Patron de control para que Muon no toque los tensores especiales del Code 2
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS", 
        "q_gain,resid_mix,skip_weights" # Aquí van los escalares aprendidos
    ).split(",") if pattern
)

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

# -----------------------------
# UTILITIES
# -----------------------------
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def get_rank(): return int(os.environ.get("RANK", 0))
def log0(msg: str):
    if get_rank() == 0: print(msg, flush=True)

def lr_mul(step: int, elapsed_ms: float) -> float:
    warmup_frac = min(step / max(HP.warmup_steps, 1), 1.0)
    if elapsed_ms <= 0: return warmup_frac
    if HP.warmdown_iters > 0 and step >= HP.iterations - HP.warmdown_iters:
        warmdown_frac = (HP.iterations - step) / HP.warmdown_iters
        return warmdown_frac * math.cos(math.pi * (1.0 - warmdown_frac) / 2.0)
    return warmup_frac

def zero_grad_all(optimizers):
    for opt in optimizers: opt.zero_grad(set_to_none=True)

# -----------------------------
# DATA LOADING (Código 1 - Estable y simple)
# -----------------------------
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520: raise ValueError(f"Bad header {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files for {pattern}")
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos : self.pos + k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> Tensor:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        return local[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True)


# -----------------------------
# OPTIMIZER MUON (Con Polar Express del Código 2)
# -----------------------------
# Coeficientes optimizados de Newton-Schulz (Polar Express)
_PE_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    was_2d = G.ndim == 2
    if was_2d: G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    coeffs = _PE_COEFFS[:steps] if steps <= len(_PE_COEFFS) else _PE_COEFFS
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed: X = X.mT
    if was_2d: X = X.squeeze(0)
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype); p.add_(g, alpha=-lr); curr += p.numel()
        return loss


# -----------------------------
# ADVANCED TRANSFORMER MODULES (Fusión Código 1 + Código 2)
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None): super().__init__(); self.eps = eps
    def forward(self, x: Tensor) -> Tensor: return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

def get_activation(act_name):
    if act_name == "silu": return nn.SiLU()
    if act_name == "xielu": return nn.SELU()
    return nn.LeakyReLU(negative_slope=0.1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq); emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)

class AdvancedAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, qk_gain, layer_idx):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.wq = CastedLinear(dim, num_heads * self.head_dim, bias=False)
        self.wk = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = CastedLinear(num_heads * self.head_dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base=HP.rope_base)
        
        # EXTRACCIÓN CODE 2: Q Gain aprendible por cabeza y QK RMSNorm
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain, dtype=torch.float32))
        
        # EXTRACCIÓN CODE 2: Inicialización Ortogonal (Q, K) y Cero (Out)
        with torch.no_grad():
            nn.init.orthogonal_(self.wq.weight, gain=qk_gain)
            nn.init.orthogonal_(self.wk.weight, gain=qk_gain)
            nn.init.zeros_(self.wo.weight)

    def forward(self, x):
        b, t, c = x.shape
        q = self.wq(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # EXTRACCIÓN CODE 2: Normalizar Q y K antes de la atención
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        
        cos, sin = self.rope(t); q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # EXTRACCIÓN CODE 2: Aplicar Q Gain
        #q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        q = q * self.q_gain.to(dtype=q.dtype).view(1, self.num_heads, 1, 1)
        #Al usar .view(1, self.num_heads, 1, 1), forzamos al tensor de 8 elementos a tener la forma [1, 8, 1, 1]. 
        #Al multiplicarlo por q ([64, 8, 1024, 44]), PyTorch ya puede hacer el broadcast correctamente:
        # problema era El compilador simbólico de torch.compile es estricto y se ha dado cuenta de que con tus hiperparámetros actuales (model_dim=354 y num_heads=8), la dimensión de la cabeza es 354 // 8 = 44.25, que PyTorch redondea a 44. 
#Al intentar multiplicar el tensor q de forma [64, 8, 1024, 44] por el q_gain de forma [1, 1, 8, 1], falla porque en la dimensión 2 intenta transmitir un 8 a un 1024, lo cual es matemáticamente imposible.
# si model_dim=512 (y 512 // 8 = 64 perfecto). , pero se usa 354   354 // 8 = 44.25, que PyTorch redondea a 44. 

        
        
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(b, self.num_heads, t, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(b, self.num_heads, t, self.head_dim)
        
        # FULL BIDIRECTIONAL (MDLM)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.wo(out.transpose(1, 2).contiguous().view(b, t, -1))

class AdvancedMLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.fc1 = CastedLinear(dim, mult * dim, bias=False)
        self.act = get_activation(HP.activation)
        self.fc2 = CastedLinear(mult * dim, dim, bias=False)
        # EXTRACCIÓN CODE 2: Inicialización
        with torch.no_grad():
            nn.init.orthogonal_(self.fc1.weight, gain=1.0)
            nn.init.zeros_(self.fc2.weight)
            
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class AdvancedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = AdvancedAttention(dim, num_heads, num_kv_heads, HP.qk_gain_init, layer_idx)
        self.mlp = AdvancedMLP(dim, HP.mlp_mult)
        
        # EXTRACCIÓN CODE 2: Mezcla residual entre estado actual y estado original (x0)
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        # EXTRACCIÓN CODE 2: Escalar la Normalización en capas tempranas
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if HP.ln_scale else 1.0

    def forward(self, x, x0, t_emb):
        mix = self.resid_mix.to(dtype=x.dtype)
        # Mezcla x actual con el embedding inicial puro
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        
        # Inyección de tiempo MDLM + LN Scale
        attn_out = self.attn(self.attn_norm(x_in + t_emb.unsqueeze(1)) * self.ln_scale_factor)
        x_out = x_in + attn_out
        
        # Inyección de tiempo MDLM + LN Scale
        mlp_out = self.mlp(self.mlp_norm(x_out + t_emb.unsqueeze(1)) * self.ln_scale_factor)
        x_out = x_out + mlp_out
        
        return x_out

# -----------------------------
# ADVANCED MDLM CORE (Con U-Net Skips)
# -----------------------------
class AdvancedMDLM(nn.Module):
    def __init__(self, hp: Hyperparameters):
        super().__init__(); self.hp = hp; self.mask_token_id = hp.vocab_size - 1
        self.tok_emb = nn.Embedding(hp.vocab_size, hp.model_dim)
        self.mask_emb = nn.Embedding(1, hp.model_dim)
        self.time_emb = TimeEmbedding(hp.model_dim)
        
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(hp.model_dim, hp.num_heads, hp.num_kv_heads, i) 
            for i in range(hp.num_layers)
        ])
        
        self.final_norm = RMSNorm()
        if hp.tie_embeddings:
            self.tok_emb.weight.data.normal_(mean=0.0, std=hp.tied_embed_init_std); self.lm_head = None 
        else: self.lm_head = CastedLinear(hp.model_dim, hp.vocab_size, bias=False)
        
        # EXTRACCIÓN CODE 2: Arquitectura U-Net (Encoder-Decoder Skips)
        self.num_encoder_layers = hp.num_layers // 2
        self.num_decoder_layers = hp.num_layers - self.num_encoder_layers
        self.num_skip_weights = self.num_encoder_layers
        if hp.skip_connections_enabled and self.num_skip_weights > 0:
            self.skip_weights = nn.Parameter(
                torch.ones(self.num_skip_weights, hp.model_dim, dtype=torch.float32)
            )
        else:
            self.skip_weights = None

    def forward(self, x_0):
        B, T = x_0.shape
        t = torch.randint(1, self.hp.diffusion_steps + 1, (B,), device=x_0.device)
        mask_ratio = t.float() / self.hp.diffusion_steps
        mask_bool = torch.rand_like(x_0.float()) < mask_ratio.unsqueeze(1)
        x_t = torch.where(mask_bool, self.mask_token_id, x_0)
        is_mask = (x_t == self.mask_token_id)
        embeddings = torch.where(is_mask.unsqueeze(-1), self.mask_emb(torch.zeros_like(x_t)), self.tok_emb(x_0))
        t_emb = self.time_emb(t.float() / self.hp.diffusion_steps)
        
        h = embeddings
        x0 = F.rms_norm(h, (h.size(-1),)) # Normalización inicial estilo Code 2
        
        skips = []
        skip_idx = 0
        
        # Fase 1: Encoder (Guardar estados para Skips)
        for i in range(self.num_encoder_layers):
            h = self.layers[i](h, x0, t_emb)
            skips.append(h)
            
        # Fase 2: Decoder (Inyectar Skips)
        for i in range(self.num_encoder_layers, self.hp.num_layers):
            if self.skip_weights is not None and skip_idx < self.num_skip_weights:
                # Conexión de salto U-Net
                h = h + self.skip_weights[skip_idx].to(dtype=h.dtype)[None, None, :] * skips[skip_idx]
                skip_idx += 1
            h = self.layers[i](h, x0, t_emb)
            
        h = self.final_norm(h)
        if self.hp.tie_embeddings: logits = F.linear(h, self.tok_emb.weight)
        else: logits = self.lm_head(h)
        
        logits = self.hp.logit_softcap * torch.tanh(logits / self.hp.logit_softcap)
        loss_matrix = F.cross_entropy(logits.reshape(-1, self.hp.vocab_size), x_0.reshape(-1), reduction='none').reshape(x_0.shape)
        return (loss_matrix * mask_bool.float()).sum() / (mask_bool.sum() + 1e-6)

class TimeEmbedding(nn.Module):
    def __init__(self, dim): 
        super().__init__(); self.dim = dim
        activacion = get_activation(HP.activation)
        self.mlp = nn.Sequential(activacion, CastedLinear(dim, dim * 4), activacion, CastedLinear(dim * 4, dim))
    def forward(self, t):
        half_dim = self.dim // 2; emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]; emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


# -----------------------------
# EVAL & QUANTIZATION (Sin cambios estructurales)
# -----------------------------
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size()); table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np, has_leading_space_np, is_boundary_token_np = np.zeros((table_size,), dtype=np.int16), np.zeros((table_size,), dtype=np.bool_), np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"): has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, device=device), torch.tensor(has_leading_space_np, device=device), torch.tensor(is_boundary_token_np, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: return None
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, luts):
    if val_tokens is None or luts[0] is None: return 0.0, 0.0
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start, seq_end = (total_seqs * rank) // world_size, (total_seqs * (rank + 1)) // world_size
    v_ls, v_tc, v_bc = torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            rs, re = bss * args.train_seq_len, bse * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x).detach()  
            btc = float(x.numel())
            v_ls += bl.to(torch.float64) * btc; v_tc += btc
            tgt_ids = x.reshape(-1); token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            v_bc += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(v_ls, op=dist.ReduceOp.SUM); dist.all_reduce(v_tc, op=dist.ReduceOp.SUM); dist.all_reduce(v_bc, op=dist.ReduceOp.SUM)
    vl = v_ls / v_tc; bpt = vl.item() / math.log(2.0); tpb = v_tc.item() / v_bc.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        return torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous(), scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    return torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous(), scale

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough, qmeta = {}, {}, {}, {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = t.to(dtype=torch.float16) if t.is_floating_point() else t; continue
        q, s = quantize_float_tensor(t); quantized[name], scales[name], dtypes[name] = q, s, str(t.dtype).removeprefix("torch.")
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
    return {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough, "qmeta": qmeta}

def dequantize_state_dict_int8(obj):
    out, qmeta = {}, obj.get("qmeta", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name]); s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32); out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else: out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items(): out[name] = t.detach().to("cpu").contiguous()
    return out

# -----------------------------
# MAIN
# -----------------------------
def main():
    local_rank = setup_ddp()
    rank = get_rank()
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    distributed = world_size > 1
    torch.set_float32_matmul_precision('high')
    max_wallclock_ms = HP.max_wallclock_seconds * 1000.0
    code_bytes = len(Path(__file__).read_text().encode("utf-8"))
    grad_accum_steps = 1

    # 1. MODELO AVANZADO
    base_model = AdvancedMDLM(HP).to(device).bfloat16()
    
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # torch.compile soporta fullgraph gracias a que reemplazamos list.pop() por índices estáticos
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # 2. OPTIMIZADORES
    matrix_params, scalar_params = [], []
    for name, p in base_model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(p)
        else:
            scalar_params.append(p)
            
    params_adam_embed = [p for n, p in base_model.named_parameters() if "tok_emb" in n or "mask_emb" in n]
    params_adam_head = [p for n, p in base_model.named_parameters() if "lm_head" in n]
    
    ids_embed = {id(p) for p in params_adam_embed}
    ids_head = {id(p) for p in params_adam_head}
    scalar_params = [p for p in scalar_params if id(p) not in ids_embed and id(p) not in ids_head]

    token_lr = HP.tied_embed_lr if HP.tie_embeddings else HP.embed_lr
    
    optimizer_tok = torch.optim.AdamW([{"params": params_adam_embed, "lr": token_lr, "base_lr": token_lr}], betas=(HP.beta1, HP.beta2), eps=HP.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=HP.matrix_lr, momentum=HP.muon_momentum, backend_steps=HP.muon_backend_steps)
    for group in optimizer_muon.param_groups: group["base_lr"] = HP.matrix_lr
    optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": HP.scalar_lr, "base_lr": HP.scalar_lr}], betas=(HP.beta1, HP.beta2), eps=HP.adam_eps, fused=True)
    
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW([{"params": params_adam_head, "lr": HP.head_lr, "base_lr": HP.head_lr}], betas=(HP.beta1, HP.beta2), eps=HP.adam_eps, fused=True)
        optimizers.insert(1, optimizer_head)

    train_loader = DistributedTokenLoader(HP.train_files, rank, world_size, device)
    sp = spm.SentencePieceProcessor(); sp.load(HP.tokenizer_path)
    val_tokens = load_validation_tokens(HP.val_files, HP.train_seq_len)
    luts = build_sentencepiece_luts(sp, HP.vocab_size, device) if val_tokens is not None else (None, None, None)

    if HP.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(HP.warmup_steps):
            zero_grad_all(optimizers)
            x = train_loader.next_batch(HP.train_batch_tokens, HP.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x)
            loss.backward()
            for opt in optimizers: opt.step()
            zero_grad_all(optimizers)
            
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states): opt.load_state_dict(state)
        zero_grad_all(optimizers)
        train_loader = DistributedTokenLoader(HP.train_files, rank, world_size, device)
        log0("Warmup compile finished. Timer starting NOW.")

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == HP.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (HP.val_loss_every > 0 and step % HP.val_loss_every == 0)
        
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(HP, model, rank, world_size, device, grad_accum_steps, val_tokens, luts)
            log0(f"step:{step}/{HP.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < HP.iterations: log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all(optimizers)
        train_loss = torch.zeros((), device=device)
        
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x = train_loader.next_batch(HP.train_batch_tokens, HP.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x)
            train_loss += loss.detach()
            loss.backward()
        train_loss /= grad_accum_steps

        frac = min(step / HP.muon_momentum_warmup_steps, 1.0) if HP.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * HP.muon_momentum_warmup_start + frac * HP.muon_momentum
        for group in optimizer_muon.param_groups: group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale

        if HP.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), HP.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all(optimizers)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = HP.train_log_every > 0 and (step <= 10 or step % HP.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(f"step:{step}/{HP.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    quant_obj  = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO(); torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    
    if get_rank() == 0:
        with open("submission.zlib", "wb") as f: f.write(quant_blob)
        log0(f"FINAL SUBMISSION SIZE: {os.path.getsize('submission.zlib') + code_bytes} bytes")

    if distributed: dist.barrier()
    with open("submission.zlib", "rb") as f: quant_blob_disk = f.read()
    base_model.load_state_dict(dequantize_state_dict_int8(torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")), strict=True)
    
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(HP, model, rank, world_size, device, grad_accum_steps, val_tokens, luts)
    torch.cuda.synchronize()
    log0(f"final_int8_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()


