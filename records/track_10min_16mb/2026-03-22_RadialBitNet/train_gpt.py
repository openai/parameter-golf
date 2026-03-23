import os
import sys
import time
import math
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import sentencepiece as spm

# -----------------------------
# HYPERPARAMETERS (16MB TITAN - RADIAL BITNET)
# -----------------------------
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    seed = int(os.environ.get("SEED", 1337))

    # We scale down to fit exactly ~16MB compressed.
    # A standard 50M parameter model in FP16 is 100MB. 
    # With BitNet 1.58b (ternary weights), zstd shrinks this dramatically.
    vocab_size = 1024
    num_layers = 12
    num_kv_heads = 2
    model_dim = 384
    num_heads = 6
    mlp_mult = 3  # Wide MLPs offset BitNet capacity reduction

    train_seq_len = 1024
    val_batch_size = 524_288
    val_loss_every = 1000
    iterations = 20000

# -----------------------------
# 1. OPTIMIZER: FRACTAL RESONANT OPTIMIZATION (FRO)
# -----------------------------
class FRO(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, 
                 scales=[0.1, 0.01, 0.001], alpha=0.1, gamma=0.5):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
                        scales=scales, alpha=alpha, gamma=gamma)
        super(FRO, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    if p.dim() == 2:
                        state['exp_avg_sq'] = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    K = len(group['scales'])
                    state['mu'] = [torch.zeros(1, device=p.device) for _ in range(K)]
                    state['sigma'] = [torch.zeros(1, device=p.device) for _ in range(K)]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                mu, sigma = state['mu'], state['sigma']
                beta1, beta2 = group['beta1'], group['beta2']
                eps = group['eps']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state['step']

                # Distributed Resonance Sync
                local_dot = torch.dot(grad.flatten(), exp_avg.flatten())
                local_gnorm_sq = grad.norm().pow(2)
                local_mnorm_sq = exp_avg.norm().pow(2)
                
                if dist.is_initialized():
                    metrics = torch.stack([local_dot, local_gnorm_sq, local_mnorm_sq])
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    global_dot, global_gnorm_sq, global_mnorm_sq = metrics[0], metrics[1], metrics[2]
                else:
                    global_dot, global_gnorm_sq, global_mnorm_sq = local_dot, local_gnorm_sq, local_mnorm_sq

                rho_t = global_dot / (torch.sqrt(global_gnorm_sq * global_mnorm_sq) + eps)
                rho_t = rho_t.clamp(-1, 1)

                for k, lam in enumerate(group['scales']):
                    mu[k].mul_(1 - lam).add_(rho_t, alpha=lam)
                    sigma[k].mul_(1 - lam).add_(rho_t**2, alpha=lam)

                log_sum = 0
                K = len(group['scales'])
                for k in range(K):
                    rk = (mu[k]**2) / (sigma[k] + eps)
                    log_sum += torch.log(rk + eps)
                Rt = torch.exp(log_sum / K).clamp(0, 1)

                if p.dim() == 2:
                    grad_sq = grad.pow(2).mean(dim=1, keepdim=True)
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)
                else:
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                
                adaptive_factor = group['alpha'] + (1 - group['alpha']) * group['gamma'] * Rt
                step_size = float(group['lr'] * adaptive_factor / bias_correction1)
                denom = exp_avg_sq.sqrt().add(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# -----------------------------
# 1.5 DISTRIBUTED SETUP (8xH100 READY)
# -----------------------------
def setup_distributed():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device, rank, local_rank, world_size

# -----------------------------
# 2. ARCHITECTURE: RADIAL BITNET
# -----------------------------
class RadialEncoding(nn.Module):
    def __init__(self, n_bits=8, alpha=0.25):
        super().__init__()
        phi = (1 + 5**0.5) / 2
        angles = torch.linspace(0, 2 * math.pi, n_bits + 1)[:n_bits]
        radii = torch.pow(phi, torch.arange(n_bits).float()) * alpha
        self.register_buffer('angles', angles)
        self.register_buffer('radii', radii)
        self.register_buffer('bit_indices', torch.arange(n_bits))
    def forward(self, x):
        bits = (x.unsqueeze(-1).long() >> self.bit_indices) & 1
        bits = bits.to(self.radii.dtype)
        re = torch.sum(bits * self.radii * torch.cos(self.angles), dim=-1)
        im = torch.sum(bits * self.radii * torch.sin(self.angles), dim=-1)
        return torch.stack([re, im, torch.sqrt(re**2 + im**2), torch.atan2(im, re)], dim=-1)

def weight_quant(w):
    scale = w.abs().mean().clamp(min=1e-5)
    return (torch.sign(w) * scale).detach() + (w - w.detach())

class BitLinear(nn.Linear):
    def forward(self, x):
        if x.dtype != self.weight.dtype: x = x.to(self.weight.dtype)
        # Weight-Only BitNet (W1.58b / A16b) to save VRAM and maintain parameter compression limit
        return F.linear(x, weight_quant(self.weight), self.bias)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(torch.mean(x.float()**2, dim=-1, keepdim=True) + self.eps)).to(x.dtype)

class BitAttention(nn.Module):
    def __init__(self, d_model, nhead, n_kv_heads):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.n_kv_heads = n_kv_heads
        
        # We use BitLinear strictly everywhere for maximum compression
        self.q_proj = BitLinear(d_model, d_model, bias=False)
        self.k_proj = BitLinear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = BitLinear(d_model, d_model, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Broadcast KV to match Q heads for older PyTorch versions
        num_kv_groups = self.nhead // self.n_kv_heads
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.out_proj(out)

class BitTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, num_kv_heads, mlp_mult):
        super().__init__()
        self.attn = BitAttention(d_model, nhead, num_kv_heads)
        self.linear1 = BitLinear(d_model, d_model * mlp_mult, bias=False)
        self.linear2 = BitLinear(d_model * mlp_mult, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.activation = nn.GELU()
    def forward(self, src):
        h = self.norm1(src)
        h = self.attn(h)
        src = src + h
        h = self.norm2(src)
        h = self.linear2(self.activation(self.linear1(h)))
        return src + h

class ParameterGolfBitNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rad8 = RadialEncoding(8)
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        # Radial projection injected into the model dim
        self.rad_proj = nn.Linear(4, args.model_dim, bias=False)
        
        self.layers = nn.ModuleList([
            BitTransformerLayer(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult) 
            for _ in range(args.num_layers)
        ])
        self.final_norm = RMSNorm(args.model_dim)
        
        # Tie embeddings
        self.lm_head = nn.Linear(args.model_dim, args.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor = None):
        # Base token embedding
        x = self.tok_emb(input_ids)
        
        # Inject pure geometric signal based on token indices (Absolute Position Bypass)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        rad_feat = self.rad8(positions) 
        x = x + self.rad_proj(rad_feat)
        
        for layer in self.layers: 
            x = layer(x)
            
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.lm_head(x)
        
        if target_ids is not None:
            targets = target_ids.reshape(-1)
            loss = F.cross_entropy(logits.float(), targets, reduction="mean")
            return loss
        return logits

# -----------------------------
# 3. EVALUATION METRICS (TOKENIZER AGNOSTIC BPB)
#    (Mirroring OpenAI starter code)
# -----------------------------
def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith(" "):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_validation_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        print(f"Warning: No validation files found for {pattern}. Returning dummy data to avoid crash.")
        return torch.zeros(seq_len * 2 + 1, dtype=torch.int64)
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1].long()

def load_training_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        print(f"Warning: No training files found.")
        return torch.zeros(seq_len * 2 + 1, dtype=torch.int64)
    # ONLY load the first shard (100M tokens = ~800MB RAM) to completely avoid Kaggle Notebook CPU OOM!
    # A 10 minute training run will only consume ~80M tokens anyway.
    print(f"Loading single dataset shard to protect Kaggle RAM: {files[0]}")
    tokens = load_data_shard(files[0]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1].long()

def eval_val(args, model, device, val_tokens, base_bytes_lut, has_space_lut, boundary_lut, rank=0, world_size=1):
    model.eval()
    local_loss_sum = 0.0
    local_token_count = 0.0
    local_byte_count = 0.0
    
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    
    # Distributed Evaluation Sharding
    seqs_per_rank = total_seqs // world_size
    start_seq = rank * seqs_per_rank
    end_seq = (rank + 1) * seqs_per_rank if rank != world_size - 1 else total_seqs
    
    with torch.inference_mode():
        for i in range(start_seq, end_seq):
            raw_start = i * seq_len
            raw_end = (i + 1) * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device)
            x = local[:-1].unsqueeze(0)
            y = local[1:].unsqueeze(0)
            
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=autocast_dtype):
                batch_loss = model(x, y)
                if isinstance(batch_loss, torch.Tensor) and batch_loss.dim() > 0:
                    batch_loss = batch_loss.mean()
                batch_loss = batch_loss.to(torch.float64)
                
            batch_token_count = float(y.numel())
            local_loss_sum += batch_loss * batch_token_count
            local_token_count += batch_token_count
            
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            t_bytes = base_bytes_lut[tgt_ids].clone()
            t_bytes += (has_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(dtype=torch.int16)
            local_byte_count += t_bytes.to(torch.float64).sum()
    
    # Aggregate results across all ranks
    metrics = torch.tensor([local_loss_sum, local_token_count, local_byte_count], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    global_loss_sum, global_token_count, global_byte_count = metrics[0], metrics[1], metrics[2]
    
    val_loss = global_loss_sum / (global_token_count + 1e-10)
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = global_token_count / (global_byte_count + 1e-10)
    val_bpb = bits_per_token * tokens_per_byte
    
    model.train()
    return float(val_loss.item()), float(val_bpb.item())

# -----------------------------
# 4. EXPORT & SIZE VALIDATION
# -----------------------------
def export_and_check_size(model, filename="golf_model.zst"):
    import zlib
    # 1. State Dict
    state = model.state_dict()
    # 2. Int8 Quantization (Ternary weights -> Int8)
    q_state = {}
    for k, v in state.items():
        if v.is_floating_point():
            # For BitNet layers, weight is heavily concentrated near -scale/0/scale
            if 'weight' in k and 'proj' in k or 'linear' in k:
                # Store mostly as ternary via INT8 (-127, 0, 127 roughly)
                scale = v.abs().mean().clamp(min=1e-5)
                # Ensure we round to perfectly compressible integers
                q = torch.clamp(torch.round(v / scale * 127.0), -127, 127).to(torch.int8)
                q_state[k] = (q, scale.item())
            else:
                q_state[k] = v.to(torch.float16) # Store embeddings in FP16
        else:
            q_state[k] = v
    # 3. Serialize and Compress
    import pickle
    raw_bytes = pickle.dumps(q_state)
    compressed = zlib.compress(raw_bytes, level=9)
    
    # OpenAI Rule: artifact = code bytes + compressed model bytes <= 16,000,000 decimal bytes
    code_bytes = Path(__file__).read_bytes()
    total_bytes = len(code_bytes) + len(compressed)
    
    print(f"\n📦 Artifact Size Audit:")
    print(f"- Source Code: {len(code_bytes)} bytes")
    print(f"- Compressed Model: {len(compressed)} bytes")
    print(f"- Total Artifact: {total_bytes} bytes")
    
    if total_bytes <= 16000000:
        print("✅ QUALIFIED FOR PARAMETER GOLF! (<= 16,000,000 bytes)")
    else:
        print(f"❌ TOO LARGE! Exceeds 16MB limit by {total_bytes - 16000000} bytes.")

# -----------------------------
# TRAINING LOOP
# -----------------------------
def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    args = Hyperparameters()
    
    device, rank, local_rank, world_size = setup_distributed()
    if rank == 0:
        print(f"✨ Initializing Radial-BitNet for Parameter Golf (Constraint: 16MB)")
    
    model = ParameterGolfBitNet(args).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    if rank == 0:
        export_and_check_size(model)
    
    optimizer = FRO(model.parameters(), lr=1e-3, gamma=0.8) # Aggressive FRO for 10-min run
    
    try:
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        base_bytes, has_space, boundary = build_sentencepiece_luts(sp, args.vocab_size, device)
        val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    except Exception as e:
        print(f"\n⚠️ Mocking SentencePiece for local testing due to missing files: {e}")
        # Dummy LUTs
        base_bytes = torch.ones(args.vocab_size, dtype=torch.int16, device=device) * 4
        has_space = torch.zeros(args.vocab_size, dtype=torch.bool, device=device)
        boundary = torch.ones(args.vocab_size, dtype=torch.bool, device=device)
        val_tokens = torch.randint(0, args.vocab_size, (10000,), device=device)

    print("⏳ Loading training tokens into memory...")
    # Load training tokens with single-shard safeguard
    train_tokens = load_training_tokens(args.train_files, args.train_seq_len)

    import time
    start_time = time.time()
    max_time = 10 * 60 - 30 # 9.5 minutes wallclock limit
    
    batch_size = 4 # VRAM safe size for Deep Graph Accumulation
    print(f"\n🚀 Starting 10-Minute Rapid Convergence Cycle on real dataset...")
    
    step = 0
    while time.time() - start_time < max_time:
        chunk_size = batch_size * args.train_seq_len
        # Rank-aware sharding: each GPU starts at a unique offset or uses a unique jump
        total_available = max(1, (train_tokens.numel() - chunk_size - 1))
        offset_per_rank = total_available // world_size
        start_token = (rank * offset_per_rank + step * chunk_size * world_size) % total_available
        chunk = train_tokens[start_token : start_token + chunk_size + 1]
        
        # Fallback to random if dataset failed to load (Mocking)
        if chunk.numel() < chunk_size + 1:
            x = torch.randint(0, args.vocab_size, (batch_size, args.train_seq_len)).to(device)
            y = torch.randint(0, args.vocab_size, (batch_size, args.train_seq_len)).to(device)
        else:
            x = chunk[:-1].reshape(batch_size, args.train_seq_len).to(device, non_blocking=True)
            y = chunk[1:].reshape(batch_size, args.train_seq_len).to(device, non_blocking=True)
        
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=autocast_dtype):
            loss = model(x, y)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if step % 50 == 0:
            # Sync Fix: ALL ranks participate in eval_val to prevent desynchronization
            val_l, val_bpb = eval_val(args, model, device, val_tokens, base_bytes, has_space, boundary, rank, world_size)
            if rank == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:04d} | Time {elapsed:.0f}s | Train Loss: {loss.item():.4f} | Val BPB: {val_bpb:.4f} ⛳")
            
        step += 1
        
    # Final Distributed Validation
    val_l, val_bpb = eval_val(args, model, device, val_tokens, base_bytes, has_space, boundary, rank, world_size)
    if rank == 0:
        print("\n⏰ 10-Minute training time budget exhausted. Validating final model...")
        print(f"FINAL RESULT | Val BPB: {val_bpb:.4f} 🏆")

if __name__ == "__main__":
    main()
