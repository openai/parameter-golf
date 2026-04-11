#!/usr/bin/env python3
from __future__ import annotations
import glob, math, os, time, io, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Config:
    data_path = "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
    train_pattern = "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"
    val_pattern = "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"
    tokenizer_path = "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
    
    seed = 1337
    vocab_size = 1024
    num_layers = 11
    model_dim = 384
    num_heads = 6
    num_kv_heads = 2
    mlp_mult = 2.5
    tie_embeddings = True
    rope_base = 10000.0
    rope_fraction = 0.5
    logit_softcap = 30.0
    batch_size = 64
    seq_len = 1024
    
    iterations = 5000
    warmup_steps = 200
    val_every = 1000
    
    lr = 0.003
    lr_min = 0.0001
    grad_clip = 1.0
    ema_decay = 0.999
    log_every = 50
    val_stride = 512
    quant_bits = 6

args = Config()

def load_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or header[0] != 20240520:
        raise ValueError(f"Bad header {file}")
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens.astype(np.int64))

def build_luts(sp, device):
    sz = sp.vocab_size()
    base = np.zeros((sz,), np.int16)
    space = np.zeros((sz,), bool)
    boundary = np.ones((sz,), bool)
    for tid in range(sz):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        boundary[tid] = False
        if sp.is_byte(tid):
            base[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            space[tid] = True
            piece = piece[1:]
        base[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base, device=device),
            torch.tensor(space, device=device),
            torch.tensor(boundary, device=device))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, fraction=1.0):
        super().__init__()
        self.dim = dim
        self.fraction = fraction
        actual_dim = int(dim * fraction)
        inv_freq = 1.0 / (base ** (torch.arange(0, actual_dim, 2).float() / actual_dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

def apply_rotary(x, cos, sin, fraction):
    head_dim = x.size(-1)
    rot_dim = int(head_dim * fraction)
    if rot_dim == 0:
        return x
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = x_rot.view(*x_rot.shape[:-1], -1, 2)
    x1, x2 = x_rot[..., 0], x_rot[..., 1]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    rotated = rotated.view(*rotated.shape[:-2], -1)
    return torch.cat([rotated, x_pass], dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, rope_frac):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_frac = rope_frac
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, rope_base, rope_frac)
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        cos, sin = self.rotary(T, x.device)
        q = apply_rotary(q, cos, sin, self.rope_frac)
        k = apply_rotary(k, cos, sin, self.rope_frac)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(y)

class SwiGLU(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, rope_frac):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, rope_frac)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config.model_dim, config.num_heads, config.num_kv_heads, config.mlp_mult, config.rope_base, config.rope_fraction) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.model_dim)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.clone().detach() for k, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * p.data
    @torch.no_grad()
    def apply(self, model):
        for k, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[k])

def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + 1e-7
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                if p.ndim >= 2:
                    g = zeropower_via_newtonschulz5(g)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)

def quantize_model(model, bits=6):
    state = model.state_dict()
    quantized = {}
    scales = {}
    for name, param in state.items():
        if param.dim() == 2 and "weight" in name:
            max_val = param.abs().max(dim=1, keepdim=True)[0]
            scale = max_val / (2**(bits-1)-1)
            scale = scale.clamp_min(1e-8)
            q = torch.round(param.float() / scale).clamp(-2**(bits-1), 2**(bits-1)-1).to(torch.int8)
            quantized[name] = q
            scales[name] = scale.squeeze()
        else:
            quantized[name] = param.to(torch.float16)
    return quantized, scales

def evaluate(model, val_tokens, luts, config, max_tokens=None):
    model.eval()
    base, space, boundary = luts
    stride = config.val_stride
    seq_len = config.seq_len
    
    total = val_tokens.numel() - 1
    if max_tokens is not None:
        total = min(total, max_tokens)
        
    loss_sum = 0.0
    tok_cnt = 0
    byte_cnt = 0
    with torch.no_grad():
        pos = 0
        while pos < total:
            start = max(0, pos - seq_len + stride)
            end = min(pos + stride, total)
            if end <= start:
                pos += stride
                continue
            chunk = val_tokens[start:end+1].cuda()
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            loss = model(x, y).item()
            cnt = min(stride, y.size(1))
            loss_sum += loss * cnt
            tok_cnt += cnt
            tgt = y[0, -cnt:].reshape(-1)
            prev = x[0, -cnt:].reshape(-1)
            bytes_t = base[tgt] + (space[tgt] & ~boundary[prev]).to(torch.int16)
            byte_cnt += bytes_t.sum().item()
            pos += stride
    avg_loss = loss_sum / tok_cnt
    bpb = (avg_loss / math.log(2)) * (tok_cnt / byte_cnt)
    return avg_loss, bpb

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    master = (rank == 0)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = build_luts(sp, device)

    val_tokens = None
    if master:
        val_files = sorted(glob.glob(args.val_pattern))
        val_tokens = torch.cat([load_shard(Path(f)) for f in val_files])
        if master: print(f"Val tokens: {val_tokens.numel()}")
    obj_list = [val_tokens]
    dist.broadcast_object_list(obj_list, src=0)
    val_tokens = obj_list[0]

    model = GPT(args).to(device)
    ema = EMA(model, args.ema_decay)
    total_params = sum(p.numel() for p in model.parameters())
    if master: print(f"Params: {total_params:,}")

    muon_params = []
    adam_2d = []
    adam_1d = []
    for n, p in model.named_parameters():
        if p.dim() >= 2 and "embedding" not in n:
            muon_params.append(p)
        elif p.dim() >= 2:
            adam_2d.append(p)
        else:
            adam_1d.append(p)
            
    muon_opt = Muon(muon_params, lr=args.lr * 0.5, momentum=0.95)
    optimizer = torch.optim.AdamW([
        {"params": adam_2d, "weight_decay": 0.1},
        {"params": adam_1d, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / (args.iterations - args.warmup_steps)
        if progress >= 1.0:
            return args.lr_min / args.lr
        return args.lr_min / args.lr + (1 - args.lr_min/args.lr) * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_muon = torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)

    train_files = sorted(glob.glob(args.train_pattern))
    my_files = train_files[rank::world_size] if train_files else train_files
    if not my_files:
        my_files = train_files
    stream = load_shard(Path(my_files[0]))
    file_idx, pos = 0, 0

    model = DDP(model, device_ids=[local_rank])
    step = 0
    loss_ema = 0.0
    t0 = time.time()

    if master: print("Training start (Target: 5000 iterations)")
    while step < args.iterations:
        batch_tokens = args.batch_size * args.seq_len + 1
        if pos + batch_tokens > stream.numel():
            file_idx = (file_idx + 1) % len(my_files)
            stream = load_shard(Path(my_files[file_idx]))
            pos = 0
        chunk = stream[pos:pos+batch_tokens].to(device)
        pos += batch_tokens
        x = chunk[:-1].view(args.batch_size, args.seq_len)
        y = chunk[1:].view(args.batch_size, args.seq_len)
        loss = model(x, y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        muon_opt.step()
        optimizer.zero_grad()
        muon_opt.zero_grad()
        scheduler.step()
        scheduler_muon.step()
        ema.update(model.module)
        step += 1
        loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
        
        if master and (step % args.log_every == 0 or step == 1):
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"step {step:6d}/{args.iterations} loss {loss_ema:.4f} lr {lr_now:.2e} time {elapsed:.0f}s")
            
        if step % args.val_every == 0:
            backup_state = {k: v.clone() for k, v in model.module.state_dict().items()}
            ema.apply(model.module)
            
            if master:
                print(f"--- 途中評価開始（時間節約のためサンプリング） ---")
                val_loss, val_bpb = evaluate(model.module, val_tokens, luts, args, max_tokens=1000000)
                print(f"--- step {step} | val_loss {val_loss:.4f} | val_bpb {val_bpb:.4f}")
            
            dist.barrier()
            model.module.load_state_dict(backup_state)
            for p in model.module.parameters():
                dist.broadcast(p.data, src=0)
            for b in model.module.buffers():
                dist.broadcast(b.data, src=0)
            dist.barrier()

    dist.barrier() 
    if master:
        ema.apply(model.module)
        print(f"\n--- 最終評価（フルスキャン）開始 ---")
        final_loss, final_bpb = evaluate(model.module, val_tokens, luts, args)
        print(f"\nFinal val_loss: {final_loss:.4f} | val_bpb: {final_bpb:.4f}")
        
        quant_state, scales = quantize_model(model.module, bits=args.quant_bits)
        save_obj = {"model": quant_state, "scales": scales, "bits": args.quant_bits}
        buf = io.BytesIO()
        torch.save(save_obj, buf)
        comp = zlib.compress(buf.getvalue(), level=9)
        size_mb = len(comp) / (1024*1024)
        with open("/workspace/parameter-golf/submission.int6.ptz", "wb") as f:
            f.write(comp)
        print(f"Compressed size: {size_mb:.2f} MB (target <=16)")
        
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
