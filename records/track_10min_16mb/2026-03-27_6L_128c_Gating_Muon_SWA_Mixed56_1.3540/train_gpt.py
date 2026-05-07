# -----------------------------------------------------------------------------
# Parameter Golf: 128-Cluster Marathon Model (800D, Specialist Architecture)
# Rule-Compliant Consolidated Script (train_gpt.py)
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, math, time, zlib, io, uuid, glob, random, contextlib, sys
from pathlib import Path
from datetime import datetime, timedelta
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

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 2048))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    iterations = int(os.environ.get("ITERATIONS", 10000))
    warmup_iters = int(os.environ.get("WARMUP_ITERS", 100)) # Default 100 iterations warmup
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288)) # STANDARD: 524k für H100 Server (8 GPUs x 64k tokens)
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 6))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 1))
    model_dim = int(os.environ.get("MODEL_DIM", 800))
    num_heads = int(os.environ.get("NUM_HEADS", 10))
    mlp_mult = float(os.environ.get("MLP_MULT", 1.2))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    
    num_clusters = int(os.environ.get("NUM_CLUSTERS", 128))
    cluster_temp = float(os.environ.get("CLUSTER_TEMP", 1.0)) 
    cluster_lambda = float(os.environ.get("CLUSTER_LAMBDA", 0.05))

    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    embed_lr = float(os.environ.get("EMBED_LR", 0.4))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    cluster_lr = float(os.environ.get("CLUSTER_LR", 0.005))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    max_wallclock_seconds = int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600)) # 10 Minute Rule Compliance
    swa_enabled = True
    use_compile = os.environ.get("USE_COMPILE", "0" if os.name == 'nt' else "1") == "1"

# -----------------------------
# OPTIMIZER (MUON)
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.half()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps))
    @torch.no_grad()
    def step(self):
        distributed = dist.is_available() and dist.is_initialized()
        world_size, rank = (dist.get_world_size(), dist.get_rank()) if distributed else (1, 0)
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.float16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank:
                    state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(p.grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    g = p.grad.add(buf, alpha=group["momentum"])
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype), alpha=-group["lr"])
                curr += p.numel()

# -----------------------------
# CLUSTER LOGIC
# -----------------------------

class FastClusterGating(nn.Module):
    def __init__(self, vocab_size, model_dim, num_clusters=8):
        super().__init__()
        self.num_clusters, self.model_dim = num_clusters, model_dim
        self.lookup_cur = nn.Embedding(vocab_size, num_clusters)
        self.lookup_prev = nn.Embedding(vocab_size, num_clusters)
        self.logits_scale = nn.Parameter(torch.ones(num_clusters))
        self.gate_scale = nn.Parameter(torch.ones(model_dim))
        self.gate_gen = nn.Linear(num_clusters, model_dim)
        self.bias_gen = nn.Linear(num_clusters, vocab_size, bias=False)
        self.context_proj = nn.Linear(model_dim, num_clusters, bias=False)
        self.cluster_norms = nn.Parameter(torch.ones(num_clusters, model_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lookup_cur.weight, std=0.01)
        nn.init.normal_(self.lookup_prev.weight, std=0.01)
        nn.init.zeros_(self.gate_gen.weight); nn.init.zeros_(self.gate_gen.bias)
        nn.init.zeros_(self.bias_gen.weight); nn.init.zeros_(self.context_proj.weight)

    def forward(self, input_ids, hidden, temp=1.0):
        prev_ids = torch.zeros_like(input_ids)
        prev_ids[:, 1:] = input_ids[:, :-1]
        cluster_logits = (self.lookup_cur(input_ids) + self.lookup_prev(prev_ids) + self.context_proj(hidden)) / temp
        mean_sq = cluster_logits.pow(2).mean(-1, keepdim=True)
        cluster_logits = cluster_logits * torch.rsqrt(mean_sq + 1e-6) * self.logits_scale
        probs = torch.softmax(cluster_logits, dim=-1)
        hidden = hidden * torch.matmul(probs, self.cluster_norms) 
        gate_h = self.gate_gen(probs)
        gate_h = gate_h * torch.rsqrt(gate_h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gate_scale
        gated_hidden = hidden * (0.5 + torch.sigmoid(gate_h)) 
        return gated_hidden, probs, self.bias_gen(probs)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class Block(nn.Module):
    def __init__(self, dim, n_head, n_kv, mlp_mult, rope_base):
        super().__init__()
        self.an, self.mn = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, dim + 2 * n_kv * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False); self.proj._zero_init = True
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_mult), bias=False), nn.Linear(dim, int(dim * mlp_mult), bias=False), nn.Linear(int(dim * mlp_mult), dim, bias=False))
        self.mlp[2]._zero_init = True
        self.inv_freq = nn.Parameter(1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)), requires_grad=False)
        self.ascl, self.mscl = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.ones(dim))

    def rotary(self, q, k):
        t = torch.arange(q.size(2), device=q.device).float()
        freqs = torch.outer(t, self.inv_freq.to(q.device))
        cos, sin = freqs.cos()[None, None, :, :].half(), freqs.sin()[None, None, :, :].half()
        h = q.size(-1) // 2
        def apply(x): return torch.cat((x[..., :h] * cos + x[..., h:] * sin, x[..., :h] * (-sin) + x[..., h:] * cos), dim=-1)
        return apply(q), apply(k)

    def forward(self, x, x0):
        nx = self.an(x + x0)
        qkv = self.qkv(nx); head_dim = self.head_dim
        n_head = qkv.size(-1) // head_dim - 2
        q, k, v = qkv.split([n_head * head_dim, head_dim, head_dim], dim=-1)
        q = q.view(x.size(0), x.size(1), n_head, head_dim).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), 1, head_dim).transpose(1, 2); v = v.view(x.size(0), x.size(1), 1, head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.ascl * self.proj(y.transpose(1, 2).reshape(x.shape))
        nx = self.mn(x); gate, up = self.mlp[0](nx), self.mlp[1](nx)
        x = x + self.mscl * self.mlp[2](F.silu(gate) * up)
        return x

class GPT(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.tok_emb = nn.Embedding(hp.vocab_size, hp.model_dim)
        self.blocks = nn.ModuleList([Block(hp.model_dim, hp.num_heads, hp.num_kv_heads, hp.mlp_mult, hp.rope_base) for _ in range(hp.num_layers)])
        self.final_norm = nn.RMSNorm(hp.model_dim)
        self.cluster_gate = FastClusterGating(hp.vocab_size, hp.model_dim, hp.num_clusters)
        self.skip_weights = nn.Parameter(torch.ones(hp.num_layers // 2, hp.model_dim))
        self.head_scale = nn.Parameter(torch.tensor(1.0))
        self._init_weights()

    def _init_weights(self):
        for n, m in self.named_modules():
            if "cluster_gate" in n: continue
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False): nn.init.zeros_(m.weight)
                else: nn.init.normal_(m.weight, std=0.02)
                if "proj" in n: m.weight.data.mul_(1.0 / math.sqrt(2 * self.hp.num_layers))
        nn.init.normal_(self.tok_emb.weight, std=self.hp.tied_embed_init_std)

    def forward(self, idx, targets, temp=None, ent_lambda=0.0, loss_mask=None):
        if temp is None: temp = self.hp.cluster_temp
        x = F.rms_norm(self.tok_emb(idx), (self.hp.model_dim,))
        x0, skips = x, []
        for i in range(self.hp.num_layers // 2): 
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.hp.num_layers // 2, self.hp.num_layers):
            if skips: x = x + self.skip_weights[i - self.hp.num_layers // 2] * skips.pop()
            x = self.blocks[i](x, x0)
        x = self.final_norm(x)
        x, probs, bias = self.cluster_gate(idx, x, temp=temp)
        logits = (F.linear(x.reshape(-1, x.size(-1)), self.tok_emb.weight) * self.head_scale) + bias.reshape(-1, self.hp.vocab_size)
        logits = self.hp.logit_softcap * torch.tanh(logits / self.hp.logit_softcap)
        loss = F.cross_entropy(logits, targets.reshape(-1), reduction='none')
        if loss_mask is not None: loss = (loss * loss_mask.reshape(-1)).sum() / loss_mask.sum()
        else: loss = loss.mean()
        if ent_lambda > 0: loss -= ent_lambda * (-(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean())
        return loss

# -----------------------------
# DATA LOADING & EVAL
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    tokens_np = np.fromfile(file, dtype="<u2", count=int(header[2]), offset=1024)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def take(self, n: int) -> Tensor:
        chunks = []
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0; continue
            k = min(n, avail); chunks.append(self.tokens[self.pos : self.pos + k]); self.pos += k; n -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device, self.stream = rank, world_size, device, TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        span = (global_tokens // (self.world_size * grad_accum_steps)) + 1
        chunk = self.stream.take(span * self.world_size)
        local = chunk[self.rank * span : (self.rank + 1) * span].to(dtype=torch.int64)
        return local[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True), local[1:].reshape(-1, seq_len).to(self.device, non_blocking=True)

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bb, hs, ib, max_batches=0):
    torch.cuda.empty_cache()
    model.eval()
    stride, seq_len = args.train_seq_len // 2, args.train_seq_len
    win_starts = list(range(0, val_tokens.numel() - seq_len, stride))
    rank_starts = win_starts[rank::world_size]
    batch_size = 64 # Faster inference
    
    vls, vtc, vbc = torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for i in range(0, len(rank_starts), batch_size):
            if max_batches > 0 and i // batch_size >= max_batches: break
            
            curr = rank_starts[i : i + batch_size]
            # Ultra-fast as_strided windowing
            s_min, s_max = curr[0], curr[-1]
            chunk = val_tokens[s_min : s_max + seq_len + 1].to(device=device, dtype=torch.int64)
            x = chunk[:-1].as_strided((len(curr), seq_len), (stride, 1))
            y = chunk[1:].as_strided((len(curr), seq_len), (stride, 1))
            
            m = torch.ones((len(curr), seq_len), dtype=torch.bool, device=device)
            # Efficient mask: First stride tokens of all but first window are masked
            if len(curr) > 1:
                # Mask out the 'stride' overlap for all windows except the very first one of the whole split
                # But here 'curr' is a list of ANY windows. 
                # Correct logic: if a window start > 0, mask its first 'stride' tokens.
                for j, s in enumerate(curr):
                    if s > 0: m[j, :stride] = False
            
            with torch.autocast("cuda"):
                loss = model(x, y, loss_mask=m).detach()
            
            n = float(m.sum().item())
            vls += loss.double() * n; vtc += n
            vbc += (bb[y[m]].to(torch.int16) + (hs[y[m]] & ~ib[x[m]]).to(torch.int16)).double().sum()
            
            if rank == 0 and (i // batch_size) % 10 == 0:
                total_b = (len(rank_starts)//batch_size) if max_batches==0 else max_batches
                tag = "[Full]" if max_batches == 0 else "[Sample]"
                print(f"Eval {tag}: {i//batch_size}/{total_b}    ", end="\r", flush=True)
        if rank == 0: print(" " * 40, end="\r", flush=True) # Clear line
                
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vls); dist.all_reduce(vtc); dist.all_reduce(vbc)
    
    loss_val = (vls / vtc).item() if vtc > 0 else 0.0
    bpb = (loss_val / math.log(2.0)) * (vtc.item() / vbc.item()) if vbc > 0 else 0.0
    model.train()
    torch.cuda.empty_cache()
    return loss_val, bpb

# -----------------------------
# QUANTIZATION
# -----------------------------

def quantize_state_dict_mixed(state_dict):
    quantized, scales, dtypes, passthrough, qmeta = {}, {}, {}, {}, {}
    patterns = ("attn_scale", "mlp_scale", "resid_mix", "skip_weight", "head_scale", "gate_scale")
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if t.ndim < 2 or t.numel() <= 65536 or any(p in name for p in patterns):
            passthrough[name] = t.half() if t.is_floating_point() else t; continue
        t32 = t.float()
        is_mlp = "mlp" in name
        is_large = any(p in name for p in ["qkv", "proj", "mlp", "tok_emb", "bias_gen", "cluster_norms", "gate_gen"])
        if is_large and t.ndim == 2:
            if is_mlp:
                c = torch.quantile(t32.abs(), 0.99999, dim=1).clamp_min(1e-5); s = (c / 15.0)
                q = (torch.clamp(torch.round(t32 / s[:, None]), -15, 15) + 16).to(torch.uint8)
                qmeta[name] = {"scheme": "uint5_s"}
            else:
                c = torch.quantile(t32.abs(), 0.99999, dim=1).clamp_min(1e-5); s = (c / 31.0)
                q = (torch.clamp(torch.round(t32 / s[:, None]), -31, 31) + 32).to(torch.uint8)
                qmeta[name] = {"scheme": "uint6_s"}
        else:
            c = torch.quantile(t32.abs(), 0.99999, dim=1).clamp_min(1e-5); s = (c / 127.0)
            q = torch.clamp(torch.round(t32 / s[:, None]), -127, 127).to(torch.int8)
            qmeta[name] = {"scheme": "int8"}
        quantized[name], scales[name], dtypes[name] = q, s.half(), str(t.dtype).removeprefix("torch.")
    return {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough, "qmeta": qmeta}

def dequantize_state_dict_mixed(obj):
    out = {}
    for name, q in obj["quantized"].items():
        dt, s, sch = getattr(torch, obj["dtypes"][name]), obj["scales"][name].float(), obj["qmeta"][name]["scheme"]
        if sch == "uint5_s": val = q.float() - 16.0
        elif sch == "uint6_s": val = q.float() - 32.0
        else: val = q.float()
        out[name] = (val * s.view(q.shape[0], *([1]*(q.ndim-1)))).to(dt)
    for name, t in obj["passthrough"].items(): out[name] = t
    return out

# -----------------------------
# MAIN
# -----------------------------

def main():
    args = Hyperparameters()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    distri = "RANK" in os.environ
    rank, world_size, local_rank = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])) if distri else (0, 1, 0)
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distri: dist.init_process_group("nccl")
    
    def log0(m): 
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        if rank == 0:
            print(msg, flush=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
    
    if rank == 0: os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"run_{args.run_id}.txt")
    if rank == 0:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"--- Run started at {datetime.now()} ---\n")
            config_dict = {k: getattr(args, k) for k in dir(args) if not k.startswith('_') and not callable(getattr(args, k))}
            f.write(f"Config: {config_dict}\n\n")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.use_compile:
        log0("H100 SERVER MODE DETECTED: Applying Power-Tweaks...")
        args.matrix_lr = 0.024
        args.muon_backend_steps = 8
    base_model = GPT(args).to(device)
    if args.use_compile:
        log0("Compiling model for server speedup...")
        base_model = torch.compile(base_model)
    model = DDP(base_model, device_ids=[local_rank]) if distri else base_model
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    
    log0("Loading validation tokens..."); files = sorted(glob.glob(args.val_files))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in files]).contiguous()
    log0(f"Loaded {val_tokens.numel()} tokens.")
    log0(f"Run configuration: {'FINALIZE (ITERATIONS=0)' if args.iterations == 0 else f'TRAIN ({args.iterations} steps)'}")
    
    log0("Building LUTs..."); vocab_size_eff = max(int(sp.vocab_size()), args.vocab_size)
    bb = torch.zeros(vocab_size_eff, dtype=torch.int16, device=device)
    hs = torch.zeros(vocab_size_eff, dtype=torch.bool, device=device)
    ib = torch.ones(vocab_size_eff, dtype=torch.bool, device=device)
    for i in range(sp.vocab_size()):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i): continue
        ib[i], p = False, sp.id_to_piece(i)
        if sp.is_byte(i): bb[i] = 1; continue
        if p.startswith("▁"): hs[i], p = True, p[1:]
        bb[i] = len(p.encode("utf-8"))
    
    m_p, s_p, c_p, t_p = [], [], [], []
    for n, p in base_model.named_parameters():
        if "cluster_gate" in n: c_p.append(p)
        elif "tok_emb" in n: t_p.append(p)
        elif p.ndim == 2 and not any(x in n for x in ["mix", "skip", "bias", "head_scale"]): m_p.append(p)
        else: s_p.append(p)
    
    opt_muon = Muon(m_p, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    opt_adam = torch.optim.Adam([{"params": t_p, "lr": args.embed_lr, "base_lr": args.embed_lr},
                                 {"params": s_p, "lr": args.scalar_lr, "base_lr": args.scalar_lr},
                                 {"params": c_p, "lr": args.cluster_lr, "base_lr": args.cluster_lr}], betas=(0.9, 0.95))
    
    swa_weights = {n: torch.zeros_like(p, device="cpu") for n, p in base_model.named_parameters()} if args.swa_enabled else None
    swa_count = 0
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    scaler, step, t0, tt = torch.amp.GradScaler("cuda"), 0, time.perf_counter(), 0.0
    t_train_start = time.perf_counter()
    torch.cuda.empty_cache()

    while step <= args.iterations:
        t_now = time.perf_counter()
        t_step = t_now
        last = (step == args.iterations)
        
        # Hard Wallclock Break (OpenAI Rule Compliance)
        if (t_now - t_train_start) > args.max_wallclock_seconds:
            log0(f"MAX_WALLCLOCK_SECONDS ({args.max_wallclock_seconds}s) reached! Stopping training at step {step}.")
            last = True
        if step % args.val_loss_every == 0 or last:
            if last and step == 0 and args.iterations == 0:
                log0("ITERATIONS=0: Skipping baseline validation. Moving directly to finalization.")
            else:
                torch.cuda.synchronize(); tt += 1000 * (time.perf_counter() - t0); t_eval = time.perf_counter()
                log0(f"Step {step}/{args.iterations} validation..."); vl, vbpb = eval_val(args, model, rank, world_size, device, 8//world_size, val_tokens, bb, hs, ib, (50 if not last else 0))
                log0(f"Validation took {time.perf_counter() - t_eval:.2f}s")
                if rank == 0: print(f"Step {step}/{args.iterations} | val_loss:{vl:.4f} | val_bpb:{vbpb:.4f} | train_time:{tt:.0f}ms")
                torch.cuda.synchronize(); t0 = time.perf_counter()
        if last: break

        # Learning Rate Schedule (Warmup + Warmdown)
        frac_up = min(step / args.warmup_iters, 1.0) if args.warmup_iters > 0 else 1.0
        frac_down = max(0.0, (args.iterations - step) / args.warmdown_iters) if step >= args.iterations - args.warmdown_iters else 1.0
        lrs = frac_up * frac_down
        
        # Muon Momentum Warmup
        frac_mom = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups: 
            g["momentum"] = (1 - frac_mom) * args.muon_momentum_warmup_start + frac_mom * args.muon_momentum
            g["lr"] = args.matrix_lr * lrs
        for g in opt_adam.param_groups: g["lr"] = g["base_lr"] * lrs

        opt_muon.zero_grad(set_to_none=True); opt_adam.zero_grad(set_to_none=True)
        for i in range(8 // world_size):
            with (model.no_sync() if distri and i < (8 // world_size - 1) else contextlib.nullcontext()):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, 8//world_size)
                with torch.autocast("cuda"): loss = model(x, y, ent_lambda=args.cluster_lambda)
                scaler.scale(loss / (8//world_size)).backward()

        scaler.unscale_(opt_muon); scaler.unscale_(opt_adam); torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        scaler.step(opt_muon); scaler.step(opt_adam); scaler.update()
        if args.swa_enabled and step >= args.iterations - args.warmdown_iters and step % 10 == 0:
            has_nan = any(torch.isnan(p).any() for p in base_model.parameters())
            if not has_nan:
                for n, p in base_model.named_parameters(): swa_weights[n].add_(p.detach().cpu())
                swa_count += 1
            else:
                log0(f"Step {step}: NaN detected in weights, skipping SWA accumulation.")
        step += 1
        if (step % args.train_log_every == 0) or (step <= 10):
            torch.cuda.synchronize()
            step_time = (time.perf_counter() - t_step) * 1000
            cur_tt = tt + 1000 * (time.perf_counter() - t0)
            rem_steps = args.iterations - step
            eta_sec = (rem_steps * step_time) / 1000 if step_time > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_sec)))
            log0(f"Step {step}/{args.iterations} | loss:{loss.item():.4f} | lr:{opt_muon.param_groups[0]['lr']:.6f} | step:{step_time:.0f}ms | total:{cur_tt/1000:.1f}s | ETA:{eta_str}")

    if args.swa_enabled and swa_count > 0:
        base_model.load_state_dict({n: (swa_weights[n] / swa_count).to(device) for n in swa_weights}, strict=False)
        if rank == 0: torch.save(base_model.state_dict(), "final_model_swa.pt")
        log0("Final SWA validation..."); vl, vbpb = eval_val(args, model, rank, world_size, device, 8//world_size, val_tokens, bb, hs, ib, 0)
        log0(f"FINAL SWA - val_loss:{vl:.4f} val_bpb:{vbpb:.4f}")
    else:
        log0("Skipping SWA loading (no weights collected). Saving current model.")
        if rank == 0: torch.save(base_model.state_dict(), "final_model_swa.pt")

    if rank == 0:
        log0("Quantizing and checking submission size...")
        q_obj = quantize_state_dict_mixed(base_model.state_dict()); buf = io.BytesIO(); torch.save(q_obj, buf)
        compressed = zlib.compress(buf.getvalue(), level=9)
        sz = len(compressed)
        code_sz = Path(__file__).stat().st_size
        total_sz = sz + code_sz
        log0(f"Final Submission Stats:")
        log0(f" - Compressed model: {sz} bytes")
        log0(f" - Code (train_gpt.py): {code_sz} bytes")
        log0(f" - TOTAL: {total_sz} bytes ({total_sz/1e6:.2f} MB)")
        
        if total_sz > 16_000_000:
            log0(f"CRITICAL ERROR: Total submission size {total_sz} exceeds 16,000,000 bytes limit!")
            sys.exit(1)
            
        with open("final_model.int8.ptz", "wb") as f: f.write(compressed)
        log0("Success: Submission file saved as final_model.int8.ptz")
        
        q_state = torch.load(io.BytesIO(zlib.decompress(compressed)), map_location="cpu")
        base_model.load_state_dict(dequantize_state_dict_mixed(q_state), strict=True)
        t_round = time.perf_counter()
        log0("Final Roundtrip validation..."); vl, vbpb = eval_val(args, model, rank, world_size, device, 8//world_size, val_tokens, bb, hs, ib, 0)
        log0(f"Roundtrip check took {time.perf_counter() - t_round:.4f}s")
        log0(f"FINAL QUANTIZED - val_loss:{vl:.4f} val_bpb:{vbpb:.4f}")

if __name__ == "__main__":
    main()
