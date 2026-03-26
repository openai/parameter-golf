"""Hymba-11 SOTA: Hybrid Attention + Mamba SSM with Parallel Muon & Banking."""

from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib, hashlib
from pathlib import Path
import zstandard as zstd
import sentencepiece as spm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn
import numpy as np, torch, torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# --- HYPERPARAMETERS ---
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "data/fineweb10b/")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "data/cl100k_base.tiktoken")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size, train_log_every = 524_288, 200
    iterations, warmup_steps, warmdown_iters = int(os.environ.get("ITERATIONS", 20000)), 20, 3500
    train_batch_tokens, train_seq_len, max_wallclock_seconds = 524_288, 2048, 600.0
    qk_gain_init = 1.5
    ttt_enabled, ttt_lr, ttt_epochs, ttt_lora_rank = True, 0.002, 3, 4
    quant_bits, qat_start_frac, gptq_lite = 4, 0.85, True
    vocab_size, num_layers, model_dim, num_heads, num_kv_heads = 1024, 11, 512, 8, 4
    mlp_mult, hymba_expand, rope_dims = 3, 1, 16
    matrix_lr, scalar_lr, muon_momentum, muon_backend_steps = 0.02, 0.01, 0.99, 3
    weight_decay, beta1, beta2, adam_eps = 0.04, 0.9, 0.95, 1e-8

# --- PARALLEL MUON ---
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315); X = G.bfloat16()
    if X.ndim == 2: X = X.unsqueeze(0)
    transposed = X.size(-2) > X.size(-1)
    if transposed: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT; B = b * A + c * (A @ A); X = a * X + B @ X
    if transposed: X = X.mT
    return X.squeeze(0) if G.ndim == 2 else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps))
        self._built = False
    def _build(self):
        self._ws = dist.get_world_size() if dist.is_initialized() else 1
        self._meta = []
        for g in self.param_groups:
            for p in g["params"]:
                B = p.shape[0]; padded_B = ((B + self._ws - 1) // self._ws) * self._ws
                shard_B = padded_B // self._ws; tail = p.shape[1:]; dev = p.device
                self._meta.append({'p': p, 'B': B, 'pg': torch.zeros(padded_B, *tail, device=dev, dtype=torch.float16),
                                   'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.float16),
                                   'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.float16),
                                   'full_upd': torch.zeros(padded_B, *tail, device=dev, dtype=torch.float16),
                                   'scale': max(1, p.shape[-2]/p.shape[-1])**0.5})
        self._meta.sort(key=lambda m: -m['p'].numel()); self._built = True
    def launch_reduce_scatters(self):
        if not self._built: self._build()
        if not dist.is_initialized(): return
        self._futs = []
        for m in self._meta:
            p = m['p']
            if p.grad is None: self._futs.append(None); continue
            m['pg'][:m['B']].copy_(p.grad.float())
            self._futs.append(dist.reduce_scatter_tensor(m['shard'], m['pg'], async_op=True))
    @torch.no_grad()
    def step(self):
        if not self._built: self._build()
        for g in self.param_groups:
            lr, mom, steps = g["lr"], g["momentum"], g["backend_steps"]
            prev_m, prev_h = None, None
            for i, m in enumerate(self._meta):
                p = m['p']
                if p.grad is None: continue
                if prev_h: prev_h.wait(); prev_m['p'].add_(prev_m['full_upd'][:prev_m['B']].to(p.dtype), alpha=-lr*prev_m['scale'])
                if hasattr(self, '_futs') and self._futs[i]: self._futs[i].wait(); grad = m['shard']
                else: grad = p.grad.float()
                m['shard_mom'].mul_(mom).add_(grad)
                upd = zeropower_via_newtonschulz5(grad.add(m['shard_mom'], alpha=mom), steps=steps)
                if dist.is_initialized(): prev_h, prev_m = dist.all_gather_into_tensor(m['full_upd'], upd, async_op=True), m
                else: p.add_(upd.to(p.dtype), alpha=-lr*m['scale'])
            if prev_h: prev_h.wait(); prev_m['p'].add_(prev_m['full_upd'][:prev_m['B']].to(p.dtype), alpha=-lr*prev_m['scale'])
        if hasattr(self, '_futs'): del self._futs

# --- DATA ---
def load_data_shard(file: Path) -> Tensor:
    h = np.fromfile(file, dtype="<i4", count=256)
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=int(h[2]), offset=1024).astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern)); self.idx, self.pos = 0, 0
        self.tokens = load_data_shard(Path(self.files[0])) if self.files else torch.randint(0, 1024, (100000,), dtype=torch.uint16)
    def take(self, n):
        res = []
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.idx = (self.idx + 1) % len(self.files); self.tokens, self.pos = load_data_shard(Path(self.files[self.idx])), 0; continue
            k = min(n, avail); res.append(self.tokens[self.pos:self.pos+k]); self.pos += k; n -= k
        return torch.cat(res) if len(res) > 1 else res[0]

class DistributedTokenLoader:
    def __init__(self, pattern, rank, ws, device):
        self.rank, self.ws, self.device, self.stream = rank, ws, device, TokenStream(pattern)
    def next_batch(self, global_t, seq_l, accum):
        local_t = global_t // (self.ws * accum); span = local_t + 1
        chunk = self.stream.take(span * self.ws); start = self.rank * span
        l = chunk[start:start+span].to(dtype=torch.int64, device=self.device)
        return l[:-1].reshape(-1, seq_l), l[1:].reshape(-1, seq_l)

# --- MODEL ---
class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    _qat_bits = 0
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self._qat_bits > 0 and self.weight.numel() > 65536:
            qmax = (1 << (self._qat_bits - 1)) - 1
            scale = w.detach().abs().amax(dim=1, keepdim=True) / qmax
            w = (torch.clamp(torch.round(w / scale), -qmax, qmax) * scale).to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000.0**(torch.arange(0, dim, 2).float() / dim)), persistent=False)
    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device).float(); f = torch.outer(t, self.inv_freq)
        return f.cos()[None,None,:,:].to(dtype), f.sin()[None,None,:,:].to(dtype)

def apply_rotary_emb(x, cos, sin):
    h = x.size(-1)//2; return torch.cat((x[...,:h]*cos + x[...,h:]*sin, x[...,:h]*(-sin) + x[...,h:]*cos), -1)

class HymbaAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_dims, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.intermediate_size, self.ssm_state_size, self.dt_rank = dim, 8, max(dim // 16, 1)
        self.kv_dim, self.rope_dims, self.q_gain = num_kv_heads*self.head_dim, rope_dims, nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.rotary, self.conv1d = Rotary(rope_dims), nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.A_log, self.D, self.dt_bias, self.merge_alpha = nn.Parameter(torch.zeros(dim, 8)), nn.Parameter(torch.ones(dim)), nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(1))
    def forward(self, x, q_w, k_w, v_w, out_w, him_w, hvg_w, hxp_w, hmo_w):
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz,seqlen,self.num_heads,self.head_dim).transpose(1,2)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim).transpose(1,2)
        v = (F.linear(x, v_w.to(x.dtype)) * torch.sigmoid(F.linear(x, hvg_w.to(x.dtype)))).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim).transpose(1,2)
        q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = torch.cat((apply_rotary_emb(q[...,:self.rope_dims], cos, sin), q[...,self.rope_dims:]), -1) * self.q_gain.to(x.dtype)[None,:,None,None]
        k = torch.cat((apply_rotary_emb(k[...,:self.rope_dims], cos, sin), k[...,self.rope_dims:]), -1)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        ya = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1,2).reshape(bsz,seqlen,dim)
        him = F.linear(x, him_w.to(x.dtype))
        _, _, x_ssm, gate = him.split([self.kv_dim, self.kv_dim, self.intermediate_size, self.intermediate_size], -1)
        x_ssm = causal_conv1d_fn(x_ssm.transpose(1,2), self.conv1d.weight.to(x.dtype).squeeze(1), self.conv1d.bias.to(x.dtype), activation="silu")
        ssm_p = F.linear(x_ssm.transpose(1,2), hxp_w.to(x.dtype))
        dt, B, C = torch.split(ssm_p, [self.dt_rank, self.ssm_state_size, self.ssm_state_size], -1)
        dt = F.linear(dt, torch.eye(self.intermediate_size, self.dt_rank, device=x.device, dtype=x.dtype)).transpose(1,2)
        scan = selective_scan_fn(x_ssm, dt, -torch.exp(self.A_log.float()), B.transpose(1,2), C.transpose(1,2), self.D.float(), z=gate.transpose(1,2), delta_bias=self.dt_bias.float(), delta_softplus=True)
        ym = F.linear(scan.transpose(1,2), hmo_w.to(x.dtype)); w = torch.sigmoid(self.merge_alpha).to(x.dtype)
        return F.linear(ya * w + ym * (1-w), out_w.to(x.dtype))

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_dims, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.q_gain, self.rope_dims, self.rotary = nn.Parameter(torch.full((num_heads,), qk_gain_init)), rope_dims, Rotary(rope_dims)
    def forward(self, x, q_w, k_w, v_w, out_w):
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz,seqlen,self.num_heads,self.head_dim).transpose(1,2)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim).transpose(1,2)
        v = F.linear(x, v_w.to(x.dtype)).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim).transpose(1,2)
        q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = torch.cat((apply_rotary_emb(q[...,:self.rope_dims], cos, sin), q[...,self.rope_dims:]), -1) * self.q_gain.to(x.dtype)[None,:,None,None]
        k = torch.cat((apply_rotary_emb(k[...,:self.rope_dims], cos, sin), k[...,self.rope_dims:]), -1)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1,2).reshape(bsz,seqlen,dim)
        return F.linear(y, out_w.to(x.dtype))

class Block(nn.Module):
    def __init__(self, i, args):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = HymbaAttention(args.model_dim, args.num_heads, args.num_kv_heads, args.rope_dims, args.qk_gain_init) if i < args.num_layers-1 else CausalSelfAttention(args.model_dim, args.num_heads, args.num_kv_heads, args.rope_dims, args.qk_gain_init)
        self.attn_scale, self.mlp_scale = nn.Parameter(torch.ones(args.model_dim)), nn.Parameter(torch.ones(args.model_dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(args.model_dim), torch.zeros(args.model_dim))))
    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, him_w=None, hvg_w=None, hxp_w=None, hmo_w=None):
        mix = self.resid_mix.to(x.dtype); xi = mix[0]*x + mix[1]*x0
        if isinstance(self.attn, HymbaAttention): ya = self.attn(self.attn_norm(xi), q_w, k_w, v_w, out_w, him_w, hvg_w, hxp_w, hmo_w)
        else: ya = self.attn(self.attn_norm(xi), q_w, k_w, v_w, out_w)
        xo = xi + self.attn_scale.to(x.dtype)*ya
        return xo + self.mlp_scale.to(x.dtype)*F.linear(F.silu(F.linear(self.mlp_norm(xo), up_w.to(x.dtype))), down_w.to(x.dtype)), None

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.blocks = nn.ModuleList([Block(i, args) for i in range(args.num_layers)])
        self.norm, self.lm_head = RMSNorm(), nn.Linear(args.model_dim, args.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight; L, D, H, M = args.num_layers, args.model_dim, args.num_heads, args.mlp_mult*args.model_dim
        self.qo_bank, self.kv_bank = nn.Parameter(torch.randn(L, D, D)*0.02), nn.Parameter(torch.randn(L, args.num_kv_heads*(D//H), D)*0.02)
        self.mlp_up_bank, self.mlp_down_bank = nn.Parameter(torch.randn(L, M, D)*0.02), nn.Parameter(torch.randn(L, D, M)*0.02)
        self.him_bank, self.hvg_bank = nn.Parameter(torch.randn(L, D*3, D)*0.02), nn.Parameter(torch.randn(L, args.num_kv_heads*(D//H), D)*0.02)
        self.hxp_bank, self.hmo_bank = nn.Parameter(torch.randn(L, max(D//16,1)+16, D)*0.02), nn.Parameter(torch.randn(L, D, D)*0.02)
    def forward(self, x, y=None):
        x = self.tok_emb(x); x0 = x
        for i, b in enumerate(self.blocks): x, _ = b(x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[i], self.qo_bank[i], self.mlp_up_bank[i], self.mlp_down_bank[i], self.him_bank[i], self.hvg_bank[i], self.hxp_bank[i], self.hmo_bank[i])
        logits = self.lm_head(self.norm(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) if y is not None else logits

# --- MAIN ---
def main():
    args = Hyperparameters(); dist.init_process_group("nccl") if "RANK" in os.environ else None
    rank, ws = int(os.environ.get("RANK",0)), int(os.environ.get("WORLD_SIZE",1)); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK",0)) if torch.cuda.is_available() else 0
    base_model = GPT(args).to(device).bfloat16()
    model = base_model
    if torch.cuda.is_available():
        compiled = torch.compile(base_model)
        model = DDP(compiled, device_ids=[local_rank]) if dist.is_initialized() else compiled
    muon_params = [base_model.qo_bank, base_model.kv_bank, base_model.mlp_up_bank, base_model.mlp_down_bank, base_model.him_bank, base_model.hvg_bank, base_model.hxp_bank, base_model.hmo_bank]
    opt_muon = Muon(muon_params, args.matrix_lr, args.muon_momentum, args.muon_backend_steps)
    opt_adam = torch.optim.AdamW([p for p in base_model.parameters() if id(p) not in {id(m) for m in muon_params}], lr=args.scalar_lr, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)
    loader = DistributedTokenLoader(args.train_files, rank, ws, device)
    for step in range(args.iterations):
        scale = (step/args.warmup_steps) if step<args.warmup_steps else (0.5*(1+math.cos(math.pi*(step-args.iterations+args.warmdown_iters)/args.warmdown_iters)) if step>args.iterations-args.warmdown_iters else 1.0)
        base_model.zero_grad(set_to_none=True); acc = 8 // ws
        for _ in range(acc):
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, acc)
            with torch.autocast(device.type, dtype=torch.bfloat16): loss = model(x, y)
            (loss/acc).backward()
        opt_muon.launch_reduce_scatters()
        for g in opt_muon.param_groups + opt_adam.param_groups: g["lr"] = g.get("base_lr", g["lr"]) * scale
        opt_adam.step(); opt_muon.step()
        if rank==0 and step % 200 == 0: print(f"step {step} loss {loss.item():.4f}")
    if rank==0: 
        torch.save(base_model.state_dict(), "final_model.pt")
        print("Success! Final model saved.")

if __name__ == "__main__": main()
