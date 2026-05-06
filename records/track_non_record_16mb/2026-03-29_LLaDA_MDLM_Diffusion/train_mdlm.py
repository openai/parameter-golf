"""
Masked Diffusion Language Model (MDLM) for OpenAI Parameter Golf.
Bidirectional transformer with log-linear noise schedule, adaLN timestep
conditioning, and discrete absorbing-mask ELBO evaluation.
"""
import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 1024; MASK_ID = 1024; TOTAL_VOCAB = 1025; PADDED_VOCAB = 1088
DEVICE = "cuda"; SEED = 42

NUM_LAYERS = 11; MODEL_DIM = 512; NUM_HEADS = 8; MLP_MULT = 3.0
SEQ_LEN = 2048; BATCH_SIZE = 8; GRAD_ACCUM = 4
TRAIN_STEPS = 6000; LR = 6e-4; WARMUP_STEPS = 300; WARMDOWN_STEPS = 1500
NOISE_EPS = 1e-3
VAR_EVAL_STEPS = 128  # higher = tighter bound

torch.manual_seed(SEED); np.random.seed(SEED)
NEG_INF = -1e6


# ─── Log-linear noise schedule (from MDLM) ───
def log_linear_noise(t, eps=NOISE_EPS):
    """sigma(t) = -log(1 - (1-eps)*t), alpha(t) = exp(-sigma(t)) = 1 - (1-eps)*t"""
    alpha = 1 - (1 - eps) * t
    sigma = -torch.log(alpha.clamp(min=1e-8))
    return sigma, alpha


# ─── Model ───
def rms_norm(x): return F.rms_norm(x, (x.size(-1),))

def apply_rotary(x, cos, sin):
    d = x.shape[3] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1*cos+x2*sin, x1*(-sin)+x2*cos], dim=3)

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000)*torch.arange(half,dtype=torch.float32)/half))
    def forward(self, sigma):
        emb = sigma[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.hd = n_heads, dim // n_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.c_q(x).view(B,T,self.n_heads,self.hd)
        k = self.c_k(x).view(B,T,self.n_heads,self.hd)
        v = self.c_v(x).view(B,T,self.n_heads,self.hd)
        q, k = apply_rotary(q,cos,sin), apply_rotary(k,cos,sin)
        q, k = rms_norm(q), rms_norm(k)
        y = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=False)
        return self.c_proj(y.transpose(1,2).contiguous().view(B,T,-1))

class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim=128):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2*dim, bias=True)
    def forward(self, x, c):
        s, sh = self.proj(c).unsqueeze(1).chunk(2, dim=-1)
        return rms_norm(x) * (1+s) + sh

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult, cond_dim=128):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.adaln_attn = AdaLN(dim, cond_dim)
        self.adaln_mlp = AdaLN(dim, cond_dim)
        hidden = int(dim * mlp_mult)
        self.mlp_fc = nn.Linear(dim, hidden, bias=False)
        self.mlp_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x, cos, sin, c):
        x = x + self.attn(self.adaln_attn(x, c), cos, sin)
        x = x + self.mlp_proj(F.relu(self.mlp_fc(self.adaln_mlp(x, c))).square())
        return x

class DiffusionLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(PADDED_VOCAB, MODEL_DIM)
        self.sigma_map = TimestepEmbedder(128)
        self.blocks = nn.ModuleList([Block(MODEL_DIM, NUM_HEADS, MLP_MULT) for _ in range(NUM_LAYERS)])
        self.head = nn.Linear(MODEL_DIM, PADDED_VOCAB, bias=False)
        hd = MODEL_DIM // NUM_HEADS
        inv_freq = 1.0/(10000**(torch.arange(0,hd,2,dtype=torch.float32)/hd))
        freqs = torch.outer(torch.arange(SEQ_LEN*2, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos()[None,:,None,:])
        self.register_buffer("sin", freqs.sin()[None,:,None,:])

    def forward_logits(self, xt, sigma):
        """Raw logits (used for training)."""
        B, T = xt.shape
        x = self.wte(xt)
        c = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        cos, sin = self.cos[:,:T], self.sin[:,:T]
        for b in self.blocks:
            x = b(x, cos, sin, c)
        logits = self.head(rms_norm(x))[...,:TOTAL_VOCAB].float()
        return logits

    def subs_log_probs(self, xt, sigma):
        """MDLM substitution log probs with frozen visible tokens."""
        logits = self.forward_logits(xt, sigma)
        logits[:, :, MASK_ID] = NEG_INF  # can't predict MASK
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Visible tokens: identity (frozen)
        frozen = torch.full_like(logits, NEG_INF)
        frozen.scatter_(-1, xt[..., None], 0.0)
        visible = (xt != MASK_ID)[..., None]
        return torch.where(visible, frozen, logits)


# ─── Training loss (MDLM continuous-time ELBO) ───
def mdlm_loss(model, x0):
    B = x0.shape[0]
    # Antithetic sampling
    t = torch.rand(B // 2 + 1, device=x0.device)
    t = torch.cat([t, 1 - t])[:B].clamp(1e-5, 1 - 1e-5)

    sigma, alpha = log_linear_noise(t)
    move_chance = 1 - alpha
    # Mask tokens
    move = torch.rand_like(x0.float()) < move_chance[:, None]
    xt = torch.where(move, MASK_ID, x0)

    log_probs = model.subs_log_probs(xt, sigma)
    log_p_x0 = torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)

    # dsigma = (1-eps) / (1 - (1-eps)*t)
    dsigma = (1 - NOISE_EPS) / alpha

    is_masked = (xt == MASK_ID).float()
    loss = (dsigma[:, None] * (-log_p_x0) * is_masked).sum() / (B * x0.shape[1])
    return loss


# ─── Discrete ELBO eval (from PR #820) ───
@torch.no_grad()
def variational_elbo_bits(model, x0, n_steps=128):
    """Proper discrete absorbing-mask ELBO. Returns total bits for the batch."""
    B, L = x0.shape
    total_bits = torch.zeros(B, device=x0.device)

    t_grid = torch.arange(1, n_steps+1, device=x0.device, dtype=torch.float32) / n_steps
    sigma_grid, alpha_grid = log_linear_noise(t_grid)

    # Terminal KL
    alpha_T = alpha_grid[-1]
    total_bits += L * float(alpha_T) * math.log(VOCAB_SIZE) / math.log(2.0)

    alpha_prev = 1.0
    for step in range(n_steps):
        alpha_curr = alpha_grid[step]
        sigma_curr = sigma_grid[step].expand(B)
        move_chance = 1 - alpha_curr

        xt = torch.where(torch.rand_like(x0.float()) < move_chance, MASK_ID, x0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_probs = model.subs_log_probs(xt, sigma_curr)
        log_p_x0 = torch.gather(log_probs.float(), -1, x0[..., None]).squeeze(-1)

        reveal_prob = (alpha_prev - float(alpha_curr)) / max(1.0 - float(alpha_curr), 1e-12)
        is_masked = (xt == MASK_ID).float()
        step_bits = reveal_prob * (-log_p_x0) * is_masked / math.log(2.0)
        total_bits += step_bits.sum(dim=-1)

        alpha_prev = float(alpha_curr)

    return total_bits  # [B] total bits per sequence


# ─── Data ───
def load_tokens(split):
    for base in [os.path.expanduser("~/data"), "data"]:
        path = os.path.join(base, f"fineweb_{split}_sp1024.bin")
        if os.path.exists(path):
            with open(path,"rb") as f:
                f.read(256*4)
                return torch.from_numpy(np.frombuffer(f.read(),dtype=np.uint16).astype(np.int64))
    raise FileNotFoundError("No data")

def get_lr(step):
    if step < WARMUP_STEPS: return LR * (step+1)/WARMUP_STEPS
    elif step < TRAIN_STEPS - WARMDOWN_STEPS: return LR
    else:
        progress = (TRAIN_STEPS - step) / WARMDOWN_STEPS
        return LR * (0.1 + 0.9*(0.5*(1+math.cos(math.pi*(1-progress)))))


# ─── Main ───
def main():
    print("="*60)
    print("  LLaDA v4: MDLM training + discrete ELBO eval")
    print("="*60, flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_tokens = load_tokens("train"); val_tokens = load_tokens("val")
    print(f"Train: {train_tokens.numel():,}, Val: {val_tokens.numel():,}")

    model = DiffusionLM().to(DEVICE).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {NUM_LAYERS}L {MODEL_DIM}d {NUM_HEADS}h — {n_params:,} params")
    print(f"Training: MDLM loss, log-linear noise, adaLN, antithetic sampling")
    print(f"Eval: discrete ELBO ({VAR_EVAL_STEPS} steps)\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1, fused=True)
    n_train = train_tokens.numel(); t0 = time.time(); losses = []
    model.train()

    for step in range(TRAIN_STEPS):
        lr = get_lr(step)
        for g in optimizer.param_groups: g['lr'] = lr
        optimizer.zero_grad(set_to_none=True); accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            idx = torch.randint(0, n_train-SEQ_LEN, (BATCH_SIZE,))
            batch = torch.stack([train_tokens[i:i+SEQ_LEN] for i in idx]).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = mdlm_loss(model, batch) / GRAD_ACCUM
            loss.backward(); accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); losses.append(accum_loss)
        if step % 100 == 0:
            avg = np.mean(losses[-100:])
            elapsed = time.time() - t0
            tok_s = (step+1)*BATCH_SIZE*GRAD_ACCUM*SEQ_LEN/elapsed
            print(f"  step {step:5d}/{TRAIN_STEPS} | loss={avg:.4f} | lr={lr:.1e} | {tok_s/1e3:.0f}K tok/s | {elapsed:.0f}s", flush=True)

    train_time = time.time() - t0
    print(f"\nTraining done: {train_time/60:.1f}m, loss={np.mean(losses[-100:]):.4f}", flush=True)
    torch.save(model.state_dict(), os.path.expanduser("~/llada_v4.pt"))

    # Discrete ELBO eval
    print(f"\nDiscrete ELBO eval ({VAR_EVAL_STEPS} steps, 500 seqs)...", flush=True)
    model.eval()
    total_bits = 0.0; total_tokens = 0
    n_seqs = min(500, (val_tokens.numel()-1)//SEQ_LEN)
    for i in range(n_seqs):
        x = val_tokens[i*SEQ_LEN:(i+1)*SEQ_LEN].unsqueeze(0).to(DEVICE)
        bits = variational_elbo_bits(model, x, n_steps=VAR_EVAL_STEPS)
        total_bits += bits.sum().item()
        total_tokens += SEQ_LEN
        if (i+1) % 50 == 0:
            # Approximate BPB: bits / (tokens * avg_bytes_per_token)
            # SP1024 avg ~4.3 bytes per token
            bpb = total_bits / (total_tokens * 4.3)
            print(f"  eval {i+1}/{n_seqs} | var_bpb≈{bpb:.4f}", flush=True)

    bpb = total_bits / (total_tokens * 4.3)
    print(f"\n{'='*60}")
    print(f"  VARIATIONAL BPB: {bpb:.4f}")
    print(f"  PR #820 MDLM:    1.625")
    print(f"  Our v2 MC ELBO:  2.41")
    print(f"  AR baseline:     1.22")
    print(f"{'='*60}", flush=True)

    json.dump({"var_bpb": bpb, "params": n_params, "train_min": train_time/60,
               "var_eval_steps": VAR_EVAL_STEPS, "train_loss": float(np.mean(losses[-100:]))},
              open(os.path.expanduser("~/v4_results.json"), "w"), indent=2)

if __name__ == "__main__": main()
